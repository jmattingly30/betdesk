"""NFL-4 scheme layer (2026-10-11, J: "caller priors first, but model all 4 options, and a way to use all 4"):
pre-game features for the opportunity and TD models, built from nflverse participation (who is on the field, offensive
and defensive personnel, coverage, box count, pass rushers) and data/play_callers.csv (2016-2026).

  python3 scheme_features.py <dl_dir> <model_dir>      -> data/scheme_features.parquet, one row per (season, posteam, pid)
                                                          plus team columns; scheme_features_log.txt
Everything comes from EARLIER seasons (Y-1, and Y-2 at half weight for player splits), because the participation file
for the current season is only published after it ends. The same definitions therefore hold in the backtest and live.

Option 1 caller priors   For a team whose play-caller changed, the expected tendency this season = half the caller's own
                         most recent team-season (or the league average for a first-time caller) + half the team's last
                         season. cl_d_* = that prior minus the team's last season (0 when the caller stayed);
                         cl_lr_pos = log ratio of the player's position target share, prior vs last season.
Option 2 coverage        Player targets per route against each look (man/zone, one-/two-high, base/nickel/dime, 5+ rushers),
                         shrunk to his overall rate (60 routes), weighted by the opponent's look mix last season:
                         ms_* = sum(f_opp x TPR) / sum(f_league x TPR). RBs: carries per on-field run by box count (light
                         <= 6, 7, heavy >= 8) vs the opponent's box mix: mc_box. Plus the opponent's raw look rates.
Option 3 game plan       Expected personnel mix tonight = team prior mix (Option 1) + game-plan shift for the opponent's
                         nickel and 5+-rusher rates (coefficients fit on 2016-2022 team-games only). on_exp = sum over
                         personnel of mix x the player's on-field rate in that personnel; on_ratio = on_exp / the same with
                         last season's mix (how much the expected personnel helps or hurts his snaps).
Option 4 red zone        rz_exp = the player's red-zone opportunities (targets + carries) per red-zone snap x his expected
                         red-zone on-field rate under the team's (caller-adjusted) red-zone personnel mix; gl_sh = his share of
                         the team's goal-to-go (inside the 5) opportunities; cl_lr_rz = caller shift in TE/RB red-zone targets."""
import sys, os, numpy as np, pandas as pd, statsmodels.api as sm

PERS = ['p11', 'p12', 'heavy', 'other']
LOOKS = {'mz': ['man', 'zone'], 'shell': ['one', 'two'], 'db': ['base', 'nickel', 'dime'], 'pr': ['r4', 'r5']}
BOX = ['light', 'b7', 'heavybox']
TEND = ['p11', 'p12', 'heavy', 'pa', 'sh_WR', 'sh_TE', 'sh_RB', 'rz_te_rb', 'rz_pass', 'proe', 'rz_p11', 'rz_heavy']


def _cnt(s, p):
    if not isinstance(s, str): return np.nan
    for part in s.split(','):
        part = part.strip()
        if ' ' not in part: continue
        k, q = part.split(' ', 1)
        if k.isdigit() and q == p: return int(k)
    return 0


def load_season(dl, y):
    cols = ['game_id', 'play_id', 'season', 'season_type', 'week', 'posteam', 'defteam', 'pass', 'rush', 'xpass', 'qb_dropback', 'wp',
            'receiver_player_id', 'rusher_player_id', 'yardline_100', 'two_point_attempt']
    P = pd.read_parquet(f'{dl}/pbp_{y}.parquet', columns=cols)
    P = P[(P.season_type == 'REG') & (P.two_point_attempt != 1) & ((P['pass'] == 1) | (P.rush == 1)) & P.posteam.notna()]
    Q = pd.read_parquet(f'{dl}/part_{y}.parquet', columns=['nflverse_game_id', 'play_id', 'offense_personnel', 'defense_personnel', 'defenders_in_box',
                                                         'number_of_pass_rushers', 'offense_players', 'defense_man_zone_type', 'defense_coverage_type'])
    P = P.merge(Q.rename(columns={'nflverse_game_id': 'game_id'}), on=['game_id', 'play_id'], how='inner')
    if os.path.exists(f'{dl}/ftn_{y}.parquet'):
        F = pd.read_parquet(f'{dl}/ftn_{y}.parquet', columns=['nflverse_game_id', 'nflverse_play_id', 'is_play_action'])
        P = P.merge(F.rename(columns={'nflverse_game_id': 'game_id', 'nflverse_play_id': 'play_id'}), on=['game_id', 'play_id'], how='left')
    else:
        P['is_play_action'] = np.nan
    rb, te, wr = (P.offense_personnel.map(lambda s: _cnt(s, k)) for k in ('RB', 'TE', 'WR'))
    P['pers'] = np.select([(rb == 1) & (te == 1) & (wr == 3), (rb == 1) & (te == 2), (rb + te) >= 3], ['p11', 'p12', 'heavy'], 'other')
    P.loc[rb.isna(), 'pers'] = None
    dbs = P.defense_personnel.map(lambda s: sum(_cnt(s, k) for k in ('CB', 'FS', 'SS', 'DB', 'S')) if isinstance(s, str) else np.nan)
    db = P.qb_dropback == 1
    P['mz'] = P.defense_man_zone_type.map({'MAN_COVERAGE': 'man', 'ZONE_COVERAGE': 'zone'}).where(db)
    P['shell'] = P.defense_coverage_type.map({'COVER_0': 'one', 'COVER_1': 'one', 'COVER_3': 'one', 'COVER_2': 'two', 'COVER_4': 'two',
                                              'COVER_6': 'two', 'COVER_9': 'two', '2_MAN': 'two'}).where(db)
    P['db'] = pd.Series(np.select([dbs <= 4, dbs == 5, dbs >= 6], ['base', 'nickel', 'dime'], None), index=P.index).where(db & dbs.notna())
    P['pr'] = pd.Series(np.where(P.number_of_pass_rushers >= 5, 'r5', 'r4'), index=P.index).where(db & P.number_of_pass_rushers.notna())
    bx = P.defenders_in_box
    P['box'] = pd.Series(np.select([bx <= 6, bx == 7, bx >= 8], ['light', 'b7', 'heavybox'], None), index=P.index).where((P.rush == 1) & bx.notna())
    P['rz'] = P.yardline_100 <= 20; P['gl'] = P.yardline_100 <= 5
    P['opp_pid'] = P.receiver_player_id.where(P.receiver_player_id.notna(), P.rusher_player_id)
    return P


def team_tendencies(P, pos):
    P = P.copy(); P['tpos'] = P.receiver_player_id.map(pos)
    tgt = P.receiver_player_id.notna(); neutral = P.wp.between(0.2, 0.8)
    v = pd.DataFrame({'season': P.season, 'posteam': P.posteam})
    for p_ in ['p11', 'p12', 'heavy']: v[p_] = (P.pers == p_).astype(float).where(P.pers.notna())
    v['pa'] = P.is_play_action.astype(float).where(P.qb_dropback == 1)
    for p_ in ['WR', 'TE', 'RB']: v[f'sh_{p_}'] = (P.tpos == p_).astype(float).where(tgt & P.tpos.isin(['WR', 'TE', 'RB']))
    v['rz_te_rb'] = P.tpos.isin(['TE', 'RB']).astype(float).where(P.rz & tgt & P.tpos.isin(['WR', 'TE', 'RB']))
    v['rz_pass'] = P['pass'].astype(float).where(P.rz)
    v['proe'] = (P['pass'] - P.xpass).where(neutral & P.xpass.notna())
    for p_, c in [('p11', 'rz_p11'), ('heavy', 'rz_heavy')]: v[c] = (P.pers == p_).astype(float).where(P.rz & P.pers.notna())
    return v.groupby(['season', 'posteam']).mean()


def defense_looks(P):
    rows = {}
    out = pd.DataFrame(index=sorted(P.defteam.dropna().unique()))
    for dim, looks in LOOKS.items():
        c = P[P[dim].notna()].groupby(['defteam', dim]).size().unstack(dim).reindex(columns=looks).fillna(0)
        for lk in looks: out[f'n_{dim}_{lk}'] = c[lk]
    c = P[P.box.notna()].groupby(['defteam', 'box']).size().unstack('box').reindex(columns=BOX).fillna(0)
    for lk in BOX: out[f'n_box_{lk}'] = c[lk]
    return out


def player_splits(P):
    """per pid: on-field counts by personnel / team counts by personnel in games he played; targets & routes by look;
    carries & on-field runs by box; red zone and goal-line usage."""
    on = P[['game_id', 'play_id', 'posteam', 'pers', 'qb_dropback', 'rush', 'receiver_player_id', 'rusher_player_id', 'opp_pid', 'rz', 'gl', 'box']
           + list(LOOKS)].copy()
    on['is_run'] = on.rush == 1
    on['pid'] = P.offense_players.str.split(';').values; on = on.explode('pid'); on = on[on.pid.notna() & (on.pid != '')]
    out = {}
    pg = on[['pid', 'game_id', 'posteam']].drop_duplicates()
    team_pers = P[P.pers.notna()].groupby(['game_id', 'posteam', 'pers']).size().unstack('pers').reindex(columns=PERS).fillna(0)
    team_rzp = P[P.pers.notna() & P.rz].groupby(['game_id', 'posteam', 'pers']).size().unstack('pers').reindex(columns=PERS).fillna(0)
    tp = pg.join(team_pers, on=['game_id', 'posteam']).groupby('pid')[PERS].sum()
    trz = pg.join(team_rzp, on=['game_id', 'posteam']).groupby('pid')[PERS].sum()
    op = on[on.pers.notna()].groupby(['pid', 'pers']).size().unstack('pers').reindex(columns=PERS).fillna(0)
    orz = on[on.pers.notna() & on.rz].groupby(['pid', 'pers']).size().unstack('pers').reindex(columns=PERS).fillna(0)
    R = pd.DataFrame(index=tp.index)
    for p_ in PERS:
        R[f'on_{p_}'] = op[p_].reindex(R.index).fillna(0); R[f'tm_{p_}'] = tp[p_]
        R[f'rzon_{p_}'] = orz[p_].reindex(R.index).fillna(0); R[f'rztm_{p_}'] = trz[p_].reindex(R.index).fillna(0)
    dbk = on[on.qb_dropback == 1]; dbk = dbk.assign(tgt=(dbk.pid == dbk.receiver_player_id).astype(int))
    R['routes'] = dbk.groupby('pid').size().reindex(R.index).fillna(0); R['tgts'] = dbk.groupby('pid').tgt.sum().reindex(R.index).fillna(0)
    for dim, looks in LOOKS.items():
        col = dim
        x = dbk[dbk[col].notna()].groupby(['pid', col]).tgt.agg(['sum', 'size']).unstack(col)
        for lk in looks:
            R[f'r_{dim}_{lk}'] = x[('size', lk)].reindex(R.index).fillna(0) if ('size', lk) in x else 0
            R[f't_{dim}_{lk}'] = x[('sum', lk)].reindex(R.index).fillna(0) if ('sum', lk) in x else 0
    run = on[on.is_run]; run = run.assign(car=(run.pid == run.rusher_player_id).astype(int))
    R['runs_on'] = run.groupby('pid').size().reindex(R.index).fillna(0); R['cars'] = run.groupby('pid').car.sum().reindex(R.index).fillna(0)
    x = run[run.box.notna()].groupby(['pid', 'box']).car.agg(['sum', 'size']).unstack('box')
    for lk in BOX:
        R[f'ro_{lk}'] = x[('size', lk)].reindex(R.index).fillna(0) if ('size', lk) in x else 0
        R[f'c_{lk}'] = x[('sum', lk)].reindex(R.index).fillna(0) if ('sum', lk) in x else 0
    rz = on[on.rz]; R['rz_on'] = rz.groupby('pid').size().reindex(R.index).fillna(0)
    R['rz_opp'] = rz[rz.pid == rz.opp_pid].groupby('pid').size().reindex(R.index).fillna(0)
    glt = P[P.gl & P.opp_pid.notna()].groupby(['game_id', 'posteam']).size().rename('gl_team')
    R['gl_team'] = pg.join(glt, on=['game_id', 'posteam']).groupby('pid').gl_team.sum().reindex(R.index).fillna(0)
    R['gl_opp'] = P[P.gl & P.opp_pid.notna()].groupby('opp_pid').size().reindex(R.index).fillna(0)
    return R


def gameplan_coefs(TG):
    """team-game personnel deviation from its season mean on the opponent's (prior-season) nickel and 5+-rusher rates."""
    out = {}
    for p_ in ['p11', 'p12', 'heavy']:
        y = TG[p_] - TG.groupby(['season', 'posteam'])[p_].transform('mean')
        X = TG[['o_nick', 'o_r5']] - TG.groupby('season')[['o_nick', 'o_r5']].transform('mean')
        ok = y.notna() & X.notna().all(axis=1)
        out[p_] = sm.OLS(y[ok], X[ok]).fit().params.to_dict()
    return out


def build(dl, M, seasons=range(2016, 2027)):
    log = open(f'{M}/scheme_features_log.txt', 'w')
    def say(*a): s = ' '.join(str(x) for x in a); print(s, flush=True); log.write(s + '\n'); log.flush()
    pos = pd.read_parquet(f'{dl}/players.parquet', columns=['gsis_id', 'position']).set_index('gsis_id').position.replace({'FB': 'RB'})
    have = [y for y in seasons if os.path.exists(f'{dl}/part_{y}.parquet') and os.path.getsize(f'{dl}/part_{y}.parquet') > 1000]
    TS, DL, PL, TGs = [], {}, {}, []
    for y in have:
        P = load_season(dl, y)
        TS.append(team_tendencies(P, pos)); DL[y] = defense_looks(P); PL[y] = player_splits(P)
        tg = P[P.pers.notna()].groupby(['season', 'game_id', 'posteam', 'defteam']).pers.value_counts(normalize=True).unstack().reset_index()
        TGs.append(tg); say(f'{y}: {len(P)} plays, {len(PL[y])} players')
    TS = pd.concat(TS); LG = TS.groupby(level=0).mean()
    C = pd.read_csv(f'{M}/data/play_callers.csv'); cur = C.set_index(['season', 'team']).caller

    def drate(y):          # defense look rates in season y (shrunk to the league, 150 plays per dimension)
        d = DL[y].copy(); out = pd.DataFrame(index=d.index)
        for dim, looks in list(LOOKS.items()) + [('box', BOX)]:
            n = sum(d[f'n_{dim}_{lk}'] for lk in looks); lg = sum(d[f'n_{dim}_{lk}'].sum() for lk in looks)
            for lk in looks:
                out[f'f_{dim}_{lk}'] = (d[f'n_{dim}_{lk}'] + 150 * d[f'n_{dim}_{lk}'].sum() / lg) / (n + 150)
        return out
    DR = {y: drate(y) for y in DL}
    # game-plan coefficients from 2016-2022 only (the backtest tests 2024-2025)
    TG = pd.concat(TGs)
    TG['o_nick'] = [DR.get(s - 1, pd.DataFrame()).get('f_db_nickel', pd.Series(dtype=float)).get(d, np.nan) + DR.get(s - 1, pd.DataFrame()).get('f_db_dime', pd.Series(dtype=float)).get(d, np.nan)
                    for s, d in zip(TG.season, TG.defteam)]
    TG['o_r5'] = [DR.get(s - 1, pd.DataFrame()).get('f_pr_r5', pd.Series(dtype=float)).get(d, np.nan) for s, d in zip(TG.season, TG.defteam)]
    GP = gameplan_coefs(TG[TG.season <= 2022])
    say('game-plan coefficients (personnel share per +1.0 opponent nickel/dime rate, per +1.0 5+-rusher rate), fit 2016-2022:',
        {k: {a: round(b, 3) for a, b in v.items()} for k, v in GP.items()})

    team_rows, opp_rows, pl_rows = [], [], []
    for Y in sorted(set(C.season)):
        if Y - 1 not in DL: continue
        for t in sorted(set(C.team)):
            if (Y, t) not in cur.index or (Y - 1, t) not in TS.index: continue
            last = TS.loc[(Y - 1, t)]; c = cur[(Y, t)]
            same = (Y - 1, t) in cur.index and cur[(Y - 1, t)] == c
            prev = [(s, tt) for (s, tt), cc in cur.items() if cc == c and s < Y and tt != t and (s, tt) in TS.index]
            prev = max(prev) if prev else None
            if same: prior, mover = last, 0
            elif prev: prior, mover = 0.5 * TS.loc[prev] + 0.5 * last, 1
            else: prior, mover = 0.5 * LG.loc[Y - 1] + 0.5 * last, 0
            r = dict(season=Y, posteam=t, cl_new=int(not same), cl_mover=mover)
            for k in TEND: r[f'cl_d_{k}'] = float(prior[k] - last[k]) if pd.notna(prior[k]) and pd.notna(last[k]) else 0.0
            for k in TEND: r[f'prior_{k}'] = float(prior[k]) if pd.notna(prior[k]) else float(LG.loc[Y - 1, k]); r[f'last_{k}'] = float(last[k]) if pd.notna(last[k]) else float(LG.loc[Y - 1, k])
            team_rows.append(r)
        d = DR[Y - 1].copy(); d['season'] = Y; d.index.name = 'opp'; opp_rows.append(d.reset_index())
        # players: Y-1 at weight 1, Y-2 at 0.5
        a = PL[Y - 1]; b = PL.get(Y - 2)
        S_ = a.add(0.5 * b, fill_value=0) if b is not None else a.copy()
        S_ = S_.copy(); S_['season'] = Y; S_.index.name = 'pid'; pl_rows.append(S_.reset_index())
    T = pd.DataFrame(team_rows); O = pd.concat(opp_rows); PLY = pd.concat(pl_rows)
    lgmix = {Y: {f'f_{dim}_{lk}': DR[Y - 1][f'f_{dim}_{lk}'].mean() for dim, looks in list(LOOKS.items()) + [('box', BOX)] for lk in looks} for Y in O.season.unique()}
    T.to_parquet(f'{M}/data/scheme_team.parquet', index=False); O.to_parquet(f'{M}/data/scheme_opp.parquet', index=False)
    PLY.to_parquet(f'{M}/data/scheme_players.parquet', index=False)
    import json; json.dump(dict(gameplan=GP, lgmix={int(k): v for k, v in lgmix.items()}), open(f'{M}/data/scheme_meta.json', 'w'), indent=1)
    say(f'team-seasons {len(T)} (new caller {int(T.cl_new.sum())}, of them movers {int(T.cl_mover.sum())}); opponent rows {len(O)}; player-seasons {len(PLY)}')


def add_scheme(df, M):
    """Attach NFL-4 features to a player-game frame (needs season, posteam, opp, pid, position, games_season)."""
    import json
    T = pd.read_parquet(f'{M}/data/scheme_team.parquet'); O = pd.read_parquet(f'{M}/data/scheme_opp.parquet')
    PLY = pd.read_parquet(f'{M}/data/scheme_players.parquet'); meta = json.load(open(f'{M}/data/scheme_meta.json'))
    GP, LGM = meta['gameplan'], {int(k): v for k, v in meta['lgmix'].items()}
    allx = {c for v in SCHEME_X.values() for c in v}
    keep = [c for c in df.columns if c not in allx]
    d = df[keep].merge(T, on=['season', 'posteam'], how='left').merge(O, on=['season', 'opp'], how='left').merge(PLY, on=['season', 'pid'], how='left')
    fade = np.exp(-d.games_season.fillna(0) / 6)
    pos = d.position
    # Option 1
    lr = np.select([pos == 'WR', pos == 'TE', pos == 'RB'],
                   [np.log(d.prior_sh_WR / d.last_sh_WR), np.log(d.prior_sh_TE / d.last_sh_TE), np.log(d.prior_sh_RB / d.last_sh_RB)], 0.0)
    out = pd.DataFrame(index=df.index)
    out['cl_new'] = d.cl_new.fillna(0).to_numpy(); out['cl_mover'] = d.cl_mover.fillna(0).to_numpy()
    out['cl_lr_pos'] = (np.nan_to_num(lr) * fade).to_numpy()
    for k in ['p11', 'p12', 'pa', 'proe', 'rz_pass']: out[f'cl_d_{k}'] = (d[f'cl_d_{k}'].fillna(0) * fade).to_numpy()
    rzl = np.where(pos.isin(['TE', 'RB']), np.log(d.prior_rz_te_rb / d.last_rz_te_rb), np.where(pos == 'WR', np.log((1 - d.prior_rz_te_rb) / (1 - d.last_rz_te_rb)), 0.0))
    out['cl_lr_rz'] = (np.nan_to_num(rzl) * fade).to_numpy()
    # Option 2: coverage-weighted targets per route, box-weighted carries per run
    base_t = (d.tgts / d.routes.clip(lower=1))
    lg_ = d.season.map(lambda y: LGM.get(int(y), {}))
    for dim, looks in LOOKS.items():
        num = 0; den = 0
        for lk in looks:
            tpr = (d[f't_{dim}_{lk}'] + 60 * base_t) / (d[f'r_{dim}_{lk}'] + 60)
            num = num + d[f'f_{dim}_{lk}'] * tpr; den = den + lg_.map(lambda m: m.get(f'f_{dim}_{lk}', np.nan)) * tpr
        out[f'ms_{dim}'] = (num / den).where(d.routes >= 60).to_numpy()
    base_c = d.cars / d.runs_on.clip(lower=1); num = 0; den = 0
    for lk in BOX:
        cr = (d[f'c_{lk}'] + 40 * base_c) / (d[f'ro_{lk}'] + 40)
        num = num + d[f'f_box_{lk}'] * cr; den = den + lg_.map(lambda m: m.get(f'f_box_{lk}', np.nan)) * cr
    out['mc_box'] = (num / den).where(d.runs_on >= 40).to_numpy()
    out['o_man'] = d.f_mz_man.to_numpy(); out['o_two'] = d.f_shell_two.to_numpy(); out['o_nick'] = (d.f_db_nickel + d.f_db_dime).to_numpy()
    out['o_r5'] = d.f_pr_r5.to_numpy(); out['o_hbox'] = d.f_box_heavybox.to_numpy()
    # Option 3: expected personnel tonight (caller prior + game plan), times the player's on-field rate by personnel
    lgn = lg_.map(lambda m: m.get('f_db_nickel', 0) + m.get('f_db_dime', 0)); lgr = lg_.map(lambda m: m.get('f_pr_r5', 0))
    mix_now, mix_last = {}, {}
    for p_ in ['p11', 'p12', 'heavy']:
        g = GP[p_]; shift = g['o_nick'] * (out.o_nick - lgn.to_numpy()) + g['o_r5'] * (out.o_r5 - lgr.to_numpy())
        mix_now[p_] = (d[f'prior_{p_}'] * fade + d[f'last_{p_}'] * (1 - fade)).to_numpy() + np.nan_to_num(shift)
        mix_last[p_] = d[f'last_{p_}'].to_numpy()
    mix_now['other'] = 1 - sum(mix_now[p] for p in ['p11', 'p12', 'heavy']); mix_last['other'] = 1 - sum(mix_last[p] for p in ['p11', 'p12', 'heavy'])
    onr = {p_: ((d[f'on_{p_}'] + 5 * d[[f'on_{q}' for q in PERS]].sum(axis=1) / d[[f'tm_{q}' for q in PERS]].sum(axis=1).clip(lower=1)) / (d[f'tm_{p_}'] + 5)).to_numpy() for p_ in PERS}
    e_now = sum(np.clip(mix_now[p], 0, 1) * onr[p] for p in PERS); e_last = sum(np.clip(mix_last[p], 0, 1) * onr[p] for p in PERS)
    ok = (d[[f'tm_{q}' for q in PERS]].sum(axis=1) >= 100).to_numpy()
    out['on_exp'] = np.where(ok, e_now, np.nan); out['on_ratio'] = np.where(ok, e_now / np.clip(e_last, 0.01, None), np.nan)
    out['gp_d11'] = (mix_now['p11'] - (d.prior_p11 * fade + d.last_p11 * (1 - fade)).to_numpy())
    # Option 4: red zone
    rzmix = {'p11': (d.prior_rz_p11 * fade + d.last_rz_p11 * (1 - fade)).to_numpy(), 'heavy': (d.prior_rz_heavy * fade + d.last_rz_heavy * (1 - fade)).to_numpy()}
    rzmix['p12'] = np.clip(1 - rzmix['p11'] - rzmix['heavy'], 0, 1) * 0.6; rzmix['other'] = np.clip(1 - rzmix['p11'] - rzmix['heavy'] - rzmix['p12'], 0, 1)
    rzon = {p_: ((d[f'rzon_{p_}'] + 3 * onr[p_]) / (d[f'rztm_{p_}'] + 3)).to_numpy() for p_ in PERS}
    rz_onexp = sum(rzmix[p] * rzon[p] for p in PERS)
    rz_rate = ((d.rz_opp + 10 * 0.1) / (d.rz_on + 10)).to_numpy()
    out['rz_exp'] = np.where((d.rz_on.fillna(0) >= 1).to_numpy(), rz_onexp * rz_rate, np.nan)
    out['gl_sh'] = ((d.gl_opp + 0.5 * 0.05) / (d.gl_team + 0.5)).where(d.gl_team.notna()).to_numpy()
    return df.drop(columns=[c for c in out.columns if c in df]).assign(**{c: out[c].to_numpy() for c in out.columns})


SCHEME_X = {1: ['cl_new', 'cl_mover', 'cl_lr_pos', 'cl_d_p11', 'cl_d_p12', 'cl_d_pa', 'cl_d_proe', 'cl_d_rz_pass', 'cl_lr_rz'],
            2: ['ms_mz', 'ms_shell', 'ms_db', 'ms_pr', 'mc_box', 'o_man', 'o_two', 'o_nick', 'o_r5', 'o_hbox'],
            3: ['on_exp', 'on_ratio', 'gp_d11'],
            4: ['rz_exp', 'gl_sh']}

if __name__ == '__main__':
    build(sys.argv[1], sys.argv[2])
