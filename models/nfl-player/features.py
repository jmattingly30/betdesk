"""Walk-forward (pre-game) features for every player-game in data/player_games.parquet.

python3 features.py <model_dir>      -> data/features.parquet

Everything is computed from games strictly before the current one. Who plays (2026-10-09 audit fix F1): when the injury
reports are passed in (build(df, inj)), teammates' availability is what was knowable BEFORE the game: each recent-rotation
player counts with his probability of playing given his final injury-report status (Out / Doubtful / Questionable /
not listed), estimated from 2016-2019, instead of who actually played. The starting QB is last game's starter unless the
report rules him Out or Doubtful (then the announced replacement), so an in-game injury no longer hands the backup the
starter's inputs. Decay is per game played, with an extra discount
when a season changes, so last year still counts early in a new season."""
import sys, numpy as np, pandas as pd

SEASON_CARRY = 0.7   # weight multiplier on everything before a season break
# 2026-10-26 (season adjustments cards 6+7): carry 0.5 -> 0.7 and rolling 3-season position priors (were fixed 2016-19).
# Back-test (fit 2017-23 test 2024-25; fit 2017-25 test 2026 wk 1-7): better than before on 14 of 15 and 12 of 15 stats, e.g. 2026 QB
# pass yds MAE 58.75 -> 58.08, WR rec yds 23.63 -> 23.34, TE rec yds 15.15 -> 15.00. Lower carry (0.3) and a carry that
# drops when the team's play-caller changes (0.7/0.4, 0.85/0.5) were both worse.


def decayed(df, by, num_cols, half, carry=SEASON_CARRY):
    """Pre-game exponentially decayed sums of num_cols within each `by` group (rows sorted by date)."""
    d = 0.5 ** (1 / half)
    out = np.zeros((len(df), len(num_cols)))
    nobs = np.zeros(len(df))
    vals = df[num_cols].to_numpy(float)
    seas = df.season.to_numpy()
    for _, idx in df.groupby(by, sort=False).indices.items():
        acc = np.zeros(len(num_cols)); n = 0.0; last_s = None
        for i in idx:
            if last_s is not None and seas[i] != last_s:
                acc *= carry; n *= carry
            out[i] = acc; nobs[i] = n
            acc = acc * d + vals[i]; n = n * d + 1; last_s = seas[i]
    return pd.DataFrame(out, columns=num_cols, index=df.index), pd.Series(nobs, index=df.index)


# ---------- player opportunity shares and efficiency (decayed sums) ----------
# no status: split by whether he played the team's previous game (a rotation player who missed it with no status is
# usually on IR or benched, which the report does not list)
P_PLAY_DEFAULT = {'Out': 0.02, 'Doubtful': 0.2, 'Questionable': 0.8, None: 0.95, 'missed_last': 0.4}

def load_injuries(dl_dir, seasons):
    import os
    fr = [pd.read_parquet(f'{dl_dir}/inj_{y}.parquet', columns=['season', 'week', 'team', 'gsis_id', 'report_status', 'game_type'])
          for y in seasons if os.path.exists(f'{dl_dir}/inj_{y}.parquet')]
    return pd.concat(fr, ignore_index=True) if fr else None


def build(df, inj=None):
    df = df.sort_values(['game_date', 'game_id', 'posteam', 'pid']).reset_index(drop=True)
    df['team_pass_att_'] = df.team_pass_att
    df['team_carries_'] = df.team_carries
    df['games_one'] = 1.0
    if 'scrambles' not in df: df['scrambles'] = 0.0
    df['scrambles'] = df.scrambles.fillna(0)
    pcols = ['scrambles', 'targets', 'team_pass_att_', 'carries', 'team_carries_', 'pass_att', 'offense_pct', 'rz_targets', 'team_rz_targets',
             'rz_carries', 'team_rz_carries', 'air_yds', 'team_air', 'rec', 'rec_yds', 'rec_td', 'rush_yds', 'rush_td', 'cmp',
             'pass_yds', 'pass_td', 'ints', 'games_one']
    for half, tag in [(3, 's'), (8, 'l')]:
        sums, n = decayed(df, 'pid', pcols, half)
        df[f'n_{tag}'] = n
        df[f'tsh_{tag}'] = sums.targets / sums.team_pass_att_.clip(lower=1)
        df[f'csh_{tag}'] = sums.carries / sums.team_carries_.clip(lower=1)
        df[f'psh_{tag}'] = sums.pass_att / sums.team_pass_att_.clip(lower=1)
        df[f'snap_{tag}'] = sums.offense_pct / n.clip(lower=1e-9)
        if tag == 'l':
            S = sums
            df['rzt_sh'] = sums.rz_targets / sums.team_rz_targets.clip(lower=1)
            df['rzc_sh'] = sums.rz_carries / sums.team_rz_carries.clip(lower=1)
            df['air_sh'] = sums.air_yds / sums.team_air.clip(lower=1)
            df['adot'] = sums.air_yds / sums.targets.clip(lower=0.5)
            # audit F4: QB rushing = designed runs (share of team carries) + scrambles (per dropback, opponent-dependent)
            df['scr_rate'] = (sums.scrambles + 30 * 0.035) / (sums.pass_att + sums.scrambles + 30)
            df['des_sh'] = (sums.carries - sums.scrambles).clip(lower=0) / sums.team_carries_.clip(lower=1)
    for c in ['tsh', 'csh', 'psh', 'snap']:
        for t in ['s', 'l']:
            df.loc[df[f'n_{t}'] == 0, f'{c}_{t}'] = np.nan
    df['last_snap'] = df.groupby('pid').offense_pct.shift(1)
    df['last_tsh'] = (df.targets / df.team_pass_att.clip(lower=1)).groupby(df.pid).shift(1)
    df['last_csh'] = (df.carries / df.team_carries.clip(lower=1)).groupby(df.pid).shift(1)
    df['games_prior'] = df.groupby('pid').cumcount()
    df['games_season'] = df.groupby(['pid', 'season']).cumcount()
    df['rookie'] = (df.season == df.rookie_season).astype(int)
    df['days_rest'] = df.groupby('pid').game_date.diff().dt.days.clip(upper=30)

    # shrunk efficiency: (decayed numerator + k * position prior) / (decayed denominator + k); priors from 2016-2019 data
    PC = ['rec', 'targets', 'rec_yds', 'rec_td', 'carries', 'rush_yds', 'rush_td', 'cmp', 'pass_att', 'pass_yds', 'pass_td', 'ints']
    # rolling priors: each season's position averages from the 3 seasons before it (2016-2019 for 2016-2019 rows)
    ss = df.groupby(['season', 'position'])[PC].sum()
    rows = []
    for s_ in sorted(df.season.unique()):
        yrs = list(range(2016, 2020)) if s_ <= 2019 else list(range(s_ - 3, s_))
        x = ss[ss.index.get_level_values(0).isin(yrs)].groupby(level=1).sum(); x['season'] = s_; rows.append(x.reset_index())
    PR = pd.concat(rows).set_index(['season', 'position'])
    key = pd.MultiIndex.from_arrays([df.season, df.position])
    pri = None
    def shrink(num, den, prior_num, prior_den, k):
        pr = (PR[num] / PR[den].clip(lower=1)).reindex(key).to_numpy()
        return (S[num] + k * pr) / (S[den] + k)
    df['catch_r'] = shrink('rec', 'targets', None, None, 25)
    df['ypr'] = shrink('rec_yds', 'rec', None, None, 20)
    df['ypt'] = df.catch_r * df.ypr
    df['rec_td_r'] = shrink('rec_td', 'targets', None, None, 80)
    df['ypc'] = shrink('rush_yds', 'carries', None, None, 40)
    df['rush_td_r'] = shrink('rush_td', 'carries', None, None, 80)
    df['cmp_r'] = shrink('cmp', 'pass_att', None, None, 60)
    df['ypa'] = shrink('pass_yds', 'pass_att', None, None, 80)
    df['pass_td_r'] = shrink('pass_td', 'pass_att', None, None, 150)
    df['int_r'] = shrink('ints', 'pass_att', None, None, 200)

    # ---------- team-game table: volume, starting QB, defense allowed ----------
    tg = df.groupby(['game_id', 'posteam']).agg(season=('season', 'first'), game_date=('game_date', 'first'), opp=('opp', 'first'),
            team_pass_att=('team_pass_att', 'first'), team_carries=('team_carries', 'first'), team_dropbacks=('team_dropbacks', 'first'),
            team_score=('team_score', 'first'), implied=('implied', 'first')).reset_index()
    tg = tg.sort_values(['game_date', 'game_id', 'posteam']).reset_index(drop=True)
    tg['plays'] = tg.team_pass_att + tg.team_carries
    # league passing environment (audit F2): recent weeks' league pass attempts per team-game and yards per attempt,
    # exponentially weighted (half-life 8 weeks, no reset between seasons), relative to 2016-2019; only completed weeks
    # before the game's week count
    wk_of = df.drop_duplicates('game_id').set_index('game_id').week
    tg['_t'] = tg.season * 100 + tg.game_id.map(wk_of).astype(float)
    done = df[(df.team_pass_att > 0) & (df.get('_new', pd.Series(0, index=df.index)).fillna(0) != 1)].assign(_t=lambda d: d.season * 100 + d.week)
    L = done.drop_duplicates(['game_id', 'posteam']).groupby('_t').team_pass_att.mean().to_frame('pa')
    yd = done.groupby('_t')[['pass_yds', 'pass_att']].sum(); L['ypa'] = yd.pass_yds / yd.pass_att.clip(lower=1)
    L = L.sort_index(); Le = L.ewm(halflife=8).mean(); base = L[(L.index // 100).isin(range(2016, 2020))].mean()
    j = np.searchsorted(Le.index.to_numpy(), tg._t.to_numpy(), side='left') - 1
    ok = j >= 0; jj = np.clip(j, 0, None)
    tg['l_lg_pa'] = np.where(ok, np.log(Le.pa.to_numpy()[jj] / base.pa), 0.0)
    tg['l_lg_ypa'] = np.where(ok, np.log(Le.ypa.to_numpy()[jj] / base.ypa), 0.0)
    tg = tg.drop(columns=['_t'])
    ts, n = decayed(tg, 'posteam', ['team_pass_att', 'team_carries', 'plays', 'team_score', 'implied'], 6)
    for c in ['team_pass_att', 'team_carries', 'plays', 'team_score', 'implied']:
        tg[f'tm_{c}'] = ts[c] / n.clip(lower=1e-9)

    # what each defense allowed, by receiver position, against what those players normally do (expected from their own shrunk rates)
    df['exp_rec_yds'] = df.targets * df.ypt
    df['exp_rec'] = df.targets * df.catch_r
    df['exp_rush_yds'] = df.carries * df.ypc
    df['exp_cmp'] = df.pass_att * df.cmp_r
    df['exp_pass_yds'] = df.pass_att * df.ypa
    df['exp_pass_td'] = df.pass_att * df.pass_td_r
    df['exp_td'] = df.targets * df.rec_td_r + df.carries * df.rush_td_r
    df['td'] = df.rec_td + df.rush_td
    dcols = []
    agg = {}
    for pos in ['WR', 'TE', 'RB']:
        m = df.position == pos
        for a, e in [('rec_yds', 'exp_rec_yds'), ('rec', 'exp_rec')]:
            agg[f'{a}_{pos}'] = df[a].where(m, 0); agg[f'{e}_{pos}'] = df[e].where(m, 0); dcols.append((f'{a}_{pos}', f'{e}_{pos}'))
        agg[f'targets_{pos}'] = df.targets.where(m, 0)
    for a, e in [('rush_yds', 'exp_rush_yds'), ('cmp', 'exp_cmp'), ('pass_yds', 'exp_pass_yds'), ('pass_td', 'exp_pass_td'), ('td', 'exp_td')]:
        agg[a] = df[a]; agg[e] = df[e]; dcols.append((a, e))
    agg['scr_allowed'] = df.scrambles
    A = pd.DataFrame(agg); A['game_id'] = df.game_id; A['defteam'] = df.opp
    A = A.groupby(['game_id', 'defteam']).sum().reset_index()
    A = A.merge(tg[['game_id', 'opp', 'game_date', 'season', 'team_pass_att', 'team_carries']].rename(columns={'opp': 'defteam', 'team_pass_att': 'pa_faced', 'team_carries': 'ca_faced'}),
                on=['game_id', 'defteam']).sort_values(['game_date', 'game_id']).reset_index(drop=True)
    tgt_cols = [f'targets_{p}' for p in ['WR', 'TE', 'RB']]
    num = [c for pair in dcols for c in pair] + tgt_cols + ['pa_faced', 'ca_faced', 'scr_allowed']
    dsum, dn = decayed(A, 'defteam', num, 10)
    K = {'rec_yds': 600, 'rec': 60, 'rush_yds': 400, 'cmp': 60, 'pass_yds': 800, 'pass_td': 6, 'td': 6}
    for a, e in dcols:
        k = K[a.split('_WR')[0].split('_TE')[0].split('_RB')[0]]
        A[f'd_{a}'] = (dsum[a] + k) / (dsum[e] + k)          # >1 = defense allows more than expected
    for p_ in ['WR', 'TE', 'RB']:
        A[f'd_tsh_{p_}'] = (dsum[f'targets_{p_}'] + 20 * 0.0) / dsum.pa_faced.clip(lower=1)
    A['d_pa'] = dsum.pa_faced / dn.clip(lower=1e-9)
    A['d_scr'] = (dsum.scr_allowed + 40 * 0.035) / (dsum.pa_faced + 40)   # scrambles allowed per pass attempt faced
    A['d_ca'] = dsum.ca_faced / dn.clip(lower=1e-9)
    dfeat = [c for c in A.columns if c.startswith('d_')]
    lg_tsh = df[df.season <= 2019].groupby('position').targets.sum() / tg[tg.season <= 2019].team_pass_att.sum()
    for p_ in ['WR', 'TE', 'RB']:
        A[f'd_tsh_{p_}'] = (A[f'd_tsh_{p_}'] * dn + 4 * lg_tsh[p_]) / (dn + 4)
    tg = tg.merge(A[['game_id', 'defteam'] + dfeat].rename(columns={'defteam': 'opp'}), on=['game_id', 'opp'], how='left')

    # starting QB = QB with most snaps (announced pre-game); his shrunk passing rates
    q = df[df.position == 'QB'].sort_values('offense_snaps', ascending=False).drop_duplicates(['game_id', 'posteam'])
    status = {}
    if inj is not None:
        ii = inj.dropna(subset=['report_status'])
        status = {(int(a), int(b), c, d): e for a, b, c, d, e in zip(ii.season, ii.week, ii.team, ii.gsis_id, ii.report_status)}
    df['_status'] = [status.get((int(a), int(b), c, d)) for a, b, c, d in zip(df.season, df.week, df.posteam, df.pid)] if status else None
    if inj is not None:
        # pre-game starter: last game's starter unless ruled Out/Doubtful this week (new rows from project.py keep theirs)
        tq = q[['game_id', 'posteam', 'pid', 'game_date', 'season', 'week']].sort_values(['posteam', 'game_date'])
        tq['prev_qb'] = tq.groupby('posteam').pid.shift(1)
        tq['prev_status'] = [status.get((int(a), int(b), c, d)) if isinstance(d, str) else None for a, b, c, d in zip(tq.season, tq.week, tq.posteam, tq.prev_qb)]
        on_roster = set(zip(df.game_id, df.posteam, df.pid))
        use_prev = tq.prev_qb.notna() & ~tq.prev_status.isin(['Out', 'Doubtful']) & np.array([(g, t, p) in on_roster for g, t, p in zip(tq.game_id, tq.posteam, tq.prev_qb)], dtype=bool)
        new_rows = set(df.loc[df.get('_new', pd.Series(0, index=df.index)).fillna(0) == 1, 'game_id']) if '_new' in df else set()
        use_prev &= ~tq.game_id.isin(new_rows)
        tq['qb_pre'] = np.where(use_prev, tq.prev_qb, tq.pid)
        print('pre-game starter differs from the in-game starter in %d of %d team-games' % ((tq.qb_pre != tq.pid).sum(), len(tq)), flush=True)
        q = df.merge(tq[['game_id', 'posteam', 'qb_pre']].rename(columns={'qb_pre': 'pid'}), on=['game_id', 'posteam', 'pid'])
    q = q[['game_id', 'posteam', 'pid', 'ypa', 'cmp_r', 'pass_td_r', 'n_l']].rename(columns={'pid': 'qb_pid', 'ypa': 'qb_ypa', 'cmp_r': 'qb_cmp', 'pass_td_r': 'qb_td_r', 'n_l': 'qb_n'})
    tg = tg.merge(q, on=['game_id', 'posteam'], how='left')
    tg['qb_ypa_rel'] = tg.qb_ypa / tg.groupby('posteam').qb_ypa.transform(lambda s: s.shift(1).rolling(8, min_periods=1).mean())
    tg['qb_changed'] = (tg.qb_pid != tg.groupby('posteam').qb_pid.shift(1)).astype(int)

    # ---------- teammate availability: who from the recent rotation is missing today ----------
    # recent rotation = anyone who played in either of the team's previous two games; their latest pre-game shares
    df['_t'] = df.groupby(['posteam']).game_date.rank(method='dense')
    team_games = df[['posteam', 'game_id', 'game_date']].drop_duplicates().sort_values(['posteam', 'game_date'])
    team_games['prev1'] = team_games.groupby('posteam').game_id.shift(1)
    team_games['prev2'] = team_games.groupby('posteam').game_id.shift(2)
    # each player's post-game share state (after game g) = pre-game of next game; use decayed long share including game g
    post = df[['pid', 'game_id', 'posteam', 'position', 'season']].copy()
    for c, numc, denc in [('tsh', 'targets', 'team_pass_att'), ('csh', 'carries', 'team_carries'), ('psh', 'pass_att', 'team_pass_att')]:
        post[c] = ((df[f'{c}_l'].fillna(0) * df.n_l * (0.5 ** (1 / 8)) + df[numc] / df[denc].clip(lower=1)) / (df.n_l * (0.5 ** (1 / 8)) + 1))
    cur = df[['game_id', 'posteam', 'pid']]
    active = cur.groupby(['game_id', 'posteam']).pid.apply(set).to_dict()
    post_by_game = {k: v for k, v in post.groupby('game_id')}
    # probability of playing by final report status, from recent-rotation players 2016-2019
    gsea = df.drop_duplicates('game_id').set_index('game_id')[['season', 'week']]
    pplay = dict(P_PLAY_DEFAULT); obs = []
    def pkey(st, pid, r):
        if st in ('Out', 'Doubtful', 'Questionable'): return st
        return None if pid in active.get((r.prev1, r.posteam), set()) else 'missed_last'
    def rotation(r):
        seen = {}
        for gprev in [r.prev2, r.prev1]:                       # later game overwrites
            if isinstance(gprev, str) and gprev in post_by_game:
                pg = post_by_game[gprev]
                for x in pg[pg.posteam == r.posteam].itertuples():
                    seen[x.pid] = x
        return seen
    if status:
        for r in team_games[team_games.game_id.map(gsea.season) <= 2019].itertuples():
            act = active.get((r.game_id, r.posteam), set()); sw = gsea.loc[r.game_id]
            for pid in rotation(r):
                obs.append((pkey(status.get((int(sw.season), int(sw.week), r.posteam, pid)), pid, r), pid in act))
        o = pd.DataFrame(obs, columns=['st', 'played'])
        est = o.groupby(o.st.fillna('none')).played.agg(['mean', 'size'])
        for k_, v_ in est['mean'].items(): pplay[None if k_ == 'none' else k_] = float(v_)
        print('P(plays) by report status, 2016-19 rotation players:', est.round(3).to_dict('index'), flush=True)
    if status:
        from types import SimpleNamespace as NS
        prev_of = {(t, g): p for t, g, p in zip(team_games.posteam, team_games.game_id, team_games.prev1)}
        df['_p_play'] = [pplay[pkey(st, pid, NS(prev1=prev_of.get((t, g)), posteam=t))] for st, pid, t, g in zip(df._status, df.pid, df.posteam, df.game_id)]
    else:
        df['_p_play'] = 1.0
    rows = []; exp_rows = []
    for r in team_games.itertuples():
        act = active.get((r.game_id, r.posteam), set())
        seen = rotation(r)
        if status:
            sw = gsea.loc[r.game_id]
            pp = {pid: pplay[pkey(status.get((int(sw.season), int(sw.week), r.posteam, pid)), pid, r)] for pid in seen}
            w = {pid: 1 - p for pid, p in pp.items()}               # expected share missing, pre-game
            exp_rows.append((r.game_id, r.posteam, sum(w[x.pid] * x.tsh for x in seen.values()), sum(w[x.pid] * x.csh for x in seen.values()),
                             {pid: pp[pid] for pid in seen}, {pid: (x.tsh, x.csh) for pid, x in seen.items()}))
        else:
            w = {pid: float(pid not in act) for pid in seen}
        miss = [x for pid, x in seen.items()]
        rows.append(dict(game_id=r.game_id, posteam=r.posteam,
                         miss_tsh=sum(w[x.pid] * x.tsh for x in miss), miss_csh=sum(w[x.pid] * x.csh for x in miss),
                         miss_tsh_WR=sum(w[x.pid] * x.tsh for x in miss if x.position == 'WR'), miss_tsh_TE=sum(w[x.pid] * x.tsh for x in miss if x.position == 'TE'),
                         miss_tsh_RB=sum(w[x.pid] * x.tsh for x in miss if x.position == 'RB'),
                         miss_csh_RB=sum(w[x.pid] * x.csh for x in miss if x.position == 'RB'), miss_qb=sum(w[x.pid] * x.psh for x in miss if x.position == 'QB'),
                         miss_tsh_actual=sum(x.tsh for pid, x in seen.items() if pid not in act)))
    av = pd.DataFrame(rows)
    tg = tg.merge(av, on=['game_id', 'posteam'], how='left')

    df = df.merge(tg.drop(columns=['season', 'game_date', 'opp', 'team_pass_att', 'team_carries', 'team_dropbacks', 'team_score', 'implied']),
                  on=['game_id', 'posteam'], how='left')
    # share of today's active group: my share / sum of active teammates' shares (missing history -> position prior)
    pri_sh = df[df.season <= 2019].groupby('position')[['tsh_l', 'csh_l']].median()
    for c in ['tsh', 'csh']:
        v = df[f'{c}_l'].fillna(df.position.map(pri_sh[f'{c}_l']) * 0.5)
        df[f'{c}_fill'] = v
        if status:
            # expected active group before the game: today's players weighted by their P(plays) (from the report), plus
            # rotation players who did not play, weighted the same way
            j = 0 if c == 'tsh' else 1
            extra = {(g, t): sum(pp[pid] * sh[pid][j] for pid in pp if pid not in active.get((g, t), set())) for g, t, _, _, pp, sh in exp_rows}
            base = (v * df._p_play).groupby([df.game_id, df.posteam]).transform('sum')
            df[f'{c}_act_sum'] = base + np.array([extra.get((g, t), 0.0) for g, t in zip(df.game_id, df.posteam)])
        else:
            df[f'{c}_act_sum'] = v.groupby([df.game_id, df.posteam]).transform('sum')
        df[f'{c}_norm'] = v / df[f'{c}_act_sum'].clip(lower=0.05)
    df['is_qb_start'] = (df.pid == df.qb_pid).astype(int)
    df['dome'] = df.roof.isin(['dome', 'closed']).astype(int)
    df['wind'] = df.wind.where(df.dome == 0, 0).fillna(df.wind.median())
    df['temp'] = df.temp.where(df.dome == 0, 70).fillna(60)
    df['playoff'] = (df.season_type == 'POST').astype(int)
    df = df.drop(columns=['_t', 'team_pass_att_', 'team_carries_', 'games_one', '_status', '_p_play'], errors='ignore')

    return df


def add_tracking(df, ngs_dir):
    """NFL-3 (2026-10-10): pre-game tracking abilities and team-changer status, used by the team-changer target model.
    ngs_dir holds nflverse ngs_rushing.csv.gz / ngs_receiving.csv.gz (weekly rows; missing files -> zeros)."""
    import os
    df = df.sort_values(['game_date', 'game_id', 'posteam', 'pid']).reset_index(drop=True)
    cols = ['rush_yards_over_expected', 'rush_attempts', 'yacoe_tot', 'ngs_rec']
    if ngs_dir and os.path.exists(f'{ngs_dir}/ngs_rushing.csv.gz'):
        ru = pd.read_csv(f'{ngs_dir}/ngs_rushing.csv.gz'); ru = ru[ru.week > 0]
        ru = ru.groupby(['player_gsis_id', 'season', 'week'], as_index=False)[['rush_yards_over_expected', 'rush_attempts']].sum()
        df = df.merge(ru.rename(columns={'player_gsis_id': 'pid'}), on=['pid', 'season', 'week'], how='left')
    if ngs_dir and os.path.exists(f'{ngs_dir}/ngs_receiving.csv.gz'):
        rc = pd.read_csv(f'{ngs_dir}/ngs_receiving.csv.gz'); rc = rc[rc.week > 0].copy()
        rc['yacoe_tot'] = rc.avg_yac_above_expectation * rc.receptions
        rc = rc.groupby(['player_gsis_id', 'season', 'week'], as_index=False)[['yacoe_tot', 'receptions']].sum().rename(columns={'receptions': 'ngs_rec'})
        df = df.merge(rc.rename(columns={'player_gsis_id': 'pid'}), on=['pid', 'season', 'week'], how='left')
    for c in cols: df[c] = df[c].fillna(0.0) if c in df else 0.0
    S, _ = decayed(df, 'pid', cols, 8)
    df['ryoe_pa'] = S.rush_yards_over_expected / (S.rush_attempts + 60)     # rush yards over expected per attempt, shrunk to 0
    df['yacoe_pr'] = S.yacoe_tot / (S.ngs_rec + 40)                          # YAC over expected per catch, shrunk to 0
    prev_team = df.groupby('pid').posteam.shift(1)
    moved = prev_team.notna() & (prev_team != df.posteam)
    stint = moved.groupby(df.pid).cumsum()
    df['games_with_team'] = df.groupby([df.pid, stint]).cumcount()
    df['changer'] = ((stint > 0) & (df.games_with_team < 8)).astype(int)     # first 8 games with a new team
    return df.drop(columns=cols)


if __name__ == '__main__':
    M = sys.argv[1]
    pg = pd.read_parquet(f'{M}/data/player_games.parquet')
    inj = load_injuries(sys.argv[2], range(2016, 2027)) if len(sys.argv) > 2 else None
    if inj is None: print('no injury reports given: availability uses who actually played (the pre-audit backtest)')
    df = build(pg, inj)
    df = add_tracking(df, sys.argv[3] if len(sys.argv) > 3 else (sys.argv[2] if len(sys.argv) > 2 else None))
    df.to_parquet(f'{M}/data/features.parquet', index=False)
    print(df.shape)
    print(df[['tsh_l', 'tsh_norm', 'csh_norm', 'miss_tsh', 'd_rec_yds_WR', 'd_tsh_TE', 'qb_ypa_rel', 'catch_r', 'ypc', 'ypa']].describe().T.round(3))
