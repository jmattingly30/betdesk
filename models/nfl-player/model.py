"""NFL player single-game model: team volume x player share (opportunity) x shrunk efficiency with matchup terms,
then a play-level simulation for the full distribution.

Layers
1. Team volume  - Poisson GLMs for team pass attempts and carries (team tendency, opponent, spread, total, weather, QB change).
2. Opportunity  - LightGBM (Poisson loss) for each player's targets and carries, and the starting QB's pass attempts, on the
                  team volume plus recent shares, snap shares, share of today's active group and missing teammates.
3. Efficiency   - Poisson GLMs with the opportunity as offset: receptions, receiving yards, rushing yards, completions,
                  passing yards, passing TDs, touchdowns, each on the player's shrunk rate plus opponent, QB and game-total terms.
4. Distribution - negative binomial opportunity, binomial catches/completions, yards summed from real per-play yardage
                  (scaled to the player's projected yards per play), Poisson TDs."""
import numpy as np, pandas as pd, statsmodels.api as sm, lightgbm as lgb

POS = {'QB': 0, 'RB': 1, 'WR': 2, 'TE': 3}


def _log(x, lo=1e-3):
    return np.log(np.clip(np.asarray(x, float), lo, None))


def add_derived(df):
    df = df.copy()
    pos = df.position
    for a in ['rec_yds', 'rec']:
        col = np.select([pos == 'WR', pos == 'TE', pos == 'RB'], [df[f'd_{a}_WR'], df[f'd_{a}_TE'], df[f'd_{a}_RB']], 1.0)
        df[f'dp_{a}'] = col
    df['d_tsh_pos'] = np.select([pos == 'WR', pos == 'TE', pos == 'RB'], [df.d_tsh_WR, df.d_tsh_TE, df.d_tsh_RB], 0.0)
    df['miss_tsh_pos'] = np.select([pos == 'WR', pos == 'TE', pos == 'RB'], [df.miss_tsh_WR, df.miss_tsh_TE, df.miss_tsh_RB], 0.0)
    df['pos_code'] = pos.map(POS)
    df['implied_rel'] = df.implied / df.tm_implied.fillna(df.implied).clip(lower=10)
    df['total_line'] = df.total_line.fillna(44)
    df['spread'] = df.spread.fillna(0)
    df['implied'] = df.implied.fillna(22)
    df['qb_ypa_rel'] = df.qb_ypa_rel.fillna(1.0)
    df['qb_cmp'] = df.qb_cmp.fillna(0.62)
    df['qb_ypa'] = df.qb_ypa.fillna(7.0)
    for c in ['l_lg_pa', 'l_lg_ypa']: df[c] = df[c].fillna(0.0) if c in df else 0.0
    return df


# ------------------------------------------------------------ layer 1: team volume
TEAM_X = ['l_tm_pa', 'l_tm_plays', 'l_d_pa', 'l_d_ca', 'spread', 'total_line', 'home', 'wind', 'dome', 'playoff', 'qb_changed', 'l_qb_ypa', 'miss_qb']
# audit F2: league passing fell in 2024-25 and a fit on older seasons kept projecting the old level (+6 passing yards per
# QB-game). A league-environment feature (l_lg_pa, built in features.py) was tried and made it worse (its coefficient is
# confounded with the team's own recent tendency). What works is weighting recent seasons more in the volume and
# efficiency GLMs (half-life SEASON_HL seasons) and refitting every season.
SEASON_HL = 2.0
def season_w(df):
    return 0.5 ** ((df.season.max() - df.season) / SEASON_HL)


def team_frame(df):
    t = df.drop_duplicates(['game_id', 'posteam']).copy()
    t['l_tm_pa'] = _log(t.tm_team_pass_att.fillna(34))
    t['l_tm_ca'] = _log(t.tm_team_carries.fillna(26))
    t['l_tm_plays'] = _log(t.tm_plays.fillna(60))
    t['l_d_pa'] = _log(t.d_pa.fillna(34))
    t['l_d_ca'] = _log(t.d_ca.fillna(26))
    t['l_qb_ypa'] = _log(t.qb_ypa)
    t['miss_qb'] = t.miss_qb.fillna(0)
    return t


def fit_team(df):
    t = team_frame(df)
    out = {}
    for y, extra in [('team_pass_att', 'l_tm_pa'), ('team_carries', 'l_tm_ca')]:
        X = sm.add_constant(t[TEAM_X + ['l_tm_ca']].fillna(0))
        out[y] = sm.GLM(t[y], X, family=sm.families.Poisson(), var_weights=season_w(t)).fit()
    return out


def predict_team(models, df):
    t = team_frame(df)
    X = sm.add_constant(t[TEAM_X + ['l_tm_ca']].fillna(0), has_constant='add')
    t['pa_hat'] = models['team_pass_att'].predict(X)
    t['ca_hat'] = models['team_carries'].predict(X)
    return df.merge(t[['game_id', 'posteam', 'pa_hat', 'ca_hat']], on=['game_id', 'posteam'], how='left')


# ------------------------------------------------------------ layer 2: opportunity
OPP_X = ['l_vol', 'tsh_s', 'tsh_l', 'last_tsh', 'tsh_norm', 'tsh_act_sum', 'csh_s', 'csh_l', 'last_csh', 'csh_norm', 'csh_act_sum',
         'psh_s', 'psh_l', 'snap_s', 'snap_l', 'last_snap', 'miss_tsh', 'miss_tsh_pos', 'miss_csh', 'miss_csh_RB', 'pos_code',
         'games_prior', 'games_season', 'rookie', 'd_tsh_pos', 'spread', 'total_line', 'adot', 'air_sh', 'rzt_sh', 'rzc_sh',
         'is_qb_start', 'n_l', 'days_rest', 'qb_changed', 'playoff', 'scr_rate', 'des_sh', 'd_scr']   # audit F4
LGB = dict(objective='poisson', learning_rate=0.03, num_leaves=31, min_data_in_leaf=80, feature_fraction=0.8,
           bagging_fraction=0.8, bagging_freq=1, lambda_l2=5.0, verbose=-1)
OPP = {'targets': ('pa_hat', lambda d: d.position != 'QB', 700),
       'carries': ('ca_hat', lambda d: d.position.isin(['RB', 'QB', 'WR']), 700),
       'pass_att': ('pa_hat', lambda d: d.is_qb_start == 1, 500)}


# NFL-3 (2026-10-10): a second target model with tracking abilities and team-changer status, used ONLY for players in
# their first 8 games with a new team. Out of sample 2024-25 it cut target error 2.2% and receiving-yard error 1.7% for
# those players (t ~5); for everyone else it did not help, so the original model stays in charge there.
OPP_X_CHG = OPP_X + ['ryoe_pa', 'yacoe_pr', 'changer', 'games_with_team']

def fit_changer_targets(df):
    d = df[OPP['targets'][1](df)].copy(); d['l_vol'] = _log(d.pa_hat)
    return lgb.train(LGB, lgb.Dataset(d[OPP_X_CHG], d.targets, categorical_feature=['pos_code'], free_raw_data=False), num_boost_round=OPP['targets'][2])


# NFL-4 (2026-10-11): extra features per opportunity model (built by scheme_features.add_scheme). backtest_nfl4.py round 2:
# play-caller priors (option 1) + expected personnel / game plan (option 3) + coverage-mix target ratios (part of option 2)
# in the TARGETS model only. Out of sample 2024-25: new play-caller teams games 1-6 target error -1.0% (t 3.1), callers who
# moved teams -1.5% (t 3.6), players whose expected personnel shifts -1.2% (t 2.5), everyone else unchanged. Box-count
# carries, opponent look rates and red-zone terms (option 4) did not help (carries got worse), so they are not used.
NFL4_TARGETS = ['cl_new', 'cl_mover', 'cl_lr_pos', 'cl_d_p11', 'cl_d_p12', 'cl_d_pa', 'cl_d_proe', 'cl_d_rz_pass', 'cl_lr_rz',
                'on_exp', 'on_ratio', 'gp_d11', 'ms_mz', 'ms_shell', 'ms_db', 'ms_pr']
OPP_EXTRA = {'targets': NFL4_TARGETS}

def fit_opportunity(df):
    models = {}
    for y, (vol, mask, rounds) in OPP.items():
        d = df[mask(df)].copy()
        d['l_vol'] = _log(d[vol]); xs = OPP_X + [c for c in OPP_EXTRA.get(y, []) if c in d]   # skipped if add_scheme was not run
        ds = lgb.Dataset(d[xs], d[y], categorical_feature=['pos_code'], free_raw_data=False)
        models[y] = lgb.train(LGB, ds, num_boost_round=rounds)
    return models


def predict_opportunity(models, df):
    df = df.copy()
    for y, (vol, mask, _) in OPP.items():
        m = mask(df)
        xs = models[y].feature_name()           # the features the model was fit with
        X = df.loc[m, [c for c in xs if c != 'l_vol']].copy() if all(c in df for c in xs if c != 'l_vol') else \
            df.loc[m].reindex(columns=[c for c in xs if c != 'l_vol']).copy()
        X['l_vol'] = _log(df.loc[m, vol])
        df[f'{y}_hat'] = 0.0
        if m.any():
            df.loc[m, f'{y}_hat'] = models[y].predict(X[xs])
    return df


# ------------------------------------------------------------ layer 3: efficiency (offset GLMs)
def eff_specs(df):
    """name -> (target, opportunity column used as offset, feature columns, row mask)."""
    rec_mask = df.position.isin(['WR', 'TE', 'RB'])
    return {
        'rec': ('rec', 'targets', {'l_rate': _log(df.catch_r), 'l_def': _log(df.dp_rec), 'l_qb': _log(df.qb_cmp / 0.63),
                                    'adot': df.adot.fillna(8).clip(0, 25) / 10}, rec_mask),
        'rec_yds': ('rec_yds', 'targets', {'l_rate': _log(df.ypt), 'l_def': _log(df.dp_rec_yds), 'l_qb': _log(df.qb_ypa_rel),
                                            'l_imp': _log(df.implied_rel), 'wind': df.wind.clip(0, 30) / 10, 'dome': df.dome}, rec_mask),
        'rush_yds': ('rush_yds', 'carries', {'l_rate': _log(df.ypc), 'l_def': _log(df.d_rush_yds), 'l_imp': _log(df.implied_rel),
                                              'spread': df.spread / 7, 'is_qb': (df.position == 'QB').astype(float),
                                              'qb_scr': (df.position == 'QB') * df.scr_rate.fillna(0.035) * 10}, df.position.isin(['RB', 'QB', 'WR'])),
        'cmp': ('cmp', 'pass_att', {'l_rate': _log(df.cmp_r), 'l_def': _log(df.d_cmp), 'wind': df.wind.clip(0, 30) / 10}, df.is_qb_start == 1),
        'pass_yds': ('pass_yds', 'pass_att', {'l_rate': _log(df.ypa), 'l_def': _log(df.d_pass_yds), 'l_imp': _log(df.implied_rel),
                                               'wind': df.wind.clip(0, 30) / 10, 'dome': df.dome}, df.is_qb_start == 1),
        'pass_td': ('pass_td', 'pass_att', {'l_rate': _log(df.pass_td_r), 'l_def': _log(df.d_pass_td), 'l_imp': _log(df.implied / 22)}, df.is_qb_start == 1),
        'td': ('td', 'td_opp', {'l_def': _log(df.d_td), 'l_imp': _log(df.implied / 22), 'l_rz': _log(df.rzt_sh.fillna(0) + df.rzc_sh.fillna(0) + 0.01),
                                'is_qb': (df.position == 'QB').astype(float)}, df.position.isin(['WR', 'TE', 'RB', 'QB'])),
    }


def _td_opp(df, tcol, ccol):
    return df[tcol] * df.rec_td_r + df[ccol] * df.rush_td_r


def fit_efficiency(df):
    df = df.copy()
    df['td_opp'] = _td_opp(df, 'targets', 'carries')
    models = {}
    for name, (y, opp, feats, mask) in eff_specs(df).items():
        m = mask & (df[opp] > 0)
        X = sm.add_constant(pd.DataFrame({k: np.asarray(v)[m.to_numpy()] for k, v in feats.items()}))
        models[name] = sm.GLM(df.loc[m, y].to_numpy(), X, family=sm.families.Poisson(), offset=_log(df.loc[m, opp]),
                              var_weights=season_w(df.loc[m]).to_numpy()).fit()
    return models


def predict_efficiency(models, df):
    df = df.copy()
    df['td_opp_hat'] = _td_opp(df, 'targets_hat', 'carries_hat')
    opp_hat = {'targets': 'targets_hat', 'carries': 'carries_hat', 'pass_att': 'pass_att_hat', 'td_opp': 'td_opp_hat'}
    for name, (y, opp, feats, mask) in eff_specs(df).items():
        X = sm.add_constant(pd.DataFrame(feats), has_constant='add')[models[name].params.index]
        rate = np.exp(X.to_numpy() @ models[name].params.to_numpy())
        df[f'{name}_hat'] = np.where(mask, rate * df[opp_hat[opp]], 0.0)
    return df


# ------------------------------------------------------------ full fit / predict
def fit(df):
    df = add_derived(df)
    team = fit_team(df)
    df = predict_team(team, df)
    opp = fit_opportunity(df)
    eff = fit_efficiency(df)
    out = dict(team=team, opp=opp, eff=eff)
    if 'changer' in df: out['opp_chg'] = fit_changer_targets(df)
    return out


def predict(model, df):
    df = add_derived(df)
    df = predict_team(model['team'], df)
    df = predict_opportunity(model['opp'], df)
    if 'opp_chg' in model and 'changer' in df:
        m = (df.changer == 1) & OPP['targets'][1](df)
        if m.any():
            X = df.loc[m, [c for c in OPP_X_CHG if c != 'l_vol']].copy(); X['l_vol'] = _log(df.loc[m, 'pa_hat'])
            df.loc[m, 'targets_hat'] = model['opp_chg'].predict(X[OPP_X_CHG])
    df = predict_efficiency(model['eff'], df)
    return df


# ------------------------------------------------------------ distribution by simulation
STATS = ['targets', 'rec', 'rec_yds', 'carries', 'rush_yds', 'pass_att', 'cmp', 'pass_yds', 'pass_td', 'td', 'rush_rec_yds']


def nb_draw(rng, mu, size, n):
    mu = np.maximum(mu, 1e-6)[:, None]
    p = size / (size + mu)
    return rng.negative_binomial(size, np.broadcast_to(p, (len(mu), n)))


def sum_pool(rng, counts, pool, scale):
    """counts: (rows, sims) ints; pool: 1-d array of real per-play yards; scale per row. Sum of `count` draws."""
    kmax = int(counts.max()) if counts.size else 0
    out = np.zeros(counts.shape)
    for k in range(kmax):
        live = counts > k
        if not live.any():
            break
        draws = pool[rng.integers(0, len(pool), size=live.sum())]
        out[live] += draws
    return out * scale[:, None]


def simulate(df, pools, disp, n=1000, seed=0):
    """Returns dict stat -> (rows, n) simulated outcomes. disp: NB size for targets/carries/pass_att."""
    rng = np.random.default_rng(seed)
    res = {}
    tg = nb_draw(rng, df.targets_hat.to_numpy(), disp['targets'], n)
    p_c = np.clip(df.rec_hat / df.targets_hat.clip(lower=1e-6), 0.05, 0.97).to_numpy()
    rec = rng.binomial(tg, p_c[:, None])
    res['targets'], res['rec'] = tg, rec
    pos = df.position.to_numpy()
    ry = np.zeros(rec.shape)
    for p_ in ['WR', 'TE', 'RB']:
        m = pos == p_
        if m.any():
            pool = pools[('catch', p_)]
            ypr = (df.rec_yds_hat / df.rec_hat.clip(lower=1e-6)).to_numpy()[m]
            ry[m] = sum_pool(rng, rec[m], pool, ypr / pool.mean())
    res['rec_yds'] = ry
    ca = nb_draw(rng, df.carries_hat.to_numpy(), disp['carries'], n)
    res['carries'] = ca
    ru = np.zeros(ca.shape)
    for p_ in ['RB', 'QB', 'WR']:
        m = pos == p_
        if m.any():
            pool = pools[('carry', p_)]
            ypc = (df.rush_yds_hat / df.carries_hat.clip(lower=1e-6)).to_numpy()[m]
            ru[m] = sum_pool(rng, ca[m], pool, ypc / pool.mean())
    res['rush_yds'] = ru
    res['rush_rec_yds'] = ru + ry
    pa = nb_draw(rng, df.pass_att_hat.to_numpy(), disp['pass_att'], n)
    p_cmp = np.clip(df.cmp_hat / df.pass_att_hat.clip(lower=1e-6), 0.3, 0.85).to_numpy()
    cm = rng.binomial(pa, p_cmp[:, None])
    pool = pools[('catch', 'ALL')]
    ypc = (df.pass_yds_hat / df.cmp_hat.clip(lower=1e-6)).to_numpy()
    res['pass_att'], res['cmp'] = pa, cm
    res['pass_yds'] = sum_pool(rng, cm, pool, ypc / pool.mean())
    res['pass_td'] = rng.poisson(np.broadcast_to(df.pass_td_hat.to_numpy()[:, None], (len(df), n)))
    res['td'] = rng.poisson(np.broadcast_to(df.td_hat.to_numpy()[:, None], (len(df), n)))
    return res


def load_pools(path, max_season=None):
    p = pd.read_parquet(path)
    if max_season is not None:
        p = p[p.season <= max_season]
    pools = {(k, pos): g.yds.to_numpy(float) for (k, pos), g in p.groupby(['kind', 'pos'])}
    pools[('catch', 'ALL')] = p[p.kind == 'catch'].yds.to_numpy(float)
    return pools


def fit_dispersion(df):
    """Method-of-moments NB size for opportunity counts, from predicted means (out of sample rows)."""
    out = {}
    for y in ['targets', 'carries', 'pass_att']:
        d = df[df[f'{y}_hat'] > 0.5]
        mu, obs = d[f'{y}_hat'], d[y]
        excess = ((obs - mu) ** 2 - mu).sum() / (mu ** 2).sum()
        out[y] = float(1 / max(excess, 1e-3))
    return out
