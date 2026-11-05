"""NBA player single-game stat model: projected minutes x adjusted per-minute rate, with a full distribution.

    minutes  : gradient-boosted regression on minutes history, starter rate, teammate/opponent absences, spread, rest
    rate     : Poisson GLM per stat, offset = log(minutes) + log(player's shrunk recent per-minute rate), adjusted for
               opponent (overall and by position), teammates missing or returning (lineup layer), market game
               environment (implied team total vs the team's normal scoring), spread, home, rest, playoffs
    spread   : negative binomial given minutes, integrated over minutes uncertainty

normalize_minutes(pred, team_key): when the full expected lineup is known (game-day use), pull each team's projected
minutes halfway toward 241 (regulation plus average overtime), which cut minutes and points error slightly in backtests.

fit(df) -> Model; Model.predict(df) -> DataFrame with min_proj, min_sd and mu_<stat>; Model.prob_over(row, stat, line)

2026-10-09 audit (P2): count noise is Var = phi*mu + a*mu^2, fit per stat and minutes band (<20, 20-30, 30+), instead of one
NB2 alpha per stat. phi > 1 captures that points come in 2s and 3s (a 20-point mean has far more spread than a Poisson
count), which the old single alpha could not, so P(over) had to be shrunk hard afterwards (temperature slope 0.67).
Each prediction uses an NB with alpha = (phi - 1)/mu + a, which matches that mean and variance.
"""
import os, json, numpy as np, pandas as pd, statsmodels.api as sm
from scipy import stats as st
from sklearn.ensemble import HistGradientBoostingRegressor

STATS = ['pts', 'reb', 'ast', 'fg3m', 'stl', 'blk', 'tov', 'pra', 'oreb', 'dreb', 'fga', 'fta']
MIN_X = ['ms', 'ml', 'last_min', 'st_rate', 'days_off', 'n_hist', 'sea_gp', 'pres_self', 'd_min', 'd_min_grp', 'od_min',
         'abs_spread', 'team_spread', 'rest', 'playoff', 'is_home', 'r_usg', 'r_pts', 'month',
         'g_back', 'miss_prev', 'ramp', 'm3', 'm_sd10', 'm_trend', 'rot_depth']   # NBA-2 (2026-10-11): minutes_ext.py
# lineup layer: ex_* = teammates' missing (or returning) share of team output beyond their share of minutes
# (team-wide, from availability); lu_* = this player's own on/off split with each missing teammate (play-by-play)
SHARE_FOR = {'pts': ['ex_usg', 'lu_pts'], 'fga': ['ex_usg', 'lu_usg'], 'fta': ['ex_usg', 'lu_usg'], 'tov': ['ex_usg', 'lu_usg'],
             'reb': ['ex_reb', 'lu_reb'], 'oreb': ['ex_reb', 'lu_reb'], 'dreb': ['ex_reb', 'lu_reb'], 'ast': ['ex_ast', 'ex_usg', 'lu_ast'],
             'fg3m': ['ex_fg3a', 'lu_fg3a'], 'stl': ['d_min'], 'blk': ['ex_reb', 'lu_reb'],
             'pra': ['ex_usg', 'ex_reb', 'ex_ast', 'lu_pts', 'lu_reb', 'lu_ast']}
RATE_OF = {'pra': 'pra'}

def prep(df):
    df = df.copy()
    df['abs_spread'] = df.team_spread.abs()
    df['month'] = df.date.dt.month
    df['days_off'] = df.days_off.clip(upper=60)
    df['b2b'] = (df.rest == 1).astype(float)
    df['env_team'] = np.log(df.team_implied / df.team_pts_ewma).clip(-0.3, 0.3)
    df['env_game'] = np.log(df.total_line / (df.team_pts_ewma + df.opp_pts_ewma)).clip(-0.3, 0.3)
    df['blowout'] = (df.abs_spread / 10) ** 2
    for x in ['usg', 'pts', 'reb', 'ast', 'fg3a']: df[f'ex_{x}'] = df[f'd_{x}'] - df.d_min
    for x in ['usg', 'pts', 'reb', 'ast', 'fg3a']: df[f'lu_{x}'] = df[f'lu_{x}'].fillna(0.0)
    return df

def usable(df):
    return df.played & df.ml.notna() & (df.n_hist >= 5) & df.total_line.notna() & df.team_pts_ewma.notna() & df.opp_pts_ewma.notna()

def rate_X(stat):
    return ['def_' + stat, 'defg_' + stat] + SHARE_FOR[stat] + ['od_reb' if stat in ('reb', 'oreb', 'dreb', 'blk', 'pra') else 'od_pts',
            'env_team', 'env_game', 'team_spread', 'blowout', 'is_home', 'b2b', 'playoff', 'st_rate']

def design(df, stat):
    X = df[rate_X(stat)].astype(float).copy()
    X['def_' + stat] = np.log(X['def_' + stat]); X['defg_' + stat] = np.log(X['defg_' + stat])
    return sm.add_constant(X, has_constant='add')

class Model:
    def predict(self, df):
        df = prep(df)
        out = pd.DataFrame(index=df.index)
        out['min_proj'] = np.clip(self.mgb.predict(df[MIN_X]), 1, 48)
        out['min_sd'] = np.clip(self.sgb.predict(df[MIN_X].assign(min_proj=out.min_proj)), 2, 15) * 1.25
        for s in STATS:
            X = design(df, s); off = np.log(df['r_' + s].clip(lower=1e-4))
            out['rate_' + s] = np.exp(X.values @ self.glm[s].params.values + off)
            out['mu_' + s] = out['rate_' + s] * out.min_proj
        return out

    def dist(self, row, stat, n=25):
        """Probability mass over 0..max for one prediction row (from predict), mixing over minutes."""
        q = (np.arange(n) + 0.5) / n
        mins = np.clip(st.norm.ppf(q, row['min_proj'], row['min_sd']), 0, 53)
        mus = row['rate_' + stat] * mins
        disp = getattr(self, 'disp', None)
        if disp: phi, a0 = disp[stat][band_of(row['min_proj'])]
        else: phi, a0 = 1.0, self.alpha[stat]
        k = np.arange(0, int(row['mu_' + stat] * 4 + 25))
        pm = np.zeros(len(k))
        for mu in mus:
            if mu <= 0: pm[0] += 1; continue
            a = max(1e-3, (phi - 1) / mu + a0)
            r = 1 / a; pm += st.nbinom.pmf(k, r, r / (r + mu))
        return k, pm / n

    def prob_over(self, row, stat, line, calibrated=True):
        k, pm = self.dist(row, stat)
        p = float(np.clip(pm[k > line].sum(), 1e-4, 1 - 1e-4))
        c = getattr(self, 'temp', {}).get(stat) if calibrated else None
        if c:  # out-of-sample temperature from calibrate.py
            z = c['intercept'] + c['slope'] * np.log(p / (1 - p)); p = 1 / (1 + np.exp(-z))
        return p

    def quantiles(self, row, stat, qs=(0.1, 0.25, 0.5, 0.75, 0.9)):
        k, pm = self.dist(row, stat); c = np.cumsum(pm)
        return [int(k[min(np.searchsorted(c, q), len(k) - 1)]) for q in qs]

BANDS = (20, 30)
def band_of(m): return 0 if m < BANDS[0] else (1 if m < BANDS[1] else 2)

def normalize_minutes(pred, team_key, target=241.0, strength=0.5):
    f = (target / pred.groupby(team_key).min_proj.transform('sum')) ** strength
    pred = pred.copy(); pred['min_proj'] *= f
    for c in [c for c in pred if c.startswith('mu_')]: pred[c] *= f
    return pred

CAL = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'calibration.json')

def fit(train):
    train = prep(train)
    m = Model()
    m.temp = json.load(open(CAL)) if os.path.exists(CAL) else {}
    m.mgb = HistGradientBoostingRegressor(max_iter=400, learning_rate=0.05, max_leaf_nodes=31, min_samples_leaf=100, random_state=0)
    m.mgb.fit(train[MIN_X], train['min'])
    res = (train['min'] - m.mgb.predict(train[MIN_X])).abs()
    m.sgb = HistGradientBoostingRegressor(max_iter=200, learning_rate=0.05, max_leaf_nodes=15, min_samples_leaf=200, random_state=0)
    m.sgb.fit(train[MIN_X].assign(min_proj=np.clip(m.mgb.predict(train[MIN_X]), 1, 48)), res)
    m.glm, m.alpha, m.disp = {}, {}, {}
    band = np.array([band_of(x) for x in train['min']])
    for s in STATS:
        X = design(train, s); off = np.log(train['min']) + np.log(train['r_' + s].clip(lower=1e-4))
        g = sm.GLM(train[s].values, X, family=sm.families.Poisson(), offset=off).fit()
        m.glm[s] = g
        mu = g.predict(X, offset=off)
        # NB2 dispersion by method of moments: Var = mu + a mu^2
        m.alpha[s] = max(1e-3, float(((train[s] - mu) ** 2 - mu).sum() / (mu ** 2).sum()))
        # Var = phi*mu + a*mu^2 by minutes band (least squares of squared residuals on mu, mu^2)
        m.disp[s] = {}
        for bnd in range(3):
            sel = band == bnd; mu_b = np.asarray(mu)[sel]; r2 = (train[s].values[sel] - mu_b) ** 2
            phi, a = np.linalg.lstsq(np.column_stack([mu_b, mu_b ** 2]), r2, rcond=None)[0]
            m.disp[s][bnd] = (max(float(phi), 0.5), max(float(a), 0.0))
        g.remove_data()
    return m
