"""NHL player single-game stat model: LightGBM (Poisson loss) means for TOI-driven counts, negative binomial
distribution with a per-stat dispersion, temperature calibration of P(over).

    python3 model.py .           backtest: fit 2023-24 -> calibrate 2024-25; fit 2023-25 -> test 2025-26
                                 then refit on everything -> data/model.pkl
Stats: skaters sog, points, goals, assists, blk, hits, pim; goalies saves (starting goalie, conditional on starting).
Every feature comes from games before the one predicted (build_dataset.py).
"""
import sys, json, pickle
import numpy as np, pandas as pd, lightgbm as lgb
from scipy.stats import nbinom, poisson
from scipy.optimize import minimize_scalar

base = sys.argv[1] if len(sys.argv) > 1 else '.'
SK_STATS = ['sog', 'points', 'goals', 'assists', 'blk', 'hits']
TEAM = ['home', 'rest', 'o_rest', 'sf_e', 'sa_e', 'gf_e', 'ga_e', 'o_sf_e', 'o_sa_e', 'o_gf_e', 'o_ga_e', 'blk_t_e', 'o_hits_a_e', 'playoff']
SK_FEATS = ['pos_code', 'n_prev', 'days_since', 'toi_e3', 'toi_e10', 'toi_last', 'toi_l10', 'toi_szn'] + TEAM + \
    [f'{s}_{k}' for s in ['sog', 'goals', 'assists', 'points', 'blk', 'hits', 'pim', 'power_play_goals'] for k in ('60', 'e5', 'l10', 'raw60')]
G_FEATS = ['n_prev', 'days_since', 'svpct', 'g_starts_prev', 'saves_l10', 'shots_against_l10', 'saves_szn'] + TEAM
LINES = {'sog': [0.5, 1.5, 2.5, 3.5, 4.5], 'points': [0.5, 1.5], 'goals': [0.5], 'assists': [0.5, 1.5], 'blk': [0.5, 1.5, 2.5],
         'hits': [0.5, 1.5, 2.5, 3.5], 'pim': [0.5, 1.5], 'saves': [20.5, 23.5, 25.5, 27.5, 30.5]}
PARAMS = dict(objective='poisson', learning_rate=0.03, num_leaves=15, min_data_in_leaf=200, feature_fraction=0.8,
              bagging_fraction=0.8, bagging_freq=1, lambda_l2=5, verbose=-1)
ROUNDS = 600


def frames(d):
    sk = d[(d.pos != 'G') & (d.n_prev >= 3)].copy()
    gl = d[(d.pos == 'G') & d.started_g & (d.n_prev >= 1)].copy()
    return sk, gl


def fit_means(tr_sk, tr_g):
    models = {}
    for s in SK_STATS:
        models[s] = lgb.train(PARAMS, lgb.Dataset(tr_sk[SK_FEATS], tr_sk[s]), ROUNDS)
    models['toi'] = lgb.train(dict(PARAMS, objective='regression'), lgb.Dataset(tr_sk[SK_FEATS], tr_sk.toi), ROUNDS)
    models['saves'] = lgb.train(dict(PARAMS, min_data_in_leaf=60), lgb.Dataset(tr_g[G_FEATS], tr_g.saves), 400)
    return models


def predict(models, sk, gl):
    for s in SK_STATS + ['toi']:
        sk[f'pred_{s}'] = models[s].predict(sk[SK_FEATS])
    gl['pred_saves'] = models['saves'].predict(gl[G_FEATS])
    return sk, gl


def nb_p_over(mu, alpha, line):
    """P(X > line) for NB with mean mu, var = mu + alpha*mu^2 (alpha ~0 -> Poisson)."""
    mu = np.asarray(mu, float)
    if alpha < 1e-4:
        return poisson.sf(np.floor(line), mu)
    n = 1 / alpha; p = n / (n + mu)
    return nbinom.sf(np.floor(line), n, p)


def fit_alpha(y, mu):
    def nll(a):
        if a < 1e-4:
            return -poisson.logpmf(y, mu).sum()
        n = 1 / a; return -nbinom.logpmf(y, n, n / (n + mu)).sum()
    return minimize_scalar(nll, bounds=(0, 2), method='bounded').x


def temp(p, t):
    z = np.log(np.clip(p, 1e-6, 1 - 1e-6) / (1 - np.clip(p, 1e-6, 1 - 1e-6))); return 1 / (1 + np.exp(-z / t))


def fit_temp(ps, ys):
    def ll(t):
        q = np.clip(temp(ps, t), 1e-6, 1 - 1e-6); return -np.mean(ys * np.log(q) + (1 - ys) * np.log(1 - q))
    return minimize_scalar(ll, bounds=(0.5, 2.0), method='bounded').x


def over_table(df, stat, mu, alpha):
    rows = []
    for L in LINES[stat]:
        rows.append(pd.DataFrame({'stat': stat, 'line': L, 'p': nb_p_over(mu, alpha, L), 'y': (df[stat] > L).astype(int).to_numpy()}))
    return pd.concat(rows)


def calibrate(sk, gl):
    cal = {}
    for s in SK_STATS + ['saves']:
        df = gl if s == 'saves' else sk
        a = fit_alpha(df[s].to_numpy(), df[f'pred_{s}'].to_numpy())
        t = over_table(df, s, df[f'pred_{s}'], a)
        cal[s] = dict(alpha=round(float(a), 5), temp=round(float(fit_temp(t.p.to_numpy(), t.y.to_numpy())), 4))
    return cal


def score(sk, gl, cal, log):
    rows, crow = [], []
    for s in SK_STATS + ['saves']:
        df = gl if s == 'saves' else sk
        y = df[s]
        r = dict(stat=s, n=len(df), mean=y.mean(), mae_model=np.mean(np.abs(y - df[f'pred_{s}'])),
                 mae_last10=np.mean(np.abs(y - df[f'{s}_l10'].fillna(df[f'pred_{s}']))),
                 mae_season=np.mean(np.abs(y - df[f'{s}_szn'].fillna(df[f'{s}_l10']).fillna(df[f'pred_{s}']))))
        r['gain_vs_last10'] = 1 - r['mae_model'] / r['mae_last10']
        # log loss at common lines: model (NB + temperature) vs a Poisson on the last-10 average
        t = over_table(df, s, df[f'pred_{s}'], cal[s]['alpha']); t['p'] = temp(t.p.to_numpy(), cal[s]['temp'])
        b = over_table(df, s, df[f'{s}_l10'].fillna(df[f'pred_{s}']).clip(0.02), cal[s]['alpha'])
        ll = lambda p, yy: -np.mean(yy * np.log(np.clip(p, 1e-6, 1)) + (1 - yy) * np.log(np.clip(1 - p, 1e-6, 1)))
        r['logloss_model'] = ll(t.p.to_numpy(), t.y.to_numpy()); r['logloss_last10'] = ll(b.p.to_numpy(), b.y.to_numpy())
        rows.append(r)
        t['bucket'] = pd.cut(t.p, [0, .1, .2, .3, .4, .5, .6, .7, .8, .9, 1])
        c = t.groupby(['bucket'], observed=True).agg(n=('y', 'size'), pred=('p', 'mean'), actual=('y', 'mean')).reset_index(); c['stat'] = s
        crow.append(c)
    acc = pd.DataFrame(rows)
    return acc, pd.concat(crow)


if __name__ == '__main__':
    d = pd.read_parquet(f'{base}/data/player_games.parquet')
    sk, gl = frames(d)
    log = open(f'{base}/backtest_log.txt', 'w')
    def say(*a):
        s = ' '.join(str(x) for x in a); print(s); log.write(s + '\n')
    # stage 1: fit 2023-24, calibrate on 2024-25
    m1 = fit_means(sk[sk.season == 2023], gl[gl.season == 2023])
    c_sk, c_gl = predict(m1, sk[sk.season == 2024].copy(), gl[gl.season == 2024].copy())
    cal = calibrate(c_sk, c_gl); say('calibration (fit on 2024-25):', cal)
    # stage 2: fit 2023-25, test on 2025-26
    m2 = fit_means(sk[sk.season <= 2024], gl[gl.season <= 2024])
    t_sk, t_gl = predict(m2, sk[sk.season == 2025].copy(), gl[gl.season == 2025].copy())
    acc, calt = score(t_sk, t_gl, cal, log)
    pd.set_option('display.width', 200)
    say('\nTest 2025-26 (out of sample; skaters with 3+ prior games, starting goalies)\n', acc.round(4).to_string(index=False))
    acc.round(4).to_csv(f'{base}/backtest_accuracy.csv', index=False); calt.round(4).to_csv(f'{base}/backtest_calibration.csv', index=False)
    say('\nCalibration at common lines (test)\n', calt[calt.n >= 200].round(3).to_string(index=False))
    # final: fit everything, calibration kept from the 2024-25 stage
    mf = fit_means(sk, gl)
    pickle.dump(dict(models=mf, cal=cal, sk_feats=SK_FEATS, g_feats=G_FEATS), open(f'{base}/data/model.pkl', 'wb'))
    json.dump(cal, open(f'{base}/calibration.json', 'w'), indent=1)
    imp = pd.Series(mf['sog'].feature_importance('gain'), index=SK_FEATS).sort_values(ascending=False)
    say('\nTop features, shots on goal\n', (imp / imp.sum()).head(12).round(3).to_string())
    log.close()
