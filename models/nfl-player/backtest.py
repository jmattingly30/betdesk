"""Walk-forward backtest. Fit 2017-2022 -> calibrate on 2023 (dispersion + P(over) temperature);
fit 2017-2023 -> test 2024, fit 2017-2024 -> test 2025 (regular + post season).

python3 backtest.py <model_dir>  -> backtest_accuracy.csv, backtest_calibration.csv, calibration.json, backtest_log.txt"""
import sys, os, json, numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import model as mdl

M = sys.argv[1]
log = open(f'{M}/backtest_log.txt', 'w')
def say(*a):
    s = ' '.join(str(x) for x in a); print(s); log.write(s + '\n'); log.flush()

f = pd.read_parquet(f'{M}/data/features.parquet')
f['td'] = f.rec_td + f.rush_td
f['rush_rec_yds'] = f.rush_yds + f.rec_yds
EVAL = ['targets', 'rec', 'rec_yds', 'carries', 'rush_yds', 'rush_rec_yds', 'pass_att', 'cmp', 'pass_yds', 'pass_td', 'td']

# baselines: player's last-8-game average (any season) and season-to-date average (falls back to last 8)
g = f.groupby('pid')
for s in EVAL:
    f[f'b8_{s}'] = g[s].transform(lambda x: x.shift(1).rolling(8, min_periods=1).mean())
    f[f'bs_{s}'] = f.groupby(['pid', 'season'])[s].transform(lambda x: x.shift(1).expanding().mean()).fillna(f[f'b8_{s}'])

pools_all = None
def run(train_seasons, test_seasons):
    tr = f[f.season.isin(train_seasons)]
    te = f[f.season.isin(test_seasons)]
    m = mdl.fit(tr)
    return m, mdl.predict(m, te)

def hat_cols(p):
    p['rush_rec_yds_hat'] = p.rush_yds_hat + p.rec_yds_hat
    return p

def relevant(p):
    rec = p.position.isin(['WR', 'TE', 'RB']) & (p.snap_l >= 0.25) & (p.games_prior >= 3)
    qb = (p.is_qb_start == 1) & (p.games_prior >= 3)
    return rec, qb

# ---- calibration season
m22, cal = run(range(2017, 2023), [2023])
cal = hat_cols(cal)
disp = mdl.fit_dispersion(cal)
say('NB dispersion (size) from 2023:', {k: round(v, 2) for k, v in disp.items()})

def sim_probs(p, pools, n=1000):
    """P(over line) for each stat where line = last-8 average rounded to .5 (a stand-in for a prop line)."""
    out = []
    for i0 in range(0, len(p), 1500):
        c = p.iloc[i0:i0 + 1500]
        sims = mdl.simulate(c, pools, disp, n=n, seed=i0)
        for s in EVAL:
            line = np.floor(c[f'b8_{s}'].to_numpy()) + 0.5 if s != 'td' else np.full(len(c), 0.5)
            pr = (sims[s] > line[:, None]).mean(1)
            q10, q90 = np.percentile(sims[s], [10, 90], axis=1)
            out.append(pd.DataFrame({'idx': c.index, 'stat': s, 'line': line, 'p_over': pr, 'q10': q10, 'q90': q90,
                                     'sim_mean': sims[s].mean(1), 'actual': c[s].to_numpy()}))
    return pd.concat(out)

pools = mdl.load_pools(f'{M}/data/play_yards.parquet', max_season=2022)
rec_m, qb_m = relevant(cal)
cal_rel = cal[rec_m | qb_m]
pc = sim_probs(cal_rel, pools)
pc = pc.merge(cal_rel[['position', 'is_qb_start']], left_on='idx', right_index=True)
keep = lambda d: d[((d.stat.isin(['pass_att', 'cmp', 'pass_yds', 'pass_td'])) & (d.is_qb_start == 1)) |
                   ((~d.stat.isin(['pass_att', 'cmp', 'pass_yds', 'pass_td'])) & (d.position != 'QB'))]
pc = keep(pc)

def logit(p): p = np.clip(p, 1e-4, 1 - 1e-4); return np.log(p / (1 - p))
def ll(p, y): p = np.clip(p, 1e-4, 1 - 1e-4); return -np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))
temps = {}
for s, d in pc.groupby('stat'):
    y = (d.actual > d.line).astype(float).to_numpy()
    best = min(np.arange(0.6, 2.51, 0.05), key=lambda T: ll(1 / (1 + np.exp(-logit(d.p_over.to_numpy()) / T)), y))
    temps[s] = round(float(best), 2)
say('temperature by stat (2023):', temps)
json.dump({'nb_size': disp, 'temperature': temps}, open(f'{M}/calibration.json', 'w'), indent=1)

# ---- test seasons
# refit every season (audit F2): 2024 from 2017-2023, 2025 from 2017-2024, as the live model is refit each year
m23, te24 = run(range(2017, 2024), [2024])
_, te25 = run(range(2017, 2025), [2025])
te = hat_cols(pd.concat([te24, te25]))
pools = mdl.load_pools(f'{M}/data/play_yards.parquet', max_season=2023)
rec_m, qb_m = relevant(te)

rows = []
def acc(d, label, stats):
    for s in stats:
        e_m = (d[s] - d[f'{s}_hat']).abs().mean()
        e_8 = (d[s] - d[f'b8_{s}']).abs().mean()
        e_s = (d[s] - d[f'bs_{s}']).abs().mean()
        rows.append(dict(subset=label, stat=s, n=len(d), mae_model=round(e_m, 3), mae_last8=round(e_8, 3), mae_season_avg=round(e_s, 3),
                         gain_vs_best_baseline=round(1 - e_m / min(e_8, e_s), 3), bias=round((d[f'{s}_hat'] - d[s]).mean(), 3)))
RS = ['targets', 'rec', 'rec_yds', 'carries', 'rush_yds', 'rush_rec_yds']
QS = ['pass_att', 'cmp', 'pass_yds', 'pass_td']
recd = te[rec_m]
acc(recd[recd.position.isin(['WR', 'TE'])], 'WR/TE', ['targets', 'rec', 'rec_yds'])
acc(recd[recd.position == 'RB'], 'RB', RS)
acc(te[qb_m], 'QB starters', QS + ['carries', 'rush_yds'])
out_rec = recd[recd.position.isin(['WR', 'TE']) & (recd.miss_tsh_pos + recd.miss_tsh_TE * 0 >= 0.12)]
acc(out_rec, 'WR/TE, same-position teammate out (>=12% target share)', ['targets', 'rec', 'rec_yds'])
out_rb = recd[(recd.position == 'RB') & (recd.miss_csh_RB >= 0.2)]
acc(out_rb, 'RB, teammate RB out (>=20% carry share)', ['carries', 'rush_yds', 'rush_rec_yds'])
qbc = recd[recd.position.isin(['WR', 'TE']) & (recd.qb_changed == 1)]
acc(qbc, 'WR/TE, starting QB changed', ['targets', 'rec', 'rec_yds'])
A = pd.DataFrame(rows)
A.to_csv(f'{M}/backtest_accuracy.csv', index=False)
say('\nOut of sample 2024-2025 (relevant players: snaps >=25%, >=3 prior games; QBs = starters)')
say(A.to_string(index=False))

# anytime TD: Brier / log loss vs player's own last-8 TD rate
d = recd.copy()
p_m = 1 - np.exp(-d.td_hat)
p_b = (d.groupby('pid').td.transform(lambda x: (x.shift(1) > 0).rolling(8, min_periods=1).mean())).fillna(0.2).clip(0.02, 0.9)
y = (d.td > 0).astype(float)
say(f'\nAnytime TD (WR/TE/RB, n={len(d)}): log loss model {ll(p_m, y):.4f} vs last-8 TD rate {ll(p_b, y):.4f} vs league rate {ll(np.full(len(y), y.mean()), y):.4f}')

# calibration of P(over) on test seasons
te_rel = te[rec_m | qb_m]
pt = sim_probs(te_rel, pools)
pt = keep(pt.merge(te_rel[['position', 'is_qb_start']], left_on='idx', right_index=True))
pt['p_cal'] = [1 / (1 + np.exp(-logit(p) / temps[s])) for p, s in zip(pt.p_over, pt.stat)]
pt['over'] = (pt.actual > pt.line).astype(float)
pt['bucket'] = pd.cut(pt.p_cal, [0, .2, .3, .4, .45, .5, .55, .6, .7, .8, 1])
C = pt.groupby(['bucket'], observed=True).agg(n=('over', 'size'), pred=('p_cal', 'mean'), actual=('over', 'mean')).round(3)
say('\nP(over) calibration, 2024-25, all stats (line = last-8 average rounded to .5):'); say(C.to_string())
S = pt.groupby('stat').apply(lambda d: pd.Series({'n': len(d), 'logloss_model': ll(d.p_cal, d.over), 'logloss_coinflip': ll(np.full(len(d), d.over.mean()), d.over),
                                                  'in_10_90': ((d.actual >= d.q10) & (d.actual <= d.q90)).mean()})).round(4)
say('\nBy stat:'); say(S.to_string())
pt.groupby(['stat', 'bucket'], observed=True).agg(n=('over', 'size'), pred=('p_cal', 'mean'), actual=('over', 'mean')).round(3).to_csv(f'{M}/backtest_calibration.csv')

# coefficient tables for the write-up
say('\nTeam volume GLMs (fit 2017-2023):')
for k, v in m23['team'].items(): say(k, v.params.round(3).to_dict())
say('\nEfficiency GLMs (fit 2017-2023), log-scale coefficients:')
for k, v in m23['eff'].items(): say(k, v.params.round(3).to_dict())
imp = pd.DataFrame({k: pd.Series(v.feature_importance('gain'), index=mdl.OPP_X) for k, v in m23['opp'].items()})
imp = (imp / imp.sum()).round(3).sort_values('targets', ascending=False)
say('\nOpportunity model feature importance (share of gain):'); say(imp.head(15).to_string())
te.to_parquet(f'{M}/data/backtest_predictions.parquet', index=False)
