"""Fit on 2018-19 to 2023-24, test on 2024-25 and 2025-26 (season labels 2025, 2026). Writes backtest_*.csv and a log."""
import sys, numpy as np, pandas as pd
sys.path.insert(0, sys.argv[1]); from model import fit, usable, STATS
md = sys.argv[1]
f = pd.read_parquet(f'{md}/data/features.parquet')
f = f[usable(f)].copy()
tr, te = f[f.season.between(2019, 2024)], f[f.season >= 2025].copy()
m = fit(tr)
pr = m.predict(te); te = te.join(pr)
te.to_parquet(f'{md}/data/test_predictions.parquet')
print('train', len(tr), 'test', len(te))
z = (te['min'] - te.min_proj) / te.min_sd
print('minutes: share inside projected 80%% band %.3f, 50%% band %.3f' % ((z.abs() < 1.2816).mean(), (z.abs() < 0.6745).mean()))
print('\nminutes MAE: model %.2f | long avg %.2f | short avg %.2f | last game %.2f' % (
    (te['min'] - te.min_proj).abs().mean(), (te['min'] - te.ml).abs().mean(), (te['min'] - te.ms).abs().mean(), (te['min'] - te.last_min).abs().mean()))
rows = []
te['ex_usg'] = te.d_usg - te.d_min
for sub_name, sub in [('all', te), ('proj 24+ min', te[te.min_proj >= 24]), ('high-usage teammate out', te[te.ex_usg < -0.05])]:
    for s in STATS:
        y = sub[s]
        cand = {'model': sub['mu_' + s], 'season avg': sub['sea_' + s], 'last 10 avg': sub['l10_' + s],
                'ewma rate x avg min': sub['r_' + s] * sub.ml, 'model rate x actual min': sub['rate_' + s] * sub['min']}
        r = {'players': sub_name, 'stat': s, 'n': len(sub), 'mean': y.mean()}
        for k, v in cand.items():
            ok = v.notna(); r[f'MAE {k}'] = (y[ok] - v[ok]).abs().mean()
        rows.append(r)
acc = pd.DataFrame(rows); acc.to_csv(f'{md}/backtest_accuracy.csv', index=False)
pd.set_option('display.width', 250); print(acc.round(3).to_string(index=False))
print('\ncoefficients (rate model, log scale):')
for s in ['pts', 'reb', 'ast', 'fg3m']:
    print(s, m.glm[s].params.round(3).to_dict())
print('NB dispersion', {k: round(v, 3) for k, v in m.alpha.items()})
# calibration on a pseudo line = season average rounded to x.5 (what a book line usually resembles)
rng = np.random.default_rng(0)
smp = te[te.min_proj >= 20].sample(6000, random_state=0)
cal = []
for s in ['pts', 'reb', 'ast', 'fg3m', 'pra']:
    for i, row in smp.iterrows():
        if np.isnan(row['sea_' + s]): continue
        line = np.floor(row['sea_' + s]) + 0.5
        k, pm = m.dist(row, s)
        cdf = np.cumsum(pm)
        lo, hi = k[np.searchsorted(cdf, 0.1)], k[np.searchsorted(cdf, 0.9)]
        cal.append((s, line, m.prob_over(row, s, line), float(row[s] > line), lo <= row[s] <= hi))
c = pd.DataFrame(cal, columns=['stat', 'line', 'p_over', 'over', 'in80'])
c.to_csv(f'{md}/backtest_calibration_sample.csv', index=False)
c['bucket'] = pd.cut(c.p_over, [0, .3, .4, .45, .5, .55, .6, .7, 1])
print('\ncalibration after temperature (P over season-avg line), proj 20+ min sample:')
print(c.groupby('bucket', observed=True).agg(n=('over', 'size'), pred=('p_over', 'mean'), actual=('over', 'mean')).round(3).to_string())
print('80% interval coverage by stat:', c.groupby('stat').in80.mean().round(3).to_dict())
print('Brier model %.4f vs coin-flip-at-base-rate %.4f' % (((c.p_over - c.over) ** 2).mean(), ((c.over.mean() - c.over) ** 2).mean()))
