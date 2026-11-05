"""Out-of-sample probability calibration: fit on 2018-19 to 2022-23, score 2023-24 against season-average lines,
regress outcomes on logit(model P(over)), lines around the median (audit P2). The slope (temperature) is stored in calibration.json and applied by
Model.prob_over. usage: python3 calibrate.py <model_dir>"""
import sys, json, numpy as np, pandas as pd, statsmodels.api as sm
sys.path.insert(0, sys.argv[1]); from model import fit, usable
md = sys.argv[1]
f = pd.read_parquet(f'{md}/data/features.parquet'); f = f[usable(f)]
m = fit(f[f.season.between(2019, 2023)]); m.temp = {}
h = f[f.season == 2024]; h = h.join(m.predict(h)); h = h[h.min_proj >= 15].sample(5000, random_state=1)
out = {}
for s in ['pts', 'reb', 'ast', 'fg3m', 'stl', 'blk', 'tov', 'pra']:
    p, y = [], []
    for _, r in h.iterrows():
        if np.isnan(r['sea_' + s]): continue
        # market-like lines: books set lines near the median outcome, so calibrate around the model's own median and the
        # last-10 average (audit P2), not the season average
        for line in (m.quantiles(r, s, (0.5,))[0] + 0.5 * (1 if m.quantiles(r, s, (0.5,))[0] < r['mu_' + s] else -1), np.floor(r['l10_' + s]) + 0.5 if r['l10_' + s] == r['l10_' + s] else np.floor(r['sea_' + s]) + 0.5):
            p.append(m.prob_over(r, s, line)); y.append(float(r[s] > line))
    p = np.clip(np.array(p), 1e-4, 1 - 1e-4); lg = np.log(p / (1 - p))
    g = sm.Logit(np.array(y), sm.add_constant(lg)).fit(disp=0)
    out[s] = {'slope': round(float(g.params[1]), 3), 'intercept': round(float(g.params[0]), 3), 'n': len(y)}
    print(s, out[s])
json.dump(out, open(f'{md}/calibration.json', 'w'), indent=1)
