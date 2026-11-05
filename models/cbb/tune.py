"""Re-tune rating hyperparameters with the roster prior on (audit fix C7), plus blowout cap and near-home neutral edge
(fixes C4, C5). Staged search on 2012-2019 margin RMSE (pre-2020 only; 2020-2026 stay out of sample).
usage: python3 tune.py   -> tuning.csv"""
import numpy as np, pandas as pd, pipeline
g, tg, links = pipeline.load()
g = g[g.season <= 2019]
def score(p, nh):
    h, *_ = pipeline.run(g, tg, links, p, nh)
    m = h[h.season.between(2012, 2019)]
    return float(np.sqrt(np.mean((m.margin - m.margin_hat) ** 2))), float(np.sqrt(np.mean((m.total - m.total_hat) ** 2)))
res = []
def trial(**kw):
    p = dict(prior_games=8, carry=0.9, tempo_carry=0.75, half_life=1000, cap=None, cap_slope=0.5); nh = kw.pop('nh', 0.0); p.update(kw)
    r = score(p, nh); res.append({**p, 'nh': nh, 'rmse_margin': r[0], 'rmse_total': r[1]}); print(res[-1], flush=True); return r[0]
base = trial()
best_cap = min([None, 15, 25, 35], key=lambda c: base if c is None else trial(cap=c))
best_nh = min([0.0, 0.5, 1.0], key=lambda n: trial(cap=best_cap, nh=n) if n else res[0]['rmse_margin'] if best_cap is None else [r for r in res if r['cap'] == best_cap and r['nh'] == 0][0]['rmse_margin'])
for pg in [4, 8, 12]:
    for carry in [0.8, 0.9, 1.0]:
        for hl in [120, 1000]:
            if (pg, carry, hl) != (8, 0.9, 1000): trial(cap=best_cap, nh=best_nh, prior_games=pg, carry=carry, half_life=hl)
r = pd.DataFrame(res).sort_values('rmse_margin'); r.to_csv('tuning.csv', index=False); print(r.head(10).to_string())
