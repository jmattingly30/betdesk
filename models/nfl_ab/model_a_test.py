"""NFL Model A distribution (J 2026-11-05 rollout). Means = rebuild drive model (stack: ratings + starting QB + weather + missing
regulars; shadow.py live). Ranges = normal x key-number weights (keynum.py, kept 2026-10-27: margin log score 3.98 -> 3.86);
the play simulator stays a viewer (research: it under-produces margins of 3: 8.5% vs 14.7%, and adds nothing to the means).
Fits weights on 2010-2017, checks 2018+ calibration (win, margin of exactly 3 / 7, push rates at 3 / 7, total buckets).
python3 model_a_test.py -> model_a_test.json (weights used by live.py)"""
import os, json, numpy as np, pandas as pd
from scipy import stats
HERE = os.path.dirname(os.path.abspath(__file__)); FB = os.path.normpath(f"{HERE}/../rebuild/football")
p = pd.read_parquet(f"{FB}/data/stack_nfl.parquet")
p["mu_m"] = p.pred_h - p.pred_a; p["mu_t"] = p.pred_h + p.pred_a; p["m"] = p.act_h - p.act_a; p["t"] = p.act_h + p.act_a
tr, te = p[(p.season >= 2010) & (p.season <= 2017)], p[p.season >= 2018]
SDM, SDT = float(np.std(tr.m - tr.mu_m)), float(np.std(tr.t - tr.mu_t))
SD_TEAM = float(np.std(np.concatenate([tr.act_h - tr.pred_h, tr.act_a - tr.pred_a])))
M = np.arange(-60, 61); TT = np.arange(0, 121)
def weights(mu, act, grid, sd):
    dens = stats.norm.pdf(grid[None, :], mu[:, None], sd).mean(0)
    freq = pd.Series(act).value_counts(normalize=True).reindex(grid, fill_value=0).values
    w = (freq + 1e-4) / (dens + 1e-4); return np.where(dens > 2e-3, w, 1.0)
wm, wt = weights(tr.mu_m.values, tr.m.values, M, SDM), weights(tr.mu_t.values, tr.t.values, TT, SDT)
def pmf(mu, grid, sd, w):
    d = stats.norm.pdf(grid[None, :], np.atleast_1d(mu)[:, None], sd) * w[None, :]; return d / d.sum(1, keepdims=True)
Pm, Pt = pmf(te.mu_m.values, M, SDM, wm), pmf(te.mu_t.values, TT, SDT, wt)
cal = {}
pw = Pm[:, M > 0].sum(1) + 0.5 * Pm[:, M == 0].sum(1); y = (te.m > 0).values.astype(float); k = te.m.values != 0
bins = pd.cut(pw, [0, .2, .35, .5, .65, .8, 1])
cal["win_by_bucket"] = {str(b): {"n": int(g.sum()), "pred": round(float(pw[g.values].mean()), 3), "act": round(float(y[(g & k).values].mean()), 3)} for b, g in pd.Series(bins).pipe(lambda s: [(b, s == b) for b in s.cat.categories])}
for kk in (3, 7, 10):
    cal[f"abs_margin_{kk}"] = {"pred": round(float((Pm[:, M == kk].sum(1) + Pm[:, M == -kk].sum(1)).mean()), 4), "act": round(float((te.m.abs() == kk).mean()), 4)}
for q in (10, 90):
    cal[f"total_below_p{q}"] = round(float((te.t.values < np.array([TT[np.searchsorted(np.cumsum(r), q / 100)] for r in Pt])).mean()), 3)
    cal[f"margin_below_p{q}"] = round(float((te.m.values < np.array([M[np.searchsorted(np.cumsum(r), q / 100)] for r in Pm])).mean()), 3)
out = {"sd_margin": round(SDM, 3), "sd_total": round(SDT, 3), "sd_team": round(SD_TEAM, 3), "fit": "2010-2017", "check": "2018+", "n_check": len(te),
       "calibration_2018plus": cal, "w_margin": [round(float(x), 4) for x in wm], "w_total": [round(float(x), 4) for x in wt]}
json.dump(out, open(f"{HERE}/model_a_test.json", "w"))
print({k: v for k, v in out.items() if not k.startswith("w_")})
