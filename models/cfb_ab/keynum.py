"""CFB key-number margin distribution for Model A / Model B pricing (normal x key-number weights fit 2014-21, checked 2022-25).
The drive sim under-produces 3- and 7-point margins (4.0% vs 9.8% at 3, model_a_test.json), so covers/win % use this curve;
the sim keeps team-score and total ranges (where it beat the bell). python3 keynum.py -> keynum.json"""
import json, numpy as np, pandas as pd
from lib import *
fit = pd.read_parquet(f"{HERE}/data/fit_feat.parquet"); te = pd.read_parquet(f"{HERE}/data/te_feat.parquet")
mf, _ = v4(fit, fit); mt, _ = v4(fit, te); SD = float(np.std(fit.am - mf)); G = np.arange(-100, 101)
def pm(mu, w): d = np.exp(-0.5 * ((G[None, :] - mu[:, None]) / SD) ** 2) * w[None, :]; return d / d.sum(1, keepdims=True)
w = np.ones(len(G)); A = np.bincount((fit.am.clip(-100, 100) + 100).astype(int), minlength=len(G))
for it in range(6):   # weights so the summed predicted histogram matches the actual margin histogram (|m| <= 21 only, rest 1)
    E = pm(mf, w).sum(0); r = (A + 2) / (E + 2); r = np.where(np.abs(G) <= 21, r, 1.0); r[G == 0] = 0.02  # no ties (overtime)
    w = w * r ** 0.8
w = np.where(np.abs(G) <= 21, w, 1.0); w[G == 0] = 0.0
def score(mu, y, ww):
    P = pm(mu, ww); i = (y.clip(-100, 100) + 100).astype(int); lp = np.log(np.maximum(P[np.arange(len(y)), i], 1e-5))
    return dict(log=round(float(lp.mean()), 4), p3=round(float(P[:, np.abs(G) == 3].sum(1).mean()), 4), p7=round(float(P[:, np.abs(G) == 7].sum(1).mean()), 4),
                p10=round(float(P[:, np.abs(G) == 10].sum(1).mean()), 4), p14=round(float(P[:, np.abs(G) == 14].sum(1).mean()), 4))
y = te.am.values
out = {"sd_margin": round(SD, 3), "grid": [-100, 100], "w": [round(float(x), 4) for x in w],
       "test_2022_25": {"keynum": score(mt, y, w), "normal": score(mt, y, np.ones(len(G))),
                        "actual": {k: round(float((np.abs(y) == v).mean()), 4) for k, v in (("p3", 3), ("p7", 7), ("p10", 10), ("p14", 14))}}}
# cover Brier vs closing spread (pushes out)
ok = te.line_close.notna() & (te.am != te.line_close)
for nm, ww in (("keynum", w), ("normal", np.ones(len(G)))):
    P = pm(mt[ok.values], ww); L = te.line_close[ok].values; pc = (P * (G[None, :] > L[:, None])).sum(1) / (1 - (P * np.isclose(G[None, :], L[:, None])).sum(1))
    out["test_2022_25"][nm]["cover_brier_close"] = round(float(((pc - (te.am[ok].values > L)) ** 2).mean()), 4)
json.dump(out, open(f"{HERE}/keynum.json", "w")); print(json.dumps(out["test_2022_25"], indent=1), "sd", SD)
