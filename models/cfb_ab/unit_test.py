"""Unit-sum term on top of Model A (v4) team scores (research matchup_prior.py 'A fin_sum' on v3: -0.027, t -3.9).
Team row: offense's drive rating + opponent defense's drive rating (unit_ratings.parquet, week w uses games before w).
Fit 2015-21 lined, test 2022-25 lined; also vs implied closing team totals. python3 unit_test.py -> unit_test.json"""
import json, numpy as np, pandas as pd
from lib import *
fit = pd.read_parquet(f"{HERE}/data/fit_feat.parquet"); te = pd.read_parquet(f"{HERE}/data/te_feat.parquet"); UR = pd.read_parquet(f"{HERE}/data/unit_ratings.parquet")
for d in (fit, te): d["m4"], d["t4"] = v4(fit, d)
te["ih"], te["ia"] = (te.total_close + te.line_close) / 2, (te.total_close - te.line_close) / 2
M = ["fin", "pts", "tdr"]
def rows(F):
    out = []
    for o, dd, act, im, sg in (("home_id", "away_id", "act_h", "ih", 1), ("away_id", "home_id", "act_a", "ia", -1)):
        x = pd.DataFrame({"game_id": F.game_id.astype(str), "season": F.season.astype(int), "wk": F.wk.astype(int), "off": F[o].astype(int), "opp": F[dd].astype(int),
                          "act": F[act], "pred": (F.t4 + sg * F.m4) / 2, "imp": F[im] if im in F else np.nan}); out.append(x)
    X = pd.concat(out, ignore_index=True); X["r"] = X.act - X.pred
    A = UR.rename(columns={"team": "off", "week": "wk", **{m + "_d": m + "_x" for m in M}}).drop(columns=[m + "_x" for m in M])
    B = UR.rename(columns={"team": "opp", "week": "wk", **{m + "_o": m + "_y" for m in M}}).drop(columns=[m + "_y" for m in M])
    X = X.merge(A, on=["season", "wk", "off"], how="left").merge(B, on=["season", "wk", "opp"], how="left")
    for m in M: X[m + "_sum"] = X[m + "_o"] + X[m + "_d"]
    return X
F, T = rows(fit), rows(te); res = {"coverage_test": round(float(T.fin_sum.notna().mean()), 3)}
F = F.dropna(subset=["fin_sum"]); T = T.copy()
for c in [m + "_sum" for m in M]: T[c] = T[c].fillna(0)
def test(nm, cols):
    X = np.c_[np.ones(len(F)), F[cols].values]; b = np.linalg.lstsq(X, F.r.values, rcond=None)[0]; p = np.c_[np.ones(len(T)), T[cols].values] @ b
    e0, e1 = np.abs(T.r), np.abs(T.r - p); d = e1 - e0; ok = T.imp.notna()
    r = {"coef": dict(zip(["c"] + cols, np.round(b, 3).tolist())), "MAE_A": round(float(e0.mean()), 4), "MAE_A_plus": round(float(e1.mean()), 4), "diff": round(float(d.mean()), 4),
         "t": round(float(d.mean() / (d.std() / np.sqrt(len(d)))), 2), "seasons": {int(k): round(float(v), 3) for k, v in d.groupby(T.season).mean().items()},
         "vs_line_MAE": round(float(np.abs(T.act - T.imp)[ok].mean()), 3), "closer_than_line_A_plus": round(float((e1 < np.abs(T.act - T.imp))[ok].mean()), 3)}
    res[nm] = r; print(nm, r)
for nm, c in (("fin_sum", ["fin_sum"]), ("pts_sum", ["pts_sum"]), ("tdr_sum", ["tdr_sum"]), ("all", ["fin_sum", "pts_sum", "tdr_sum"])): test(nm, c)
json.dump(res, open(f"{HERE}/unit_test.json", "w"), indent=1)
