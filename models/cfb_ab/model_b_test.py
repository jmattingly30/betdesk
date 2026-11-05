"""CFB Model B out of sample (2022-25 lined games, Model A = v4 means fit 2014-21): spreads and totals at the OPENER and
the CLOSE by gap; share of our gap that is real (b); confidence fit (research model-b-confidence CFB_open_2fold features:
mid favourite 7.5-24, week 8+, neutral, we like the dog, home side, engines agree = desk v3 on the same side) fit on two
seasons, tested on the other two; tiers A = top 10% confidence, B = next 23%, C = rest.
python3 model_b_test.py -> model_b_test.json (seconds)"""
import json, numpy as np, pandas as pd
from lib import *
fit = pd.read_parquet(f"{HERE}/data/fit_feat.parquet"); te = pd.read_parquet(f"{HERE}/data/te_feat.parquet")
te["m4"], te["t4"] = v4(fit, te)
J = -110; WIN = 100 / 110
def rec(win_mask, n):
    w = int(win_mask.sum()); return {"n": int(n), "win": round(w / n, 4) if n else None, "roi": round((w * WIN - (n - w)) / n, 4) if n else None, "se": round(float(np.sqrt(.25 / n)), 4) if n else None}
def ats(z, ours, line, act):
    e = z[ours] - z[line]; r = z[act] - z[line]; ok = (r != 0) & z[line].notna(); return e[ok], r[ok], z[ok]
def by_gap(e, r, gaps):
    return {f"{g}+": rec((np.sign(e) == np.sign(r))[e.abs() >= g], int((e.abs() >= g).sum())) for g in gaps}
def slope(e, r): b = np.polyfit(e, r, 1)[0]; return round(float(b), 3)
out = {}
for when, L, T in (("open", "line_open", "total_open"), ("close", "line_close", "total_close")):
    e, r, z = ats(te, "m4", L, "am"); e3, r3, _ = ats(te, "model_margin", L, "am")
    o = {"spread_v4": by_gap(e, r, (2, 3, 4, 6, 8, 10)), "spread_v3": by_gap(e3, r3, (2, 3, 4, 6, 8)), "b_v4": slope(e, r), "b_v3": slope(e3, r3)}
    o["spread_v4_by_season_4plus"] = {int(s): rec((np.sign(e) == np.sign(r))[(e.abs() >= 4) & (z.season == s)], int(((e.abs() >= 4) & (z.season == s)).sum())) for s in sorted(z.season.unique())}
    et, rt, zt = ats(te, "t4", T, "at"); et3, rt3, _ = ats(te, "model_total", T, "at")
    o["total_v4"] = by_gap(et, rt, (2, 3, 4, 5, 6, 8)); o["total_v3"] = by_gap(et3, rt3, (2, 4, 6)); o["b_total_v4"] = slope(et, rt)
    o["total_v4_over_under_4plus"] = {s: rec((np.sign(et) == np.sign(rt))[(et.abs() >= 4) & ((et > 0) == (s == "over"))], int(((et.abs() >= 4) & ((et > 0) == (s == "over"))).sum())) for s in ("over", "under")}
    if when == "open":
        zc = z.line_close; o["clv_pts_by_gap"] = {f"{g}+": round(float((np.sign(e) * (zc - z[L]))[e.abs() >= g].mean()), 3) for g in (2, 4, 6, 8)}
    out[when] = o; print(when, json.dumps(o))
# ---------- confidence fit at the opener (2-fold by season pairs)
e, r, z = ats(te, "m4", "line_open", "am"); z = z.copy(); z["edge"] = e; z["res"] = r
z["fav"] = z.line_open.abs(); z["mid_fav"] = z.fav.between(7.5, 24).astype(float); z["late"] = (z.wk >= 8).astype(float)
z["dog"] = (np.sign(z.edge) != np.sign(z.line_open)).astype(float); z["home_side"] = (z.edge > 0).astype(float)
z["agree"] = (np.sign(z.model_margin - z.line_open) == np.sign(z.edge)).astype(float)
F = ["mid_fav", "late", "neu", "dog", "home_side", "agree"]
def X(d): return np.c_[d.edge.values, *[d.edge.values * d[f].values for f in F]]
def fitb(d, lam=50.0):
    A = X(d); return np.linalg.solve(A.T @ A + lam * np.eye(A.shape[1]), A.T @ d.res.values)
folds = [(z.season <= 2023, z.season >= 2024), (z.season >= 2024, z.season <= 2023)]
z["conf"] = np.nan; coefs = []
for fm, tm in folds:
    b = fitb(z[fm]); coefs.append(b); z.loc[tm, "conf"] = X(z[tm]) @ b   # = trusted part of the gap in points (signed)
z["ac"] = z.conf * np.sign(z.edge)   # trusted points on our side (negative = conf says the other side)
ok = z.ac > 0
q90, q67 = np.quantile(z.ac, .90), np.quantile(z.ac, .67)
win = np.sign(z.edge) == np.sign(z.res)
tiers = {"A": z.ac >= q90, "B": (z.ac >= q67) & (z.ac < q90), "C": z.ac < q67}
o = {"features": F, "coef_mean": dict(zip(["edge"] + ["x_" + f for f in F], np.round(np.mean(coefs, 0), 4).tolist())),
     "cut_A_trusted_pts": round(float(q90), 3), "cut_B_trusted_pts": round(float(q67), 3),
     "tiers": {k: {**rec(win[m], int(m.sum())), "clv": round(float((np.sign(z.edge) * (z.line_close - z.line_open))[m].mean()), 3),
                   "dog_share": round(float(z.dog[m].mean()), 3)} for k, m in tiers.items()},
     "tier_A_by_season": {int(s): rec(win[tiers["A"] & (z.season == s)], int((tiers["A"] & (z.season == s)).sum())) for s in sorted(z.season.unique())},
     "raw_gap_top10": rec(win[z.edge.abs() >= np.quantile(z.edge.abs(), .9)], int((z.edge.abs() >= np.quantile(z.edge.abs(), .9)).sum())),
     "raw_gap_cut_top10": round(float(np.quantile(z.edge.abs(), .9)), 2)}
# is each feature pulling its weight? drop-one: tier A win rate without it
drop = {}
for f in F:
    F2 = [x for x in F if x != f]; c2 = np.full(len(z), np.nan)
    for fm, tm in folds:
        A_ = np.c_[z[fm].edge.values, *[z[fm].edge.values * z[fm][x].values for x in F2]]; b2 = np.linalg.solve(A_.T @ A_ + 50 * np.eye(A_.shape[1]), A_.T @ z[fm].res.values)
        c2[tm.values] = np.c_[z[tm].edge.values, *[z[tm].edge.values * z[tm][x].values for x in F2]] @ b2
    a2 = c2 * np.sign(z.edge); m = a2 >= np.quantile(a2, .9); drop[f] = rec(win[m], int(m.sum()))
o["tier_A_without_feature"] = drop
# share used live: one fit on all four seasons (cut points from the 2-fold test distribution above)
o["live_coef"] = dict(zip(["edge"] + ["x_" + f for f in F], np.round(fitb(z), 4).tolist()))
out["open_confidence"] = o; print(json.dumps(o, indent=1))
# close-like lines: does a gap there still hold? (tier B at most)
e, r, zc = ats(te, "m4", "line_close", "am")
out["close_gap_by_fav"] = {nm: rec((np.sign(e) == np.sign(r))[(e.abs() >= 4) & m], int(((e.abs() >= 4) & m).sum())) for nm, m in (("mid_fav", zc.line_close.abs().between(7.5, 24)), ("other", ~zc.line_close.abs().between(7.5, 24)))}
# moneyline: implied from the spread's trusted edge, untested on prices (no ML history) -> lean at most
json.dump(out, open(f"{HERE}/model_b_test.json", "w"), indent=1)
