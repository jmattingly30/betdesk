"""Model B CBB tier rules, graded out of sample (leave-one-month-out on the 2026 DK opener, the only CBB opener season on file).
real edge = gap x (b0 + sum b_k z_k)  (research/model-b-confidence-2026-11 confidence.py, ridge 0.1).
Needs /tmp/f_cbb_open.parquet: run research/model-b-confidence-2026-11/cbb.py first."""
import json, numpy as np, pandas as pd
F_CBB = ["we_like_dog", "neu", "nonconf", "tier_min", "engines_same_side", "av_abs", "nov", "fav_size"]
d = pd.read_parquet("/tmp/f_cbb_open.parquet"); d["mon"] = pd.to_datetime(d.date).dt.month; d["nov"] = (d.mon == 11).astype(float)
def fit(F, lam=0.1):
    mu = {f: float(F[f].astype(float).mean()) for f in F_CBB}; sd = {f: float(F[f].astype(float).std() or 1) for f in F_CBB}
    def X(z): return np.column_stack([z.edge.values] + [z.edge.values * ((z[f].astype(float).values - mu[f]) / sd[f]) for f in F_CBB])
    A = np.nan_to_num(X(F)); y = F.res.values; P = np.eye(A.shape[1]) * lam * np.mean(np.diag(A.T @ A)); P[0, 0] = 0
    return np.linalg.solve(A.T @ A + P, A.T @ y), X, mu, sd
rows = []
for m in sorted(d.mon.unique()):
    b, X, _, _ = fit(d[d.mon != m]); T = d[d.mon == m].copy(); T["real"] = np.nan_to_num(X(T)) @ b; rows.append(T)
T = pd.concat(rows)
def ats(z, side):
    s = np.sign(side); r = np.sign(z.res); k = (r != 0) & (s != 0); w = (s[k] == r[k]).mean(); n = int(k.sum())
    return dict(n=n, win=round(float(w), 4), roi=round(float(w * 100 / 110 - (1 - w)), 4), se=round(float(np.sqrt(w * (1 - w) / max(n, 1))), 4),
                clv=round(float((np.sign(side) * (z.line_close_dk - z.line_open))[k].mean()), 2))
q90, q67 = T.real.abs().quantile([0.9, 0.67]).values
same = np.sign(T.real) == np.sign(T.edge)
out = {"real_edge_pts_q90": round(float(q90), 2), "real_edge_pts_q67": round(float(q67), 2), "n": len(T)}
# R1 research tiers (side from real edge)
tier1 = np.select([T.real.abs() >= q90, T.real.abs() >= q67], ["A", "B"], "C")
out["R1_research_quantile_tiers"] = {t: ats(T[tier1 == t], T.real[tier1 == t]) for t in "ABC"}
# R2: A = top-10% real edge AND same side as our raw gap; B = raw gap >= 3 not A; C = rest
A2 = (T.real.abs() >= q90) & same; B2 = ~A2 & (T.edge.abs() >= 3)
tier2 = np.select([A2, B2], ["A", "B"], "C")
out["R2_A_conf_top10_same_side__B_raw_gap3"] = {t: ats(T[tier2 == t], T.edge[tier2 == t]) for t in "ABC"}
# R3: A = top-10% real & raw gap>=3; B = raw gap>=3 rest; C = rest
A3 = (T.real.abs() >= q90) & same & (T.edge.abs() >= 3); B3 = ~A3 & (T.edge.abs() >= 3)
tier3 = np.select([A3, B3], ["A", "B"], "C")
out["R3_A_conf_top10_and_gap3__B_gap3"] = {t: ats(T[tier3 == t], T.edge[tier3 == t]) for t in "ABC"}
# raw gap only reference
for gp in (2, 3, 4, 5): out[f"raw_gap_{gp}+"] = ats(T[T.edge.abs() >= gp], T.edge[T.edge.abs() >= gp])
# flips: games where real edge disagrees with raw side
fl = ~same & (T.real.abs() >= q67); out["flipped_side_q67"] = ats(T[fl], T.real[fl])
# final live coefficients: fit on every 2026 opener game
b, X, mu, sd = fit(d)
dd = d.copy(); dd["real"] = np.nan_to_num(X(dd)) @ b
out["live"] = dict(features=F_CBB, coef=dict(zip(["edge"] + F_CBB, [round(float(v), 4) for v in b])), mu=mu, sd=sd,
                   q90=round(float(dd.real.abs().quantile(0.9)), 3), q67=round(float(dd.real.abs().quantile(0.67)), 3))
json.dump(out, open("model_b_test.json", "w"), indent=1); print(json.dumps({k: v for k, v in out.items() if k != "live"}, indent=1)); print(out["live"])
# R4 (chosen): A = R2's A OR raw gap >= 4 (same side); B = raw gap 3+ not A; C = rest
A4 = A2 | (T.edge.abs() >= 4); B4 = ~A4 & (T.edge.abs() >= 3); tier4 = np.select([A4, B4], ["A", "B"], "C")
out["R4_A_conf_top10_or_gap4__B_gap3"] = {t: ats(T[tier4 == t], T.edge[tier4 == t]) for t in "ABC"}
out["R4_by_month_A"] = {int(m): ats(T[(tier4 == "A") & (T.mon == m)], T.edge[(tier4 == "A") & (T.mon == m)]) for m in sorted(T.mon.unique())}
json.dump(out, open("model_b_test.json", "w"), indent=1); print(out["R4_A_conf_top10_or_gap4__B_gap3"]); print(out["R4_by_month_A"])
