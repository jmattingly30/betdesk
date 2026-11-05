"""NBA Model A research fix #1 (model-b-confidence nba.json component regression: market underprices our OREB and 3PA share,
our 3P% (t -3.7) and FT rate (t -2.6) are overweighted). Test: walk-forward (fit seasons < s, test s) correction of each team's
projected points from its predicted four-factor components, on J's yardstick (implied closing team totals).
python3 comp_test.py -> comp_test.json"""
import json, numpy as np, pandas as pd
M = "/mnt/project-files/sports-analytics/models/"
s = pd.read_parquet(M + "rebuild/basketball/data/stack_nba.parquet")
G = pd.read_parquet(M + "nba-team/data/games.parquet", columns=["game_id", "close", "total_close", "sbr_open", "sbr_total_open"])
d = s.merge(G, on="game_id", how="left"); d = d[(d.season >= 2016) & (d.season <= 2026) & d.act_h.notna()].copy()
C = ["tov", "orb", "fta", "p3", "p2", "s3", "ftp"]
rows = []
for side, o in (("h", "a"), ("a", "h")):
    z = pd.DataFrame({"game_id": d.game_id, "season": d.season, "act": d[f"act_{side}"], "pred": d[f"pred_{side}"], "poss": d.poss_hat,
                      "imp": (d.total_close + (d.close if side == "h" else -d.close)) / 2,
                      "imp_o": (d.sbr_total_open + (d.sbr_open if side == "h" else -d.sbr_open)) / 2})
    for c in C: z[c] = d[f"{side}_{c}"] * d.poss_hat     # component x pace = counts per game
    rows.append(z)
z = pd.concat(rows, ignore_index=True)
X = lambda q: np.c_[np.ones(len(q)), (q[C] - mu).to_numpy()]
out = {"by_season": {}, "pooled": {}}
E = {"base": [], "fix": [], "line": [], "n": 0}; coefs = {}
for lam in (0.0, 50.0, 500.0):
    eb, ef, el = [], [], []
    for se in range(2019, 2027):
        tr, te = z[z.season < se], z[(z.season == se) & z.imp.notna()]
        mu = tr[C].mean(); Xt = X(tr); y = (tr.act - tr.pred).to_numpy()
        R = np.eye(Xt.shape[1]) * lam; R[0, 0] = 0
        b = np.linalg.solve(Xt.T @ Xt + R, Xt.T @ y)
        f = te.pred + X(te) @ b
        eb += list((te.pred - te.act).abs()); ef += list((f - te.act).abs()); el += list((te.imp - te.act).abs())
        if lam == 50.0:
            out["by_season"][se] = {"n": len(te), "base": round(float((te.pred - te.act).abs().mean()), 3), "fix": round(float((f - te.act).abs().mean()), 3), "line": round(float((te.imp - te.act).abs().mean()), 3)}
            coefs[se] = dict(zip(["const"] + C, np.round(b, 3).tolist()))
    out["pooled"][f"ridge_{lam:g}"] = {"n": len(eb), "base_MAE": round(float(np.mean(eb)), 3), "fix_MAE": round(float(np.mean(ef)), 3), "line_MAE": round(float(np.mean(el)), 3)}
out["coefs_ridge50"] = coefs
print(json.dumps(out["pooled"], indent=1)); print(json.dumps(out["by_season"]))
json.dump(out, open("comp_test.json", "w"), indent=1)
