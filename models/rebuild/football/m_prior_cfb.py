"""CFB module test: preseason prior from returning production and the roster (transfer-aware QB and skill
players, from cfb-v3 roster_prior.csv / team-strength cfb_returning.csv). Pass 1 records how each team's final
offense ratings moved away from last season's carried rating; a ridge fit on 2015-2019 maps the roster features to
that move; pass 2 adds it to the preseason prior. Test 2021+. Usage: python3 m_prior_cfb.py [out_tag]"""
import sys, os, json, numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE); sys.path.insert(0, os.path.dirname(HERE))
import harness as H, engine as E, ridge_engine as RE
out = sys.argv[1] if len(sys.argv) > 1 else "ridge1p"
SUF = os.environ.get("SUF", ""); E.MAP.update(json.load(open(f"{HERE}/data/map_cfb{SUF}.json")))
prm = json.load(open(f"{HERE}/data/tuned_cfb_ridge.json")); prm["lam"] = tuple(prm["lam"])
T = pd.read_parquet(f"{HERE}/data/teamgames_cfb{SUF}.parquet")
TS = os.path.join(HERE, "../../team-strength/data"); V3 = os.path.join(HERE, "../../cfb-v3")
ret = pd.read_csv(f"{TS}/cfb_returning.csv").rename(columns={"team_id": "team"})
ros = pd.read_csv(f"{V3}/roster_prior.csv").rename(columns={"team_id": "team"})
F = ret.merge(ros, on=["season", "team"], how="outer")
FEAT = ["ret_off", "ret_pass", "ret_skill", "qb_anya", "qb_new", "qb_none", "avail_skill", "xfer_skill"]
F[FEAT] = F[FEAT].astype(float)
F = F.set_index(["season", "team"])
F = F[FEAT].fillna(F[FEAT].mean())
OFFC = [0, 1, 2, 6]   # sr, epp, xr, fin get the offense prior shift
rec = {}
def pass1(season, idx, pri):
    rec[season] = {c: (pri[c][0].copy(), pri[c][1].copy()) for c in E.COMP}; return pri
RE.run(T, prm, pass1)
inv = None
# final ratings of season s = carried prior of season s+1 / carry
rows = []
for s in sorted(rec):
    if s + 1 not in rec or s < 2015: continue
    for t, i in idx_map.items() if False else []: pass
seasons = sorted(rec); teams = pd.unique(pd.concat([T.team, T.opp])); idx = {t: i for i, t in enumerate(pd.unique(pd.concat([E.games(T).team, E.games(T).team_a])))}
X, Y, S = [], [], []
for s in seasons:
    if s + 1 not in rec or s - 1 not in rec: continue
    for t, i in idx.items():
        if (s, t) not in F.index: continue
        fin = np.array([rec[s + 1][E.COMP[k]][0][i] / prm["carry"] for k in OFFC]); prior = np.array([rec[s][E.COMP[k]][0][i] for k in OFFC])
        if np.all(fin == 0): continue
        X.append(F.loc[(s, t)].values); Y.append(fin - prior); S.append(s)
X, Y, S = np.array(X), np.array(Y), np.array(S)
fit = (S >= 2015) & (S <= 2019)
mu = X[fit].mean(0); sd = X[fit].std(0) + 1e-9; Z = (X[fit] - mu) / sd
G = np.linalg.solve(Z.T @ Z + 50 * np.eye(len(FEAT)), Z.T @ (Y[fit] - Y[fit].mean(0)))
print("R2 by component", [round(1 - np.var(Y[fit][:, k] - Z @ G[:, k]) / np.var(Y[fit][:, k]), 3) for k in range(len(OFFC))])
def pass2(season, idx_, pri):
    for t, i in idx_.items():
        if (season, t) not in F.index: continue
        z = (F.loc[(season, t)].values - mu) / sd; sh = z @ G
        for j, k in enumerate(OFFC): pri[E.COMP[k]][0][i] += sh[j]
    return pri
p = RE.run(T, prm, pass2)
b = pd.read_parquet(f"{HERE}/data/baseline_cfb_points.parquet").set_index("game_id").loc[p.game_id].reset_index()
m = p.pred_h - p.pred_a; t = 0.1 * (p.pred_h + p.pred_a) + 0.9 * (b.pred_h + b.pred_a)
q = p.copy(); q["pred_h"] = (t + m) / 2; q["pred_a"] = (t - m) / 2; q.to_parquet(f"{HERE}/data/{out}_cfb.parquet")
a = pd.read_parquet(f"{HERE}/data/{os.environ.get('BASE', 'ridge1b')}_cfb.parquet")
s = H.score(q[q.season >= 2021]); d2 = q[(q.season >= 2022) & (q.season <= 2025) & q.line_margin.notna()]; s2 = H.score(d2)
print({x: s[x] for x in ("team_mae", "margin_rmse", "margin_mae", "total_mae")}, "| lined 22-25 margin MAE", s2["margin_mae"], "close", s2["margin_mae_close"])
for l in ("team_abs", "margin_sq", "total_abs"):
    r = H.compare(a[a.season >= 2021], q[q.season >= 2021], l); print(r)
    e = H.compare(a[(a.season >= 2021) & (a.week <= 4)], q[(q.season >= 2021) & (q.week <= 4)], l); print("   weeks 1-4", e)
    H.log_result("CFB", "module", "roster/returning-production prior" + SUF, os.environ.get("BASE", "ridge1b"), r, "KEEP" if r["delta"] < 0 and r["p"] < 0.10 else ("HOLD" if r["delta"] < 0 else "DISCARD"), "fit 2015-19 test 2021+")
