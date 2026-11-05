"""NFL module test: the starting QB. Each QB's value = shrunk EPA per dropback, walked forward game by game
(decay HL_QB dropback-games, shrink K_QB dropbacks toward a backup level). Feature = tonight's starter minus the
team's recent starters (the team ratings already hold its usual QB), plus a 'new starter' flag.
Fit on 2008-2017 residuals, tested on 2018+. Usage: python3 m_qb.py base_tag [out_tag]"""
import sys, os, numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE); sys.path.insert(0, os.path.dirname(HERE))
import harness as H
tag = sys.argv[1]; out = sys.argv[2] if len(sys.argv) > 2 else tag + "q"
HL = float(os.environ.get("HL_QB", 24)); K = float(os.environ.get("K_QB", 250)); B0 = float(os.environ.get("B0", -0.08))
Q = pd.read_parquet(f"{HERE}/data/qb_games_nfl.parquet")
T = pd.read_parquet(f"{HERE}/data/teamgames_nfl.parquet")
T["starter"] = np.where(T.is_home == 1, T.home_qb_id, T.away_qb_id)
T = T.sort_values(["date", "game_id"])
lg = Q.merge(T[["game_id", "season"]].drop_duplicates(), on="game_id").groupby("season").apply(lambda d: d.epa.sum() / d.db.sum())
a = 0.5 ** (1 / HL); a_t = 0.5 ** (1 / 6)
S = {}            # qb -> [sum w*(epa - lg*db), sum w*db]
team_ew = {}; last = {}
val = lambda q: (S[q][0] + B0 * K) / (S[q][1] + K) if q in S else B0
Qg = {k: v for k, v in Q.groupby("game_id")}
rows = []
for gid, g in T.groupby("game_id", sort=False):
    f = {}
    for r in g.itertuples():
        v = val(r.starter); base = team_ew.get(r.team, v)
        f[r.is_home] = (v - base, float(last.get(r.team) not in (None, r.starter)), v)
    if len(f) == 2:
        rows.append({"game_id": gid, "h_dq": f[1][0], "h_new": f[1][1], "h_v": f[1][2], "a_dq": f[0][0], "a_new": f[0][1], "a_v": f[0][2]})
    for r in g.itertuples():
        v = val(r.starter); team_ew[r.team] = a_t * team_ew.get(r.team, v) + (1 - a_t) * v; last[r.team] = r.starter
    qg = Qg.get(gid); s_lg = lg.get(g.season.iloc[0], 0)
    if qg is not None:
        for r in qg.itertuples():
            s = S.setdefault(r.qb, [0.0, 0.0]); s[0] = a * s[0] + (r.epa - s_lg * r.db); s[1] = a * s[1] + r.db
F = pd.DataFrame(rows).set_index("game_id")
p = pd.read_parquet(f"{HERE}/data/{tag}_nfl.parquet").join(F, on="game_id").fillna(0)
X = lambda o: np.column_stack([p[o + "dq"], p[o + "new"]])
Xh, Xa = X("h_"), X("a_")
fit = (p.season >= 2008) & (p.season <= 2017)
XX = np.vstack([Xh[fit], Xa[fit]]); yy = np.concatenate([(p.act_h - p.pred_h)[fit], (p.act_a - p.pred_a)[fit]])
mu = XX.mean(0); Z = XX - mu; lam = 20.0; sd = Z.std(0)
beta = np.linalg.solve((Z / sd).T @ (Z / sd) + lam * np.eye(2), (Z / sd).T @ (yy - yy.mean())) / sd
print("beta (pts per 1.0 EPA/db change, new starter)", beta.round(3), "sd", sd.round(3))
q = p.copy(); q["pred_h"] = p.pred_h + (Xh - mu) @ beta; q["pred_a"] = p.pred_a + (Xa - mu) @ beta
t, tq = p[p.season >= 2018], q[q.season >= 2018]
chg = t[(t.h_new + t.a_new) > 0].game_id
for loss in ("team_abs", "margin_sq", "total_abs"):
    r = H.compare(t, tq, loss); r2 = H.compare(t[t.game_id.isin(chg)], tq[tq.game_id.isin(chg)], loss)
    print(loss, r, "\n   QB-change games", r2)
    if not os.environ.get("NOLOG"):
        H.log_result("NFL", "module", f"starting QB (HL {HL}, K {K}, backup {B0})", tag, r,
                     "KEEP" if r["delta"] < 0 and r["p"] < 0.10 else ("HOLD" if r["delta"] < 0 else "DISCARD"), "fit 2008-17 test 2018+")
q.drop(columns=F.columns).to_parquet(f"{HERE}/data/{out}_nfl.parquet")
