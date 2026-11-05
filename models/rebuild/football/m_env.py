"""NFL module test: weather and schedule. Wind, cold, dome/closed roof, rest days (bye, short week).
Additive shift on team points, fit on 2008-2017 residuals, tested on 2018+. Usage: python3 m_env.py base_tag [out]"""
import sys, os, numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE); sys.path.insert(0, os.path.dirname(HERE))
import harness as H
tag = sys.argv[1]; out = sys.argv[2] if len(sys.argv) > 2 else tag + "e"
SP = os.environ.get("NFLDL", "/tmp/claude-0/-home-user/157a0a45-f501-5d44-bea4-68a0646dd4e5/scratchpad/nfl")
gm = pd.read_csv(f"{SP}/games.csv")[["game_id", "roof", "temp", "wind", "home_rest", "away_rest"]].set_index("game_id")
p = pd.read_parquet(f"{HERE}/data/{tag}_nfl.parquet").join(gm, on="game_id")
indoor = p.roof.isin(["dome", "closed"]).astype(float)
wind = np.where(indoor == 1, 0, p.wind.fillna(p.wind.median())); temp = np.where(indoor == 1, 68, p.temp.fillna(p.temp.median()))
env = {"dome": indoor, "wind_over10": np.clip(wind - 10, 0, None), "cold_under30": np.clip(30 - temp, 0, None)}
FS = os.environ.get("ENV_F", "dome,wind_over10,cold_under30,bye,short").split(",")
def X(o, d):
    rest_o, rest_d = (p.home_rest, p.away_rest) if o == "h" else (p.away_rest, p.home_rest)
    cols = {**env, "bye": (rest_o >= 13).astype(float) - (rest_d >= 13).astype(float),
            "short": (rest_o <= 5).astype(float) - (rest_d <= 5).astype(float)}
    return np.column_stack([np.asarray(cols[f], float) for f in FS])
Xh, Xa = X("h", "a"), X("a", "h")
fit = (p.season >= 2008) & (p.season <= 2017)
XX = np.vstack([Xh[fit], Xa[fit]]); yy = np.concatenate([(p.act_h - p.pred_h)[fit], (p.act_a - p.pred_a)[fit]])
mu = XX.mean(0); Z = XX - mu; sd = Z.std(0); lam = 20.0
beta = np.linalg.solve((Z / sd).T @ (Z / sd) + lam * np.eye(len(FS)), (Z / sd).T @ (yy - yy.mean())) / sd
print(dict(zip(FS, beta.round(3))))
q = p.copy(); q["pred_h"] = p.pred_h + (Xh - mu) @ beta; q["pred_a"] = p.pred_a + (Xa - mu) @ beta
t, tq = p[p.season >= 2018], q[q.season >= 2018]
for loss in ("team_abs", "margin_sq", "total_abs"):
    r = H.compare(t, tq, loss); print(loss, r)
    H.log_result("NFL", "module", "weather+rest " + "+".join(FS), tag, r, "KEEP" if r["delta"] < 0 and r["p"] < 0.10 else ("HOLD" if r["delta"] < 0 else "DISCARD"), "fit 2008-17 test 2018+")
q.drop(columns=gm.columns).to_parquet(f"{HERE}/data/{out}_nfl.parquet")
