"""Hockey module test: schedule (back-to-back, rest) as an additive goal shift on the frame, fit on 2012-2021,
tested on 2022+. Features for a team's goals: its own fatigue (offense) and the opponent's (defense).
Usage: python3 m_sched.py frame_tag"""
import sys, os, numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE); sys.path.insert(0, os.path.dirname(HERE))
import harness as H
tag = sys.argv[1]
tg = pd.read_parquet(f"{HERE}/data/teamgames_nhl.parquet")[["game_id", "season", "date", "team", "ha"]].sort_values("date")
tg["rest"] = tg.groupby(["team", "season"]).date.diff().dt.days
tg["b2b"] = (tg.rest == 1).astype(float); tg["rest3"] = (tg.rest >= 3).astype(float)
tg["g3in4"] = 0.0
d = tg.groupby(["team", "season"]).date.shift(2); tg["g3in4"] = ((tg.date - d).dt.days <= 3).astype(float)
TZ = {**{t: -5 for t in "ATL BOS BUF CAR CBJ DET FLA MTL NJD NYI NYR OTT PHI PIT TBL TOR WSH".split()},
      **{t: -6 for t in "CHI DAL MIN NSH STL WPG".split()}, **{t: -7 for t in "COL CGY EDM ARI PHX UTA".split()},
      **{t: -8 for t in "ANA LAK SJS SEA VAN VGK".split()}}
host = tg[tg.ha == "home"].set_index("game_id").team
tg["loc"] = tg.game_id.map(host).map(TZ); tg["own"] = tg.team.map(TZ)
prev = tg.groupby(["team", "season"])["loc"].shift(1).fillna(tg.own)
tg["tz_move"] = (tg["loc"] - prev).abs()                      # hours shifted since the last game
tg["east"] = ((tg["loc"] - prev) > 0).astype(float) * tg.tz_move
tg["far"] = (tg["loc"] - tg.own).abs()                        # body clock vs home time
tg["b2b_move"] = tg.b2b * (tg.tz_move > 0)
F = os.environ.get("NHL_SCHED_F", "b2b,rest3,g3in4").split(",")
h = tg[tg.ha == "home"].set_index("game_id")[F]; a = tg[tg.ha == "away"].set_index("game_id")[F]
p = pd.read_parquet(f"{HERE}/data/{tag}_nhl.parquet")
p = p.join(h.add_prefix("h_"), on="game_id").join(a.add_prefix("a_"), on="game_id").fillna(0)
def X(side):
    o, dd = ("h_", "a_") if side == "h" else ("a_", "h_")
    return np.column_stack([p[o + f] for f in F] + [p[dd + f] for f in F] + [np.full(len(p), side == "h", float)])
Xh, Xa = X("h"), X("a")
fit = (p.season >= 2012) & (p.season <= 2021)
XX = np.vstack([Xh[fit], Xa[fit]]); yy = np.concatenate([(p.act_h - p.pred_h)[fit], (p.act_a - p.pred_a)[fit]])
mu = XX.mean(0); Z = XX - mu; lam = 50.0
beta = np.linalg.solve(Z.T @ Z + lam * np.eye(Z.shape[1]), Z.T @ (yy - yy.mean()))
names = [f"own_{f}" for f in F] + [f"opp_{f}" for f in F] + ["home"]
print(pd.Series(beta, names).round(4).to_string())
q = p.copy(); q["pred_h"] = p.pred_h + (Xh - mu) @ beta; q["pred_a"] = p.pred_a + (Xa - mu) @ beta
t, tq = p[p.season >= 2022], q[q.season >= 2022]
for loss in ("team_abs", "margin_abs", "total_abs", "margin_sq"):
    r = H.compare(t, tq, loss); print(loss, r)
    H.log_result("NHL", "module", "schedule " + "+".join(F), tag, r,
                 "KEEP" if r["delta"] < 0 and r["p"] < 0.10 else ("HOLD" if r["delta"] < 0 else "DISCARD"), "fit 2012-21 test 2022+, ridge 50")
q.to_parquet(f"{HERE}/data/{tag}{os.environ.get('NHL_SCHED_OUT', 's')}_nhl.parquet")
