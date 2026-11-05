"""Hockey module test: who is dressed tonight. Each skater gets a 5v5 impact (on-ice xG for / against per 60,
relative to his team in the same games, shrunk toward 0), walked forward game by game. A team's lineup value is
the sum of impact x expected 5v5 minutes over tonight's dressed skaters; the feature is tonight's value minus the
team's recent average value (so it only says what changed), plus the share of regular minutes missing.
Fit on 2012-2021 residuals of the base predictions, tested on 2022+.
Usage: python3 m_lineup.py base_tag [out_tag]   (env HL_IMP, K_IMP)"""
import sys, os, glob, numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE); sys.path.insert(0, os.path.dirname(HERE))
import harness as H
tag = sys.argv[1]; out = sys.argv[2] if len(sys.argv) > 2 else tag + "l"
HL = float(os.environ.get("HL_IMP", 60)); K = float(os.environ.get("K_IMP", 8)) * 3600   # shrink: K hours of 5v5 at 0
pl = pd.concat([pd.read_parquet(f) for f in sorted(glob.glob(f"{HERE}/data/players_nhl/*.parquet"))]).drop_duplicates(["game_id", "team", "player"])
tg = pd.read_parquet(f"{HERE}/data/teamgames_nhl.parquet")[["game_id", "date", "season", "team", "ha"]]
pl = pl.merge(tg, on=["game_id", "team"], how="inner").sort_values(["date", "game_id"])
tt = pl.groupby(["game_id", "team"])[["toi5", "xgf5", "xga5"]].transform("sum")
pl["rf"] = pl.xgf5 - pl.toi5 * tt.xgf5 / tt.toi5.clip(lower=1)     # on-ice minus team rate over the same time
pl["ra"] = pl.xga5 - pl.toi5 * tt.xga5 / tt.toi5.clip(lower=1)
a = 0.5 ** (1 / HL); a_t = 0.5 ** (1 / 10)
P = {}   # player -> [sum w*rf, sum w*ra, sum w*toi5, ew toi5 per game]
T = {}   # team -> dict(last games list of player sets, ew lineup values, ew toi of regulars)
rows = []
for gid, g in pl.groupby("game_id", sort=False):
    feats = {}
    for team, gt in g.groupby("team"):
        st = T.setdefault(team, {"recent": [], "vo": None, "vd": None, "reg": {}})
        vo = vd = 0.0
        for pid in gt.player.values:
            s = P.get(pid)
            if s is None: continue
            vo += s[0] / (s[2] + K) * s[3]; vd += s[1] / (s[2] + K) * s[3]
        regs = {p for gs in st["recent"][-5:] for p in gs if sum(p in x for x in st["recent"][-5:]) >= 3}
        missing = sum(P[p][3] for p in regs - set(gt.player) if p in P)
        tot = sum(P[p][3] for p in regs if p in P) or 1.0
        feats[team] = dict(d_off=vo - (st["vo"] if st["vo"] is not None else vo), d_def=vd - (st["vd"] if st["vd"] is not None else vd),
                           lost=missing / tot, ha=gt.ha.iloc[0])
        st["vo"] = vo if st["vo"] is None else a_t * st["vo"] + (1 - a_t) * vo
        st["vd"] = vd if st["vd"] is None else a_t * st["vd"] + (1 - a_t) * vd
        st["recent"] = (st["recent"] + [set(gt.player)])[-5:]
    if len(feats) == 2:
        h = [k for k in feats if feats[k]["ha"] == "home"]; aw = [k for k in feats if feats[k]["ha"] == "away"]
        if h and aw:
            fh, fa = feats[h[0]], feats[aw[0]]
            rows.append({"game_id": gid, **{f"h_{k}": fh[k] for k in ("d_off", "d_def", "lost")}, **{f"a_{k}": fa[k] for k in ("d_off", "d_def", "lost")}})
    for r in g.itertuples():   # update after the game
        s = P.setdefault(r.player, [0.0, 0.0, 0.0, r.toi5])
        s[0] = a * s[0] + r.rf; s[1] = a * s[1] + r.ra; s[2] = a * s[2] + r.toi5; s[3] = a_t * s[3] + (1 - a_t) * r.toi5
F = pd.DataFrame(rows).set_index("game_id")
F[[c for c in F if "d_" in c]] /= 3600 / 60   # xG per game units -> (impact per sec * sec) already xG; keep scale sane
p = pd.read_parquet(f"{HERE}/data/{tag}_nhl.parquet").join(F, on="game_id").fillna(0)
def X(o, d):   # features for side o's goals: o's offense change, d's defense change, missing regulars
    return np.column_stack([p[o + "d_off"], p[d + "d_def"], p[o + "lost"], p[d + "lost"]])
Xh, Xa = X("h_", "a_"), X("a_", "h_")
fit = (p.season >= 2012) & (p.season <= 2021)
XX = np.vstack([Xh[fit], Xa[fit]]); yy = np.concatenate([(p.act_h - p.pred_h)[fit], (p.act_a - p.pred_a)[fit]])
mu = XX.mean(0); Z = XX - mu; sd = Z.std(0); lam = 50.0
beta = np.linalg.solve((Z / sd).T @ (Z / sd) + lam * np.eye(Z.shape[1]), (Z / sd).T @ (yy - yy.mean())) / sd
print(pd.Series(beta, ["own_off_change", "opp_def_change", "own_lost_share", "opp_lost_share"]).round(4).to_string())
print("feature sd", np.round(sd, 4))
q = p.copy(); q["pred_h"] = p.pred_h + (Xh - mu) @ beta; q["pred_a"] = p.pred_a + (Xa - mu) @ beta
t, tq = p[p.season >= 2022], q[q.season >= 2022]
for loss in ("team_abs", "margin_abs", "total_abs", "margin_sq"):
    r = H.compare(t, tq, loss); print(loss, r)
    H.log_result("NHL", "module", f"lineup (dressed skaters 5v5 impact, HL {HL}, K {K/3600}h)", tag, r,
                 "KEEP" if r["delta"] < 0 and r["p"] < 0.10 else ("HOLD" if r["delta"] < 0 else "DISCARD"), "fit 2012-21 test 2022+")
q.drop(columns=F.columns).to_parquet(f"{HERE}/data/{out}_nhl.parquet")
