"""NFL module test: missing regulars (non-QB). A regular = 50%+ of his unit's snaps in 3 of the team's last 5
games; missing = no snaps tonight (inactives are known 90 minutes before kickoff). Lost share by position group,
weighted by the player's usual snap share. Own offense groups and the opponent's defense groups shift a team's
points. Snap data starts 2013: fit 2013-2017, test 2018+. Usage: python3 m_inj.py base_tag [out]  (env INJ_MODE=unit|group)"""
import sys, os, glob, numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE); sys.path.insert(0, os.path.dirname(HERE))
import harness as H
tag = sys.argv[1]; out = sys.argv[2] if len(sys.argv) > 2 else tag + "i"; MODE = os.environ.get("INJ_MODE", "group")
SP = os.environ.get("NFLDL", "/tmp/claude-0/-home-user/157a0a45-f501-5d44-bea4-68a0646dd4e5/scratchpad/nfl")
s = pd.concat([pd.read_parquet(f) for f in sorted(glob.glob(f"{SP}/snaps_*.parquet"))])
s = s[s.season >= 2013]
GRP = {"T": "OL", "G": "OL", "C": "OL", "OL": "OL", "WR": "WR", "TE": "TE", "RB": "RB", "FB": "RB",
       "DE": "DL", "DT": "DL", "NT": "DL", "DL": "DL", "LB": "LB", "ILB": "LB", "OLB": "LB", "MLB": "LB",
       "CB": "DB", "S": "DB", "FS": "DB", "SS": "DB", "DB": "DB"}
s["grp"] = s.position.map(GRP); s = s[s.grp.notna()]
s["pct"] = np.where(s.grp.isin(["OL", "WR", "TE", "RB"]), s.offense_pct, s.defense_pct)
gd = pd.read_csv(f"{SP}/games.csv")[["game_id", "gameday"]]; s = s.merge(gd, on="game_id")
s = s.sort_values(["gameday", "game_id"])
GR = ["OL", "WR", "TE", "RB", "DL", "LB", "DB"]
hist = {}; rows = []
for (gid, team), g in s.groupby(["game_id", "team"], sort=False):
    h = hist.get(team, [])
    today = set(g.pfr_player_id[g.pct > 0])
    f = {k: 0.0 for k in GR}; tot = {k: 0.0 for k in GR}
    if len(h) >= 3:
        cnt = {}; pctsum = {}; grp = {}
        for gm in h[-5:]:
            for pid, (pc, gr) in gm.items():
                if pc >= 0.5: cnt[pid] = cnt.get(pid, 0) + 1
                pctsum[pid] = pctsum.get(pid, 0) + pc; grp[pid] = gr
        for pid, c in cnt.items():
            if c >= 3:
                w = pctsum[pid] / min(len(h), 5); tot[grp[pid]] += w
                if pid not in today: f[grp[pid]] += w
    rows.append({"game_id": gid, "team": team, **{k: f[k] for k in GR}})
    hist[team] = (h + [dict(zip(g.pfr_player_id, zip(g.pct, g.grp)))])[-5:]
F = pd.DataFrame(rows)
if MODE == "unit":
    F["OFF"] = F[["OL", "WR", "TE", "RB"]].sum(1); F["DEF"] = F[["DL", "LB", "DB"]].sum(1); OG, DG = ["OFF"], ["DEF"]
else:
    OG, DG = ["OL", "WR", "TE", "RB"], ["DL", "LB", "DB"]
p = pd.read_parquet(f"{HERE}/data/{tag}_nfl.parquet")
fh = F.rename(columns={"team": "home"}).set_index(["game_id", "home"]); fa = F.rename(columns={"team": "away"}).set_index(["game_id", "away"])
H_ = fh.reindex(pd.MultiIndex.from_frame(p[["game_id", "home"]])).fillna(0).values; A_ = fa.reindex(pd.MultiIndex.from_frame(p[["game_id", "away"]])).fillna(0).values
cols = list(F.columns[2:])
ix = lambda names: [cols.index(c) for c in names]
Xh = np.column_stack([H_[:, ix(OG)], A_[:, ix(DG)]]); Xa = np.column_stack([A_[:, ix(OG)], H_[:, ix(DG)]])
fit = (p.season >= 2013) & (p.season <= 2017)
XX = np.vstack([Xh[fit], Xa[fit]]); yy = np.concatenate([(p.act_h - p.pred_h)[fit], (p.act_a - p.pred_a)[fit]])
mu = XX.mean(0); Z = XX - mu; sd = Z.std(0) + 1e-9; lam = 30.0
beta = np.linalg.solve((Z / sd).T @ (Z / sd) + lam * np.eye(Z.shape[1]), (Z / sd).T @ (yy - yy.mean())) / sd
print(dict(zip(["own_" + c for c in OG] + ["opp_" + c for c in DG], beta.round(2))))
q = p.copy(); q["pred_h"] = p.pred_h + (Xh - mu) @ beta; q["pred_a"] = p.pred_a + (Xa - mu) @ beta
t, tq = p[p.season >= 2018], q[q.season >= 2018]
for loss in ("team_abs", "margin_sq", "total_abs"):
    r = H.compare(t, tq, loss); print(loss, r)
    H.log_result("NFL", "module", f"missing regulars ({MODE})", tag, r, "KEEP" if r["delta"] < 0 and r["p"] < 0.10 else ("HOLD" if r["delta"] < 0 else "DISCARD"), "fit 2013-17 test 2018+")
q.to_parquet(f"{HERE}/data/{out}_nfl.parquet")
