"""Grade the rebuild on games Bet Desk already projected (GAMES.json.gz), using the rebuild's walk-forward
predictions (made from data before each game, so the same information a live run would have had).
NBA: team-ratings frame (the live version). NFL: full stack (QB, weather, missing regulars known 90 min before)."""
import sys, os, json, gzip, numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__)); RB = os.path.dirname(HERE); ROOT = os.path.normpath(f"{RB}/../..")
sys.path.insert(0, f"{RB}/basketball"); sys.path.insert(0, RB)
import engine as E
today = json.load(gzip.open(f"{ROOT}/models/grading/history/GAMES.json.gz"))
tg = pd.read_parquet(f"{RB}/basketball/data/teamgames_nba.parquet")
p, _ = E.run(tg, {**E.DEFAULT, **json.load(open(f"{RB}/basketball/data/tuned_stack_nba.json"))})
nfl = pd.read_parquet(f"{RB}/football/data/stack_nfl.parquet")
rows = []
for k, v in today.items():
    sp, gid = v["sp"], v["id"]
    if sp == "NBA":
        r = p[p.game_id.astype(str) == gid]
        if r.empty: continue
        r = r.iloc[0]; flip = r.home != v["h"]
        rh, ra, ah, aa = (r.pred_a, r.pred_h, r.act_a, r.act_h) if flip else (r.pred_h, r.pred_a, r.act_h, r.act_a)
    elif sp == "NFL":
        r = nfl[nfl.game_id == gid]
        if r.empty: continue
        r = r.iloc[0]; rh, ra, ah, aa = r.pred_h, r.pred_a, r.act_h, r.act_a
    else: continue
    if v.get("hs") is None or pd.isna(ah): continue
    L = (v.get("ln") or {}); line = L.get("L"); tot = L.get("T")
    rows.append({"sp": sp, "id": gid, "d": v["d"], "rh": rh, "ra": ra, "th": v["hs"], "ta": v["as"], "ah": ah, "aa": aa,
                 "line_m": (-line if (line is not None and sp == "NFL") else (-line if line is not None else None)), "line_t": tot})
D = pd.DataFrame(rows)
out = {}
for sp, d in D.groupby("sp"):
    res = {}
    for nm, h, a in (("rebuild", "rh", "ra"), ("today", "th", "ta")):
        res[nm] = {"n": len(d), "team_mae": round(float(((d[h] - d.ah).abs() + (d[a] - d.aa).abs()).mean() / 2), 2),
                   "margin_mae": round(float(((d[h] - d[a]) - (d.ah - d.aa)).abs().mean()), 2),
                   "total_mae": round(float(((d[h] + d[a]) - (d.ah + d.aa)).abs().mean()), 2),
                   "winner_right": round(float((np.sign(d[h] - d[a]) == np.sign(d.ah - d.aa)).mean()), 3)}
    x = d[d.line_t.notna()]
    res["line"] = {"n": len(x), "total_mae": round(float((x.line_t - (x.ah + x.aa)).abs().mean()), 2),
                   "margin_mae": round(float((x.line_m - (x.ah - x.aa)).abs().mean()), 2)}
    out[sp] = res
print(json.dumps(out, indent=1)); json.dump({"note": "rebuild walk-forward predictions vs Bet Desk unblended projections on the same games", **out}, open(f"{HERE}/backfill_report.json", "w"), indent=1)
