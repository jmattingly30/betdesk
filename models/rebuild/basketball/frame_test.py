import sys, os, time, numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE); sys.path.insert(0, os.path.dirname(HERE))
import harness as H, engine as E
import json
sport = sys.argv[1]
prm = dict(E.DEFAULT); tag = "frame0"
if len(sys.argv) > 2:
    prm.update(json.load(open(sys.argv[2]))); tag = sys.argv[3] if len(sys.argv) > 3 else "frame1"
tg = pd.read_parquet(f"{HERE}/data/teamgames_{sport}.parquet")
t0 = time.time(); p, R = E.run(tg, prm); print("run secs", round(time.time() - t0, 1))
p["sport"] = sport.upper(); p.to_parquet(f"{HERE}/data/{tag}_{sport}.parquet")
# NBA benchmark is the honest version (no hindsight on who sat)
base = pd.read_parquet(f"{HERE}/data/baseline_{'nba_known' if sport == 'nba' else sport}.parquet")
start = 2017 if sport == "cbb" else 2018
common = set(p.game_id) & set(base.game_id)
pf = p[p.game_id.isin(common) & (p.season >= start)]; bf = base[base.game_id.isin(common) & (base.season >= start)]
# home/away orientation may differ: align the frame to the baseline's home team
bmap = bf.set_index("game_id").home
flip = pf.home.values != bmap.loc[pf.game_id].values
for a, b in (("pred_h", "pred_a"), ("act_h", "act_a"), ("home", "away")):
    pf.loc[flip, [a, b]] = pf.loc[flip, [b, a]].values
pf = pf.merge(bf[["game_id", "line_margin", "line_total"]], on="game_id")
print("flipped", int(flip.sum()), "of", len(pf))
print(pd.DataFrame({"today": H.score(bf), "frame": H.score(pf)}).to_string())
for loss in ("team_abs", "margin_abs", "total_abs"):
    print(H.compare(bf, pf, loss))
