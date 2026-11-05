"""NFL module test: offensive play-caller changes (J's rule: separate coordinator change from team change).
At each new season, a team whose play-caller changed carries less of last year's offense (CC instead of carry),
and if the new caller called plays elsewhere last season, a share W of that offense's rating travels with him.
Callers table: models/nfl-player/data/play_callers.csv (2016+). Usage: python3 m_caller.py CC W tag"""
import sys, os, json, numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE); sys.path.insert(0, os.path.dirname(HERE))
import harness as H, engine as E
CC, W, tag = float(sys.argv[1]), float(sys.argv[2]), sys.argv[3]
C = pd.read_csv(os.path.join(HERE, "../../nfl-player/data/play_callers.csv"))
caller = {(r.season, r.team): r.caller for r in C.itertuples()}
prm = json.load(open(f"{HERE}/data/tuned_nfl.json"))
OFF = [0, 1, 2, 3]   # sr, epp, xr, tor offense components travel with the caller
def season_start(R, season):
    c = prm["carry"]; prev = {v: t for (s, t), v in caller.items() if s == season - 1}
    snapO = R.O.copy()
    for t, i in R.idx.items():
        new, old = caller.get((season, t)), caller.get((season - 1, t))
        if new is None or old is None or new == old: continue
        R.O[i, OFF] *= CC / c                     # O already carried by c: rescale to CC
        R.PO[i, OFF] += 0.1 * R.var0[OFF]          # less sure about this offense
        if W > 0 and new in prev and prev[new] in R.idx:
            R.O[i, OFF] = (1 - W) * R.O[i, OFF] + W * snapO[R.idx[prev[new]], OFF]
T = pd.read_parquet(f"{HERE}/data/teamgames_nfl.parquet")
p, _ = E.run(T, prm, {"season_start": season_start}); p.to_parquet(f"{HERE}/data/{tag}_nfl.parquet")
b = pd.read_parquet(f"{HERE}/data/frame1_nfl.parquet")
for lo, nm in ((2017, "2017+"), (2018, "test 2018+")):
    for loss in ("team_abs", "margin_sq", "total_abs"):
        r = H.compare(b[b.season >= lo], p[p.season >= lo], loss); print(nm, loss, r)
