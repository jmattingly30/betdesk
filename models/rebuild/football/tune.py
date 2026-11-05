"""Tune the NFL frame on 2008-2017 (objective team MAE + 0.5 x margin RMSE); report 2018+ for the winner only."""
import sys, os, json, itertools, random, pandas as pd, numpy as np
from multiprocessing import Pool
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE); sys.path.insert(0, os.path.dirname(HERE))
import harness as H, engine as E
SPORT = os.environ.get("SPORT", "nfl"); TUNE = (2008, 2017) if SPORT == "nfl" else (2015, 2019)
if SPORT == "cfb": E.MAP.update(json.load(open(f"{HERE}/data/map_cfb.json")))
T = pd.read_parquet(f"{HERE}/data/teamgames_{SPORT}.parquet")
def one(kw):
    p, _ = E.run(T, kw); s = H.score(p[(p.season >= TUNE[0]) & (p.season <= TUNE[1])])
    return {**{k: (str(v) if isinstance(v, tuple) else v) for k, v in kw.items()}, "team": s["team_mae"], "mrmse": s["margin_rmse"], "total": s["total_mae"], "obj": s["team_mae"] + 0.5 * s["margin_rmse"]}
random.seed(int(sys.argv[1]) if len(sys.argv) > 1 else 0)
space = dict(p0=[8, 16, 32], carry=[0.85, 0.9, 0.95], q_off=[0.3, 0.6, 1.2], q=[0.01, 0.02, 0.04],
             s_tor=[0.1, 0.3], s_start=[0.3, 1], s_fin=[0.5, 1, 2], s_drv=[0.25, 0.5], tau_epp=[0], hca_gain=[0.002, 0.005])
grid = []
for _ in range(int(sys.argv[2]) if len(sys.argv) > 2 else 80):
    c = {k: random.choice(v) for k, v in space.items()}
    kw = dict(p0=c["p0"], carry=c["carry"], q_off=c["q_off"], q=c["q"], hca_gain=c["hca_gain"],
              slow=(1, 1, 1, c["s_tor"], c["s_start"], c["s_drv"], c["s_fin"]), tau=(0, c["tau_epp"], 0, 0, 0, 0, 0))
    grid.append(kw)
with Pool(4) as pool: rows = pool.map(one, grid)
r = pd.DataFrame(rows).sort_values("obj"); print(r.head(10).to_string())
r.to_csv(f"{HERE}/data/tune_{SPORT}_{sys.argv[1] if len(sys.argv) > 1 else 0}.csv", index=False)
