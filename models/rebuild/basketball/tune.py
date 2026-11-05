"""Coordinate search of frame settings on tuning seasons only (never the test seasons)."""
import sys, os, json, itertools, numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE); sys.path.insert(0, os.path.dirname(HERE))
import harness as H, engine as E
sport = sys.argv[1]; seasons = [int(s) for s in sys.argv[2].split(",")]
tg = pd.read_parquet(f"{HERE}/data/teamgames_{sport}.parquet")
tg = tg[tg.season <= max(seasons)]
grid = {"tau": [0.1, 0.2, 0.3, 0.45], "q": [0.0, 0.00003, 0.0001, 0.0003], "carry": [0.6, 0.8, 0.9, 1.0], "q_off": [0.003, 0.01, 0.03],
        "p0": [0.04, 0.08, 0.15], "pace_tau": [0.06, 0.08, 0.12, 0.16], "pace_q": [0.0, 0.00002, 0.00005, 0.0002],
        "mu_gain": [0.003, 0.01, 0.03], "hca_gain": [0.0005, 0.002, 0.006]}
best = json.load(open(sys.argv[3])) if len(sys.argv) > 3 else {}
def loss(prm):
    p, _ = E.run(tg, prm); p = p[p.season.isin(seasons)]
    s = H.score(p); return s["team_mae"], s
cur, s = loss(best); print("start", round(cur, 4), s["margin_mae"], s["total_mae"], flush=True)
for rnd in range(3):
    for k, vals in grid.items():
        for v in vals:
            if best.get(k, E.DEFAULT[k]) == v: continue
            trial = {**best, k: v}; l, s = loss(trial)
            if l < cur - 1e-4: cur, best = l, trial; print("better", k, v, round(l, 4), s["margin_mae"], s["total_mae"], flush=True)
    json.dump(best, open(f"{HERE}/data/tuned_{sport}.json", "w"))
print("final", best, round(cur, 4))
