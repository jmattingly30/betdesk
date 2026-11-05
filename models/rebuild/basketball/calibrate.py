"""Calibrate the possession simulator's spread on tuning seasons (score effects, shared shooting nights,
possession count noise, rate wobble). Usage: python3 calibrate.py sport frame_tag seasons
Grades margin + total CRPS of the simulated distribution (means from the sim itself)."""
import sys, os, json, itertools, numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE); sys.path.insert(0, os.path.dirname(HERE))
import harness as H, sim as S
sport, tag, seasons = sys.argv[1], sys.argv[2], [int(s) for s in sys.argv[3].split(",")]
p = pd.read_parquet(f"{HERE}/data/{tag}_{sport}.parquet")
p = p[p.season.isin(seasons)].dropna(subset=["pred_h"])
if len(p) > 3000: p = p.sample(3000, random_state=1)
grid = {"k_score": [0.0, 0.04, 0.07, 0.1], "w_shared": [0.0, 0.05, 0.09, 0.13], "poss_sd": [1.0, 2.5, 4.0], "wob": [0.0, 0.04]}
if sport == "cbb":   # college blowouts: wider score-effect range, fewer of the settled ones
    grid = {"k_score": [0.0, 0.07, 0.12, 0.18, 0.25], "w_shared": [0.0, 0.05, 0.1], "poss_sd": [1.0, 2.5], "wob": [0.0, 0.04]}
rows = []
for k, w, ps, wb in itertools.product(*grid.values()):
    q, _ = S.simulate(p, [wb] * 7, ps, nsim=300, k_score=k, w_shared=w)
    q = q.assign(pred_h=q.sim_h, pred_a=q.sim_a)
    s = H.score(q); s.update(k_score=k, w_shared=w, poss_sd=ps, wob=wb)
    rows.append(s); print({x: s[x] for x in ("k_score", "w_shared", "poss_sd", "wob", "margin_crps", "total_crps", "margin_cover80", "total_cover80", "team_bias")}, flush=True)
r = pd.DataFrame(rows); r["crps_sum"] = r.margin_crps + r.total_crps
r.to_csv(f"{HERE}/data/calib_{sport}.csv", index=False)
best = r.sort_values("crps_sum").iloc[0]
print("BEST", best.to_dict())
json.dump({c: float(best[c]) for c in grid}, open(f"{HERE}/data/simcal_{sport}.json", "w"))
