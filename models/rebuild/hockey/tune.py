"""Coordinate search of hockey frame settings on tuning seasons 2017-2021 (test seasons 2022+ untouched).
Objective: team MAE + 0.5 x margin RMSE (hockey margins are small; RMSE separates better)."""
import sys, os, json, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE); sys.path.insert(0, os.path.dirname(HERE))
import harness as H, engine as E
tg = pd.read_parquet(f"{HERE}/data/teamgames_nhl.parquet"); tg = tg[tg.season <= 2021]
best = json.load(open(f"{HERE}/data/tuned_nhl.json")) if os.path.exists(f"{HERE}/data/tuned_nhl.json") else {}
grid = {"carry": [0.4, 0.6, 0.75, 0.9], "p0": [0.005, 0.01, 0.02, 0.04], "q": [0.0, 0.00001, 0.00003, 0.0001], "q_off": [0.003, 0.01, 0.03],
        "fin_scale": [0.05, 0.15, 0.3, 0.6], "g_p0": [0.002, 0.005, 0.01, 0.02], "g_carry": [0.5, 0.7, 0.9], "g_scale": [0.5, 1.0, 2.0],
        "tau": [(0.05, 0.2, 0.15, 0.05), (0.08, 0.25, 0.2, 0.1), (0.12, 0.35, 0.3, 0.15), (0.16, 0.45, 0.4, 0.2)]}
def loss(prm):
    p, _ = E.run(tg, prm); p = p[p.season >= 2017]; s = H.score(p)
    return s["team_mae"] + 0.5 * s["margin_rmse"], s
cur, s = loss(best); print("start", round(cur, 5), s["team_mae"], s["margin_rmse"], flush=True)
for k, vals in grid.items():
    for v in vals:
        if best.get(k, E.DEFAULT[k]) == v: continue
        l, s = loss({**best, k: v})
        if l < cur - 1e-5: cur, best = l, {**best, k: v}; print("better", k, v, round(l, 5), s["team_mae"], s["margin_rmse"], flush=True)
    json.dump(best, open(f"{HERE}/data/tuned_nhl.json", "w"))
print("final", best, round(cur, 5))
