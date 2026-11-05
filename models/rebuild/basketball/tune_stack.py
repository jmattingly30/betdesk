"""Re-tune frame settings with the kept modules in place (tuning seasons only).
Objective: team MAE + 0.5 x margin MAE. Usage: python3 tune_stack.py sport seasons"""
import sys, os, json, numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE); sys.path.insert(0, os.path.dirname(HERE))
import harness as H, engine as E, ctx
sport = sys.argv[1]; seasons = [int(s) for s in sys.argv[2].split(",")]
mods = {"nba": ["lineup", "schedule"], "cbb": ["prior", "avail"]}[sport]
fit_to = max(seasons)
tg = pd.read_parquet(f"{HERE}/data/teamgames_{sport}.parquet"); tg = tg[tg.season <= fit_to]
best = {**json.load(open(f"{HERE}/data/tuned_{sport}.json")), "d_scale": {"p3": 0.3, "ftp": 0.05}}
grid = {"tau": [0.2, 0.3, 0.45, 0.6], "carry": [0.6, 0.7, 0.8, 0.9], "p0": [0.08, 0.15, 0.25, 0.4], "q": [0.0, 0.00001, 0.00003, 0.0001],
        "q_off": [0.003, 0.01, 0.03, 0.06], "hca_gain": [0.0005, 0.002], "mu_gain": [0.003, 0.01]}
def loss(prm):
    p, _, _ = ctx.model(tg, {**E.DEFAULT, **prm}, mods, fit_to); p = p[p.season.isin(seasons)]
    s = H.score(p); return s["team_mae"] + 0.5 * s["margin_mae"], s
cur, s = loss(best); print("start", round(cur, 4), s["team_mae"], s["margin_mae"], s["total_mae"], flush=True)
for rnd in range(2):
    for k, vals in grid.items():
        for v in vals:
            if best.get(k, E.DEFAULT[k]) == v: continue
            trial = {**best, k: v}; l, s = loss(trial)
            if l < cur - 1e-4: cur, best = l, trial; print("better", k, v, round(l, 4), s["team_mae"], s["margin_mae"], s["total_mae"], flush=True)
    json.dump(best, open(f"{HERE}/data/tuned_stack_{sport}.json", "w"))
print("final", best, round(cur, 4))
