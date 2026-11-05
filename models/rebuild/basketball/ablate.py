"""Ablation: the full kept stack vs the same stack with one piece removed, on test seasons.
A piece earns its place if removing it makes things worse. Usage: python3 ablate.py sport piece"""
import sys, os, json, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE); sys.path.insert(0, os.path.dirname(HERE))
import harness as H, engine as E, ctx
sport, piece = sys.argv[1], sys.argv[2]
fit_to = 2017 if sport == "nba" else 2016
tg, nc = ctx.load_tg(sport)
mods = ctx.STACK[sport][0]; prm = ctx.stack_params(sport, nc)
full = os.path.join(HERE, f"data/ablate_full_{sport}.parquet")
if piece == "full" or not os.path.exists(full):
    pf, _, _ = ctx.model(tg, prm, mods, fit_to); pf.to_parquet(full)
    if piece == "full": sys.exit()
pf = pd.read_parquet(full)
m2, p2 = list(mods), dict(prm)
if piece in mods: m2.remove(piece)
elif piece == "luck": p2["d_scale"] = None
elif piece == "conf": p2["n_conf"] = 0
pa, _, _ = ctx.model(tg, p2, m2, fit_to)
base = pd.read_parquet(f"{HERE}/data/baseline_{'nba_known' if sport == 'nba' else sport}.parquet")
keep = lambda d: d[(d.season > fit_to) & d.game_id.isin(base.game_id)]
A, B = keep(pa), keep(pf)
res = {l: H.compare(A, B, l) for l in ("team_abs", "margin_abs", "total_abs")}
print(sport, "without", piece, "-> full stack:", {l: (r["a"], r["b"], r["delta"], round(r["p"], 4)) for l, r in res.items()}, flush=True)
json.dump(res, open(f"{HERE}/data/ablate_{sport}_{piece}.json", "w"), default=float)
