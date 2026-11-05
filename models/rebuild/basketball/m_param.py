"""Module test for engine-parameter modules (e.g. luck split = d_scale) on top of kept context modules.
Usage: python3 m_param.py sport params.json kept_modules '{"d_scale": {...}}' label"""
import sys, os, json, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE); sys.path.insert(0, os.path.dirname(HERE))
import harness as H, engine as E, ctx
sport, pf, kept, over, label = sys.argv[1], sys.argv[2], [m for m in sys.argv[3].split(",") if m], json.loads(sys.argv[4]), sys.argv[5]
prm = {**E.DEFAULT, **json.load(open(pf))}
fit_to = 2017 if sport == "nba" else 2016
tg = pd.read_parquet(f"{HERE}/data/teamgames_{sport}.parquet")
pa, _, _ = ctx.model(tg, prm, kept, fit_to)
pb, _, _ = ctx.model(tg, {**prm, **over}, kept, fit_to)
base = pd.read_parquet(f"{HERE}/data/baseline_{'nba_known' if sport == 'nba' else sport}.parquet")
keep = lambda d: d[(d.season > fit_to) & d.game_id.isin(base.game_id)]
A, B = keep(pa), keep(pb)
tune = lambda d: d[(d.season <= fit_to) & (d.season > d.season.min())]
print(label, "fit seasons team MAE", round(H.score(tune(pa))["team_mae"], 4), "->", round(H.score(tune(pb))["team_mae"], 4))
print(pd.DataFrame({"before": H.score(A), label: H.score(B)}).loc[["n", "team_mae", "team_bias", "margin_mae", "total_mae"]].to_string())
out = {l: H.compare(A, B, l) for l in ("team_abs", "margin_abs", "total_abs")}
for v in out.values(): print(v)
json.dump(out, open(f"{HERE}/data/param_{sport}_{label}.json", "w"), default=float)
