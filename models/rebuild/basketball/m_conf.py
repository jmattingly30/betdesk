"""Module test: CBB conference layer (shared conference offense/defense component ratings, learned from games
between conferences). Usage: python3 m_conf.py pc0 carry_c"""
import sys, os, json, numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE); sys.path.insert(0, os.path.dirname(HERE))
import harness as H, engine as E, ctx
pc0, cc = float(sys.argv[1]), float(sys.argv[2])
CM = sys.argv[3].split(",") if len(sys.argv) > 3 else None
tg = pd.read_parquet(f"{HERE}/data/teamgames_cbb.parquet")
G = pd.read_parquet(f"{HERE}/../../cbb/data/cbb_games.parquet")[["game_id", "home", "away", "home_conference_id", "away_conference_id"]]
cf = pd.concat([G[["game_id", "home", "home_conference_id"]].set_axis(["game_id", "team", "c"], axis=1),
                G[["game_id", "away", "away_conference_id"]].set_axis(["game_id", "team", "c"], axis=1)]).dropna()
cf["team"] = cf.team.astype(int); cf["game_id"] = cf.game_id.astype(tg.game_id.dtype)
codes = {c: i for i, c in enumerate(sorted(cf.c.unique()))}
cf["conf"] = cf.c.map(codes)
tg = tg.merge(cf[["game_id", "team", "conf"]].drop_duplicates(["game_id", "team"]), on=["game_id", "team"], how="left")
print("conf matched", round(tg.conf.notna().mean(), 3), "n_conf", len(codes), flush=True)
tg["conf"] = tg.conf.fillna(-1).astype(int)
prm = {**E.DEFAULT, **json.load(open(f"{HERE}/data/tuned_stack_cbb.json"))}
mods = ["prior", "avail"]
pa, _, _ = ctx.model(tg, prm, mods, 2016)
pb, _, _ = ctx.model(tg, {**prm, "n_conf": len(codes), "pc0": pc0, "carry_c": cc, "qc_off": 0.002, "conf_comp": CM}, mods, 2016)
base = pd.read_parquet(f"{HERE}/data/baseline_cbb.parquet")
keep = lambda d: d[(d.season > 2016) & d.game_id.isin(base.game_id)]
A, B = keep(pa), keep(pb)
tune = lambda d: d[(d.season <= 2016) & (d.season >= 2012)]
print("tuning seasons margin", round(H.score(tune(pa))["margin_mae"], 4), "->", round(H.score(tune(pb))["margin_mae"], 4))
print(pd.DataFrame({"stack": H.score(A), "stack+conf": H.score(B)}).loc[["n", "team_mae", "team_bias", "margin_mae", "total_mae"]].to_string())
out = {l: H.compare(A, B, l) for l in ("team_abs", "margin_abs", "total_abs")}
for v in out.values(): print(v)
json.dump(out, open(f"{HERE}/data/m_conf_{pc0}_{cc}_{sys.argv[3] if CM else 'all'}.json", "w"), default=float)
