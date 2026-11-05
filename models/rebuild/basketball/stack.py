"""Current kept stack for a sport, graded on test seasons three ways: formula means, simulated means (layer 2 with
calibrated score effects), and today's model. Usage: python3 stack.py sport"""
import sys, os, json, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE); sys.path.insert(0, os.path.dirname(HERE))
import harness as H, engine as E, ctx, sim as S
sport = sys.argv[1]
KEPT = {"nba": (["lineup", "schedule"], {"d_scale": {"p3": 0.3, "ftp": 0.05}}, "data/tuned_nba.json"),
        "cbb": (["prior", "avail"], {"d_scale": {"p3": 0.3, "ftp": 0.05}}, "data/tuned_cbb.json")}[sport]
mods, over, pf = KEPT
if os.path.exists(f"{HERE}/data/tuned_stack_{sport}.json") and "--old" not in sys.argv: pf = f"data/tuned_stack_{sport}.json"
prm = {**E.DEFAULT, **json.load(open(f"{HERE}/{pf}")), **over}
fit_to = 2017 if sport == "nba" else 2016
tg = pd.read_parquet(f"{HERE}/data/teamgames_{sport}.parquet")
p, _, _ = ctx.model(tg, prm, mods, fit_to)
p.to_parquet(f"{HERE}/data/stack_{sport}.parquet"); print("params", pf)
base = pd.read_parquet(f"{HERE}/data/baseline_{'nba_known' if sport == 'nba' else sport}.parquet")
t = p[(p.season > fit_to) & p.game_id.isin(base.game_id)].copy()
cal = json.load(open(f"{HERE}/data/simcal_{sport}.json"))
q, _ = S.simulate(t, [cal["wob"]] * 7, cal["poss_sd"], nsim=300, k_score=cal["k_score"], w_shared=cal["w_shared"])
q = q.assign(pred_h=q.sim_h, pred_a=q.sim_a); q.to_parquet(f"{HERE}/data/stack_sim_{sport}.parquet")
bt = base.set_index("game_id").loc[t.game_id].reset_index()
rows = ["n", "team_mae", "team_bias", "margin_mae", "margin_rmse", "total_mae", "margin_crps", "margin_cover80", "total_crps", "total_cover80", "win_logloss",
        "margin_mae_close", "total_mae_close"]
print(pd.DataFrame({"today": H.score(bt), "stack formula": H.score(t), "stack sim": H.score(q)}).reindex(rows).to_string())
for l in ("team_abs", "margin_abs", "total_abs"):
    print("sim vs today", H.compare(bt, q, l)); print("sim vs formula", H.compare(t, q, l))
