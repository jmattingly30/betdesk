"""Test layer 2: run the tuned frame and the classic goals-Kalman through the same calibrated simulator on the
test seasons (2022+), so any difference in win log loss / CRPS comes from the ratings, not the simulator.
Usage: python3 simtest.py frame_tag [nsim]"""
import sys, os, json, numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE); sys.path.insert(0, os.path.dirname(HERE))
import harness as H, sim as S
tag = sys.argv[1]; nsim = int(sys.argv[2]) if len(sys.argv) > 2 else 400
cal = json.load(open(f"{HERE}/data/simcal_nhl.json")); RS = 0.97
Y0, Y1 = int(os.environ.get("Y0", 2022)), int(os.environ.get("Y1", 2099)); cal["rate_sd"] = float(os.environ.get("RATE_SD", cal.get("rate_sd", 0)))
f = pd.read_parquet(f"{HERE}/data/{tag}_nhl.parquet"); f = f[(f.season >= Y0) & (f.season <= Y1)].reset_index(drop=True)
base = pd.read_parquet(f"{HERE}/data/frame1_nhl.parquet").set_index("game_id").loc[f.game_id].reset_index()
b = pd.read_parquet(f"{HERE}/data/baseline_nhl_goals.parquet").set_index("game_id").loc[f.game_id].reset_index()
# simulator input = non-empty-net scoring rate; give the baseline the same empty-net/shootout allowance as the frame
off_h = base.pred_h - (base.evg_h + base.ppg_h + 0.075); off_a = base.pred_a - (base.evg_a + base.ppg_a + 0.075)
out = {}
for name, d in (("baseline", b.assign(playoff=f.playoff.values)), (tag, f)):
    d = d.assign(rate_h=(d.pred_h - off_h.values).clip(lower=0.5) * RS, rate_a=(d.pred_a - off_a.values).clip(lower=0.5) * RS)
    q, _, _ = S.simulate(d, nsim=nsim, seed=7, **cal)
    out[name] = q
    print(name, H.score(q))
if os.environ.get("NOLOG"): sys.exit()
for loss in ("win_ll", "margin_crps", "total_crps", "team_abs", "margin_abs", "total_abs"):
    r = H.compare(out["baseline"], out[tag], loss)
    print(loss, r)
    H.log_result("NHL", "layer2 sim", f"{tag} ratings through sim", "goals-Kalman through same sim", r,
                 "KEEP" if r["delta"] < 0 and r["p"] < 0.10 else ("HOLD" if r["delta"] < 0 else "LOSES"),
                 f"test 2022+, nsim {nsim}, simcal {cal}")
out[tag].to_parquet(f"{HERE}/data/{tag}_sim_nhl.parquet")
