"""Test the drive simulator's spread on 2018+ (margin/total CRPS, 80% coverage, win log loss, key-number shape).
Usage: python3 simtest.py tag [nsim] [clock_scale]"""
import sys, os, numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE); sys.path.insert(0, os.path.dirname(HERE))
import harness as H, sim as S
tag = sys.argv[1]; nsim = int(sys.argv[2]) if len(sys.argv) > 2 else 400; cs = float(sys.argv[3]) if len(sys.argv) > 3 else 1.0; ks = float(sys.argv[4]) if len(sys.argv) > 4 else 0.0; kl = float(sys.argv[5]) if len(sys.argv) > 5 else 0.0; tsd = float(sys.argv[6]) if len(sys.argv) > 6 else 0.0
Y0, Y1 = int(os.environ.get("Y0", 2018)), int(os.environ.get("Y1", 2099))
p = pd.read_parquet(f"{HERE}/data/{tag}_nfl.parquet"); p = p[(p.season >= Y0) & (p.season <= Y1)].reset_index(drop=True)
q, mh = S.simulate(p, nsim=nsim, clock_scale=cs, k_score=ks, k_late=kl, th_sd=tsd)
print("sim mean vs projection", round((q.sim_h - q.pred_h).mean(), 3), round((q.sim_a - q.pred_a).mean(), 3), "| sd margin", round(q.sd_margin.mean(), 2), "sd total", round(q.sd_total.mean(), 2))
s = H.score(q); print({k: s[k] for k in ("margin_crps", "margin_cover80", "total_crps", "total_cover80", "win_logloss", "win_brier")})
# normal approximation with the residual sd from tuning seasons, for comparison
n = q.assign(sd_margin=13.4, sd_total=13.4, p_home=np.nan).drop(columns=["p_home"])
s2 = H.score(n); print("normal sd 13.4", {k: s2[k] for k in ("margin_crps", "margin_cover80", "total_crps", "total_cover80", "win_logloss")})
act = (q.act_h - q.act_a).clip(-40, 40).astype(int).value_counts(normalize=True)
pmf = mh.mean(0)
print("margin share  |m|=3: sim", round(pmf[37] + pmf[43], 3), "act", round(act.get(3, 0) + act.get(-3, 0), 3),
      "| |m|=7: sim", round(pmf[33] + pmf[47], 3), "act", round(act.get(7, 0) + act.get(-7, 0), 3), "| tie: sim", round(pmf[40], 4), "act", round(act.get(0, 0), 4))
q.to_parquet(f"{HERE}/data/{tag}_sim_nfl.parquet")
