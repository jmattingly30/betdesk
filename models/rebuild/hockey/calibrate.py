"""Calibrate the hockey simulator on tuning seasons 2017-2021: match the league's overtime share, empty-net goals
and margin distribution, and minimise win log loss. Usage: python3 calibrate.py frame_tag"""
import sys, os, json, itertools, numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE); sys.path.insert(0, os.path.dirname(HERE))
import harness as H, sim as S
p = pd.read_parquet(f"{HERE}/data/{sys.argv[1]}_nhl.parquet")
p = p[(p.season >= 2017) & (p.season <= 2021)].sample(1000, random_state=2)
p["rate_h"] = p.evg_h + p.ppg_h + 0.075; p["rate_a"] = p.evg_a + p.ppg_a + 0.075
tg = pd.read_parquet(f"{HERE}/data/teamgames_nhl.parquet"); h = tg[(tg.ha == "home") & tg.game_id.isin(p.game_id)]
tgt_ot, tgt_en = h.ot.mean(), (h.g_en.mean() + h.o_g_en.mean()) / 2
act = (p.act_h - p.act_a).clip(-10, 10).value_counts(normalize=True).reindex(range(-10, 11), fill_value=0).values
grid = dict(s_eff=[0.04, 0.08, 0.12], pull1=[1.9, 2.4], pull_boost=[1.7, 2.1, 2.5], en_rate=[16.0, 20.0], ot_mult=[2.2])
rows = []
for vals in itertools.product(*grid.values()):
    kw = dict(zip(grid, vals)); kw["pull2"] = kw["pull1"] + 1.0
    RS = 0.97   # rates are fit on all game states; trim so simulated means match the projection
    q, _, mh = S.simulate(p.assign(rate_h=p.rate_h * RS, rate_a=p.rate_a * RS), nsim=150, **kw)
    m = q.act_h != q.act_a; y = (q.act_h > q.act_a)[m].astype(float); pr = q.p_home[m].clip(1e-3, 1 - 1e-3)
    ll = -np.mean(y * np.log(pr) + (1 - y) * np.log(1 - pr))
    pmf = mh.mean(0); err = np.sum((pmf - act) ** 2) * 100
    r = {**kw, "logloss": ll, "pmf_err": err, "ot": q.p_ot.mean(), "en": q.en_goals.mean() / 2, "bias": (q.sim_h + q.sim_a - q.act_h - q.act_a).mean() / 2}
    r["score"] = ll + 0.02 * err + 2 * abs(r["ot"] - tgt_ot) + 2 * abs(r["en"] - tgt_en)
    rows.append(r)
r = pd.DataFrame(rows).sort_values("score"); r.to_csv(f"{HERE}/data/calib_nhl.csv", index=False)
print("targets ot", round(tgt_ot, 3), "en", round(tgt_en, 3)); print(r.head(8).round(4).to_string())
json.dump({k: float(r.iloc[0][k]) for k in list(grid) + ["pull2"]}, open(f"{HERE}/data/simcal_nhl.json", "w"))
