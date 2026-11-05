"""NBA Model A tests on J's yardstick (implied closing team totals; also the 2026 DraftKings opener's implied team totals).
  base = rebuild stack means (what the desk shows today as Model A)
  fix  = base + component fix (comp_test.py: our 3P% and FT-rate terms overweighted, OREB/3PA underweighted), walk-forward
Also: measured miss (margin / total / team sd) for the sim, and 80% range coverage of the sim on 2025-26.  -> model_a_test.json"""
import json, numpy as np, pandas as pd, sys, os
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from lib import sim_game, C
M = "/mnt/project-files/sports-analytics/"
s = pd.read_parquet(M + "models/rebuild/basketball/data/stack_nba.parquet")
G = pd.read_parquet(M + "models/nba-team/data/games.parquet", columns=["game_id", "close", "total_close"])
d = s.merge(G, on="game_id", how="left"); d = d[(d.season >= 2016) & (d.season <= 2026) & d.act_h.notna()].copy()
X = lambda z, side, mu: np.c_[np.ones(len(z)), np.column_stack([z[f"{side}_{c}"] * z.poss_hat - mu[c] for c in C])]
def fitc(tr, lam=50.0):
    mu = {c: float(pd.concat([tr[f"h_{c}"] * tr.poss_hat, tr[f"a_{c}"] * tr.poss_hat]).mean()) for c in C}
    A = np.vstack([X(tr, "h", mu), X(tr, "a", mu)]); y = np.r_[tr.act_h - tr.pred_h, tr.act_a - tr.pred_a]
    P = np.eye(A.shape[1]) * lam; P[0, 0] = 0; return np.linalg.solve(A.T @ A + P, A.T @ y), mu
d["fix_h"], d["fix_a"] = d.pred_h, d.pred_a
for se in range(2019, 2027):
    b, mu = fitc(d[d.season < se]); m = d.season == se
    d.loc[m, "fix_h"] = d.pred_h[m] + np.clip(X(d[m], "h", mu) @ b, -4, 4); d.loc[m, "fix_a"] = d.pred_a[m] + np.clip(X(d[m], "a", mu) @ b, -4, 4)
t = d[(d.season >= 2019) & d.close.notna()].copy()
t["ih"], t["ia"] = (t.total_close + t.close) / 2, (t.total_close - t.close) / 2
def yard(z, ph, pa, ih, ia):
    e = np.r_[(z[ph] - z.act_h).abs(), (z[pa] - z.act_a).abs()]; l = np.r_[(z[ih] - z.act_h).abs(), (z[ia] - z.act_a).abs()]
    act = np.r_[z.act_h, z.act_a]
    return dict(n=len(e), team_MAE=round(float(e.mean()), 3), line_team_MAE=round(float(l.mean()), 3), team_APE=round(float((e / act).mean() * 100), 2),
                line_team_APE=round(float((l / act).mean() * 100), 2), closer_share=round(float((e < l).mean()), 3))
out = {"yardstick_close_2019_2026": {"base": yard(t, "pred_h", "pred_a", "ih", "ia"), "fix": yard(t, "fix_h", "fix_a", "ih", "ia")},
       "by_season": {int(se): {"base": yard(z, "pred_h", "pred_a", "ih", "ia")["team_MAE"], "fix": yard(z, "fix_h", "fix_a", "ih", "ia")["team_MAE"],
                               "line": yard(z, "pred_h", "pred_a", "ih", "ia")["line_team_MAE"]} for se, z in t.groupby("season")}}
# totals (game) MAE too
for k, (h, a) in {"base": ("pred_h", "pred_a"), "fix": ("fix_h", "fix_a")}.items():
    out["yardstick_close_2019_2026"][k]["game_total_MAE"] = round(float(((t[h] + t[a]) - (t.act_h + t.act_a)).abs().mean()), 3)
out["yardstick_close_2019_2026"]["line_game_total_MAE"] = round(float((t.total_close - (t.act_h + t.act_a)).abs().mean()), 3)
# 2026 DK opener implied team totals
sys.argv = [sys.argv[0]]; exec(open(f"{HERE}/model_b_test.py").read().split("corr_home =")[0].split("# ---- 2026 DK opener frame")[0].split("sbr = pd.read_parquet")[0])
L = pd.read_csv(M + "op/us/lines_nba.csv"); L = L[L.book == "draftkings"].copy(); L["h"], L["a"] = L.p1.map(TEAM), L.p2.map(TEAM)
L["day"] = pd.to_datetime(L.start).dt.tz_convert("America/New_York").dt.strftime("%Y-%m-%d"); d["day"] = pd.to_datetime(d.date).dt.strftime("%Y-%m-%d")
k = d[d.season == 2026].merge(L, left_on=["home", "away", "day"], right_on=["h", "a", "day"])
k["ih"], k["ia"] = (k.total_open - k.spread_open) / 2, (k.total_open + k.spread_open) / 2
out["yardstick_dk_open_2026"] = {"base": yard(k, "pred_h", "pred_a", "ih", "ia"), "fix": yard(k, "fix_h", "fix_a", "ih", "ia")}
# measured miss of the shipped means (fix) for the sim
em = (t.act_h - t.act_a) - (t.fix_h - t.fix_a); et = (t.act_h + t.act_a) - (t.fix_h + t.fix_a)
out["sd_margin"], out["sd_total"], out["sd_team"] = round(float(em.std()), 2), round(float(et.std()), 2), round(float(np.r_[t.act_h - t.fix_h, t.act_a - t.fix_a].std()), 2)
# sim coverage on a 600-game sample of 2025-26
z = t[t.season >= 2025].sample(600, random_state=1); cov = {"team": [], "total": [], "margin": []}; ot = []
for i, r in enumerate(z.itertuples()):
    H, A, po = sim_game(r.fix_h, r.fix_a, r.poss_hat, out["sd_margin"], out["sd_total"], n=1500, seed=i); ot.append(po)
    q = lambda x, v: np.percentile(x, 10) <= v <= np.percentile(x, 90)
    cov["team"] += [q(H, r.act_h), q(A, r.act_a)]; cov["total"].append(q(H + A, r.act_h + r.act_a)); cov["margin"].append(q(H - A, r.act_h - r.act_a))
out["sim_80pct_coverage_2025_26"] = {k_: round(float(np.mean(v)), 3) for k_, v in cov.items()}; out["sim_ot_rate"] = round(float(np.mean(ot)), 3)
# final live component fix: fit on 2016-2026
b, mu = fitc(d)
out["final"] = {"const": round(float(b[0]), 4), "b": dict(zip(C, [round(float(x), 4) for x in b[1:]])), "mu": {c: round(v, 4) for c, v in mu.items()}, "cap": 4.0}
json.dump(out, open(f"{HERE}/model_a_test.json", "w"), indent=1); print(json.dumps(out, indent=1))
