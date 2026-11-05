"""CFB Model A on J's yardstick (2026-10-31): team scores vs the closing implied team totals (total/spread split; posted 2026
team totals sit within ~0.5 of the split). Test 2022-25 lined games (fit 2014-21). v3 (desk, unblended) vs v4 means.
Then the drive simulator on v4 means: 80% bands, exact-score log score vs a normal bell, team-total over Brier, win calibration,
key numbers. python3 model_a_test.py [N sims] -> model_a_test.json (about 4 min at N=1000)"""
import sys, json, numpy as np, pandas as pd
from scipy import stats
from lib import *
N = int(sys.argv[1]) if len(sys.argv) > 1 else 1000
fit = pd.read_parquet(f"{HERE}/data/fit_feat.parquet"); te = pd.read_parquet(f"{HERE}/data/te_feat.parquet")
out = {"n_test_games": len(te)}
m4, t4 = v4(fit, te); te["m4"], te["t4"] = m4, t4
te["h4"], te["a4"] = (t4 + m4) / 2, (t4 - m4) / 2; te["h3"], te["a3"] = te.pred_h, te.pred_a
ok = te.line_close.notna() & te.total_close.notna()
te["ih"], te["ia"] = (te.total_close + te.line_close) / 2, (te.total_close - te.line_close) / 2
def yard(z, tag):
    a = np.r_[z.act_h, z.act_a]; L = np.r_[z.ih, z.ia]; r = {}
    for nm, h, aa in (("v3", "h3", "a3"), ("v4", "h4", "a4")):
        p = np.r_[z[h], z[aa]]; e = np.abs(p - a); el = np.abs(L - a); pos = a > 0
        r[nm] = {"MAE": round(float(e.mean()), 3), "APE": round(float((e[pos] / a[pos]).mean()), 4), "closer_than_line": round(float((e < el).mean()), 3),
                 "minus_line_mean": round(float((p - L).mean()), 2)}
    r["line"] = {"MAE": round(float(np.abs(L - a).mean()), 3), "APE": round(float((np.abs(L - a)[a > 0] / a[a > 0]).mean()), 4)}
    r["game_total_MAE"] = {"v3": round(float(np.abs(z.model_total - z["at"]).mean()), 3), "v4": round(float(np.abs(z.t4 - z["at"]).mean()), 3), "line": round(float(np.abs(z.total_close - z["at"]).mean()), 3)}
    r["margin_MAE"] = {"v3": round(float(np.abs(z.model_margin - z.am).mean()), 3), "v4": round(float(np.abs(z.m4 - z.am).mean()), 3), "line": round(float(np.abs(z.line_close - z.am).mean()), 3)}
    r["n_team_scores"] = len(a); print(tag, json.dumps(r)); return r
Z = te[ok]
out["yardstick_2022_25"] = yard(Z, "all")
out["yardstick_by_season"] = {int(s): yard(g, str(s)) for s, g in Z.groupby("season")}
out["yardstick_wk1_3"] = yard(Z[Z.wk <= 3], "wk1-3"); out["yardstick_wk4p"] = yard(Z[Z.wk > 3], "wk4+")
# margin slope on the fit set (sim SHRINK), residual sds
mf, tf = v4(fit, fit); X = np.c_[np.ones(len(fit)), mf]; b = np.linalg.lstsq(X, fit.am.values, rcond=None)[0]
SH = round(float(b[1]), 3); out["fit_margin_slope"] = SH
sdm, sdt = float(np.std(fit.am - mf)), float(np.std(fit["at"] - tf)); sdh = float(np.std(np.r_[fit.act_h - (tf + mf) / 2, fit.act_a - (tf - mf) / 2]))
out["normal_sd"] = {"margin": round(sdm, 2), "total": round(sdt, 2), "team": round(sdh, 2)}
# ---------------- drive sim on v4 means
S = Sim(); rows = []
for i, g in enumerate(Z.itertuples()):
    h, a = S.run(g.h4, g.a4, N, SH); m, t = h - a, h + a
    rows.append(dict(season=g.season, wk=g.wk, am=g.am, at=g.at, ah=g.act_h, aa=g.act_a, ih=g.ih, ia=g.ia, m4=g.m4, t4=g.t4, h4=g.h4, a4=g.a4,
                     p_home=(m > 0).mean() + 0.5 * (m == 0).mean(), h10=np.quantile(h, .1), h90=np.quantile(h, .9), a10=np.quantile(a, .1), a90=np.quantile(a, .9),
                     m10=np.quantile(m, .1), m90=np.quantile(m, .9), t10=np.quantile(t, .1), t90=np.quantile(t, .9),
                     lp_h=np.log(max((h == g.act_h).mean(), 1 / (5 * N))), lp_a=np.log(max((a == g.act_a).mean(), 1 / (5 * N))),
                     lp_m=np.log(max((m == g.am).mean(), 1 / (5 * N))),
                     p_hover=(h > g.ih).mean() + 0.5 * (h == g.ih).mean(), p_aover=(a > g.ia).mean() + 0.5 * (a == g.ia).mean(),
                     p3=(np.abs(m) == 3).mean(), p7=(np.abs(m) == 7).mean(), sim_h=h.mean(), sim_a=a.mean(), sd_m=m.std(), sd_t=t.std(), sd_h=h.std()))
    if i % 500 == 0: print(i, flush=True)
R = pd.DataFrame(rows); R.to_parquet(f"{HERE}/data/sim_te.parquet")
cov = lambda lo, hi, x: float(((R[x] >= R[lo]) & (R[x] <= R[hi])).mean())
sim = {"N": N, "shrink": SH, "mean_bias_team": round(float((np.r_[R.sim_h, R.sim_a] - np.r_[R.h4, R.a4]).mean()), 2),
       "band80": {"home": round(cov("h10", "h90", "ah"), 3), "away": round(cov("a10", "a90", "aa"), 3), "margin": round(cov("m10", "m90", "am"), 3), "total": round(cov("t10", "t90", "at"), 3)},
       "sd": {"margin": round(float(R.sd_m.mean()), 2), "total": round(float(R.sd_t.mean()), 2), "team": round(float(R.sd_h.mean()), 2)}}
nh = np.r_[stats.norm.logpdf(R.ah, R.h4, sdh), stats.norm.logpdf(R.aa, R.a4, sdh)]
sim["team_score_log"] = {"sim": round(float(np.r_[R.lp_h, R.lp_a].mean()), 3), "normal": round(float(nh.mean()), 3)}
sim["margin_log"] = {"sim": round(float(R.lp_m.mean()), 3), "normal": round(float(stats.norm.logpdf(R.am, R.m4, sdm).mean()), 3)}
# team-total over Brier (J yardstick market): sim vs normal vs coin
yo = np.r_[(R.ah > R.ih), (R.aa > R.ia)].astype(float); push = np.r_[R.ah == R.ih, R.aa == R.ia]
ps = np.r_[R.p_hover, R.p_aover]; pn = np.r_[stats.norm.sf(R.ih, R.h4, sdh), stats.norm.sf(R.ia, R.a4, sdh)]
sim["team_total_over_brier"] = {"sim": round(float(((ps - yo) ** 2)[~push].mean()), 4), "normal": round(float(((pn - yo) ** 2)[~push].mean()), 4), "coin": 0.25}
y = (R.am > 0).astype(float); sim["win_brier"] = {"sim": round(float(((R.p_home - y) ** 2).mean()), 4), "normal": round(float(((stats.norm.cdf(R.m4 / sdm) - y) ** 2).mean()), 4)}
R["pb"] = pd.cut(R.p_home, [0, .1, .2, .3, .4, .5, .6, .7, .8, .9, 1])
sim["win_calibration"] = {str(k): [round(float(v.p_home.mean()), 3), round(float((v.am > 0).mean()), 3), int(len(v))] for k, v in R.groupby("pb", observed=True)}
sim["key_numbers"] = {"p3_sim": round(float(R.p3.mean()), 3), "p3_actual": round(float((R.am.abs() == 3).mean()), 3), "p7_sim": round(float(R.p7.mean()), 3), "p7_actual": round(float((R.am.abs() == 7).mean()), 3)}
out["sim"] = sim
json.dump(out, open(f"{HERE}/model_a_test.json", "w"), indent=1); print(json.dumps(sim, indent=1))
