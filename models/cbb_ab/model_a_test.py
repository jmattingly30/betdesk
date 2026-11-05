"""Model A CBB tests (J 2026-11-02): does the possession-sim end-game state (fouling when the margin is 4-12 at 2:00) and
overtime chance improve CBB v2 team scores?  Fit 2013-2019, test 2020-2026. Yardstick (J 2026-10-31): closing team totals;
implied from closing spread + total (research showed implied = real team-total closes, 8.07 vs 8.07 MAE)."""
import json, numpy as np, pandas as pd
from lib import v2_frame, endgame_terms
g = v2_frame(); g = g[g.season >= 2013].copy()
fit, test = g.season <= 2019, g.season >= 2020
sd_m = float((g.act_m - g.v2_m)[fit].std()); sd_t = float((g.act_t - g.v2_t)[fit].std())
ex, pot = endgame_terms(g.v2_m.values, sd_m); g["eg"] = ex - ex[fit.values].mean(); g["ot"] = pot - pot[fit.values].mean()
out = {"sd_margin_fit": round(sd_m, 2), "sd_total_fit": round(sd_t, 2), "sd_margin_test": round(float((g.act_m - g.v2_m)[test].std()), 2),
       "sd_total_test": round(float((g.act_t - g.v2_t)[test].std()), 2)}
# fit total = v2_t + a*eg + b*ot on 2013-19 residual
F = g[fit]; X = np.c_[F.eg, F.ot]; y = (F.act_t - F.v2_t).values
k, *_ = np.linalg.lstsq(X, y, rcond=None); out["fit_k_eg_ot"] = [round(float(v), 3) for v in k]
# also the theory version: k = (1, OT_PTS)
from lib import OT_PTS
for tag, kk in (("v2", (0, 0)), ("fit", tuple(k)), ("theory", (1.0, OT_PTS))):
    g[f"t_{tag}"] = g.v2_t + kk[0] * g.eg + kk[1] * g.ot
    g[f"h_{tag}"], g[f"a_{tag}"] = (g[f"t_{tag}"] + g.v2_m) / 2, (g[f"t_{tag}"] - g.v2_m) / 2
T = g[test]; L = T[T.line_margin.notna() & T.line_total.notna()].copy()
L["ih"], L["ia"] = (L.line_total + L.line_margin) / 2, (L.line_total - L.line_margin) / 2
res = {}
for tag in ("v2", "fit", "theory"):
    e_tot = (T[f"t_{tag}"] - T.act_t).abs().mean()
    eh = pd.concat([L[f"h_{tag}"] - L.act_h, L[f"a_{tag}"] - L.act_a]); act = pd.concat([L.act_h, L.act_a])
    line_e = pd.concat([L.ih - L.act_h, L.ia - L.act_a])
    # by season (stability)
    bys = {int(s): round(float((q[f"t_{tag}"] - q.act_t).abs().mean()), 3) for s, q in T.groupby("season")}
    res[tag] = dict(total_MAE_all=round(float(e_tot), 3), team_MAE_lined=round(float(eh.abs().mean()), 3), team_APE_lined=round(float((eh.abs() / act).mean()), 4),
                    closer_than_line=round(float((eh.abs() < line_e.abs()).mean()), 4), total_MAE_by_season=bys)
res["line"] = dict(team_MAE_lined=round(float(line_e.abs().mean()), 3), team_APE_lined=round(float((line_e.abs() / act).mean()), 4), n_team_scores=int(len(line_e)))
# paired t for fit vs v2 on totals
d = (T.t_fit - T.act_t).abs() - (T.t_v2 - T.act_t).abs(); res["fit_vs_v2_total_t"] = round(float(d.mean() / (d.std() / np.sqrt(len(d)))), 2)
d = (T.t_theory - T.act_t).abs() - (T.t_v2 - T.act_t).abs(); res["theory_vs_v2_total_t"] = round(float(d.mean() / (d.std() / np.sqrt(len(d)))), 2)
# does the adjustment track where the market's total sits? r(eg, line_total - v2_t)
res["r_eg_with_line_minus_v2"] = round(float(np.corrcoef(L.eg, L.line_total - L.t_v2)[0, 1]), 3)
out["test_2020_2026"] = res
# spread of results: calibration of cover chance using sd_m (normal) on test
z = (T.act_m - T.v2_m) / sd_m; out["test_z_sd"] = round(float(z.std()), 3)
json.dump(out, open("model_a_test.json", "w"), indent=1); print(json.dumps(out, indent=1))
