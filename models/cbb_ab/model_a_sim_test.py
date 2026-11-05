"""Does the possession sim's shape (end-game fouling, overtime) price closing lines better than a plain normal around v2?
Test 2020-2026 lined games (sample). Brier + log loss of P(home covers close), P(over close total), P(team over implied team total)."""
import json, numpy as np, pandas as pd
from lib import v2_frame, sim_game, ncdf
g = v2_frame(); T = g[(g.season >= 2020) & g.line_margin.notna() & g.line_total.notna() & g.poss_hat.notna()].sample(6000, random_state=1)
SDM, SDT = 10.93, 17.14
SDTT = None
rows = []
for x in T.itertuples():
    hs, as_ = (x.v2_t + x.v2_m) / 2, (x.v2_t - x.v2_m) / 2
    for eg in (True, False):
        H, A = sim_game(hs, as_, x.poss_hat, SDM, SDT, n=3000, seed=int(x.game_id) % 100000, endgame=eg)
        m, t = H - A, H + A; L, TT = x.line_margin, x.line_total; ih = (TT + L) / 2
        rows.append(dict(eg=eg, pcov=((m > L).mean() + 0.5 * (m == L).mean()), ov=((t > TT).mean() + 0.5 * (t == TT).mean()), tto=(H > ih).mean() + 0.5 * (H == ih).mean(),
                         y_cov=np.sign(x.act_m - L), y_ov=np.sign(x.act_t - TT), y_tt=np.sign(x.act_h - ih),
                         n_cov=ncdf((x.v2_m - L) / SDM), n_ov=ncdf((x.v2_t - TT) / SDT), n_tt=ncdf((hs - ih) / np.hypot(SDT, SDM) * 2), win=(m > 0).mean(), y_win=float(x.act_m > 0), n_win=ncdf(x.v2_m / SDM)))
R = pd.DataFrame(rows)
def score(p, y):
    m = y != 0; p = np.clip(p[m], 1e-4, 1 - 1e-4); yy = (y[m] > 0).astype(float)
    return dict(brier=round(float(((p - yy) ** 2).mean()), 5), logloss=round(float(-(yy * np.log(p) + (1 - yy) * np.log(1 - p)).mean()), 5), n=int(m.sum()))
out = {}
for eg in (True, False):
    Q = R[R.eg == eg]; k = "sim_endgame" if eg else "sim_no_endgame"
    out[k] = {"cover": score(Q.pcov.values, Q.y_cov.values), "over": score(Q.ov.values, Q.y_ov.values), "team_over": score(Q.tto.values, Q.y_tt.values), "win": score(Q.win.values, Q.y_win.values * 2 - 1)}
Q = R[R.eg]
out["normal"] = {"cover": score(Q.n_cov.values, Q.y_cov.values), "over": score(Q.n_ov.values, Q.y_ov.values), "team_over": score(Q.n_tt.values, Q.y_tt.values), "win": score(Q.n_win.values, Q.y_win.values * 2 - 1)}
out["coinflip_ref"] = 0.25
json.dump(out, open("model_a_sim_test.json", "w"), indent=1); print(json.dumps(out, indent=1))
