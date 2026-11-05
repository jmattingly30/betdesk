"""Football layer 1, re-fit version (built for CFB): every week, re-solve each component's offense/defense ratings
from all of this season's games so far (plus last season's at weight W_PREV), shrunk toward a preseason prior
(last season's final rating x carry, plus optional prior terms). Unlike the forward Kalman, early-season results are
re-read once we know more about those opponents (college schedules are sparse, so this matters).
Same components and points map as engine.py. run(T, prm, prior_fn) -> predictions frame."""
import numpy as np, pandas as pd
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import lsqr
import engine as E

DEFAULT = dict(lam=(30, 30, 30, 120, 60, 40, 200), carry=0.6, w_prev=0.15, hfa_lam=50)

def solve(rows, y, w, nT, prior, lam):
    """rows: (off_idx, def_idx, home_sign); minimise sum w (y - mu - O - D - h*s)^2 + lam |[O;D] - prior|^2."""
    n = len(y); cols_o = rows[:, 0]; cols_d = nT + rows[:, 1]
    data = np.concatenate([np.ones(n), np.ones(n), rows[:, 2].astype(float), np.ones(n)])
    ri = np.concatenate([np.arange(n)] * 4)
    ci = np.concatenate([cols_o, cols_d, np.full(n, 2 * nT), np.full(n, 2 * nT + 1)])
    sw = np.sqrt(w)
    A = csr_matrix((data * np.tile(sw, 4), (ri, ci)), shape=(n, 2 * nT + 2))
    # prior rows
    k = 2 * nT; P = csr_matrix((np.full(k, np.sqrt(lam)), (np.arange(k), np.arange(k))), shape=(k, 2 * nT + 2))
    from scipy.sparse import vstack
    M = vstack([A, P]); b = np.concatenate([y * sw, np.sqrt(lam) * prior])
    x = lsqr(M, b, atol=1e-8, btol=1e-8)[0]
    return x[:nT], x[nT:2 * nT], x[2 * nT], x[2 * nT + 1]

def run(T, prm=None, prior_fn=None):
    prm = {**DEFAULT, **(prm or {})}
    g = E.games(T); teams = pd.unique(pd.concat([g.team, g.team_a])); idx = {t: i for i, t in enumerate(teams)}; nT = len(teams)
    g["hi"] = g.team.map(idx); g["ai"] = g.team_a.map(idx); g["hs"] = np.where(g.location == "Neutral", 0, 1)
    C = E.COMP; NC = len(C)
    last = {c: (np.zeros(nT), np.zeros(nT)) for c in C}
    rows_out = []; lvl = 0.0
    for season, gs in g.groupby("season", sort=True):
        prev = g[g.season == season - 1]
        pri = {c: (prm["carry"] * last[c][0], prm["carry"] * last[c][1]) for c in C}
        if prior_fn: pri = prior_fn(season, idx, pri)
        for wk, gw in gs.groupby("week", sort=True):
            done = gs[gs.week < wk]; hist = pd.concat([prev, done])
            wts = np.where(hist.season == season, 1.0, prm["w_prev"])
            R = np.vstack([np.column_stack([hist.hi, hist.ai, hist.hs]), np.column_stack([hist.ai, hist.hi, -hist.hs])]).astype(int)
            pred = {}
            for k, c in enumerate(C):
                y = np.concatenate([hist["y_" + c].values, hist["y_" + c + "_a"].values])
                n = np.concatenate([hist["n_" + c].values, hist["n_" + c + "_a"].values])
                w = np.concatenate([wts, wts]) * n / E.SIG[k] ** 2
                ok = np.isfinite(y)
                lam = prm["lam"][k] / (E.SIG[k] ** 2 / [62, 62, 62, 62, 11.4, 1, 11.4][k])
                if ok.sum() < 20:
                    O, D, mu, h = pri[c][0], pri[c][1], np.nanmean(g["y_" + c]), 0.0
                else:
                    # the solver centres on the prior, so solve for deviations from the league mean
                    m0 = np.average(y[ok], weights=w[ok])
                    O, D, h, mu_d = solve(R[ok], y[ok] - m0, w[ok], nT, np.concatenate(pri[c]), lam)
                    mu = m0 + mu_d
                pred[c] = (O, D, mu, h)
                if wk == gs.week.max(): last[c] = (O, D)
            recs = gw.to_dict("records"); out_rows = []
            for r in recs:
                lh = np.array([pred[c][2] + pred[c][0][r["hi"]] + pred[c][1][r["ai"]] + r["hs"] * pred[c][3] for c in C])
                la = np.array([pred[c][2] + pred[c][0][r["ai"]] + pred[c][1][r["hi"]] - r["hs"] * pred[c][3] for c in C])
                o = {"game_id": r["game_id"], "season": season, "week": wk, "date": r["date"], "home": r["team"], "away": r["team_a"],
                     "playoff": r["season_type"] == "POST", "act_h": r["final"], "act_a": r["final_a"],
                     "pred_h": E.points(lh, 0) + lvl, "pred_a": E.points(la, 0) + lvl, "line_margin": r["spread_line"], "line_total": r["total_line"]}
                out_rows.append(o)
            res = np.mean([(o["act_h"] - o["pred_h"] + o["act_a"] - o["pred_a"]) / 2 for o in out_rows])
            lvl += 0.03 * min(len(recs) / 50, 3) * res
            rows_out += out_rows
        # season end: final ratings (refit including the last week)
    return pd.DataFrame(rows_out)
