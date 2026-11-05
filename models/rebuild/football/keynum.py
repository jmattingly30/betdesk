"""NFL score distribution test: plain normal vs normal x key-number weights (weights = how much more often each
exact margin / total happens than a smooth normal says, fit on 2010-2017) vs the drive simulator.
Scores on 2018+: win log loss, margin/total CRPS (discrete), log score of the exact margin."""
import sys, os, numpy as np, pandas as pd
from scipy import stats
HERE = os.path.dirname(os.path.abspath(__file__))
p = pd.read_parquet(f"{HERE}/data/stack_nfl.parquet")
p["mu_m"] = p.pred_h - p.pred_a; p["mu_t"] = p.pred_h + p.pred_a; p["m"] = p.act_h - p.act_a; p["t"] = p.act_h + p.act_a
tr, te = p[(p.season >= 2010) & (p.season <= 2017)], p[p.season >= 2018]
SDM, SDT = np.std(tr.m - tr.mu_m), np.std(tr.t - tr.mu_t)
M = np.arange(-60, 61); TT = np.arange(0, 121)
def weights(mu, act, grid, sd):
    dens = stats.norm.pdf(grid[None, :], mu[:, None], sd).mean(0)
    freq = pd.Series(act).value_counts(normalize=True).reindex(grid, fill_value=0).values
    w = (freq + 1e-4) / (dens + 1e-4)
    w = np.where(dens > 2e-3, w, 1.0)            # only trust where there is data
    return w
wm = weights(tr.mu_m.values, tr.m.values, M, SDM); wt = weights(tr.mu_t.values, tr.t.values, TT, SDT)
print("key margin weights:", {int(k): round(float(wm[k + 60]), 2) for k in (1, 2, 3, 4, 6, 7, 8, 10, 14)})
def pmf(mu, grid, sd, w=None):
    d = stats.norm.pdf(grid[None, :], mu[:, None], sd)
    if w is not None: d = d * w[None, :]
    return d / d.sum(1, keepdims=True)
def crps(P, grid, y):
    F = np.cumsum(P, 1); H = (grid[None, :] >= y[:, None]).astype(float)
    return ((F - H) ** 2).sum(1).mean()
def evals(Pm, Pt, d):
    m, t = d.m.values.astype(int), d.t.values.astype(int)
    pw = Pm[:, M > 0].sum(1) + 0.5 * Pm[:, M == 0].sum(1); y = (m > 0).astype(float); k = m != 0
    pw = np.clip(pw, 1e-4, 1 - 1e-4)
    return {"win_ll": round(float(-np.mean((y * np.log(pw) + (1 - y) * np.log(1 - pw))[k])), 4),
            "margin_crps": round(crps(Pm, M, m), 4), "total_crps": round(crps(Pt, TT, t), 4),
            "margin_logscore": round(float(-np.mean(np.log(Pm[np.arange(len(m)), np.clip(m + 60, 0, 120)] + 1e-9))), 4),
            "total_logscore": round(float(-np.mean(np.log(Pt[np.arange(len(t)), np.clip(t, 0, 120)] + 1e-9))), 4)}
res = {"normal": evals(pmf(te.mu_m.values, M, SDM), pmf(te.mu_t.values, TT, SDT), te),
       "normal x key numbers": evals(pmf(te.mu_m.values, M, SDM, wm), pmf(te.mu_t.values, TT, SDT, wt), te)}
print("sd", round(SDM, 2), round(SDT, 2)); print(pd.DataFrame(res).T.to_string())
