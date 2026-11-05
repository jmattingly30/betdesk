"""Hockey layer 2: game-state simulator (numba).

Regulation is played in 10-second ticks. Each team scores at its projected rate (even strength + power play +
short-handed goals per 60, without empty-net goals), adjusted by:
  score effects   the trailing team presses (rate x (1 + s_eff)), the leader sits back (x (1 - s_eff))
  pulled goalie   trailing by 1 with `pull1` minutes left, or by 2 with `pull2` left: the trailer's rate x pull_boost
                  and the leader scores into the empty net at en_rate per 60
  overtime        regular season: 5 minutes of 3-on-3 at ot_mult x the combined rate, then a shootout
                  (home wins it with so_home); playoffs: 20-minute 5-on-5 periods until a goal
Returns per game: mean goals each side, sd of margin and total, home win %, overtime %, and score histograms.
"""
import numpy as np
from numba import njit

@njit(cache=True)
def sim_games(rh, ra, playoff, nsim, seed, s_eff, pull1, pull2, pull_boost, en_rate, ot_mult, so_home, rate_sd):
    np.random.seed(seed)
    G = rh.shape[0]
    out = np.zeros((G, 7)); hist = np.zeros((G, 2, 12)); mh = np.zeros((G, 21))
    ticks = 360; dt = 1.0 / 360.0          # fraction of 60 minutes per tick
    for g in range(G):
        sh = 0.0; sa = 0.0; sm2 = 0.0; st2 = 0.0; wins = 0.0; ots = 0.0; en_tot = 0.0
        for s in range(nsim):
            h = 0; a = 0
            # game-level rate noise: tonight's true rates differ from the projection (lineups, form, refs)
            uh = np.exp(rate_sd * np.random.randn() - rate_sd * rate_sd / 2); ua = np.exp(rate_sd * np.random.randn() - rate_sd * rate_sd / 2)
            for k in range(ticks):
                left = (ticks - k) / 6.0      # minutes left
                d = h - a
                lh = rh[g] * uh; la = ra[g] * ua; enh = 0.0; ena = 0.0
                if d > 0: lh *= (1 - s_eff); la *= (1 + s_eff)
                elif d < 0: lh *= (1 + s_eff); la *= (1 - s_eff)
                if (d == -1 and left <= pull1) or (d == -2 and left <= pull2):
                    lh *= pull_boost; ena = en_rate
                if (d == 1 and left <= pull1) or (d == 2 and left <= pull2):
                    la *= pull_boost; enh = en_rate
                if np.random.random() < lh * dt: h += 1
                if np.random.random() < enh * dt: h += 1; en_tot += 1
                if np.random.random() < la * dt: a += 1
                if np.random.random() < ena * dt: a += 1; en_tot += 1
            if h == a:
                ots += 1
                if playoff[g]:
                    tot = rh[g] + ra[g]
                    # sudden death: first goal wins, share by rates
                    if np.random.random() < rh[g] / tot: h += 1
                    else: a += 1
                else:
                    tot = (rh[g] + ra[g]) * ot_mult
                    if np.random.random() < 1 - np.exp(-tot * 5.0 / 60.0):
                        if np.random.random() < rh[g] / (rh[g] + ra[g]): h += 1
                        else: a += 1
                    else:
                        if np.random.random() < so_home: h += 1
                        else: a += 1
            sh += h; sa += a; sm2 += (h - a) ** 2; st2 += (h + a) ** 2
            if h > a: wins += 1
            hist[g, 0, min(h, 11)] += 1; hist[g, 1, min(a, 11)] += 1
            mh[g, min(max(h - a + 10, 0), 20)] += 1
        m1 = sh / nsim; m2 = sa / nsim
        out[g, 0] = m1; out[g, 1] = m2
        out[g, 2] = np.sqrt(max(sm2 / nsim - (m1 - m2) ** 2, 1e-6)); out[g, 3] = np.sqrt(max(st2 / nsim - (m1 + m2) ** 2, 1e-6))
        out[g, 4] = wins / nsim; out[g, 5] = ots / nsim; out[g, 6] = en_tot / nsim
        hist[g] /= nsim; mh[g] /= nsim
    return out, hist, mh

DEFAULT = dict(s_eff=0.12, pull1=1.6, pull2=2.6, pull_boost=2.0, en_rate=14.0, ot_mult=2.3, so_home=0.5, rate_sd=0.0)

def simulate(p, nsim=400, seed=1, **kw):
    """p needs rate_h, rate_a (non-empty-net goals per 60) and playoff. Adds sim columns."""
    c = {**DEFAULT, **kw}
    out, hist, mh = sim_games(p.rate_h.values.astype(np.float64), p.rate_a.values.astype(np.float64),
                              p.playoff.values.astype(np.bool_), nsim, seed, c["s_eff"], c["pull1"], c["pull2"],
                              c["pull_boost"], c["en_rate"], c["ot_mult"], c["so_home"], c["rate_sd"])
    q = p.copy()
    q["sim_h"], q["sim_a"], q["sd_margin"], q["sd_total"], q["p_home"], q["p_ot"], q["en_goals"] = out.T
    return q, hist, mh
