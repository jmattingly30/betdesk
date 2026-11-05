"""Layer 2 game engine: play every possession out, many times per game (numba).

Each possession: turnover?  -> else scoring attempts until a make, a defensive rebound or FT trip:
  FT trip (2 shots at FT%), or a FGA (3 with prob s3 at p3, else 2 at p2); a missed FG is rebounded
  by the offense with prob orb and the possession continues.
Game-level wobble: each team's rates for this game are drawn around the projection (hot/cold nights beyond luck of
individual shots), sd `wob` in logit units. Possessions per team ~ Normal(poss_hat, poss_sd).
Overtime: tied after regulation -> extra periods of ot_frac x possessions until someone leads.
End-game fouling (module, off by default): in the last `foul_poss` possessions, a team trailing by 3..`foul_max`
fouls on defense: the leader's possession becomes a 2-shot FT trip with no turnover chance.
"""
import numpy as np
from numba import njit

@njit(cache=True)
def _sigm(x): return 1.0 / (1.0 + np.exp(-x))

@njit(cache=True)
def _possession(t, f, s3, p3, p2, ftp, orb, fouled):
    if fouled:
        return (np.random.random() < ftp) * 1 + (np.random.random() < ftp) * 1
    if np.random.random() < t:
        return 0
    pts = 0
    for _ in range(12):
        if np.random.random() < f:
            pts += (np.random.random() < ftp) * 1 + (np.random.random() < ftp) * 1
            if np.random.random() < 0.27272727:
                pts += (np.random.random() < ftp) * 1
            return pts
        if np.random.random() < s3:
            if np.random.random() < p3: return pts + 3
        else:
            if np.random.random() < p2: return pts + 2
        if np.random.random() >= orb:
            return pts
    return pts

@njit(cache=True)
def sim_games(rh, ra, poss_hat, poss_sd, wob, nsim, ot_frac, foul_poss, foul_max, seed, k_score, w_shared):
    """rh, ra: (G, 7) projected rates (tov, fta, s3, p3, p2, ftp, orb). Returns per game:
    mean_h, mean_a, sd_margin, sd_total, p_home, corr, and a (G, 41) margin histogram for -20..20."""
    np.random.seed(seed)
    G = rh.shape[0]
    out = np.zeros((G, 6)); hist = np.zeros((G, 41))
    for g in range(G):
        lh = np.log(rh[g] / (1 - rh[g])); la = np.log(ra[g] / (1 - ra[g]))
        sh = 0.0; sa = 0.0; sm2 = 0.0; st2 = 0.0; wins = 0.0; sha = 0.0; sh2 = 0.0; sa2 = 0.0
        for s in range(nsim):
            xh = np.empty(7); xa = np.empty(7)
            for c in range(7):
                xh[c] = _sigm(lh[c] + wob[c] * np.random.standard_normal())
                xa[c] = _sigm(la[c] + wob[c] * np.random.standard_normal())
            zsh = w_shared * np.random.standard_normal()
            for c in (3, 4):
                xh[c] = _sigm(np.log(xh[c] / (1 - xh[c])) + zsh)
                xa[c] = _sigm(np.log(xa[c] / (1 - xa[c])) + zsh)
            n = int(round(poss_hat[g] + poss_sd * np.random.standard_normal()))
            if n < 10: n = 10
            ph = 0; pa = 0
            for k in range(n):
                left = n - k
                # home offense: away fouls if away trails
                foul_h = foul_poss > 0 and left <= foul_poss and (ph - pa) >= 3 and (ph - pa) <= foul_max
                L = ph - pa
                if L > 25: L = 25
                if L < -25: L = -25
                e = -k_score * L / 10.0
                ph += _possession(xh[0], _sigm(np.log(xh[1]/(1-xh[1]))), xh[2], _sigm(np.log(xh[3]/(1-xh[3])) + e), _sigm(np.log(xh[4]/(1-xh[4])) + e), xh[5], xh[6], foul_h)
                foul_a = foul_poss > 0 and left <= foul_poss and (pa - ph) >= 3 and (pa - ph) <= foul_max
                L = pa - ph
                if L > 25: L = 25
                if L < -25: L = -25
                e = -k_score * L / 10.0
                pa += _possession(xa[0], _sigm(np.log(xa[1]/(1-xa[1]))), xa[2], _sigm(np.log(xa[3]/(1-xa[3])) + e), _sigm(np.log(xa[4]/(1-xa[4])) + e), xa[5], xa[6], foul_a)
            while ph == pa:
                m = int(round(poss_hat[g] * ot_frac))
                if m < 3: m = 3
                for k in range(m):
                    ph += _possession(xh[0], xh[1], xh[2], xh[3], xh[4], xh[5], xh[6], False)
                    pa += _possession(xa[0], xa[1], xa[2], xa[3], xa[4], xa[5], xa[6], False)
            mg = ph - pa; tt = ph + pa
            sh += ph; sa += pa; sm2 += mg * mg; st2 += tt * tt; sha += ph * pa; sh2 += ph * ph; sa2 += pa * pa
            if mg > 0: wins += 1
            b = mg + 20
            if b < 0: b = 0
            if b > 40: b = 40
            hist[g, b] += 1
        mh = sh / nsim; ma = sa / nsim
        mm = mh - ma; mt = mh + ma
        out[g, 0] = mh; out[g, 1] = ma
        out[g, 2] = np.sqrt(max(sm2 / nsim - mm * mm, 1e-6)); out[g, 3] = np.sqrt(max(st2 / nsim - mt * mt, 1e-6))
        out[g, 4] = wins / nsim
        vh = sh2 / nsim - mh * mh; va = sa2 / nsim - ma * ma
        out[g, 5] = (sha / nsim - mh * ma) / np.sqrt(max(vh * va, 1e-6))
        hist[g] /= nsim
    return out, hist

COMP = ["tov", "fta", "s3", "p3", "p2", "ftp", "orb"]

def simulate(p, wob, poss_sd, nsim=400, ot_frac=0.125, foul_poss=0, foul_max=8, seed=1, k_score=0.0, w_shared=0.0):
    rh = p[["h_" + c for c in COMP]].values.astype(np.float64)
    ra = p[["a_" + c for c in COMP]].values.astype(np.float64)
    out, hist = sim_games(rh, ra, p.poss_hat.values.astype(np.float64), float(poss_sd), np.asarray(wob, np.float64),
                          nsim, ot_frac, foul_poss, foul_max, seed, float(k_score), float(w_shared))
    q = p.copy()
    q["sim_h"], q["sim_a"], q["sd_margin"], q["sd_total"], q["p_home"], q["rho"] = out.T
    return q, hist
