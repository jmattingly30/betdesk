"""Football layer 2: drive simulator (numba).

A game is two 30-minute halves of alternating drives. Each drive starts at a field position (yards from the end
zone) and ends in one of: touchdown, field goal, punt, turnover (incl. downs / missed FG), opponent return
touchdown, safety. Outcome odds come from league tables by start position (fit on 2008-2017 drives), shifted for
the offense by one scoring-odds term solved per game so the simulated mean matches the projection (the point
estimate comes from the stack; the simulator supplies the shape: key numbers, correlation, clock).
The next drive's start is drawn from what actually followed that outcome in the data. Drive length (clock) is
drawn per outcome. Overtime: one possession each, then sudden death (simplified).
"""
import os, numpy as np, pandas as pd
from numba import njit
HERE = os.path.dirname(os.path.abspath(__file__))
OUT = ["TD", "FG", "PUNT", "TO", "OPPTD", "SAF"]
BINS = np.arange(0, 101, 5)

def tables(lo=2008, hi=2017):
    D = pd.read_parquet(f"{HERE}/data/drives_nfl.parquet")
    D = D[(D.season >= lo) & (D.season <= hi) & D.start_y100.notna()].sort_values(["game_id", "fixed_drive"]).copy()
    m = {"Touchdown": 0, "Field goal": 1, "Punt": 2, "Turnover": 3, "Turnover on downs": 3, "Missed field goal": 3,
         "Opp touchdown": 4, "Safety": 5}
    D["o"] = D.result.map(m)
    D["next_start"] = D.groupby(["game_id", "half"]).start_y100.shift(-1)
    D = D[D.o.notna()]
    D["b"] = np.clip((D.start_y100 // 5).astype(int), 0, 19)
    P = np.zeros((20, 6))
    for b in range(20):
        c = D[D.b == b].o.value_counts().reindex(range(6), fill_value=0).values + 0.5
        P[b] = c / c.sum()
    # next start samples per outcome (the other team's start), and drive duration samples per outcome
    ns = [D[(D.o == k) & D.next_start.notna()].next_start.values.astype(np.float64) for k in range(6)]
    du = [D[D.o == k].dur.values.astype(np.float64) for k in range(6)]
    tdp = D[D.o == 0].pts_off.round()
    pts = np.array([(tdp == 7).mean(), (tdp == 6).mean(), (tdp == 8).mean()])
    L = max(len(x) for x in ns)
    NS = np.full((6, L), 75.0); NN = np.array([len(x) for x in ns]); DU = np.full((6, max(len(x) for x in du)), 120.0); DN = np.array([len(x) for x in du])
    for k in range(6): NS[k, :NN[k]] = ns[k]; DU[k, :DN[k]] = du[k]
    return P, NS, NN, DU, DN, pts

@njit(cache=True)
def _drive(P, th, start):
    b = min(max(int(start // 5), 0), 19)
    w = np.empty(6); s = 0.0
    for k in range(6):
        w[k] = P[b, k] * (np.exp(th) if k < 2 else 1.0) * (np.exp(-th) if k == 4 else 1.0); s += w[k]
    u = np.random.random() * s; acc = 0.0
    for k in range(6):
        acc += w[k]
        if u <= acc: return k
    return 5

@njit(cache=True)
def sim_games(thh, tha, P, NS, NN, DU, DN, td_pts, nsim, seed, clock_scale, k_score, k_late, th_sd):
    np.random.seed(seed); G = thh.shape[0]
    out = np.zeros((G, 6)); mh = np.zeros((G, 81))
    for g in range(G):
        sh = 0.0; sa = 0.0; s2m = 0.0; s2t = 0.0; win = 0.0
        for s in range(nsim):
            sc = np.zeros(2)
            eh = th_sd * np.random.randn(); ea = th_sd * np.random.randn()   # tonight's form vs the projection
            for half in range(2):
                t = 1800.0; pos = half    # home receives the 1st half kickoff in half the sims (alternate by half)
                if np.random.random() < 0.5: pos = 1 - pos
                start = 75.0
                while t > 0:
                    th = (thh[g] + eh) if pos == 0 else (tha[g] + ea)
                    lead = sc[pos] - sc[1 - pos]
                    # score effects: the leader sits on it, the trailer presses (more so late in the 2nd half)
                    th -= k_score * min(max(lead, -21.0), 21.0) / 7.0
                    if half == 1 and t < 480 and lead < 0 and lead >= -8: th += k_late
                    k = _drive(P, th, start)
                    if k == 0 or k == 4:     # touchdown + try: 7 (PAT), 6 (miss / failed 2-pt), 8 (2-pt)
                        u = np.random.random(); tp = 7.0 if u < td_pts[0] else (6.0 if u < td_pts[0] + td_pts[1] else 8.0)
                        if k == 0: sc[pos] += tp
                        else: sc[1 - pos] += tp
                    elif k == 1: sc[pos] += 3
                    elif k == 5: sc[1 - pos] += 2
                    t -= DU[k, np.random.randint(DN[k])] * clock_scale
                    start = NS[k, np.random.randint(NN[k])]
                    if k != 4 and k != 5: pos = 1 - pos
                    else: pos = pos if k == 5 else pos   # after a return TD / safety, scorer kicks/receives per table
            if sc[0] == sc[1]:   # overtime, simplified: first score wins, by relative scoring odds
                ph = np.exp(thh[g]) / (np.exp(thh[g]) + np.exp(tha[g]))
                if np.random.random() < 0.94:
                    if np.random.random() < ph: sc[0] += 3
                    else: sc[1] += 3
            sh += sc[0]; sa += sc[1]; d = sc[0] - sc[1]; s2m += d * d; s2t += (sc[0] + sc[1]) ** 2
            if d > 0: win += 1
            elif d == 0: win += 0.5
            mh[g, min(max(int(d) + 40, 0), 80)] += 1
        m1 = sh / nsim; m2 = sa / nsim
        out[g, 0] = m1; out[g, 1] = m2
        out[g, 2] = np.sqrt(max(s2m / nsim - (m1 - m2) ** 2, 1e-6)); out[g, 3] = np.sqrt(max(s2t / nsim - (m1 + m2) ** 2, 1e-6))
        out[g, 4] = win / nsim
        mh[g] /= nsim
    return out, mh

_T = None
def simulate(p, nsim=400, seed=1, clock_scale=1.0, iters=5, k_score=0.0, k_late=0.0, th_sd=0.0):
    """p needs pred_h, pred_a. Solves each side's scoring-odds shift so simulated means match the projection."""
    global _T
    if _T is None: _T = tables()
    P, NS, NN, DU, DN, pts = _T
    lg = 21.9
    thh = np.log(np.clip(p.pred_h.values, 3, 60) / lg) * 1.6; tha = np.log(np.clip(p.pred_a.values, 3, 60) / lg) * 1.6
    for it in range(iters):
        out, mh = sim_games(thh, tha, P, NS, NN, DU, DN, pts, max(nsim // 2, 200), seed + it, clock_scale, k_score, k_late, th_sd)
        thh += np.log(np.clip(p.pred_h.values, 3, 60) / np.clip(out[:, 0], 1, None)) * 1.6
        tha += np.log(np.clip(p.pred_a.values, 3, 60) / np.clip(out[:, 1], 1, None)) * 1.6
    out, mh = sim_games(thh, tha, P, NS, NN, DU, DN, pts, nsim, seed + 99, clock_scale, k_score, k_late, th_sd)
    q = p.copy(); q["sim_h"], q["sim_a"], q["sd_margin"], q["sd_total"], q["p_home"] = out[:, 0], out[:, 1], out[:, 2], out[:, 3], out[:, 4]
    q["th_h"], q["th_a"] = thh, tha
    return q, mh
