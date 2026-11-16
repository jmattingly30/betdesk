"""Football layer 2: drive simulator (numba).

A game is two 30-minute halves of alternating drives. Each drive starts at a field position (yards from the end
zone) and ends in one of: touchdown, field goal, punt, turnover (incl. downs / missed FG), opponent return
touchdown, safety. Outcome odds come from league tables by start position (fit on 2008-2017 drives), shifted for
the offense by one scoring-odds term solved per game so the simulated mean matches the projection (the point
estimate comes from the stack; the simulator supplies the shape: key numbers, correlation, clock).
The next drive's start is drawn from what actually followed that outcome in the data. Drive length (clock) is
drawn per outcome. Overtime (roadmap step 3, 2026-11-15): 2025+ regular-season rules (10:00, both teams get the ball, sudden death, ties).

GAME STATE (J 2026-11-15, football sim roadmap step 2): drive outcomes, drive lengths and the try after a touchdown depend
on the game state, from real 2008-18 drives: 7 states = normal, last 2:00 of the 1st half, and the last 5:00 of the game
trailing 1-3 / 4-8 / 9+, leading, tied. State tables = the normal table x that state's outcome mix (smoothed, 4 field
zones), including an "end of half" outcome (kneel / clock runs out). The try (PAT 7, miss 6, 2-pt 8) depends on the half
and the score after the 6. Clock x1.12 so a game has the real ~20.9 drives (was ~24). Safety: the team that scored the 2
gets the free kick. Test 2019-25 (1,908 games, rebuild means): exact-final log loss 6.966 -> 6.880, team score 3.512 ->
3.495, margin 3.900 -> 3.895; margin of 3 10.8% -> 13.9% (real 14.5%); field goal when down 4-8 in the last 5:00 17% -> 0.3%
(real 0.3%); 1st-half last 2:00 field goals 12% -> 22% (real 22%). The old k_score / k_late score effects are replaced by
the state tables (still accepted, default 0).
"""
import os, numpy as np, pandas as pd
try: from numba import njit
except ImportError:   # no numba (e.g. a container without it): plain Python; trace.py and models/sims/build.py only need the tables
    njit = lambda *a, **k: (lambda f: f)
HERE = os.path.dirname(os.path.abspath(__file__))
OUT = ["TD", "FG", "PUNT", "TO", "OPPTD", "SAF", "EOH"]
NST = 7; CLOCK = 1.12
STATES = ["normal", "1st half last 2:00", "last 5:00 trailing 1-3", "last 5:00 trailing 4-8", "last 5:00 trailing 9+", "last 5:00 leading", "last 5:00 tied"]

@njit(cache=True)
def state(half, t, lead):
    """half 0/1, t = seconds left in the half, lead = offense score minus defense score"""
    if half == 0: return 1 if t <= 120 else 0
    if t > 300: return 0
    if lead >= 1: return 5
    if lead == 0: return 6
    if lead >= -3: return 2
    if lead >= -8: return 3
    return 4

def tables(lo=2008, hi=2018):
    D = pd.read_parquet(f"{HERE}/data/drives_nfl.parquet")
    D = D[(D.season >= lo) & (D.season <= hi) & (D.season_type == "REG") & (D.qtr <= 4) & D.start_y100.notna() & D.start_diff.notna()].sort_values(["game_id", "fixed_drive"]).copy()
    m = {"Touchdown": 0, "Field goal": 1, "Punt": 2, "Turnover": 3, "Turnover on downs": 3, "Missed field goal": 3,
         "Opp touchdown": 4, "Safety": 5, "End of half": 6}
    D["o"] = D.result.map(m); D["next_start"] = D.groupby(["game_id", "half"]).start_y100.shift(-1)
    D = D[D.o.notna()].copy(); D["o"] = D.o.astype(int)
    t = np.where(D.half == "Half1", D.sec0 - 1800, D.sec0).astype(float)
    D["s"] = [state(int(h), float(x), int(l)) for h, x, l in zip((D.half == "Half2").astype(int), t, D.start_diff)]
    D["b"] = np.clip((D.start_y100 // 5).astype(int), 0, 19); D["bb"] = np.digitize(D.start_y100, [40, 60, 75])
    base = np.zeros((20, 7))
    for b in range(20):
        c = D[(D.b == b) & (D.s == 0)].o.value_counts().reindex(range(7), fill_value=0).values + 0.5; base[b] = c / c.sum()
    P = np.zeros((NST, 20, 7)); P[0] = base
    for s in range(1, NST):   # state = normal shape x the state's outcome mix by field zone (shrunk toward normal)
        for bb in range(4):
            X = D[(D.s == s) & (D.bb == bb)]; y0 = D[(D.s == 0) & (D.bb == bb)].o.value_counts(normalize=True).reindex(range(7), fill_value=0).values
            cs = X.o.value_counts().reindex(range(7), fill_value=0).values + 2.0 * y0; mult = (cs / cs.sum()) / (y0 + 1e-3)
            for b in range(20):
                if np.digitize([b * 5 + 2.5], [40, 60, 75])[0] == bb: v = base[b] * mult; P[s, b] = v / v.sum()
    ns = [D[(D.o == k) & D.next_start.notna()].next_start.values.astype(np.float64) for k in range(6)] + [np.array([75.0])]
    NS = np.full((7, max(len(x) for x in ns)), 75.0); NN = np.array([len(x) for x in ns])
    for k in range(7): NS[k, :NN[k]] = ns[k]
    L = 4000; DU = np.full((NST, 7, L), 120.0); DN = np.zeros((NST, 7), np.int64)
    for s in range(NST):
        for k in range(7):
            v = D[(D.s == s) & (D.o == k)].dur.values
            if len(v) < 20: v = D[D.o == k].dur.values
            v = v[:L].astype(np.float64); DU[s, k, :len(v)] = v; DN[s, k] = len(v)
    T = D[D.o == 0].copy(); T["d6"] = np.clip(T.start_diff + 6, -20, 20).astype(int); T["h2"] = (T.half == "Half2").astype(int)
    TRY = np.zeros((2, 41, 3))   # P(7, 6, 8) by half and the offense's lead after the 6; 1st half pooled (2-pt chasing is a 2nd-half thing)
    for hh in range(2):
        for d in range(-20, 21):
            x = T[(T.h2 == hh) & T.d6.between(d - (0 if hh else 30), d + (0 if hh else 30))].pts_off
            c = np.array([(x == 7).sum(), (x == 6).sum(), (x == 8).sum()], float) + np.array([0.9, 0.06, 0.04]) * 3; TRY[hh, d + 20] = c / c.sum()
    return P, NS, NN, DU, DN, TRY

@njit(cache=True)
def _drive(P, s, th, start):
    b = min(max(int(start // 5), 0), 19)
    w = np.empty(7); tot = 0.0
    for k in range(7):
        w[k] = P[s, b, k] * (np.exp(th) if k < 2 else 1.0) * (np.exp(-th) if k == 4 else 1.0); tot += w[k]
    u = np.random.random() * tot; acc = 0.0
    for k in range(7):
        acc += w[k]
        if u <= acc: return k
    return 6

@njit(cache=True)
def _try(TRY, half, lead6):
    d = min(max(lead6, -20), 20) + 20; u = np.random.random()
    if u < TRY[half, d, 0]: return 7.0
    if u < TRY[half, d, 0] + TRY[half, d, 1]: return 6.0
    return 8.0

@njit(cache=True)
def _ot(sc, th0, th1, P, NS, NN, DU, DN, TRY, clock_scale):
    """Regular-season overtime under the 2025+ rules: 10:00, both teams get the ball (unless the defense scores), then
    sudden death; still tied at 0:00 = tie. Drives use the late-game state tables (tied / trailing), so the 2nd team
    down 3 kicks, down 7 or 8 goes for the TD, and a team down 1 after its TD goes for 2 as often as real teams do."""
    t = 600.0; pos = 0 if np.random.random() < 0.5 else 1; start = NS[0, np.random.randint(NN[0])]; n = 0
    while t > 0:
        th = th0 if pos == 0 else th1
        lead = sc[pos] - sc[1 - pos]
        s = state(1, min(t, 300.0), int(lead))
        k = _drive(P, s, th, start)
        if k == 6: k = 2                                         # no kneels in a tied overtime: a dead drive = a punt
        if k == 0: sc[pos] += _try(TRY, 1, int(lead) + 6)
        elif k == 4: sc[1 - pos] += _try(TRY, 1, int(-lead) + 6); return
        elif k == 1: sc[pos] += 3
        elif k == 5: sc[1 - pos] += 2; return
        n += 1
        if n >= 2 and sc[0] != sc[1]: return                     # both have had the ball (or sudden death): a lead ends it
        t -= DU[s, k, np.random.randint(DN[s, k])] * clock_scale
        start = NS[k, np.random.randint(NN[k])]; pos = 1 - pos

@njit(cache=True)
def sim_games(thh, tha, P, NS, NN, DU, DN, TRY, nsim, seed, clock_scale, k_score, k_late, th_sd):
    np.random.seed(seed); G = thh.shape[0]
    out = np.zeros((G, 6)); mh = np.zeros((G, 81))
    for g in range(G):
        sh = 0.0; sa = 0.0; s2m = 0.0; s2t = 0.0; win = 0.0
        for it in range(nsim):
            sc = np.zeros(2)
            eh = th_sd * np.random.randn(); ea = th_sd * np.random.randn()   # tonight's form vs the projection
            first = 0 if np.random.random() < 0.5 else 1                     # 2nd half: the other team receives
            for half in range(2):
                t = 1800.0; pos = first if half == 0 else 1 - first; start = 75.0
                while t > 0:
                    th = (thh[g] + eh) if pos == 0 else (tha[g] + ea)
                    lead = sc[pos] - sc[1 - pos]
                    th -= k_score * min(max(lead, -21.0), 21.0) / 7.0
                    if half == 1 and t < 480 and lead < 0 and lead >= -8: th += k_late
                    s = state(half, t, int(lead))
                    k = _drive(P, s, th, start)
                    if k == 6: break                                     # kneel / clock runs out
                    if k == 0: sc[pos] += _try(TRY, half, int(lead) + 6)
                    elif k == 4: sc[1 - pos] += _try(TRY, half, int(-lead) + 6)
                    elif k == 1: sc[pos] += 3
                    elif k == 5: sc[1 - pos] += 2
                    t -= DU[s, k, np.random.randint(DN[s, k])] * clock_scale
                    start = NS[k, np.random.randint(NN[k])]
                    if k != 4: pos = 1 - pos   # return TD: the scorer kicks off, the same offense gets it back; safety: the scorer gets the free kick
            if sc[0] == sc[1]:
                _ot(sc, thh[g] + eh, tha[g] + ea, P, NS, NN, DU, DN, TRY, clock_scale)
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
def simulate(p, nsim=400, seed=1, clock_scale=CLOCK, iters=5, k_score=0.0, k_late=0.0, th_sd=0.3):
    """p needs pred_h, pred_a. Solves each side's scoring-odds shift so simulated means match the projection."""
    global _T
    if _T is None: _T = tables()
    P, NS, NN, DU, DN, TRY = _T
    lg = 21.9
    thh = np.log(np.clip(p.pred_h.values, 3, 60) / lg) * 1.6; tha = np.log(np.clip(p.pred_a.values, 3, 60) / lg) * 1.6
    for it in range(iters):
        out, mh = sim_games(thh, tha, P, NS, NN, DU, DN, TRY, max(nsim // 2, 200), seed + it, clock_scale, k_score, k_late, th_sd)
        thh += np.log(np.clip(p.pred_h.values, 3, 60) / np.clip(out[:, 0], 1, None)) * 1.6
        tha += np.log(np.clip(p.pred_a.values, 3, 60) / np.clip(out[:, 1], 1, None)) * 1.6
    out, mh = sim_games(thh, tha, P, NS, NN, DU, DN, TRY, nsim, seed + 99, clock_scale, k_score, k_late, th_sd)
    q = p.copy(); q["sim_h"], q["sim_a"], q["sd_margin"], q["sd_total"], q["p_home"] = out[:, 0], out[:, 1], out[:, 2], out[:, 3], out[:, 4]
    q["th_h"], q["th_a"] = thh, tha
    return q, mh
