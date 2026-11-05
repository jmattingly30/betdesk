"""Play one simulated NFL game drive by drive (same rules and tables as sim.py) and record every drive, for the
Drive Lab viewer. trace_game(th_h, th_a, seed, k_score, k_late, th_sd) -> dict with drives and final score."""
import numpy as np, sim as S
OUTS = ["Touchdown", "Field goal", "Punt", "Turnover", "Return TD", "Safety"]

def trace_game(th_h, th_a, seed=0, k_score=0.1, k_late=0.6, th_sd=0.3, clock_scale=1.0):
    if S._T is None: S._T = S.tables()
    P, NS, NN, DU, DN, tdp = S._T
    rng = np.random.default_rng(seed)
    eh, ea = th_sd * rng.standard_normal(), th_sd * rng.standard_normal()
    sc = [0, 0]; drives = []
    first = int(rng.random() < 0.5)
    for half in range(2):
        t = 1800.0; pos = first if half == 0 else 1 - first; start = 75.0
        while t > 0:
            th = (th_h + eh) if pos == 0 else (th_a + ea)
            lead = sc[pos] - sc[1 - pos]
            th -= k_score * min(max(lead, -21), 21) / 7
            if half == 1 and t < 480 and -8 <= lead < 0: th += k_late
            b = min(max(int(start // 5), 0), 19)
            w = P[b] * np.array([np.exp(th), np.exp(th), 1, 1, np.exp(-th), 1]); k = int(rng.choice(6, p=w / w.sum()))
            pts = 0; who = pos
            if k in (0, 4):
                u = rng.random(); pts = 7 if u < tdp[0] else (6 if u < tdp[0] + tdp[1] else 8)
                if k == 4: who = 1 - pos
            elif k == 1: pts = 3
            elif k == 5: pts = 2; who = 1 - pos
            sc[who] += pts
            dur = float(DU[k, rng.integers(DN[k])]) * clock_scale
            drives.append({"half": half + 1, "clock": round(t / 60, 2), "side": "H" if pos == 0 else "A", "start": int(round(start)),
                           "res": OUTS[k], "pts": pts, "who": "H" if who == 0 else "A", "dur": round(dur / 60, 1), "h": sc[0], "a": sc[1]})
            t -= dur; start = float(NS[k, rng.integers(NN[k])])
            if k not in (4, 5): pos = 1 - pos
    ot = 0
    if sc[0] == sc[1] and rng.random() < 0.94:
        ph = np.exp(th_h) / (np.exp(th_h) + np.exp(th_a)); w = 0 if rng.random() < ph else 1; sc[w] += 3; ot = 1
        drives.append({"half": 3, "clock": 10.0, "side": "H" if w == 0 else "A", "start": 75, "res": "Field goal (OT)", "pts": 3,
                       "who": "H" if w == 0 else "A", "dur": 5.0, "h": sc[0], "a": sc[1]})
    return {"home": sc[0], "away": sc[1], "ot": ot, "drives": drives}
