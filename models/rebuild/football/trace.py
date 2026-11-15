"""Play one simulated NFL game drive by drive (same rules and tables as sim.py, incl. the game-state tables of roadmap
step 2 and the overtime rules of step 3) and record every drive, for the Drive Lab viewer and Watch the sim. trace_game(th_h, th_a, seed, ...) -> dict
with drives and final score. k_score / k_late are the old score effects (replaced by the state tables; default 0)."""
import numpy as np, sim as S
OUTS = ["Touchdown", "Field goal", "Punt", "Turnover", "Return TD", "Safety", "End of half"]

def _st(half, t, lead):
    if half == 0: return 1 if t <= 120 else 0
    if t > 300: return 0
    return 5 if lead >= 1 else 6 if lead == 0 else 2 if lead >= -3 else 3 if lead >= -8 else 4

def trace_game(th_h, th_a, seed=0, k_score=0.0, k_late=0.0, th_sd=0.3, clock_scale=S.CLOCK):
    if S._T is None: S._T = S.tables()
    P, NS, NN, DU, DN, TRY = S._T
    rng = np.random.default_rng(seed)
    eh, ea = th_sd * rng.standard_normal(), th_sd * rng.standard_normal()
    sc = [0, 0]; drives = []
    first = int(rng.random() < 0.5)
    def tp(half, lead6):
        p = TRY[half, min(max(lead6, -20), 20) + 20]; u = rng.random()
        return 7 if u < p[0] else (6 if u < p[0] + p[1] else 8)
    for half in range(2):
        t = 1800.0; pos = first if half == 0 else 1 - first; start = 75.0
        while t > 0:
            th = (th_h + eh) if pos == 0 else (th_a + ea)
            lead = sc[pos] - sc[1 - pos]
            th -= k_score * min(max(lead, -21), 21) / 7
            if half == 1 and t < 480 and -8 <= lead < 0: th += k_late
            s = _st(half, t, lead); b = min(max(int(start // 5), 0), 19)
            w = P[s, b] * np.array([np.exp(th), np.exp(th), 1, 1, np.exp(-th), 1, 1]); k = int(rng.choice(7, p=w / w.sum()))
            pts = 0; who = pos
            if k == 0: pts = tp(half, lead + 6)
            elif k == 4: who = 1 - pos; pts = tp(half, -lead + 6)
            elif k == 1: pts = 3
            elif k == 5: pts = 2; who = 1 - pos
            sc[who] += pts
            dur = min(float(DU[s, k, rng.integers(DN[s, k])]) * clock_scale, t) if k != 6 else t
            drives.append({"half": half + 1, "clock": round(t / 60, 2), "side": "H" if pos == 0 else "A", "start": int(round(start)),
                           "res": OUTS[k], "pts": pts, "who": "H" if who == 0 else "A", "dur": round(dur / 60, 1), "h": sc[0], "a": sc[1]})
            if k == 6: break
            t -= dur; start = float(NS[k, rng.integers(NN[k])])
            if k != 4: pos = 1 - pos
    ot = 0
    if sc[0] == sc[1]:   # overtime, same rules as sim._ot: 10:00, both teams get the ball, then sudden death; tie at 0:00
        ot = 1; t = 600.0; pos = int(rng.random() < 0.5); start = float(NS[0, rng.integers(NN[0])]); n = 0
        while t > 0:
            th = (th_h + eh) if pos == 0 else (th_a + ea); lead = sc[pos] - sc[1 - pos]
            s = _st(1, min(t, 300.0), lead); b = min(max(int(start // 5), 0), 19)
            w = P[s, b] * np.array([np.exp(th), np.exp(th), 1, 1, np.exp(-th), 1, 1]); k = int(rng.choice(7, p=w / w.sum()))
            if k == 6: k = 2
            pts = 0; who = pos
            if k == 0: pts = tp(1, lead + 6)
            elif k == 4: who = 1 - pos; pts = tp(1, -lead + 6)
            elif k == 1: pts = 3
            elif k == 5: pts = 2; who = 1 - pos
            sc[who] += pts; n += 1
            dur = min(float(DU[s, k, rng.integers(DN[s, k])]) * clock_scale, t)
            drives.append({"half": 3, "clock": round(t / 60, 2), "side": "H" if pos == 0 else "A", "start": int(round(start)),
                           "res": OUTS[k] + (" (OT)" if k == 1 else ""), "pts": pts, "who": "H" if who == 0 else "A", "dur": round(dur / 60, 1), "h": sc[0], "a": sc[1]})
            if k in (4, 5) or (n >= 2 and sc[0] != sc[1]): break
            t -= dur; start = float(NS[k, rng.integers(NN[k])]); pos = 1 - pos
    return {"home": sc[0], "away": sc[1], "ot": ot, "drives": drives}
