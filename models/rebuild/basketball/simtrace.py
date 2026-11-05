"""Possession-by-possession trace of single simulated games (same rules as sim.py, in plain Python so every
event is recorded), for the simulation viewer. trace_game(...) -> list of possessions with events and score."""
import numpy as np

def _sig(x): return 1 / (1 + np.exp(-x))
def _lg(p): return np.log(p / (1 - p))

def _poss(rng, r, e, fouled):
    """r: 7 rates (tov, fta, s3, p3, p2, ftp, orb); e: score-effect logit shift on p3/p2. Returns (pts, events)."""
    t, f, s3, p3, p2, ftp, orb = r
    p3, p2 = _sig(_lg(p3) + e), _sig(_lg(p2) + e)
    ev = []
    if fouled:
        m = int(rng.random() < ftp) + int(rng.random() < ftp)
        return m, [("FOUL", f"intentional foul, FT {m}/2", m)]
    if rng.random() < t:
        return 0, [("TO", "turnover", 0)]
    pts = 0
    for _ in range(12):
        if rng.random() < f:
            n = 3 if rng.random() < 0.27272727 else 2
            m = sum(int(rng.random() < ftp) for _ in range(n))
            ev.append(("FT", f"free throws {m}/{n}", m)); return pts + m, ev
        if rng.random() < s3:
            if rng.random() < p3: ev.append(("3M", "3-pointer made", 3)); return pts + 3, ev
            ev.append(("3X", "3-pointer missed", 0))
        else:
            if rng.random() < p2: ev.append(("2M", "2-pointer made", 2)); return pts + 2, ev
            ev.append(("2X", "2-pointer missed", 0))
        if rng.random() >= orb:
            ev.append(("DR", "defensive rebound", 0)); return pts, ev
        ev.append(("OR", "offensive rebound", 0))
    return pts, ev

def trace_game(rh, ra, poss_hat, seed, poss_sd=1.0, wob=0.04, k_score=0.07, w_shared=0.05, minutes=48.0, ot_frac=0.125):
    rng = np.random.default_rng(seed)
    xh = _sig(_lg(np.asarray(rh)) + wob * rng.standard_normal(7)); xa = _sig(_lg(np.asarray(ra)) + wob * rng.standard_normal(7))
    z = w_shared * rng.standard_normal()
    for x in (xh, xa): x[3] = _sig(_lg(x[3]) + z); x[4] = _sig(_lg(x[4]) + z)
    n = max(10, int(round(poss_hat + poss_sd * rng.standard_normal())))
    ph = pa = 0; out = []
    def one(side, k, total, period_len, base_min, ot):
        nonlocal ph, pa
        lead = (ph - pa) if side == "H" else (pa - ph)
        e = -k_score * max(-25, min(25, lead)) / 10
        pts, ev = _poss(rng, xh if side == "H" else xa, e, False)
        if side == "H": ph += pts
        else: pa += pts
        clock = base_min + period_len * (k + (0.5 if side == "A" else 0)) / total
        out.append({"i": len(out), "t": round(clock, 2), "side": side, "pts": pts, "ev": [[a, b] for a, b, _ in ev],
                    "h": ph, "a": pa, "ot": ot})
    for k in range(n):
        one("H", k, n, minutes, 0.0, 0); one("A", k, n, minutes, 0.0, 0)
    ot = 0
    while ph == pa:
        ot += 1; m = max(3, int(round(poss_hat * ot_frac)))
        for k in range(m):
            one("H", k, m, 5.0, minutes + 5.0 * (ot - 1), ot); one("A", k, m, 5.0, minutes + 5.0 * (ot - 1), ot)
    return {"seed": int(seed), "home": ph, "away": pa, "ot": ot, "poss": out,
            "rates_h": [round(float(v), 4) for v in xh], "rates_a": [round(float(v), 4) for v in xa]}
