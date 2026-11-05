"""Shared pieces for NBA Model A (scores) and Model B (bets) -- J 2026-11-05 rollout (third sport after CBB and NFL).
Model A = rebuild stack means (+ component fix, comp_test.py) with a possession sim around them for ranges and overtime.
Sim research: research/model-a-sim-2026-10/nba-possession (total sd 18.8 = real, needs 2nd-half blowout easing for the margin,
sim means run ~3 low without and-ones, so the sim only shapes the spread around Model A's mean, as CBB)."""
import math, numpy as np
C = ["tov", "orb", "fta", "p3", "p2", "s3", "ftp"]          # comp_test.py order
COMP = ["tov", "fta", "s3", "p3", "p2", "ftp", "orb"]       # engine.py COMP order (live_future_nba returns rates in this order)
def ncdf(x): return 0.5 * (1 + math.erf(x / math.sqrt(2)))

def comp_fix(F, rates, poss):
    """points correction for one team from its predicted component rates (dict name -> rate) x pace (comp_test.json 'final')."""
    if not rates or not poss: return 0.0
    v = F["const"] + sum(F["b"][c] * (rates[c] * poss - F["mu"][c]) for c in C)
    return float(max(-F["cap"], min(F["cap"], v)))

K = np.array([0, 1, 2, 3.])
BASE = np.array([0.46, 0.05, 0.32, 0.17])   # NBA possession outcome shares (0/1/2/3 pts), ppp ~1.14 after the tilt
def _tilt(ppp):
    lo, hi = -3.0, 3.0
    for _ in range(40):
        t = (lo + hi) / 2; q = BASE * np.exp(t * K); q /= q.sum()
        lo, hi = (t, hi) if (q * K).sum() < ppp else (lo, t)
    q = BASE * np.exp(lo * K); return q / q.sum()
EASE = 0.0015 * 1.14       # research: leader's make chance -0.0015 per point of lead in the 2nd half (x ~1.14 pts per trip)
TIE_W, OT_PTS = 1.0, 21.0  # |margin| <= 1 after regulation counts as a tie (about 6% of games, the NBA overtime rate); OT ~ 21 pts total

def sim_game(hs, as_, poss, sd_m, sd_t, n=4000, seed=0):
    """Possession sim anchored to Model A's means (hs, as_): first half possession by possession, then the second half with
    blowout easing (the leader eases off), overtime on a tie. Luck is rescaled so the spread of results matches Model A's
    measured miss (sd_m margin, sd_t total, model_a_test.json), which includes rating error, not just game luck."""
    rng = np.random.default_rng(seed)
    N = np.clip(np.rint(rng.normal(poss, 3.5, n)), 80, 125).astype(int); h1 = N // 2
    qh, qa = _tilt(hs / poss), _tilt(as_ / poss)
    pts = lambda q, m: rng.multinomial(m, q) @ K if np.isscalar(m) else np.array([rng.multinomial(k, q) @ K for k in m])
    H1, A1 = pts(qh, h1), pts(qa, h1); H2, A2 = pts(qh, N - h1), pts(qa, N - h1)
    lead = (H1 - A1) - (H1 - A1).mean(); ease = EASE * np.abs(lead) * (N - h1) / 2
    H2 = H2 - np.where(lead > 0, ease, 0); A2 = A2 - np.where(lead < 0, ease, 0)
    H, A = H1 + H2, A1 + A2; m, t = H - A, H + A
    fm, ft = sd_m / max(m.std(), 1e-9), sd_t / max(t.std(), 1e-9)
    m = (hs - as_) + (m - m.mean()) * fm; t = (hs + as_) + (t - t.mean()) * ft
    tie = np.abs(m) <= TIE_W
    ot = rng.normal(0, 4.5, n); m = np.where(tie, np.round(ot) + np.sign(ot + 1e-9), m)
    t = t + tie * OT_PTS; t = t - (tie * OT_PTS).mean()    # Model A's mean total is kept
    return (t + m) / 2, (t - m) / 2, float(tie.mean())
