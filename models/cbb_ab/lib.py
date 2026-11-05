"""Shared pieces for CBB Model A (scores) and Model B (bets) -- J 2026-11-02 rollout (CBB first).
Model A = CBB v2 mean scores + possession sim around them (variance calibrated, end-game fouling + overtime state).
Model B = edge finder on Model A: sharp-weighted devig, real-edge share by timing, confidence tiers."""
import json, math, os, numpy as np, pandas as pd
M = "/mnt/project-files/sports-analytics/models/"
HERE = os.path.dirname(os.path.abspath(__file__))
# end-game table from research/cbb-sim-2026-10/endgame.json: |margin| with 2:00 left -> extra points vs normal pace (both teams)
EG_BINS = [(0, 3, 1.66), (3, 6, 3.69), (6, 9, 5.01), (9, 12, 3.84), (12, 20, 0.61), (20, 99, -0.82)]
OT_PTS = 14.6   # 5-minute overtime ~ 7.3 possessions a side at ~1.0 pts

def ncdf(x): return 0.5 * (1 + math.erf(x / math.sqrt(2)))
_nc = np.vectorize(ncdf)

def endgame_terms(m, sd):
    """expected end-game extra points and overtime chance from the projected margin m (array) and its spread sd.
    margin with 2:00 left ~ N(0.95 m, 0.95 sd) (most of the game is played)."""
    m = np.asarray(m, float); mu, s = 0.95 * m, 0.95 * sd
    ex = np.zeros_like(m)
    for lo, hi, x in EG_BINS:   # P(lo < |M| <= hi)
        p = (_nc((hi - mu) / s) - _nc((lo - mu) / s)) + (_nc((-lo - mu) / s) - _nc((-hi - mu) / s))
        ex += p * x
    pot = np.exp(-0.5 * (m / sd) ** 2) / (sd * math.sqrt(2 * math.pi)) * 1.0 * 1.25   # tie window ~1 pt; 1.25 matches the 5.3% OT rate
    return ex, pot

def v2_frame():
    """CBB v2 history: one row per game with v2 margin/total (fit 2013-2019), both engines, closes, opener (2026 DK)."""
    b = pd.read_parquet(M + "rebuild/basketball/data/baseline_cbb.parquet")
    r = pd.read_parquet(M + "rebuild/basketball/data/ablate_full_cbb.parquet", columns=["game_id", "pred_h", "pred_a", "poss_hat"]).rename(columns={"pred_h": "rb_h", "pred_a": "rb_a"})
    g = b.merge(r, on="game_id", how="left")
    P = pd.read_parquet(M + "cbb/data/cbb_predictions.parquet", columns=["game_id", "line_open", "total_open", "line_close_dk", "total_close_dk"])
    g = g.merge(P, on="game_id", how="left")
    A = pd.read_parquet(M + "cbb/data/cbb_availability.parquet", columns=["game_id", "team", "abs_val_cont"]).set_index(["game_id", "team"]).abs_val_cont.to_dict()
    g["x_av"] = [A.get((a, h), 0) - A.get((a, w), 0) for a, h, w in zip(g.game_id, g.home, g.away)]
    C = json.load(open(M + "rebuild/basketball/data/cbb_v2_coef.json")); cm, ct = C["margin"], C["total"]
    g["tm"], g["rm"] = g.pred_h - g.pred_a, g.rb_h - g.rb_a
    g["v2_m"] = cm["const"] + cm["tm"] * g.tm + cm["rm"] * g.rm + cm["x_av"] * g.x_av
    g["v2_t"] = ct["const"] + ct["tt"] * (g.pred_h + g.pred_a) + ct["rt"] * (g.rb_h + g.rb_a) + ct["poss_hat"] * g.poss_hat
    g["act_m"], g["act_t"] = g.act_h - g.act_a, g.act_h + g.act_a
    g["dt"] = pd.to_datetime(g.date)
    return g[g.v2_m.notna() & g.v2_t.notna()].sort_values("dt").reset_index(drop=True)

K = np.array([0, 1, 2, 3.])
BASE = np.array([0.47, 0.06, 0.30, 0.17])   # CBB possession outcome shares (0/1/2/3 pts), ppp ~1.03
def _tilt(ppp):
    lo, hi = -3.0, 3.0
    for _ in range(40):
        t = (lo + hi) / 2; q = BASE * np.exp(t * K); q /= q.sum()
        lo, hi = (t, hi) if (q * K).sum() < ppp else (lo, t)
    q = BASE * np.exp(lo * K); return q / q.sum()

def sim_game(hs, as_, poss, sd_m, sd_t, n=4000, seed=0, endgame=True):
    """Possession sim anchored to Model A's mean scores (hs, as_). Regulation is played possession by possession up to the
    last 2:00 (7 trips a side), then the end-game: when the margin is 4-12 the trailing side fouls (extra points from
    research/cbb-sim-2026-10/endgame.json), then overtime on a tie. The luck part is rescaled so the spread of results
    matches Model A's measured miss (sd_m margin, sd_t total), which includes our rating error, not just game luck.
    Returns simulated home and away points (arrays of length n)."""
    rng = np.random.default_rng(seed)
    N = np.clip(np.rint(rng.normal(poss, 3.0, n)), 50, 95).astype(int)
    qh, qa = _tilt(hs / poss), _tilt(as_ / poss)
    pts = lambda q, m: rng.multinomial(m, q) @ K
    reg = N - 7
    H1, A1 = pts(qh, reg), pts(qa, reg); H2, A2 = pts(qh, np.full(n, 7)), pts(qa, np.full(n, 7))
    H, A = H1 + H2, A1 + A2
    # rescale the luck to the measured miss: margin and total deviations separately
    m, t = H - A, H + A; mm, tt = m.mean(), t.mean()
    fm, ft = sd_m / max(m.std(), 1e-9), sd_t / max(t.std(), 1e-9)
    m2 = (hs - as_) + ((H1 - A1) - (H1 - A1).mean()) * fm      # margin with 2:00 left (regulation part)
    m = (hs - as_) + (m - mm) * fm; t = (hs + as_) + (t - tt) * ft
    if endgame:
        ex = np.zeros(n)
        for lo_, hi_, x in EG_BINS: ex += np.where((np.abs(m2) > lo_) & (np.abs(m2) <= hi_), x, 0.0)
        t = t + ex - ex.mean()        # fouling adds points to close games; centred per game so Model A's mean total is kept
        # (model_a_test.py: moving the MEAN total by end-game closeness made totals worse, t +9.8, so it only shapes the spread)
        tie = np.abs(m) < 0.5
        ot = rng.normal(0, 4.0, n) * tie; m = np.where(tie, np.round(ot) + np.sign(ot + 1e-9), m)   # overtime decides a tie
        t = t + tie * OT_PTS; t = t - (tie * OT_PTS).mean()
    return (t + m) / 2, (t - m) / 2
