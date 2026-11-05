"""Tennis Model A engine (point -> game -> set -> match, exact Markov), built on research/model-a-sim-2026-10/tennis-pbp/markov.py.
One engine for every output: win %, set scores, total games, each player's games, games handicap.
Match = serve level mu (logit of the average serve-point win rate in this match) and gap d (logit A serve minus logit B serve,
A's edge). pa = sg(mu + d/2), pb = sg(mu - d/2). Match-day form: d mixed over N(0, SD) (research: sd ~.35 logit, essential).
Pressure shifts tour-wide only (research: men break point -.036, game point +.032; women deuce +.037; clutch not repeatable).
A grid over (mu, d) is built once per format and cached in grid_bo{3,5}_{m,w}.npz next to this file."""
import os, numpy as np
from functools import lru_cache
lg = lambda p: np.log(p / (1 - p)); sg = lambda x: 1 / (1 + np.exp(-x))
HERE = os.path.dirname(os.path.abspath(__file__))
ADJ = {'m': {'bp': -0.036, 'gp': 0.032}, 'w': {'dc': 0.037}}
SD = 0.35
MU = np.round(np.arange(-0.30, 1.21, 0.05), 3)          # serve-point win 43% .. 77%
DG = np.round(np.arange(-3.0, 3.0001, 0.025), 3)         # gap grid
GMAX = 70

def game_win(p, a):
    @lru_cache(None)
    def f(i, j):
        if i >= 4 and i - j >= 2: return 1.
        if j >= 4 and j - i >= 2: return 0.
        if i >= 3 and j >= 3 and i == j:
            pd_ = sg(lg(p) + a.get('dc', 0)); pa_ = sg(lg(p) + a.get('gp', 0)); pb_ = sg(lg(p) + a.get('bp', 0))
            w = pd_ * pa_; l = (1 - pd_) * (1 - pb_); return w / (w + l)
        x = lg(p)
        if j >= 3 and j > i: x += a.get('bp', 0)
        elif i >= 3 and i > j: x += a.get('gp', 0)
        elif i == 2 and j == 2: x += a.get('dc', 0)
        q = sg(x); return q * f(i + 1, j) + (1 - q) * f(i, j + 1)
    return f(0, 0)

def tiebreak(pa, pb):
    @lru_cache(None)
    def f(i, j):
        if i >= 7 and i - j >= 2: return 1.
        if j >= 7 and j - i >= 2: return 0.
        if i >= 6 and j >= 6 and i == j:
            w = pa * (1 - pb); l = (1 - pa) * pb; return w / (w + l)
        q = pa if (i + j) % 4 in (0, 3) else 1 - pb
        return q * f(i + 1, j) + (1 - q) * f(i, j + 1)
    return f(0, 0)

def set_dist(ha, hb, ta):
    out = {}; st = {(0, 0): 1.}
    while st:
        nx = {}
        for (i, j), pr in st.items():
            if i == 6 and j == 6:
                out[(7, 6)] = out.get((7, 6), 0) + pr * ta; out[(6, 7)] = out.get((6, 7), 0) + pr * (1 - ta); continue
            pA = ha if (i + j) % 2 == 0 else 1 - hb
            for (ni, nj), q in (((i + 1, j), pA), ((i, j + 1), 1 - pA)):
                if (ni >= 6 or nj >= 6) and abs(ni - nj) >= 2: out[(ni, nj)] = out.get((ni, nj), 0) + pr * q
                else: nx[(ni, nj)] = nx.get((ni, nj), 0) + pr * q
        st = nx
    return out

def match_joint(pa, pb, bo, adj):
    """returns P(A wins), joint array J[ga, gb], set-score dict"""
    ha = game_win(pa, adj); hb = game_win(pb, adj); ta = tiebreak(pa, pb)
    sA = set_dist(ha, hb, ta)
    sB = {(y, x): v for (x, y), v in set_dist(hb, ha, 1 - ta).items()}
    need = (bo + 1) // 2
    J = np.zeros((GMAX, GMAX)); SC = {}; win = 0.
    st = {(0, 0, 0): {(0, 0): .5}, (0, 0, 1): {(0, 0): .5}}
    while st:
        nx = {}
        for (x, y, sv), gd in st.items():
            sd = sA if sv == 0 else sB
            for (i, j), q in sd.items():
                nsv = sv if (i + j) % 2 == 0 else 1 - sv
                ns = (x + (i > j), y + (j > i), nsv)
                for (ga, gb), pr in gd.items():
                    k = (ga + i, gb + j); pp = pr * q
                    if ns[0] == need or ns[1] == need:
                        if ns[0] == need: win += pp
                        J[min(k[0], GMAX - 1), min(k[1], GMAX - 1)] += pp
                        SC[(ns[0], ns[1])] = SC.get((ns[0], ns[1]), 0) + pp
                    else:
                        dd = nx.setdefault(ns, {}); dd[k] = dd.get(k, 0) + pp
        st = nx
    return win, J, SC

def build_grid(bo, sex):
    f = f'{HERE}/grid_bo{bo}_{sex}.npz'
    if os.path.exists(f): return dict(np.load(f))
    need = (bo + 1) // 2
    scores = [(need, k) for k in range(need)] + [(k, need) for k in range(need)]
    W = np.zeros((len(MU), len(DG))); TG = np.zeros((len(MU), len(DG), 2 * GMAX)); MG = np.zeros((len(MU), len(DG), 2 * GMAX + 1))
    GA = np.zeros((len(MU), len(DG), GMAX)); SS = np.zeros((len(MU), len(DG), len(scores)))
    ii, jj = np.indices((GMAX, GMAX))
    for a, mu in enumerate(MU):
        for b, d in enumerate(DG):
            w, J, SC = match_joint(sg(mu + d / 2), sg(mu - d / 2), bo, ADJ[sex])
            W[a, b] = w
            TG[a, b] = np.bincount((ii + jj).ravel(), J.ravel(), 2 * GMAX)
            MG[a, b] = np.bincount((ii - jj + GMAX).ravel(), J.ravel(), 2 * GMAX + 1)
            GA[a, b] = J.sum(1)
            SS[a, b] = [SC.get(s, 0) for s in scores]
    # match-day form: mix the gap over N(0, SD) on the grid
    z, wz = np.polynomial.hermite_e.hermegauss(15); wz = wz / wz.sum()
    out = {}
    for nm, A in (('W', W), ('TG', TG), ('MG', MG), ('GA', GA), ('SS', SS)):
        M = np.zeros_like(A)
        for zz, ww in zip(z, wz):
            idx = np.clip(np.round((DG + zz * SD - DG[0]) / (DG[1] - DG[0])).astype(int), 0, len(DG) - 1)
            M += ww * A[:, idx]
        out[nm] = M
    out['scores'] = np.array(scores); out['MU'] = MU; out['DG'] = DG
    np.savez_compressed(f, **{k: (v.astype(np.float32) if v.dtype == np.float64 and k not in ('MU', 'DG') else v) for k, v in out.items()})
    return out

_G = {}
def grid(bo, sex):
    if (bo, sex) not in _G: _G[(bo, sex)] = build_grid(bo, sex)
    return _G[(bo, sex)]

def project(p, mu, bo=3, sex='m'):
    """p = Model A win chance for A (from our ratings), mu = serve level. Returns dict of distributions/summary."""
    g = grid(bo, sex)
    mu = float(np.clip(mu, MU[0], MU[-1])); a = np.interp(mu, MU, np.arange(len(MU)))
    a0 = int(min(np.floor(a), len(MU) - 2)); fa = a - a0
    def at(nm, b0, fb):
        A = g[nm]
        return ((1 - fa) * ((1 - fb) * A[a0, b0] + fb * A[a0, b0 + 1]) + fa * ((1 - fb) * A[a0 + 1, b0] + fb * A[a0 + 1, b0 + 1]))
    wrow = (1 - fa) * g['W'][a0] + fa * g['W'][a0 + 1]
    p = float(np.clip(p, wrow[0] + 1e-6, wrow[-1] - 1e-6))
    b = np.interp(p, wrow, np.arange(len(DG))); b0 = int(min(np.floor(b), len(DG) - 2)); fb = b - b0
    tg = at('TG', b0, fb); mg = at('MG', b0, fb); ga = at('GA', b0, fb); ss = at('SS', b0, fb)
    tg /= tg.sum(); mg /= mg.sum(); ga /= ga.sum()
    k = np.arange(len(tg)); m = np.arange(len(mg)) - GMAX
    def q(pr, x, qq): return float(x[np.searchsorted(np.cumsum(pr), qq)])
    return dict(p=p, d=float(DG[b0] + fb * (DG[1] - DG[0])), mu=mu, tg=tg, mg=mg, ga=ga, ss=ss, scores=g['scores'],
                tot_mean=float((k * tg).sum()), tot_med=q(tg, k, .5), tot_lo=q(tg, k, .1), tot_hi=q(tg, k, .9),
                mar_mean=float((m * mg).sum()), mar_med=q(mg, m, .5), ga_mean=float((np.arange(len(ga)) * ga).sum()),
                ga_med=q(ga, np.arange(len(ga)), .5))

def p_over(r, line):
    k = np.arange(len(r['tg'])); return float(r['tg'][k > line].sum()), float(r['tg'][k < line].sum())
def p_cover(r, hcp):
    """A covers games handicap hcp (A's line, e.g. -3.5 means A must win by 4+)"""
    m = np.arange(len(r['mg'])) - GMAX; return float(r['mg'][m + hcp > 0].sum()), float(r['mg'][m + hcp < 0].sum())
