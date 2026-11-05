"""iid point model: serve-point win probabilities -> game, tiebreak, set, match distributions.

pa = P(A wins a point on A's serve), pb = P(B wins a point on B's serve).
Exact dynamic programming (Barnett & Clarke style), with grid tables for speed.
"""
import numpy as np
from functools import lru_cache


def p_game(p):
    """P(server holds) from point-on-serve p."""
    q = 1 - p
    deuce = p * p / (1 - 2 * p * q)
    return p**4 * (1 + 4 * q + 10 * q * q) + 20 * p**3 * q**3 * deuce


def p_tiebreak(pa, pb, target=7):
    """P(A wins tiebreak to `target`); A serves first point, then alternate in pairs."""
    # state (a, b) points; server of point k: k=0 A, k=1,2 B, k=3,4 A ...
    @lru_cache(None)
    def f(a, b):
        if a >= target and a - b >= 2:
            return 1.0
        if b >= target and b - a >= 2:
            return 0.0
        if a >= target - 1 and b >= target - 1 and a == b:
            # deuce-like state: two points, one on each serve
            k = a + b
            s1 = 'A' if (k % 4 in (0, 3)) else 'B'
            x = pa if s1 == 'A' else 1 - pb
            y = pa if s1 == 'B' else 1 - pb  # next point other server
            win2 = x * y
            lose2 = (1 - x) * (1 - y)
            return win2 / (win2 + lose2)
        k = a + b
        x = pa if (k % 4 in (0, 3)) else 1 - pb
        return x * f(a + 1, b) + (1 - x) * f(a, b + 1)
    return f(0, 0)


def set_dist(pa, pb, a_serves_first=True, tb_at=6, final_adv=False):
    """Distribution over set scores. Returns dict {(ga, gb): prob} and P(next set A serves first)."""
    ha, hb = p_game(pa), p_game(pb)
    tba = p_tiebreak(pa, pb)          # A serves first in TB
    tbb = 1 - p_tiebreak(pb, pa)      # B serves first in TB
    out = {}
    # state probs over (ga, gb)
    from collections import defaultdict
    cur = {(0, 0): 1.0}
    while cur:
        nxt = defaultdict(float)
        for (ga, gb), pr in cur.items():
            n = ga + gb
            a_srv = (n % 2 == 0) == a_serves_first
            if ga == tb_at and gb == tb_at and not final_adv:
                ptb = tba if a_srv else tbb
                out[(ga + 1, gb)] = out.get((ga + 1, gb), 0) + pr * ptb
                out[(ga, gb + 1)] = out.get((ga, gb + 1), 0) + pr * (1 - ptb)
                continue
            pw = ha if a_srv else 1 - hb
            for (na, nb), q in (((ga + 1, gb), pw), ((ga, gb + 1), 1 - pw)):
                if (na >= 6 and na - nb >= 2) or (nb >= 6 and nb - na >= 2):
                    out[(na, nb)] = out.get((na, nb), 0) + pr * q
                elif final_adv and na + nb > 40:
                    # cap very long advantage sets: split by hold probs
                    out[(na, nb)] = out.get((na, nb), 0) + pr * q
                else:
                    nxt[(na, nb)] += pr * q
        cur = nxt
    return out


def match_dist(pa, pb, best_of=3):
    """Return dict with P(A wins), set-score probs, total-games pmf, games-margin pmf (A minus B)."""
    need = best_of // 2 + 1
    s_first = set_dist(pa, pb, True)
    s_second = set_dist(pa, pb, False)
    # server alternation: next set's first server depends on parity of games in set
    from collections import defaultdict
    states = {(0, 0, True, 0, 0): 1.0}  # sets a, sets b, A serves first, games a, games b
    final = defaultdict(float)
    while states:
        nxt = defaultdict(float)
        for (sa, sb, af, ga, gb), pr in states.items():
            dist = s_first if af else s_second
            for (x, y), q in dist.items():
                na, nb = sa + (x > y), sb + (y > x)
                naf = af if (x + y) % 2 == 0 else (not af)
                key = (na, nb, naf, ga + x, gb + y)
                if na == need or nb == need:
                    final[(na, nb, ga + x, gb + y)] += pr * q
                else:
                    nxt[key] += pr * q
        states = nxt
    pwin = sum(v for (a, b, _, _), v in final.items() if a == need)
    sets = defaultdict(float)
    tot = defaultdict(float)
    mar = defaultdict(float)
    for (a, b, x, y), v in final.items():
        sets[(a, b)] += v
        tot[x + y] += v
        mar[x - y] += v
    return {'p': pwin, 'sets': dict(sets), 'total': dict(tot), 'margin': dict(mar)}


GRID = np.round(np.arange(0.30, 0.951, 0.005), 3)


def build_table(best_of):
    T = np.zeros((len(GRID), len(GRID)))
    for i, a in enumerate(GRID):
        for j, b in enumerate(GRID):
            T[i, j] = match_dist(a, b, best_of)['p'] if False else _fast_p(a, b, best_of)
    return T


def _fast_p(pa, pb, best_of):
    """P(A wins match) via set-level recursion (exact; tracks first server parity)."""
    s1 = set_dist(pa, pb, True)
    s2 = set_dist(pa, pb, False)
    # per set: P(win & even games), P(win & odd), P(lose & even), P(lose & odd)
    def agg(d):
        we = sum(v for (x, y), v in d.items() if x > y and (x + y) % 2 == 0)
        wo = sum(v for (x, y), v in d.items() if x > y and (x + y) % 2 == 1)
        le = sum(v for (x, y), v in d.items() if x < y and (x + y) % 2 == 0)
        lo = sum(v for (x, y), v in d.items() if x < y and (x + y) % 2 == 1)
        return we, wo, le, lo
    A1, A2 = agg(s1), agg(s2)
    need = best_of // 2 + 1

    @lru_cache(None)
    def f(sa, sb, af):
        if sa == need:
            return 1.0
        if sb == need:
            return 0.0
        we, wo, le, lo = A1 if af else A2
        return (we * f(sa + 1, sb, af) + wo * f(sa + 1, sb, not af)
                + le * f(sa, sb + 1, af) + lo * f(sa, sb + 1, not af))
    return f(0, 0, True)


class Table:
    def __init__(self, best_of, path=None):
        import os
        path = path or os.path.join(os.path.dirname(__file__), '..', 'data', f'table_bo{best_of}.npy')
        self.T = np.load(path) if os.path.exists(path) else build_table(best_of)

    def __call__(self, pa, pb):
        pa = np.clip(pa, GRID[0], GRID[-1] - 1e-9)
        pb = np.clip(pb, GRID[0], GRID[-1] - 1e-9)
        fa = (pa - GRID[0]) / 0.005
        fb = (pb - GRID[0]) / 0.005
        i, j = np.floor(fa).astype(int), np.floor(fb).astype(int)
        da, db = fa - i, fb - j
        T = self.T
        return (T[i, j] * (1 - da) * (1 - db) + T[i + 1, j] * da * (1 - db)
                + T[i, j + 1] * (1 - da) * db + T[i + 1, j + 1] * da * db)
