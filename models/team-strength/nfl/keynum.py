"""NFL final-margin distribution with key numbers (J 2026-10-11 blowout ask). The app priced every NFL spread with a smooth
normal curve (sigma 13.3). Real margins pile up on 3, 7, 10, 14 and the normal curve over-states "wins by 8+" by 5-8 points
at every spread. This builds P(favourite margin = k | expected margin s) from games with similar closing spreads (Gaussian
kernel on the spread, bandwidth BW), blended with a discretised normal for the tails, and tests it leave-one-season-out.
usage: python3 nfl/keynum.py [out.json]      -> nfl/keynum_log.txt (stdout) and the app table (default ../../app/data/nfl_margin.json)"""
import sys, os, json, numpy as np, pandas as pd
from scipy.stats import norm
H = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
out = sys.argv[1] if len(sys.argv) > 1 else os.path.normpath(f'{H}/../../app/data/nfl_margin.json')
g = pd.read_parquet(f'{H}/data/nfl_games_features.parquet')
g = g[g.margin.notna() & g.spread_line.notna() & g.season.between(2006, 2025)].copy()
g['s'] = g.spread_line.abs(); g['m'] = np.where(g.spread_line >= 0, g.margin, -g.margin)
K = np.arange(-60, 61); SIG = 13.3

def normal_pmf(s, sig=SIG):
    c = norm.cdf((K + 0.5 - s) / sig) - norm.cdf((K - 0.5 - s) / sig); c[K == 0] = 0   # NFL ties ~0.2%: fold into +-1
    return c / c.sum()

def emp_pmf(d, s, bw, lam):
    w = np.exp(-0.5 * ((d.s.values - s) / bw) ** 2)
    # residual-shift: each game's margin moved by (s - its spread) rounded, so key numbers stay where they are in
    # absolute terms only for games with nearly the same spread (the kernel does that), tails come from the normal mix
    h = np.bincount(np.clip(d.m.values.astype(int) + 60, 0, 120), weights=w, minlength=121)
    h[K == 0] = 0; h = h / h.sum()
    p = (1 - lam) * h + lam * normal_pmf(s)
    return p / p.sum()

def score(pf, d):
    """log loss of the favourite's cover at line + offset (alt lines -6.5..+6.5 around the spread), win by 8+, by 14+."""
    ll = {}
    for name, thr in [('cover', 0), ('alt-3.5', -3.5), ('alt+3.5', 3.5), ('alt+6.5', 6.5), ('8+', None), ('14+', None)]:
        L = []
        for s, m in zip(d.s.values, d.m.values):
            p = pf(s)
            if name in ('8+', '14+'): cut = 7.5 if name == '8+' else 13.5
            else: cut = s + thr
            # whole-number lines: drop pushes (bet refunded)
            if cut == int(cut) and m == cut: continue
            pw = p[K > cut].sum(); pl = p[K < cut].sum(); q = pw / (pw + pl); y = m > cut
            L.append(-np.log(q if y else 1 - q))
        ll[name] = np.mean(L)
    return ll

res = {}
for bw in [0.75, 1.0, 1.5]:
    for lam in [0.1, 0.25, 0.4]:
        tot = {}
        for y in range(2010, 2026):
            tr, te = g[g.season != y], g[g.season == y]
            cache = {}
            def pf(s, tr=tr):
                k = round(s * 2) / 2
                if k not in cache: cache[k] = emp_pmf(tr, k, bw, lam)
                return cache[k]
            for kk, v in score(pf, te).items(): tot.setdefault(kk, []).append(v * len(te))
        res[(bw, lam)] = {k: sum(v) / len(g[g.season >= 2010]) for k, v in tot.items()}
te_all = g[g.season >= 2010]
base = score(lambda s: normal_pmf(s), te_all)
print(f'leave-one-season-out log loss, 2010-2025 ({len(te_all)} games); lower is better. Normal curve sigma {SIG}:')
print('  normal          ' + '  '.join(f'{k} {v:.4f}' for k, v in base.items()))
for (bw, lam), v in res.items():
    print(f'  bw {bw:4} mix {lam:4}  ' + '  '.join(f'{k} {x:.4f}' for k, x in v.items()))
best = min(res, key=lambda k: sum(res[k].values())); bw, lam = best
print('chosen', best, 'gain vs normal (log loss x1000):', {k: round(1000 * (base[k] - res[best][k]), 2) for k in base})

# calibration of the chosen table for "wins by 8+" and push chances at 3 and 7
print('\nchance the favourite wins by 8+ / pushes 3 / pushes 7, by closing spread: actual vs table vs normal')
for lo, hi in [(0, 2.5), (2.5, 3.5), (3.5, 6.5), (6.5, 7.5), (7.5, 9.5), (9.5, 13.5), (13.5, 30)]:
    d = te_all[(te_all.s > lo) & (te_all.s <= hi)]
    pt = np.mean([emp_pmf(g[g.season != y], round(s * 2) / 2, bw, lam)[K >= 8].sum() for s, y in zip(d.s, d.season)])
    pn = np.mean([normal_pmf(s)[K >= 8].sum() for s in d.s])
    print(f'  spread {lo:4}-{hi:4} n {len(d):4d}: actual {np.mean(d.m >= 8):.3f}  table {pt:.3f}  normal {pn:.3f}')
for k in (3, 7, 10):
    d = te_all[te_all.s == k]
    print(f'  spread exactly {k}: push actual {np.mean(d.m == k):.3f} (n {len(d)}), table {emp_pmf(g, k, bw, lam)[K == k][0]:.3f}, normal {normal_pmf(k)[K == k][0]:.3f}')

# app table: favourite-margin pmf for expected margins 0..24 in half points, all 2006-2025 games, as cumulative P(margin > k)
tab = {}
for s in np.arange(0, 24.5, 0.5):
    p = emp_pmf(g, s, bw, lam); tab[f'{s:.1f}'] = [round(float(x), 5) for x in p[(K >= -45) & (K <= 60)]]
json.dump(dict(made='2026-10-11', kmin=-45, kmax=60, bw=bw, mix=lam, sigma=SIG, note='P(favourite final margin = k) for k = kmin..kmax, '
               'keyed by the favourite\'s expected margin (0 to 24 by 0.5); interpolate between keys; ties folded out',
               pmf=tab), open(out, 'w'), separators=(',', ':'))
print('->', out, os.path.getsize(out) // 1024, 'KB')
