"""Accuracy of prediction versions on the same matches: pick rate, log loss, Brier; closing line as yardstick. Usage: cmp_pred.py tour v1 v2 ..."""
import sys
import numpy as np, pandas as pd
tour, vs = sys.argv[1], sys.argv[2:]
P = {v: pd.read_parquet(f'../data/predx26{v}_{tour}.parquet') for v in vs}
common = set.intersection(*[set(p[p.p.notna()].fid) for p in P.values()])
for c in [c for c in ['Challenger', 'ITF', 'Tour'] if c in set(next(iter(P.values())).cat)]:
    print('==', c)
    for v, p in P.items():
        g = p[p.fid.isin(common) & (p.cat == c) & p.close.notna() & (p.start >= '2026-04-01')]
        y, q = g.y.values, g.p.clip(.01, .99).values
        exp = g.gelo_n.values >= 20
        print(f'  v{v or "0"}: n {len(g)} pick {np.mean((q > .5) == (y == 1)):.3f} LL {-np.mean(y*np.log(q)+(1-y)*np.log(1-q)):.4f} Brier {np.mean((q-y)**2):.4f} | close pick {np.mean((g.close > .5) == (y == 1)):.3f} | experienced pick {np.mean(((q > .5) == (y == 1))[exp]):.3f}')
