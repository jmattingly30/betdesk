"""Live 2026 model: old 'Last F.' name mapping (a) vs tennisexplorer-id identity (b); log loss vs the close by level."""
import numpy as np, pandas as pd
import warnings; warnings.filterwarnings('ignore')
from model26 import walk, ll
from te_vs_mkt import fair
FULL = ['elo', 'elos', 'conf', 'new_a', 'layoff', 'm21', 'n90', 'age', 'age2', 'gelo', 'gelo_x_new']
res = {}
for v in ('a', 'b'):
    for tour in ('atp', 'wta'):
        F = pd.read_parquet(f'../data/featx26{v}_{tour}.parquet')
        G, p, _ = walk(F, FULL); G['p'] = p
        ok = (G.oa > 1) & (G.ob > 1) & (1 / G.oa + 1 / G.ob < 1.15)
        G['close'] = np.where(ok, fair(G.oa.fillna(2).values, G.ob.fillna(2).values), np.nan)
        G = G[G.p.notna() & G.close.notna()]
        G.to_parquet(f'../data/predx26{v}_{tour}.parquet')
        for c, z in G.groupby('cat'):
            res.setdefault((tour, c), {})[v] = (len(z), ll(z.p, z.y), ll(z.close, z.y))
for k, d in res.items():
    print(k, 'n', d['a'][0], d['b'][0], 'old', round(d['a'][1], 4), 'ids', round(d['b'][1], 4), 'close', round(d['b'][2], 4))
