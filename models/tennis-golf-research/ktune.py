import sys, numpy as np, pandas as pd, bridge, bridge2
from model26 import walk, ll
FULL = ['elo', 'elos', 'conf', 'new_a', 'layoff', 'm21', 'n90', 'age', 'age2']
for kn, ko, kp in [(250, 5, .4), (350, 5, .4), (250, 5, .3), (400, 5, .35), (200, 5, .4)]:
    bridge.K.update(k_num=kn, k_off=ko, k_pow=kp); bridge2.K.update(k_num=kn, k_off=ko, k_pow=kp)
    out = []
    for t in ('atp', 'wta'):
        import contextlib, io
        with contextlib.redirect_stdout(io.StringIO()): bridge2.run(t)
        F, p, _ = walk(pd.read_parquet(f'../data/feat26_{t}.parquet'), FULL)
        g = pd.DataFrame(dict(p=p, y=F.y, c=F.cat)).dropna()
        out += [f'{c} {ll(z.p, z.y):.4f}' for c, z in g.groupby('c')]
    print(kn, ko, kp, ' | '.join(out), flush=True)
