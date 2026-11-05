"""Baseline accuracy of Elo / surface Elo / point model by level, 2012-2024 (no retirements)."""
import sys
import numpy as np, pandas as pd
import iid

tour = sys.argv[1]
F = pd.read_parquet(f'../data/feat_{tour}.parquet')
T3, T5 = iid.Table(3), iid.Table(5)
bo = F.bo.fillna(3).values
F['p_pt'] = np.where(bo == 5, T5(F.pa.values, F.pb.values), T3(F.pa.values, F.pb.values))
F['p_elo'] = 1 / (1 + 10 ** ((F.elo_b - F.elo_a) / 400))
F['p_elos'] = 1 / (1 + 10 ** ((F.elos_b - F.elos_a) / 400))
F['p_eb'] = 0.5 * F.p_elo + 0.5 * F.p_elos
F.to_parquet(f'../data/feat_{tour}.parquet')


def ll(p, y):
    p = np.clip(p, 1e-4, 1 - 1e-4)
    return -np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))


x = F[(F.date >= '2012-01-01') & (~F.ret) & (F.n_a >= 10) & (F.n_b >= 10)]
lvl = x.level.astype(str)
grp = np.where(lvl.isin(['Grand Slams']), 'Slam', np.where(lvl.isin(['Masters', '250 or 500', 'Tour Finals', 'Premier', 'Premier Mandatory', 'International']), 'Tour',
               np.where(lvl.str.startswith('C') & (lvl != 'Challenger'), 'ITF', np.where(lvl == 'Challenger', 'Challenger', np.where(lvl == 'Futures', 'ITF', 'Other')))))
x = x.assign(g=grp)
res = []
for g, z in x.groupby('g'):
    r = {'group': g, 'n': len(z)}
    for c in ['p_elo', 'p_elos', 'p_eb', 'p_pt']:
        zz = z if c != 'p_pt' else z[(z.vs_a < 0.2) & (z.vs_b < 0.2)]
        r[c + '_ll'] = round(ll(zz[c], zz.y), 4)
        r[c + '_acc'] = round(((zz[c] > .5) == (zz.y == 1)).mean(), 4)
    r['n_pt'] = int(((z.vs_a < 0.2) & (z.vs_b < 0.2)).sum())
    res.append(r)
print(pd.DataFrame(res).to_string())
