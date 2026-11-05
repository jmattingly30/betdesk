"""Add a market rating (Kalman on logits of each player's PAST closing prices, never the match's own) to the 2026
tennisexplorer live model. Compare model, model+mkt, close, and blends with the close by level."""
import numpy as np, pandas as pd
from collections import defaultdict
from sklearn.linear_model import LogisticRegression
import warnings; warnings.filterwarnings('ignore')
from model26 import walk, lg, ll, feats
import model26
from te_vs_mkt import fair
FULL = ['elo', 'elos', 'conf', 'new_a', 'layoff', 'm21', 'n90', 'age', 'age2', 'gelo', 'gelo_x_new']

def mkt(G, q=0.003, obs=0.05):
    m = defaultdict(float); v = defaultdict(lambda: 1.0); last = {}
    mk = np.zeros(len(G)); va = np.ones(len(G)); vb = np.ones(len(G))
    G = G.sort_values('start')
    for i, r in enumerate(G[['a', 'b', 'start', 'close']].itertuples(index=False)):
        a, b, d = r.a, r.b, r.start
        for p in (a, b):
            if p in last: v[p] = min(1.0, v[p] + q * (d - last[p]).total_seconds() / 86400)
        mk[i], va[i], vb[i] = m[a] - m[b], v[a], v[b]
        if r.close == r.close:
            res = lg(r.close) - (m[a] - m[b]); S = v[a] + v[b] + obs
            m[a] += v[a] / S * res; m[b] -= v[b] / S * res
            v[a] *= 1 - v[a] / S; v[b] *= 1 - v[b] / S
        last[a] = last[b] = d
    return G.assign(mk=mk, mva=va, mvb=vb)

out = []
for tour in ('atp', 'wta'):
    F = pd.read_parquet(f'../data/featx26_{tour}.parquet')
    ok = (F.oa > 1) & (F.ob > 1) & (1 / F.oa + 1 / F.ob < 1.15)
    F['close'] = np.where(ok, fair(F.oa.fillna(2).values, F.ob.fillna(2).values), np.nan)
    F = mkt(F)
    known = np.clip(1 - np.maximum(F.mva, F.mvb), 0, 1)
    F['mkx'] = F.mk * known          # shrink rating toward 0 when either player has little price history
    F['mkn'] = known
    _feats = model26.feats
    def feats2(D):
        X = _feats(D); X['mkx'] = D.mkx.values; X['mkx_g'] = D.mkx.values * D.gelo.values; return X
    model26.feats = feats2
    res = {}
    for name, cols in (('model', FULL), ('model+mkt', FULL + ['mkx'])):
        G, p, lr = walk(F, cols); res[name] = p
    model26.feats = _feats
    G['p0'] = res['model']; G['p1'] = res['model+mkt']
    G = G[G.p0.notna() & G.p1.notna()]
    G.to_parquet(f'../data/predxm26_{tour}.parquet')
    print(tour, 'mkx coef', round(lr.coef_[0][-1], 3))
    for c, z in G[G.close.notna()].groupby('cat'):
        z = z.copy()
        for col in ('p0', 'p1'):
            z['bl_' + col] = np.nan
            for m in range(4, 11):
                tr, te = z[z.mon < m], z.mon == m
                if te.sum() == 0 or len(tr) < 300: continue
                X = np.c_[lg(tr.close), lg(tr[col])]
                l2 = LogisticRegression(C=10).fit(np.r_[X, -X], np.r_[tr.y, 1 - tr.y])
                z.loc[te, 'bl_' + col] = l2.predict_proba(np.c_[lg(z.close[te]), lg(z[col][te])])[:, 1]
        k = z.bl_p0.notna() & z.bl_p1.notna()
        out.append(dict(tour=tour, cat=c, n=int(k.sum()), model=ll(z.p0[k], z.y[k]), model_mkt=ll(z.p1[k], z.y[k]),
                        mkt_only=ll(1 / (1 + np.exp(-z.mkx[k])), z.y[k]), close=ll(z.close[k], z.y[k]),
                        bl_model=ll(z.bl_p0[k], z.y[k]), bl_model_mkt=ll(z.bl_p1[k], z.y[k])))
print(pd.DataFrame(out).round(4).to_string())
