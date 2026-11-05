"""Is the slam edge just favourite-longshot correction? Compare market-only recalibration vs market+model."""
import sys
import numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression
from model import ll, lgt
tour = sys.argv[1] if len(sys.argv) > 1 else 'atp'
S = pd.read_parquet(f'../data/slam_{tour}.parquet')
S = S[S.p_model.notna()]
for name, cols in (('recal', ['l_mkt']), ('blend', ['l_mkt', 'l_mod'])):
    S['p_' + name] = np.nan
    for t in range(2013, 2024):
        tr = S.yr < t; te = S.yr == t
        if te.sum() == 0: continue
        m = LogisticRegression(C=10).fit(S.loc[tr, cols], S.y[tr])
        S.loc[te, 'p_' + name] = m.predict_proba(S.loc[te, cols])[:, 1]
z = S.p_blend.notna()
print('n', z.sum(), 'logloss mkt', round(ll(S.p_mkt_pow[z], S.y[z]), 4), 'recal', round(ll(S.p_recal[z], S.y[z]), 4), 'blend', round(ll(S.p_blend[z], S.y[z]), 4))
def roi(pc, price, th):
    out = []
    for sd in 'ab':
        p = S[pc] if sd == 'a' else 1 - S[pc]; o = S[price + sd]; w = (S.y == 1) if sd == 'a' else (S.y == 0)
        sel = (p * o - 1 > th) & p.notna() & o.notna()
        out.append(pd.DataFrame({'r': np.where(w[sel], o[sel] - 1, -1.0), 'yr': S.yr[sel]}))
    r = pd.concat(out)
    return len(r), r.r.mean(), r.r.std() / np.sqrt(max(len(r), 1)), (r.groupby('yr').r.mean() > 0).mean()
rows = []
for pc in ('p_recal', 'p_blend'):
    for pr in ('PS', 'Max'):
        for th in (0, .02, .05):
            n, m, se, yrs = roi(pc, pr, th)
            rows.append(dict(model=pc, price=pr, edge=th, bets=n, roi=round(m, 4), se=round(se, 4), share_years_positive=round(yrs, 2)))
print(pd.DataFrame(rows).to_string())
S.to_parquet(f'../data/slam_{tour}.parquet')
