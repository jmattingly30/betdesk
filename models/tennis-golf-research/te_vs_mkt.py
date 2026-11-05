"""Our model vs the market at every level, 2018-24 (te odds). Market fair = power de-vig of the closing price
(Betfair, else bet365), opener the same. Blend = logistic on [logit market, logit model], fit on earlier years only."""
import numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression
import warnings; warnings.filterwarnings('ignore')


def lg(p): p = np.clip(p, 1e-4, 1 - 1e-4); return np.log(p / (1 - p))
def ll(p, y): p = np.clip(p, 1e-4, 1 - 1e-4); return -np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))
def fair(a, b):
    ia, ib = 1 / a, 1 / b; lo = np.ones_like(ia); hi = np.full_like(ia, 3.0)
    for _ in range(40):
        k = (lo + hi) / 2; big = ia ** k + ib ** k >= 1; lo = np.where(big, k, lo); hi = np.where(big, hi, k)
    return ia ** ((lo + hi) / 2)


LV = {'Grand Slams': 'Slam', 'Masters': 'Tour', '250 or 500': 'Tour', 'Challenger': 'Challenger', 'Premier': 'Tour', 'Premier Mandatory': 'Tour',
      'International': 'Tour', 'C100': 'ITF 60-100k', 'C75': 'ITF 60-100k', 'C50': 'ITF 25-50k', 'C25': 'ITF 25-50k', 'C10': 'ITF 10-15k'}
if __name__ == '__main__':
    res = []
    for tour in ('atp', 'wta'):
        T = pd.read_parquet(f'../data/te_{tour}.parquet')
        T = T[(T.completed == 'completed') & T.p_model.notna()]
        T['lv'] = T.level.map(LV); T = T[T.lv.notna()]
        T['yr'] = T.date.dt.year
        for sfx, nm in (('', 'close'), ('_opening', 'open')):
            pb = fair(T[f'Betfair_a{sfx}'].values, T[f'Betfair_b{sfx}'].values)
            p3 = fair(T[f'bet365_a{sfx}'].values, T[f'bet365_b{sfx}'].values)
            ok_b = (T[f'Betfair_a{sfx}'] > 1) & (T[f'Betfair_b{sfx}'] > 1) & (1 / T[f'Betfair_a{sfx}'] + 1 / T[f'Betfair_b{sfx}'] < 1.08)
            T['m_' + nm] = np.where(ok_b, pb, p3)
        T = T[T.m_close.between(0.01, 0.99) & T.m_open.between(0.01, 0.99)]
        for lv, z in T.groupby('lv'):
            out = {}
            for yr in range(2018, 2025):
                tr, te = z[(z.yr < yr) & (z.yr >= yr - 4)], z[z.yr == yr]
                if len(te) == 0 or len(tr) < 500: continue
                for k in ('close', 'open'):
                    X = np.c_[lg(tr['m_' + k]), lg(tr.p_model)]
                    lr = LogisticRegression(C=10).fit(np.r_[X, -X], np.r_[tr.y, 1 - tr.y])
                    Xt = np.c_[lg(te['m_' + k]), lg(te.p_model)]
                    out.setdefault('blend_' + k, []).append(pd.Series(lr.predict_proba(Xt)[:, 1], te.index))
                    out.setdefault('w_' + k, []).append(lr.coef_[0][1] / max(lr.coef_[0][0], 1e-6))
            if not out: continue
            idx = pd.concat(out['blend_close']).index; zz = z.loc[idx]
            res.append(dict(tour=tour, level=lv, n=len(zz), model=ll(zz.p_model, zz.y), mkt_open=ll(zz.m_open, zz.y), mkt_close=ll(zz.m_close, zz.y),
                            blend_open=ll(pd.concat(out['blend_open']).loc[idx], zz.y), blend_close=ll(pd.concat(out['blend_close']).loc[idx], zz.y),
                            model_weight_vs_open=np.mean(out['w_open']), model_weight_vs_close=np.mean(out['w_close'])))
    R = pd.DataFrame(res)
    R['gain_open'] = R.mkt_open - R.blend_open; R['gain_close'] = R.mkt_close - R.blend_close
    print(R.round(4).to_string())
    R.to_csv('../../op/te_vs_mkt.csv', index=False)
    