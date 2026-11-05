"""2026 model test: walk-forward logistic on bridge features. Train on all earlier months, predict the next month
(Mar..Oct). Compare Elo only vs full feature model, by level, and vs Pinnacle where 2026 prices are pulled."""
import sys, json
import numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression

def lg(p): p = np.clip(p, 1e-4, 1 - 1e-4); return np.log(p / (1 - p))
def ll(p, y): p = np.clip(p, 1e-4, 1 - 1e-4); return -np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))


def feats(F):
    X = pd.DataFrame(index=F.index)
    pe = 1 / (1 + 10 ** ((F.elo_b - F.elo_a) / 400)); ps = 1 / (1 + 10 ** ((F.elos_b - F.elos_a) / 400))
    X['elo'] = lg(pe); X['elos'] = lg(ps)
    X['conf'] = X.elo * np.log1p(np.minimum(F.n_a, F.n_b)) / 5
    X['new_a'] = (F.n_a < 10).astype(float) - (F.n_b < 10).astype(float)
    for c in ('days', 'hrs', 'lastmin'):
        pass
    lay = lambda d: ((d > 45) & (d < 400)).astype(float)
    X['layoff'] = lay(F.days_a) - lay(F.days_b)
    X['tm'] = F.tm_a - F.tm_b
    X['tmin'] = (F.tmin_a - F.tmin_b) / 100
    X['lastmin'] = (F.lastmin_a.fillna(0) - F.lastmin_b.fillna(0)) / 100
    X['quick'] = ((F.hrs_a < 20).astype(float) - (F.hrs_b < 20).astype(float))
    X['m21'] = F.m21_a - F.m21_b
    X['n90'] = (F.n90_a - F.n90_b) / 10
    aa, ab = F.age_a.fillna(25), F.age_b.fillna(25)
    X['tr'] = (F.tr_a - F.tr_b) / 100; X['tr180'] = (F.tr180_a - F.tr180_b) / 100
    X['age'] = (aa - ab) / 5; X['age2'] = ((aa - 26) ** 2 - (ab - 26) ** 2) / 25
    if 'home' in F: X['home'] = F.home
    if 'gelo' in F: X['gelo'] = F.gelo; X['gelo_x_new'] = F.gelo / np.sqrt(1 + F.gelo_n / 20)
    return X


def walk_ym(F, cols, first='2026-01', min_train_from=None):
    """walk-forward by calendar month across years: each test month (from `first`) is fit on all earlier rows"""
    F = F[F.y.notna()].copy(); F['ym'] = F.start.dt.strftime('%Y-%m'); F['mon'] = F.start.dt.month
    X = feats(F); out = pd.Series(np.nan, index=F.index); lr = None
    for ym in sorted(F.ym.unique()):
        if ym < first: continue
        tr, te = F.ym < ym, F.ym == ym
        Xt = pd.concat([X.loc[tr, cols], -X.loc[tr, cols]]); yt = np.r_[F.y[tr], 1 - F.y[tr]]
        lr = LogisticRegression(C=1.0, fit_intercept=False, max_iter=500).fit(Xt, yt)
        out[te] = lr.predict_proba(X.loc[te, cols])[:, 1]
    return F, out, lr


def walk(F, cols):
    F = F[F.y.notna()].copy(); F['mon'] = F.start.dt.month
    X = feats(F); out = pd.Series(np.nan, index=F.index)
    for m in range(3, 11):
        tr, te = F.mon < m, F.mon == m
        if te.sum() == 0: continue
        # symmetric augmentation: flip sides
        Xt = pd.concat([X.loc[tr, cols], -X.loc[tr, cols]]); yt = np.r_[F.y[tr], 1 - F.y[tr]]
        lr = LogisticRegression(C=1.0, fit_intercept=False, max_iter=500).fit(Xt, yt)
        out[te] = lr.predict_proba(X.loc[te, cols])[:, 1]
    return F, out, lr


if __name__ == '__main__':
    BASE = ['elo']
    FULL = ['elo', 'elos', 'conf', 'new_a', 'layoff', 'm21', 'n90', 'age', 'age2']
    FULL = FULL + ['home']
    STEPS = [FULL, FULL + ['gelo', 'gelo_x_new']]
    _ = [['elo', 'elos'], ['elo', 'elos', 'conf', 'new_a'], ['elo', 'elos', 'conf', 'new_a', 'layoff'],
             ['elo', 'elos', 'conf', 'new_a', 'layoff', 'tm', 'tmin', 'lastmin', 'quick'],
             ['elo', 'elos', 'conf', 'new_a', 'layoff', 'tm', 'tmin', 'lastmin', 'quick', 'm21', 'n90'],
             ['elo', 'elos', 'conf', 'new_a', 'layoff', 'tm', 'tmin', 'lastmin', 'quick', 'm21', 'n90', 'age', 'age2']]
    for tour in ('atp', 'wta'):
        F = pd.read_parquet(f'../data/feat26_{tour}.parquet')
        F, pb, _ = walk(F, BASE)
        res = {'elo': pb}
        for i, cols in enumerate(STEPS):
            _, p, lr = walk(pd.read_parquet(f'../data/feat26_{tour}.parquet'), cols)
            res[f's{i}'] = p
        R = pd.DataFrame(res); R['y'] = F.y; R['cat'] = F.cat; R = R.dropna()
        print(tour, 'Mar-Oct 2026, s0 = current model, s1 + games rating')
        print(R.groupby('cat').apply(lambda g: pd.Series({k: ll(g[k], g.y) for k in res} | {'n': len(g)})).round(4).to_string())
        print('coef last fit:', dict(zip(STEPS[-1], np.round(lr.coef_[0], 3))))
        R.join(F[['fid']]).to_parquet(f'../data/pred26_{tour}.parquet')
