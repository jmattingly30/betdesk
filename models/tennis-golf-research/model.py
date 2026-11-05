"""Walk-forward match-win model: blend Elo / surface Elo / point model + context, add factors one at a time.
Each test year is fit on the previous 8 years only."""
import sys
import numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression

tour = sys.argv[1] if len(sys.argv) > 1 else 'atp'


def lgt(p):
    p = np.clip(p, 1e-4, 1 - 1e-4)
    return np.log(p / (1 - p))


def ll(p, y):
    p = np.clip(p, 1e-4, 1 - 1e-4)
    return -np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))


def groups(level):
    lvl = level.astype(str)
    return np.where(lvl == 'Grand Slams', 'Slam',
           np.where(lvl.isin(['Masters', '250 or 500', 'Tour Finals', 'Premier', 'Premier Mandatory', 'International']), 'Tour',
           np.where(lvl == 'Challenger', 'Challenger',
           np.where((lvl == 'Futures') | lvl.str.match(r'^C\d+'), 'ITF', 'Other'))))


def features(F):
    X = pd.DataFrame(index=F.index)
    X['l_elo'] = lgt(F.p_elo); X['l_elos'] = lgt(F.p_elos)
    ok = (F.vs_a < 0.2) & (F.vs_b < 0.2)
    X['l_pt'] = np.where(ok, lgt(F.p_pt), 0); X['pt_ok'] = ok.astype(float)
    X['l_pt_x_bo5'] = X.l_pt * (F.bo == 5)
    X['l_elo_x_bo5'] = X.l_elo * (F.bo == 5)
    X['conf'] = (np.log1p(F.n_a) - np.log1p(F.n_b))
    X['l_elo_x_new'] = X.l_elo / np.sqrt(1 + np.minimum(F.n_a, F.n_b) / 20)
    ra, rb = F.rank_a.fillna(2000), F.rank_b.fillna(2000)
    X['lrank'] = np.log(rb) - np.log(ra)
    # context
    X['age'] = (F.age_a - F.age_b).fillna(0)
    X['age_old'] = (np.maximum(F.age_a - 30, 0) - np.maximum(F.age_b - 30, 0)).fillna(0)
    X['age_young'] = (np.maximum(21 - F.age_a, 0) - np.maximum(21 - F.age_b, 0)).fillna(0)
    X['ht'] = (F.ht_a - F.ht_b).fillna(0) / 10
    X['ht_x_grass'] = X.ht * (F.surface == 'Grass'); X['ht_x_clay'] = X.ht * (F.surface == 'Clay')
    X['lefty'] = (F.hand_a == 'L').astype(float) - (F.hand_b == 'L').astype(float)
    for e in ['Q', 'WC', 'LL', 'PR']:
        X['ent_' + e] = (F.entry_a == e).astype(float) - (F.entry_b == e).astype(float)
    da, db = F.days_a.fillna(365), F.days_b.fillna(365)
    X['layoff'] = (np.log1p(np.minimum(da, 400)) - np.log1p(np.minimum(db, 400)))
    X['long_lay'] = (da > 60).astype(float) - (db > 60).astype(float)
    X['m21'] = F.m21_a - F.m21_b
    X['g21'] = (F.g21_a - F.g21_b) / 20
    X['tm'] = F.tm_a - F.tm_b
    X['tg'] = (F.tg_a - F.tg_b) / 20
    X['last_tg'] = (F.last_tg_a - F.last_tg_b) / 10
    X['last_tg_x_bo5'] = X.last_tg * (F.bo == 5)
    X['n90'] = (F.n90_a - F.n90_b) / 10
    X['h2h'] = np.sign(F.h2h) * np.log1p(np.abs(F.h2h))
    import os
    if os.path.exists(f'../data/gelo_{tour}.parquet'):
        G = pd.read_parquet(f'../data/gelo_{tour}.parquet').reindex(F.index)
        X['gelo'] = G.gelo.fillna(0); X['gelo_x_new'] = X.gelo / np.sqrt(1 + G.gelo_n.fillna(0) / 20)
    return X


BASE = ['l_elo', 'l_elos']
STEPS = [['gelo', 'gelo_x_new'], ['l_pt', 'pt_ok', 'l_pt_x_bo5', 'l_elo_x_bo5'], ['conf', 'l_elo_x_new'], ['lrank'],
         ['age', 'age_old', 'age_young'], ['ht', 'ht_x_grass', 'ht_x_clay'], ['lefty'],
         ['ent_Q', 'ent_WC', 'ent_LL', 'ent_PR'], ['layoff', 'long_lay'], ['m21', 'g21', 'n90'],
         ['tm', 'tg', 'last_tg', 'last_tg_x_bo5'], ['h2h']]


def walk(F, X, cols, years, train_years=8, C=1.0):
    preds = pd.Series(np.nan, index=F.index)
    yr = F.date.dt.year
    for t in years:
        tr = (yr < t) & (yr >= t - train_years) & F.use
        te = (yr == t) & F.pred_ok
        m = LogisticRegression(C=C, max_iter=500)
        m.fit(X.loc[tr, cols], F.y[tr])
        preds[te] = m.predict_proba(X.loc[te, cols])[:, 1]
    return preds, m


if __name__ == '__main__':
    F = pd.read_parquet(f'../data/feat_{tour}.parquet')
    F['g'] = groups(F.level)
    F['use'] = (~F.ret) & (F.n_a >= 5) & (F.n_b >= 5) & (F.date.dt.year >= 2002)
    F['pred_ok'] = (F.n_a >= 5) & (F.n_b >= 5)  # predict retired matches too (graded only on completed ones)
    X = features(F)
    years = range(2012, 2025)
    test = F.use & (F.date.dt.year >= 2012)
    cols = list(BASE)
    out = []
    p0, _ = walk(F, X, cols, years)
    F['p_base'] = p0
    def rep(name, p):
        r = {'step': name}
        for g in ['Slam', 'Tour', 'Challenger', 'ITF', 'all']:
            z = test & ((F.g == g) if g != 'all' else True)
            r[g] = round(ll(p[z], F.y[z]), 4)
        return r
    out.append(rep('elo+surface elo (refit)', p0))
    best = p0
    for st in STEPS:
        p, m = walk(F, X, cols + st, years)
        r = rep('+ ' + ','.join(st), p)
        val = test & (F.date.dt.year < 2018)
        gain = ll(best[val], F.y[val]) - ll(p[val], F.y[val])
        r['gain_val12_17'] = round(gain, 5)
        tst = test & (F.date.dt.year >= 2018)
        r['gain_test18_24'] = round(ll(best[tst], F.y[tst]) - ll(p[tst], F.y[tst]), 5)
        keep = gain > 0.0002
        r['kept'] = keep
        out.append(r)
        if keep:
            cols += st; best = p
    print(pd.DataFrame(out).to_string())
    F['p_model'] = best
    print('final cols', cols)
    print(dict(zip(cols, np.round(m.coef_[0], 3))) if False else '')
    p, m = walk(F, X, cols, [2024])
    print(dict(zip(cols, np.round(m.coef_[0], 3))))
    F[['p_base', 'p_model']].to_parquet(f'../data/pred_{tour}.parquet')
