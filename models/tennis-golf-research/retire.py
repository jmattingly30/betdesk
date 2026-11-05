"""How predictable are retirements? (player-level risk of retiring in a match), ATP/WTA 2010-2024."""
import sys
import numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression
from model import ll
tour = sys.argv[1] if len(sys.argv) > 1 else 'atp'
F = pd.read_parquet(f'../data/feat_{tour}.parquet', columns=['date', 'tid', 'level', 'surface', 'bo', 'rnd', 'a', 'b', 'y', 'ret', 'score',
                                                          'age_a', 'age_b', 'days_a', 'days_b', 'last_tg_a', 'last_tg_b', 'tm_a', 'tm_b', 'm21_a', 'm21_b', 'n_a', 'n_b'])
F = F[F.date >= '2005-01-01'].reset_index(drop=True)
# who retired: the loser of a RET match
loser = np.where(F.y == 1, F.b, F.a)
F['ret_player'] = np.where(F.ret, loser, -1)
# long format: one row per player-match
rows = []
for side, o in (('a', 'b'), ('b', 'a')):
    rows.append(pd.DataFrame({'date': F.date, 'level': F.level.astype(str), 'surface': F.surface, 'bo': F.bo, 'p': F[side],
                              'age': F['age_' + side], 'days': F['days_' + side], 'last_tg': F['last_tg_' + side], 'tm': F['tm_' + side],
                              'm21': F['m21_' + side], 'n': F['n_' + side], 'retired': (F.ret_player == F[side]).astype(int)}))
L = pd.concat(rows).sort_values('date', kind='stable').reset_index(drop=True)
# prior retirements by player (strictly before this date)
L['ret_cum'] = L.groupby('p').retired.cumsum() - L.retired
L['m_cum'] = L.groupby('p').cumcount()
L['prev_date_ret'] = L.date.where(L.retired == 1)
L['last_ret_date'] = L.groupby('p').prev_date_ret.transform(lambda s: s.shift(1).ffill())
L['days_since_ret'] = (L.date - L.last_ret_date).dt.days
X = pd.DataFrame({'ret_rate': (L.ret_cum + 0.5) / (L.m_cum + 25), 'recent_ret': (L.days_since_ret < 60).astype(float),
                  'mid_ret': ((L.days_since_ret >= 60) & (L.days_since_ret < 365)).astype(float),
                  'age30': np.maximum(L.age.fillna(25) - 30, 0), 'layoff': (L.days.fillna(365) > 60).astype(float),
                  'last_tg': L.last_tg.fillna(0) / 10, 'm21': L.m21, 'bo5': (L.bo == 5).astype(float),
                  'clay': (L.surface == 'Clay').astype(float), 'itf': L.level.str.match(r'^(Futures|C\d+)').astype(float),
                  'chall': (L.level == 'Challenger').astype(float)})
tr = (L.date < '2018-01-01') & (L.date >= '2008-01-01'); te = L.date >= '2018-01-01'
m = LogisticRegression(max_iter=1000).fit(X[tr], L.retired[tr])
p = m.predict_proba(X[te])[:, 1]
y = L.retired[te].values
print('retire rate per player-match (2018-24)', round(y.mean(), 4))
print('coefs', dict(zip(X.columns, np.round(m.coef_[0], 3))))
print('logloss base', round(ll(np.full(len(y), y.mean()), y), 5), 'model', round(ll(p, y), 5))
q = pd.qcut(p, 20, labels=False, duplicates='drop')
t = pd.DataFrame({'q': q, 'p': p, 'y': y}).groupby('q').agg(n=('y', 'size'), pred=('p', 'mean'), act=('y', 'mean'))
print(t.tail(5).round(4).to_string())
top = p > np.quantile(p, 0.99)
print('top 1% risk: n', top.sum(), 'pred', round(p[top].mean(), 4), 'actual', round(y[top].mean(), 4))
