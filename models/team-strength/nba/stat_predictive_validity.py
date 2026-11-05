"""Which NBA team stats over games 1-41 predict results in games 42-82? (regular seasons 2008-2026)"""
import sys, numpy as np, pandas as pd
o = pd.read_parquet(sys.argv[1]); o = o[o.playoff == 0].sort_values(['season','team','date'])
o['margin'] = o.pts - o.o_pts; o['win'] = (o.margin > 0).astype(float)
o['close_win'] = np.where(o.margin.abs() <= 5, o.win, np.nan)
o['n'] = o.groupby(['season','team']).cumcount()
for a in ['efg','tov_pct','orb_pct','ftr','tp_pct','two_pct','ft_pct','tpa_rate']:
    o['net_'+a] = o[a] - o['o_'+a]
o['net_tov_pct'] = -o.net_tov_pct
STATS = ['margin','win','net','ortg','drtg','net_efg','efg','o_efg','net_tov_pct','net_orb_pct','net_ftr','tp_pct','o_tp_pct','net_tp_pct','two_pct','o_two_pct',
         'net_two_pct','ft_pct','o_ft_pct','tpa_rate','o_tpa_rate','net_tpa_rate','pace','close_win']
first, rest = o[o.n < 41], o[o.n >= 41]
F = first.groupby(['season','team'])[STATS].mean()
pf = first.groupby(['season','team']).agg(p=('pts','sum'), a=('o_pts','sum')).astype(float); F['pythag'] = pf.p**14/(pf.p**14 + pf.a**14)
R = rest.groupby(['season','team']).agg(fut_margin=('margin','mean'), fut_win=('win','mean'))
X = F.join(R, how='inner')
odd = o[o.n % 2 == 1].groupby(['season','team'])[STATS].mean(); even = o[o.n % 2 == 0].groupby(['season','team'])[STATS].mean()
rows = []
for k in STATS + ['pythag']:
    rel = odd[k].corr(even[k]) if k in STATS else np.nan
    rows.append(dict(stat=k, r_future_margin=X[k].corr(X.fut_margin), r_future_win=X[k].corr(X.fut_win), split_half_reliability=rel))
res = pd.DataFrame(rows).sort_values('r_future_margin', key=abs, ascending=False)
print(res.round(3).to_string(index=False)); res.to_csv(sys.argv[2], index=False)
# does shooting luck (3P% for and against) carry forward once margin is known?
import statsmodels.api as sm
Z = X[['margin','tp_pct','o_tp_pct','ft_pct','o_ft_pct','close_win','fut_margin']].dropna()
f = sm.OLS(Z.fut_margin, sm.add_constant(Z.drop(columns='fut_margin'))).fit(cov_type='HC1'); print(f.summary().tables[1])
