"""NBA matchup model judged against the closing spread (lines 2017-18 to 2025-26).
Ratings (walk-forward): score margin, 3-point-luck-adjusted margin, offense/defense per-100 ratings, market rating from past closes.
Situational: rest days and back-to-backs. Style matchups: 3-point rate vs opponent 3-point defence, pace clash."""
import sys, numpy as np, pandas as pd, statsmodels.api as sm
sys.path.insert(0, __file__.rsplit('/', 2)[0])
from common import *
g = pd.read_parquet(sys.argv[1]); o = pd.read_parquet(sys.argv[2]); outdir = sys.argv[3]
# luck-adjusted margin: replace both teams' 3P makes with attempts x league-season 3P% (keeps 40% of own-shooting skill)
lg3 = o.groupby('season').apply(lambda d: d.tpm.sum()/d.tpa.sum())
o['lg3'] = o.season.map(lg3)
o['pts_adj'] = o.pts - 3*(o.tpm - (0.4*o.tpm + 0.6*o.tpa*o.lg3))
o['opts_adj'] = o.o_pts - 3*(o.o_tpm - o.o_tpa*o.lg3)          # opponent 3P% allowed is ~all noise
oh = o[o.is_home == 1].set_index('game_id')
g['margin_adj'] = (oh.pts_adj - oh.opts_adj).reindex(g.game_id).values
g['margin_adj'] = g.margin_adj.fillna(g.margin)
TR = lambda h: (h.season >= 2010) & (h.season <= 2017)
b1 = tune(g, lambda d, **p: margin_ratings(d, neutral='neutral', **p), grid(k=[0.03,0.04,0.05,0.07], hfa=[2,2.5,3], carry=[0.5,0.6,0.7,0.8]), mask=TR)
print('score rating', b1); g = margin_ratings(g, neutral='neutral', **b1[1])
b2 = tune(g, lambda d, **p: margin_ratings(d, margin='margin_adj', neutral='neutral', out='adj', **p), grid(k=[0.03,0.04,0.05,0.07], hfa=[2,2.5,3], carry=[0.5,0.6,0.7,0.8]),
          pred_col='adj_pred', mask=TR)
print('luck-adjusted rating', b2); g = margin_ratings(g, margin='margin_adj', neutral='neutral', out='adj', **b2[1])
gl = g[g.line.notna()].copy()
bm = tune(gl, lambda d, **p: market_ratings(d, 'line', neutral='neutral', out='mkt', **p), grid(k=[0.1,0.2,0.3,0.4], carry=[0.5,0.7,0.85]),
          target='line', pred_col='mkt_pred', mask=lambda h: h.season <= 2019)
print('market rating', bm); gl = market_ratings(gl, 'line', neutral='neutral', out='mkt', **bm[1])
g = g.merge(gl[['game_id','mkt_pred']], on='game_id', how='left')
# style ratings: 3PA rate (offense) vs opponent's 3PA-rate allowed; pace
oo = o.merge(g[['game_id','date']], on='game_id', suffixes=('','_g')).sort_values(['date','game_id']).reset_index(drop=True)
for stat in ['tpa_rate','pace','ortg']:
    r = od_ratings(oo.assign(value=oo[stat]), k=0.04, carry=0.6, out=stat)
    oo[stat+'_O'] = r[stat+'_O']; oo[stat+'_D'] = r[stat+'_Dopp']
x = oo.set_index(['game_id','team'])
def side(team_col, c): return x[c].reindex(pd.MultiIndex.from_arrays([g.game_id, g[team_col]])).values
g['exp_pace'] = side('home','pace_O') + side('home','pace_D')
g['h_3rate_exp'] = side('home','tpa_rate_O') + side('home','tpa_rate_D'); g['a_3rate_exp'] = side('away','tpa_rate_O') + side('away','tpa_rate_D')
g['three_matchup'] = g.h_3rate_exp - g.a_3rate_exp           # who gets to take more threes in THIS matchup
g['pace_x_line'] = (g.exp_pace - 98) * g.line.fillna(0) / 100  # do fast games spread out favourites more?
g['rest_diff'] = (g.home_rest - g.away_rest).clip(-3, 3)
g['b2b_diff'] = (g.home_rest == 1).astype(int) - (g.away_rest == 1).astype(int)
g['early'] = (g.groupby('season').cumcount() < 150).astype(int)
E_ = g[g.line.notna() & (g.season >= 2018)].copy()
def wf(data, feats, start=2020, base=True):
    pr = pd.Series(np.nan, index=data.index)
    for s in range(start, 2027):
        tr, te = data[data.season < s], data[data.season == s]
        X = sm.add_constant(tr[feats]).dropna().astype(float); f = sm.OLS(tr.loc[X.index, 'margin'], X).fit()
        pr[te.index] = f.predict(sm.add_constant(te[feats], has_constant='add').astype(float))
    return pr
acc = []
T = E_.season >= 2020
for name, pr in [('closing line', E_.line), ('score rating only', wf(E_, ['rt_pred'])), ('luck-adjusted rating only', wf(E_, ['adj_pred'])),
                 ('ratings + rest', wf(E_, ['rt_pred','adj_pred','rest_diff','b2b_diff'])),
                 ('ratings + rest + market rating', wf(E_, ['rt_pred','adj_pred','mkt_pred','rest_diff','b2b_diff'])),
                 ('line + everything', wf(E_, ['line','rt_pred','adj_pred','mkt_pred','rest_diff','b2b_diff','three_matchup','pace_x_line']))]:
    m = T & pr.notna()
    acc.append(dict(predictor=name, n=int(m.sum()), mae=(E_.margin[m]-pr[m]).abs().mean(), rmse=np.sqrt(((E_.margin[m]-pr[m])**2).mean())))
    if name == 'line + everything': E_['stk'] = pr
acc = pd.DataFrame(acc); print(acc.round(3).to_string(index=False))
ET = E_[T].copy(); ET['resid'] = ET.margin - ET.line
rows = []
for f in ['rt_pred','adj_pred','mkt_pred','rest_diff','b2b_diff','three_matchup','pace_x_line','early']:
    xx = ET[f] - ET.line if f in ('rt_pred','adj_pred','mkt_pred') else ET[f]
    d = pd.DataFrame({'x': xx, 'y': ET.resid}).dropna(); fit = sm.OLS(d.y, sm.add_constant(d.x)).fit(cov_type='HC1')
    rows.append(dict(feature=f + (' - line' if f in ('rt_pred','adj_pred','mkt_pred') else ''), coef=fit.params.x, t=fit.tvalues.x, p=fit.pvalues.x, n=len(d)))
inf = pd.DataFrame(rows).sort_values('p'); print(inf.round(4).to_string(index=False))
ET['edge'] = ET['stk'] - ET.line
ats = []
for thr in [0.5, 1, 2, 3]:
    m = (ET.edge.abs() >= thr).values; w, l, _ = ats_record((ET.edge[m] > 0).values, ET.margin[m].values, ET.line[m].values)
    ats.append(dict(threshold=thr, bets=w+l, win_pct=w/max(w+l,1), p_vs_52_4=binom_p(w, w+l, 0.524)))
ats = pd.DataFrame(ats); print(ats.round(4).to_string(index=False))
sig = (E_[~T].margin - E_[~T].line).std(); y = (ET.margin > 0).astype(float)
for name, pr in [('closing line', ET.line), ('stack', ET['stk'])]:
    p = win_prob_from_margin(pr, sig); print(f'win prob {name:12s} logloss {logloss(p, y):.4f} brier {brier(p, y):.4f}')
acc.to_csv(f'{outdir}/accuracy_vs_close.csv', index=False); inf.to_csv(f'{outdir}/features_vs_close_residual.csv', index=False); ats.to_csv(f'{outdir}/ats_stack.csv', index=False)
g.to_parquet(sys.argv[4])
