"""MLB matchup model (2015-2025 features, odds 2021-03 to 2025-08) judged against opening and closing moneylines.
Walk-forward logistic regression on point-in-time features; tests which stats add information beyond the market."""
import sys, numpy as np, pandas as pd, statsmodels.api as sm
sys.path.insert(0, __file__.rsplit('/', 2)[0])
from common import logloss, brier, binom_p
G = pd.read_parquet(sys.argv[1]); O = pd.read_csv(sys.argv[2]); outdir = sys.argv[3]
R2S = {'ANA':'LAA','CHA':'CHW','CHN':'CHC','KCA':'KC','LAN':'LAD','NYA':'NYY','NYN':'NYM','SDN':'SD','SFN':'SF','SLN':'STL','TBA':'TB','ATH':'OAK'}
O['home'] = O.home.replace({'AZ':'ARI','ATH':'OAK'}); O['away'] = O.away.replace({'AZ':'ARI','ATH':'OAK'})
G['h'] = G.home.replace(R2S); G['a'] = G.away.replace(R2S)
O['date'] = pd.to_datetime(O.date)
O = O.drop_duplicates(['date','home','away','hs','as_'])
G = G.merge(O[['date','home','away','hs','as_','ph_open','ph_close','total_open','total_close']].rename(columns={'home':'h','away':'a'}),
            left_on=['date','h','a','hscore','vscore'], right_on=['date','h','a','hs','as_'], how='left')
G = G[G.season >= 2015].copy()
G = G[G.margin != 0]
G['y'] = (G.margin > 0).astype(int)
lgt = lambda p: np.log(p/(1-p))
# features: positive = good for home
G['sp_diff'] = (G.away_sp_r_woba - G.home_sp_r_woba)*100
G['sp_kbb_diff'] = (G.home_sp_r_kbb - G.away_sp_r_kbb)*100
G['lineup_diff'] = (G.home_lineup_woba - G.away_lineup_woba)*100
G['bp_diff'] = (G.away_bp_woba - G.home_bp_woba)*100
G['platoon_diff'] = G.home_platoon_share - G.away_platoon_share
G['bp_load_diff'] = (G.away_bp_load3 - G.home_bp_load3)/10
G['sp_raw_diff'] = (G.away_sp_raw_woba - G.home_sp_raw_woba).fillna(0)*100
G['lineup_raw_diff'] = (G.home_lineup_raw_woba - G.away_lineup_raw_woba).fillna(0)*100
G['sp_exp_diff'] = np.log1p(G.home_sp_r_n) - np.log1p(G.away_sp_r_n)
SETS = {
 'J-style: raw season-to-date SP + lineup, no home field': (['sp_raw_diff','lineup_raw_diff'], False),
 'shrunk SP + lineup (adds home field)': (['sp_diff','lineup_diff'], True),
 '+ SP K-BB%, bullpen, platoon': (['sp_diff','sp_kbb_diff','lineup_diff','bp_diff','platoon_diff'], True),
 '+ team run rating, bullpen fatigue': (['sp_diff','sp_kbb_diff','lineup_diff','bp_diff','platoon_diff','rt_pred','bp_load_diff'], True),
}
def wf(feats, const=True, data=G, start=2021, extra=None):
    pr = pd.Series(np.nan, index=data.index)
    for s in range(start, 2026):
        tr, te = data[(data.season < s) & (data.season != 2020)], data[data.season == s]
        if extra is not None: tr = tr[tr[extra].notna()]
        X = tr[feats].astype(float); X = sm.add_constant(X) if const else X
        f = sm.Logit(tr.y, X).fit(disp=0)
        Xt = te[feats].astype(float); Xt = sm.add_constant(Xt, has_constant='add') if const else Xt
        pr[te.index] = f.predict(Xt)
    return pr
T = G[(G.season >= 2021) & G.ph_close.notna()].index
rows = [dict(model='coin flip at home rate', logloss=logloss(np.full(len(T), G.loc[T].y.mean()), G.loc[T].y), brier=brier(np.full(len(T), G.loc[T].y.mean()), G.loc[T].y)),
        dict(model='opening line', logloss=logloss(G.loc[T].ph_open, G.loc[T].y), brier=brier(G.loc[T].ph_open, G.loc[T].y)),
        dict(model='closing line', logloss=logloss(G.loc[T].ph_close, G.loc[T].y), brier=brier(G.loc[T].ph_close, G.loc[T].y))]
for name, (feats, c) in SETS.items():
    p = wf(feats, const=c); G['p_'+str(len(rows))] = p
    rows.append(dict(model=name, logloss=logloss(p[T], G.loc[T].y), brier=brier(p[T], G.loc[T].y)))
best_feats = SETS['+ team run rating, bullpen fatigue'][0]
G['p_model'] = wf(best_feats)
G['lo_open'] = lgt(G.ph_open); G['lo_close'] = lgt(G.ph_close)
G['p_stack_open'] = wf(['lo_open'] + best_feats, data=G, start=2022, extra='lo_open')
T2 = G[(G.season >= 2022) & G.ph_close.notna() & G.p_stack_open.notna()].index
rows.append(dict(model='opening line + model (2022-25)', logloss=logloss(G.loc[T2].p_stack_open, G.loc[T2].y), brier=brier(G.loc[T2].p_stack_open, G.loc[T2].y)))
rows.append(dict(model='opening line alone (same 2022-25 games)', logloss=logloss(G.loc[T2].ph_open, G.loc[T2].y), brier=brier(G.loc[T2].ph_open, G.loc[T2].y)))
rows.append(dict(model='closing line (same 2022-25 games)', logloss=logloss(G.loc[T2].ph_close, G.loc[T2].y), brier=brier(G.loc[T2].ph_close, G.loc[T2].y)))
acc = pd.DataFrame(rows); acc['n'] = [len(T)]*(len(rows)-3) + [len(T2)]*3
print(acc.round(4).to_string(index=False))
# information beyond the close: logit(y) ~ logit(close) + feature, 2021-2025
D_ = G.loc[T].copy(); D_['model_minus_close'] = lgt(D_.p_model) - D_.lo_close
inf = []
for f in best_feats + ['model_minus_close','sp_exp_diff','pf','temp','wind_out','ump_factor','day_game']:
    d = D_[['y','lo_close',f]].dropna()
    fit = sm.Logit(d.y, sm.add_constant(d[['lo_close', f]].astype(float))).fit(disp=0)
    inf.append(dict(feature=f, coef=fit.params[f], z=fit.tvalues[f], p=fit.pvalues[f], n=len(d)))
inf = pd.DataFrame(inf).sort_values('p'); print(inf.round(4).to_string(index=False))
# betting at open with the stack; CLV in implied-probability points
D2 = G.loc[T2].copy(); D2['edge'] = D2.p_stack_open - D2.ph_open; D2['move'] = D2.ph_close - D2.ph_open
print('corr(edge at open, move open->close) =', round(np.corrcoef(D2.edge, D2.move)[0,1], 3))
bets = []
for thr in [0.01, 0.02, 0.03, 0.05]:
    m = D2.edge.abs() >= thr; home = D2.edge[m] > 0
    win = np.where(home, D2.y[m], 1 - D2.y[m]); p_open = np.where(home, D2.ph_open[m], 1 - D2.ph_open[m]); p_close = np.where(home, D2.ph_close[m], 1 - D2.ph_close[m])
    # flat-stake profit at no-vig open price minus ~2.2% vig
    dec = 1/p_open*(1 - 0.022); profit = np.where(win == 1, dec - 1, -1)
    bets.append(dict(threshold=thr, bets=int(m.sum()), win_pct=win.mean(), expected_at_open=p_open.mean(), expected_at_close=p_close.mean(),
                     mean_clv_prob_pts=(p_close - p_open).mean()*100, roi_after_vig=profit.mean()))
bets = pd.DataFrame(bets); print(bets.round(4).to_string(index=False))
# totals: model runs vs market totals
G['exp_total_feat'] = G.pf
acc.to_csv(f'{outdir}/accuracy_vs_market.csv', index=False); inf.to_csv(f'{outdir}/features_vs_close_residual.csv', index=False); bets.to_csv(f'{outdir}/bets_at_open.csv', index=False)
G.to_parquet(sys.argv[4])
