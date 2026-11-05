"""CFB matchup model (2006-2025) judged against OPENING and CLOSING lines.
Ratings (walk-forward): score margin (capped), market rating learned from past closing lines, CFBD pregame Elo,
offense/defense points ratings for totals. Stack trained on prior seasons only, then tested on 2014-2025 (no 2020 opens).
Key questions: does a public-data model beat the opener? Does the line move toward it (CLV)? Does it hold at the close?

2026-10-09 audit (T1): preseason prior. Instead of shrinking every rating by the same carry factor in the off-season, each
team starts from a fitted prior: end-of-season ratings for the last two seasons and returning offensive production
(cfb/prep_returning.py), regression fit walk-forward on earlier seasons only. Pass the returning file as a 5th argument;
without it the old carry rule runs.
usage: python3 cfb/game_model.py data/cfb_games.parquet cfb data/cfb_games_features.parquet data/cfb_team_games_epa.csv [data/cfb_returning.csv]"""
import sys, numpy as np, pandas as pd, statsmodels.api as sm
sys.path.insert(0, __file__.rsplit('/', 2)[0])
from common import *
g = pd.read_parquet(sys.argv[1]); outdir = sys.argv[2]
g['m_cap'] = g.margin.clip(-28, 28)
g['elo_diff'] = (g.home_pregame_elo - g.away_pregame_elo)
TR = lambda h: (h.season >= 2008) & (h.season <= 2013)
best = tune(g, lambda d, **p: margin_ratings(d, margin='m_cap', neutral='neutral', **p),
            grid(k=[0.06,0.09,0.12,0.15], hfa=[2,2.5,3], carry=[0.5,0.6,0.7,0.8]), target='margin', mask=TR)
print('capped score rating', best)
g_carry = margin_ratings(g, margin='m_cap', neutral='neutral', **best[1])
if len(sys.argv) > 5:
    ret = pd.read_csv(sys.argv[5])
    name_id = {**dict(zip(g.home, g.home_id)), **dict(zip(g.away, g.away_id))}
    fin = {}; margin_ratings(g, margin='m_cap', neutral='neutral', finals=fin, **best[1])
    rows = []
    rk = ret.set_index(['season', 'team_id'])
    for s_ in sorted(fin):
        if s_ - 1 not in fin: continue
        for t, y in fin[s_].items():
            l1 = fin[s_ - 1].get(t); l2 = fin.get(s_ - 2, {}).get(t, l1); key_ = (s_, name_id.get(t))
            if l1 is None or key_ not in rk.index: continue
            r_ = rk.loc[key_]; rows.append((s_, t, y, l1, l2, r_.ret_pass, r_.ret_skill))
    PR = pd.DataFrame(rows, columns=['season', 'team', 'y', 'l1', 'l2', 'ret_pass', 'ret_skill'])
    PR['ret_off'] = (PR.ret_pass + PR.ret_skill) / 2 - PR.groupby('season').ret_pass.transform('mean') / 2 - PR.groupby('season').ret_skill.transform('mean') / 2
    PR['l1_x_ret'] = PR.l1 * PR.ret_off
    XF = ['l1', 'l2', 'ret_off', 'l1_x_ret']
    priors, coefs = {}, []
    for s_ in sorted(PR.season.unique()):
        tr = PR[(PR.season < s_) & (PR.season >= 2008)]
        if tr.season.nunique() < 2: continue
        f_ = sm.OLS(tr.y, sm.add_constant(tr[XF])).fit(); te = PR[PR.season == s_]
        priors[s_] = dict(zip(te.team, f_.predict(sm.add_constant(te[XF], has_constant='add'))))
        coefs.append({'season': s_, **f_.params.round(3).to_dict(), 'r2': round(f_.rsquared, 3)})
    print('preseason prior coefficients (walk-forward)\n' + pd.DataFrame(coefs).to_string(index=False))
    # the update speed is re-tuned with the prior in place (early-season ratings no longer start from a shrunk carry)
    bp = tune(g, lambda d, **p: margin_ratings(d, margin='m_cap', neutral='neutral', priors=priors, **p),
              grid(k=[0.04, 0.06, 0.09, 0.12], hfa=[best[1]['hfa']], carry=[best[1]['carry']]), target='margin', mask=TR)
    print('with preseason prior', bp, '(carry-rule MSE was %.2f)' % best[0])
    g = margin_ratings(g, margin='m_cap', neutral='neutral', priors=priors, **bp[1])
    g['rt_pred_carry'] = g_carry.rt_pred.values
else:
    g = g_carry; g['rt_pred_carry'] = g.rt_pred
b2 = tune(g, lambda d, **p: margin_ratings(d, margin='margin', neutral='neutral', out='rraw', **p),
          grid(k=[0.04,0.06,0.09,0.12], hfa=[2,2.5,3], carry=[0.5,0.6,0.7,0.8]), target='margin', pred_col='rraw_pred', mask=TR)
print('uncapped score rating', b2); g = margin_ratings(g, margin='margin', neutral='neutral', out='rraw', **b2[1])
gl = g[g.line_close.notna()].copy()
bm = tune(gl, lambda d, **p: market_ratings(d, 'line_close', neutral='neutral', out='mkt', **p), grid(k=[0.2,0.3,0.4], carry=[0.7,0.85], hfa=[2.0,2.5,3.0,3.5]),
          target='line_close', pred_col='mkt_pred', mask=TR)
print('market rating', bm)
gl = market_ratings(gl, 'line_close', neutral='neutral', out='mkt', **bm[1])
g = g.merge(gl[['game_id','mkt_pred']], on='game_id', how='left')
# offense/defense points ratings for totals
rows = pd.concat([g[['game_id','season','home','away','home_points']].set_axis(['game_id','season','team','opp','value'], axis=1),
                  g[['game_id','season','away','home','away_points']].set_axis(['game_id','season','team','opp','value'], axis=1)])
rows = rows.set_index('game_id').loc[g.game_id].reset_index()
r = od_ratings(rows, k=0.08, carry=0.6, out='pts')
r['exp_pts'] = r.pts_O + r.pts_Dopp
e = r.groupby(['game_id','team']).exp_pts.first()
g['exp_home_pts'] = e.reindex(pd.MultiIndex.from_arrays([g.game_id, g.home])).values
g['exp_away_pts'] = e.reindex(pd.MultiIndex.from_arrays([g.game_id, g.away])).values
g['exp_total'] = g.exp_home_pts + g.exp_away_pts
# play-level EPA ratings (offense vs opponent defense), 2013+
ep = pd.read_csv(sys.argv[4])
ep = ep.merge(g[['game_id','date']], on='game_id').sort_values(['date','game_id']).reset_index(drop=True)
ep = ep.rename(columns={'pos_team_id':'team','def_pos_team_id':'opp'})
for stat, k in [('epa', 0.08), ('ng_epa', 0.08), ('sr', 0.08)]:
    rr = od_ratings(ep.assign(value=ep[stat]), k=k, carry=0.6, out=stat)
    ep[stat+'_exp'] = rr[stat+'_O'] + rr[stat+'_Dopp']
    x = ep.set_index(['game_id','team'])[stat+'_exp']
    h = x.reindex(pd.MultiIndex.from_arrays([g.game_id, g.home_id])).values; a = x.reindex(pd.MultiIndex.from_arrays([g.game_id, g.away_id])).values
    g[stat+'_edge'] = h - a
g['week_early'] = (g.week <= 4).astype(int)
g['fbs_both'] = ((g.home_division == 'fbs') & (g.away_division == 'fbs')).astype(int)
g.to_parquet(sys.argv[3])
# ---- evaluation set: games with an opening and closing line
E_ = g[g.line_open.notna() & g.line_close.notna() & g.rt_pred.notna() & ((g.line_open - g.line_close).abs() <= 10) & (g.season >= 2012)].copy()
for c in ['epa_edge','ng_epa_edge','sr_edge']: E_[c] = E_[c].fillna(0)
FE = ['rt_pred','rraw_pred','mkt_pred','elo_diff']
def wf(data, feats, target='margin', start=2014):
    pr = pd.Series(np.nan, index=data.index)
    for s in sorted(data.season.unique()):
        if s < start: continue
        tr, te = data[(data.season < s)], data[data.season == s]
        X = sm.add_constant(tr[feats]).dropna().astype(float); f = sm.OLS(tr.loc[X.index, target], X).fit()
        pr[te.index] = f.predict(sm.add_constant(te[feats], has_constant='add').astype(float))
    return pr
E_['elo_diff'] = E_.elo_diff.fillna(0)
E_['model'] = wf(E_, ['rt_pred','rraw_pred','mkt_pred','elo_diff'])
E_['model_ratings_only'] = wf(E_, ['rt_pred','rraw_pred','elo_diff'])
E_['carry_only'] = wf(E_, ['rt_pred_carry','rraw_pred','elo_diff'])
E_['stack_open_scores'] = wf(E_, ['line_open','rt_pred','rraw_pred','mkt_pred','elo_diff'])
E_['model_epa'] = wf(E_, ['rt_pred','rraw_pred','elo_diff','epa_edge','ng_epa_edge','sr_edge'])
E_['stack_open'] = wf(E_, ['line_open','rt_pred','rraw_pred','mkt_pred','elo_diff','epa_edge','ng_epa_edge','sr_edge'])
T = E_[E_.model.notna()].copy()
acc = []
for name, col in [('opening line','line_open'), ('closing line','line_close'), ('ratings only (score+Elo)','model_ratings_only'),
                  ('ratings + market rating','model'), ('ratings + play-level EPA','model_epa'), ('open line + score ratings','stack_open_scores'), ('open line + ratings + EPA','stack_open')]:
    acc.append(dict(predictor=name, n=len(T), mae=(T.margin - T[col]).abs().mean(), rmse=np.sqrt(((T.margin - T[col])**2).mean())))
acc = pd.DataFrame(acc); print(acc.round(3).to_string(index=False))
for lab, d_ in [('weeks 1-3', T[T.week <= 3]), ('weeks 4+', T[T.week > 3])]:
    print('  %s (n %d): MAE open %.2f | ratings only %.2f (score rating with carry rule only %.2f) | open+ratings+EPA %.2f' % (lab, len(d_),
          (d_.margin - d_.line_open).abs().mean(), (d_.margin - d_.model_ratings_only).abs().mean(), (d_.margin - d_.carry_only).abs().mean(), (d_.margin - d_.stack_open).abs().mean()))
# betting at the opener with the stack's view; grade at open and at close; CLV = close moved toward pick
T['edge'] = T.stack_open - T.line_open; T['move'] = T.line_close - T.line_open
print('\ncorr(edge vs open, line move open->close) =', round(np.corrcoef(T.edge, T.move)[0,1], 3))
e2 = T.stack_open_scores - T.line_open
print('score-only stack (no EPA): corr(edge, move) =', round(np.corrcoef(e2, T.move)[0,1], 3),
      '| mean CLV on 1+ pt edges =', round(np.where(e2[e2.abs() >= 1] > 0, T.move[e2.abs() >= 1], -T.move[e2.abs() >= 1]).mean(), 3), 'n =', int((e2.abs() >= 1).sum()))
bets = []
for thr in [1, 2, 3, 4, 5, 7]:
    m = (T.edge.abs() >= thr).values; side = (T.edge[m] > 0).values
    w, l, pu = ats_record(side, T.margin[m].values, T.line_open[m].values)
    wc, lc, _ = ats_record(side, T.margin[m].values, T.line_close[m].values)
    clv = np.where(side, T.move[m], -T.move[m])
    bets.append(dict(threshold=thr, bets=w+l, win_pct_at_open=w/max(w+l,1), p_open=binom_p(w, w+l, 0.524), win_pct_at_close=wc/max(wc+lc,1),
                     mean_clv_pts=clv.mean(), clv_t=clv.mean()/(clv.std(ddof=1)/np.sqrt(len(clv))), share_positive_clv=(clv > 0).mean()))
bets = pd.DataFrame(bets); print(bets.round(4).to_string(index=False))
by = []
for s, d in T.groupby('season'):
    m = (d.edge.abs() >= 3).values; side = (d.edge[m] > 0).values
    w, l, _ = ats_record(side, d.margin[m].values, d.line_open[m].values)
    clv = np.where(side, d.move[m], -d.move[m])
    by.append(dict(season=s, bets=w+l, win_pct_open=w/max(w+l,1), mean_clv=clv.mean()))
by = pd.DataFrame(by); print(by.round(3).to_string(index=False))
# totals at the opener
Tt = g[((g.total_open - g.total_close).abs() <= 10) & g.total_open.notna() & g.total_close.notna() & g.exp_total.notna() & (g.season >= 2014)].copy()
Tt['stack_t'] = wf(g[g.total_open.notna() & g.exp_total.notna()].copy(), ['total_open','exp_total'], target='total').reindex(Tt.index)
Tt = Tt[Tt.stack_t.notna()]
Tt['edge'] = Tt.stack_t - Tt.total_open; Tt['move'] = Tt.total_close - Tt.total_open
print('\nTOTALS: MAE open %.2f close %.2f stack %.2f; corr(edge, move) %.3f' % ((Tt.total-Tt.total_open).abs().mean(), (Tt.total-Tt.total_close).abs().mean(), (Tt.total-Tt.stack_t).abs().mean(), np.corrcoef(Tt.edge, Tt.move)[0,1]))
tb = []
for thr in [1, 2, 3, 4, 6]:
    m = Tt.edge.abs() >= thr; over = Tt.edge[m] > 0
    res_o = Tt.total[m] - Tt.total_open[m]; w = ((over & (res_o > 0)) | (~over & (res_o < 0))).sum(); l = ((over & (res_o < 0)) | (~over & (res_o > 0))).sum()
    res_c = Tt.total[m] - Tt.total_close[m]; wc = ((over & (res_c > 0)) | (~over & (res_c < 0))).sum(); lc = ((over & (res_c < 0)) | (~over & (res_c > 0))).sum()
    clv = np.where(over, Tt.move[m], -Tt.move[m])
    tb.append(dict(threshold=thr, bets=int(w+l), win_pct_open=w/max(w+l,1), p_open=binom_p(int(w), int(w+l), 0.524), win_pct_close=wc/max(wc+lc,1), mean_clv=clv.mean()))
tb = pd.DataFrame(tb); print(tb.round(4).to_string(index=False))
acc.to_csv(f'{outdir}/accuracy.csv', index=False); bets.to_csv(f'{outdir}/spread_bets_at_open.csv', index=False)
by.to_csv(f'{outdir}/spread_bets_by_season.csv', index=False); tb.to_csv(f'{outdir}/total_bets_at_open.csv', index=False)
