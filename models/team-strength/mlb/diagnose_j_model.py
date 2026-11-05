"""Diagnose J's MLB run projections (master_scores_combined.csv, 426 games) against the betting market.
Joins each projected game to SBR consensus opening/closing moneylines and asks:
  1. How biased and how spread-out are the projections?
  2. How good is the Pythagorean win prob vs the market (Brier, log loss)?
  3. Does the model add anything once the market is known (logistic regression on both)?
"""
import sys, numpy as np, pandas as pd, statsmodels.api as sm
odds_csv, master_csv = sys.argv[1], sys.argv[2]
NICK = {'Diamondbacks':'ARI','D-backs':'ARI','Braves':'ATL','Orioles':'BAL','Red Sox':'BOS','Cubs':'CHC','White Sox':'CHW','Reds':'CIN',
 'Guardians':'CLE','Indians':'CLE','Rockies':'COL','Tigers':'DET','Astros':'HOU','Royals':'KC','Angels':'LAA','Dodgers':'LAD','Marlins':'MIA',
 'Brewers':'MIL','Twins':'MIN','Mets':'NYM','Yankees':'NYY','Athletics':'OAK','As':'OAK',"A's":'OAK','Phillies':'PHI','Pirates':'PIT','Padres':'SD',
 'Giants':'SF','Mariners':'SEA','Cardinals':'STL','Rays':'TB','Rangers':'TEX','Blue Jays':'TOR','Nationals':'WAS','Nats':'WAS','DBacks':'ARI','Dbacks':'ARI','Os':'BAL','O’s':'BAL','A’s':'OAK'}
o = pd.read_csv(odds_csv)
alias = {'CWS':'CHW','KCR':'KC','SDP':'SD','SFG':'SF','TBR':'TB','WSH':'WAS','WSN':'WAS','ATH':'OAK','AZ':'ARI'}
for c in ('home','away'): o[c] = o[c].replace(alias)
m = pd.read_csv(master_csv)
m = m[(m.projected_runs_team1 < 30) & (m.projected_runs_team2 < 30)].copy()
m['away'] = m.team1.map(NICK); m['home'] = m.team2.map(NICK)
print('unmapped teams:', set(m.team1[m.away.isna()]) | set(m.team2[m.home.isna()]))
j = m.merge(o, left_on=['date','home','away'], right_on=['date','home','away'], how='left')
j = j.drop_duplicates(['sheet_name','date'])
# scores must match the market row (guards doubleheaders / bad joins)
j['score_ok'] = (j.actual_runs_team2 == j.hs) & (j.actual_runs_team1 == j.as_)
print(f'games {len(m)}, joined to odds {j.ph_close.notna().sum()}, scores agree {j.score_ok.sum()}')
j = j[j.score_ok & j.ph_close.notna() & (j.hs != j.as_)].copy()
ra, rh = j.projected_runs_team1, j.projected_runs_team2
j['p_pyth'] = rh**1.83/(rh**1.83 + ra**1.83)
j['y'] = (j.hs > j.as_).astype(int)
def brier(p): return np.mean((p - j.y)**2)
def ll(p): p = np.clip(p, 1e-6, 1-1e-6); return -np.mean(j.y*np.log(p) + (1-j.y)*np.log(1-p))
res = {}
res['n'] = len(j)
res['home_win_rate'] = j.y.mean()
res['mean_model_home_prob'] = j.p_pyth.mean(); res['mean_close_home_prob'] = j.ph_close.mean()
res['sd_model_prob'] = j.p_pyth.std(); res['sd_close_prob'] = j.ph_close.std()
res['corr_model_close'] = np.corrcoef(j.p_pyth, j.ph_close)[0,1]
for k, p in (('model', j.p_pyth), ('open', j.ph_open), ('close', j.ph_close), ('coin_home', np.full(len(j), j.y.mean()))):
    res[f'brier_{k}'] = brier(p); res[f'logloss_{k}'] = ll(p)
# total runs vs market total
j['proj_total'] = ra + rh; j['act_total'] = j.hs + j.as_
res['proj_total_mean'] = j.proj_total.mean(); res['close_total_mean'] = j.total_close.mean(); res['actual_total_mean'] = j.act_total.mean()
res['total_mae_model'] = (j.proj_total - j.act_total).abs().mean(); res['total_mae_close'] = (j.total_close - j.act_total).abs().mean()
res['corr_projtotal_closetotal'] = np.corrcoef(j.proj_total, j.total_close)[0,1]
# does model add info beyond the close? logistic on logits
lg = lambda p: np.log(p/(1-p))
X = sm.add_constant(pd.DataFrame({'close': lg(j.ph_close), 'model': lg(j.p_pyth)}))
f = sm.Logit(j.y, X).fit(disp=0)
res['logit_coef_close'] = f.params['close']; res['logit_coef_model'] = f.params['model']; res['logit_p_model'] = f.pvalues['model']
X2 = sm.add_constant(pd.DataFrame({'open': lg(j.ph_open), 'model': lg(j.p_pyth)}))
f2 = sm.Logit(j.y, X2).fit(disp=0)
res['vs_open_coef_model'] = f2.params['model']; res['vs_open_p_model'] = f2.params['model'] and f2.pvalues['model']
# does model predict the line move (open->close)?
mv = lg(j.ph_close) - lg(j.ph_open); edge_open = lg(j.p_pyth) - lg(j.ph_open)
res['corr_edge_at_open_vs_line_move'] = np.corrcoef(edge_open, mv)[0,1]
# where is the model's disagreement? home vs away lean
res['model_minus_close_home_prob_mean'] = (j.p_pyth - j.ph_close).mean()
# picks: model edge > 3pts vs close
for thr in (0.03, 0.06):
    e = j.p_pyth - j.ph_close
    pick_home = e > thr; pick_away = e < -thr
    win = np.r_[j.y[pick_home], 1 - j.y[pick_away]]
    imp = np.r_[j.ph_close[pick_home], 1 - j.ph_close[pick_away]]
    res[f'picks_{thr}_n'] = len(win); res[f'picks_{thr}_winrate'] = win.mean(); res[f'picks_{thr}_market_expected'] = imp.mean()
for k, v in res.items(): print(f'{k:36s} {v:.4f}' if isinstance(v, float) else f'{k:36s} {v}')
j.to_csv(sys.argv[3], index=False)
