"""Projected score for each match: set-score probabilities (2-0, 2-1, 1-2, 0-2) and total games / games margin.
Fit on earlier months, tested on later months. Usage: score_proj.py PRED_PARQUET [CUT_DAY]"""
import sys
import numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression, LinearRegression
R = pd.read_csv('../../tex/results_all.csv').drop_duplicates('mid'); R['fid'] = 'tex' + R.mid.astype(str)
p = pd.read_parquet(sys.argv[1]); p = p[p.p.notna()]
cut = sys.argv[2] if len(sys.argv) > 2 else '2026-07-01'
x = p.merge(R[['fid', 'odds_a', 'games_a', 'games_b', 'sets_a', 'sets_b', 'nsets']], on='fid')
S = pd.read_parquet('../data/style26.parquet'); x = x.merge(S, on='fid', how='left')
x = x[x.nsets.isin([2, 3]) & (x.sets_a.fillna(0) + x.sets_b.fillna(0) == x.nsets)].copy()
same = np.isclose(x.oa, x.odds_a)
x['ga'] = np.where(same, x.games_a, x.games_b); x['gb'] = np.where(same, x.games_b, x.games_a)
x['sa'] = np.where(same, x.sets_a, x.sets_b); x['sb'] = np.where(same, x.sets_b, x.sets_a)
x['act'] = np.select([(x.sa == 2) & (x.sb == 0), (x.sa == 2) & (x.sb == 1), (x.sa == 1) & (x.sb == 2)], [0, 1, 2], 3)
x['tot'] = x.ga + x.gb; x['mar'] = x.ga - x.gb
lp = np.log(x.p.clip(.01, .99) / (1 - x.p.clip(.01, .99)))
x['day'] = x.start.astype(str).str[:10]
isW = (x.tour == 'wta').astype(float) if 'tour' in x and x.tour.isin(['wta']).any() else 0.
lvl = pd.get_dummies(x.cat).astype(float)
sty = np.c_[x.gps_w + x.gps_l, x.tb_w + x.tb_l, np.abs(lp) * (x.tb_w + x.tb_l)] if 'gps_w' in x else np.zeros((len(x), 0))
F = np.c_[lp, np.abs(lp), lp ** 2, x.gelo.fillna(0), lvl.values, sty]
tr = (x.day < cut).values; te = ~tr
m = LogisticRegression(C=10, max_iter=2000).fit(F[tr], x.act[tr])
pr = m.predict_proba(F[te])
act = x.act.values[te]
ll = -np.mean(np.log(np.clip(pr[np.arange(len(act)), act], 1e-6, 1)))
base = np.bincount(x.act[tr], minlength=4) / tr.sum()
llb = -np.mean(np.log(base[act]))
# win prob from the score model must match the match model
print(f'test {te.sum()} matches from {cut}: set-score most-likely right {np.mean(pr.argmax(1) == act):.3f}; log loss {ll:.3f} vs base rates {llb:.3f}')
print('   predicted 3-set rate', round((pr[:, 1] + pr[:, 2]).mean(), 3), 'actual', round(np.isin(act, [1, 2]).mean(), 3))
G = np.c_[F, pr[:, 1] + pr[:, 2]] if False else F
for tgt in ['tot', 'mar']:
    lr = LinearRegression().fit(F[tr], x[tgt][tr]); e = lr.predict(F[te]) - x[tgt].values[te]
    b = x[tgt][tr].mean() if tgt == 'tot' else 0
    print(f'   {tgt}: MAE {np.abs(e).mean():.2f} games (naive {np.abs(x[tgt].values[te] - b).mean():.2f}); within 3 games {np.mean(np.abs(e) <= 3):.2f}')
# market comparison: same score model fed the closing price instead of our model
if 'close' in x:
    xc = x[x.close.notna()]
    lc = np.log(xc.close.clip(.01, .99) / (1 - xc.close.clip(.01, .99)))
    Fc = np.c_[lc, np.abs(lc), lc ** 2, xc.gelo.fillna(0), pd.get_dummies(xc.cat).astype(float).values, np.c_[xc.gps_w + xc.gps_l, xc.tb_w + xc.tb_l, np.abs(lc) * (xc.tb_w + xc.tb_l)]]
    trc = (xc.day < cut).values; tec = ~trc
    mc = LogisticRegression(C=10, max_iter=2000).fit(Fc[trc], xc.act[trc]); pc = mc.predict_proba(Fc[tec])
    print(f'   yardstick (closing line through same score model): set-score right {np.mean(pc.argmax(1) == xc.act.values[tec]):.3f}', end='')
    lr = LinearRegression().fit(Fc[trc], xc.tot[trc]); print(f', total games MAE {np.abs(lr.predict(Fc[tec]) - xc.tot.values[tec]).mean():.2f}')
    ok = x.close.notna().values[te]
    print(f'   win pick right: model {np.mean((x.p.values[te] > .5) == (x.act.values[te] < 2)):.3f}, close {np.mean((xc.close.values[tec] > .5) == (xc.act.values[tec] < 2)):.3f}')
for c in x.cat.unique():
    s = (x.cat.values[te] == c)
    if s.sum() > 300: print(f'   {c}: n {s.sum()} set-score right {np.mean(pr[s].argmax(1) == act[s]):.3f}, win pick {np.mean((x.p.values[te][s] > .5) == (act[s] < 2)):.3f}')
