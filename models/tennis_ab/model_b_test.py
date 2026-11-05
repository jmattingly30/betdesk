"""Model B tests, out of sample 2021-25 (weights fit on earlier years only).
1) Moneyline at the bet365 OPENER: blend logit(open no-vig) + logit(Model A win %), bet edge >= x at the opening price,
   ROI (retirements settle as the file's winner) and CLV vs bet365 close. Confidence features tested for tiers.
2) Total games at the bet365 CLOSE: Model A sim P(over line) blended with the book's no-vig over, ROI by edge.
3) Games handicap at the CLOSE (favourite -|h|), same.
Usage: model_b_test.py PRED_ALL_PARQUET PRED_TOT_PARQUET"""
import sys, json, numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression
import lib
lgt = lambda p: np.log(np.clip(p, 1e-4, 1 - 1e-4) / (1 - np.clip(p, 1e-4, 1 - 1e-4)))
T = pd.read_parquet(sys.argv[1]); G = pd.read_parquet(sys.argv[2])
RES = {}
def roi_table(df, edge, price, win, clv=None, cuts=(.03, .06, .10)):
    out = {}
    for c in cuts:
        s = edge >= c
        if s.sum() < 30: continue
        pnl = np.where(win[s], price[s] - 1, -1.)
        r = dict(n=int(s.sum()), roi=round(pnl.mean() * 100, 1), se=round(pnl.std() / np.sqrt(s.sum()) * 100, 1))
        if clv is not None: r['clv'] = round(np.nanmean(clv[s]) * 100, 1)
        out[f'{int(c * 100)}%+'] = r
    return out

# ---------- 1) moneyline at the opener ----------
M = T[T.open.notna() & T.close.notna() & (T.bet365_odds_a_opening > 1)].copy()
M['lo'] = lgt(M.open); M['lm'] = lgt(M.p); M['yr'] = M.t.dt.year
M['pb'] = np.nan
for (sx, lv), g in M.groupby(['sex', 'level']):
    for yr in range(2021, 2026):
        tr = g[(g.yr < yr) & (g.yr >= yr - 3)]
        if len(tr) < 300:   # p only exists from 2021: fit on the earlier test years only (expanding)
            continue
        te = g[g.yr == yr]
        m = LogisticRegression(C=10, fit_intercept=False).fit(np.r_[tr[['lo', 'lm']].values, -tr[['lo', 'lm']].values], np.r_[tr.y, 1 - tr.y])
        M.loc[te.index, 'pb'] = m.predict_proba(te[['lo', 'lm']])[:, 1]
        RES.setdefault('ml_weights', {})[f'{sx}|{lv}|{yr}'] = [round(float(x), 3) for x in m.coef_[0]]
B = M[M.pb.notna()].copy()
rows = []
for side in 'ab':
    q = B.pb if side == 'a' else 1 - B.pb
    pr = B.bet365_odds_a_opening if side == 'a' else B.bet365_odds_b_opening
    cl = B.close if side == 'a' else 1 - B.close
    w = (B.y == 1) if side == 'a' else (B.y == 0)
    nmin = np.minimum(B.n_a, B.n_b)
    rows.append(pd.DataFrame(dict(sex=B.sex, level=B.level, yr=B.yr, edge=q * pr - 1, price=pr, win=w, clv=cl * pr - 1, nmin=nmin,
                                  gap=np.abs(B.lm - B.lo), agree=np.sign(B.lm - B.lo) == (1 if side == 'a' else -1),
                                  mv=np.where(side == 'a', B.bet365_odds_a_opening - B.bet365_odds_a, B.bet365_odds_b_opening - B.bet365_odds_b))))
R = pd.concat(rows); R = R[R.price.between(1.2, 5)]
ml = {}
for (sx, lv), z in R.groupby(['sex', 'level']):
    ml[f'{sx}|{lv}'] = roi_table(z, z.edge.values, z.price.values, z.win.values, z.clv.values)
    # confidence splits at 6%+
    s = z[z.edge >= .06]
    if len(s) > 100:
        for nm, cond in (('both 30+ matches', s.nmin >= 30), ('a player under 30', s.nmin < 30), ('price 1.6-3.0', s.price.between(1.6, 3)),
                         ('price outside 1.6-3.0', ~s.price.between(1.6, 3)), ('2024-25 only', s.yr >= 2024)):
            ss = s[cond]
            if len(ss) >= 30:
                pnl = np.where(ss.win, ss.price - 1, -1.)
                ml[f'{sx}|{lv}']['6%+ ' + nm] = dict(n=len(ss), roi=round(pnl.mean() * 100, 1), se=round(pnl.std() / np.sqrt(len(ss)) * 100, 1), clv=round(ss.clv.mean() * 100, 1))
RES['moneyline_at_open'] = ml
# by year for the levels the live desk flags
RES['moneyline_by_year_6pct'] = {f'{k[0]}|{k[1]}|{k[2]}': dict(n=len(z), roi=round(np.where(z.win, z.price - 1, -1.).mean() * 100, 1), clv=round(z.clv.mean() * 100, 1))
                                 for k, z in R[R.edge >= .06].groupby(['sex', 'level', 'yr'])}

# ---------- 2) totals and 3) handicap at the close ----------
G = G[G.bet365_odds_over.gt(1) & G.bet365_odds_under.gt(1)].copy()
po, pu, hc, hd = [], [], [], []
fa = (G.bet365_odds_a <= G.bet365_odds_b).values
for r, f in zip(G.itertuples(index=False), fa):
    o = lib.project(r.p, r.mu, r.bo, r.sex)
    a, b = lib.p_over(o, r.bet365_total_games_line); po.append(a); pu.append(b)
    h = abs(r.bet365_handicap) if r.bet365_handicap == r.bet365_handicap else np.nan
    if h == h:
        # fav covers -h: fav = A -> margin A - h > 0 ; fav = B -> A margin + h < 0
        c, nc = lib.p_cover(o, -h) if f else lib.p_cover(o, h)[::-1]
        hc.append(c); hd.append(nc)
    else: hc.append(np.nan); hd.append(np.nan)
G['po'] = np.array(po) / (np.array(po) + np.array(pu)); G['pc'] = np.array(hc) / (np.array(hc) + np.array(hd))
G['fo'] = (1 / G.bet365_odds_over) / (1 / G.bet365_odds_over + 1 / G.bet365_odds_under)
G['yr'] = G.t.dt.year
over = G.total_games > G.bet365_total_games_line; push = G.total_games == G.bet365_total_games_line
G['fav_m'] = np.where(fa, G.total_games_a - G.total_games_b, G.total_games_b - G.total_games_a)
cov = G.fav_m - G.bet365_handicap.abs() > 0
G['fh'] = (1 / G.bet365_odds_handicap_a) / (1 / G.bet365_odds_handicap_a + 1 / G.bet365_odds_handicap_b)
def blend_test(G, pm, pmk, won, po_, pu_, name):
    G = G.copy(); G['lm'] = lgt(pm); G['lk'] = lgt(pmk); G['q'] = np.nan
    for yr in range(2022, 2026):
        tr = G[(G.yr < yr)]; te = G[G.yr == yr]
        ok = tr.lm.notna() & tr.lk.notna() & tr.won.notna()
        m = LogisticRegression(C=10).fit(tr.loc[ok, ['lk', 'lm']], tr.won[ok].astype(int))
        okt = te.lm.notna() & te.lk.notna()
        G.loc[te.index[okt], 'q'] = m.predict_proba(te.loc[okt, ['lk', 'lm']])[:, 1]
        RES.setdefault(name + '_weights', {})[str(yr)] = [round(float(x), 3) for x in m.coef_[0]]
    Z = G[G.q.notna()]
    out = {}
    for lv, z in list(Z.groupby(['sex', 'level'])) + [(('all', 'all'), Z)]:
        e1 = z.q * z[po_] - 1; e2 = (1 - z.q) * z[pu_] - 1
        side1 = e1 >= e2; edge = np.where(side1, e1, e2); price = np.where(side1, z[po_], z[pu_])
        win = np.where(side1, z.won == 1, z.won == 0)
        out['|'.join(lv)] = roi_table(z, edge, price, win, None, (.02, .04, .06, .10))
        # model alone LL vs market LL (accuracy of the probability)
        ok = z.won.notna()
        llf = lambda p: -np.mean(np.where(z.won[ok] == 1, np.log(np.clip(p[ok], 1e-4, 1)), np.log(np.clip(1 - p[ok], 1e-4, 1))))
        out['|'.join(lv)]['logloss'] = dict(model=round(llf(z.pmod), 4), book=round(llf(z.pmk), 4), blend=round(llf(z.q), 4), coin=0.6931)
    RES[name] = out
G['won'] = np.where(push, np.nan, over.astype(float)); G['pmod'] = G.po; G['pmk'] = G.fo
blend_test(G, G.po, G.fo, None, 'bet365_odds_over', 'bet365_odds_under', 'totals_at_close')
H = G[G.pc.notna() & G.fh.notna() & G.bet365_odds_handicap_a.gt(1)].copy()
H['won'] = np.where(H.fav_m - H.bet365_handicap.abs() == 0, np.nan, cov[H.index].astype(float)); H['pmod'] = H.pc; H['pmk'] = H.fh
blend_test(H, H.pc, H.fh, None, 'bet365_odds_handicap_a', 'bet365_odds_handicap_b', 'handicap_at_close')
# opener totals prices at the closing line (line at open not in the file: shown for reference only)
json.dump(RES, open(f'{lib.HERE}/model_b_test.json', 'w'), indent=1)
for k, v in RES.items():
    if 'weights' in k: continue
    print('==', k)
    for kk, vv in v.items(): print('  ', kk, vv)
