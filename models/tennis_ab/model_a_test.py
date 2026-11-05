"""Model A yardstick (J 2026-10-31): grade NUMBERS against closing lines. Total games vs bet365 closing total, favourite's games
won vs the line-implied number ((total + |handicap|)/2), set score. Win % background only.
Walk-forward: win model (Elo, surface Elo, games-won rating) fit on the 4 prior years per sex+level; serve level mu fit on
prior years (group base + player scoring style). Test 2021-25.  Usage: model_a_test.py FEAT_PARQUET OUT_PRED_PARQUET"""
import sys, json, numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression
import lib
F = pd.read_parquet(sys.argv[1]); OUTP = sys.argv[2]
F['yr'] = F.t.dt.year
lgt = lambda p: np.log(np.clip(p, 1e-4, 1 - 1e-4) / (1 - np.clip(p, 1e-4, 1 - 1e-4)))
F['f_elo'] = lgt(1 / (1 + 10 ** ((F.elo_b - F.elo_a) / 400))); F['f_elos'] = lgt(1 / (1 + 10 ** ((F.elos_b - F.elos_a) / 400)))
F['f_g'] = F.g_a - F.g_b
new = ((np.minimum(F.n_a, F.n_b) < 10)).astype(float)
F['f_gnew'] = F.f_g * new
F['f_lay'] = (np.minimum(F.lay_a, 120) - np.minimum(F.lay_b, 120)) / 100
X = ['f_elo', 'f_elos', 'f_g', 'f_gnew', 'f_lay']
def fair(oa, ob):
    ia, ib = 1 / oa, 1 / ob; return ia / (ia + ib)
F['close'] = np.where((F.bet365_odds_a > 1) & (F.bet365_odds_b > 1), fair(F.bet365_odds_a, F.bet365_odds_b), np.nan)
F['open'] = np.where((F.bet365_odds_a_opening > 1) & (F.bet365_odds_b_opening > 1), fair(F.bet365_odds_a_opening, F.bet365_odds_b_opening), np.nan)
# style deviations (shrunk) vs sex+level means of established players
for p in 'ab':
    F[f'w_{p}'] = F[f'sets_{p}'] / (F[f'sets_{p}'] + 25)
F['p'] = np.nan
TEST = range(2021, 2026)
for (sx, lv), g in F.groupby(['sex', 'level']):
    for yr in TEST:
        tr = g[(g.yr < yr) & (g.yr >= yr - 4) & (np.minimum(g.n_a, g.n_b) >= 3)]; te = g[g.yr == yr]
        if len(tr) < 500 or not len(te): continue
        Xt = pd.concat([tr[X], -tr[X]]); yt = np.r_[tr.y, 1 - tr.y]
        m = LogisticRegression(C=1, fit_intercept=False, max_iter=500).fit(Xt, yt)
        F.loc[te.index, 'p'] = m.predict_proba(te[X])[:, 1]
T = F[F.yr.isin(TEST) & F.p.notna()].copy()
print('win picks (background):', T.groupby(['sex', 'level']).apply(lambda z: f"model {np.mean((z.p > .5) == (z.y == 1)):.3f} close {np.mean((z.close[z.close.notna()] > .5) == (z.y[z.close.notna()] == 1)):.3f} (same rows model {np.mean((z.p[z.close.notna()] > .5) == (z.y[z.close.notna()] == 1)):.3f}) n {len(z)}").to_string())

# ---- serve level mu: group base + style. Fit on 2017-20 (before every test year) ----
def proj(rows, mu, pcol='p'):
    out = []
    for r, m in zip(rows.itertuples(index=False), mu):
        o = lib.project(getattr(r, pcol), m, r.bo, r.sex); out.append((o['tot_med'], o['tot_mean'], o['ga_mean'], o['mar_mean'], o['ss'][0] + o['ss'][len(o['ss']) // 2]))
    return np.array(out)
C = F[(F.yr.between(2017, 2020)) & ~F.ret & F.close.notna()].copy()
C = C.sample(min(len(C), 20000), random_state=1)
# group means of style for deviations (established players), stored for live use
STY = {}
for (sx, lv), g in F[(F.yr.between(2017, 2020))].groupby(['sex', 'level']):
    e = pd.concat([g[['gps_a', 'tb_a', 'sets_a']].set_axis(['gps', 'tb', 'sets'], axis=1), g[['gps_b', 'tb_b', 'sets_b']].set_axis(['gps', 'tb', 'sets'], axis=1)])
    e = e[e.sets >= 20]; STY[f'{sx}|{lv}'] = [float(e.gps.mean()), float(e.tb.mean())]
def sty(rows):
    k = rows.sex + '|' + rows.level
    g0 = np.array([STY.get(x, [9.3, .12])[0] for x in k]); t0 = np.array([STY.get(x, [9.3, .12])[1] for x in k])
    dg = sum(((rows[f'gps_{p}'].fillna(pd.Series(g0, index=rows.index)) - g0) * rows[f'w_{p}']).values for p in 'ab')
    dt = sum(((rows[f'tb_{p}'].fillna(pd.Series(t0, index=rows.index)) - t0) * rows[f'w_{p}']).values for p in 'ab')
    return dg, dt
# serve level from published tour serve-point win rates (fitting mu to mean totals is ill-posed: total games barely move with mu)
SP = {('m', 'Tour'): .64, ('m', 'Challenger'): .62, ('m', 'ITF'): .60, ('w', 'Tour'): .56, ('w', 'ITF'): .545}
SURF = {'clay': -.015, 'grass': +.025, 'hard': 0., 'carpet': +.01}
BASE = {f'{sx}|{lv}|{sf}|{bo}': round(float(lib.lg(v + SURF[sf])), 4) for (sx, lv), v in SP.items() for sf in SURF for bo in (3, 5)}
print('base mu', BASE)
def base(rows):
    out = []
    for r in rows.itertuples(index=False):
        k = f'{r.sex}|{r.level}|{r.surf}|{r.bo}'
        if k not in BASE: k = next((x for x in BASE if x.startswith(f'{r.sex}|{r.level}|') and x.endswith(f'|{r.bo}')), None) or next(x for x in BASE if x.startswith(f'{r.sex}|'))
        out.append(BASE[k])
    return np.array(out)
s = C.sample(4000, random_state=3); b0 = base(s); dg, dt = sty(s)
best = None
for c1 in (-.3, -.15, 0, .15, .3, .6, 1.0):
    for c2 in (-3, -1.5, 0, 1.5, 3, 6, 10):
        pr = proj(s, b0 + c1 * dg + c2 * dt, 'close'); e = np.mean(np.abs(pr[:, 0] - s.total_games))
        if best is None or e < best[0]: best = (e, c1, c2)
print('style fit (MAE, c_gps, c_tb):', best)
C1, C2 = best[1], best[2]
def mu_of(rows): dg, dt = sty(rows); return base(rows) + C1 * dg + C2 * dt
json.dump(dict(base=BASE, style_means=STY, c_gps=C1, c_tb=C2, sd=lib.SD, fit='2017-20 bet365-close-fed, Kaggle alimoh89'), open(f'{lib.HERE}/params.json', 'w'), indent=1)

# ---- grade on 2021-25 ----
G = T[~T.ret & T.bet365_total_games_line.notna() & T.close.notna()].copy()
mu = mu_of(G); P = proj(G, mu, 'p'); G['a_med'], G['a_mean'], G['a_ga'], G['a_mar'], G['a_straight'] = P.T
Pc = proj(G, mu, 'close'); G['c_med'], G['c_mean'], G['c_ga'] = Pc[:, 0], Pc[:, 1], Pc[:, 2]
Pn = proj(G, base(G), 'p'); G['n_med'] = Pn[:, 0]    # no style term
G['mu'] = mu
# favourite games: fav = lower close ML; line-implied fav games = (total + |h|)/2
fa = G.bet365_odds_a <= G.bet365_odds_b
h = G.bet365_handicap.abs()
G['fav_line'] = (G.bet365_total_games_line + h) / 2
G['fav_act'] = np.where(fa, G.total_games_a, G.total_games_b)
# A's expected games is a_ga (A = player_a); fav = A or B
G['fav_ours'] = np.where(fa, G.a_ga, G.a_mean - G.a_ga)
G['fav_eng_close'] = np.where(fa, G.c_ga, G.c_mean - G.c_ga)
G.to_parquet(OUTP)
T.to_parquet(OUTP.replace('.parquet', '_all.parquet'))
res = {}
for key, z in list(G.groupby(['sex', 'level'])) + [(('all', 'all'), G)]:
    if len(z) < 200: continue
    act = z.total_games
    r = dict(n=len(z),
             tot_mae_ours=round(np.mean(np.abs(z.a_med - act)), 3), tot_mae_nostyle=round(np.mean(np.abs(z.n_med - act)), 3),
             tot_mae_line=round(np.mean(np.abs(z.bet365_total_games_line - act)), 3), tot_mae_engine_on_close_ml=round(np.mean(np.abs(z.c_med - act)), 3),
             tot_ape_ours=round(np.mean(np.abs(z.a_med - act) / act) * 100, 1), tot_ape_line=round(np.mean(np.abs(z.bet365_total_games_line - act) / act) * 100, 1),
             tot_bias_ours=round(np.mean(z.a_mean - act), 2), r_ours_line=round(np.corrcoef(z.a_med, z.bet365_total_games_line)[0, 1], 3))
    zz = z[z.bet365_handicap.notna()]
    if len(zz) > 200:
        r.update(fav_games_mae_ours=round(np.mean(np.abs(zz.fav_ours - zz.fav_act)), 3), fav_games_mae_line=round(np.mean(np.abs(zz.fav_line - zz.fav_act)), 3),
                 fav_games_mae_engine_close=round(np.mean(np.abs(zz.fav_eng_close - zz.fav_act)), 3))
    stra = (np.minimum(z.total_sets_a, z.total_sets_b) == 0)
    r['straight_sets_pred'] = round(z.a_straight.mean(), 3); r['straight_sets_real'] = round(stra.mean(), 3)
    res['|'.join(key)] = r
for k, v in res.items(): print(k, v)
json.dump(res, open(f'{lib.HERE}/model_a_test.json', 'w'), indent=1)
