"""NBA score / win / cover model: the college-basketball tempo x efficiency engine (../cbb/ratings.py) plus a lineup layer.

  points per 100 for team T vs U = mu + O_T - D_U + hca + k * (lineup_T,U - average)
  lineup_T,U = weights . [box-score line of T's expected lineup, per 100 possessions] + weights . [U's expected lineup]
The weights are fit (walk-forward, earlier seasons only) by regressing each team-game's points per 100 on both lineups.
The team ratings O, D are fit on what the lineup does not explain, so O/D are "team beyond its players" (system, coaching,
fit), and a missing star moves the prediction through the lineup term only.
Possessions come from the same engine's tempo ratings.

2026-10-09 audit changes: RAPM column in both lineups (N2, rapm.py), 'known at the open' lineup and the opener test split
into continuing vs new absences (N1), market-anchored win probability with sigma by total (N3), playoff home-edge term (N4),
opener blend weight fit walk-forward instead of in-sample.

usage: python3 game_model.py      (run from models/nba-team after prep.py, rapm.py and lineup.py)"""
import sys, json, numpy as np, pandas as pd, statsmodels.api as sm
from scipy.stats import norm
sys.path.insert(0, '../cbb'); import ratings
STATS = ['pts', 'fga', 'fta', 'fg3a', 'fg3m', 'ast', 'tov', 'oreb', 'dreb', 'stl', 'blk', 'pf']   # same as lineup.py
log = open('run_log.txt', 'w')
def say(*a):
    s = ' '.join(str(x) for x in a); print(s, flush=True); log.write(s + '\n')
def rmse(a, b): m = a.notna() & b.notna(); return float(np.sqrt(np.mean((a[m] - b[m])**2)))
RC = ['r_' + s for s in STATS]

g = pd.read_parquet('data/games.parquet'); tg = pd.read_parquet('data/team_games.parquet')
la = pd.read_parquet('data/lineup_agg.parquet')
if 'r_rapm' in la.columns: STATS = STATS + ['rapm']; RC = RC + ['r_rapm']
say('lineup columns:', RC)
g = g[g.season >= 2015].reset_index(drop=True); tg = tg[tg.game_id.isin(g.game_id)].copy()
tg['ppp'] = 100 * tg.pts / tg.poss; tg['loc'] = np.where(tg.neutral, 0, np.where(tg.ha == 'home', 1, -1))
own = la.set_index(['game_id', 'team']); 
def lx(df, team_col, suffix=''):
    cols = [c + suffix for c in RC]
    return own.reindex(pd.MultiIndex.from_arrays([df.game_id, df[team_col]]))[cols].to_numpy()
tg_X, tg_Y = lx(tg, 'team'), lx(tg, 'opp')   # offense lineup, defense lineup (actual players)
ok = ~np.isnan(tg_X).any(1) & ~np.isnan(tg_Y).any(1)
say('team-games with both lineups', ok.sum(), 'of', len(tg))

# ---- 1. lineup weights, walk-forward by season (fit on earlier seasons; 2015 uses 2015 itself = burn-in, not evaluated)
def design(X, Y, loc): return np.column_stack([np.ones(len(X)), X, Y, loc])
W = {}
for y in sorted(g.season.unique()):
    tr = ok & (tg.season < y).to_numpy() if y > 2015 else ok & (tg.season == 2015).to_numpy()
    if y == 2016: tr = ok & (tg.season == 2015).to_numpy()
    D = design(tg_X[tr], tg_Y[tr], tg['loc'].to_numpy()[tr])
    W[y] = np.linalg.lstsq(D, tg.ppp.to_numpy()[tr], rcond=None)[0]
wl = pd.DataFrame({y: w for y, w in W.items()}, index=['const'] + ['off_' + s for s in STATS] + ['def_' + s for s in STATS] + ['loc']).T
say('lineup weights (points per 100 per unit of lineup rate per 100), by fit season\n' + wl.round(3).to_string())
def lineup_pts(X, Y, season):
    w = np.vstack([W[s] for s in season])
    return (w[:, 1:1 + len(RC)] * X).sum(1) + (w[:, 1 + len(RC):1 + 2 * len(RC)] * Y).sum(1)
tg['lu'] = np.where(ok, lineup_pts(np.nan_to_num(tg_X), np.nan_to_num(tg_Y), tg.season.to_numpy()), np.nan)
tg['lu'] = tg.lu - tg.groupby('season').lu.transform('mean')   # centred: points per 100 above an average lineup matchup
tg['lu'] = tg.lu.fillna(0)

def run(k, params):
    t = tg.copy(); t['pts'] = (t.ppp - k * t.lu) * t.poss / 100
    h, final = ratings.walk_forward(g, t, **params)
    lu = t.set_index(['game_id', 'team']).lu
    h['lu_h'] = lu.reindex(pd.MultiIndex.from_arrays([h.game_id, h.home])).values
    h['lu_a'] = lu.reindex(pd.MultiIndex.from_arrays([h.game_id, h.away])).values
    h['ppp_h'] += k * h.lu_h; h['ppp_a'] += k * h.lu_a
    h['pts_h_hat'] = h.ppp_h * h.poss_hat / 100; h['pts_a_hat'] = h.ppp_a * h.poss_hat / 100
    h['margin_hat'] = h.pts_h_hat - h.pts_a_hat; h['total_hat'] = h.pts_h_hat + h.pts_a_hat
    return h, final

# ---- 2. tune on 2017-2021 closing-line games (k, prior weight, carry, recency); 2022-2026 held out
import itertools, os
grid = [] if ('--tune' in sys.argv or not os.path.exists('tuning.csv')) else None
for k, pgm, carry, hl in [(k, 10, 0.7, 90) for k in [0.0, 0.5, 0.8, 1.0, 1.2]] + list(itertools.product([0.0, 'best'], [5, 20], [0.5, 0.85], [45, 1000])) if grid is not None else []:
                if k == 'best': k = max(grid[:5], key=lambda r: -r[4])[0]
                p = dict(prior_games=pgm, carry=carry, tempo_carry=0.75, half_life=hl)
                h, _ = run(k, p); e = h[h.season.between(2017, 2021)]
                grid.append((k, pgm, carry, hl, rmse(e.margin, e.margin_hat), rmse(e.total, e.total_hat)))
                print(grid[-1], flush=True)
if grid is not None:
    grid = pd.DataFrame(grid, columns=['k', 'prior_games', 'carry', 'half_life', 'rmse_margin', 'rmse_total']).sort_values('rmse_margin'); grid.to_csv('tuning.csv', index=False)
grid = pd.read_csv('tuning.csv'); say('tuning (2017-2021)\n' + grid.head(12).round(3).to_string(index=False))
best = grid.iloc[0]; P = dict(prior_games=float(best.prior_games), carry=float(best.carry), tempo_carry=0.75, half_life=float(best.half_life))
k_best = float(best.k)
b0 = grid[grid.k == 0].iloc[0]; P0 = dict(prior_games=float(b0.prior_games), carry=float(b0.carry), tempo_carry=0.75, half_life=float(b0.half_life))
h, final = run(k_best, P); h0, _ = run(0.0, P0)
h['margin_team_only'] = h0.margin_hat.values; h['total_team_only'] = h0.total_hat.values
say('best', P, 'k', k_best, '| team-only best', P0)

# lineup absence: points per 100 lost vs the full roster (regulars who sat), for both teams
full = own[[c + '_full' for c in RC]].copy(); full.columns = RC
def lu_of(idx_df, team_col, opp_col, which):
    A = (own if which == 'act' else full).reindex(pd.MultiIndex.from_arrays([idx_df.game_id, idx_df[team_col]]))[RC].to_numpy()
    return A
for side, tm, op in [('h', 'home', 'away'), ('a', 'away', 'home')]:
    Xa, Ya = lu_of(h, tm, op, 'act'), lu_of(h, op, tm, 'act'); Xf = lu_of(h, tm, op, 'full'); Yf = lu_of(h, op, tm, 'full')
    s = h.season.to_numpy()
    # own absences: offense loss when own regulars sit, and defense loss (opponent scores more) for own sitting players
    w = np.vstack([W[x] for x in s]); wo, wd = w[:, 1:1 + len(RC)], w[:, 1 + len(RC):1 + 2 * len(RC)]
    h['abs_' + side] = ((wo * (Xa - Xf)).sum(1) - (wd * (Xa - Xf)).sum(1))   # own pts/100 change minus opponent pts/100 change
# expressed in points of margin for the home team (per 100 possessions x possessions/100)
h['abs_net_pts'] = (h.abs_h - h.abs_a) * h.poss_hat / 100
h['abs_any'] = np.maximum(-h.abs_h, -h.abs_a) * h.poss_hat / 100   # biggest single-team loss, points
h[['abs_h', 'abs_a', 'abs_net_pts', 'abs_any']] = h[['abs_h', 'abs_a', 'abs_net_pts', 'abs_any']].fillna(0)

# absence boost: the k-shrunk lineup term moves the line less than the market does when regulars sit. Fit, walk-forward on
# earlier seasons, how much of the full-weight absence value (abs_net_pts) to add on top
h['margin_pre_boost'] = h.margin_hat
boost = {}
for y in sorted(h.season.unique()):
    tr = h[(h.season < y) & (h.season >= 2016)] if y > 2016 else h[h.season == 2015]
    boost[y] = float(sm.OLS(tr.margin - tr.margin_hat, tr[['abs_net_pts']]).fit().params.iloc[0])
h['boost'] = h.season.map(boost)
h['margin_hat'] = h.margin_hat + h.boost * h.abs_net_pts
h['pts_h_hat'] += h.boost * h.abs_net_pts / 2; h['pts_a_hat'] -= h.boost * h.abs_net_pts / 2
say('absence boost by season (extra share of full absence value):', {k_: round(v, 3) for k_, v in boost.items()})

# N1: the prediction a bettor could make at the open, before same-day injury news: lineup = full roster minus CONTINUING
# absences (also missed the previous game). Same weights, k and boost; only the lineup inputs change.
known = own[[c + '_known' for c in RC]].copy(); known.columns = RC
def lu_pair(src, df, tm, op):
    X = src.reindex(pd.MultiIndex.from_arrays([df.game_id, df[tm]]))[RC].to_numpy(); Y = src.reindex(pd.MultiIndex.from_arrays([df.game_id, df[op]]))[RC].to_numpy()
    return lineup_pts(np.nan_to_num(X), np.nan_to_num(Y), df.season.to_numpy()), ~(np.isnan(X).any(1) | np.isnan(Y).any(1))
mu_lu = tg.assign(raw=np.where(ok, lineup_pts(np.nan_to_num(tg_X), np.nan_to_num(tg_Y), tg.season.to_numpy()), np.nan)).groupby('season').raw.mean()
for side, tm, op in [('h', 'home', 'away'), ('a', 'away', 'home')]:
    v, okk = lu_pair(known, h, tm, op); h['luk_' + side] = np.where(okk, v - h.season.map(mu_lu).to_numpy(), h['lu_' + side])
    w = np.vstack([W[x] for x in h.season.to_numpy()]); wo, wd = w[:, 1:1 + len(RC)], w[:, 1 + len(RC):1 + 2 * len(RC)]
    Xk = known.reindex(pd.MultiIndex.from_arrays([h.game_id, h[tm]]))[RC].to_numpy(); Xf = full.reindex(pd.MultiIndex.from_arrays([h.game_id, h[tm]]))[RC].to_numpy()
    h['absk_' + side] = np.nan_to_num((wo * (Xk - Xf)).sum(1) - (wd * (Xk - Xf)).sum(1))
h['abs_net_known'] = (h.absk_h - h.absk_a) * h.poss_hat / 100
h['margin_known'] = h.margin_pre_boost + k_best * ((h.luk_h - h.lu_h) - (h.luk_a - h.lu_a)) * h.poss_hat / 100 + h.boost * h.abs_net_known
h['total_known'] = h.total_hat + k_best * ((h.luk_h - h.lu_h) + (h.luk_a - h.lu_a)) * h.poss_hat / 100
nm = la.set_index(['game_id', 'team']).miss_min_new
h['new_abs'] = (nm.reindex(pd.MultiIndex.from_arrays([h.game_id, h.home])).fillna(0).values + nm.reindex(pd.MultiIndex.from_arrays([h.game_id, h.away])).fillna(0).values) > 0

# N4: playoff home edge. Mean residual of playoff games (home side), fit on earlier seasons, added to playoff games.
h['playoff'] = h.playoff.fillna(False).astype(bool)
po = {}
for y in sorted(h.season.unique()):
    tr = h[(h.season < y) & h.playoff]
    po[y] = float((tr.margin - tr.margin_hat).mean()) if len(tr) > 150 else 0.0
h['po_adj'] = np.where(h.playoff, h.season.map(po), 0.0)
e_ = h[h.playoff & (h.season >= 2017)]
say('playoff home-edge term by season:', {k_: round(v, 2) for k_, v in po.items()}, '| playoff margin RMSE before %.3f after %.3f (n %d)' % (
    rmse(e_.margin, e_.margin_hat), rmse(e_.margin, e_.margin_hat + e_.po_adj), len(e_)))
for c_ in ['margin_hat', 'margin_known']: h[c_] = h[c_] + h.po_adj

# ---- 3. accuracy vs market
x = h[h.close.notna() & (h.season >= 2017)]
rows = []
for s, d in list(x.groupby('season')) + [('2017-2026', x), ('2022-2026 (held out)', x[x.season >= 2022])]:
    rows.append((s, len(d), rmse(d.margin, d.close), rmse(d.margin, d.margin_hat), rmse(d.margin, d.margin_team_only), rmse(d.margin, d.margin_pre_boost),
                 rmse(d.total, d.total_close), rmse(d.total, d.total_hat), rmse(d.margin, d.sbr_open) if d.sbr_open.notna().any() else np.nan))
acc = pd.DataFrame(rows, columns=['season', 'games', 'rmse_close', 'rmse_model', 'rmse_team_only', 'rmse_no_boost', 'rmse_total_close', 'rmse_total_model', 'rmse_open'])
acc.round(3).to_csv('accuracy_vs_close.csv', index=False); say('\nMargin / total RMSE vs closing line\n' + acc.round(3).to_string(index=False))

# star absences: does the lineup layer track the market?  close - team-only model vs the lineup model's absence adjustment
z = x.copy(); z['mkt_gap'] = z.close - z.margin_team_only; z['lu_gap'] = z.margin_hat - z.margin_team_only
r = sm.OLS(z.mkt_gap, sm.add_constant(z[['lu_gap']])).fit()
say('\nclose minus team-only model, regressed on lineup adjustment: b %.3f (1 = market moves exactly as the lineup layer) t %.1f n %d' % (r.params.lu_gap, r.tvalues.lu_gap, len(z)))
bins = [0, 1, 2, 3, 4, 6, 99]; z['absb'] = pd.cut(z.abs_any, bins, right=False)
ab = z.groupby('absb', observed=True).apply(lambda d: pd.Series({'games': len(d), 'rmse_close': rmse(d.margin, d.close), 'rmse_lineup_model': rmse(d.margin, d.margin_hat),
      'rmse_team_only': rmse(d.margin, d.margin_team_only), 'mean_abs_close_minus_team_only': (d.close - d.margin_team_only).abs().mean(),
      'mean_abs_close_minus_lineup': (d.close - d.margin_hat).abs().mean()}))
ab.round(3).to_csv('star_absence_accuracy.csv'); say('by size of the biggest absence (points of margin lost by the weakened team)\n' + ab.round(3).to_string())

# ---- 4. win probability
h['win'] = (h.margin > 0).astype(float)
sig = {y: (rmse(h[h.season.between(2016, y - 1)].margin, h[h.season.between(2016, y - 1)].margin_hat) if y > 2016 else 13.0) for y in h.season.unique()}
h['sigma'] = h.season.map(sig); h['p_win'] = norm.cdf(h.margin_hat / h.sigma)
def mlp(ml): return np.where(ml < 0, -ml / (100 - ml), 100 / (100 + ml))
ph, pa = mlp(h.sbr_ml_home), mlp(h.sbr_ml_away); h['p_ml'] = ph / (ph + pa)
def ll(p, y): p = np.clip(p, 1e-4, 1 - 1e-4); return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))
w = h[h.p_ml.notna() & (h.season >= 2017)]
w = w.assign(p_close=norm.cdf(w.close / 13.0))
say('\nWin probability 2017-2022 (%d games with SBR moneylines): logloss model %.4f | moneyline %.4f | brier model %.4f moneyline %.4f' % (
    len(w), ll(w.p_win, w.win), ll(w.p_ml, w.win), np.mean((w.p_win - w.win)**2), np.mean((w.p_ml - w.win)**2)))
# N3: market-anchored win probability: centre = line + b * (model - line) (b fit on earlier seasons vs the close), and
# sigma grows with the game's total (more points, more variance): sigma = s0 * sqrt(total / 220), s0 fit on earlier seasons
c_all = h[h.close.notna()].copy(); c_all['gap'] = c_all.margin_hat - c_all.close
bw, s0 = {}, {}
for y in sorted(h.season.unique()):
    tr = c_all[c_all.season < y]
    if len(tr) < 2000: bw[y], s0[y] = 0.0, 13.0; continue
    bw[y] = float(np.polyfit(tr.gap, tr.margin - tr.close, 1)[0])
    ctr = tr.close + bw[y] * tr.gap; sc = np.sqrt(tr.total_close.fillna(220) / 220)
    s0[y] = float(np.sqrt(np.mean(((tr.margin - ctr) / sc) ** 2)))
h['p_win_mkt'] = np.where(h.close.notna(), norm.cdf((h.close + h.season.map(bw) * (h.margin_hat - h.close)) / (h.season.map(s0) * np.sqrt(h.total_close.fillna(220) / 220))), h.p_win)
w = h[h.p_ml.notna() & (h.season >= 2017)]
say('win logloss: model alone %.4f | market-anchored, sigma by total %.4f | no-vig moneyline %.4f (n %d)' % (ll(w.p_win, w.win), ll(w.p_win_mkt, w.win), ll(w.p_ml, w.win), len(w)))
cal = w.assign(b=pd.cut(w.p_win, np.linspace(0, 1, 11))).groupby('b', observed=True).agg(n=('win', 'size'), predicted=('p_win', 'mean'), actual=('win', 'mean'))
cal.round(3).to_csv('win_prob_calibration.csv'); say(cal.round(3).to_string())

# ---- 5. betting tests: vs close (2017-26) and vs SBR open (2017-22), plus closing-line value
def ats(gap, res, thr):
    res = res.loc[gap.index]; s_ = gap.abs() >= thr; c = np.sign(gap[s_]) * np.sign(res[s_]); wn, ls = int((c > 0).sum()), int((c < 0).sum()); n = wn + ls
    return n, (wn / n if n else np.nan), (1 - norm.cdf((wn - 0.5238 * n) / np.sqrt(n * 0.25)) if n else np.nan)
o = h[(h.season >= 2017) & h.sbr_open.notna() & h.sbr_close.notna()].copy()
o['gap_o'] = o.margin_hat - o.sbr_open; o['gap_o0'] = o.margin_team_only - o.sbr_open; o['move'] = o.sbr_close - o.sbr_open
o['tgap_o'] = o.total_hat - o.sbr_total_open; o['tmove'] = o.sbr_total_close - o.sbr_total_open
c = h[(h.season >= 2017) & h.close.notna()].copy(); c['gap'] = c.margin_hat - c.close; c['tgap'] = c.total_hat - c.total_close
out = []
for thr in [0, 1, 2, 3, 4, 5]:
    for lab, gap, res in [('spread vs close 2017-26', c.gap, c.margin - c.close), ('total vs close 2017-26', c.tgap, c.total - c.total_close),
                          ('spread vs OPEN 2017-22, lineup model', o.gap_o, o.margin - o.sbr_open), ('spread vs OPEN 2017-22, team-only', o.gap_o0, o.margin - o.sbr_open),
                          ('total vs OPEN 2017-22', o.tgap_o, o.total - o.sbr_total_open)]:
        n, wp, p = ats(gap.dropna(), res, thr); out.append((lab, thr, n, wp, p))
bets = pd.DataFrame(out, columns=['test', 'min_gap', 'bets', 'win_pct', 'p_vs_52.4']); bets.round(4).to_csv('bets_by_threshold.csv', index=False)
say('\nAgainst the spread / totals by minimum model gap\n' + bets.round(4).to_string(index=False))
clv = []
for lab, d in [('all', o), ('no regular missing either side', o[o.abs_any < 0.5]), ('a regular missing', o[o.abs_any >= 0.5])]:
    for gcol, mcol in [('gap_o', 'move'), ('gap_o0', 'move'), ('tgap_o', 'tmove')]:
        dd = d[d[gcol].notna() & d[mcol].notna()]; rr = sm.OLS(dd[mcol], sm.add_constant(dd[gcol])).fit()
        for thr in [2, 3, 4]:
            s_ = dd[dd[gcol].abs() >= thr]; cv = np.sign(s_[gcol]) * s_[mcol]
            clv.append((lab, {'gap_o': 'spread, lineup model', 'gap_o0': 'spread, team-only', 'tgap_o': 'total'}[gcol], len(dd), rr.params[gcol], rr.tvalues[gcol], thr, len(s_), cv.mean(),
                        cv.mean() / cv.std() * np.sqrt(len(cv)) if len(cv) > 2 else np.nan, (cv > 0).mean(), (cv < 0).mean()))
clv = pd.DataFrame(clv, columns=['games', 'model', 'n', 'move_on_gap_b', 't', 'min_gap', 'bets', 'avg_clv_pts', 't_clv', 'beat_close', 'worse']).round(3)
clv.to_csv('clv_open.csv', index=False); say('\nClosing-line value from the SBR opener, 2017-2022\n' + clv.to_string(index=False))

# market-anchored cover / over probabilities: E[result - line] = b * (model - line), b fit on earlier seasons
rows = []
for y in range(2018, 2027):
    tr = c[c.season < y]; b = float(np.polyfit(tr.gap, tr.margin - tr.close, 1)[0]); s = rmse(tr.margin - tr.close, b * tr.gap)
    t2 = tr[tr.tgap.notna()]; bt = float(np.polyfit(t2.tgap, t2.total - t2.total_close, 1)[0]); st = rmse(t2.total - t2.total_close, bt * t2.tgap)
    rows.append((y, b, s, bt, st))
cb = pd.DataFrame(rows, columns=['season', 'b_spread', 'sd_ats', 'b_total', 'sd_total']).set_index('season'); say('\nblend weights vs close\n' + cb.round(3).to_string())
ro = sm.OLS(o.margin - o.sbr_open, sm.add_constant(o.gap_o)).fit(); oo = o[o.tgap_o.notna()]; rto = sm.OLS(oo.total - oo.sbr_total_open, sm.add_constant(oo.tgap_o)).fit()
# N1: re-run the opener test with only what was knowable at the open (continuing absences), split by new-absence games
o['gap_k'] = o.margin_known - o.sbr_open; o['tgap_k'] = o.total_known - o.sbr_total_open
say('\nOpener test, model priced with the lineup known at the open (continuing absences only):')
for lab, d in [('all games', o), ('no new absence either side', o[~o.new_abs]), ('a new absence (news after the open)', o[o.new_abs])]:
    for gc, mc, nm_ in [('gap_o', 'move', 'spread, actual lineup (knows who sat)'), ('gap_k', 'move', 'spread, lineup known at open'), ('tgap_k', 'tmove', 'total, lineup known at open')]:
        dd = d[d[gc].notna() & d[mc].notna()]; s3 = dd[dd[gc].abs() >= 3]; cv = np.sign(s3[gc]) * s3[mc]
        n3, w3, _ = ats(dd[gc], (dd.margin - dd.sbr_open) if 'spread' in nm_ else (dd.total - dd.sbr_total_open), 3)
        say('  %-36s %-40s gap>=3: %5d bets, CLV %+.2f (t %.1f), ATS %.1f%%' % (lab, nm_, len(s3), cv.mean(), cv.mean() / cv.std() * np.sqrt(len(cv)), 100 * w3))
# open blend weight, walk-forward by season (the old saved weight was fit on all 2017-22 games at once)
bo = {}
for y in sorted(o.season.unique()):
    tr = o[o.season < y]
    if len(tr) > 1000: bo[y] = float(np.polyfit(tr.gap_k, tr.margin - tr.sbr_open, 1)[0])
say('opener blend weight on the known-at-open gap, fit on earlier seasons:', {k_: round(v, 3) for k_, v in bo.items()})
rk = sm.OLS(o.margin - o.sbr_open, sm.add_constant(o.gap_k)).fit()
say('result vs open on model gap (2017-22): spread b %.3f t %.1f | total b %.3f t %.1f' % (ro.params.gap_o, ro.tvalues.gap_o, rto.params.tgap_o, rto.tvalues.tgap_o))
h = h.join(cb, on='season'); h['p_cover_home'] = norm.cdf(h.b_spread * (h.margin_hat - h.close) / h.sd_ats)

# ---- 6. save
keep = ['game_id', 'season', 'date', 'home', 'away', 'playoff', 'home_score', 'away_score', 'margin', 'total', 'poss', 'O_h', 'D_h', 'O_a', 'D_a', 'P_h', 'P_a',
        'mu', 'hca', 'T0', 'lu_h', 'lu_a', 'abs_h', 'abs_a', 'abs_net_pts', 'poss_hat', 'pts_h_hat', 'pts_a_hat', 'margin_hat', 'total_hat', 'margin_team_only',
        'p_win', 'p_win_mkt', 'margin_known', 'total_known', 'new_abs', 'po_adj', 'close', 'total_close', 'sbr_open', 'sbr_close', 'sbr_total_open', 'sbr_total_close', 'p_ml', 'p_cover_home']
h[keep].to_parquet('data/predictions.parquet')
last = max(final); cur = final[last].sort_values('net', ascending=False); cur['tempo'] = cur.T0 + cur.P
cur.rename_axis('team').reset_index()[['team', 'net', 'O', 'D', 'tempo', 'mu', 'hca', 'T0']].round(2).to_csv('team_residual_ratings.csv', index=False)
json.dump({'params': P, 'k': k_best, 'absence_boost': boost[max(boost)], 'lineup_weights': {str(y): list(map(float, W[y])) for y in W}, 'stats': STATS, 'sigma_margin': sig[max(sig)],
           'blend_close': cb.iloc[-1].round(4).to_dict(), 'rc': RC, 'win_market': {'b': bw[max(bw)], 's0': s0[max(s0)]}, 'playoff_home_edge': po[max(po)],
           'blend_open': {'b_spread_open': float(np.mean(list(bo.values())[-2:])) if bo else float(rk.params.gap_k), 'sd_ats_open': float(np.std(rk.resid)),
           'in_sample_b_spread_open_actual_lineup': float(ro.params.gap_o),
           'b_total_open': float(rto.params.tgap_o), 'sd_total_open': float(np.std(rto.resid))}, 'season': int(last)}, open('model_config.json', 'w'), indent=1)
say('saved')
