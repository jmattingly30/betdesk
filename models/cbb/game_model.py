"""College basketball: tempo-and-efficiency score model -> margin, total, win probability, cover probability.
Backtest against closing lines (2013-2023 consensus, 2026 DraftKings) and opening lines (2026), walk-forward throughout.

usage: python3 game_model.py            (reads data/, writes data/cbb_predictions.parquet, ratings_cbb.csv, *.csv, run_log.txt)"""
import sys, json, numpy as np, pandas as pd
from scipy.stats import norm, t as tdist
import ratings, priors, pipeline

# 2026-10-09 audit: hyperparameters re-tuned with the roster prior on (tune.py, 2012-2019 only -> tuning.csv best row);
# blowout cap (C4) and near-home 'neutral' edge (C5) are tuned there too and kept only if they lowered the error.
_t = pd.read_csv('tuning.csv').sort_values('rmse_margin').iloc[0]
PARAMS = dict(prior_games=int(_t.prior_games), carry=float(_t.carry), tempo_carry=float(_t.tempo_carry), half_life=int(_t.half_life),
              cap=None if pd.isna(_t.cap) else float(_t.cap), cap_slope=float(_t.cap_slope))
NH = float(_t.nh)
log = open('run_log.txt', 'w')
def say(*a):
    s = ' '.join(str(x) for x in a); print(s, flush=True); log.write(s + '\n')
say('tuned params', PARAMS, 'near-home neutral', NH)

g, tg, links = pipeline.load()
L = pd.read_parquet('data/cbb_lines.parquet')

# ---- 1. ratings: pass 1 (carry rule) gives end-of-season ratings; pass 2 uses roster-based preseason priors
h, final, pr, coefs = pipeline.run(g, tg, links, PARAMS, NH)
coefs.to_csv('prior_coefficients.csv', index=False)
h = h.merge(L, on='game_id', how='left')

# ---- 2. clean lines: drop spreads whose side disagrees with the same book's moneyline (ESPN feed errors, ~1%)
def ml_prob(ml): return np.where(ml < 0, -ml / (100 - ml), 100 / (100 + ml))
ph, pa = ml_prob(h.ml_home), ml_prob(h.ml_away)
h['p_ml'] = ph / (ph + pa)
ml_margin = 11.0 * norm.ppf(np.clip(h.p_ml, 0.005, 0.995))
bad = h.line.notna() & h.p_ml.notna() & ((h.line - ml_margin).abs() > 6)
say('lines', int(h.line.notna().sum()), 'dropped as inconsistent with moneyline', int(bad.sum()))
h.loc[bad, ['line', 'total_line']] = np.nan
# 'verified' close: consensus and TeamRankings agree within half a point. Where they disagree, one number is usually stale
# (an earlier line), and the model 'beats' it; that is an opener effect, not an edge at the close.
h['verified'] = h.line.notna() & (((h.line_cons - h.line_tr).abs() <= 0.5) | (h.line_src == 'DraftKings'))
say('verified closes (two sources agree) %d of %d' % (h.verified.sum(), h.line.notna().sum()))
h['hca_used'] = np.where(h.neutral, 0, h.hca)

# ---- 2b. player availability (audit fix C1). abs_val = margin points lost from regulars who sat (prep_availability.py).
#      k (spread) and kt (total) fit walk-forward on earlier seasons using every absence (the true effect of a missing player).
#      The model's prices then use only CONTINUING absences (also missed the previous game), which a bettor knows before
#      the line opens; using every absence would credit the backtest with injury news nobody had yet.
A = pd.read_parquet('data/cbb_availability.parquet')[['game_id', 'team', 'abs_val', 'abs_val_cont']]
for side, col in [('h', 'home'), ('a', 'away')]:
    h = h.merge(A.rename(columns={'team': col, 'abs_val': 'abs_' + side, 'abs_val_cont': 'absc_' + side}), on=['game_id', col], how='left')
for c in ['abs_h', 'abs_a', 'absc_h', 'absc_a']: h[c] = h[c].fillna(0.0)
h['margin_hat_base'] = h.margin_hat; h['total_hat_base'] = h.total_hat
h['d_abs'] = h.abs_a - h.abs_h; h['s_abs'] = h.abs_a + h.abs_h
kk = {}
for y in sorted(h.season.unique()):
    tr = h[(h.season < y) & (h.season >= 2012) & (h.s_abs > 0)]
    if len(tr) < 2000: kk[y] = (0.0, 0.0); continue
    k = float(np.sum(tr.d_abs * (tr.margin - tr.margin_hat_base)) / np.sum(tr.d_abs ** 2))
    kt = float(np.polyfit(tr.s_abs, tr.total - tr.total_hat_base, 1)[0])
    kk[y] = (k, kt)
h['k_av'] = h.season.map(lambda y: kk[y][0]); h['kt_av'] = h.season.map(lambda y: kk[y][1])
say('\nAvailability weights by season (k: margin pts per pt of value lost; kt: total pts per pt lost)\n' +
    pd.DataFrame(kk, index=['k', 'kt']).T.round(3).to_string())
# every absence (needs injury news; upper bound) vs continuing absences (known before the open; what predict.py prices)
h['margin_hat_all'] = h.margin_hat_base + h.k_av * h.d_abs; h['total_hat_all'] = h.total_hat_base + h.kt_av * h.s_abs
h['margin_hat'] = h.margin_hat_base + h.k_av * (h.absc_a - h.absc_h); h['total_hat'] = h.total_hat_base + h.kt_av * (h.absc_a + h.absc_h)

# ---- 3. accuracy vs the market
def rmse(a, b): return float(np.sqrt(np.nanmean((a - b)**2)))
def mae(a, b): return float(np.nanmean(np.abs(a - b)))
x = h[h.line.notna()]
acc = x.groupby('season').apply(lambda d: pd.Series({
    'games': len(d), 'rmse_line': rmse(d.margin, d.line), 'rmse_model': rmse(d.margin, d.margin_hat),
    'mae_line': mae(d.margin, d.line), 'mae_model': mae(d.margin, d.margin_hat),
    'rmse_total_line': rmse(d.total, d.total_line), 'rmse_total_model': rmse(d.total, d.total_hat),
    'corr_model_line': np.corrcoef(d.line, d.margin_hat)[0, 1]})).round(3)
acc.loc['all'] = [len(x), rmse(x.margin, x.line), rmse(x.margin, x.margin_hat), mae(x.margin, x.line), mae(x.margin, x.margin_hat),
                  rmse(x.total, x.total_line), rmse(x.total, x.total_hat), np.corrcoef(x.line, x.margin_hat)[0, 1]]
acc.to_csv('accuracy_vs_close.csv'); say('\nAccuracy vs closing line (lined D1 games)\n' + acc.round(3).to_string())
m = h[h.season >= 2012]
say('\nAll D1 games 2012-2026: margin RMSE %.2f MAE %.2f, total RMSE %.2f, bias margin %.2f total %.2f, n %d' % (
    rmse(m.margin, m.margin_hat), mae(m.margin, m.margin_hat), rmse(m.total, m.total_hat), (m.margin - m.margin_hat).mean(), (m.total - m.total_hat).mean(), len(m)))
bym = m.groupby(m.date.dt.month).apply(lambda d: pd.Series({'n': len(d), 'rmse': rmse(d.margin, d.margin_hat), 'rmse_line': rmse(d.margin[d.line.notna()], d.line[d.line.notna()])})).round(2)
say('by month\n' + bym.to_string())
mm = m[(m.abs_h + m.abs_a) > 0]
say('availability (games 2013+ where a regular sat, n %d): margin RMSE no adjustment %.3f | continuing absences %.3f | every absence %.3f' % (
    len(mm[mm.season >= 2013]), *[rmse(mm[mm.season >= 2013].margin, mm[mm.season >= 2013][c]) for c in ['margin_hat_base', 'margin_hat', 'margin_hat_all']]))

# ---- 4. win probability: P(home win) = Phi(margin_hat / sigma), sigma fit walk-forward on earlier seasons
def ll(p, y): p = np.clip(p, 1e-4, 1 - 1e-4); return float(-np.mean(y*np.log(p) + (1 - y)*np.log(1 - p)))
h['win'] = (h.margin > 0).astype(float)
# sigma varies by month (early-season ratings are noisier) and by pace (more possessions, more variance) - audit fix C3.
#   sigma = s_month x (poss_hat / 68) ** e, s_month and e fit walk-forward on earlier seasons
def mon(d): return np.clip(np.where(d.dt.month >= 10, d.dt.month - 10, d.dt.month + 2), 1, 5)   # 1 Nov .. 5 Mar+
h['mon'] = mon(h.date)
SIG = {}
for y in sorted(h.season.unique()):
    tr = h[(h.season < y) & (h.season >= 2010)]
    if len(tr) < 1000: SIG[y] = ({k: 11.5 for k in range(1, 6)}, 0.0); continue
    r2 = (tr.margin - tr.margin_hat) ** 2; pr_ = tr.poss_hat / 68
    best = None
    for e in [0, 0.25, 0.5, 0.75, 1.0]:
        sm_ = {k: float(np.sqrt(np.mean(r2[tr.mon == k] / pr_[tr.mon == k] ** (2 * e)))) for k in range(1, 6)}
        sg = tr.mon.map(sm_) * pr_ ** e; nll = float(np.mean(np.log(sg) + r2 / (2 * sg ** 2)))
        if best is None or nll < best[0]: best = (nll, sm_, e)
    SIG[y] = (best[1], best[2])
h['sigma'] = [SIG[y][0][k] * (p / 68) ** SIG[y][1] for y, k, p in zip(h.season, h.mon, h.poss_hat)]
h['sig_flat'] = h.season.map({y: float(np.sqrt(np.mean([v ** 2 for v in SIG[y][0].values()]))) for y in SIG})
ls_ = SIG[max(SIG)]
say('\nsigma by month (Nov..Mar) %s, pace exponent %.2f' % ([round(v, 2) for v in ls_[0].values()], ls_[1]))
h['p_win'] = norm.cdf(h.margin_hat / h.sigma)
w = h[h.p_ml.notna() & h.season.between(2013, 2026)]
wp = pd.DataFrame({
    'model': [ll(w.p_win, w.win), float(np.mean((w.p_win - w.win)**2))],
    'moneyline (no-vig)': [ll(w.p_ml, w.win), float(np.mean((w.p_ml - w.win)**2))],
    'ESPN BPI': [ll(w.bpi_home.fillna(w.p_win), w.win), float(np.mean((w.bpi_home.fillna(w.p_win) - w.win)**2))]}, index=['logloss', 'brier']).round(4)
say('\nWin probability, %d games with moneylines\n' % len(w) + wp.to_string())
say('win logloss with one sigma per season %.4f vs by month and pace %.4f' % (ll(norm.cdf(w.margin_hat / w.sig_flat), w.win), ll(w.p_win, w.win)))
cal = w.assign(b=pd.cut(w.p_win, np.linspace(0, 1, 11))).groupby('b', observed=True).agg(n=('win', 'size'), predicted=('p_win', 'mean'), actual=('win', 'mean')).round(3)
cal.to_csv('win_prob_calibration.csv'); say('calibration\n' + cal.to_string())

# ---- 5. cover probability: market-anchored. E[margin] = line + b*(model - line); P(home covers) = Phi(b*(model-line)/s)
#      b and s fit walk-forward on earlier seasons (b is how much of the model's disagreement with the close is real)
h['gap'] = h.margin_hat - h.line; h['tgap'] = h.total_hat - h.total_line
rows = []
for y in sorted(h.season[h.line.notna()].unique()):
    tr = h[(h.season < y) & h.verified]
    if len(tr) < 3000: b, s, bt, st = 0.0, 11.0, 0.0, 17.0
    else:
        b = float(np.polyfit(tr.gap, tr.margin - tr.line, 1)[0]); s = rmse(tr.margin - tr.line, b*tr.gap)
        t2 = tr[tr.total_line.notna()]; bt = float(np.polyfit(t2.tgap, t2.total - t2.total_line, 1)[0]); st = rmse(t2.total - t2.total_line, bt*t2.tgap)
    rows.append((y, b, s, bt, st))
cb = pd.DataFrame(rows, columns=['season', 'b_spread', 'sd_ats', 'b_total', 'sd_total']).set_index('season')
say('\nWalk-forward market-blend weights (share of model-minus-line that is real)\n' + cb.round(3).to_string())
h = h.join(cb, on='season')
# spread/total uncertainty scales with the game's sigma (month and pace), relative to that season's average sigma
h['sd_scale'] = h.sigma / h.sig_flat
h['p_cover_home'] = norm.cdf(h.b_spread * h.gap / (h.sd_ats * h.sd_scale))
h['p_over'] = norm.cdf(h.b_total * h.tgap / (h.sd_total * h.sd_scale))
# in-sample check of the whole period: does the gap predict the result vs the close?
z = h[h.line.notna()]
import statsmodels.api as sm
r = sm.OLS(z.margin - z.line, sm.add_constant(z.gap)).fit(cov_type='cluster', cov_kwds={'groups': z.season})
rt = sm.OLS((z.total - z.total_line)[z.total_line.notna()], sm.add_constant(z.tgap[z.total_line.notna()])).fit()
say('margin-minus-close on model gap: b %.3f t %.2f | total: b %.3f t %.2f' % (r.params.gap, r.tvalues.gap, rt.params.tgap, rt.tvalues.tgap))
for lab, d_ in [('verified close 2014-23', z[z.verified & z.season.between(2014, 2023)]), ('DK close 2026', z[z.season == 2026])]:
    r_ = sm.OLS(d_.margin - d_.line, sm.add_constant(d_.gap)).fit(); d2 = d_[d_.total_line.notna()]; rt_ = sm.OLS(d2.total - d2.total_line, sm.add_constant(d2.tgap)).fit()
    say('  %s (n %d): spread b %.3f t %.2f | total b %.3f t %.2f' % (lab, len(d_), r_.params.gap, r_.tvalues.gap, rt_.params.tgap, rt_.tvalues.tgap))

def ats(d, gapcol, res, thr):
    s_ = d[d[gapcol].abs() >= thr]; pick = np.sign(s_[gapcol]); c = np.sign(res[s_.index]) * pick
    w_, l_ = int((c > 0).sum()), int((c < 0).sum()); n = w_ + l_
    return n, (w_ / n if n else np.nan), (1 - norm.cdf((w_ - 0.5238*n) / np.sqrt(n*0.25)) if n else np.nan)
out = []
for thr in [0, 1, 2, 3, 4, 5, 6]:
    for lab, d_ in [('2014-2023 close, all', z[z.season.between(2014, 2023)]), ('2014-2023 close, verified', z[z.season.between(2014, 2023) & z.verified]),
                    ('2014-2023 close, sources disagree', z[z.season.between(2014, 2023) & ~z.verified & z.line_tr.notna()]), ('2026 close (DK)', z[z.season == 2026])]:
        n, wp_, p = ats(d_, 'gap', d_.margin - d_.line, thr); out.append(('spread', lab, thr, n, wp_, p))
        dt = d_[d_.total_line.notna()]; n, wp_, p = ats(dt, 'tgap', dt.total - dt.total_line, thr); out.append(('total', lab, thr, n, wp_, p))
# opening lines 2026: bet the model vs the open, grade vs result, and measure closing-line value
o = h[h.line_open.notna() & h.line_close_dk.notna()].copy()
o['gap_o'] = o.margin_hat - o.line_open; o['tgap_o'] = o.total_hat - o.total_open
for thr in [0, 1, 2, 3, 4, 5, 6]:
    n, wp_, p = ats(o, 'gap_o', o.margin - o.line_open, thr); out.append(('spread', '2026 OPEN (DK)', thr, n, wp_, p))
    n, wp_, p = ats(o, 'tgap_o', o.total - o.total_open, thr); out.append(('total', '2026 OPEN (DK)', thr, n, wp_, p))
bets = pd.DataFrame(out, columns=['market', 'line', 'min_gap', 'bets', 'win_pct', 'p_vs_52.4']).round(4)
bets.to_csv('bets_by_threshold.csv', index=False); say('\nAgainst the spread / totals by minimum model gap\n' + bets.to_string())
# CLV: does the model-minus-open gap predict how the line moves to the close?
o['move'] = o.line_close_dk - o.line_open; o['tmove'] = o.total_close_dk - o.total_open
rc = sm.OLS(o.move, sm.add_constant(o.gap_o)).fit(); rtc = sm.OLS(o.tmove[o.tmove.notna()], sm.add_constant(o.tgap_o[o.tmove.notna()])).fit()
say('\nCLV 2026 (n %d): line move on model gap b %.3f t %.2f; total move b %.3f t %.2f' % (len(o), rc.params.gap_o, rc.tvalues.gap_o, rtc.params.tgap_o, rtc.tvalues.tgap_o))
clv = []
for thr in [1, 2, 3, 4, 5]:
    s_ = o[o.gap_o.abs() >= thr]; c = np.sign(s_.gap_o) * s_.move
    st_ = o[o.tgap_o.abs() >= thr]; ct = np.sign(st_.tgap_o) * st_.tmove
    clv.append((thr, len(s_), c.mean(), c.mean() / c.std() * np.sqrt(len(c)), (c > 0).mean(), (c < 0).mean(), len(st_), ct.mean(), ct.mean() / ct.std() * np.sqrt(len(ct))))
clv = pd.DataFrame(clv, columns=['min_gap', 'spread_bets', 'avg_clv_pts', 't', 'beat_close', 'worse_than_close', 'total_bets', 'avg_total_clv', 't_total']).round(3)
clv.to_csv('clv_2026_open.csv', index=False); say(clv.to_string())
ro = sm.OLS(o.margin - o.line_open, sm.add_constant(o.gap_o)).fit(); rto = sm.OLS((o.total - o.total_open)[o.total_open.notna()], sm.add_constant(o.tgap_o[o.total_open.notna()])).fit()
say('Result vs OPEN on model gap (2026, in-sample): spread b %.3f t %.2f | total b %.3f t %.2f' % (ro.params.gap_o, ro.tvalues.gap_o, rto.params.tgap_o, rto.tvalues.tgap_o))
# audit fix C2: the old 0.35 weight was fit on the same 2026 games it was judged on. Cross-fit: fit on one half of the
# season, test on the other. The weight saved for bet sizing is the cross-fit average shrunk halfway to the close weight
# (one season of openers), and ONE pre-registered rule is reported: |model - open| >= 3, judged on each half separately.
o = o.sort_values('date'); half = np.where(np.arange(len(o)) < len(o) // 2, 'A', 'B'); o['half'] = half
cf = {}
for fit_, test_ in [('A', 'B'), ('B', 'A')]:
    f_, t_ = o[o.half == fit_], o[o.half == test_]
    b_ = float(np.polyfit(f_.gap_o, f_.margin - f_.line_open, 1)[0]); ft = f_[f_.total_open.notna()]; bt_ = float(np.polyfit(ft.tgap_o, ft.total - ft.total_open, 1)[0])
    tt = t_[t_.total_open.notna()]
    oos = rmse(t_.margin - t_.line_open, b_ * t_.gap_o); base_ = rmse(t_.margin, t_.line_open)
    n3, w3, p3 = ats(t_, 'gap_o', t_.margin - t_.line_open, 3); n3t, w3t, p3t = ats(tt, 'tgap_o', tt.total - tt.total_open, 3)
    c3 = (np.sign(t_.gap_o) * t_.move)[t_.gap_o.abs() >= 3].mean()
    cf[test_] = (b_, bt_)
    say('  cross-fit: weights from half %s (spread %.3f, total %.3f) -> half %s: margin RMSE open %.3f, open+blend %.3f | rule gap>=3: spread %d bets %.1f%% CLV %+.2f, totals %d bets %.1f%%' % (
        fit_, b_, bt_, test_, base_, oos, n3, 100 * w3, c3, n3t, 100 * w3t))
bc_ = cb.iloc[-1]
b_sp = 0.5 * (np.mean([v[0] for v in cf.values()]) + bc_.b_spread); b_to = 0.5 * (np.mean([v[1] for v in cf.values()]) + bc_.b_total)
sd_o = rmse(o.margin - o.line_open, b_sp * o.gap_o); tt = o[o.total_open.notna()]; sdt_o = rmse(tt.total - tt.total_open, b_to * tt.tgap_o)
b_open = dict(b_spread_open=float(b_sp), sd_ats_open=float(sd_o), b_total_open=float(b_to), sd_total_open=float(sdt_o),
              in_sample_b_spread=float(ro.params.gap_o), in_sample_b_total=float(rto.params.tgap_o))
say('opener weights saved (cross-fit, shrunk halfway to the close): spread %.3f total %.3f (in-sample was %.3f / %.3f)' % (b_sp, b_to, ro.params.gap_o, rto.params.tgap_o))
# opener test by availability version: none / continuing absences (what a bettor knows at the open) / every absence (needs news)
for lab, mc, tc in [('no availability', 'margin_hat_base', 'total_hat_base'), ('continuing absences', 'margin_hat', 'total_hat'), ('every absence', 'margin_hat_all', 'total_hat_all')]:
    gg = o[mc] - o.line_open; s3 = o[gg.abs() >= 3]; c = (np.sign(gg) * o.move)[gg.abs() >= 3]
    n3, w3, _ = ats(o.assign(_g=gg), '_g', o.margin - o.line_open, 3)
    say('  opener, model %-20s: margin RMSE %.3f | gap>=3: %d bets, %.1f%% ATS, CLV %+.2f pts (t %.1f)' % (lab, rmse(o.margin, o[mc]), n3, 100 * w3, c.mean(), c.mean() / c.std() * np.sqrt(len(c))))
r_open = rmse(o.margin, o.line_open); r_close = rmse(o.margin, o.line_close_dk)
say('2026 RMSE: open %.3f close %.3f model %.3f | totals open %.3f close %.3f model %.3f' % (r_open, r_close, rmse(o.margin, o.margin_hat),
    rmse(o.total, o.total_open), rmse(o.total, o.total_close_dk), rmse(o.total, o.total_hat)))

# ---- 6. save predictions and current ratings
keep = ['game_id', 'season', 'date', 'home', 'away', 'home_location', 'away_location', 'neutral', 'home_score', 'away_score', 'margin', 'total', 'poss',
        'O_h', 'D_h', 'O_a', 'D_a', 'P_h', 'P_a', 'mu', 'hca', 'T0', 'poss_hat', 'pts_h_hat', 'pts_a_hat', 'margin_hat', 'total_hat', 'p_win',
        'line', 'verified', 'line_cons', 'line_tr', 'total_line', 'line_open', 'total_open', 'line_close_dk', 'total_close_dk', 'p_ml', 'bpi_home', 'p_cover_home', 'p_over',
        'margin_hat_base', 'total_hat_base', 'margin_hat_all', 'abs_h', 'abs_a', 'absc_h', 'absc_a', 'sigma']
h[keep].to_parquet('data/cbb_predictions.parquet')
names = pd.concat([g[['season', 'home', 'home_location']].rename(columns={'home': 'team', 'home_location': 'name'}),
                   g[['season', 'away', 'away_location']].rename(columns={'away': 'team', 'away_location': 'name'})]).drop_duplicates(['team'], keep='last').set_index('team').name
allr = pd.concat([f.assign(season=s_) for s_, f in final.items()]).rename_axis('team').reset_index()
allr['name'] = allr.team.map(names); allr.to_csv('data/cbb_ratings_by_season.csv', index=False)
last = max(final); cur = final[last].assign(name=lambda d: d.index.map(names)).sort_values('net', ascending=False)
cur['tempo'] = cur.T0 + cur.P   # expected possessions vs an average D1 team
cur.rename_axis('team_id').reset_index()[['team_id', 'name', 'net', 'O', 'D', 'tempo', 'mu', 'hca', 'T0']].round(2).to_csv('ratings_cbb.csv', index=False)
json.dump({'params': PARAMS, 'near_home_neutral': NH, 'sigma_margin': float(np.sqrt(np.mean([v ** 2 for v in ls_[0].values()]))),
           'sigma_by_month': {str(k): round(v, 3) for k, v in ls_[0].items()}, 'sigma_pace_exp': ls_[1],
           'availability': {'k_margin': round(kk[max(kk)][0], 4), 'k_total': round(kk[max(kk)][1], 4)}, 'blend_close': cb.iloc[-1].round(4).to_dict(), 'blend_open': {k: round(v, 4) for k, v in b_open.items()}, 'season': int(last)}, open('model_config.json', 'w'), indent=1)
say('\nTop 15 end of', last); say(cur.head(15)[['name', 'net', 'O', 'D', 'tempo']].round(1).to_string())
