"""NFL matchup model, 2006-2025, judged against the closing spread.
1. Point-in-time team ratings: score margin, EPA margin, pass/rush offense & defense EPA, market (closing-line) ratings.
2. Walk-forward stack: margin ~ closing line + (model - line) + situational features; trained on prior seasons only.
3. Tests: does anything add information beyond the closing line? Do style matchups (pass vs rush) matter?

2026-10-09 audit (T2): QB term. Each starter's value = shrunk EPA per play (dropbacks + QB runs) from nflverse weekly player
stats, walk-forward (decayed by game, carried across seasons, shrunk toward backup level). Features: qb_val_diff (home
starter minus away starter) and qb_delta_diff (each starter vs his team's recent starters, which the team ratings already
contain: this is the backup-QB effect). 'playoff' was constant in the fits (EPA features exist for regular season only),
which made every design matrix rank deficient; it is dropped.
usage: python3 nfl/game_model.py data/nfl_games_nflverse.csv data/nfl_team_games.csv nfl data/nfl_games_features.parquet <nflverse player_stats dir>"""
import sys, numpy as np, pandas as pd, statsmodels.api as sm
sys.path.insert(0, __file__.rsplit('/', 2)[0])
from common import *
games_csv, tg_csv, outdir = sys.argv[1:4]
g = pd.read_csv(games_csv)
g = g[(g.season >= 2006) & g.result.notna()].copy()
g['margin'] = g.home_score - g.away_score
g['line'] = g.spread_line                     # expected home margin at close
g['neutral'] = g.location.eq('Neutral')
g['home'], g['away'] = g.home_team, g.away_team
g = g.sort_values(['gameday','gametime','game_id']).reset_index(drop=True)
TRAIN = g.season <= 2013
# --- 1a score-margin rating (tuned on 2006-2013)
best = tune(g, lambda d, **p: margin_ratings(d, neutral='neutral', **p), grid(k=[0.05,0.08,0.1,0.12,0.15], hfa=[1.5,2,2.5], carry=[0.4,0.5,0.6,0.7]),
            mask=lambda h: (h.season <= 2013) & (h.season > 2006))
print('score rating best', best)
g = margin_ratings(g, neutral='neutral', **best[1])
# --- 1b market rating: learns from closing lines; predicts next game's line
bm = tune(g, lambda d, **p: market_ratings(d, 'line', neutral='neutral', out='mkt', **p), grid(k=[0.2,0.3,0.4,0.5], carry=[0.5,0.7,0.8,0.9]),
          target='line', pred_col='mkt_pred', mask=lambda h: (h.season <= 2013) & (h.season > 2006))
print('market rating best', bm)
g = market_ratings(g, 'line', neutral='neutral', out='mkt', **bm[1])
# --- 1c EPA ratings from team-game table
tg = pd.read_csv(tg_csv)
key = g[['season','week','home','away','game_id','gameday','gametime']].rename(columns={'game_id':'gid'})
tg = tg.merge(key, on=['season','week','home','away'], how='inner').sort_values(['gameday','gametime','gid','team']).reset_index(drop=True)
tg['game_id'] = tg.gid
comp = {}
for stat, k in [('epa',0.06), ('pass_epa',0.06), ('rush_epa',0.05), ('sr',0.06), ('ng_epa',0.06)]:
    r = od_ratings(tg.assign(value=tg[stat]), k=k, carry=0.5, out=stat)
    comp[stat+'_O'] = r[stat+'_O'].values; comp[stat+'_Dopp'] = r[stat+'_Dopp'].values
for c, v in comp.items(): tg[c] = v
side = tg.set_index(['gid','team'])
def pick(team_col, cols):
    idx = pd.MultiIndex.from_arrays([g.game_id, g[team_col]])
    return side.reindex(idx)[cols].values
cols = list(comp)
H = pd.DataFrame(pick('home_team', cols), columns=['h_'+c for c in cols]); A = pd.DataFrame(pick('away_team', cols), columns=['a_'+c for c in cols])
g = pd.concat([g, H, A], axis=1)
# expected EPA/play edge for home: (home offense vs away defense) - (away offense vs home defense)
for s in ['epa','pass_epa','rush_epa','sr','ng_epa']:
    g[f'{s}_edge'] = (g[f'h_{s}_O'] + g[f'h_{s}_Dopp']) - (g[f'a_{s}_O'] + g[f'a_{s}_Dopp'])
# offense-only and defense-only edges (offense is more stable; test weighting)
g['off_edge'] = g.h_epa_O - g.a_epa_O
g['def_edge'] = g.h_epa_Dopp - g.a_epa_Dopp      # home team's opponent-faced D... (h_*_Dopp is the AWAY defense faced by home)
# style leverage: how much better is a team's best phase vs the opponent than its average
hp = g.h_pass_epa_O + g.h_pass_epa_Dopp; hr = g.h_rush_epa_O + g.h_rush_epa_Dopp
ap = g.a_pass_epa_O + g.a_pass_epa_Dopp; ar = g.a_rush_epa_O + g.a_rush_epa_Dopp
g['style_leverage'] = (np.maximum(hp, hr) - (hp + hr)/2) - (np.maximum(ap, ar) - (ap + ar)/2)
g['pass_matchup_x'] = (g.h_pass_epa_O * g.h_pass_epa_Dopp) - (g.a_pass_epa_O * g.a_pass_epa_Dopp)
# --- situational features
g['rest_diff'] = (g.home_rest - g.away_rest).clip(-7, 7)
g['home_off_bye'] = (g.home_rest >= 13).astype(int) - (g.away_rest >= 13).astype(int)
g['short_week'] = (g.home_rest <= 5).astype(int) - (g.away_rest <= 5).astype(int)
g['div'] = g.div_game
g['wind'] = g.wind.fillna(0); g['dome'] = g.roof.isin(['dome','closed']).astype(int)
g['late_season'] = (g.week >= 13).astype(int)
long = pd.concat([g[['game_id','gameday','season','home_team','home_qb_id']].set_axis(['gid','d','s','team','qb'], axis=1),
                  g[['game_id','gameday','season','away_team','away_qb_id']].set_axis(['gid','d','s','team','qb'], axis=1)]).sort_values('d')
long['qb_change'] = (long.groupby(['s','team']).qb.shift() != long.qb) & long.groupby(['s','team']).cumcount().gt(0)
qc = long.set_index(['gid','team']).qb_change
g['qb_change_diff'] = qc.reindex(pd.MultiIndex.from_arrays([g.game_id, g.home_team])).values.astype(float) - qc.reindex(pd.MultiIndex.from_arrays([g.game_id, g.away_team])).values.astype(float)
g['playoff'] = (g.game_type != 'REG').astype(int)
# --- QB value (T2)
psd = sys.argv[5] if len(sys.argv) > 5 else None
g['qb_val_diff'] = 0.0; g['qb_delta_diff'] = 0.0
if psd:
    import os
    ps = pd.read_parquet(f'{psd}/player_stats.parquet').rename(columns={'recent_team': 'team', 'sacks': 'sk'})
    if os.path.exists(f'{psd}/stats_player_week_2025.parquet'):
        p25 = pd.read_parquet(f'{psd}/stats_player_week_2025.parquet').rename(columns={'sacks_suffered': 'sk'})
        ps = pd.concat([ps[ps.season < 2025], p25], ignore_index=True)
    ps = ps[ps.attempts.fillna(0) + ps.carries.fillna(0) > 0]
    ps['plays'] = ps.attempts.fillna(0) + ps.sk.fillna(0) + np.where(ps.position == 'QB', ps.carries.fillna(0), 0)
    ps['epa'] = ps.passing_epa.fillna(0) + np.where(ps.position == 'QB', ps.rushing_epa.fillna(0), 0)
    ps = ps[(ps.position == 'QB') & (ps.plays > 0)]
    wk = {(a, int(b), int(c)): (e, n) for a, b, c, e, n in zip(ps.player_id, ps.season, ps.week, ps.epa, ps.plays)}
    # backup level: EPA per play of QBs with under 100 plays in a season (2006-2012 only, so it is fixed before testing)
    sea = ps[ps.season.between(2006, 2012)].groupby(['player_id', 'season'])[['epa', 'plays']].sum()
    REPL = float(sea[sea.plays < 100].epa.sum() / sea[sea.plays < 100].plays.sum())
    KQ, DQ, CQ = 200.0, 0.5 ** (1 / 16), 0.75      # prior plays, per-game decay (half-life 16 starts), season carry
    st_ = {}; tb = {}; vh, va, bh, ba = (np.full(len(g), np.nan) for _ in range(4))
    for i, r in enumerate(g.itertuples()):
        for side, q, team in (('h', r.home_qb_id, r.home_team), ('a', r.away_qb_id, r.away_team)):
            if not isinstance(q, str): continue
            S_, N_, ls = st_.get(q, (0.0, 0.0, r.season))
            if ls != r.season: S_ *= CQ; N_ *= CQ
            v = (S_ + KQ * REPL) / (N_ + KQ); base = tb.get(team, v)
            if side == 'h': vh[i], bh[i] = v, base
            else: va[i], ba[i] = v, base
            e, n = wk.get((q, int(r.season), int(r.week)), (0.0, 0.0))
            st_[q] = (S_ * DQ + e, N_ * DQ + n, r.season)
            tb[team] = 0.85 * base + 0.15 * v      # team's recent-starter level (what the team ratings already contain)
    g['qb_val_diff'] = np.nan_to_num(vh - va); g['qb_delta_diff'] = np.nan_to_num((vh - bh) - (va - ba))
    print('QB term: backup level %.3f EPA/play; sd of qb_val_diff %.3f, qb_delta_diff %.3f' % (REPL, g.qb_val_diff.std(), g.qb_delta_diff.std()))
g['early'] = (g.week <= 4).astype(int)
# --- convert ratings to points on training seasons (OLS margin ~ ratings), walk-forward stack vs the line
FEATS_MODEL = ['rt_pred', 'mkt_pred', 'epa_edge', 'pass_epa_edge', 'rush_epa_edge', 'sr_edge', 'ng_epa_edge']
SITU = ['rest_diff','home_off_bye','short_week','div','qb_change_diff','late_season']
QB = ['qb_val_diff', 'qb_delta_diff']
g = g[g.season >= 2007].reset_index(drop=True)
def wf(feats, base='line', start=2012, data=g):
    preds = pd.Series(np.nan, index=data.index)
    for s in range(start, 2026):
        tr, te = data[data.season < s], data[data.season == s]
        X = sm.add_constant(tr[[base] + feats] if base else tr[feats]).dropna(); y = tr.loc[X.index, 'margin']
        f = sm.OLS(y, X.astype(float)).fit()
        Xt = sm.add_constant(te[[base] + feats] if base else te[feats], has_constant='add')
        preds[te.index] = f.predict(Xt)
    return preds
T = g.season >= 2012
out = []
def report(name, p):
    m = T & p.notna()
    mae = (g.margin[m] - p[m]).abs().mean(); rmse = np.sqrt(((g.margin[m] - p[m])**2).mean())
    out.append(dict(model=name, n=int(m.sum()), mae=mae, rmse=rmse))
report('closing line', g.line)
report('score rating only', wf(['rt_pred'], base=None))
report('EPA rating only', wf(['epa_edge'], base=None))
report('score + EPA ratings', wf(['rt_pred','epa_edge'], base=None))
report('pass/rush/SR ratings', wf(['rt_pred','pass_epa_edge','rush_epa_edge','sr_edge'], base=None))
report('market rating (prior lines) only', wf(['mkt_pred'], base=None))
report('all ratings, no line', wf(FEATS_MODEL, base=None))
p_stack0 = wf(FEATS_MODEL + SITU)
report('line + all ratings + situational (no QB term)', p_stack0)
report('all ratings + QB term, no line', wf(FEATS_MODEL + QB, base=None))
p_stack = wf(FEATS_MODEL + SITU + QB)
report('line + all ratings + situational + QB term', p_stack)
report('line + situational only', wf(SITU))
res = pd.DataFrame(out); print(res.round(3).to_string(index=False))
# out-of-sample information test: regress (margin - line) on (each feature), 2012-2025, using walk-forward predicted residuals
gT = g[T].copy(); gT['resid'] = gT.margin - gT.line
print('\nWhich features explain the closing-line miss (2012-2025)? OLS of margin-line on each, HC1 errors')
rows = []
for f in FEATS_MODEL + SITU + QB + ['style_leverage','pass_matchup_x','off_edge','def_edge','wind','dome']:
    x = gT[f] - (gT.line if f in ('rt_pred','mkt_pred') else 0) if f in ('rt_pred','mkt_pred') else gT[f]
    d = pd.DataFrame({'x': x, 'y': gT.resid}).dropna()
    fit = sm.OLS(d.y, sm.add_constant(d.x)).fit(cov_type='HC1')
    rows.append(dict(feature=f + (' - line' if f in ('rt_pred','mkt_pred') else ''), coef=fit.params.x, t=fit.tvalues.x, p=fit.pvalues.x, n=len(d)))
inf = pd.DataFrame(rows).sort_values('p'); print(inf.round(4).to_string(index=False))
# ATS from the stack when it disagrees with the line
gT['stack'] = p_stack[T]; gT['edge'] = gT['stack'] - gT.line
print('\nATS when walk-forward stack disagrees with the close:')
ats = []
for thr in [0.5, 1, 1.5, 2, 3]:
    m = (gT.edge.abs() >= thr).values
    w, l, pu = ats_record(gT.edge[m] > 0, gT.margin[m].values, gT.line[m].values)
    ats.append(dict(threshold=thr, bets=w+l, wins=w, win_pct=w/max(w+l,1), p_vs_50=binom_p(w, w+l), p_vs_52_4=binom_p(w, w+l, 0.524)))
print(pd.DataFrame(ats).round(4).to_string(index=False))
# win probability: model vs line-implied (sigma fit on train)
sig = (g[~T].margin - g[~T].line).std()
y = (gT.margin > 0).astype(float); keep = gT.margin != 0
for name, pr in [('closing line', gT.line), ('stack', gT['stack']), ('ratings only', wf(FEATS_MODEL, base=None)[T])]:
    p = win_prob_from_margin(pr[keep], sig)
    print(f'win prob {name:14s} logloss {logloss(p, y[keep]):.4f} brier {brier(p, y[keep]):.4f}')
res.to_csv(f'{outdir}/accuracy_vs_close.csv', index=False); inf.to_csv(f'{outdir}/features_vs_close_residual.csv', index=False)
pd.DataFrame(ats).to_csv(f'{outdir}/ats_stack.csv', index=False)
if len(sys.argv) > 4: g.to_parquet(sys.argv[4])
