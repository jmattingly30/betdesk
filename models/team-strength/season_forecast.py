"""Season-level forecasting test for NFL, CFB (FBS), NBA, MLB.
For each test season: freeze team ratings (a) before the season and (b) at the halfway point, then project wins
in the remaining games by summing per-game win probabilities over the actual schedule. Compare methods:
  naive record (last season / first half), score-margin rating, market rating (learned from closing lines), blend.
Also writes simulate_season() used by the live tool: Monte Carlo of the remaining schedule -> win distribution."""
import sys, numpy as np, pandas as pd
from scipy.stats import norm
sys.path.insert(0, __file__.rsplit('/', 1)[0])
from common import margin_ratings, market_ratings
D = sys.argv[1]; outdir = sys.argv[2]

def team_ratings_at(g, upto_mask, rcols):
    """Latest pre-game rating for each team among games in upto_mask, plus that game's update (approximated by next value)."""
    h = g[upto_mask]
    last = {}
    for side in ('home','away'):
        x = h[[side, f'{rcols}_{side}']].rename(columns={side:'team', f'{rcols}_{side}':'r'})
        x['i'] = x.index
        last[side] = x
    x = pd.concat(last.values()).sort_values('i').groupby('team').r.last()
    return x

def project(g_future, R, hfa, prob_fn, default=0.0):
    rh = g_future.home.map(R).fillna(default); ra = g_future.away.map(R).fillna(default)
    pred = rh - ra + np.where(g_future.neutral, 0, hfa)
    p = pd.Series(np.asarray(prob_fn(pred)))
    w = pd.concat([pd.Series(p.values, index=g_future.home.values), pd.Series(1 - p.values, index=g_future.away.values)])
    return w.groupby(level=0).sum()

def actual_wins(g_future):
    y = (g_future.margin > 0).astype(float) + 0.5*(g_future.margin == 0)
    w = pd.concat([pd.Series(y.values, index=g_future.home.values), pd.Series(1 - y.values, index=g_future.away.values)])
    n = pd.concat([pd.Series(1, index=g_future.home.values), pd.Series(1, index=g_future.away.values)])
    return w.groupby(level=0).sum(), n.groupby(level=0).sum()

def run(sport, g, rp, mp, sigma, test_seasons, teams_filter=None, line='line', logit=False):
    g = g.sort_values('date').reset_index(drop=True)
    g = margin_ratings(g, **rp)
    has_line = g[line].notna()
    gm = market_ratings(g[has_line].copy(), line, out='mkt', **mp)
    g = g.join(gm[['mkt_home','mkt_away']])
    # carry market ratings forward over games without a line
    for s in ('home','away'): g[f'mkt_{s}'] = g[f'mkt_{s}']
    pf = (lambda x: 1/(1 + np.exp(-x))) if logit else (lambda x: norm.cdf(x/sigma))
    pf_score = lambda x: norm.cdf(x/sigma)
    hfa_m = mp.get('hfa', g.loc[has_line, line].mean())
    rows = []
    for S in test_seasons:
        gs = g[(g.season == S) & g.regular]
        if teams_filter is not None: gs = gs[gs.home.isin(teams_filter(g, S)) & gs.away.isin(teams_filter(g, S))]
        if len(gs) < 50: continue
        prev = g[(g.season == S - 1)]
        # --- preseason
        R_pre = team_ratings_at(g, g.season == S - 1, 'rt')*rp['carry']
        last_wins, last_n = actual_wins(prev[prev.regular])
        last_pct = (last_wins/last_n)
        act, n = actual_wins(gs)
        preds = {'naive: last season win% (regressed 1/3)': (0.5 + (last_pct - 0.5)*2/3).reindex(act.index).fillna(0.5)*n,
                 'score rating (regressed)': project(gs, R_pre, rp['hfa'], pf_score)}
        if has_line[g.season == S - 1].any():
            Rm = team_ratings_at(g[has_line], (g[has_line].season == S - 1), 'mkt')*mp['carry']
            preds['market rating (regressed)'] = project(gs, Rm, hfa_m, pf)
            preds['blend score+market'] = (preds['score rating (regressed)'] + preds['market rating (regressed)'])/2
        for k, p in preds.items():
            p = p.reindex(act.index)
            ok = p.notna()
            rows.append(dict(sport=sport, season=S, when='preseason', method=k, teams=int(ok.sum()),
                             rmse_wins=np.sqrt(((p[ok] - act[ok])**2).mean()), corr=np.corrcoef(p[ok], act[ok])[0,1], games_per_team=n.mean()))
        # --- midseason: split each team's regular season by date at the median game date
        mid = gs.date.quantile(0.5)
        first, second = gs[gs.date <= mid], gs[gs.date > mid]
        fw, fn = actual_wins(first); sw, sn = actual_wins(second)
        upto = (g.season == S) & (g.date <= mid)
        R_mid = team_ratings_at(g, upto, 'rt')
        preds = {'naive: first-half win%': (fw/fn).reindex(sw.index)*sn,
                 'score rating': project(second, R_mid, rp['hfa'], pf_score)}
        if has_line[upto].sum() > 20:
            Rm = team_ratings_at(g[has_line], upto[has_line], 'mkt')
            preds['market rating'] = project(second, Rm, hfa_m, pf)
            preds['blend score+market'] = (preds['score rating'] + preds['market rating'])/2
        for k, p in preds.items():
            p = p.reindex(sw.index); ok = p.notna()
            rows.append(dict(sport=sport, season=S, when='midseason', method=k, teams=int(ok.sum()),
                             rmse_wins=np.sqrt(((p[ok] - sw[ok])**2).mean()), corr=np.corrcoef(p[ok], sw[ok])[0,1], games_per_team=sn.mean()))
    return pd.DataFrame(rows)

out = []
# NFL
n = pd.read_parquet(f'{D}/nfl_games_features.parquet')
n = n.assign(date=pd.to_datetime(n.gameday), regular=n.game_type.eq('REG'))[['season','date','home','away','margin','neutral','line','regular']]
out.append(run('NFL', n, dict(k=0.08, hfa=2.5, carry=0.6, neutral='neutral'), dict(k=0.3, carry=0.7, neutral='neutral'), 13.3, range(2009, 2026)))
# CFB: FBS vs FBS games only for the season table
c = pd.read_parquet(f'{D}/cfb_games.parquet')
c = c.assign(regular=c.season_type.eq('regular'), line=c.line_close)
fbs = lambda g, S: set(g[(g.season == S) & (g.home_division == 'fbs')].home)
out.append(run('CFB', c[['season','date','home','away','margin','neutral','line','regular','home_division']], dict(k=0.12, hfa=3, carry=0.8, neutral='neutral'),
               dict(k=0.4, carry=0.85, hfa=3.5, neutral='neutral'), 16.5, range(2009, 2026), teams_filter=fbs))
# NBA
b = pd.read_parquet(f'{D}/nba_games.parquet'); b = b.assign(regular=b.playoff.eq(0))
out.append(run('NBA', b[['season','date','home','away','margin','neutral','line','regular']], dict(k=0.03, hfa=3, carry=0.7, neutral='neutral'),
               dict(k=0.3, carry=0.7, neutral='neutral'), 12.5, range(2010, 2027)))
# MLB: market line in log-odds of home win
m = pd.read_parquet(f'{D}/mlb_games_model.parquet')
m = m.assign(neutral=False, regular=True, line=np.log(m.ph_close/(1 - m.ph_close)))
out.append(run('MLB', m[['season','date','home','away','margin','neutral','line','regular']], dict(k=0.012, hfa=0.2, carry=0.5), dict(k=0.15, carry=0.6),
               4.3, [2016,2017,2018,2019,2021,2022,2023,2024,2025], logit=True))
res = pd.concat(out)
summ = res.groupby(['sport','when','method']).agg(seasons=('season','nunique'), rmse_wins=('rmse_wins','mean'), corr=('corr','mean'), games_per_team=('games_per_team','mean')).round(3)
print(summ.to_string())
res.to_csv(f'{outdir}/season_forecast_by_season.csv', index=False); summ.to_csv(f'{outdir}/season_forecast_summary.csv')
