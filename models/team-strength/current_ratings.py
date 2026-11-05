"""Current team ratings and a matchup predictor (NFL, CFB, NBA) built from the validated rating filters.

  python3 current_ratings.py <data dir> <out dir>            -> writes ratings_<sport>.csv and blend_weights.json
  python3 current_ratings.py --predict NFL KC BUF [--neutral] [--line -2.5]

Rating = expected margin vs an average team on a neutral field. Blend = OLS weights of score rating and market rating
fitted on all past games. If you pass the current line, the output also shows the line-anchored estimate (what the
backtests say to trust most): the market already contains almost everything public."""
import sys, json, os, numpy as np, pandas as pd
from scipy.stats import norm
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import margin_ratings, market_ratings
HERE = os.path.dirname(os.path.abspath(__file__))
P = {  # tuned in nfl/cfb/nba game_model.py (training seasons only)
 'NFL': dict(rt=dict(k=0.08, hfa=2.5, carry=0.6), mkt=dict(k=0.3, carry=0.7), sigma=13.3, line_weight=0.92),
 'CFB': dict(rt=dict(k=0.12, hfa=3.0, carry=0.8), mkt=dict(k=0.4, carry=0.85, hfa=3.5), sigma=16.5, line_weight=0.95),
 'NBA': dict(rt=dict(k=0.03, hfa=3.0, carry=0.7), mkt=dict(k=0.3, carry=0.7), sigma=12.5, line_weight=0.97),
}
def final_ratings(g, prefix, k, margin='margin'):
    """Ratings after the last game (applies that game's update)."""
    R = {}
    last = pd.concat([g[['date','home',f'{prefix}_home']].set_axis(['d','team','r'], axis=1).assign(side=1, i=g.index),
                      g[['date','away',f'{prefix}_away']].set_axis(['d','team','r'], axis=1).assign(side=-1, i=g.index)]).sort_values('i')
    lastrow = last.groupby('team').tail(1)
    out = {}
    for _, r in lastrow.iterrows():
        row = g.loc[r.i]
        err = row[margin] - row[f'{prefix}_pred'] if pd.notna(row[margin]) else 0.0
        out[r.team] = r.r + r.side*k*err
    return pd.Series(out)
def build(D, outdir):
    n = pd.read_csv(f'{D}/nfl/games.csv'); n = n[n.season >= 2006].copy()
    n['date'] = pd.to_datetime(n.gameday); n['home'], n['away'] = n.home_team, n.away_team; n['margin'] = n.home_score - n.away_score
    n['neutral'] = n.location.eq('Neutral'); n['line'] = n.spread_line; n['done'] = n.result.notna()
    c = pd.read_parquet(f'{D}/cfb_games.parquet'); c['line'] = c.line_close; c['done'] = True
    b = pd.read_parquet(f'{D}/nba_games.parquet'); b['done'] = True
    weights = {}
    for sport, g in (('NFL', n), ('CFB', c), ('NBA', b)):
        p = P[sport]
        g = g[g.done].sort_values('date').reset_index(drop=True)
        g = margin_ratings(g, neutral='neutral', **p['rt'])
        gl = market_ratings(g[g.line.notna()].copy(), 'line', neutral='neutral', out='mkt', **p['mkt'])
        g = g.join(gl[['mkt_pred','mkt_home','mkt_away']])
        both = g[g.mkt_pred.notna() & g.rt_pred.notna()]
        X = np.column_stack([np.ones(len(both)), both.rt_pred, both.mkt_pred]); w = np.linalg.lstsq(X, both.margin, rcond=None)[0]
        weights[sport] = dict(const=w[0], score=w[1], market=w[2], hfa_score=p['rt']['hfa'], hfa_market=float(p['mkt'].get('hfa', gl.line.mean())), sigma=p['sigma'],
                              line_weight=p['line_weight'], last_game=str(g.date.max().date()))
        rs = final_ratings(g, 'rt', p['rt']['k'])
        gm = gl.reset_index(drop=True); rm = final_ratings(gm, 'mkt', p['mkt']['k'], margin='line')
        cur_season = g.season.max()
        active = set(g[g.season == cur_season].home) | set(g[g.season == cur_season].away)
        if sport == 'NBA': active = set(g[(g.season == cur_season) & (g.playoff == 0)].home)
        if sport == 'CFB': active = set(g[(g.season == cur_season) & (g.home_division == 'fbs')].home)
        # a new season that has not started yet gets the between-season regression
        R = pd.DataFrame({'score_rating': rs, 'market_rating': rm}).loc[lambda d: d.index.isin(active)]
        # a rating from a finished season is regressed toward the mean before it is used for the next one
        if gm.season.max() < cur_season: R['market_rating'] *= p['mkt']['carry']
        if sport == 'NBA' and g.date.max().month in (6, 7, 8, 9, 10):
            R['score_rating'] *= p['rt']['carry']; R['market_rating'] *= p['mkt']['carry']
        R['market_rating'] = R.market_rating - R.market_rating.mean(); R['score_rating'] = R.score_rating - R.score_rating.mean()
        R['blended_rating'] = w[1]*R.score_rating + w[2]*R.market_rating.fillna(R.score_rating)
        R = R.sort_values('blended_rating', ascending=False).round(2)
        R.to_csv(f'{outdir}/ratings_{sport.lower()}.csv', index_label='team')
        print(sport, 'through', g.date.max().date(), 'teams', len(R)); print(R.head(8).to_string())
    json.dump(weights, open(f'{outdir}/blend_weights.json', 'w'), indent=1)
def predict(sport, home, away, neutral=False, line=None):
    w = json.load(open(f'{HERE}/blend_weights.json'))[sport]
    R = pd.read_csv(f'{HERE}/ratings_{sport.lower()}.csv', index_col='team')
    rh, ra = R.loc[home], R.loc[away]
    s = rh.score_rating - ra.score_rating + (0 if neutral else w['hfa_score'])
    m = (rh.market_rating - ra.market_rating) + (0 if neutral else w['hfa_market'])
    est = w['const']*(not neutral) + w['score']*s + w['market']*(m if pd.notna(m) else s)
    out = {'sport': sport, 'home': home, 'away': away, 'model_margin_home': round(est, 1), 'model_home_win_prob': round(norm.cdf(est/w['sigma']), 3),
           'fair_home_spread': round(-est, 1)}
    if line is not None:
        anchored = w['line_weight']*(-line) + (1 - w['line_weight'])*est
        out.update(line_home=line, line_anchored_margin=round(anchored, 1), line_anchored_win_prob=round(norm.cdf(anchored/w['sigma']), 3),
                   model_minus_market_pts=round(est + line, 1),
                   note='Backtests: disagreements with the CLOSING line are mostly noise; check against the OPENING line and track CLV before betting.')
    return out
if __name__ == '__main__':
    if sys.argv[1] == '--predict':
        a = sys.argv[2:]; line = float(a[a.index('--line') + 1]) if '--line' in a else None
        print(json.dumps(predict(a[0], a[1], a[2], '--neutral' in a, line), indent=1))
    else:
        build(sys.argv[1], sys.argv[2])
