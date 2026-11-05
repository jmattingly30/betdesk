"""Does the NFL team card lean too hard on last season? (J 2026-10-18, PHI vs CAR.)
python3 season_weight.py <games.csv> [week ranges, e.g. 2-4,5-7,8-10]

The card's ratings carry 60-70% of last season's rating into the new one. This adds a this-season-only rating (same
filter, k 0.15, nothing carried over) to the score + market ratings and checks, leaving one season out at a time
(2012-2025), whether it predicts the final margin better by week range.
Result 2026-10-18: weeks 2-4 worse (13.35 -> 13.38), weeks 5-18 better (13.141 -> 13.114, coef +0.37, both halves of
the season). So from week 5 the card uses refit weights with this-season form: fit_current() gives them to
app/export_games.py."""
import sys, os, numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.dirname(HERE))
from common import margin_ratings, market_ratings
from current_ratings import P
CS = dict(k=0.15, hfa=2.5, carry=0.0)

def prep(path):
    n = pd.read_csv(path); n = n[n.season >= 2006].copy()
    n['date'] = pd.to_datetime(n.gameday); n = n.sort_values(['date', 'game_id']).reset_index(drop=True)
    n['home'], n['away'] = n.home_team, n.away_team; n['margin'] = n.home_score - n.away_score
    n['neutral'] = n.location.eq('Neutral'); n['line'] = n.spread_line; n['done'] = n.result.notna(); n = n[n.done].reset_index(drop=True)
    p = P['NFL']; g = margin_ratings(n, neutral='neutral', **p['rt']); gl = market_ratings(g[g.line.notna()].copy(), 'line', neutral='neutral', out='mkt', **p['mkt'])
    g = g.join(gl[['mkt_pred']])
    g['cs_pred'] = margin_ratings(n, neutral='neutral', out='cs', **CS).cs_pred
    return g

def fit_current(path, wk_from=5):
    """weights [const, score, market, this-season] fit on 2012-2025 weeks wk_from+, and each team's this-season rating now"""
    g = prep(path); m = g.season.between(2012, 2025) & g.mkt_pred.notna() & g.week.ge(wk_from)
    X = np.column_stack([np.ones(m.sum()), g.rt_pred[m], g.mkt_pred[m], g.cs_pred[m]]); b = np.linalg.lstsq(X, g.margin[m], rcond=None)[0]
    cur = g[g.season == g.season.max()]; R = {}
    for r in cur.itertuples():   # replay this season to get end ratings
        rh, ra = R.get(r.home, 0.0), R.get(r.away, 0.0); e = r.margin - (rh - ra + (0 if r.neutral else CS['hfa']))
        R[r.home] = rh + CS['k'] * e; R[r.away] = ra - CS['k'] * e
    return dict(from_week=wk_from, const=round(float(b[0]), 3), score=round(float(b[1]), 3), market=round(float(b[2]), 3), season=round(float(b[3]), 3), hfa_season=CS['hfa']), {t: round(v, 2) for t, v in R.items()}

def test(path, ranges):
    g = prep(path); m = g.season.between(2012, 2025) & g.mkt_pred.notna()
    for lo, hi in ranges:
        k = m & g.week.between(lo, hi)
        X0 = lambda q: np.column_stack([np.ones(q.sum()), g.rt_pred[q], g.mkt_pred[q]]); X1 = lambda q: np.column_stack([X0(q), g.cs_pred[q]])
        e0, e1, co = [], [], []
        for s in range(2012, 2026):   # leave one season out
            tr, te = k & (g.season != s), k & (g.season == s)
            for X, e in ((X0, e0), (X1, e1)):
                b = np.linalg.lstsq(X(tr), g.margin[tr], rcond=None)[0]; e.append(g.margin[te].values - X(te) @ b)
            co.append(np.linalg.lstsq(X1(tr), g.margin[tr], rcond=None)[0][-1])
        e0, e1 = np.concatenate(e0), np.concatenate(e1)
        print(f'weeks {lo}-{hi} + this season: RMSE {np.sqrt(np.mean(e0**2)):.3f} -> {np.sqrt(np.mean(e1**2)):.3f}  coef {np.mean(co):+.3f}  n {k.sum()}')

if __name__ == '__main__':
    rg = sys.argv[2] if len(sys.argv) > 2 else '2-4,5-7,8-10,11-18,5-18'
    test(sys.argv[1], [tuple(map(int, x.split('-'))) for x in rg.split(',')])
    print(fit_current(sys.argv[1]))
