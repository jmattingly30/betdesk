"""US moneyline devig + consensus test on Kaggle odds archives (pulled 2026-10-31).
python3 us_ml_test.py -> results/us_ml_devig.csv, results/us_ml_consensus.csv
  CBB  archangel0x01 closing moneylines 2009-10 to 2025-26, ~15 books per game (no Pinnacle; BetInAsia = sharp Asian broker)
  NBA  cviaxmiwnptr NBA moneylines 2007-08 to 2025-26 (one consensus line per game)"""
import os, sys, math, warnings; warnings.filterwarnings('ignore')
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import devig as D
from devig_test import score, M
HERE = os.path.dirname(os.path.abspath(__file__)); K = f'{HERE}/data/kaggle'; OUT = f'{HERE}/results'
rows = []
g = pd.read_parquet(f'{K}/nba-nhl-and-college-basketball-results-and-odds/games.parquet')
o = pd.read_parquet(f'{K}/nba-nhl-and-college-basketball-results-and-odds/closing_odds.parquet')
o = o[(o.market == 'moneyline') & (o.line_active.astype(str) != 'False')].merge(g, on='game_id')
o = o[(o.competition == 'NCAA Men') & (o.home_score != o.away_score)]
o['y'] = (o.home_score > o.away_score).astype(int)
for bk in ['bet365', '888sport', 'Betsson', 'bet-at-home', '1xBet', '22Bet', 'BetInAsia']:
    b = o[o.bookmaker == bk]
    r = score(f'CBB ML {bk} close', b.home_odds.values.astype(float), b.away_odds.values.astype(float), b.y.values, b.date.astype(str).values)
    if r: r.update(sport='CBB', book=bk); rows.append(r)
n = pd.read_csv(f'{K}/nba-betting-data-october-2007-to-june-2024/nba_2008-2026.csv')
n = n[n.moneyline_home.notna() & n.moneyline_away.notna() & (n.moneyline_home.abs() >= 100) & (n.moneyline_away.abs() >= 100)]
r = score('NBA ML 2008-26', n.moneyline_home.map(D.dec_from_am).values, n.moneyline_away.map(D.dec_from_am).values,
          (n.score_home > n.score_away).astype(int).values, n.date.astype(str).values)
if r: r.update(sport='NBA', book='consensus'); rows.append(r)
T = pd.DataFrame(rows); T.to_csv(f'{OUT}/us_ml_devig.csv', index=False)
pd.set_option('display.width', 250)
T['mult-power'] = T['mult'] - T['power']; T['z'] = T['mult-power'] / T['mult_vs_power_se']
print(T[['set', 'n', 'hold', 'side1_won', 'side1_power'] + M + ['mult-power', 'z', 'b', 'power_test', 'cal_test']].round(5).to_string(index=False))

# consensus across books (power devig each, log-odds mean): equal vs capped sharp (BetInAsia 1.5) vs best single book
w = o.assign(p=[D.power([a, b])[0] if a > 1 and b > 1 else np.nan for a, b in zip(o.home_odds, o.away_odds)])
P = w.pivot_table(index='game_id', columns='bookmaker', values='p', aggfunc='first'); y = w.groupby('game_id').y.first()
def ll(p, yy): p = np.clip(p, 1e-6, 1 - 1e-6); return float(-(yy * np.log(p) + (1 - yy) * np.log(1 - p)).mean())
books = ['bet365', '888sport', 'Betsson', 'bet-at-home', '1xBet', 'BetInAsia']
sub = P[books].dropna(); yy = y.loc[sub.index].values; L = np.log(sub.values / (1 - sub.values))
c = dict(n=len(sub), **{b: ll(sub[b].values, yy) for b in books})
c['equal'] = ll(1 / (1 + np.exp(-L.mean(1))), yy)
ws = np.array([1.5 if b == 'BetInAsia' else 1.0 for b in books]); c['capped_sharp'] = ll(1 / (1 + np.exp(-(L @ ws) / ws.sum())), yy)
print('\nCBB consensus close:', {k: round(v, 5) for k, v in c.items()})
pd.DataFrame([c]).to_csv(f'{OUT}/us_ml_consensus.csv', index=False)
