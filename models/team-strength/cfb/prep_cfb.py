"""Build one row per CFB game (2006-2025): score, venue, CFBD pregame Elo, and consensus opening and closing
spread and total from the cfbfastR-data betting file (CFBD lines, many books). Home line is expressed as
expected HOME margin (positive = home favoured)."""
import sys, glob, numpy as np, pandas as pd
odds_pq, sched_dir, out = sys.argv[1:4]
s = pd.concat([pd.read_csv(f) for f in sorted(glob.glob(f'{sched_dir}/sched_*.csv'))], ignore_index=True)
s = s[s.completed & s.home_points.notna()].copy()
o = pd.read_parquet(odds_pq)
o = o[o.market_type.isin(['spread','total'])]
# map betting abbr -> team id: the id common to every game the abbr appears in
sp = o[o.market_type == 'spread'][['game_id','abbr','home_team_id','away_team_id']].drop_duplicates()
cand = {}
for ab, grp in sp.groupby('abbr'):
    ids = pd.concat([grp.home_team_id, grp.away_team_id]).value_counts()
    if ids.empty: continue
    cand[ab] = ids.index[0] if ids.iloc[0] >= 0.8*grp.game_id.nunique() else np.nan
sp_rows = o[o.market_type == 'spread'].copy()
sp_rows['team_id'] = sp_rows.abbr.map(cand)
sp_rows = sp_rows[sp_rows.team_id == sp_rows.home_team_id]          # keep the home team's side
sp_rows['home_close'] = -sp_rows.lines; sp_rows['home_open'] = -sp_rows.opening_lines
sp_rows.loc[sp_rows.opening_lines.eq(0) & sp_rows.opening_odds.isna(), 'home_open'] = np.nan
tot = o[(o.market_type == 'total') & (o.abbr == 'over')]
G = pd.DataFrame({
    'line_close': sp_rows.groupby('game_id').home_close.median(),
    'line_open': sp_rows.groupby('game_id').home_open.median(),
    'n_books': sp_rows.groupby('game_id').home_close.count(),
    'pinnacle_close': sp_rows[sp_rows.book.eq('PINNACLE')].groupby('game_id').home_close.median(),
    'total_close': tot.groupby('game_id').lines.median(),
    'total_open': tot.groupby('game_id').opening_lines.median(),
})
s = s.merge(G, left_on='game_id', right_index=True, how='left')
s['margin'] = s.home_points - s.away_points; s['total'] = s.home_points + s.away_points
s['home'], s['away'] = s.home_team, s.away_team
s['neutral'] = s.neutral_site.astype(bool)
s['date'] = pd.to_datetime(s.start_date)
s = s.sort_values('date').reset_index(drop=True)
s.to_parquet(out)
print(s.shape)
print(s.groupby('season').agg(games=('game_id','size'), close=('line_close', lambda x: x.notna().sum()), open=('line_open', lambda x: x.notna().sum()),
      fbs_fbs=('home_division', lambda x: 0)).T.to_string())
# sanity: closing line should track the margin with slope ~1
m = s.line_close.notna(); print('corr(close, margin)', np.corrcoef(s.line_close[m], s.margin[m])[0,1], 'MAE', (s.margin[m]-s.line_close[m]).abs().mean())
m = s.line_open.notna(); print('corr(open, margin)', np.corrcoef(s.line_open[m], s.margin[m])[0,1], 'MAE open', (s.margin[m]-s.line_open[m]).abs().mean(), 'MAE close same games', (s.margin[m]-s.line_close[m]).abs().mean())
