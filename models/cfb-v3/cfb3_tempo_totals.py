"""CFB-3 (2026-10-10, Edge research thread): tempo x efficiency game totals, as an accuracy check on v3's total.

total = sum over both offenses of expected plays x expected points per play, where
  expected plays  = league + (offense's plays per game - league) + (plays the defense allows - league)
  points per play = league + (offense's points per play - league) + (defense's points per play allowed - league)
each team number a pre-game, exponentially weighted average (this season weight 1 per game with half-life 6 games, last
season's games at 0.3), shrunk toward league with 4 games of prior. Walk-forward, nothing from the game itself.
Calibrated on 2014-2021 (total = a + b x tempo total), judged on 2022-2025 against v3's total (sum of projected team
scores) and the opening/closing totals.

usage: python3 cfb3_tempo_totals.py <cfb_games.parquet> <cfb_team_games_epa.csv> <v3 game_predictions.parquet> <log.txt>"""
import sys, numpy as np, pandas as pd
import statsmodels.api as sm
games_pq, epa_csv, v3_pq, out = sys.argv[1:5]
G = pd.read_parquet(games_pq)
G = G[G.home_points.notna() & G.away_points.notna() & (G.season >= 2013)].copy()
G['date'] = pd.to_datetime(G.date, utc=True); G = G.sort_values(['date', 'game_id'])
E = pd.read_csv(epa_csv)[['game_id', 'pos_team_id', 'def_pos_team_id', 'plays']]
# one row per offense-game
rows = pd.concat([G[['game_id', 'season', 'date', 'home_id', 'away_id', 'home_points']].set_axis(['game_id', 'season', 'date', 'off', 'def', 'pts'], axis=1),
                  G[['game_id', 'season', 'date', 'away_id', 'home_id', 'away_points']].set_axis(['game_id', 'season', 'date', 'off', 'def', 'pts'], axis=1)])
rows = rows.merge(E.rename(columns={'pos_team_id': 'off', 'def_pos_team_id': 'def'}), on=['game_id', 'off', 'def'], how='inner')
rows = rows[rows.plays >= 30].sort_values(['date', 'game_id']).reset_index(drop=True)
rows['ppp'] = rows.pts / rows.plays

HALF, PREV_W, K = 6, 0.3, 4.0
dec = 0.5 ** (1 / HALF)
lg = rows.groupby('season')[['plays', 'ppp']].mean()

def pregame(by, cols):
    """pre-game weighted means of cols for each row's `by` team, shrunk to the league of that season."""
    out = np.full((len(rows), len(cols)), np.nan)
    state = {}
    for i, (t, s) in enumerate(zip(rows[by], rows.season)):
        st = state.get(t)
        if st is not None and st['s'] != s:
            st = {'s': s, 'sum': st['sum'] * PREV_W, 'n': st['n'] * PREV_W}
        prior = lg.loc[s - 1, cols].to_numpy() if (s - 1) in lg.index else lg.loc[s, cols].to_numpy()
        if st is None: st = {'s': s, 'sum': np.zeros(len(cols)), 'n': 0.0}
        out[i] = (st['sum'] + K * prior) / (st['n'] + K)
        st = {'s': s, 'sum': st['sum'] * dec + rows.loc[i, cols].to_numpy(float), 'n': st['n'] * dec + 1}
        state[t] = st
    return pd.DataFrame(out, columns=cols, index=rows.index)

o = pregame('off', ['plays', 'ppp']); d = pregame('def', ['plays', 'ppp'])
L = lg.shift(1).reindex(rows.season).fillna(lg.reindex(rows.season)).set_index(rows.index)
rows['x_plays'] = L.plays + (o.plays - L.plays) + (d.plays - L.plays)
rows['x_ppp'] = (L.ppp + (o.ppp - L.ppp) + (d.ppp - L.ppp)).clip(lower=0.05)
rows['x_pts'] = rows.x_plays * rows.x_ppp
tempo = rows.groupby('game_id').agg(tempo_total=('x_pts', 'sum'), x_plays=('x_plays', 'sum'), n=('x_pts', 'size'))
tempo = tempo[tempo.n == 2]

V = pd.read_parquet(v3_pq).merge(tempo, left_on='game_id', right_index=True)
TR = V.season.between(2014, 2021) & (V.season != 2020); TE = V.season.between(2022, 2025)
log = []
def say(*a): s = ' '.join(str(x) for x in a); print(s); log.append(s)
c = sm.OLS(V.total[TR], sm.add_constant(V.tempo_total[TR])).fit(); V['cfb3_total'] = c.params.iloc[0] + c.params.iloc[1] * V.tempo_total
both = sm.OLS(V.total[TR], sm.add_constant(V[['model_total', 'tempo_total']][TR])).fit()
V['combo_total'] = both.predict(sm.add_constant(V[['model_total', 'tempo_total']]))
say('calibration (2014-21): total = %.2f + %.3f x tempo total' % tuple(c.params))
say('combined (2014-21): total =', {k: round(v, 3) for k, v in both.params.items()})
H = V[TE & V.total_open.notna() & V.total_close.notna() & ((V.total_open - V.total_close).abs() <= 10)]
say(f'\nHOLDOUT 2022-2025, {len(H)} games with opening and closing totals: total MAE')
for lab, col in [('opening total', 'total_open'), ('closing total', 'total_close'), ('v3 (team A + team B scores)', 'model_total'),
                 ('CFB-3 tempo x efficiency', 'cfb3_total'), ('v3 + CFB-3 combined', 'combo_total')]:
    say(f'  {lab:32s} {(H.total - H[col]).abs().mean():.3f}')
for lab, col in [('v3', 'model_total'), ('CFB-3', 'cfb3_total'), ('combined', 'combo_total')]:
    g = H[col] - H.total_open
    f_o = sm.OLS(H.total - H.total_open, sm.add_constant(g)).fit(cov_type='HC1')
    f_c = sm.OLS(H.total - H.total_close, sm.add_constant(H[col] - H.total_close)).fit(cov_type='HC1')
    clv = ((H.total_close - H.total_open) * np.sign(g))[g.abs() >= 3]
    res = ((H.total - H.total_open) * np.sign(g))[g.abs() >= 3]; w, l = int((res > 0).sum()), int((res < 0).sum())
    say(f'  {lab}: gap vs opener predicts the miss, coef {f_o.params.iloc[1]:.3f} (p {f_o.pvalues.iloc[1]:.3f}); vs close {f_c.params.iloc[1]:.3f} (p {f_c.pvalues.iloc[1]:.3f}); '
        f'3+ pt gaps vs opener {w}-{l} ({w / max(w + l, 1):.3f}), CLV {clv.mean():+.2f} pts (t {clv.mean() / clv.std() * np.sqrt(len(clv)):.1f})')
open(out, 'w').write('\n'.join(log) + '\n')
V[['game_id', 'tempo_total', 'cfb3_total', 'combo_total']].to_parquet(out.replace('.txt', '.parquet'), index=False)
