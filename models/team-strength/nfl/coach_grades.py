"""J 2026-10-11: can we measure a team being OUTCOACHED, grade coaches, and find coaches who are great at / against a
scheme? Judged on predicting actual results (closing line as a second check). NFL 2007-2025.

Residual of a team-game = the team's actual margin minus the pre-game expectation. Three expectations:
  model  walk-forward blend of score + market ratings (weights fit on earlier seasons only), what the app uses
  score  score-margin rating alone (no market information)
  close  closing spread
Tests
  1 Does a head coach's residual repeat next season? Same coach + same team vs coach who MOVED (does it follow him?)
    vs team whose coach changed (does it stay with the team?). Plus a variance split: how big is the true coach effect.
  2 Career grade (shrunk mean residual from earlier seasons only) -> does it predict this season's results? Overall and in
    games rated close (|expected margin| <= 3).
  3 Coach vs coach: the residual in earlier meetings of the same two head coaches -> this meeting.
  4 Second-half adjustments: 2nd-half minus 1st-half EPA/play margin (neutral game states), coach-season repeatability.
  5 One-score games and 4th-quarter comebacks: coach-season close-game record minus expectation, repeatability.
usage: python3 nfl/coach_grades.py <dir with pbp_YYYY.parquet>     -> nfl/coach_grades_log.txt, data/nfl_coach_games.parquet"""
import sys, os, numpy as np, pandas as pd, statsmodels.api as sm
from scipy.stats import norm
H = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
dl = sys.argv[1]
g = pd.read_parquet(f'{H}/data/nfl_games_features.parquet')
g = g[g.margin.notna() & g.season.between(2007, 2025)].copy().sort_values(['gameday', 'gametime', 'game_id']).reset_index(drop=True)
est = np.full(len(g), np.nan)
for y in range(2008, 2026):
    tr, te = (g.season < y).values, (g.season == y).values
    X = np.column_stack([np.ones(tr.sum()), g.rt_pred[tr], g.mkt_pred[tr]]); b = np.linalg.lstsq(X, g.margin[tr], rcond=None)[0]
    est[te] = b[0] + b[1] * g.rt_pred[te] + b[2] * g.mkt_pred[te]
g['e_model'] = est; g['e_score'] = g.rt_pred; g['e_close'] = g.spread_line
g = g[g.season >= 2008]
# team-game table (one row per team per game)
rows = []
for side, opp_side, sgn in (('home', 'away', 1), ('away', 'home', -1)):
    d = pd.DataFrame({'game_id': g.game_id, 'season': g.season, 'week': g.week, 'gameday': g.gameday, 'team': g[f'{side}_team'], 'opp': g[f'{opp_side}_team'],
                      'coach': g[f'{side}_coach'], 'opp_coach': g[f'{opp_side}_coach'], 'home': int(side == 'home'), 'margin': sgn * g.margin,
                      'e_model': sgn * g.e_model, 'e_score': sgn * g.e_score, 'e_close': sgn * g.e_close, 'game_type': g.game_type})
    rows.append(d)
T = pd.concat(rows, ignore_index=True).sort_values(['gameday', 'game_id', 'home']).reset_index(drop=True)
for e in ['model', 'score', 'close']: T[f'r_{e}'] = T.margin - T[f'e_{e}']
T['win'] = (T.margin > 0).astype(float).where(T.margin != 0, 0.5)
print(f'NFL 2008-2025: {T.game_id.nunique()} games, {T.coach.nunique()} head coaches; residual SD model {T.r_model.std():.2f}, close {T.r_close.std():.2f}')

# ---------------- 1. repeatability ----------------
CS = T.groupby(['coach', 'season']).agg(team=('team', lambda s: s.mode()[0]), n=('margin', 'size'), **{f'm_{e}': (f'r_{e}', 'mean') for e in ['model', 'score', 'close']}).reset_index()
CS = CS[CS.n >= 8]
print('\n1. Does a coach-season residual repeat? (correlation with the next season he coaches, games >= 8)')
nx = CS.copy(); nx['season'] -= 1
J = CS.merge(nx, on=['coach', 'season'], suffixes=('', '_n'))
J['moved'] = J.team != J.team_n
# coach leaves: team's next season under a NEW coach
TS = T.groupby(['team', 'season']).agg(coach=('coach', lambda s: s.mode()[0]), **{f'm_{e}': (f'r_{e}', 'mean') for e in ['model', 'score', 'close']}).reset_index()
tn = TS.copy(); tn['season'] -= 1
K = TS.merge(tn, on=['team', 'season'], suffixes=('', '_n')); K = K[K.coach != K.coach_n]
for e in ['model', 'score', 'close']:
    st = J[~J.moved]; mv = J[J.moved]
    print(f'  vs {e:5s}: same coach same team r {np.corrcoef(st[f"m_{e}"], st[f"m_{e}_n"])[0, 1]:+.2f} (n {len(st)})   coach moved r '
          f'{np.corrcoef(mv[f"m_{e}"], mv[f"m_{e}_n"])[0, 1]:+.2f} (n {len(mv)})   team after coach left r {np.corrcoef(K[f"m_{e}"], K[f"m_{e}_n"])[0, 1]:+.2f} (n {len(K)})')
# movers with any gap: last season at old team -> first at new team, all career moves
car = CS.sort_values('season')
mv_rows = []
for c, d in car.groupby('coach'):
    d = d.reset_index(drop=True)
    for i in range(1, len(d)):
        if d.team[i] != d.team[i - 1]: mv_rows.append((c, d.season[i - 1], d.season[i], d.team[i - 1], d.team[i], d.m_model[:i].mean(), d.m_model[i], d.m_close[:i].mean(), d.m_close[i]))
M = pd.DataFrame(mv_rows, columns=['coach', 's0', 's1', 't0', 't1', 'old_model', 'new_model', 'old_close', 'new_close'])
print(f'  every coach move with a gap allowed (career mean at old teams -> first season at new team, n {len(M)}): '
      f'model r {np.corrcoef(M.old_model, M.new_model)[0, 1]:+.2f}, close r {np.corrcoef(M.old_close, M.new_close)[0, 1]:+.2f}')
print('    movers: ' + '; '.join(f'{r.coach} {r.t0}->{r.t1} {r.s1}: {r.old_close:+.1f} -> {r.new_close:+.1f}' for r in M.sort_values('s1').tail(12).itertuples()))
# variance split: true coach-season effect SD = sqrt(var(means) - mean(noise var / n))
for e in ['model', 'close']:
    v = CS[f'm_{e}'].var(); noise = (T[f'r_{e}'].var() / CS.n).mean()
    print(f'  coach-season mean residual vs {e}: SD {np.sqrt(v):.2f} pts; pure luck would give {np.sqrt(noise):.2f}; implied true spread {np.sqrt(max(v - noise, 0)):.2f} pts per game')
CC = T.groupby('coach').agg(n=('margin', 'size'), m=('r_close', 'mean')); CC = CC[CC.n >= 48]
v = CC.m.var(); noise = (T.r_close.var() / CC.n).mean()
print(f'  CAREER mean residual vs close (coaches with 48+ games, n {len(CC)}): SD {np.sqrt(v):.2f}; luck {np.sqrt(noise):.2f}; implied true {np.sqrt(max(v - noise, 0)):.2f} pts per game')
print('    top / bottom career residual vs close (pts per game, games):')
CC = CC.sort_values('m')
print('    ' + ', '.join(f'{c} {r.m:+.1f} ({r.n})' for c, r in CC.tail(8)[::-1].iterrows()))
print('    ' + ', '.join(f'{c} {r.m:+.1f} ({r.n})' for c, r in CC.head(8).iterrows()))

# ---------------- 2. walk-forward career grade ----------------
print('\n2. Career grade from EARLIER seasons only (mean residual, shrunk with k games of 0) -> this season')
def grade(T, col, k):
    out = np.zeros(len(T)); cum = {}
    for y in sorted(T.season.unique()):
        idx = T.season == y
        out[idx.values] = [cum.get(c, (0, 0))[0] / (cum.get(c, (0, 0))[1] + k) for c in T.coach[idx]]
        oc = [cum.get(c, (0, 0))[0] / (cum.get(c, (0, 0))[1] + k) for c in T.opp_coach[idx]]
        T.loc[idx, 'og_' + col] = oc
        for c, d in T[idx].groupby('coach'): s, n = cum.get(c, (0, 0)); cum[c] = (s + d[col].sum(), n + len(d))
    return out
for e in ['model', 'close']:
    for k in [16, 48, 96]:
        T['g'] = grade(T, f'r_{e}', k); T['gd'] = T.g - T['og_r_' + e]
        t = T[(T.season >= 2010) & T.home.eq(1)]
        r = sm.OLS(t[f'r_{e}'], sm.add_constant(t.gd)).fit()
        print(f'  vs {e:5s} k {k:3d}: coef {r.params.iloc[1]:+.2f} (t {r.tvalues.iloc[1]:+.2f}), grade-gap SD {t.gd.std():.2f} pts')
k = 48
T['g'] = grade(T, 'r_model', k); T['gd'] = T.g - T['og_r_model']
t = T[(T.season >= 2010) & T.home.eq(1)].copy()
base = np.sqrt(np.mean(t.r_model ** 2))
for a in [0.25, 0.5, 1.0]:
    print(f'  margin RMSE with the coach grade added x{a}: {np.sqrt(np.mean((t.r_model - a * t.gd) ** 2)):.3f} (model alone {base:.3f})')
print('  close games (|model margin| <= 3): win rate of the team with the better coach grade, by grade gap:')
c3 = t[t.e_model.abs() <= 3]
for lo, hi in [(0, 1), (1, 2), (2, 9)]:
    s = c3[c3.gd.abs().between(lo, hi, inclusive='left')]
    w = np.where(s.gd > 0, s.win, 1 - s.win); ex = np.where(s.gd > 0, norm.cdf(s.e_model / 13.3), 1 - norm.cdf(s.e_model / 13.3))
    se = np.sqrt(0.25 / max(len(s), 1))
    print(f'    gap {lo}-{hi} pts: n {len(s):4d}  better-graded coach won {w.mean():.1%} (+-{1.96 * se:.1%}), model expected {ex.mean():.1%}')

# ---------------- 3. coach vs coach ----------------
print('\n3. Coach vs coach: residual in their earlier meetings -> this meeting (vs model, then vs close)')
t = T[T.home == 1].sort_values('gameday')
for e in ['model', 'close']:
    hist = {}; X = []
    for r in t.itertuples():
        key = (r.coach, r.opp_coach); rk = (r.opp_coach, r.coach)
        prev = hist.get(key, []) + [-x for x in hist.get(rk, [])]
        if prev: X.append((np.mean(prev), len(prev), getattr(r, f'r_{e}'), r.team == r.team, r.season))
        hist.setdefault(key, []).append(getattr(r, f'r_{e}'))
    X = pd.DataFrame(X, columns=['prev', 'n', 'y', 'x', 'season'])
    rr = sm.OLS(X.y, sm.add_constant(X.prev)).fit()
    m3 = X[X.n >= 3]; r3 = sm.OLS(m3.y, sm.add_constant(m3.prev)).fit()
    print(f'  vs {e:5s}: {len(X)} games with an earlier meeting, coef {rr.params.iloc[1]:+.3f} (t {rr.tvalues.iloc[1]:+.2f});  3+ earlier meetings n {len(m3)}: coef {r3.params.iloc[1]:+.3f} (t {r3.tvalues.iloc[1]:+.2f})')
# same-season division rematch: game 2 residual on game 1 residual
t2 = t.copy(); t2['pair'] = [tuple(sorted([a, b])) + (s,) for a, b, s in zip(t2.team, t2.opp, t2.season)]
pr = []
for p, d in t2[t2.game_type == 'REG'].groupby('pair'):
    if len(d) == 2:
        a, b = d.iloc[0], d.iloc[1]; s = 1 if a.team == b.team else -1
        pr.append((a.r_model, s * b.r_model, a.r_close, s * b.r_close))
pr = np.array(pr)
print(f'  same-season rematches (n {len(pr)}): game-1 residual -> game-2 residual r {np.corrcoef(pr[:, 0], pr[:, 1])[0, 1]:+.3f} (model), {np.corrcoef(pr[:, 2], pr[:, 3])[0, 1]:+.3f} (close)')

# ---------------- 4/5. in-game coaching from play-by-play ----------------
P = pd.concat([pd.read_parquet(f'{dl}/pbp_{y}.parquet', columns=['game_id', 'season', 'posteam', 'defteam', 'qtr', 'epa', 'wp', 'pass', 'rush', 'home_team',
                                                                 'total_home_score', 'total_away_score', 'game_seconds_remaining']) for y in range(2016, 2026)])
P = P[((P['pass'] == 1) | (P.rush == 1)) & P.epa.notna() & P.posteam.notna()]
N = P[P.wp.between(0.1, 0.9)]
half = np.where(N.qtr <= 2, 'h1', 'h2')
E = N.assign(half=half).groupby(['game_id', 'posteam', 'half']).epa.mean().unstack()
D = N.assign(half=half).groupby(['game_id', 'defteam', 'half']).epa.mean().unstack()
E.index.names = ['game_id', 'team']; D.index.names = ['game_id', 'team']
A = (E - D).dropna(); A['adj'] = A.h2 - A.h1      # EPA/play margin, 2nd half minus 1st half
A = A.reset_index().merge(T[['game_id', 'team', 'coach', 'season']], on=['game_id', 'team'])
ca = A.groupby(['coach', 'season']).agg(adj=('adj', 'mean'), team=('team', 'first'), n=('adj', 'size')).reset_index(); ca = ca[ca.n >= 8]
c2 = ca.copy(); c2['season'] -= 1; j = ca.merge(c2, on=['coach', 'season'], suffixes=('', '_n'))
print(f'\n4. Second-half adjustment (2H minus 1H EPA/play margin, game states 10-90% win prob), 2016-2025:')
print(f'   coach-season repeat r {np.corrcoef(j.adj, j.adj_n)[0, 1]:+.2f} (n {len(j)}); same team {np.corrcoef(j[j.team == j.team_n].adj, j[j.team == j.team_n].adj_n)[0, 1]:+.2f}; '
      f'odd/even games split-half below')
A['k'] = A.groupby(['coach', 'season']).cumcount() % 2
sh = A.groupby(['coach', 'season', 'k']).adj.mean().unstack().dropna()
print(f'   split-half r {np.corrcoef(sh[0], sh[1])[0, 1]:+.2f} (coach-seasons {len(sh)})')
# 5: one-score games
g1 = T[(T.margin.abs() <= 8)].copy()
g1['wexp'] = norm.cdf(g1.e_model / 13.3)
cs = g1.groupby(['coach', 'season']).apply(lambda d: pd.Series(dict(n=len(d), over=(d.win - d.wexp).sum() / len(d), team=d.team.mode()[0]))).reset_index()
cs = cs[cs.n >= 4]; c2 = cs.copy(); c2['season'] -= 1; j = cs.merge(c2, on=['coach', 'season'], suffixes=('', '_n'))
print(f'\n5. One-score games (final margin <= 8): win rate minus model expectation, coach-season repeat r {np.corrcoef(j.over, j.over_n)[0, 1]:+.2f} (n {len(j)})')
cc = g1.groupby('coach').apply(lambda d: pd.Series(dict(n=len(d), over=(d.win - d.wexp).mean()))); cc = cc[cc.n >= 40]
print(f'   career (40+ one-score games, n {len(cc)}): SD {cc.over.std():.3f}, luck alone {np.sqrt((0.25 / cc.n).mean()):.3f}; best ' +
      ', '.join(f'{c} {r.over:+.2f} ({r.n:.0f})' for c, r in cc.sort_values('over').tail(5)[::-1].iterrows()) + '; worst ' +
      ', '.join(f'{c} {r.over:+.2f} ({r.n:.0f})' for c, r in cc.sort_values('over').head(5).iterrows()))
T.drop(columns=['g', 'gd'], errors='ignore').to_parquet(f'{H}/data/nfl_coach_games.parquet')
