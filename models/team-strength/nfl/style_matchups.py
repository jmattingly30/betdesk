"""J 2026-10-11 (follow-up to scheme_game.py): pooled scheme STYLE matchups and coach-vs-style grades, judged on predicting
actual game results. Team splits vs single looks were noise; here every team is described by its style (rates over all its
snaps, which repeat), and the question is whether certain offense styles beat certain defense styles across the league
(rock-paper-scissors), and whether a particular play-caller / head coach beats a defensive style more than the pool says.
Offense style (prior season half weight + season to date): 11 personnel rate, heavy (2+ TE or 2+ RB) rate, pass rate over
expected, deep (20+ air yards) share, shotgun rate, no-huddle rate, average time to throw.
Defense style: nickel+dime rate, man rate, two-high rate, 5+ rusher rate, heavy box (8+) rate on runs.
Outcome per team-game: offense EPA/play (win prob 10-90%) minus expectation (walk-forward offense + defense EPA ratings),
and team points minus closing-line implied points. Interactions fit 2018-2021, tested 2022-2025.
usage: python3 nfl/style_matchups.py <dl dir: pbp_YYYY, part_YYYY> [play_callers.csv]   -> nfl/style_matchups_log.txt"""
import sys, os, numpy as np, pandas as pd, statsmodels.api as sm
H = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
dl = sys.argv[1]; pc = sys.argv[2] if len(sys.argv) > 2 else os.path.normpath(f'{H}/../nfl-player/data/play_callers.csv')
def cnt(s, p):
    if not isinstance(s, str): return np.nan
    for part in s.split(','):
        part = part.strip()
        if ' ' in part:
            k, q = part.split(' ', 1)
            if k.isdigit() and q == p: return int(k)
    return 0
fr = []
for y in range(2017, 2026):
    P = pd.read_parquet(f'{dl}/pbp_{y}.parquet', columns=['game_id', 'play_id', 'season', 'season_type', 'week', 'posteam', 'defteam', 'epa', 'pass', 'rush', 'qb_dropback',
                                                         'wp', 'xpass', 'air_yards', 'shotgun', 'no_huddle', 'two_point_attempt', 'qb_kneel', 'qb_spike'])
    P = P[((P['pass'] == 1) | (P.rush == 1)) & P.epa.notna() & P.posteam.notna() & (P.two_point_attempt != 1) & (P.qb_kneel != 1) & (P.qb_spike != 1)]
    Q = pd.read_parquet(f'{dl}/part_{y}.parquet', columns=['nflverse_game_id', 'play_id', 'offense_personnel', 'defense_personnel', 'defenders_in_box',
                                                         'number_of_pass_rushers', 'defense_man_zone_type', 'defense_coverage_type', 'time_to_throw'])
    P = P.merge(Q.rename(columns={'nflverse_game_id': 'game_id'}), on=['game_id', 'play_id'], how='left')
    rb, te, wr = (P.offense_personnel.map(lambda s: cnt(s, k)) for k in ('RB', 'TE', 'WR'))
    P['o11'] = np.where(rb.notna(), ((rb == 1) & (te == 1) & (wr == 3)).astype(float), np.nan)
    P['oheavy'] = np.where(rb.notna(), ((rb + te) >= 3).astype(float), np.nan)
    nd = P.defense_personnel.map(lambda s: sum(cnt(s, k) for k in ('CB', 'FS', 'SS', 'DB', 'S')) if isinstance(s, str) else np.nan)
    P['dnick'] = np.where(nd.notna(), (nd >= 5).astype(float), np.nan)
    db = P.qb_dropback == 1
    mz = P.defense_man_zone_type.fillna('')
    P['dman'] = np.where(db & mz.isin(['MAN_COVERAGE', 'ZONE_COVERAGE']), (mz == 'MAN_COVERAGE').astype(float), np.nan)
    ct = P.defense_coverage_type.fillna('')
    P['dtwo'] = np.where(db & (ct != '') & (ct != 'BLOWN') & (ct != 'COMBO'), ct.isin(['COVER_2', 'COVER_4', 'COVER_6', '2_MAN', 'COVER_9']).astype(float), np.nan)
    P['dblitz'] = np.where(db & P.number_of_pass_rushers.notna(), (P.number_of_pass_rushers >= 5).astype(float), np.nan)
    P['dbox'] = np.where((P.rush == 1) & P.defenders_in_box.notna(), (P.defenders_in_box >= 8).astype(float), np.nan)
    P['oproe'] = np.where(P.xpass.notna() & P.wp.between(0.2, 0.8), P['pass'] - P.xpass, np.nan)
    P['odeep'] = np.where((P['pass'] == 1) & P.air_yards.notna(), (P.air_yards >= 20).astype(float), np.nan)
    P['ogun'] = P.shotgun.astype(float); P['onh'] = P.no_huddle.astype(float)
    P['ottt'] = np.where(db, P.time_to_throw, np.nan)
    P['wk'] = np.where(P.season_type == 'POST', P.week.where(P.week > 18, P.week + 18), P.week)
    fr.append(P[['game_id', 'season', 'wk', 'posteam', 'defteam', 'epa', 'wp', 'o11', 'oheavy', 'oproe', 'odeep', 'ogun', 'onh', 'ottt', 'dnick', 'dman', 'dtwo', 'dblitz', 'dbox']])
P = pd.concat(fr, ignore_index=True)
OS = ['o11', 'oheavy', 'oproe', 'odeep', 'ogun', 'onh', 'ottt']; DS = ['dnick', 'dman', 'dtwo', 'dblitz', 'dbox']
N = P[P.wp.between(0.1, 0.9)]
# team-game aggregates
OG = P.groupby(['game_id', 'season', 'wk', 'posteam'])[OS].mean().join(N.groupby(['game_id', 'season', 'wk', 'posteam']).epa.agg(['mean', 'size']).rename(columns={'mean': 'oepa', 'size': 'on'}))
DG = P.groupby(['game_id', 'season', 'wk', 'defteam'])[DS].mean().join(N.groupby(['game_id', 'season', 'wk', 'defteam']).epa.agg(['mean', 'size']).rename(columns={'mean': 'depa', 'size': 'dn'}))
OG = OG.reset_index().rename(columns={'posteam': 'team'}); DG = DG.reset_index().rename(columns={'defteam': 'team'})
pair = P.groupby(['game_id', 'posteam']).defteam.first().reset_index().rename(columns={'posteam': 'team', 'defteam': 'opp'})
def prior(G, cols, team):
    """prior-only style/rating per team-game: earlier games this season (weight 1 each) + last season's games (weight 0.5 each)"""
    G = G.sort_values(['season', 'wk']).copy(); out = {c: np.full(len(G), np.nan) for c in cols}
    for t, d in G.groupby(team):
        idx = d.index.values; S = d.season.values; W = d.wk.values; X = d[cols].values
        for j in range(len(d)):
            w = np.where(S == S[j], (W < W[j]) * 1.0, np.where(S == S[j] - 1, 0.5, 0.0))
            for ci, c in enumerate(cols):
                ok = ~np.isnan(X[:, ci]) & (w > 0)
                if ok.sum(): out[c][np.searchsorted(G.index.values, idx[j]) if False else G.index.get_loc(idx[j])] = np.average(X[ok, ci], weights=w[ok])
    for c in cols: G['p_' + c] = out[c]
    return G
OG = prior(OG, OS + ['oepa'], 'team'); DG = prior(DG, DS + ['depa'], 'team')
X = OG.merge(pair, on=['game_id', 'team']).merge(DG.drop(columns=['season', 'wk']).rename(columns={'team': 'opp'}), on=['game_id', 'opp'])
X = X[X.season >= 2018].dropna(subset=['p_oepa', 'p_depa'] + ['p_' + c for c in OS + DS] + ['oepa'])
lg = X.groupby('season').oepa.mean()
X['res'] = X.oepa - (X.p_oepa + X.p_depa - X.season.map(lg))
# standardise styles within season
for c in OS + DS: X['z_' + c] = X.groupby('season')['p_' + c].transform(lambda s: (s - s.mean()) / s.std())
INT = []
for o in OS:
    for d in DS: X[f'{o}x{d}'] = X['z_' + o] * X['z_' + d]; INT.append(f'{o}x{d}')
tr = X.season <= 2021; te = X.season >= 2022
print(f'team-games: train 2018-21 {tr.sum()}, test 2022-25 {te.sum()}; residual SD (EPA/play) {X.res.std():.3f}')
base = X.res[te].var()
Z = ['z_' + c for c in OS + DS]
for nm, cols in [('main style effects', Z), ('main + all 35 interactions', Z + INT)]:
    m = sm.OLS(X.res[tr], sm.add_constant(X[tr][cols])).fit()
    pr = m.predict(sm.add_constant(X[te][cols]))
    print(f'  {nm:28s}: out-of-sample R2 {1 - np.mean((X.res[te] - pr) ** 2) / base:+.4f}')
ridge = None
from sklearn.linear_model import RidgeCV
r = RidgeCV(alphas=np.logspace(0, 4, 20)).fit(X[tr][Z + INT], X.res[tr]); pr = r.predict(X[te][Z + INT])
print(f'  ridge (alpha {r.alpha_:.0f})            : out-of-sample R2 {1 - np.mean((X.res[te] - pr) ** 2) / base:+.4f}')
full = sm.OLS(X.res, sm.add_constant(X[INT])).fit()
tv = full.tvalues.drop('const').sort_values()
print('  strongest interactions, all years (t): ' + ', '.join(f'{k} {v:+.1f}' for k, v in list(tv.head(4).items()) + list(tv.tail(4).items())))
# sign stability of each interaction: first half vs second half of years
a = sm.OLS(X.res[tr], sm.add_constant(X[tr][INT])).fit().params.drop('const'); b = sm.OLS(X.res[te], sm.add_constant(X[te][INT])).fit().params.drop('const')
print(f'  interaction coefficients 2018-21 vs 2022-25: r {np.corrcoef(a, b)[0, 1]:+.2f} (1 = same rock-paper-scissors both periods, 0 = noise)')

# coach vs style: offence play-caller's slope vs each defence style, repeat next season?
cl = pd.read_csv(pc)[['season', 'team', 'caller']]
X = X.merge(cl, on=['season', 'team'], how='left')
print('\n  play-caller vs defence style: residual slope on the opponent style, caller-season vs next caller-season')
for d in DS:
    rows = []
    for (c, s), g in X.groupby(['caller', 'season']):
        if len(g) >= 10: rows.append((c, s, np.polyfit(g['z_' + d], g.res, 1)[0], len(g)))
    R = pd.DataFrame(rows, columns=['caller', 'season', 'b', 'n']); R2 = R.copy(); R2['season'] -= 1
    J = R.merge(R2, on=['caller', 'season'], suffixes=('', '_n'))
    # split half: odd/even games
    sh = []
    for (c, s), g in X.groupby(['caller', 'season']):
        g = g.sort_values('wk'); o, e = g.iloc[::2], g.iloc[1::2]
        if len(o) >= 6 and len(e) >= 6: sh.append((np.polyfit(o['z_' + d], o.res, 1)[0], np.polyfit(e['z_' + d], e.res, 1)[0]))
    sh = np.array(sh)
    print(f'    vs {d:6s}: next season r {np.corrcoef(J.b, J.b_n)[0, 1]:+.2f} (n {len(J)}), split-half r {np.corrcoef(sh[:, 0], sh[:, 1])[0, 1]:+.2f}')
# defence head coach vs offence style
T = pd.read_parquet(f'{H}/data/nfl_coach_games.parquet')[['game_id', 'team', 'coach']].rename(columns={'team': 'opp', 'coach': 'dcoach'})
X = X.merge(T, on=['game_id', 'opp'], how='left')
print('  defensive-side head coach vs offence style (offence residual slope; negative = his defence handles that style):')
for o in OS:
    sh = []; rows = []
    for (c, s), g in X.groupby(['dcoach', 'season']):
        if len(g) >= 10:
            rows.append((c, s, np.polyfit(g['z_' + o], g.res, 1)[0]))
            g = g.sort_values('wk'); a1, b1 = g.iloc[::2], g.iloc[1::2]
            if len(a1) >= 6 and len(b1) >= 6: sh.append((np.polyfit(a1['z_' + o], a1.res, 1)[0], np.polyfit(b1['z_' + o], b1.res, 1)[0]))
    R = pd.DataFrame(rows, columns=['c', 's', 'b']); R2 = R.copy(); R2['s'] -= 1; J = R.merge(R2, on=['c', 's']); sh = np.array(sh)
    print(f'    vs {o:6s}: next season r {np.corrcoef(J.b_x, J.b_y)[0, 1]:+.2f} (n {len(J)}), split-half r {np.corrcoef(sh[:, 0], sh[:, 1])[0, 1]:+.2f}')
