"""Play-callers and game plans (J's follow-up, 2026-10-10).

python3 scheme_callers.py <dl_dir> <model_dir>
  dl_dir: nflverse pbp_YYYY, part_YYYY (participation), ftn_YYYY (FTN charting), players.parquet; 2023-2025
  model_dir/data/play_callers.csv: each team's primary offensive play-caller by season (hand-built 2026-10-10 from
  ESPN / The 33rd Team / Fantasy Index play-caller lists and season records; mid-season changes noted, not split)

A. Which offensive tendencies belong to the PLAY-CALLER? Year-to-year correlation of team-season tendencies when the
   caller stayed vs changed, and for callers who moved teams: does the new team look like his old team?
B. Do callers change the game plan for the opponent? Each team-game's deviation from its own season average,
   regressed on the opponent's coverage habits (man rate, two-high rate, nickel/dime rate, blitz rate, from its other games).
Writes scheme_callers_log.txt."""
import sys, numpy as np, pandas as pd, statsmodels.api as sm
dl, M = sys.argv[1], sys.argv[2]
log = open(f'{M}/scheme_callers_log.txt', 'w')
def say(*a): s = ' '.join(str(x) for x in a); print(s, flush=True); log.write(s + '\n'); log.flush()
YRS = tuple(range(int(sys.argv[3]), 2026)) if len(sys.argv) > 3 else (2023, 2024, 2025)   # 2016-2022 callers added 2026-10-11
import os
cols = ['game_id', 'play_id', 'season', 'week', 'posteam', 'defteam', 'pass', 'rush', 'xpass', 'qb_dropback', 'air_yards', 'receiver_player_id',
        'yardline_100', 'two_point_attempt', 'season_type', 'yards_gained', 'wp']
P = pd.concat([pd.read_parquet(f'{dl}/pbp_{y}.parquet', columns=cols) for y in YRS], ignore_index=True)
P = P[(P.season_type == 'REG') & (P.two_point_attempt != 1) & ((P['pass'] == 1) | (P.rush == 1))]
Q = pd.concat([pd.read_parquet(f'{dl}/part_{y}.parquet', columns=['nflverse_game_id', 'play_id', 'offense_personnel', 'defense_personnel',
                                                                     'defense_man_zone_type', 'defense_coverage_type']) for y in YRS])
F = pd.concat([pd.read_parquet(f'{dl}/ftn_{y}.parquet', columns=['nflverse_game_id', 'nflverse_play_id', 'is_play_action', 'is_motion', 'is_screen_pass',
                                                                    'n_blitzers', 'is_rpo', 'is_no_huddle']) for y in YRS if os.path.exists(f'{dl}/ftn_{y}.parquet')])   # FTN starts 2022
P = P.merge(Q.rename(columns={'nflverse_game_id': 'game_id'}), on=['game_id', 'play_id'], how='left')
P = P.merge(F.rename(columns={'nflverse_game_id': 'game_id', 'nflverse_play_id': 'play_id'}), on=['game_id', 'play_id'], how='left')
pos = pd.read_parquet(f'{dl}/players.parquet', columns=['gsis_id', 'position']).set_index('gsis_id').position
P['tpos'] = P.receiver_player_id.map(pos).replace({'FB': 'RB'})
def cnt(s, p):
    if not isinstance(s, str): return np.nan
    for part in s.split(','):
        if ' ' not in part.strip(): continue
        k, q = part.strip().split(' ', 1)
        if not k.isdigit(): continue
        if q == p: return int(k)
    return 0
P['n_rb'] = P.offense_personnel.map(lambda s: cnt(s, 'RB')); P['n_te'] = P.offense_personnel.map(lambda s: cnt(s, 'TE')); P['n_wr'] = P.offense_personnel.map(lambda s: cnt(s, 'WR'))
P['dbs'] = P.defense_personnel.map(lambda s: sum(cnt(s, p) for p in ('CB', 'FS', 'SS', 'DB', 'S')) if isinstance(s, str) else np.nan)
neutral = P.wp.between(0.2, 0.8)
P['is11'] = ((P.n_rb == 1) & (P.n_te == 1) & (P.n_wr == 3)).astype(float).where(P.n_rb.notna())
P['is12'] = ((P.n_rb == 1) & (P.n_te == 2)).astype(float).where(P.n_rb.notna())
P['heavy'] = ((P.n_te + P.n_rb) >= 3).astype(float).where(P.n_rb.notna())
P['proe'] = (P['pass'] - P.xpass).where(neutral & P.xpass.notna())
db = P.qb_dropback == 1
P['pa'] = P.is_play_action.astype(float).where(db & P.is_play_action.notna()); P['motion'] = P.is_motion.astype(float).where(P.is_motion.notna())
P['screen'] = P.is_screen_pass.astype(float).where(db & P.is_screen_pass.notna())
tgt = P.receiver_player_id.notna()
for p_ in ['WR', 'TE', 'RB']: P[f'sh_{p_}'] = (P.tpos == p_).astype(float).where(tgt)
P['adot'] = P.air_yards.where(tgt)
rz = P.yardline_100 <= 20
P['rz_pass'] = P['pass'].astype(float).where(rz); P['rz_te_rb'] = P.tpos.isin(['TE', 'RB']).astype(float).where(rz & tgt)
# defense looks (for B and for opponent habits)
P['d_man'] = P.defense_man_zone_type.map({'MAN_COVERAGE': 1.0, 'ZONE_COVERAGE': 0.0})
P['d_two'] = P.defense_coverage_type.map({'COVER_0': 0.0, 'COVER_1': 0.0, 'COVER_3': 0.0, 'COVER_2': 1.0, 'COVER_4': 1.0, 'COVER_6': 1.0, 'COVER_9': 1.0, '2_MAN': 1.0})
P['d_nick'] = (P.dbs >= 5).astype(float).where(P.dbs.notna()); P['d_blitz'] = (P.n_blitzers > 0).astype(float).where(P.n_blitzers.notna() & db)
P['ypp'] = P.yards_gained.where(db)
# offense results by look (does this offense do better vs man than zone?)
P['ypp_man'] = P.ypp.where(P.d_man == 1); P['ypp_zone'] = P.ypp.where(P.d_man == 0)
P['ypp_one'] = P.ypp.where(P.d_two == 0); P['ypp_two'] = P.ypp.where(P.d_two == 1)

TEND = ['proe', 'is11', 'is12', 'heavy', 'pa', 'motion', 'screen', 'sh_WR', 'sh_TE', 'sh_RB', 'adot', 'rz_pass', 'rz_te_rb']
TS = P.groupby(['season', 'posteam'])[TEND + ['ypp_man', 'ypp_zone', 'ypp_one', 'ypp_two']].mean()
TS['man_minus_zone'] = TS.ypp_man - TS.ypp_zone; TS['one_minus_two'] = TS.ypp_one - TS.ypp_two
TS = TS.drop(columns=['ypp_man', 'ypp_zone', 'ypp_one', 'ypp_two']).reset_index()
C = pd.read_csv(f'{M}/data/play_callers.csv')
TS = TS.merge(C[['season', 'team', 'caller']].rename(columns={'team': 'posteam'}), on=['season', 'posteam'], how='left')
say(f'team-seasons {len(TS)}, callers matched {TS.caller.notna().sum()}')
ALL = TEND + ['man_minus_zone', 'one_minus_two']
def rr(a, b):
    ok = a.notna() & b.notna(); return np.corrcoef(a[ok], b[ok])[0, 1] if ok.sum() > 4 else np.nan
a = TS.rename(columns={c: c + '_1' for c in ALL + ['caller']}); a['season'] += 1
pair = TS.merge(a, on=['season', 'posteam'])
same = pair.caller == pair.caller_1
say(f'\nA1. Same team, next season: correlation of each tendency, caller stayed ({same.sum()} team-pairs) vs caller changed ({(~same).sum()})')
say(f'   {"tendency":16s} {"stayed":>8s} {"changed":>8s}')
for c in ALL:
    r1 = rr(pair[c][same], pair[c + '_1'][same]); r2 = rr(pair[c][~same], pair[c + '_1'][~same])
    say(f'   {c:16s} {r1:+8.2f} {r2:+8.2f}')
# movers: caller on a new team; compare the new team-season to the caller's previous team-season
# the caller's most recent earlier season as primary caller (gap years allowed, e.g. Pederson PHI 2020 -> JAX 2022)
prev = TS.rename(columns={c: c + '_old' for c in ALL + ['posteam']}).rename(columns={'season': 'season_old'})
mv = TS.merge(prev, on='caller'); mv = mv[mv.season_old < mv.season].sort_values('season_old').groupby(['season', 'posteam']).tail(1)
mv = mv[mv.posteam != mv.posteam_old]
# and the new team's own previous season (other caller)
own = TS[['season', 'posteam'] + ALL].rename(columns={c: c + '_team' for c in ALL}); own['season'] += 1
mv = mv.merge(own, on=['season', 'posteam'], how='left')
say(f'\nA2. Callers who moved teams ({len(mv)}: ' + ', '.join(f'{r.caller} {r.posteam_old}->{r.posteam} {r.season}' for r in mv.itertuples()) + ')')
say('   does the new team look like the CALLER\'s old team, or like the TEAM\'s own last season?')
say(f'   {"tendency":16s} {"caller old team":>16s} {"team last year":>15s}')
for c in ALL:
    r1 = rr(mv[c], mv[c + '_old']); r2 = rr(mv[c], mv[c + '_team'])
    say(f'   {c:16s} {r1:+16.2f} {r2:+15.2f}')

# ---------- B: game-plan adjustment to the opponent ----------
TG = P.groupby(['season', 'game_id', 'posteam', 'defteam'])[TEND].mean().reset_index()
DG = P.groupby(['season', 'game_id', 'defteam'])[['d_man', 'd_two', 'd_nick', 'd_blitz']].agg(['sum', 'count'])
DG.columns = [f'{a}_{b}' for a, b in DG.columns]; DG = DG.reset_index()
DS = DG.groupby(['season', 'defteam']).sum(numeric_only=True)
DG = DG.join(DS, on=['season', 'defteam'], rsuffix='_s')
for c in ['d_man', 'd_two', 'd_nick', 'd_blitz']:       # opponent's habit from its OTHER games that season
    DG[f'opp_{c}'] = (DG[f'{c}_sum_s'] - DG[f'{c}_sum']) / (DG[f'{c}_count_s'] - DG[f'{c}_count']).clip(lower=1)
TG = TG.merge(DG[['game_id', 'defteam'] + [f'opp_{c}' for c in ['d_man', 'd_two', 'd_nick', 'd_blitz']]], on=['game_id', 'defteam'])
say('\nB. Game plans: team-game deviation from the team\'s own season average, on the opponent\'s coverage habits')
say('   (coefficient = change in the tendency for +10 points of opponent man / two-high / nickel+ / blitz rate; * = p < 0.01)')
X = TG[[f'opp_{c}' for c in ['d_man', 'd_two', 'd_nick', 'd_blitz']]]
X = (X - X.groupby(TG.season).transform('mean')) * 10
for c in TEND:
    y = TG[c] - TG.groupby(['season', 'posteam'])[c].transform('mean')
    ok = y.notna() & X.notna().all(1)
    f = sm.OLS(y[ok], sm.add_constant(X[ok])).fit(cov_type='cluster', cov_kwds={'groups': TG.posteam[ok]})
    sd_game = y[ok].std()
    say(f'   {c:9s} game-to-game sd {sd_game:.3f} | ' + ', '.join(f'{k[6:]} {f.params[k]:+.3f}{"*" if f.pvalues[k] < 0.01 else " "}' for k in X.columns) + f' | R2 {f.rsquared:.3f}')
