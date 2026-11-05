"""Player availability for every D1 team-game (2026-10-09 audit fix C1) and venue context (fix C5).

data/cbb_availability.parquet, one row per (game_id, team):
  abs_val       points of margin the team is expected to lose from regulars who did not play: sum over missing regulars of
                projected minutes x (his shrunk per-minute Game Score - replacement level). Fitted to margin in game_model.py.
  abs_val_cont  the part from regulars who ALSO missed the team's previous game (likely known before the line opened)
  n_missing, miss_names
data/cbb_player_values.csv: each player's latest value (same units), for predict.py --out-home / --out-away
  A regular = played in 2+ of the team's previous 4 games, averaging 15+ minutes in them, this season.
  Pre-game only: each player's rating and projected minutes use games before this one.
data/cbb_venue.parquet: game_id, venue_state, home_state_h, home_state_a (each team's usual home-arena state).
usage: python3 prep_availability.py <hoopR dir> data"""
import os, sys, numpy as np, pandas as pd
# CBB_LAST=2027 adds the in-progress season once hoopR posts its box files (missing years are skipped)
LAST = int(os.environ.get('CBB_LAST', 2026))
YEARS = lambda a: [y for y in range(a, LAST + 1) if y < 2027 or os.path.exists(f'{sys.argv[1]}/team_box_{y}.parquet')]
d, out = sys.argv[1:3]
COLS = ['game_id', 'season', 'game_date', 'athlete_id', 'athlete_display_name', 'team_id', 'minutes', 'points', 'field_goals_made', 'field_goals_attempted',
        'free_throws_made', 'free_throws_attempted', 'offensive_rebounds', 'defensive_rebounds', 'steals', 'assists', 'blocks', 'fouls', 'turnovers']
B = []
for y in YEARS(2012):
    b = pd.read_parquet(f'{d}/player_box_{y}.parquet', columns=COLS)
    B.append(b)
B = pd.concat(B, ignore_index=True).dropna(subset=['athlete_id'])
num = COLS[6:]
B[num] = B[num].apply(pd.to_numeric, errors='coerce').fillna(0)
B['date'] = pd.to_datetime(B.game_date)
B['gms'] = (B.points + 0.4 * B.field_goals_made - 0.7 * B.field_goals_attempted - 0.4 * (B.free_throws_attempted - B.free_throws_made)
            + 0.7 * B.offensive_rebounds + 0.3 * B.defensive_rebounds + B.steals + 0.7 * B.assists + 0.7 * B.blocks - 0.4 * B.fouls - B.turnovers)
B = B.drop_duplicates(['game_id', 'athlete_id'])
P = B[B.minutes > 0].sort_values(['athlete_id', 'date', 'game_id']).reset_index(drop=True)
# replacement level: per-minute Game Score of players averaging under 12 minutes in a season
sea = P.groupby(['athlete_id', 'season']).minutes.transform('mean')
REPL = float(P.gms[sea < 12].sum() / P.minutes[sea < 12].sum())
print('replacement Game Score per minute %.4f; all players %.4f' % (REPL, P.gms.sum() / P.minutes.sum()))
K, HL_R, HL_M, CARRY = 150.0, 20.0, 4.0, 0.5
dr, dm = 0.5 ** (1 / HL_R), 0.5 ** (1 / HL_M)
pid = P.athlete_id.values; sy = P.season.values; g_ = P.gms.values; m_ = P.minutes.values
RATE = np.empty(len(P)); MIN = np.empty(len(P))
S = SM = 0.0; EM = EW = 0.0; last = None; ls = None
for i in range(len(P)):
    if pid[i] != last: S = SM = EM = EW = 0.0; last = pid[i]; ls = sy[i]
    elif sy[i] != ls: S *= CARRY; SM *= CARRY; EM *= 0.3; EW *= 0.3; ls = sy[i]
    RATE[i] = (S + K * REPL) / (SM + K); MIN[i] = EM / EW if EW > 0 else np.nan
    S = dr * S + g_[i]; SM = dr * SM + m_[i]; EM = dm * EM + m_[i]; EW = dm * EW + 1
P['rate'] = RATE; P['proj_min'] = MIN
# state AFTER each game = the pre-game state of his next game that season (for his last game of a season: this game's)
P['rate_after'] = P.groupby('athlete_id').rate.shift(-1); P['min_after'] = P.groupby('athlete_id').proj_min.shift(-1)
newseason = P.season.ne(P.groupby('athlete_id').season.shift(-1))
P.loc[newseason, ['rate_after', 'min_after']] = np.nan
P['rate_after'] = P.rate_after.fillna(P.rate); P['min_after'] = P.min_after.fillna(P.proj_min).fillna(P.minutes)
tgames = B[['game_id', 'team_id', 'season', 'date']].drop_duplicates(['game_id', 'team_id']).sort_values(['team_id', 'date', 'game_id'])
mins_d = {(g, a): m for g, a, m in zip(P.game_id.values, P.athlete_id.values, P.minutes.values)}
val_d = {(g, a): (mp * (r - REPL), nm) for g, a, r, mp, nm in zip(P.game_id.values, P.athlete_id.values, P.rate_after.values, P.min_after.values, P.athlete_display_name.values)}
played = {}
for g, t, a in zip(P.game_id.values, P.team_id.values, P.athlete_id.values): played.setdefault((g, t), set()).add(a)
rows = []
for (team, season), tgm in tgames.groupby(['team_id', 'season'], sort=False):
    hist = []   # [(game_id, set of players who played)]
    for gid in tgm.game_id.tolist():
        today = played.get((gid, team), set())
        cand = {}
        for gg, h in hist[-4:]:
            for a in h: cand.setdefault(a, []).append((gg, mins_d[(gg, a)]))
        val = valc = 0.0; names = []
        prev = hist[-1][1] if hist else set()
        if len(today) >= 5:          # skip games with a broken box score
            for a, lst in cand.items():
                if len(lst) >= 2 and np.mean([m for _, m in lst]) >= 15 and a not in today:
                    v, nm = val_d[(lst[-1][0], a)]
                    val += v; names.append(str(nm))
                    if a not in prev: valc += v
        rows.append((gid, team, val, valc, len(names), ';'.join(names)))
        hist.append((gid, today))
A = pd.DataFrame(rows, columns=['game_id', 'team', 'abs_val', 'abs_val_cont', 'n_missing', 'miss_names'])
# latest value per player (for predict.py --out-home/--out-away): points of margin his absence costs, from his state
# after his most recent game
last = P.groupby('athlete_id').tail(1)
pv = pd.DataFrame({'athlete_id': last.athlete_id.astype('int64'), 'name': last.athlete_display_name, 'team_id': last.team_id, 'season': last.season,
                   'last_game': last.date.dt.date, 'proj_min': last.min_after.round(1), 'value': (last.min_after * (last.rate_after - REPL)).round(3)})
pv.to_csv(f'{out}/cbb_player_values.csv', index=False)
print('player values saved', len(pv))
A.to_parquet(f'{out}/cbb_availability.parquet')
print('team-games', len(A), 'with a regular missing', int((A.n_missing > 0).sum()), 'mean value lost when missing %.2f' % A.abs_val[A.n_missing > 0].mean())
# venue context
S_ = pd.concat([pd.read_parquet(f'{d}/mbb_schedule_{y}.parquet', columns=['game_id', 'season', 'neutral_site', 'venue_address_state', 'home_id', 'away_id'])
                for y in YEARS(2012)], ignore_index=True)
S_['game_id'] = pd.to_numeric(S_.game_id); S_['home_id'] = pd.to_numeric(S_.home_id, errors='coerce'); S_['away_id'] = pd.to_numeric(S_.away_id, errors='coerce')
hs = S_[~S_.neutral_site.astype(bool) & S_.venue_address_state.notna()].groupby('home_id').venue_address_state.agg(lambda s: s.mode().iloc[0])
S_['home_state_h'] = S_.home_id.map(hs); S_['home_state_a'] = S_.away_id.map(hs)
S_[['game_id', 'venue_address_state', 'home_state_h', 'home_state_a']].rename(columns={'venue_address_state': 'venue_state'}).drop_duplicates('game_id').to_parquet(f'{out}/cbb_venue.parquet')
print('venue rows', len(S_), 'neutral with state', int((S_.neutral_site.astype(bool) & S_.venue_address_state.notna()).sum()))
