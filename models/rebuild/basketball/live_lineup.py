"""Rebuild-owned copy of models/nba-team/lineup.py (2026-10-27) so this season's lineup feed can be built without
touching the nba-team thread's files. Same method; inputs: nba-player player_games (current), possessions from the
rebuild's teamgames_nba, RAPM from nba-team/data/rapm_by_month.parquet (read only: latest fit carries forward).
Writes data/live/{player_rates,current_players,lineup_agg}.parquet. Run from models/rebuild/basketball."""
import os, numpy as np, pandas as pd
STATS = ['pts', 'fga', 'fta', 'fg3a', 'fg3m', 'ast', 'tov', 'oreb', 'dreb', 'stl', 'blk', 'pf']
HL_RATE, HL_MIN, CARRY, K, NEW_MIN = 40, 6, 0.6, 400.0, 14.0   # half-lives in games; season carry; prior possessions; rookie minutes

pg = pd.read_parquet('../../nba-player/data/player_games.parquet')
g = pd.read_parquet('data/teamgames_nba.parquet').groupby('game_id', as_index=False).poss.first()
pg = pg[pg.game_id.isin(g.game_id)].merge(g, on='game_id')
pl = pg[pg.played & (pg['min'] > 0)].copy()
pl['on_poss'] = pl['min'] / 48 * pl.poss
pl = pl.sort_values(['pid', 'date', 'game_id']).reset_index(drop=True)
# bench-level prior: per-100 rates of players averaging under 16 minutes in a season
sea = pl.groupby(['pid', 'season']).agg(mpg=('min', 'mean'))
bench = pl.join(sea, on=['pid', 'season'])
bench = bench[bench.mpg < 16]
prior = {s: 100 * bench[s].sum() / bench.on_poss.sum() for s in STATS}
print('bench prior per 100:', {k: round(v, 1) for k, v in prior.items()})

dr, dm = 0.5 ** (1 / HL_RATE), 0.5 ** (1 / HL_MIN)
X = pl[STATS].to_numpy(float); P = pl.on_poss.to_numpy(); M = pl['min'].to_numpy()
pid = pl.pid.to_numpy(); sea_ = pl.season.to_numpy()
R = np.full((len(pl), len(STATS)), np.nan); MP = np.full(len(pl), np.nan); NP = np.zeros(len(pl))
pv = np.array([prior[s] for s in STATS]) / 100
S = np.zeros(len(STATS)); SP = 0.0; SM = 0.0; SW = 0.0; last_pid = None; last_sea = None
cur = []   # state after each player's latest game, for pricing upcoming games
def _save(i):
    cur.append(np.r_[pid[i], sea_[i], 100 * (S + K * pv) / (SP + K), (SM + NEW_MIN) / (SW + 1), SP])
for i in range(len(pl)):
    if pid[i] != last_pid:
        if last_pid is not None: _save(i - 1)
        S[:] = 0; SP = SM = SW = 0.0; last_pid = pid[i]; last_sea = sea_[i]
    elif sea_[i] != last_sea: S *= CARRY; SP *= CARRY; SM *= 0.5; SW *= 0.5; last_sea = sea_[i]
    R[i] = 100 * (S + K * pv) / (SP + K); MP[i] = (SM + NEW_MIN) / (SW + 1); NP[i] = SP
    S = dr * S + X[i]; SP = dr * SP + P[i]; SM = dm * SM + M[i]; SW = dm * SW + 1
_save(len(pl) - 1)
cur = pd.DataFrame(cur, columns=['pid', 'season_last'] + ['r_' + s for s in STATS] + ['min_proj', 'poss_hist'])
lastrow = pl.groupby('pid').tail(1).set_index('pid')[['name', 'team', 'date', 'pos']]
cur = cur.join(lastrow, on='pid'); cur['pid'] = cur.pid.astype(int); cur['season_last'] = cur.season_last.astype(int)
cur.to_parquet('data/live/current_players.parquet')
rates = pd.DataFrame(R, columns=['r_' + s for s in STATS]); rates['min_proj'] = MP; rates['poss_hist'] = NP
pl = pd.concat([pl[['game_id', 'season', 'date', 'pid', 'name', 'team', 'opp', 'min', 'starter']], rates], axis=1)
# state after the player's last game (for pricing future games)
# RAPM (N2): latest monthly fit made on or before the game date; players never seen in a fit get 0 (league average)
if os.path.exists('../../nba-team/data/rapm_by_month.parquet'):
    R = pd.read_parquet('../../nba-team/data/rapm_by_month.parquet').sort_values('fit_date'); R['fit_date'] = R.fit_date.astype('datetime64[ns]')
    pl['_d'] = pd.to_datetime(pl.date).astype('datetime64[ns]'); pl['_i'] = np.arange(len(pl))
    j = pd.merge_asof(pl.sort_values('_d'), R.rename(columns={'fit_date': '_d'}), on='_d', by='pid', direction='backward').set_index('_i').rapm
    pl['r_rapm'] = j.reindex(np.arange(len(pl))).fillna(0.0).to_numpy(); pl = pl.drop(columns=['_d', '_i'])
    latest = R.groupby('pid').tail(1).set_index('pid').rapm
    cur['r_rapm'] = cur.pid.map(latest).fillna(0.0); cur.to_parquet('data/live/current_players.parquet')
    STATS = STATS + ['rapm']
    print('RAPM joined: %.0f%% of player-games have a fitted value' % (100 * (pl.r_rapm != 0).mean()))
pl.to_parquet('data/live/player_rates.parquet')

# team schedule order, to find regulars missing from a game
tsch = pg[['game_id', 'date', 'season', 'team']].drop_duplicates().sort_values(['team', 'date'])
tsch['tidx'] = tsch.groupby(['team', 'season']).cumcount()
pl = pl.merge(tsch[['game_id', 'team', 'tidx']], on=['game_id', 'team'])
rcols = ['r_' + s for s in STATS]
def agg(df):
    # a ten-man rotation (more only if 11+ players project 15+ minutes): end-of-bench players who get in only in
    # blowouts would otherwise leak the result into the lineup
    df = df.nlargest(max(10, int((df.min_proj >= 15).sum())), 'min_proj')
    w = df.min_proj.to_numpy(); w = w / w.sum() * 5          # minutes share x 5 = players on the floor
    return pd.Series(np.r_[w @ df[rcols].to_numpy(), len(df)], index=rcols + ['n'])
# actual lineup
act = pl.groupby(['game_id', 'team']).apply(agg)
# full lineup: add regulars (played 2+ of this team's previous 5 games this season, 12+ projected minutes) who are absent today, with their latest state
# a player's next team after each date (to drop traded players from the 'missing' list)
nxt = pl.sort_values('date')[['pid', 'date', 'team']]
next_team_lookup = {p_: (d_.date.to_numpy(), d_.team.to_numpy()) for p_, d_ in nxt.groupby('pid')}
class _NT(dict):
    def get(self, key, default=None):
        p_, dt = key; ds, ts = next_team_lookup.get(p_, (None, None))
        if ds is None: return default
        j = np.searchsorted(ds, np.datetime64(dt), side='left')
        return ts[j] if j < len(ds) else default
next_team = _NT()
rows = []
by_team = {k: v for k, v in pl.groupby(['team', 'season'])}
for (team, season), d in by_team.items():
    d = d.sort_values('tidx')
    for t_, today in d.groupby('tidx'):
        if t_ == 0: continue
        recent = d[(d.tidx < t_) & (d.tidx >= t_ - 5)]
        cnt = recent.groupby('pid').size()
        prev_played = set(d[d.tidx == t_ - 1].pid)
        miss = recent[~recent.pid.isin(today.pid) & recent.pid.map(cnt).ge(2)].sort_values('tidx').groupby('pid').tail(1)
        # regulars only (12+ projected minutes), and not players who have since moved to another team
        miss = miss[(miss.min_proj >= 12) & ~miss.pid.map(lambda p: next_team.get((p, today.date.iloc[0]), team) != team)]
        if len(miss):
            miss = miss.copy(); miss['game_id'] = today.game_id.iloc[0]; miss['cont'] = ~miss.pid.isin(prev_played); rows.append(miss)
missing = pd.concat(rows, ignore_index=True)
full_in = pd.concat([pl, missing])
full = full_in.groupby(['game_id', 'team']).apply(agg)
known = pd.concat([pl, missing[~missing.cont]]).groupby(['game_id', 'team']).apply(agg)   # new absentees still expected to play
out = act.join(full, rsuffix='_full').join(known.add_suffix('_known')).reset_index()
mins = missing.groupby(['game_id', 'team']).agg(miss_min=('min_proj', 'sum'), miss_names=('name', lambda s: ';'.join(s)),
                                                 miss_min_new=('min_proj', lambda s: s[~missing.loc[s.index, 'cont']].sum()))
out = out.merge(mins.reset_index(), on=['game_id', 'team'], how='left'); out['miss_min'] = out.miss_min.fillna(0); out['miss_min_new'] = out.miss_min_new.fillna(0)
out.to_parquet('data/live/lineup_agg.parquet')
print('team-games', len(out), '| with a regular missing', int((out.miss_min > 0).sum()), '| missing 25+ projected min', int((out.miss_min >= 25).sum()))
