"""Project NHL player props for each team's next scheduled game -> Bet Desk JSON.

usage: python3 project.py <dl> <out_json> [days_ahead=2]
<dl> needs pb_2024..pb_2027.parquet (fetch_data.py) and sched_2027.parquet:
  https://github.com/sportsdataverse/sportsdataverse-data/releases/download/nhl_schedules/nhl_schedule_2027.parquet
Players: everyone who dressed in the team's most recent game (skaters), plus goalies who appeared in its last
10 games (each goalie's line assumes he starts; check the confirmed starter). Projections assume the player plays.
"""
import sys, json, pickle, datetime as dt
import numpy as np, pandas as pd
sys.path.insert(0, '.')
from build_dataset import load_raw, build
from model import SK_STATS, nb_p_over, temp

dl, out_json = sys.argv[1], sys.argv[2]
days = int(sys.argv[3]) if len(sys.argv) > 3 else 2
M = pickle.load(open('data/model.pkl', 'rb')); models, cal = M['models'], M['cal']
raw = load_raw(dl)
raw['game_date'] = raw.game_date.astype(str)
sched = pd.read_parquet(f'{dl}/sched_2027.parquet')
sched = sched[(sched.game_type == 'R') & (sched.game_state.isin(['FUT', 'PRE']))]
today = dt.date.today().isoformat(); last = (dt.date.today() + dt.timedelta(days=days)).isoformat()
sched = sched[(sched.game_date >= today) & (sched.game_date <= last)].sort_values('game_date')
# each team's next game; keep a game only if it is the next game for BOTH teams (otherwise the opponent's
# pre-game form is not known yet, e.g. the second night of a back-to-back: rerun after the first game)
first = {}
for _, g in sched.iterrows():
    for t in (g.home_team_abbr, g.away_team_abbr):
        first.setdefault(t, g.game_id)
nxt = {}
for _, g in sched.iterrows():
    if first[g.home_team_abbr] == g.game_id and first[g.away_team_abbr] == g.game_id:
        nxt[g.home_team_abbr] = g; nxt[g.away_team_abbr] = g
rows = []
raw = raw.sort_values('game_date')
for t, g in nxt.items():
    tr = raw[raw.team_abbrev == t]
    if tr.empty:
        continue
    last_gid = tr.game_id.iloc[-1]
    sk = tr[(tr.game_id == last_gid) & tr.saves.isna() & tr.position.notna()]
    recent = tr.game_id.drop_duplicates().iloc[-10:]
    gk = tr[tr.game_id.isin(recent) & (tr.saves.notna() | tr.position.isna())].drop_duplicates('player_id', keep='last')
    for _, p in pd.concat([sk, gk]).iterrows():
        r = p.copy()
        for c in ['goals', 'assists', 'points', 'shots_on_goal', 'blocked_shots', 'hits', 'power_play_goals', 'pim', 'shots_against', 'goals_against']:
            r[c] = np.nan
        is_g = pd.notna(p.saves) or pd.isna(p.position)
        r['saves'] = np.nan; r['position'] = None if is_g else p.position
        r['game_id'] = g.game_id; r['game_date'] = g.game_date; r['home_away'] = 'home' if g.home_team_abbr == t else 'away'
        r['toi'] = None; r['future'] = True; r['season'] = 20262027
        rows.append(r)
fut = pd.DataFrame(rows)
raw['future'] = False
d = build(pd.concat([raw, fut], ignore_index=True))
f = d[d.game_id.isin(fut.game_id) & d.toi.eq(0)].copy() if False else d[d.game_id.isin(set(fut.game_id))].copy()
f['home'] = (f.home_away == 'home').astype(int) if 'home_away' in f else f.home
# goalies flagged by position G
sk = f[f.pos != 'G'].copy(); gl = f[f.pos == 'G'].copy()
from model import SK_FEATS, G_FEATS, LINES
for s in SK_STATS:
    sk[f'pred_{s}'] = models[s].predict(sk[SK_FEATS])
sk['pred_toi'] = models['toi'].predict(sk[SK_FEATS])
gl['pred_saves'] = models['saves'].predict(gl[G_FEATS]) if len(gl) else []
NAMES = {'sog': 'Shots on goal', 'points': 'Points', 'goals': 'Goals', 'assists': 'Assists', 'blk': 'Blocked shots', 'hits': 'Hits', 'saves': 'Saves'}
# ladders run past the usual book lines so alt lines are priced from the model, not extrapolated (Bet Desk audit C3, 2026-10-17)
LN = {'sog': np.arange(0.5, 10, 1), 'points': np.arange(0.5, 5, 1), 'goals': np.arange(0.5, 4, 1), 'assists': np.arange(0.5, 4, 1), 'blk': np.arange(0.5, 8, 1),
      'hits': np.arange(0.5, 10, 1), 'saves': np.arange(12.5, 46, 1)}


def stat_block(mu, s):
    a, t = cal[s]['alpha'], cal[s]['temp']
    c = [[float(L), round(float(temp(nb_p_over(mu, a, L), t)), 4)] for L in LN[s]]
    # 10/50/90 percentiles from the NB
    from scipy.stats import nbinom, poisson
    if a < 1e-4:
        q = poisson.ppf([0.1, 0.5, 0.9], mu)
    else:
        n = 1 / a; q = nbinom.ppf([0.1, 0.5, 0.9], n, n / (n + mu))
    return dict(m=round(float(mu), 2), q=[float(x) for x in q], c=c)


players = []
names = raw.drop_duplicates('player_id', keep='last').set_index('player_id').player_name
for _, r in sk.iterrows():
    players.append(dict(n=names.get(r.player_id, str(r.player_id)), t=r.team, o=r.opp, p=r.position, g=str(r.game_id), h=int(r.home), d=str(r.game_date)[:10],
                        toi=round(float(r.pred_toi), 1), s={s: stat_block(r[f'pred_{s}'], s) for s in SK_STATS}))
for _, r in gl.iterrows():
    players.append(dict(n=names.get(r.player_id, str(r.player_id)) + ' (G)', t=r.team, o=r.opp, p='G', g=str(r.game_id), h=int(r.home), d=str(r.game_date)[:10],
                        s={'saves': stat_block(r.pred_saves, 'saves')}))
games = sorted({f"{g.game_date} {g.away_team_abbr} @ {g.home_team_abbr}" for g in nxt.values()})
assert not any(p['o'] != p['o'] for p in players), 'missing opponent'
J = dict(sport='NHL', season='2026-27', week=f'{today} to {last}', made=dt.datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC'),
         note="Assumes the player plays. Skaters: everyone dressed in the team's last game. Goalie lines assume that goalie starts; check the confirmed starter.",
         stat_names=NAMES, games=games, players=players)
json.dump(J, open(out_json, 'w'), separators=(',', ':'), allow_nan=False)
print(len(players), 'players,', len(games), 'games')
