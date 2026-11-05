"""Add each player's recent and season averages to a Bet Desk props file, so the app can flag projections that sit well
above or below how the player has actually been doing (J 2026-10-15), then fold in the pre-game condition shifts.

python3 add_recent.py NFL|NBA|NHL props.json          (rewrites the file in place)

For every stat on the slate: s[stat].a = [recent average, season average, games in recent, games in season].
Recent = last 5 comparable games (NFL) or last 10 (NBA, NHL), regular season only. Season = this season once the player
has 3+ comparable games in it, else last season. The app flags a projection 15%+ away from BOTH averages (FLAG_MIN in
bet-desk.html) and needs 3+ recent games, so a player with fewer comparable games is not flagged.

Comparable games (J 2026-10-17: don't flag a backup off averages from games where he only played in relief, but do flag
a player back from injury on a full load against his pre-injury form). The workload is minutes (NBA), ice time (NHL) or
snap share (NFL). E = the workload we expect today. A past game counts when
    (his expected role going into that game, or what he actually played, is within 75%-133% of E)
    and he actually played 50%-150% of E (drops early exits and relief cameos, and games with far more work than today).
So the flag compares today's projection with games at a similar workload in both directions: a backup who starts today is
compared with his earlier starts, a starter on a reduced load today with his reduced-load games.
His expected role going into a game = median workload of his previous 5 games. Filtering on role rather than only on the
minutes he ended up playing avoids inflating the averages (tested on 2024-26 NBA). Comparable games are looked for
across the last 3 seasons, so a backup's earlier starts count and a returning starter is compared with his healthy form.
p['wl'] = {exp, unit, recent workload over all games, comparable games used, games left out}; s['a_all'] = the old averages.

Then models/parlay/conditions.py folds the condition shifts (spread, total, month, expected goals) into each mapped stat's
projection and over-curve, so flags, prices and parlays all use the adjusted numbers (s['cs'] = shift, p['cond'] = why)."""
import sys, os, re, json, unicodedata, numpy as np, pandas as pd
here = os.path.dirname(os.path.abspath(__file__)); M = os.path.normpath(f'{here}/../models')
sys.path.insert(0, f'{M}/parlay'); import conditions
LO, HI, FLOOR, CEIL, SEASONS = 0.75, 1 / 0.75, 0.50, 1.50, 3
sport, path = sys.argv[1].upper(), sys.argv[2]
doc = json.load(open(path))
def norm(s):
    s = unicodedata.normalize('NFKD', str(s)).encode('ascii', 'ignore').decode().lower().replace(' (g)', '')
    return ' '.join(re.sub(r'\b(jr|sr|ii|iii|iv)\b', '', re.sub(r'[^a-z. ]', '', s)).split())
if sport == 'NFL':
    h = pd.read_parquet(f'{M}/nfl-player/data/player_games.parquet')
    h = h[(h.season_type == 'REG') & (h.offense_snaps > 0)].copy()
    h['td'] = h.rush_td + h.rec_td; h['rush_rec_yds'] = h.rush_yds + h.rec_yds
    h['key'] = h.name.map(norm); h['tm'] = h.posteam; h['when'] = h.game_date; h['id'] = h.pid; N = 5
    h['wl'] = h.offense_pct; unit = 'snap share'
elif sport == 'NBA':
    h = pd.read_parquet(f'{M}/nba-player/data/player_games.parquet')
    h = h[h.played & (h.playoff == 0) & (h['min'] > 0)].copy(); h['pra'] = h.pts + h.reb + h.ast
    h['key'] = h.name.map(norm); h['tm'] = h.team; h['when'] = h.date; h['id'] = h.pid; N = 10
    h['wl'] = h['min']; unit = 'minutes'
else:
    h = pd.read_parquet(f'{M}/nhl-player/data/player_games.parquet')
    h = h[(h.playoff == 0) & (h.toi > 0)].copy()
    h['key'] = h.player_name.map(norm); h['tm'] = h.team; h['when'] = h.game_date; h['id'] = h.player_id; N = 10
    h['wl'] = h.toi; unit = 'minutes on ice'
h = h.sort_values('when')
h['role'] = h.groupby('id').wl.transform(lambda x: x.shift(1).rolling(5, min_periods=1).median())

def expected(p, d):
    """Today's expected workload, in the same unit as h.wl."""
    if sport == 'NBA':
        m = re.search(r'([\d.]+) min', p.get('p', '')); return float(m.group(1)) if m else None
    if sport == 'NHL':
        return 60.0 if p['p'] == 'G' else p.get('toi')
    if p['p'] == 'QB': return 1.0                       # the props file lists each team's expected starter
    s = p['s']; opp = s.get('targets', {}).get('m', 0) + (s.get('carries', {}).get('m', 0) if p['p'] == 'RB' else 0)
    q = d[d.wl >= 0.2].tail(16)
    o = q.targets.fillna(0) + (q.carries.fillna(0) if p['p'] == 'RB' else 0)
    if not len(q) or o.sum() <= 0: return None
    return float(np.clip(opp / (o.sum() / q.wl.sum()), 0.05, 1.0))   # projected opportunities / his opportunities per unit of snap share

by = {k: d for k, d in h.groupby('key')}
done = miss = 0
for p in doc['players']:
    d = by.get(norm(p['n']))
    if d is None: miss += 1; continue
    if d.id.nunique() > 1:      # two players with the same (abbreviated) name: take the one whose latest team is this team
        lastteam = d.groupby('id').tm.last(); ids = lastteam[lastteam == p['t']].index
        d = d[d.id == (ids[0] if len(ids) else d.id.iloc[-1])]
    d = d[d.season > d.season.max() - SEASONS]
    E = expected(p, d)
    near = lambda x: (x >= LO * E) & (x <= HI * E)
    ok = ((near(d.role) | near(d.wl)) & (d.wl >= FLOOR * E) & (d.wl <= CEIL * E)) if E else pd.Series(True, index=d.index)
    q = d[ok]
    cur = d.season.max()
    def season_rows(x):
        s = x[x.season == cur]
        if len(s) < 3: s = x[x.season == x[x.season < cur].season.max()] if (x.season < cur).any() else x.iloc[0:0]
        return s
    rec, sea, allr, alls = q.tail(N), season_rows(q), d.tail(N), season_rows(d)
    p['wl'] = dict(exp=round(E, 2) if E else None, unit=unit, recent_all=round(float(allr.wl.mean()), 2),
                   used=int(len(rec)), left_out=int((~ok.tail(40)).sum()))
    for s, v in p['s'].items():
        if s not in d: continue
        v['a'] = [round(float(rec[s].mean()), 2) if len(rec) else 0.0, round(float(sea[s].mean()), 2) if len(sea) else 0.0, int(len(rec)), int(len(sea))]
        v['a_all'] = [round(float(allr[s].mean()), 2), round(float(alls[s].mean()), 2) if len(alls) else 0.0, int(len(allr)), int(len(alls))]
    done += 1
moved = conditions.apply_to_props(sport, doc)
json.dump(doc, open(path, 'w'), separators=(',', ':'), allow_nan=False)
print(f'{sport}: averages added for {done} players, no history found for {miss}; condition shifts applied to {moved} stats')
