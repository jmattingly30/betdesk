"""Parse Retrosheet event files (.EVA/.EVN) into plate appearances and games.
PA row: game, date, bat_home, batter, pitcher, is_starter_pitcher, result class (K, BB, IBB, HBP, 1B, 2B, 3B, HR, OUT, ROE).
Game row: date, teams, starters, starting lineups, park, day/night, temperature, wind, home-plate umpire."""
import sys, glob, re, pandas as pd
src, out_pa, out_g = sys.argv[1:4]
NONPA = ('NP', 'BK', 'CS', 'DI', 'OA', 'PB', 'WP', 'PO', 'SB', 'FLE', 'C/', 'C')
def classify(ev):
    e = ev.split('/')[0].split('.')[0]
    if e.startswith('K'): return 'K'
    if e.startswith('IW') or (e.startswith('I') and not e.startswith('IW') and len(e) <= 2): return 'IBB'
    if e.startswith('W') and not e.startswith('WP'): return 'BB'
    if e.startswith('HP'): return 'HBP'
    if e.startswith('HR') or (e.startswith('H') and not e.startswith('HP')): return 'HR'
    if e.startswith('S') and not e.startswith('SB'): return '1B'
    if e.startswith('DGR') or (e.startswith('D') and not e.startswith('DI')): return '2B'
    if e.startswith('T'): return '3B'
    if e.startswith('E'): return 'ROE'
    if e.startswith('FC') or e[:1].isdigit(): return 'OUT'
    if e.startswith('C'): return None              # catcher interference: not counted
    return None
pas, games = [], []
for f in sorted(glob.glob(f'{src}/*.EV[AN]')):
    g = None
    for line in open(f, encoding='latin-1'):
        p = line.rstrip('\n').split(',')
        t = p[0]
        if t == 'id':
            if g: games.append(g)
            g = dict(gid=p[1], home_sp=None, away_sp=None, home_lu={}, away_lu={}); cur = {0: None, 1: None}; spid = {0: None, 1: None}
        elif t == 'info':
            if p[1] in ('visteam','hometeam','date','number','daynight','temp','windspeed','winddir','umphome','site','fieldcond','sky'):
                g[p[1]] = p[2]
        elif t in ('start', 'sub'):
            pid, team, order, pos = p[1], int(p[3]), int(p[4]), int(p[5])
            if pos == 1:
                cur[team] = pid
                if t == 'start': spid[team] = pid; g['home_sp' if team == 1 else 'away_sp'] = pid
            if t == 'start' and order > 0:
                (g['home_lu'] if team == 1 else g['away_lu'])[order] = pid
        elif t == 'play':
            ev = p[6]; c = classify(ev)
            if c is None: continue
            bt = int(p[2]); pit = cur[1 - bt]
            pas.append((g['gid'], g.get('date'), bt, p[3], pit, int(pit == spid[1 - bt]), int(p[1]), c))
    if g: games.append(g)
pa = pd.DataFrame(pas, columns=['gid','date','bat_home','batter','pitcher','vs_starter','inning','res'])
pa['date'] = pd.to_datetime(pa.date, format='%Y/%m/%d')
pa.to_parquet(out_pa)
G = pd.DataFrame(games)
for side in ('home','away'):
    for i in range(1, 10): G[f'{side}_b{i}'] = G[f'{side}_lu'].map(lambda d: d.get(i))
G = G.drop(columns=['home_lu','away_lu'])
G['date'] = pd.to_datetime(G.date, format='%Y/%m/%d')
G.to_parquet(out_g)
print(pa.shape, G.shape); print(pa.res.value_counts(normalize=True).round(4).to_dict())
