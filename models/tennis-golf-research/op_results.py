"""2026 results from OddsPapi fixtures: the player who appears in a later match of the same event won.
Checked against winners read from in-play prices (hist.jsonl)."""
import json, collections, datetime as dt
import pandas as pd
F = json.load(open('../../op/fixtures_2026.json'))
KEEP = {'ATP', 'WTA', 'Challenger', 'WTA 125K', 'ITF Men', 'ITF Women', 'Davis Cup', 'Billie Jean King Cup', 'United Cup'}
rows = []
for f in F:
    if f['categoryName'] not in KEEP or 'Doubles' in f['tournamentName'] or '/' in f['participant1Name']: continue
    if f.get('statusId') in (0, 1): continue
    rows.append(dict(fid=f['fixtureId'], tid=f['tournamentId'], tour=f['tournamentName'], cat=f['categoryName'],
                     start=pd.Timestamp(f['startTime']), ts=f.get('trueStartTime'), te=f.get('trueEndTime'), status=f.get('statusId'),
                     a=f['participant1Id'], b=f['participant2Id'], an=f['participant1Name'], bn=f['participant2Name']))
M = pd.DataFrame(rows).sort_values('start').reset_index(drop=True)
M['ts'] = pd.to_datetime(M.ts, errors='coerce'); M['te'] = pd.to_datetime(M.te, errors='coerce')
M['mins'] = (M.te - M.ts).dt.total_seconds() / 60
# event = tournamentId + ISO week block (events recur yearly under same id)
M['ev'] = M.tid.astype(str) + '_' + (M.start.dt.tz_convert(None) - pd.to_timedelta(M.start.dt.weekday, 'D')).dt.strftime('%Y%m%d')
# allow events spanning 2 weeks: link by tid and gap <= 16 days
nxt = collections.defaultdict(list)
for i, r in M.iterrows():
    nxt[(r.tid, r.a)].append(r.start); nxt[(r.tid, r.b)].append(r.start)
def later(t, p, s):
    return any(s < x <= s + pd.Timedelta(days=16) for x in nxt[(t, p)])
W = []
for r in M.itertuples():
    la, lb = later(r.tid, r.a, r.start), later(r.tid, r.b, r.start)
    W.append(1 if la and not lb else 0 if lb and not la else None)
M['win_next'] = W
# check vs in-play price winners
H = {}
for l in open('../../op/hist.jsonl'):
    x = json.loads(l)
    if 'books' not in x: continue
    for bk in ('pinnacle', 'draftkings', 'fanduel'):
        o = x['books'].get(bk, {}).get('121', {}).get('o', {})
        if '1' in o and '2' in o and o['1'].get('last') and o['2'].get('last'):
            a, b = o['1']['last'][1], o['2']['last'][1]
            if a <= 1.06 and b >= 6: H[x['fid']] = 1; break
            if b <= 1.06 and a >= 6: H[x['fid']] = 0; break
M['win_price'] = M.fid.map(H)
c = M[M.win_next.notna() & M.win_price.notna()]
print('check vs price winners: n', len(c), 'agree', (c.win_next == c.win_price).mean().round(4))
M['win'] = M.win_price.fillna(M.win_next)
print(M.groupby('cat').agg(n=('fid', 'size'), known=('win', lambda s: s.notna().mean()), mins=('mins', 'median')).round(3).to_string())
M.to_parquet('../data/op_results_2026.parquet')
