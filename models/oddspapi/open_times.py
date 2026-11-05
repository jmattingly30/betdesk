"""When do books first post lines? From op/us/hist_<lg>.jsonl: per fixture and book, the earliest posting of the game
lines (spread/total/moneyline) and of player props -> clock time (Mountain, J is in Colorado) and lead before start.
  python3 open_times.py nfl nba nhl cbb"""
import os, sys, json, datetime as dt, zoneinfo
import pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__)); OP = os.path.normpath(f'{HERE}/../../op'); US = f'{OP}/us'
MT = zoneinfo.ZoneInfo('America/Denver')
M = {str(m['marketId']): m for m in json.load(open(f'{OP}/markets.json'))}

def rows(lg):
    for line in open(f'{US}/hist_{lg}.jsonl'):
        try: r = json.loads(line)
        except ValueError: continue
        if 'books' not in r: continue
        st = dt.datetime.fromisoformat(r['start'].replace('Z', '+00:00')).timestamp()
        for bk, bm in r['books'].items():
            first = {}
            for mid, outs in bm.items():
                m = M.get(mid)
                if not m or m['period'] not in ('result', 'fulltime', 'fulltime+overtime'): continue
                kind = 'prop' if m['playerProp'] else 'game' if (m['marketType'] in ('spreads', 'totals') or m['marketName'].startswith('Winner')) else None
                if not kind: continue
                for o in outs.values():
                    for h in o.values():
                        if isinstance(h, list) and h and isinstance(h[0], list): first[kind] = max(first.get(kind, 0), h[0][0])
            for kind, secs in first.items():
                t = dt.datetime.fromtimestamp(st - secs, MT)
                yield dict(lg=lg, book=bk, kind=kind, lead_h=secs / 3600, open_mt=t, hour=t.hour + t.minute / 60, dow=t.strftime('%a'),
                           game_dow=dt.datetime.fromtimestamp(st, MT).strftime('%a'))

D = pd.DataFrame([x for lg in sys.argv[1:] for x in rows(lg)])
D.to_csv(f'{US}/open_times.csv', index=False)
for (lg, kind, bk), g in D.groupby(['lg', 'kind', 'book']):
    if len(g) < 20: continue
    hb = g.hour.round().astype(int).value_counts(normalize=True).sort_index()
    top = ', '.join(f'{h:02d}:00 {p:.0%}' for h, p in hb.sort_values(ascending=False).head(4).items())
    print(f'{lg:4} {kind:4} {bk:10} n={len(g):5} lead median {g.lead_h.median():6.1f} h (p25 {g.lead_h.quantile(.25):5.1f}, p75 {g.lead_h.quantile(.75):6.1f}) | top MT hours: {top}')
