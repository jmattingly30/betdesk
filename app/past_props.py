"""Earlier-day props for Bet Desk (J 2026-10-18: log a bet placed without the Desk, after its day dropped off the slate).
python3 past_props.py      # run at the START of every daily refresh, before any props file is replaced

Copies every NHL and NBA player whose game day (US Eastern) is before today out of the current data/{sport}_props.json
into data/past_props.json, keeping the last 3 days. These are the projections exactly as the Desk showed them before
the game, so a bet priced from them is priced as it would have been then. The app loads the file only when J taps
"Earlier" on the day bar."""
import json, os, datetime as dt
from zoneinfo import ZoneInfo
HERE = os.path.dirname(os.path.abspath(__file__)); D = f'{HERE}/data'; OUT = f'{D}/past_props.json'
TODAY = dt.datetime.now(ZoneInfo('America/New_York')).date(); LO = (TODAY - dt.timedelta(days=3)).isoformat(); TODAY = TODAY.isoformat()
old = json.load(open(OUT)) if os.path.exists(OUT) else {}
J = dict(made=dt.datetime.now(dt.timezone.utc).strftime('%Y-%m-%d %H:%M UTC'))
for sp in ('NHL', 'NBA'):
    keep = {(p['g'], p['n']): p for p in old.get(sp, {}).get('players', []) if LO <= (p.get('d') or '') < TODAY}
    f = f'{D}/{sp.lower()}_props.json'
    if os.path.exists(f):
        for p in json.load(open(f))['players']:
            if LO <= (p.get('d') or '') < TODAY: keep[(p['g'], p['n'])] = p
    if keep: J[sp] = dict(players=sorted(keep.values(), key=lambda p: (p['d'], str(p['g']))))
    print(sp, len(keep), 'earlier-day players', sorted({p['d'] for p in keep.values()}))
json.dump(J, open(OUT, 'w'), separators=(',', ':'))
print('wrote', OUT, os.path.getsize(OUT) // 1024, 'KB')
