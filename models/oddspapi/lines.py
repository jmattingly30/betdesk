"""Main game lines per book from the OddsPapi history (op/us/hist_<lg>.jsonl) -> op/us/lines_<lg>.csv
  python3 lines.py cbb [nba nhl nfl ...]
One row per fixture and book: p1/p2 = OddsPapi participant1/participant2 (participant1 is the home team except at
neutral sites); spread = participant1's handicap. For spread and total: the OPENING main line (of the lines first
posted, the one priced closest to even) and the CLOSING main line (of the lines still up at the start, closest to
even), each with both prices and seconds before the start; moneyline open/close. Prices decimal."""
import os, sys, json
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__)); OP = os.path.normpath(f'{HERE}/../../op'); US = f'{OP}/us'
SPORT = {'cbb': 11, 'nba': 11, 'nhl': 15, 'nfl': 14, 'cfb': 14, 'cfbp': 14, 'mlb': 13}


def meta(sp):
    M = {}
    for m in json.load(open(f'{OP}/markets.json')):
        if m['sportId'] != sp or m['playerProp'] or m['period'] not in ('result', 'fulltime'): continue
        if m['marketType'] in ('spreads', 'totals') or m['marketName'].startswith('Winner'):
            M[str(m['marketId'])] = (m['marketType'] if not m['marketName'].startswith('Winner') else 'ml', m['handicap'],
                                     {str(o['outcomeId']): o['outcomeName'] for o in m['outcomes']})
    return M


def main_line(cands, when):
    """cands: [(hcp, [a_hist, b_hist])] with hist = [open, d24, close, n, alive?] -> (hcp, a, b, secs) closest to even."""
    k = 0 if when == 'open' else 2
    pool = []
    for h, a, b in cands:
        if when == 'open':
            pool.append((h, a[0], b[0], max(a[0][0], b[0][0])))
        else:
            alive = (a[4] if len(a) > 4 else 1) and (b[4] if len(b) > 4 else 1)
            pool.append((h, a[2], b[2], min(a[2][0], b[2][0]), alive))
    if not pool: return None
    if when == 'open':
        first = max(p[3] for p in pool); pool = [p for p in pool if p[3] >= first - 1800]
    else:
        alive = [p for p in pool if p[4]]
        latest = min(p[3] for p in pool); pool = alive or [p for p in pool if p[3] <= latest + 1800]
    h, a, b, secs = min(pool, key=lambda p: abs(1 / p[1][1] - 1 / p[2][1]))[:4]
    return h, a[1], b[1], secs


def run(lg):
    M = meta(SPORT[lg]); rows = []
    for line in open(f'{US}/hist_{lg}.jsonl'):
        try: r = json.loads(line)
        except ValueError: continue
        if 'err' in r: continue
        for bk, bm in r['books'].items():
            c = {'spreads': [], 'totals': [], 'ml': None}
            for mid, outs in bm.items():
                if mid not in M: continue
                typ, h, on = M[mid]
                o = {on[k]: v.get('0') for k, v in outs.items() if k in on and v.get('0')}
                if typ == 'ml':
                    if '1' in o and '2' in o: c['ml'] = (o['1'], o['2'])
                elif typ == 'spreads' and '1' in o and '2' in o: c['spreads'].append((h, o['1'], o['2']))
                elif typ == 'totals' and 'Over' in o and 'Under' in o: c['totals'].append((h, o['Over'], o['Under']))
            row = dict(fid=r['fid'], start=r['start'], p1=r['home'], p2=r['away'], book=bk)
            for typ in ('spreads', 'totals'):
                for when in ('open', 'close'):
                    x = main_line(c[typ], when)
                    if x: row.update({f'{typ[:-1]}_{when}': x[0], f'{typ[:-1]}_{when}_a': x[1], f'{typ[:-1]}_{when}_b': x[2], f'{typ[:-1]}_{when}_secs': x[3]})
            if c['ml']:
                a, b = c['ml']; row.update(ml_open_1=a[0][1], ml_open_2=b[0][1], ml_close_1=a[2][1], ml_close_2=b[2][1], ml_open_secs=max(a[0][0], b[0][0]))
            if len(row) > 5: rows.append(row)
    D = pd.DataFrame(rows); D.to_csv(f'{US}/lines_{lg}.csv', index=False)
    print(lg, len(D), 'rows', D.fid.nunique(), 'games', D.groupby('book').spread_open.count().to_dict())


if __name__ == '__main__':
    for lg in sys.argv[1:]: run(lg)
