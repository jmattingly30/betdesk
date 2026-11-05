"""Does tennis behave like independent points? Grand Slam point-by-point 2011-2024.
Compares server point-win rate in score states against that server's own rate in the same match."""
import numpy as np, pandas as pd

d = pd.read_parquet('../data/gs_point_by_point.parquet',
                    columns=['match_id', 'SetNo', 'GameNo', 'PointNumber', 'PointWinner', 'PointServer', 'P1Score', 'P2Score',
                             'P1GamesWon', 'P2GamesWon', 'SetWinner', 'GameWinner', 'P1BreakPoint', 'P2BreakPoint', 'match_num', 'year', 'slam'])
for c in ['PointServer', 'PointWinner', 'SetNo', 'GameNo']:
    d[c] = pd.to_numeric(d[c], errors='coerce')
d = d[(d.PointServer.isin([1, 2])) & (d.PointWinner.isin([1, 2]))].copy()
d['men'] = d.match_num.astype(str).str[0] == '1'
d['mkey'] = d.match_id.astype(str)
d['sw'] = (d.PointServer == d.PointWinner).astype(int)
# server's own match rate (leave-one-out)
g = d.groupby(['mkey', 'PointServer']).sw
d['n_srv'] = g.transform('count'); d['k_srv'] = g.transform('sum')
d['base'] = (d.k_srv - d.sw) / (d.n_srv - 1)
d = d[d.n_srv >= 40]
# scores in the file are AFTER the point: take the previous point's score within the game
d['pn'] = pd.to_numeric(d.PointNumber, errors='coerce')
d = d.sort_values(['mkey', 'SetNo', 'GameNo', 'pn'])
gg = d.groupby(['mkey', 'SetNo', 'GameNo'])
p1 = gg.P1Score.shift(1).fillna('0').astype(str); p2 = gg.P2Score.shift(1).fillna('0').astype(str)
d['ss'] = np.where(d.PointServer == 1, p1, p2); d['rs'] = np.where(d.PointServer == 1, p2, p1)
tb = pd.to_numeric(d.GameNo, errors='coerce') == 13
d['state'] = np.where(tb, 'tiebreak', d.ss + '-' + d.rs)
d['prev_sw'] = d.groupby(['mkey', 'SetNo', 'GameNo']).sw.shift(1)
# was server broken in their previous service game? / did server just break?
out = []
for men, z in d.groupby('men'):
    tot = {'tour': 'ATP' if men else 'WTA', 'points': len(z), 'serve_win': z.sw.mean()}
    print(tot)
    rows = []
    for st, q in z.groupby('state'):
        if len(q) < 5000:
            continue
        rows.append({'state': st, 'n': len(q), 'actual': q.sw.mean(), 'expected': q.base.mean(), 'diff_pp': 100 * (q.sw.mean() - q.base.mean())})
    print(pd.DataFrame(rows).sort_values('diff_pp').round(4).to_string())
    for pv in (0, 1):
        q = z[z.prev_sw == pv]
        print('previous point won by server =', pv, 'n', len(q), 'diff pp', round(100 * (q.sw.mean() - q.base.mean()), 3))
d[['mkey', 'men', 'SetNo', 'GameNo', 'PointServer', 'sw', 'base', 'state']].to_parquet('../data/pbp_small.parquet')
