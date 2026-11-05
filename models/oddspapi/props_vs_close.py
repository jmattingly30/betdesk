"""Report-card file for the Bet Desk: our prop chances vs the books' closing no-vig price -> app/data/props_vs_close.json
  python3 props_vs_close.py      (reads results/*_joined.parquet written by props_backtest.py nba | nhl | live)
Rows: one per sport (stat 'All') + one per stat with n >= 100. ll_ = log loss vs the result at the closing main lines
(pushes dropped). Bets = our side at 5%+ EV at the opening price, one per player-stat-game (best book), on lines still
up at the close; clv_ = mean(fair close chance x opening price - 1), beat_close_ = share of those bets with clv > 0.
Jan-Jun 2026 = backtest models (NBA/NHL); Oct 2026 = Bet Desk snapshots (NBA/NHL/NFL)."""
import os, json, datetime as dt
import numpy as np, pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__)); R = f'{HERE}/results'
OUT = os.path.normpath(f'{HERE}/../../app/data/props_vs_close.json')

def load(sp):
    D = [pd.read_parquet(f'{R}/{f}', columns=['fid', 'book', 'stat', 'line', 'key', 'o_open', 'u_open', 'p_close', 'is_open', 'is_close', 'p_model', 'actual'])
         for f in (f'{sp}_joined.parquet', f'live_{sp}_joined.parquet') if os.path.exists(f'{R}/{f}')]
    D = pd.concat(D, ignore_index=True)
    return D[D.actual.notna() & D.p_model.between(1e-4, 1 - 1e-4) & (D.actual != D.line)].assign(y=lambda x: (x.actual > x.line).astype(int))

def ll(p, y): p = np.clip(p, 1e-4, 1 - 1e-4); return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))

def row(sp, stat, D):
    C = D[D.is_close]
    if len(C) == 0: return None
    r = dict(sp=sp.upper(), stat=stat, n=int(len(C)), ll_ours=round(ll(C.p_model, C.y), 4), ll_book=round(ll(C.p_close, C.y), 4))
    B = D[D.is_open & D.is_close]
    for side, price, pm, pc in (('over', B.o_open, B.p_model, B.p_close), ('under', B.u_open, 1 - B.p_model, 1 - B.p_close)):
        b = B.assign(ev=pm * price - 1, clv=pc * price - 1)
        b = b[b.ev >= 0.05].sort_values('ev').groupby(['fid', 'key', 'stat']).tail(1)
        r[f'bets_{side}'] = int(len(b))
        r[f'beat_close_{side}'] = round(float((b.clv > 0).mean()), 3) if len(b) else None
        r[f'clv_{side}'] = round(float(b.clv.mean()), 4) if len(b) else None
    return r

by = []
for sp in ('nba', 'nhl', 'nfl'):
    D = load(sp); by.append(row(sp, 'All', D))
    for st, g in D.groupby('stat'):
        if g.is_close.sum() >= 100: by.append(row(sp, st, g))
by = [r for r in by if r]
json.dump(dict(made=dt.datetime.now(dt.timezone.utc).isoformat(timespec='minutes'), window='2026-01-13..2026-10-26',
               note='Jan-Jun 2026 backtest models (NBA, NHL) + Oct 16-26 Bet Desk snapshots (NBA, NHL, NFL); books Pinnacle/DraftKings/FanDuel',
               by=by), open(OUT, 'w'), indent=1)
print(pd.DataFrame(by).to_string(index=False))
