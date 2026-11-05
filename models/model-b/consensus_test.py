"""Is the multi-book consensus close a better yardstick than Pinnacle alone? (Model B item 1: beat the consensus close)

python3 consensus_test.py   -> results/consensus_test.csv
Same player, stat, line and game priced at 2+ books (OddsPapi props 2026). Each book devigged (power), then:
single books, equal-weight log-odds mean, sharp-weighted mean (devig.WEIGHT), and the log-loss-optimal weights
fitted on the first half of dates and scored on the second half."""
import os, sys, math, itertools
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import devig as D
HERE = os.path.dirname(os.path.abspath(__file__)); R = f'{HERE}/../oddspapi/results'

def ll(p, y): p = np.clip(p, 1e-6, 1 - 1e-6); return -(y * np.log(p) + (1 - y) * np.log(1 - p))
def lg(p): p = np.clip(p, 1e-6, 1 - 1e-6); return np.log(p / (1 - p))
rows = []
for sp, f in (('NBA', 'nba_joined.parquet'), ('NHL', 'nhl_joined.parquet'), ('NFL', 'live_nfl_joined.parquet')):
    d = pd.read_parquet(f'{R}/{f}')
    if 'did_not_play' in d: d = d[~d.did_not_play.fillna(False).astype(bool)]
    d = d[d.actual.notna() & (d.actual != d.line)]
    dcol = 'date' if 'date' in d else ('game_date' if 'game_date' in d else 'fid')
    for t in ('open', 'close'):
        d['p'] = [D.power([a, b])[0] if a > 1 and b > 1 else np.nan for a, b in zip(d[f'o_{t}'], d[f'u_{t}'])]
        w = d.pivot_table(index=['fid', 'key', 'stat', 'line'], columns='book', values='p', aggfunc='first')
        d['y'] = (d.actual > d.line).astype(int); y = d.groupby(['fid', 'key', 'stat', 'line']).y.first()
        dt = d.groupby(['fid', 'key', 'stat', 'line'])[dcol].first().astype(str)
        books = [b for b in ('pinnacle', 'draftkings', 'fanduel') if b in w]
        for combo in [c for k in (2, 3) for c in itertools.combinations(books, k)]:
            sub = w[list(combo)].dropna()
            if len(sub) < 300: continue
            yy = y.loc[sub.index].values.astype(float); L = lg(sub.values)
            r = dict(sport=sp, time=t, books='+'.join(combo), n=len(sub))
            for i, b in enumerate(combo): r[b] = float(ll(sub.values[:, i], yy).mean())
            r['equal'] = float(ll(1 / (1 + np.exp(-L.mean(1))), yy).mean())
            ws = np.array([D.WEIGHT[b] for b in combo]); r['sharp_w'] = float(ll(1 / (1 + np.exp(-(L @ ws) / ws.sum())), yy).mean())
            # fitted weights (grid, sum to 1) on early half, scored late half
            order = np.argsort(dt.loc[sub.index].values, kind='stable'); h = len(order) // 2; tr, te = order[:h], order[h:]
            grid = [g for g in itertools.product(np.arange(0, 1.01, 0.1), repeat=len(combo)) if abs(sum(g) - 1) < 1e-9]
            best = min(grid, key=lambda g: ll(1 / (1 + np.exp(-(L[tr] @ np.array(g)))), yy[tr]).mean())
            r['fit_w'] = '/'.join(f'{x:.1f}' for x in best)
            r['fit_test'] = float(ll(1 / (1 + np.exp(-(L[te] @ np.array(best)))), yy[te]).mean())
            r['pin_test'] = float(ll(sub.values[te, 0], yy[te]).mean()) if combo[0] == 'pinnacle' else None
            r['equal_test'] = float(ll(1 / (1 + np.exp(-L[te].mean(1))), yy[te]).mean())
            rows.append(r)
T = pd.DataFrame(rows); T.to_csv(f'{HERE}/results/consensus_test.csv', index=False)
pd.set_option('display.width', 250); print(T.round(4).to_string(index=False))
