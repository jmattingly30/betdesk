"""Soccer (3-way and over/under 2.5) devig + consensus + soft-book value test on football-data.co.uk (2012-13 to 2025-26,
18 leagues). python3 soccer_test.py -> results/soccer_devig.csv, soccer_consensus.csv, soccer_value.csv

1. Devig: which method makes Pinnacle / bet365 / average closing 1X2 and O/U 2.5 most accurate (log loss)?
2. Consensus: Pinnacle alone vs equal-weight books vs J's capped sharp weights (devig.WEIGHT style, Pinnacle 1.5).
3. Kaunitz-style value: bet the best soft-book closing price when it beats the consensus fair price by X%.
   Opening-price version too (prices without C = opening/early in football-data)."""
import os, sys, glob, math
import warnings; warnings.filterwarnings('ignore')
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import devig as D
HERE = os.path.dirname(os.path.abspath(__file__)); OUT = f'{HERE}/results'

fr = []
for f in sorted(glob.glob(f'{HERE}/data/fd/*.csv')):
    try: d = pd.read_csv(f, encoding='latin-1', on_bad_lines='skip')
    except Exception: continue
    if 'FTR' not in d: continue
    d['file'] = os.path.basename(f); fr.append(d)
A = pd.concat(fr, ignore_index=True); A = A[A.FTR.isin(['H', 'D', 'A'])].copy()
for c in A.columns:
    if c not in ('Div', 'Date', 'Time', 'HomeTeam', 'AwayTeam', 'FTR', 'HTR', 'Referee', 'file'): A[c] = pd.to_numeric(A[c], errors='coerce')
A['date'] = pd.to_datetime(A.Date, dayfirst=True, errors='coerce'); A = A.sort_values('date')
A['res'] = A.FTR.map({'H': 0, 'D': 1, 'A': 2}); A['ov'] = ((A.FTHG + A.FTAG) > 2.5).astype(int)
M = ['mult', 'add', 'power', 'shin', 'odds_ratio']
def lls(p, k): return -math.log(max(p[k], 1e-9))

def cols3(pref, close):
    c = 'C' if close else ''
    if pref == 'PS': return [f'PS{c}H', f'PS{c}D', f'PS{c}A']
    if pref == 'Avg': return [f'Avg{c}H', f'Avg{c}D', f'Avg{c}A']
    return [f'{pref}{c}H', f'{pref}{c}D', f'{pref}{c}A']

rows = []
for pref in ('PS', 'B365', 'Avg', 'BW', 'WH'):
    for close in (False, True):
        cs = cols3(pref, close)
        if not all(c in A for c in cs): continue
        d = A.dropna(subset=cs); d = d[(d[cs] > 1).all(1)]
        if len(d) < 1000: continue
        r = dict(market='1X2', book=pref, time='close' if close else 'open', n=len(d), hold=float((1 / d[cs]).sum(1).mean() - 1))
        O = d[cs].values; y = d.res.values
        for m in M: r[m] = float(np.mean([lls(D.METHODS[m](list(o)), k) for o, k in zip(O, y)]))
        rows.append(r)
# O/U 2.5
for a, b, bk, t in (('P>2.5', 'P<2.5', 'PS', 'open'), ('PC>2.5', 'PC<2.5', 'PS', 'close'), ('B365>2.5', 'B365<2.5', 'B365', 'open'),
                    ('B365C>2.5', 'B365C<2.5', 'B365', 'close'), ('Avg>2.5', 'Avg<2.5', 'Avg', 'open'), ('AvgC>2.5', 'AvgC<2.5', 'Avg', 'close')):
    if a not in A: continue
    d = A.dropna(subset=[a, b]); d = d[(d[a] > 1) & (d[b] > 1)]
    r = dict(market='OU2.5', book=bk, time=t, n=len(d), hold=float((1 / d[a] + 1 / d[b]).mean() - 1))
    O = d[[a, b]].values; y = 1 - d.ov.values   # k=0 over
    for m in M: r[m] = float(np.mean([lls(D.METHODS[m](list(o)), k) for o, k in zip(O, y)]))
    rows.append(r)
T = pd.DataFrame(rows); T.to_csv(f'{OUT}/soccer_devig.csv', index=False)
pd.set_option('display.width', 220); print(T.round(5).to_string(index=False))

# ---- consensus at the close (1X2), power devig per book, log-odds-of-each-outcome mean ----
books = {'PS': 1.5, 'B365': 1.0, 'BW': 0.9, 'WH': 0.9, '1XB': 0.9}
cs = sum([cols3(b, True) for b in books], [])
d = A.dropna(subset=cs); d = d[(d[cs] > 1).all(1)]; y = d.res.values
P = {b: np.array([D.power(list(o)) for o in d[cols3(b, True)].values]) for b in books}
def comb(ws):
    L = sum(w * np.log(P[b]) for b, w in ws.items()) / sum(ws.values()); E = np.exp(L); return E / E.sum(1, keepdims=True)
def LL(p): return float(-np.mean(np.log(p[np.arange(len(y)), y])))
crow = dict(n=len(d), pinnacle=LL(P['PS']), bet365=LL(P['B365']), equal=LL(comb({b: 1 for b in books})), capped_sharp=LL(comb(books)),
            soft_only=LL(comb({b: 1 for b in books if b != 'PS'})))
print('\nconsensus close 1X2:', {k: round(v, 5) if isinstance(v, float) else v for k, v in crow.items()})
pd.DataFrame([crow]).to_csv(f'{OUT}/soccer_consensus.csv', index=False)

# ---- Kaunitz-style value: best soft price vs capped consensus fair (both at close, and both at open) ----
vrows = []
for close in (True, False):
    cs = sum([cols3(b, close) for b in books], [])
    d = A.dropna(subset=cs); d = d[(d[cs] > 1).all(1)]; y = d.res.values
    P = {b: np.array([D.power(list(o)) for o in d[cols3(b, close)].values]) for b in books}
    L = sum(w * np.log(P[b]) for b, w in books.items()) / sum(books.values()); F = np.exp(L); F /= F.sum(1, keepdims=True)
    soft = np.max(np.stack([d[cols3(b, close)].values for b in books if b != 'PS']), axis=0)   # best soft price per outcome
    ev = F * soft - 1
    for thr in (0.0, 0.02, 0.04, 0.06, 0.10):
        sel = ev > thr; k = np.where(sel)
        won = (y[k[0]] == k[1]); pnl = np.where(won, soft[k] - 1, -1.0)
        vrows.append(dict(time='close' if close else 'open', ev_min=thr, bets=int(sel.sum()), roi=float(pnl.mean()) if len(pnl) else None,
                          se=float(pnl.std() / math.sqrt(len(pnl))) if len(pnl) > 1 else None, exp_ev=float(ev[sel].mean()) if len(pnl) else None))
V = pd.DataFrame(vrows); V.to_csv(f'{OUT}/soccer_value.csv', index=False); print('\nvalue bets (best soft price vs consensus):'); print(V.round(4).to_string(index=False))
