"""Score-engine consistency test on 2026 OddsPapi prices.
Feed the engine Pinnacle's no-vig match price + main total-games line -> solve hold level s -> price every game
handicap and the first-set winner. Compare with Pinnacle's own prices, with DK/FD prices, and with results
(settled from last in-play prices)."""
import os, json, sys
import numpy as np, pandas as pd
import iid
import datetime as dt


def ts(x):
    return dt.datetime.fromisoformat(x.replace('Z', '+00:00')).timestamp()

G = iid.GRID
D3 = np.load('../data/dist_bo3.npz')
Z, W = np.polynomial.hermite_e.hermegauss(9); W = W / W.sum()
SIG, SHIFT = 0.07, -0.03
TMAX = 40


def set1_table():
    path = '../data/set1.npy'
    if os.path.exists(path):
        return np.load(path)
    T = np.zeros((len(G), len(G)))
    for i, a in enumerate(G):
        for j, b in enumerate(G):
            x = sum(v for (ga, gb), v in iid.set_dist(a, b, True).items() if ga > gb)
            y = sum(v for (ga, gb), v in iid.set_dist(a, b, False).items() if ga > gb)
            T[i, j] = (x + y) / 2
    np.save(path, T)
    return T


S1 = set1_table()


def interp(T, pa, pb):
    pa = np.clip(pa, G[0], G[-1] - 1e-6); pb = np.clip(pb, G[0], G[-1] - 1e-6)
    fa = (pa - G[0]) / 0.005; fb = (pb - G[0]) / 0.005
    i = np.floor(fa).astype(int); j = np.floor(fb).astype(int)
    da = (fa - i)[..., None] if T.ndim == 3 else fa - i
    db = (fb - j)[..., None] if T.ndim == 3 else fb - j
    return T[i, j] * (1 - da) * (1 - db) + T[i + 1, j] * da * (1 - db) + T[i, j + 1] * (1 - da) * db + T[i + 1, j + 1] * da * db


def solve_mu(s, p):
    lo, hi = np.full_like(s, -0.5), np.full_like(s, 0.5)
    for _ in range(30):
        m = (lo + hi) / 2
        q = sum(w * interp(D3['win'], s + (m + SIG * z) / 2, s - (m + SIG * z) / 2) for z, w in zip(Z, W))
        lo = np.where(q < p, m, lo); hi = np.where(q >= p, m, hi)
    return (lo + hi) / 2


def dist(s, p):
    s = s + SHIFT
    mu = solve_mu(s, p)
    out = {k: sum(w * interp(T, s + (mu + SIG * z) / 2, s - (mu + SIG * z) / 2) for z, w in zip(Z, W))
           for k, T in (('tot', D3['tot']), ('mar', D3['mar']), ('set1', S1))}
    return out


def p_over(tot, L):
    k = np.arange(tot.shape[1])
    a = tot[:, k > L].sum(1); b = tot[:, k < L].sum(1)
    return a / (a + b)


def p_cover(mar, h):  # player 1 with handicap h covers: margin + h > 0
    k = np.arange(-TMAX, TMAX + 1)
    a = mar[:, k + h > 0].sum(1); b = mar[:, k + h < 0].sum(1)
    return a / (a + b)


def power_fair(a, b):
    ia, ib = 1 / a, 1 / b
    lo, hi = 1.0, 3.0
    for _ in range(50):
        k = (lo + hi) / 2
        if ia ** k + ib ** k < 1: hi = k
        else: lo = k
    return ia ** ((lo + hi) / 2)


def settle(B, mid_filter, a, b):
    for bk in ('pinnacle', 'draftkings', 'fanduel'):
        for mid, m in B.get(bk, {}).items():
            if not mid_filter(mid, m): continue
            o = m['o']
            if a in o and b in o:
                x, y = o[a]['last'][1], o[b]['last'][1]
                if x <= 1.06 and y >= 6: return 1
                if y <= 1.06 and x >= 6: return 0
    return None


if __name__ == '__main__':
    LEVEL = {'ATP': 'ATP', 'WTA': 'WTA', 'Challenger': 'Challenger', 'WTA 125K': 'WTA 125'}
    SLAMS = ('Australian Open', 'French Open', 'Roland Garros', 'Wimbledon', 'US Open')
    recs = [json.loads(l) for l in open('../../op/hist.jsonl')]
    rows = []
    for r in recs:
        if 'books' not in r: continue
        if r['cat'] == 'ATP' and any(x in r['tour'] for x in SLAMS): continue  # bo5 handled later
        B = r['books']; pin = B.get('pinnacle', {})
        ml = pin.get('121', {}).get('o', {})
        if not ('1' in ml and '2' in ml and ml['1'].get('close') and ml['2'].get('close')): continue
        p = power_fair(ml['1']['close'][1], ml['2']['close'][1])
        # Pinnacle main total line (closest to even)
        best = None
        for mid, m in pin.items():
            if m['t'] != 'totals-games': continue
            o = m['o']
            if not (o.get('Over', {}).get('close') and o.get('Under', {}).get('close')): continue
            f = power_fair(o['Over']['close'][1], o['Under']['close'][1])
            if best is None or abs(f - 0.5) < abs(best[1] - 0.5): best = (m['h'], f)
        if best is None: continue
        rows.append(dict(r=r, p=p, L=best[0], fo=best[1], level=LEVEL.get(r['cat'], r['cat'])))
    print('matches with Pinnacle match price + total', len(rows))
    P = np.array([x['p'] for x in rows]); L = np.array([x['L'] for x in rows]); FO = np.array([x['fo'] for x in rows])
    # solve hold level s so model P(over main line) = Pinnacle fair over
    lo, hi = np.full(len(rows), 0.45), np.full(len(rows), 0.85)
    for _ in range(25):
        m = (lo + hi) / 2
        d = dist(m, P)
        q = np.array([p_over(d['tot'][i:i + 1], L[i])[0] for i in range(len(rows))])
        lo = np.where(q < FO, m, lo); hi = np.where(q >= FO, m, hi)
    s = (lo + hi) / 2
    d = dist(s, P)
    print('hold level s: median', np.median(s).round(3), 'stuck at bounds', int(((s < 0.46) | (s > 0.84)).sum()))

    items = []
    for i, x in enumerate(rows):
        r = x['r']; B = r['books']
        # spreads
        for bk in ('pinnacle', 'draftkings', 'fanduel'):
            for mid, m in B.get(bk, {}).items():
                if m['t'] not in ('spreads-games', 'totals-games') and mid != '123': continue
                o = m['o']
                if mid == '123':
                    a, b, typ, h = '1', '2', 'first_set', 0.0
                    pm = float(d['set1'][i])
                elif m['t'] == 'spreads-games':
                    a, b, typ, h = '1', '2', 'spread', m['h']
                    pm = float(p_cover(d['mar'][i:i + 1], h)[0])
                else:
                    a, b, typ, h = 'Over', 'Under', 'total', m['h']
                    pm = float(p_over(d['tot'][i:i + 1], h)[0])
                if not (o.get(a, {}).get('close') and o.get(b, {}).get('close')): continue
                ca, cb = o[a]['close'][1], o[b]['close'][1]
                if min(ca, cb) <= 1.0: continue
                age = (ts(r['start']) - min(ts(o[a]['close'][0]), ts(o[b]['close'][0]))) / 3600
                res = settle(B, (lambda mm, M, typ=typ, h=h: (mm == '123') if typ == 'first_set' else (M['t'] == ('spreads-games' if typ == 'spread' else 'totals-games') and M['h'] == h)), a, b)
                items.append(dict(fid=r['fid'], level=x['level'], book=bk, typ=typ, h=h, ca=ca, cb=cb, p_model=pm,
                                  p_book=power_fair(ca, cb), hold=1 / ca + 1 / cb - 1, res=res, age=age, s=float(s[i]), main=(typ == 'total' and h == x['L'])))
    I = pd.DataFrame(items)
    pin = I[I.book == 'pinnacle'][['fid', 'typ', 'h', 'p_book']].rename(columns={'p_book': 'p_pin'})
    I = I.merge(pin, on=['fid', 'typ', 'h'], how='left')
    I = I[(I.age < 3) & (I.s > 0.46) & (I.s < 0.84)]
    print('items after fresh-close (<3h) and hold-solve filters', len(I))
    I.to_pickle('../../op/score_items.pkl')


    def ll(p, y):
        p = np.clip(p, 1e-4, 1 - 1e-4); return -np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))


    print('\n[A] Engine (from Pinnacle ML + total) vs Pinnacle own prices, same line')
    q = I[(I.book == 'pinnacle') & ~I.main & (I.p_book.between(0.15, 0.85))]
    print(q.groupby('typ').apply(lambda z: pd.Series(dict(n=len(z), mean_diff=(z.p_model - z.p_book).mean(), mae=(z.p_model - z.p_book).abs().mean()))).round(4).to_string())
    print('\n[B] Settled outcomes, log loss: Pinnacle fair vs engine (lines priced 15-85% at Pinnacle)')
    q = I[(I.book == 'pinnacle') & ~I.main & I.res.notna() & I.p_book.between(0.15, 0.85)]
    for typ, z in q.groupby('typ'):
        print(f'  {typ}: n={len(z)} pin {ll(z.p_book, z.res):.4f} engine {ll(z.p_model, z.res):.4f} blend {ll((z.p_book + z.p_model) / 2, z.res):.4f} | pin_pred {z.p_book.mean():.3f} eng_pred {z.p_model.mean():.3f} act {z.res.mean():.3f}')
    print('\n[C] DK/FD: hold, and bets where price beats engine fair (and Pinnacle fair) by X')
    out = []
    U = I[I.book != 'pinnacle'].copy()
    for (bk, typ), z in U.groupby(['book', 'typ']):
        for src in ('p_model', 'p_pin', 'both'):
            for th in (0.0, 0.02, 0.05):
                ev, ret, clv = [], [], []
                for side in ('a', 'b'):
                    pr = z['c' + side]
                    pm = z.p_model if side == 'a' else 1 - z.p_model
                    pp = z.p_pin if side == 'a' else 1 - z.p_pin
                    e_m, e_p = pm * pr - 1, pp * pr - 1
                    sel = (e_m > th) if src == 'p_model' else (e_p > th) if src == 'p_pin' else ((e_m > th) & (e_p > th))
                    sel = sel & z.p_pin.notna() if src != 'p_model' else sel
                    sel = sel & pr.between(1.3, 4.0)
                    zz = z[sel]
                    ev += list((e_m if src == 'p_model' else e_p)[sel])
                    clv += list((pp * pr - 1)[sel].dropna())
                    w = zz.res if side == 'a' else 1 - zz.res
                    ret += list(np.where(w.isna(), np.nan, np.where(w == 1, pr[sel] - 1, -1.0)))
                ret = np.array(ret, float); ns = int(np.sum(~np.isnan(ret)))
                out.append(dict(book=bk, typ=typ, chosen_by=src, edge_over=th, bets=len(ev), hold=z.hold.mean(),
                                pin_edge=np.mean(clv) if clv else np.nan, settled=ns, roi=np.nanmean(ret) if ns else np.nan,
                                roi_se=np.nanstd(ret) / np.sqrt(max(ns, 1))))
    O = pd.DataFrame(out)
    print(O.round(4).to_string())
    O.to_csv('../../op/score_bets.csv', index=False)
