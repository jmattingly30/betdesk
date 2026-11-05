"""Line shopping, +EV, middles, arbs, low holds, bonus-bet hedges and line moves (J 2026-10-27) -> app/data/line_tools.json.

python3 line_tools.py            # after export_slates.py (reads game_slates.json ln.bk = every book's line per game)
python3 line_tools.py --landing  # rebuild data/landing.json (how often games land on each margin/total) from the archive

How the fair price is set (the OddsJam / Outlier method): the sharpest book on the board sets the no-vig price, Pinnacle
first, then BetOnline/LowVig, else every book. Its two prices are devigged (power method, same as Bet Desk devig()), then
we find the expected margin (or total) that gives that cover chance on the historical landing curve. That one number
prices every other book's line, even at a different point: +4 at -110 vs Pinnacle's +4.5 gets the value of the half point
right, including pushes and key numbers (NFL 3, 7, 10; totals 37, 41, 44, 47, 51).
Landing curve: a normal around the expected margin (spread of results grows with the line), times how much more often
each exact margin happens than the normal says, pooled over history (nflverse 2015+, CFB, NBA, CBB closing lines).

Outputs per game: every book's prices, the fair line, EV of every book side, the shopped hold, middles (both sides at two
books, priced on the landing curve, stakes set so both sides return the same), arbs (worst case >= 0), bonus-bet hedges
(bonus at one book, cash at another), and line moves since we first saw the game (data/line_history.json)."""
import json, math, os, sys, datetime as dt
from statistics import median
from zoneinfo import ZoneInfo

HERE = os.path.dirname(os.path.abspath(__file__)); D = f'{HERE}/data'
ARC = f'{HERE}/../archive/pre-audit-2026-10-09/models'
ET = ZoneInfo('America/New_York'); NOW = dt.datetime.now(dt.timezone.utc)
SHARP = ['pinnacle', 'betonlineag', 'lowvig']       # fair-price source, in order
NAMES = dict(pinnacle='Pinnacle', betonlineag='BetOnline', lowvig='LowVig', draftkings='DraftKings', fanduel='FanDuel', betmgm='BetMGM',
             williamhill_us='Caesars', espnbet='ESPN BET', fanatics='Fanatics', betrivers='BetRivers', hardrockbet='Hard Rock',
             ballybet='Bally Bet', betus='BetUS', mybookieag='MyBookie', bovada='Bovada', fliff='Fliff', betparx='BetParx')
MY_BOOKS = ['draftkings', 'fanduel', 'betmgm', 'williamhill_us', 'espnbet', 'fanatics', 'betrivers', 'ballybet', 'bet365', 'hardrockbet', 'circasports']   # J: Colorado, any legal book (2026-11-02: all CO books, no picking)
MOVE = dict(NFL=1.0, CFB=1.5, NBA=1.0, CBB=1.0)        # sharp line move worth flagging (points)
KEYS = dict(NFL=[3, 7, 10, 6, 4, 14])
MIN_EV = -0.005                                        # list middles down to -0.5% EV
W_TEAM = 0.30                                          # Bet Desk default model weight for team bets (Settings)

def dec(a): return 1 + (a / 100 if a > 0 else 100 / -a)
def am(d): return round((d - 1) * 100) if d >= 2 else round(-100 / (d - 1))
def devig(a, b):
    """Power method: p1^k + p2^k = 1 (Bet Desk devig())."""
    pa, pb = 1 / dec(a), 1 / dec(b); lo, hi = 0.5, 3.0
    for _ in range(60):
        k = (lo + hi) / 2
        if pa ** k + pb ** k > 1: lo = k
        else: hi = k
    return pa ** k / (pa ** k + pb ** k)
def Phi(x): return 0.5 * (1 + math.erf(x / math.sqrt(2)))

# ---------- landing curve ----------
def build_landing():
    import pandas as pd, numpy as np
    g = pd.read_csv(f'{ARC}/team-strength/data/nfl_games_nflverse.csv'); g = g[g.season >= 2015]
    c = pd.read_parquet(f'{ARC}/team-strength/data/cfb_games_features.parquet')
    n = pd.read_parquet(f'{ARC}/team-strength/data/nba_games_features.parquet'); n['total'] = n.home_score + n.away_score
    b = pd.read_parquet(f'{ARC}/cbb/data/cbb_games.parquet').merge(pd.read_parquet(f'{ARC}/cbb/data/cbb_lines.parquet'), on='game_id')
    src = dict(NFL=(g, 'result', 'spread_line', 'total', 'total_line'), CFB=(c, 'margin', 'line_close', 'total', 'total_close'),
               NBA=(n, 'margin', 'line', 'total', 'total_line'), CBB=(b, 'margin', 'line', 'total', 'total_line'))
    out = {}
    for sp, (df, m, l, t, tl) in src.items():
        o = {}
        for kind, y, x, sym in (('m', m, l, False), ('t', t, tl, False)):
            d = df[[y, x]].dropna(); d = d[(d[x] % 0.5 == 0)]
            yv, xv = d[y].round().astype(int).values, d[x].values
            if kind == 'm':   # margins seen from the favourite's side: favourites win by 3 more often than underdogs do
                sg = np.where(xv < 0, -1, 1); yv, xv = yv * sg, np.abs(xv); sym = False
            r = yv - xv
            ax = np.abs(xv) if kind == 'm' else xv
            A = np.vstack([np.ones_like(ax), ax]).T; beta = np.linalg.lstsq(A, r ** 2, rcond=None)[0]   # variance grows with the line
            sd = np.sqrt(np.clip(A @ beta, 25, None))
            lo, hi = (0, 75) if sym else (int(yv.min()), int(yv.max()))
            O, E = np.zeros(hi - lo + 1), np.zeros(hi - lo + 1)
            for k in range(lo, hi + 1):
                if sym:
                    O[k - lo] = np.sum(np.abs(yv) == k)
                    e = Phi_v(k + .5, xv, sd) - Phi_v(k - .5, xv, sd)
                    if k: e = e + Phi_v(-k + .5, xv, sd) - Phi_v(-k - .5, xv, sd)
                else:
                    O[k - lo] = np.sum(yv == k); e = Phi_v(k + .5, xv, sd) - Phi_v(k - .5, xv, sd)
                E[k - lo] = e.sum()
            mult = (O + 3) / (E + 3)                     # shrink thin cells toward 1
            if kind == 'm': mult[-lo] = O[-lo] / max(E[-lo], 1e-9)   # ties: NBA/CBB never, NFL rare
            o[kind] = dict(sd=[round(float(beta[0]), 2), round(float(beta[1]), 3)], lo=lo, mult=[round(float(v), 3) for v in mult], n=int(len(d)))
        out[sp] = o
    out['made'] = NOW.strftime('%Y-%m-%d'); out['note'] = ('P(result = k) = normal(k; expected, sd) x mult, renormalised. '
        'm: home margin, mult by margin from the favourite side (home margin x -1 when the home side is the underdog, mu < 0) from lo; t: total, mult by total from lo. sd^2 = sd[0] + sd[1] x |line|.')
    json.dump(out, open(f'{D}/landing.json', 'w'), separators=(',', ':'))
    for sp in src: m = out[sp]['m']; print(sp, 'margin n', m['n'], 'fav by 3/7:', m['mult'][3 - m['lo']], m['mult'][7 - m['lo']], 'dog by 3:', m['mult'][-3 - m['lo']], 'total n', out[sp]['t']['n'])
    return out

def Phi_v(x, mu, sd):
    import numpy as np
    return 0.5 * (1 + np.vectorize(math.erf)((x - mu) / (sd * math.sqrt(2))))

LAND = None
def pmf(sp, kind, mu):
    """{k: P(result = k)} for an expected home margin (kind 'm') or total ('t') of mu."""
    L = LAND[sp][kind]; v = max(L['sd'][0] + L['sd'][1] * abs(mu), 25); sd = math.sqrt(v)
    out = {}
    for k in range(int(mu - 6 * sd), int(mu + 6 * sd) + 2):
        i = (k * (-1 if mu < 0 else 1) if kind == 'm' else k) - L['lo']
        m = L['mult'][i] if 0 <= i < len(L['mult']) else 1.0
        out[k] = (Phi((k + .5 - mu) / sd) - Phi((k - .5 - mu) / sd)) * m
    s = sum(out.values()); return {k: p / s for k, p in out.items()}

def wpl(P, a, side):
    """Win, push, loss chances. side 'H': home margin + a > 0. 'A': away, margin < a (a = away point). 'O': total > a. 'U': total < a."""
    w = p = 0.0
    for k, q in P.items():
        x = {'H': k + a, 'A': a - k, 'O': k - a, 'U': a - k}[side]
        if x > 0: w += q
        elif x == 0: p += q
    return w, p, 1 - w - p

def solve(sp, kind, side, a, fair):
    """Expected margin/total whose no-push win chance for this side at point a equals the devigged fair chance."""
    lo, hi = (-60, 60) if kind == 'm' else (a - 60, a + 60)
    for _ in range(40):
        mu = (lo + hi) / 2; w, p, l = wpl(pmf(sp, kind, mu), a, side)
        if w / max(w + l, 1e-9) < fair: lo, hi = (mu, hi) if side in 'HO' else (lo, mu)
        else: lo, hi = (lo, mu) if side in 'HO' else (mu, hi)
    return (lo + hi) / 2

# ---------- per game ----------
def fair_lines(sp, bk):
    """Expected margin, expected total and fair home ML chance from the sharpest books present."""
    def from_books(keys):
        ms, ts, mls = [], [], []
        for b in keys:
            r = bk.get(b)
            if not r: continue
            if r[0] is not None and r[1] and r[2]: ms.append(solve(sp, 'm', 'H', r[0], devig(r[1], r[2])))
            if r[3] is not None and r[4] and r[5]: ts.append(solve(sp, 't', 'O', r[3], devig(r[4], r[5])))
            if r[6] and r[7]: mls.append(devig(r[6], r[7]))
        return ms, ts, mls
    for keys, lab in (([SHARP[0]], NAMES[SHARP[0]]), (SHARP[1:], 'BetOnline/LowVig'), (list(bk), 'all books')):
        ms, ts, mls = from_books(keys)
        if ms or ts or mls: break
    # a sharp book may miss a market: fall back per market to every book
    am_, at_, aml = from_books(list(bk))
    f = dict(src=lab, mu=sum(ms) / len(ms) if ms else (median(am_) if am_ else None), mt=sum(ts) / len(ts) if ts else (median(at_) if at_ else None),
             pml=sum(mls) / len(mls) if mls else (median(aml) if aml else None))
    f['cons_mu'] = median(am_) if am_ else None; f['cons_mt'] = median(at_) if at_ else None
    return f

def sides(r):
    """(market, side, point, price) for one book row [hPt, hPr, aPr, tPt, oPr, uPr, mlH, mlA]."""
    out = []
    if r[0] is not None:
        if r[1]: out.append(('spread', 'H', r[0], r[1]))
        if r[2]: out.append(('spread', 'A', -r[0], r[2]))
    if r[3] is not None:
        if r[4]: out.append(('total', 'O', r[3], r[4]))
        if r[5]: out.append(('total', 'U', r[3], r[5]))
    if r[6]: out.append(('ml', 'H', None, r[6]))
    if r[7]: out.append(('ml', 'A', None, r[7]))
    return out

def chances(sp, f, mk, side, pt, Pm, Pt):
    if mk == 'ml':
        if f['pml'] is None: return None
        return (f['pml'], 0.0, 1 - f['pml']) if side == 'H' else (1 - f['pml'], 0.0, f['pml'])
    P = Pm if mk == 'spread' else Pt
    return None if P is None else wpl(P, pt, side)

def middle(o1, o2, P):
    """Two opposite bets (side, point, price) at two books on one result curve. Stakes set so both sides return the same
    (100 total). Returns chance both win, EV, worst loss, best case."""
    (s1, a1, p1), (s2, a2, p2) = o1, o2; d1, d2 = dec(p1), dec(p2)
    x1 = 100 * d2 / (d1 + d2); x2 = 100 - x1
    def res(side, a, k):
        x = {'H': k + a, 'A': a - k, 'O': k - a, 'U': a - k}[side]; return 1 if x > 0 else (0 if x == 0 else -1)
    ev = both = 0.0; outs = []
    for k, q in P.items():
        r1, r2 = res(s1, a1, k), res(s2, a2, k)
        g = (x1 * (d1 - 1) if r1 > 0 else (0 if r1 == 0 else -x1)) + (x2 * (d2 - 1) if r2 > 0 else (0 if r2 == 0 else -x2))
        ev += q * g; outs.append(g)
        if r1 > 0 and r2 > 0: both += q
    return dict(both=both, ev=ev / 100, worst=min(outs) / 100, best=max(outs) / 100, stake=[round(x1, 1), round(x2, 1)])

def logit(p): p = min(max(p, 1e-6), 1 - 1e-6); return math.log(p / (1 - p))
def blend(pm, pf, w=W_TEAM): return 1 / (1 + math.exp(-(w * logit(pm) + (1 - w) * logit(pf))))

def game_tools(sp, g, hist, model=None):
    bk = {b: r for b, r in g['ln']['bk'].items() if r}
    f = fair_lines(sp, bk)
    Pm = pmf(sp, 'm', f['mu']) if f['mu'] is not None else None
    Pt = pmf(sp, 't', f['mt']) if f['mt'] is not None else None
    # our model (Bet Desk's card in "Our model" mode): expected scores priced on the same landing curve, ML = the card's chance
    fm = None
    if model:
        fm = dict(mu=model['hs'] - model['as'], mt=model['hs'] + model['as'], pml=model['rows'][0][2] if model['rows'] and model['rows'][0][4] == '' else None)
        MPm, MPt = pmf(sp, 'm', fm['mu']), pmf(sp, 't', fm['mt'])
    rows = []
    for b, r in bk.items():
        for mk, side, pt, pr in sides(r):
            c = chances(sp, f, mk, side, pt, Pm, Pt)
            if c is None: continue
            w, p, l = c; ev = w * dec(pr) + p - 1
            fair_am = am(1 / (w / (w + l))) if 0 < w < 1 else None
            x = dict(b=b, mk=mk, s=side, pt=pt, pr=pr, ev=round(ev, 4), w=round(w, 4), p=round(p, 4), fair=fair_am)
            if fm and (mk != 'ml' or fm['pml'] is not None):
                cm = chances(sp, fm, mk, side, pt, MPm, MPt); wm, pmu, lm = cm
                x['mw'] = round(wm, 4); x['mev'] = round(wm * dec(pr) + pmu - 1, 4)
                # blended (Bet Desk default: logit blend, 30% our model) on the no-push chance, pushes from the market curve
                nb = blend(wm / max(wm + lm, 1e-9), w / max(w + l, 1e-9)); wb = nb * (1 - p)
                x['bev'] = round(wb * dec(pr) + p - 1, 4)
            rows.append(x)
    # shopped board: best price per market side and point
    best = {}
    for x in rows:
        key = (x['mk'], x['s'], x['pt'])
        if key not in best or dec(x['pr']) > dec(best[key]['pr']): best[key] = x
    holds = []
    for (mk, s, pt), x in best.items():
        if s in 'HO':
            o = best.get((mk, 'A' if s == 'H' else 'U', -pt if mk == 'spread' else pt))
            if o: holds.append(dict(mk=mk, pt=pt, h=round(1 / dec(x['pr']) + 1 / dec(o['pr']) - 1, 4), a=[x['b'], x['pr']], b=[o['b'], o['pr']]))
    # middles and arbs across books (opposite sides, any points)
    mids = []
    for mk, S1, S2, P in (('spread', 'H', 'A', Pm), ('total', 'O', 'U', Pt), ('ml', 'H', 'A', None)):
        A = [x for x in rows if x['mk'] == mk and x['s'] == S1]; B = [x for x in rows if x['mk'] == mk and x['s'] == S2]
        for x in A:
            for y in B:
                if x['b'] == y['b']: continue
                if mk == 'ml':
                    s = 1 / dec(x['pr']) + 1 / dec(y['pr'])
                    if s < 1: mids.append(dict(mk=mk, kind='arb', a=[x['b'], 'H', None, x['pr']], b=[y['b'], 'A', None, y['pr']], both=0, ev=round(1 / s - 1, 4), worst=round(1 / s - 1, 4), stake=[round(100 / dec(x['pr']) / s, 1), round(100 / dec(y['pr']) / s, 1)]))
                    continue
                gap = (x['pt'] + y['pt']) if mk == 'spread' else (y['pt'] - x['pt'])   # >0 = a window both can win, 0 = same number
                if gap < 0 or P is None: continue
                m = middle((S1, x['pt'], x['pr']), (S2, y['pt'], y['pr']), P)
                kind = 'arb' if m['worst'] >= 0 else ('middle' if m['both'] > 0 else ('push' if gap > 0 else 'same'))
                if kind in ('same', 'push') and m['ev'] < MIN_EV: continue
                if kind == 'middle' and m['ev'] < MIN_EV and m['both'] < 0.08: continue
                mids.append(dict(mk=mk, kind=kind, gap=gap, a=[x['b'], S1, x['pt'], x['pr']], b=[y['b'], S2, y['pt'], y['pr']],
                                 both=round(m['both'], 4), ev=round(m['ev'], 4), worst=round(m['worst'], 4), best=round(m['best'], 4), stake=m['stake']))
    # a leg whose point sits far from every other book's is more often a stale or mislabelled price than a gift: flag it
    far = dict(NFL=2, CFB=2.5, NBA=3, CBB=3)[sp]
    for mk, col in (('spread', 0), ('total', 3)):
        pts = [r[col] for r in bk.values() if r[col] is not None]
        if len(pts) < 3: continue
        md = median(pts)
        for z in mids:
            if z['mk'] == mk and any(abs((leg[2] if mk == 'total' or leg[1] == 'H' else -leg[2]) - md) >= far for leg in (z['a'], z['b'])): z['check'] = 1
    for z in mids:
        if z['mk'] == 'ml': z['check'] = 1 if z['kind'] == 'arb' else z.get('check')
    mids.sort(key=lambda z: (z.get('check') or 0, -z['ev']))
    # bonus-bet hedge: bonus on one side at book X, cash on the other side at book Y (moneyline, no pushes)
    conv = []
    for x in rows:
        if x['mk'] != 'ml': continue
        for y in rows:
            if y['mk'] == 'ml' and y['s'] != x['s'] and y['b'] != x['b']:
                dA, dB = dec(x['pr']), dec(y['pr'])
                conv.append(dict(bonus=[x['b'], x['s'], x['pr']], cash=[y['b'], y['s'], y['pr']], conv=round((dA - 1) * (1 - 1 / dB), 4), hedge=round((dA - 1) / dB, 3)))
    conv.sort(key=lambda z: -z['conv'])
    # line moves since first seen
    mv = None; h = hist.get(f"{sp}:{g['id']}")
    if h and len(h['s']) > 1:
        f0 = h['s'][0]; mv = dict(since=f0['t'], mu0=f0.get('mu'), mt0=f0.get('mt'), src0=f0.get('src'))
        if f0.get('mu') is not None and f['mu'] is not None: mv['dmu'] = round(f['mu'] - f0['mu'], 2)
        if f0.get('mt') is not None and f['mt'] is not None: mv['dmt'] = round(f['mt'] - f0['mt'], 2)
        mv['flag'] = bool(abs(mv.get('dmu') or 0) >= MOVE[sp] or abs(mv.get('dmt') or 0) >= MOVE[sp] * 1.5)
    # sharp vs everyone else: where the sharp number sits against the median of the other books
    gapc = None
    if f['src'] != 'all books' and f['mu'] is not None and f['cons_mu'] is not None: gapc = round(f['mu'] - f['cons_mu'], 2)
    return dict(id=g['id'], k=g['k'], d=g['d'], h=g['h'], a=g['a'], nb=len(bk),
                fair=dict(src=f['src'], mu=r2(f['mu']), mt=r2(f['mt']), pml=r2(f['pml'], 4), L=fair_pt(-f['mu']) if f['mu'] is not None else None, T=fair_pt(f['mt']) if f['mt'] is not None else None, sharp_vs_books=gapc),
                model=None if not fm else dict(mu=r2(fm['mu']), mt=r2(fm['mt']), pml=r2(fm['pml'], 4)),
                bk=bk, ev=sorted([x for x in rows if x['ev'] > 0], key=lambda z: -z['ev']), best=list(best.values()),
                hold=sorted(holds, key=lambda z: z['h']), mid=mids[:30], conv=conv[:20], move=mv,
                # Look ahead (J 2026-10-28): every book price our model beats by 3%+, with when that book first posted the line
                la=[{**x, 'po': (g['ln'].get('po') or {}).get(x['b'], [None] * 3)[{'spread': 0, 'total': 1, 'ml': 2}[x['mk']]]}
                    for x in rows if (x.get('mev') or -1) >= 0.03 and (g['ln'].get('po') or {}).get(x['b'])])

def r2(x, n=2): return None if x is None else round(x, n)
def fair_pt(x): return round(x * 4) / 4

def main():
    global LAND
    if '--landing' in sys.argv or not os.path.exists(f'{D}/landing.json'): build_landing()
    LAND = json.load(open(f'{D}/landing.json'))
    S = json.load(open(f'{D}/game_slates.json'))
    hp = f'{D}/line_history.json'; hist = json.load(open(hp)) if os.path.exists(hp) else {}
    stamp = S.get('made')
    models = {}
    try:   # our model's expected scores per game, read from Bet Desk itself (models/promo/probe_games.js, headless)
        import subprocess, tempfile
        tmp = os.path.join(tempfile.mkdtemp(), 'probe.json')
        subprocess.run(['node', f'{HERE}/../models/promo/probe_games.js', tmp], check=True, stderr=subprocess.DEVNULL, stdout=subprocess.DEVNULL, timeout=300)
        models = {(x['sp'], x['id']): x for x in json.load(open(tmp))['games']}
    except Exception as e: print('model probe failed, market-only:', e)
    out = dict(made=NOW.strftime('%Y-%m-%d %H:%M UTC'), lines_from=stamp, books=NAMES, my_books=MY_BOOKS, sports={})
    for sp in ('NFL', 'CFB', 'NBA', 'CBB'):
        x = S.get(sp)
        if not x or sp not in LAND: continue
        G = []
        for g in x['games']:
            if not (g.get('ln') or {}).get('bk') or dt.datetime.fromisoformat(g['k'].replace('Z', '+00:00')) <= NOW: continue
            t = game_tools(sp, g, hist, models.get((sp, g['id'])))
            G.append(t)
            key = f"{sp}:{g['id']}"; e = hist.setdefault(key, dict(k=g['k'], s=[]))
            snap = dict(t=stamp, src=t['fair']['src'], mu=t['fair']['mu'], mt=t['fair']['mt'], bk=t['bk'])
            if not e['s'] or e['s'][-1]['t'] != stamp: e['s'].append(snap)
        out['sports'][sp] = G
        print(sp, len(G), 'games |', sum(len(t['ev']) for t in G), '+EV sides |', sum(1 for t in G for x in t['best'] if x.get('bev', -1) > 0), 'blended +EV best prices |', sum(1 for t in G for m in t['mid'] if m['kind'] == 'middle' and m['ev'] > 0), '+EV middles |',
              sum(1 for t in G for m in t['mid'] if m['kind'] == 'arb'), 'arbs')
    # keep history for games that kicked off under 3 days ago
    cut = NOW - dt.timedelta(days=3)
    hist = {k: v for k, v in hist.items() if dt.datetime.fromisoformat(v['k'].replace('Z', '+00:00')) > cut}
    json.dump(hist, open(hp, 'w'), separators=(',', ':'))
    json.dump(out, open(f'{D}/line_tools.json', 'w'), separators=(',', ':'))
    print('wrote', f'{D}/line_tools.json')

if __name__ == '__main__': main()
