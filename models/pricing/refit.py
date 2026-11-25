"""Weekly trust refit (J 2026-11-25, desk simplification step 2). For each sport x market x timing (and each sport's props),
find the trust that best explains the graded results of EVERY priced row (Pass included):

    P(win) = sigmoid(logit(fair) + t x (logit(ours) - logit(fair)))      t in [0, 1]

scored by log loss on results, pulled toward the research seed (prior: normal, sd PRIOR_SD) so a few weeks of noise cannot
swing it. Rows: open = the first snapshot when it was opener-like; close = the last pregame snapshot. Pushes are dropped.
Each trust row keeps `seed` (the research value, the prior forever), `fit` (n, data-only best t, what the close says, date).

Also reported, never applied: `close_says` = the t that best explains where the market CLOSED (first snapshot vs the last
snapshot's no-vig chance). It is far less noisy than results, so it shows the direction early; results decide.

Caps are J's call and are never lifted here: the report lists buckets that have met their cap's bar (e.g. 100 live bets).
python3 models/pricing/refit.py [--dry]   (morning chain, Mondays) -> models/pricing/trust.json + history/refit.json"""
import json, os, sys, gzip, math, datetime as dt
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
TRUST, GRADED, LOG = f"{HERE}/trust.json", f"{HERE}/history/graded.json.gz", f"{HERE}/history/refit.json"
PRIOR_SD = 0.15        # how far a season of evidence must push before trust moves a lot
MIN_N = 30             # fewer graded rows than this: leave trust alone
GAP_CLIP = 1.5         # logit gap clipped so one wild row (a missed injury) cannot dominate
GRID = np.round(np.arange(0, 1.0001, 0.01), 2)
lg = lambda p: np.log(np.clip(p, 1e-6, 1 - 1e-6) / (1 - np.clip(p, 1e-6, 1 - 1e-6)))

def fit(rows, seed):
    """(posterior t, data-only t, close-says t) for rows [(ours, fair, y, close_fair)]"""
    o = np.array([r[0] for r in rows]); f = np.array([r[1] for r in rows]); y = np.array([r[2] for r in rows], float)
    lf = lg(f); gap = np.clip(lg(o) - lf, -GAP_CLIP, GAP_CLIP)
    P = 1 / (1 + np.exp(-(lf[None, :] + GRID[:, None] * gap[None, :])))
    ll = (y * np.log(P) + (1 - y) * np.log(1 - P)).sum(1)
    post = ll - (GRID - seed) ** 2 / (2 * PRIOR_SD ** 2)
    c = [(i, r[3]) for i, r in enumerate(rows) if r[3] is not None]
    cs = None
    if len(c) >= 10:
        idx = np.array([i for i, _ in c]); cf = np.clip(np.array([v for _, v in c]), 1e-4, 1 - 1e-4)
        llc = (cf * np.log(P[:, idx]) + (1 - cf) * np.log(1 - P[:, idx])).sum(1); cs = float(GRID[llc.argmax()])
    return float(GRID[post.argmax()]), float(GRID[ll.argmax()]), cs

def main():
    dry = '--dry' in sys.argv
    T = json.load(open(TRUST)); rows = json.load(gzip.open(GRADED, 'rt')) if os.path.exists(GRADED) else []
    today = dt.date.today().isoformat(); rep = dict(made=today, prior_sd=PRIOR_SD, min_n=MIN_N, changes=[], held=[], caps_ready=[])
    rows = [r for r in rows if r['res'] in 'WL' and r.get('ours') is not None and r.get('fair') is not None]
    # ---- games: sp x market x timing
    for sp, mk in T['games'].items():
        for market, t in mk.items():
            for tm in ('open', 'close'):
                R = []
                for r in rows:
                    if r['sp'] != sp or r['market'] != market: continue
                    y = r['res'] == 'W'
                    if tm == 'open' and r['timing'] == 'open': R.append((r['ours'], r['fair'], y, r.get('close_fair')))
                    if tm == 'close':
                        # the last pregame snapshot (or the first, if it was already close-like and nothing came after)
                        if r.get('close_ours') is not None and r.get('close_fair') is not None and r['close_line'] == r['line']: R.append((r['close_ours'], r['close_fair'], y, None))
                        elif r['timing'] == 'close': R.append((r['ours'], r['fair'], y, None))
                seed = t.setdefault('seed', {}).setdefault(tm, t[tm])
                if len(R) < MIN_N: rep['held'].append(f"{sp} {market} {tm}: {len(R)} graded"); continue
                new, data, cs = fit(R, seed)
                t.setdefault('fit', {})[tm] = dict(n=len(R), data_only=data, close_says=cs, date=today)
                if abs(new - t[tm]) >= 0.005: rep['changes'].append(dict(sp=sp, market=market, timing=tm, old=t[tm], new=new, n=len(R), data_only=data, close_says=cs, seed=seed))
                t[tm] = round(new, 3)
            if t.get('cap'):
                n = sum(1 for r in rows if r['sp'] == sp and r['market'] == market and r['would_be'] == 'Bet')
                if n >= 100: rep['caps_ready'].append(f"{sp} {market}: {n} graded would-be Bets (cap {t['cap']}: J decides)")
    # ---- props: one trust per sport
    P = T['props'].setdefault('by_sport', {})
    for sp in ('NFL', 'NBA', 'NHL'):
        R = [(r['ours'], r['fair'], r['res'] == 'W', r.get('close_fair')) for r in rows if r['sp'] == sp and r['market'] == 'prop']
        e = P.setdefault(sp, dict(trust=T['props']['default']['trust'], seed=T['props']['default']['trust']))
        if len(R) < MIN_N: rep['held'].append(f"{sp} props: {len(R)} graded"); continue
        new, data, cs = fit(R, e['seed'])
        e['fit'] = dict(n=len(R), data_only=data, close_says=cs, date=today)
        if abs(new - e['trust']) >= 0.005: rep['changes'].append(dict(sp=sp, market='props', old=e['trust'], new=new, n=len(R), data_only=data, close_says=cs, seed=e['seed']))
        e['trust'] = round(new, 3)
    T['refit'] = dict(date=today, changes=len(rep['changes']))
    if not dry:
        json.dump(T, open(TRUST, 'w'), indent=1)
        L = json.load(open(LOG)) if os.path.exists(LOG) else []
        L.append(rep); json.dump(L[-60:], open(LOG, 'w'), indent=1)
    print(json.dumps(rep, indent=1))

if __name__ == '__main__':
    main()
