"""One price check for every sport (J 2026-11-24, step 1 of the desk simplification; runs in SHADOW beside the old tiers).

There is one model: our projection (Model A, never blended). The price check is the same for every sport and market:
  1. fair   = the books' no-vig chance of this exact bet: a sharp-weighted consensus (devig.WEIGHT, capped 1.5x) of every
              book posting the same line; with fewer than two, the books' average line moved onto this line along our curve.
  2. ours   = our projection's chance of the same bet (push removed).
  3. trusted = logit(fair) + trust x (logit(ours) - logit(fair)); trust per sport x market x timing from trust.json.
  4. EV at the best Colorado price on the trusted chance -> Bet (5%+ trusted EV and 5%+ on ours alone), Lean (2%+), else Pass,
     then caps (trust.json), the far check, longest price +300, and auto-demote (50+ live bets, ROI -5% or worse, CLV not
     clearly positive). One plain reason and a quarter-Kelly stake per row.
Game lines: NFL, NBA, CBB, CFB, NHL, MLB (spread / run line / puck line, total, moneyline). Props: NFL, NBA, NHL (pro only;
Colorado books post no college props).

  python3 models/pricing/price.py [--now 2026-11-24T22:31Z]   -> app/data/priced_bets.json + models/pricing/history/priced.json.gz
Nothing on the desk reads priced_bets.json yet; step 4 (new front end) does. The old *_ab tiers and prop flags are untouched."""
import json, os, sys, math, gzip, statistics, datetime as dt, unicodedata, re
from zoneinfo import ZoneInfo
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.normpath(f"{HERE}/../.."); APP = f"{ROOT}/app/data"
sys.path.insert(0, f"{ROOT}/models/model-b")
import devig as DV
import ml_anchor as MLA

CFG = json.load(open(f"{HERE}/trust.json"))
LAB, TIM = CFG["labels"], CFG["timing"]
# J 2026-11-02: every Colorado-legal book counts (no 'my books' subset). Pinnacle / BetOnline only set the fair price.
CO_BOOKS = ['draftkings', 'fanduel', 'betmgm', 'williamhill_us', 'espnbet', 'fanatics', 'betrivers', 'ballybet', 'bet365', 'hardrockbet', 'circasports']
NAMES = dict(pinnacle='Pinnacle', betonlineag='BetOnline', draftkings='DraftKings', fanduel='FanDuel', betmgm='BetMGM', williamhill_us='Caesars',
             espnbet='ESPN BET', fanatics='Fanatics', betrivers='BetRivers', ballybet='Bally Bet', hardrockbet='Hard Rock', bet365='bet365', circasports='Circa')
RANK = {'Bet': 2, 'Lean': 1, 'Pass': 0}
dec = DV.dec_from_am
lg, sg = DV.logit, DV.sigmoid
NOW = dt.datetime.now(dt.timezone.utc)
for i, a in enumerate(sys.argv):
    if a == '--now' and i + 1 < len(sys.argv): NOW = dt.datetime.fromisoformat(sys.argv[i + 1].replace('Z', '+00:00'))

def jload(p, default=None):
    try:
        return json.load(gzip.open(p, 'rt')) if p.endswith('.gz') else json.load(open(p))
    except (OSError, ValueError):
        return default

def ts(s):
    """'2026-11-24T01:15:00Z' / '2026-11-13 19:50 UTC' -> aware datetime"""
    if not s: return None
    s = str(s).replace(' UTC', '+00:00').replace('Z', '+00:00')
    if 'T' not in s and ' ' in s: s = s.replace(' ', 'T', 1)
    try:
        d = dt.datetime.fromisoformat(s)
    except ValueError:
        return None
    return d if d.tzinfo else d.replace(tzinfo=dt.timezone.utc)

def hours(a, b): return (a - b).total_seconds() / 3600

# ---------------------------------------------------------------- score distributions (our projection's shape) -------------
def _pmf(mu, grid, sd, w=None):
    d = np.exp(-0.5 * ((grid - mu) / sd) ** 2)
    if w is not None: d = d * w
    return d / d.sum()

NFL_TA = jload(f"{ROOT}/models/nfl_ab/model_a_test.json")
CFB_KN = jload(f"{ROOT}/models/cfb_ab/keynum.json"); CFB_TA = jload(f"{ROOT}/models/cfb_ab/model_a_test.json")
NBA_TA = jload(f"{ROOT}/models/nba_ab/model_a_test.json"); CBB_TA = jload(f"{ROOT}/models/cbb_ab/model_a_test.json")
GRID = {   # margin grid, margin sd(mu), margin weights, total grid, total sd, total weights (same shapes the *_ab models use)
    'NFL': (np.arange(-60, 61), lambda mu: NFL_TA['sd_margin'], np.array(NFL_TA['w_margin']), np.arange(0, 121), NFL_TA['sd_total'], np.array(NFL_TA['w_total'])),
    'CFB': (np.arange(CFB_KN['grid'][0], CFB_KN['grid'][1] + 1), lambda mu: CFB_KN['sd_margin'], np.array(CFB_KN['w']), np.arange(0, 161), CFB_TA['normal_sd']['total'], None),
    'NBA': (np.arange(-80, 81), lambda mu: NBA_TA['sd_margin'], None, np.arange(120, 341), NBA_TA['sd_total'], None),
    # CBB: miss grows with the spread (cbb_ab/live.py sd_m_for)
    'CBB': (np.arange(-90, 91), lambda mu: float(np.interp(abs(mu), [0, 12, 15, 21.5, 30], [CBB_TA['sd_margin_fit']] * 2 + [11.1, 11.9, 12.9])), None,
            np.arange(70, 261), CBB_TA['sd_total_fit'], None),
}

class Dist:
    """discrete distribution of the home margin and of the total: chance (win, push) of any spread / total side"""
    def __init__(self, mg, pm, tg, pt):
        self.mg, self.pm, self.tg, self.pt = mg, pm, tg, pt
    @classmethod
    def normal(cls, sp, mu_m, mu_t):
        mg, sdm, wm, tg, sdt, wt = GRID[sp]
        return cls(mg, _pmf(mu_m, mg, sdm(mu_m), wm), tg, _pmf(mu_t, tg, sdt, wt))
    def side(self, market, side, pt):
        """market 'spread': side 'home' gets pt (home handicap); 'away' gets -pt. market 'total': 'Over' / 'Under' at pt."""
        if market == 'spread':
            v = self.mg + pt; P = self.pm
            win = P[v > 1e-9].sum() if side == 'home' else P[v < -1e-9].sum()
        else:
            v = self.tg - pt; P = self.pt
            win = P[v > 1e-9].sum() if side == 'Over' else P[v < -1e-9].sum()
        push = P[np.abs(v) < 1e-9].sum()
        return float(win), float(push)

def nb_pmf(m, a, K=30):
    k = np.arange(K); n = 1 / a; p = n / (n + m)
    lp = np.array([math.lgamma(n + i) - math.lgamma(n) - math.lgamma(i + 1) for i in k]) + n * math.log(p) + k * math.log(1 - p)
    d = np.exp(lp); return d / d.sum()

# ---------------------------------------------------------------- the price check --------------------------------------------
def trust_for(sp, market, timing):
    t = CFG['games'].get(sp, {}).get(market)
    return (t['open'] if timing == 'open' else t['close'], t.get('cap'), t.get('why') or t.get('src')) if t else (0.0, 'Lean', 'no trust row')

def label(ev_t, ev_o):
    if ev_t >= LAB['bet_ev'] and (not LAB['ours_must_clear_bet_ev'] or ev_o >= LAB['bet_ev']): return 'Bet'
    return 'Lean' if ev_t >= LAB['lean_ev'] else 'Pass'

def cap(lab, c): return lab if c is None or RANK[lab] <= RANK[c] else c

def stake(lab, pw, pl, d):
    if lab != 'Bet': return 0.0
    b = d - 1; f = max(0.0, (b * pw - pl) / b) * LAB['stake']['kelly']
    return round(min(LAB['stake']['max_u'], LAB['stake']['bankroll_u'] * f), 1)

def price_row(ours_c, fair_c, push, trust, am):
    """ours_c / fair_c = chance of winning with pushes removed. Returns trusted chance, EV on trusted and on ours, stake inputs."""
    tr = sg(lg(fair_c) + trust * (lg(ours_c) - lg(fair_c))) if fair_c is not None else ours_c
    d = dec(am); nb = 1 - push
    ev = lambda c: c * nb * (d - 1) - (1 - c) * nb
    return tr, ev(tr), ev(ours_c), tr * nb, (1 - tr) * nb, d

def pct(p): return f"{100 * p:.0f}%"
def amS(a): return f"+{a}" if a > 0 else str(a)

def reason(r):
    k = r['why']
    if k == 'far': return f"Our chance ({pct(r['ours'])}) is far from the books' ({pct(r['fair'])}): usually news we are missing, so no bet"
    if k == 'off_line': return f"This line ({r['line']}) is more than {r.get('tol', 3):g} off the other books' line: usually stale or a different market, so no bet"
    if k == 'long': return f"Price {amS(r['price'])} is longer than +300: our long-shot chances are untested"
    if k == 'demoted': return f"Held to Lean: {r['demoted']}"
    if k == 'capped': return f"Held to {r['label']}: {r['cap_why']}"
    if r['label'] == 'Bet': return f"We have {pct(r['ours'])}, the books {pct(r['fair'])}; after the price check it is still {r['ev']:+.1%} at {r['book_name']} {amS(r['price'])}"
    if r['label'] == 'Lean': return f"Small edge after the price check ({r['ev']:+.1%} at {r['book_name']} {amS(r['price'])}): tracked, not staked"
    if r['ev_ours'] >= LAB['bet_ev']: return f"Our number likes it ({r['ev_ours']:+.1%}) but the market has been right more often here: {r['ev']:+.1%} after the price check"
    return 'No edge at the best Colorado price'

def finish(r, cap_lab, cap_why, demoted, guards=()):
    """label + reason + stake. A GUARD (far from the books, longer than +300, line off the books' median, one-sided prop
    price) never stakes, but the row keeps the label it would have had (would_be) and is graded on its own
    (J 2026-11-25: these may be our best bets if we disagree with the line and keep being right, e.g. big underdogs)."""
    lab = label(r['ev'], r['ev_ours']); r['why'] = None; g = list(guards)
    if r['fair'] is not None and abs(r['ours'] - r['fair']) > LAB['far_prob']: g.append('far')
    if r['price'] > LAB['max_price']: g.append('long')
    r['would_be'] = cap(lab, cap_lab) if cap_lab != 'Pass' else lab
    r['guards'] = g
    if 'far' in g: lab, r['why'] = 'Pass', 'far'
    elif 'long' in g: lab, r['why'] = 'Pass', 'long'
    elif 'off_line' in g: lab, r['why'] = 'Pass', 'off_line'
    else:
        c = cap(lab, cap_lab)
        if c != lab: lab, r['why'], r['cap_why'] = c, 'capped', cap_why
        if demoted and RANK[lab] > RANK['Lean']: lab, r['why'], r['demoted'] = 'Lean', 'demoted', demoted
    r['label'] = lab; r['reason'] = reason(r)
    r['stake_u'] = stake(lab, r.pop('_pw'), r.pop('_pl'), r.pop('_d'))
    for k in ('why', 'cap_why'): r.pop(k, None)
    return r

# ---------------------------------------------------------------- auto-demote (stopgap until step 2 grades the new labels) ----
def demotions():
    """sport x market buckets whose staked bets (old A/B tiers, old prop flags) lost money with no clear CLV: max Lean"""
    D, rule = {}, CFG['demote']
    for sp, f in (('NFL', 'nfl_ab'), ('NBA', 'nba_ab'), ('CBB', 'cbb_ab'), ('CFB', 'cfb_ab')):
        B = (jload(f"{ROOT}/models/{f}/history/grade.json") or {}).get('B') or {}
        for mk in ('spread', 'total', 'ml'):
            rows = [B[k] for k in (f"{mk}_A", f"{mk}_B") if k in B and B[k].get('n')]
            n = sum(x['n'] for x in rows)
            if n < rule['min_bets']: continue
            roi = sum(x['n'] * x['roi'] for x in rows) / n
            cl = [(x.get('clv_n') or x['n'], x['clv_pts']) for x in rows if x.get('clv_pts') is not None and mk != 'ml']
            clv = sum(a * b for a, b in cl) / sum(a for a, _ in cl) if cl else None
            if roi <= rule['roi_below'] and (clv is None or clv < rule['games_clv_pts_below']):
                D[(sp, mk)] = f"old {sp} {mk} bets went {roi:+.1%} ROI over {n} live bets" + (f", closing line value {clv:+.2f} pts" if clv is not None else '')
    lf = (jload(f"{APP}/scorecard.json") or {}).get('lineflags') or {}
    by = {}
    for x in lf.get('by') or []:
        if x['sp'] != 'All' and x['kind'] in ('flag', 'proven'): by.setdefault(x['sp'], []).append(x)
    for sp, rows in by.items():
        n = sum(x['n'] for x in rows)
        if n < rule['min_bets']: continue
        roi = sum(x['units'] for x in rows) / n
        nc = sum(x.get('nclv') or 0 for x in rows); clv = sum((x.get('nclv') or 0) * (x.get('clv') or 0) for x in rows) / nc if nc else None
        if roi <= rule['roi_below'] and (clv is None or clv < rule['props_clv_below']):
            D[(sp, 'props')] = f"old {sp} prop flags went {roi:+.1%} ROI over {n} live bets" + (f", closing line value {clv:+.1%}" if clv is not None else '')
    # step 2: once a bucket has 50+ graded Bets under the NEW labels, their record replaces the old tiers' (either way)
    G = (jload(f"{HERE}/history/grade.json") or {}).get('staked') or {}
    for key, x in G.items():
        sp, mk = key.split('|'); mk = 'props' if mk == 'prop' else mk
        if x['n'] < rule['min_bets']: continue
        D.pop((sp, mk), None)
        bar = rule['props_clv_below'] if mk == 'props' else rule.get('games_clv_below', 0.01)
        if x['roi'] <= rule['roi_below'] and (x.get('clv') is None or x['clv'] < bar):
            D[(sp, mk)] = f"{sp} {mk} Bets went {x['roi']:+.1%} ROI over {x['n']} graded bets" + (f", closing line value {x['clv']:+.1%}" if x.get('clv') is not None else '')
    return D

# ---------------------------------------------------------------- timing: opener-like or close-like -----------------------
LH = jload(f"{APP}/line_history.json", {}) or {}
def first_seen(sp, gid, nhl_first=None):
    if sp == 'NHL': return (nhl_first or {}).get(gid)
    h = LH.get(f"{sp}:{gid}")
    return ts(h['s'][0]['t']) if h and h.get('s') else None

def timing(sp, k, seen):
    """open = the line went up 24 h ago or less and the game is 8+ h away; NFL lines 96+ h out (look-ahead) count as open"""
    to_start = hours(k, NOW)
    if sp == 'NFL' and to_start >= TIM['nfl_lookahead_hours']: return 'open'
    if seen is None: return 'close'   # unknown post time: treat as close-like (the cautious trust)
    return 'open' if hours(NOW, seen) <= TIM['open_hours'] and to_start >= TIM['min_hours_to_start'] else 'close'

# ---------------------------------------------------------------- game lines --------------------------------------------------
def fair_two_way(rows):
    """rows: {book: (price1, price2)} on one line -> consensus no-vig chance of outcome 1, or None with fewer than two books"""
    books = {b: [dec(a), dec(c)] for b, (a, c) in rows.items() if a and c}
    if len(books) < 2: return None
    return DV.consensus(books)[0]

def market_mean(bk, market, sd, kind_idx):
    """books' average home margin / total implied by line + devigged price (normal approx, as the *_ab market() functions)"""
    acc = tw = 0.0
    for b, r in bk.items():
        pt, p1, p2 = r[kind_idx], r[kind_idx + 1], r[kind_idx + 2]
        if pt is None or not p1 or not p2: continue
        p = min(max(DV.power([dec(p1), dec(p2)])[0], 0.02), 0.98)
        z = math.sqrt(2) * _erfinv(2 * p - 1)
        mu = (-pt + sd * z) if market == 'spread' else (pt + sd * z)
        w = min(DV.WEIGHT.get(b, 0.8), DV.MAX_RATIO); acc += w * mu; tw += w
    return acc / tw if tw else None

def _erfinv(y):
    a = 0.147; s = math.copysign(1, y); l = math.log(1 - y * y); t = 2 / (math.pi * a) + l / 2
    return s * math.sqrt(math.sqrt(t * t - l / a) - t)

GAMES = []
def _move(sp, gid, keep=16):
    """[[time, home margin, total], ...] from line_history.json, thinned to `keep` points (first and last always kept)"""
    s = (LH.get(f"{sp}:{gid}") or {}).get('s') or []
    now = NOW.strftime('%Y-%m-%d %H:%M UTC')   # a replay (--now) must not show later snapshots
    s = [x for x in s if (x.get('mu') is not None or x.get('mt') is not None) and str(x.get('t', '')) <= now]
    if len(s) > keep: s = [s[round(i * (len(s) - 1) / (keep - 1))] for i in range(keep)]
    return [[x['t'], None if x.get('mu') is None else round(x['mu'], 2), None if x.get('mt') is None else round(x['mt'], 2)] for x in s]

def price_game(sp, g, ours, dist_mkt, demote):
    """g: {id, k, h, a, bk}; ours: Dist + p_home; dist_mkt(market, mu) -> Dist centred on the books' number. Returns rows."""
    bk, k = g['bk'], ts(g['k']); tm = g['timing']; out = []
    # one summary per game for the front end's Games tab: our score (never blended) beside the books' median line
    med = lambda i: (lambda v: statistics.median(v) if v else None)([r[i] for r in bk.values() if r and len(r) > i and r[i] is not None])
    sc = getattr(ours, 'score', None)
    GAMES.append(dict(sp=sp, id=g['id'], k=g['k'], h=g['h'], a=g['a'], hs=None if not sc else round(sc[0], 1), as_=None if not sc else round(sc[1], 1),
                      p_home=None if ours.p_home is None else round(ours.p_home, 3), spread=med(0), total=med(3), ml=[med(6), med(7)], timing=tm,
                      # owner desk Details (J 2026-11-25): every Colorado book plus Pinnacle as [home spread, home price, away price,
                      # total, over, under, home ML, away ML], and the consensus line since it was first seen (home margin, total)
                      books={NAMES.get(b, b): bk[b] for b in list(CO_BOOKS) + ['pinnacle'] if bk.get(b)},
                      move=_move(sp, g['id'])))
    names = {'home': g['h'], 'away': g['a'], 'Over': 'Over', 'Under': 'Under'}
    for market, idx, sides in (('spread', 0, ('home', 'away')), ('total', 3, ('Over', 'Under'))):
        lines = [r[idx] for r in bk.values() if r and r[idx] is not None and r[idx + 1] and r[idx + 2]]
        if not lines: continue
        med = statistics.median(lines); tol = 1.0 if sp in ('NHL', 'MLB') else 3.0
        trust, cap_lab, cap_src = trust_for(sp, market, tm)
        mu_mkt = None
        best, best_off = {}, {}
        for b in CO_BOOKS:
            r = bk.get(b)
            if not r or r[idx] is None or not r[idx + 1] or not r[idx + 2]: continue
            off = abs(r[idx] - med) > tol
            pt = r[idx]
            same = {bb: (rr[idx + 1], rr[idx + 2]) for bb, rr in bk.items() if rr and rr[idx] == pt}
            f1 = fair_two_way(same)
            if f1 is None:
                if mu_mkt is None: mu_mkt = ('x', dist_mkt(market, bk))
                dm = mu_mkt[1]
                if dm is None: continue
                w1, p1 = dm.side(market, sides[0], pt); w2, _ = dm.side(market, sides[1], pt); f1 = w1 / max(w1 + w2, 1e-9)
            for j, s in enumerate(sides):
                am = r[idx + 1 + j]
                win, push = ours.side(market, s, pt); lose = 1 - win - push
                oc = win / max(win + lose, 1e-9); fc = f1 if j == 0 else 1 - f1
                oc, fc = min(max(oc, 1e-4), 1 - 1e-4), min(max(fc, 1e-4), 1 - 1e-4)
                tr, ev_t, ev_o, pw, pl, d = price_row(oc, fc, push, trust, am)
                line = (pt if s == 'home' else -pt) if market == 'spread' else pt
                row = dict(market=market, side=s, pick=names[s], line=line, book=b, book_name=NAMES.get(b, b), price=am,
                           ours=round(oc, 4), fair=round(fc, 4), trusted=round(tr, 4), push=round(push, 4), ev=round(ev_t, 4), ev_ours=round(ev_o, 4),
                           trust=trust, _pw=pw, _pl=pl, _d=d)
                B = best_off if off else best
                if s not in B or row['ev'] > B[s]['ev']: B[s] = row
        if best:   # one row per market: the side with the better price-checked EV (the old tiers were one per market too)
            out.append(finish(max(best.values(), key=lambda r: r['ev']), cap_lab, cap_src, demote.get((sp, market))))
        if best_off:   # a book far off the median line: guarded, tracked only if it would have been a Bet or Lean
            x = finish(dict(max(best_off.values(), key=lambda r: r['ev']), tol=tol), cap_lab, cap_src, demote.get((sp, market)), ('off_line',))
            if x['would_be'] != 'Pass': out.append(x)
    # moneyline: ours = the projection's win chance; fair = books' devigged moneyline (ml_anchor)
    pm = MLA.market_home(bk, sp) if sp != 'NHL' else _nhl_ml_fair(bk)
    if pm is not None and ours.p_home is not None:
        trust, cap_lab, cap_src = trust_for(sp, 'ml', tm); best = {}
        for b in CO_BOOKS:
            r = bk.get(b)
            if not r or len(r) < 8 or not r[6] or not r[7]: continue
            for j, s in enumerate(('home', 'away')):
                am = r[6 + j]; oc = ours.p_home if j == 0 else 1 - ours.p_home; fc = pm if j == 0 else 1 - pm
                tr, ev_t, ev_o, pw, pl, d = price_row(oc, fc, 0.0, trust, am)
                row = dict(market='ml', side=s, pick=names[s], line=None, book=b, book_name=NAMES.get(b, b), price=am, ours=round(oc, 4),
                           fair=round(fc, 4), trusted=round(tr, 4), push=0.0, ev=round(ev_t, 4), ev_ours=round(ev_o, 4), trust=trust, _pw=pw, _pl=pl, _d=d)
                if s not in best or row['ev'] > best[s]['ev']: best[s] = row
        if best: out.append(finish(max(best.values(), key=lambda r: r['ev']), cap_lab, cap_src, demote.get((sp, 'ml'))))
    for r in out:
        r.update(sp=sp, game=g['id'], matchup=f"{g['a']} @ {g['h']}", k=g['k'], timing=tm, hours_to_start=round(hours(k, NOW), 1))
        r['big_dog'] = bool((r['market'] == 'spread' and sp not in ('NHL', 'MLB') and r['line'] >= BIG_DOG_PTS.get(sp, 10))
                            or (r['market'] == 'ml' and r['price'] >= 250))
    return out
BIG_DOG_PTS = {'NFL': 7, 'CFB': 14, 'NBA': 9, 'CBB': 15}   # spread a big underdog gets (and moneylines +250 or longer)

def _nhl_ml_fair(bk):
    return MLA.market_home(bk, 'NHL')

# ---- per-sport adapters: our projection + book rows -------------------------------------------------------------------------
def ab_games(sp, fname, slate, demote):
    V = jload(f"{APP}/{fname}") or {}; rows = []; MW[sp] = market_wide(sp, V)
    S = {str(g['id']): g for g in (slate.get(sp) or {}).get('games') or []}
    mg, sdm, wm, tg, sdt, wt = GRID[sp]
    for gid, v in (V.get('games') or {}).items():
        x = S.get(str(gid)); A = v.get('A')
        if not x or not A or not (x.get('ln') or {}).get('bk'): continue
        k = ts(x['k'])
        if not k or k <= NOW: continue
        mu_m, mu_t = A['hs'] - A['as'], A['hs'] + A['as']
        ours = Dist.normal(sp, mu_m, mu_t); ours.p_home = A.get('p_home'); ours.score = (A['hs'], A['as'])
        def dist_mkt(market, bk, sp=sp, mu_m=mu_m, mu_t=mu_t):
            mm = market_mean(bk, 'spread', sdm(mu_m), 0) if market == 'spread' else mu_m
            mt = market_mean(bk, 'total', sdt, 3) if market == 'total' else mu_t
            return None if mm is None or mt is None else Dist.normal(sp, mm, mt)
        g = dict(id=str(gid), k=x['k'], h=x['h'], a=x['a'], bk=x['ln']['bk'])
        g['timing'] = timing(sp, k, first_seen(sp, gid))
        for r in price_game(sp, g, ours, dist_mkt, demote):
            r['old'] = old_game_label(v, r['market']); r['ctx'] = context(v, r, MW.get(sp, set())); rows.append(r)
    return rows

# ---------------------------------------------------------------- confidence (J 2026-11-25) ----
# One 1-5 number per Bet/Lean, the same on every desk, shown only in Details and as a sort. It starts from the label and the
# edge after the price check, then moves one step for the context signals the old tiers used (model-b-confidence research
# 2026-11-01: both engines agree, news-driven gaps, basketball underdogs, game types). Timing is already in the trust number,
# so timing reasons are left out, and so are caveats that hit every game in a sport and market (they are not about this game).
# Props and NHL/MLB games have no context feed: their rating is the edge alone. Graded by bucket in price_grades.json (conf).
CTX_SKIP = ('Kickoff in', 'No post time', 'Moneyline ', 'Chance = ', 'Measured against', 'Key numbers', 'Tempo:')
MW = {}   # sport -> reasons seen on 90%+ of that sport's games (market-wide caveats)
def market_wide(sp, V):
    from collections import Counter
    n = 0; c = Counter()
    for v in (V.get('games') or {}).values():
        B = v.get('B') or {}
        if not B: continue
        n += 1; c.update({t for b in B.values() if isinstance(b, dict) for _, t in b.get('reasons') or []})
    return {t for t, k in c.items() if n and k >= 0.9 * n}

def context(v, r, wide):
    """the old tier's +/- reasons for this game and market, only when they argued for the same side we pick"""
    b = (v.get('B') or {}).get('spread' if r['market'] == 'ml' else r['market'])
    if not isinstance(b, dict) or not b.get('side'): return []
    side = b['side']
    mine = {'home': v.get('h'), 'away': v.get('a')}.get(r['side'], r['side'])
    if str(side).lower() != str(mine).lower(): return []
    return [[sg_, t] for sg_, t in b.get('reasons') or [] if sg_ in '+-' and t not in wide and not t.startswith(CTX_SKIP)]

def confidence(r):
    if r['label'] == 'Pass': return None, []
    why = []
    if r['label'] == 'Bet': c = 4 if r['ev'] >= 0.08 else 3; why.append(['=', f"Bet, {r['ev']:+.1%} after the price check: starts at {c}"])
    else: c = 2; why.append(['=', f"Lean ({r['ev']:+.1%} after the price check): starts at 2"])
    ctx = r.get('ctx') or []; net = sum(1 if x[0] == '+' else -1 for x in ctx)
    if net >= 2: c += 1; why.append(['+', f"{net} more signals for than against: up one"])
    elif net <= -1: c -= 1; why.append(['-', f"More signals against than for: down one"])
    elif net == 1: why.append(['=', 'One more signal for than against: not enough to move it'])
    elif ctx: why.append(['=', 'Signals for and against even out'])
    else: why.append(['=', 'No context signals for this kind of bet: edge only'])
    return max(1, min(5, c)), why + ctx

def old_game_label(v, market):
    b = ((v.get('B') or {}).get(market) or {})
    return {'A': 'Bet', 'B': 'Lean', 'C': 'Pass'}.get(b.get('tier')) if b else None

PO_TEAM = {'JAX': 'JACKSONVILLE', 'WSH': 'WASHINGTON', 'LA': 'LOS ANGELES', 'NJ': 'NEW JERSEY'}
def _team_ok(ab, full):
    """does a team abbreviation (GB, VGK) name this full team name? prefix, initials, or letters in order (desk poTeam)"""
    A = str(ab or '').upper(); F = re.sub(r'[^A-Z ]', '', str(full or '').upper()); w = F.split()
    if not A or not w: return False
    if A in PO_TEAM and F.startswith(PO_TEAM[A]): return True
    if F.startswith(A) or ''.join(x[0] for x in w).startswith(A): return True
    i = 0
    for ch in w[0]:
        if i < len(A) and ch == A[i]: i += 1
    return i == len(A)

def odds_api_rows(ev):
    """Odds API event -> {book: [spread_pt_home, home, away, total, over, under, ml_home, ml_away]}"""
    out = {}
    for b in ev.get('bookmakers') or []:
        r = [None] * 8; M = {m['key']: m['outcomes'] for m in b.get('markets') or []}
        for o in M.get('spreads') or []:
            if o['name'] == ev['home_team']: r[0], r[1] = o.get('point'), o.get('price')
            elif o['name'] == ev['away_team']: r[2] = o.get('price')
        for o in M.get('totals') or []:
            if o['name'] == 'Over': r[3], r[4] = o.get('point'), o.get('price')
            elif o['name'] == 'Under': r[5] = o.get('price')
        for o in M.get('h2h') or []:
            if o['name'] == ev['home_team']: r[6] = o.get('price')
            elif o['name'] == ev['away_team']: r[7] = o.get('price')
        out[b['key']] = r
    return out

def nhl_games(demote):
    N = jload(f"{APP}/nhl_sim.json") or {}; O = (jload(f"{APP}/odds/NHL.json") or {}).get('events') or []
    H = jload(f"{APP}/odds/nhl_line_history.json.gz", {}) or {}
    seen = {}
    for t in sorted(H):
        for e in H[t] or []: seen.setdefault(e.get('id'), ts(t))
    rows = []
    for x in N.get('games') or []:
        k = ts(x.get('k'))
        if not k or k <= NOW: continue
        ev = next((e for e in O if abs(hours(ts(e['commence_time']), k)) < 2 and _team_ok(x['h'], e['home_team']) and _team_ok(x['a'], e['away_team'])), None)
        if not ev: continue
        dh, da, dtot = np.array(x['dh']), np.array(x['da']), np.array(x['dt'])
        # margin pmf: the sim's goal distributions treated as independent, regulation ties settled by the sim's own win chance
        M = np.outer(dh, da); mgs = np.arange(-(len(da) - 1), len(dh))
        pm = np.zeros(len(mgs))
        for i in range(len(dh)):
            for j in range(len(da)): pm[i - j + len(da) - 1] += M[i, j]
        # puck line straight from the sim (pl = P(home by 2+), P(away by 2+)); rescale the pmf tails to match it
        ours = _NHLDist(mgs, pm, np.arange(len(dtot)), dtot, x.get('pl')); ours.p_home = x.get('pw'); ours.score = (x['hs'], x['as'])
        g = dict(id=str(x['id']), k=x['k'], h=x['h'], a=x['a'], bk=odds_api_rows(ev))
        g['timing'] = timing('NHL', k, seen.get(ev['id']))
        def dist_mkt(market, bk, ours=ours):
            # NHL: no market-mean shift (puck line is always 1.5, totals 5.5-6.5); fall back to the median book's own devig
            return None
        rows += [dict(r, old=None) for r in price_game('NHL', g, ours, dist_mkt, demote)]   # NHL games had no tiers
    return rows

class _NHLDist(Dist):
    def __init__(self, mg, pm, tg, pt, pl):
        super().__init__(mg, pm, tg, pt); self.pl = pl
    def side(self, market, side, pt):
        if market == 'spread' and self.pl and abs(abs(pt) - 1.5) < 1e-9:
            h2, a2 = self.pl
            if side == 'home': return (1 - a2, 0.0) if pt > 0 else (h2, 0.0)
            return (1 - h2, 0.0) if pt < 0 else (a2, 0.0)
        return super().side(market, side, pt)

def mlb_games(slate, demote):
    M = jload(f"{APP}/mlb_slate.json") or {}; meta = M.get('meta') or {}
    a = (meta.get('nbAlpha') or {}).get('eff', 0.2429); tie_home = meta.get('tieHomeWin', 0.52)
    S = {str(g['id']): g for g in (slate.get('MLB') or {}).get('games') or []}; rows = []
    for x in M.get('games') or []:
        e = (x.get('models') or {}).get('eff'); s = S.get(str(x.get('gamePk')))
        if not e or not s or not (s.get('ln') or {}).get('bk'): continue
        k = ts(s['k'])
        if not k or k <= NOW: continue
        ph, pa = nb_pmf(e['runsHome'], a), nb_pmf(e['runsAway'], a)
        K = len(ph); mgs = np.arange(-(K - 1), K); pm = np.zeros(len(mgs)); tg = np.arange(2 * K - 1); pt = np.zeros(len(tg))
        for i in range(K):
            for j in range(K): pm[i - j + K - 1] += ph[i] * pa[j]; pt[i + j] += ph[i] * pa[j]
        z = pm[K - 1]; pm[K - 1] = 0; pm[K] += z * tie_home; pm[K - 2] += z * (1 - tie_home)   # extra innings: one-run games
        ours = _MLBDist(mgs, pm, tg, pt, e.get('totals') or []); ours.p_home = e.get('homeWinProb'); ours.score = (e['runsHome'], e['runsAway'])
        g = dict(id=str(x['gamePk']), k=s['k'], h=s['h'], a=s['a'], bk=s['ln']['bk'])
        g['timing'] = timing('MLB', k, first_seen('MLB', g['id']))
        rows += [dict(r, old=None) for r in price_game('MLB', g, ours, lambda market, bk: None, demote)]   # MLB had no tiers
    return rows

class _MLBDist(Dist):
    """J's efficiency model: totals from its own table (over / under / push at each line), run line from the NB convolution"""
    def __init__(self, mg, pm, tg, pt, table):
        super().__init__(mg, pm, tg, pt); self.T = {float(r['line']): r for r in table}
    def side(self, market, side, pt):
        if market == 'total' and float(pt) in self.T:
            r = self.T[float(pt)]; return (r['over'] if side == 'Over' else r['under']), r.get('push', 0.0)
        return super().side(market, side, pt)

# ---------------------------------------------------------------- props (NFL, NBA, NHL) -------------------------------------
PO_MAP = {
    'NFL': {'Pass Yards': 'pass_yds', 'Pass Attempts': 'pass_att', 'Pass Completions': 'cmp', 'TD Passes': 'pass_td', 'Rush Yards': 'rush_yds',
            'Rush Attempts': 'carries', 'Receiving Yards': 'rec_yds', 'Receptions': 'rec', 'To Score TD': 'td'},
    'NBA': {'Points': 'pts', 'Rebounds': 'reb', 'Assists': 'ast', '3 Point FG': 'fg3m', 'Points + Assists + Rebounds': 'pra', 'Steals': 'stl', 'Blocks': 'blk', 'Turnovers': 'tov'},
    'NHL': {'Shots On Goal': 'sog', 'Points': 'points', 'Goals': 'goals', 'Anytime Goal Scorer': 'goals', 'Assists': 'assists', 'Blocks': 'blk', 'Saves': 'saves'}}
YARDS = {'rec_yds', 'rush_yds', 'pass_yds', 'rush_rec_yds'}
DESK_TZ = ZoneInfo('America/New_York')   # same clock the desk snapshot runs in (snap_flags.js)

def norm(n):
    s = unicodedata.normalize('NFD', str(n).lower()); s = ''.join(c for c in s if not unicodedata.combining(c))
    s = re.sub(r"[.'’,-]", ' ', s); s = re.sub(r'\b(jr|sr|ii|iii|iv|v)\b', '', s); return re.sub(r'\s+', ' ', s).strip()
def init(n):
    w = norm(n).split(' '); return w[0][0] + ' ' + ' '.join(w[1:]) if len(w) > 1 else w[0]

def interp(c, L):
    """desk interp(): logit-linear between ladder points, end segments extended at 0.1+ logit per unit"""
    n = len(c)
    if n == 1: return c[0][1]
    def ext(i, j):
        sl = min(-0.1, (lg(c[j][1]) - lg(c[i][1])) / (c[j][0] - c[i][0])); return sg(lg(c[i][1]) + sl * (L - c[i][0]))
    if L < c[0][0]: return ext(0, 1)
    if L > c[-1][0]: return ext(n - 1, n - 2)
    for (a, pa), (b, pb) in zip(c, c[1:]):
        if a <= L <= b:
            t = (L - a) / (b - a) if b != a else 0; return sg(lg(pa) + (lg(pb) - lg(pa)) * t)
    return 0.5

def model_probs(stat, s, L):
    c = s['c']
    if stat not in YARDS and float(L).is_integer():
        po, pu = interp(c, L + 0.5), 1 - interp(c, L - 0.5); return po, pu, max(0.0, 1 - po - pu)
    po = interp(c, L); return po, 1 - po, 0.0

def cond(stat, s, L):
    po, pu, pp = model_probs(stat, s, L); return po / max(po + pu, 1e-9), pp

def prop_trust(sp):
    return (CFG['props'].get('by_sport') or {}).get(sp, CFG['props']['default'])['trust']

def prop_caps(sp, stat, side):
    for c in CFG['props']['caps']:
        if c['sp'] != sp: continue
        if 'stats' in c and stat not in c['stats']: continue
        if 'not_stats' in c and stat in c['not_stats']: continue
        if 'side' in c and c['side'] != side: continue
        return c['cap'], c['why']
    return None, None

def prop_rows(sp, P, stat, s):
    """every line for one player stat: {L: {book: [over, under]}} (desk poRows: ladders are alt overs, Yes/No = 0.5)"""
    at = {}
    for name, lines in P.items():
        if PO_MAP[sp].get(name.replace(' ladder', '')) != stat: continue
        for l0, B in lines.items():
            for b, S in B.items():
                for sd, pr in S.items():
                    try: pr = float(pr)
                    except (TypeError, ValueError): continue
                    m = re.match(r'^(\d+)\+$', sd)
                    if m: L, j = int(m.group(1)) - 0.5, 0
                    elif sd in ('Over', 'Under'): L, j = float(l0), 0 if sd == 'Over' else 1
                    elif sd in ('Yes', 'No'): L, j = 0.5, 0 if sd == 'Yes' else 1
                    else: continue
                    r = at.setdefault((L, b), [None, None])
                    if r[j] is None: r[j] = int(pr)
    rows = [dict(L=L, b=b, o=v[0], u=v[1]) for (L, b), v in at.items()]
    ph = lambda r: r['b'] == 'bet365' and r['o'] == -120 and r['u'] == -120
    mains = sorted(r['L'] for r in rows if r['o'] is not None and r['u'] is not None and not ph(r))
    med = mains[len(mains) >> 1] if mains else None
    for r in rows:
        r['main'] = r['o'] is not None and r['u'] is not None
        r['odd'] = ph(r) or (med is not None and r['main'] and abs(r['L'] - med) > max(1, 0.08 * abs(med)) and all(-140 <= v <= 120 for v in (r['o'], r['u'])))
    return rows

def prop_fair(stat, s, rows, L):
    """books' no-vig chance of the over at L (desk pfMarket): sharp-weighted consensus of two-sided prices at L; other
    two-sided lines moved to L along our curve; one-sided only -> implied price less ~7% hold"""
    same = {r['b']: [dec(r['o']), dec(r['u'])] for r in rows if r['main'] and not r['odd'] and r['L'] == L}
    if len(same) >= 2: return DV.consensus(same)[0], 'two-sided'
    est = []
    cL = cond(stat, s, L)[0]
    for r in rows:
        if not r['main'] or r['odd']: continue
        q = DV.power([dec(r['o']), dec(r['u'])])[0]
        est.append(q if r['L'] == L else sg(lg(q) + lg(cL) - lg(cond(stat, s, r['L'])[0])))
    if est: return statistics.median(est), 'two-sided'
    est = [min(0.99, (1 / dec(r['o'])) / 1.07) for r in rows if r['L'] == L and r['o'] is not None]
    return (statistics.median(est), 'one-sided') if est else (None, None)

def game_of(sp, g):
    """id of this sport's priced game (GAMES) for an odds-feed event: same teams, start within 3 h. Prop projections
    carry their own game key (NBA: a slate index), so the desk and guest pages need this to put a prop under its game."""
    k = ts(g['k'])
    for G in GAMES:
        if G['sp'] != sp or not k or not ts(G['k']) or abs((ts(G['k']) - k).total_seconds()) > 3 * 3600: continue
        same = lambda ab, full: str(ab).lower() == str(full).lower() or _team_ok(ab, full)
        if same(G['h'], g['h']) and same(G['a'], g['a']): return G['id']
    return None

def props(sp, demote):
    PJ, OD = jload(f"{APP}/{sp.lower()}_props.json") or {}, jload(f"{APP}/odds/props_{sp}.json") or {}
    idx = {}
    for g in OD.get('games') or []:
        for n, P in (g.get('players') or {}).items():
            for key in ('f:' + norm(n), 'i:' + init(n)): idx.setdefault(key, []).append((g, n, P))
    old = {}
    for r in (jload(f"{APP}/prop_picks.json") or {}).get('rows') or []:
        old['|'.join(str(r.get(x)) for x in ('sp', 'g', 'n', 'k', 'side'))] = {'proven': 'Bet', 'flag': 'Bet', 'lean': 'Lean', 'unproven': 'Pass', 'check': 'Pass'}.get(r.get('kind'))
    trust = prop_trust(sp); out = []; n_pass = 0; quiet = []
    for p in PJ.get('players') or []:
        cand = idx.get('f:' + norm(p['n']), []) + idx.get('i:' + init(p['n']), [])
        ok = [x for x in cand if (_team_ok(p['t'], x[0]['h']) and _team_ok(p['o'], x[0]['a'])) or (_team_ok(p['t'], x[0]['a']) and _team_ok(p['o'], x[0]['h']))]
        hit = next((x for x in ok if p.get('d') and ts(x[0]['k']).astimezone(DESK_TZ).date().isoformat() == p['d']), None) or (ok[0] if ok else None)
        if not hit: continue
        g, _, P = hit; k = ts(g['k'])
        if not k or k <= NOW: continue
        for stat, s in (p.get('s') or {}).items():
            if not s.get('c'): continue
            allr = prop_rows(sp, P, stat, s)
            mine = [r for r in allr if r['b'] in CO_BOOKS]
            if not mine: continue
            ml = sorted(r['L'] for r in allr if r['main'] and not r['odd']); med = ml[len(ml) >> 1] if ml else None
            pool = [r for r in mine if r['main'] and not r['odd'] and (med is None or abs(r['L'] - med) <= max(0.5, 0.08 * abs(med)))]
            if not pool:
                cnt = {}
                for r in mine: cnt[r['L']] = cnt.get(r['L'], 0) + 1
                top = max(cnt, key=cnt.get); pool = [r for r in mine if r['L'] == top and not r['odd']]
            for side in ('Over', 'Under'):
                O = side == 'Over'; best = best_long = None
                side_books = [[NAMES.get(q['b'], q['b']), q['L'], q['o'] if O else q['u']] for q in mine
                              if q['main'] and not q['odd'] and (q['o'] if O else q['u']) is not None]   # owner desk Details: every CO book's main line
                for r in pool + [q for q in mine if q not in pool and not q['odd']]:
                    am = r['o'] if O else r['u']
                    if am is None: continue
                    is_long = am > LAB['max_price']   # long-shot rungs are never the pick, but are tracked on their own
                    if not is_long and r not in pool: continue
                    imp = sorted(1 / dec(q['o'] if O else q['u']) for q in allr if q['L'] == r['L'] and not q['odd'] and (q['o'] if O else q['u']) is not None)
                    if len(imp) >= 3 and 1 / dec(am) < 0.8 * imp[len(imp) >> 1]: continue   # stale or different market
                    c, pp = cond(stat, s, r['L']); fo, fsrc = prop_fair(stat, s, allr, r['L'])
                    oc = c if O else 1 - c; fc = None if fo is None else (fo if O else 1 - fo)
                    oc = min(max(oc, 1e-4), 1 - 1e-4); fc = None if fc is None else min(max(fc, 1e-4), 1 - 1e-4)
                    tr, ev_t, ev_o, pw, pl, d = price_row(oc, fc, pp, trust, am)
                    row = dict(market='prop', stat=stat, side=side, pick=f"{p['n']} {side.lower()} {r['L']} {stat}", player=p['n'], team=p['t'], line=r['L'],
                               book=r['b'], book_name=NAMES.get(r['b'], r['b']), price=am, proj=s.get('m'), ours=round(oc, 4),
                               fair=None if fc is None else round(fc, 4), trusted=round(tr, 4), push=round(pp, 4), ev=round(ev_t, 4), ev_ours=round(ev_o, 4),
                               trust=trust, fair_from=fsrc, books=side_books, _pw=pw, _pl=pl, _d=d)
                    if is_long:
                        if best_long is None or row['ev'] > best_long['ev']: best_long = row
                    elif best is None or row['ev'] > best['ev']: best = row
                tag = dict(sp=sp, game=str(p.get('g')), gid=game_of(sp, g), d=p.get('d'), matchup=f"{g['a']} @ {g['h']}", k=g['k'], timing=None, hours_to_start=round(hours(k, NOW), 1), big_dog=False)
                if best_long:
                    x = finish(best_long, *prop_caps(sp, stat, side), demote.get((sp, 'props')))
                    if x['would_be'] != 'Pass': x.update(tag, old=None); out.append(x)
                if not best: continue
                r_far = best['line'] >= 3 and abs(s['m'] - best['line']) / best['line'] > 0.25
                cap_lab, cap_why = prop_caps(sp, stat, side); gd = []
                if best['fair_from'] != 'two-sided' and cap_lab != 'Pass':   # ladders / anytime scorer: the fair price is a guess
                    cap_lab, cap_why = 'Lean', 'only one-sided prices are posted, so the books\' fair price is an estimate'; gd.append('one_sided')
                best = finish(best, cap_lab, cap_why, demote.get((sp, 'props')), gd)
                if r_far and 'far' not in best['guards']: best['guards'].append('far_proj')
                if r_far and best['label'] != 'Pass':
                    best['label'], best['stake_u'] = 'Pass', 0.0
                    best['reason'] = f"We project {s['m']} against a line of {best['line']}: a gap this size usually means news we are missing, so no bet"
                best.update(tag, old=old.get('|'.join(str(x) for x in (sp, p.get('g'), p['n'], stat, side))) or 'Pass')
                if best['label'] == 'Pass' and best['old'] == 'Pass' and best['would_be'] == 'Pass': n_pass += 1; quiet.append(best); continue   # keep the file small: quiet passes counted only
                out.append(best)
    return out, n_pass, quiet

# ---------------------------------------------------------------- run ---------------------------------------------------------
def compare(rows):
    """new label vs old tier, per sport x market: how many rows moved where"""
    C = {}
    for r in rows:
        key = f"{r['sp']} {r['market']}"; c = C.setdefault(key, {})
        o = r.get('old') or 'none'; c[f"{o}->{r['label']}"] = c.get(f"{o}->{r['label']}", 0) + 1
    return {k: dict(sorted(v.items())) for k, v in sorted(C.items())}

def main():
    slate = jload(f"{APP}/game_slates.json") or {}
    D = demotions(); rows = []; quiet = {}; quiet_rows = []; errors = {}
    jobs = [('NFL', lambda: ab_games('NFL', 'nfl_ab.json', slate, D)), ('NBA', lambda: ab_games('NBA', 'nba_ab.json', slate, D)),
            ('CBB', lambda: ab_games('CBB', 'cbb_ab.json', slate, D)), ('CFB', lambda: ab_games('CFB', 'cfb_ab.json', slate, D)),
            ('NHL', lambda: nhl_games(D)), ('MLB', lambda: mlb_games(slate, D))]
    for sp, f in jobs:
        try: rows += f()
        except Exception as e: errors[sp] = repr(e); print(sp, 'games failed:', repr(e))
    for sp in ('NFL', 'NBA', 'NHL'):
        try:
            r, q, qr = props(sp, D); rows += r; quiet[sp] = q; quiet_rows += qr
        except Exception as e: errors[f"{sp} props"] = repr(e); print(sp, 'props failed:', repr(e))
    for r in rows:
        r['conf'], r['conf_why'] = confidence(r); r.pop('ctx', None)
    rows.sort(key=lambda r: (-RANK[r['label']], -r['ev']))
    # guarded = would have been a Bet or Lean but a guard held it to Pass: kept apart and graded per guard
    blocked = lambda r: r['label'] == 'Pass' and r['would_be'] != 'Pass' and r['guards']
    guarded = [r for r in rows if blocked(r)]; rows = [r for r in rows if not blocked(r)]
    for r in guarded: r['conf'] = confidence(dict(r, label=r['would_be']))[0]
    gsum = {}
    for r in guarded:
        for gd in r['guards']:
            c = gsum.setdefault(gd, {'Bet': 0, 'Lean': 0, 'big_dog': 0}); c[r['would_be']] += 1; c['big_dog'] += r['big_dog']
    out = dict(made=NOW.strftime('%Y-%m-%d %H:%M UTC'), step='1 (shadow: nothing on the desk reads this yet)',
               rules=dict(trust='trusted = logit(fair) + trust x (logit(ours) - logit(fair))', labels=LAB, timing=TIM, demote=CFG['demote']),
               demoted=[dict(sp=k[0], market=k[1], why=v) for k, v in sorted(D.items())],
               counts={l: sum(1 for r in rows if r['label'] == l) for l in ('Bet', 'Lean', 'Pass')}, quiet_prop_passes=quiet,
               compare=compare(rows), guarded_counts=gsum, errors=errors, games=sorted(GAMES, key=lambda x: (x['k'], x['sp'])), bets=rows, guarded=guarded)
    hf = f"{HERE}/history/priced.json.gz"; os.makedirs(os.path.dirname(hf), exist_ok=True)
    H = jload(hf, {}) or {}; n = 0
    # owner desk Details: when we first priced this bet, at what line and price (from the history, else now)
    hkey = lambda r: '|'.join(str(r.get(x) or '') for x in ('sp', 'game', 'market', 'stat', 'side'))
    for r in rows:
        f0 = (H.get(hkey(r)) or {}).get('first') or {}
        r['first'] = dict(made=f0.get('made') or out['made'], line=f0.get('line', r.get('line')), price=f0.get('price', r.get('price')), label=f0.get('label', r['label']))
    json.dump(out, open(f"{APP}/priced_bets.json", 'w'), separators=(',', ':'))
    # history: last pregame row per bet (key sp|game|market|stat|side), first-seen label and price kept for opener grading
    # EVERY priced row is kept, Pass included: step 2's trust refit needs the bets we passed on too (fitting only on the
    # rows we liked would overstate how much of our gap is real)
    for r in rows + guarded + quiet_rows:
        key = '|'.join(str(r.get(x) or '') for x in ('sp', 'game', 'market', 'stat', 'side'))
        if any(x in r['guards'] for x in ('off_line', 'long')): key += '|' + ('off_line' if 'off_line' in r['guards'] else 'long')   # a second row for the same bet
        first = (H.get(key) or {}).get('first') or dict(made=out['made'], **{x: r.get(x) for x in ('label', 'would_be', 'guards', 'line', 'book', 'price', 'ev', 'ev_ours',
                                                                                                 'ours', 'fair', 'trusted', 'push', 'trust', 'timing', 'stake_u', 'conf')})
        H[key] = {**{x: v for x, v in r.items() if x not in ('books', 'conf_why')}, 'made': out['made'], 'first': first}; n += 1   # per-book prices: display only
    json.dump(H, gzip.open(hf, 'wt'), separators=(',', ':'))
    print('guarded', len(guarded), gsum)
    print('priced', len(rows), 'rows', out['counts'], 'quiet prop passes', quiet, 'history', len(H), f'(+{n})')
    for d in out['demoted']: print('demoted', d['sp'], d['market'], '-', d['why'])
    for k, v in out['compare'].items(): print(' ', k, v)
    if errors: sys.exit(1)

if __name__ == '__main__':
    main()
