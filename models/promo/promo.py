"""Promo optimizer (J 2026-10-18): which bets turn sportsbook promo dollars into the most cash, with no hedging or arbitrage.

Promo dollars are treated as bonus bets: the stake is NOT returned (DraftKings / FanDuel style), so a win pays only the profit.
    conversion = P(hit, OUR model) x (decimal odds - 1)
e.g. a $100 bonus bet converting at 60% is worth $60 cash on average. Only odds of +100 or longer count (J: never shorter than
even money); +300 and up is J's preferred range. Picks are ranked by conversion inside three odds bands, so within a band the
higher pick is also the likelier one to hit.

Two kinds of pick:
  priced    = real sportsbook odds from The Odds API (game lines: moneyline, spread, total) for NFL/CFB/CBB/NBA. Parlays here are
              cross-game only, one book per parlay, priced the way books price them (legs multiplied). Our chance per leg comes
              from Bet Desk's own game cards in "Our model" mode (probe_games.js), so it matches the app exactly.
  estimated = player props. The free odds plan has no prop odds, so the book price is estimated: the book is assumed to centre
              the player halfway between our projection and his comparable-workload average (add_recent.py, J's 75-133% workload
              rule; shift capped at 10%) and charge a 20% margin per leg on alt lines (cross-game parlays multiply legs, like books do). Type the real price before betting.
Same-game stacks come from parlay_suggestions.json (models/parlay/suggest.py), with a 25% same-game-parlay margin assumed.

Market check: every priced pick also shows the conversion if the books' no-vig chances are right. Our models showed no edge
against closing lines, so a big gap between the two is more often model error than value ("model_gap" marks it).

Run (after app/export_slates.py and the props refresh):  python3 promo.py      -> app/data/promo_picks.json"""
import json, os, sys, math, itertools, subprocess, datetime as dt
from zoneinfo import ZoneInfo

HERE = os.path.dirname(os.path.abspath(__file__)); D = os.path.join(HERE, '../../app/data')
ET = ZoneInfo('America/New_York'); NOW = dt.datetime.now(dt.timezone.utc); TODAY = NOW.astimezone(ET).date().isoformat()
BOOKS = {'draftkings': 'DraftKings', 'fanduel': 'FanDuel', 'betmgm': 'BetMGM', 'betrivers': 'BetRivers', 'williamhill_us': 'Caesars',
         'espnbet': 'ESPN BET', 'fanatics': 'Fanatics', 'hardrockbet': 'Hard Rock', 'ballybet': 'Bally Bet'}   # US regulated books only
BANDS = [('+100 to +299', 2.0, 4.0), ('+300 to +599', 4.0, 7.0), ('+600 to +1500', 7.0, 16.0)]
PER_BAND, MAX_LEG_USE, PROP_HOLD, SGP_HOLD, GAP, TAKE = 6, 2, 1.20, 1.25, 1.5, 0.70
W_TEAM, W_PROP = 0.30, 0.50     # Bet Desk default model weights (Settings: team bets 30%, props 50%)
NFL_ML_NOTE = 'NFL moneyline chances from our model have run optimistic'

def dec(a): return 1 + (a / 100 if a > 0 else 100 / -a)
def am(d): return f'+{round((d - 1) * 100)}' if d >= 2 else f'{round(-100 / (d - 1))}'
def band(d): return next((b for b, lo, hi in BANDS if lo <= d < hi), None)
def novig(a, b):
    """Bet Desk's devig(): power method."""
    if a is None or b is None: return None
    pa, pb = 1 / dec(a), 1 / dec(b); lo, hi = 0.5, 3.0
    for _ in range(60):
        k = (lo + hi) / 2
        if pa ** k + pb ** k > 1: lo = k
        else: hi = k
    return pa ** ((lo + hi) / 2)
def lg(p): p = min(max(p, 1e-6), 1 - 1e-6); return math.log(p / (1 - p))
def blend(pm, pf, w):
    """Bet Desk's blended chance (priceSides): logit blend, model weight w."""
    return 1 / (1 + math.exp(-(w * lg(pm) + (1 - w) * lg(pf))))

# ---------- game legs (real odds) ----------
def game_legs():
    out = os.path.join(HERE, '.probe.json')
    subprocess.run(['node', os.path.join(HERE, 'probe_games.js'), out], check=True, stderr=subprocess.DEVNULL)
    G = json.load(open(out)); os.remove(out)
    legs = []
    for g in G['games']:
        if dt.datetime.fromisoformat(g['k'].replace('Z', '+00:00')) <= NOW: continue
        bk = {b: r for b, r in (g['ln'].get('bk') or {}).items() if b in BOOKS}
        if not bk: continue
        for r in g['rows']:
            name_a, name_b, pa, kind = r[0], r[1], r[2], (r[5] if len(r) > 5 else ('ml' if 'win' in r[0] else 'total'))
            push = r[6] if len(r) > 6 and r[6] else 0.0
            if kind == 'ml': idx, line = (6, 7), None
            elif kind == 'spread': idx, line = (1, 2), r[4]
            else: idx, line = (4, 5), r[4]
            for side, nm, p in ((0, name_a, pa), (1, name_b, 1 - pa)):
                if kind == 'spread': p *= (1 - push)        # a push is counted as no win (a parlay leg that pushes just drops out)
                odds = {}
                for b, row in bk.items():
                    if kind == 'spread' and row[0] != line: continue
                    if kind == 'total' and row[3] != line: continue
                    if row[idx[side]] is not None: odds[b] = (row[idx[side]], novig(row[idx[side]], row[idx[1 - side]]))
                if not odds: continue
                gm = f"{g['a']} @ {g['h']}"
                legs.append(dict(sp=g['sp'], game=gm, gid=g['sp'] + g['id'], k=g['k'], d=g['d'], kind=kind,
                                 pick=nm if kind != 'total' else f'{gm} {nm}', pm=p, odds=odds, model_score=f"{g['h']} {g['hs']:.1f}, {g['a']} {g['as']:.1f}"))
    return legs

def best_book(odds):
    b = max(odds, key=lambda b: dec(odds[b][0])); return b, odds[b]
def pb_at(l, b): return blend(l['pm'], l['odds'][b][1], W_TEAM)

def priced_singles(legs):
    out = []
    for l in legs:
        b, (a, mk) = best_book(l['odds']); d = dec(a)
        if d < 2.0 or d >= 16 or l['kind'] != 'ml': continue      # singles: moneylines (spreads/totals sit near -110)
        out.append(pick('single', [l], d, pb_at(l, b), mk, b, priced=True, pm=l['pm']))
    return out

def priced_parlays(legs, k_max=4):
    # pool: the legs our model likes best per dollar (P x decimal odds at the best book), max 2 sides per game, 1 per parlay
    for l in legs: b, (a, _) = best_book(l['odds']); l['p'] = pb_at(l, b); l['ev'] = l['p'] * dec(a)
    pool = sorted([l for l in legs if l['p'] >= 0.35], key=lambda l: -l['ev'])
    seen = {}; P = []
    for l in pool:
        if seen.get(l['gid'], 0) >= 2: continue
        seen[l['gid']] = seen.get(l['gid'], 0) + 1; P.append(l)
        if len(P) >= 36: break
    out = []
    for k in range(2, k_max + 1):
        for c in itertools.combinations(P, k):
            if len({l['gid'] for l in c}) < k: continue
            books = set.intersection(*(set(l['odds']) for l in c))
            if not books: continue
            b = max(books, key=lambda b: math.prod(dec(l['odds'][b][0]) for l in c))
            d = math.prod(dec(l['odds'][b][0]) for l in c)
            if not 2.0 <= d < 16: continue
            p = math.prod(pb_at(l, b) for l in c); mk = math.prod(l['odds'][b][1] for l in c)
            out.append(pick('parlay', list(c), d, p, mk, b, priced=True, pm=math.prod(l['pm'] for l in c)))
    return out

# ---------- prop legs (estimated odds) ----------
GRID = {'NBA': {'pts': [9.5, 14.5, 19.5, 24.5, 29.5, 34.5, 39.5], 'reb': [3.5, 5.5, 7.5, 9.5, 11.5, 13.5], 'ast': [3.5, 5.5, 7.5, 9.5, 11.5],
                'fg3m': [0.5, 1.5, 2.5, 3.5, 4.5, 5.5], 'blk': [0.5, 1.5, 2.5, 3.5], 'stl': [0.5, 1.5, 2.5]},
        'NHL': {'goals': [0.5, 1.5], 'points': [0.5, 1.5, 2.5], 'assists': [0.5, 1.5], 'sog': [1.5, 2.5, 3.5, 4.5, 5.5]},
        'NFL': {'rec_yds': [39.5, 49.5, 59.5, 69.5, 79.5, 99.5, 124.5], 'rush_yds': [39.5, 49.5, 59.5, 69.5, 79.5, 99.5, 124.5],
                'pass_yds': [224.5, 249.5, 274.5, 299.5, 324.5], 'rec': [3.5, 4.5, 5.5, 6.5, 7.5, 8.5, 9.5], 'pass_td': [1.5, 2.5, 3.5], 'td': [0.5]}}
LABEL = {'pts': 'points', 'reb': 'rebounds', 'ast': 'assists', 'fg3m': 'threes', 'blk': 'blocks', 'stl': 'steals', 'goals': 'goals',
         'points': 'points', 'assists': 'assists', 'sog': 'shots on goal', 'rec_yds': 'receiving yds', 'rush_yds': 'rushing yds',
         'pass_yds': 'passing yds', 'rec': 'receptions', 'pass_td': 'passing TDs', 'td': 'anytime TD'}
sys.path.insert(0, os.path.join(HERE, '../parlay'))
from suggest import p_over   # Bet Desk's own ladder interpolation (logit, extrapolated past the ends)

def role_ok(sp, p):
    if sp == 'NBA':
        try: return int(p['p'].split('·')[-1].split()[0]) >= 15          # J's flag rule: a real role, 15+ min
        except Exception: return False
    if sp == 'NHL': return p['p'] != 'G' and p.get('toi', 0) >= 12
    return True

def prop_legs():
    sl = json.load(open(f'{D}/game_slates.json')); kick = {g['id']: g['k'] for g in (sl.get('NFL') or {}).get('games', [])}
    legs = []
    for sp, fn in (('NBA', 'nba_props.json'), ('NHL', 'nhl_props.json'), ('NFL', 'nfl_props.json')):
        if not os.path.exists(f'{D}/{fn}'): continue
        F = json.load(open(f'{D}/{fn}'))
        for p in F['players']:
            if not role_ok(sp, p): continue
            if sp == 'NFL':
                k = kick.get(p['g'])
                if not k or dt.datetime.fromisoformat(k.replace('Z', '+00:00')) <= NOW: continue
                d = dt.datetime.fromisoformat(k.replace('Z', '+00:00')).astimezone(ET).date().isoformat()
            else:
                d = p.get('d'); k = None
                if not d or d < TODAY: continue
            for st, lines in GRID[sp].items():
                s = p['s'].get(st)
                if not s: continue
                if st == 'td' and not s['m'] > 0.5: continue                    # J's rule: anytime TD only when we project 0.5+ TD
                a = s.get('a'); avg = (a[0] + a[1]) / 2 if a and a[2] >= 3 else None
                for line in lines:
                    pr = p_over(s['c'], line)
                    if not 0.12 <= pr <= 0.6: continue
                    # the book's view: our ladder re-centred halfway toward the player's comparable-workload average
                    # (books have projections too), the shift capped at 10% either way: the price is a guess, so it stays near ours
                    r = s['m'] / (0.5 * s['m'] + 0.5 * avg) if avg and avg > 0 and s['m'] > 0 else 1.0
                    pb = p_over(s['c'], line * min(1.1, max(0.9, r)))
                    d_est = 1 / (min(0.97, pb * PROP_HOLD))
                    legs.append(dict(sp=sp, game=f"{p['t']} vs {p['o']}", gid=f"{sp}{p['g']}", k=k, d=d, kind='prop',
                                     pick=f"{p['n']} ({p['t']}) {math.ceil(line)}+ {LABEL[st]}", pm=pr, p=blend(pr, pb, W_PROP), p_book=pb, d_est=d_est,
                                     proj=s['m'], avg=None if avg is None else round(avg, 2), player=p['n']))
    return legs

def prop_singles(legs):
    return [pick('single', [l], l['d_est'], l['p'], None, None, priced=False, pm=l['pm'], pbook=l['p_book']) for l in legs if 2.0 <= l['d_est'] < 16]

def prop_parlays(legs, k_max=3):
    for l in legs: l['ev'] = l['p'] * l['d_est']
    pool, seen = [], set()
    for l in sorted([l for l in legs if l['p'] >= 0.3], key=lambda l: -l['ev']):
        if l['player'] in seen: continue
        seen.add(l['player']); pool.append(l)
        if len(pool) >= 30: break
    out = []
    for k in range(2, k_max + 1):
        for c in itertools.combinations(pool, k):
            if len({l['gid'] for l in c}) < k: continue                       # cross-game only: independent, books multiply
            d = math.prod(l['d_est'] for l in c)
            if 2.0 <= d < 16: out.append(pick('parlay', list(c), d, math.prod(l['p'] for l in c), None, None, priced=False, pm=math.prod(l['pm'] for l in c), pbook=math.prod(l['p_book'] for l in c)))
    return out

def sgp_stacks():
    f = f'{D}/parlay_suggestions.json'
    if not os.path.exists(f): return []
    S = json.load(open(f)); out = []
    for sp, v in S['sports'].items():
        for c in v.get('correlation', []) + v.get('flag', []):
            if c.get('date') and c['date'] < TODAY: continue
            pbk = 1 / dec(int(c['book_proxy_odds'])); d = 1 / min(0.97, pbk * SGP_HOLD)
            if not 2.0 <= d < 16: continue
            legs = [dict(sp=sp, game=' vs '.join(c['teams']), gid=sp + str(c['game']), d=c.get('date'), kind='prop', p=l['p'], pm=l['p'],
                         pick=f"{l['player']} ({l['team']}) {l['dir']} {l['line']} {l['stat']}", player=l['player'], proj=l['proj']) for l in c['legs']]
            x = pick('same-game', legs, d, blend(c['fair_joint'], pbk, W_PROP), None, None, priced=False, pm=c['fair_joint'], pbook=pbk)
            x['why'] = c.get('why'); out.append(x)
    return out

def pick(kind, legs, d, p, mk, book, priced, pm, pbook=None):
    """p = Bet Desk's blended chance (what its EV uses), pm = our model alone."""
    conv = p * (d - 1)
    x = dict(type=kind, priced=priced, sport=sorted({l['sp'] for l in legs}), day=min(l['d'] for l in legs if l.get('d')) if any(l.get('d') for l in legs) else None,
             odds=am(d), dec=round(d, 3), band=band(d), book=BOOKS.get(book) if book else None, hit=round(p, 4), model_hit=round(pm, 4), fair_odds=am(1 / p) if p > 0 else None,
             conversion=round(conv, 4), model_conversion=round(pm * (d - 1), 4),
             legs=[{kk: (round(vv, 4) if isinstance(vv, float) else vv) for kk, vv in l.items() if kk in ('sp', 'game', 'pick', 'p', 'pm', 'k', 'proj', 'avg', 'model_score')} for l in legs])
    if priced:
        for l, j in zip(legs, x['legs']): j['odds'] = l['odds'][book][0]; j['book_p'] = round(l['odds'][book][1], 4); j['p'] = round(pb_at(l, book), 4)
        x['market_hit'] = round(mk, 4); x['market_conversion'] = round(mk * (d - 1), 4)
        x['model_gap'] = pm > GAP * mk
    else:
        x['est_book_implied'] = round(1 / d, 4)             # with the assumed margin
        x['est_book_hit'] = round(pbook, 4) if pbook else None   # the book's view without margin (hit sits between this and model_hit)
    notes = []
    if any(l['sp'] == 'NFL' and l['kind'] == 'ml' for l in legs): notes.append(NFL_ML_NOTE)
    if not priced:
        x['take_at'] = am(1 + TAKE / p)
        notes.append(f"estimated book price: check the real odds; worth it at {x['take_at']} or longer (converts {TAKE:.0%})")
    if notes: x['notes'] = notes
    return x

def top(picks, n=PER_BAND):
    """Best conversion per band, each leg in at most MAX_LEG_USE picks so the list is not one leg in ten wrappers."""
    out = []
    for b, _, _ in BANDS:
        use, got = {}, 0
        for x in sorted([x for x in picks if x['band'] == b], key=lambda x: -x['conversion']):
            if not x['priced'] and x['conversion'] < TAKE: continue   # estimated picks must clear their own take_at at the estimated price
            keys = [l['pick'] for l in x['legs']]
            if any(use.get(kk, 0) >= MAX_LEG_USE for kk in keys): continue
            for kk in keys: use[kk] = use.get(kk, 0) + 1
            out.append(x); got += 1
            if got >= n: break
    return out

if __name__ == '__main__':
    gl, pl, sg = game_legs(), prop_legs(), sgp_stacks()
    for l in gl: l['d'] = dt.datetime.fromisoformat(l['k'].replace('Z', '+00:00')).astimezone(ET).date().isoformat()   # ET day of kickoff
    days = sorted({l['d'] for l in gl + pl if l.get('d')} | {x['day'] for x in sg if x.get('day')})
    by_day = {}
    for d in days:   # J: the best longshots and parlays on any given day, so every list is built from that day's games only
        g1, p1 = [l for l in gl if l['d'] == d], [l for l in pl if l['d'] == d]
        groups = {'game_singles': top(priced_singles(g1)), 'game_parlays': top(priced_parlays(g1)),
                  'prop_singles': top(prop_singles(p1)), 'prop_parlays': top(prop_parlays(p1)), 'same_game': top([x for x in sg if x['day'] == d], 4)}
        if any(groups.values()): by_day[d] = groups
    J = dict(made=NOW.strftime('%Y-%m-%d %H:%M UTC'), rule='bonus bet, stake not returned: conversion = hit chance x (decimal odds - 1)',
             weights=dict(team=W_TEAM, props=W_PROP), bands=[b for b, _, _ in BANDS], books=list(BOOKS.values()), days=by_day,
             note=('Game picks use real odds (best US regulated book). Prop picks use an estimated book price (player centred halfway to his '
                   'comparable-workload average, 20% margin per leg on alt lines; same-game stacks 25%): type the real price first. '
                   'hit / conversion = Bet Desk\'s blended chance (our model at the Settings weight, 30% team bets and 50% props, the rest the market), '
                   'the same chance every Bet Desk edge uses; model_hit / model_conversion = our model alone. '
                   'market_conversion = conversion if the books are right; model_gap = our model alone is 1.5x+ the market, more often model error than value.'))
    json.dump(J, open(f'{D}/promo_picks.json', 'w'), separators=(',', ':'))
    for d, groups in by_day.items():
        print(f'######## {d}')
        for g, xs in groups.items():
            print(f'== {g} ({len(xs)})')
            for x in xs:
                print(f"  [{x['band']}] {x['odds']:>6} {x['book'] or 'est.':10} hit {x['hit']:.1%} (model {x['model_hit']:.1%}) conv {x['conversion']:.0%}"
                      + (f" mkt {x['market_conversion']:.0%}{' GAP' if x['model_gap'] else ''}" if x['priced'] else '') + ' | ' + ' + '.join(f"{l['pick']} ({l.get('pm', l['p']):.0%})" for l in x['legs']))
