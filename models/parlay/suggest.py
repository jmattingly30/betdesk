"""Parlay suggester: reads this week's props files, finds correlated same-game stacks from our projections.
Two kinds of leg line until J types the real one:
  - "fair line": our projection set to the nearest .5 (where a sharp book would post it); the edge then comes from skew, condition shifts and correlation.
  - "flag line": for legs that pass the Bet Desk flag rule (projection 15%+ away from BOTH recent and season averages), the line is set at the average, where books tend to hang it.
Book proxy = flag legs at -110 (52.4%, the book thinks the average is fair); fair-line legs at our base chance plus vig (the book agrees with our projection);
only the OBVIOUS links priced in (same player, QB with his own pass catchers). So a fair-line stack's edge is only hidden links, condition shifts and tail shape.
Leg chance = our ladder at that line, nudged by condition shifts; joint = Gaussian copula on our latent correlations x tail factor.
Writes app/data/parlay_suggestions.json.  Run: python3 suggest.py"""
import json, itertools, math, datetime, numpy as np, sys, os
from scipy.stats import norm
sys.path.insert(0, os.path.dirname(__file__)); from copula import joint_gauss
D = os.path.join(os.path.dirname(__file__), '../../app/data')
T = json.load(open(f'{D}/parlay_corr.json'))
MIN_P, MAX_P, MIN_MULT, MIN_EV, POOL, PER_GAME, MAX_USE, FLAG = 0.50, 0.85, 1.05, 0.05, 14, 3, 2, 0.15
BOOK_P = 1 / 1.9091
STAT = {'NBA': {'pts': 'pts', 'reb': 'reb', 'ast': 'ast', 'fg3m': '3pm', 'stl': 'stl', 'blk': 'blk', 'tov': 'tov'},
        'NHL': {'points': 'pts', 'sog': 'sog', 'goals': 'goals', 'assists': 'ast', 'blk': 'blk', 'hits': 'hits', 'saves': 'saves'},
        'NFL': {'pass_yds': 'pass_yds', 'cmp': 'cmp', 'pass_td': 'pass_td', 'rush_yds': 'rush_yds', 'rec': 'rec', 'rec_yds': 'rec_yds', 'td': 'td'}}
NAME = {'pts': 'points', 'reb': 'rebounds', 'ast': 'assists', '3pm': 'threes', 'stl': 'steals', 'blk': 'blocks', 'tov': 'turnovers', 'sog': 'shots', 'goals': 'goals',
        'hits': 'hits', 'saves': 'saves', 'pass_yds': 'pass yds', 'cmp': 'completions', 'pass_td': 'pass TDs', 'rush_yds': 'rush yds', 'rec': 'receptions', 'rec_yds': 'rec yds', 'td': 'anytime TD'}

def roles(sp, players):
    """Role per player, ranked the same way the correlation tables were built."""
    out = {}
    if sp == 'NBA':
        for i, p in enumerate(sorted(players, key=lambda p: -p['s']['pts']['m'])[:5]): out[id(p)] = f'P{i+1}'
    elif sp == 'NHL':
        sk = [p for p in players if p['p'] != 'G']; gl = [p for p in players if p['p'] == 'G']
        for i, p in enumerate(sorted(sk, key=lambda p: -p['s']['points']['m'])[:6]): out[id(p)] = f'S{i+1}'
        if gl: out[id(max(gl, key=lambda p: p['s']['saves']['m']))] = 'GOALIE'
    else:
        qb = [p for p in players if p['p'] == 'QB']; rb = [p for p in players if p['p'] == 'RB']; rc = [p for p in players if p['p'] in ('WR', 'TE')]
        if qb: out[id(max(qb, key=lambda p: p['s']['pass_att']['m']))] = 'QB'
        if rb: out[id(max(rb, key=lambda p: p['s']['rush_yds']['m']))] = 'RB1'
        for i, p in enumerate(sorted(rc, key=lambda p: -p['s']['rec_yds']['m'])[:3]): out[id(p)] = ['WR1', 'WR2', 'REC3'][i]
    return out

def _lg(p): p = min(max(p, 1e-4), 1 - 1e-4); return float(np.log(p / (1 - p)))
def _il(x): return float(1 / (1 + np.exp(-x)))
# same rule as interp() in bet-desk.html: logit interpolation, and past either end of the ladder the end segment's
# logit slope (at least 0.1 per unit) instead of the end point (Bet Desk audit C3, 2026-10-17)
def p_over(ladder, line):
    xs = [c[0] for c in ladder]; ys = [c[1] for c in ladder]; n = len(xs)
    if n == 1: return ys[0]
    def ext(i, j): sl = min(-0.1, (_lg(ys[j]) - _lg(ys[i])) / (xs[j] - xs[i])); return _il(_lg(ys[i]) + sl * (line - xs[i]))
    if line < xs[0]: return ext(0, 1)
    if line > xs[-1]: return ext(n - 1, n - 2)
    k = max(i for i in range(n - 1) if xs[i] <= line)
    t = (line - xs[k]) / (xs[k + 1] - xs[k]); return _il(_lg(ys[k]) + (_lg(ys[k + 1]) - _lg(ys[k])) * t)

# the Bet Desk flag rule (flagOf in bet-desk.html): projection 15%+ beyond BOTH comparable-workload averages, a minimum gap,
# a size floor, 3+ comparable recent games (add_recent.py), anytime TD only when we project over 0.5
FLAG_MIN = {'NFL': dict(rec_yds=8, rush_yds=8, rush_rec_yds=10, pass_yds=20, targets=1, rec=0.8, carries=2, pass_att=3, cmp=2, pass_td=0.25, td=0.1),
            'NBA': dict(pts=2.5, reb=1, ast=1, fg3m=0.5, pra=3, stl=0.3, blk=0.3, tov=0.4), 'NHL': dict(sog=0.6, points=0.2, goals=0.12, assists=0.15, blk=0.5, hits=0.6, saves=2)}
FLAG_FLOOR = {'NFL': dict(rec_yds=20, rush_yds=20, rush_rec_yds=25, pass_yds=150, targets=3, rec=2, carries=6, pass_att=20, cmp=12, pass_td=0.8, td=0.25),
              'NBA': dict(pts=8, reb=3, ast=2, fg3m=1, pra=12, stl=0.6, blk=0.5, tov=1), 'NHL': dict(sog=1.5, points=0.4, goals=0.2, assists=0.25, blk=1, hits=1.5, saves=20)}
def flag_dir(sp, k, s, thr=FLAG):
    a = s.get('a')
    if not a or a[2] < 3: return None
    if sp == 'NFL' and k == 'td' and not s['m'] > 0.5: return None
    mn, fl = FLAG_MIN[sp].get(k, 1), FLAG_FLOOR[sp].get(k, 0)
    b = max(a[0], a[1])
    if b > 0 and s['m'] >= (1 + thr) * b and s['m'] - b >= mn and s['m'] >= fl: return 'over'
    b = min(a[0], a[1])
    if b > 0 and s['m'] <= (1 - thr) * b and b - s['m'] >= mn and b >= fl: return 'under'
    return None

def book_line(st, side):
    """Line at the player's average, rounded to .5 on the side that is WORSE for our bet (lower for an under, higher for an over)."""
    a = st.get('a') or [st['m'], st['m'], 0, 0]
    avg = (a[0] + a[1]) / 2
    return max(0.5, math.floor(avg - 0.5) + 0.5 if side == 'under' else math.ceil(avg - 0.5) + 0.5 if avg - math.floor(avg) != 0.5 else avg)

def active(sp, ctx):
    OPS = {'<=': lambda x, v: x <= v, '>=': lambda x, v: x >= v, '==': lambda x, v: x == v, 'in': lambda x, v: x in v}
    rules = T['shifts'].get(sp, {}).get('rules', {})
    return [name for name, conds in rules.items() if all(ctx.get(var) is not None and OPS[op](ctx[var], val) for var, op, val in conds)]

def american(dec):
    return f'+{round((dec - 1) * 100)}' if dec >= 2 else f'{round(-100 / (dec - 1))}'

def run_sport(sp, fn):
    d = json.load(open(f'{D}/{fn}')); C = T[sp]; idx = {n: i for i, n in enumerate(C['legs'])}; R = np.array(C['R'])
    eff = {}
    for e in T['shifts'].get(sp, {}).get('effects', []): eff.setdefault(e['leg'], []).append(e)
    games = {}
    for p in d['players']: games.setdefault(p['g'], []).append(p)
    week = d.get('week') if isinstance(d.get('week'), int) else None
    sugg = []
    for g, ps in games.items():
        teams = sorted({p['t'] for p in ps})
        if len(teams) != 2: continue
        legs = []; ctxs = {}
        for t in teams:
            tp = [p for p in ps if p['t'] == t]; op = [p for p in ps if p['t'] != t]
            ctx = {'h': tp[0].get('h'), 'sp': tp[0].get('sp'), 'week': week}
            if tp[0].get('d'): ctx['month'] = int(tp[0]['d'][5:7])
            if 'imp' in tp[0] and op and 'imp' in op[0]: ctx['tot'] = tp[0]['imp'] + op[0]['imp']
            if sp == 'NHL':
                gf = sum(p['s']['goals']['m'] for p in tp if 'goals' in p['s']); ga = sum(p['s']['goals']['m'] for p in op if 'goals' in p['s'])
                ctx.update(xg_diff=gf - ga, xg_tot=gf + ga)
            on = active(sp, ctx); ctxs[t] = on
            rl = roles(sp, tp)
            for p in tp:
                if id(p) not in rl: continue
                for sk, st in p['s'].items():
                    ls = STAT[sp].get(sk)
                    if not ls: continue
                    role = f'{rl[id(p)]}_{ls}' if not (sp == 'NFL' and ls == 'td' and rl[id(p)] == 'QB') else None
                    if not role or role not in idx: continue
                    a = st.get('a') or [st['m'], st['m'], 0, 0]
                    flag = flag_dir(sp, sk, st) is not None and (sp != 'NBA' or int(p['p'].split('·')[-1].split()[0] if '·' in p['p'] else 0) >= 15) and (sp != 'NHL' or p['p'] == 'G' or p.get('toi', 99) >= 12)
                    line = book_line(st, flag_dir(sp, sk, st)) if flag else max(0.5, round(st['m'] - 0.5) + 0.5)
                    po = p_over(st['c'], line)
                    dz = 0.0 if 'cs' in st else max(-0.3, min(0.3, sum(e['dz'] for e in eff.get(role, []) if e['cond'] in on)))   # cs = already folded in by add_recent.py
                    zc = lambda x: norm.ppf(min(max(x, 1e-4), 1 - 1e-4))
                    p0 = float(norm.cdf(zc(po) - st['cs'])) if 'cs' in st else po   # the book is assumed to know our base projection, not the condition shift
                    po = float(norm.cdf(zc(po) + dz))
                    for dr, pr, pb in (('over', po, p0), ('under', 1 - po, 1 - p0)):
                        if ls == 'td' and (dr == 'under' or st['m'] <= 0.5): continue     # J's rule: anytime TD only when we project more than 0.5 TD
                        if flag and dr != flag_dir(sp, sk, st): continue          # flagged legs only in the flag's direction
                        if MIN_P <= pr <= MAX_P:
                            legs.append(dict(player=p['n'], team=t, stat=NAME[ls], role=role, dir=dr, line=line, p=round(pr, 3), dz=round(dz, 3), proj=st['m'],
                                             line_type='flag' if flag else 'fair', avgs=a[:2],
                                             p_book=BOOK_P if flag else min(0.97, pb * 1.0476)))
        # one direction per player-stat already; keep the strongest legs
        legs = sorted(legs, key=lambda l: -l['p'])[:POOL]
        def corr(a, b):
            ra, rb = a['role'], (b['role'] if a['team'] == b['team'] else 'OPP_' + b['role'])
            if rb not in idx: return 0.0
            s = (1 if a['dir'] == 'over' else -1) * (1 if b['dir'] == 'over' else -1)
            return s * R[idx[ra], idx[rb]]
        def obvious(a, b):
            if a['player'] == b['player']: return True
            if sp == 'NFL' and a['team'] == b['team'] and 'QB' in (a['role'].split('_')[0], b['role'].split('_')[0]): return True
            return False
        cands = []
        for k in (2, 3, 4):
            for combo in itertools.combinations(legs, k):
                if max(sum(l['player'] == x['player'] for l in combo) for x in combo) > 2: continue          # at most two legs from one player
                if any(a['player'] == b['player'] and a['stat'] == b['stat'] for a, b in itertools.combinations(combo, 2)): continue
                M = np.array([[1.0 if i == j else corr(a, b) for j, b in enumerate(combo)] for i, a in enumerate(combo)])
                if (M[np.triu_indices(k, 1)] < 0).any(): continue           # every pair must help
                pr = np.array([l['p'] for l in combo]); ind = float(pr.prod())
                if k >= 3 and ind < 0.08: continue
                jt = joint_gauss(pr, M)
                dirs = {l['dir'] for l in combo}
                if k >= 3 and len(dirs) == 1:
                    jt *= T['tail_factor'][dirs.pop()].get(sp, {}).get(str(k), 1.0)
                mult = jt / ind
                if mult < MIN_MULT: continue
                Mo = np.array([[1.0 if i == j else (corr(a, b) if obvious(a, b) else 0.0) for j, b in enumerate(combo)] for i, a in enumerate(combo)])
                book = joint_gauss(np.array([l['p_book'] for l in combo]), Mo); ev_sep = jt / book - 1
                if ev_sep < MIN_EV: continue
                hidden = jt / joint_gauss(pr, Mo)
                cands.append(dict(k=k, legs=list(combo), fair_joint=round(jt, 4), corr_mult=round(mult, 3), hidden_mult=round(hidden, 3), ev_est=round(ev_sep, 3), book_proxy_odds=american(1 / book),
                                  fair_odds=american(1 / jt), bet_if_at_least=american(1.05 / jt),
                                  strongest_link=max(((corr(a, b), a, b) for a, b in itertools.combinations(combo, 2)), key=lambda x: x[0])))
        for c in cands: c['kind'] = 'flag' if any(l['line_type'] == 'flag' for l in c['legs']) else 'correlation'
        cands.sort(key=lambda c: -c['ev_est'])
        picked = []; use = {}
        for kind, cap in (('correlation', 2), ('flag', 1)):
            got = 0
            for c in cands:
                if got >= cap: break
                if c['kind'] != kind: continue
                keys = [(l['player'], l['stat']) for l in c['legs']]
                if any(use.get(x, 0) >= MAX_USE for x in keys): continue
                for x in keys: use[x] = use.get(x, 0) + 1
                r, a, b = c.pop('strongest_link')
                c['why'] = f"{a['player']} {a['stat']} {a['dir']} and {b['player']} {b['stat']} {b['dir']} move together (latent r {r:.2f})"
                c['all_under'] = all(l['dir'] == 'under' for l in c['legs'])
                c['conditions'] = {t: ctxs[t] for t in teams}
                c['game'] = g; c['teams'] = teams; c['date'] = next((p.get('d') for p in ps if p.get('d')), None)
                picked.append(c); got += 1
        sugg += picked
    sugg.sort(key=lambda c: -c['ev_est'])
    for c in sugg:
        for l in c['legs']: l.pop('p_book', None)
    return dict(made_from=d.get('made'), week=d.get('week'), correlation=[c for c in sugg if c['kind'] == 'correlation'], flag=[c for c in sugg if c['kind'] == 'flag'])

if __name__ == '__main__':
    out = {'made': datetime.datetime.now(datetime.timezone.utc).strftime('%Y-%m-%d %H:%M UTC'),
           'note': ('Correlated same-game stacks from our projections. Lines are our guess at the book line (recent and season average, nearest .5): '
                    'type the real line and the book SGP price before betting. fair_odds = our true price; bet only at bet_if_at_least or longer (+5% EV). '
                    'ev_est = edge if the book hangs each leg at -110 and prices only the obvious links (same player, QB with his receivers). hidden_mult = how much the links books ignore add. '
                    'line_type fair = our projection rounded to .5; flag = the leg passes the Bet Desk flag rule and the line is set at the player average.'),
           'sports': {}}
    for sp, fn in (('NBA', 'nba_props.json'), ('NFL', 'nfl_props.json'), ('NHL', 'nhl_props.json')):
        if os.path.exists(f'{D}/{fn}'): out['sports'][sp] = run_sport(sp, fn)
    json.dump(out, open(f'{D}/parlay_suggestions.json', 'w'), indent=1)
    for sp, v in out['sports'].items():
      for kind in ('correlation', 'flag'):
        print(sp, kind, len(v[kind]))
        for c in v[kind][:6]:
            print('  ', c['k'], ' + '.join(f"{l['player']} {l['stat']} {l['dir']} {l['line']}{'F' if l['line_type']=='flag' else ''} ({l['p']:.2f})" for l in c['legs']), '| fair', c['fair_odds'], 'book~', c['book_proxy_odds'], 'mult', c['corr_mult'], 'hid', c['hidden_mult'], 'ev', c['ev_est'])
