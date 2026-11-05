"""Shared by app/add_recent.py (single props + flags) and models/parlay/suggest.py (parlays).

roles():   the role each player holds on his team (P1-P5, S1-S6/GOALIE, QB/RB1/WR1/WR2/REC3), ranked the way the
           correlation and condition tables were built.
contexts(): the pre-game conditions for each team in a props file (spread, total, home, month, expected goals).
shift():   the condition shift (probit dz) for one role-stat, from app/data/parlay_corr.json -> shifts.
apply_to_props(): folds those shifts into every mapped stat's projection, percentiles and over-curve, once
           (marks s['cs'] = dz and p['cond'] = active conditions, so it never applies twice)."""
import os, json, numpy as np
from scipy.stats import norm

D = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../app/data'))
STAT = {'NBA': {'pts': 'pts', 'reb': 'reb', 'ast': 'ast', 'fg3m': '3pm', 'stl': 'stl', 'blk': 'blk', 'tov': 'tov'},
        'NHL': {'points': 'pts', 'sog': 'sog', 'goals': 'goals', 'assists': 'ast', 'blk': 'blk', 'hits': 'hits', 'saves': 'saves'},
        'NFL': {'pass_yds': 'pass_yds', 'cmp': 'cmp', 'pass_td': 'pass_td', 'rush_yds': 'rush_yds', 'rec': 'rec', 'rec_yds': 'rec_yds', 'td': 'td'}}
OPS = {'<=': lambda x, v: x <= v, '>=': lambda x, v: x >= v, '==': lambda x, v: x == v, 'in': lambda x, v: x in v}
CAP = 0.3

def tables():
    return json.load(open(f'{D}/parlay_corr.json'))

def roles(sp, players):
    """id(player) -> role for one team's players."""
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

def role_stat(sp, role, stat):
    ls = STAT[sp].get(stat)
    if not ls or (sp == 'NFL' and ls == 'td' and role == 'QB'): return None
    return f'{role}_{ls}'

def contexts(sp, doc):
    """(game, team) -> context dict, and the game's players by team."""
    games = {}
    for p in doc['players']: games.setdefault(p['g'], {}).setdefault(p['t'], []).append(p)
    week = doc.get('week') if isinstance(doc.get('week'), int) else None
    out = {}
    for g, teams in games.items():
        if len(teams) != 2: continue
        for t, tp in teams.items():
            op = [x for o, xs in teams.items() if o != t for x in xs]
            ctx = {'h': tp[0].get('h'), 'sp': tp[0].get('sp'), 'week': week}
            if tp[0].get('d'): ctx['month'] = int(tp[0]['d'][5:7])
            if 'imp' in tp[0] and op and 'imp' in op[0]: ctx['tot'] = tp[0]['imp'] + op[0]['imp']
            if sp == 'NHL':
                gf = sum(x['s']['goals']['m'] for x in tp if 'goals' in x['s']); ga = sum(x['s']['goals']['m'] for x in op if 'goals' in x['s'])
                ctx.update(xg_diff=gf - ga, xg_tot=gf + ga)
            out[(g, t)] = (ctx, tp)
    return out

def active(T, sp, ctx):
    rules = T['shifts'].get(sp, {}).get('rules', {})
    return [n for n, conds in rules.items() if all(ctx.get(v) is not None and OPS[op](ctx[v], val) for v, op, val in conds)]

def shift(T, sp, role_stat_name, on):
    dz = sum(e['dz'] for e in T['shifts'].get(sp, {}).get('effects', []) if e['leg'] == role_stat_name and e['cond'] in on)
    return max(-CAP, min(CAP, dz))

def apply_to_props(sp, doc, T=None):
    """Fold condition shifts into the props file in place. Returns how many stats moved."""
    T = T or tables(); moved = 0
    for (g, t), (ctx, tp) in contexts(sp, doc).items():
        on = active(T, sp, ctx); rl = roles(sp, tp)
        for p in tp:
            if id(p) not in rl: continue
            used = set()
            for k, s in p['s'].items():
                if 'cs' in s: continue
                rs = role_stat(sp, rl[id(p)], k)
                if not rs: continue
                dz = shift(T, sp, rs, on)
                if abs(dz) < 1e-6: continue
                sd = max((s['q'][2] - s['q'][0]) / 2.563, 1e-6) if s.get('q') else 0.0   # q = 10/50/90 percentiles
                s['c'] = [[ln, round(float(norm.cdf(norm.ppf(min(max(pv, 1e-4), 1 - 1e-4)) + dz)), 4)] for ln, pv in s['c']]
                dec = 2 if s['m'] < 5 else 1
                s['m'] = round(max(0.0, s['m'] + dz * sd), dec)
                if s.get('q'): s['q'] = [round(max(0.0, x + dz * sd), 1) for x in s['q']]
                s['cs'] = round(dz, 3); moved += 1
                used |= {e['cond'] for e in T['shifts'][sp]['effects'] if e['leg'] == rs and e['cond'] in on}
            if used: p['cond'] = sorted(used)
    return moved
