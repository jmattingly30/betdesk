"""Walk-forward tennis ratings. One pass over matches in time order, pre-match features out.

Ratings kept per player:
  elo, elo by surface (538-style k), match counts
  point ratings: serve s and return r on the logit scale, each with a variance (Kalman style):
     logit P(A wins a point on serve vs B) = mu[surface] + s_A + ss_A[surf] - r_B - rs_B[surf]
Context kept per player: last tournament date, recent matches/minutes/sets, tournament run so far.
"""
import sys, math, random
import numpy as np, pandas as pd
from collections import defaultdict

ROUND_ORD = {'Q1': 0, 'Q2': 1, 'Q3': 2, 'Q4': 3, 'ER': 4, 'R128': 5, 'R64': 6, 'R32': 7, 'R16': 8,
             'QF': 9, 'SF': 10, 'BR': 11, 'F': 12, 'RR': 8}


def lg(p):
    return math.log(p / (1 - p))


def ex(x):
    return 1 / (1 + math.exp(-x))


def load(tour):
    d = pd.read_parquet(f'../data/{tour}_matches.parquet')
    d = d[d.year >= (1991 if tour == 'atp' else 1984)].copy()
    for c in ['w_svpt', 'w_1stWon', 'w_2ndWon', 'l_svpt', 'l_1stWon', 'l_2ndWon', 'minutes', 'winner_rank', 'loser_rank',
              'winner_age', 'loser_age', 'winner_ht', 'loser_ht', 'best_of', 'W1', 'L1', 'W2', 'L2', 'W3', 'L3', 'W4', 'L4', 'W5', 'L5']:
        d[c] = pd.to_numeric(d[c], errors='coerce')
    d['Retirement'] = d.Retirement.astype(str).isin(['True', '1', 'TRUE'])
    d['date'] = pd.to_datetime(d.tourney_start_date)
    d['ro'] = d['round'].map(ROUND_ORD).fillna(7)
    wo = d.score.str.contains('W/O|w/o|Walkover|DEF|Def', na=False) | d.score.isna()
    d = d[~wo]
    # the source repeats many matches (same event, players, round): keep one copy, or later copies leak results
    d = d.drop_duplicates(['tourney_id', 'winner_id', 'loser_id', 'round'])
    d = d.sort_values(['date', 'tourney_id', 'ro', 'match_num']).reset_index(drop=True)
    return d


DEF = dict(k_num=250., k_off=5., k_pow=0.4, surf_w=0.5,
           v0=0.30, vmin=0.0, tau=0.12, drift=0.00012, surf_v0=0.06, surf_share=0.35,
           mu_lr=0.002)


def run(tour, P=DEF, seed=7, out=None, verbose=True):
    P = {**DEF, **P}
    d = load(tour)
    rng = random.Random(seed)
    elo = defaultdict(lambda: 1500.); eloS = defaultdict(lambda: 1500.)
    n = defaultdict(int); nS = defaultdict(int)
    s = defaultdict(float); r = defaultdict(float)
    vs = defaultdict(lambda: P['v0']); vr = defaultdict(lambda: P['v0'])
    ss = defaultdict(float); rs = defaultdict(float)
    last = {}                      # player -> last date seen (for variance drift)
    hist = defaultdict(list)       # player -> list of (date, tourney_id, minutes, sets, games)
    run_t = defaultdict(lambda: [None, 0, 0, 0])  # player -> [tourney_id, matches, sets, games] this event
    h2h = defaultdict(int)
    mu = {'Hard': lg(0.63 if tour == 'atp' else 0.56), 'Clay': lg(0.60 if tour == 'atp' else 0.54),
          'Grass': lg(0.66 if tour == 'atp' else 0.58), 'Carpet': lg(0.65 if tour == 'atp' else 0.57)}
    rows = []
    cols = d[['date', 'tourney_id', 'tourney_level', 'tourney_name', 'surface', 'best_of', 'round', 'winner_id', 'loser_id',
              'winner_name', 'loser_name', 'winner_rank', 'loser_rank', 'winner_age', 'loser_age', 'winner_ht', 'loser_ht',
              'winner_hand', 'loser_hand', 'winner_entry', 'loser_entry', 'winner_ioc', 'loser_ioc', 'w_svpt', 'w_1stWon', 'w_2ndWon',
              'l_svpt', 'l_1stWon', 'l_2ndWon', 'minutes', 'Retirement', 'score',
              'W1', 'L1', 'W2', 'L2', 'W3', 'L3', 'W4', 'L4', 'W5', 'L5', 'year']].itertuples(index=False)
    for m in cols:
        w, l = m.winner_id, m.loser_id
        surf = m.surface if m.surface in mu else 'Hard'
        # variance drift since last seen
        for p in (w, l):
            if p in last:
                dd = (m.date - last[p]).days
                if dd > 0:
                    vs[p] = min(P['v0'], vs[p] + P['drift'] * dd)
                    vr[p] = min(P['v0'], vr[p] + P['drift'] * dd)
        # pre-match features with A/B random
        a, b = (w, l) if rng.random() < 0.5 else (l, w)
        ya = 1 if a == w else 0
        sk = (surf, )
        pa = ex(mu[surf] + s[a] + ss[(a, surf)] - r[b] - rs[(b, surf)])
        pb = ex(mu[surf] + s[b] + ss[(b, surf)] - r[a] - rs[(a, surf)])
        f = dict(date=m.date, tid=m.tourney_id, level=m.tourney_level, tname=m.tourney_name, surface=surf, bo=m.best_of, rnd=m.round,
                 a=a, b=b, y=ya, ret=bool(m.Retirement), score=m.score,
                 elo_a=elo[a], elo_b=elo[b], elos_a=eloS[(a, surf)], elos_b=eloS[(b, surf)], n_a=n[a], n_b=n[b],
                 ns_a=nS[(a, surf)], ns_b=nS[(b, surf)], pa=pa, pb=pb,
                 vs_a=vs[a], vr_a=vr[a], vs_b=vs[b], vr_b=vr[b])
        for tag, p in (('a', a), ('b', b)):
            isw = p == w
            f['rank_' + tag] = m.winner_rank if isw else m.loser_rank
            f['age_' + tag] = m.winner_age if isw else m.loser_age
            f['ht_' + tag] = m.winner_ht if isw else m.loser_ht
            f['hand_' + tag] = m.winner_hand if isw else m.loser_hand
            f['entry_' + tag] = m.winner_entry if isw else m.loser_entry
            f['ioc_' + tag] = m.winner_ioc if isw else m.loser_ioc
            hl = hist[p]
            f['days_' + tag] = (m.date - hl[-1][0]).days if hl else np.nan
            rec = [x for x in hl[-40:] if (m.date - x[0]).days <= 21 and x[1] != m.tourney_id]
            f['m21_' + tag] = len(rec)
            f['g21_' + tag] = sum(x[4] for x in rec)
            rt = run_t[p]
            if rt[0] == m.tourney_id:
                f['tm_' + tag], f['ts_' + tag], f['tg_' + tag] = rt[1], rt[2], rt[3]
            else:
                f['tm_' + tag] = f['ts_' + tag] = f['tg_' + tag] = 0
            f['last_tg_' + tag] = rt[4] if (rt[0] == m.tourney_id and len(rt) > 4) else 0
            f['n90_' + tag] = sum(1 for x in hl[-60:] if (m.date - x[0]).days <= 90)
        f['h2h'] = h2h[(a, b)] - h2h[(b, a)]
        # outcome detail (winner perspective) for score models
        ws = [m.W1, m.W2, m.W3, m.W4, m.W5]; ls = [m.L1, m.L2, m.L3, m.L4, m.L5]
        gw = sum(x for x in ws if x == x); gl = sum(x for x in ls if x == x)
        sw = sum(1 for x, z in zip(ws, ls) if x == x and z == z and x > z)
        sl = sum(1 for x, z in zip(ws, ls) if x == x and z == z and z > x)
        f.update(games_w=gw, games_l=gl, sets_w=sw, sets_l=sl)
        has_pts = m.w_svpt == m.w_svpt and m.l_svpt == m.l_svpt and m.w_svpt and m.l_svpt and m.w_svpt > 10 and m.l_svpt > 10 and (m.w_1stWon + m.w_2ndWon) == (m.w_1stWon + m.w_2ndWon) and (m.l_1stWon + m.l_2ndWon) == (m.l_1stWon + m.l_2ndWon) and m.w_1stWon + m.w_2ndWon <= m.w_svpt and m.l_1stWon + m.l_2ndWon <= m.l_svpt
        if has_pts:
            wsw = m.w_1stWon + m.w_2ndWon; lsw = m.l_1stWon + m.l_2ndWon
            f['spw_a'] = (wsw / m.w_svpt) if ya else (lsw / m.l_svpt)
            f['spw_b'] = (lsw / m.l_svpt) if ya else (wsw / m.w_svpt)
            f['svpt_a'] = m.w_svpt if ya else m.l_svpt
            f['svpt_b'] = m.l_svpt if ya else m.w_svpt
        rows.append(f)
        # ---------- updates ----------
        # Elo (538 style)
        for E, N, kw, kl in ((elo, n, w, l), (eloS, nS, (w, surf), (l, surf))):
            pe = 1 / (1 + 10 ** ((E[kl] - E[kw]) / 400))
            Kw = P['k_num'] / (N[kw] + P['k_off']) ** P['k_pow']
            Kl = P['k_num'] / (N[kl] + P['k_off']) ** P['k_pow']
            if m.Retirement:
                Kw *= 0.5; Kl *= 0.5
            E[kw] += Kw * (1 - pe); E[kl] -= Kl * (1 - pe)
            N[kw] += 1; N[kl] += 1
        # point ratings (Kalman-like on logit scale)
        if has_pts and not m.Retirement:
            for srv, ret_, won, tot in ((w, l, wsw, m.w_svpt), (l, w, lsw, m.l_svpt)):
                pp = ex(mu[surf] + s[srv] + ss[(srv, surf)] - r[ret_] - rs[(ret_, surf)])
                obs = lg(min(max(won / tot, 0.02), 0.98))
                pred = lg(pp)
                ov = 1 / (tot * pp * (1 - pp)) + P['tau'] ** 2
                vsum = vs[srv] + vr[ret_]
                tot_v = vsum * (1 + P['surf_share']) + ov
                resid = obs - pred
                gs_ = vs[srv] / tot_v; gr_ = vr[ret_] / tot_v
                s[srv] += gs_ * resid; r[ret_] -= gr_ * resid
                ss[(srv, surf)] += P['surf_share'] * gs_ * resid; rs[(ret_, surf)] -= P['surf_share'] * gr_ * resid
                vs[srv] = max(vs[srv] * (1 - gs_), P['vmin']); vr[ret_] = max(vr[ret_] * (1 - gr_), P['vmin'])
                mu[surf] += P['mu_lr'] * resid
        # context
        mins = m.minutes if m.minutes == m.minutes else np.nan
        for p in (w, l):
            hist[p].append((m.date, m.tourney_id, mins, sw + sl, gw + gl))
            if len(hist[p]) > 80:
                hist[p] = hist[p][-60:]
            rt = run_t[p]
            if rt[0] != m.tourney_id:
                run_t[p] = [m.tourney_id, 1, sw + sl, gw + gl, gw + gl]
            else:
                run_t[p] = [m.tourney_id, rt[1] + 1, rt[2] + sw + sl, rt[3] + gw + gl, gw + gl]
            last[p] = m.date
        h2h[(w, l)] += 1
    F = pd.DataFrame(rows)
    if out:
        F.to_parquet(out)
    return F


def score(F, start='2012-01-01', end='2025-01-01', levels=None):
    x = F[(F.date >= start) & (F.date < end) & (~F.ret)]
    if levels is not None:
        x = x[x.level.isin(levels)]
    pe = 1 / (1 + 10 ** ((x.elo_b - x.elo_a) / 400))
    return x, pe


if __name__ == '__main__':
    tour = sys.argv[1]
    F = run(tour, out=f'../data/feat_{tour}.parquet')
    print(F.shape)
