"""Bridge ratings to 2026: deuce results (to 2024-12) + TML ATP tour (2025-01..2026-01) + OddsPapi 2026 results
(winner from next-round appearance / in-play prices). Players keyed by normalized name. Elo only (538 k).
Writes ../data/bridge_{men,women}.parquet with pre-match Elo for each 2026 match."""
import unicodedata, re, sys
import numpy as np, pandas as pd
from collections import defaultdict


def norm(s):
    if not isinstance(s, str): return ''
    if ',' in s:
        last, first = s.split(',', 1); s = first + ' ' + last
    s = unicodedata.normalize('NFKD', s).encode('ascii', 'ignore').decode().lower()
    s = re.sub(r'[^a-z ]', ' ', s.replace('-', ' '))
    return ' '.join(s.split())


K = dict(k_num=250., k_off=5., k_pow=0.4)
MEN = {'ATP', 'Challenger', 'ITF Men', 'Davis Cup', 'United Cup'}
WOMEN = {'WTA', 'WTA 125K', 'ITF Women', 'Billie Jean King Cup'}


def score_games(sc):
    """'6-3 6-7(5) 4-6' -> (games first player, games second player); None if unparseable."""
    a = b = 0
    for st in str(sc).split():
        m = re.match(r'^(\d+)-(\d+)', st)
        if m: a += int(m.group(1)); b += int(m.group(2))
    return (a, b) if a + b > 0 else (np.nan, np.nan)


def short_key(s):
    """'Bharadwaj, Parashar' / 'Parashar Bharadwaj' / 'Bharadwaj P.' -> 'bharadwaj p' (last name + first initial)."""
    s = str(s)
    if ',' in s:
        last, first = s.split(',', 1)
    elif re.match(r'^(.*?)\s+([A-Z])\.[A-Z.\- ]*$', s.strip()):
        m = re.match(r'^(.*?)\s+([A-Z])\.[A-Z.\- ]*$', s.strip()); last, first = m.group(1), m.group(2)
    else:
        parts = s.split(); first, last = parts[0], ' '.join(parts[1:])
    n = lambda x: ' '.join(re.sub(r'[^a-z ]', ' ', unicodedata.normalize('NFKD', x).encode('ascii', 'ignore').decode().lower().replace('-', ' ')).split())
    return n(last) + ' ' + n(first)[:1]


def hist_rows(tour):
    d = pd.read_parquet(f'../data/{tour}_matches.parquet', columns=['tourney_id', 'tourney_start_date', 'round', 'match_num', 'winner_name', 'loser_name', 'score', 'surface', 'tourney_level', 'Retirement'] + [f'{x}{i}' for x in 'WL' for i in range(1, 6)])
    d['date'] = pd.to_datetime(d.tourney_start_date)
    d = d[d.date >= '2008-01-01']
    d = d[~d.score.str.contains('W/O|w/o|Walkover|DEF|Def', na=True)].drop_duplicates(['tourney_id', 'winner_name', 'loser_name', 'round'])
    gw = sum(pd.to_numeric(d[f'W{i}'], errors='coerce').fillna(0) for i in range(1, 6)); gl = sum(pd.to_numeric(d[f'L{i}'], errors='coerce').fillna(0) for i in range(1, 6))
    out = pd.DataFrame(dict(date=d.date, w=d.winner_name.map(norm), l=d.loser_name.map(norm), src='deuce', level=d.tourney_level, surface=d.surface,
                            ret=d.Retirement.astype(str).isin(['True', '1', 'TRUE']), gw=np.where(gw + gl > 0, gw, np.nan), gl=np.where(gw + gl > 0, gl, np.nan)))
    # 2025: tennisexplorer set (ATP tour + Challenger, WTA tour + ITF) with winners from sets; men's ITF 2025 is missing
    te = pd.read_csv('../../kg/alimoh89_tennis-results-and-betting-odds-20142025/tennis_matches_2014_2025.csv', low_memory=False,
                     usecols=['match_date_formatted', 'tour', 'tournament', 'surface', 'player_a', 'player_b', 'comment', 'total_sets_a', 'total_sets_b', 'total_games_a', 'total_games_b'])
    te['date'] = pd.to_datetime(te.match_date_formatted, errors='coerce')
    te = te[(te.tour == tour.upper()) & (te.date >= '2025-01-01') & (te.comment == 'completed') & (te.total_sets_a != te.total_sets_b)]
    aw = te.total_sets_a > te.total_sets_b
    sf = te.surface.str.lower().map({'hard': 'Hard', 'clay': 'Clay', 'grass': 'Grass', 'indoors': 'Hard'}).fillna('Hard')
    out = pd.concat([out, pd.DataFrame(dict(date=te.date.dt.normalize(), w=np.where(aw, te.player_a, te.player_b), l=np.where(aw, te.player_b, te.player_a),
                                            src='te', level=te.tournament, surface=sf.values, ret=False,
                                            gw=np.where(aw, te.total_games_a, te.total_games_b), gl=np.where(aw, te.total_games_b, te.total_games_a)))])
    if tour == 'atp':  # TML has the full ATP tour 2025; keep te only for Challengers there
        out = out[(out.src != 'te') | out.level.astype(str).str.contains('challenger', case=False)]
        t = pd.read_csv('../../dl/Tennismylife_TML-Database_master_2025.csv')
        t = t[~t.score.astype(str).str.contains('W/O|w/o|DEF', na=True)]
        t['date'] = pd.to_datetime(t.tourney_date.astype(str), format='%Y%m%d')
        out = pd.concat([out, pd.DataFrame(dict(date=t.date, w=t.winner_name, l=t.loser_name, src='tml', level=t.tourney_level,
                                                surface=t.surface, ret=t.score.astype(str).str.contains('RET'),
                                                gw=[score_games(x)[0] for x in t.score], gl=[score_games(x)[1] for x in t.score]))])
    out['w'] = out.w.astype(str).map(norm); out['l'] = out.l.astype(str).map(norm)
    return out.sort_values('date', kind='stable')


def games_2026(tour):
    """Tour-level 2026 scores (Kaggle dissfya daily files): {(date, frozenset(short keys)): {short_key: games}}."""
    f = '../../kg/dissfya_atp-tennis-2000-2023daily-pull/atp_tennis.csv' if tour == 'atp' else '../../kg/dissfya_wta-tennis-2007-2023-daily-update/wta.csv'
    d = pd.read_csv(f); d['Date'] = pd.to_datetime(d.Date, errors='coerce'); d = d[d.Date >= '2025-12-20']
    out = {}
    import glob
    for f in glob.glob('../../tex/results_*.csv'):  # tennisexplorer 2026 daily results (all levels except men's ITF)
        x = pd.read_csv(f)
        x = x[x.ev.str.contains('/atp-men/' if tour == 'atp' else '/wta-women/') & (x.nsets > 0) & (x.games_a + x.games_b > 5)]
        for r in x.itertuples(index=False):
            k1, k2 = short_key(r.pa), short_key(r.pb)
            out[(frozenset((k1, k2)), pd.Timestamp(r.day))] = {k1: r.games_a, k2: r.games_b}
    for r in d.itertuples(index=False):
        a, b = score_games(r.Score)
        if a != a: continue
        k1, k2 = short_key(r.Player_1), short_key(r.Player_2)
        out[(frozenset((k1, k2)), r.Date.normalize())] = {k1: a, k2: b}
    return out


def run(tour):
    H = hist_rows(tour)
    O = pd.read_parquet('../data/op_results_2026.parquet')
    O = O[O.cat.isin(MEN if tour == 'atp' else WOMEN)].copy()
    O['an_'] = O.an.map(norm); O['bn_'] = O.bn.map(norm)
    elo = defaultdict(lambda: 1500.); n = defaultdict(int); last = {}
    def upd(w, l, ret=False):
        pe = 1 / (1 + 10 ** ((elo[l] - elo[w]) / 400))
        kw = K['k_num'] / (n[w] + K['k_off']) ** K['k_pow']; kl = K['k_num'] / (n[l] + K['k_off']) ** K['k_pow']
        if ret: kw *= .5; kl *= .5
        elo[w] += kw * (1 - pe); elo[l] -= kl * (1 - pe); n[w] += 1; n[l] += 1
    for r in H.itertuples(index=False):
        upd(r.w, r.l, r.ret); last[r.w] = last[r.l] = r.date
    known_end = set(n)
    rows = []
    O = O.sort_values('start')
    for r in O.itertuples(index=False):
        a, b = r.an_, r.bn_
        rows.append(dict(fid=r.fid, start=r.start, cat=r.cat, tour=r.tour, a=a, b=b, elo_a=elo[a], elo_b=elo[b], n_a=n[a], n_b=n[b],
                         seen_a=a in known_end, seen_b=b in known_end, y=r.win, mins=r.mins))
        if r.win == r.win and r.win is not None:
            upd(a, b) if r.win == 1 else upd(b, a)
    R = pd.DataFrame(rows)
    R['p_elo'] = 1 / (1 + 10 ** ((R.elo_b - R.elo_a) / 400))
    R.to_parquet(f'../data/bridge_{tour}.parquet')
    z = R[R.y.notna()]
    ll = lambda p, y: -np.mean(y * np.log(np.clip(p, 1e-4, 1)) + (1 - y) * np.log(np.clip(1 - p, 1e-4, 1)))
    z = z.assign(mon=z.start.dt.month)
    print(tour, 'share of 2026 players already rated at end of 2025 data:', round(pd.concat([R.seen_a, R.seen_b]).mean(), 3))
    print(z.groupby('cat').apply(lambda g: pd.Series(dict(n=len(g), ll=ll(g.p_elo, g.y), acc=((g.p_elo > .5) == (g.y == 1)).mean()))).round(4).to_string())
    print(z.groupby(pd.cut(z.mon, [0, 3, 6, 10])).apply(lambda g: round(ll(g.p_elo, g.y), 4)).to_string())


if __name__ == '__main__':
    for t in sys.argv[1:] or ['atp', 'wta']:
        run(t)
