"""2026 feature engine: Elo + surface Elo carried from deuce/TML history into OddsPapi 2026 results, plus schedule
features from fixture times (rest days, recent load, minutes on court, matches already played this event), age.
Writes ../data/feat26_{atp,wta}.parquet (one row per 2026 match, a = participant 1)."""
import sys, math, os
import numpy as np, pandas as pd
from collections import defaultdict
from bridge import norm, hist_rows, MEN, WOMEN, K, games_2026, short_key
from surfmap import surface
from home import host, nationality


def ages(tour):
    p = pd.read_parquet(f'../data/{tour}_players.parquet')
    p = p[p.player_id != 'player_id']
    nm = (p.first_name.astype(str) + ' ' + p.last_name.astype(str)).map(norm)
    dob = pd.to_datetime(p.dob, errors='coerce')
    out = {}
    for a, b in zip(nm, dob):
        if b == b and b.year > 1975: out[a] = b
    return out


def te25_stream(tour):
    """2025 matches from the te set as a holdout stream (a/b orientation randomised)."""
    te = pd.read_csv('../../kg/alimoh89_tennis-results-and-betting-odds-20142025/tennis_matches_2014_2025.csv', low_memory=False)
    te['start'] = pd.to_datetime(te.match_date_formatted, errors='coerce').dt.tz_localize('Asia/Tokyo', ambiguous='NaT', nonexistent='NaT').dt.tz_convert('UTC')
    te = te[(te.tour == tour.upper()) & (te.start >= '2025-01-01') & te.comment.isin(['completed', 'retired_or_abandoned'])].copy()
    rng = np.random.default_rng(3); sw = rng.random(len(te)) < 0.5
    aw = (te.total_sets_a > te.total_sets_b).values
    nm = te.tournament.astype(str)
    cat = np.where(nm.str.contains('challenger', case=False), 'Challenger', np.where(nm.str.contains('ITF'), 'ITF', 'Tour'))
    sf = te.surface.str.lower().map({'hard': 'Hard', 'clay': 'Clay', 'grass': 'Grass', 'indoors': 'Hard'}).fillna('Hard')
    O = pd.DataFrame(dict(fid=te.match_link.values, tid=(nm + '_' + te.tournament_year.astype(str)).values, tour=nm.values, cat=cat,
                          start=te.start.values, an=np.where(sw, te.player_b, te.player_a), bn=np.where(sw, te.player_a, te.player_b),
                          win=np.where(te.comment == 'completed', np.where(sw, ~aw, aw).astype(float), np.nan), mins=np.nan, surface=sf.values,
                          ga=np.where(sw, te.total_games_b, te.total_games_a), gb=np.where(sw, te.total_games_a, te.total_games_b)))
    O['start'] = pd.to_datetime(O.start, utc=True)
    return O.dropna(subset=['start']).sort_values('start')


def tex26_stream(tour):
    """2026 matches from tennisexplorer daily pages (set scores + closing odds), orientation randomised."""
    x = pd.read_csv(os.environ.get('TEX_CSV', '../../tex/results_20251201_20261026.csv'))
    if tour == 'atp' and os.environ.get('TEX_FUT'):
        # men's Futures (ITF) with scores from tennisexplorer player pages (fut_rows.py)
        x = pd.concat([x, pd.read_csv(os.environ['TEX_FUT'])]).drop_duplicates('mid').reset_index(drop=True)
    x = x[x.ev.str.contains('/atp-men/' if tour == 'atp' else '/wta-women/') & (x.nsets > 0) & (pd.to_datetime(x.day) >= os.environ.get('TEX_START', '2025-12-09'))].copy()
    sa, sb = pd.to_numeric(x.sets_a, errors='coerce'), pd.to_numeric(x.sets_b, errors='coerce')
    x = x[sa.notna() & sb.notna() & (sa != sb)]; sa, sb = sa[x.index], sb[x.index]
    # upcoming (unplayed) matches: priced from the final ratings, never used to update them
    up = os.environ.get('TEX_UPCOMING')
    if up:
        u = pd.read_csv(up)
        u = u[u.ev.str.contains('/atp-men/' if tour == 'atp' else '/wta-women/') & (u.nsets == 0)].copy()
        u.index = u.index + 10 ** 7
        x = pd.concat([x, u]); sa = pd.concat([sa, pd.Series(np.nan, u.index)]); sb = pd.concat([sb, pd.Series(np.nan, u.index)])
    rng = np.random.default_rng(5); sw = rng.random(len(x)) < 0.5
    aw = (sa > sb).values
    slug = x.ev.str.extract(r'^/([^/]+)/')[0]
    cat = np.where(x.ev.str.contains('itf'), 'ITF', np.where(x.ev.str.contains('challenger'), 'Challenger', 'Tour'))
    O = pd.DataFrame(dict(fid='tex' + x.mid.astype(str).values, tid=x.ev.values, tour=slug.str.replace('-', ' ').values, cat=cat,
                          start=pd.to_datetime(x.day).dt.tz_localize('UTC').values + pd.Timedelta(hours=12),
                          an=np.where(sw, x.pb, x.pa), bn=np.where(sw, x.pa, x.pb), ida=np.where(sw, x.pid_b, x.pid_a), idb=np.where(sw, x.pid_a, x.pid_b), win=np.where(sa.isna().values, np.nan, np.where(sw, ~aw, aw).astype(float)), mins=np.nan,
                          ga=np.where(sw, x.games_b, x.games_a), gb=np.where(sw, x.games_a, x.games_b),
                          oa=np.where(sw, x.odds_b, x.odds_a), ob=np.where(sw, x.odds_a, x.odds_b)))
    O['start'] = pd.to_datetime(O.start, utc=True)
    O['surf_tex'] = x.surf.str.capitalize().replace({'Indoors': 'Hard'}).values if 'surf' in x else None
    if tour == 'atp' and os.environ.get('TEX_ITFM', '1') == '1' and not os.environ.get('TEX_FUT'):
        # men's ITF is not on tennisexplorer: add 2026 OddsPapi ITF Men results (winner only, no scores or prices)
        M = pd.read_parquet('../data/op_results_2026.parquet')
        M = M[(M.cat == 'ITF Men') & M.win.notna() & ~M.an.str.contains('/') & ~M.tour.str.contains('Doubles')]
        O = pd.concat([O, pd.DataFrame(dict(fid=M.fid.values, tid=M.tid.astype(str).values, tour=M.tour.values, cat='ITF', start=M.start.values,
                                            an=M.an.values, bn=M.bn.values, ida=('op:' + M.an).values, idb=('op:' + M.bn).values,
                                            win=M.win.astype(float).values, mins=np.nan, ga=np.nan, gb=np.nan, oa=np.nan, ob=np.nan))])
        O['start'] = pd.to_datetime(O.start, utc=True)
    O['surface'] = [st if isinstance(st, str) and st else (surface(t, s.month) or 'Hard') for t, s, st in zip(O.tour, O.start, O.surf_tex)]
    return O.drop(columns='surf_tex').sort_values('start')


def run(tour, mode='op'):
    H = hist_rows(tour)
    H = H[H.date >= '2012-01-01']
    if mode == 'te25':
        H = H[H.date < '2025-01-01']
        O = te25_stream(tour)
    elif mode == 'tex26':
        O = tex26_stream(tour)
        H = H[H.date < os.environ.get('TEX_START', '2025-12-09')]
        # tex names are 'Last F.': map to the full names used in history (most recent player with that last name + initial)
        rec = H.sort_values('date')
        if os.environ.get('TEX_IDS') == '0':
            full = {}
            for nm in pd.concat([rec.w, rec.l]).iloc[::-1]:
                k = short_key(nm)
                if k not in full: full[k] = nm
            O['an'] = [full.get(short_key(x), short_key(x)) for x in O.an]; O['bn'] = [full.get(short_key(x), short_key(x)) for x in O.bn]
        else:
            # identity = tennisexplorer player id; carry history only when 'Last F.' points to exactly one recent
            # history name and exactly one tennisexplorer player (else start the player fresh)
            from collections import Counter
            hn = pd.concat([rec[rec.date >= '2019-01-01'].w, rec[rec.date >= '2019-01-01'].l]).unique()
            hk = Counter(short_key(n) for n in hn); one = {short_key(n): n for n in hn}
            ids = pd.concat([pd.DataFrame(dict(i=O.ida, k=O.an.map(short_key))), pd.DataFrame(dict(i=O.idb, k=O.bn.map(short_key)))]).drop_duplicates()
            tk = ids[~ids.i.astype(str).str.startswith('op:')].groupby('k').i.nunique().to_dict()
            full_hist = set(norm(n) for n in hn)
            def ident(nm, pid):
                k = short_key(nm)
                if str(pid).startswith('op:'):  # OddsPapi rows carry full names
                    if norm(nm) in full_hist: return nm
                    return one[k] if hk.get(k) == 1 and tk.get(k, 1) == 1 else str(pid)
                return one[k] if hk.get(k) == 1 and tk.get(k) == 1 else 'tex:' + str(pid)
            O['an'] = [ident(n, i) for n, i in zip(O.an, O.ida)]; O['bn'] = [ident(n, i) for n, i in zip(O.bn, O.idb)]
    else:
        O = pd.read_parquet('../data/op_results_2026.parquet')
        O = O[O.cat.isin(MEN if tour == 'atp' else WOMEN) & ~O.cat.str.contains('Cup')].copy().sort_values('start')
        O['surface'] = [surface(t, s.month) or 'Hard' for t, s in zip(O.tour, O.start)]
    DOB = ages(tour); NAT = nationality(tour)
    elo = defaultdict(lambda: 1500.); n = defaultdict(int); eS = defaultdict(lambda: 1500.); nS = defaultdict(int)
    def upd(w, l, surf, ret=False):
        for E, N, kw, kl in ((elo, n, w, l), (eS, nS, (w, surf), (l, surf))):
            pe = 1 / (1 + 10 ** ((E[kl] - E[kw]) / 400))
            a = K['k_num'] / (N[kw] + K['k_off']) ** K['k_pow']; b = K['k_num'] / (N[kl] + K['k_off']) ** K['k_pow']
            if ret: a *= .5; b *= .5
            E[kw] += a * (1 - pe); E[kl] -= b * (1 - pe); N[kw] += 1; N[kl] += 1
    g = defaultdict(float); ng = defaultdict(int)
    def gupd(a, b, ga, gb):
        tot = ga + gb
        if not (tot == tot and tot > 5): return
        e = 1 / (1 + np.exp(-(g[a] - g[b]))); sa = ga / tot
        for p, sgn in ((a, 1), (b, -1)):
            g[p] += sgn * 4 * 0.9 / (min(ng[p], 60) + 8) ** 0.5 * (sa - e)
        ng[a] += 1; ng[b] += 1
    for r in H.itertuples(index=False):
        upd(r.w, r.l, r.surface if r.surface in ('Hard', 'Clay', 'Grass', 'Carpet') else 'Hard', r.ret)
        if not r.ret: gupd(r.w, r.l, r.gw, r.gl)
    G26 = games_2026(tour) if mode == 'op' else {}
    lastdate = {}
    for r in H.itertuples(index=False):
        lastdate[r.w] = r.date; lastdate[r.l] = r.date
    hist = defaultdict(list)  # player -> [(start, tid, mins)]
    eh = defaultdict(list)  # player -> [(start, elo after match)]
    if os.environ.get('STREAM_OUT'):
        S = pd.DataFrame(dict(start=O.start.values, a=[norm(x) for x in O.an], b=[norm(x) for x in O.bn], surface=O.surface.values,
                              win=O.win.values, ga=O.ga.values, gb=O.gb.values, fid=O.fid.values))
        S.to_parquet(os.environ['STREAM_OUT'].format(tour=tour))
    rows = []
    for r in O.itertuples(index=False):
        a, b = norm(r.an), norm(r.bn); s = r.start.tz_convert(None); surf = r.surface
        f = dict(fid=r.fid, start=r.start, cat=r.cat, tour=r.tour, surface=surf, a=a, b=b, y=r.win, mins=r.mins,
                 gelo=g[a] - g[b], gelo_n=min(ng[a], ng[b]), g_a=g[a], g_b=g[b],
                 elo_a=elo[a], elo_b=elo[b], n_a=n[a], n_b=n[b], elos_a=eS[(a, surf)], elos_b=eS[(b, surf)], ns_a=nS[(a, surf)], ns_b=nS[(b, surf)])
        for t, p in (('a', a), ('b', b)):
            hl = hist[p]
            prev = hl[-1][0] if hl else lastdate.get(p)
            f['days_' + t] = (s - prev).total_seconds() / 86400 if prev is not None else np.nan
            ev = [x for x in hl if x[1] == r.tid and (s - x[0]).days <= 16]
            f['tm_' + t] = len(ev)
            f['tmin_' + t] = np.nansum([x[2] for x in ev]) if ev else 0.0
            f['lastmin_' + t] = ev[-1][2] if ev else np.nan
            f['hrs_' + t] = (s - ev[-1][0]).total_seconds() / 3600 if ev else np.nan
            rec = [x for x in hl[-40:] if (s - x[0]).days <= 21 and x[1] != r.tid]
            f['m21_' + t] = len(rec)
            f['n90_' + t] = sum(1 for x in hl[-80:] if (s - x[0]).days <= 90)
            e90 = [x for x in eh[p] if (s - x[0]).days <= 90]
            f['tr_' + t] = elo[p] - e90[0][1] if e90 else 0.0
            f['tr180_' + t] = elo[p] - next((x[1] for x in eh[p] if (s - x[0]).days <= 180), elo[p])
            d = DOB.get(p)
            f['age_' + t] = (s - d).days / 365.25 if d is not None else np.nan
        h = host(r.tour); f['host'] = h
        f['home'] = float(h is not None and NAT.get(a) == h) - float(h is not None and NAT.get(b) == h)
        if mode == 'tex26': f['oa'], f['ob'] = r.oa, r.ob
        rows.append(f)
        for p in (a, b):
            hist[p].append((s, r.tid, r.mins if r.mins == r.mins else np.nan))
        if r.win == r.win and r.win is not None:
            upd(a, b, surf) if r.win == 1 else upd(b, a, surf)
            if mode in ('te25', 'tex26'):
                gupd(a, b, r.ga, r.gb)
            else:
                ka, kb = short_key(r.an), short_key(r.bn); key = frozenset((ka, kb))
                for dd in (0, -1, 1, -2):
                    hit = G26.get((key, s.normalize() + pd.Timedelta(days=dd)))
                    if hit and ka in hit and kb in hit:
                        gupd(a, b, hit[ka], hit[kb]); break
            eh[a].append((s, elo[a])); eh[b].append((s, elo[b]))
    R = pd.DataFrame(rows)
    if mode == 'tex26' and os.environ.get('TEX_KEEP') != '1': R = R[pd.to_datetime(R.start, utc=True) >= pd.Timestamp('2025-12-09', tz='UTC')]  # earlier rows only warm up ratings
    R.to_parquet({'op': f'../data/feat26_{tour}.parquet', 'te25': f'../data/feat25_{tour}.parquet', 'tex26': os.environ.get('FEAT_OUT', '../data/featx26_{tour}.parquet').format(tour=tour)}[mode])
    print(tour, len(R), 'age known', round(pd.concat([R.age_a, R.age_b]).notna().mean(), 3))


if __name__ == '__main__':
    mode = sys.argv[1] if len(sys.argv) > 1 else 'op'
    for t in os.environ.get('TOURS', 'atp,wta').split(','):
        run(t, mode)
