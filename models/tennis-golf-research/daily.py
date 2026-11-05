"""Daily tennis pricer. 1) add tennisexplorer results from the last scraped day to yesterday, 2) rebuild ratings
(Elo, surface Elo, games-won) through yesterday, 3) price today's scheduled matches with the live model, 4) blend
with the current price (2025 opener weights) and flag edges. Writes ../../tex/daily_YYYYMMDD.csv + .json.
Usage: daily.py [YYYY-MM-DD]   (default: today UTC)"""
import os, sys, json, glob, datetime as dt, urllib.request, time
import numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression
import warnings; warnings.filterwarnings('ignore')
import tex_scores
from te_vs_mkt import fair
TEX = '../../tex'
FULL = ['elo', 'elos', 'conf', 'new_a', 'layoff', 'm21', 'n90', 'age', 'age2', 'gelo', 'gelo_x_new', 'tr', 'tr180', 'tm', 'quick']  # +form (exp_fit.py 10-29)
W = {('wta', 'ITF'): (0.749, 0.43), ('atp', 'Challenger'): (0.798, 0.198), ('wta', 'Tour'): (0.703, 0.296), ('atp', 'Tour'): (0.82, 0.096)}
lg = lambda p: np.log(p / (1 - p))

day = dt.date.fromisoformat(sys.argv[1]) if len(sys.argv) > 1 else dt.datetime.utcnow().date()
ALL = f'{TEX}/results_all.csv'
R = pd.read_csv(ALL) if os.path.exists(ALL) else pd.read_csv(f'{TEX}/results_20251201_20261026.csv')
last = pd.to_datetime(R.day).max().date()
d = last + dt.timedelta(days=1)
new = []
while d < day:
    p = f'{TEX}/day_{d:%Y%m%d}.html'
    if d == day - dt.timedelta(days=1) and os.path.exists(p): os.remove(p)  # yesterday may have been cached incomplete
    new += tex_scores.parse(tex_scores.fetch(d), d); d += dt.timedelta(days=1)
if new:
    R = pd.concat([R, pd.DataFrame(new)]).drop_duplicates('mid', keep='last')
    R.to_csv(ALL, index=False)
print('results through', pd.to_datetime(R.day).max().date(), len(R), 'matches', flush=True)
# today's schedule + current prices
U = []
for dd in (day, day + dt.timedelta(days=1)):  # today + tomorrow, so the desk sees the next day's openers
    url = f'https://www.tennisexplorer.com/matches/?type=all&year={dd.year}&month={dd.month:02d}&day={dd.day:02d}'
    req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
    U += tex_scores.parse(urllib.request.urlopen(req, timeout=60).read().decode('utf-8', 'ignore'), dd); time.sleep(3)
U = pd.DataFrame(U)
U = U[U.nsets == 0]
U.to_csv(f'{TEX}/upcoming_{day:%Y%m%d}.csv', index=False)
print('scheduled', len(U), flush=True)
# ratings warm up from 2025-01-01 (2025 results + men's Futures from player pages); model trained on 2025-26
pd.concat([pd.read_csv(f'{TEX}/results_20250101_20251130.csv'), R]).drop_duplicates('mid', keep='last').to_csv(f'{TEX}/results_2025_live.csv', index=False)
os.environ['TEX_CSV'] = f'{TEX}/results_2025_live.csv'; os.environ['TEX_START'] = '2025-01-01'; os.environ['TEX_KEEP'] = '1'
os.environ['TEX_FUT'] = f'{TEX}/futures_men.csv'
os.environ['TEX_UPCOMING'] = f'{TEX}/upcoming_{day:%Y%m%d}.csv'
os.environ['FEAT_OUT'] = '../data/featd_{tour}.parquet'; os.environ['STREAM_OUT'] = '../data/streamd_{tour}.parquet'
import bridge2, model26
out = []
for tour in ('atp', 'wta'):
    bridge2.run(tour, 'tex26')
    F = pd.read_parquet(f'../data/featd_{tour}.parquet')
    X = model26.feats(F)
    tr = F.y.notna() & (F.start >= pd.Timestamp('2025-01-01', tz='UTC'))
    up = F.y.isna() & F.start.dt.date.isin([day, day + dt.timedelta(days=1)])
    Xt = pd.concat([X.loc[tr, FULL], -X.loc[tr, FULL]]); yt = np.r_[F.y[tr], 1 - F.y[tr]]
    m = LogisticRegression(C=1.0, fit_intercept=False, max_iter=500).fit(Xt, yt)
    G = F[up].copy(); G['p_model'] = m.predict_proba(X.loc[up, FULL])[:, 1]
    C = X.loc[up, FULL] * m.coef_[0]  # per-term pull on the logit, for the 'why' line
    G['why_terms_all'] = [[(float(v), k) for k, v in row.items()] for _, row in C.iterrows()]
    G['form_a'] = X.loc[up, 'tr'] if 'tr' in X else np.nan
    G['tour_'] = tour
    out.append(G)
G = pd.concat(out)
ok = (G.oa > 1) & (G.ob > 1)
G['fair'] = np.where(ok, fair(G.oa.fillna(2).values, G.ob.fillna(2).values), np.nan)
G['level'] = G.cat
rows = []
for r in G.itertuples(index=False):
    wo, wm = W.get((r.tour_, r.cat), (0.8, 0.1))
    known = min(r.n_a, r.n_b) >= 10
    bl = 1 / (1 + np.exp(-(wo * lg(r.fair) + wm * lg(np.clip(r.p_model, .01, .99))))) if r.fair == r.fair else np.nan
    for side, nm, pr, q in (('a', r.a, r.oa, bl), ('b', r.b, r.ob, 1 - bl if bl == bl else np.nan)):
        if not (pr == pr and q == q): continue
        rows.append(dict(event=r.tour, level=f'{r.tour_.upper()} {r.cat}', player=nm, opponent=r.b if side == 'a' else r.a,
                         model=round(r.p_model if side == 'a' else 1 - r.p_model, 3), market=round(r.fair if side == 'a' else 1 - r.fair, 3),
                         blend=round(q, 3), price=pr, edge=round(q * pr - 1, 3), both_rated=known))
D = pd.DataFrame(rows).sort_values('edge', ascending=False)
D.to_csv(f'{TEX}/daily_{day:%Y%m%d}.csv', index=False)
pick = D[(D.edge > 0.06) & D.price.between(1.2, 5) & D.both_rated & D.level.isin(['WTA ITF', 'ATP Challenger'])]
pick.to_json(f'{TEX}/daily_{day:%Y%m%d}.json', orient='records')
print('priced', len(G), 'matches;', len(pick), 'flags (6%+ edge, ITF women / Challenger, both players rated)')
print(pick.head(30).to_string(index=False))

# shared file for the Bet Desk Tennis tab (paper-tracked): model chance per match, matched to the OddsPapi list by
# date + both surnames/initials; edge = blend vs the best current price at J's books in that list
OP = '/mnt/project-files/sports-analytics/app/data/odds/tennis.json'
WB = {'ITF': (0.8, 0.2), 'Challenger': (0.8, 0.2), 'Tour': (0.9, 0.1)}  # 2026 in-sample refit, model weight small
import re, unicodedata
def toks(nm):
    nm = unicodedata.normalize('NFKD', str(nm)).encode('ascii', 'ignore').decode().lower()
    return [t for t in re.split(r'[^a-z]+', nm) if t]
def same(tex, opn):
    """tex 'Last F.' (or 'Last-Name F.') vs OddsPapi 'Last, First' or 'First Last' in any order"""
    t = toks(tex); o = toks(opn)
    if not t or not o: return False
    ini = t[-1] if len(t[-1]) == 1 else ''
    sur = t[:-1] if ini else t
    if not all(x in o for x in sur): return False
    rest = [x for x in o if x not in sur]
    return not ini or not rest or any(x[0] == ini for x in rest)
ops = json.load(open(OP)) if os.path.exists(OP) else []
bydate = {}
for m in ops: bydate.setdefault(m['start'][:10], []).append(m)
U2 = pd.read_csv(f'{TEX}/upcoming_{day:%Y%m%d}.csv').drop_duplicates('mid')
# recent form by rating identity (2025-26 stream incl. men's Futures), newest first
S = pd.concat([pd.read_parquet(f'../data/streamd_{t}.parquet') for t in ('atp', 'wta')])
S = S[S.win.notna()].sort_values('start')
FORM = {}
for t in S.itertuples(index=False):
    for p, w in ((t.a, t.win == 1), (t.b, t.win == 0)):
        FORM.setdefault(p, []).append((w, t.surface))
# scoring style by tennisexplorer id from set scores (games per set, tiebreak-set rate), 2025-26
SC = pd.concat([pd.read_csv(os.environ['TEX_CSV'], usecols=['ev', 'pid_a', 'pid_b', 'score']), pd.read_csv(f'{TEX}/futures_men.csv', usecols=['ev', 'pid_a', 'pid_b', 'score'])])
TOUR = {}
STY = {}
for t in SC.itertuples(index=False):
    sets = [(int(x), int(y)) for x, y in re.findall(r'(\d+)-(\d+)', str(t.score)) if 6 <= max(int(x), int(y)) <= 7]
    for p in (t.pid_a, t.pid_b):
        TOUR[p] = 'm' if 'atp-men' in t.ev else 'w'
        st = STY.setdefault(p, [0, 0, 0])
        for x, y in sets: st[0] += 1; st[1] += x + y; st[2] += (x + y == 13)
# style per tour: serve-dominance score = z(tiebreak-set rate) + z(games per set); top quarter = big server,
# bottom quarter = grinder / returner (breaks of serve are common in their matches), rest all-court
_q = pd.DataFrame([(TOUR[p], v[1] / v[0], v[2] / v[0]) for p, v in STY.items() if v[0] >= 80], columns=['t', 'gps', 'tb'])  # established players only
MS = {t: (g.gps.mean(), g.gps.std(), g.tb.mean(), g.tb.std()) for t, g in _q.groupby('t')}
_q['z'] = [(x.gps - MS[x.t][0]) / MS[x.t][1] + (x.tb - MS[x.t][2]) / MS[x.t][3] for x in _q.itertuples()]
CUT = {t: (g.z.quantile(.75), g.z.quantile(.25)) for t, g in _q.groupby('t')}
def style(pid):
    st = STY.get(pid)
    if not st or st[0] < 15: return None, None, None
    gps, tb = st[1] / st[0], st[2] / st[0]
    t = TOUR[pid]; m = MS[t]; z = (gps - m[0]) / m[1] + (tb - m[2]) / m[3]
    lab = 'big server' if z >= CUT[t][0] else 'grinder / returner' if z <= CUT[t][1] else 'all-court'
    return lab, round(gps, 2), round(tb, 3)
def info(r, side, U_r, a_is_pa):
    ident = getattr(r, side)
    pid = (U_r.pid_a if (side == 'a') == a_is_pa else U_r.pid_b)
    fl = FORM.get(ident, [])
    last = fl[-10:][::-1]; ls = [x for x in fl if x[1] == r.surface][-10:]
    lab, gps, tb = style(pid)
    o = 'b' if side == 'a' else 'a'
    return dict(elo=round(float(getattr(r, 'elo_' + side))), elo_surf=round(float(getattr(r, 'elos_' + side))), gw=round(float(getattr(r, 'g_' + side)), 3),
                srv=None, ret=None, srv_pct=None, ret_pct=None, style=lab, games_per_set=gps, tiebreak_rate=tb,
                n=int(getattr(r, 'n_' + side)), form=[sum(w for w, _ in last), sum(not w for w, _ in last)],
                form_str=''.join('W' if w else 'L' for w, _ in last),
                form_rating=round(float(getattr(r, 'tr_' + side)), 1) if getattr(r, 'tr_' + side) == getattr(r, 'tr_' + side) else None,
                surf_form=[sum(w for w, _ in ls), sum(not w for w, _ in ls)], rank=None, tex_id=pid)
PHRASE = {'elo': 'Elo', 'elos': 'surface Elo', 'conf': 'Elo', 'new_a': 'one player being new to the ratings',
          'layoff': 'a recent layoff', 'm21': 'recent match load', 'n90': 'matches in the last 90 days', 'age': 'age', 'age2': 'age curve',
          'gelo': 'games-won rating', 'gelo_x_new': 'games-won rating', 'tr': 'recent form', 'tr180': '6-month form', 'tm': 'matches already won this event',
          'quick': 'a quick turnaround'}
def why(r, p1, p2, sw):
    agg = {}
    for v, k in r.why_terms_all:
        agg[PHRASE.get(k, k)] = agg.get(PHRASE.get(k, k), 0) + v
    parts = []
    for ph, v in sorted(agg.items(), key=lambda t: -abs(t[1]))[:3]:
        if abs(v) < .08: continue
        fav = 'a' if v > 0 else 'b'
        who = (p2 if fav == 'a' else p1) if sw else (p1 if fav == 'a' else p2)
        parts.append(f"{ph} favours {str(who).split(',')[0].rstrip('.')}")
    return ('Main drivers: ' + '; '.join(parts) + '.') if parts else 'Ratings are close; no single input dominates.'

nm = {('tex' + str(r.mid)): (r.pa, r.pb) for r in U2.itertuples()}
outm = []
for r in G.itertuples(index=False):
    pa_pb = nm.get(r.fid)
    if not pa_pb: continue
    # G rows use the bridge's randomised a/b; recover which tex name is 'a' via the average price
    U_r = U2[U2.mid == int(r.fid[3:])].iloc[0]
    a_is_pa = not (r.oa == r.oa and np.isclose(r.oa, U_r.odds_b) and not np.isclose(r.oa, U_r.odds_a))
    na, nb = (pa_pb if a_is_pa else pa_pb[::-1])
    om = None
    d0 = r.start.date()
    for dd in (d0, d0 + dt.timedelta(days=1), d0 - dt.timedelta(days=1)):
        for m in bydate.get(f'{dd}', []):
            if (same(na, m['p1']) and same(nb, m['p2'])) or (same(na, m['p2']) and same(nb, m['p1'])): om = m; break
        if om: break
    p1, p2, p = na, nb, float(r.p_model)
    if om:
        p1, p2 = om['p1'], om['p2']
        if not same(na, om['p1']): p = 1 - p
    rated = bool(min(r.n_a, r.n_b) >= 10)
    rec = dict(p1=p1, p2=p2, start=om['start'] if om else None, cat=om['cat'] if om else f'{r.tour_.upper()} {r.cat}',
               tournament=om['tournament'] if om else r.tour, fid=om['fid'] if om else None, tex_mid=r.fid[3:],
               p_model=round(p, 3), p_blend=None, edge_book=None, edge=None, flag=False, rated=rated)
    if om:
        pin = om['books'].get('pinnacle', {}).get('now') or om['books'].get('pinnacle', {}).get('open')
        if pin:
            fp = fair(np.array([pin[1]]), np.array([pin[2]]))[0]
            wo, wm = WB.get(r.cat, (0.85, 0.15))
            pb = 1 / (1 + np.exp(-(wo * lg(fp) + wm * lg(np.clip(p, .01, .99)))))
            rec['p_blend'] = round(float(pb), 3)
            best = None
            for bk, v in om['books'].items():
                q = v.get('now') or v.get('open')
                if not q or bk == 'pinnacle': continue
                for side, pr, pp in ((0, q[1], pb), (1, q[2], 1 - pb)):
                    e = pp * pr - 1
                    if pr <= 5 and (best is None or e > best[0]): best = (e, bk, side)
            if best:
                rec['edge'], rec['edge_book'], rec['edge_side'] = round(best[0], 3), best[1], 'p1' if best[2] == 0 else 'p2'
                rec['flag'] = bool(best[0] >= .06 and rated and r.cat in ('ITF', 'Challenger'))
    # player tiles: model inputs per player (a/b = model sides; swap when the OddsPapi P1 is side b)
    sw = bool(om) and not same(na, om['p1'])
    ia, ib = info(r, 'a', U_r, a_is_pa), info(r, 'b', U_r, a_is_pa)
    rec['surface'] = r.surface
    rec['info'] = dict(p1=ib if sw else ia, p2=ia if sw else ib)
    rec['why'] = why(r, p1, p2, sw)
    outm.append(rec)
J = dict(made=dt.datetime.utcnow().strftime('%Y-%m-%dT%H:%M:%SZ'),
         note='Tennis model (Elo + surface Elo + games-won rating + form; 2025-26 results incl. men\'s Futures). info: gw = games-won rating (log-odds of winning a game vs average), form_rating = Elo gained over recent matches, style from games/set and tiebreak rate vs the tour; no live serve/return stats (srv null). p_model = P1 win chance from our model alone. '
              'p_blend = 2026 weights with the Pinnacle no-vig price. Paper-tracked only: no CLV edge proven in 2026.',
         matches=outm)
json.dump(J, open('../data/tennis_model.json', 'w'))
if os.environ.get('PUBLISH_SHARED', '0') == '1':
    json.dump(J, open('/mnt/project-files/sports-analytics/app/data/tennis_model.json', 'w'))
print('tennis_model.json', len(outm), 'matches,', sum(1 for m in outm if m['fid']), 'matched to the OddsPapi list,', sum(m['flag'] for m in outm), 'flags')
