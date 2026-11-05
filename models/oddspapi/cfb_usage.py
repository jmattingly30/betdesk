"""CFB usage priors + missing-star flags from college player props (J 2026-10-29: implement usage priors; track the
missing-star flag and show it only on bets where an elite player is out). Props are game-model inputs, never bets.
  python3 cfb_usage.py        run after app/op_live.py (reads app/data/odds/model_props_CFB.json) -> app/data/cfb_usage.json
USAGE PRIOR: each team's prop-implied offense = its QB's pass-yards line + every rusher's rush-yards line. The home-minus-away
gap (per 100 yds) shifts our CFB margin by a coefficient refit every run on completed 2026 games against CFB v3's walk-forward
margin (models/cfb-v3/game_predictions.parquet), through the origin so it only adds what our model misses. Tested 2026-10-29:
walk-forward margin MAE 12.65 -> 12.58 (112 games); props predict team pass/rush yards 2-3x better than season averages.
MISSING STAR: the team's usual top rusher/receiver prop (top in 2+ of its last 4 games) has no prop this game while the team
has 3+ prop players posted. Elite = his average line 75+ rush yds / 60+ rec yds. Graded each run (margin vs our model and
the line) in 'track'; ~45 games so far ran ~4 pts under the close (not significant)."""
import os, json, collections, datetime as dt, statistics as stt
import numpy as np, pandas as pd
import college_props as cp

US, SA = cp.US, cp.SA; APP = f'{SA}/app/data'; STORE = f'{US}/cfbp/live_store.json'
NOW = dt.datetime.now(dt.timezone.utc)
LIVE_STAT = {'Pass Yards': 'pass', 'Rush Yards': 'rush', 'Receiving Yards': 'rec'}
ELITE = {'rush': 75, 'rec': 60}
CAP = 6.0   # largest margin shift in points


def prob(a): return 100 / (a + 100) if a > 0 else -a / (-a + 100)


def live_rows():
    """model_props_CFB.json -> {fid: dict(k, h, a, rows=[[pid, stat, line]])}: per book the line priced closest to even, median of books."""
    f = f'{APP}/odds/model_props_CFB.json'
    if not os.path.exists(f): return {}
    fx = {x['fixtureId']: x for x in json.load(open(f'{APP}/odds/op/fixtures_14.json'))['fx']}
    out = {}
    for g in json.load(open(f))['games']:
        x = fx.get(g['id'])
        if not x: continue
        rows = []
        for pid, st in g['p'].items():
            for stat, lines in st.items():
                if stat not in LIVE_STAT: continue
                per = collections.defaultdict(list)
                for ln, bks in lines.items():
                    for b, sd in bks.items():
                        if 'Over' in sd and 'Under' in sd: per[b].append((abs(prob(sd['Over']) - prob(sd['Under'])), float(ln)))
                if per: rows.append([pid, LIVE_STAT[stat], stt.median(min(v)[1] for v in per.values())])
        out[g['id']] = dict(k=g['k'], h=x['participant1ShortName'], a=x['participant2ShortName'], rows=rows)
    return out


def spreads():
    """fid -> median home spread now (data/odds/CFB.json, OddsPapi sweep)."""
    f = f'{APP}/odds/CFB.json'; out = {}
    if not os.path.exists(f): return out
    for e in json.load(open(f)).get('events', []):
        v = [o['point'] for b in e['bookmakers'] for m in b['markets'] if m['key'] == 'spreads' for o in m['outcomes'] if o['name'] == e['home_team']]
        if v: out[e['id']] = stt.median(v)
    return out


def main():
    S = json.load(open(STORE)) if os.path.exists(STORE) else {}
    sp_now = spreads()
    for fid, g in live_rows().items():   # keep the last snapshot before kickoff
        if dt.datetime.fromisoformat(g['k'].replace('Z', '+00:00')) > NOW:
            g['sp'] = sp_now.get(fid, S.get(fid, {}).get('sp')); S[fid] = g
    json.dump(S, open(STORE, 'w'), separators=(',', ':'))
    # history: the OddsPapi price-history pull (op/us/hist_cfbp.jsonl), closing main line, median of books
    fx = {f['fixtureId']: f for f in json.load(open(f'{US}/fx_cfbp.json'))}
    H = cp.props(); H = H[H.stat.isin(['pass', 'rush', 'rec'])]
    H = H.assign(l=H.l_close.fillna(H.l_open)).groupby(['fid', 'stat', 'pid']).l.median().reset_index()
    L = pd.read_csv(f'{US}/lines_cfbp.csv').groupby('fid').spread_close.median()
    G = {}
    for fid, g in H.groupby('fid'):
        f = fx[fid]; G[fid] = dict(k=f['startTime'].replace('.000Z', 'Z'), h=f['participant1ShortName'], a=f['participant2ShortName'],
                                  rows=g[['pid', 'stat', 'l']].values.tolist(), sp=L.get(fid))
    for fid, g in S.items(): G.setdefault(fid, g)
    order = sorted(G, key=lambda f: G[f]['k'])
    # player -> team: college_props' map (names + box scores), else the one team common to all his games
    pteam = json.load(open(f'{US}/cfbp/pid_team.json'))
    seen = collections.defaultdict(list)
    for fid in order:
        for pid, _, _ in G[fid]['rows']: seen[pid].append({G[fid]['h'], G[fid]['a']})
    for pid, sets in seen.items():
        if pid not in pteam:
            c = set.intersection(*sets)
            if len(c) == 1: pteam[pid] = c.pop()
    # outcomes (CFBD, at most one call a day) and our model's walk-forward margins
    gf = f'{US}/cfbp/games.json'
    if NOW - dt.datetime.fromtimestamp(os.path.getmtime(gf), dt.timezone.utc) > dt.timedelta(hours=20) and os.environ.get('CFBD_API_KEY'):
        import requests
        r = requests.get('https://api.collegefootballdata.com/games', params=dict(year=2026, seasonType='regular'),
                         headers={'Authorization': f"Bearer {os.environ['CFBD_API_KEY']}"}, timeout=120)
        if r.ok: json.dump(r.json(), open(gf, 'w'))
    cg = collections.defaultdict(list)
    for g in json.load(open(gf)):
        if g.get('completed') and g.get('homePoints') is not None: cg[frozenset((g['homeTeam'], g['awayTeam']))].append(g)
    V = pd.read_parquet(f'{SA}/models/cfb-v3/game_predictions.parquet'); V = V[V.season == 2026].set_index('game_id').model_margin.to_dict()

    def result(fid):
        g0 = G[fid]
        for g in cg.get(frozenset((g0['h'], g0['a'])), []):
            if abs((dt.datetime.fromisoformat(g['startDate'][:19]) - dt.datetime.fromisoformat(g0['k'][:19])).total_seconds()) < 1.5 * 86400:
                m = g['homePoints'] - g['awayPoints']; return (m if g['homeTeam'] == g0['h'] else -m), V.get(g['id'])
        return None, None
    # features, stars, flags (chronological, so stars come only from earlier games)
    tops = collections.defaultdict(list); avg = collections.defaultdict(list); rec = []
    for fid in order:
        g = G[fid]; by = collections.defaultdict(lambda: collections.defaultdict(dict))
        for pid, stat, l in g['rows']:
            t = pteam.get(pid)
            if t in (g['h'], g['a']): by[t][stat][pid] = l
        feat, stars_out = {}, []
        for t in (g['h'], g['a']):
            d = by[t]; n = len({p for s in d.values() for p in s})
            feat[t] = dict(pass_l=max(d['pass'].values()) if d['pass'] else None, rush_l=round(sum(d['rush'].values()), 1) if d['rush'] else None,
                           rec_l=round(sum(d['rec'].values()), 1) if d['rec'] else None, n=n)
            for kind in ('rush', 'rec'):
                last = tops[(t, kind)][-4:]; c = collections.Counter(last).most_common(1)
                if c and c[0][1] >= 2 and n >= 3 and c[0][0] not in d[kind] and c[0][0] not in {p for s in d.values() for p in s}:
                    a = stt.mean(avg[c[0][0]]); stars_out.append(dict(team=t, kind=kind, pid=c[0][0], avg_line=round(a, 1), elite=a >= ELITE[kind]))
                if d[kind]: tops[(t, kind)].append(max(d[kind], key=d[kind].get))
            for s in d.values():
                for p, l in s.items(): avg[p].append(l)
        fh, fa = feat[g['h']], feat[g['a']]
        x = None if None in (fh['pass_l'], fh['rush_l'], fa['pass_l'], fa['rush_l']) else (fh['pass_l'] + fh['rush_l'] - fa['pass_l'] - fa['rush_l']) / 100
        m, mm = result(fid)
        rec.append(dict(fid=fid, k=g['k'], h=g['h'], a=g['a'], home=fh, away=fa, x=x, sp=g.get('sp'), margin=m, model=mm, stars_out=stars_out))
    R = pd.DataFrame(rec)
    done = R[R.margin.notna() & R.model.notna() & R.x.notna()]
    r = done.margin - done.model; coef = float((done.x * r).sum() / (done.x ** 2).sum()) if len(done) >= 30 else 0.0
    # walk-forward check of the coefficient (refit before each week's games)
    wf0, wf1 = [], []
    for wk, g in done.groupby(pd.to_datetime(done.k).dt.isocalendar().week):
        tr = done[pd.to_datetime(done.k) < pd.to_datetime(g.k).min()]
        if len(tr) < 30: continue
        c = float((tr.x * (tr.margin - tr.model)).sum() / (tr.x ** 2).sum())
        wf0 += list((g.margin - g.model).abs()); wf1 += list((g.margin - g.model - np.clip(c * g.x, -CAP, CAP)).abs())
    # missing-star tracking: margin vs our model and vs the line, flagged games only
    fl = [dict(fid=x.fid, team=s['team'], kind=s['kind'], elite=s['elite'], home=s['team'] == x.h, margin=x.margin, model=x.model, sp=x.sp)
          for x in R.itertuples() for s in x.stars_out if x.margin == x.margin and x.margin is not None]
    F = pd.DataFrame(fl)

    def grade(F):
        if not len(F): return dict(n=0)
        sgn = np.where(F.home, 1, -1); out = dict(n=len(F))
        a = F.dropna(subset=['sp']); out['vs_line'] = round(float((sgn[F.sp.notna()] * (a.margin + a.sp)).mean()), 2) if len(a) else None
        b = F.dropna(subset=['model']); out['vs_model'] = round(float((sgn[F.model.notna()] * (b.margin - b.model)).mean()), 2) if len(b) else None
        return out
    up = R[pd.to_datetime(R.k, utc=True) > NOW]; names = json.load(open(f'{APP}/odds/op/players.json'))['ids']
    games = []
    for x in up.itertuples():
        shift = round(float(np.clip(coef * x.x, -CAP, CAP)), 2) if x.x is not None and x.x == x.x else 0.0
        games.append(dict(id=x.fid, k=x.k, h=x.h, a=x.a, home=x.home, away=x.away, gap_yds=None if x.x is None or x.x != x.x else round(x.x * 100),
                          shift=shift, stars_out=[dict(s, name=names.get(s['pid'])) for s in x.stars_out]
                          if pd.to_datetime(x.k, utc=True) - NOW < pd.Timedelta(hours=36) else []))   # props aren't all up until ~a day out
    out = dict(made=NOW.strftime('%Y-%m-%d %H:%M UTC'),
               note='shift = points added to our CFB margin (home view) from prop-implied offense; stars_out elite=true -> show on bets',
               coef_pts_per_100yds=round(coef, 2), fit_games=len(done),
               check=dict(walk_forward_games=len(wf0), mae_model=round(float(np.mean(wf0)), 3) if wf0 else None, mae_with=round(float(np.mean(wf1)), 3) if wf1 else None),
               track=dict(all=grade(F), elite=grade(F[F.elite]) if len(F) else dict(n=0)), games=games)
    json.dump(out, open(f'{APP}/cfb_usage.json', 'w'), separators=(',', ':'))
    print('cfb_usage:', len(games), 'upcoming games,', sum(g['shift'] != 0 for g in games), 'with a shift,',
          sum(len(g['stars_out']) for g in games), 'stars out (', sum(s['elite'] for g in games for s in g['stars_out']), 'elite ) | coef', out['coef_pts_per_100yds'],
          '| check', out['check'], '| track', out['track'])


if __name__ == '__main__':
    main()
