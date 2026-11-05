"""College player props as GAME-model inputs (J 2026-10-28: treat them as market player-stat projections, not bets).
CFB 2026 FBS-vs-FBS games (op/us/hist_cfbp.jsonl: DK/FD/bet365 history; lines_cfbp.csv from lines.py) joined to CFBD
2026 box scores (op/us/cfbp/games.json, players.json). Tests, each judged against the market game line:
  A  prop-implied team offense (QB pass yds + rushers' rush yds) vs the spread and total
  B  starting-QB change (the team's pass-yards prop player differs from its usual one)
  C  missing star (the team's top prop player from earlier games has no prop this game)
  D  props as usage/production priors (prop-implied team pass/rush yards vs season-to-date averages)
  python3 college_props.py   -> op/us/cfbp/report.json + printout"""
import os, json, re, collections, statistics as stt, datetime as dt
import numpy as np, pandas as pd
from lines import main_line

HERE = os.path.dirname(os.path.abspath(__file__)); OP = os.path.normpath(f'{HERE}/../../op'); US = f'{OP}/us'; SA = os.path.dirname(OP)
STATS = {'Pass Yards': 'pass', 'Rush Yards': 'rush', 'Receiving Yards': 'rec', 'Receptions': 'recs', 'Pass Attempts': 'att',
         'Rush Attempts': 'ratt', 'TD Passes': 'ptd'}


def props():
    M = {}
    for m in json.load(open(f'{OP}/markets.json')):
        n = m['marketName']
        if m['sportId'] == 14 and m['playerProp'] and n.startswith('Over Under Player '):
            base = n[len('Over Under Player '):].replace(' (incl. overtime)', '')
            if base in STATS: M[str(m['marketId'])] = (STATS[base], m['handicap'], {str(o['outcomeId']): o['outcomeName'] for o in m['outcomes']})
    rows = []
    for line in open(f'{US}/hist_cfbp.jsonl'):
        try: r = json.loads(line)
        except ValueError: continue
        if 'err' in r: continue
        for bk, bm in r['books'].items():
            c = collections.defaultdict(list)
            for mid, outs in bm.items():
                if mid not in M: continue
                st, h, on = M[mid]
                side = {on[k]: v for k, v in outs.items() if k in on}
                if 'Over' in side and 'Under' in side:
                    for pid in set(side['Over']) & set(side['Under']):
                        c[(st, pid)].append((h, side['Over'][pid], side['Under'][pid]))
            for (st, pid), cands in c.items():
                o, cl = main_line(cands, 'open'), main_line(cands, 'close')
                rows.append((r['fid'], bk, st, pid, o[0], o[3], cl[0] if cl else None))
    return pd.DataFrame(rows, columns=['fid', 'book', 'stat', 'pid', 'l_open', 'open_secs', 'l_close'])


def cfbd():
    G = {g['id']: g for g in json.load(open(f'{US}/cfbp/games.json')) if g.get('completed')}
    box = {}
    for g in json.load(open(f'{US}/cfbp/players.json')):
        for t in g['teams']:
            d = {'pts': t.get('points'), 'pl': collections.defaultdict(dict)}
            for c in t['categories']:
                for ty in c['types']:
                    if (c['name'], ty['name']) == ('passing', 'YDS'): d['pass'] = sum(float(a['stat']) for a in ty['athletes'])
                    if (c['name'], ty['name']) == ('rushing', 'YDS'): d['rush'] = sum(float(a['stat']) for a in ty['athletes'])
                    for a in ty['athletes']:
                        d['pl'][a['name']][(c['name'], ty['name'])] = a['stat']
            box[(g['id'], t['team'])] = d
    return G, box


def norm(s): return re.sub(r'[^a-z]', '', re.sub(r'\b(jr|sr|ii|iii|iv)\b\.?', '', s.lower()))


def main():
    fx = {f['fixtureId']: f for f in json.load(open(f'{US}/fx_cfbp.json'))}
    L = pd.read_csv(f'{US}/lines_cfbp.csv')
    L = L.groupby('fid').agg(sp_open=('spread_open', 'median'), sp_close=('spread_close', 'median'), to_open=('total_open', 'median'),
                             to_close=('total_close', 'median'), sp_open_secs=('spread_open_secs', 'max')).reset_index()
    P = props(); G, box = cfbd()
    names = json.load(open(f'{SA}/app/data/odds/op/players.json'))['ids']
    # fixture -> CFBD game (same short names, start within a day)
    by = collections.defaultdict(list)
    for g in G.values(): by[frozenset((g['homeTeam'], g['awayTeam']))].append(g)
    gm = {}
    for fid, f in fx.items():
        for g in by.get(frozenset((f['participant1ShortName'], f['participant2ShortName'])), []):
            if abs((dt.datetime.fromisoformat(g['startDate'][:19]) - dt.datetime.fromisoformat(f['startTime'][:19])).total_seconds()) < 86400 * 1.5: gm[fid] = g
    print('fixtures', len(fx), 'matched to CFBD', len(gm))
    # player -> team: exact name in the box score of that game, else the team common to all its games
    pteam = {}
    pg = P.groupby('pid').fid.unique()
    for pid, fids in pg.items():
        nm = names.get(pid)
        if nm:
            for fid in fids:
                g = gm.get(fid)
                if not g: continue
                for t in (g['homeTeam'], g['awayTeam']):
                    if any(norm(n) == norm(nm) for n in box.get((g['id'], t), {}).get('pl', {})): pteam[pid] = t
            if pid in pteam: continue
        sets = [{fx[f]['participant1ShortName'], fx[f]['participant2ShortName']} for f in fids]
        common = set.intersection(*sets)
        if len(common) == 1: pteam[pid] = common.pop()
    P['team'] = P.pid.map(pteam); json.dump(pteam, open(f'{US}/cfbp/pid_team.json', 'w'))   # cfb_usage.py starts from this map
    print('prop players', P.pid.nunique(), 'with team', len(pteam), '| rows with team', round(P.team.notna().mean(), 3))
    C = P.dropna(subset=['team']).groupby(['fid', 'team', 'stat', 'pid']).agg(lo=('l_open', 'median'), lc=('l_close', 'median'), secs=('open_secs', 'max')).reset_index()
    C['lc'] = C.lc.fillna(C.lo)
    # team-game features
    T = []
    for (fid, team), g in C.groupby(['fid', 'team']):
        if fid not in gm: continue
        cg = gm[fid]; home = team == fx[fid]['participant1ShortName']
        qb = g[g.stat == 'pass'].sort_values('lc', ascending=False)
        ru = g[g.stat == 'rush']; rc = g[g.stat == 'rec']
        b = box.get((cg['id'], team), {}); opp = cg['awayTeam'] if team == cg['homeTeam'] else cg['homeTeam']
        T.append(dict(fid=fid, gid=cg['id'], week=cg['week'], team=team, opp=opp, home=home, start=fx[fid]['startTime'],
                      qb=qb.pid.iloc[0] if len(qb) else None, pass_l=qb.lc.iloc[0] if len(qb) else np.nan,
                      rush_l=ru.lc.sum() if len(ru) else np.nan, n_rush=len(ru), rec_l=rc.lc.sum() if len(rc) else np.nan, n_rec=len(rc),
                      top_rush=ru.sort_values('lc').pid.iloc[-1] if len(ru) else None, top_rec=rc.sort_values('lc').pid.iloc[-1] if len(rc) else None,
                      prop_secs=g.secs.max(), pts=b.get('pts'), pass_a=b.get('pass'), rush_a=b.get('rush'), opp_pts=box.get((cg['id'], opp), {}).get('pts'),
                      players=set(g.pid)))
    T = pd.DataFrame(T).sort_values('start').reset_index(drop=True)
    T = T.merge(L, on='fid', how='left')
    T['mkt_margin'] = np.where(T.home, -T.sp_close, T.sp_close); T['mkt_margin_o'] = np.where(T.home, -T.sp_open, T.sp_open)
    T['margin'] = T.pts - T.opp_pts
    print('team-games', len(T), 'with QB pass prop', T.pass_l.notna().sum(), 'with rush props', T.rush_l.notna().sum())
    R = {}
    # ---------- season-to-date production (for D and as a baseline) ----------
    for col in ('pass_a', 'rush_a', 'pts'):
        T[f'{col}_std'] = T.groupby('team')[col].transform(lambda s: s.shift().expanding().mean())
    # ---------- A: prop-implied offense vs the game line ----------
    T['impl'] = T.pass_l + T.rush_l
    W = T.merge(T[['fid', 'team', 'impl', 'pass_l', 'rush_l']].rename(columns={'team': 'opp', 'impl': 'o_impl', 'pass_l': 'o_pass', 'rush_l': 'o_rush'}), on=['fid', 'opp'])
    W = W[W.home].dropna(subset=['impl', 'o_impl', 'mkt_margin', 'margin', 'to_close', 'pts', 'opp_pts']).copy()
    W['x'] = W.impl - W.o_impl; W['tot_x'] = W.impl + W.o_impl; W['total'] = W.pts + W.opp_pts
    R['A_games'] = len(W)

    def cv(y, base, extra, df):
        """leave-one-week-out: RMSE of y ~ base vs y ~ base + extra; also ATS-style hit rate where the extra moves us 3+ pts off the line."""
        e0, e1, picks = [], [], []
        for w in sorted(df.week.unique()):
            tr, te = df[df.week != w], df[df.week == w]
            X0 = np.c_[np.ones(len(tr)), tr[base]]; X1 = np.c_[X0, tr[extra]]
            b0 = np.linalg.lstsq(X0, tr[y], rcond=None)[0]; b1 = np.linalg.lstsq(X1, tr[y], rcond=None)[0]
            p0 = np.c_[np.ones(len(te)), te[base]] @ b0; p1 = np.c_[np.ones(len(te)), te[base], te[extra]] @ b1
            e0 += list(te[y] - p0); e1 += list(te[y] - p1)
            for pv, ln, act in zip(p1, te[base], te[y]):
                if abs(pv - ln) >= 3 and act != ln: picks.append((pv > ln) == (act > ln))
        b = np.linalg.lstsq(np.c_[np.ones(len(df)), df[base], df[extra]], df[y], rcond=None)[0]
        return dict(n=len(df), rmse_line=round(float(np.sqrt(np.mean(np.square(e0)))), 3), rmse_with=round(float(np.sqrt(np.mean(np.square(e1)))), 3),
                    coef_per_100yds=round(float(b[2] * 100), 2), picks_3plus=len(picks), hit=round(float(np.mean(picks)), 3) if picks else None)
    R['A_margin_vs_close'] = cv('margin', 'mkt_margin', 'x', W)
    R['A_margin_vs_open'] = cv('margin', 'mkt_margin_o', 'x', W.dropna(subset=['mkt_margin_o']))
    R['A_total_vs_close'] = cv('total', 'to_close', 'tot_x', W)
    R['A_total_vs_open'] = cv('total', 'to_open', 'tot_x', W.dropna(subset=['to_open']))
    # does the closing line move toward the props? (game opener -> close, against prop-implied gap measured vs the opener)
    W['mv'] = W.mkt_margin - W.mkt_margin_o
    R['A_open_to_close_move_corr'] = round(float(W[['mv', 'x']].dropna().corr().iloc[0, 1]), 3)
    R['A_prop_posted_hours_before'] = round(float(W.prop_secs.median() / 3600), 1); R['A_line_posted_hours_before'] = round(float(W.sp_open_secs.median() / 3600), 1)
    # ---------- B: QB change ----------
    T['usual_qb'] = None; T['qb_n'] = 0
    seen = collections.defaultdict(list)
    for i, r in T.iterrows():
        h = seen[r.team]
        if len(h) >= 2:
            c = collections.Counter(h).most_common(1)[0]; T.at[i, 'usual_qb'] = c[0]; T.at[i, 'qb_n'] = c[1]
        if r.qb: h.append(r.qb)
    B = T[T.usual_qb.notna() & T.qb.notna()].copy(); B['changed'] = B.qb != B.usual_qb
    B['ats_close'] = np.sign(B.margin - B.mkt_margin); B['ats_open'] = np.sign(B.margin - B.mkt_margin_o); B['mv'] = B.mkt_margin - B.mkt_margin_o
    R['B'] = {str(k): dict(n=len(g), cover_vs_open=round(float((g.ats_open > 0).sum() / max(1, (g.ats_open != 0).sum())), 3),
                          cover_vs_close=round(float((g.ats_close > 0).sum() / max(1, (g.ats_close != 0).sum())), 3),
                          line_move_pts=round(float(g.mv.mean()), 2), margin_minus_close=round(float((g.margin - g.mkt_margin).mean()), 2),
                          pass_yds_line=round(float(g.pass_l.mean()), 1)) for k, g in B.groupby('changed')}
    # ---------- C: missing star ----------
    star = {}; Cr = []
    for i, r in T.iterrows():
        for kind in ('rush', 'rec'):
            s = star.get((r.team, kind))
            if s and s[1] >= 2:
                Cr.append(dict(kind=kind, missing=s[0] not in r.players, margin=r.margin, mm=r.mkt_margin, mmo=r.mkt_margin_o, pts=r.pts, team=r.team))
        sub = C[(C.fid == r.fid) & (C.team == r.team)]
        for kind in ('rush', 'rec'):
            q = sub[sub.stat == kind].sort_values('lc')
            if len(q):
                top = q.pid.iloc[-1]; s = star.get((r.team, kind))
                star[(r.team, kind)] = (top, s[1] + 1 if s and s[0] == top else 1) if not (s and s[1] >= 2 and s[0] != top and s[0] not in r.players) else s
    Cd = pd.DataFrame(Cr)
    R['C'] = {f'{k}|{m}': dict(n=len(g), margin_minus_close=round(float((g.margin - g.mm).mean()), 2), margin_minus_open=round(float((g.margin - g.mmo).mean()), 2),
                               line_move=round(float((g.mm - g.mmo).mean()), 2)) for (k, m), g in Cd.groupby(['kind', 'missing'])} if len(Cd) else {}
    # ---------- D: props as production priors ----------
    D = T.dropna(subset=['pass_l', 'pass_a', 'pass_a_std', 'rush_l', 'rush_a', 'rush_a_std'])
    D = D[D.n_rush >= 2]
    R['D'] = dict(n=len(D), pass_mae_prop=round(float((D.pass_a - D.pass_l).abs().mean()), 1), pass_mae_season=round(float((D.pass_a - D.pass_a_std).abs().mean()), 1),
                  pass_corr_prop=round(float(D[['pass_a', 'pass_l']].corr().iloc[0, 1]), 3), pass_corr_season=round(float(D[['pass_a', 'pass_a_std']].corr().iloc[0, 1]), 3),
                  rush_corr_prop_sum=round(float(D[['rush_a', 'rush_l']].corr().iloc[0, 1]), 3), rush_corr_season=round(float(D[['rush_a', 'rush_a_std']].corr().iloc[0, 1]), 3),
                  pass_share_corr_prop=round(float(np.corrcoef(D.pass_a / (D.pass_a + D.rush_a), D.pass_l / (D.pass_l + D.rush_l))[0, 1]), 3),
                  pass_share_corr_season=round(float(np.corrcoef(D.pass_a / (D.pass_a + D.rush_a), D.pass_a_std / (D.pass_a_std + D.rush_a_std))[0, 1]), 3))
    R['made'] = dt.datetime.now(dt.timezone.utc).strftime('%Y-%m-%d %H:%M UTC')
    json.dump(R, open(f'{US}/cfbp/report.json', 'w'), indent=1)
    T.drop(columns=['players']).to_csv(f'{US}/cfbp/team_games.csv', index=False)
    print(json.dumps(R, indent=1))


if __name__ == '__main__':
    main()
