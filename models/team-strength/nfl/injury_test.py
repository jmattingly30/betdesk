"""Do non-QB injuries move NFL margins beyond the ratings + QB term? (J 2026-10-18: PHI without its top 3 RBs, WR1
back from injury, line and secondary hurt.)
python3 injury_test.py <games.csv> [nflverse dir with inj_/snaps_ YYYY.parquet + players.parquet]

A missing regular = a player listed Out or Doubtful on the final injury report, weighted by his average snap share in
the team's previous 3 games. Summed by group (OL, RB, WR/TE, DL, LB, DB), home minus away, then added to the ratings +
QB model; leave-one-season-out over 2017-2025 for the final margin, plus the same fit on the closing line (does the
market price it?).
Result 2026-10-18 (2017-25 held out): total missing regulars helps the margin a little (13.041 -> 13.023, -0.87 pts
per full-time starter out; by group is noisier, 13.053); the closing line moves -0.53 per starter. In the card as
-0.87 x (home missing - away missing); Questionable players are not counted."""
import sys, os, numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.dirname(HERE)); sys.path.insert(0, HERE)
import season_weight, qb_change
D = sys.argv[2] if len(sys.argv) > 2 and __name__ == '__main__' else '/tmp/claude-0/nfldl'
INJ_COEF = -0.87
GRP = {'T': 'OL', 'G': 'OL', 'C': 'OL', 'OT': 'OL', 'OG': 'OL', 'RB': 'RB', 'FB': 'RB', 'WR': 'WRTE', 'TE': 'WRTE', 'DE': 'DL', 'DT': 'DL', 'NT': 'DL',
       'LB': 'LB', 'OLB': 'LB', 'ILB': 'LB', 'MLB': 'LB', 'CB': 'DB', 'S': 'DB', 'FS': 'DB', 'SS': 'DB', 'DB': 'DB'}
G = ['OL', 'RB', 'WRTE', 'DL', 'LB', 'DB']

def missing(seasons, regular=False):
    """regular=False (the fit): share = snaps over the team's previous 3 weeks. regular=True (this week's card, 2026-10-27,
    J: PHI's WR1 out again after missing weeks 4-5 counted as only 0.22 of a starter): share = his average over the games
    he played in the team's previous 5 weeks (2+ games), so a starter who was hurt recently still counts as a starter."""
    pl = pd.read_parquet(f'{D}/players.parquet', columns=['gsis_id', 'pfr_id']).dropna(); g2p = dict(zip(pl.gsis_id, pl.pfr_id))
    rows = []
    for y in seasons:
        fi, fs = f'{D}/inj_{y}.parquet', f'{D}/snaps_{y}.parquet'
        if not (os.path.exists(fi) and os.path.exists(fs)): continue
        inj = pd.read_parquet(fi); inj = inj[inj.report_status.isin(['Out', 'Doubtful']) & (inj.get('game_type', inj.get('season_type')) == 'REG')]
        sn = pd.read_parquet(fs); sn = sn[sn.game_type == 'REG']; sn['pct'] = sn[['offense_pct', 'defense_pct']].max(axis=1)
        sn = sn.sort_values('week')
        for (t, w), x in inj.groupby(['team', 'week']):
            if regular:
                prev = sn[(sn.team == t) & (sn.week < w) & (sn.week >= w - 5) & (sn.pct > 0)]
                c = prev.groupby('pfr_player_id').pct; share = c.mean()[c.count() >= 2]
            else:
                prev = sn[(sn.team == t) & (sn.week < w) & (sn.week >= w - 3)]
                share = prev.groupby('pfr_player_id').pct.sum() / max(1, prev.week.nunique())
            v = dict.fromkeys(G, 0.0); who = []
            for r in x.itertuples():
                gp = GRP.get(r.position); pid = g2p.get(r.gsis_id)
                if gp and pid in share.index:
                    v[gp] += float(share[pid])
                    if share[pid] >= 0.3: who.append(f'{r.full_name} {r.position}')
            rows.append(dict(season=y, week=w, team=t, who=who, **v))
    return pd.DataFrame(rows)

def current(season, week):
    """this week's missing regulars per team: {team: [starter-equivalents, [names]]}"""
    import urllib.request; B = 'https://github.com/nflverse/nflverse-data/releases/download'; os.makedirs(D, exist_ok=True)
    for u, f in ((f'{B}/injuries/injuries_{season}.parquet', f'inj_{season}'), (f'{B}/snap_counts/snap_counts_{season}.parquet', f'snaps_{season}'), (f'{B}/players/players.parquet', 'players')):
        f = f'{D}/{f}.parquet'
        if not os.path.exists(f) or os.path.getmtime(f) < __import__('time').time() - 12 * 3600:   # fresh this morning
            try: urllib.request.urlretrieve(u, f)
            except Exception as e: print('download failed', u, e)
    x = missing([season], regular=True); x = x[x.week == week] if len(x) else x
    return {r.team: [round(sum(getattr(r, k) for k in G), 2), r.who] for r in x.itertuples()}

def main():
    g = season_weight.prep(sys.argv[1]); g = g[g.season >= 2017].reset_index(drop=True)
    mi = missing(range(2017, int(g.season.max()) + 1)).set_index(['season', 'week', 'team'])
    for side, col in (('h', 'home'), ('a', 'away')):
        j = mi.reindex(pd.MultiIndex.from_arrays([g.season, g.week, g[col]])).fillna(0.0).reset_index(drop=True)
        for k in G: g[f'{side}_{k}'] = j[k].values
    for k in G: g[f'd_{k}'] = g[f'h_{k}'] - g[f'a_{k}']
    # QB term as in qb_change (value gap where a non-usual starter starts)
    n = pd.read_csv(sys.argv[1]); n = n[n.season >= 2006].copy(); n['date'] = pd.to_datetime(n.gameday); n = n.sort_values(['date', 'game_id'])
    qid = dict(zip(n.game_id, zip(n.home_qb_id, n.away_qb_id)))
    m = g.season.between(2017, 2025) & g.mkt_pred.notna() & g.week.ge(2)
    print('mean missing starter-equivalents per team-game:', {k: round(float(g.loc[m, f'h_{k}'].mean()), 2) for k in G})
    base = lambda q: [np.ones(q.sum()), g.rt_pred[q], g.mkt_pred[q], g.cs_pred[q]]
    for tgt in ('margin', 'line'):
        for name, extra in (('all groups', G), ('total', None)):
            cols1 = (lambda q: base(q) + [g[f'd_{k}'][q] for k in G]) if extra else (lambda q: base(q) + [sum(g[f'd_{k}'][q] for k in G)])
            e0, e1 = [], []
            for s in range(2017, 2026):
                tr, te = m & (g.season != s), m & (g.season == s)
                for cols, e in ((base, e0), (cols1, e1)):
                    b = np.linalg.lstsq(np.column_stack(cols(tr)), g[tgt][tr], rcond=None)[0]; e.append(g[tgt][te].values - np.column_stack(cols(te)) @ b)
            b = np.linalg.lstsq(np.column_stack(cols1(m)), g[tgt][m], rcond=None)[0]
            e0, e1 = np.concatenate(e0), np.concatenate(e1)
            coefs = dict(zip(G, np.round(b[4:], 2))) if extra else round(float(b[4]), 2)
            print(f'{tgt:6s} + {name:10s}: held-out RMSE {np.sqrt(np.mean(e0**2)):.3f} -> {np.sqrt(np.mean(e1**2)):.3f}; pts per missing regular {coefs}')

if __name__ == '__main__': main()
