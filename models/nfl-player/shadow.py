"""NFL scheme shadow projections (J 2026-10-18, matchup tracking option 4): candidate scheme terms run LIVE next to our
official props, never shown as the projection, graded side by side by models/grading/grade.py (scorecard `shadow`).
A term is worth adopting only if its shadow beats the official projection over a real sample.

python3 shadow.py            # after the NFL props export (app/data/nfl_props.json); before grade.py

Candidates (the NFL-4 Option 2 and 3 features from scheme_features.add_scheme, which the backtest left out of the
yards/receptions models because the gain was ~0 in 2024-25; a live test checks that on 2026 games):
  cov   receivers (targets, rec, rec_yds, rush_rec_yds for RBs' receiving part not split): x ms, the geometric mean of the
        player's coverage-split multipliers vs tonight's defence (man/zone, one/two-high, base/nickel/dime, 4/5+ rushers);
        RBs (carries, rush_yds): x mc_box (his carries per run by box count vs the defence's box mix)
  pers  targets, rec, rec_yds, carries, rush_yds: x on_ratio (his expected on-field rate under tonight's expected
        personnel, from the caller prior + game plan vs the defence's nickel/blitz rates, vs last season's mix)
Writes ../grading/history/NFL_SHADOW.json.gz: {"game|team|name|stat": {cov, pers, m, made}}; a game's rows are frozen
once that game has a box score, like the official snapshot."""
import os, io, json, gzip, datetime as dt, urllib.request, numpy as np, pandas as pd
import scheme_features as sf

HERE = os.path.dirname(os.path.abspath(__file__))
APP = os.path.normpath(f'{HERE}/../../app/data'); OUT = os.path.normpath(f'{HERE}/../grading/history/NFL_SHADOW.json.gz')
SEASON = 2026
STATS = f'https://github.com/nflverse/nflverse-data/releases/download/stats_player/stats_player_week_{SEASON}.parquet'
REC, RUSH = ['targets', 'rec', 'rec_yds'], ['carries', 'rush_yds']

def main():
    d = json.load(open(f'{APP}/nfl_props.json'))
    with urllib.request.urlopen(STATS, timeout=90) as r: W = pd.read_parquet(io.BytesIO(r.read()))
    W = W[W.season_type == 'REG']
    played = set(W.game_id.astype(str))
    ids = {}; gp = W.groupby('player_id').game_id.nunique().to_dict()
    for x in W.sort_values('week').itertuples():
        for nm in {x.player_display_name, x.player_name}: ids[(x.team, str(nm))] = (x.player_id, x.position)
    rows = []
    for p in d['players']:
        pid, pos = ids.get((p['t'], p['n']), (None, p.get('p')))
        if pid is None: continue
        rows.append(dict(season=SEASON, posteam=p['t'], opp=p.get('o'), pid=pid, position=pos, games_season=gp.get(pid, 0), key=f"{p['g']}|{p['t']}|{p['n']}", g=str(p['g']), s=p['s']))
    if not rows: print('shadow: no players matched'); return
    F = pd.DataFrame(rows); X = sf.add_scheme(F.drop(columns=['s']), HERE)
    ms = np.exp(np.nanmean(np.log(X[['ms_mz', 'ms_shell', 'ms_db', 'ms_pr']].clip(0.5, 2)), axis=1))
    H = json.load(gzip.open(OUT, 'rt')) if os.path.exists(OUT) else {}
    made = dt.datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC'); n = 0
    for i, r in F.iterrows():
        if r.g in played: continue
        cov_r, cov_c, pers = ms[i], X.mc_box.iloc[i], X.on_ratio.iloc[i]
        for st, s in r.s.items():
            if st in REC: c = cov_r
            elif st in RUSH: c = cov_c
            else: continue
            c = 1.0 if c != c else float(c); q = 1.0 if pers != pers else float(np.clip(pers, 0.7, 1.3))
            H[f'{r.key}|{st}'] = dict(m=s['m'], cov=round(s['m'] * c, 3), pers=round(s['m'] * q, 3), made=made); n += 1
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    json.dump(H, gzip.open(OUT, 'wt'))
    moved = pd.Series([abs(v['cov'] / v['m'] - 1) for v in H.values() if v['m']]).describe(percentiles=[.5, .9]).round(4).to_dict()
    print('shadow', n, 'rows written, history', len(H), 'cov change', moved)

if __name__ == '__main__':
    main()
