"""Project every offensive skill player for an upcoming NFL week.

python3 project.py <model_dir> <dl_dir> SEASON WEEK [--overrides overrides.csv] [--lines lines.csv] [--out projections.csv]

<dl_dir>: games.csv (github.com/nflverse/nfldata data/games.csv: schedule + spread/total) and injuries_SEASON.parquet
          (nflverse-data release `injuries`, saved as inj_SEASON.parquet).
Expected players per team: anyone who played offense for the team in its last two games, plus anyone listed Questionable
who has played for the team this season, minus anyone ruled Out or Doubtful on the injury report.
overrides.csv (optional): team,out,add,qb   (out/add = names separated by ';', qb = starting QB name)
lines.csv (optional):     name,stat,line    (stat one of targets rec rec_yds carries rush_yds rush_rec_yds pass_att cmp pass_yds pass_td td)
Projections assume the player plays (props are void otherwise)."""
import sys, os, json, pickle, argparse, numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import model as mdl, features, scheme_features

ap = argparse.ArgumentParser()
ap.add_argument('model_dir'); ap.add_argument('dl_dir'); ap.add_argument('season', type=int); ap.add_argument('week', type=int)
ap.add_argument('--overrides'); ap.add_argument('--lines'); ap.add_argument('--out')
ap.add_argument('--sims', type=int, default=4000)
a = ap.parse_args()
M = a.model_dir

hist = pd.read_parquet(f'{M}/data/player_games.parquet')
sched = pd.read_csv(f'{a.dl_dir}/games.csv')
wk = sched[(sched.season == a.season) & (sched.week == a.week) & sched.home_score.isna()]
if wk.empty:
    sys.exit(f'no unplayed games for {a.season} week {a.week} in games.csv')
inj_path = f'{a.dl_dir}/inj_{a.season}.parquet'
inj = pd.read_parquet(inj_path) if os.path.exists(inj_path) else pd.DataFrame(columns=['week', 'gsis_id', 'team', 'report_status'])
inj = inj[(inj.week == a.week)]
ov = pd.read_csv(a.overrides).fillna('') if a.overrides else pd.DataFrame(columns=['team', 'out', 'add', 'qb'])
names = hist.drop_duplicates('pid', keep='last').set_index('pid')
by_name = {n.lower(): pid for pid, n in names.name.items()}

def split(s):
    return [x.strip() for x in str(s).split(';') if x.strip()]

rows = []
for g in wk.itertuples():
    for team, opp, home in [(g.home_team, g.away_team, 1), (g.away_team, g.home_team, 0)]:
        th = hist[hist.posteam == team]
        last_games = th.drop_duplicates('game_id').sort_values('game_date').game_id.tail(2)
        recent = th[th.game_id.isin(last_games) & (th.offense_snaps > 0)].pid.unique().tolist()
        latest_team = hist.sort_values('game_date').drop_duplicates('pid', keep='last').set_index('pid').posteam
        q_ok = inj[(inj.team == team) & (inj.report_status == 'Questionable')].gsis_id
        recent += [p for p in q_ok if p in latest_team.index and latest_team[p] == team and p not in recent
                   and ((hist.pid == p) & (hist.season == a.season)).any()]
        recent = [p for p in recent if latest_team.get(p) == team]
        out_ids = set(inj[(inj.team == team) & inj.report_status.isin(['Out', 'Doubtful'])].gsis_id)
        o = ov[ov.team == team]
        for n in (split(o.out.iloc[0]) if len(o) else []):
            out_ids.add(by_name.get(n.lower(), n))
        adds = [by_name.get(n.lower()) for n in (split(o['add'].iloc[0]) if len(o) else [])]
        if len(o) and any(x is None for x in adds):
            print('unknown name in add list for', team, split(o['add'].iloc[0]))
        players = [p for p in recent if p not in out_ids] + [p for p in adds if p and p not in recent]
        # starting QB: override, else the QB with most snaps in the team's last game who is available
        qbs = [p for p in players if names.position.get(p) == 'QB']
        qb = by_name.get(o.qb.iloc[0].lower()) if len(o) and o.qb.iloc[0] else None
        if qb is None and qbs:
            last = th[th.pid.isin(qbs)].sort_values(['game_date', 'offense_snaps']).groupby('pid').tail(1)
            qb = last.sort_values(['game_date', 'offense_snaps']).pid.iloc[-1] if len(last) else qbs[0]
        if qb and qb not in players:
            players.append(qb)
        spread = g.spread_line if home else -g.spread_line
        for p in players:
            rows.append(dict(game_id=g.game_id, posteam=team, pid=p, name=names.name.get(p, p), position=names.position.get(p),
                             rookie_season=names.rookie_season.get(p), season=a.season, season_type='REG', week=a.week,
                             game_date=pd.Timestamp(g.gameday), home_team=g.home_team, away_team=g.away_team,
                             spread_line=g.spread_line, total_line=g.total_line, roof=g.roof, wind=np.nan, temp=np.nan,
                             home=home, opp=opp, spread=spread, implied=g.total_line / 2 + spread / 2,
                             offense_snaps=100.0 if p == qb else (1.0 if names.position.get(p) == 'QB' else 50.0),
                             offense_pct=np.nan, _new=1))
new = pd.DataFrame(rows)
new = new[new.position.isin(['QB', 'RB', 'WR', 'TE'])]
full = pd.concat([hist.assign(_new=0), new], ignore_index=True)
for c in ['pass_att', 'cmp', 'pass_yds', 'pass_td', 'ints', 'carries', 'rush_yds', 'rush_td', 'rz_carries', 'targets', 'rec', 'rec_yds',
          'rec_td', 'air_yds', 'rz_targets']:
    full[c] = full[c].fillna(0)
# audit F1: the same pre-game availability as the backtest: starting QB and teammates' chance of playing from the
# injury report (Questionable ~69%, no status but missed last game ~31%), for history and this week alike
feat = features.build(full, features.load_injuries(a.dl_dir, range(2016, a.season + 1)))
feat = features.add_tracking(feat, a.dl_dir)           # NFL-3 team-changer target model (ngs_*.csv.gz from fetch_data.py)
if os.path.exists(f'{M}/data/scheme_team.parquet'):     # NFL-4 play-caller / personnel / coverage features (scheme_features.py)
    feat = scheme_features.add_scheme(feat, M)
cur = feat[feat._new == 1].copy()

pk = f'{M}/data/model.pkl'
if os.path.exists(pk):
    model = pickle.load(open(pk, 'rb'))
else:
    tr = feat[(feat._new == 0) & (feat.season >= 2017)]
    print(f'fitting on {len(tr)} player-games {tr.season.min()}-{tr.season.max()} ...')
    model = mdl.fit(tr)
    pickle.dump(model, open(pk, 'wb'))
if 'cl_new' in feat and 'cl_new' not in model['opp']['targets'].feature_name():   # a model saved before NFL-4: refit
    tr = feat[(feat._new == 0) & (feat.season >= 2017)]
    print(f'refitting with the NFL-4 scheme features on {len(tr)} player-games ...')
    model = mdl.fit(tr); pickle.dump(model, open(pk, 'wb'))
if 'opp_chg' not in model:                              # a model saved before 2026-10-10: add the team-changer target model
    tr = mdl.predict_team(model['team'], mdl.add_derived(feat[(feat._new == 0) & (feat.season >= 2017)]))
    model['opp_chg'] = mdl.fit_changer_targets(tr)
    pickle.dump(model, open(pk, 'wb'))
cal = json.load(open(f'{M}/calibration.json'))
pred = mdl.predict(model, cur)
pred['rush_rec_yds_hat'] = pred.rush_yds_hat + pred.rec_yds_hat
pools = mdl.load_pools(f'{M}/data/play_yards.parquet')
sims = mdl.simulate(pred.reset_index(drop=True), pools, cal['nb_size'], n=a.sims, seed=1)

SHOW = {'QB': ['pass_att', 'cmp', 'pass_yds', 'pass_td', 'carries', 'rush_yds'],
        'RB': ['carries', 'rush_yds', 'targets', 'rec', 'rec_yds', 'rush_rec_yds', 'td'],
        'WR': ['targets', 'rec', 'rec_yds', 'td'], 'TE': ['targets', 'rec', 'rec_yds', 'td']}
pred = pred.reset_index(drop=True)
out = []
for i, r in pred.iterrows():
    if r.position == 'QB' and r.is_qb_start != 1:
        continue
    d = dict(game=r.game_id, team=r.posteam, opp=r.opp, name=r['name'], pos=r.position, starter_qb=int(r.is_qb_start))
    for s in SHOW[r.position]:
        if s == 'td':
            d['p_anytime_td'] = round(float((sims['td'][i] > 0).mean()), 3)
            continue
        hat = r[f'{s}_hat']
        q10, q90 = np.percentile(sims[s][i], [10, 90])
        d[s] = round(float(hat), 1); d[f'{s}_10_90'] = f'{q10:.0f}-{q90:.0f}'
    out.append(d)
P = pd.DataFrame(out)
order = {'QB': 0, 'RB': 1, 'WR': 2, 'TE': 3}
P['_o'] = P.pos.map(order)
P['_v'] = P.get('rec_yds', 0).fillna(0) + P.get('rush_yds', 0).fillna(0) + P.get('pass_yds', 0).fillna(0)
P = P.sort_values(['game', 'team', '_o', '_v'], ascending=[True, True, True, False]).drop(columns=['_o', '_v'])
outp = a.out or f'{M}/projections_{a.season}_wk{a.week:02d}.csv'
P.to_csv(outp, index=False)
print(f'{len(P)} player projections -> {outp}')

if a.lines:
    L = pd.read_csv(a.lines)
    res = []
    def logit(p): p = np.clip(p, 1e-4, 1 - 1e-4); return np.log(p / (1 - p))
    for r in L.itertuples():
        m = pred.name.str.lower() == r.name.lower()
        if not m.any():
            print('not found:', r.name); continue
        i = pred.index[m][0]
        raw = float((sims[r.stat][i] > r.line).mean())
        p = 1 / (1 + np.exp(-logit(raw) / cal['temperature'].get(r.stat, 1.0)))
        res.append(dict(name=r.name, stat=r.stat, line=r.line, projection=round(float(pred.loc[i, f'{r.stat}_hat']), 1), p_over=round(p, 3)))
    R = pd.DataFrame(res)
    R.to_csv(outp.replace('.csv', '_lines.csv'), index=False)
    print(R.to_string(index=False))
