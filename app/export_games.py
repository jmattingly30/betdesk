"""Export the game (team) models for the Bet Desk Games tab: NFL and NBA. (CFB: models/cfb-v3/export_app.py; CBB:
models/cbb/export_app.py; soccer: models/soccer/export_app.py; MLB uses the MLB app's projected runs.)

python3 export_games.py <nfl games.csv> [out.json] [--nba-rosters rosters_2027.parquet]

NFL: score + market ratings from models/team-strength/current_ratings.py (rebuilt here on the fresh nflverse schedule,
written to a temp folder, team-strength files untouched), plus walk-forward offense/defense points ratings for totals.
Spread and total line weights (how much of the model-vs-line gap is real) are fit on 2012-2025 closing lines.
NBA: models/nba-team (predict.py ported): team tempo/O/D, lineup weights, and each team's current roster (ESPN
2026-27 rosters) with every player's per-100 rates and projected minutes, so the app can price any players out."""
import sys, os, json, re, datetime, tempfile, unicodedata, urllib.request, numpy as np, pandas as pd
here = os.path.dirname(os.path.abspath(__file__))
TS = os.path.normpath(f'{here}/../models/team-strength'); NT = os.path.normpath(f'{here}/../models/nba-team')
sys.path.insert(0, TS)
import current_ratings as cr
args = sys.argv[1:]
ros_path = None
if '--nba-rosters' in args:
    k = args.index('--nba-rosters'); ros_path = args[k + 1]; del args[k:k + 2]
games_csv = args[0]; out = args[1] if len(args) > 1 else f'{here}/data/games_models.json'

# ---------------- NFL ----------------
D = tempfile.mkdtemp(); os.makedirs(f'{D}/nfl'); os.symlink(os.path.abspath(games_csv), f'{D}/nfl/games.csv')
for f in ['cfb_games.parquet', 'nba_games.parquet']: os.symlink(f'{TS}/data/{f}', f'{D}/{f}')
O = tempfile.mkdtemp(); cr.build(D, O)
W = json.load(open(f'{O}/blend_weights.json'))['NFL']
R = pd.read_csv(f'{O}/ratings_nfl.csv')
g = pd.read_csv(games_csv); g = g[(g.season >= 2006)].copy()
g['date'] = pd.to_datetime(g.gameday); g = g.sort_values(['date', 'game_id']).reset_index(drop=True)
done = g.result.notna().values

def points_od(g, k, carry):
    """walk-forward points scored = mu + O[team] + D[opp] (+ home edge h); pre-game expected points for each side."""
    Om, Dm = {}, {}; last = None; eh, ea = np.zeros(len(g)), np.zeros(len(g)); mu, hh = 22.0, 1.0
    for i, r in enumerate(g.itertuples()):
        if r.season != last:
            Om = {t: carry * v for t, v in Om.items()}; Dm = {t: carry * v for t, v in Dm.items()}; last = r.season
        h, a = r.home_team, r.away_team; neu = r.location == 'Neutral'
        ph = mu + Om.get(h, 0) + Dm.get(a, 0) + (0 if neu else hh / 2); pa = mu + Om.get(a, 0) + Dm.get(h, 0) - (0 if neu else hh / 2)
        eh[i], ea[i] = ph, pa
        if done[i]:
            e1, e2 = r.home_score - ph, r.away_score - pa
            Om[h] = Om.get(h, 0) + k * e1; Dm[a] = Dm.get(a, 0) + k * e1; Om[a] = Om.get(a, 0) + k * e2; Dm[h] = Dm.get(h, 0) + k * e2
            mu += 0.002 * (e1 + e2) / 2; hh += 0.002 * (e1 - e2)
    return eh, ea, Om, Dm, mu, hh
best = None
for k in [0.03, 0.05, 0.07, 0.09]:
    for c in [0.4, 0.6, 0.8]:
        eh, ea, *_ = points_od(g, k, c)
        m = done & (g.season.between(2008, 2019)).values
        mse = np.mean(((g.home_score + g.away_score).values[m] - (eh + ea)[m]) ** 2)
        if best is None or mse < best[0]: best = (mse, k, c)
_, k, c = best
eh, ea, Om, Dm, mu, hh = points_od(g, k, c)
g['mt'] = eh + ea
f = g[done & g.total_line.notna() & (g.season.between(2012, 2025))]
X = np.column_stack([np.ones(len(f)), f.total_line, f.mt]); b = np.linalg.lstsq(X, f.total, rcond=None)[0]
wt = float(min(1.0, max(0.0, b[1] / (b[1] + b[2])))); sd_t = float(np.std(f.total - X @ b))
sd_model_t = float(np.std(f.total - f.mt))
cur = g.season.max(); act = set(g[g.season == cur].home_team) | set(g[g.season == cur].away_team)
nfl = dict(made=W['last_game'], w=W, total=dict(k=k, carry=c, mu=round(mu, 3), hfa=round(hh, 3), line_weight=round(wt, 3), sd=round(sd_t, 2), sd_model=round(sd_model_t, 2)),
           teams=[[t.team, t.score_rating, None if pd.isna(t.market_rating) else t.market_rating, round(Om.get(t.team, 0), 2), round(Dm.get(t.team, 0), 2)]
                  for t in R.itertuples() if t.team in act])
print(f'NFL totals: k {k} carry {c}; line weight {wt:.2f}, sd {sd_t:.1f} (model alone {sd_model_t:.1f})')
# starting QB (J 2026-10-18): today's projected starter vs the usual one, priced by models/team-strength/nfl/qb_change.py
sys.path.insert(0, f'{TS}/nfl'); import qb_change
Q = qb_change.main(games_csv); F = Q['fit']['margin']
nfl['qb'] = dict(new=F['new_per_epa'], cont=F['cont_per_epa'], games={
    gid.split('_')[3] + '@' + gid.split('_')[2]: dict(hq=v[2], aq=v[3], hf=v[0], af=v[1], pts=round((F['new_per_epa'] if max(v[0], v[1]) == 1 else F['cont_per_epa']) * v[4], 2))
    for gid, v in Q['upcoming'].items() if (v[0] or v[1])}, usual=Q['usual'])
import season_weight
Wi, CSR = season_weight.fit_current(games_csv)   # from week 5: refit weights with this-season form (season_weight.py)
nfl['w_in'] = Wi; nfl['week'] = int(g[~done & (g.season == cur)].week.min()) if (~done & (g.season == cur)).any() else None
for t in nfl['teams']: t.append(CSR.get(t[0], 0.0))
import injury_test   # non-QB regulars Out or Doubtful on this week's injury report (injury_test.py)
nfl['inj'] = dict(coef=injury_test.INJ_COEF, teams=injury_test.current(int(cur), nfl['week']) if nfl['week'] else {})
print('NFL missing regulars', {t: v[0] for t, v in nfl['inj']['teams'].items()})
print('NFL in-season weights', Wi, 'week', nfl['week'])
print('NFL QB changes:', {k: x['pts'] for k, x in nfl['qb']['games'].items()})

# ---------------- NBA ----------------
cfg = json.load(open(f'{NT}/model_config.json')); season = cfg['season']
RC = cfg.get('rc'); Wl = np.array(cfg['lineup_weights'][str(season)]); wo, wd = Wl[1:1 + len(RC)], Wl[1 + len(RC):1 + 2 * len(RC)]
la = pd.read_parquet(f'{NT}/data/lineup_agg.parquet'); gm = pd.read_parquet(f'{NT}/data/games.parquet')[['game_id', 'season', 'home', 'away']]
la = la.merge(gm, on='game_id'); la = la[la.season == season]; la['opp'] = np.where(la.team == la.home, la.away, la.home)
oth = la.set_index(['game_id', 'team'])[RC]; Y = oth.reindex(pd.MultiIndex.from_arrays([la.game_id, la.opp])).to_numpy()
centre = float(np.nanmean(la[RC].to_numpy() @ wo + Y @ wd))
tr = pd.read_csv(f'{NT}/team_residual_ratings.csv').set_index('team')
cp = pd.read_parquet(f'{NT}/data/current_players.parquet')
if not ros_path:
    ros_path = '/tmp/nba_rosters_2027.parquet'
    urllib.request.urlretrieve('https://raw.githubusercontent.com/sportsdataverse/hoopR-nba-data/main/nba/rosters/parquet/rosters_2027.parquet', ros_path)
ros = pd.read_parquet(ros_path); cur_team = dict(zip(ros.athlete_id.astype('int64'), ros.team_abbreviation))
cp = cp[cp.season_last >= season - 1].copy(); cp['now'] = cp.pid.map(cur_team)
teams = {}
for t, r in tr.iterrows():
    p = cp[cp.now == t].sort_values('min_proj', ascending=False).head(15)
    teams[t] = dict(tempo=r.tempo, O=r.O, D=r.D, mu=r.mu, hca=r.hca, T0=r.T0,
                    players=[[n, round(float(m), 1)] + [round(float(x), 4) for x in v] for n, m, v in zip(p.name, p.min_proj, p[RC].to_numpy())])
bo, bc = cfg['blend_open'], cfg['blend_close']
nba = dict(made=str(pd.Timestamp(cp.date.max()).date()), k=cfg['k'], boost=cfg['absence_boost'], sigma=cfg['sigma_margin'], centre=centre,
           wo=[round(float(x), 5) for x in wo], wd=[round(float(x), 5) for x in wd], playoff=cfg.get('playoff_home_edge', 0),
           win_market=cfg.get('win_market'),
           open=dict(b=bo['b_spread_open'], s=bo['sd_ats_open'], bt=bo['b_total_open'], st=bo['sd_total_open']),
           close=dict(b=bc['b_spread'], s=bc['sd_ats'], bt=bc['b_total'], st=bc['sd_total']), teams=teams)
print('NBA teams', len(teams), 'players', sum(len(v['players']) for v in teams.values()))

doc = dict(made=datetime.datetime.now(datetime.timezone.utc).strftime('%Y-%m-%d %H:%M UTC'), NFL=nfl, NBA=nba)
json.dump(doc, open(out, 'w'), separators=(',', ':'), allow_nan=False)
print('->', out, os.path.getsize(out) // 1024, 'KB')
