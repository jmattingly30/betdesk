"""Export the NFL player prop model's calibrated over/under curves for the Bet Desk app.

python3 export_nfl_props.py <dl_dir> SEASON WEEK [out.json] [--overrides overrides.csv]
WEEK can be 'auto': the next week with games left, moving on once 3/4 of a week's games are final (J 2026-10-20: be
as early as possible on lines, so next week's props build Monday and are not held back by Monday night's game).

Runs models/nfl-player/project.py unchanged (same injury report, overrides and calibration), then writes, for every
projected player and stat, the calibrated P(over) at half-point lines across the player's range. The app reads that
curve to price any line J types in: integer lines get exact push handling for count stats, yards lines are
interpolated in logit space between grid points 5 yards apart."""
import sys, os, json, runpy, datetime, numpy as np

here = os.path.dirname(os.path.abspath(__file__))
M = os.path.normpath(f'{here}/../models/nfl-player')
args = [a for a in sys.argv[1:]]
ov = None
if '--overrides' in args:
    k = args.index('--overrides'); ov = args[k + 1]; del args[k:k + 2]
dl, season = args[0], int(args[1])

def auto_week(dl, season):
    import pandas as pd
    g = pd.read_csv(f'{dl}/games.csv'); g = g[g.season == season]
    left = g[g.home_score.isna()].week.unique()
    for w in sorted(left):
        if g[(g.week == w)].home_score.notna().mean() < 0.75: return int(w)
    return int(max(left))

week = auto_week(dl, season) if args[2] == 'auto' else int(args[2])
print('NFL props week', week)
out = args[3] if len(args) > 3 else f'{here}/data/nfl_props.json'
tmp = f'/tmp/nfl_proj_{season}_{week}.csv'
if ov and os.path.exists(ov):
    # J 2026-11-05: an override row with an 'until' date (YYYY-MM-DD, the player's game day) expires after that day (Denver)
    import pandas as pd
    o = pd.read_csv(ov, dtype=str).fillna('')
    if 'until' in o:
        today = pd.Timestamp.now(tz='America/Denver').strftime('%Y-%m-%d')
        keep = o[(o.until == '') | (o.until >= today)]
        if len(keep) < len(o): print('expired overrides dropped:', o[~o.index.isin(keep.index)].team.tolist())
        ov = f'/tmp/nfl_overrides_live.csv'; keep.drop(columns='until').to_csv(ov, index=False)
    print('overrides:', pd.read_csv(ov).fillna('').to_dict('records'))
sys.argv = ['project.py', M, dl, str(season), str(week), '--out', tmp] + (['--overrides', ov] if ov else [])
g = runpy.run_path(f'{M}/project.py', run_name='__main__')
pred, sims, cal, P = g['pred'], g['sims'], g['cal'], g['P']
T = cal['temperature']

def calp(raw, stat):
    raw = np.clip(raw, 1e-4, 1 - 1e-4)
    return 1 / (1 + np.exp(-np.log(raw / (1 - raw)) / T.get(stat, 1.0)))

YARDS = {'rec_yds', 'rush_yds', 'pass_yds', 'rush_rec_yds'}
SHOW = g['SHOW']
players = []
for i, r in pred.iterrows():
    if r.position == 'QB' and r.is_qb_start != 1:
        continue
    st = {}
    for s in SHOW[r.position]:
        x = np.asarray(sims[s][i], dtype=float)
        if s == 'td':
            st['td'] = dict(m=round(float(x.mean()), 2), c=[[0.5, round(float(calp((x > 0.5).mean(), 'td')), 4)],
                                                            [1.5, round(float(calp((x > 1.5).mean(), 'td')), 4)]])
            continue
        lo, hi = np.percentile(x, [1, 99])
        step = 5 if s in YARDS else 1
        start = np.floor(lo / step) * step - (0.5 if step == 1 else 0) + (0.5 if step == 5 else 0)
        grid = np.arange(max(start, -0.5 if step == 1 else -4.5), hi + step, step)
        curve = [[round(float(L), 1), round(float(calp((x > L).mean(), s)), 4)] for L in grid]
        q10, q50, q90 = np.percentile(x, [10, 50, 90])
        st[s] = dict(m=round(float(r[f'{s}_hat']), 1), q=[round(float(q10), 1), round(float(q50), 1), round(float(q90), 1)], c=curve)
    players.append(dict(n=r['name'], t=r.posteam, o=r.opp, p=r.position, g=r.game_id, h=int(r.home),
                        imp=round(float(r.implied), 1), sp=round(float(r.spread), 1), s=st))
games = sorted({p['g'] for p in players})
doc = dict(sport='NFL', season=season, week=week, made=datetime.datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC'),
           note='Assumes the player plays (props are void otherwise). Injury report as of the run time.',
           games=games, players=players)
os.makedirs(os.path.dirname(out), exist_ok=True)
json.dump(doc, open(out, 'w'), separators=(',', ':'))
print(f'{len(players)} players, {len(games)} games -> {out} ({os.path.getsize(out) / 1024:.0f} KB)')
