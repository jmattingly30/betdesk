"""Game-day projections for NBA players: minutes, mean and range for each stat, and P(over) for any prop line.

usage: python3 project.py <model_dir> <games.csv> [lines.csv] [out.csv]
games.csv : one row per team per game: date,team,opp,home,spread,total,out,add
            spread = expected margin for `team` (+ favoured), total = game total; take both from the betting market.
            out = players not playing (names separated by ';'); add = players new to the team (signings/trades).
            optional column q = questionable players with your chance they play, 'Name:0.6;Name2:0.25' (audit P1):
            teammates' projections become the mixture mean over him playing or sitting; his own projection is
            conditional on him playing (a prop on a player who sits is void).
            The expected lineup is the team's latest roster in the box scores, minus out, plus add.
lines.csv : optional prop lines: name,stat,line  (stat in pts reb ast fg3m stl blk tov pra)
Writes out.csv (default projections.csv) and prints a summary with team points built from the player projections."""
import sys, os, re, pickle, subprocess, unicodedata, numpy as np, pandas as pd
md = sys.argv[1]; sys.path.insert(0, md)
from model import fit, usable, normalize_minutes, STATS
games = pd.read_csv(sys.argv[2], dtype=str).fillna('')
lines = pd.read_csv(sys.argv[3]) if len(sys.argv) > 3 and sys.argv[3] not in ('', '-') else None
outp = sys.argv[4] if len(sys.argv) > 4 else 'projections.csv'
def norm(s):
    s = unicodedata.normalize('NFKD', str(s)).encode('ascii', 'ignore').decode().lower()
    return ' '.join(re.sub(r'\b(jr|sr|ii|iii|iv)\b', '', re.sub(r'[^a-z ]', '', s)).split())
pg = pd.read_parquet(f'{md}/data/player_games.parquet')
pl = pg[pg.played].sort_values('date')
last = pl.groupby('pid').tail(1).set_index('pid')
team_last = pl.groupby('team').agg(season=('season', 'max'), date=('date', 'max'))
by_name = {norm(n): pid for pid, n in zip(last.index, last.name)}
rows, gids = [], {}
for k, g in games.iterrows():
    date = pd.Timestamp(g.date); team, opp = g.team.upper(), g.opp.upper()
    tl = team_last.loc[team]
    season = int(tl.season) + (1 if (date - tl.date).days > 90 else 0)
    roster = last[(last.team == team) & (last.season == tl.season)]
    out = {norm(x) for x in g['out'].split(';') if x.strip()}
    qp = {}
    for x in str(g.get('q', '')).split(';'):
        if ':' in x: nm_, pv_ = x.rsplit(':', 1); qp[norm(nm_)] = min(1.0, max(0.0, float(pv_)))
    add = [by_name[norm(x)] for x in g['add'].split(';') if x.strip() and norm(x) in by_name]
    miss = [x for x in g['add'].split(';') if x.strip() and norm(x) not in by_name]
    if miss: print('no NBA history, skipped:', miss)
    unknown = out - {norm(n) for n in roster.name}
    if unknown: print(f'{team}: out names not on the latest roster (ignored):', sorted(unknown))
    ids = [p for p in roster.index if norm(last.loc[p, 'name']) not in out] + [p for p in add if p not in roster.index]
    spread, total = float(g.spread), float(g.total)
    for pid in ids:
        r = last.loc[pid]
        gid = gids.setdefault((g.date, frozenset([team, opp])), -(len(gids) + 1))
        rows.append({'game_id': gid, 'season': season, 'date': date, 'playoff': 0,
                     'pid': pid, 'name': r['name'], 'team': team, 'opp': opp, 'is_home': int(g.home), 'pos': r.pos, 'pos_grp': r.pos_grp,
                     'starter': False, 'played': True, 'did_not_play': False, 'dnp_injury': False, 'reason': '',
                     'total_line': total, 'team_spread': spread, 'team_implied': (total + spread) / 2,
                     'rest': min(7.0, max(1.0, (date - tl.date).days)), 'future': True, 'p_play': qp.get(norm(r['name']), 1.0)})
up = pd.DataFrame(rows)
for c in pg.columns:
    if c not in up: up[c] = np.nan
tmp_in, tmp_out = f'{md}/data/_upcoming_in.parquet', f'{md}/data/_upcoming_feat.parquet'
up[list(pg.columns) + ['future', 'p_play']].to_parquet(tmp_in)
subprocess.run([sys.executable, f'{md}/features.py', md, tmp_in, tmp_out], check=True, stdout=subprocess.DEVNULL)
fu = pd.read_parquet(tmp_out); os.remove(tmp_in); os.remove(tmp_out)
mp = f'{md}/data/model.pkl'
if os.path.exists(mp): m = pickle.load(open(mp, 'rb'))
else:
    f = pd.read_parquet(f'{md}/data/features.parquet'); m = fit(f[usable(f) & (f.season >= 2019)]); pickle.dump(m, open(mp, 'wb'))
fu['n_hist'] = fu.n_hist.fillna(0)
ok = fu.ml.notna()
if (~ok).any(): print('no minutes history, not projected:', fu.loc[~ok, 'name'].tolist())
fu = fu[ok].copy(); pr = m.predict(fu)
pr = normalize_minutes(pr, [fu.game_id, fu.team]); fu = fu.join(pr)
res = fu[['date', 'team', 'opp', 'name', 'min_proj', 'min_sd']].copy()
for s in ['pts', 'reb', 'ast', 'fg3m', 'stl', 'blk', 'tov', 'pra']:
    res[s] = fu['mu_' + s]
    q = [m.quantiles(r, s, (0.1, 0.9)) for _, r in fu.iterrows()]
    res[s + '_p10'] = [a for a, b in q]; res[s + '_p90'] = [b for a, b in q]
if lines is not None:
    lines['key'] = lines.name.map(norm); fu['key'] = fu.name.map(norm)
    lj = lines.merge(fu, on='key', suffixes=('', '_f'))
    lj['p_over'] = [m.prob_over(r, r.stat, r.line) for _, r in lj.iterrows()]
    lj['mean'] = [r['mu_' + r.stat] for _, r in lj.iterrows()]
    # audit P3: unders beat overs in the 2023 sample (57% vs 47%); until J's own logged lines say otherwise, an over needs
    # a bigger edge. Flags assume -110 both ways (break-even 52.4%).
    OVER_MIN, UNDER_MIN = 0.58, 0.55
    lj['lean'] = np.where(lj.p_over >= OVER_MIN, 'over', np.where(1 - lj.p_over >= UNDER_MIN, 'under', ''))
    print(lj[['name', 'stat', 'line', 'mean', 'p_over', 'lean']].round(3).to_string(index=False))
    print(f'lean = P(over) >= {OVER_MIN} or P(under) >= {UNDER_MIN} (overs need more until your own results say otherwise)')
    lj[['name', 'stat', 'line', 'mean', 'p_over', 'lean']].to_csv(outp.replace('.csv', '_lines.csv'), index=False)
res = res.sort_values(['date', 'team', 'min_proj'], ascending=[True, True, False])
res.round(2).to_csv(outp, index=False)
pd.set_option('display.width', 200)
for (d, t), g in res.groupby(['date', 'team']):
    tm = games[(games.team.str.upper() == t)].iloc[0]
    print(f"\n{t} vs {g.opp.iloc[0]} {d.date()}  team points from players {g.pts.sum():.1f} (market implied {(float(tm.total) + float(tm.spread)) / 2:.1f}), minutes {g.min_proj.sum():.0f}")
    print(g[['name', 'min_proj', 'pts', 'reb', 'ast', 'fg3m', 'pra', 'pts_p10', 'pts_p90']].round(1).head(11).to_string(index=False))
