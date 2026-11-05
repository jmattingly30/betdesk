"""Regularised adjusted plus-minus (RAPM) from ESPN play-by-play, walk-forward (audit fix N2).

Box-score rates miss most defence. RAPM credits each player with the team's scoring margin while he is on the floor,
adjusted for the other nine players, and shrunk toward zero by a ridge penalty. Public all-in-one metrics (EPM, LEBRON,
DARKO) blend exactly this kind of term with a box prior.

1. stints: from the same on-court tracking as ../nba-player/lineups.py, every stretch with no substitution gives
   (5 home players, 5 away players, minutes, home points, away points).
2. fits: at the start of each month, ridge on all earlier stints (age-weighted, half-life 365 days, 3 seasons back):
   y = home margin per 48 minutes, weight = minutes, x = +1 home players, -1 away players, plus a home-court column.
   The rating used for a game is the latest fit made BEFORE the game's month: nothing from the game or later.

usage: python3 rapm.py <dir with pbp_YYYY.parquet> [first last]  (run from models/nba-team)
writes data/stints.parquet, data/rapm_by_month.parquet (pid, fit_date, rapm = net points per 48 on-floor minutes)"""
import sys, numpy as np, pandas as pd
from scipy.sparse import csr_matrix, hstack
from scipy.sparse.linalg import lsqr
d = sys.argv[1]
seasons = range(int(sys.argv[2]), int(sys.argv[3]) + 1) if len(sys.argv) > 3 else range(2016, 2027)
LAMBDA, HL_DAYS, BACK_DAYS = 3000.0, 365.0, 3 * 365

box = pd.read_parquet('../nba-player/data/player_games.parquet', columns=['game_id', 'pid', 'team', 'starter', 'played', 'season', 'date', 'is_home'])
box = box[box.played]

def stints_for(season):
    p = pd.read_parquet(f'{d}/pbp_{season}.parquet', columns=['game_id', 'sequence_number', 'type_text', 'period_number',
                        'start_game_seconds_remaining', 'scoring_play', 'score_value', 'athlete_id_1', 'athlete_id_2', 'athlete_id_3', 'season_type'])
    p = p[p.season_type.isin([2, 3])]
    p['game_id'] = p.game_id.astype(int); p['seq'] = pd.to_numeric(p.sequence_number, errors='coerce')
    p = p.sort_values(['game_id', 'period_number', 'start_game_seconds_remaining', 'seq'], ascending=[True, True, False, True])
    bs = box[box.season == season]
    team_of = {(g, a): t for g, a, t in zip(bs.game_id, bs.pid, bs.team)}
    starters = bs[bs.starter.astype(bool)].groupby(['game_id', 'team']).pid.apply(list).to_dict()
    home_of = bs[bs.is_home.astype(bool)].groupby('game_id').team.first().to_dict()
    out = []
    for gid, g in p.groupby('game_id', sort=False):
        teams = [t for t in {team_of.get((gid, a)) for a in bs.pid[bs.game_id == gid]} if t]
        hm = home_of.get(gid)
        if len(teams) != 2 or hm is None or any(len(starters.get((gid, t), [])) != 5 for t in teams): continue
        aw = [t for t in teams if t != hm][0]
        typ = g.type_text.fillna('').values; per = g.period_number.values.astype(int); t0 = g.start_game_seconds_remaining.values.astype(float)
        a1, a2, a3 = g.athlete_id_1.values, g.athlete_id_2.values, g.athlete_id_3.values
        sv = pd.to_numeric(g.score_value, errors='coerce').fillna(0).values; sc = g.scoring_play.fillna(False).values.astype(bool)
        n = len(g)
        def aid(x): return int(x) if x == x and x is not None else None
        is_sub = np.array(['Substitution' in s for s in typ])
        on = {t: set(starters[(gid, t)]) for t in teams}
        st = {'last': t0[0] if n else 2880.0, 'pts': {hm: 0.0, aw: 0.0}}
        def flush(now):
            dt = max(0.0, st['last'] - now)
            if dt > 0 and len(on[hm]) == 5 and len(on[aw]) == 5:
                out.append((gid, *sorted(on[hm]), *sorted(on[aw]), dt / 60, st['pts'][hm], st['pts'][aw]))
            st['pts'] = {hm: 0.0, aw: 0.0}; st['last'] = now
        def period_start(k, start_idx):
            found = {t: [] for t in teams}; entered = set()
            for m in range(start_idx, n):
                if per[m] != k: break
                if is_sub[m]:
                    inn, outp = aid(a1[m]), aid(a2[m])
                    if outp is not None and outp not in entered:
                        t = team_of.get((gid, outp))
                        if t and outp not in found[t]: found[t].append(outp)
                    if inn is not None: entered.add(inn)
                    continue
                for q in (aid(a1[m]), aid(a2[m]), aid(a3[m])):
                    if q is None or q in entered: continue
                    t = team_of.get((gid, q))
                    if t and q not in found[t]: found[t].append(q)
            for t in teams:
                L = found[t][:5]
                for q in on[t]:
                    if len(L) >= 5: break
                    if q not in L and q not in entered: L.append(q)
                for q in on[t]:
                    if len(L) >= 5: break
                    if q not in L: L.append(q)
                on[t] = set(L)
        cur = 1
        for m in range(n):
            if per[m] != cur:
                flush(t0[m]); cur = per[m]; period_start(cur, m); st['last'] = t0[m]
            if is_sub[m]:
                flush(t0[m]); inn, outp = aid(a1[m]), aid(a2[m])
                t = team_of.get((gid, inn)) or team_of.get((gid, outp))
                if t:
                    on[t].discard(outp)
                    if inn is not None: on[t].add(inn)
                continue
            if sc[m]:
                x1 = aid(a1[m]); t = team_of.get((gid, x1))
                if t in st['pts']: st['pts'][t] += sv[m]
        flush(0.0 if n == 0 else min(t0[-1], st['last']))
    cols = ['game_id'] + [f'h{i}' for i in range(5)] + [f'a{i}' for i in range(5)] + ['min', 'pts_h', 'pts_a']
    return pd.DataFrame(out, columns=cols)

S = []
for y in seasons:
    s = stints_for(y); s['season'] = y; S.append(s); print(y, 'stints', len(s), 'minutes per game %.1f' % (s['min'].sum() / s.game_id.nunique()), flush=True)
S = pd.concat(S, ignore_index=True)
S = S.merge(box[['game_id', 'date']].drop_duplicates('game_id'), on='game_id')
S.to_parquet('data/stints.parquet')

# ---- walk-forward monthly ridge fits
S = S[S['min'] > 0].reset_index(drop=True)
P = np.unique(S[[f'h{i}' for i in range(5)] + [f'a{i}' for i in range(5)]].to_numpy())
col = {p: i for i, p in enumerate(P)}
rows = np.repeat(np.arange(len(S)), 10)
cols = np.array([col[x] for x in S[[f'h{i}' for i in range(5)] + [f'a{i}' for i in range(5)]].to_numpy().ravel()])
vals = np.tile(np.r_[np.ones(5), -np.ones(5)], len(S))
y_all = (S.pts_h - S.pts_a).to_numpy() / S['min'].to_numpy() * 48
dates = S.date.to_numpy().astype('datetime64[D]')
fit_dates = pd.date_range(S.date.min() + pd.Timedelta(days=60), S.date.max() + pd.Timedelta(days=31), freq='MS')
res = []
for fd in fit_dates:
    fd64 = np.datetime64(fd.date())
    age = (fd64 - dates).astype(float)
    m = (age > 0) & (age <= BACK_DAYS)
    if m.sum() < 5000: continue
    idx = np.where(m)[0]; w = S['min'].to_numpy()[idx] * 0.5 ** (age[idx] / HL_DAYS)
    sel = np.isin(rows, idx)
    rmap = -np.ones(len(S), int); rmap[idx] = np.arange(len(idx))
    X = hstack([csr_matrix((vals[sel], (rmap[rows[sel]], cols[sel])), shape=(len(idx), len(P))), csr_matrix(np.ones((len(idx), 1)))]).tocsr()   # + home court
    sw = np.sqrt(w); Xw = X.multiply(sw[:, None]).tocsr(); yw = y_all[idx] * sw
    beta = lsqr(Xw, yw, damp=np.sqrt(LAMBDA), atol=1e-8, btol=1e-8, iter_lim=3000)[0]
    seen = np.zeros(len(P), bool); seen[np.unique(cols[sel])] = True
    res.append(pd.DataFrame({'pid': P[seen], 'fit_date': fd, 'rapm': beta[:len(P)][seen]}))
    print(fd.date(), 'stints', len(idx), 'players', int(seen.sum()), 'home %.2f' % beta[-1], 'sd rapm %.2f' % beta[:len(P)][seen].std(), flush=True)
R = pd.concat(res, ignore_index=True); R['pid'] = R.pid.astype(np.int64)
R.to_parquet('data/rapm_by_month.parquet')
last = R[R.fit_date == R.fit_date.max()].merge(box.drop_duplicates('pid', keep='last')[['pid']], on='pid')
print('latest fit', R.fit_date.max().date(), '| top 10 rapm pids:', last.nlargest(10, 'rapm').round(2).to_dict('records'))
