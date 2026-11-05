"""Walk-forward features for every player-game (only information from earlier games is used).
usage: python3 features.py <model_dir> [upcoming_rows.parquet out.parquet]
reads data/player_games.parquet, writes data/features.parquet. With upcoming rows (made by project.py: expected
players for future games, stats empty, future=True) it writes their pre-game features to out.parquet instead.

Per player: exponentially weighted per-minute rates (shrunk to a position prior), short/long minutes averages,
starter rate, days since last game, season and last-10 averages (baselines).
Per team-game (the lineup layer): for every rostered teammate, a presence average p_j (share of recent team games
played). Today's change vs that baseline is sum_j (present_j - p_j) * expected_j, so a missing star frees his
expected minutes/shots and a returning star takes them back. Expressed as a share of the team's expected output.
On/off layer (play-by-play lineups, 2018-19 on): for every teammate pair, player i's per-minute usage, points,
rebounds, assists and 3PA with j on the floor vs off it (decayed, shrunk to i's overall rate). For today's game,
lu_<stat> = sum_j (p_j - present_j) * share of i's minutes shared with j * log(rate without j / rate with j):
a player-specific estimate of how i's production moves when a given teammate sits.
Per opponent: how much each defense allows of each stat relative to the offense players' own rates, overall and
by position group (G/F/C).

2026-10-09 audit (P1): teammate presence is now what was knowable BEFORE the game. The old version used who actually
played, which nobody knows for a questionable player at bet time. A rotation teammate who played the team's previous game
counts as P(plays | played last game); one who missed it as P(plays | missed last game), both estimated walk-forward from
earlier games. For upcoming games (project.py) the presence is the probability J gives (1 = playing, 0 = out, or e.g.
0.6 for a questionable player), so the projection is the mixture mean over 'he plays' / 'he sits'.
Evaluation-only flags: new_abs_min / cont_abs_min = expected minutes of teammates who sat today after playing (new
absence, needs same-day news) or after also missing the last game (continuing absence). AVAIL=actual restores the old way."""
import os, sys, numpy as np, pandas as pd
from collections import defaultdict, deque
md = sys.argv[1]
b = pd.read_parquet(f'{md}/data/player_games.parquet'); b['future'] = False
UP = len(sys.argv) > 3
if UP: b = pd.concat([b, pd.read_parquet(sys.argv[2])], ignore_index=True); b['future'] = b.future.fillna(False).astype(bool)
b['usg'] = b.fga + 0.44 * b.fta + b.tov
b['pra'] = b.pts + b.reb + b.ast
STATS = ['pts', 'reb', 'oreb', 'dreb', 'ast', 'fg3m', 'fg3a', 'fga', 'fta', 'stl', 'blk', 'tov', 'usg', 'pra']
SHARE = ['min', 'usg', 'pts', 'reb', 'ast', 'fg3a']          # teammate-absence shares
NS = len(STATS); IX = {s: i for i, s in enumerate(STATS)}
H_RATE, H_MS, H_ML, H_P, H_DEF, SEASON_CARRY = 25, 4, 12, 10, 30, 0.6
dr, dms, dml, dp, ddf = (0.5 ** (1 / h) for h in (H_RATE, H_MS, H_ML, H_P, H_DEF))
K = 150.0                                                    # prior pseudo-minutes
LU = ['usg', 'pts', 'reb', 'ast', 'fg3a']; NL = len(LU); K_PAIR, D_PAIR = 100.0, 0.5 ** (1 / 46)
lp = pd.read_parquet(f'{md}/data/lineup_pairs.parquet').sort_values('game_id')
lp_vals = np.column_stack([lp.t_with, lp.t_wo] + [lp[f'{s}_with'] for s in LU] + [lp[f'{s}_wo'] for s in LU])
lp_g = lp.game_id.values; lp_i = lp.pid.values; lp_j = lp.mate.values
lp_cut = np.flatnonzero(np.r_[True, lp_g[1:] != lp_g[:-1], True])
lp_index = {lp_g[a]: (a, z) for a, z in zip(lp_cut[:-1], lp_cut[1:])}
pair = {}                                                    # (i, j) -> [values, i's game count at last update]
p = b[b.played & (b.season <= 2017)]
prior = {g: (x[STATS].sum() / x['min'].sum()).values for g, x in p.groupby('pos_grp')}

class P:  # player state
    __slots__ = ('S', 'M', 'ms', 'ml', 'st', 'last', 'season', 'sea', 'seam', 'seag', 'prev', 'l10', 'team', 'grp', 'n')
    def __init__(s, grp):
        s.S = np.zeros(NS); s.M = 0.0; s.ms = s.ml = None; s.st = 0.5; s.last = None; s.season = None
        s.sea = np.zeros(NS); s.seam = 0.0; s.seag = 0; s.prev = None; s.l10 = deque(maxlen=10); s.team = None; s.grp = grp; s.n = 0
    def rate(s): return (s.S + prior[s.grp] * K) / (s.M + K)

ps = {}
pres = defaultdict(dict)               # team -> {pid: presence average}
deff = defaultdict(lambda: [np.zeros(NS), 0.0])   # (team, grp or '') -> [log-ratio sums, weight]
tenv = defaultdict(lambda: [np.nan, np.nan])      # team -> [pts scored ewma, pts allowed ewma]
ACTUAL = os.environ.get('AVAIL') == 'actual'
prevset = {}                                      # team -> players who played the team's previous game
pcount = {True: [93.0, 100.0], False: [30.0, 100.0]}   # played last game? -> [played today, rotation players] (prior + running)
if 'p_play' not in b: b['p_play'] = np.nan
rows = []
arr = {c: b[c].values for c in b.columns}
order = np.lexsort((arr['pid'], arr['team'], arr['game_id'], arr['date']))
# group indices by date, then game
dates = arr['date'][order]
cuts = np.flatnonzero(np.r_[True, dates[1:] != dates[:-1], True])
out = {k: np.full(len(b), np.nan) for k in
       [f'r_{s}' for s in STATS] + [f'sea_{s}' for s in STATS] + [f'l10_{s}' for s in STATS] +
       [f'd_{s}' for s in SHARE] + [f'lu_{s}' for s in LU] + ['d_min_grp'] + [f'od_{s}' for s in ('min', 'reb', 'pts')] +
       [f'def_{s}' for s in STATS] + [f'defg_{s}' for s in STATS] +
       ['ms', 'ml', 'last_min', 'st_rate', 'days_off', 'gp', 'n_hist', 'sea_gp', 'pres_self', 'team_pts_ewma', 'opp_pts_ewma', 'opp_allow_ewma', 'team_allow_ewma',
        'new_abs_min', 'cont_abs_min']}

def state(pid, grp):
    if pid not in ps: ps[pid] = P(grp)
    return ps[pid]

for a, z in zip(cuts[:-1], cuts[1:]):
    idx = order[a:z]
    day = dates[a]
    by_team = defaultdict(list)
    for i in idx: by_team[(arr['game_id'][i], arr['team'][i])].append(i)
    # ---- pass 1: features (pre-game state) ----
    team_info = {}
    for (gid, team), ii in by_team.items():
        played = {arr['pid'][i] for i in ii if arr['played'][i]}
        fut = bool(arr['future'][ii[0]])
        pgiven = {arr['pid'][i]: arr['p_play'][i] for i in ii if arr['p_play'][i] == arr['p_play'][i]}
        roster = pres[team]; last_ = prevset.get(team, set())
        exp = {}; nab = cab = 0.0
        for pid, pj in roster.items():
            s = ps[pid]
            if s.ml is None: continue
            r = s.rate(); e = {'min': s.ml}
            for x in SHARE[1:]: e[x] = s.ml * r[IX[x]]
            if fut: pre = pgiven.get(pid, 1.0 if pid in played else 0.0)
            elif ACTUAL: pre = 1.0 if pid in played else 0.0
            elif pj >= 0.3:
                c_ = pcount[pid in last_]; pre = c_[0] / c_[1]
            else: pre = pj        # fringe player: no pre-game signal, his own baseline
            if not fut and pid not in played and pj >= 0.3:
                if pid in last_: nab += s.ml
                else: cab += s.ml
            exp[pid] = (pj, pre, e, s.grp)
        tot = {x: sum(v[0] * v[2][x] for v in exp.values()) for x in SHARE}
        team_info[(gid, team)] = (exp, tot, nab, cab)
    for (gid, team), ii in by_team.items():
        exp, tot, nab, cab = team_info[(gid, team)]
        opp = arr['opp'][ii[0]]
        oexp, otot, _, _ = team_info.get((gid, opp), ({}, {}, 0, 0))
        odel = {x: (sum((v[1] - v[0]) * v[2][x] for v in oexp.values()) / otot[x]) if otot and otot[x] > 0 else 0.0 for x in ('min', 'reb', 'pts')}
        te, oe = tenv[team], tenv[opp]
        for i in ii:
            pid = arr['pid'][i]; s = state(pid, arr['pos_grp'][i])
            r = s.rate()
            for k, x in enumerate(STATS): out[f'r_{x}'][i] = r[k]
            if s.seag >= 5 and s.seam > 0: sea = s.sea / s.seag
            elif s.prev is not None: sea = s.prev
            else: sea = np.full(NS, np.nan)
            l10 = np.mean(s.l10, axis=0)[:NS] if s.l10 else np.full(NS, np.nan)
            for k, x in enumerate(STATS): out[f'sea_{x}'][i] = sea[k]; out[f'l10_{x}'][i] = l10[k]
            for x in SHARE:
                out[f'd_{x}'][i] = (sum((v[1] - v[0]) * v[2][x] for q, v in exp.items() if q != pid) / tot[x]) if tot[x] > 0 else 0.0
            lu = np.zeros(NL)
            for q, v in exp.items():
                if q == pid or abs(v[1] - v[0]) < 0.05: continue
                pv = pair.get((pid, q))
                if pv is None: continue
                a_ = pv[0] * D_PAIR ** (s.n - pv[1]); tw, to = a_[0], a_[1]
                if tw + to <= 0: continue
                base = (a_[2:2 + NL] + a_[2 + NL:]) / (tw + to)
                lr = np.log((a_[2 + NL:] + base * K_PAIR + 1e-3) / (to + K_PAIR)) - np.log((a_[2:2 + NL] + base * K_PAIR + 1e-3) / (tw + K_PAIR))
                lu += (v[0] - v[1]) * tw / (tw + to) * lr
            for k, x in enumerate(LU): out[f'lu_{x}'][i] = lu[k]
            totg = sum(v[0] * v[2]['min'] for v in exp.values() if v[3] == s.grp)
            out['d_min_grp'][i] = (sum((v[1] - v[0]) * v[2]['min'] for q, v in exp.items() if q != pid and v[3] == s.grp) / totg) if totg > 0 else 0.0
            for x in ('min', 'reb', 'pts'): out[f'od_{x}'][i] = odel[x]
            dd = deff[(opp, '')]; dg = deff[(opp, s.grp)]
            out_d = np.exp(dd[0] / (dd[1] + 5.0)); out_g = np.exp(dg[0] / (dg[1] + 5.0))
            for k, x in enumerate(STATS): out[f'def_{x}'][i] = out_d[k]; out[f'defg_{x}'][i] = out_g[k]
            out['ms'][i] = s.ms if s.ms is not None else np.nan; out['ml'][i] = s.ml if s.ml is not None else np.nan
            out['last_min'][i] = s.l10[-1][-1] if s.l10 else np.nan
            out['st_rate'][i] = s.st
            out['days_off'][i] = (day - s.last) / np.timedelta64(1, 'D') if s.last is not None else np.nan
            out['n_hist'][i] = s.n; out['sea_gp'][i] = s.seag
            out['pres_self'][i] = pres[team].get(pid, 0.0)
            out['new_abs_min'][i] = nab; out['cont_abs_min'][i] = cab
            out['team_pts_ewma'][i], out['team_allow_ewma'][i] = te; out['opp_pts_ewma'][i], out['opp_allow_ewma'][i] = oe
    # ---- pass 2: update states ----
    for (gid, team) in by_team:
        if gid not in lp_index: continue
        a_, z_ = lp_index.pop(gid)
        for r_ in range(a_, z_):
            i_, j_ = lp_i[r_], lp_j[r_]
            n_ = ps[i_].n if i_ in ps else 0
            pv = pair.get((i_, j_))
            if pv is None: pair[(i_, j_)] = [lp_vals[r_].copy(), n_]
            else: pv[0] = pv[0] * D_PAIR ** (n_ - pv[1]) + lp_vals[r_]; pv[1] = n_
    for (gid, team), ii in by_team.items():
        opp = arr['opp'][ii[0]]
        exp_sum = np.zeros(NS); act = np.zeros(NS); grp_e = defaultdict(lambda: np.zeros(NS)); grp_a = defaultdict(lambda: np.zeros(NS))
        played_now = set()
        if arr['future'][ii[0]]: continue
        for i in ii:
            if not arr['played'][i]: continue
            pid = arr['pid'][i]; s = ps[pid]; m = arr['min'][i]
            y = np.array([arr[x][i] for x in STATS])
            y = np.nan_to_num(y)
            e = s.rate() * m
            exp_sum += e; act += y; grp_e[s.grp] += e; grp_a[s.grp] += y
            played_now.add(pid)
            season = arr['season'][i]
            if s.season is not None and season != s.season:
                s.S *= SEASON_CARRY; s.M *= SEASON_CARRY
                s.prev = s.sea / s.seag if s.seag > 0 else s.prev
                s.sea = np.zeros(NS); s.seam = 0.0; s.seag = 0
            s.season = season
            s.S = dr * s.S + y; s.M = dr * s.M + m
            s.ms = m if s.ms is None else dms * s.ms + (1 - dms) * m
            s.ml = m if s.ml is None else dml * s.ml + (1 - dml) * m
            s.st = dml * s.st + (1 - dml) * float(arr['starter'][i])
            s.sea += y; s.seam += m; s.seag += 1
            s.l10.append(np.r_[y, m])
            s.last = day; s.n += 1
            if s.team is not None and s.team != team: pres[s.team].pop(pid, None)
            s.team = team
            pres[team].setdefault(pid, 0.0)
        # running P(plays) for rotation players (presence 0.3+), by whether they played the team's previous game
        last_ = prevset.get(team, set())
        for pid, pj in pres[team].items():
            if pj >= 0.3 and ps[pid].ml is not None:
                c_ = pcount[pid in last_]; c_[0] += float(pid in played_now); c_[1] += 1
        prevset[team] = played_now
        for pid in list(pres[team]):
            pres[team][pid] = dp * pres[team][pid] + (1 - dp) * (1.0 if pid in played_now else 0.0)
            if pres[team][pid] < 0.02 and pid not in played_now: pres[team].pop(pid)
        # defense allowed by `opp` (this team is the offense)
        ok = exp_sum > 0
        lr = np.zeros(NS); lr[ok] = np.log(np.maximum(act[ok], 0.5) / exp_sum[ok])
        dd = deff[(opp, '')]; dd[0] = ddf * dd[0] + lr; dd[1] = ddf * dd[1] + 1
        for g in grp_e:
            ok = grp_e[g] > 0; lr = np.zeros(NS); lr[ok] = np.log(np.maximum(grp_a[g][ok], 0.5) / grp_e[g][ok])
            dg = deff[(opp, g)]; dg[0] = ddf * dg[0] + lr; dg[1] = ddf * dg[1] + 1
        pts_for = arr['team_score'][ii[0]]; pts_against = arr['opponent_team_score'][ii[0]]
        te = tenv[team]
        te[0] = pts_for if np.isnan(te[0]) else 0.9 * te[0] + 0.1 * pts_for
        te[1] = pts_against if np.isnan(te[1]) else 0.9 * te[1] + 0.1 * pts_against

b = pd.concat([b, pd.DataFrame(out, index=b.index)], axis=1)
import minutes_ext; b = minutes_ext.add(b)      # NBA-2 minutes features (return-from-absence ramp, trend, rotation depth)
if UP:
    b[b.future].to_parquet(sys.argv[3]); print('upcoming rows', int(b.future.sum())); sys.exit()
b.to_parquet(f'{md}/data/features.parquet')
print('features', b.shape, '| P(plays) for a rotation player who played the last game %.3f, who missed it %.3f' % (pcount[True][0] / pcount[True][1], pcount[False][0] / pcount[False][1]))
print(b[b.played & (b.season >= 2018)][['d_min', 'd_usg', 'd_pts', 'od_min', 'def_pts', 'defg_reb', 'ml', 'st_rate']].describe().round(3).to_string())
