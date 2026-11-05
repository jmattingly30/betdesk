"""ESPN play-by-play -> on-court lineups -> per game, for every pair of teammates (i, j): seconds i played with j on
the floor and without j, and i's usage/points/rebounds/assists/3PA in each state.
usage: python3 lineups.py <download_dir> <model_dir> [first_season last_season]
download_dir holds play_by_play_YYYY.parquet saved as pbp_YYYY.parquet (sportsdataverse-data release espn_nba_pbp).
Period starters come from the box score (Q1) or from who is involved before being subbed in (later periods)."""
import sys, numpy as np, pandas as pd
from collections import defaultdict
d, md = sys.argv[1], sys.argv[2]
seasons = range(int(sys.argv[3]), int(sys.argv[4]) + 1) if len(sys.argv) > 4 else range(2019, 2027)
box = pd.read_parquet(f'{md}/data/player_games.parquet', columns=['game_id', 'pid', 'team', 'starter', 'played', 'min', 'season'])
box = box[box.played]
ST = ['usg', 'pts', 'reb', 'ast', 'fg3a']
out, check = [], []
for season in seasons:
    p = pd.read_parquet(f'{d}/pbp_{season}.parquet', columns=['game_id', 'game_play_number', 'sequence_number', 'type_text', 'text', 'period_number',
                        'start_game_seconds_remaining', 'end_game_seconds_remaining', 'scoring_play', 'score_value', 'shooting_play',
                        'athlete_id_1', 'athlete_id_2', 'athlete_id_3', 'season_type'])
    p = p[p.season_type.isin([2, 3])]
    p['game_id'] = p.game_id.astype(int)
    p['seq'] = pd.to_numeric(p.sequence_number, errors='coerce')
    # sequence numbers are reassigned when ESPN corrects a play, so order by period, game clock, then sequence
    p = p.sort_values(['game_id', 'period_number', 'start_game_seconds_remaining', 'seq'], ascending=[True, True, False, True])
    bs = box[box.season == season]
    team_of = {(g, a): t for g, a, t in zip(bs.game_id, bs.pid, bs.team)}
    starters = bs[bs.starter.astype(bool)].groupby(['game_id', 'team']).pid.apply(list).to_dict()
    for gid, g in p.groupby('game_id', sort=False):
        tt = bs[bs.game_id == gid]
        if tt.empty: continue
        teams = tt.team.unique().tolist()
        if len(teams) != 2 or any(len(starters.get((gid, t), [])) != 5 for t in teams): continue
        typ = g.type_text.fillna('').values; txt = g.text.fillna('').values
        per = g.period_number.values.astype(int)
        t_end = g.end_game_seconds_remaining.values.astype(float); t0 = g.start_game_seconds_remaining.values.astype(float)
        a1 = g.athlete_id_1.values; a2 = g.athlete_id_2.values; a3 = g.athlete_id_3.values
        sv = pd.to_numeric(g.score_value, errors='coerce').fillna(0).values; sc = g.scoring_play.fillna(False).values.astype(bool)
        sh = g.shooting_play.fillna(False).values.astype(bool)
        n = len(g)
        def aid(x): return int(x) if x == x and x is not None else None
        is_sub = np.array(['Substitution' in s for s in typ])
        # who is on court at the start of each period
        on = {t: set(starters[(gid, t)]) for t in teams}
        acc = defaultdict(lambda: np.zeros(2 + 2 * len(ST)))   # (i, j) -> [t_with, t_wo, stats_with..., stats_wo...]
        solo_t = defaultdict(float)
        cur_per = 1; clock = {'last': t0[0] if n else 2880}
        pstats = defaultdict(lambda: np.zeros(len(ST)))         # stats accrued in current stint
        def flush(now):
            dt = max(0.0, clock['last'] - now)
            for t in teams:
                L = on[t]; mates = [q for q in roster[t]]
                for i in L:
                    s = pstats.pop(i, np.zeros(len(ST)))
                    solo_t[i] += dt
                    for j in mates:
                        if j == i: continue
                        v = acc[(i, j)]
                        if j in L: v[0] += dt; v[2:2 + len(ST)] += s
                        else: v[1] += dt; v[2 + len(ST):] += s
            clock['last'] = now
        roster = {t: tt[tt.team == t].pid.tolist() for t in teams}
        def period_start(k, start_idx):
            # starters of period k: involved before being subbed in; fill from previous lineup
            found = {t: [] for t in teams}; entered = set()
            for m in range(start_idx, n):
                if per[m] != k: break
                ids = [aid(a1[m]), aid(a2[m])] if is_sub[m] else [aid(a1[m]), aid(a2[m]), aid(a3[m])]
                if is_sub[m]:
                    inn, outp = aid(a1[m]), aid(a2[m])
                    if outp is not None and outp not in entered:
                        t = team_of.get((gid, outp))
                        if t and outp not in found[t]: found[t].append(outp)
                    if inn is not None: entered.add(inn)
                    continue
                for q in ids:
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
        for m in range(n):
            if per[m] != cur_per:
                flush(t0[m]); cur_per = per[m]; period_start(cur_per, m); clock['last'] = t0[m]
            now = t0[m]
            if is_sub[m]:
                flush(now)
                inn, outp = aid(a1[m]), aid(a2[m])
                t = team_of.get((gid, inn)) or team_of.get((gid, outp))
                if t:
                    on[t].discard(outp)
                    if inn is not None: on[t].add(inn)
                continue
            # credit event stats to the actor(s) currently in pstats (time is credited at the next flush)
            x1 = aid(a1[m]); s = typ[m]
            if x1 is None: continue
            if 'Free Throw' in s:
                pstats[x1][0] += 0.44; pstats[x1][1] += sv[m] if sc[m] else 0
            elif sh[m]:
                pstats[x1][0] += 1
                if sc[m]: pstats[x1][1] += sv[m]
                if 'three point' in txt[m]: pstats[x1][4] += 1
                if sc[m] and 'assists' in txt[m]:
                    x2 = aid(a2[m])
                    if x2 is not None: pstats[x2][3] += 1
            elif 'Rebound' in s:
                pstats[x1][2] += 1
            elif 'Turnover' in s:
                pstats[x1][0] += 1
        flush(t_end[-1] if n else 0)
        for (i, j), v in acc.items():
            out.append((gid, i, j, *v))
        for i, sec in solo_t.items(): check.append((gid, i, sec / 60))
    print(season, 'pairs so far', len(out), flush=True)
cols = ['game_id', 'pid', 'mate', 't_with', 't_wo'] + [f'{s}_with' for s in ST] + [f'{s}_wo' for s in ST]
pr = pd.DataFrame(out, columns=cols)
pr[['t_with', 't_wo']] /= 60
for c in cols[1:3]: pr[c] = pr[c].astype(np.int64)
pr.to_parquet(f'{md}/data/lineup_pairs.parquet')
ck = pd.DataFrame(check, columns=['game_id', 'pid', 'pbp_min']).merge(box[['game_id', 'pid', 'min']], on=['game_id', 'pid'])
print('pairs', pr.shape, '| pbp minutes vs box minutes: MAE %.2f, corr %.4f, games %d' % ((ck.pbp_min - ck['min']).abs().mean(), np.corrcoef(ck.pbp_min, ck['min'])[0, 1], ck.game_id.nunique()))
