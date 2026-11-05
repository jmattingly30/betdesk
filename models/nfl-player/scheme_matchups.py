"""NFL scheme matchups (J's follow-up, 2026-10-10): does how a player does against a TYPE of defense, weighted by how
often tonight's defense plays that type, improve the prop model?

python3 scheme_matchups.py <dl_dir> <model_dir>
  dl_dir: nflverse play_by_play_YYYY (pbp_YYYY.parquet), pbp_participation (part_YYYY.parquet), ftn_charting (ftn_YYYY.parquet)
  model_dir: models/nfl-player (uses data/backtest_predictions.parquet, the 2024-25 out-of-sample projections)

Scheme dimensions (participation = NGS/FTN charting, 2023-2025; FTN alone has 2026):
  passing  man vs zone; one-high (cover 0/1/3) vs two-high (cover 2/4/6/9, 2-man) safeties; base (<=4 DBs) / nickel (5) /
           dime (6+); blitz (FTN: 1+ blitzers) vs no blitz
  rushing  light (<=6), even (7) or stacked (8+) box; base / nickel / dime
For each player-game in 2024-25, using only earlier plays (from 2023 on):
  player split   yards per target (or carry) against each look, shrunk to his overall rate (k = 40 targets / 50 carries)
  defense mix    how often tonight's defense used each look in its earlier games (decayed, prior season x 0.5, shrunk to league)
  matchup        M = sum over looks of mix x split / sum over looks of league mix x split   (1.00 = no edge either way)
Tests: (1) are the splits a real, repeatable skill (2023 vs 2024 split, players with 30+ targets on both sides)?
(2) does log(M) add to the model's projection (Poisson, offset = projection; fit 2024, judged on 2025)?
Writes scheme_matchups_log.txt."""
import sys, numpy as np, pandas as pd, statsmodels.api as sm
dl, M = sys.argv[1], sys.argv[2]
log = open(f'{M}/scheme_matchups_log.txt', 'w')
def say(*a): s = ' '.join(str(x) for x in a); print(s, flush=True); log.write(s + '\n'); log.flush()

cols = ['game_id', 'play_id', 'season', 'week', 'game_date', 'posteam', 'defteam', 'pass_attempt', 'rush_attempt', 'qb_scramble', 'qb_kneel',
        'receiver_player_id', 'rusher_player_id', 'receiving_yards', 'rushing_yards', 'sack', 'two_point_attempt', 'season_type']
P = pd.concat([pd.read_parquet(f'{dl}/pbp_{y}.parquet', columns=cols) for y in (2023, 2024, 2025)], ignore_index=True)
Q = pd.concat([pd.read_parquet(f'{dl}/part_{y}.parquet', columns=['nflverse_game_id', 'play_id', 'defense_man_zone_type', 'defense_coverage_type',
                                                                     'defense_personnel', 'defenders_in_box']) for y in (2023, 2024, 2025)], ignore_index=True)
F = pd.concat([pd.read_parquet(f'{dl}/ftn_{y}.parquet', columns=['nflverse_game_id', 'nflverse_play_id', 'n_blitzers']) for y in (2023, 2024, 2025)], ignore_index=True)
P = P.merge(Q.rename(columns={'nflverse_game_id': 'game_id'}), on=['game_id', 'play_id'], how='left')
P = P.merge(F.rename(columns={'nflverse_game_id': 'game_id', 'nflverse_play_id': 'play_id'}), on=['game_id', 'play_id'], how='left')
P['game_date'] = pd.to_datetime(P.game_date)

def n_db(s):
    if not isinstance(s, str) or not s: return np.nan
    n = 0
    for part in s.split(','):
        k, pos = part.strip().split(' ', 1)
        if pos in ('CB', 'FS', 'SS', 'DB', 'S'): n += int(k)
    return n
P['dbs'] = P.defense_personnel.map(n_db)
P['look_db'] = np.select([P.dbs <= 4, P.dbs == 5, P.dbs >= 6], ['base', 'nickel', 'dime'], None)
P['look_mz'] = P.defense_man_zone_type.map({'MAN_COVERAGE': 'man', 'ZONE_COVERAGE': 'zone'})
P['look_shell'] = P.defense_coverage_type.map({'COVER_0': 'one', 'COVER_1': 'one', 'COVER_3': 'one', 'COVER_2': 'two', 'COVER_4': 'two',
                                               'COVER_6': 'two', 'COVER_9': 'two', '2_MAN': 'two'})
P['look_blitz'] = np.where(P.n_blitzers.isna(), None, np.where(P.n_blitzers > 0, 'blitz', 'none'))
P['look_box'] = np.select([P.defenders_in_box <= 6, P.defenders_in_box == 7, P.defenders_in_box >= 8], ['light', 'even', 'stacked'], None)

tg = P[(P.pass_attempt == 1) & P.receiver_player_id.notna() & (P.sack != 1) & (P.two_point_attempt != 1)].copy()
tg['pid'] = tg.receiver_player_id; tg['y'] = tg.receiving_yards.fillna(0)
ru = P[(P.rush_attempt == 1) & P.rusher_player_id.notna() & (P.qb_scramble != 1) & (P.qb_kneel != 1) & (P.two_point_attempt != 1)].copy()
ru['pid'] = ru.rusher_player_id; ru['y'] = ru.rushing_yards.fillna(0)
DIMS = {'rec': (tg, ['look_mz', 'look_shell', 'look_db', 'look_blitz'], 40), 'rush': (ru, ['look_box', 'look_db'], 50)}

# ---------- (1) reliability of player splits: 2023 split vs 2024 split ----------
def split_table(d, dim, season):
    x = d[(d.season == season) & d[dim].notna()]
    g = x.groupby(['pid', dim]).y.agg(['sum', 'size']).unstack(dim)
    return g
say('(1) Is a player\'s split against a look a repeatable skill? Correlation of (yards per play vs look A - vs look B), 2023 vs 2024, 30+ plays each side')
for kind, (d, dims, _) in DIMS.items():
    for dim in dims:
        looks = sorted(d[dim].dropna().unique())
        a, b = split_table(d, dim, 2023), split_table(d, dim, 2024)
        for i in range(len(looks)):
            for j in range(i + 1, len(looks)):
                A_, B_ = looks[i], looks[j]
                def gap(t):
                    ok = (t[('size', A_)] >= 30) & (t[('size', B_)] >= 30)
                    return (t[('sum', A_)] / t[('size', A_)] - t[('sum', B_)] / t[('size', B_)])[ok]
                g1, g2 = gap(a), gap(b); common = g1.index.intersection(g2.index)
                if len(common) >= 20:
                    r = np.corrcoef(g1[common], g2[common])[0, 1]
                    say(f'   {kind:4s} {dim[5:]:5s} {A_} vs {B_}: r = {r:+.3f} (n {len(common)} players)')

# ---------- (2) walk-forward matchup multipliers ----------
games = P[['game_id', 'game_date', 'season', 'week']].drop_duplicates('game_id').sort_values('game_date')
def matchup(kind):
    d, dims, k = DIMS[kind]
    d = d.sort_values(['game_date', 'game_id'])
    out = []
    lg_all = {dim: d[dim].value_counts(normalize=True) for dim in dims}
    gids = games[games.season >= 2024].game_id.tolist()
    # cumulative player sums and decayed defense counts, advanced game by game
    by_game = {g: x for g, x in d.groupby('game_id')}
    psum = {}      # (pid, dim, look) -> [yds, n];  (pid, 'all') -> [yds, n]
    dmix = {}      # (team, dim, look) -> decayed count
    last_season = {}
    for g in games.game_id:
        x = by_game.get(g)
        s = games.loc[games.game_id == g, 'season'].iloc[0]
        if x is None: continue
        if g in gids:
            # pre-game multipliers for every player in this game
            for (pid, team, opp), _ in x.groupby(['pid', 'posteam', 'defteam']):
                tot = psum.get((pid, 'all', ''))
                if not tot or tot[1] < 10: continue
                base = tot[0] / tot[1]
                row = dict(game_id=g, pid=pid)
                for dim in dims:
                    looks = lg_all[dim].index
                    num = den = 0.0
                    mix_n = sum(dmix.get((opp, dim, lk), 0.0) for lk in looks)
                    for lk in looks:
                        ps = psum.get((pid, dim, lk), [0.0, 0])
                        split = ((ps[0] + k * base) / (ps[1] + k)) / max(base, 0.1)
                        f_d = (dmix.get((opp, dim, lk), 0.0) + 150 * lg_all[dim][lk]) / (mix_n + 150)
                        num += f_d * split; den += lg_all[dim][lk] * split
                    row[f'M_{dim[5:]}'] = num / den
                out.append(row)
        # advance state with this game's plays
        for r in x.itertuples():
            t = psum.setdefault((r.pid, 'all', ''), [0.0, 0]); t[0] += r.y; t[1] += 1
            for dim in dims:
                lk = getattr(r, dim)
                if lk is None or (isinstance(lk, float) and np.isnan(lk)): continue
                t = psum.setdefault((r.pid, dim, lk), [0.0, 0]); t[0] += r.y; t[1] += 1
        for team in x.defteam.unique():
            if last_season.get(team) not in (None, s):
                for key in [k_ for k_ in dmix if k_[0] == team]: dmix[key] *= 0.5
            last_season[team] = s
            for key in [k_ for k_ in dmix if k_[0] == team]: dmix[key] *= 0.5 ** (1 / 8)
            xt = x[x.defteam == team]
            for dim in dims:
                for lk, c in xt[dim].value_counts().items():
                    dmix[(team, dim, lk)] = dmix.get((team, dim, lk), 0.0) + c
    return pd.DataFrame(out)

Mrec, Mrush = matchup('rec'), matchup('rush')
B = pd.read_parquet(f'{M}/data/backtest_predictions.parquet')
rel = (B.snap_l >= 0.25) & (B.games_prior >= 3)
say('\n(2) Does the matchup multiplier add to the projection? Poisson GLM, offset = model projection; fit 2024, judged on 2025')
for kind, Mdf, stat, pos in [('rec', Mrec, 'rec_yds', ['WR', 'TE', 'RB']), ('rush', Mrush, 'rush_yds', ['RB'])]:
    d = B[rel & B.position.isin(pos)].merge(Mdf, on=['game_id', 'pid'], how='inner')
    d = d[d[f'{stat}_hat'] > 1]
    mcols = [c for c in Mdf.columns if c.startswith('M_')]
    say(f'  {stat}: {len(d)} player-games with a multiplier; spread of M: ' + ', '.join(f'{c[2:]} 10-90% {d[c].quantile(.1):.3f}-{d[c].quantile(.9):.3f}' for c in mcols))
    for use in [[c] for c in mcols] + [mcols]:
        X = np.log(d[use].clip(0.5, 2.0)); tr, te = d.season == 2024, d.season == 2025
        g = sm.GLM(d[stat][tr], X[tr], family=sm.families.Poisson(), offset=np.log(d[f'{stat}_hat'][tr])).fit(cov_type='HC1')
        pred = d[f'{stat}_hat'][te] * np.exp(X[te] @ g.params)
        mae0 = (d[stat][te] - d[f'{stat}_hat'][te]).abs().mean(); mae1 = (d[stat][te] - pred).abs().mean()
        gp = sm.GLM(d[stat], X, family=sm.families.Poisson(), offset=np.log(d[f'{stat}_hat'])).fit(cov_type='HC1')
        say(f'    {"+".join(c[2:] for c in use):22s} coef ' + ', '.join(f'{c[2:]} {gp.params[c]:+.2f} (p {gp.pvalues[c]:.3f})' for c in use) +
            f' | 2025 MAE {mae0:.3f} -> {mae1:.3f} ({100 * (1 - mae1 / mae0):+.2f}%)')
