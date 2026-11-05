"""NFL scheme -> target share (J's follow-up, 2026-10-10): does the defense's coverage mix change WHO gets targeted, and
does an offense's (coordinator's) record against that coverage change how well its receivers do?

python3 scheme_share.py <dl_dir> <model_dir>     (same inputs as scheme_matchups.py)

Per pass play (2023-2025) we know the coverage look (man/zone, one-/two-high, base/nickel/dime, blitz) and, from the
participation file, every offensive player on the field. So for each receiver and look:
  targets per route   TPR(look) = targets / pass plays he was on the field for, shrunk to his overall rate (k = 60 plays)
For each offense and look:
  offense yards per pass play against that look, shrunk to the offense's overall rate (k = 120 plays)
For tonight's game (all inputs from earlier games; this season x1, earlier seasons x0.5), with tonight's defense mix f(look):
  M_share  = sum f x TPR / sum league-mix x TPR       (does tonight's coverage mix favour this receiver's targets?)
  M_oc     = sum f x offense ypp / sum league-mix x offense ypp   (does it favour this offense's passing game?)
Tests: year-to-year repeatability of the man-vs-zone TPR gap, and whether log(M) adds to the model's projected targets and
receiving yards (Poisson, offset = projection; fit 2024, judged on 2025). Writes scheme_share_log.txt."""
import sys, numpy as np, pandas as pd, statsmodels.api as sm
dl, M = sys.argv[1], sys.argv[2]
log = open(f'{M}/scheme_share_log.txt', 'w')
def say(*a): s = ' '.join(str(x) for x in a); print(s, flush=True); log.write(s + '\n'); log.flush()

cols = ['game_id', 'play_id', 'season', 'game_date', 'posteam', 'defteam', 'pass_attempt', 'qb_dropback', 'sack', 'receiver_player_id',
        'receiving_yards', 'yards_gained', 'two_point_attempt']
P = pd.concat([pd.read_parquet(f'{dl}/pbp_{y}.parquet', columns=cols) for y in (2023, 2024, 2025)], ignore_index=True)
Q = pd.concat([pd.read_parquet(f'{dl}/part_{y}.parquet', columns=['nflverse_game_id', 'play_id', 'defense_man_zone_type', 'defense_coverage_type',
                                                                     'defense_personnel', 'offense_players']) for y in (2023, 2024, 2025)], ignore_index=True)
F = pd.concat([pd.read_parquet(f'{dl}/ftn_{y}.parquet', columns=['nflverse_game_id', 'nflverse_play_id', 'n_blitzers']) for y in (2023, 2024, 2025)], ignore_index=True)
P = P[(P.qb_dropback == 1) & (P.two_point_attempt != 1)]
P = P.merge(Q.rename(columns={'nflverse_game_id': 'game_id'}), on=['game_id', 'play_id'], how='inner')
P = P.merge(F.rename(columns={'nflverse_game_id': 'game_id', 'nflverse_play_id': 'play_id'}), on=['game_id', 'play_id'], how='left')
P['game_date'] = pd.to_datetime(P.game_date)
def n_db(s):
    if not isinstance(s, str) or not s: return np.nan
    return sum(int(p.strip().split(' ', 1)[0]) for p in s.split(',') if p.strip().split(' ', 1)[1] in ('CB', 'FS', 'SS', 'DB', 'S'))
P['dbs'] = P.defense_personnel.map(n_db)
LOOK = {'mz': P.defense_man_zone_type.map({'MAN_COVERAGE': 'man', 'ZONE_COVERAGE': 'zone'}),
        'shell': P.defense_coverage_type.map({'COVER_0': 'one', 'COVER_1': 'one', 'COVER_3': 'one', 'COVER_2': 'two', 'COVER_4': 'two', 'COVER_6': 'two', 'COVER_9': 'two', '2_MAN': 'two'}),
        'db': pd.Series(np.select([P.dbs <= 4, P.dbs == 5, P.dbs >= 6], ['base', 'nickel', 'dime'], None), index=P.index),
        'blitz': pd.Series(np.where(P.n_blitzers.isna(), None, np.where(P.n_blitzers > 0, 'blitz', 'none')), index=P.index)}
for k, v in LOOK.items(): P[k] = v
P['ypp'] = P.yards_gained.fillna(0)
games = P[['game_id', 'game_date', 'season']].drop_duplicates('game_id').sort_values('game_date').reset_index(drop=True)
gord = {g: i for i, g in enumerate(games.game_id)}; P['gi'] = P.game_id.map(gord)

# on-field rows: one per (play, offensive player)
on = P[['game_id', 'gi', 'season', 'play_id', 'posteam', 'receiver_player_id', 'offense_players'] + list(LOOK)].copy()
on['pid'] = on.offense_players.str.split(';'); on = on.explode('pid'); on = on[on.pid.notna() & (on.pid != '')]
on['tgt'] = (on.pid == on.receiver_player_id).astype(int)

def prior_weighted(tab, key):
    """tab: rows (key..., gi, season, values) per game; returns pre-game weighted sums (this season x1, earlier x0.5)."""
    tab = tab.sort_values('gi'); v = [c for c in tab.columns if c not in key + ['gi', 'season']]
    allc = tab.groupby(key)[v].cumsum() - tab[v]
    seas = tab.groupby(key + ['season'])[v].cumsum() - tab[v]
    out = tab[key + ['gi', 'season']].copy(); out[v] = 0.5 * allc + 0.5 * seas
    return out

say('Repeatability: man-vs-zone targets-per-route gap, 2023 vs 2024 (players with 80+ routes vs each)')
for dim in LOOK:
    looks = sorted(P[dim].dropna().unique())
    if len(looks) != 2 and dim != 'db': continue
    A_, B_ = (looks[0], looks[1]) if dim != 'db' else ('base', 'nickel')
    def gap(season):
        x = on[(on.season == season) & on[dim].isin([A_, B_])].groupby(['pid', dim]).tgt.agg(['sum', 'size']).unstack(dim)
        ok = (x[('size', A_)] >= 80) & (x[('size', B_)] >= 80)
        return (x[('sum', A_)] / x[('size', A_)] - x[('sum', B_)] / x[('size', B_)])[ok]
    g1, g2 = gap(2023), gap(2024); c = g1.index.intersection(g2.index)
    say(f'   {dim:5s} {A_} vs {B_}: r = {np.corrcoef(g1[c], g2[c])[0, 1]:+.3f} (n {len(c)} players)')
    # offense (coordinator) version: yards per dropback vs A minus vs B, by team
    def tgap(season):
        x = P[(P.season == season) & P[dim].isin([A_, B_])].groupby(['posteam', dim]).ypp.mean().unstack(dim)
        return x[A_] - x[B_]
    t1, t2 = tgap(2023), tgap(2024); c = t1.index.intersection(t2.index)
    say(f'         offense ypp {A_} minus {B_}: r = {np.corrcoef(t1[c], t2[c])[0, 1]:+.3f} (32 teams)')

rows = []
lgmix = {dim: P[dim].value_counts(normalize=True) for dim in LOOK}
# player look counts per game, offense look yards per game, defense look counts per game
B = pd.read_parquet(f'{M}/data/backtest_predictions.parquet')
B = B[(B.snap_l >= 0.25) & (B.games_prior >= 3) & B.position.isin(['WR', 'TE', 'RB'])].copy()
B['gi'] = B.game_id.map(gord); B = B[B.gi.notna()]
B = B.merge(games[['game_id']], on='game_id')
for dim in LOOK:
    looks = list(lgmix[dim].index)
    pl = on[on[dim].notna()].assign(lk=lambda d: d[dim]).pivot_table(index=['pid', 'gi', 'season'], columns='lk', values='tgt', aggfunc=['sum', 'size'], fill_value=0)
    pl.columns = [f'{a}_{b}' for a, b in pl.columns]; pl = pl.reset_index()
    for lk in looks:
        for c in (f'sum_{lk}', f'size_{lk}'):
            if c not in pl: pl[c] = 0
    pw = prior_weighted(pl, ['pid'])
    pw['all_t'] = sum(pw[f'sum_{lk}'] for lk in looks); pw['all_n'] = sum(pw[f'size_{lk}'] for lk in looks)
    of = P[P[dim].notna()].assign(lk=lambda d: d[dim]).pivot_table(index=['posteam', 'gi', 'season'], columns='lk', values='ypp', aggfunc=['sum', 'size'], fill_value=0)
    of.columns = [f'{a}_{b}' for a, b in of.columns]; ow = prior_weighted(of.reset_index(), ['posteam'])
    ow['all_y'] = sum(ow[f'sum_{lk}'] for lk in looks); ow['all_n'] = sum(ow[f'size_{lk}'] for lk in looks)
    df_ = P[P[dim].notna()].assign(lk=lambda d: d[dim]).pivot_table(index=['defteam', 'gi', 'season'], columns='lk', values='ypp', aggfunc='size', fill_value=0)
    dw = prior_weighted(df_.reset_index(), ['defteam'])
    # attach: player's latest pre-game state = his row for this game if he played it (pre-game sums), via merge on (pid, gi)
    x = B[['game_id', 'gi', 'pid', 'posteam', 'opp']].merge(pw, on=['pid', 'gi'], how='left')
    x = x.merge(ow.rename(columns={'posteam': 'posteam', 'all_n': 'o_all_n', **{f'sum_{lk}': f'osum_{lk}' for lk in looks}, **{f'size_{lk}': f'osize_{lk}' for lk in looks}}).drop(columns=['season']), on=['posteam', 'gi'], how='left')
    x = x.merge(dw.rename(columns={'defteam': 'opp', **{lk: f'd_{lk}' for lk in looks}}).drop(columns=['season']), on=['opp', 'gi'], how='left')
    base_t = (x.all_t / x.all_n.clip(lower=1))
    dn = sum(x[f'd_{lk}'].fillna(0) for lk in looks)
    ms, mo_num, mo_den = 0, 0, 0; ms_den = 0
    base_o = x.all_y / x.o_all_n.clip(lower=1)
    for lk in looks:
        tpr = (x[f'sum_{lk}'] + 60 * base_t) / (x[f'size_{lk}'] + 60)
        oy = (x[f'osum_{lk}'] + 120 * base_o) / (x[f'osize_{lk}'] + 120)
        f = (x[f'd_{lk}'].fillna(0) + 150 * lgmix[dim][lk]) / (dn + 150)
        ms = ms + f * tpr; ms_den = ms_den + lgmix[dim][lk] * tpr
        mo_num = mo_num + f * oy; mo_den = mo_den + lgmix[dim][lk] * oy
    B[f'Ms_{dim}'] = (ms / ms_den).where(x.all_n.to_numpy() >= 60).to_numpy()
    B[f'Mo_{dim}'] = (mo_num / mo_den).where(x.o_all_n.to_numpy() >= 100).to_numpy()

say('\nDoes it add to the projection? Poisson GLM, offset = model projection; coefficient 1 = the multiplier is right as is')
d = B.dropna(subset=[c for c in B.columns if c.startswith('Ms_') or c.startswith('Mo_')])
say(f'{len(d)} WR/TE/RB player-games 2024-25. Spread (10-90%): ' + ', '.join(f'{c} {d[c].quantile(.1):.3f}-{d[c].quantile(.9):.3f}' for c in d.columns if c[:3] in ('Ms_', 'Mo_')))
tr, te = d.season == 2024, d.season == 2025
for stat in ['targets', 'rec_yds']:
    d = d[d[f'{stat}_hat'] > 0.3]; tr, te = d.season == 2024, d.season == 2025
    sets = [[f'Ms_{k}'] for k in LOOK] + [[f'Mo_{k}'] for k in LOOK] + [[f'Ms_{k}' for k in LOOK] + [f'Mo_{k}' for k in LOOK]]
    for use in sets:
        X = np.log(d[use].clip(0.5, 2.0))
        g = sm.GLM(d[stat][tr], X[tr], family=sm.families.Poisson(), offset=np.log(d[f'{stat}_hat'][tr])).fit()
        pred = d[f'{stat}_hat'][te] * np.exp(X[te] @ g.params)
        mae0 = (d[stat][te] - d[f'{stat}_hat'][te]).abs().mean(); mae1 = (d[stat][te] - pred).abs().mean()
        gp = sm.GLM(d[stat], X, family=sm.families.Poisson(), offset=np.log(d[f'{stat}_hat'])).fit(cov_type='HC1')
        say(f'  {stat:8s} {"+".join(use) if len(use) < 3 else "all eight":22s} ' + ', '.join(f'{c} {gp.params[c]:+.2f} (p {gp.pvalues[c]:.3f})' for c in use[:2]) +
            f'{" ..." if len(use) > 2 else ""} | 2025 MAE {mae0:.3f} -> {mae1:.3f} ({100 * (1 - mae1 / mae0):+.2f}%)')
