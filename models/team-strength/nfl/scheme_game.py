"""J 2026-10-11: game-level (scores, not props) scheme matchups. Example: Jaguars vs Eagles; the Eagles play nickel more than
base, the Jaguars' offense gains more EPA/play against nickel -> does that move the projected score?

For every offense: its EPA/play against each defensive look, minus its own average (defence quality removed play by play),
shrunk toward 0. For every defence: how often it shows each look, given the offence's personnel. The matchup term for a
game = sum over looks of (how much MORE this defence will show the look than what the offence usually faces) x (the
offence's split against that look). The defence side is the mirror: the defence's split against offensive personnel
(11 / 12 / heavy) x how much more of it this offence will use. Families (2016+ unless noted):
  db     base (<=4 DBs) / nickel (5) / dime (6+), all snaps     cov  man / zone, dropbacks (2018+)
  shell  one-high (C0/C1/C3) / two-high (C2/C4/C6/2-man) (2018+) prush 4-or-fewer / 5+ rushers, dropbacks
  box    light (<=6) / 7 / heavy (8+) defenders in box, runs     pers (defence side) 11 / 12 / heavy / other
Parts: (1) does an offence's split repeat (odd vs even weeks; next season, same play-caller vs new)? (2) does a defence's
look mix repeat? (3) walk-forward game test 2018-2025: every term is built from earlier games only (prior season at half
weight, a quarter if the play-caller changed; current season to date at full weight) and tested against the CLOSING line.
usage: python3 nfl/scheme_game.py <dl dir with pbp_YYYY, part_YYYY> [play_callers.csv]   -> nfl/scheme_game_log.txt"""
import sys, os, numpy as np, pandas as pd, statsmodels.api as sm
H = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
dl = sys.argv[1]
pc = sys.argv[2] if len(sys.argv) > 2 else os.path.normpath(f'{H}/../nfl-player/data/play_callers.csv')
YRS = range(2016, 2026)

def cnt(s, p):
    if not isinstance(s, str): return np.nan
    for part in s.split(','):
        part = part.strip()
        if ' ' in part:
            k, q = part.split(' ', 1)
            if k.isdigit() and q == p: return int(k)
    return 0

def load(y):
    P = pd.read_parquet(f'{dl}/pbp_{y}.parquet', columns=['game_id', 'play_id', 'season', 'season_type', 'week', 'posteam', 'defteam', 'epa', 'pass',
                                                         'rush', 'qb_dropback', 'wp', 'two_point_attempt', 'qb_kneel', 'qb_spike'])
    P = P[((P['pass'] == 1) | (P.rush == 1)) & P.epa.notna() & P.posteam.notna() & (P.two_point_attempt != 1) & (P.qb_kneel != 1) & (P.qb_spike != 1)]
    Q = pd.read_parquet(f'{dl}/part_{y}.parquet', columns=['nflverse_game_id', 'play_id', 'offense_personnel', 'defense_personnel', 'defenders_in_box',
                                                         'number_of_pass_rushers', 'defense_man_zone_type', 'defense_coverage_type'])
    P = P.merge(Q.rename(columns={'nflverse_game_id': 'game_id'}), on=['game_id', 'play_id'], how='inner')
    P = P[P.offense_personnel.notna() & P.defense_personnel.notna()]
    rb, te, wr = (P.offense_personnel.map(lambda s: cnt(s, k)) for k in ('RB', 'TE', 'WR'))
    P['pers'] = np.select([(rb == 1) & (te == 1) & (wr == 3), (rb == 1) & (te == 2), (rb + te) >= 3], ['p11', 'p12', 'heavy'], 'other')
    nd = P.defense_personnel.map(lambda s: sum(cnt(s, k) for k in ('CB', 'FS', 'SS', 'DB', 'S')))
    P['db'] = np.select([nd <= 4, nd == 5], ['base', 'nickel'], 'dime')
    dbk = P.qb_dropback == 1
    mz = P.defense_man_zone_type.fillna('')
    P['cov'] = np.where(dbk & (mz == 'MAN_COVERAGE'), 'man', np.where(dbk & (mz == 'ZONE_COVERAGE'), 'zone', None))
    ct = P.defense_coverage_type.fillna('')
    P['shell'] = np.where(dbk & ct.isin(['COVER_0', 'COVER_1', 'COVER_3']), 'one', np.where(dbk & ct.isin(['COVER_2', 'COVER_4', 'COVER_6', '2_MAN']), 'two', None))
    P['prush'] = np.where(dbk & P.number_of_pass_rushers.notna(), np.where(P.number_of_pass_rushers >= 5, 'r5', 'r4'), None)
    run = P.rush == 1
    P['box'] = np.where(run & P.defenders_in_box.notna(), np.select([P.defenders_in_box <= 6, P.defenders_in_box == 7], ['light', 'b7'], 'heavy'), None)
    P['neutral'] = P.wp.between(0.1, 0.9)
    return P[['game_id', 'season', 'season_type', 'week', 'posteam', 'defteam', 'epa', 'neutral', 'pers', 'db', 'cov', 'shell', 'prush', 'box']]

P = pd.concat([load(y) for y in YRS], ignore_index=True)
P['wk'] = np.where(P.season_type == 'POST', P.week.where(P.week > 18, P.week + 18), P.week)
print(f'{len(P):,} plays with personnel, {P.season.min()}-{P.season.max()}')
# defence / offence quality, leave-this-game-out season means
for side, col in (('d', 'defteam'), ('o', 'posteam')):
    a = P.groupby(['season', col]).epa.agg(['sum', 'count']); b = P.groupby(['season', col, 'game_id']).epa.agg(['sum', 'count'])
    j = b.join(a, rsuffix='_s'); j[f'{side}q'] = (j['sum_s'] - j['sum']) / (j['count_s'] - j['count'])
    P = P.merge(j[[f'{side}q']].reset_index(), on=['season', col, 'game_id'])
lg = P.groupby('season').epa.mean().rename('lg'); P = P.join(lg, on='season')
P['r_o'] = P.epa - (P.dq - P.lg)    # offence play value with defence quality removed
P['r_d'] = P.epa - (P.oq - P.lg)    # defence side: offence quality removed
FAM = {'db': ['base', 'nickel', 'dime'], 'cov': ['man', 'zone'], 'shell': ['one', 'two'], 'prush': ['r4', 'r5'], 'box': ['light', 'b7', 'heavy']}
PERS = ['p11', 'p12', 'heavy', 'other']
N = P[P.neutral]

print('\n(1) Do OFFENCE splits repeat?  r between odd-week and even-week splits of the same team-season (shrunk, k=60 plays)')
def split_tab(D, fam, looks, by, rc='r_o', k=60):
    d = D[D[fam].notna()]
    g = d.groupby(by + [fam])[rc].agg(['sum', 'count']).unstack(fam).fillna(0)
    tot_s = sum(g[('sum', l)] for l in looks); tot_n = sum(g[('count', l)] for l in looks)
    out = {}
    for l in looks:
        n = g[('count', l)]; m = g[('sum', l)] / n.where(n > 0); allm = tot_s / tot_n
        out[l] = (n / (n + k) * (m - allm)).fillna(0)
    return pd.DataFrame(out), tot_n
rep = []
for fam, looks in FAM.items():
    odd, _ = split_tab(N[N.wk % 2 == 1], fam, looks, ['season', 'posteam']); even, _ = split_tab(N[N.wk % 2 == 0], fam, looks, ['season', 'posteam'])
    j = odd.join(even, lsuffix='_o', rsuffix='_e', how='inner')
    row = [fam] + [np.corrcoef(j[l + '_o'], j[l + '_e'])[0, 1] for l in looks]
    # within 11 personnel only (removes "offence in 12 faces base" confounding)
    o11, _ = split_tab(N[(N.wk % 2 == 1) & (N.pers == 'p11')], fam, looks, ['season', 'posteam']); e11, _ = split_tab(N[(N.wk % 2 == 0) & (N.pers == 'p11')], fam, looks, ['season', 'posteam'])
    j2 = o11.join(e11, lsuffix='_o', rsuffix='_e', how='inner')
    print(f'  {fam:6s} ' + '  '.join(f'{l} r {np.corrcoef(j[l + "_o"], j[l + "_e"])[0, 1]:+.2f}' for l in looks) +
          '   | 11-personnel only: ' + '  '.join(f'{l} {np.corrcoef(j2[l + "_o"], j2[l + "_e"])[0, 1]:+.2f}' for l in looks) + f'   (team-seasons {len(j)})')
print('  defence side (defence split vs offence personnel, offence quality removed):')
odd, _ = split_tab(N[N.wk % 2 == 1], 'pers', PERS, ['season', 'defteam'], 'r_d'); even, _ = split_tab(N[N.wk % 2 == 0], 'pers', PERS, ['season', 'defteam'], 'r_d')
j = odd.join(even, lsuffix='_o', rsuffix='_e', how='inner')
print('  pers   ' + '  '.join(f'{l} r {np.corrcoef(j[l + "_o"], j[l + "_e"])[0, 1]:+.2f}' for l in PERS))

print('\n    season to next season, same play-caller vs new play-caller (offence splits, all neutral snaps):')
cl = pd.read_csv(pc)[['season', 'team', 'caller']]
cl = cl.rename(columns={'team': 'posteam'})
for fam, looks in FAM.items():
    s, _ = split_tab(N, fam, looks, ['season', 'posteam']); s = s.reset_index().merge(cl, on=['season', 'posteam'], how='left')
    nx = s.copy(); nx['season'] -= 1
    j = s.merge(nx, on=['season', 'posteam'], suffixes=('', '_n'))
    same = j.caller == j.caller_n
    print(f'  {fam:6s} same caller (n {same.sum()}): ' + '  '.join(f'{l} {np.corrcoef(j[same][l], j[same][l + "_n"])[0, 1]:+.2f}' for l in looks) +
          f'   new caller (n {(~same).sum()}): ' + '  '.join(f'{l} {np.corrcoef(j[~same][l], j[~same][l + "_n"])[0, 1]:+.2f}' for l in looks))

print('\n(2) Does a DEFENCE\'s look mix repeat? (share of snaps in the look; odd vs even weeks / next season)')
for fam, looks in FAM.items():
    d = P[P[fam].notna()]
    for half in ['split', 'year']:
        if half == 'split':
            a = d[d.wk % 2 == 1].groupby(['season', 'defteam'])[fam].value_counts(normalize=True).unstack(fill_value=0)
            b = d[d.wk % 2 == 0].groupby(['season', 'defteam'])[fam].value_counts(normalize=True).unstack(fill_value=0)
        else:
            a = d.groupby(['season', 'defteam'])[fam].value_counts(normalize=True).unstack(fill_value=0)
            b = a.copy(); b.index = pd.MultiIndex.from_arrays([b.index.get_level_values(0) - 1, b.index.get_level_values(1)])
        j = a.join(b, lsuffix='_a', rsuffix='_b', how='inner')
        print(f'  {fam:6s} {half:5s} ' + '  '.join(f'{l} r {np.corrcoef(j[l + "_a"], j[l + "_b"])[0, 1]:+.2f}' for l in looks if l + '_a' in j))

# (3) walk-forward game test ------------------------------------------------------------------------------------------
print('\n(3) Walk-forward game test 2018-2025 vs the closing line')
games = pd.read_parquet(f'{H}/data/nfl_games_features.parquet')
games = games[games.season.between(2016, 2025) & games.margin.notna()][['game_id', 'season', 'week', 'home_team', 'away_team', 'home_score', 'away_score',
                                                                         'spread_line', 'total_line', 'margin', 'total']]
callers = cl.set_index(['season', 'posteam']).caller.to_dict()
# aggregate to team-game x look (neutral snaps for splits; all snaps for mixes)
OA = {f: N[N[f].notna()].groupby(['posteam', 'season', 'wk', f]).r_o.agg(['sum', 'count']) for f in FAM}
DM = {f: P[P[f].notna()].groupby(['defteam', 'season', 'wk', 'pers', f]).size() for f in FAM}          # defence look counts by offence personnel
OF = {f: P[P[f].notna()].groupby(['posteam', 'season', 'wk', f]).size() for f in FAM}                   # looks this offence faced
OP = P.groupby(['posteam', 'season', 'wk', 'pers']).size()                                               # offence personnel use
OPF = {f: P[P[f].notna()].groupby(['posteam', 'season', 'wk', 'pers']).size() for f in FAM}              # personnel on the family's snaps
DA = N.groupby(['defteam', 'season', 'wk', 'pers']).r_d.agg(['sum', 'count'])                           # defence vs personnel
DF = P.groupby(['defteam', 'season', 'wk', 'pers']).size()
LGM = {f: P[P[f].notna()].groupby(['season', 'pers', f]).size() for f in FAM}
def wts(team, season, wk, idx_team, idx_season, idx_wk):
    """prior-only weights: current season earlier weeks 1, last season 0.5 (0.25 if the play-caller changed), two back 0.25/0.1"""
    same = callers.get((season, team)) == callers.get((season - 1, team))
    w = np.where(idx_season == season, (idx_wk < wk) * 1.0, np.where(idx_season == season - 1, 0.5 if same else 0.25,
                 np.where(idx_season == season - 2, 0.25 if same else 0.1, 0.0)))
    return w
def sub(S, team, season):
    try: x = S.loc[team]
    except KeyError: return None
    return x[x.index.get_level_values(0).isin([season, season - 1, season - 2])]
rows = []
for gr in games[games.season >= 2018].itertuples():
    for O, D, home in ((gr.home_team, gr.away_team, 1), (gr.away_team, gr.home_team, 0)):
        rec = dict(game_id=gr.game_id, season=gr.season, team=O, opp=D, home=home)
        # offence personnel mix (prior)
        op = sub(OP, O, gr.season)
        if op is None: continue
        w = wts(O, gr.season, gr.week, None, op.index.get_level_values(0).values, op.index.get_level_values(1).values)
        pm = (op * w).groupby(level=2).sum(); pm = pm / pm.sum()
        for f, looks in FAM.items():
            oa = sub(OA[f], O, gr.season); dm = sub(DM[f], D, gr.season); of = sub(OF[f], O, gr.season); opf = sub(OPF[f], O, gr.season)
            if oa is None or dm is None or of is None or len(oa) == 0 or len(dm) == 0: rec[f] = 0.0; continue
            w = wts(O, gr.season, gr.week, None, oa.index.get_level_values(0).values, oa.index.get_level_values(1).values)
            ss = (oa['sum'] * w).groupby(level=2).sum(); nn = (oa['count'] * w).groupby(level=2).sum()
            allm = ss.sum() / max(nn.sum(), 1); spl = {l: (nn.get(l, 0) / (nn.get(l, 0) + 60)) * ((ss.get(l, 0) / nn[l]) - allm) if nn.get(l, 0) > 0 else 0 for l in looks}
            wd = wts(D, gr.season, gr.week, None, dm.index.get_level_values(0).values, dm.index.get_level_values(1).values)
            dmx = (dm * wd).groupby(level=[2, 3]).sum()      # (pers, look)
            lgp = LGM[f].loc[gr.season - 1] if gr.season - 1 in LGM[f].index.get_level_values(0) else LGM[f].loc[gr.season]
            # expected look rate vs this offence = sum_pers P(pers | O, on this family's snaps) x P(look | D, pers) shrunk to league (40 snaps)
            wf = wts(O, gr.season, gr.week, None, opf.index.get_level_values(0).values, opf.index.get_level_values(1).values)
            pmf = (opf * wf).groupby(level=2).sum(); pmf = pmf / pmf.sum()
            exp = {l: 0.0 for l in looks}
            for pe, pp in pmf.items():
                tot = sum(dmx.get((pe, l), 0) for l in looks); lt = sum(lgp.get((pe, l), 0) for l in looks)
                for l in looks:
                    prior = lgp.get((pe, l), 0) / lt if lt else 1 / len(looks)
                    exp[l] += pp * (dmx.get((pe, l), 0) + 40 * prior) / (tot + 40)
            wo = wts(O, gr.season, gr.week, None, of.index.get_level_values(0).values, of.index.get_level_values(1).values)
            fc = (of * wo).groupby(level=2).sum(); fc = fc / fc.sum()
            rec[f] = sum((exp[l] - fc.get(l, 0)) * spl[l] for l in looks)
            rec[f + '_mixdev'] = exp.get('nickel', exp.get('man', exp.get('one', exp.get('r5', exp.get('heavy', 0))))) - fc.get(looks[1] if f == 'db' else looks[0], 0)
        # defence side: D's split vs personnel x how much more of each personnel O will use than D usually sees
        da = sub(DA, D, gr.season); df = sub(DF, D, gr.season)
        if da is not None and len(da):
            w = wts(D, gr.season, gr.week, None, da.index.get_level_values(0).values, da.index.get_level_values(1).values)
            ss = (da['sum'] * w).groupby(level=2).sum(); nn = (da['count'] * w).groupby(level=2).sum(); allm = ss.sum() / max(nn.sum(), 1)
            wf = wts(D, gr.season, gr.week, None, df.index.get_level_values(0).values, df.index.get_level_values(1).values)
            fd = (df * wf).groupby(level=2).sum(); fd = fd / fd.sum()
            rec['pers'] = sum((pm.get(p, 0) - fd.get(p, 0)) * (nn.get(p, 0) / (nn.get(p, 0) + 60)) * ((ss.get(p, 0) / nn[p]) - allm) for p in PERS if nn.get(p, 0) > 0)
        else: rec['pers'] = 0.0
        rec['pts'] = gr.home_score if home else gr.away_score
        rec['imp'] = (gr.total_line + (gr.spread_line if home else -gr.spread_line)) / 2
        rows.append(rec)
R = pd.DataFrame(rows).fillna(0)
R.to_parquet('/tmp/claude-0/nflwork/scheme_game_terms.parquet')
TERMS = list(FAM) + ['pers']
R['all'] = R[TERMS].sum(axis=1)
R['res'] = R.pts - R.imp
print(f'  team-games {len(R)}; term SD (EPA/play): ' + '  '.join(f'{t} {R[t].std():.4f}' for t in TERMS + ['all']))
print('  team points minus closing-line implied points ~ term (EPA/play x ~62 plays would be ~62 if fully real and unpriced):')
for t in TERMS + ['all']:
    r = sm.OLS(R.res, sm.add_constant(R[t])).fit(cov_type='cluster', cov_kwds={'groups': R.game_id.astype('category').cat.codes})
    print(f'    {t:6s} coef {r.params.iloc[1]:+7.1f} pts per EPA/play  (t {r.tvalues.iloc[1]:+.2f})   1 SD of term = {r.params.iloc[1] * R[t].std():+.2f} pts')
r = sm.OLS(R.res, sm.add_constant(R[TERMS])).fit(cov_type='cluster', cov_kwds={'groups': R.game_id.astype('category').cat.codes})
print('  all terms together:\n' + r.summary().tables[1].as_text())
# game level: margin vs spread, total vs total
G = R.pivot_table(index='game_id', columns='home', values='all').rename(columns={1: 'h', 0: 'a'}).join(games.set_index('game_id'))
G = G.dropna(subset=['h', 'a']); G['diff'] = G.h - G.a; G['sum'] = G.h + G.a
F = pd.read_parquet(f'{H}/data/nfl_games_features.parquet').set_index('game_id')
G['est'] = (-0.429 + 0.51 * F.rt_pred + 0.561 * F.mkt_pred).reindex(G.index)
for nm, y, x in [('margin - spread', G.margin - G.spread_line, G['diff']), ('margin - our model', G.margin - G.est, G['diff']), ('total - total line', G.total - G.total_line, G['sum'])]:
    r = sm.OLS(y, sm.add_constant(x)).fit()
    q = x.abs() > x.abs().quantile(0.8); side = np.sign(x[q]); res = np.sign(y[q]); hit = np.mean((side == res)[res != 0])
    print(f'  {nm:18s} coef {r.params.iloc[1]:+.1f} (t {r.tvalues.iloc[1]:+.2f}); top-20% largest terms (n {q.sum()}): right side {hit:.1%}')
print('  by season (team points coef on the summed term):', '  '.join(f'{y}: {sm.OLS(s.res, sm.add_constant(s["all"])).fit().params.iloc[1]:+.0f}' for y, s in R.groupby('season')))
