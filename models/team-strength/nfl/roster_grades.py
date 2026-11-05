"""NFL player grades -> team roster grade from who is expected to play -> does it improve game prediction?

Built 2026-10-11/15 (J: "grade teams and athletes and include that in models, especially target share and scheme").
Judged on predicting actual margins/totals (RMSE, win-prob log loss); beating the close is a secondary check.

usage: python3 -I nfl/roster_grades.py <pbp_dir> <grades_dl_dir> <team-strength data dir> <cache dir> [log path]
  pbp_dir:   pbp_YYYY.parquet (2016-2026), part_YYYY.parquet (nflverse participation 2016-2025)
  grades_dl: nflverse players.parquet, ngs_rushing.parquet, snaps_YYYY (snap_counts), rw_YYYY (weekly rosters),
             inj_YYYY (injuries), advdef_YYYY (pfr_advstats advstats_week_def, 2018+)
  data dir:  nfl_games_features.parquet (walk-forward rt_pred / mkt_pred, QB terms, lines, results)
  cache:     scratch folder for intermediate parquet files (not the shared folder)
  log path:  default roster_grades_log.txt next to this script

Steps
1. Player-game stats (2016-2026). Every grade is a (numerator, denominator) pair:
   QB      EPA per dropback (qb_epa), CPOE per attempt; composite weights fit on 2016-17 games only
   WR/TE/RB targets per route (route = on field for a dropback, participation, 2016-25), yards per route,
           EPA per target, EPA per target relative to the team's other targets that game, coverage-adjusted EPA per
           target (minus league-season mean for that man/zone label), targets per offensive snap, target share
   RB      designed-run EPA and success per carry, NGS rush yards over expected per carry (2018+), carries per snap
   OL      the team's sack/hit EPA cost per dropback and designed-run EPA per carry, credited by snap share
   DEF     sacks + QB hits per pass snap (pbp, all years), PFR pressures per pass snap (2018+), PFR coverage
           targets and yards allowed per pass snap (2018+), on-field defensive EPA allowed per play (snap weighted)
   all     on-field offensive EPA per play (snap weighted "plus-minus")
2. Walk-forward grades: decayed sums of earlier games only (half-life 16 games, x0.75 at each new season), shrunk
   to the league mean with prior weight K = n_typical*(1-r)/r, r = Spearman-Brown split-half reliability (odd vs even
   games within a season, 2016-2019 seasons). Year-to-year r of each grade reported.
3. Team grade per game: players expected to play = that week's active roster (nflverse weekly rosters, status ACT)
   minus Out/Doubtful on the injury report; expected snap share = recent snap share (last 3 appearances,
   0.5/0.3/0.2, any team; rookies by draft round); actual starting QB as in the existing QB term. Shares normalised to
   each position group's usual on-field count. Target/carry weights = snap share x targets (carries) per snap grade,
   normalised -> the pass-game grade is expected target share x receiver grade x the team's expected targets per game
   (team volume = scheme). Components in EPA per game above average. "delta" = expected roster minus the team's
   recent ACTUAL roster (snap weighted, decayed, evaluated with the same current grades): the personnel change the
   team ratings have not seen.
4. Tests 2018-2025 against actual results, closing line, totals (see log).
"""
import sys, os, numpy as np, pandas as pd, statsmodels.api as sm
from scipy.stats import norm

PBP, DL, DATA, CACHE = sys.argv[1:5]
LOGPATH = sys.argv[5] if len(sys.argv) > 5 else os.path.join(os.path.dirname(os.path.abspath(__file__)), 'roster_grades_log.txt')
os.makedirs(CACHE, exist_ok=True)
YEARS = range(2016, 2027)
TEAMFIX = {'SD': 'LAC', 'OAK': 'LV', 'STL': 'LA', 'LAR': 'LA', 'JAC': 'JAX'}
fixteam = lambda s: s.replace(TEAMFIX)
HALF, CARRY = 16, 0.75
DECAY = 0.5 ** (1 / HALF)
LOG = []
def log(*a):
    s = ' '.join(str(x) for x in a); print(s, flush=True); LOG.append(s)

GROUP = {'QB': 'QB', 'RB': 'RB', 'HB': 'RB', 'FB': 'RB', 'WR': 'WR', 'TE': 'TE',
         'T': 'OL', 'G': 'OL', 'C': 'OL', 'OL': 'OL', 'OT': 'OL', 'OG': 'OL',
         'DE': 'DL', 'DT': 'DL', 'NT': 'DL', 'DL': 'DL', 'LB': 'LB', 'OLB': 'LB', 'ILB': 'LB', 'MLB': 'LB',
         'CB': 'DB', 'S': 'DB', 'FS': 'DB', 'SS': 'DB', 'DB': 'DB', 'SAF': 'DB'}
OFF_G, DEF_G = ['QB', 'RB', 'WR', 'TE', 'OL'], ['DL', 'LB', 'DB']

# ------------------------------------------------------------------ 1. player-game stats
def build_pg():
    f = f'{CACHE}/pg.parquet'
    if os.path.exists(f): return pd.read_parquet(f), pd.read_parquet(f'{CACHE}/tg.parquet')
    players = pd.read_parquet(f'{DL}/players.parquet')
    pfr2g = players.dropna(subset=['pfr_id']).drop_duplicates('pfr_id').set_index('pfr_id').gsis_id
    cols = ['game_id', 'play_id', 'season', 'week', 'game_date', 'posteam', 'defteam', 'play_type', 'qb_dropback', 'qb_scramble',
            'sack', 'qb_hit', 'pass_attempt', 'passer_id', 'receiver_player_id', 'rusher_player_id', 'epa', 'qb_epa', 'cpoe',
            'success', 'yards_gained', 'two_point_attempt', 'sack_player_id', 'half_sack_1_player_id', 'half_sack_2_player_id',
            'qb_hit_1_player_id', 'qb_hit_2_player_id']
    rows = []
    for y in YEARS:
        p = pd.read_parquet(f'{PBP}/pbp_{y}.parquet', columns=cols)
        p = p[p.play_type.isin(['pass', 'run']) & (p.two_point_attempt != 1) & p.epa.notna() & p.posteam.notna()].copy()
        p['posteam'] = fixteam(p.posteam); p['defteam'] = fixteam(p.defteam)
        p['db'] = p.qb_dropback.fillna(0).eq(1)
        p['dcar'] = p.play_type.eq('run') & p.qb_scramble.fillna(0).eq(0)
        p['tgt'] = p.pass_attempt.eq(1) & p.sack.fillna(0).eq(0) & p.receiver_player_id.notna()
        p['sk'] = p.sack.fillna(0).eq(1); p['hit'] = p.qb_hit.fillna(0).eq(1) & ~p.sk
        ppath = f'{PBP}/part_{y}.parquet'
        p['play_id'] = p.play_id.astype(float)
        if os.path.exists(ppath):
            pa = pd.read_parquet(ppath, columns=['nflverse_game_id', 'play_id', 'offense_players', 'defense_man_zone_type'])
            pa = pa.rename(columns={'nflverse_game_id': 'game_id'}); pa['play_id'] = pa.play_id.astype(float)
            p = p.merge(pa.drop_duplicates(['game_id', 'play_id']), on=['game_id', 'play_id'], how='left')
        else:
            p['offense_players'] = np.nan; p['defense_man_zone_type'] = np.nan
        p['cov'] = p.defense_man_zone_type.replace('', np.nan).fillna('NA')
        p['date'] = pd.to_datetime(p.game_date)
        rows.append(p); log(f'pbp {y}: {len(p)} plays, participation {p.offense_players.notna().mean():.2f}')
    p = pd.concat(rows, ignore_index=True); del rows
    b = p[(p.season <= 2017) & p.db]
    mu_db = b.epa.mean(); SKC = b[b.sk].epa.mean() - mu_db; HITC = b[b.hit].epa.mean() - mu_db
    log(f'EPA cost vs average dropback (2016-17): sack {SKC:.2f}, non-sack QB hit {HITC:.2f}')
    p['ppcost'] = np.where(p.sk, SKC, 0.0) + np.where(p.hit, HITC, 0.0)
    p['covlab'] = p['cov'].ne('NA'); p['run_epa_v'] = np.where(p.dcar, p.epa, 0.0)
    t = p[p.tgt]
    cm = t.groupby(['season', 'cov']).epa.mean().rename('cov_mu')
    p = p.join(cm, on=['season', 'cov'])
    p['epa_cov'] = np.where(p.tgt, p.epa - p.cov_mu, np.nan)
    gt = p[p.tgt].groupby(['game_id', 'posteam']).epa.agg(['sum', 'count'])
    p = p.join(gt.rename(columns={'sum': 'tm_tgt_epa', 'count': 'tm_tgt'}), on=['game_id', 'posteam'])
    p['epa_rel'] = np.where(p.tgt, p.epa - (p.tm_tgt_epa - p.epa) / (p.tm_tgt - 1).clip(lower=1), np.nan)
    key = ['game_id', 'season', 'week', 'date']
    q = p[p.db & p.passer_id.notna()].groupby(key + ['posteam', 'passer_id']).agg(
        qb_db=('qb_epa', 'size'), qb_epa=('qb_epa', 'sum'), cpoe_s=('cpoe', 'sum'), cpoe_n=('cpoe', 'count')).reset_index().rename(columns={'passer_id': 'gsis', 'posteam': 'team'})
    r = p[p.tgt].groupby(key + ['posteam', 'receiver_player_id']).agg(
        tgt=('epa', 'size'), rec_epa=('epa', 'sum'), rec_rel=('epa_rel', 'sum'), rec_cov=('epa_cov', 'sum'), rec_yds=('yards_gained', 'sum'),
        cov_n=('covlab', 'sum')).reset_index().rename(columns={'receiver_player_id': 'gsis', 'posteam': 'team'})
    d = p[p.db & p.offense_players.notna()][key + ['posteam', 'offense_players']]
    d = d.assign(gsis=d.offense_players.str.split(';')).explode('gsis')
    rt = d.groupby(key + ['posteam', 'gsis']).size().rename('routes').reset_index().rename(columns={'posteam': 'team'})
    u = p[p.dcar & p.rusher_player_id.notna()].groupby(key + ['posteam', 'rusher_player_id']).agg(
        car=('epa', 'size'), rush_epa=('epa', 'sum'), rush_succ=('success', 'sum')).reset_index().rename(columns={'rusher_player_id': 'gsis', 'posteam': 'team'})
    pr = []
    for c in ['sack_player_id', 'half_sack_1_player_id', 'half_sack_2_player_id', 'qb_hit_1_player_id', 'qb_hit_2_player_id']:
        pr.append(p[p[c].notna()][key + ['play_id', 'defteam', c]].rename(columns={c: 'gsis'}))
    pr = pd.concat(pr).drop_duplicates(['game_id', 'play_id', 'gsis'])
    pr = pr.groupby(key + ['defteam', 'gsis']).size().rename('pr_pbp').reset_index().rename(columns={'defteam': 'team'})
    tg = p.groupby(key + ['posteam', 'defteam']).agg(plays=('epa', 'size'), epa=('epa', 'sum'), db=('db', 'sum'),
        ppcost=('ppcost', 'sum'), dcar=('dcar', 'sum'), run_epa=('run_epa_v', 'sum'),
        tgts=('tgt', 'sum')).reset_index().rename(columns={'posteam': 'team', 'defteam': 'opp'})
    tg.to_parquet(f'{CACHE}/tg.parquet')
    sn = []
    for y in YEARS:
        s = pd.read_parquet(f'{DL}/snaps_{y}.parquet'); s['team'] = fixteam(s.team); s['opponent'] = fixteam(s.opponent)
        s['gsis'] = s.pfr_player_id.map(pfr2g); sn.append(s)
    sn = pd.concat(sn); log(f'snap rows {len(sn)}, mapped to gsis {sn.gsis.notna().mean():.4f}')
    sn = sn.dropna(subset=['gsis'])
    sn['grp'] = sn.position.map(GROUP)
    sn = sn[['game_id', 'gsis', 'team', 'grp', 'offense_pct', 'defense_pct', 'offense_snaps', 'defense_snaps']].drop_duplicates(['game_id', 'gsis'])
    gk = tg[key].drop_duplicates('game_id')
    sn = sn.merge(gk, on='game_id', how='inner')
    to = tg.rename(columns={'team': 'tm'}).set_index(['game_id', 'tm'])
    td = tg.rename(columns={'opp': 'tm'}).set_index(['game_id', 'tm'])
    ix = pd.MultiIndex.from_arrays([sn.game_id, sn.team])
    o = to.reindex(ix); dd = td.reindex(ix)
    op, dp = sn.offense_pct.fillna(0).values, sn.defense_pct.fillna(0).values
    sn['o_plays'] = o.plays.values * op; sn['o_epa'] = o.epa.values * op
    sn['d_plays'] = dd.plays.values * dp; sn['d_epa'] = dd.epa.values * dp
    sn['d_pass'] = dd.db.values * dp                 # pass snaps faced (opponent dropbacks x snap share)
    isol = sn.grp.eq('OL').values
    sn['ol_db'] = np.where(isol, o.db.values * op, 0); sn['ol_pp'] = np.where(isol, o.ppcost.values * op, 0)
    sn['ol_car'] = np.where(isol, o.dcar.values * op, 0); sn['ol_run'] = np.where(isol, o.run_epa.values * op, 0)
    sn['tm_tgts'] = o.tgts.values; sn['tm_car'] = o.dcar.values
    adv = []
    for y in range(2018, 2027):
        f2 = f'{DL}/advdef_{y}.parquet'
        if os.path.exists(f2): adv.append(pd.read_parquet(f2))
    adv = pd.concat(adv); adv['gsis'] = adv.pfr_player_id.map(pfr2g)
    adv = adv.dropna(subset=['gsis']).groupby(['game_id', 'gsis'])[['def_pressures', 'def_targets', 'def_yards_allowed']].sum().reset_index()
    adv['pfr'] = 1.0
    ng = pd.read_parquet(f'{DL}/ngs_rushing.parquet'); ng = ng[(ng.week > 0) & ng.rush_yards_over_expected.notna()]
    ng = ng.rename(columns={'player_gsis_id': 'gsis'})[['season', 'week', 'gsis', 'rush_yards_over_expected', 'rush_attempts']]
    ng.columns = ['season', 'week', 'gsis', 'ryoe', 'ryoe_att']
    pg = sn
    for x in [q.drop(columns=['team']), r.drop(columns=['team']), rt.drop(columns=['team']), u.drop(columns=['team']), pr.drop(columns=['team'])]:
        pg = pg.merge(x, on=key + ['gsis'], how='left')
    pg = pg.merge(adv, on=['game_id', 'gsis'], how='left').merge(ng.drop_duplicates(['season', 'week', 'gsis']), on=['season', 'week', 'gsis'], how='left')
    num = pg.columns.difference(['game_id', 'gsis', 'team', 'grp', 'season', 'week', 'date'])
    pg[num] = pg[num].fillna(0)
    pg['has_part'] = pg.season <= 2025
    pg['pr_den'] = np.where(pg.grp.isin(DEF_G), pg.d_pass, 0)
    pg['pfr_den'] = np.where(pg.pfr > 0, pg.pr_den, 0)
    pg['route_den'] = np.where(pg.has_part & pg.grp.isin(['WR', 'TE', 'RB']), pg.routes, 0)
    pg['skill_snaps'] = np.where(pg.grp.isin(['WR', 'TE', 'RB']), pg.o_plays, 0)
    pg['rb_snaps'] = np.where(pg.grp.eq('RB'), pg.o_plays, 0)
    pg['tgt_den_share'] = np.where(pg.grp.isin(['WR', 'TE', 'RB']), pg.tm_tgts, 0)
    pg = pg.sort_values(['gsis', 'date', 'game_id']).reset_index(drop=True)
    pg.to_parquet(f); return pg, tg

# grade definitions: name -> (numerator, denominator, min denominator per season for repeatability)
GRADES = {
    'qb_epa_db':   ('qb_epa', 'qb_db', 150),
    'qb_cpoe':     ('cpoe_s', 'cpoe_n', 150),
    'rec_tprr':    ('tgt', 'route_den', 150),
    'rec_yprr':    ('rec_yds', 'route_den', 150),
    'rec_epa_tgt': ('rec_epa', 'tgt', 40),
    'rec_rel_tgt': ('rec_rel', 'tgt', 40),
    'rec_cov_tgt': ('rec_cov', 'tgt', 40),
    'rec_tgt_snap':('tgt', 'skill_snaps', 150),
    'rec_tshare':  ('tgt', 'tgt_den_share', 150),
    'rush_epa':    ('rush_epa', 'car', 60),
    'rush_succ':   ('rush_succ', 'car', 60),
    'rush_ryoe':   ('ryoe', 'ryoe_att', 60),
    'rush_car_snap':('car', 'rb_snaps', 100),
    'ol_pp':       ('ol_pp', 'ol_db', 200),
    'ol_run':      ('ol_run', 'ol_car', 150),
    'def_pr_pbp':  ('pr_pbp', 'pr_den', 150),
    'def_press_pfr':('def_pressures', 'pfr_den', 150),
    'def_cov_tgt': ('def_targets', 'pfr_den', 150),
    'def_cov_yds': ('def_yards_allowed', 'pfr_den', 150),
    'def_onfield': ('d_epa', 'd_plays', 300),
    'off_onfield': ('o_epa', 'o_plays', 300),
}

def reliability(pg):
    """split-half (odd/even games) and year-to-year r for each grade; returns K (prior weight) per grade."""
    pg = pg.copy(); pg['gno'] = pg.groupby(['gsis', 'season']).cumcount() % 2
    K, rep, mu = {}, [], {}
    for g, (n, d, mn) in GRADES.items():
        x = pg[pg[d] > 0]
        mu[g] = x[n].sum() / x[d].sum()
        h = x.groupby(['gsis', 'season', 'gno'])[[n, d]].sum().unstack('gno')
        h = h[(h[d][0] >= mn / 2) & (h[d][1] >= mn / 2)]
        hs = h[h.index.get_level_values('season') <= 2019]
        r = np.corrcoef(hs[n][0] / hs[d][0], hs[n][1] / hs[d][1])[0, 1]
        rf = max(2 * r / (1 + r), 0.02)
        nt = (hs[d][0] + hs[d][1]).median()
        K[g] = nt * (1 - rf) / rf
        s = x.groupby(['gsis', 'season'])[[n, d]].sum(); s = s[s[d] >= mn]
        s['rate'] = s[n] / s[d]
        s = s.reset_index(); s2 = s.assign(season=s.season - 1)
        m = s.merge(s2, on=['gsis', 'season'], suffixes=('', '_next'))
        yy = np.corrcoef(m.rate, m.rate_next)[0, 1] if len(m) > 10 else np.nan
        rep.append(dict(grade=g, split_half_r=round(r, 3), full_season_r=round(rf, 3), K_prior=round(K[g], 1), n_typ=round(nt, 0),
                        yoy_r=round(yy, 3), yoy_pairs=len(m), min_den=mn, league_mean=round(mu[g], 4)))
    return K, mu, pd.DataFrame(rep)

def states(pg):
    """walk-forward decayed sums after each player-game (used only for LATER games)."""
    nums = [GRADES[g][0] for g in GRADES]; dens = [GRADES[g][1] for g in GRADES]
    Xn, Xd = pg[nums].values.astype(float), pg[dens].values.astype(float)
    S, N = np.zeros_like(Xn), np.zeros_like(Xd)
    pid, sea = pg.gsis.values, pg.season.values
    s = np.zeros(len(nums)); n = np.zeros(len(nums))
    for i in range(len(pg)):
        if i == 0 or pid[i] != pid[i - 1]: s[:] = 0; n[:] = 0
        elif sea[i] != sea[i - 1]: f = CARRY ** (sea[i] - sea[i - 1]); s *= f; n *= f
        s = s * DECAY + Xn[i]; n = n * DECAY + Xd[i]
        S[i] = s; N[i] = n
    st = pd.DataFrame(np.hstack([S, N]), columns=[f'S_{g}' for g in GRADES] + [f'N_{g}' for g in GRADES])
    st['gsis'] = pid; st['date'] = pg.date.values; st['st_season'] = sea
    return st

def lookup(rows, st, K, mu):
    """grades as of BEFORE each row's game date (strictly earlier games)."""
    rows = rows.reset_index(drop=True); rows['_i'] = np.arange(len(rows))
    a = rows[['_i', 'gsis', 'date']].sort_values('date')
    b = st.sort_values('date')
    m = pd.merge_asof(a, b, on='date', by='gsis', allow_exact_matches=False).sort_values('_i')
    f = CARRY ** (rows.season.values - m.st_season.fillna(rows.season).values).clip(0, 10)
    out = pd.DataFrame(index=rows.index)
    for g in GRADES:
        S_ = m[f'S_{g}'].fillna(0).values * f; N_ = m[f'N_{g}'].fillna(0).values * f
        out[g] = (S_ + K[g] * mu[g]) / (N_ + K[g]); out[g + '_n'] = N_
    return pd.concat([rows.drop(columns='_i'), out], axis=1)

# ------------------------------------------------------------------ 3. expected and baseline rosters
def rosters(pg, games):
    f = f'{CACHE}/rost.parquet'
    if os.path.exists(f): return pd.read_parquet(f)
    players = pd.read_parquet(f'{DL}/players.parquet')
    dr = players.drop_duplicates('gsis_id').set_index('gsis_id')[['draft_year', 'draft_round']]
    tgl = pd.concat([games[['game_id', 'season', 'week', 'date', 'home_team', 'home_qb_id']].set_axis(['game_id', 'season', 'week', 'date', 'team', 'qb'], axis=1),
                     games[['game_id', 'season', 'week', 'date', 'away_team', 'away_qb_id']].set_axis(['game_id', 'season', 'week', 'date', 'team', 'qb'], axis=1)])
    app = pg[['gsis', 'date', 'season', 'grp', 'offense_pct', 'defense_pct']].copy()
    app['pct'] = np.where(app.grp.isin(DEF_G), app.defense_pct, app.offense_pct)
    app = app.sort_values(['gsis', 'date'])
    g1 = app.groupby('gsis')
    app['e1'] = app.pct; app['e2'] = g1.pct.shift(1); app['e3'] = g1.pct.shift(2)
    w = np.array([0.5, 0.3, 0.2]); E = app[['e1', 'e2', 'e3']].values
    W = np.where(np.isnan(E), 0, w); app['exp_share'] = np.nansum(np.nan_to_num(E) * W, 1) / W.sum(1)
    app['grp_last'] = app.grp
    out = []
    for y in YEARS:
        rw = pd.read_parquet(f'{DL}/rw_{y}.parquet'); rw['team'] = fixteam(rw.team)
        rw = rw[rw.status.eq('ACT') & rw.gsis_id.notna()][['team', 'week', 'gsis_id', 'position', 'depth_chart_position']].rename(columns={'gsis_id': 'gsis'})
        inj = pd.read_parquet(f'{DL}/inj_{y}.parquet'); inj['team'] = fixteam(inj.team)
        out_ = inj[inj.report_status.isin(['Out', 'Doubtful'])][['team', 'week', 'gsis_id']].rename(columns={'gsis_id': 'gsis'}).assign(out=1)
        t = tgl[tgl.season == y].merge(rw, on=['team', 'week'], how='inner')
        t = t.merge(out_.drop_duplicates(), on=['team', 'week', 'gsis'], how='left'); t = t[t.out.isna()].drop(columns='out')
        out.append(t)
    ex = pd.concat(out, ignore_index=True).drop_duplicates(['game_id', 'team', 'gsis'])
    ex = ex.sort_values('date')
    ex = pd.merge_asof(ex, app[['gsis', 'date', 'exp_share', 'grp_last']].sort_values('date'), on='date', by='gsis', allow_exact_matches=False)
    ex['grp'] = ex.grp_last.fillna(ex.depth_chart_position.map(GROUP)).fillna(ex.position.map(GROUP))
    rk = dr.reindex(ex.gsis)
    rook = (rk.draft_year.values == ex.season.values)
    dflt = np.select([rk.draft_round.values == 1, rk.draft_round.values == 2, rk.draft_round.values == 3], [0.6, 0.4, 0.25], 0.08)
    ex['share'] = np.where(ex.exp_share.notna(), ex.exp_share, np.where(rook, dflt, 0.03))
    ex = ex[ex.grp.notna()].copy()
    isqb = ex.grp.eq('QB'); ex.loc[isqb, 'share'] = np.where(ex.loc[isqb, 'gsis'] == ex.loc[isqb, 'qb'], 1.0, 0.0)
    ex = ex[~((ex.gsis == ex.qb) & ~isqb)]
    have = ex[ex.grp.eq('QB') & ex.share.gt(0)].set_index(['game_id', 'team']).index
    missq = tgl[~tgl.set_index(['game_id', 'team']).index.isin(have)]
    missq = missq[missq.qb.notna()].assign(gsis=lambda d: d.qb, grp='QB', share=1.0)
    cols = ['game_id', 'season', 'week', 'date', 'team', 'gsis', 'grp', 'share']
    ex = pd.concat([ex[cols], missq[cols]]); ex['kind'] = 'exp'
    # baseline: team's recent ACTUAL roster, decayed snap share over its earlier games (half-life 4 games, x0.5 new season)
    a = pg[['game_id', 'date', 'season', 'team', 'gsis', 'grp', 'offense_pct', 'defense_pct']].copy()
    a['pct'] = np.where(a.grp.isin(DEF_G), a.defense_pct, a.offense_pct); a = a[a.grp.notna() & (a.pct > 0)]
    bl = []; dec = 0.5 ** (1 / 4)
    tl = tgl.sort_values('date')
    for team, tt in tl.groupby('team'):
        aa = a[a.team == team]; byg = {k: v for k, v in aa.groupby('game_id')}
        acc, tot, last_s = {}, 0.0, None
        for r in tt.itertuples():
            if last_s is not None and r.season != last_s:
                acc = {k: v * 0.5 for k, v in acc.items()}; tot *= 0.5
            if tot > 0:
                for (pid, grp), v in acc.items():
                    if v / tot > 0.01: bl.append((r.game_id, r.season, r.week, r.date, team, pid, grp, v / tot))
            acc = {k: v * dec for k, v in acc.items()}; tot = tot * dec + 1
            if r.game_id in byg:
                for pid, grp, pc in zip(byg[r.game_id].gsis, byg[r.game_id].grp, byg[r.game_id].pct):
                    acc[(pid, grp)] = acc.get((pid, grp), 0) + pc
            last_s = r.season
    bl = pd.DataFrame(bl, columns=cols); bl['kind'] = 'base'
    ac = a.merge(tgl[['game_id', 'team', 'week']], on=['game_id', 'team'])[cols[:-1] + ['pct']].rename(columns={'pct': 'share'})
    ac['kind'] = 'act'
    R = pd.concat([ex, bl, ac], ignore_index=True)
    R.to_parquet(f); return R

def team_components(R, slots, tm_vol):
    """components per (game_id, team, kind), EPA per game vs average. O_* positive = better offense; D_* positive = allows more."""
    R = R[R.share > 0].copy()
    gs = R.groupby(['game_id', 'team', 'kind', 'grp']).share.transform('sum')
    R['sh'] = np.where(R.grp.eq('QB'), R.share / gs, R.share / gs * R.grp.map(slots))
    R = R.merge(tm_vol, on=['game_id', 'team'], how='left')
    for c, v in [('v_db', 36.0), ('v_tgt', 32.0), ('v_car', 24.0), ('v_plays', 62.0), ('v_odb', 36.0)]:
        R[c] = R[c].fillna(v)
    M = MU; k = ['game_id', 'team', 'kind']; out = {}
    G = lambda df: [df[x] for x in k]
    q = R[R.grp.eq('QB')]
    out['O_QB'] = ((q.qb_comp - M['qb_epa_db']) * q.sh * q.v_db).groupby(G(q)).sum()
    s = R[R.grp.isin(['WR', 'TE', 'RB'])].copy(); s['w'] = s.sh * s.rec_tgt_snap; s['w'] = s.w / s.groupby(k).w.transform('sum')
    out['O_PASSGAME'] = (s.w * s.rec_rel_tgt * s.v_tgt).groupby(G(s)).sum()
    out['O_PASSGAME_cov'] = (s.w * (s.rec_cov_tgt - M['rec_cov_tgt']) * s.v_tgt).groupby(G(s)).sum()
    out['O_PASSGAME_raw'] = (s.w * (s.rec_epa_tgt - M['rec_epa_tgt']) * s.v_tgt).groupby(G(s)).sum()
    out['O_YPRR'] = (s.sh * (s.rec_yprr - M['rec_yprr'])).groupby(G(s)).sum()
    out['O_TPRR'] = (s.sh * (s.rec_tprr - M['rec_tprr'])).groupby(G(s)).sum()
    b = R[R.grp.eq('RB')].copy(); b['w'] = b.sh * b.rush_car_snap; b['w'] = b.w / b.groupby(k).w.transform('sum')
    out['O_RUSH'] = (b.w * (b.rush_epa - M['rush_epa']) * b.v_car).groupby(G(b)).sum()
    out['O_RYOE'] = (b.w * (b.rush_ryoe - M['rush_ryoe']) * 0.06 * b.v_car).groupby(G(b)).sum()
    o = R[R.grp.eq('OL')]
    out['O_OL'] = ((o.ol_pp - M['ol_pp']) * o.sh / 5 * o.v_db + (o.ol_run - M['ol_run']) * o.sh / 5 * o.v_car).groupby(G(o)).sum()
    oo = R[R.grp.isin(OFF_G)]
    out['O_ONFIELD'] = ((oo.off_onfield - M['off_onfield']) * oo.sh / 11 * oo.v_plays).groupby(G(oo)).sum()
    d = R[R.grp.isin(DEF_G)]
    # PRC < 0 (EPA cost to the offense per credited sack/hit): more pass rush -> negative = allows less
    out['D_PASSRUSH'] = ((d.def_pr_pbp - M['def_pr_pbp']) * d.sh * d.v_odb * PRC).groupby(G(d)).sum()
    out['D_PRESS_PFR'] = ((d.def_press_pfr - M['def_press_pfr']) * d.sh * d.v_odb * PRC * 0.5).groupby(G(d)).sum()
    out['D_COVERAGE'] = ((d.def_cov_yds - M['def_cov_yds']) * d.sh * d.v_odb * 0.06).groupby(G(d)).sum()
    out['D_ONFIELD'] = ((d.def_onfield - M['def_onfield']) * d.sh / 11 * d.v_plays).groupby(G(d)).sum()
    C = pd.DataFrame(out).fillna(0)
    C['O_UNITS'] = C.O_QB + C.O_PASSGAME + C.O_RUSH + C.O_OL
    C['D_UNITS'] = C.D_PASSRUSH + C.D_COVERAGE
    C['NET_UNITS'] = C.O_UNITS - C.D_UNITS
    C['NET_ONFIELD'] = C.O_ONFIELD - C.D_ONFIELD
    return C.reset_index()

# ------------------------------------------------------------------ main
pg, tg = build_pg()
log(f'player-games {len(pg)}, seasons {pg.season.min()}-{pg.season.max()}')
K, MU, rep = reliability(pg)
log('\n=== 1. Grade reliability (split-half r: odd vs even games in a season, 2016-19; full_season_r = Spearman-Brown;\n'
    'K_prior = shrinkage prior weight in denominator units; yoy_r = season t vs t+1 raw rate, players with >= min_den)')
log(rep.to_string(index=False))
st = states(pg)
games = pd.read_parquet(f'{DATA}/nfl_games_features.parquet')
games = games[games.season >= 2016].copy()
for c in ['home_team', 'away_team', 'home', 'away']: games[c] = fixteam(games[c])
games['date'] = pd.to_datetime(games.gameday)
R = rosters(pg, games)
Gq = lookup(R[['gsis', 'date', 'season']].copy(), st, K, MU)
R = pd.concat([R.reset_index(drop=True), Gq.drop(columns=['gsis', 'date', 'season'])], axis=1)

qg = pg[(pg.qb_db >= 10)][['gsis', 'date', 'season', 'qb_epa', 'qb_db']].copy()
qq = lookup(qg[['gsis', 'date', 'season']].copy(), st, K, MU)
qq['y'] = qg.qb_epa.values / qg.qb_db.values; qq['w'] = qg.qb_db.values
tr = qq[qq.season <= 2017]
fq = sm.WLS(tr.y, sm.add_constant(tr[['qb_epa_db', 'qb_cpoe']]), weights=tr.w).fit()
log(f'\nQB composite fit on 2016-17 games: EPA/db = {fq.params.const:.3f} + {fq.params.qb_epa_db:.2f}*EPA grade + {fq.params.qb_cpoe:.4f}*CPOE grade '
    f'(t {fq.tvalues.qb_epa_db:.1f}, {fq.tvalues.qb_cpoe:.1f})')
R['qb_comp'] = MU['qb_epa_db'] + fq.params.qb_epa_db * (R.qb_epa_db - MU['qb_epa_db']) + fq.params.qb_cpoe * (R.qb_cpoe - MU['qb_cpoe'])
# out-of-sample check of every grade: does the pre-game grade predict the player's rate in that game, 2018-25?
log('\nPre-game grade vs that game\'s rate, 2018-25 (denominator-weighted corr; player-games with den >= 5):')
chkrows = pg[pg.season.between(2018, 2025)][['gsis', 'date', 'season']].copy()
cg = lookup(chkrows, st, K, MU); pgt = pg[pg.season.between(2018, 2025)].reset_index(drop=True)
for g_, (n, d, _) in GRADES.items():
    m = (pgt[d] >= 5).values
    if m.sum() < 100: continue
    yv = (pgt[n] / pgt[d].where(pgt[d] > 0)).values[m]; xv = cg[g_].values[m]; wv = pgt[d].values[m]
    xm, ym = np.average(xv, weights=wv), np.average(yv, weights=wv)
    r = np.average((xv - xm) * (yv - ym), weights=wv) / np.sqrt(np.average((xv - xm) ** 2, weights=wv) * np.average((yv - ym) ** 2, weights=wv))
    log(f'  {g_:14s} r {r:+.3f}  (n {m.sum()})')
PRC = float(tg[tg.season <= 2017].ppcost.sum() / pg[pg.season <= 2017].pr_pbp.sum())
log(f'EPA per credited pass-rush event (sack/hit), 2016-17: {PRC:.2f}')
a = pg[pg.grp.notna()].copy(); a['pct'] = np.where(a.grp.isin(DEF_G), a.defense_pct, a.offense_pct)
slots = a.groupby(['game_id', 'team', 'grp']).pct.sum().groupby('grp').median()
log('position-group slots (median summed snap share):', slots.round(2).to_dict())
tgs = tg.sort_values('date').copy()
for c, src in [('v_db', 'db'), ('v_tgt', 'tgts'), ('v_car', 'dcar'), ('v_plays', 'plays')]:
    tgs[c] = tgs.groupby('team')[src].transform(lambda s: s.shift().ewm(halflife=8, min_periods=1).mean())
od = tg.sort_values('date').copy(); od['v_odb'] = od.groupby('opp').db.transform(lambda s: s.shift().ewm(halflife=8, min_periods=1).mean())
tm_vol = tgs[['game_id', 'team', 'v_db', 'v_tgt', 'v_car', 'v_plays']].merge(od[['game_id', 'opp', 'v_odb']].rename(columns={'opp': 'team'}), on=['game_id', 'team'])
C = team_components(R, slots, tm_vol)
C.to_parquet(f'{CACHE}/components.parquet')
COMP = [c for c in C.columns if c[:2] in ('O_', 'D_', 'NE')]
W = C.pivot_table(index=['game_id', 'team'], columns='kind', values=COMP)
W.columns = [f'{a}_{b}' for a, b in W.columns]; W = W.reset_index()
for c in COMP: W[f'{c}_delta'] = W[f'{c}_exp'] - W[f'{c}_base']

TGt = tg.merge(tg[['game_id', 'team', 'epa']].rename(columns={'team': 'opp', 'epa': 'epa_a'}), on=['game_id', 'opp'])
chk = W.merge(TGt[['game_id', 'team', 'epa', 'epa_a', 'season']], on=['game_id', 'team'])
chk = chk[chk.season.between(2018, 2025)]
log('\n=== 2. Team components (EPA/game vs average). sd across team-games; corr of the EXPECTED-roster value with that game\'s\n'
    'actual offensive EPA (O_*), EPA allowed (D_*), or net EPA (NET_*), 2018-25')
for c in COMP:
    y = chk.epa if c.startswith('O_') else (chk.epa_a if c.startswith('D_') else chk.epa - chk.epa_a)
    log(f'  {c:16s} sd {chk[c + "_exp"].std():5.2f}  delta sd {chk[c + "_delta"].std():5.2f}  r(actual) exp {np.corrcoef(chk[c + "_exp"], y)[0, 1]:+.3f}  delta {np.corrcoef(chk[c + "_delta"], y)[0, 1]:+.3f}')

# ------------------------------------------------------------------ 4. game tests
g = games.merge(W.add_prefix('h_').rename(columns={'h_game_id': 'game_id', 'h_team': 'home_team'}), on=['game_id', 'home_team'], how='left')
g = g.merge(W.add_prefix('a_').rename(columns={'a_game_id': 'game_id', 'a_team': 'away_team'}), on=['game_id', 'away_team'], how='left')
g = g[g.margin.notna()].copy()
g['est'] = -0.429 + 0.51 * g.rt_pred + 0.561 * g.mkt_pred
for c in COMP:
    for k in ['exp', 'delta']:
        g[f'x_{c}_{k}'] = (g[f'a_{c}_{k}'] - g[f'h_{c}_{k}']) if c.startswith('D_') else (g[f'h_{c}_{k}'] - g[f'a_{c}_{k}'])
        g[f't_{c}_{k}'] = g[f'h_{c}_{k}'] + g[f'a_{c}_{k}']
lg = pd.concat([g[['game_id', 'date', 'home_team', 'home_qb_id']].set_axis(['gid', 'd', 'team', 'qb'], axis=1),
                g[['game_id', 'date', 'away_team', 'away_qb_id']].set_axis(['gid', 'd', 'team', 'qb'], axis=1)]).sort_values('d')
lg['chg'] = lg.groupby('team').qb.shift().ne(lg.qb) & lg.groupby('team').cumcount().gt(0)
g['qb_change'] = g.game_id.map(lg.groupby('gid').chg.any()).fillna(False).astype(bool)
ov = R[R.kind.isin(['exp', 'base'])].pivot_table(index=['game_id', 'team', 'gsis'], columns='kind', values='share', aggfunc='sum').fillna(0)
ov['mn'] = np.minimum(ov.exp.clip(0, 1), ov.base.clip(0, 1))
tov = (1 - ov.groupby(['game_id', 'team']).mn.sum() / ov.groupby(['game_id', 'team']).base.sum().clip(lower=1e-6)).rename('turn').reset_index()
g = g.merge(tov.rename(columns={'team': 'home_team', 'turn': 'h_turn'}), on=['game_id', 'home_team'], how='left')
g = g.merge(tov.rename(columns={'team': 'away_team', 'turn': 'a_turn'}), on=['game_id', 'away_team'], how='left')
g['turn'] = g[['h_turn', 'a_turn']].max(axis=1)

T = g[g.season.between(2017, 2025)].copy()
TEST = T.season >= 2018
thr = T.loc[TEST, 'turn'].quantile(0.8)
log(f'\nturnover = share of the team\'s recent snap-weighted roster not expected to play (max of the two teams); top-20% threshold {thr:.2f} '
    f'(week 1 median {T[TEST & (T.week == 1)].turn.median():.2f}, weeks 5+ median {T[TEST & (T.week >= 5)].turn.median():.2f})')
SEG = {'all': TEST, 'weeks 1-4': TEST & (T.week <= 4), 'weeks 5-18': TEST & T.week.between(5, 18) & T.game_type.eq('REG'),
       'week 1': TEST & (T.week == 1), 'QB change': TEST & T.qb_change, 'high turnover': TEST & (T.turn >= thr),
       'wk1-4 hi-turn': TEST & (T.week <= 4) & (T.turn >= T.loc[TEST & (T.week <= 4), 'turn'].median())}
log('segment sizes: ' + ', '.join(f'{k} {int(v.sum())}' for k, v in SEG.items()))

def wf(df, feats, y='res', start=2018):
    p = pd.Series(np.nan, index=df.index); coefs = []
    for s in range(start, 2026):
        tr, te = df[(df.season < s) & df[feats].notna().all(axis=1)], df[df.season == s]
        f = sm.OLS(tr[y], tr[feats].astype(float)).fit()      # no intercept: residual model on top of the base
        p[te.index] = te[feats].fillna(0).astype(float).values @ f.params.values; coefs.append(f.params.values)
    return p, np.array(coefs)

def ll_vec(pred, margin, sd):
    pr = norm.cdf(pred / sd).clip(1e-6, 1 - 1e-6); yy = (margin > 0)
    return np.where(margin == 0, np.nan, -(yy * np.log(pr) + (~yy) * np.log(1 - pr)))

SD = 13.5
T['res'] = T.margin - T.est
# second baseline: est + the existing QB term (qb_val_diff, qb_delta_diff from game_model.py), walk-forward residual
TQ = g[g.season <= 2025].assign(res=g.margin - g.est)
pq, _ = wf(TQ, ['qb_val_diff', 'qb_delta_diff'], start=2017); T['estq'] = T.est + pq.reindex(T.index)
BASES = {'est': T.est, 'est+QB term': T.estq}
log(f'\n=== 3. Game margin, walk-forward 2018-25. Base A = est (-0.429 + 0.51*rt_pred + 0.561*mkt_pred). Base B = est + existing QB term\n'
    f'(qb_val_diff, qb_delta_diff). Roster features enter as a residual on top of the base, coefficient refit each season on\n'
    f'2017..s-1. Win prob = Phi(pred/{SD}). Cells: RMSE change / log-loss change x1000 (negative = better); t = paired t of squared-error change')
for bn, bp in BASES.items():
    log(f'{"  base " + bn:32s}' + ''.join(f'{k[:13]:>21s}' for k in SEG))
    log(f'{"  RMSE / logloss":32s}' + ''.join(f'{np.sqrt(((T.margin[m] - bp[m]) ** 2).mean()):>10.3f}/{np.nanmean(ll_vec(bp[m], T.margin[m], SD)):<10.4f}' for m in SEG.values()))
log(f'{"  closing spread (reference)":32s}' + ''.join(f'{np.sqrt(((T.margin[m] - T.spread_line[m]) ** 2).mean()):>10.3f}/{np.nanmean(ll_vec(T.spread_line[m], T.margin[m], SD)):<10.4f}' for m in SEG.values()))
CANDS = [[f'x_{c}_{k}'] for k in ['delta', 'exp'] for c in COMP] + [
    ['x_O_QB_delta', 'x_O_PASSGAME_delta', 'x_O_RUSH_delta', 'x_O_OL_delta', 'x_D_PASSRUSH_delta', 'x_D_COVERAGE_delta'],
    ['x_NET_UNITS_delta', 'x_NET_ONFIELD_delta'], ['x_O_QB_delta', 'x_NET_ONFIELD_delta'],
    ['x_O_PASSGAME_cov_delta', 'x_O_TPRR_delta', 'x_D_PASSRUSH_delta'], ['x_NET_UNITS_delta', 'x_NET_UNITS_exp']]
res_rows = []; best = {}
for bn, bp in BASES.items():
    log(f'\n  -- on top of base {bn} --')
    TT = T.assign(res=T.margin - bp)
    for feats in CANDS:
        p, cf = wf(TT, feats); pred = bp + p
        name = '+'.join(f[2:] for f in feats)
        row = dict(base=bn, feats=name, coef_2025fit=np.round(cf[-1], 3).tolist()); line = f'  {name[:30]:30s}'
        for sname, m in SEG.items():
            e0 = (T.margin[m] - bp[m]) ** 2; e1 = (T.margin[m] - pred[m]) ** 2
            d_r = np.sqrt(e1.mean()) - np.sqrt(e0.mean()); dd = e1 - e0; tt = dd.mean() / (dd.std() / np.sqrt(len(dd)))
            d_l = 1000 * (np.nanmean(ll_vec(pred[m], T.margin[m], SD)) - np.nanmean(ll_vec(bp[m], T.margin[m], SD)))
            row.update({f'{sname}_drmse': round(d_r, 4), f'{sname}_dll': round(d_l, 3), f'{sname}_t': round(tt, 2)})
            line += f'{d_r:>+8.3f}/{d_l:<+6.1f}t{tt:<+5.1f}'
        log(line + f' b={row["coef_2025fit"]}'); res_rows.append(row); best[(bn, name)] = pred

log('\nprojection SD weeks 1-4 (2018-25): est %.2f, est+QB term %.2f, est+QB term+NET_UNITS_delta %.2f, closing spread %.2f' % (
    T.est[SEG['weeks 1-4']].std(), T.estq[SEG['weeks 1-4']].std(), best[('est+QB term', 'NET_UNITS_delta')][SEG['weeks 1-4']].std(), T.spread_line[SEG['weeks 1-4']].std()))

log('\n=== 4. vs closing spread, 2018-25: (margin - spread_line) ~ feature [coef, t, HC1]; with (est - spread) also in;\n'
    'and how much the closing line already moved off est per unit of the feature')
for c in COMP:
    for k in ['delta', 'exp']:
        x = f'x_{c}_{k}'; m = TEST & T[x].notna()
        f1 = sm.OLS((T.margin - T.spread_line)[m], sm.add_constant(T[[x]][m])).fit(cov_type='HC1')
        f2 = sm.OLS((T.margin - T.spread_line)[m], sm.add_constant(pd.concat([T[[x]], (T.estq - T.spread_line).rename('est_gap')], axis=1)[m])).fit(cov_type='HC1')
        f3 = sm.OLS(T.spread_line[m] - T.estq[m], sm.add_constant(T[[x]][m])).fit(cov_type='HC1')
        f4 = sm.OLS((T.margin - T.spread_line)[m & (T.week <= 4)], sm.add_constant(T[[x]][m & (T.week <= 4)])).fit(cov_type='HC1')
        log(f'  {x[2:]:22s} resid coef {f1.params[x]:+.3f} (t {f1.tvalues[x]:+.2f}) wk1-4 {f4.params[x]:+.3f} (t {f4.tvalues[x]:+.2f}) | with est gap {f2.params[x]:+.3f} (t {f2.tvalues[x]:+.2f}) | '
            f'line moved off est+QB {f3.params[x]:+.3f}/unit (t {f3.tvalues[x]:+.1f})')

log('\n=== 5. Totals, 2018-25: sum of both teams\' grades vs (total - total_line); walk-forward total_line + b*x')
T['tres'] = T.total - T.total_line
for c in ['O_UNITS', 'O_ONFIELD', 'O_QB', 'O_PASSGAME', 'O_PASSGAME_cov', 'O_TPRR', 'O_OL', 'D_UNITS', 'D_PASSRUSH', 'D_ONFIELD']:
    for k in ['exp', 'delta']:
        x = f't_{c}_{k}'; m = TEST & T[x].notna()
        f1 = sm.OLS(T.tres[m], sm.add_constant(T[[x]][m])).fit(cov_type='HC1')
        p, cf = wf(T.assign(res=T.tres), [x]); pr = T.total_line + p
        e4 = SEG['weeks 1-4']
        r0, r1 = [np.sqrt(((T.total - z)[TEST] ** 2).mean()) for z in (T.total_line, pr)]
        r0e, r1e = [np.sqrt(((T.total - z)[e4] ** 2).mean()) for z in (T.total_line, pr)]
        log(f'  {x[2:]:20s} coef {f1.params[x]:+.3f} (t {f1.tvalues[x]:+.2f}); RMSE {r0:.3f} -> {r1:.3f} ({r1 - r0:+.3f}); weeks 1-4 {r0e:.3f} -> {r1e:.3f} ({r1e - r0e:+.3f})')
Xc = ['t_O_UNITS_exp', 't_D_UNITS_exp', 't_O_ONFIELD_exp', 't_D_ONFIELD_exp']
pm = pd.Series(np.nan, index=T.index); base_t = pd.Series(np.nan, index=T.index)
for s in range(2018, 2026):
    tr, te = T[T.season < s], T[T.season == s]
    mu_t = tr[tr.season == s - 1].total.mean()
    f = sm.OLS(tr.total - tr.groupby('season').total.transform('mean'), tr[Xc]).fit()
    pm[te.index] = mu_t + te[Xc].values @ f.params.values; base_t[te.index] = mu_t
log(f'  no-market check: total = last season mean + grades ({", ".join(x[2:] for x in Xc)}): RMSE {np.sqrt(((T.total - base_t)[TEST] ** 2).mean()):.3f} -> '
    f'{np.sqrt(((T.total - pm)[TEST] ** 2).mean()):.3f}; total_line {np.sqrt(((T.total - T.total_line)[TEST] ** 2).mean()):.3f}')

log('\n=== 6. Big discrepancies, 2018-25: roster-implied margin (walk-forward OLS margin ~ grades + neutral) vs base; top 5% |gap|')
for bn, bp in BASES.items():
    for feats in [['x_NET_UNITS_exp'], ['x_NET_ONFIELD_exp'], ['x_O_QB_exp', 'x_NET_UNITS_exp', 'x_NET_ONFIELD_exp']]:
        pr = pd.Series(np.nan, index=T.index)
        for s in range(2018, 2026):
            tr, te = T[T.season < s], T[T.season == s]
            f = sm.OLS(tr.margin, sm.add_constant(tr[feats + ['neutral']].astype(float))).fit()
            pr[te.index] = sm.add_constant(te[feats + ['neutral']].astype(float), has_constant='add').values @ f.params.values
        gap = (pr - bp)[TEST]; top = gap.abs() >= gap.abs().quantile(0.95); idx = gap[top].index
        sgn = np.sign(gap[top]); act = (T.margin - bp)[idx] * sgn
        log(f'  base {bn}, {"+".join(f[2:] for f in feats)}: n {top.sum()}, |gap| mean {gap[top].abs().mean():.1f} pts (>= {gap[top].abs().min():.1f}); '
            f'result beat the base toward the roster side by {act.mean():+.2f} pts (se {act.std() / np.sqrt(len(act)):.2f}), {(act > 0).mean():.0%} of games; '
            f'RMSE base {np.sqrt(((T.margin - bp)[idx] ** 2).mean()):.2f}, roster-only {np.sqrt(((T.margin - pr)[idx] ** 2).mean()):.2f}, spread '
            f'{np.sqrt(((T.margin - T.spread_line)[idx] ** 2).mean()):.2f}; closing line sided with roster by {((T.spread_line - bp)[idx] * sgn).mean():+.2f} pts; '
            f'weeks 1-4 {(T.week[idx] <= 4).mean():.0%}')

pd.DataFrame(res_rows).to_csv(f'{CACHE}/roster_grade_tests.csv', index=False)
rep.to_csv(f'{CACHE}/grade_reliability.csv', index=False)
open(LOGPATH, 'w').write('\n'.join(LOG) + '\n')
print('log ->', LOGPATH)
