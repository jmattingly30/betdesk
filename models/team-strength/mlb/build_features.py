"""Point-in-time MLB features per game (2014-2025): shrunk starter quality, lineup quality, platoon edge, bullpen quality,
bullpen fatigue, team run-differential rating, park, weather, umpire; joined to scores (game logs) and SBR odds (2021-2025)."""
import sys, glob, numpy as np, pandas as pd
sys.path.insert(0, __file__.rsplit('/', 2)[0])
from common import margin_ratings
D = sys.argv[1]; out = sys.argv[2]
pa = pd.read_parquet(f'{D}/mlb_pa.parquet'); G = pd.read_parquet(f'{D}/mlb_games_ev.parquet')
W = {'BB': 0.69, 'IBB': 0.0, 'HBP': 0.72, '1B': 0.89, '2B': 1.27, '3B': 1.62, 'HR': 2.10}
pa = pa[pa.res != 'IBB'].copy()
pa['w'] = pa.res.map(W).fillna(0.0); pa['k'] = (pa.res == 'K').astype(int); pa['bb'] = pa.res.isin(['BB','HBP']).astype(int); pa['hr'] = (pa.res == 'HR').astype(int)
pa['season'] = pa.date.dt.year
lg = pa.groupby('season').agg(w=('w','mean'), k=('k','mean'), bb=('bb','mean'), hr=('hr','mean'))
def rolling_rating(df, who, prior_n, decay=0.6):
    """Shrunk per-PA rates for `who` as of the morning of each date: (cur season to date + decay*last + decay^2*two ago + prior) / n."""
    d = df.groupby([who, 'season', 'date']).agg(n=('w','size'), w=('w','sum'), k=('k','sum'), bb=('bb','sum'), hr=('hr','sum')).reset_index().sort_values([who,'date'])
    for c in ['n','w','k','bb','hr']:
        d['c_'+c] = d.groupby([who,'season'])[c].cumsum() - d[c]          # through previous day
    S = d.groupby([who,'season'])[['n','w','k','bb','hr']].sum()
    for lag, wt in ((1, decay), (2, decay**2)):
        prev = S.copy(); prev.index = prev.index.set_levels(prev.index.levels[1] + lag, level=1)
        prev = prev.reindex(pd.MultiIndex.from_arrays([d[who], d.season])).fillna(0).values
        for j, c in enumerate(['n','w','k','bb','hr']): d['c_'+c] = d['c_'+c] + wt*prev[:, j]
    L = lg.reindex(d.season)
    n = d.c_n + prior_n
    d['r_woba'] = (d.c_w + prior_n*L.w.values)/n; d['r_k'] = (d.c_k + prior_n*L.k.values)/n
    d['r_bb'] = (d.c_bb + prior_n*L.bb.values)/n; d['r_hr'] = (d.c_hr + prior_n*L.hr.values)/n; d['r_n'] = d.c_n
    d['r_kbb'] = d.r_k - d.r_bb
    return d.set_index([who,'date'])[['r_woba','r_k','r_bb','r_hr','r_kbb','r_n']]
bat = rolling_rating(pa, 'batter', 220); pit = rolling_rating(pa, 'pitcher', 250)
# 'raw' versions mimic a season-to-date sheet: current season only, almost no shrinkage
bat_raw = rolling_rating(pa, 'batter', 10, decay=0.0); pit_raw = rolling_rating(pa, 'pitcher', 10, decay=0.0)
def lookup(tab, ids, dates, col):
    """Rating as of `dates` for players `ids`; for a first appearance that day use the latest earlier value (asof)."""
    x = tab[col].reindex(pd.MultiIndex.from_arrays([ids, dates])).values.copy()
    miss = np.isnan(x) & pd.notna(np.asarray(ids))
    if miss.any():
        t = tab[[col]].reset_index().rename(columns={tab.index.names[0]:'pid'}).sort_values('date')
        q = pd.DataFrame({'pid': np.asarray(ids)[miss], 'date': np.asarray(dates)[miss], 'i': np.where(miss)[0]}).sort_values('date')
        m = pd.merge_asof(q, t, on='date', by='pid', allow_exact_matches=False)
        x[m.i.values] = m[col].values
    return x
G['season'] = G.date.dt.year
for side in ('home','away'):
    for c in ['r_woba','r_kbb','r_n']:
        G[f'{side}_sp_{c}'] = lookup(pit, G[f'{side}_sp'], G.date, c)
    G[f'{side}_sp_raw_woba'] = lookup(pit_raw, G[f'{side}_sp'], G.date, 'r_woba')
    rv = np.column_stack([lookup(bat_raw, G[f'{side}_b{i}'], G.date, 'r_woba') for i in range(1, 10)])
    G[f'{side}_lineup_raw_woba'] = np.nanmean(rv, 1)
    vals = np.column_stack([lookup(bat, G[f'{side}_b{i}'], G.date, 'r_woba') for i in range(1, 10)])
    L = lg.w.reindex(G.season).values
    vals = np.where(np.isnan(vals), L[:, None]*0.92, vals)               # unknown debutants: below-average prior
    G[f'{side}_lineup_woba'] = vals.mean(1)
# fill unknown starters with a replacement-level prior
for side in ('home','away'):
    G[f'{side}_sp_r_woba'] = G[f'{side}_sp_r_woba'].fillna(pd.Series(lg.w.reindex(G.season).values*1.05, index=G.index))
    G[f'{side}_sp_r_kbb'] = G[f'{side}_sp_r_kbb'].fillna(pd.Series((lg.k - lg.bb).reindex(G.season).values*0.8, index=G.index))
    G[f'{side}_sp_r_n'] = G[f'{side}_sp_r_n'].fillna(0)
# handedness and platoon advantage share of lineup
bio = pd.read_csv(f'{D}/retroev/biofile.csv', usecols=['PLAYERID','BATS','THROWS']).set_index('PLAYERID')
for side, opp in (('home','away'), ('away','home')):
    th = G[f'{opp}_sp'].map(bio.THROWS)
    adv = np.zeros(len(G))
    for i in range(1, 10):
        b = G[f'{side}_b{i}'].map(bio.BATS)
        adv += ((b == 'B') | ((b == 'L') & (th == 'R')) | ((b == 'R') & (th == 'L'))).astype(int).values
    G[f'{side}_platoon_share'] = adv/9
    G[f'{opp}_sp_lefty'] = (th == 'L').astype(int)
# bullpen quality (team relievers, decayed) and fatigue (relief PA faced in the previous 3 days)
gid_side = pa[['gid','bat_home']].copy()
G['home_team'] = G.hometeam; G['away_team'] = G.visteam
tmap = G.set_index('gid')[['home_team','away_team']]
rel = pa[pa.vs_starter == 0].copy()
rel['pteam'] = np.where(rel.bat_home == 1, tmap.away_team.reindex(rel.gid).values, tmap.home_team.reindex(rel.gid).values)
bp = rolling_rating(rel.rename(columns={'pteam':'tm'}), 'tm', 600)
daily = rel.groupby(['pteam','date']).size().rename('rpa').reset_index()
daily = daily.set_index('date').groupby('pteam').rpa.apply(lambda s: s.asfreq('D', fill_value=0).rolling(3, min_periods=1).sum().shift(1)).rename('bp_load3').reset_index()
for side in ('home','away'):
    G[f'{side}_bp_woba'] = lookup(bp, G[f'{side}_team'], G.date, 'r_woba')
    G[f'{side}_bp_load3'] = daily.set_index(['pteam','date']).bp_load3.reindex(pd.MultiIndex.from_arrays([G[f'{side}_team'], G.date])).fillna(0).values
# scores and context from game logs
gl = []
for f in sorted(glob.glob(f'{D}/retro/GL*.TXT')):
    x = pd.read_csv(f, header=None, usecols=[0,1,3,6,9,10,12,16], names=['date','number','visteam','hometeam','vscore','hscore','daynight_gl','park'], dtype={0:str})
    gl.append(x)
gl = pd.concat(gl); gl['date'] = pd.to_datetime(gl.date, format='%Y%m%d'); gl['number'] = gl.number.astype(str)
G = G.merge(gl, on=['date','hometeam','visteam','number'], how='inner')
G['margin'] = G.hscore - G.vscore; G['total'] = G.hscore + G.vscore
G = G.sort_values(['date','gid']).reset_index(drop=True)
G['home'], G['away'] = G.home_team, G.away_team
G = margin_ratings(G, k=0.012, hfa=0.2, carry=0.5, out='rt')
# park factor: runs per game at park vs league, previous 3 seasons + season to date, shrunk
pk = G.groupby(['park','season']).agg(r=('total','sum'), n=('total','size')).reset_index()
lgr = G.groupby('season').total.mean()
def park_factor(row):
    h = pk[(pk.park == row.park) & pk.season.between(row.season - 3, row.season - 1)]
    r, n = h.r.sum(), h.n.sum(); lgm = lgr.reindex(h.season).mean() if len(h) else lgr.get(row.season - 1, 9)
    return (r + 200*lgm)/(n + 200)/lgm if n else 1.0
pf = pk[['park','season']].drop_duplicates().copy(); pf['pf'] = pf.apply(park_factor, axis=1)
G = G.merge(pf, on=['park','season'], how='left')
G['temp'] = pd.to_numeric(G.temp, errors='coerce').where(lambda s: s > 0); G['windspeed'] = pd.to_numeric(G.windspeed, errors='coerce').where(lambda s: s >= 0)
G['wind_out'] = G.winddir.isin(['tocf','tolf','torf']).astype(int)*G.windspeed.fillna(0)
G['wind_in'] = G.winddir.isin(['fromcf','fromlf','fromrf']).astype(int)*G.windspeed.fillna(0)
# umpire run tendency (walk-forward, shrunk): his previous games' total runs vs league
G['ump_runs'] = G.groupby('umphome').total.transform(lambda s: s.shift().expanding().sum()).fillna(0)
G['ump_n'] = G.groupby('umphome').cumcount()
G['ump_factor'] = (G.ump_runs + 150*G.season.map(lgr))/((G.ump_n + 150)*G.season.map(lgr))
G['day_game'] = (G.daynight == 'day').astype(int)
G.to_parquet(out)
print(G.shape); print(G[['home_sp_r_woba','home_lineup_woba','home_bp_woba','home_platoon_share','pf','temp','ump_factor','home_bp_load3']].describe().round(3).T)
