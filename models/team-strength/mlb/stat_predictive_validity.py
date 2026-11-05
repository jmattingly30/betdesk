"""Which MLB team stats over games 1-81 predict run differential and win % in games 82-162? (2014-2025, no 2020)
Stats from Retrosheet plate appearances: wOBA for/against, K%, BB%, HR rate, plus runs, Pythagorean, one-run record."""
import sys, numpy as np, pandas as pd
D = sys.argv[1]
pa = pd.read_parquet(f'{D}/mlb_pa.parquet'); G = pd.read_parquet(f'{D}/mlb_features.parquet')
W = {'BB': 0.69, 'HBP': 0.72, '1B': 0.89, '2B': 1.27, '3B': 1.62, 'HR': 2.10}
pa['w'] = pa.res.map(W).fillna(0); pa['k'] = pa.res.eq('K'); pa['bb'] = pa.res.isin(['BB','HBP']); pa['hr'] = pa.res.eq('HR')
pa['bip'] = ~pa.res.isin(['K','BB','HBP','HR','IBB']); pa['hit_bip'] = pa.res.isin(['1B','2B','3B'])
pa = pa[pa.res != 'IBB']
t = G.set_index('gid')[['home','away']]
pa['bat_team'] = np.where(pa.bat_home == 1, t.home.reindex(pa.gid).values, t.away.reindex(pa.gid).values)
pa['pit_team'] = np.where(pa.bat_home == 1, t.away.reindex(pa.gid).values, t.home.reindex(pa.gid).values)
off = pa.groupby(['gid','bat_team']).agg(pa_=('w','size'), w=('w','sum'), k=('k','sum'), bb=('bb','sum'), hr=('hr','sum'), bip=('bip','sum'), hbip=('hit_bip','sum')).reset_index()
long = pd.concat([G[['gid','date','season','home','away','hscore','vscore']].set_axis(['gid','date','season','team','opp','rf','ra'], axis=1),
                  G[['gid','date','season','away','home','vscore','hscore']].set_axis(['gid','date','season','team','opp','rf','ra'], axis=1)])
long = long.merge(off.rename(columns={'bat_team':'team'}), on=['gid','team']).merge(
       off.rename(columns={'bat_team':'opp'}).add_prefix('o_').rename(columns={'o_gid':'gid','o_opp':'opp'}), on=['gid','opp'])
long = long[long.season != 2020].sort_values(['season','team','date'])
long['n'] = long.groupby(['season','team']).cumcount()
long['win'] = (long.rf > long.ra).astype(float); long['onerun_win'] = np.where((long.rf - long.ra).abs() == 1, long.win, np.nan)
first, rest = long[long.n < 81], long[long.n >= 81]
def agg(x):
    s = x.groupby(['season','team']).sum(numeric_only=True); m = x.groupby(['season','team'])
    out = pd.DataFrame({'run_diff_pg': (s.rf - s.ra)/m.size(), 'win_pct': m.win.mean(),
        'pythag': s.rf**1.83/(s.rf**1.83 + s.ra**1.83), 'woba_for': s.w/s.pa_, 'woba_against': s.o_w/s.o_pa_,
        'net_woba': s.w/s.pa_ - s.o_w/s.o_pa_, 'k_pct_pitching': s.o_k/s.o_pa_, 'bb_pct_pitching': s.o_bb/s.o_pa_,
        'kbb_pitching': (s.o_k - s.o_bb)/s.o_pa_, 'kbb_hitting': (s.k - s.bb)/s.pa_, 'hr_rate_allowed': s.o_hr/s.o_pa_, 'hr_rate_hit': s.hr/s.pa_,
        'babip_for': s.hbip/s.bip, 'babip_against': s.o_hbip/s.o_bip, 'runs_per_woba_pt': (s.rf/s.pa_)/(s.w/s.pa_), 'one_run_win_pct': m.onerun_win.mean()})
    return out
F = agg(first.astype({'k':float,'bb':float,'hr':float,'o_k':float,'o_bb':float,'o_hr':float}))
R = rest.groupby(['season','team']).agg(rf=('rf','sum'), ra=('ra','sum'), g=('rf','size'), fut_win=('win','mean'))
R['fut_rd'] = (R.rf - R.ra)/R.g
X = F.join(R[['fut_rd','fut_win']], how='inner')
odd = agg(long[long.n % 2 == 1].astype(float, errors='ignore')); even = agg(long[long.n % 2 == 0].astype(float, errors='ignore'))
rows = [dict(stat=c, r_future_run_diff=X[c].corr(X.fut_rd), r_future_win=X[c].corr(X.fut_win), split_half_reliability=odd[c].corr(even[c])) for c in F.columns]
res = pd.DataFrame(rows).sort_values('r_future_run_diff', key=abs, ascending=False); print(res.round(3).to_string(index=False))
res.to_csv(sys.argv[2], index=False)
import statsmodels.api as sm
Z = X[['run_diff_pg','net_woba','kbb_pitching','babip_against','one_run_win_pct','fut_rd']].dropna()
Zs = Z.drop(columns='fut_rd'); Zs = (Zs - Zs.mean())/Zs.std()
print(sm.OLS(Z.fut_rd, sm.add_constant(Zs)).fit(cov_type='HC1').summary().tables[1])
