"""Fit the coefficients J's baseball-win-tool can use in place of the Tester-sheet chain:
  runs_team = exp(b0 + b1*lineup_wOBA + b2*opp_SP_wOBA + b3*opp_SP_KBB + b4*opp_bullpen_wOBA + b5*home + b6*ln(park factor))
with all player rates shrunk as in build_features.py. Fitted on 2015-2025 (no 2020); win prob by Poisson/Skellam."""
import sys, json, numpy as np, pandas as pd, statsmodels.api as sm
from scipy.stats import poisson
G = pd.read_parquet(sys.argv[1]); G = G[(G.season >= 2015) & (G.season != 2020)]
rows = []
for side, opp, home, runs in (('home','away',1,'hscore'), ('away','home',0,'vscore')):
    rows.append(pd.DataFrame({'season': G.season, 'runs': G[runs], 'lineup': G[f'{side}_lineup_woba']*100, 'opp_sp': G[f'{opp}_sp_r_woba']*100,
        'opp_sp_kbb': G[f'{opp}_sp_r_kbb']*100, 'opp_bp': G[f'{opp}_bp_woba']*100, 'home': home, 'ln_pf': np.log(G.pf), 'gid': G.gid}))
L = pd.concat(rows)
X = sm.add_constant(L[['lineup','opp_sp','opp_sp_kbb','opp_bp','home','ln_pf']])
f = sm.GLM(L.runs, X, family=sm.families.Poisson()).fit()
print(f.summary().tables[1])
L['mu'] = f.predict(X)
print('mean predicted runs %.3f vs actual %.3f' % (L.mu.mean(), L.runs.mean()))
# win prob check with negative binomial-ish dispersion: compare Poisson vs observed variance
print('runs variance / mean = %.2f (Poisson = 1)' % (L.runs.var()/L.runs.mean()))
json.dump({'model': 'Poisson GLM on team runs per game, 2015-2025 excl. 2020', 'coefficients': f.params.round(5).to_dict(),
           'inputs': {'lineup': 'mean of 9 starters shrunk wOBA x100 (prior 220 PA of league avg; last two seasons weighted 0.6, 0.36)',
                      'opp_sp': 'starter shrunk wOBA allowed x100 (prior 250 PA)', 'opp_sp_kbb': 'starter shrunk (K-BB)/PA x100',
                      'opp_bp': 'team bullpen shrunk wOBA allowed x100 (prior 600 PA)', 'home': '1 if batting team is home', 'ln_pf': 'ln(3-yr shrunk park run factor)'},
           'overdispersion': float(L.runs.var()/L.runs.mean()),
           'note': 'Use a negative binomial or the hierarchical MC with stage-1 sd ~0.4-0.5 runs to match the observed spread; then blend 85/15 with the market price.'},
          open(sys.argv[2], 'w'), indent=1)
