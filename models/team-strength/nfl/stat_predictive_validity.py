"""Which NFL team stats in games 1-8 predict results in games 9+? (2006-2025 regular seasons)
For each stat: correlation with future point differential per game and future win %, and odd/even-game reliability.
Then a cross-validated multivariate model to see which stats carry independent signal."""
import sys, numpy as np, pandas as pd
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import GroupKFold, cross_val_predict
t = pd.read_csv(sys.argv[1]).sort_values(['season','team','week'])
t['margin'] = t.pf - t.pa; t['win'] = (t.margin > 0) + 0.5*(t.margin == 0)
t['to_margin'] = (t.d_ints + t.d_fumbles_lost) - (t.ints + t.fumbles_lost)
t['fum_rec_rate'] = 1 - (t.fumbles_lost + t.d_fumbles - t.d_fumbles_lost) / (t.fumbles + t.d_fumbles).replace(0, np.nan)  # share of all fumbles we recovered
t['close'] = t.margin.abs() <= 8
t['close_win'] = np.where(t.close, t.win, np.nan)
for k in ['epa','sr','pass_epa','rush_epa','early_epa','ng_epa','ng_sr','explosive','third_conv','rz_td_rate','start_y100']:
    t['net_'+k] = t[k] - t['d_'+k]
t['net_sack_rate'] = t.d_sack_rate - t.sack_rate
t['net_int'] = t.d_ints - t.ints
t['net_st_epa'] = t.st_epa - t.d_st_epa
t['n'] = t.groupby(['season','team']).cumcount()
STATS = {'point_diff':'margin','win_pct':'win','pythag_pts':None,'net_epa':'net_epa','net_success_rate':'net_sr','net_pass_epa':'net_pass_epa',
 'net_rush_epa':'net_rush_epa','net_early_down_epa':'net_early_epa','net_epa_non_garbage':'net_ng_epa','net_sr_non_garbage':'net_ng_sr',
 'off_epa':'epa','def_epa_allowed':'d_epa','off_pass_epa':'pass_epa','def_pass_epa_allowed':'d_pass_epa','off_rush_epa':'rush_epa','def_rush_epa_allowed':'d_rush_epa',
 'net_explosive_rate':'net_explosive','net_sack_rate':'net_sack_rate','net_3rd_down_conv':'net_third_conv','net_red_zone_td':'net_rz_td_rate',
 'turnover_margin':'to_margin','net_interceptions':'net_int','fumble_recovery_rate':'fum_rec_rate','close_game_win_pct':'close_win',
 'net_special_teams_epa':'net_st_epa','net_start_field_pos':'net_start_y100'}
first = t[t.n < 8]; rest = t[t.n >= 8]
F = first.groupby(['season','team']).agg({v:'mean' for v in STATS.values() if v})
F['pythag_pts'] = first.groupby(['season','team']).apply(lambda g: g.pf.sum()**2.37/(g.pf.sum()**2.37 + g.pa.sum()**2.37))
F.columns = [k for k, v in STATS.items() if v] + ['pythag_pts']
R = rest.groupby(['season','team']).agg(fut_margin=('margin','mean'), fut_win=('win','mean'))
odd = t[t.n % 2 == 1].groupby(['season','team']); even = t[t.n % 2 == 0].groupby(['season','team'])
rows = []
X = F.join(R, how='inner')
for k, v in STATS.items():
    col = v if v else None
    rel = np.nan
    if v:
        a, b = odd[v].mean(), even[v].mean(); ab = pd.concat([a, b], axis=1).dropna(); rel = ab.corr().iloc[0,1]
    rows.append(dict(stat=k, r_future_margin=X[k].corr(X.fut_margin), r_future_win=X[k].corr(X.fut_win), split_half_reliability=rel))
res = pd.DataFrame(rows).sort_values('r_future_margin', key=abs, ascending=False)
print(res.round(3).to_string(index=False))
# multivariate: which stats survive together (standardized ridge, CV by season)
feats = ['point_diff','net_epa','net_success_rate','net_pass_epa','net_rush_epa','net_early_down_epa','net_epa_non_garbage','turnover_margin',
         'fumble_recovery_rate','close_game_win_pct','net_special_teams_epa','net_3rd_down_conv','net_red_zone_td','net_sack_rate','net_explosive_rate']
Z = X[feats + ['fut_margin']].dropna(); Zs = (Z[feats] - Z[feats].mean()) / Z[feats].std()
grp = Z.index.get_level_values(0)
def cvr2(cols):
    m = RidgeCV(alphas=np.logspace(-2, 3, 20))
    p = cross_val_predict(m, Zs[cols], Z.fut_margin, cv=GroupKFold(5), groups=grp)
    return 1 - ((Z.fut_margin - p)**2).sum() / ((Z.fut_margin - Z.fut_margin.mean())**2).sum()
print('\nCV R^2 for future point diff/game:')
for name, cols in [('point diff only', ['point_diff']), ('net EPA only', ['net_epa']), ('net non-garbage EPA only', ['net_epa_non_garbage']),
                   ('point diff + net EPA', ['point_diff','net_epa']), ('pass/rush split + SR', ['net_pass_epa','net_rush_epa','net_success_rate']),
                   ('all stats', feats)]:
    print(f'  {name:28s} {cvr2(cols):.3f}')
m = RidgeCV(alphas=np.logspace(-2, 3, 20)).fit(Zs, Z.fut_margin)
print('\nstandardized ridge weights (all stats):'); print(pd.Series(m.coef_, feats).sort_values(key=abs, ascending=False).round(2).to_string())
res.to_csv(sys.argv[2], index=False)
