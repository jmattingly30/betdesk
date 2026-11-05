"""NFL-2 / NFL-3 upgrade test (2026-10-10, Edge research thread): does tracking ability that travels improve the
existing prop model?

python3 backtest_nfl23.py <model_dir> <ngs_dir>     (ngs_dir holds nflverse ngs_rushing.csv.gz, ngs_receiving.csv.gz)

NFL-2 rushing: + rush yards over expected per attempt (NGS, pre-game, decayed like the other rates, shrunk toward 0),
               + team-changer flag and its interaction with the old-team YPC (YPC belongs partly to the old line).
NFL-3 receiving: + aDOT and intended-air-yard share in the yards GLM, + YAC over expected per catch (NGS), + team-changer
               interaction with the old-team yards per target.
Same walk-forward as backtest.py: fit 2017-2023 -> test 2024, fit 2017-2024 -> test 2025. Compares MAE overall, for
team changers, and in weeks 1-6. Writes backtest_nfl23_log.txt and nfl23_features.parquet (new columns only)."""
import sys, os, numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import model as mdl
from features import decayed

M, NGS = sys.argv[1], sys.argv[2]
log = open(f'{M}/backtest_nfl23_log.txt', 'w')
def say(*a):
    s = ' '.join(str(x) for x in a); print(s, flush=True); log.write(s + '\n'); log.flush()

f = pd.read_parquet(f'{M}/data/features.parquet')
f['td'] = f.rec_td + f.rush_td
f = f.sort_values(['game_date', 'game_id', 'posteam', 'pid']).reset_index(drop=True)

# ---------- new pre-game features ----------
ru = pd.read_csv(f'{NGS}/ngs_rushing.csv.gz'); ru = ru[ru.week > 0]
ru = ru.groupby(['player_gsis_id', 'season', 'week'], as_index=False)[['rush_yards_over_expected', 'rush_attempts', 'expected_rush_yards']].sum()
rc = pd.read_csv(f'{NGS}/ngs_receiving.csv.gz'); rc = rc[rc.week > 0].copy()
rc['yacoe_tot'] = rc.avg_yac_above_expectation * rc.receptions
rc = rc.groupby(['player_gsis_id', 'season', 'week'], as_index=False)[['yacoe_tot', 'receptions']].sum().rename(columns={'receptions': 'ngs_rec'})
f = f.merge(ru.rename(columns={'player_gsis_id': 'pid'}), on=['pid', 'season', 'week'], how='left')
f = f.merge(rc.rename(columns={'player_gsis_id': 'pid'}), on=['pid', 'season', 'week'], how='left')
for c in ['rush_yards_over_expected', 'rush_attempts', 'expected_rush_yards', 'yacoe_tot', 'ngs_rec']: f[c] = f[c].fillna(0.0)
S, _ = decayed(f, 'pid', ['rush_yards_over_expected', 'rush_attempts', 'expected_rush_yards', 'yacoe_tot', 'ngs_rec'], 8)
f['ryoe_pa'] = S.rush_yards_over_expected / (S.rush_attempts + 60)            # shrunk toward 0 (60 attempts)
f['ryoe_n'] = S.rush_attempts
f['yacoe_pr'] = S.yacoe_tot / (S.ngs_rec + 40)
prev_team = f.groupby('pid').posteam.shift(1)
f['moved'] = (prev_team.notna() & (prev_team != f.posteam)).astype(int)
# "team changer" status lasts his first 8 games with the new team
f['_stint'] = (f.moved == 1).groupby(f.pid).cumsum()
f['games_with_team'] = f.groupby(['pid', '_stint']).cumcount()
f['changer'] = ((f._stint > 0) & (f.games_with_team < 8)).astype(int)
f.loc[f.groupby('pid').cumcount() == 0, 'changer'] = 0
f[['pid', 'game_id', 'ryoe_pa', 'ryoe_n', 'yacoe_pr', 'moved', 'changer']].to_parquet(f'{M}/data/nfl23_features.parquet', index=False)
say('team-changer player-games (first 8 with new team):', int(f.changer.sum()), '| RB rows with any NGS rushing history:', int(((f.position == 'RB') & (f.ryoe_n > 0)).sum()))

# ---------- the upgraded efficiency specs ----------
base_specs = mdl.eff_specs
def new_specs(df):
    sp = base_specs(df)
    y, opp, feats, mask = sp['rush_yds']
    feats = dict(feats); feats['ryoe'] = df.ryoe_pa.fillna(0).clip(-2, 2)
    feats['l_rate_chg'] = mdl._log(df.ypc) * df.changer
    feats['changer'] = df.changer.astype(float)
    sp['rush_yds'] = (y, opp, feats, mask)
    y, opp, feats, mask = sp['rec_yds']
    feats = dict(feats); feats['adot'] = df.adot.fillna(8).clip(0, 25) / 10
    feats['l_air'] = mdl._log(df.air_sh.fillna(0) + 0.02)
    feats['yacoe'] = df.yacoe_pr.fillna(0).clip(-3, 3)
    feats['l_rate_chg'] = mdl._log(df.ypt) * df.changer
    feats['changer'] = df.changer.astype(float)
    sp['rec_yds'] = (y, opp, feats, mask)
    return sp

BASE_OPP_X = list(mdl.OPP_X)
def run(specs, tr_seasons, te_seasons, opp_x=None):
    mdl.eff_specs = specs; mdl.OPP_X = opp_x or BASE_OPP_X
    m = mdl.fit(f[f.season.isin(tr_seasons)])
    p = mdl.predict(m, f[f.season.isin(te_seasons)])
    mdl.eff_specs = base_specs; mdl.OPP_X = BASE_OPP_X
    return m, p

res = {}
# variant 'opp': the new abilities also go into the opportunity model (do effective backs earn more carries?)
OPP_PLUS = BASE_OPP_X + ['ryoe_pa', 'yacoe_pr', 'changer', 'games_with_team']
for name, sp, ox in [('current', base_specs, None), ('nfl23', new_specs, None), ('opp', base_specs, OPP_PLUS)]:
    m24, p24 = run(sp, range(2017, 2024), [2024], ox)
    _, p25 = run(sp, range(2017, 2025), [2025], ox)
    res[name] = pd.concat([p24, p25]); res[name + '_m'] = m24
    say(f'\n{name} efficiency GLMs (fit 2017-2023):')
    for k in ('rush_yds', 'rec_yds'):
        mm = m24['eff'][k]; say(' ', k, {c: f'{v:+.3f} (p {mm.pvalues[c]:.3f})' for c, v in mm.params.items()})

a = res['current']
rel = (a.snap_l >= 0.25) & (a.games_prior >= 3)
rows = []
def cmp(label, mask, stat, vname):
    b = res[vname]; d = mask & rel
    ea, eb = (a[stat] - a[f'{stat}_hat'])[d].abs(), (b[stat] - b[f'{stat}_hat'])[d].abs()
    diff = ea - eb; se = diff.std() / np.sqrt(d.sum())
    rows.append(dict(variant=vname, subset=label, stat=stat, n=int(d.sum()), mae_current=round(ea.mean(), 3), mae_variant=round(eb.mean(), 3),
                     gain_pct=round(100 * (1 - eb.mean() / ea.mean()), 2), t=round(diff.mean() / se, 2)))
rb, wr = a.position == 'RB', a.position.isin(['WR', 'TE'])
early = a.week <= 6
for vname in ('nfl23', 'opp'):
    for lab, m_ in [('RB all', rb), ('RB team changers', rb & (a.changer == 1)), ('RB weeks 1-6', rb & early)]:
        for st in ('carries', 'rush_yds'): cmp(lab, m_, st, vname)
    for lab, m_ in [('WR/TE all', wr), ('WR/TE team changers', wr & (a.changer == 1)), ('WR/TE weeks 1-6', wr & early)]:
        for st in ('targets', 'rec_yds'): cmp(lab, m_, st, vname)
R = pd.DataFrame(rows)
say('\nOut of sample 2024-2025 (snaps >= 25%, 3+ prior games). gain = MAE improvement; t > 2 = real')
say(R.to_string(index=False))
R.to_csv(f'{M}/backtest_nfl23_accuracy.csv', index=False)
