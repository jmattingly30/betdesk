"""NFL-4 scheme layer backtest (2026-10-11): the four options alone and together, against the current prop model.

python3 backtest_nfl4.py <model_dir>        (needs data/scheme_*.parquet from scheme_features.py)
Walk-forward like backtest.py: fit 2017-2023 -> test 2024, fit 2017-2024 -> test 2025. Options 1-3 add features to the
LightGBM opportunity models (targets, carries); option 4 adds them to the opportunity models and to the touchdown GLM.
Reports MAE (targets, receptions, receiving yards, carries, rushing yards) and anytime-TD log loss overall and on the
games each option is about: new play-caller teams in weeks 1-6, receivers facing a coverage mix far from their usual,
backs facing an unusual box mix, players whose expected personnel changes, and red-zone heavy roles.
python3 backtest_nfl4.py <model_dir> round2   targets-only combinations (after round 1)
Writes backtest_nfl4_<round>_log.txt, backtest_nfl4_<round>_accuracy.csv, data/backtest_nfl4_<round>_predictions.parquet."""
import sys, os, numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import model as mdl, scheme_features as sf

M = sys.argv[1]
ROUND_ = sys.argv[2] if len(sys.argv) > 2 else 'round1'
log = open(f'{M}/backtest_nfl4_{ROUND_}_log.txt', 'w')
def say(*a):
    s = ' '.join(str(x) for x in a); print(s, flush=True); log.write(s + '\n'); log.flush()

f = pd.read_parquet(f'{M}/data/features.parquet')
f['td'] = f.rec_td + f.rush_td
f = f[f.season >= 2017].sort_values(['game_date', 'game_id', 'posteam', 'pid']).reset_index(drop=True)
f = sf.add_scheme(f, M)

BASE_OPP_X, base_specs = list(mdl.OPP_X), mdl.eff_specs
def td_specs(df):
    sp = base_specs(df); y, opp, feats, mask = sp['td']; feats = dict(feats)
    feats['l_rzx'] = mdl._log(df.rz_exp.fillna(df.rz_exp.median()) + 0.01); feats['l_gl'] = mdl._log(df.gl_sh.fillna(0.02) + 0.01)
    sp['td'] = (y, opp, feats, mask); return sp

ALLM = lambda xs: {'targets': xs, 'carries': xs, 'pass_att': xs}
X1, X2, X3, X4 = (sf.SCHEME_X[k] for k in (1, 2, 3, 4))
ROUND = sys.argv[2] if len(sys.argv) > 2 else 'round1'
if ROUND == 'round1':     # each option in every opportunity model, and all four together
    VARIANTS = {'current': ({}, False), 'opt1 caller priors': (ALLM(X1), False), 'opt2 coverage/box': (ALLM(X2), False),
                'opt3 game plan': (ALLM(X3), False), 'opt4 red zone': (ALLM(X4), True), 'all four': (ALLM(X1 + X2 + X3 + X4), True)}
else:                     # round 2: round 1 showed the gains are in TARGETS and the carries model got worse, so targets only
    VARIANTS = {'current': ({}, False), 'opt1 targets': ({'targets': X1}, False), 'opt1+3 targets': ({'targets': X1 + X3}, False),
                'opt1+2+3 targets': ({'targets': X1 + X2 + X3}, False), 'opt1+3 targets, coverage ratios': ({'targets': X1 + X3 + ['ms_mz', 'ms_shell', 'ms_db', 'ms_pr']}, False),
                'opt1+3 targets + opt4 TD': ({'targets': X1 + X3}, True)}
res = {}
for name, (xs, td) in VARIANTS.items():
    ps = []
    for tr, te in [(range(2017, 2024), [2024]), (range(2017, 2025), [2025])]:
        mdl.OPP_EXTRA = xs; mdl.eff_specs = td_specs if td else base_specs
        m = mdl.fit(f[f.season.isin(tr)]); ps.append(mdl.predict(m, f[f.season.isin(te)]))
        if te == [2025] and xs.get('targets'):
            mt = m['opp']['targets']; imp = pd.Series(mt.feature_importance('gain'), index=mt.feature_name()); imp = imp / imp.sum()
            say(f'{name}, targets model, share of split gain from the new features:', {k: round(v, 4) for k, v in imp[xs['targets']].sort_values(ascending=False).head(12).items()})
            if td: say(f'{name}, TD GLM:', {k: f'{v:+.3f} (p {m["eff"]["td"].pvalues[k]:.3f})' for k, v in m['eff']['td'].params.items()})
        mdl.OPP_EXTRA = {}; mdl.eff_specs = base_specs
    res[name] = pd.concat(ps); say('done', name)

a = res['current']
rel = ((a.snap_l >= 0.25) & (a.games_prior >= 3)).to_numpy()
pos = a.position
wr = pos.isin(['WR', 'TE']).to_numpy(); rb = (pos == 'RB').to_numpy(); skill = pos.isin(['WR', 'TE', 'RB']).to_numpy()
early_new = ((a.cl_new == 1) & (a.games_season < 6)).to_numpy()
mover_early = ((a.cl_mover == 1) & (a.games_season < 6)).to_numpy()
ms = np.log(a[['ms_mz', 'ms_shell', 'ms_db']].fillna(1)).abs().sum(axis=1).to_numpy(); ms_x = ms >= np.nanquantile(ms[wr & rel], 0.9)
mc_x = (np.abs(np.log(a.mc_box.fillna(1))) >= np.nanquantile(np.abs(np.log(a.mc_box.fillna(1)))[rb & rel], 0.9)).to_numpy()
on_x = (np.abs(np.log(a.on_ratio.fillna(1))) >= 0.03).to_numpy()
rz_x = (a.rz_exp.fillna(0) >= np.nanquantile(a.rz_exp[skill & rel], 0.8)).to_numpy()
SUBS = [('all WR/TE', wr, ['targets', 'rec', 'rec_yds']), ('all RB', rb, ['carries', 'rush_yds', 'targets']),
        ('new caller, games 1-6, WR/TE/RB', early_new & skill, ['targets', 'rec', 'carries']),
        ('caller who moved teams, games 1-6', mover_early & skill, ['targets', 'rec', 'carries']),
        ('WR/TE facing unusual coverage mix (top 10%)', wr & ms_x, ['targets', 'rec', 'rec_yds']),
        ('RB facing unusual box mix (top 10%)', rb & mc_x, ['carries', 'rush_yds']),
        ('expected personnel shifts his snaps 3%+', skill & on_x, ['targets', 'rec', 'carries']),
        ('top-20% red-zone roles', skill & rz_x, ['td'])]
rows = []
def ll(y, mu):
    p = np.clip(1 - np.exp(-np.asarray(mu)), 1e-4, 1 - 1e-4); y = (np.asarray(y) > 0).astype(float)
    return -(y * np.log(p) + (1 - y) * np.log(1 - p))
for vname in VARIANTS:
    if vname == 'current': continue
    b = res[vname]
    for lab, m_, stats in SUBS:
        d = m_ & rel
        for st in stats + (['td'] if lab.startswith('all') else []):
            if st == 'td':
                ea, eb = ll(a.td, a.td_hat)[d], ll(b.td, b.td_hat)[d]
            else:
                ea, eb = np.abs(a[st] - a[f'{st}_hat']).to_numpy()[d], np.abs(b[st] - b[f'{st}_hat']).to_numpy()[d]
            diff = ea - eb; se = diff.std() / np.sqrt(max(1, d.sum()))
            rows.append(dict(variant=vname, subset=lab, stat=st if st != 'td' else 'anytime TD (log loss)', n=int(d.sum()), current=round(ea.mean(), 4),
                             variant_err=round(eb.mean(), 4), gain_pct=round(100 * (1 - eb.mean() / ea.mean()), 2), t=round(diff.mean() / se, 2) if se > 0 else 0))
R = pd.DataFrame(rows)
say('\nOut of sample 2024-2025 (snaps >= 25%, 3+ prior games). gain = error reduction vs the current model; t > 2 = real')
for v in R.variant.unique(): say(f'\n{v}\n' + R[R.variant == v].drop(columns='variant').to_string(index=False))
R.to_csv(f'{M}/backtest_nfl4_{ROUND}_accuracy.csv', index=False)
keep = ['game_id', 'pid', 'season', 'week', 'posteam', 'position', 'name'] + [c for v in sf.SCHEME_X.values() for c in v]
out = a[keep].copy()
for v in VARIANTS:
    for st in ['targets', 'rec', 'rec_yds', 'carries', 'rush_yds', 'td']: out[f'{st}_hat__{v}'] = res[v][f'{st}_hat'].to_numpy()
for st in ['targets', 'rec', 'rec_yds', 'carries', 'rush_yds', 'td']: out[st] = a[st].to_numpy()
out.to_parquet(f'{M}/data/backtest_nfl4_{ROUND}_predictions.parquet', index=False)
