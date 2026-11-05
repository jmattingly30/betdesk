import pandas as pd, numpy as np, sys
sys.path.insert(0, "/mnt/project-files/sports-analytics/models/rebuild"); import harness as H
M = "/mnt/project-files/sports-analytics/models"
v = pd.read_parquet(f'{M}/cfb-v3/game_predictions.parquet'); r = pd.read_parquet(f'{M}/rebuild/football/data/ridge1p_cfb.parquet')
m = r.merge(v, on='game_id', suffixes=('', '_v')); m['rm'] = m.pred_h - m.pred_a; m['am'] = m.act_h - m.act_a
m['rt'] = m.pred_h + m.pred_a; m['at'] = m.act_h + m.act_a
R = pd.read_csv(f'{M}/cfb-v3/roster_prior.csv'); RC = ['qb_anya', 'qb_lvl', 'qb_new', 'qb_none', 'avail_skill', 'xfer_skill']
for side, idc in (('h', 'home_id'), ('a', 'away_id')):
    m = m.merge(R.rename(columns={'team_id': idc, **{c: f'{c}_{side}' for c in RC}}), on=['season', idc], how='left')
for c in RC: m[f'd_{c}'] = m[f'{c}_h'].fillna(0) - m[f'{c}_a'].fillna(0)
m['neu'] = m.neutral.astype(float)
m = m[m.line_close.notna() | (m.season == 2026)]
fit = m[(m.season <= 2021) & (m.season != 2020)]; te = m[(m.season >= 2022) & (m.season <= 2026)].copy()
def run(cols, split=True, y='am', lam=0.0):
    pred = np.zeros(len(te))
    groups = [(fit.wk <= 3, te.wk <= 3), (fit.wk > 3, te.wk > 3)] if split else [(fit.wk > 0, te.wk > 0)]
    for fm, tm in groups:
        X = np.c_[np.ones(fm.sum()), fit.loc[fm, cols].values]; A = X.T @ X + lam * np.diag([0] + [1] * len(cols))
        b = np.linalg.solve(A, X.T @ fit[y][fm].values); pred[tm.values] = np.c_[np.ones(tm.sum()), te.loc[tm, cols].values] @ b
    return pred
def rep(name, pred, y='am', base=None):
    e = np.abs(pred - te[y].values); w = te.wk.values <= 3
    s = f"{name:42s} all {e.mean():.3f}  wk1-3 {e[w].mean():.3f}  wk4+ {e[~w].mean():.3f}"
    if base is not None:
        d = e - np.abs(base - te[y].values); s += f"  vs v3 {d.mean():+.3f} (t {d.mean() / (d.std() / np.sqrt(len(d))):.1f})"
    print(s)
v3 = te.model_margin.values
rep("v3", v3); rep("close", te.line_close.values)
D = ['d_' + c for c in RC]
for nm, cols in [("rebuild", ['rm']), ("rebuild + market", ['rm', 'mk']), ("rebuild + roster", ['rm'] + D),
                 ("rebuild + market + roster", ['rm', 'mk'] + D), ("rebuild + market + roster + neutral", ['rm', 'mk', 'neu'] + D)]:
    rep(nm, run(cols), base=v3)
p = run(['rm', 'mk'] + D); rep("avg(stack, v3)", (p + v3) / 2, base=v3)
print("--- stacks with v3")
rep("v3 + rebuild + market + roster", run(['model_margin', 'rm', 'mk'] + D), base=v3)
rep("v3 + rebuild", run(['model_margin', 'rm']), base=v3)
rep("v3 + market", run(['model_margin', 'mk']), base=v3)
# returning production
TS = f"{M}/team-strength/data"; ret = pd.read_csv(f"{TS}/cfb_returning.csv")
print(ret.columns.tolist())
