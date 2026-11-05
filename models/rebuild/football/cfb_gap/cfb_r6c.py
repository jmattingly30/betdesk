exec(open('cfb_r6b.py').read().split('\nE4 = [')[0])
PR = P + RT + D + ['c_cwin']
E4 = ['model_margin'] + PR; L4 = ['model_margin'] + P + ['c_cwin']
def v4(e=()): return np.where(w, run(E4 + list(e), lam=1e-4), run(L4 + list(e), lam=1e-4))
B = v4(); rep("v4+cwin", B, base=v3); BA = v4(['x_att']); rep("v4+cwin+crowd", BA, base=v3)
for d in (fit, te):
    d["iw"] = 1 / d.wk.clip(lower=1)
    for c in PR + ['x_att']: d["i_" + c] = d[c] * d.iw
def vs(name, p, base):
    rep(name, p, base=v3); d_ = np.abs(p - te.am.values) - np.abs(base - te.am.values); print(f"      vs v4+cwin+crowd {d_.mean():+.4f} (t {d_.mean() / (d_.std() / np.sqrt(len(d_))):.1f})")
# one model, priors decaying with 1/week
vs("decay 1/wk, one fit", run(['model_margin', 'x_att'] + PR + ['i_' + c for c in PR], split=False, lam=1e-4), BA)
# three splits
def run3(cols, edges=(3, 6)):
    p = np.zeros(len(te)); lo = 0
    for hi in list(edges) + [99]:
        Fm = fit[(fit.wk > lo) & (fit.wk <= hi)].reset_index(drop=True); m = ((te.wk > lo) & (te.wk <= hi)).values
        p[m] = run(cols, split=False, lam=1e-4, F=Fm, Tt=te[m].reset_index(drop=True)); lo = hi
    return p
vs("3 splits wk1-3/4-6/7+, all priors", run3(['model_margin', 'x_att'] + PR), BA)
vs("4 splits wk1-2/3-4/5-7/8+, all priors", run3(['model_margin', 'x_att'] + PR, (2, 4, 7)), BA)
