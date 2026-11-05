"""Expanding window: each test season fit on all lined seasons before it (no 2020)."""
exec(open('cfb_r5b.py').read().split('\nT0 = run(')[0])
for d in (fit, te): ins(d, 3)
ALL = pd.concat([fit, te], ignore_index=True)
def ew(cols, y='am', split=True):
    pred = np.zeros(len(te))
    for s in sorted(te.season.unique()):
        F = ALL[(ALL.season < s) & (ALL.season != 2020)]; m = (te.season == s).values
        Tt = te[m]; pred[m] = run(cols, split=split, y=y, F=F.reset_index(drop=True), Tt=Tt.reset_index(drop=True))
    return pred
E4 = ['model_margin'] + P + RT + D + ['c_cwin']; L4 = ['model_margin'] + P + ['c_cwin']
base = np.where(w, run(E4), run(L4)); rep("v4+cwin fixed fit", base, base=v3)
p = np.where(w, ew(E4), ew(L4)); rep("v4+cwin expanding", p, base=v3)
v3t = te.model_total.values
rep("tempo total fixed", run(['model_total', 'x_ip'], y='at'), y='at', base=v3t)
rep("tempo total expanding", ew(['model_total', 'x_ip'], y='at'), y='at', base=v3t)
