exec(open('cfb_r6b.py').read().split('\nE4 = [')[0])
E4 = ['model_margin'] + P + RT + D + ['c_cwin', 'x_att']; L4 = ['model_margin'] + P + ['c_cwin', 'x_att']
def v4(e=()): return np.where(w, run(E4 + list(e), lam=1e-4), run(L4 + list(e), lam=1e-4))
B = v4(); rep("v4+cwin+crowd", B, base=v3)
m0 = np.nanmean(fit.x_ip)
for d in (fit, te):
    d["mxp"] = d.model_margin * (d.x_ip / m0 - 1); d["mxt"] = d.model_margin * (d.model_total / 55 - 1)
for nm, e in [("margin x tempo", ['mxp']), ("margin x v3 total", ['mxt']), ("both", ['mxp', 'mxt'])]:
    p = v4(e); rep("v4+ " + nm, p, base=v3); d_ = np.abs(p - te.am.values) - np.abs(B - te.am.values); print(f"      vs base {d_.mean():+.4f} (t {d_.mean() / (d_.std() / np.sqrt(len(d_))):.1f})")
