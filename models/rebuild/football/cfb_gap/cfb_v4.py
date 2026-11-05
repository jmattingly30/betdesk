from cfb_lib import *
pe = run(['model_margin'] + P + RT + D); pl = run(['model_margin'] + P); v4 = np.where(w, pe, pl)
rep("v3", v3); rep("v4 = v3 + early prior stack + program", v4, base=v3)
pe = run(['model_margin'] + P + RT + D + ['q_dq', 'q_new']); pl = run(['model_margin'] + P + ['q_dq', 'q_new'])
rep("v4 + known starting QB", np.where(w, pe, pl), base=v3)
rep("close (yardstick)", te.line_close.values)
for nm, p in (("v3", v3), ("v4", v4)):
    g = p - te.line_open.values; k = te.line_open.notna().values & (np.abs(g) >= 2) & (te.am.values != te.line_open.values)
    cov = np.sign(te.am.values - te.line_open.values) == np.sign(g)
    print(nm, "2+ pt gaps vs opener:", k.sum(), "ATS", round(cov[k].mean(), 3), "CLV", round(np.mean(np.sign(g[k]) * (te.line_close.values[k] - te.line_open.values[k])), 3))
