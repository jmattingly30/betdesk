exec(open('cfb_gap11.py').read().split('rep("v3", v3)\nrep("v3 + C", run')[0])
rep("v3", v3)
def early_only(cols, lam=0.0):
    p = run(['model_margin'] + cols, lam=lam); w = te.wk.values <= 3; return np.where(w, p, v3)
P = ['s1', 's23']; RT = ['d_ro', 'd_rp']
for nm, cols in [("program", P), ("program + returning", P + RT), ("program + returning + roster", P + RT + D),
                 ("program + returning + roster + market", P + RT + D + ['mk'])]:
    p = early_only(cols); rep("v3, wk1-3 + " + nm, p, base=v3)
    e = np.abs(p - te.am.values) - np.abs(v3 - te.am.values); print("   by season", pd.Series(e).groupby(te.season.values).mean().round(3).to_dict())
p = early_only(P + RT + D, lam=50); rep("v3, wk1-3 + prog+ret+roster ridge50", p, base=v3)
