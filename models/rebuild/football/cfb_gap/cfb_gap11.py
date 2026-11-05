exec(open('cfb_gap10.py').read().split('rep("v3", v3)')[0])
rep("v3", v3)
rep("v3 + C", run(['model_margin'] + base), base=v3)
rep("v3 + C, ridge 50", run(['model_margin'] + base, lam=50), base=v3)
rep("v3 + C + rule", run(['model_margin'] + base + ['q2_dq_c']), base=v3)
# where does weeks 1-3 gain come from: drop groups
for nm, drop in [("no program", ['s1', 's23']), ("no roster", D), ("no market", ['mk']), ("no rebuild engines", ['rm', 'pk', 'kal', 'r0']), ("no returning", ['d_ro', 'd_rp'])]:
    rep("v3 + C " + nm, run(['model_margin'] + [c for c in base if c not in drop]), base=v3)
# by season stability
p = run(['model_margin'] + base); e = np.abs(p - te.am.values) - np.abs(v3 - te.am.values)
print(pd.Series(e).groupby(te.season.values).mean().round(3).to_dict())
