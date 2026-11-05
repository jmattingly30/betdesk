exec(open('cfb_gap3.py').read().split('v3 = te.model_margin.values')[0])
for d in (fit, te):
    d["rm2"] = d.rm * d.rm.abs() / 30; d["mk2"] = d.mk * d.mk.abs() / 30; d["dis"] = (d.rm - d.mk)
v3 = te.model_margin.values; D = ['d_' + c for c in RC]
base = ['rm', 'mk'] + D + ['pk', 'kal', 'r0', 'd_ro', 'd_rp', 's1', 's23']
rep("v3", v3)
rep("C", run(base), base=v3)
rep("C + curvature", run(base + ['rm2', 'mk2']), base=v3)
rep("C one fit (no week split)", run(base, split=False), base=v3)
rep("C ridge 20", run(base, lam=20), base=v3)
rep("C ridge 200", run(base, lam=200), base=v3)
