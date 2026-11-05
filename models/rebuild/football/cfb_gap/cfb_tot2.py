from cfb_lib import *
for d in (fit, te):
    d["mm2"] = d.rm.abs(); d["v3m2"] = d.model_margin.abs()
v3t = te.model_total.values
rep("v3 total", v3t, y='at')
for nm, cols in [("rebuild total", ['rt']), ("rebuild total + mismatch", ['rt', 'mm2']), ("+ points-kalman", ['rt', 'mm2', 'pk']),
                 ("+ QB both sides", ['rt', 'q_dq'])]:
    rep("v3 + " + nm, run(['model_total'] + cols, y='at'), y='at', base=v3t)
