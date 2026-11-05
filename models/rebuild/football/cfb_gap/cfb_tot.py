exec(open('cfb_gap3.py').read().split('v3 = te.model_margin.values')[0])
for d in (fit, te): d["mm2"] = d.rm.abs()
v3t = te.model_total.values
rep("v3 total", v3t, y='at'); rep("close total", te.total_close.values, y='at')
rep("rebuild total raw", te.rt.values, y='at', base=v3t)
rep("rebuild total stacked", run(['rt'], y='at'), y='at', base=v3t)
rep("rebuild total + mismatch", run(['rt', 'mm2'], y='at'), y='at', base=v3t)
rep("rebuild + v3 total", run(['rt', 'model_total'], y='at'), y='at', base=v3t)
