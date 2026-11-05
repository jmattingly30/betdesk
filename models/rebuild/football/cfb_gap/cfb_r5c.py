exec(open('cfb_r5b.py').read().split('\nT0 = run(')[0])
v3t = te.model_total.values
for K in (1, 3, 6):
    for d in (fit, te): ins(d, K)
    p = run(['model_total', 'x_ip'], y='at'); rep(f"total+ in-season plays K={K}", p, y='at', base=v3t)
# margins: v4 + cwin on this te
B = np.where(w, run(['model_margin'] + P + RT + D + ['c_cwin']), run(['model_margin'] + P + ['c_cwin'])); rep("v4+cwin margin", B, base=v3)
