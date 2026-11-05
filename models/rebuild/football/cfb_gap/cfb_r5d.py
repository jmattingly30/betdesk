exec(open('cfb_r5b.py').read().split('\nT0 = run(')[0])
for nm, p in [("v3", te.model_total.values), ("v3+tempo", run(['model_total', 'x_ip'], y='at'))]:
    for g in (3, 5):
        m = te.total_open.notna() & te.total_close.notna() & (np.abs(p - te.total_open) >= g) & (te["at"] != te.total_open)
        side = np.sign(p - te.total_open)[m]; hit = (np.sign(te["at"] - te.total_open)[m] == side).mean()
        clv = (side * (te.total_close - te.total_open)[m]).mean()
        print(f"{nm:9s} gap {g}+ n {m.sum():4d} hit {hit:.3f} CLV {clv:+.2f}")
