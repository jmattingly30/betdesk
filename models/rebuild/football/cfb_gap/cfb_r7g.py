exec(open('cfb_r7b.py').read().split('\np = run(')[0])
for nm, p in [("tempo", T1), ("tempo + opp-adj wk4+", run(['model_total', 'x_ip', 'x_oip4'], y='at', lam=1e-4))]:
    for g in (3, 5):
        m = te.total_open.notna() & te.total_close.notna() & (np.abs(p - te.total_open) >= g) & (te["at"] != te.total_open)
        side = np.sign(p - te.total_open)[m]; hit = (np.sign(te["at"] - te.total_open)[m] == side).mean(); clv = (side * (te.total_close - te.total_open)[m]).mean()
        print(f"OPEN {nm:22s} gap {g}+ n {m.sum():4d} hit {hit:.3f} CLV {clv:+.2f}")
