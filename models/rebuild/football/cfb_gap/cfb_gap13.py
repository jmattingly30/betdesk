exec(open('cfb_gap12.py').read().split('rep("v3", v3)\ndef early_only')[0])
P = ['s1', 's23']; RT = ['d_ro', 'd_rp']
w = te.wk.values <= 3
def late_only(cols, lam=0.0):
    p = run(['model_margin'] + cols, lam=lam); return np.where(w, v3, p)
rep("v3", v3)
for nm, cols in [("program", P), ("returning", RT), ("roster", D), ("market", ['mk']), ("rebuild engines", ['rm', 'pk', 'kal', 'r0']),
                 ("rebuild ridge only", ['rm']), ("program+returning+roster", P + RT + D), ("everything", base)]:
    p = late_only(cols); rep("v3, wk4+ + " + nm, p, base=v3)
    e = (np.abs(p - te.am.values) - np.abs(v3 - te.am.values))[~w]; print("   wk4+ by season", pd.Series(e).groupby(te.season.values[~w]).mean().round(3).to_dict())
