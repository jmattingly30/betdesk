exec(open('cfb_r6b.py').read().split('\nE4 = [')[0])
import pickle; UO = pickle.load(open("cfb_units2.pkl", "rb"))
MET = ["epa", "sr", "expl", "repa", "pepa"]
WK = pd.concat([pd.read_csv(f"cfbros/s_{y}.csv", usecols=["game_id", "week", "season_type"]) for y in range(2014, 2027)]).drop_duplicates("game_id")
WK = pd.Series(np.where(WK.season_type == "postseason", 20, WK.week), index=WK.game_id)
for d in (fit, te):
    wks = d.game_id.map(WK).values
    for m in MET:
        xs, ts = [], []
        for s, wv, h, a in zip(d.season.values.astype(int), wks, d.home_id.values.astype(int), d.away_id.values.astype(int)):
            u1 = UO.get((s, int(wv), h)) if wv == wv else None; u2 = UO.get((s, int(wv), a)) if wv == wv else None
            if u1 is None or u2 is None: xs.append(np.nan); ts.append(np.nan); continue
            eh = u1[m][0] + u2[m][1]; ea = u2[m][0] + u1[m][1]; xs.append(eh - ea); ts.append(eh + ea)
        d["u_" + m] = np.nan_to_num(xs); d["ut_" + m] = np.nan_to_num(ts)
    d["u_miss"] = (d.u_epa == 0).astype(float)
print("coverage te", 1 - te.u_miss.mean())
E4 = ['model_margin'] + P + RT + D + ['c_cwin', 'x_att']; L4 = ['model_margin'] + P + ['c_cwin', 'x_att']
def v4(e=()): return np.where(w, run(E4, lam=1e-4), run(L4 + list(e), lam=1e-4))
B = v4(); rep("v4+cwin+crowd", B, base=v3)
def vs(name, p, base, y='am'):
    rep(name, p, y=y, base=v3 if y == 'am' else te.model_total.values); d_ = np.abs(p - te[y].values) - np.abs(base - te[y].values); print(f"      vs base {d_.mean():+.4f} (t {d_.mean() / (d_.std() / np.sqrt(len(d_))):.1f})")
for nm, e in [("epa", ['u_epa']), ("success", ['u_sr']), ("explosive", ['u_expl']), ("rush+pass epa", ['u_repa', 'u_pepa']), ("all units", ['u_' + m for m in MET])]:
    vs("v4 wk4+ + " + nm, v4(e), B)
T1 = run(['model_total', 'x_ip'], y='at')
for nm, c in [("explosive sum", ['ut_expl']), ("epa sum", ['ut_epa']), ("all unit sums", ['ut_' + m for m in MET])]:
    vs("tempo total + " + nm, run(['model_total', 'x_ip'] + c, y='at'), T1, y='at')
