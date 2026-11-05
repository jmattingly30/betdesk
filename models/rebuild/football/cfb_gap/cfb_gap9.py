exec(open('cfb_gap3.py').read().split('v3 = te.model_margin.values')[0])
QF = pd.read_parquet("cfb_qb_feat.parquet").set_index(["game_id", "team"])
def feat5(df):
    for c in ("dq", "new", "dql"):
        h = QF[c].reindex(list(zip(df.game_id, df.home_id))).fillna(0).values; a = QF[c].reindex(list(zip(df.game_id, df.away_id))).fillna(0).values
        df["q_" + c] = h - a
    return df
fit = feat5(fit); te = feat5(te)
v3 = te.model_margin.values; D = ['d_' + c for c in RC]
base = ['rm', 'mk'] + D + ['pk', 'kal', 'r0', 'd_ro', 'd_rp', 's1', 's23']
rep("v3", v3); rep("C", run(base), base=v3)
rep("C + QB (actual starter; upper bound)", run(base + ['q_dq', 'q_new']), base=v3)
rep("C + QB (last game's starter; live-usable)", run(base + ['q_dql']), base=v3)
rep("v3 + QB actual", run(['model_margin', 'q_dq', 'q_new']), base=v3)
rep("v3 + C + QB actual", run(['model_margin'] + base + ['q_dq', 'q_new']), base=v3)
chg = (te.q_new != 0).values
p1 = run(base + ['q_dq', 'q_new']); e1 = np.abs(p1 - te.am.values); ev3 = np.abs(v3 - te.am.values)
print("QB-change games n", chg.sum(), "C+QB", e1[chg].mean().round(3), "v3", ev3[chg].mean().round(3))
