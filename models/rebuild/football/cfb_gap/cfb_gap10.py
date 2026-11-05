exec(open('cfb_gap9.py').read().split('v3 = te.model_margin.values; D')[0])
Q2 = pd.read_parquet("cfb_qb_feat2.parquet").set_index(["game_id", "team"])
for d in (fit, te):
    for c in ("dq_last", "dq_c", "hurt", "dq_hurt"):
        h = Q2[c].reindex(list(zip(d.game_id, d.home_id))).fillna(0).values; a = Q2[c].reindex(list(zip(d.game_id, d.away_id))).fillna(0).values
        d["q2_" + c] = h - a
v3 = te.model_margin.values; D = ['d_' + c for c in RC]
base = ['rm', 'mk'] + D + ['pk', 'kal', 'r0', 'd_ro', 'd_rp', 's1', 's23']
rep("v3", v3)
for nm, cols in [("v3 + last game's finishing QB", ['q2_dq_last']), ("v3 + starter-left-early rule", ['q2_dq_c']),
                 ("v3 + hurt flag + value drop", ['q2_hurt', 'q2_dq_hurt']), ("v3 + rule + hurt", ['q2_dq_c', 'q2_hurt', 'q2_dq_hurt'])]:
    rep(nm, run(['model_margin'] + cols), base=v3)
rep("v3 + C + rule + hurt", run(['model_margin'] + base + ['q2_dq_c', 'q2_hurt', 'q2_dq_hurt']), base=v3)
rep("C + rule + hurt (rebuild only)", run(base + ['q2_dq_c', 'q2_hurt', 'q2_dq_hurt']), base=v3)
h = (te.q2_hurt != 0).values; print("games where a starter left early last game:", h.sum())
p = run(['model_margin', 'q2_dq_c', 'q2_hurt', 'q2_dq_hurt']); print("  those games: v3", np.abs(v3 - te.am.values)[h].mean().round(3), "v3+rule", np.abs(p - te.am.values)[h].mean().round(3))
