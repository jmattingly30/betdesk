import pandas as pd, numpy as np, sys
HL, K, B0 = 24.0, 250.0, -0.08
if len(sys.argv) > 1: HL, K, B0 = map(float, sys.argv[1:4])
Q = pd.read_parquet("cfb_qb_games.parquet")
T = pd.read_parquet("/mnt/project-files/sports-analytics/models/rebuild/football/data/teamgames_cfb.parquet")[["game_id", "team", "date", "season"]]
T["date"] = pd.to_datetime(T.date)
lg = Q.groupby("season").apply(lambda d: d.epa.sum() / d.db.sum())
st = pd.read_parquet("cfb_qb_first.parquet").set_index(["game_id", "pos_team_id"]).passer_player_id   # first dropback = starter (no in-game injury leak)
T["starter"] = [st.get((g, t)) for g, t in zip(T.game_id, T.team)]
T = T.sort_values(["date", "game_id"])
a = 0.5 ** (1 / HL); a_t = 0.5 ** (1 / 4); S = {}; ew = {}; last = {}
val = lambda q: (S[q][0] + B0 * K) / (S[q][1] + K) if q in S else B0
Qg = {k: v for k, v in Q.groupby("game_id")}; rows = []
for gid, g in T.groupby("game_id", sort=False):
    for r in g.itertuples():
        if r.starter is None or r.starter != r.starter: continue
        v = val(r.starter); base = ew.get(r.team, v); lq = last.get(r.team)
        vl = val(lq) if lq is not None else base
        rows.append({"game_id": gid, "team": r.team, "dq": v - base, "new": float(lq not in (None, r.starter)), "dql": vl - base})
    for r in g.itertuples():
        if r.starter is None or r.starter != r.starter: continue
        v = val(r.starter); ew[r.team] = a_t * ew.get(r.team, v) + (1 - a_t) * v; last[r.team] = r.starter
    qg = Qg.get(gid)
    if qg is not None:
        s_lg = lg.get(qg.season.iloc[0], 0)
        for r in qg.itertuples():
            s = S.setdefault(r.passer_player_id, [0.0, 0.0]); s[0] = a * s[0] + (r.epa - s_lg * r.db); s[1] = a * s[1] + r.db
pd.DataFrame(rows).to_parquet("cfb_qb_feat.parquet"); print(len(rows))
