import pandas as pd, numpy as np
HL, K, B0 = 24.0, 250.0, -0.08
Q = pd.read_parquet("cfb_qb_games.parquet")
FL = pd.read_parquet("cfb_qb_firstlast.parquet").set_index(["game_id", "pos_team_id"])
T = pd.read_parquet("/mnt/project-files/sports-analytics/models/rebuild/football/data/teamgames_cfb.parquet")[["game_id", "team", "date", "season"]]
T["date"] = pd.to_datetime(T.date); T = T.sort_values(["date", "game_id"])
lg = Q.groupby("season").apply(lambda d: d.epa.sum() / d.db.sum())
a = 0.5 ** (1 / HL); a_t = 0.5 ** (1 / 4); S = {}; ew = {}; prev = {}
val = lambda q: (S[q][0] + B0 * K) / (S[q][1] + K) if q in S else B0
Qg = {k: v for k, v in Q.groupby("game_id")}; rows = []
for gid, g in T.groupby("game_id", sort=False):
    for r in g.itertuples():
        p = prev.get(r.team)
        if p is None or p[3] != r.season: continue   # first game of a season: handled by the preseason prior
        first, last, h2 = p[0], p[1], p[2]
        base = ew.get(r.team, val(first))
        pred_c = first if h2 else last          # starter left before halftime -> his replacement
        rows.append({"game_id": gid, "team": r.team, "dq_last": val(last) - base, "dq_c": val(pred_c) - base,
                     "hurt": float(not h2), "dq_hurt": (val(last) - val(first)) * (not h2)})
    for r in g.itertuples():
        k = (gid, r.team)
        if k in FL.index:
            f = FL.loc[k]; v = val(f.first_qb); ew[r.team] = a_t * ew.get(r.team, v) + (1 - a_t) * v
            prev[r.team] = (f.first_qb, f.last_qb, bool(f.starter_h2), r.season)
    qg = Qg.get(gid)
    if qg is not None:
        s_lg = lg.get(qg.season.iloc[0], 0)
        for r in qg.itertuples():
            s = S.setdefault(r.passer_player_id, [0.0, 0.0]); s[0] = a * s[0] + (r.epa - s_lg * r.db); s[1] = a * s[1] + r.db
D2 = pd.DataFrame(rows); D2.to_parquet("cfb_qb_feat2.parquet"); print(len(D2), D2.hurt.mean().round(3))
