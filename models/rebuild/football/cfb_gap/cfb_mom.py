from cfb_lib import *
G = pd.read_parquet('/mnt/project-files/sports-analytics/models/cfb-v3/game_predictions.parquet')
G = G[G.margin.notna() & G.model_margin.notna()].copy()
dates = pd.read_parquet("/mnt/project-files/sports-analytics/models/rebuild/football/data/teamgames_cfb.parquet")[["game_id", "date"]].drop_duplicates("game_id")
G = G.merge(dates, on="game_id", how="left"); G["date"] = pd.to_datetime(G.date).fillna(pd.Timestamp("2000-01-01"))
r = G.margin - G.model_margin
TG = pd.concat([pd.DataFrame({"game_id": G.game_id, "team": G.home_id, "season": G.season, "wk": G.wk, "res": r}),
                pd.DataFrame({"game_id": G.game_id, "team": G.away_id, "season": G.season, "wk": G.wk, "res": -r})]).sort_values(["team", "season", "wk"])
for n in (1, 3, 5):
    TG[f"m{n}"] = TG.groupby(["team", "season"]).res.transform(lambda s: s.shift(1).rolling(n, min_periods=1).mean()).fillna(0)
TG["mall"] = TG.groupby(["team", "season"]).res.transform(lambda s: s.shift(1).expanding().mean()).fillna(0)
MI = TG.set_index(["game_id", "team"])
for d in (fit, te):
    for c in ("m1", "m3", "m5", "mall"):
        d["mo_" + c] = MI[c].reindex(list(zip(d.game_id, d.home_id))).fillna(0).values - MI[c].reindex(list(zip(d.game_id, d.away_id))).fillna(0).values
rep("v3", v3)
for c in (["mo_m1"], ["mo_m3"], ["mo_m5"], ["mo_mall"], ["mo_m3", "mo_mall"]):
    rep("v3 + momentum " + "+".join(c), run(['model_margin'] + c), base=v3)
