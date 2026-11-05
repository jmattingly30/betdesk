"""Totals: in-season tempo (plays/game, neutral seconds per play) blended with coach pace prior."""
import sys
exec(open('cfb_r5.py').read().split('\ndef v4(')[0])
PG = pd.read_parquet("cfb_pace_games.parquet")
dates = pd.concat([pd.read_csv(f"cfbros/s_{y}.csv", usecols=["game_id", "start_date"]) for y in range(2014, 2027)]).drop_duplicates("game_id")
PG = PG.merge(dates, on="game_id", how="left"); PG["dt"] = pd.to_datetime(PG.start_date, utc=True); PG = PG.sort_values("dt")
g = PG.groupby(["season", "team"])
PG["n_prev"] = g.cumcount(); PG["cp"] = g.gplays.cumsum() - PG.gplays
PG["cs"] = g.nsec.cumsum() - PG.nsec; PG["cn"] = g.nn.cumsum() - PG.nn
SPP_prior = {}  # prior-season seconds per play
ss = PG.groupby(["season", "team"])[["nsec", "nn"]].sum(); ss = (ss.nsec / ss.nn)
IDX = PG.set_index(["game_id", "team"])
def ins(d, K=3):
    for side, col in (("h", "home_id"), ("a", "away_id")):
        k = list(zip(d.game_id.values, d[col].values.astype(int)))
        r = IDX.reindex(k)
        n = r.n_prev.fillna(0).values; avg = (r.cp / r.n_prev.replace(0, np.nan)).values
        prior = d["cpace_" + side].values / 2   # coach pace is plays per game for both teams of his games -> same scale as gplays? keep raw
        d["ipn_" + side] = n
        d["ip_" + side] = np.where(n > 0, (np.nan_to_num(avg) * n + d["cpace_" + side].values * K) / (n + K), d["cpace_" + side].values)
        sp_prior = np.array([ss.get((s - 1, t), np.nan) for s, t in zip(d.season.values, d[col].values.astype(int))]); sp_prior = np.where(np.isnan(sp_prior), np.nanmean(ss.values), sp_prior)
        sp_avg = (r.cs / r.cn.replace(0, np.nan)).values
        d["is_" + side] = np.where(n > 0, (np.nan_to_num(sp_avg) * r.cn.fillna(0).values + sp_prior * 60) / (r.cn.fillna(0).values + 60), sp_prior)
    d["x_ip"] = d.ip_h + d.ip_a; d["x_is"] = d.is_h + d.is_a
for d in (fit, te): ins(d)
T0 = run(['model_total'], y='at'); v3t = te.model_total.values
def vs(name, p, base=T0, y='at'):
    rep(name, p, y=y, base=v3t); d_ = np.abs(p - te[y].values) - np.abs(base - te[y].values); print(f"      vs stacked v3 {d_.mean():+.4f} (t {d_.mean() / (d_.std() / np.sqrt(len(d_))):.1f})")
vs("total+ coach pace", run(['model_total', 'x_cpace'], y='at'))
vs("total+ in-season plays (coach prior)", run(['model_total', 'x_ip'], y='at'))
vs("total+ in-season sec/play", run(['model_total', 'x_is'], y='at'))
vs("total+ in-season plays + sec/play", run(['model_total', 'x_ip', 'x_is'], y='at'))
vs("total+ coach pace + in-season plays + sec/play", run(['model_total', 'x_cpace', 'x_ip', 'x_is'], y='at'))
for d in (fit, te): d["mm"] = d.model_margin.abs()
vs("total+ coach pace + ip + is + |margin|", run(['model_total', 'x_cpace', 'x_ip', 'x_is', 'mm'], y='at'))
vs("total+ ... + rebuild total rt", run(['model_total', 'x_cpace', 'x_ip', 'x_is', 'rt'], y='at'))
import pickle; te[["game_id", "x_cpace", "x_ip", "x_is"]].to_parquet("cfb_r5_te_tot.parquet")
