exec(open('cfb_gap7.py').read().split('v3 = te.model_margin.values; D')[0])
conf = pd.concat([S[["season", "home_id", "home_conference"]].set_axis(["season", "tid", "conf"], axis=1),
                  S[["season", "away_id", "away_conference"]].set_axis(["season", "tid", "conf"], axis=1)]).dropna().drop_duplicates(["season", "tid"])
conf["srs_prev"] = [srs.get((s - 1, t), np.nan) for s, t in zip(conf.season, conf.tid)]
cm = conf.groupby(["season", "conf"]).srs_prev.mean()
conf["cstr"] = [cm.get((s, c), np.nan) for s, c in zip(conf.season, conf.conf)]
CI = conf.set_index(["season", "tid"])
def feat4(df):
    h = CI.reindex(list(zip(df.season, df.home_id))); a = CI.reindex(list(zip(df.season, df.away_id)))
    df["d_conf"] = (h.cstr.values - a.cstr.values); df["d_conf"] = df.d_conf.fillna(0)
    df["nonconf"] = (h.conf.values != a.conf.values).astype(float); df["d_conf_nc"] = df.d_conf * df.nonconf
    return df
fit = feat4(fit); te = feat4(te)
v3 = te.model_margin.values; D = ['d_' + c for c in RC]
base = ['rm', 'mk'] + D + ['pk', 'kal', 'r0', 'd_ro', 'd_rp', 's1', 's23']
rep("v3", v3); rep("C", run(base), base=v3)
rep("C + conference strength", run(base + ['d_conf']), base=v3)
rep("C + conf strength (non-conf games)", run(base + ['d_conf_nc']), base=v3)
rep("C + conf + Elo + roster", run(base + ['d_conf_nc', 'd_elo', 'd_stay', 'd_xin', 'd_up', 'd_yr']), base=v3)
