exec(open('cfb_gap3.py').read().split('v3 = te.model_margin.values')[0])
RD = "/tmp/claude-0/-home-user/157a0a45-f501-5d44-bea4-68a0646dd4e5/scratchpad/cfbros"
S = pd.concat([pd.read_csv(f"{RD}/s_{y}.csv") for y in range(2014, 2027)])
elo = S.set_index("game_id")[["home_pregame_elo", "away_pregame_elo"]]
name2id = pd.concat([S[["home_team", "home_id"]].set_axis(["t", "i"], axis=1), S[["away_team", "away_id"]].set_axis(["t", "i"], axis=1)]).dropna().drop_duplicates("t").set_index("t").i
ros = pd.concat([pd.read_csv(f"{RD}/r_{y}.csv", usecols=["athlete_id", "team", "year", "position", "season"]) for y in range(2013, 2027)])
ros["tid"] = ros.team.map(name2id); ros = ros.dropna(subset=["tid"]); ros["tid"] = ros.tid.astype(int)
ros = ros[ros.year.between(1, 6)]
prev = ros[["athlete_id", "season", "tid"]].assign(season=lambda d: d.season + 1).rename(columns={"tid": "ptid"})
ros = ros.merge(prev, on=["athlete_id", "season"], how="left")
ros["stay"] = ros.ptid == ros.tid; ros["xin"] = ros.ptid.notna() & (ros.ptid != ros.tid); ros["up"] = ros.year >= 3
TF = ros.groupby(["season", "tid"]).agg(n=("athlete_id", "size"), stay=("stay", "mean"), xin=("xin", "sum"), up=("up", "mean"), yr=("year", "mean"))
print(TF.describe().round(2).T[["mean", "std"]])
def feat3(df):
    e = elo.reindex(df.game_id); df["d_elo"] = (e.home_pregame_elo.values - e.away_pregame_elo.values) / 25
    df["has_elo"] = np.isfinite(df.d_elo).astype(float); df["d_elo"] = df.d_elo.fillna(0)
    for c in ("stay", "xin", "up", "yr"):
        h = TF[c].reindex(list(zip(df.season, df.home_id))).values; a = TF[c].reindex(list(zip(df.season, df.away_id))).values
        mu = TF[c].mean(); df["d_" + c] = np.nan_to_num(h, nan=mu) - np.nan_to_num(a, nan=mu)
    return df
fit = feat3(fit); te = feat3(te)
print("elo coverage fit/test", fit.has_elo.mean().round(2), te.has_elo.mean().round(2))
v3 = te.model_margin.values; D = ['d_' + c for c in RC]
base = ['rm', 'mk'] + D + ['pk', 'kal', 'r0', 'd_ro', 'd_rp', 's1', 's23']
rep("v3", v3)
rep("C", run(base), base=v3)
rep("C + Elo", run(base + ['d_elo']), base=v3)
rep("C + roster continuity/seniority", run(base + ['d_stay', 'd_xin', 'd_up', 'd_yr']), base=v3)
rep("C + Elo + roster", run(base + ['d_elo', 'd_stay', 'd_xin', 'd_up', 'd_yr']), base=v3)
rep("Elo alone", run(['d_elo']), base=v3)
