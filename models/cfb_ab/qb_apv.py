"""CFB QB values from APV, to date this season (J 2026-11-16, football roadmap step 4: SHADOW only, live numbers unchanged).
Plays valued exactly as research/football-2026-11/apv_players.py (own EP model apv/ep_model.txt, fumble / interception zone
averages, no cap); expected value per play = league average + home + the opponent's defensive rating for that play type from
EARLIER weeks only (ridge, prior = half of 2025's final ratings, apv/qb_apv_prior.json), as research/apv-projections-2026-11/
cfb_weekly_apv.py. QB credit = his dropbacks and runs (QB-rec 50/50 split not used: value per QB play = all his pass + rush
plays, role slopes pass 1.37 / rush 1.45). QB value = (sum credit - 0.05 x 203) / (plays + 203): shrunk toward a replacement
level of -0.05 per play. The 2024-25 test sized a QB change at 16.7 team points per 1.0 of value (roster_cfb2.json:
QB-change games 10.09 -> 10.01 MAE, 2025 -0.15, 2024 +0.01), so it runs in shadow before replacing the flat -2.5 / -0.5.
python3 models/cfb_ab/qb_apv.py [pbp parquet]  -> models/cfb_ab/apv/qb_apv.json"""
import sys, os, re, json, datetime as dt, numpy as np, pandas as pd, lightgbm as lgb
HERE = os.path.dirname(os.path.abspath(__file__)); D_ = f"{HERE}/apv"
PBP = sys.argv[1] if len(sys.argv) > 1 else f"{HERE}/../cfb-v3/pbp_2026.parquet"
META = json.load(open(f"{D_}/ep_meta.json")); EP = lgb.Booster(model_file=f"{D_}/ep_model.txt")
SLOPE = {"rush": 1.45, "pass": 1.37}; LAM = 150.0; K, B = 203, -0.05; PTS_PER_VALUE = 16.7

# ---- plays and their value (verbatim from apv_players.py)
C = ['game_id', 'game_play_number', 'half', 'period', 'pos_team_id', 'def_pos_team_id', 'start.down', 'start.distance', 'start.yardsToEndzone', 'start.TimeSecsRem',
     'start.homeScore', 'start.awayScore', 'homeTeamId', 'awayTeamId', 'homeFinalScore', 'awayFinalScore', 'type.text', 'start.posTeamTimeouts', 'start.defPosTeamTimeouts',
     'drive.id', 'week', 'seasonType', 'rush', 'pass', 'text', 'passer_player_id', 'passer_player_name', 'rusher_player_id', 'rusher_player_name', 'receiver_player_id',
     'receiver_player_name', 'sack_player_name', 'sack_player_name2', 'interception_player_name', 'pass_breakup_player_name', 'fumble_forced_player_name', 'start.pos_team.name', 'statYardage']
DROP = {'Timeout', 'End Period', 'End of Half', 'End of Game', 'End of Regulation', 'Coin Toss', 'Unknown', 'Two Point Pass', 'Two Point Rush', 'Defensive 2pt Conversion', 'Extra Point Good', 'Extra Point Missed'}
FEAT = ['down', 'dist', 'yte', 'secs', 'half2', 'sdiff', 'to_pos', 'to_def', 'gtg']
d = pd.read_parquet(PBP, columns=C)
d = d[(d.period <= 4) & ~d['type.text'].isin(DROP)].sort_values(['game_id', 'game_play_number']).reset_index(drop=True)
d = d.rename(columns={'start.down': 'down', 'start.distance': 'dist', 'start.yardsToEndzone': 'yte', 'start.TimeSecsRem': 'secs', 'type.text': 'type',
                      'start.posTeamTimeouts': 'to_pos', 'start.defPosTeamTimeouts': 'to_def', 'drive.id': 'drive'})
d['kick'] = d.type.str.startswith('Kickoff'); d = d[d.kick | d.down.between(1, 4)].reset_index(drop=True)
d['is_home'] = d.pos_team_id == d.homeTeamId
g = d.groupby('game_id'); nh = g['start.homeScore'].shift(-1).fillna(d.homeFinalScore); na = g['start.awayScore'].shift(-1).fillna(d.awayFinalScore)
nhm = (nh - d['start.homeScore']).clip(-8, 8) - (na - d['start.awayScore']).clip(-8, 8); d['pts'] = np.where(d.is_home, nhm, -nhm)
d['half2'] = (d.half == 2).astype(int); d['gtg'] = (d.dist >= d.yte).astype(int)
d['sdiff'] = np.where(d.is_home, d['start.homeScore'] - d['start.awayScore'], d['start.awayScore'] - d['start.homeScore'])
d['ep'] = np.where(d.kick, META['kickoff_ep'], EP.predict(d[FEAT].astype(float)))
g = d.groupby(['game_id', 'half']); nep, npos = g.ep.shift(-1), g.pos_team_id.shift(-1)
d['V'] = d.pts + np.where(nep.isna(), 0.0, np.where(npos == d.pos_team_id, 1.0, -1.0) * nep.fillna(0)) - d.ep
d['fum'] = d.type.str.contains('Fumble'); d['intc'] = d.type.str.contains('Interception'); z = np.digitize(d.yte, [20, 50, 80])
fm = {int(k): v for k, v in META['fum_mu'].items()}; im = {int(k): v for k, v in META['int_mu'].items()}
CAP = float(os.environ.get('APV_CAP', '99'))   # per-play cap; default none (J 2026-11-14: a 99-yd TD should count fully; cap cost only ~.01-.03 in stability)
d['X4'] = np.where(d.fum, pd.Series(z).map(fm), np.where(d.intc, pd.Series(z).map(im), d.V)).clip(-CAP, CAP)
SCR = ['Rush', 'Pass Reception', 'Pass Incompletion', 'Sack', 'Rushing Touchdown', 'Passing Touchdown', 'Pass Completion', 'Pass']
d['scrim'] = (d.type.isin(SCR) | d.fum | d.intc) & ~d.kick & ~d.type.str.contains('Kickoff|Punt|Field Goal')
s = d[d.scrim & (d.rush | d['pass'])].copy(); s['kind'] = np.where(s['pass'] == True, 'P', 'R'); s['team'] = s.pos_team_id; s['opp'] = s.def_pos_team_id

# ---- walk-forward opponent ratings (earlier weeks only), prior = half of 2025's final ratings
def ridge(tg, prior):
    teams = sorted(set(tg.team) | set(tg.opp) | set(prior[0].keys())); T = len(teams); ix = {t: i for i, t in enumerate(teams)}; p = 2 + 2 * T
    A = np.zeros((p, p)); b = np.zeros(p); P0 = np.zeros(p); P0[0] = prior[2]
    for t, i in ix.items(): P0[2 + i] = prior[0].get(t, 0.0); P0[2 + T + i] = prior[1].get(t, prior[3])
    if len(tg):
        k = np.arange(len(tg)); X = np.zeros((len(tg), p)); X[:, 0] = 1; X[:, 1] = np.where(tg.home, .5, -.5)
        X[k, 2 + np.array([ix[t] for t in tg.team])] = 1; X[k, 2 + T + np.array([ix[t] for t in tg.opp])] = 1
        w = tg.n.values.astype(float); A = X.T @ (X * w[:, None]); b = X.T @ (tg.y.values * w)
    pen = np.r_[1e-2, 30.0, np.full(2 * T, LAM)]; A[np.diag_indices(p)] += pen
    c = np.linalg.solve(A, b + pen * P0); return c[0], c[1], {t: c[2 + i] for t, i in ix.items()}, {t: c[2 + T + i] for t, i in ix.items()}
PR = json.load(open(f"{D_}/qb_apv_prior.json"))["ratings"]
s["wk"] = np.where(s.seasonType == 2, s.week, 16 + s.week).astype(int)
TG = s.groupby(["kind", "wk", "game_id", "team", "opp"]).agg(y=("X4", "mean"), n=("X4", "size"), home=("is_home", "first")).reset_index()
eo = np.zeros(len(s)); mu_ = np.zeros(len(s))
for kind in ("R", "P"):
    pv = PR[kind]; dv = np.array(list(pv["dfn"].values()))
    pr = ({int(t): .5 * v for t, v in pv["off"].items()}, {int(t): .5 * v for t, v in pv["dfn"].items()}, pv["mu"], float(np.quantile(dv, .9)))
    for w in sorted(s.wk.unique()):
        mu, hm, o_, d_ = ridge(TG[(TG.kind == kind) & (TG.wk < w)], pr)
        m = ((s.kind == kind) & (s.wk == w)).values
        eo[m] = mu + hm * np.where(s.is_home.values[m], .5, -.5) + np.array([d_.get(o, pr[3]) for o in s.opp.values[m]]); mu_[m] = mu
s["exp_off"] = eo; s["mu"] = mu_
s["qb_txt"] = s.text.str.extract(r"#\d+ ([A-Z][\w'\-]*\.[\w'\-\. ]+?) (?:sacked|pass)")[0].str.strip()
qmap = s[s.passer_player_id.notna()].dropna(subset=["qb_txt"]).groupby(["game_id", "team", "qb_txt"]).agg(pid=("passer_player_id", "first"))
sk = s.type.eq("Sack") & s.passer_player_id.isna(); j = s[sk].join(qmap, on=["game_id", "team", "qb_txt"]); s.loc[sk, "passer_player_id"] = j.pid
P = s[s.kind == "P"]; R = s[(s.kind == "R") & s.rusher_player_id.notna()]
O = pd.concat([P[P.passer_player_id.notna()].assign(pid=lambda x: x.passer_player_id, name=lambda x: x.passer_player_name, role="pass"),
               R.assign(pid=lambda x: x.rusher_player_id, name=lambda x: x.rusher_player_name, role="rush")])
O["c"] = O.X4 - (O.mu + O.role.map(SLOPE) * (O.exp_off - O.mu))
qbs = set(O[O.role == "pass"].pid)                     # a QB = anyone with a pass attempt; his runs count too
Q = O[O.pid.isin(qbs)].groupby(["team", "pid"]).agg(name=("name", lambda x: x.dropna().mode().iat[0] if x.notna().any() else ""), plays=("c", "size"),
                                                   credit=("c", "sum"), passes=("role", lambda x: int((x == "pass").sum())), games=("game_id", "nunique"), last_wk=("wk", "max")).reset_index()
Q["v"] = (Q.credit + B * K) / (Q.plays + K)
names = s.groupby("team")["start.pos_team.name"].agg(lambda x: x.dropna().mode().iat[0] if x.notna().any() else "")
out = {"made": dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%d %H:%M UTC"), "through_week": int(s.wk.max()), "replacement_v": B, "shrink_plays": K,
       "pts_per_value": PTS_PER_VALUE, "teams": {}}
for t, g in Q.sort_values("plays", ascending=False).groupby("team"):
    out["teams"][str(int(t))] = {"name": names.get(t, ""), "qbs": [dict(pid=str(r.pid), name=r.name, v=round(float(r.v), 4), plays=int(r.plays), passes=int(r.passes),
                                 games=int(r.games), last_wk=int(r.last_wk)) for r in g.itertuples()]}
json.dump(out, open(f"{D_}/qb_apv.json", "w"), indent=0)
print(f"qb_apv.json: {len(out['teams'])} teams, {len(Q)} QBs, through week {out['through_week']}")
