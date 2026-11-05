"""Data for the CFB Drive Lab: this week's CFB games with the tested CFB projection (Bet Desk v3 + v4 margin terms +
crowd size, totals + in-season tempo; built by cfb_gap/cfb_live.py -> cfb_live.json), a key-number curve for margin and
total (fit on 2014-2021 lined games), and 2,000 drive-by-drive simulations per game using CFB drive tables from ESPN
play-by-play 2023-2025 (today's clock rules). Expected plays set each game's clock speed, so faster teams get more drives.
python build_cfb_drive_viewer.py <cfb_live.json> <cfb_fit.parquet> <espn pbp dir>  -> ../viewer/cfb_drives.json"""
import sys, os, json, datetime as dt, numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from scipy import stats
import sim as S, trace as TR
LIVE, FIT, PBP = sys.argv[1:4]
RES = {"TD": 0, "FG": 1, "PUNT": 2, "DOWNS": 3, "INT": 3, "FUMBLE": 3, "MISSED FG": 3, "INT TD": 4, "FUMBLE RETURN TD": 4,
       "PUNT TD": 4, "PUNT RETURN TD": 4, "FUMBLE TD": 4, "MISSED FG TD": 4, "FG TD": 4, "DOWNS TD": 4, "SF": 5}

def drives_cfb(years=(2023, 2024, 2025)):
    out = []
    for y in years:
        d = pd.read_parquet(f"{PBP}/espn_{y}.parquet", columns=["game_id", "drive.id", "drive.result", "drive.start.period.number",
                            "drive.timeElapsed.displayValue", "start.yardsToEndzone", "game_play_number"])
        d = d.sort_values(["game_id", "game_play_number"]).groupby("drive.id", sort=False).first().reset_index()
        d = d[d["drive.start.period.number"] <= 4]
        d["half"] = np.where(d["drive.start.period.number"] <= 2, 1, 2)
        d["o"] = d["drive.result"].map(RES)
        d["start_y100"] = d["start.yardsToEndzone"].astype(float)
        d["next_start"] = d.groupby(["game_id", "half"]).start_y100.shift(-1)
        t = d["drive.timeElapsed.displayValue"].astype(str).str.extract(r"(\d+):(\d+)").astype(float)
        d["dur"] = (t[0] * 60 + t[1]).clip(5, 600)
        out.append(d[["game_id", "half", "o", "start_y100", "next_start", "dur"]])
    return pd.concat(out)

D = drives_cfb(); D = D[D.o.notna() & D.start_y100.between(1, 99)]
P = np.zeros((20, 6)); b = np.clip((D.start_y100 // 5).astype(int), 0, 19)
for k in range(20):
    c = D[b == k].o.value_counts().reindex(range(6), fill_value=0).values + 0.5; P[k] = c / c.sum()
ns = [D[(D.o == k) & D.next_start.notna()].next_start.values for k in range(6)]; du = [D[D.o == k].dur.values for k in range(6)]
NN = np.array([len(x) for x in ns]); DN = np.array([len(x) for x in du])
NS = np.full((6, NN.max()), 75.0); DU = np.full((6, DN.max()), 150.0)
for k in range(6): NS[k, :NN[k]] = ns[k]; DU[k, :DN[k]] = du[k]
TDP = np.array([0.88, 0.05, 0.07])   # try after a CFB TD: PAT 7, miss/failed 2-pt 6, 2-pt 8 (approximate)
S._T = (P, NS, NN, DU, DN, TDP)
print("drive tables:", len(D), "drives; outcome shares", np.round(D.o.value_counts(normalize=True).sort_index().values, 3))
# league plays per game under today's rules: clock speed for each game = league plays / this game's expected plays
PG = pd.read_parquet(os.path.join(os.path.dirname(LIVE), "cfb_pace_games.parquet"))
lg_plays = PG[PG.season.isin([2023, 2024, 2025])].drop_duplicates("game_id").gplays.mean()
U = pd.DataFrame(json.load(open(LIVE))).sort_values("k").reset_index(drop=True)
KW = dict(k_score=0.1, k_late=0.6, th_sd=0.3)
# key-number curve on CFB lined games 2014-2021 (stack fit set): normal x empirical weights
F = pd.read_parquet(FIT); mu_m, mu_t = F.model_margin.values, F.model_total.values; am, at = F["am"].values, F["at"].values
SDM, SDT = np.std(am - mu_m), np.std(at - mu_t); M = np.arange(-90, 91); TT = np.arange(0, 141)
def W(mu, act, grid, sd):
    dens = stats.norm.pdf(grid[None, :], mu[:, None], sd).mean(0); freq = pd.Series(act).value_counts(normalize=True).reindex(grid, fill_value=0).values
    return np.where(dens > 2e-3, (freq + 1e-4) / (dens + 1e-4), 1.0)
wm, wt = W(mu_m, am, M, SDM), W(mu_t, at, TT, SDT)
def pmf(mu, grid, sd, w): d = stats.norm.pdf(grid, mu, sd) * w; return d / d.sum()
print(f"sd margin {SDM:.2f} total {SDT:.2f}; league plays/game {lg_plays:.1f}")
out = {"made": dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%d %H:%M UTC"), "sport": "CFB", "nsim": 2000,
       "note": "Projections: Bet Desk's CFB v3 (unblended) plus the terms that beat it in testing on 2022-2025 lined games: 3-year program rating, the head coach's career win rate and home crowd size for the margin (12.25 vs v3 12.36 average miss), and in-season tempo for the total (12.76 vs 12.85). Win chances and margin/total shapes use a key-number curve fit on 2014-2021 games; the simulator supplies team-score shapes and the drive-by-drive replays, with drive odds from 2023-2025 play-by-play and each game's clock speed set by its expected plays.",
       "games": []}
for i, g in U.iterrows():
    mm, tt = g.m_new, g.t_new
    plays = g.x_ip / 2; cs = float(np.clip(lg_plays / plays, 0.8, 1.25))
    p = pd.DataFrame({"pred_h": [g.hs_new], "pred_a": [g.as_new]})
    q, _ = S.simulate(p, nsim=2000, clock_scale=cs, **KW)
    pm, pt = pmf(mm, M, SDM, wm), pmf(tt, TT, SDT, wt)
    ph = float(pm[M > 0].sum() + 0.5 * pm[M == 0].sum())
    sims = [TR.trace_game(q.th_h.iat[0], q.th_a.iat[0], seed=s + 7919 * (i + 1), clock_scale=cs, **KW) for s in range(2000)]
    Hs = np.array([s["home"] for s in sims]); As = np.array([s["away"] for s in sims])
    hist_ = lambda x: [[int(v), int(c)] for v, c in zip(*np.unique(x, return_counts=True))]
    ln = g.ln or {}
    out["games"].append({"id": g.id, "kick": g.k, "home": g.h, "away": g.a, "neutral": int(g.n),
        "proj": {"home": round(g.hs_new, 1), "away": round(g.as_new, 1), "margin": round(mm, 1), "total": round(tt, 1)},
        "v3": {"home": round(g.hs, 1), "away": round(g["as"], 1), "margin": round(g.hs - g["as"], 1), "total": round(g.hs + g["as"], 1)},
        "plays": round(plays, 1), "p_home": round(ph, 3),
        "key": {k: round(float(pm[M == k].sum() + pm[M == -k].sum()), 3) for k in (3, 7, 10, 14)},
        "margin_pmf": [[int(m), round(float(v), 4)] for m, v in zip(M, pm) if abs(m) <= 56 and v > 1e-5],
        "total_pmf": [[int(t), round(float(v), 4)] for t, v in zip(TT, pt) if 10 <= t <= 110 and v > 1e-5],
        "q_margin": [int(M[np.searchsorted(np.cumsum(pm), z)]) for z in (0.1, 0.5, 0.9)],
        "q_total": [int(TT[np.searchsorted(np.cumsum(pt), z)]) for z in (0.1, 0.5, 0.9)],
        "hist_home": hist_(Hs), "hist_away": hist_(As),
        "drives_mean": round(float(np.mean([len(s["drives"]) for s in sims])), 1),
        "sim_mean": [round(float(Hs.mean()), 1), round(float(As.mean()), 1)],
        "line": {"spread": ln.get("L"), "total": ln.get("T")}, "samples": sims[:12]})
    print(g.a, "@", g.h, out["games"][-1]["proj"], "sim", out["games"][-1]["sim_mean"], "drives", out["games"][-1]["drives_mean"], flush=True)
json.dump(out, open(f"{HERE}/../viewer/cfb_drives.json", "w"), separators=(",", ":"))
print("wrote", len(out["games"]), "games", os.path.getsize(f"{HERE}/../viewer/cfb_drives.json") // 1024, "KB")
