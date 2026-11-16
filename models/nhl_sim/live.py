"""Daily NHL sim: refresh this season from play-by-play, rate teams and goalies, simulate the next games and log them.
Usage: python3 live.py <scratch dir> [days ahead, default 1]
Reads data/hist.parquet + data/sums.json (finished seasons, see hist.py). Downloads this season's play-by-play and schedule.
Writes app/data/nhl_sim.json (latest slate) and appends history/log.jsonl. A game is logged only before its start time ("k"),
so the grader never sees a row made after puck drop. Starting goalie = the team's most used goalie of its last 10 games."""
import sys, os, json, glob, subprocess, datetime as dt, urllib.request, time, numpy as np, pandas as pd
from scipy.optimize import least_squares
from scipy.stats import poisson
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import lib, sim
SA = os.path.abspath(f"{HERE}/../..")
B = "https://github.com/sportsdataverse/sportsdataverse-data/releases/download"
FILE_YEAR = 2027; SEASON = FILE_YEAR - 1      # file year = the year the season ends; season label = start year
NSIM = 4000


def fetch(url, path, tries=4):
    for i in range(tries):
        try: return urllib.request.urlretrieve(url, path)
        except Exception as e:
            if i < tries - 1: time.sleep(2 ** (i + 1))
            elif os.path.exists(path): print(f"::warning::download failed ({e}); using the existing copy:", path, flush=True)
            else: raise


def noviG(a, b):
    f = lambda x: 100 / (x + 100) if x > 0 else -x / (100 - x)
    pa, pb = f(a), f(b); return pa / (pa + pb)


def market(path):
    """Team goals the market implies, from the moneyline and the game total (Pinnacle if listed, else the median book).
    Our odds pulls carry no team-total market, so this is the line we grade beside."""
    out = {}
    try: ev = json.load(open(path))["events"]
    except Exception: return out
    for e in ev:
        got = {}
        for bk in e.get("bookmakers", []):
            m = {x["key"]: x["outcomes"] for x in bk.get("markets", [])}
            try:
                pr = {o["name"]: o["price"] for o in m["h2h"]}; ov = {o["name"]: o for o in m["totals"]}
                got[bk["key"]] = (noviG(pr[e["home_team"]], pr[e["away_team"]]), float(ov["Over"]["point"]), noviG(ov["Over"]["price"], ov["Under"]["price"]))
            except Exception: continue
        if not got: continue
        if "pinnacle" in got: p, L, q = got["pinnacle"]
        else:
            v = list(got.values()); L = float(np.median([x[1] for x in v])); p = float(np.median([x[0] for x in v]))
            q = float(np.median([x[2] for x in v if x[1] == L] or [0.5]))
        k = np.arange(16)
        def res(x):      # regulation scores are Poisson; a tie adds one goal to the winner; pushes are refunded
            J = np.outer(poisson.pmf(k, x[0]), poisson.pmf(k, x[1])); tot = k[:, None] + k[None, :] + np.eye(16)
            return [np.tril(J, -1).sum() + 0.5 * np.trace(J) - p, (J * (tot > L)).sum() / max(1 - (J * (tot == L)).sum(), 1e-9) - q]
        x = least_squares(res, [L / 2, L / 2], bounds=(0.8, 7)).x
        J = np.outer(poisson.pmf(k, x[0]), poisson.pmf(k, x[1])); tie = np.trace(J)
        out[(e["home_team"], e["away_team"], e["commence_time"][:10])] = {"lh": round(float(x[0] + tie / 2), 2), "la": round(float(x[1] + tie / 2), 2), "lt": L, "lp": round(p, 3)}
    return out


def main():
    SP = sys.argv[1]; AHEAD = int(sys.argv[2]) if len(sys.argv) > 2 else 1
    dl, ev = f"{SP}/nhlsim_dl", f"{SP}/nhlsim_ev"; os.makedirs(dl, exist_ok=True); os.makedirs(ev, exist_ok=True)
    fetch(f"{B}/nhl_pbp_full/play_by_play_{FILE_YEAR}.parquet", f"{dl}/pbp_{FILE_YEAR}.parquet")
    fetch(f"{B}/nhl_schedules/nhl_schedule_{FILE_YEAR}.parquet", f"{SP}/nhlsim_sched.parquet")
    subprocess.run([sys.executable, f"{HERE}/prep_events.py", dl, ev], check=True, capture_output=True)
    s = pd.read_parquet(f"{ev}/shots.parquet"); t = pd.read_parquet(f"{ev}/segs.parquet")
    s = s[s.season == SEASON]; t = t[t.season == SEASON]
    hist = pd.read_parquet(f"{HERE}/data/hist.parquet"); S = json.load(open(f"{HERE}/data/sums.json"))
    ng = s.game_id.nunique()
    for name, tab in lib.sums(s, t).items(): S[name].update(tab)
    bases = {SEASON: lib.base_of(lib.pool(S, SEASON, prev_w=max(0.0, 1 - ng / 656)))}      # last season fades out by mid-season
    TG = pd.concat([hist[hist.season < SEASON], lib.team_games(s, t, bases)], ignore_index=True).sort_values(["season", "game_id"])
    _, state, gstate, last = lib.rate(TG)
    cur = TG[TG.season == SEASON]
    usual = {tm: d.goalie.dropna().tail(10).mode().iat[0] for tm, d in cur.groupby("team") if d.goalie.notna().any()}
    sch = pd.read_parquet(f"{SP}/nhlsim_sched.parquet"); sch = sch[sch.game_type == "R"].copy()
    sch["k"] = pd.to_datetime(sch.game_time, utc=True); sch["date"] = pd.to_datetime(sch.game_date)
    now = pd.Timestamp.now(tz="UTC")
    today = pd.Timestamp(now.tz_convert("America/New_York").date())
    up = sch[~sch.game_state.isin(["OFF", "FINAL"]) & (sch.k > now) & (sch.date <= today + pd.Timedelta(days=AHEAD))].sort_values("k")
    played = pd.concat([sch[["date", "home_team_abbr"]].rename(columns={"home_team_abbr": "team"}), sch[["date", "away_team_abbr"]].rename(columns={"away_team_abbr": "team"})])
    yest = set(zip(played.team, played.date + pd.Timedelta(days=1)))       # (team, date) pairs that are a second night
    up = up[up.home_team_abbr.isin(state) & up.away_team_abbr.isin(state)].reset_index(drop=True)
    made = now.isoformat(timespec="seconds"); rows = []
    if len(up):
        rt = lambda col: pd.DataFrame([{**lib.now_rating(tm, SEASON, state, last), "gk": lib.goalie_rating(usual.get(tm), gstate)} for tm in up[col]])
        g = pd.DataFrame({"game_id": up.game_id, "season": SEASON,
                          "b2b_h": [float((h, d) in yest) for h, d in zip(up.home_team_abbr, up.date)],
                          "b2b_a": [float((a, d) in yest) for a, d in zip(up.away_team_abbr, up.date)]})
        G = lib.frame(g, rt("home_team_abbr"), rt("away_team_abbr"))
        o = sim.run(G, lib.sim_base(bases), {**lib.PARAMS, "dist": 1}, N=NSIM, seed=int(now.timestamp()) % 100000)
        mk = market(f"{SA}/app/data/odds/NHL.json")
        for i, r in up.iterrows():
            x = o.iloc[i]
            row = {"id": str(int(r.game_id)), "made": made, "k": r.k.isoformat(timespec="seconds"), "d": str(r.game_date)[:10], "h": r.home_team_abbr, "a": r.away_team_abbr,
                   "hs": round(float(x.pred_h), 2), "as": round(float(x.pred_a), 2), "pw": round(float(x.p_home), 3), "pot": round(float(x.p_ot), 3),
                   "dh": [round(float(v), 4) for v in x.dist_h], "da": [round(float(v), 4) for v in x.dist_a],
                   "b2b": [int(G.b2b_h[i]), int(G.b2b_a[i])], "gk": [usual.get(r.home_team_abbr), usual.get(r.away_team_abbr)]}
            m = mk.get((f"{r.home_team_name}", f"{r.away_team_name}", str(r.k)[:10]))
            if m is None:      # odds feed uses full names ("New York Rangers"); match on the city / place part
                for (hn, an, d_), v in mk.items():
                    if d_ == r.k.isoformat()[:10] and hn.startswith(_plain(r.home_team_name)) and an.startswith(_plain(r.away_team_name)): m = v
            if m: row.update(m)
            rows.append(row)
    os.makedirs(f"{HERE}/history", exist_ok=True)
    with open(f"{HERE}/history/log.jsonl", "a") as f:
        for r in rows: f.write(json.dumps({k_: v for k_, v in r.items() if k_ not in ("dh", "da")}) + "\n")
    json.dump({"made": made, "season": SEASON, "games_rated": int(ng), "rest_mult": lib.REST_MULT, "games": rows}, open(f"{SA}/app/data/nhl_sim.json", "w"))
    print("NHL sim:", len(rows), "games,", sum("lh" in r for r in rows), "with a line,", sum(sum(r["b2b"]) for r in rows), "teams on a second night", flush=True)


def _plain(name):
    import unicodedata
    return unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode()


if __name__ == "__main__":
    main()
