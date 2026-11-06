"""CFB Model A + Model B for Bet Desk (J 2026-11-05 rollout; fourth sport on the two-model system after CBB, NFL, NBA).
Model A = projected scores: CFB rebuild v4 means (app/data/cfb_live.json from cfb_gap/live/run_live.sh: desk v3 unblended + program
          SRS + coach career win% + crowd size; total v3 + in-season tempo), PLUS two inputs v4 lacked:
          - player usage prior (app/data/cfb_usage.json 'shift', prop-implied offense, fitted on 2026 games), split across both teams;
          - missing stars (absence_test.json): starting QB out (elite 225+ yds/g -2.5 team pts, other -0.5), top skill player out -1.0.
            Sources: absences.json (manual) and cfb_usage.json stars_out (no prop posted within 36 h of kickoff). Size from 2022-25
            play-by-play: elite QB first game out, team scored 2.1 under Model A and the close was already 1.9 under.
          Ranges from the drive simulator (lib.Sim, beats the bell on exact team scores), margin / cover / win % from the key-number
          curve (keynum.json: sim under-produces 3- and 7-point margins).
Model B = bets on Model A: sharp-weighted devigged market (devig.WEIGHT, capped 1.5x; books 3+ pts off the median dropped), line timing
          (look-ahead = kickoff 4+ days out, opener-like), trusted share of the gap (b: opener .44, close .34; model_b_test.json), EV at
          every Colorado book, tier A bet / B lean / C pass.
          Spreads: look-ahead gap 4+ = A (2022-25 openers 58.2%, 866 bets, every season 52-63%), 3-4 = B; later lines gap 4+ = B
          (closes 55.0%, 1005), else C. Totals: look-ahead UNDER gap 4+ = B (openers 58.0%, 343; 3 of 4 seasons), overs and later C.
          ML lean max. Confidence model (research features) was a NET NEGATIVE for v4 (tier A 58.5% vs raw top-10% gap 61.5%) -> raw gap.
          College player props are game-model inputs only (no CO college props).
Run after run_live.sh (cfb_live.json), cfb_usage.py and export_slates.py:  python3 models/cfb_ab/live.py -> app/data/cfb_ab.json"""
import json, os, sys, math, datetime as dt, numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE); sys.path.insert(0, f"{HERE}/../model-b")
import devig as DV
from lib import Sim
ROOT = os.path.normpath(f"{HERE}/../.."); APP = f"{ROOT}/app/data"
MY_BOOKS = ['draftkings', 'fanduel', 'betmgm', 'williamhill_us', 'espnbet', 'fanatics', 'betrivers', 'ballybet', 'bet365', 'hardrockbet', 'circasports']   # J: every CO-legal book
NAMES = dict(pinnacle='Pinnacle', betonlineag='BetOnline', draftkings='DraftKings', fanduel='FanDuel', betmgm='BetMGM', williamhill_us='Caesars',
             espnbet='ESPN BET', fanatics='Fanatics', betrivers='BetRivers', ballybet='Bally Bet', hardrockbet='Hard Rock', bet365='bet365', circasports='Circa')
KN = json.load(open(f"{HERE}/keynum.json")); TA = json.load(open(f"{HERE}/model_a_test.json")); TB = json.load(open(f"{HERE}/model_b_test.json"))
SDM, WM = KN["sd_margin"], np.array(KN["w"]); MG = np.arange(KN["grid"][0], KN["grid"][1] + 1); SDT = TA["normal_sd"]["total"]
EARLY_H = 96
SHARE = {"early": TB["open"]["b_v4"], "late": TB["close"]["b_v4"], "total_early": TB["open"]["b_total_v4"]}
ABS = {"qb_e": -2.5, "qb_n": -0.5, "skill": -1.0}   # team points (absence_test.json, shrunk; skill elite/non-elite pooled)
NOW = dt.datetime.now(dt.timezone.utc); dec = DV.dec_from_am
NSIM = int(os.environ.get("CFB_NSIM", 4000))

def inv(p):
    a = 0.147; y = 2 * p - 1; s = math.copysign(1, y); l = math.log(1 - y * y); t = 2 / (math.pi * a) + l / 2
    return math.sqrt(2) * s * math.sqrt(math.sqrt(t * t - l / a) - t)
def pmf(mu):
    d = np.exp(-0.5 * ((MG - mu) / SDM) ** 2) * WM; return d / d.sum()
def cover(mu, line):
    P = pmf(mu); v = MG + line; return float(P[v > 0].sum()), float(P[np.isclose(v, 0)].sum())
def over(mu, line):
    z = (line - mu) / SDT; p = 0.5 * math.erfc(z / math.sqrt(2)); return p, 0.0   # totals: no key numbers
def hours_to(ts): return (dt.datetime.fromisoformat(ts.replace("Z", "+00:00")) - NOW).total_seconds() / 3600
def hours_since(ts):
    if not ts: return None
    try: return (NOW - dt.datetime.fromisoformat(ts.replace("Z", "+00:00"))).total_seconds() / 3600
    except ValueError: return None
norm = lambda s: "".join(ch for ch in (s or "").lower().replace("state", "st").replace("&", "and") if ch.isalnum())

def market(bk, kind):
    i = 0 if kind == "m" else 3
    pts = [r[i] for r in bk.values() if r[i] is not None and r[i + 1] and r[i + 2]]
    if not pts: return None, []
    med = float(np.median(pts)); acc = tw = 0.0; used = []
    for b, r in bk.items():
        pt, p1, p2 = r[i], r[i + 1], r[i + 2]
        if pt is None or not p1 or not p2 or abs(pt - med) >= 3: continue
        p = min(max(DV.power([dec(p1), dec(p2)])[0], 0.02), 0.98)
        mu = (-pt + SDM * inv(p)) if kind == "m" else (pt + SDT * inv(p))
        w = min(DV.WEIGHT.get(b, 0.8), DV.MAX_RATIO); acc += w * mu; tw += w; used.append(b)
    return (acc / tw, used) if tw else (None, [])

def best_price(bk, po, side_home, kind, mu):
    out = None
    for b in MY_BOOKS:
        r = bk.get(b)
        if not r: continue
        pt, ph, pa = (r[0], r[1], r[2]) if kind == "m" else (r[3], r[4], r[5])
        if pt is None or not ph or not pa: continue
        if kind == "m": pc, pp = cover(mu, pt); p, pr, lab = ((pc, ph, pt) if side_home else (1 - pc - pp, pa, -pt))
        else: po_, pp = over(mu, pt); p, pr, lab = ((po_, ph, pt) if side_home else (1 - po_ - pp, pa, pt))
        ev = p * (dec(pr) - 1) - (1 - p - pp)
        age = hours_since((po.get(b) or [None] * 3)[0 if kind == "m" else 1])
        if out is None or ev > out["ev"]:
            out = dict(book=b, book_name=NAMES.get(b, b), pt=lab, price=pr, p=round(p, 4), push=round(pp, 4), ev=round(ev, 4), age_h=None if age is None else round(age, 1))
    return out

def star_table():
    """2026 play-by-play: each team's current starting QB and top skill player with per-game yards (elite rule as absence_test)"""
    f = f"{HERE}/data/usage_games.parquet"
    if not os.path.exists(f): return {}
    U = pd.read_parquet(f); U = U[U.season == U.season.max()].copy(); U["scr"] = U.rush_yds + U.rec_yds; out = {}
    for t, g in U.groupby("team"):
        P = g.groupby("player").agg(pa=("pass_att", "sum"), py=("pass_yds", "sum"), scr=("scr", "sum"), n=("game_id", "nunique"), npass=("pass_att", lambda x: (x >= 8).sum()))
        out[int(t)] = {p: dict(qb_ypg=round(float(r.py / max(r.npass, 1)), 1) if r.npass else 0.0, scr_ypg=round(float(r.scr / r.n), 1), games=int(r.n)) for p, r in P.iterrows()}
    return out

def absences(games, use_by_game):
    """{gid: [dict(team, side, player, kind, elite, pts, src)]}"""
    O = json.load(open(f"{HERE}/absences.json"))["out"]; ST = star_table()
    S26 = {}
    try:
        s = pd.read_csv(f"{ROOT}/models/rebuild/football/cfb_gap/live/cfbros/s_2026.csv", usecols=["game_id", "home_id", "away_id"]); S26 = {str(r.game_id): (int(r.home_id), int(r.away_id)) for r in s.itertuples()}
    except Exception: pass
    out = {}
    for gid, g in games.items():
        L = []; ids = S26.get(gid, (None, None)); kick_d = g["k"][:10]
        cand = [dict(team=o["team"], player=o["player"], kind=o["kind"], elite=o.get("elite"), src=o.get("src", "manual")) for o in O if o.get("until", "9999") >= kick_d]
        u = use_by_game.get(gid)
        if u:
            for s in u.get("stars_out") or []:
                cand.append(dict(team=s["team"], player=s.get("name") or s["pid"], kind="qb" if s["kind"] == "pass" else "skill", elite=s.get("elite"),
                                 src=f"no {s['kind']} prop posted (line usually about {s.get('avg_line')})"))
        seen = set()
        for c in cand:
            side = "h" if norm(c["team"]) == norm(g["h"]) else "a" if norm(c["team"]) == norm(g["a"]) else None
            if side is None or (side, norm(c["player"])) in seen: continue
            seen.add((side, norm(c["player"])))
            st = ST.get(ids[0 if side == "h" else 1], {}).get(c["player"], {}) if ids[0] else {}
            el = c["elite"] if c["elite"] is not None else (st.get("qb_ypg", 0) >= 225 if c["kind"] == "qb" else st.get("scr_ypg", 0) >= 90)
            pts = ABS["qb_e" if el else "qb_n"] if c["kind"] == "qb" else ABS["skill"]
            L.append(dict(team=g["h" if side == "h" else "a"], side=side, player=c["player"], kind=c["kind"], elite=bool(el), pts=pts, src=c["src"],
                          ypg=st.get("qb_ypg") if c["kind"] == "qb" else st.get("scr_ypg")))
        if L: out[gid] = L
    return out

def main():
    CL = json.load(open(f"{APP}/cfb_live.json")); S = json.load(open(f"{APP}/game_slates.json"))["CFB"]; slate = {x["id"]: x for x in S["games"]}
    U = json.load(open(f"{APP}/cfb_usage.json")) if os.path.exists(f"{APP}/cfb_usage.json") else {"games": []}
    G = {g["id"]: g for g in CL["games"] if hours_to(g["k"]) > -4}
    use_by_game = {}
    for gid, g in G.items():   # cfb_usage keys are OddsPapi fixtures: match by team names (either order) and kickoff within 1.5 days
        for u in U.get("games", []):
            if abs((dt.datetime.fromisoformat(u["k"].replace("Z", "+00:00")) - dt.datetime.fromisoformat(g["k"].replace("Z", "+00:00"))).total_seconds()) > 1.5 * 86400: continue
            if {norm(u["h"]), norm(u["a"])} == {norm(g["h"]), norm(g["a"])}:
                use_by_game[gid] = dict(u, flip=norm(u["h"]) != norm(g["h"])); break
    AB = absences(G, use_by_game); SIM = Sim()
    out = {"made": NOW.strftime("%Y-%m-%d %H:%M UTC"), "sport": "CFB", "from": {"model_a": CL.get("made"), "slates": S.get("src"), "usage": U.get("made")},
           "how": {"A": "CFB rebuild v4 (desk v3 unblended + program SRS + coach career win% + crowd size; total + in-season tempo) + player usage prior + missing starters; "
                        f"drive-simulator ranges, key-number margin curve (sd {SDM:.1f})",
                   "B": "Sharp-weighted devigged market; look-ahead lines (kickoff 4+ days out) trust 44% of a spread gap, later lines 34%. Spread A = look-ahead gap 4+, "
                        "B = look-ahead 3-4 or later gap 4+; totals: look-ahead unders 4+ lean, the rest pass; moneyline lean at most."},
           "tested": {"A_yardstick_2022_25": TA["yardstick_2022_25"], "sim": {k: TA["sim"][k] for k in ("band80", "team_score_log", "team_total_over_brier")},
                      "keynum": KN["test_2022_25"]["keynum"], "open": {k: TB["open"][k] for k in ("spread_v4", "total_v4_over_under_4plus", "clv_pts_by_gap")},
                      "close": {k: TB["close"][k] for k in ("spread_v4",)}, "confidence_net_negative": {"tier_A": TB["open_confidence"]["tiers"]["A"], "raw_top10": TB["open_confidence"]["raw_gap_top10"]},
                      "absences": "absence_test.json"},
           "absence_pts": ABS, "games": {}}
    tg = []
    for gid, v in sorted(G.items(), key=lambda z: z[1]["k"]):
        u = use_by_game.get(gid); ush = (-1 if u and u["flip"] else 1) * float(u["shift"]) if u and u.get("shift") else 0.0
        ab = AB.get(gid, []); dh = sum(a["pts"] for a in ab if a["side"] == "h"); da = sum(a["pts"] for a in ab if a["side"] == "a")
        hs, as_ = v["hs"] + ush / 2 + dh, v["as"] - ush / 2 + da; mA, tA = hs - as_, hs + as_
        h, a = SIM.run(hs, as_, NSIM); P = pmf(mA)
        A = dict(hs=round(hs, 2), p_home=round(float(P[MG > 0].sum()), 4), m10=int(np.quantile(h - a, .1)), m90=int(np.quantile(h - a, .9)),
                 t10=int(np.quantile(h + a, .1)), t90=int(np.quantile(h + a, .9)), h10=int(np.quantile(h, .1)), h90=int(np.quantile(h, .9)),
                 a10=int(np.quantile(a, .1)), a90=int(np.quantile(a, .9)), p3=round(float(P[np.abs(MG) == 3].sum()), 3), p7=round(float(P[np.abs(MG) == 7].sum()), 3),
                 v4=[v["hs"], v["as"]], usage=round(ush, 2), absences=ab)
        A["as"] = round(as_, 2)
        x = slate.get(gid) or {}; kick_h = hours_to(v["k"]); early = kick_h >= EARLY_H
        dm = (v["v3_hs"] - v["v3_as"]) if v.get("v3_hs") is not None else None
        rec = {"h": v["h"], "a": v["a"], "d": x.get("d") or v["k"][:10], "k": v["k"], "w": v.get("wk"), "n": v.get("n"), "A": A, "desk_m": None if dm is None else round(dm, 2), "B": None}
        ln = x.get("ln")
        if ln and ln.get("bk"):
            bk, po = ln["bk"], ln.get("po") or {}; B = {}
            mu_m, used = market(bk, "m")
            if mu_m is not None:
                gap = mA - mu_m; side_home = gap > 0; share = SHARE["early" if early else "late"]; real = gap * share
                agree = dm is not None and np.sign(dm - mu_m) == np.sign(gap); fav = abs(mu_m)
                bp = best_price(bk, po, side_home, "m", mu_m + real)
                tier = ("A" if abs(gap) >= 4 else "B" if abs(gap) >= 3 else "C") if early else ("B" if abs(gap) >= 4 else "C")
                R = [["+" if early else "·", f"Kickoff in {kick_h / 24:.1f} days: " + ("look-ahead line, opener-like: 2022-25 openers with a gap of 4+ covered 58% (866 bets, 52-63% every season)" if early
                      else "game-week line: gaps of 4+ at the close still covered 55% (1,005 bets, 2022-25), so a lean at most")],
                     ["+" if agree else "-", "The desk model (v3) is on the same side of the line" if agree else "The desk model (v3) is on the other side of the line"]]
                if 7.5 <= fav <= 24: R.append(["+", "Favourite of 7.5-24: the size where our CFB gaps have held up best at the opener"])
                if (v.get("wk") or 0) >= 8: R.append(["+", "Week 8+: Model A's in-season ratings carry the most weight by now"])
                if ab: R.append(["·", "Missing starters priced into Model A: " + "; ".join(f"{z['player']} ({z['team']}, {'QB' if z['kind'] == 'qb' else 'top skill player'}{', elite' if z['elite'] else ''}) {z['pts']:+.1f}" for z in ab)])
                if ush: R.append(["·", f"Player usage prior (prop-implied offense): {ush:+.1f} pts to {v['h']}, included"])
                if abs(gap) >= 3: R.append(["·", f"Key numbers: Model A puts {A['p3']:.0%} on a 3-point margin and {A['p7']:.0%} on 7"])
                if tier != "C" and (not bp or bp["ev"] < 0): tier = "C"; R.append(["-", "No Colorado book offers a positive-EV price on this side right now"])
                B["spread"] = dict(market=round(mu_m, 2), books=used, ours=round(mA, 2), gap=round(gap, 2), share=share, real=round(real, 2), fresh=early, agree=bool(agree),
                                   tier=tier, side=v["h"] if side_home else v["a"], bet=bp, reasons=R)
            mu_t, used_t = market(bk, "t")
            if mu_t is not None:
                tg.append(tA - mu_t); gap_t = tA - mu_t; ov = gap_t > 0; real_t = gap_t * (SHARE["total_early"] if early else 0.0)
                bp = best_price(bk, po, ov, "t", mu_t + real_t)
                tier = "B" if early and not ov and gap_t <= -4 and bp and bp["ev"] >= 0 else "C"
                RT = [["+" if tier == "B" else "-", "Look-ahead under with a gap of 4+: 2022-25 openers hit 58% (343 bets, 3 of 4 seasons); a lean until live tracking" if tier == "B" else
                       ("Overs at the opener lost (49.5%, 489 bets)" if early and ov else "Total gaps below 4 or on game-week lines have not held up (2022-25 closes 51%)")],
                      ["·", f"Tempo: Model A's total uses both teams' plays per game this season, shrunk toward each head coach's pace"]]
                B["total"] = dict(market=round(mu_t, 2), books=used_t, ours=round(tA, 2), gap=round(gap_t, 2), real=round(real_t, 2), share=SHARE["total_early"] if early else 0.0,
                                  fresh=early, tier=tier, side="Over" if ov else "Under", bet=bp, reasons=RT)
            sp_ = B.get("spread")
            if sp_:
                mu_B = sp_["market"] + sp_["real"]; PB = pmf(mu_B); ph = float(PB[MG > 0].sum()); best = None
                for b in MY_BOOKS:
                    r = bk.get(b)
                    if not r or not r[6] or not r[7]: continue
                    for home, pr in ((True, r[6]), (False, r[7])):
                        if home != (sp_["gap"] > 0): continue
                        p = ph if home else 1 - ph; ev = p * dec(pr) - 1
                        if best is None or ev > best["ev"]: best = dict(book=b, book_name=NAMES.get(b, b), pt=None, price=pr, p=round(p, 4), ev=round(ev, 4), age_h=None)
                if best:
                    tier = "B" if sp_["tier"] in "AB" and best["ev"] >= 0.03 and abs(mu_B) <= 7 else "C"
                    B["ml"] = dict(market=sp_["market"], ours=sp_["ours"], gap=sp_["gap"], real=sp_["real"], fresh=sp_["fresh"], tier=tier, side=sp_["side"], bet=best,
                                   reasons=[["·", "Moneyline prices the spread's trusted edge as a win chance (key-number margin curve); untested on prices, a lean at most, only near pick'em"]])
            ts = [z for z in B.values() if z["tier"] != "C" and z["bet"]]
            B["best"] = min(ts, key=lambda z: ("AB".index(z["tier"]), -z["bet"]["ev"]))["tier"] if ts else "C"
            rec["B"] = B
        out["games"][gid] = rec
    # slate total check (NBA rule: a guard only when the whole slate sits 3+ pts off; a normal slate is left alone)
    med = float(np.median(tg)) if len(tg) >= 8 else 0.0; out["total_gap_median"] = round(med, 2)
    if abs(med) >= 3:
        out["total_guard"] = round(med, 2)
        for g in out["games"].values():
            z = g["B"] and g["B"].get("total")
            if z: z["gap"] = round(z["gap"] - med, 2); z["tier"] = "C"; z["reasons"].append(["-", f"Whole slate sits {med:+.1f} pts off the market on totals: a level problem, so totals pass today"])
    clean = lambda o: {k: clean(x) for k, x in o.items()} if isinstance(o, dict) else [clean(x) for x in o] if isinstance(o, list) else (None if isinstance(o, float) and o != o else o)
    # J 2026-11-06: a started game keeps its last pre-game record (frozen) so the desk can list the whole day (models/model-b/pregame.py)
    from pregame import keep
    json.dump(clean(dict(out, games=keep(f"{APP}/cfb_ab.json", out["games"], kick=lambda i, r: r.get("k")))), open(f"{APP}/cfb_ab.json", "w"), separators=(",", ":"), allow_nan=False)
    os.makedirs(f"{HERE}/history", exist_ok=True)
    with open(f"{HERE}/history/runs.jsonl", "a") as fh:
        for gid, g in out["games"].items():
            fh.write(json.dumps({"t": out["made"], "id": gid, "h": g["h"], "a": g["a"], "d": g["d"], "k": g["k"], "A": [g["A"]["hs"], g["A"]["as"]], "v4": g["A"]["v4"], "desk_m": g["desk_m"],
                                 "abs": [[z["side"], z["player"], z["pts"]] for z in g["A"]["absences"]],
                                 "B": {k: {kk: z.get(kk) for kk in ("market", "gap", "real", "fresh", "tier", "side", "bet")} for k, z in (g["B"] or {}).items() if isinstance(z, dict)}}) + "\n")
    tiers = {}
    for g in out["games"].values():
        for k in ("spread", "total", "ml"):
            z = g["B"] and g["B"].get(k)
            if z: tiers[f"{k}_{z['tier']}"] = tiers.get(f"{k}_{z['tier']}", 0) + 1
    print(len(out["games"]), "games ->", f"{APP}/cfb_ab.json |", sum(1 for g in out["games"].values() if g["B"]), "with lines | tiers", tiers, "| usage matched", len(use_by_game),
          "| absences", sum(len(v) for v in AB.values()), "| slate total gap median", round(med, 2))
    for gid, g in out["games"].items():
        for z in g["A"]["absences"]: print("  OUT", g["a"], "@", g["h"], "|", z["player"], z["team"], z["kind"], "elite" if z["elite"] else "", z["pts"], "|", z["src"])
    for gid, g in out["games"].items():
        z = g["B"] and g["B"].get("spread")
        if z and z["tier"] != "C": print(f"  {g['a']}@{g['h']} A {g['A']['as']:.1f}-{g['A']['hs']:.1f} mkt {z['market']:+.1f} gap {z['gap']:+.1f} {z['tier']} {z['side']} early={z['fresh']} ev={z['bet'] and z['bet']['ev']}")

if __name__ == "__main__": main()
