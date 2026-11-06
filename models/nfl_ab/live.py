"""NFL Model A + Model B for Bet Desk (J 2026-11-05 rollout; second sport on the two-model system after CBB).
Model A = projected scores: the rebuild drive model (models/rebuild/shadow/history.json from shadow.py: Kalman unit ratings +
          depth-chart starting QB + missing regulars + dome), with key-number score distributions (model_a_test.json: normal x
          key-number weights, calibrated 2018+) for win %, margin / total ranges and the chance of landing on 3 or 7.
Model B = bets built on Model A: sharp-weighted devigged market number (devig.WEIGHT, capped 1.5x; books far off the consensus
          line dropped), the gap after a slate-level total guard, line timing (look-ahead = kickoff 4+ days away, where the 2026
          opener test saw the line move our way), engines agreeing (desk model on the same side), EV at every Colorado book,
          and a tier: A = bet, B = lean (tracked, not a bet), C = pass.
          NFL game tiers are capped at B until live tracking proves them (open_test.json: 2026 openers moved our way 82-86% at
          gaps 3+ but covered only 49-56% on 27-35 games; closes and totals lost). Player props carry NFL's proven edge and are
          tiered on the desk (prop_odds.js, rules in this file's PROPS block, from model-b-confidence props.json).
Run after shadow.py and export_slates.py:  python3 models/nfl_ab/live.py  -> app/data/nfl_ab.json (+ history/runs.jsonl)"""
import json, os, sys, math, datetime as dt, numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, f"{HERE}/../model-b")
import devig as DV
import ml_anchor as MLA
ROOT = os.path.normpath(f"{HERE}/../.."); APP = f"{ROOT}/app/data"
MY_BOOKS = ['draftkings', 'fanduel', 'betmgm', 'williamhill_us', 'espnbet', 'fanatics', 'betrivers', 'ballybet', 'bet365', 'hardrockbet', 'circasports']   # J: every CO-legal book
NAMES = dict(pinnacle='Pinnacle', betonlineag='BetOnline', draftkings='DraftKings', fanduel='FanDuel', betmgm='BetMGM', williamhill_us='Caesars',
             espnbet='ESPN BET', fanatics='Fanatics', betrivers='BetRivers', ballybet='Bally Bet', hardrockbet='Hard Rock', bet365='bet365', circasports='Circa')
TA = json.load(open(f"{HERE}/model_a_test.json")); TO = json.load(open(f"{HERE}/open_test.json"))
SDM, SDT, SDTEAM = TA["sd_margin"], TA["sd_total"], TA["sd_team"]
WM, WT = np.array(TA["w_margin"]), np.array(TA["w_total"]); MG, TG = np.arange(-60, 61), np.arange(0, 121)
EARLY_H = 96            # kickoff 4+ days away = look-ahead line (before the Wednesday injury report), opener-like
B_EARLY, B_LATE = 0.35, 0.0   # share of our spread gap trusted: early from the 2026 opener CLV (moved ~1/3 of a 3+ gap), at the close 0 (2018+ gaps lost)
NOW = dt.datetime.now(dt.timezone.utc)
dec = DV.dec_from_am
# PROPS (desk applies these in prop_odds.js; numbers = model-b-confidence props.json NFL_live, desk projections vs OddsPapi, Oct 2026):
PROPS = {"A": "5%+ EV (ours and blended) on an UNDER, or on receptions either side",
         "B": "5%+ EV on an over of another stat (opener 53.5% / +1.9%, close 43% / -16%): lean, tracked",
         "C": "pass attempts, rushing yards and carries (lost at 5%+ EV at open and close)",
         "tested": {"open_5pct": "854 bets 59.6% +12.8% ROI", "unders_open": "628 bets 61.8% +16.7%", "rec_open": "392 bets 66.6% +26.6%",
                    "unders_close": "519 bets 58.4% +10.6%", "overs_close": "188 bets 43.1% -16.4%", "caveat": "two weeks of games (Oct 16-26 2026), bets share games"}}

def ncdf(x): return 0.5 * (1 + math.erf(x / math.sqrt(2)))
def inv(p):
    a = 0.147; y = 2 * p - 1; s = math.copysign(1, y); l = math.log(1 - y * y); t = 2 / (math.pi * a) + l / 2
    return math.sqrt(2) * s * math.sqrt(math.sqrt(t * t - l / a) - t)
def pmf(mu, grid, sd, w):
    d = np.exp(-0.5 * ((grid - mu) / sd) ** 2) * w; return d / d.sum()
def cover(mu, line):
    """P(home margin + line > 0), P(push) from Model A's key-number margin pmf centred on mu"""
    P = pmf(mu, MG, SDM, WM); v = MG + line; return float(P[v > 0].sum()), float(P[np.isclose(v, 0)].sum())
def over(mu, line):
    P = pmf(mu, TG, SDT, WT); return float(P[TG > line].sum()), float(P[np.isclose(TG, line)].sum())
def q(P, grid, x): return int(grid[np.searchsorted(np.cumsum(P), x)])

def hours_since(ts):
    if not ts: return None
    try: return (NOW - dt.datetime.fromisoformat(ts.replace("Z", "+00:00"))).total_seconds() / 3600
    except ValueError: return None

def market(bk, kind):
    """sharp-weighted devigged market number from every book (home margin 'm' or total 't'); a book whose line sits 3+ pts
    off the median line is dropped (stale or a different market, e.g. a BetOnline alt line)."""
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

def desk_margin(D, h, a, neu=False):
    """the desk NFL model's own margin (bet-desk.html renderNfl 'est', no line): the second engine for 'engines agree'"""
    if not D: return None
    by = {t[0]: t for t in D["teams"]}; H, A = by.get(h), by.get(a)
    if not H or not A: return None
    w, Wi = D["w"], D.get("w_in"); s = H[1] - A[1] + (0 if neu else w["hfa_score"])
    m = H[2] - A[2] + (0 if neu else w["hfa_market"]) if H[2] is not None and A[2] is not None else s
    QG = (D.get("qb") or {}).get("games", {}).get(f"{h}@{a}"); qa = QG["pts"] if QG else 0
    IJ = (D.get("inj") or {}).get("teams", {}); ia = D["inj"]["coef"] * (IJ.get(h, [0])[0] - IJ.get(a, [0])[0]) if IJ else 0
    inS = Wi and D.get("week", 0) >= Wi["from_week"] and H[5] is not None and A[5] is not None
    if inS: est = Wi["const"] * (0 if neu else 1) + Wi["score"] * s + Wi["market"] * m + Wi["season"] * (H[5] - A[5] + (0 if neu else Wi["hfa_season"]))
    else: est = w["const"] * (0 if neu else 1) + w["score"] * s + w["market"] * m
    return est + qa + ia

def best_price(bk, po, side_home, kind, mu):
    out = None
    for b in MY_BOOKS:
        r = bk.get(b)
        if not r: continue
        pt, ph, pa = (r[0], r[1], r[2]) if kind == "m" else (r[3], r[4], r[5])
        if pt is None or not ph or not pa: continue
        if kind == "m": pc, pp = cover(mu, pt); p, pr, lab = ((pc, ph, pt) if side_home else (1 - pc - pp, pa, -pt))
        else: po_, pp = over(mu, pt); p, pr, lab = ((po_, ph, pt) if side_home else (1 - po_ - pp, pa, pt))
        ev = p * (dec(pr) - 1) - (1 - p - pp)   # a push returns the stake
        age = hours_since((po.get(b) or [None] * 3)[0 if kind == "m" else 1])
        if out is None or ev > out["ev"]:
            out = dict(book=b, book_name=NAMES.get(b, b), pt=lab, price=pr, p=round(p, 4), push=round(pp, 4), ev=round(ev, 4), age_h=None if age is None else round(age, 1))
    return out

def main():
    hist = json.load(open(f"{ROOT}/models/rebuild/shadow/history.json"))
    S = json.load(open(f"{APP}/game_slates.json"))["NFL"]; slate = {x["id"]: x for x in S["games"]}
    D = json.load(open(f"{APP}/games_models.json")).get("NFL")
    G = {v["id"]: v for v in hist.values() if v.get("sp") == "NFL" and v.get("k") and dt.datetime.fromisoformat(v["k"].replace("Z", "+00:00")) > NOW - dt.timedelta(hours=4)}
    # SLATE TOTAL GUARD: 2026 Model A team scores run +1.1 above the market's implied team totals (open_test.json), so total
    # gaps are measured against the slate's median gap (as CBB's level guard). Model A itself is not changed.
    tg = []
    for gid, v in G.items():
        x = slate.get(gid); ln = x and x.get("ln")
        if ln and ln.get("bk"):
            t, _ = market(ln["bk"], "t")
            if t is not None: tg.append(v["hs"] + v["as"] - t)
    TSHIFT = round(float(np.median(tg)), 2) if len(tg) >= 8 else round(2 * TO["A_yardstick_2026_close_consensus"]["A_minus_line_mean"], 2)
    out = {"made": NOW.strftime("%Y-%m-%d %H:%M UTC"), "sport": "NFL", "from": {"model_a": "models/rebuild/shadow/history.json", "slates": S.get("made") or S.get("src")},
           "how": {"A": f"Rebuild drive model (ratings, depth-chart QB, missing regulars, dome) + key-number score distribution (margin sd {SDM:.1f}, total sd {SDT:.1f})",
                   "B": "Sharp-weighted devigged market number; look-ahead lines (kickoff 4+ days out) trust about a third of a spread gap, later lines none; "
                        "slate total guard. NFL game tiers max B (lean) until live tracking proves them; player props carry the tier A bets."},
           "tested": {"A_vs_close_2018_2026": TO["A_yardstick_2018_2026_nflverse_close"], "A_vs_close_2026": TO["A_yardstick_2026_close_consensus"],
                      "open_2026_spread_dk": TO["spread_open_dk"]["by_gap"], "open_2026_total_dk": TO["total_open_dk"]["by_gap"]},
           "props": PROPS, "total_guard": TSHIFT, "games": {}}
    for gid, v in sorted(G.items(), key=lambda z: z[1]["k"]):
        hs, as_ = v["hs"], v["as"]; mA, tA = hs - as_, hs + as_
        Pm, Pt = pmf(mA, MG, SDM, WM), pmf(tA, TG, SDT, WT)
        p_home = float(Pm[MG > 0].sum() + 0.5 * Pm[MG == 0].sum())
        A = dict(hs=hs, p_home=round(p_home, 4), m10=q(Pm, MG, .1), m90=q(Pm, MG, .9), t10=q(Pt, TG, .1), t90=q(Pt, TG, .9),
                 h10=round(max(0, hs - 1.2816 * SDTEAM)), h90=round(hs + 1.2816 * SDTEAM), a10=round(max(0, as_ - 1.2816 * SDTEAM)), a90=round(as_ + 1.2816 * SDTEAM),
                 p3=round(float(Pm[np.abs(MG) == 3].sum()), 3), p7=round(float(Pm[np.abs(MG) == 7].sum()), 3), made=v.get("made"))
        A["as"] = as_
        x = slate.get(gid); kick_h = -hours_since(v["k"]); early = kick_h >= EARLY_H
        dm = desk_margin(D, v["h"], v["a"], bool(x and x.get("n")))
        rec = {"h": v["h"], "a": v["a"], "d": v["d"], "k": v["k"], "w": x and x.get("w"), "A": A, "desk_m": None if dm is None else round(dm, 2), "B": None}
        ln = x and x.get("ln")
        if ln and ln.get("bk"):
            bk, po = ln["bk"], ln.get("po") or {}; B = {}
            mu_m, used = market(bk, "m")
            if mu_m is not None:
                gap = mA - mu_m; side_home = gap > 0
                agree = dm is not None and np.sign(dm - mu_m) == np.sign(gap)
                share = B_EARLY if early else B_LATE; real = gap * share
                bp = best_price(bk, po, side_home, "m", mu_m + real)
                tier = "B" if early and abs(gap) >= 3 and agree else "C"
                R = [["+" if early else "-", f"Kickoff in {kick_h / 24:.1f} days: " + ("look-ahead line; in the 2026 opener test lines moved our way 82-86% of the time at gaps of 3+" if early else "late line; our NFL spread gaps have not held up at the close (2018-26)")],
                     ["+" if agree else "-", "The desk model is on the same side of the line" if agree else "The desk model is on the other side of the line"],
                     ["·", "NFL spreads are a lean at most: the 2026 openers covered 49-56% (27-35 games) and the backtest knew the starting QB, which an opener often does not"]]
                if abs(gap) >= 3: R.append(["·", f"Key numbers: Model A puts {A['p3']:.0%} on a 3-point margin and {A['p7']:.0%} on 7"])
                B["spread"] = dict(market=round(mu_m, 2), books=used, ours=round(mA, 2), gap=round(gap, 2), share=share, real=round(real, 2), fresh=early, agree=bool(agree),
                                   tier=tier, side=v["h"] if side_home else v["a"], bet=bp, reasons=R)
            mu_t, used_t = market(bk, "t")
            if mu_t is not None:
                gap_t = tA - mu_t - TSHIFT; ov = gap_t > 0
                bp = best_price(bk, po, ov, "t", mu_t)
                B["total"] = dict(market=round(mu_t, 2), books=used_t, ours=round(tA, 2), gap=round(gap_t, 2), real=0.0, fresh=early, tier="C", side="Over" if ov else "Under", bet=bp,
                                  reasons=[["·", f"Measured against this slate: Model A sits {TSHIFT:+.1f} pts vs the market's totals on the median game, so only the part beyond that counts"],
                                           ["-", "NFL total gaps have lost: 2026 openers 3+ hit 39-44%, 2026 closes 43-48%. Shown for the number only, never a bet"]])
            sp_ = B.get("spread")
            if sp_:
                mu_B = sp_["market"] + sp_["real"]; best = None
                F = lambda mu: float(pmf(mu, MG, SDM, WM)[MG > 0].sum() + 0.5 * pmf(mu, MG, SDM, WM)[MG == 0].sum())
                ph, pm = MLA.home_chance(bk, "NFL", F, mu_B, sp_["market"])   # anchored on the books' moneyline (J 2026-11-06)
                for b in MY_BOOKS:
                    r = bk.get(b)
                    if not r or not r[6] or not r[7]: continue
                    for home, pr in ((True, r[6]), (False, r[7])):
                        if home != (sp_["gap"] > 0): continue
                        p = ph if home else 1 - ph; ev = p * dec(pr) - 1
                        if best is None or ev > best["ev"]: best = dict(book=b, book_name=NAMES.get(b, b), pt=None, price=pr, p=round(p, 4), ev=round(ev, 4), age_h=None)
                if best:
                    tier = "B" if sp_["tier"] == "B" and best["ev"] >= 0.03 and abs(mu_B) <= 7 else "C"
                    B["ml"] = dict(mkt_p=None if pm is None else round(pm if sp_["gap"] > 0 else 1 - pm, 4), market=sp_["market"], ours=sp_["ours"], gap=sp_["gap"], real=sp_["real"], fresh=sp_["fresh"], tier=tier, side=sp_["side"], bet=best,
                                   reasons=[["·", "Moneyline prices the spread's trusted edge as a win chance (key-number margin curve); a lean at most, only near pick'em. Win chance = the books' no-vig moneyline moved only by that trusted edge"]])
            ts = [z for z in B.values() if z["tier"] != "C" and z["bet"]]
            B["best"] = min(ts, key=lambda z: ("AB".index(z["tier"]), -z["bet"]["ev"]))["tier"] if ts else "C"
            rec["B"] = B
        out["games"][gid] = rec
    clean = lambda o: {k: clean(x) for k, x in o.items()} if isinstance(o, dict) else [clean(x) for x in o] if isinstance(o, list) else (None if isinstance(o, float) and o != o else o)
    import unproven; unproven.annotate("NFL", out)   # unproven rating on tier C passes (J 2026-11-06)
    # J 2026-11-06: a started game keeps its last pre-game record (frozen) so the desk can list the whole day (models/model-b/pregame.py)
    from pregame import keep
    json.dump(clean(dict(out, games=keep(f"{APP}/nfl_ab.json", out["games"], kick=lambda i, r: r.get("k")))), open(f"{APP}/nfl_ab.json", "w"), separators=(",", ":"), allow_nan=False)
    os.makedirs(f"{HERE}/history", exist_ok=True)
    with open(f"{HERE}/history/runs.jsonl", "a") as fh:
        for gid, g in out["games"].items():
            fh.write(json.dumps({"t": out["made"], "id": gid, "h": g["h"], "a": g["a"], "d": g["d"], "A": [g["A"]["hs"], g["A"]["as"]], "desk_m": g["desk_m"],
                                 "B": {k: {kk: z.get(kk) for kk in ("market", "gap", "real", "fresh", "tier", "side", "bet")} for k, z in (g["B"] or {}).items() if isinstance(z, dict)}}) + "\n")
    tiers = {}
    for g in out["games"].values():
        for k in ("spread", "total", "ml"):
            z = g["B"] and g["B"].get(k)
            if z: tiers[f"{k}_{z['tier']}"] = tiers.get(f"{k}_{z['tier']}", 0) + 1
    print(len(out["games"]), "games ->", f"{APP}/nfl_ab.json |", sum(1 for g in out["games"].values() if g["B"]), "with lines | tiers", tiers, "| total guard", TSHIFT)
    for gid, g in out["games"].items():
        z = g["B"] and g["B"].get("spread")
        if z: print(f"  {g['a']}@{g['h']} A {g['A']['as']:.1f}-{g['A']['hs']:.1f} desk {g['desk_m']} mkt {z['market']:+.1f} gap {z['gap']:+.1f} {z['tier']} {z['side']} early={z['fresh']} agree={z['agree']} | tot gap {g['B'].get('total', {}).get('gap')}")

if __name__ == "__main__": main()
