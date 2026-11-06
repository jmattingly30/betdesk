"""Unproven confidence rating for Model B passes (J 2026-11-06: "include a confidence rating or a recommendation for bets even
on our less proven models ... so I can track them"). Tier A/B logic is untouched: this only rates tier C (pass) picks that have a
price, so J can see and track them. Every rating is labelled UNPROVEN and carries the reason plus the sample it rests on.

How the score works (0-100, 50 = no edge):
  1. Model A alone: its full gap is trusted (Model B trusts only `real` of it). Chance at the shown line/price:
     pA = Phi(Phi^-1(pB) + (|gap| - |real|) / sd), sd = the sport's margin or total spread. evA = pA * decimal - 1.
     Model A alone is overconfident (that is why Model B shrinks it), so the score uses the half-way chance
     pU = Phi(Phi^-1(pB) + (|gap| - |real|) / (2 sd)) and evU = pU * decimal - 1.
  2. score = 50 + 40 * tanh(evU / 6%), +5 if Model B's own EV is >= 0, -10 if it is below -3%,
     then the tested record of this kind of bet (HIST): lost -15, flat -5, won +5, untested -5. Capped at 95 (60 for moneylines).
  3. rec: "play" (small, unproven) = score 65+, evU 4%+, Model B EV >= 0, the bucket did not lose in testing, not a moneyline
     (ML chances are converted from the spread and untested); "watch" = score 50+ and evU 2%+; else "skip".
The rating is a pure function of fields already logged in each sport's history/runs.jsonl (gap, real, fresh, bet), so grade.py
re-derives it and grades each rec bucket (keys like spread_U_play) without changing what live.py logs.
Use: from unproven import annotate; annotate("CBB", out)   (after Model B tiers are set, before json.dump)"""
import math

SD = {"CBB": (10.93, 17.14), "NBA": (13.7, 18.72), "NFL": (13.4, 13.5), "CFB": (16.24, 16.89)}   # (margin, total) from each Model A test

# What testing says about this kind of pass: (verdict, n bets or 0 when the source gives no count, text); verdict = won (52.4%+),
# flat (50-52.4%), lost (under 50%) or None (untested); optional 4th item = smallest gap the test covered. A side-specific key
# (sport, kind, early, side) wins over the plain one. Keys: (sport, kind, early?) with early = opener /
# look-ahead line (z["fresh"]). Numbers come from each sport's *_ab.json "tested" block and README.
HIST = {
    ("CBB", "spread", True):  ("flat", 3967, "CBB spread passes at the 2026 opener won 51.3% (3,967)"),
    ("CBB", "spread", False): (None, 0, "CBB spread gaps under 6 at older lines: only big gaps held up at the close"),
    ("CBB", "total", True):   (None, 0, "CBB total gaps under 3 at the opener: untested size"),
    ("CBB", "total", False):  ("lost", 0, "CBB totals at close-like lines lost at every gap size (2020-26)"),
    ("NBA", "spread", True):  ("lost", 466, "NBA spread passes at the 2026 DK opener won 48.1% (466)"),
    ("NBA", "spread", False): ("lost", 0, "NBA spreads at close-like lines lost at every gap size (2018-26)"),
    ("NBA", "total", True):   (None, 0, "NBA total gaps under 3 at the opener: untested size"),
    ("NBA", "total", False):  ("lost", 0, "NBA totals at close-like lines lost at every gap size (2018-26)"),
    ("NFL", "spread", True):  ("lost", 35, "NFL spread gaps 3+ at 2026 openers covered 48.6% (35)", 3),
    ("NFL", "spread", False): ("lost", 0, "NFL spread gaps have not held up at the close (2018-26)"),
    ("NFL", "total", True):   ("lost", 31, "NFL total gaps 3+ at 2026 openers hit 38.7% (31)", 3),
    ("NFL", "total", False):  ("lost", 0, "NFL total gaps at 2026 closes hit 43-48%"),
    ("CFB", "spread", True):  ("won", 1842, "CFB look-ahead spread gaps 2+ covered 54.6% (1,842, 2022-25)", 2),
    ("CFB", "spread", False): ("won", 2062, "CFB game-week spread gaps 2+ covered 53.4% (2,062, 2022-25)", 2),
    ("CFB", "total", True, "Over"): ("lost", 489, "CFB look-ahead overs 4+ hit 49.5% (489, 2022-25)", 4),
    ("CFB", "total", True, "Under"): (None, 0, "CFB look-ahead unders under 4: untested size (4+ is tier B)"),
    ("CFB", "total", False):  ("flat", 0, "CFB game-week totals have not held up (2022-25 closes 51%)"),
}
ML_NOTE = ("Moneyline chance converted from the spread with a normal curve, never tested in any sport; that conversion makes big "
           "underdogs look +EV on its own, so ML ratings are capped at 60 and never a small play")


def _ncdf(x): return 0.5 * math.erfc(-x / math.sqrt(2))


def _inv(p):
    p = min(max(p, 1e-6), 1 - 1e-6); lo, hi = -8.0, 8.0
    for _ in range(60):
        m = (lo + hi) / 2
        if _ncdf(m) < p: lo = m
        else: hi = m
    return (lo + hi) / 2


def _dec(price): return 1 + (price / 100 if price > 0 else 100 / -price)


def _score(evU, evB, verdict, ml=False):
    s = 50 + 40 * math.tanh(evU / 0.06)   # smooth: evU 4% -> 73, 7% -> 83, 15% -> 90; keeps big edges in order
    s += 5 if evB >= 0 else -10 if evB < -0.03 else 0
    s += {"won": 5, "flat": -5, "lost": -15}.get(verdict, -5)
    s = int(round(max(1, min(60 if ml else 95, s))))   # moneylines capped lower (see ML_NOTE)
    lost = verdict == "lost"
    rec = "play" if s >= 65 and evU >= 0.04 and evB >= 0 and not lost and not ml else "watch" if s >= 50 and evU >= 0.02 else "skip"
    return s, rec


def rate(sport, kind, z):
    """Unproven rating for one game-line pick dict z (spread / total / ml) from a *_ab.json or runs.jsonl row. None if not rateable."""
    b = z.get("bet") if z else None
    if not b or b.get("p") is None or not b.get("price") or z.get("tier") != "C" or sport not in SD: return None
    sd = SD[sport][1 if kind == "total" else 0]
    pB = b["p"]; x = (abs(z.get("gap") or 0) - abs(z.get("real") or 0)) / sd
    pA, pU = _ncdf(_inv(pB) + x), _ncdf(_inv(pB) + x / 2)
    d = _dec(b["price"]); evA, evU = pA * d - 1, pU * d - 1; evB = b.get("ev", pB * d - 1)
    if kind == "ml": hw, n, txt = None, 0, ML_NOTE
    else:
        e = bool(z.get("fresh")); h = HIST.get((sport, kind, e, z.get("side"))) or HIST.get((sport, kind, e)) or (None, 0, "No test of this kind of bet")
        hw, n, txt = h[:3]
        if len(h) > 3 and abs(z.get("gap") or 0) < h[3]: hw, n, txt = None, 0, f"Gap under {h[3]}: smaller than the tested sizes ({txt})"
    s, rec = _score(evU, evB, hw, kind == "ml")
    return {"score": s, "rec": rec, "pA": round(pA, 4), "evA": round(evA, 4), "evU": round(evU, 4), "evB": round(evB, 4), "n": n, "tested": hw, "why": txt}


def rate_tennis(m):
    """Unproven rating for a tennis Model B pass (tennis_ab.json matches[]): Model A = our point sim's win %, B = market blend."""
    B, A = m.get("B") or {}, m.get("A") or {}
    if B.get("tier") != "C" or not B.get("price") or A.get("win_p1") is None or B.get("edge") is None: return None
    pA = A["win_p1"] if B.get("side") == 1 else 1 - A["win_p1"]; pB = (1 + B["edge"]) / B["price"]; pU = (pA + pB) / 2
    evA, evU, evB = pA * B["price"] - 1, pU * B["price"] - 1, B["edge"]
    women = m.get("sex") == "w"
    if women and evB >= 0.10: hw, n, txt = "lost", 141, "Women's blended edges of 10%+ lost 10% (141 bets, 2022-25)"
    elif women: hw, n, txt = None, 0, "Women's match outside the lean rules (edge under 6% or a player with under 30 rated matches)"
    else: hw, n, txt = None, 0, "Men's model weight is small (4-14%): men's ML was net negative at the bet365 opener (2022-25)"
    s, rec = _score(evU, evB, hw)
    if not women and rec == "play": rec = "watch"
    return {"score": s, "rec": rec, "pA": round(pA, 4), "evA": round(evA, 4), "evU": round(evU, 4), "evB": round(evB, 4), "n": n, "tested": hw, "why": txt}


def annotate(sport, out):
    """Add z["u"] (the unproven rating) to every rateable pass in a Model B output dict, in place. Returns how many."""
    n = 0
    if sport == "TENNIS":
        for m in out.get("matches", []):
            u = rate_tennis(m)
            if u: m["B"]["u"] = u; n += 1
        return n
    for g in (out.get("games") or {}).values():
        for k, z in ((g.get("B") or {}) if isinstance(g.get("B"), dict) else {}).items():
            if not isinstance(z, dict): continue
            u = rate(sport, k, z)
            if u: z["u"] = u; n += 1
    return n


def grade_tier(sport, kind, z):
    """Tier label for grade.py: 'A'/'B' as before, 'U_play' / 'U_watch' / 'U_skip' for rated passes, None otherwise."""
    if z.get("tier") in ("A", "B"): return z["tier"]
    u = z.get("u") or rate(sport, kind, z)
    return f"U_{u['rec']}" if u else None
