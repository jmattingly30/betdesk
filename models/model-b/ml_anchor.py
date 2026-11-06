"""Moneyline win chance for Model B, anchored on the books' own moneyline (J 2026-11-06, code audit fix).

Before: every sport turned Model B's spread into a win chance with a fixed margin curve and compared that with the
book price. The curve itself disagrees with the books' moneylines (11-05 data, games with no trusted spread gap:
NFL/CFB underdogs +3.8/+4.3 pts over the books' no-vig ML, CBB the other way), so big underdogs showed +20-30% EV
with no edge at all (BAL +425: curve 24.4%, books 19.2%).

Now: our win chance = the books' devigged moneyline + (curve at our margin - curve at the market margin).
The curve only prices the CHANGE our trusted spread edge makes, its own bias cancels, and a game with no trusted gap
shows the market's chance (no moneyline edge). Market ML = sharp-weighted (devig.WEIGHT, capped) consensus of each
book's two-way devigged price; books 6+ pts of win chance off the median are dropped (stale or a different market).
Devig method by sport from models/model-b research (us_ml_test.py): NBA ML multiplicative, everything else power.
"""
import statistics
import devig as DV

METHOD = {'NBA': 'mult'}   # default power (CBB ML power >> mult; NFL/CFB untested -> power, J's default)

def _dec(a): return 1 + (a / 100 if a > 0 else 100 / -a)

def market_home(bk, sport):
    """sharp-weighted devigged home win chance from book rows (r[6] home ML, r[7] away ML); None if no book has both."""
    f = DV.mult if METHOD.get(sport) == 'mult' else DV.power
    rows = []
    for b, r in bk.items():
        if not r or len(r) < 8 or not r[6] or not r[7]: continue
        try: p = f([_dec(r[6]), _dec(r[7])])[0]
        except Exception: continue
        if 0 < p < 1: rows.append((b, p))
    if not rows: return None
    med = statistics.median(p for _, p in rows)
    acc = tw = 0.0
    for b, p in rows:
        if abs(p - med) > 0.06: continue
        w = min(DV.WEIGHT.get(b, 0.8), DV.MAX_RATIO); acc += w * p; tw += w
    return acc / tw if tw else None

def home_chance(bk, sport, curve, mu_ours, mu_market):
    """(our home win chance, market home win chance or None). curve(mu) = the sport's P(home wins | mean margin mu)."""
    pm = market_home(bk, sport)
    if pm is None: return curve(mu_ours), None
    return min(max(pm + curve(mu_ours) - curve(mu_market), 0.01), 0.99), pm
