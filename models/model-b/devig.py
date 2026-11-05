"""Devig (no-vig fair price) toolkit for Model B (J 2026-10-30: build around power devigging).

Every function takes DECIMAL odds for all outcomes of one market (2-way or n-way) and returns fair probabilities
that sum to 1. American odds: use dec_from_am() first.

Methods
  mult     multiplicative / normalise: p = q / sum(q). Spreads vig in proportion to price. Ignores favourite-longshot bias.
  add      additive: p = q - (sum(q)-1)/n. Takes the same number of points off every side (can go negative on longshots).
  power    p = q^k with k solved so sum = 1 (Clarke, Kovalchik & Ingram 2017; Vovk & Zhdanov 2009). Takes MORE off the
           longshot, which matches the favourite-longshot bias. Default method in Bet Desk and line_tools.
  shin     Shin (1992/93) insider-trading model; solve z (share of informed money). 2-way it equals add.
  odds_ratio  fair odds-ratio form p = q / (c + q - c*q) (Cheung; Buchdahl "Wisdom of the Crowd"). 2-way = a constant
           shift in log-odds.
  mpto     margin proportional to odds (Buchdahl): fair = n*o / (n - M*o), M = sum(q)-1. Strongest longshot cut.
  worst    for EV checks: the LOWEST fair chance any method gives each outcome (conservative, CrazyNinjaMike style).
  calibrated  power devig, then p = sigmoid(a + b*logit(p_power)) with (a, b) fitted on our own graded history for that
           book + market (see CAL below and devig_test.py). a < 0 on overs = the book shades overs.

consensus(books) builds the "market" price for one outcome from several books: devig each book, then average in
log-odds with weights (Pinnacle/Circa heavier, soft books lighter, stale prices lighter). This is the yardstick for CLV
("consensus close") in Model B.
"""
import math

def dec_from_am(a):
    a = float(a)
    return 1 + (a / 100 if a > 0 else 100 / -a)

def am_from_dec(d):
    return round((d - 1) * 100) if d >= 2 else round(-100 / (d - 1))

def _q(odds): return [1 / o for o in odds]
def logit(p): p = min(max(p, 1e-9), 1 - 1e-9); return math.log(p / (1 - p))
def sigmoid(x): return 1 / (1 + math.exp(-x))

def _bisect(f, lo, hi, n=80):
    """f increasing in x; find f(x)=0."""
    for _ in range(n):
        m = (lo + hi) / 2
        if f(m) > 0: hi = m
        else: lo = m
    return (lo + hi) / 2

def mult(odds):
    q = _q(odds); s = sum(q); return [x / s for x in q]

def add(odds):
    q = _q(odds); m = (sum(q) - 1) / len(q); return [x - m for x in q]

def power(odds):
    q = _q(odds)
    if abs(sum(q) - 1) < 1e-12: return q
    # sum q^k decreases in k; overround -> k > 1, underround (arb) -> k < 1
    k = _bisect(lambda k: 1 - sum(x ** k for x in q), 0.2, 5.0)
    p = [x ** k for x in q]; s = sum(p); return [x / s for x in p]

def power_k(odds):
    q = _q(odds); return _bisect(lambda k: 1 - sum(x ** k for x in q), 0.2, 5.0)

def shin(odds):
    q = _q(odds); S = sum(q)
    def probs(z): return [(math.sqrt(z * z + 4 * (1 - z) * x * x / S) - z) / (2 * (1 - z)) for x in q]
    if S <= 1: return mult(odds)
    z = _bisect(lambda z: 1 - sum(probs(z)), 0.0, 0.4)   # sum(probs) decreases in z
    p = probs(z); s = sum(p); return [x / s for x in p]

def odds_ratio(odds):
    q = _q(odds)
    f = lambda c: 1 - sum(x / (c + x - c * x) for x in q)   # increasing in c
    c = _bisect(f, 0.2, 5.0)
    p = [x / (c + x - c * x) for x in q]; s = sum(p); return [x / s for x in p]

def mpto(odds):
    q = _q(odds); n = len(odds); M = sum(q) - 1
    fair = [n * o / (n - M * o) if n - M * o > 0 else float('inf') for o in odds]
    p = [1 / f if f != float('inf') else 0.0 for f in fair]; s = sum(p); return [x / s for x in p]

METHODS = dict(mult=mult, add=add, power=power, shin=shin, odds_ratio=odds_ratio, mpto=mpto)

def worst(odds):
    """Lowest fair chance any method gives each outcome (does NOT sum to 1; use only to check a bet's EV)."""
    allp = [f(odds) for f in METHODS.values()]
    return [min(p[i] for p in allp) for i in range(len(odds))]

# Calibration on our own graded history (devig_test.py --fit writes cal.json): key 'book|market' -> (a, b)
CAL = {}
def calibrated(odds, key=None):
    p = power(odds)
    if key not in CAL or len(odds) != 2: return p
    a, b = CAL[key]; p1 = sigmoid(a + b * logit(p[0])); return [p1, 1 - p1]

# Book weights for the consensus line. J 2026-10-30: sharp-weighted consensus, but "don't weigh it too heavily" ->
# sharp books capped at 1.5x a Colorado book, soft books never below 0.8x. Tested on 2026 props: Pinnacle 1.5 vs 3.0
# costs at most .001 log loss (NBA close), nothing elsewhere.
WEIGHT = dict(pinnacle=1.5, circasports=1.4, betonlineag=1.25, bookmaker=1.25, lowvig=1.1, draftkings=1.0, fanduel=1.0,
              bet365=1.0, betmgm=0.9, williamhill_us=0.9, caesars=0.9, espnbet=0.9, fanatics=0.9, hardrockbet=0.9,
              betrivers=0.8, ballybet=0.8)
MAX_RATIO = 1.5   # no book may count more than 1.5x a DraftKings/FanDuel price

def consensus(books, method='power', weights=None, stale_secs=None, half_life=3600.0):
    """books: {book: [dec odds outcome1, outcome2, ...]} all on the SAME line. Returns fair probs (log-odds weighted mean).
    stale_secs: optional {book: seconds since that price last moved}; older prices are down-weighted."""
    w = weights or WEIGHT; f = METHODS[method]; acc = None; tw = 0.0
    for b, odds in books.items():
        if not odds or any(o is None or o <= 1 for o in odds): continue
        p = f(odds); wt = min(w.get(b, 0.8), MAX_RATIO)
        if stale_secs and stale_secs.get(b) is not None: wt *= 0.5 ** (stale_secs[b] / half_life)
        lp = [math.log(max(x, 1e-9)) for x in p]
        acc = [wt * x for x in lp] if acc is None else [a + wt * x for a, x in zip(acc, lp)]
        tw += wt
    if acc is None: return None
    e = [math.exp(x / tw) for x in acc]; s = sum(e); return [x / s for x in e]

def ev(price_dec, p_fair):
    """Expected profit per 1 unit staked at decimal price with fair chance p."""
    return p_fair * (price_dec - 1) - (1 - p_fair)

def kelly(price_dec, p_fair, frac=0.25):
    b = price_dec - 1; f = (b * p_fair - (1 - p_fair)) / b; return max(0.0, f * frac)

if __name__ == '__main__':
    for case in ([1.91, 1.91], [1.50, 2.70], [1.20, 4.80], [1.05, 11.0], [2.10, 3.40, 3.60]):
        print(case, {m: [round(x, 4) for x in f(case)] for m, f in METHODS.items()},
              'worst', [round(x, 4) for x in worst(case)])
