# Mathax math in the Bet Desk (2026-10-19)

Source: J's Google Sheet "Copy of Mathax!" (Odds and Futures tabs). Formulas were rebuilt from the sheet's own example values, and every one reproduces them exactly: -310 = 75.61%, 17% = +488, 46.5% at +135 = 9.28% edge, Kelly 53.14% at +118 x1 on $1,000 = $134.28, -1667/+3000 hold = -2.49%, futures hold 17.27%, parlay true odds +172 and edge 37.75%.

## Mathax formulas

| Tool | Formula |
|---|---|
| American to probability | implied = 1 / decimal (vig left in) |
| Probability to American | p >= 50%: -100p/(1-p); else +100(1-p)/p |
| Calculate your edge | edge = p x decimal - 1 (EV per unit staked) |
| Kelly betsizer | f = (b*p - q)/b, b = decimal - 1; bet = f x multiplier x bankroll |
| Parlay edge | p = legs' chances multiplied; true odds from p; edge = p x book decimal - 1 |
| Theoretical hold | 1 - 1/(sum of every outcome's implied chance), 2 to 40 outcomes |

## How the desk compares

The desk already used the same edge and Kelly formulas: its "EV per unit" is Mathax's edge, and its stake is Mathax's Kelly times the 0.25 multiplier. Two things made it look off:

1. The desk's "Edge" row was a different number: blended chance minus break-even, in percentage points. Mathax's edge is chance x decimal - 1. At +185 the same bet reads +5.1 on the desk and +14.6% in Mathax terms.
2. The desk feeds a blended chance into the formula (30% our model and 70% no-vig market on team bets; 50/50 on props). Mathax uses whatever chance you type in, so typing our model's number gives a far bigger edge.

Six of J's logged bets:

| Bet | Odds | Our model | Blended | Book implied | Desk "Edge" | Desk EV (Mathax on blended) | Mathax on our model | Stake at 0.25 Kelly, our model / blended |
|---|---|---|---|---|---|---|---|---|
| Penn State ML (CFB) | +185 | 59.4% | 40.2% | 35.1% | +5.1 | +14.6% | +69.3% | 3.00u (cap) / 1.97u |
| Southern Miss ML (CFB) | +185 | 47.0% | 36.3% | 35.1% | +1.2 | +3.5% | +33.9% | 3.00u (cap) / 0.47u |
| NY -4.5 (NBA) | -112 | 65.2% | 55.1% | 52.8% | +2.3 | +4.3% | +23.4% | 3.00u (cap) / 1.20u |
| GB ML (NFL) | +100 | 51.9% | 50.3% | 50.0% | +0.3 | +0.6% | +3.8% | 0.95u / 0.15u |
| Wembanyama o2.5 threes | +185 | 40.6% | 37.0% | 35.1% | +1.9 | +5.4% | +15.7% | 2.12u / 0.74u |
| Huntley 250+ pass yds | +650 | 18.9% | 15.8% | 13.3% | +2.5 | +18.5% | +41.7% | 1.61u / 0.71u |

Mathax has no no-vig step. The desk keeps its power-method no-vig price for the blend, and now also shows the raw book implied chance and the hold the way Mathax does.

## Changes (mathax.patch, tested on a served copy of the desk with no page errors)

- Every price card (props, manual pricer, game cards) shows "Edge, our model" (Mathax on our raw chance) and "Edge, blended" (Mathax on the blended chance; this one drives flags and stakes, unchanged), plus the book's implied chance. The old percentage-point "Edge" row is gone. Side cards also show full Kelly % and the stake.
- Book hold (Mathax theoretical hold) shows on any market with both prices typed or fed.
- Parlay builder gets a "Mathax parlay edge" block: legs multiplied, true odds, book odds, edge with legs independent next to our correlated edge, Kelly, and the book's theoretical hold on the parlay. Parlay hold extends Mathax's definition: 1 - (every leg's no-vig chance multiplied) x the parlay's decimal odds, so it needs both sides of every leg. Game legs and prop legs (when over and under odds were typed) now carry their no-vig chance and hold.
- Settings gets a "Calculators (Mathax)" card: moneyline / probability / decimal conversions, edge plus Kelly betsizer (multiplier and bankroll default to the desk's settings), and theoretical hold for up to 40 outcomes.

Apply with `python3 mathax_patch.py` from app/ (exact-match replacements, stops if any text has moved) or `patch -p1 < mathax.patch`.

Open decision for J: which chance should flag and stake bets. Blended (current) or our model alone (same as setting both model weights in Settings to 100%).
