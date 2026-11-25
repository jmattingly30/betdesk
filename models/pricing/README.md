# One price check (desk simplification, step 1)

J 2026-11-24: "Model B" was never a second model. It devigs the books and handicaps Model A. The desk now has
**one model** (our projection, never blended) and **one price check** that works the same way for every sport and market.

| File | What it is |
|---|---|
| `trust.json` | How much of our gap from the market the price check keeps, per sport × market × timing (open / close). Also holds caps, label thresholds, stake rules and the auto-demote rule. Seeds come from each sport's earlier research (`src` on every row). Step 2 refits it weekly from live grades. |
| `grade.py` | Daily (morning C8a). Grades every priced row, Pass included, at its first line and price: W/L/P, units, and closing line value against the last pregame snapshot. Buckets: Bets per sport × market (these feed auto-demote), each label, each guard by the label it would have had, big dogs, open vs close. Writes `history/grade.json`, `history/graded.json.gz` and `app/data/price_grades.json` (for the monitor). |
| `refit.py` | Weekly (Mondays, before pricing). For each sport × market × timing and each sport's props, finds the trust that best explains graded results, pulled toward the research seed (prior sd 0.15). It needs 30+ graded rows before it changes anything. It never lifts a cap: buckets that reach their cap's bar are listed for J. `history/refit.json` logs every change, including `close_says`, the trust implied by where the market closed. |
| `price.py` | Prices every upcoming game line (NFL, NBA, CBB, CFB, NHL, MLB: spread / puck line / run line, total, moneyline) and every pro prop (NFL, NBA, NHL). Writes `app/data/priced_bets.json` and keeps the last pregame row of each Bet/Lean in `history/priced.json.gz`, along with the first price and label seen. |

## The steps, for every bet

1. **fair**: the books' no-vig chance of this exact bet. It is a sharp-weighted consensus (`devig.WEIGHT`, capped at 1.5×) of every book posting the same line. With fewer than two books on the line, the books' average number is moved onto the line along our curve. Moneylines use `ml_anchor.market_home`.
2. **ours**: our projection's chance of the same bet, with pushes removed.
   - NFL and CFB use the *_ab key-number margin grids.
   - NBA and CBB use the *_ab normal fits.
   - NHL uses the sim's goal distributions and puck-line chances.
   - MLB uses J's efficiency model: its totals table, plus negative-binomial runs for the run line.
   - Props use the desk's ladder curve (`interp` / `modelProbs`, ported unchanged).
3. **trusted** = `logit(fair) + trust × (logit(ours) − logit(fair))`.
4. **EV** is taken at the best Colorado price on the trusted chance, then labelled:
   - **Bet**: trusted EV 5%+ *and* EV on our own chance 5%+.
   - **Lean**: trusted EV 2%+. Tracked, never staked.
   - **Pass**: everything else.
5. Then, in order:
   - **Far check**: our chance is 15+ points from the books' chance, or a prop projection is 25%+ off a line of 3 or more. The result is Pass.
   - **Long prices**: anything longer than +300 is Pass.
   - **Caps** from `trust.json`.
   - **One-sided props**: ladders and anytime scorer prices are held to Lean, because their fair price is an estimate.
   - **Auto-demote**.
6. **Guarded rows are graded, not dropped** (J 2026-11-25). A row held to Pass by the far check, the +300 limit or an
   off-median line keeps `would_be` (the label it would have had) and `guards`. It goes to `guarded` in
   `priced_bets.json` and into the history, so step 2 can grade each guard on its own. Rows also carry a `big_dog` flag:
   a spread of NFL 7+, CFB 14+, NBA 9+ or CBB 15+ points, or a moneyline of +250 or longer. If a guard keeps winning
   against the close, it gets lifted.
7. **Stake**: quarter Kelly on the trusted chance, 100-unit bankroll, 3u max.

**Timing.** A line is *open* if it was first seen 24 h ago or less and the game starts 8+ h from now. NFL lines 96+ h before kickoff (look-ahead) also count as open. Everything else is *close*, including a line whose post time is unknown.

**Auto-demote.** Once a sport × market has 50+ graded Bets under the new labels, that record replaces the old tiers' (price CLV bar +1% for games and props). Until then, the stopgap below applies.

**Stopgap.** A sport × market is held to Lean when the old staked bets meet all three conditions:
- 50 or more live bets,
- ROI of −5% or worse,
- CLV not clearly positive: under +0.5 pts for games, under +1% for props.

The old bets are the *_ab tiers A and B, and the prop flags marked flag or proven.

## Shadow

Nothing on the desk reads `priced_bets.json` yet. The old tiers and flags are untouched. Each row carries `old` (the old tier or flag, mapped to Bet/Lean/Pass) so the two can be compared, and `compare` totals the moves for each sport × market. Step 4 (the new front end) is the first reader.

```
python3 models/pricing/price.py                          # now
python3 models/pricing/price.py --now 2026-11-24T22:31Z  # replay a snapshot
```

## Names (step 5, J 2026-11-25)

| Old name | Now |
|---|---|
| Model A | Our projection |
| Model B | The price check |
| Tier A, prop flag, proven | Bet |
| Tier B, lean | Lean |
| Tier C, unproven, check, long shot | Pass |
| Guest "Strong pick" | Bet |

Old records keep their old fields; the Monitor and the guest page show them under the new names ("was tier A").
The guest pick sheet (`worker/picks.js`) reads `priced_bets.json`, so guests and J see the same labels. Guests see
Bets at the Simple level and Leans from Guided. Prop rows carry `gid`, the priced game they belong to.
