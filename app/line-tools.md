# Line shopping tools: Bet Desk handoff (2026-10-27)

J asked (2026-10-27 05:37) for OddsJam / OddsShopper style tools in Bet Desk: middles and book shopping for +EV first, then
arbs, low holds, promo conversion and sharp money. The data side is built. This file is the spec for the Bet Desk display.

## Files (publish both at the artifact root)
- `data/line_tools.json`: written by `app/line_tools.py` (about 17 s; needs node for `models/promo/probe_games.js`).
- `data/landing.json`: how often games land on each margin and total, per sport (NFL, CFB, NBA, CBB). Static; rebuild
  with `python3 line_tools.py --landing`.
- `data/line_history.json`: every pull's book lines per game (for line moves). Not needed by the app.

Refresh order: `export_slates.py` -> `line_tools.py` (any time after the odds pull; reruns are free, no credits).

## Odds pull change (export_slates.py)
The pull now names 10 books instead of `regions=us`. Ten named books cost the same credits as one region (tested: 1 credit
per market), so Pinnacle comes free. Books: pinnacle, draftkings, fanduel, betmgm, williamhill_us (Caesars), espnbet,
fanatics, betrivers, hardrockbet, betonlineag. Dropped: betus, mybookieag, bovada, lowvig. J (2026-10-27): Colorado, any legal book, no paid feed. Final list: pinnacle and betonlineag (fair price only) plus the
Colorado books draftkings, fanduel, betmgm, williamhill_us, espnbet, fanatics, betrivers, ballybet. hardrockbet dropped.

## line_tools.json
`{made, lines_from, books: {key: name}, my_books: [Colorado book keys, the My books default], sports: {NFL|CFB|NBA|CBB: [game]}}`. Each game:
- `id, k, d, h, a` match game_slates.json (open the same game card).
- `fair`: `src` (Pinnacle, BetOnline/LowVig, or all books), `mu` expected home margin, `mt` expected total, `pml` fair home
  ML chance, `L`/`T` fair line (home spread, total, to the quarter point), `sharp_vs_books` (sharp margin minus the
  median of all books; + = sharp likes home more than the market).
- `model`: our model's `mu`, `mt`, `pml` from the game card in "Our model" mode.
- `bk`: every book's row `[homePt, homePr, awayPr, totPt, overPr, underPr, mlH, mlA]` (same as ln.bk).
- `ev`: every book side with EV > 0 against the fair price. `best`: the best price per market, side and point. Each row:
  `b` book, `mk` spread|total|ml, `s` H|A|O|U, `pt`, `pr` American, `ev` vs fair, `w` win chance, `p` push chance,
  `fair` fair American price at that point, `mev` EV with our model alone, `bev` EV blended (logit, 30% our model, same as
  the Settings team weight default; recompute in the app with the user's weight from `mw` and `w`/`p`).
- `hold`: shopped hold per market and point (best price each side, any books): `h` hold, `a`/`b` = [book, price].
- `mid`: pairs of opposite bets at two books, sorted by EV. `kind`: `middle` (a result where both win), `push` (half-point
  window: one side pushes, the other wins), `same` (same number, low hold), `arb` (no result loses money). `a`/`b` =
  [book, side, point, price], `both` chance both win, `ev` per dollar staked in total, `worst`/`best` per dollar,
  `stake` split of 100 (both sides return the same). `check: 1` = verify before betting: a leg's point is far from the
  other books (NFL 2+, CFB 2.5+, NBA/CBB 3+ pts from the median, usually a stale or mislabelled price), or a moneyline
  arb. Show these with a "verify at the book" mark, sorted after clean rows.
- `conv`: bonus-bet hedges, moneyline: `bonus` [book, side, price] bet with the bonus, `cash` [book, side, price] hedge,
  `conv` guaranteed share of the bonus kept, `hedge` cash stake per $1 of bonus.
- `move`: since first pull: `since`, `dmu`/`dmt` change in the sharp expected margin/total, `flag` (NFL/NBA/CBB 1+ pt,
  CFB 1.5+; totals x1.5).

## Display asked for (keep it simple, OddsJam-style)
1. **Slate > Line shop** subview per sport (NFL, CFB, NBA, CBB). Five short lists, each row opens the game card:
   - **+EV**: best price per side where EV > 0 vs fair, sorted by EV; columns: game, bet, book, price, fair price, EV,
     our-model EV and blended EV. Toggle "vs fair / vs blended model".
   - **Middles**: `mid` kinds middle and push, sorted by EV; show both legs, middle chance, EV, worst case, stakes.
   - **Arbs**: kind arb; note "prices are from the morning pull, check both books before betting".
   - **Low hold**: `hold` under 2.5%.
   - **Bonus hedge**: `conv` sorted, filtered to J's promo book in the bonus leg.
   - **Line moves** (sharp money for now): games with `move.flag`, and `sharp_vs_books` of 0.5+ pt.
2. **Game card**: an "Every book" table (spread, total, ML per book; best price bold; fair line from Pinnacle on top).
3. **Settings > My books**: tick the books J can bet (default: `my_books`, J's Colorado books; Pinnacle and BetOnline are
   fair-price references only). Every list above shows only legs at ticked books.
4. **Bets > Middle watch**: for each pending spread or total bet, find the best opposite price in line_tools.json at
   J's books and price the pair with landing.json (same math as `middle()` in line_tools.py). Show it when a window
   opens (`both` > 0) or the pair is +EV.
5. **Settings > Calculators**: middle calculator (sport, two lines, two prices -> middle chance, EV, stakes, using
   landing.json), arb calculator, bonus hedge calculator (bonus price, hedge price -> conversion, hedge stake).

## landing.json (JS)
For sport `sp`, kind `m` (home margin) or `t` (total), expected value `mu`:
`v = max(sd[0] + sd[1]*|mu|, 25)`; for each integer k from mu-6*sqrt(v) to mu+6*sqrt(v):
`P(k) = (Phi((k+.5-mu)/sqrt(v)) - Phi((k-.5-mu)/sqrt(v))) * mult[i]`, i = k*(mu<0 ? -1 : 1) - lo for margins (seen from
the favourite's side), k - lo for totals (mult 1 outside the array); then normalise. NFL favourites win by exactly 3
2.8x and by 7 1.8x as often as the normal curve says. Checked on 2015+ NFL closing lines: at lines of 3 the curve gives
8.9% for the favourite by 3 against 10.6% seen (444 games), so middles and pushes on 3 are priced a little low (cautious).

## Prop odds (added 2026-10-27, J: odds for every prop daily matter most)
`app/op_live.py` (OddsPapi, free) writes `data/odds/props_{NFL,NBA,NHL,CFB,CBB}.json`; publish them at the artifact root.
Shape: `{made, src, games: [{id (OddsPapi fixture), k, h, a (full team names), players: {"First Last": {stat: {line: {book: {Over|Under|Yes|No|"250+": American}}}}}}]}`.
Stats as OddsPapi names them without "Player"/"(incl. overtime)": NFL Pass Yards, Pass Attempts, Pass Completions, TD
Passes, Interceptions, Rush Yards, Rush Attempts, Receiving Yards, Receptions, Longest Rush/Pass, To Score TD (+ First
TD / First Half / Second Half / First Quarter), kicking. `... ladder` = alt ladders keyed "250+". Book keys match
line_tools (williamhill_us = Caesars). First NFL sweep: 14 games, 397 players, 9 books (bet365, DK, Bally, BetRivers,
FD, Caesars, Fanatics, BetMGM, Pinnacle).

Display asked for:
1. On every prop row and prop card: the book lines and prices at J's books (`my_books`), best Over and best Under bold,
   and EV of each against our model's P(over) curve at that book's line (the card already prices any line). Default the
   card's line/price to the best-EV book instead of making J type it.
2. Props +EV list (Slate > Line shop, or the Flags view): every prop side at J's books with EV >= the Settings
   threshold, sorted by EV, with book.
3. Prop middles: Over at a lower line at one book + Under at a higher line at another (e.g. Love o259.5 DK, u263.5 BetMGM).
   Middle chance = P(over low) - P(over high) from our model curve; EV and stakes as in `middle()` (equal returns).
4. Name matching: OddsPapi names are "First Last" with suffixes (Jr., III); match on normalized name + team.
