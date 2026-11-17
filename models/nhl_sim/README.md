# NHL shot-state sim (team model)

Built 2026-11-16 from the design in `docs/design/nhl-soccer-models.md`. Research: `docs/research/nhl-soccer.md` and the October sim study.

What it does: plays each game in 5-second steps. Shot and goal rates depend on the score, the skaters on the ice and an empty net; team shot volume, shot quality, special teams, penalties and finishing come from recent games (no look-ahead); the starting goalie has his own rating; overtime is 3-on-3 and a shootout is a coin flip. Output is each team's full goal distribution.

Rest is the one change from the October sim: a team on the second night of a back-to-back scores at 0.92 of its usual rate.

## Files

- `prep_events.py` play-by-play -> shots, time by game state, final scores.
- `lib.py` league base rates, team-game table, ratings, the frame the sim reads. Settings are at the top.
- `sim.py` the simulator.
- `live.py <scratch>` daily: downloads this season, rates, simulates today's and tomorrow's games, writes `app/data/nhl_sim.json`, appends `history/log.jsonl`. Only games that have not started are logged.
- `grade.py <scratch>` grades the last row logged before each game started -> `history/grade.json`.
- `hist.py` rebuilds `data/hist.parquet` and `data/sums.json` (finished seasons). Run once after each season, then raise `FILE_YEAR` in `live.py`.
- `backtest.py` the rest test below.

## Test of the rest step (`backtest.json`)

Regular seasons 2022-23 to Nov 2026, 5,578 games, ratings walk-forward. The multiplier used for each season was fitted on the other seasons.

| Team scores | n | Miss, no rest | Miss, with rest | Bias, no rest | Bias, with rest |
|---|---|---|---|---|---|
| All | 11,156 | 1.3883 | 1.3832 | +0.07 | +0.04 |
| Second night | 1,675 | 1.427 | 1.393 | +0.27 | +0.03 |
| Rested | 9,481 | 1.3815 | 1.3815 | +0.04 | +0.04 |

Better in all five seasons on second-night teams (t -6.0 overall).

## Limits

- Starting goalie is the team's most used goalie of its last ten games, not the confirmed starter.
- The line beside each game is team goals implied by the moneyline and game total (Pinnacle when listed). Our odds pulls do not carry team-total markets.
- No injuries or line-ups. Those are research candidates in the design doc, not part of this build.

## On the Bet Desk (2026-11-17)

NHL > Games lists every game in `nhl_sim.json`: the sim's score and win chance (Model A, not blended), the market's implied team goals beside it, and in the game tile the 80% ranges, overtime chance, and our chance for the moneyline, puck line, total and team totals 2.5/3.5 (from `dt`, `pl`, `dh`, `da`). Started games stay listed for 30 h with the final (`fs`). Data > Model A vs B shows the grade (`grade` in `nhl_sim.json`): team goals and totals vs the market, by team, latest games. No bet engine (Model B) for NHL yet.
