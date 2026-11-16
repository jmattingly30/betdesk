# NHL and soccer model spec, Phase 4 (2026-11-16)

A design for approval. Nothing here is built.

Background: `docs/research/nhl-soccer.md` (Phase 3c) and the October sim research (project folders `research/nhl-sim-2026-10/`, `research/soccer-sim-2026-10/`). Quick checks for this spec: project folder `research/nhl-soccer-2026-11/` (`nhl_p4.py`, `soc_p4.py`).

## The starting point, stated plainly

- NHL: the shot-state sim already ties the closing team totals on average (1.385 goals miss each). No team-level stat tested in two rounds improves it.
- Soccer: ratings from goals and shots trail the closing odds by 0.02 to 0.03 goals per team. The market's goal expectation shows no average miss in any situation checked for this spec (see the soccer section).
- Neither model gets better by adding another team rating.

## Rule for this spec

The design uses only pieces that have been researched in depth: tested on several seasons, held out, and compared with the closing line. Anything checked once or not at all is listed separately as a research candidate and is not part of the design.

## NHL model

### Design (researched)

| Piece | Evidence | Source |
|---|---|---|
| Simulation in 5-second steps, with shot and goal rates by score, skater counts and empty net | Ties the closing team totals (1.385 each) on 928 team scores; calibrated on 5,464 games | October sim research |
| Team shot quality for and against from recent games | Expected goals improve on goals alone by 0.008 in all six seasons | October, Phase 3c |
| Starting goalie rating | The most valuable single input in October's tests | October |
| Score effects applied league-wide, the same for every team | Shot share drops about 4.5 points when leading; differences between teams do not repeat (split-half -0.18) | October, checked again for this spec |
| Rest: lower scoring for a team on the second night of a back-to-back | -0.21 goals over the sim, t -3.6, four seasons of four, 6,964 team scores | Phase 3c |

Rest is the only change to the existing sim. It is one branch, tested alone.

### Left out on evidence

Zone time and possession time, possession chains (the public feed sees about a third of true possessions), team-specific score effects, shot-type mix, splitting 5-on-5 from special teams in the team rating, penalties drawn.

### Outputs and grading

Team totals with full distribution and the game total. Graded on J's yardstick: team totals against their closing lines, with rows logged before the game starts only. Player outputs stay with the existing NHL player model.

## Soccer model

### Design (researched)

| Piece | Evidence | Source |
|---|---|---|
| Minute-by-minute engine with game state | Leading by one: 20% fewer shots of 21% better quality. Down a man: about 45% fewer shots. Beats the flat engine (log loss 0.9847 vs 0.9863) on 12,304 matches | October |
| Team strength from expected goals, goals and shots | Best of the team ratings tested; expected goals improve on goals alone | October, Phase 3c |
| Starting eleven ratings | Improves on team ratings (0.9820 vs 0.9839) on seven seasons | October |
| Possession share and tempo set how many possessions each side gets, not how good it is | Possession stability 0.89, worth 0.001 goals as a strength input on 33,058 team scores | Phase 3c |
| Several shots in one possession combine as the chance that at least one scores | Summing overstates those possessions (0.27 expected vs 0.18 scored) | October |
| No situational adjustments to match goals | The market's goal expectation shows no average miss on rest, breaks, season phase, promoted sides or heavy favourites (39,520 team scores) | This spec |

### What to expect

The closing odds stay ahead of this design by 0.02 to 0.03 goals per team, and October's work traced that gap to line-ups and match-day news. The researched design will not close it.

### Open risk

The shot-location source used in October (Understat) has been down since Nov 12. The design needs it or a replacement, confirmed before any build.

## Research candidates (not in the design)

Each needs its own full study before it can be considered. The quick checks below are single passes, not research.

| Candidate | What exists so far | What a proper study would need |
|---|---|---|
| NHL: build team rates from the skaters who are dressed | One pass: with two or more of a team's six highest-ice-time skaters out (840 team-games) the sim is 0.19 goals high; -0.07 per missing skater (t -2.8), not consistent by season | Player ratings solved jointly, held-out seasons, test against the close and against the current player model |
| NHL: separate empty-net module | One pass: 0.28 goals per team per game come with a net empty (9% of scoring); pull timing is a mild team habit (0.35 split-half) | Effect on team-total tails at real lines, held out |
| NHL: three games in four nights, travel | Not tested | Same method as the back-to-back test |
| NHL: referee penalty rates | Not tested | One season of officials data first |
| NHL: unconfirmed starting goalie | Not tested | History of confirmation times |
| Soccer: predicted line-ups before team news | Not tested; it is where the gap to the market is | Line-up history, held-out seasons, test against opening and closing odds |
| Soccer: player shots and scorer outputs | October: scorer log loss 0.283 vs 0.317 flat; no test against real lines | A history of player lines, which we do not have |

## Decisions needed

1. NHL: approve adding rest to the sim as the one design change.
2. Pick which research candidates, if any, get a full study next.
3. Soccer: build the researched design now, knowing it trails the close, or hold it until the line-up study is done.
