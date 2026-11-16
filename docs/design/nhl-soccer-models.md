# NHL and soccer model spec, Phase 4 (2026-11-16)

A design for approval. Nothing here is built. Each layer is listed with the evidence for it, how it would be tested, and what result would make us drop it.

Background: `docs/research/nhl-soccer.md` (Phase 3c) and the October sim research (project folders `research/nhl-sim-2026-10/`, `research/soccer-sim-2026-10/`). New tests for this spec: project folder `research/nhl-soccer-2026-11/` (`nhl_p4.py`, `soc_p4.py`).

## The starting point, stated plainly

- NHL: the shot-state sim already ties the closing team totals on average (1.385 goals miss each). No team-level stat tested in two rounds improves it.
- Soccer: ratings from goals and shots trail the closing odds by 0.02 to 0.03 goals per team. The market's goal expectation shows no average miss in any situation checked for this spec (see the soccer section).
- So neither model gets better by adding another team rating. What is left is conditional accuracy: the specific games where a team-average model is known to be wrong, and the player outputs that a team model does not produce at all.

## New tests run for this spec

| Question | Result | Decision |
|---|---|---|
| NHL: is the sim wrong when regular skaters are out? | Yes. With two or more of a team's six highest-ice-time skaters out (840 team-games, 9%) the sim is 0.19 goals too high for that team. Each missing top-six skater is worth -0.07 goals (t -2.8). The opponent gains about as much per share of ice time missing (t 2.0). | Build a line-up layer |
| NHL: how much scoring happens with a net empty? | 0.28 goals per team per game, 9% of all scoring: 0.19 into the empty net and 0.09 by the side with the extra skater. | Model it as its own module |
| NHL: is the timing of pulling the goalie a team habit? | Mildly. Down one, the average pull comes with 1:43 left; teams range from 1:22 to 2:02. Split-half 0.35, year to year 0.37. The sim uses a fixed 2:00. | League curve plus a small team offset |
| NHL: do some teams sit on a lead more than others? | No. Every team's shot share drops about 4.5 points when leading, and the differences between teams do not repeat (split-half -0.18). | Do not build; keep league-wide score effects |
| NHL: back-to-backs (Phase 3c) | -0.21 goals for the tired team, four seasons of four. | Build |
| Soccer: does the closing market miss on short rest, after breaks, late in the season, for promoted sides, or for heavy favourites? | No. 39,520 team scores, 17 situations, average miss 0.003 goals overall and nothing beyond chance in any group. | Do not build situational adjustments for match goals |

## NHL model

### What exists (layer 0)

A simulation in 5-second steps. Shot and goal rates depend on score, skater counts and an empty net, scaled by each team's recent shot quality for and against and by the starting goalie. It produces a full distribution for each team's goals.

### Layers to add, in build order

**1. Who is dressed.** Replace the team's rolling average with the sum of the players expected to play.
- Each skater gets an on-ice rating for shot quality created and allowed per minute, solved jointly across all players with heavy shrinkage (the interval table with all ten skaters on the ice already exists for 2021-22 onward).
- A team's rate for a game is the ice-time-weighted sum over the skaters dressed. Injuries, trades, call-ups and returns then move the projection on the day, not three weeks later.
- The same run produces each player's shots, points and ice time, so player props and the team total come from one model and agree with each other.
- Test: bias on games with two or more regulars out falls from 0.19 to under 0.05; overall miss no worse than 1.381; player shots no worse than the current NHL player model against prop closes.
- Drop it if: the bias on those games does not fall, or player shots are worse than the current model.

**2. Rest.** Lower the scoring rate for a team on the second night of a back-to-back (measured: -0.21). Test alone, then check three games in four nights and long road trips with the same method. Drop any that do not hold in every season.

**3. Empty net.** A separate module for the last minutes of a one or two goal game.
- League timing curve by deficit (1:43 down one, 2:48 down two on average), with each team shrunk toward it.
- Separate scoring rates for the leader into the empty net and for the side with the extra skater.
- This matters most for the high tail of team totals (4.5 lines) and for a favourite's team total, where a ninth of the goals come from this phase.
- Test: over rates at 3.5 and 4.5 team-total lines calibrated to within 1 point; total distribution no worse.

**4. Officials (untested, test before building).** Referee pairs differ in penalties called, and penalties drive power-play time. Assignments are public on game day. Test first on one season: does the pair's past penalty rate predict power-play chances after team tendencies? If it explains less than a tenth of a penalty per team, drop it.

**5. Goalie uncertainty.** When the starter is not confirmed, run both goalies weighted by the chance each starts, in place of assuming the usual starter.

### Not building

Zone time and possession time, possession chains (the public feed sees about a third of true possessions), team-specific score effects, shot-type mix, splitting 5-on-5 from special teams in the team rating.

### Outputs and grading

Team totals with full distribution, game total, and player shots, points and saves. Graded on J's yardstick: team totals and player stats against their closing lines, with rows logged before the game starts only.

## Soccer model

### What the evidence allows

- Match goals: the closing market is unbiased everywhere we looked, and the remaining gap is who starts. A team model will tie the close at best.
- Possession is the most stable thing a team does (0.89) and is worth 0.001 goals as a strength input. It is useful for one thing: setting how many possessions and shots each side gets.
- So the model is worth building for line-up-aware projections ahead of the market's information (before team news) and for shot and player outputs, not for beating closing match odds.

### What exists (layer 0)

A minute-by-minute engine with game state: a side leading by one takes 20% fewer shots of 21% better quality; a side down a man loses about 45% of its shots.

### Layers to add, in build order

**1. Players.** Each player's shots, shot quality and chance creation per 90 minutes, plus a shrunk on-off rating for defence. Team rates are the sum over the eleven expected starters and likely substitutes. October's first version of this closed about an eighth of the gap to the close.

**2. Predicted line-ups.** Before team news, a probability that each squad player starts, from recent starts, days since the last match, and the next fixture. The projection is the average over likely line-ups, and is re-run when the real line-up is announced an hour before kick-off. This is the only layer aimed at where the gap is.

**3. Possession engine for shot outputs.** Possessions per side from possession share and tempo (both very stable), then for each possession: where it starts, whether it produces a shot, and the shot's quality. Possessions with several shots are combined as the chance that at least one scores, not as a sum. Used for team shots, shots on target and player shots, not for the match result.

**4. State.** Score, red cards and substitution timing change the rates, as in layer 0.

### Outputs and grading

Team goal distributions, team and player shots and shots on target, scorer chances. Graded against closing team totals and player shot lines. We have no history of soccer player lines, so the first step is to start logging them.

### Open risk

The shot-location source used in October (Understat) has been down since Nov 12. Layer 1 needs it or a replacement. ESPN carries line-ups and player shots per match and is the fallback, to be confirmed before any build.

## Recommended order and cost

| Step | Work | Cost |
|---|---|---|
| 1 | NHL rest (layer 2) | S |
| 2 | NHL line-up layer with player outputs (layer 1) | L |
| 3 | NHL empty net (layer 3) | M |
| 4 | NHL officials test | S |
| 5 | Soccer: confirm a data source, start logging player lines | S |
| 6 | Soccer players and predicted line-ups (layers 1 and 2) | L |
| 7 | Soccer possession engine for shots (layer 3) | L |

Each step is one branch and one pull request, tested alone before the next.

## Decisions needed

1. Approve NHL steps 1 to 4.
2. Soccer: build steps 5 to 7 after the NHL, or leave soccer on hold. The honest expectation is a model that matches the close on match goals and may add value on player shots, which is untested.
