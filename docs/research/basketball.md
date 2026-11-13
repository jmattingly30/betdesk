# Basketball research, Phase 3b: CBB and NBA (2026-11-13)

Nothing here has been built into a live model. Each idea needs approval first.

Targets came from the model audit (`docs/audit/models.md`): CBB margins and November totals, NBA player projections running low, and the directive to study time of possession and time spent attacking.

Code and results: project folder `research/basketball-2026-11/` (`cbb_cal.py`, `cbb_tail.py`, `cbb_poss.py`, `nba_poss.py`, `nba_minutes.py`).

## What was tested

- CBB history: 21,948 lined games 2020 to 2026 with Model A (v2) and the closing line; 40,966 team scores 2023 to 2026 with possession detail from play-by-play.
- CBB live: 324 lined games from this season, using only numbers logged before game day.
- NBA: 7,376 team-games of possession logs (2022-23 to 2024-25), joined to our team-score model and the close (6,926 team scores); 3,192 player projections from this season with box scores.
- Tests of added value hold out one season at a time. "MAE" is the average miss in points. A difference is treated as real only when it is large against its noise and has the same sign in every season.

## Results

### 1. CBB: this season's ratings are too compressed, and it is specific to this season

Live, pre-game, by size of the closing line:

| Line | Games | Our margin vs line (toward favourite) | Result vs line | Margin MAE ours | Line |
|---|---|---|---|---|---|
| Under 7 | 107 | -0.3 | -1.7 | 8.75 | 8.68 |
| 7 to 14 | 52 | -0.6 | -1.1 | 10.09 | 9.26 |
| 14 to 21 | 56 | -1.8 | +0.4 | 10.85 | 9.44 |
| 21 and up | 109 | -4.4 | +0.9 | 12.78 | 10.85 |

- Close games are fine. The whole gap to the line is in mismatches: we have the favourite 4.4 points short of the line and the favourite then beats the line.
- Our spread of projected margins is 85% as wide as the market's this season. In back-tests of the first ten days (2019 to 2023) it was 96% to 99%, and the shortfall at lines of 21 and up was 1.4 to 2.5 points.
- The weight on our disagreement with the line that best fits this season's results is negative (-0.28). Where we differ from the market on a CBB margin right now, the market has been right.
- A third of this season's lined games (109 of 324) have lines of 21 or more. In the back-test only 3% to 12% of early-season lined games did, because the old line data skipped those games. The model was never tested where it is used most in November.
- A general mismatch correction learned from history does not fix it: +0.002 overall on 37,632 games, and only 0.16 of the 1.9 gap on this season's 21-and-up games. The model's history is calibrated (see 2), so this is a problem with how this season's ratings were built: the preseason roster prior, its clip on transfer minutes, or the switch to in-season ratings on Nov 6 with only a game or two per team.

### 2. CBB: the model is calibrated in history; a seasonal shrink does not help

- Slope of actual on our margin is 0.97 to 1.01 at every stage of the season, and totals 0.92 to 1.07, both the same as the line's.
- A walk-forward shrink by days into the season changes team-score MAE from 7.988 to 7.989.
- November totals: the first-day level problem (4.8 low on Nov 2) was gone after the Nov 6 change (within 1.3 since, apart from single days). Opening-week pace runs 2.5 to 3 possessions above the rest of the season every year (70.2 to 71.3 vs 67.5 to 68.4; 72.2 so far this season), which the model should expect.

### 3. NBA player projections: the shortfall is minutes, not scoring rate

| | Projected | Actual (players who played) |
|---|---|---|
| Points | 10.15 | 10.77 |
| Minutes | 20.8 | 22.8 |
| Points per minute | 0.487 | 0.473 |
| Team total of projected players' minutes | 213 | 233 |

- Per-minute rates are slightly high, not low. Minutes are 9% low for the players who take the floor, because the model spreads a team's minutes over more players than end up playing. The gap is largest on the bench (expected 9.4 minutes, played 13.4) and gone for starters above 33 minutes.
- A prop on a player who does not play is void, so the right projection is "if he plays".
- Test: learn expected-to-actual minutes on games to Oct 31, apply to Nov 1 to 11 (1,387 player-games). Bias goes from -0.64 to +0.10 on points, -0.35 to -0.01 on rebounds, -1.20 to +0.03 on points+rebounds+assists. Minutes MAE improves from 5.08 to 4.81.
- Point MAE does not improve (4.57 to 4.66). Scoring is skewed, so a projection that is right on average sits above the typical game. The gain is in the over/under chances, which have been 6 to 7 points too low on the over, not in the headline miss.

**Correction to the Phase 2 audit, finding 5.** The "under flags hit 39.9%" figure is measured against our own fair line, not a sportsbook line. It shows the model is low; it is not a betting record. The bias and calibration numbers in that finding stand.

### 4. Time of possession and time spent attacking: stable traits, no predictive value

Definitions. Transition = a possession after a defensive rebound or steal that ends within 10 seconds (8 in the NBA). Attacking possession = one that reaches the rim or the foul line. Attack time share = share of a team's possession seconds spent on those possessions.

| Measure | CBB stability (odd vs even games) | NBA stability | Added to our CBB team score (MAE 7.98) | Added to our NBA team score (9.12) |
|---|---|---|---|---|
| Seconds per possession | 0.88 | 0.92 | -0.001 | -0.013 |
| Time to first shot | 0.89 | n/a | -0.002 | n/a |
| Transition share | 0.84 | 0.62 | -0.001 | +0.013 (worse) |
| Early-offense share | 0.87 | 0.82 | -0.002 | 0.000 |
| Attacking possession share | 0.79 | n/a | 0.000 | n/a |
| Attack time share | 0.75 | n/a | 0.000 | n/a |
| Total possession time | 0.76 | 0.86 | -0.003 | -0.002 |
| Pace (possessions) | 0.78 | 0.82 | -0.004 | -0.004 |
| All of the above together | | | -0.006 | +0.004 |

- These are among the most repeatable things a team does. They still add nothing measurable to score prediction on top of the current models, and nothing on top of the close (largest change 0.004 CBB, 0.006 NBA).
- In the same game, attacking share goes with efficiency (0.28) and half-court points per possession explains most of total efficiency (0.90). Transition efficiency itself is mostly noise (stability 0.30).
- One small lead: in the first 27 days of a CBB season, transition share improves our team score by 0.02 in all four seasons (t -2.1). Not enough to act on alone.
- NBA shot-location attacking time (paint or frontcourt time) could not be tested: the league's tracking pages are blocked from here.

### 5. NBA team margins

History says our NBA margins are slightly too narrow, not too wide (slope of actual on ours 1.04 to 1.13 in 2024 to 2026, margin MAE within 0.2 to 0.3 of the line). The 0.88 slope in the Phase 2 audit came from 170 early-season desk snapshots and should be treated as small-sample.

## Ranked ideas

| # | Idea | Evidence | Expected gain | Cost | Type |
|---|---|---|---|---|---|
| 1 | **Find and fix why this season's CBB ratings are compressed.** Check the roster-prior clip on transfer minutes, teams with missing roster data falling back to average, and the in-season switch with one or two games per team. Target: our margin spread back to 96% or more of the market's. | Whole live gap to the line is in games with lines of 14 and up (MAE 12.78 vs 10.85 at 21 and up); history shows no such gap. | Most of the 1.0 point live margin gap in November | M | Fix to existing model |
| 2 | **Until then, no CBB spread picks when the line is 14 or more.** | Best-fit weight on our disagreement is negative this season. | Removes the losing tier A picks (41 of 96 were on lines of 20 and up, winning 39%) | S | Model B rule |
| 3 | **NBA minutes "if he plays".** Rebuild the team minutes step so minutes go to the players expected to be active, and re-measure over/under calibration. | Bias -0.64 to +0.10 on points; minutes MAE 5.08 to 4.81. | Fixes the low over-chances; no gain in headline MAE | M | Fix to existing model |
| 4 | **Back-test CBB on every game, including big mismatches.** Rebuild the early-season test with this season's full line coverage as it accumulates. | Back-test covered 3% to 12% of the games that are 34% of the live slate. | Stops this class of surprise | S | Testing |
| 5 | CBB early-season transition share as a small input. | -0.02 in the first 27 days, four seasons of four. | Small | S | Adjusts existing model |

Held from Phase 2 for the same batch of CBB changes: the grader fix and the definition of a fresh price.

## Tested and not worth building

- Possession time, seconds per possession, time to first shot, transition share, early offense, attacking share and attack time share as predictors, CBB and NBA.
- A days-into-season shrink for CBB margins or totals.
- A general mismatch correction for CBB from history.
- Narrowing NBA margins.

## Limits

- CBB possession detail covers 2023 to 2026; closing lines in that span exist for 2023 and 2026 only.
- This season's CBB sample is ten days.
- NBA possession logs end with 2024-25 and have no shot locations.
- The NBA minutes test is three weeks of one season.
