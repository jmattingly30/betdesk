# Football research, Phase 3a: CFB and NFL (2026-11-13)

Nothing here has been built into a live model. Each idea needs approval first.

Two directives shaped this round: (1) time of possession and time spent attacking, (2) our own EPA-style stat that is schedule, situation and leverage weighted, in the spirit of baseball's xwOBA, college first.

Code and results: project folder `research/football-2026-11/` (README, scripts, `results.json`).

## What was tested

- College: every play 2014 to 2026 (1.85 million kept plays, 21,953 team-games). NFL: every play 2014 to 2025 (6,590 team-games).
- All ratings are walk-forward: a team's number going into a week uses only games before that week, plus half of last season's final number.
- Prediction tests fit on 2015 to 2021 and are scored on 2022 to 2025, week 4 on unless stated. College test: 5,104 team scores (rating-only tests), 7,026 lined team scores (tests on top of the desk model). NFL: 1,894 and 2,278.
- "MAE" is the average miss in points on one team's score. Lower is better. A difference is treated as real only when it is large against its noise (t of 4 or more) and has the same sign in all four test seasons. About 60 comparisons were run, so smaller effects are likely chance.

## The stat we built

Working name: **Adjusted Play Value (APV)**.

1. **Our own expected points.** A model fit by us on 2014 to 2021 plays gives the expected next score from down, distance, field position, clock, score and timeouts. A play's value is the change in expected points plus any points scored. It agrees closely with the public college EPA (correlation 0.92 on 2022 to 2025 plays it never saw).
2. **Luck taken out (the xwOBA step).** A fumble gets the average value of all fumbles in that part of the field whether or not it was lost. An interception gets the average interception value, so the return does not count. One play cannot be worth more than 4 points either way, which limits long-touchdown breaks. Field goals and punts get the average for that distance.
3. **Schedule and level.** Each team's offense and defense number is solved together with every opponent's, with separate baselines for power conferences, other FBS and FCS.
4. **Leverage.** Weights for decided games and for late, close situations were learned from what predicts best, not assumed.
5. **Index.** 100 is the FBS average for that season and 15 points is one standard deviation, so seasons with different scoring are comparable.

## Results

### 1. Does the stat do what it was designed for? Yes, as a comparison tool.

| Check | Public EPA | Our raw value | Luck out, capped at 4 | Capped at 2.5 |
|---|---|---|---|---|
| Stability, odd vs even games (college offense) | 0.545 | 0.539 | 0.553 | 0.572 |
| Rating vs next game's actual value | 0.435 | 0.405 | 0.434 | 0.457 |
| NFL stability, odd vs even games | 0.591 | same source | 0.610 | 0.632 |

- **Across levels.** Without level baselines, ratings miss by 20.9 points when a power team hosts an FCS team and 5.9 when it hosts another FBS team. With them the misses are 1.5 and 0.1 (222 and 336 games). A team's number now means the same thing whatever its schedule.
- **Quarterbacks.** 1,614 quarterback seasons with 150 or more attempts. Season-to-season carry: public EPA 0.38, our schedule-adjusted number 0.42. For the 18 quarterbacks who moved from another FBS conference up to a power conference, raw EPA fell by 0.072 per attempt and tracked poorly (0.20). The adjusted number changed by 0.002 and tracked at 0.44. Small sample, but it is the result the stat was built to produce.

### 2. Does it predict scores better than what we have? No.

College, next-game team score from ratings alone:

| Rating built on | MAE |
|---|---|
| Points scored and allowed (opponent-adjusted) | 9.06 |
| "Deserved points" (game total of luck-stripped value) | 9.25 |
| Points per drive | 9.31 |
| Public EPA per play | 9.38 |
| APV per play | 9.41 |
| Our raw value per play | 9.46 |
| Success rate | 9.56 |
| Half actual points, half deserved points | 9.09 |
| Points + APV per play + plays per game | 9.01 to 9.02 |

- In college the scoreboard beats every per-play stat, including ours, and blending in deserved points does not help. Turnovers and long plays that the stat treats as luck carry real information about talent gaps.
- In the NFL every basis lands between 7.38 and 7.45 and none differs from another beyond noise. Deserved points equal actual points there (7.41 vs 7.39).
- On top of the desk's college model (8.97): APV plus plays per game improves team scores by 0.06 to 0.08 in all four seasons. Most of that is pace, which the live model already has. The new part is about 0.03.
- On top of the closing line (college 8.69, NFL 7.16): nothing adds anything. The largest change was 0.007.
- On top of our NFL model (7.28): nothing adds anything.

### 3. Leverage weighting

- Dropping plays from decided games makes college ratings much worse (9.92 vs 9.41). Half weight equals full weight. In the NFL it makes no difference.
- Double weight on late, close plays does not help in either league.
- Conclusion: the weights the data chooses are close to flat. Blowout plays still tell us who is good.

### 4. Time of possession and time spent attacking

Attacking time = seconds of possession inside the opponent's half. Red zone time = inside the 20.

| Measure | College stability | NFL stability | Same-game link to points (college) |
|---|---|---|---|
| Time of possession | 0.77 | 0.30 | 0.05 |
| Seconds per play | 0.81 | 0.60 | 0.00 |
| Attacking time | 0.67 | 0.49 | 0.34 |
| Attacking share of own time | 0.31 | 0.53 | 0.44 |
| Red zone time | 0.30 | 0.37 | 0.38 |
| Plays per game | 0.54 | 0.38 | 0.25 |

- These are real team traits, and attacking time does go with scoring in the same game.
- They do not predict the next game's score beyond what plays per game and efficiency already give. Added to efficiency plus plays: time of possession 0.00, attacking time +0.01 (worse), red zone time 0.00, attacking share -0.02 (t -2, not enough). NFL: all slightly worse.
- On college totals over the desk model: plays per game -0.14 (all four seasons, already live as tempo). Attacking time and share add nothing after plays.
- Verdict: a null for prediction in football. Pace matters; where on the field the time is spent does not add to it.

### 5. Other things found along the way

- **College totals.** When our total differs from the close by 5 or more (845 games), the result lands 0.2 points from the close and 7.3 from ours. Our total disagreements have been noise.
- **College totals are spread too wide.** In 2023 to 2025 the actual total moved only 0.64 to 0.78 points for each point of our projected total. Margins are fine (0.96 to 1.00). Pulling totals toward the average using the last two seasons gains 0.04, which is not enough to act on yet.
- **Where the gap to the line is.** The line beats our models by 0.13 (NFL) to 0.28 (college) per team score, and no team play-by-play stat closes any of it. What the market has and these stats lack is who is playing: quarterback and injury news.

## Ranked ideas

Promise is judged on J's yardstick: accuracy of projected scores and stats.

| # | Idea | Evidence | Expected gain | Cost | Type |
|---|---|---|---|---|---|
| 1 | **Use the quarterback version of APV to size quarterback changes and missing starters in college Model A.** Today a missing quarterback is worth a fixed 2.5 or 0.5 points from a yards-per-game rule. | The adjusted number carries across seasons and levels better than EPA (0.42 vs 0.38; transfers up 0.44 vs 0.20). The remaining gap to the line is personnel, not team stats. Not yet tested on scores. | Unknown until tested; this is the only direction pointing at the actual gap | M | Adjusts existing input |
| 2 | **Add APV offense and defense to college Model A team scores.** | -0.03 beyond tempo, all four seasons. Matches the earlier unit-rating result. | Small but dependable, about a tenth of the gap to the line | S | Adjusts existing model |
| 3 | **Publish APV as a desk stat for teams and quarterbacks**, indexed to 100, for comparing teams and players across conferences and levels. | Level bias removed (20.9 to 1.5 points); more stable than EPA. | A comparison tool, not an accuracy gain | M | New display |
| 4 | **Stop trusting college total disagreements.** Keep total picks at pass when the price is not a true opener, and test a narrower total spread next season. | Close is right when we differ by 5 or more; slope 0.64 to 0.78. | Removes bad picks; +0.04 accuracy at most | S | Adjusts Model B |
| 5 | Early-season college prior from points per drive. | -0.06 in weeks 1 to 3, three of four seasons. | Small, early season only | S | Adjusts existing model |

## Tested and not worth building

- Deserved points in place of actual points (college worse, NFL equal).
- Zero or reduced weight for decided games; extra weight for late, close plays.
- Time of possession, attacking time, red zone time and seconds per play as predictors beyond plays per game.
- Explosive-play rate, success rate and points per drive on top of a points rating.
- Our own expected-points model as a replacement for the public one: equal for prediction. Its value is that we control it and can strip luck from it.
- Any team stat on top of the NFL model or either closing line.

## Limits

- The NFL replication used the public NFL expected-points values; we did not fit our own for the NFL.
- Quarterback credit is not separated from his line and receivers.
- FCS teams share one baseline because they appear only once or twice a season in this data.
- Only 18 quarterbacks moved up a level with enough attempts in both seasons.
