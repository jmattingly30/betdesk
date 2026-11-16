# NHL and soccer research, Phase 3c (2026-11-16)

Nothing here has been built into a live model. Each idea needs approval first.

This round covers the directive to study time of possession and time spent attacking in the two sports that have no desk model yet, and checks what else moves score accuracy. It builds on the October sim research for each sport (project folders `research/nhl-sim-2026-10/` and `research/soccer-sim-2026-10/`).

Code and results: project folder `research/nhl-soccer-2026-11/` (README, scripts, four results files).

## What was tested

- NHL: every regular-season game from 2021-22 to Nov 2026 (6,858 games, 11,700 team scores in the tests) from the public event feed. The league's own puck tracking (zone time by team and season, 162 team-seasons) was used to check our measures. Our shot-state sim's saved predictions (9,250 team scores) and closing team totals (928 team scores, Jan to Oct 2026).
- Soccer, long sample: 19,763 matches in the five big leagues, 2015-16 to 2025-26, with possession %, shots, shots on target, corners and closing odds (33,058 team scores in the tests).
- Soccer, event detail: every pass, carry and shot of 1,516 matches (Premier League, La Liga, Serie A, Ligue 1, 2015-16), which gives true possession time and time in the final third.
- Each team's number going into a game uses only earlier games. Models are fit with one season (or one league, for the event sample) held out. "MAE" is the average miss in goals on one team's score. A difference is treated as real only when it is large against its noise and has the same sign in nearly every season.
- The market's goal expectation for each soccer team was backed out of the closing home, draw, away and over 2.5 prices.

## Results

### 1. NHL: attacking-zone time is a stable trait that tracks shot share and adds nothing

Measures (from the event feed, checked against tracking): attacking-zone time share = share of located puck time in the team's offensive zone. Possession share = share of time credited to the team with the puck. Attacking possession = both at once.

| Measure | Stability (odd vs even games) | Same-game link to goals | Same-game link to expected goals | Added to goals + expected goals + attempts (MAE 1.381) |
|---|---|---|---|---|
| Attacking-zone time share | 0.82 | -0.12 | 0.35 | -0.002 |
| Possession share | 0.83 | 0.00 | 0.41 | -0.001 |
| Attacking possession share | 0.84 | -0.07 | 0.41 | -0.002 |
| Share of own possession spent attacking | 0.71 | -0.16 | 0.28 | 0.000 |
| Sequences per game (pace) | 0.90 | 0.12 | 0.47 | 0.000 |
| Seconds per sequence | 0.65 | -0.06 | 0.28 | 0.000 |
| Rush share of shots | 0.57 | 0.09 | -0.01 | 0.000 |
| Expected goals (for reference) | 0.82 | 0.36 | 1.00 | -0.008 over goals alone, every season |

- Our zone-time measure matches the league's tracked zone time closely (0.90 across 162 team-seasons). Tracked zone time is itself nearly the same thing as shot-attempt share (0.89).
- In the same game, more time in the attacking zone goes with fewer goals, not more. Teams that are behind spend the rest of the game attacking. Any possession model has to know the score.
- Season level, tracked zone time into next season's goals per game: expected-goal share explains 23%, zone time 16%, both together 23%.
- Over the sim (MAE 1.381): the largest change from any time measure is -0.003, and it holds in only three of five seasons. Over the closing team total (1.385): nothing.
- All the time measures together: -0.001.

### 2. NHL: a team on the second night of a back-to-back scores about 0.2 fewer goals, and the sim does not know it

| | Goals per game |
|---|---|
| Team on a back-to-back (15% of team-games) | 2.85 |
| Team with a day or more of rest | 3.09 |
| Against a back-to-back opponent | 3.14 |
| Against a rested opponent | 3.04 |

- Over the sim, with home ice held fixed: -0.21 goals for the tired team (t -3.6), same sign in all four seasons with dates (6,964 team scores). The gain in average miss is small (-0.002) because it applies to one game in seven, but on those games the sim is 0.2 goals high.
- The opponent's extra scoring is already in the sim through the starting goalie (+0.05, not distinguishable from zero).
- Over the closing team total the same -0.19 shows up, but on 154 team scores it is not distinguishable from zero. We cannot say yet whether the market prices it fully.

### 3. NHL: other things tested

- Splitting expected goals into 5-on-5 and special teams: -0.001. By shot type (rush, cycle, forecheck, rebound): 0.000. Penalties drawn: 0.000.
- The sim and the closing team total are level on this sample (1.3845 and 1.3852). No team stat tested adds to either.
- The public feed shows about 80 sequences per team per game. Hand-tracked games (28 with every puck touch logged) have about 235 true possessions per team, with a median length of 6 seconds. A true possession-by-possession NHL model cannot be fed from public data. The shot-state sim is the right level of detail.

### 4. Soccer: possession is the most stable thing a team does and is worth about a thousandth of a goal

Eleven seasons:

| Rating built on | MAE |
|---|---|
| Home advantage only | 0.985 |
| Possession % only | 0.930 |
| Shots on target | 0.918 |
| Goals | 0.915 |
| Goals + shots on target + shots | 0.909 |
| Closing odds (market's goal expectation) | 0.889 |

- Possession stability is 0.89, against 0.66 for goals and 0.72 for shots on target.
- Added to goals and shots: -0.0012 (t -3.3, nine of eleven seasons). The sign is real and the size is too small to matter. Over the close: -0.0002.
- Winners average 51.5% possession, losers 48.5%.

### 5. Soccer: time in the final third behaves the same way

Event detail, 1,516 matches. A team has the ball in play about 22 minutes a match, about 5 of them in the final third and under 30 seconds in the box.

| Measure | Stability | Same-game link to goals | Same-game link to expected goals | Added to goals + expected goals (MAE 0.915) |
|---|---|---|---|---|
| Possession time share | 0.88 | 0.11 | 0.31 | -0.005 |
| Final-third time share (field tilt) | 0.72 | 0.11 | 0.39 | -0.005 |
| Final-third time, seconds | 0.79 | 0.10 | 0.39 | +0.002 (worse) |
| Share of own possession in the final third | 0.72 | 0.04 | 0.30 | 0.000 |
| Box time share | 0.64 | 0.18 | 0.42 | -0.005 |
| Seconds per possession | 0.93 | 0.17 | 0.25 | -0.002 |
| Possessions per match | 0.73 | -0.07 | 0.11 | -0.001 |
| Passes | 0.95 | 0.16 | 0.30 | -0.007 |
| Expected goals (for reference) | 0.68 | 0.59 | | -0.011 over goals alone |

- The -0.005 results all have t near -2 on one season of data and are the same signal (how much of the ball a team has). On the eleven-season sample that signal is worth -0.001.
- Over the close (0.883 on these matches): nothing adds. The largest change is -0.001.
- How directly a team attacks (speed toward goal, time to first shot) is less stable and adds nothing.

### 6. Where the gap to the market is

- NHL: there is no gap on team totals. The sim ties the close.
- Soccer: ratings from goals and shots are 0.020 behind the close per team score, and 0.03 with expected goals on the one-season sample. October's research found the same gap and traced it to line-ups and match-day news, not team strength. Nothing in this round changes that.

## Ranked ideas

| # | Idea | Evidence | Expected gain | Cost | Type |
|---|---|---|---|---|---|
| 1 | **Add rest to the NHL sim**: lower a team's scoring rate on the second night of a back-to-back. | -0.21 goals over the sim, t -3.6, four seasons of four. | Removes a 0.2 goal bias on 15% of team-games; -0.002 overall | S | Adjusts existing sim |
| 2 | **Design the NHL model (Phase 4) around the shot-state sim, not a possession chain.** Use shot rates and quality by score and strength, starting goalie, rest. Leave zone time out. | Zone time is shot share (0.89 to 0.94) and adds nothing; public data sees a third of true possessions. | Avoids building a layer that cannot be fed | - | Design decision |
| 3 | **Design the soccer model (Phase 4) as a possession engine driven by expected goals, with possession share setting how many possessions each side gets, not how good it is.** Score state has to change the rates. | Possession is the most stable input (0.89) and worth 0.001 on strength; expected goals carry the strength; trailing teams attack more in both sports. | Structure for match, team-total and player outputs | L | New model, needs approval |
| 4 | **Soccer line-ups before kick-off** as the first input after the base model. | Gap to the close is 0.02 to 0.03 goals and is not team stats (this round and October). | The only direction pointing at the gap | M | New input |
| 5 | Track the back-to-back effect against closing team totals as lines accumulate. | -0.19 over the close on 154 team scores, not yet distinguishable from zero. | Tells us if it is a pricing gap or only a sim fix | S | Testing |

## Tested and not worth building

- NHL: attacking-zone time, possession time, attacking possession, sequence length, pace, rush share, shot-type splits, 5-on-5 vs special-teams splits, penalties drawn, as predictors beyond goals, expected goals and attempts.
- Soccer: possession share, final-third time, field tilt, box time, tempo, directness, pressing counts, corners, as predictors beyond goals, expected goals and shots.
- With football, college basketball and the NBA, that makes time of possession and attacking time a null for score prediction in all six sports. They are stable style traits. They are useful for setting pace and shape inside a simulation, not for rating teams.

## Limits

- NHL zone and possession time are estimates from event locations, not tracking. Per-game tracking is not public; the check against tracking is at team-season level.
- Game dates, and so the rest test, cover 2023-24 onward.
- NHL closing team totals cover Jan to Oct 2026 only.
- Soccer event detail is one season and four leagues. The usual expected-goals source (Understat) was down, so the eleven-season sample uses shots on target in its place.
- Possession % is missing for 4% of matches.
