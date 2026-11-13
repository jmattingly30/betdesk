# Adjusted Play Value (APV): how it is built

This is the process and the math for our own EPA-style football stat, written so the pieces can be changed one at a time. Results are in `football.md`.

Scripts and saved data: project folder `research/football-2026-11/`. The `data/` folder there holds everything needed to re-run steps 4 to 6 without the raw play-by-play:
- `team_games.parquet` (college) and `nfl_team_games.parquet`: one row per team per game with every sum used below.
- `preds.parquet`: each metric's expected value for each team-game, known before the game.
- `final_ratings.parquet`: season-end offense and defense numbers per team.
- `apv_teams_2025.csv`, `apv_teams_2026.csv`, `apv_qbs_2025.csv`: readable rankings.
- `ep_model.txt`, `ep_meta.json`: the fitted expected-points model and the luck constants.

Steps 1 to 3 need the raw plays (about 600 MB, free download, see the README).

## Step 1. Expected points (EP)

For every play we know the state `s` = (down, distance, yards to goal, seconds left in the half, half, score difference, timeouts for each side, goal-to-go flag).

Target: the next score in the same half, signed for the team with the ball. A touchdown with its extra point is +7 if they score next and -7 if the opponent does; 0 if nobody scores before the half ends.

`EP(s)` = the average of that target for plays in state `s`. We estimate it with gradient-boosted trees (LightGBM, 400 trees, 63 leaves, at least 400 plays per leaf) fit on 2014 to 2021 plays only. A kickoff is one fixed state worth +0.535 to the receiving team.

Things to try: add season as an input so the scoring environment moves with the year; fit by era; use drive points as the target in place of next score.

## Step 2. Play value

For play `i` by team `p`, with the next play `j` in the same half:

`V_i = points_i + sign * EP(s_j) - EP(s_i)`

- `points_i` = points `p` scored on the play minus points the opponent scored on it.
- `sign` = +1 if `p` still has the ball on play `j`, otherwise -1.
- If the half ends, the `EP(s_j)` term is 0.

This is the same idea as public EPA; the correlation with it is 0.92.

## Step 3. Take the luck out (X)

- **Fumbles.** Every fumble play, lost or kept, gets the average value of all fumble plays in that field zone. Zones are by yards to goal: 0 to 20, 20 to 50, 50 to 80, 80 to 100. Averages: -3.68, -3.04, -2.81, -2.84.
- **Interceptions.** Every interception gets the zone average, so the return is ignored: -5.04, -4.00, -3.88, -3.78.
- **Cap.** `X4 = clip(X, -4, +4)`. Caps of 3 and 2.5 were also stored (`X3`, `X2`). Tighter caps are more stable and predict the next game's value better, but did not predict points better.
- **Kicks** (used only in the game-total version): a field goal or punt gets the average value of all such kicks from the same 5-yard band.

Things to try: credit passes broken up as part-interceptions; treat sacks and fumbles on sacks separately; keep part of the return value; different caps for runs and passes.

## Step 4. Leverage weights

Each scrimmage play falls in one of three groups using win probability `wp` before the play:
- decided: `wp < 0.05` or `wp > 0.95`
- late and close: fourth quarter and `0.2 <= wp <= 0.8`
- everything else

Team-game value per play with weights `w1` (decided) and `w2` (late and close):

`(X4_rest + w1 * X4_decided + w2 * X4_late) / (n_rest + w1 * n_decided + w2 * n_late)`

Tested `w1` in {0, 0.5, 1} and `w2` in {1, 2}. Best was `w1` between 0.5 and 1 with `w2 = 1`, which is nearly flat weighting. `w1 = 0` was much worse in college.

Things to try: a smooth weight in `wp`; weights that differ for offense and defense; score-and-time groups in place of `wp`.

## Step 5. Schedule and level adjustment

For team-game `g` (offense `i`, defense `j`), the per-play value `y_g` with `n_g` plays is modelled as

`y_g = mu + h * home_g + L_off[level(i)] + o_i + L_def[level(j)] + d_j + error`

- `home_g` is +0.5 at home and -0.5 away.
- `level` is power conference, other FBS, or FCS. `L_off` and `L_def` are the level baselines (power is the zero point).
- `o_i` is the offense's number and `d_j` is what the defense allows (lower is a better defense).

We choose the numbers to minimise

`sum over games of n_g * (y_g - fitted)^2 + lambda * sum over teams of [(o_i - prior_o_i)^2 + (d_i - prior_d_i)^2]`

- `lambda = 250` plays, about three and a half games. Values from 80 to 1000 gave the same conclusions; 150 to 500 were best.
- `prior` = half of the team's final number from last season (0 for a team with no last season).
- Going into week `w`, only games before week `w` are used. Nothing from the game being predicted is used.

A team's full offense rating is `L_off[level] + o_i`; its defense rating is `L_def[level] + d_i`.

Things to try: conference baselines in place of three levels; a different carry than one half, or one that depends on returning players; separate run and pass numbers; a larger `lambda` early in the season.

## Step 6. The index

`APV+ = 100 + 15 * (rating - FBS average that season) / (FBS standard deviation that season)`

Defense is flipped so that higher is better. Because each season is scaled to itself, a season with more scoring does not inflate the number.

A ratio form like wRC+ (100 times the team's value over the league value) does not work per play because the league average value per play is about zero. It does work per drive: 100 times adjusted points per drive over the league's points per drive. That form is a small change to step 6 if preferred.

## Quarterback version

- Value per pass attempt: mean of `X4` over the quarterback's attempts and interceptions (sacks carry no passer in this data).
- Schedule: subtract the average defense rating he faced, measured against the FBS average that season. `adjusted = X4 per attempt - (mean d_j faced - FBS mean d)`.
- Index as in step 6 among quarterbacks with 150 or more attempts.
- Not yet separated from his line and receivers.

## Game-total version ("deserved points")

`XT` = sum of luck-stripped, capped value over every play a team ran with the ball in a game. `EP0` = sum of the expected points at the start of each of its drives.

`deserved points = 22.06 + 0.687 * XT + 0.592 * EP0` (fit on 2014 to 2021 college games; same-game correlation with real points 0.90).

This did not predict future scores better than real points in college and tied in the NFL.

## How each choice was judged

1. Stability: the same team's number in odd games vs even games.
2. Next-game prediction: fit points on the two teams' expected values using 2015 to 2021, score on 2022 to 2025.
3. On top of the desk model and on top of the closing line.
4. Across levels: the average miss when teams from different levels meet.

`quick.py` runs test 2 for any list of metrics in a few seconds, for example `python3 quick.py X4 E PTS X4+N`.
