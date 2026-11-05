# NHL player single-game prop model

Built 2026-10-09 (J's app plan item 4). Write-up: [library/01-prediction/new-models-2026-10.md](../../library/01-prediction/new-models-2026-10.md).

Stats: shots on goal, points, goals, assists, blocked shots, hits (skaters); saves (starting goalie). Every projection
assumes the player plays; goalie lines assume that goalie starts.

## How it works
- `build_dataset.py`: NHL API box scores (sportsdataverse-data releases; only 2023-24 on have game ids) -> pre-game
  features: TOI recency averages, per-60 rates shrunk to the previous season's position average (Marcel style),
  last-10/season averages, team and opponent shots/goals for and against (recency weighted), rest days, goalie save %
  shrunk to .900.
- `model.py`: LightGBM (Poisson) mean per stat, negative binomial spread with a fitted dispersion, temperature
  calibration fit on 2024-25. Backtest: fit 2023-24 -> calibrate 2024-25; fit 2023-25 -> test 2025-26.
- `project.py`: each team's next game where it is also the opponent's next game (rerun daily) -> `nhl_props.json` for Bet Desk.

## Results (2025-26, out of sample)
| stat | MAE model | MAE last-10 | log loss model | log loss last-10 |
|---|---|---|---|---|
| shots on goal | 1.013 | 1.055 | 0.406 | 0.425 |
| points | 0.524 | 0.529 | 0.430 | 0.465 |
| goals | 0.267 | 0.267 | 0.393 | 0.433 |
| assists | 0.404 | 0.401 | 0.337 | 0.371 |
| blocks | 0.729 | 0.737 | 0.419 | 0.447 |
| hits | 0.880 | 0.901 | 0.396 | 0.416 |
| goalie saves | 5.01 | 5.55 | 0.588 | 0.669 |
No market test (no free NHL prop archive). PIM dropped (badly calibrated). No power-play TOI split yet (needs the 65 MB/season play-by-play).

## Daily use
```
python3 fetch_data.py <dl> 2027 2027      # refresh current-season box scores
curl -L -o <dl>/sched_2027.parquet https://github.com/sportsdataverse/sportsdataverse-data/releases/download/nhl_schedules/nhl_schedule_2027.parquet
python3 project.py <dl> ../../app/data/nhl_props.json 2
```
Then republish Bet Desk with `nhl_props.json` (app/README.md). Refit (`build_dataset.py` + `model.py`) every few weeks.
