# NBA player single-game stat model

Built 2026-10-09 in the "NBA player stat projections" thread. Predicts minutes, points, rebounds, assists, threes,
steals, blocks, turnovers and PRA for a player in a specific game, as a full distribution (mean, range, P(over) any line).
Write-up: [library/01-prediction/nba-player-model-2026-10.md](../../library/01-prediction/nba-player-model-2026-10.md).

## How it works

1. **Minutes**: gradient-boosted model on recent minutes (short and long averages, last game), starter rate,
   days off, which teammates are missing or returning, opponent absences, spread (blowout risk), rest, playoffs.
   On game day each team's projected minutes are pulled halfway toward 241.
2. **Per-minute rate**: each player's recent rate (exponentially weighted, half-life 25 games, 60% carried across
   seasons, shrunk to a position average), then a Poisson model per stat adjusts it for:
   - opponent: what the defense allows of that stat relative to the players it faced, overall and by position (G/F/C)
   - **lineup layer**, two parts: (a) availability: teammates out or returning, measured as their share of team
     shots/rebounds/assists beyond their share of minutes; (b) on/off: this player's own rate with vs without each
     missing teammate on the floor, from lineups rebuilt from play-by-play (2018-19 on)
   - market game environment: implied team total vs the team's normal scoring; game total vs both teams' scoring
   - spread, home, back-to-back, playoffs
3. **Distribution**: negative binomial given minutes, mixed over minutes uncertainty; P(over) is temperature-calibrated
   out of sample (calibration.json, fit on 2023-24).

## Use

```
python3 project.py . games.csv [lines.csv] [out.csv]
```
`games.csv`: one row per team per game: `date,team,opp,home,spread,total,out,add` (spread = that team's expected
margin, + favoured; out/add = player names separated by `;`). `lines.csv`: `name,stat,line`. See `examples/`
(San Antonio vs New York, with and without Wembanyama). The expected lineup is the team's latest box-score roster
minus `out` plus `add`, so early in a season list offseason arrivals in `add` and departures in `out`.
Output also prints **team points built from the player projections** (hook for the NBA team model).

## Rebuild

```
python3 build_dataset.py <dl> .     # player_box_YYYY.parquet -> data/player_games.parquet (+ lines from team-strength)
python3 lineups.py <dl> . 2019 2026 # pbp_YYYY.parquet -> data/lineup_pairs.parquet (on/off by teammate pair)
python3 features.py .               # -> data/features.parquet (walk-forward, ~2 min; not kept in the shared folder, 150 MB)
python3 backtest.py .               # fit 2018-19..2023-24, test 2024-25 + 2025-26 -> backtest_*.csv, backtest_log.txt
python3 calibrate.py .              # -> calibration.json
python3 props_test.py . <props_dir> # vs 2023 sportsbook props -> props_2023_*.csv
rm data/model.pkl                   # project.py refits on 2018-19..2025-26 when the pickle is missing
```
Downloads (into `<dl>`): `github.com/sportsdataverse/sportsdataverse-data/releases/download/espn_nba_player_boxscores/player_box_YYYY.parquet`
and `.../espn_nba_pbp/play_by_play_YYYY.parquet` (save as `pbp_YYYY.parquet`). The 2026-27 files appear there as games
are played; ESPN's own API is blocked from the cloud sandbox. Needs pandas, pyarrow, scikit-learn, statsmodels, scipy.

## Results (2024-25 and 2025-26, 55k player-games, out of sample)

| | model | season avg | last 10 |
|---|---|---|---|
| minutes MAE | 4.47 | (avg minutes 5.06) | 5.87 (last game) |
| points MAE | 4.46 | 4.72 | 4.73 |
| PRA MAE | 5.79 | 6.29 | 6.25 |
| points MAE, high-usage teammate out | 4.76 | 5.21 | 5.10 |
| PRA MAE, high-usage teammate out | 6.24 | 7.18 | 6.96 |

Details: `backtest_log.txt`, `backtest_accuracy.csv`. Calibrated P(over) matches outcomes within ~2 points per bucket.
Against 2023 sportsbook props (612 player-games, almost all 3-pointers made): log loss 0.682 vs market 0.677,
50/50 blend 0.676; i.e. about as good as the market, not shown to beat it (`props_test_log.txt`).

## 2026-10-09 audit changes

Results: [audit, before and after](../../library/01-prediction/model-audit-2026-10.md#before-and-after-the-fixes). Original: `../../archive/pre-audit-2026-10-09/models/nba-player/`.
- Teammate availability in the features is pre-game (chance of playing), not who actually played; `AVAIL=actual`
  restores the old way. `backtest_by_absence_type.csv` scores new and continuing absences separately.
- `project.py` games.csv takes an optional `q` column: `Name:0.6;Name2:0.25` (chance each questionable player plays).
- Count spread is fit per stat and minutes band; `calibrate.py` calibrates around median-like lines.
- `project.py` flags a lean only at P(over) ≥ 0.58 or P(under) ≥ 0.55 (overs need more, audit P3).
- `data/model.pkl` was refit on the new features.

## NBA-2 minutes upgrade (2026-10-11)
`minutes_ext.py` adds seven pre-game minutes features (games since returning from a 3+ game absence, the absence length,
a fading ramp, last-3 minutes and its trend vs the last 15, minutes volatility, tonight's rotation depth). `features.py`
calls it; `model.MIN_X` uses them. Out of sample (2024-25 + 2025-26, 55k player-games):
| | before | after |
|---|---|---|
| minutes MAE | 4.80 | 4.61 (-3.9%, t 26) |
| points / PRA MAE | 4.565 / 5.982 | 4.514 / 5.881 (-1.1% / -1.7%) |
| first 3 games back from a 3+ game absence: minutes / PRA | 5.06 / 5.13 | 4.71 / 4.92 (-6.9% / -4.1%) |
| spread 10+ (blowout risk): minutes | 4.81 | 4.57 (-5.0%) |
Calibration refit (`calibration.json`); model.pkl refit. Pre-change copy: `archive/pre-nba2-2026-10-11/`.
