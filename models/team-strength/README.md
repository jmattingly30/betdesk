# Team strength, matchup and season models (NFL, CFB, NBA, MLB)

Built 2026-10-08 in the "Prediction model sweep" thread. Write-up: [library/01-prediction/model-sweep-2026-10.md](../../library/01-prediction/model-sweep-2026-10.md).

## Use

```
python3 current_ratings.py data/ .                      # rebuild ratings_*.csv + blend_weights.json (expects the raw inputs listed below)
python3 current_ratings.py --predict NFL KC BUF --line -2.5
python3 current_ratings.py --predict CFB Georgia Alabama --neutral
```

`--line` is the home team's spread in betting notation (home -2.5 = favoured by 2.5).

**NBA: retired from the bet workflow (audit T3, 2026-10-09).** Use `models/nba-team` (lineup-dependent, with injuries,
on/off plus-minus and market-anchored probabilities). The NBA files here stay only as the sweep's record.

**2026-10-09 audit changes:** NFL QB term (`nfl/game_model.py`, needs the nflverse weekly player stats as a 5th argument);
CFB preseason prior from returning offensive production (`cfb/prep_returning.py` → `data/cfb_returning.csv`, 5th
argument to `cfb/game_model.py`); `common.margin_ratings` takes optional `priors` and `finals`. Results:
[model-audit-2026-10.md](../../library/01-prediction/model-audit-2026-10.md#before-and-after-the-fixes).

## Files

| Path | What it does |
|---|---|
| `common.py` | Walk-forward margin ratings, market ratings (learned from closing lines), offense/defense filters, metrics |
| `nfl/build_team_games.py` | nflverse play-by-play → team-game EPA, success rate, turnovers, drives |
| `nfl/stat_predictive_validity.py` | Which first-half stats predict second-half results |
| `nfl/game_model.py` | Ratings + situational features vs the closing spread, ATS test |
| `cfb/prep_cfb.py`, `cfb/build_team_games_epa.py` | CFBD games + opening/closing lines; ESPN play-by-play EPA |
| `cfb/game_model.py` | Ratings + EPA vs opening and closing lines, CLV test, totals |
| `nba/prep_nba.py`, `nba/stat_predictive_validity.py`, `nba/game_model.py` | Same for the NBA (four factors, 3-point luck adjustment, rest). Retired: use `models/nba-team` |
| `cfb/prep_returning.py` | ESPN CFB player box scores → returning passing and rush/receiving yards share per team-season |
| `prep_mlb_odds.py` | SBR moneyline/total odds JSON → one row per game, consensus no-vig open/close |
| `mlb/parse_retrosheet.py` | Retrosheet event files → plate appearances, starters, lineups, umpires, weather |
| `mlb/build_features.py` | Shrunk starter/lineup/bullpen ratings, platoon share, bullpen load, park, umpire |
| `mlb/game_model.py` | Logistic model vs opening/closing moneylines, including a "J-style" raw-stats version |
| `mlb/diagnose_j_model.py` | Your 426 logged projections vs market odds |
| `mlb/fit_tool_upgrade.py` → `tool_upgrade_coefficients.json` | Replacement run model for the baseball-win-tool |
| `season_forecast.py` | Preseason and midseason win projections vs naive records |
| `ratings_*.csv`, `blend_weights.json` | Current ratings (points vs an average team, neutral field) |
| `*/run_log.txt`, `*.csv` | Outputs of the last run |
| `data/` | Processed tables (58 MB) so later work doesn't need to re-download |

## Raw sources (all public, reachable from the cloud sandbox via GitHub)

- NFL: `github.com/nflverse/nflverse-data` releases (`pbp/play_by_play_YYYY.parquet`), `raw.githubusercontent.com/nflverse/nfldata/master/data/games.csv`
- CFB: `sportsdataverse/cfbfastR-data` (`schedules/csv/cfb_schedules_YYYY.csv`, `betting/parquet/cfb_line_odds.parquet`), `sportsdataverse-data` release `espn_cfb_pbp`, release `espn_cfb_player_box` (`player_box_YYYY.parquet`, saved as `pb_YYYY.parquet`)
- NFL QB values: nflverse release `player_stats/player_stats.parquet` (1999-2024) and `stats_player/stats_player_week_YYYY.parquet` (2025+)
- NBA: `sportsdataverse/hoopR-nba-data` (`nba/schedules/nba_schedule_master.parquet`, `nba/betting_lines/*`), `sportsdataverse-data` release `espn_nba_team_boxscores`
- MLB: `chadwickbureau/retrosheet` (`seasons/YYYY/*.EVA|EVN`, `GLYYYY.TXT`, `reference/biofile.csv`), odds `ArnavSaraogi/mlb-odds-scraper` release `dataset/mlb_odds_dataset.json`

Scripts take the download folder as an argument; raw downloads were kept in the session scratchpad, not here.
