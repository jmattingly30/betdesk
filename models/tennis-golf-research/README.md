# Tennis + golf matchup research (2026-10-27)

Research only, nothing wired into Bet Desk. Findings page: see project memory `tennis-golf-research`.

## Data (all reachable from the cloud, downloaded to scratchpad, not stored here)
- Tennis results 1877-2024, ATP incl. Challenger/Futures, WTA incl. ITF, with serve stats: R package `deuce` (skoval/deuce) `data/*.RData` on raw.githubusercontent.com (read with `pyreadr`). Jeff Sackmann's own repos return 404 now.
- Grand Slam odds (tennis-data.co.uk columns incl. Pinnacle PSW/PSL, B365, Max, Avg): `deuce` `atp_odds` 2000-2023, `wta_odds` 2007-2023. Slams only.
- Grand Slam point-by-point 2011-2024: `deuce` `gs_point_by_point`.
- Charted matches point by point: JeffSackmann/tennis_MatchChartingProject (raw still works).
- ATP tour 2025-2026: Tennismylife/TML-Database (`2025.csv`, `2026.csv`).
- PGA hole-by-hole 2010-2026 (ESPN based): github.com/array-carpenter/golfastr releases `holes/holes_YYYY.rds`.
- IMPORTANT: deuce match tables repeat ~170k ATP matches (same event, players, round). engine.py drops them; without that, results leak.

## Run order (from a folder with ../data)
1. `python3 build_tables.py 3` and `5` (iid grid tables), `engine.py atp|wta` (walk-forward ratings, ~6 min each)
2. `evalbase.py`, `model.py` (factor tests, val 2012-17, test 2018-24)
3. `slamtest.py`, `slam2.py` (vs Grand Slam odds), `scoremix.py` (score model with day-to-day form), `iidtest.py`, `retire.py`
4. golf: `golf.py` in a folder with rounds.parquet (built from holes_*.rds)
