# NBA team score / win / cover model (lineup dependent)

Built 2026-10-09 in the "NBA team score, win and cover model" thread (J's basketball item 3). Write-up:
[library/01-prediction/nba-team-model-2026-10.md](../../library/01-prediction/nba-team-model-2026-10.md).

## Use

```
python3 predict.py SA NY                                              # home team first, ESPN codes
python3 predict.py SA NY --out-home "Victor Wembanyama" --line -3.5 --total 226.5      # star out, priced vs a line
python3 predict.py BOS NY --out-away "Jalen Brunson;Karl-Anthony Towns" --at close
python3 predict.py SA NY --out-home "Victor Wembanyama" --minutes ../nba-player/projections.csv   # player model's minutes
```
Prints projected score for each team, margin, total, P(home win), and with `--line` / `--total` market-anchored
P(cover) / P(over); when anyone is out it also prints the full-strength price, so the difference is what the
absence is worth. `--line` is the home spread in betting notation (home -3.5 = favoured by 3.5). `--add-home/--add-away`
adds players who changed teams (rosters come from the latest box scores, so before 2026-27 games are played,
list offseason arrivals and departures). Example output: `examples_wemby_out.txt`.

## How it works

1. **Engine**: the college model's tempo x efficiency ratings (`../cbb/ratings.py`, walk-forward ridge with a fading
   preseason prior), fit on NBA team-games.
2. **Lineup layer** (`lineup.py`): every player's pre-game box-score rates per 100 possessions on the floor (recency
   weighted, shrunk toward bench level) and projected minutes. A team's expected ten-man rotation, weighted by minutes,
   gives a lineup line per 100 possessions; regression weights (fit on earlier seasons only) turn the two lineups into
   expected points per 100 for each side. Team O/D ratings are fit on what the lineup does not explain.
3. **Absence weight**: the lineup term enters at k = 0.5 (tuned), plus an extra 0.22 of the full value of any regular
   who sits (fit walk-forward on results), so a missing star is worth about 0.72 of his full box-score value.
4. **Probabilities**: win = Normal(margin / 13.2); cover/over = market-anchored (only 19% of the model-minus-opener gap
   and ~3% of the model-minus-close gap is real, from the backtest).

## Rebuild (run from this folder)

```
python3 prep.py data/sbr_nba_archive_10Y.json   # games, possessions, ESPN/Odds API closes, SBR opens 2011-12..2021-22
python3 lineup.py                               # player rates, current player state, lineup aggregates (~3 min)
python3 game_model.py                           # backtest + outputs (~2 min); add --tune to redo the 21-run grid (~15 min)
```
Inputs: `../team-strength/data/nba_games.parquet`, `nba_team_games.parquet`, `../nba-player/data/player_games.parquet`.
SBR archive: `raw.githubusercontent.com/flancast90/sportsbookreview-scraper/main/data/nba_archive_10Y.json` (copy kept in data/).
To update for 2026-27, refresh the player box scores per `../nba-player/README.md`, rebuild team-strength's NBA tables, rerun all three.

## Files

| Path | What it is |
|---|---|
| `prep.py`, `lineup.py`, `game_model.py`, `predict.py` | pipeline and pricing tool |
| `model_config.json` | tuned parameters, lineup weights by season, absence boost, sigma, open/close blend weights |
| `team_residual_ratings.csv` | end-of-2025-26 team ratings beyond their players (O, D per 100, tempo) |
| `accuracy_vs_close.csv` | margin/total RMSE by season: close, open, lineup model, team-only model |
| `star_absence_accuracy.csv` | the same by size of the biggest absence in the game |
| `clv_open.csv`, `bets_by_threshold.csv` | closing-line value from the opener; ATS / totals records by model gap |
| `win_prob_calibration.csv`, `tuning.csv`, `run_log.txt` | outputs of the last run |
| `data/` | games, team-games, player rates, lineup aggregates, current players, predictions for every game (2015-2026) |

## 2026-10-09 audit changes

Results: [audit, before and after](../../library/01-prediction/model-audit-2026-10.md#before-and-after-the-fixes). Original: `../../archive/pre-audit-2026-10-09/models/nba-team/`.
- `rapm.py <pbp dir>`: on/off plus-minus from play-by-play stints, refit monthly. Run it before `lineup.py`.
- `lineup.py` adds a "known at the open" lineup (only continuing absences); `game_model.py` re-runs the opener test on it.
- `predict.py --q-home "Name:0.6" --q-away "Name:0.3"` prices questionable players as probabilities; `--playoff` adds
  the playoff home edge; with `--line`, a market-anchored P(home win) is shown.
- Honest opener result: spreads CLV +0.43 at 3+ points (was +0.87 when the backtest knew who sat); totals hold (+0.84, 53.6%).

## NBA-1 and NBA-3 (2026-10-11)
- **NBA-1 news repricer** `reprice.py HOME AWAY --date D --line L --total T --news "TEAM:Name:out|in|0.4" [--props props.csv]`:
  reprices the game (lineup model with the player model's minutes) and every player's minutes / points / rebounds /
  assists / PRA before and after the news, plus P(over) at posted prop lines. The edge it targets: CLV +1.18 pts at 3+
  point gaps on games where news came after the open (vs +0.35 without news). Example (Wembanyama out, SA -4.5):
  fair spread SA -1.9 -> NY by 2.8, a 4.7-point move; Castle 15.5 points over 39% -> 45%. First run refits the player
  model (~7 min); later runs ~1-2 min.
- **NBA-3 roster continuity** (tested, not adopted): scaling each team's carried-over residual rating by how much of last
  season's minutes returned changed early-season accuracy by under 0.002 RMSE. The lineup layer already builds team
  strength from player box-score + RAPM ratings times minutes, which is what NBA-3 proposed, so new rosters are already
  priced from their players. Test: scratch `nba3_test.py` logic described here; continuity^0.5..1.5 all flat.
