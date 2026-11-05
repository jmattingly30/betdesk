# College basketball score / win / cover model

Built 2026-10-09 in the "College basketball model" thread. Write-up and results: [library/01-prediction/cbb-model-2026-10.md](../../library/01-prediction/cbb-model-2026-10.md).

## Use

```
python3 predict.py "Duke" "North Carolina"                                   # first team is home
python3 predict.py "Duke" "Houston" --neutral --line -2.5 --total 131.5 --at open
python3 predict.py "Kansas" "Kentucky" --preseason --line -3                  # 2026-27 preseason ratings
```

`--line` is the home team's spread in betting notation (home -2.5 = favoured by 2.5). `--at open` (default) uses the opener blend weight (35% of the model's disagreement counts); `--at close` uses 7%.

## Rebuild (run from this folder)

```
python3 prep_cbb.py <hoopR dir> data            # games + team-game possessions/efficiency
python3 prep_returning.py <hoopR dir> data      # roster continuity: returning minutes, transfers
python3 fetch_lines.py <hoopR dir> lines.jsonl 2026,2023,...   # ESPN pick-center lines from hoopR-mbb-raw game JSONs (~35 min)
python3 prep_lines.py lines.jsonl data/cbb_lines.parquet
python3 game_model.py                           # ratings, priors, backtest, outputs
```

`<hoopR dir>` holds `mbb_schedule_YYYY.parquet`, `team_box_YYYY.parquet` and `player_box_YYYY.parquet` from `raw.githubusercontent.com/sportsdataverse/hoopR-mbb-data/main/mbb/{schedules,team_box,player_box}/parquet/`.

## Files

| Path | What it is |
|---|---|
| `ratings.py` | Walk-forward opponent-adjusted efficiency (O, D, home court) and tempo ridge regressions with a fading preseason prior |
| `priors.py` | Preseason O/D/pace from last two seasons, returning minutes, transfers in; expanding-window fit |
| `game_model.py` | Full pipeline + backtest: accuracy, win probability, ATS/totals by threshold, CLV at the 2026 open |
| `predict.py` | Price one game |
| `ratings_cbb.csv` | End-of-2025-26 ratings (net, O, D per 100 possessions vs average D1; tempo = possessions vs an average team) |
| `model_config.json` | Tuned parameters, margin sigma, open/close blend weights |
| `prior_coefficients.csv` | Prior regression coefficients by season |
| `accuracy_vs_close.csv`, `win_prob_calibration.csv`, `bets_by_threshold.csv`, `clv_2026_open.csv`, `tuning.csv`, `run_log.txt` | Outputs of the last run |
| `data/` | Processed tables (27 MB): games, team-games, lines, roster links, returning minutes, predictions for every game, ratings by season |

## Data notes

- Lines: ESPN consensus closing spreads/totals 2013–2023 (TeamRankings as fallback), DraftKings open + close for 2026; none for 2024–2025 in this source. 1,166 spreads that contradicted the same book's moneyline were dropped. Consensus and TeamRankings disagree on ~1/3 of games; only "verified" closes (agree within 0.5) are used to judge edge at the close.
- D1 filter: both teams play 15+ games that season against such teams.
- `ratings_cbb.csv` IDs are ESPN team IDs.

## 2026-10-09 audit changes

Results: [audit, before and after](../../library/01-prediction/model-audit-2026-10.md#before-and-after-the-fixes). Original: `../../archive/pre-audit-2026-10-09/models/cbb/`.
- Player availability (`prep_availability.py` → `data/cbb_availability.parquet`, `data/cbb_player_values.csv`). The model
  prices continuing absences only (known before the open). `predict.py --out-home "Name;Name2"` / `--out-away` use the saved values.
- Sigma by month and pace (`--month`); the opener weight is cross-fit and shrunk (0.27 spread, 0.22 total), not 0.35.
- Ratings re-tuned with the roster prior (`tune.py`, `tuning.csv`). The blowout cap and near-home neutral edge were
  tested and rejected; `--near-home` stays available but adds nothing while the tuned value is 0.
- `pipeline.py` holds the shared two-pass rating run. `prep_roster_2027.py` refuses ESPN's 2027 roster while it is still
  last season's; rerun it once the real rosters appear.

## 2026-10-10: CBB-1 + CBB-2 merged (player-value priors, conference tiers)
- `prep_player_prior.py` -> `data/cbb_player_seasons.csv`: each player's shrunk per-minute Game Score by season (150-minute
  shrinkage toward replacement, 0.148). `priors.py` adds three preseason inputs from them: value of the players on this
  season's roster (`r_val`), the same weighted by their previous team's level (`r_val_lvl`), and value brought in by
  transfers (`xfer_val`). `CBB_PLAYER_PRIOR=0` turns them off. Pre-change copy: `archive/pre-cbb2-2026-10-10/`.
- 2026 DraftKings openers, continuing-absences model: margin RMSE 11.375 -> 11.351; gap >= 3: 631 bets 56.3% -> 56.1% ATS,
  CLV +1.45 -> +1.55 pts (t 10.3 -> 11.0); gap >= 4 CLV +2.60 -> +2.88; Nov-Dec 56.6% -> 58.9% ATS (gap >= 3), CLV +1.54 -> +1.77.
- `tier_split.py` (log `tier_split_log.txt`): the model's edge over the opener is LARGEST in high-major games (77% of the gap
  shows up in results) and smallest when a low-major team plays (30%); non-conference games 62% ATS at gap >= 3 (258 bets).
  The merge helped low-major and non-conference games most. Per-tier opener weights (cross-fit, shrunk halfway to the
  close) are in `model_config.json` `blend_open_tier`; `predict.py --at open` uses the lower team's tier; team tiers in
  `data/cbb_team_tier.csv`; `export_app.py` writes `open_tier` and a tier letter per team.
- `predict.py --preseason` now adds last season's league-average value for roster inputs it can't know yet (transfer
  quality and the player terms; `data/cbb_prior_feature_means.csv`). Before, those were left out, which shifted every
  team's preseason rating down (mean net -2.6, now -0.8). Real 2027 rosters (`prep_roster_2027.py`) still pending at ESPN.

## Shooting-luck correction (2026-10-11, stat hunt)
`luck.py`: over each team's last 10 games, points opponents scored above league-average FT% and 3P% (defence has little
control) and the team's own FT%/3P% swing. The ratings take those points at face value; the correction shifts O and D
(model_config `shooting_luck`: +0.173 per point of defensive luck, -0.103 per point of own shooting luck on the margin,
-0.124 on the total). Walk-forward 2015-2026, refit each season on earlier ones (coefficients stable):
margin MAE 8.722 -> 8.701 (t 5.1); ATS at 3+ pt gaps vs close 52.2% -> 53.2% (5,000 games); 2026 DK opener at 3+ pt
gaps CLV +1.55 -> +1.65 pts (t 12.8), ATS 55.9% -> 58.0%. predict.py and export_app.py apply it to in-season ratings
automatically (`--no-luck` to compare); preseason ratings are unchanged. Archive: archive/pre-luck-2026-10-11/.

Data caution found in the same work: about 6.6% of the historical **consensus** lines at neutral sites (mostly
November-December events) have the sign flipped (they disagree with the model and the result by the line's full size).
Any neutral-site historical test must drop rows where |line - margin_hat| exceeds |-line - margin_hat| by 6+.
The 2026 DraftKings lines and non-neutral games are clean (0.1%). The model's own blend fits use 'verified' closes
(consensus and TeamRankings agree), which are clean (0.07% flagged), so earlier results stand; 16% of UNverified neutral closes are flipped.
