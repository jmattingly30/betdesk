# NFL player single-game stat model

Built 2026-10-09 in the "Player stat models for other sports" thread (J picked NFL first). Projects passing, rushing,
receiving and touchdown props for every offensive skill player in a specific game, as a full distribution (mean,
10th–90th percentile range, P(over) any line). Write-up: [library/01-prediction/nfl-player-model-2026-10.md](../../library/01-prediction/nfl-player-model-2026-10.md).

## How it works

1. **Team volume**: Poisson GLMs for team pass attempts and carries from the team's recent tendencies, what the opponent
   faces, spread, total, home, weather, playoff, and a change of starting QB.
2. **Opportunity** (LightGBM, Poisson loss): each player's targets and carries, and the starting QB's attempts, from team
   volume, recent target/carry shares (half-lives 3 and 8 games, last season carried at half weight), snap shares,
   **share of today's active group** (my share / sum of shares of the teammates who are playing) and the share held by
   missing teammates at the same position.
3. **Efficiency** (Poisson GLMs, opportunity as offset): receptions, receiving yards, rushing yards, completions,
   passing yards, passing TDs and touchdowns, on the player's shrunk rate (shrunk to a position average), what the
   opponent allows to that position relative to the players it faced, the starting QB's accuracy, the implied team
   total and wind.
4. **Distribution**: negative binomial targets/carries/attempts, binomial catches and completions, yards summed from
   real per-play gains (2016–2026 play-by-play, by position) scaled to the player's projection, Poisson TDs;
   P(over) temperature-calibrated on 2023 (`calibration.json`).

## Weekly use

```
python3 fetch_data.py <dl>                 # refresh 2026 play-by-play, snaps, injuries, schedule (past seasons cached)
python3 build_dataset.py <dl> .            # -> data/player_games.parquet, data/play_yards.parquet
rm data/model.pkl                          # optional: refit including the newest games (~30 s)
python3 project.py . <dl> 2026 6 [--overrides overrides.csv] [--lines lines.csv]
```
Players included: anyone who played offense for the team in its last two games, plus Questionable players who have
played for the team this season, minus anyone ruled Out or Doubtful on the nflverse injury report for that week.
The starting QB is last game's starter unless that QB is out. Override with `overrides.csv` (`team,out,add,qb`, names
separated by `;`). `lines.csv` (`name,stat,line`; stats: targets rec rec_yds carries rush_yds rush_rec_yds pass_att
cmp pass_yds pass_td td) adds calibrated P(over). Output: `projections_SEASON_wkWW.csv`.
Spread and total come from the nflverse schedule (games.csv), which carries current lines.

`projections_2026_wk05.csv` is the first run (Sunday/Monday week 5 games, made Friday Oct 9 before the Friday injury
reports were posted, so only Thursday's report was applied; re-run after `fetch_data.py` to apply Sunday's).

## Rebuild / backtest

```
python3 features.py .        # -> data/features.parquet (walk-forward, ~20 s; not kept, 18 MB)
python3 backtest.py .        # fit 2017-22 -> calibrate 2023; fit 2017-23 -> test 2024-25
```
Needs pandas, pyarrow, numpy, statsmodels, lightgbm.

## Results (2024 and 2025 seasons, out of sample; players with >=25% snaps and 3+ prior games; QBs = starters)

| | n | model MAE | last-8 avg | season avg | gain |
|---|---|---|---|---|---|
| WR/TE receptions | 6,971 | 1.37 | 1.45 | 1.46 | 5% |
| WR/TE receiving yards | 6,971 | 19.2 | 20.1 | 20.2 | 5% |
| RB rushing yards | 2,072 | 21.8 | 23.3 | 23.3 | 6% |
| RB carries | 2,072 | 3.55 | 3.88 | 3.86 | 8% |
| RB rush+rec yards | 2,072 | 25.7 | 27.6 | 27.5 | 7% |
| QB pass attempts | 1,102 | 6.3 | 7.6 | 8.0 | 17% |
| QB passing yards | 1,102 | 55.8 | 64.3 | 67.5 | 13% |
| QB passing TDs | 1,102 | 0.89 | 0.95 | 0.98 | 6% |
| RB carries, RB teammate out | 428 | 3.65 | 4.24 | 4.23 | 14% |
| WR/TE rec yards, starting QB changed | 969 | 18.1 | 19.6 | 19.4 | 7% |

Anytime TD (WR/TE/RB): log loss 0.457 vs 0.549 for the player's last-8 TD rate and 0.518 for the league rate.
QB rushing yards: no gain over averages. Passing yards ran 6 yards high in 2024-25 (league passing fell).
P(over) against a stand-in line (last-8 average rounded to .5) is calibrated within ~3 points in every bucket.
Full tables: `backtest_log.txt`, `backtest_accuracy.csv`, `backtest_calibration.csv`.

No free archive of historical NFL prop lines was found, so there is no market test yet; the next step is logging
prop lines each week and scoring the model against them.

## 2026-10-09 audit changes

Results: [audit, before and after](../../library/01-prediction/model-audit-2026-10.md#before-and-after-the-fixes). Original: `../../archive/pre-audit-2026-10-09/models/nfl-player/`.
- `features.py . <dl_dir>` uses the injury reports (`inj_YYYY.parquet`) for the starting QB (last week's unless Out or
  Doubtful) and each teammate's chance of playing; `project.py` does the same for the coming week.
- League passing drift: team volume and efficiency GLMs weight recent seasons more (half-life 2 seasons) and the backtest
  refits every season. A league-environment feature was tried first and made the passing bias worse.
- QB scrambles split from designed runs (`build_dataset.py` adds `scrambles`).
- Route participation was not added: current-season participation is not published until after the season.
- `data/model.pkl` was removed; `project.py` refits it on first run.

## 2026-10-10 NFL-2 / NFL-3 upgrade (Edge research thread)

`python3 backtest_nfl23.py . <ngs_dir>` tested tracking abilities that "travel" on top of this model, 2024-25 out of
sample (fit 2017-23 -> 2024, 2017-24 -> 2025). Log: `backtest_nfl23_log.txt`, table `backtest_nfl23_accuracy.csv`.

| Change | Result |
|---|---|
| NFL-2: rush yards over expected (NGS) + team-changer terms in the rushing-yards GLM | no gain (RB rush yds MAE 21.95 -> 22.01); given the back's shrunk YPC, RYOE adds nothing |
| NFL-2: same abilities in the carries model | carries 1.2% better overall but rushing yards not better; not adopted |
| NFL-3: aDOT, air-yard share, YAC over expected in the receiving-yards GLM | 0.4% worse overall, 0.6% better for team changers; not adopted |
| **NFL-3: target model with tracking abilities + team-changer status, used only for a player's first 8 games with a new team** | **adopted**: those players' targets MAE 1.671 -> 1.634 (-2.2%), receiving yards 16.38 -> 16.10 (-1.7%), t ~5 on 1,039 player-games; everyone else unchanged |

The existing model already carried most of NFL-3 (air-yard share and aDOT drive its target model). Wiring:
`features.add_tracking()` (ngs_rushing/ngs_receiving from `fetch_data.py`), `model.fit_changer_targets()`;
`project.py` adds the team-changer model to an older `model.pkl` on its first run. Pre-change copies:
`../../archive/pre-nfl23-2026-10-10/`.

## 2026-10-10 scheme matchups (J's follow-up): tested, not adopted
`python3 scheme_matchups.py <dl_dir> .` (nflverse participation + FTN charting 2023-25, log `scheme_matchups_log.txt`).
Each player's yards per target/carry against each defensive look (man/zone, one-/two-high safeties, base/nickel/dime,
blitz; for runs light/even/stacked box and DB count), shrunk, weighted by how often tonight's defense shows each look.
- Splits barely repeat year to year: man-vs-zone r 0.15 (47 WR/TE/RB), one- vs two-high 0.13; blitz vs no blitz 0.46
  and RB base vs nickel 0.40 look real but on only 22-31 players.
- Matchup multipliers are small (10-90% range about +/-1%) and added nothing to 2025 projections (rec yds MAE
  18.888 -> 18.902 with all four; rush yds worse), coefficients the wrong sign.
- Not in the free data: where a player lines up (slot/wide/inline/backfield), safety depth beyond the coverage label,
  and 2026 participation (nflverse publishes it after the season; FTN box/blitz/motion is current).

### Follow-up: scheme -> target share (`scheme_share.py`, log `scheme_share_log.txt`)
J's point: scheme decides who gets the ball. Targets per route by coverage look, from on-field participation:
- **Receivers' target-share splits are a real, repeatable trait**: man-vs-zone gap r 0.48 year to year (334 players),
  one- vs two-high 0.49, base vs nickel 0.77 (58), blitz vs none 0.39.
- **Offense (coordinator) splits are not**: yards per dropback vs man minus vs zone r -0.19 across 32 teams; others -0.09 to -0.16.
- Weighted by tonight's defense mix the effect is about +/-2% of targets (defenses' mixes are similar). Man/zone and
  shell multipliers come out with the right size (coefficient ~1, p 0.03-0.05) but 2025 MAE does not move
  (targets 1.721 -> 1.720, rec yds unchanged). Real but too small to matter for a single game; not adopted.

## NFL-4 scheme and play-caller layer (2026-10-11)
J: caller priors first, but model all four scheme options and a way to use them together. Pre-change copy:
`archive/pre-nfl4-2026-10-11/`.
- `data/play_callers.csv` now covers 2016-2026 (2016-2022 added from coaching records; mid-season changes in `note`;
  primary caller = most games). 2016-2025: 50 caller moves. Personnel (11 r 0.57 vs team-last-year 0.32), play action
  (0.52 vs -0.03) and screens (0.56 vs 0.06) follow the caller to his new team; pass rate over expected stays with the team
  (0.09 vs 0.45); position target shares only weakly follow the caller (WR 0.24 vs 0.16). `scheme_callers.py <dl> . 2016`.
- `scheme_features.py <dl> .` builds `data/scheme_team|opp|players.parquet` + `scheme_meta.json` from participation
  (every feature from EARLIER seasons, because the current season's participation file appears only after it ends):
  option 1 caller priors, option 2 coverage/box matchups, option 3 expected personnel and game plan, option 4 red zone.
  `add_scheme(df, M)` attaches them; `project.py` calls it automatically.
- `backtest_nfl4.py . round1` (each option in every opportunity model; all four together) and `round2` (targets model
  only). Round 1: caller priors helped the games they are about; coverage/box features made carries and rushing yards
  worse (rush yds t -2.8); all four together overfit (WR/TE targets -0.2%, RB rush yds t -2.2).
- ADOPTED (round 2 best): options 1 + 3 + the coverage-mix target ratios in the TARGETS model only
  (`model.NFL4_TARGETS`). Out of sample 2024-25, targets / receptions error:
  new play-caller teams games 1-6 -1.0% (t 3.1) / -0.6% (t 2.0); callers who moved teams -1.5% (t 3.6) / -1.1% (t 2.8);
  players whose expected personnel shifts their snaps 3%+ -1.2% (t 2.5); receivers facing an unusual coverage mix -0.4%
  (t 1.2); all WR/TE unchanged (+0.05%). Carries, efficiency and the TD model are unchanged: red-zone terms (option 4) in
  the TD GLM had coefficients near 0 (p 0.36, 0.88).
- Each new season: add its rows to `play_callers.csv`, download the finished season's `pbp_participation_YYYY.parquet`,
  rerun `scheme_features.py`, then delete `data/model.pkl` (or let `project.py` refit).
