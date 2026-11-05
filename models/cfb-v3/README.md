# CFB v3 (college football sides and totals, rebuilt 2026-10-09)

J asked for the CFB model to be rebuilt from what the project has. J's v2 (CFBD PPA ridge) lives on J's work PC and
CFBD's API is blocked from the cloud, so v3 rebuilds the same idea from ESPN play-by-play EPA (sportsdataverse).

## How it works (`build.py`)
Every week, walk-forward, three ridge regressions toward a preseason prior (last season's final rating x 0.6 + returning
offensive production): margin ratings, opponent-adjusted non-garbage EPA/play (offense and defense), points scored/allowed.
Last season's games count 15%. A stack (fit 2014-2021, no 2020) turns them into a margin (separate fits for weeks 1-3
and 4+) and a total. Holdout 2022-2025.

## Results (holdout 2022-2025, 3,388 games with CFBD opening and closing lines) — `run_log.txt`, `holdout_bets.csv`
| | opener | close | v3 |
|---|---|---|---|
| margin MAE | 12.17 | 12.04 | 12.69 |
| weeks 1-3 | 12.38 | 12.25 | 13.45 |
| weeks 4+ | 12.10 | 11.97 | 12.44 |
| total MAE | 12.62 | 12.48 | 12.93 |

For comparison: J's v2 holdout margin MAE was 13.76 vs the close's 11.90; the public team-strength model's weeks 4+ ratings MAE was 13.07.
Betting: 2+ point gaps at the opener went 51.0% ATS (2,064 bets), +0.21 pts CLV (t 4.3); 51% at the close. Totals: no edge.
So v3 is the project's most accurate CFB model but not a standalone betting edge; the app treats only 3% of a spread gap
and 29% of a total gap at the opener as real (fit on 2014-2021).

## CFB-2 roster inputs (2026-10-10, now in the model)
`roster_prior.py` follows every player to wherever he plays next (ESPN player box scores), so transfers count:
starting QB's last-season adjusted net yards per attempt (any team, shrunk, scaled by his old team's Elo), whether he is
a transfer or has no passing history, and last season's skill-position yards on this season's roster (all of it, and the
transfer part). `build.py --roster roster_prior.csv` adds the home-minus-away values to the margin stack.

| Holdout 2022-2025 | v3 before | v3 + roster |
|---|---|---|
| margin MAE, all (opener 12.17) | 12.69 | 12.61 |
| weeks 1-3 (opener 12.38) | 13.45 | 13.15 |
| 2+ pt gaps vs opener, ATS | 51.0% (2,064) | 52.6% (2,197) |
| closing-line value of those bets | +0.21 pts (t 4.3) | +0.30 pts (t 6.6) |
| weeks 1-4 bets: ATS / CLV | 51.4% / -0.10 | 52.5% / +0.14 |
| share of the opener gap that is real (b_open) | 3% | 9% |

Still short of the 52.4% break-even with margin to spare, but the closest the rebuilt CFB model has been. Totals:
roster terms made them worse (12.93 -> 12.97), so totals are unchanged. Transfers credited by the model grew from 1% of
skill yards (2014) to 21% (2026); 2026 transfer volumes exceed the fitted range, so `export_app.py` caps them at the
tested range and folds the roster terms into each team's margin rating (the app's formula is unchanged).

## Season adjustments (2026-10-26, J's picks 1, 2 and 5 from the season-adjustments review)
- Market rating: each team's rating learned walk-forward from its past closing lines (`common.margin_ratings`, k 0.5,
  carry 0.85, home 3.0) is a stack input (`mk`). The stacks are now fit only on games with a line (FCS games without lines
  had drowned the market term). Holdout margin MAE 12.60 -> 12.33 (fitting on lined games alone: 12.49); weeks 1-3 12.84;
  2+ pt opener gaps 53.4% ATS (1,914 bets), CLV +0.29. 2026 weeks 1-8 margin MAE 12.65 (ESPN line 11.52).
- Win chance: the spread of misses is measured on lined games (sd 15.6, was 17-19 on all games), which fixes the
  overstated underdog chances. A fitted curve was checked and logged but not adopted (holdout log loss 0.4850 vs 0.4847).
- Totals mismatch term: |model margin| and margin^2/100 in the total stack (totals ran low in 24+ pt mismatches every season
  since 2018). Holdout total MAE 12.89; 2026 12.75, 24+ pt game bias +0.5.
- `export_app.py` folds the market rating into each team's exported rating and the home term, so the app formula is
  unchanged (checked against the direct stack formula to the hundredth of a point).

## CFB-3 tempo totals check (2026-10-10, not added)
`cfb3_tempo_totals.py`: expected plays x points per play for both offenses. Holdout total MAE 12.90 vs v3 12.93 and
the opener 12.64; combined 12.89. Its gap to the opener has a little signal (coef 0.17, p 0.03) but no closing-line
value (-0.03 pts), so it is not built in.

## Weekly refresh
```
curl -L -o pbp_2026.parquet https://github.com/sportsdataverse/sportsdataverse-data/releases/download/espn_cfb_pbp/play_by_play_2026.parquet
curl -L -o sched_2026.csv https://raw.githubusercontent.com/sportsdataverse/cfbfastR-data/main/schedules/csv/cfb_schedules_2026.csv
D=../team-strength/data; python3 build.py $D/cfb_games.parquet $D/cfb_team_games_epa.csv $D/cfb_returning.csv . pbp_2026.parquet sched_2026.csv --roster roster_prior.csv
# roster_prior.csv only changes before a season: python3 roster_prior.py <dir with pb_YYYY.parquet from espn_cfb_player_box> $D/cfb_games.parquet roster_prior.csv
python3 export_app.py      # -> ../../app/data/cfb_model.json, then republish Bet Desk
```
## Not in v3
Recruiting rankings and CFBD PPA (API blocked from the cloud, even with a key); 2026 betting lines (cfbfastR's betting
file stops at 2025; ESPN's spread is in the 2026 play-by-play and logged in run_log as a check); injuries/QB changes
(enter them as a margin adjustment in the app).
