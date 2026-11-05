# CFB Model A + Model B (J 2026-11-05 rollout; fourth sport on the two-model system after CBB, NFL, NBA)
Pattern copied from models/cbb_ab, nfl_ab, nba_ab. Model A = projected numbers, graded on J's yardstick (closing team totals).
Model B = bets on Model A (edge, tier A bet / B lean / C pass). Colorado has no college props: CFB props are Model A inputs only.

- prep.py -> data/te_feat.parquet, data/fit_feat.parquet (cfb_gap rebuild features), data/drives_2023_25.parquet (drive sim tables).
- lib.py: v4 means (same terms as cfb_gap/live/cfb_live.py) and the drive simulator (research/cfb-sim-2026-10/drive_sim.py).
- model_a_test.py -> model_a_test.json. 2022-25 lined games vs implied closing team totals: team MAE v3 9.18, v4 8.88, line 8.69
  (v4 closer than the line 47.8%; every season v4 beats v3 by 0.25-0.38). Weeks 1-3 v4 8.82 vs line 8.61.
  Drive sim on v4 means: exact team-score log -3.54 vs bell -3.83; 80% bands 82-85%; team-total over Brier .2563 vs bell .2559 (coin .25).
- keynum.py -> keynum.json: the sim makes too few 3- and 7-point margins (4.0% vs 9.8% at 3) -> NET NEGATIVE for margins; margins,
  covers and win % use normal x key-number weights (fit 2014-21): margin log -4.046 vs sim -4.128 vs normal -4.160; p3 8.9% vs 9.8% actual.
- model_b_test.py -> model_b_test.json (2022-25 OOS). Spreads at the OPENER, v4: 3+ 56.5% (1291), 4+ 58.2% (866; 2022 59, 2023 57,
  2024 52, 2025 63), 6+ 58.7%, 8+ 64.0% (200); CLV +0.68 pts at 4+. At the CLOSE: 4+ 55.0% (1005), 6+ 55.9%. Share of gap real b:
  open .44 (v3 .37), close .34. Totals: opener unders 4+ 58.0% (343; 3 of 4 seasons), overs 4+ 49.5%; closes ~51%.
  NET NEGATIVE: the research confidence model (mid favourite, week 8+, neutral, dog, home, engines agree) for v4: tier A 58.5% vs the
  raw top-10% gap (6.7+) 61.5%; dropping any one feature but 'agree' raised tier A -> tiers use the raw gap; the features are reasons only.
- usage_prep.py <pbp dir> [years] -> data/usage_games.parquet (player rush/rec/pass per team-game; run_live.sh refreshes 2026).
- absence_test.py -> absence_test.json (J 2026-11-05, Malachi Toney / Julian Sayin): starters missing from play-by-play, 2022-25.
  Elite QB (225+ yds/g) first game out: team scored 2.1 under Model A (125 team-games), the close already 1.9 under; other QB -0.7
  (close -2.0); elite skill (90+ scrimmage yds/g) -1.1 (close -1.2). OOS team MAE change ~0 (pooled +0.004, t 0.4; first-game -0.002):
  no measurable accuracy gain, but leaving it out would show fake Model B edges on the team missing its star once the market moves.
  Live sizes: elite QB -2.5, other QB -0.5, top skill player -1.0 team pts. Caveat: the test knew who sat; live depends on news.
- absences.json: manual outs (team, player, kind, until). Props-based outs (app/data/cfb_usage.json stars_out, within 36 h) added automatically.
- live.py -> app/data/cfb_ab.json (+ history/runs.jsonl). Run after run_live.sh (cfb_live.json), cfb_usage.py and export_slates.py.
  Model A = v4 + usage prior shift + missing starters; ranges from the drive sim; win %/covers from keynum. Model B: look-ahead (kickoff 4+
  days) spread gap 4+ = A, 3-4 = B; game-week gap 4+ = B; look-ahead under 4+ = B; tiers need a positive-EV CO price; ML lean max.
  Slate total guard only when the whole slate sits 3+ pts off (NBA rule).
- grade.py -> history/grade.json: Model A vs actual and vs implied team totals (v4 alone beside it, missing-star games split out);
  Model B by tier and early/late (first time seen).
- unit_ratings.py + unit_test.py -> unit_test.json: research unit-sum term (fin_sum -0.027 on v3) rebuilt with walk-forward drive ratings:
  on v4 it adds nothing (fin_sum +0.001, t 0.6; all three +0.004) -> NET NEGATIVE/neutral, not used (v4's stack already carries it).
NOT BUILT: play-level unit ratings (pass EPA; play table not kept, and the drive version showed v4 already has it);
opponent-adjusted tempo (r7, -0.015 total MAE, borderline); play-state engine inside the game sim; red-zone trait.
