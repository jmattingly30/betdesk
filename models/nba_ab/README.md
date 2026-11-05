# NBA Model A + Model B (J 2026-11-05 rollout; third sport on the two-model system after CBB and NFL)
Pattern copied from models/cbb_ab and models/nfl_ab. Model A = projected numbers, graded on J's yardstick (closing team totals +
player prop closes). Model B = bets built on Model A (edge, confidence tier A bet / B lean / C pass).

- comp_test.py -> comp_test.json: research fix from model-b-confidence nba.json (our 3P% t -3.7 and FT rate t -2.6 overweighted,
  OREB / 3PA share underpriced). Walk-forward correction of each team's points from its predicted components x pace.
- model_a_test.py -> model_a_test.json: yardstick. Team MAE vs actual 2019-26: base 9.228, with fix 9.208 (7 of 8 seasons better),
  implied closing team totals 8.962. 2026 DK opener implied team totals 9.196: base 9.284, fix 9.253 (A closer 50.6%).
  Game totals 14.83 -> 14.79 (close 14.34). Sim (lib.py) 80% coverage team .79, total .80, margin .78; OT 5.1%. 'final' = live fix.
- model_b_test.py -> model_b_test.json: opener tiers out of sample. Needs /tmp/f_nba_open_sbr.parquet (run
  research/model-b-confidence-2026-11/nba.py first). Spreads, tier A = confidence top 10% same side, not early season:
  SBR 2021-22 56.9% (255, CLV +1.0), 2026 DK opener 61.9% (42, CLV +1.6); tier B (gap 3+) 48-52%; C 48-52%.
  Raw gap 5+ alone: 61% SBR but 53% DK 2026 -> not used for A. Early season (either team <= 5 games) gap 3+ 46% -> A waits.
  Totals at the opener: gap 4+ 56.1% (808, SBR 21-22), 63.2% (310, DK 2026; frame without lineup 62%, so not lineup hindsight).
  A slate-median guard made openers worse -> only used when the whole slate sits 3+ pts off (preseason level problem).
  NOTE: OddsPapi 'close' totals for NBA 2026 miss actual by 16.1 vs open 14.9 (close snapshot looks contaminated), so CLV
  there is unreliable; win rates are vs the opener.
- live.py -> app/data/nba_ab.json (+ history/runs.jsonl). Run after shadow.py nba (refreshes teamgames) and export_slates.py.
  Needs rebuild/basketball/live_future_nba.py returning components (tuple items 4-6, added 2026-11-05).
- grade.py -> history/grade.json: Model A (and base) vs actual and vs implied team totals; Model B by tier (first time seen).
NET NEGATIVES / NOT USED: slate total guard on normal slates (worse on both openers); raw gap 5+ as tier A (did not hold on 2026
DK); close-like lines (b .04 spreads, -.06 totals: every gap size lost) -> tier C; moneyline untested -> lean max.
NOT BUILT (data not live): player-level lineup+defender possession layer (research +0.06 APE, needs tracking matchups each game);
steady-minutes prop feature (needs minutes CV in nba_props.json export).
