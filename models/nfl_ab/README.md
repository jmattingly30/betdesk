# NFL Model A + Model B (J 2026-11-05 rollout; second sport on the two-model system after CBB)
Pattern copied from models/cbb_ab. Model A = projected numbers (graded on J's yardstick: closing team totals + player prop closes).
Model B = bets built on Model A (edge, confidence tier A bet / B lean / C pass).

- carry_test.py -> carry_test.json: season-start offense carry by play-caller / QB case (research "efficiency carries only if QB
  or caller stays"). NET NEGATIVE at every strength (team MAE +0.002 to +0.018 on 2018+, weeks 1-6 +0.007 to +0.038); NOT used.
- model_a_test.py -> model_a_test.json: key-number score distribution (normal x key-number weights fit 2010-17), calibrated 2018+
  (|margin| = 3: 14.4% predicted vs 15.0% actual; 7: 9.0 vs 8.6). Used by live.py for win %, ranges, push chances.
- open_test.py -> open_test.json: first NFL OPENER test (OddsPapi 2026, 106 games weeks 1-7, Pinnacle/DK/FD; op/us/lines_nfl.csv
  from models/oddspapi/lines.py nfl). DK opener spread gaps 3+: line moved our way 86%, CLV +4.2 pts, but covered 48.6% (35);
  consensus opener 3+: 55.6% (27). Totals lost (3+ open 39-44%, close 43-48%). Model A 2026 runs +1.1 pts/team over the market.
  Caveat: the backtest stack knew the actual starting QB; openers ~12 days out often did not.
- live.py -> app/data/nfl_ab.json (+ history/runs.jsonl). Run after shadow.py (Model A means) and export_slates.py (lines).
  Spreads: tier B (lean) only when the kickoff is 4+ days away, gap 3+, and the desk model agrees; never A until tracked.
  Totals: always C (slate total guard shown). ML: lean at most. Props: tier rules in PROPS (applied on the desk, redesign.js).
- grade.py [games.csv] -> history/grade.json: Model A vs actual and vs implied team totals; Model B by tier (first time seen).
