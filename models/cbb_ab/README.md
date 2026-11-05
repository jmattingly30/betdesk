# CBB Model A + Model B (J 2026-11-02, CBB first on the two-model system)
Doc: https://claude.ai/artifact/KKp8zU6M1g8V8wrzG2xTC4
- lib.py: v2 history frame, end-game terms, possession sim (sim_game) anchored to Model A means.
- model_a_test.py -> model_a_test.json: end-game shift on mean totals (net negative, not used), v2 vs implied closing team totals.
- model_a_sim_test.py -> model_a_sim_test.json: sim shape vs normal on closing lines (no gain).
- model_b_test.py -> model_b_test.json: tier rules, month-CV on 2026 DK opener; 'live' = coefficients used by live.py.
  Needs /tmp/f_cbb_open.parquet: run research/model-b-confidence-2026-11/cbb.py first.
- live.py -> app/data/cbb_ab.json (+ history/runs.jsonl). Run after live_cbb.py and export_slates.py.
- grade.py -> history/grade.json (ESPN finals).
