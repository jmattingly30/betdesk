# Tennis Model A + Model B (J 2026-11-05 rollout; fifth sport on the two-model system after CBB, NFL, NBA, CFB)
Paper only. Model A = projected numbers, graded on J's yardstick (closing total-games and games-handicap lines). Model B = bets
built on Model A (edge, confidence tier A bet / B lean / C pass). The win model itself (Elo, surface Elo, games-won rating, form)
stays in models/tennis-golf-research/daily.py (tennis thread) -> app/data/tennis_model.json.

- lib.py: exact point -> game -> set -> match Markov engine (from research/model-a-sim-2026-10/tennis-pbp/markov.py) tracking
  both players' games. Tour-wide pressure shifts only (men break point -.036, game point +.032; women deuce +.037), match-day
  form sd .35 logit. Grids grid_bo{3,5}_{m,w}.npz (built in ~2 min if missing). project(p, mu) finds the serve gap that gives our
  win % p at serve level mu, then reads total games, each player's games, margin, set scores.
- prep.py: walk-forward Elo / surface Elo / games-won rating / scoring style on Kaggle alimoh89 (2014-25, bet365 open+close ML,
  totals, handicaps; download with KAGGLE_API_TOKEN). model_a_test.py -> model_a_test.json + params.json.
- MODEL A (2021-25 OOS, 49,382 matches with a bet365 closing total): total games miss 4.86 vs closing line 4.84 (APE 20.4% vs
  20.5%); favourite's games won 2.45 vs line-implied 2.41. The SAME engine fed the closing moneyline misses 4.83 (beats the
  line) -> the engine is right; the gap is our win %. Straight sets 63.0% predicted vs 63.7% real (ATP 60.4 vs 60.2, WTA 65.7
  vs 65.9). Serve level from tour serve-point rates (fitting it to mean totals was ill-posed and worse: 4.875).
  Style: tiebreak-set rate helps a little (4.864 -> 4.856); games-per-set term fit to 0 (net zero, dropped).
- model_b_test.py -> model_b_test.json (weights fit on earlier years only, test 2022-25):
  Moneyline at the bet365 opener, blend weight on our model: men's Challenger .04-.07, ATP .10-.14, women's ITF .20-.23, WTA .2-.33.
  Women's ITF 6%+: +2.3% ROI (465, se 4.9), CLV +3.9%; 6-10% about +7.7% (324); 10%+ -10% (141); by year -12, -13, +11, +15.
  WTA 6%+ +2.4% (215). Men: almost no 3%+ edges (model weight too small to clear the vig).
  Totals at the close: model alone log loss .6952 vs book .6924 vs coin .6931 -> no edge at any size. NO TOTALS BETS.
  Handicap at the close: blend 'ROI' +3% comes from the book's own price (weight 2.5, ours ~0): Kaggle closing handicap prices look
  stale vs their line, so that ROI is a data artifact, not our model. NO HANDICAP BETS.
- TIERS (live.py): no tier A anywhere (nothing holds both 2022-25 and the 2026 walk-forward in [[tennis-model-2026]]).
  Tier B lean = women (ITF or WTA), blended edge 6-10% at the best Colorado price, both players 30+ rated matches, price 1.2-5.
  Everything else C. Market reference = Pinnacle no-vig (else bet365).
- live.py -> app/data/tennis_ab.json + history/runs.jsonl. Run after tennis_model.json is rebuilt (16:30 UTC) and after each
  tennis odds pull: noon + night pulls (odds_pulls.md 4f). grade.py -> history/grade.json (night pull).
NET NEGATIVES / NOT USED: fitted serve level from mean totals; games-per-set style term; totals and handicap bets; edges 10%+
(market knows something); men's moneyline leans. NOT BUILT: aces / double faults / breaks props (no prop lines in our feed);
live serve stats (no free 2025-26 source).
