# Bet Desk audit, Phase 1 (2026-11-13)

Scope: the site (`public/`, built from `app/bet-desk.html` + `app/redesign.js`), the exporters in `app/`, the pipeline in `tools/` and `.github/workflows/`, the Worker in `worker/`, and the shared grading path. Model quality is Phase 2 and is only noted here where it showed up.

How it was checked: failing workflow logs read; `export_slates.py`, `line_tools.py`, `check_data.py`, `model-b/devig.py`, `ml_anchor.py`, `unproven.py`, `pregame.py`, `cbb_ab/grade.py` and the desk's core odds, bet and storage functions read line by line; a headless phone and desktop walk of every tab, every sport and 12 game tiles (no page errors, no NaN/undefined text, no horizontal overflow); the CBB grader re-run on a copy against ESPN finals; repo history measured.

Impact: H = wrong numbers, lost data or the desk not updating. M = misleading or fragile. L = polish. Effort: S under an hour, M a few hours, L a day or more.

## Fixed in this phase

| # | Finding | Impact | Status |
|---|---|---|---|
| F1 | College basketball tournament slots listed as "TBD vs TBD" failed the data check, so every run was marked failed. | H | Fixed, PR #18 merged. `export_slates.py real_game()`. |
| F2 | ESPN swapped home and away on a finished neutral-site game (Saint Joseph's vs South Florida, 11-11). The slate kept the old line on the swapped teams, so it showed the wrong favourite, and `cbb_ab/grade.py` graded the game with the two scores crossed. Re-running the grader with team names found 4 more games this season graded crossed. | H | Fix in PR #19 (`merge_started()`, grader reads scores by team name, data check warns instead of failing on a post-start swap). |

## Open, ranked

| # | Finding | Evidence | Impact | Effort | Suggested fix |
|---|---|---|---|---|---|
| 1 | **Scheduled runs are not reliable.** No scheduled workflow of any kind ran between Nov 7 14:14 UTC and Nov 12 23:23 UTC. The only runs in that window were started by hand. On Nov 12 the two noon crons (18:00 and 19:00 UTC) arrived at 23:23 and 23:57 UTC, about five hours late. All workflows show as active, so this is GitHub dropping or delaying cron events, which it documents as possible on busy times such as the top and half of the hour. | Actions run list filtered to `schedule`. | H | M | Needs J's choice. (a) Recommended: trigger the workflows from outside GitHub on time (a Cloudflare cron on the existing Worker calling `workflow_dispatch`; needs one GitHub token saved as a Worker secret) and keep the GitHub crons as backup. (b) Cheaper, weaker: move crons off :00/:15/:30, let either daylight-saving cron run when the other has not produced a data commit in 3 hours, and show a "pipeline has not run in N hours" banner on the desk. |
| 2 | **The repo grows about 40 to 50 MB per morning run** and is already 1.06 GB packed. `models/rebuild/basketball/data/live/player_rates.parquet` alone is 314 MB of history across 10 versions; other rebuilt parquet and pickle files are recommitted daily. At this rate clones and pushes slow down within weeks and GitHub starts warning. | `git count-objects`, per-path history size. | H | M | Stop committing files the run rebuilds from scratch (`.gitignore` them and rebuild or cache them with `actions/cache`); keep only small state (`history/*.jsonl`, `*.json`). Check each file is truly rebuilt before untracking it. Shrinking existing history needs a history rewrite, which is J's call. |
| 3 | **Edge numbers on Home are far above live results.** Best bets shows +15% to +20% edges (for example a CBB total at +19.6%, NFL reception props at +15% to +19%). Live record so far: CBB Model B tier A spreads 42 of 91 (46.2%, CLV -0.20), tier A totals 34 of 67 (50.7%, CLV -0.44). | `cbb_ab/history/grade.json` after the F2 fix; Home screenshot. | H | Phase 2 | Calibration problem in the models, not a display bug. Listed here so it is not lost; handled in the model audit. Until then the desk could show the live record of each tier beside the edge. |
| 4 | Other graders may have the F2 crossing. `cfb_ab/grade.py`, `nba_ab/grade.py`, `nfl_ab/grade.py` and `models/grading/grade.py` were not re-run. NBA and NFL have true home teams, so the risk is CFB neutral-site games and the main scorecard's CBB rows. | Same pattern as F2 (scores read by home/away position). | M | S each | Read each grader's finals join in Phase 2 and apply the by-name check where a swap is possible. |
| 5 | **A failed save is silent.** `lsSave()` wraps the write in `try {} catch(e) {}`. If phone storage is full or blocked, logging a bet shows success and the bet is gone on reload. On the hosted site bets go to the account instead, so this mainly affects use without sign-in. | `bet-desk.html` `lsSave`. | M | S | Show a toast when the write throws. Change belongs to the thread that owns `bet-desk.html`. |
| 6 | **The phone downloads 11.7 MB on every open** (33 requests), most of it JSON: `player_results.json` 2.7 MB, `props_NFL.json` 2.6 MB, `past_props.json` 2.0 MB, `props_NHL.json` 1.9 MB, `props_NBA.json` 1.4 MB, `line_tools.json` 0.9 MB. On a weak signal this is the load time. | Headless load numbers. | M | M | Load the prop and history files only when their view is opened; confirm Cloudflare serves them compressed (JSON compresses about 8 to 1). |
| 7 | The site depends on two outside hosts at run time: Chart.js from cdnjs and team logos from ESPN's image server. If either is blocked or slow, charts or logos are missing. | Smoke test request log. | L | S | Ship `chart.umd.min.js` in `public/`; keep logos but make sure a missing logo leaves no broken image. |
| 8 | Silent fallbacks in the pipeline. `line_tools.py` drops to market-only when the headless model probe fails (prints one line, job stays green; Look ahead then goes empty). `export_slates.py` prints "failed" for a sport and carries on with that sport missing from the file. | Code read. | M | S | Count these as failed steps (write to the fail list) so the job turns red while still publishing what did refresh. |
| 9 | `tools/check_data.py` warnings are mostly noise: about 30 "game is not on the slate (stale?)" lines per run for games that have simply finished. Real warnings are hard to see. | Run logs. | L | S | Do not warn for games whose day is in the past. |
| 10 | Averages that include missing values. In the A+B graders `clv_pts` adds 0 for picks with no closing number, which pulls the average toward zero. | `cbb_ab/grade.py` (`clv or 0`). | L | S | Average only picks that have a closing number and report that count. Check the other three graders. |
| 11 | `devig.consensus()` averages log-probabilities although its docstring says log-odds; `ml_anchor.home_chance()` adds the curve difference in probability space, which is clipped at 1% and 99% and behaves poorly for heavy favourites. Small numeric effects. | Code read. | L | S | Phase 2: test a probit-space shift and a true log-odds average against history before changing. |
| 12 | `esc()` in the desk escapes `& < > "` but not `'`. Bet text and imported bets are escaped where used, so this is only a risk inside single-quoted attributes. | Code read. | L | S | Add `'` to the escape map. |
| 13 | Until J sets the Worker secrets, the Worker serves `public/` with no sign-in (by design, per README), and the repo is public. Anyone with the address can read every model file. | README, `worker/index.js`. | M | J | Already on J's list (SESSION_SECRET, OWNER_TOKEN). Noted for completeness. |
| 14 | Dependencies are unpinned (`requirements.txt` has bare names; Playwright is pinned, Python packages are not). A new pandas or lightgbm release can break a run or change a model output with no code change. | `requirements.txt`. | M | S | Pin to the versions the runner uses today (`pip freeze` from one green run). |

## Checked and found sound
- Odds math: American/decimal conversion, power devig (Python and browser versions agree), EV with pushes, Kelly, hold, middle staking, arbitrage and bonus-bet conversion formulas in `line_tools.py` and the desk.
- Bet tracker math: profit, promo handling, expected P/L, ROI.
- Worker access rules: guests are limited to `/guest/*`, icons and their own `/api` data (re-read of the route table only; the 11-06 review covered the rest).
- Front end: no script errors, no "NaN/undefined/null" text and no sideways overflow at phone size across every tab, every sport and 12 game tiles.

## Usability and capability notes (not yet built; for J to pick from)
1. Show each tier's live record next to its edge ("Model B · A, 46% on 91 this season") so the number can be judged at a glance. Highest value, small.
2. A visible "last updated" state for the whole pipeline on Home, turning amber or red when the data is older than expected (ties to finding 1).
3. NHL has no Games tab and MLB shows a placeholder panel. Hide MLB out of season and label NHL "props only" so the sport bar does not lead to dead ends.
4. A game with no lines shows an empty "Model B · Bets" box. Collapse it to one line.
5. Long team names are cut off in the model card inputs ("Texas A&M-Cor"); allow wrapping.
6. Bets: one-tap "settle from final score" for game bets, since finals are already in the slate file.
