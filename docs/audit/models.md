# Model audit, Phase 2 (2026-11-13)

Scope: the live Model A (scores, player stats) and Model B (edges, tiers) for CBB, CFB, NFL and NBA. No model code was changed. Research into why Model A misses is Phase 3.

How it was checked: each sport's `history/runs.jsonl` re-graded from scratch against finals with paired samples (same games for ours and the line), `models/<sport>_ab/live.py` tier and share rules read, `app/data/scorecard.json` and `props_vs_close.json` read. Live samples run Nov 2 to Nov 12 for CBB and Nov 5 to Nov 12 for the others, so CFB, NFL and NBA game numbers are one week and should be read as early signals. Scheduled runs did not fire Nov 7 to 12 (see `betdesk.md` finding 1), so many picks in this window were logged at odd times.

Yardstick (J's rule): Model A is graded on error against actual results, with the closing line's error beside it. Model B is graded on win rate, shown edge and closing line value (CLV).

## Summary

1. The CBB grader has been grading numbers that were recomputed after the games were played. It reports a team-score miss of 7.55. The true pre-game miss is 8.83, against 8.34 for the line.
2. Model B shows edges the live results do not support. CBB tier A spreads show an average edge of +11.7% and have won 32 of 76. The market has moved against our tier A picks on both spreads and totals.
3. Model A is behind the line in all four sports live. The largest gaps are CBB margins and CFB totals.
4. NBA player projections run low on every stat, which is what creates most NBA under flags. Those flags have hit 40%.
5. Grading is done three different ways across the repo, so the same model gets different accuracy numbers depending on the file.

## Findings, ranked

Impact: H = wrong numbers shown or bets pointed the wrong way. M = misleading or fragile. L = small. Effort: S under an hour, M a few hours, L a day or more.

| # | Finding | Evidence | Impact | Effort | Suggested change (for approval) |
|---|---|---|---|---|---|
| 1 | **CBB grading uses post-game numbers.** `cbb_ab/live.py` logs every game in `cbb_v2.json` on every run, including games already played, and `grade.py` grades the last logged row. After a game the ratings include that game, so the logged score moves toward the result. NBA, CFB and NFL logs do not have this (no rows after game day). | 322 CBB games have a row before game day and a row after it; Model A changed in all 322, by 4.5 points per game on average. On 40 games matched to the scorecard: last logged row misses by 6.66 per team, first logged row by 8.89. Four spread picks first "reached tier A" after the game was over and all four graded as wins. | H | S | Do not log a game once it has started (the slate has the start time). Grade the last row before the start. Regrade the season. This is a measurement fix, not a model change. |
| 2 | **CBB Model B tier A is not beating the market.** Clean pre-game record: spreads 32 of 76 (42.1%) with shown edge +11.7%; totals 40 of 68 (58.8%) with shown edge +15.2%. CLV is negative on both: spreads -0.34 points (52 picks with a later market), totals -0.55 (56 picks). The totals win rate is above 50% but the market moved against those picks, so it is more likely luck than edge. | Re-grade of `runs.jsonl`. Shown edge comes from trusting up to 85% of our gap (`SHARE_CAP`, `conf_share`) on spreads and 50.5% on totals when a price is "fresh". | H | M | Until the clean record supports it: show tier A as a lean, and compute the shown edge from a share the live data supports. Show each tier's live record beside the edge. J's call. |
| 3 | **"Fresh" is measured on the wrong clock (leading suspect for finding 2).** The backtest that justified the trusted share used the first DraftKings line of the season's games. Live, a price counts as fresh when the one book with the best price posted it within 24 hours, even if the market has been open and bet into for a day. Slow books posting late look like openers. | Tier A spreads on a price under 6 hours old at that book: 16 of 46 (34.8%). 6 to 24 hours: 21 of 39. Backtest CLV at the true opener was +1.7 to +5.7 points; live CLV is negative. The backtest opener cannot be reached by a pipeline that pulls three times a day. | H | M | Define fresh as hours since any book first posted the game. Re-test on the logged history before changing live. One change, validated alone. |
| 4 | **Model A is behind the line live in every sport.** Paired, pre-game, same games. CBB (324 games): team 8.83 vs 8.34, margin 10.68 vs 9.64, total 13.93 vs 13.35, our totals 2.2 low on average. CFB (58): team 9.40 vs 8.92, total 14.12 vs 13.20. NBA (49): team 9.09 vs 8.83, total 12.73 vs 12.41. NFL (15): team 7.22 vs 6.53. The season scorecard agrees: CFB total 14.09 vs 12.91 (209 games), NBA total 15.35 vs 13.80 (156), CBB margin 10.39 vs 9.26 (417). | `runs.jsonl` re-grade; `scorecard.json`. Scorecard slopes: CBB 1.11 and NFL 1.12 (our margins are too compressed), NBA 0.88 (too wide). | H | Phase 3 | Research targets, in order: CBB margins and blowouts, CFB totals, NBA margin spread, CBB November total level. Tier A picks are chosen where we disagree most with the line, which is also where our error is largest. |
| 5 | **NBA player projections are low across the board.** Every stat has a negative bias (points -0.63, rebounds -0.41, points+rebounds+assists -1.26). When we said 46% chance of the over it went over 53%; when we said 53% it went over 61%. Under flags have hit 39.9% (1,618 flags) and over flags 46.0% (565). | `scorecard.json` props, calib, flags. | H | M | Find the source in Phase 3 (minutes, pace or scoring level this season) and fix that one input. NBA unders are already off the desk per J's earlier decision; this explains why they failed. |
| 6 | **Desk-flagged game edges are losing in three sports.** CBB 67 wins, 98 losses. CFB 8 and 28. NBA 21 and 28. NFL 4 and 4. Together 100 of 258 (38.8%). | `scorecard.json` games.edges. | H | with 2 and 4 | Same root as findings 2 to 4. No separate fix. |
| 7 | **Grading is inconsistent.** (a) CBB A+B grade has no line comparison at all. (b) NFL reports our miss on 30 team scores and the line's on 12. (c) CLV averages count a missing close as zero (CBB tier A spread shows -0.21; on picks that have a close it is -0.34). (d) The scorecard grades a desk snapshot while the A+B graders grade the run log, so CBB team miss reads 8.76 in one file and 7.55 in the other. | Graders read; numbers above. | M | M | One rule in all four graders: last row before the start, the same games for ours and the line, CLV only where a close exists with its count. |
| 8 | **Player prop bets rarely beat the close.** Share of flagged props that beat the closing price: NBA 24 to 36%, NFL 20 to 24%, NHL 11 to 17%, with negative CLV in every group. The file has not been refreshed since Oct 28. | `props_vs_close.json`. | M | S | Put the file back on the morning run so the number is current, then judge prop tiers on it. |
| 9 | **Quarterback volume and goalie saves are spread too wide.** Slope of actual on projected: completions 0.59, pass attempts 0.45, pass yards 0.73, NHL saves 0.28. A slope under 1 means our high projections come in lower and our low ones higher. | `scorecard.json` props. | M | S | Shrink these projections toward the league average by the measured slope, then re-check. Small samples (108 QB games), so re-measure first. |
| 10 | **Slate total guards differ by sport and stack on the model.** CBB shifts only when the slate median is 2.5 off; NBA at 3; NFL always subtracts the slate median; CFB uses its own rule. Project notes list slate-median guards as a tested dead end. | `live.py` in each folder. | M | S | Pick one rule or remove them, tested once on logged history. Conflicts with "no fix on top of a fix". |
| 11 | CBB and NBA spread and total edges ignore pushes on whole-number lines (CFB and NFL handle them). | `best_price()` in `cbb_ab/live.py`, `nba_ab/live.py`. | L | S | Use the CFB push handling. |
| 12 | Model B math notes carried from Phase 1 (consensus averages log-probabilities; moneyline anchor shifts in probability space). | `model-b/devig.py`, `ml_anchor.py`. | L | S | Test against history before changing. |

## Sound or encouraging
- NFL and NHL prop probabilities are well calibrated (said 53%, went 56%; said 62%, went 63% for NFL; NHL within 2 points in every band).
- NFL, NHL and NBA prop projections beat a season-average baseline on error in every stat.
- NBA Model B has not shown the CBB problem in its first week (spread leans 14 of 25, CLV about zero; total tier A 7 of 14, CLV +0.22). Too few to judge.
- NFL game tiers are capped at lean and CFB tier A needs a look-ahead gap, so those sports are not showing large game edges.

## Suggested order (nothing done yet)
1. Fix the CBB log and grader, and make all four graders follow one rule (findings 1 and 7). Everything else depends on trustworthy numbers.
2. Decide how the desk should show CBB tier A until the clean record supports it (finding 2).
3. Test the market-age definition of fresh on logged history (finding 3).
4. Phase 3 research takes findings 4 and 5: CBB margins, CFB totals, NBA prop level.
5. Small ones after that: 8, 9, 10, 11.
