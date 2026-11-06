# Watch the sim (J 2026-11-06)

Every game tile on the Bet Desk gets a **Watch the sim** button that replays a simulated game from our Model A:
possessions for NBA and CBB, drives for NFL and CFB, points for tennis. It is labeled as a simulation everywhere: one game the
simulator played, not a prediction of the real game.

## Pieces
- `build.py [NBA CBB NFL CFB TENNIS]` -> `app/data/sims_{sport}.json`. It reads the Model A numbers in `app/data/{sport}_ab.json`
  and plays games with each sport's own simulator (no new model):
  - NBA / CBB: possession outcome shares from `models/{nba,cbb}_ab/lib.py` (`_tilt`) at Model A's points per trip and pace,
    overtime on a tie. A rating / pace shock keeps the pool a bit wider than Model A before reweighting.
  - NFL: `models/rebuild/football/trace.py` (same rules and tables as the tested drive simulator, `sim.py`), scoring odds solved
    to Model A's means.
  - CFB: `models/cfb_ab/lib.py` `Sim` (the engine behind CFB Model A's ranges), recorded drive by drive through `rec=`.
    Recording draws no random numbers, so CFB Model A's own results are unchanged (checked: identical arrays with and without it).
  - Tennis: point by point with Model A's serve-point odds, the match-day form sd and pressure shifts from `models/tennis_ab/lib.py`.
    Skipped until `models/tennis_ab` and `tennis_ab.json` are in the repo.
  The team-sport replays are drawn from a pool of 800-3,000 games, weighted to Model A's margin and total spread (its 10th-90th
  ranges), so across replays the results match Model A. Each game carries an `audit` block (Model A vs the weighted pool) and the
  build log lists any game that misses Model A's win % by more than 3 standard errors, or its margin by 3+ points.
  Window: today (Denver) and tomorrow; NFL and CFB the next 6 days. Games that already started stay in the file for the day.
- `../../app/sim.js`: the viewer. Loaded only on the first tap (`app/sim_patch.py` adds the button and loader to the desk), so the desk
  page grows by about 1 KB. Three small adapters (possession, drive, point) turn a stored replay into one event list, and one
  renderer draws every sport with the desk's theme tokens: scoreboard, game-flow line, play / step / final, speed, replay picker, log.
- `test_sim.js [data dir]`: plays every stored replay back through the viewer's adapters and checks it ends on the stored final score
  (tennis: set scores), with every event inside the game clock.

## Sizes (2026-11-06 slate)
NBA 20 KB (11 games), CBB 120 KB (101), NFL 49 KB (15), CFB 184 KB (58), tennis 66 KB (60 matches); about 4-5x smaller gzipped.
Only the sport you tap is downloaded. Build time about 25 s for all five.

## Known limits
- NFL replays are drive by drive (the tested NFL simulator); a play-by-play engine would be new modeling work.
- The CFB drive simulator does not play college overtime: a tie after four quarters stays a tie in the replay.
- Extreme mismatches (FCS at a power team, Model A margin 45+) sit past the drive simulator's range: the replays run a few points
  short of Model A's margin there. The build log names them.
- Possessions show points only (0-3). The simulator does not model shot type, so the log does not invent it.
