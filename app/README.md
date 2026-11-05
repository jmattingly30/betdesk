# Bet Desk (J's private betting app)

Live: https://claude.ai/artifact/Pdq3m2UxoriBf2gBdzgBp4 (private to J; works on phone, add to home screen from the browser).
Built 2026-10-09 in thread "Prop edge tool, phone app and tracker" (J's plan items 2 and 3).

## Tabs
- **Props**: pick player + stat, type the line and both odds. Shows model P(over/under) (calibrated curve from the model,
  exact push handling on whole-number count lines), no-vig market (power method), blended chance (logit blend,
  model weight 50% by default: the 2023 NBA prop test found a 50/50 blend beat model and market alone), edge, EV,
  and a stake (0.25 Kelly on 100u bankroll, max 3u; overs need +2% extra EV per audit P3). Every line checked is saved
  to the `lines` collection for a future model-vs-market test.
- **Price**: any side/total/ML: model chance for side A + both odds. Team models default to 30% weight (none beat the close).
- **Tracker**: bets by sport (db collection `bets`), with units, odds, book, model/market/blended chance, open, close
  line and both closing odds, result. Auto: P/L, EV at bet, CLV (no-vig close at same line) and line points gained.
- **Charts**: actual vs expected units, ROI by EV bucket, calibration, CLV over time; tables by sport, market, model,
  side, stat, book and month.
- **Models**: what's loaded + staking settings (db doc `settings/main`).

## Refresh NFL props (weekly, after Friday injury reports)
```
python3 export_nfl_props.py <dl_dir> 2026 WEEK      # dl_dir needs games.csv + inj_2016..2026.parquet (models/nfl-player/fetch_data.py)
```
Writes `data/nfl_props.json`. Copy `bet-desk.html` to a scratch folder as `index.html`, then republish with the Artifact
tool to the URL above (`url` + `files: {"nfl_props.json": ...}`); bets and settings live in the artifact db and survive.
Rebuild `models/nfl-player/data/player_games.parquet` first (build_dataset.py) so last week's games count.

## NBA props (exporter added 2026-10-09)
```
python3 export_nba_props.py 2026-10-22 [data/nba_props.json] [--games overrides.csv]
```
Pulls ESPN's 2026-27 schedule and current rosters (sportsdataverse mirrors), works out offseason moves, and trims
bench minutes when a team's full roster projects well over 240. Spread/total come from the NBA team model unless
`--games` gives market numbers (`team,spread,total,out,q`); injuries are only what you list in `out`/`q`. Rookies
with no NBA games are skipped. Republish with `files: {"nba_props.json": ...}`. Version 3 shipped the opening-night
(2026-10-20) slate as a preview; rerun on game days.

## CBB game card (Price tab, added 2026-10-09)
`models/cbb/export_app.py --preseason` writes `data/cbb_ratings.json` (2026-27 prior). Once teams have about 8 games,
rebuild the ratings with the 2027 hoopR files (models/cbb README) and export without `--preseason`; repeat weekly.
The card ports `predict.py`: projected score, win chance, market-anchored cover/over at open or close weights, and
flags a 3+ point gap at the opener. Its spread/total "Use" sets model "CBB card" at 100% weight because the number
already blends model and line.

## Why this bet (added 2026-10-09)
Every bet has a reason field (`why`); Charts has a table by reason. See library/01-prediction/prop-edges-2026-10.md.

## NHL props (added 2026-10-09)
Daily, any game day (models/nhl-player/README.md):
```
python3 fetch_data.py <dl> 2027 2027; curl -L -o <dl>/sched_2027.parquet <nhl_schedule_2027 release URL>
python3 project.py <dl> ../../app/data/nhl_props.json 2
```
Republish with `files: {"nhl_props.json": ...}`. Only games that are the next game for both teams are projected
(rerun after the first night of a back-to-back). Goalie lines assume that goalie starts.

## Soccer (added 2026-10-09)
- Props tab, Soccer: pick the EPL match and enter the 1X2 and over/under 2.5 odds; team expected goals come from those
  odds (otherwise from the soccer model). Players from `soccer_players.json` (models/soccer-player/export_app.py).
- Price tab, "Soccer model": 1X2, totals, BTTS, handicaps for any big-five league match or any two clubs by Elo
  (`soccer_match.json`, models/soccer/export_app.py). Model weight capped at 10% in code (`SOC_CAP`): the market beat it.

## MLB props
Built (models/mlb-player). Not in the app: no 2026 stats source is reachable from the cloud. A playoff mode is possible if J uploads FanGraphs 2026 batting and pitching CSVs.

## Games tab (added 2026-10-09, version 4; the Price tab became Games)
Sport buttons: NFL, CFB, CBB, NBA, MLB, Soccer. Each card projects score, margin, total, win chance and (with a line)
cover/over chances; Use copies a market into the Price-it form below.
- NFL + NBA: `python3 export_games.py <nflverse games.csv>` -> `data/games_models.json` (NFL ratings rebuilt from the
  fresh schedule with team-strength/current_ratings.py in a temp folder, plus points O/D for totals; NBA team model with
  ESPN 2026-27 rosters, players-out buttons, predict.py ported). Rerun weekly (NFL) / after roster moves (NBA).
- CFB: models/cfb-v3 (rebuilt model, README there) -> `data/cfb_model.json`.
- MLB (2026-10-19 late, J): primary is J's efficiency model (models.eff). The card prefills eff runs; win chance = the app's own
  calibrated homeWinProb (typed runs: Pythagorean 1.83, then 0.1419 + 1.0974 x logit); run line NB alpha 0.2429 (raw grid);
  total = eff's calibrated pOver for that line, else NB + logit slope 0.1776. Weight = J's team-bet weight (not eff's 3% market
  blend). Sheet models shown as "Your original"; v3/v3d/v3f folded under "Comparison models". Falls back to v3 (NB 0.2556,
  calibration 0.0671/1.32, 15% cap) when the slate has no eff. v2 retired, never read.
- Rows priced against a line are already market-anchored, so their models end in " card" and get 100% weight.

## Version 5 (2026-10-10)
- Every line and odds box has a ± button (iPhone number keypads have no minus key).
- CFB card: v3 / v2 toggle. v2 is J's cfb_v2_model.py ported (checked against run_week on test data: same numbers).
  Weekly data: J runs models/cfb-v2/export_v2_ratings.py on their laptop, uploads cfb_v2.json to Drive;
  copy it to data/cfb_v2.json and republish (publish folder needs cfb_v2.json beside index.html).

## Version 10 (2026-10-11): simpler inputs, latest models
- Games tab: pick teams, type the book's spread/total. Each card shows the projected score, model spread and total, then
  one block per market (win, spread, total) with the model chance, odds boxes beside it (spreads/totals assume -110 when
  blank), blended edge, verdict, stake and a Log button. The old "Use -> Price it" step is gone; manual pricing is in a
  collapsed "Price any other bet by hand" section.
- Props: edge shows at assumed -110 until odds are typed (logged lines keep only typed odds).
- Data re-exported 2026-10-11: nfl_props.json (week 5, NFL-3 team-changer + NFL-4 scheme targets), nba_props.json
  (opening night, NBA-2 minutes), nhl_props.json (Oct 11-13). CBB/CFB from v8/v9.

## Version 11 (2026-10-11)
- NFL card: spread and moneyline chances come from real NFL final margins (`data/nfl_margin.json`, built by
  `models/team-strength/nfl/keynum.py`), with push chances on whole-number spreads and "wins by 3+/4+/7+/8+/11+/14+" chances.
  Rebuild once a season (or never; it changes slowly).
- NFL card: wind forecast box; 10-19 mph outdoors lowers the total 1.3 pts (nfl/totals_check.py).
- Game rows accept a push chance as the 7th field (`r[6]`), passed to priceSides.

## MLB full model feed (2026-10-17, from thread "MLB model re-audit")
Bet Desk cannot reach MLB Stats API / Savant / Baseball-Reference, and neither can the cloud. J's Apps Script app can, so it
is the engine: `exportSlate()` (timer `installSlateTimer`, every 30 min) writes `mlb_slate.json` to the Drive folder
"Bet Desk MLB". Bet Desk reads it with the `mcp` capability on J's Google Drive connector:
`search_files({query: "title = 'mlb_slate.json'"})` -> fileId -> `download_file_content({fileId})` (base64 JSON).
Shape (sample: data/mlb_slate_sample.json): `{updatedAt, date, pending, meta: {nbAlpha: {eff,v3,v3d,v3f}, totalsCalSlope (v3's), totalsCalSlopeByModel,
tieHomeWin, marketBlend (eff's), marketBlendByModel, primary: 'eff', loggingOnly: ['v3d','v3f']}, games: [{gamePk, startUtc, away, home, awaySP, homeSP,
status: ok|waiting|pending|missing|error, notes?, models: {eff (J's efficiency model, primary: + total, totals[], pythHomeWinProb), v3 (+ total, totals[] with pOver/pUnder per line), v3d, v3f,
sheetPyth, sheetMC} each {runsAway, runsHome, homeWinProb}, inputs: {away/home: {efficiency (eff model's), efficiencySource, seasonRpg, framing, catcher,
fielding, platoon, pairing, starterThrows}, tempF, parkFactor}, warnings}]}`.
2026-10-19 late (J): J's EFFICIENCY MODEL (models.eff) is primary and sets picks from its own calibrated homeWinProb (its market
blend gives the model only 0.03 weight, so it is logged, not used); v2 retired; v3/v3d/v3f comparison only. Games > MLB should
prefill eff runs, show J's sheet models (sheetPyth, sheetMC) beside it, and v3 / v3 + defense / v3 full folded away (v3d/v3f logging only). Typed runs stay as the fallback when the file is missing or stale.
Finals (2026-10-19): `finals: [{gamePk, date, away, home, awayRuns, homeRuns, pregame}]`, last 3 days. `pregame` is null when the
game was never run before first pitch, else `{madeAt, startUtc, lineupsPosted, awaySP, homeSP, models: {eff, v3, v3d, v3f}}` (entries saved before J's 2026-10-19 repaste may still carry v2; ignore it) with each
model `{runsAway, runsHome, homeWinProb}` (v3 adds `total`; v3d/v3f null when Savant defense data was missing). Totals odds: use
`meta.nbAlpha` and `meta.totalsCalSlope` as for the slate games. Grade MLB from `pregame`, not from the morning slate.
Built 2026-10-17 (Bet Desk v15): Games > MLB loads the newest mlb_slate.json on opening the tab (and on Refresh / return after 10 min);
stale = updatedAt over 2 h old or date before today. Bets logged from a feed game carry `mlb: {gamePk, slateAt, v2/v3/v3d/v3f: [runsAway, runsHome, homeWinProb], v3total, v3over}`.
