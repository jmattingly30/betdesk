# Game win-probability tool

Type a team, see today's game (opponent, venue, start time, probable starters, posted lineups), **confirm it**,
and get projected runs plus four win probabilities, each with a no-vig fair moneyline. Every run is saved to a
play log, where you record what you actually bet.

## Model v2 (2026-10-09) - now the primary model
The 2026-10-09 audit found the sheet chain projects **fewer** runs for a better lineup (it divides by tonight's lineup
QPA), uses raw stats with no regression, lets the starter pitch the whole game, has no home edge, and adds wRC+/FRV
in the wrong units. Model v2 (`src/projection_v2.js`, coefficients in `src/config.js` -> `v2`) replaces the run formula:
every hitter and starter regressed toward league average (this season + last season at 0.6 weight), park-neutral
player rates, starter 58% / bullpen 42%, starter K-BB%, home edge, negative-binomial runs, a fitted calibration, and
once market odds are on a log entry a **"Model v2 + market"** blend that sets the pick (`primaryModel: 'v2mkt'`).

Backtest on 9,991 games 2021-2025 (both versions rebuilt on Retrosheet so they see the same games; Sports Analytics
project `models/mlb-app/backtest_chain.py`): log loss sheet chain 0.6831, v2 0.6762, opening line 0.6736, closing
line 0.6674. Correlation with the closing line 0.73 -> 0.86. On 2023-2025, v2-plus-market picks (2+ points vs the
opener) beat the close by +2.8 points on average, but only 111 of them, so treat it as unproven.

The sheet models still run next to v2 for comparison. If one of their inputs is missing (for example a hitter with
no stats), they are skipped with a note and v2 runs anyway. The original version of this whole folder is archived
in the Sports Analytics project at `archive/pre-audit-2026-10-09/sources/baseball-win-tool/`.

## Model v3 (2026-10-10) - testing next to v2, v2 stays primary
Model v3 is v2 with four more things in the run formula, all refit together on Retrosheet 2015-2025: tonight's
lineup strikeout rate, walk rate and power (ISO), each hitter regressed toward league average (60 PA for K%, 120 for
BB%, 160 AB for ISO, last season at 0.6 weight), and the game-time temperature (+3.6% runs per 10F). The wOBA/xwOBA
lineup, starter, bullpen, park and home terms are the same as v2. Pythagorean (1.83) and a Monte Carlo also run on
v3's projected runs, and v3 shows the chance of over/under each total from 6.5 to 11.5 with fair odds.

Backtest 2021-2025 (9,991 games, `models/mlb-app/backtest_v3.py`): win-prob log loss v2 0.6762, v3 0.6763 (no change
on the moneyline); total-runs error 3.489 -> 3.470; how much v3's disagreement with the opening total predicts the
final total roughly doubles (0.22 -> 0.43, p < 0.001), but nothing beyond the closing total. When v3 said 55%+ for a
side of the opening total in 2023-2025, that side won 54.9% of 1,957 games (+4.9% at -110); v2 52.6%. Not yet
statistically proven, so it is logged (`v3_*` columns) to be graded live. The over/under chances are shrunk toward
50% (slope 0.50, fit on 2021-22 openers) because the raw distribution was overconfident against a real line.

New data the app pulls: Savant batter columns `ab, k_percent, bb_percent, isolated_power` (new cache names) and the
MLB schedule's `weather`. Without a posted temperature v3 uses 73.5F; without ISO it drops that term (both say so).
Pre-v3 copy of this folder: `archive/pre-mlb-v3-2026-10-10/`.

### Redeploy the phone app (do this after any update)

Paste the new `apps_script/Bundle.gs`, `apps_script/Index.html` and
`apps_script/Server.gs` into the Apps Script editor, save, then Deploy -> Manage deployments -> pencil -> New version.
No need to re-run `setup`. Without Windows, `node build_apps_script.js` rebuilds them, `node test/run_gas_test.js`
and `node test/run_gas_ui.js` run the phone-version tests (the UI one needs Playwright).

## Run it on a computer
1. Install Node.js LTS (https://nodejs.org). No other packages are needed.
2. In this folder: `npm test` (expect all tests to pass), then `npm start` and open http://127.0.0.1:3000.
   The local version only answers on this computer (127.0.0.1).
3. Optional: set `ODDS_API_KEY` (free key from https://the-odds-api.com, 500 credits/month) before `npm start` to
   list every book's odds at run time and capture closing lines automatically. Without it, type odds in yourself.
   The Odds API is a licensed feed; the tool does not scrape any sportsbook or odds site.

What has NOT run for real yet: the computer version's `server.js` / `src/io.js` under Node (syntax-checked only;
no Node on the machine this was written on), and the phone version on Google's actual servers (tested end to end
on stand-ins for Google's services, see Tests). The odds parser is tested on a sample built from The Odds API's
documented format; `checkOddsFeed` (setup step 5) is the first check against the live feed.

## The models (all use the same projected runs)
| Model | What it computes |
|---|---|
| Hierarchical Monte Carlo | Projected runs x park factor (lambda), plus a Normal error (mean -0.58, std 3.03; a 2.2 line is printed as a sensitivity check), then Poisson runs; 200,000 simulations, ties excluded |
| Pythagorean | Your sheet's "Probability of Win": score^1.83 / (score^1.83 + opp^1.83) on the 162-game scores (Tester C36/K36). A park factor would cancel, so none is used |
| Pythagenpat, projected | Exponent = (lambda away + lambda home)^0.287 (David Smyth's constant), applied to the same runs |
| Pythagenpat, season R/G | Exponent = (away R/G + home R/G)^0.287 (like your Tester K44, with 0.287) |

Fair moneyline: p > 50% gives -100p/(1-p); p < 50% gives +100(1-p)/p; exactly 50% gives +100 (Tester H53).

**Backtest verdict** (`mlb-backtest/step3_methods_report.md`): Pythagorean, both Pythagenpats and the MC at std 2.2
are statistically tied; the MC at 3.03 is very slightly worse (too cautious); the closing line beats all of them.
Model v2 + market is now the log's primary model (`primaryModel` in `src/config.js`); this verdict was about the sheet models.

## The play log
One row per game run, played or not. "Log" tab: summaries, edit, delete, **Download CSV**.

| Filled in | Fields |
|---|---|
| Automatically at run time | date, matchup, venue, starting pitchers, which inputs (FanGraphs or not), projected runs (sheet C37/K37 and after park factor), every model's win probability and fair moneylines; with an odds feed, every book's price (listed for shopping) and the best price on each side |
| By you | market odds you found shopping (optional, both sides - gives each model's pick and edge); played or passed; side; odds you took (American); stake in units (optional; P/L assumes 1u when blank); sportsbook (optional); notes (optional); closing line (optional, if there's no feed) |
| Automatically | each model's pick and edge (percentage points and cents) whenever market odds are on the entry; with a feed, the closing line (every book, captured once within 10 minutes of first pitch; CLV uses the average no-vig probability across books); final score and result (MLB Stats API); P/L in units; CLV = closing no-vig probability of your side minus the implied probability of your odds (also as expected value %), and the same for the model's pick |

With the odds feed, the run screen lists every book; tap a price to fill in your side, odds and book.

A model "picks" a side when its probability beats the price's vig-included implied probability by 2+ points
(`pickThresholdPp`). Summaries: your record/units/ROI/CLV for games played; the model's pick on games you passed
(1u at the price when you ran it); each model's own picks; your plays by edge size (primary model, in cents);
average CLV. Re-running a game adds a row; model and passed summaries use the latest run of each game.

Local storage: `data/playlog.json` (the previous version is kept as `data/playlog.previous.json`). Back it up by
copying that file or with Download CSV.

## Where each number comes from
| Data | Source | How |
|---|---|---|
| Game, venue, start time, probable starters, lineups, final scores | MLB Stats API | automatic |
| Starter wOBA / xwOBA against, PA (TBF), IP | Baseball Savant | automatic (CSV) |
| Team games played, PA, runs | Baseball-Reference | automatic (cached 6 h; keep to a few runs/hour, the site rate-limits) |
| Batters: PA, wOBA, xwOBA | Baseball Savant (default) or FanGraphs | automatic, or **you export a CSV** - see `batterSource` |
| Bullpen IP, TBF and wOBA against (relief split) | FanGraphs | **you export a CSV** |
| Team wRC+, team FRV | FanGraphs | **you export a CSV** |
| Market and closing odds (optional) | The Odds API | automatic with `ODDS_API_KEY` |

**No-FanGraphs mode** (switch on the Run screen, remembered): no bullpen term, wRC+ 100, FRV 0 - the setup the
full-season backtest validated - so no exports are needed at all. With FanGraphs on, upload the three exports on
the Log tab ("FanGraphs files"; phone or computer), or on the computer version put them in `data/fangraphs/`.
FanGraphs sits behind a Cloudflare bot check; the tool does not try to get around it. Files, named `<kind>_<season>.csv`:
- `bullpen_2026.csv` : Team, IP, TBF, wOBA (split = Reliever, team level)
- `team_wrc_2026.csv` : Team, wRC+
- `team_frv_2026.csv` : Team, FRV
- `batters_2026.csv` : Name, Team, PA, wOBA, xwOBA (+ MLBAMID) - only if `batterSource: 'fangraphs-csv'`

When a needed file is missing, the tool lists every missing one at once and refuses to run rather than guess.

## Which seasons of stats (`src/config.js`)
- Current season has >= 100 games (the smaller of the two teams') -> current season only.
- Fewer -> current season-to-date blended with the full prior season.

## Which games
Regular season and the postseason (Wild Card, Division Series, LCS, World Series; `gameTypes` in `src/config.js`).
Postseason games are flagged before you confirm and use regular-season stats. Spring training, exhibitions and the
All-Star Game are never run. Each log entry records the game type.

## What it refuses to do
It will not run unless both lineups (9 names) and both probable starters are in, and every stat it needs is
found. It prints exactly what is missing. Probable starters can still change before first pitch.

## Projection formulas
`src/projection.js` is the Tester sheet cell for cell (result `cells` uses the sheet's cell names).
Corrections from the sheet as received: bullpen weights `E17`->`F17` in M2/K19/H18; P18 BF-weighted like H18;
C33/K33 use FRV/G x 162. R/QPA divides by tonight's lineup QPA by design (`rqpaBasis: 'tonight'`).

## Phone version (Google Apps Script) - setup, once, about 10 minutes, any computer's browser
**Already set up (2026-09-28):** Apps Script project "MLB Win Probability", web app (Execute as Me, Only myself):
https://script.google.com/macros/s/AKfycbwBKDERx4JtaUZNk9F7xSf8maDWHO3wKu-d_cxURvQtyqRIHmjVK2qxETUR_EIU9Kk/exec
- log: the "MLB Play Log" Sheet in your Drive;
- timer: every 5 minutes;
- odds key: in Script Properties;
- odds feed books: 15 US books (Bet365 and Caesars aren't in it - type those in).

The steps below are for setting it up again from scratch. Tip for step 2: the left menu expands when the mouse
is over it and covers the checkbox - move the mouse to the right first.

The phone version is the same page and the same calculation code (`apps_script/Bundle.gs` is generated from
`src/`), running on Google's servers under your account. Free. Only you can open it (Google sign-in).

1. Go to https://script.google.com -> **New project**. Rename it (top left), e.g. "MLB win probability".
2. **Project Settings** (gear, left) -> tick **Show "appsscript.json" manifest file in editor**.
3. **Editor** (`<>`, left). Replace the contents of `appsscript.json` with `apps_script/appsscript.json`.
   Replace `Code.gs` with `apps_script/Server.gs` (rename the file to `Server`). Add a script file (+ -> Script)
   named `Bundle` with `apps_script/Bundle.gs`. Add an HTML file (+ -> HTML) named `Index` with
   `apps_script/Index.html`. Save (Ctrl+S).
4. Pick `setup` in the function menu at the top and click **Run**. Google asks for permission: it lists
   spreadsheets (the log), connecting to external services (MLB, Savant, Baseball-Reference, odds) and running on
   a timer. Allow it. The log shows the link to a new Sheet, "MLB Play Log", in your Drive.
5. Optional odds feed: **Project Settings** -> **Script Properties** -> Add property `ODDS_API_KEY` = your key
   from the-odds-api.com. Then run `checkOddsFeed` from the editor: it should print PASS with the list of books.
   Without a key you type odds in yourself; everything else works the same.
6. **Deploy** -> **New deployment** -> gear -> **Web app**. Execute as: **Me**. Who has access: **Only myself**.
   Deploy. Copy the **Web app URL** - that's your link.
7. On your phone, open the link (signed in to the same Google account).
   iPhone (Safari): Share -> **Add to Home Screen**. Android (Chrome): menu -> **Add to Home screen**.

**Privacy and keys:** "Only myself" means Google refuses anyone who isn't signed in as you. The odds key sits in
Script Properties on Google's side and is never sent to the page. The log is an ordinary Sheet in your Drive:
Drive keeps its version history, and Download CSV (Log tab) or File -> Download in Sheets gives you a copy.

**Updating later:** replace the changed files in the editor (rebuild `Bundle.gs` / `Index.html` first with
`build_apps_script.ps1` if `src/` or `public/index.html` changed), save, then **Deploy -> Manage deployments ->
pencil -> Version: New version -> Deploy**. The link stays the same. Re-run `setup` only if told to (it is safe to
re-run; it never deletes the log).

**If a data source blocks Google's servers,** runs stop with a plain message naming the site; nothing is guessed.
`deploy/test_sources.gs` checks every source from Google's servers in 2 minutes.

**Limits (free Google account):** 20,000 web requests/day and 90 minutes/day of timer run time - the 5-minute timer
uses a few seconds per run. Odds feed: each request costs 1 credit per region (`oddsRegions`, default `us,us2` = 2);
one request per game run plus one closing-line capture per game start. Credits left are shown by `checkOddsFeed`.

## Tests
- `npm test` (Node): the calculation code, the log, the odds parser, and end-to-end on recorded real data.
  Without Node (Windows): `powershell -File test\run_tests_in_edge.ps1` runs the same tests in headless Edge.
- `test/run_gas_test.ps1`: the Apps Script server on stand-ins for Google's services (Sheets, fetch, cache,
  timer), end to end on the same recorded data. `test/run_gas_ui.ps1`: the real page clicked through against it
  at phone width. Both run in headless Microsoft Edge (Windows) with no installs.
