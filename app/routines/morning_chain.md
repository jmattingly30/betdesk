# Bet Desk morning chain (one fresh session, daily 6:30 am Mountain)

J 2026-10-29: one short session on a smaller model does the morning odds pull, the NFL props build and the
props refresh in one run, so the desk is fresh by about 7:45 and the Bet Desk thread stays free for J.
It replaced three scheduled tasks: the morning odds pull, "Bet Desk NFL props build" (trig_016RWNGAevSphRweZfypFP11)
and "Bet Desk props refresh" (trig_01KpmS38qyg6gExqsVC1EEfo). The scheduled task's prompt only says "follow this file",
so the Bet Desk thread changes the steps here, never in the task.

Rules: work in /mnt/project-files/sports-analytics; run each Python step in its own Bash call starting with
`cd /mnt/project-files/sports-analytics` (never chain later steps after a cd). Never print or store API keys.
No model math changes, no MLB model changes. Today = the date in America/Denver.

## A. Start the odds sweep in the background (about 100 min with CBB in season; the key is free during it)
Setup first: `python3 -m pip install -q pyarrow pandas numpy scipy statsmodels scikit-learn lightgbm requests`.
A1. `cd /mnt/project-files/sports-analytics/app && python3 op_live.py NFL CFB NBA CBB NHL` with run_in_background (timeout 7200000).
    All 12 books, NFL/CFB/NBA/CBB/NHL game lines plus NFL/NBA/NHL props; it holds data/odds/oddspapi.lock. Took ~100 min on 11-01 (CFB + CBB).
A2. When A1 exits: `cd /mnt/project-files/sports-analytics/app && python3 op_live.py TENNIS` with run_in_background (timeout 3600000)
    -> app/data/odds/tennis.json (about 45 min on 11-01, 164 matches; its own call so a long team sweep can't cut it off). Start part C while it runs;
    it must finish before D (publish tennis.json). The tennis thread rebuilds tennis_model.json at 16:30 UTC from it.
Do part B while it runs. Never poll with sleep; you are woken when it exits.

## B. Projections (no odds needed), while the sweep runs
B0. `python3 app/past_props.py` FIRST, before any props file is replaced (last 3 days of NHL/NBA players). Publish past_props.json.
B1. NHL: models/nhl-player fetch_data.py /tmp/claude-0/nhldl 2027 2027; re-download sched_2027.parquet
    (curl -L -o /tmp/claude-0/nhldl/sched_2027.parquet https://github.com/sportsdataverse/sportsdataverse-data/releases/download/nhl_schedules/nhl_schedule_2027.parquet);
    project.py /tmp/claude-0/nhldl <json> 2; app/add_recent.py NHL <json>; copy to app/data/nhl_props.json.
B2. NBA: download every season 2015-2027 as /tmp/claude-0/nbadl/pb_YYYY.parquet from
    https://github.com/sportsdataverse/sportsdataverse-data/releases/download/espn_nba_player_boxscores/player_box_YYYY.parquet,
    then (cd models/nba-player && python3 build_dataset.py /tmp/claude-0/nbadl .); app/export_nba_props.py <today> <json>
    ("if he plays" projections); add_recent.py NBA (rerun once on FileNotFoundError; copy only after it succeeds) -> app/data/nba_props.json.
B3. NFL props, Tuesday to Sunday only (J 2026-10-28; skip Mondays): download nflverse games.csv
    (https://raw.githubusercontent.com/nflverse/nfldata/master/data/games.csv) to /tmp/claude-0/nfldl/games.csv; week = the week of
    the next 2026 regular-season game with no result; (cd models/nfl-player && python3 fetch_data.py /tmp/claude-0/nfldl 2026);
    (cd models/nfl-player && python3 build_dataset.py /tmp/claude-0/nfldl .); python3 app/export_nfl_props.py /tmp/claude-0/nfldl 2026 <week> /tmp/claude-0/nfl_props.json;
    python3 app/add_recent.py NFL /tmp/claude-0/nfl_props.json (rerun once on FileNotFoundError); copy to app/data/nfl_props.json.
    Spot-check a few teams' opponents against games.csv for that week.
B3a. Every day: python3 models/nfl-player/shadow.py (grading only; needs statsmodels).
B4. Every day: download games.csv fresh (as in B3) and python3 app/export_games.py /tmp/claude-0/nfldl/games.csv (about 2 min).
    Check its "starters changed from games.csv" and "NFL QB changes" lines look sane (a few teams, each under about 8 points).
B5. Every day (J 2026-11-02 CBB two-model rollout; was Mon/Thu, daily so every newly posted CBB game gets Model A): python3 models/rebuild/basketball/live_cbb.py (about 20 min, run_in_background) -> app/data/cbb_v2.json.
B6. Sundays (and any day pbp_2026.parquet gained a full week since the last rebuild): CFB v3 rebuild per the "Weekly refresh" block
    in models/cfb-v3/README.md (pbp_2026.parquet + cfb_schedules_2026.csv, build.py --roster roster_prior.csv, export_app.py) -> cfb_model.json.
    The CBB ratings rebuild (models/cbb/export_app.py) waits until ~8 games per team (late November).
B7. Google Drive: search_files for title = 'mlb_slate.json' (folder "Bet Desk MLB"), download_file_content, decode base64 to
    app/data/mlb_slate.json. Skip quietly if missing or its updatedAt is over 2 days old. Never publish it.
B8. Model A (J 2026-10-30 redesign, the rebuild shown beside the desk model on every game card): NBA lineup feed
    `cd /mnt/project-files/sports-analytics/models/rebuild/basketball && python3 live_lineup.py` (about 8 min, run_in_background),
    then `cd /mnt/project-files/sports-analytics/models/rebuild/shadow && DL=<scratchpad> python3 shadow.py nba nfl nhl cbb`
    (one sport at a time if it runs long). Writes shadow/history.json + report.json.

## C. After the sweep exits
C1. Check op_live's output: every sport with games printed "N games from OddsPapi". Note any "failed:" line.
C2. `python3 app/export_slates.py` (Odds API credits only for a sport whose sweep failed or is over 6 h old; CBB never before 2026-11-02).
    If a sport with games shows 0 lines, note it for the report.
C3. Credits: read the highest "used" and lowest "left" across app/data/odds/*.json (strings; convert), compare with yesterday in
    models/grading/odds_credits.json and append {date, used, left}. The noon pull spends 3 to 6 a day, so expect 0 to 15. If used rose
    more than 20 in a day or left is under 60, report the numbers and which sport printed "<SP> odds pulled".
C4. `python3 app/line_tools.py`.
C4d. CBB Model A + Model B (J 2026-11-02, after B5 and C2): `python3 models/cbb_ab/live.py` -> app/data/cbb_ab.json. It prints games, tiers and
    'teams without tier match'; report if it errors or prints 0 games with lines while CBB has lines.
    Then `python3 models/cbb_ab/grade.py` (ESPN finals -> models/cbb_ab/history/grade.json: Model A team-score miss, Model B win/ROI/CLV by tier).
C4a. `cd /mnt/project-files/sports-analytics/models/oddspapi && python3 cfb_usage.py` (J 2026-10-29: rebuilds app/data/cfb_usage.json, the CFB usage prior and missing-star flags, from this sweep's CFB props).
C4b. CFB Model A: `bash /mnt/project-files/sports-analytics/models/rebuild/football/cfb_gap/live/run_live.sh` (writes app/data/cfb_live.json).
C4c. `python3 app/export_model_a.py` (after B8 and C4b) -> app/data/model_a.json.
C5. Props files: for each app/data/odds/props_{NFL,NBA,NHL}.json made in the last 30 hours, count players and named players
    (ids look like "id1222502"); under half named is worth reporting.
C6. `python3 models/parlay/suggest.py`, then `python3 models/promo/promo.py`.
C7. `node models/grading/snap_games.js`, then `node models/grading/snap_flags.js`.
C8. `python3 models/grading/grade.py` (last).

## D. One publish
Copy app/bet-desk.html to <scratchpad>/betdesk/index.html (never edit bet-desk.html; it already holds the 2026-10-30 Concept E redesign, never rerun app/redesign_patch.py) and every file below that changed this run into
<scratchpad>/betdesk/ at its bare name: past_props.json, nhl_props.json, nba_props.json, nfl_props.json, games_models.json, cbb_v2.json,
cfb_model.json, game_slates.json, line_tools.json, props_NFL.json, props_NBA.json, props_NHL.json, parlay_suggestions.json,
promo_picks.json, scorecard.json, player_results.json, cfb_usage.json, model_a.json, cbb_ab.json, tennis_model.json (file time in the last 30 h),
tennis.json (from app/data/odds/tennis.json, a bare array: use its file time, last 30 h), and props_vs_close.json if its "made" is newer than the published copy.
Publish with the Artifact tool: url https://claude.ai/artifact/Pdq3m2UxoriBf2gBdzgBp4, file_path <scratchpad>/betdesk/index.html,
root <scratchpad>/betdesk, files map of those names. Data files go at the ROOT path (the page fetches nhl_props.json, not data/...).
If refused for files you have not read, read them with the Artifact read action (paths) and publish again. Do not pass capabilities.

## E. Report
All fine: no reply; set_thread_resolved(resolved=true) and end with no_reply_needed.
Anything failed or tripped (a step error, a sport with games but no lines, the credit check, the name check, the publish):
post ONE short message saying exactly what (no secrets), leave the thread open, and also send it to the Bet Desk thread
(send_message to session_01JsbVsY24TteeBqiGur7frJ) so the owner can fix it.
