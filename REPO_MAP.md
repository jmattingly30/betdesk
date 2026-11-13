# Repo map (written 2026-11-13 for the deep audit; update when files move)

Read this instead of re-reading code. Sport-model goals and rules live in project memory, not here.

## What the repo is
- `public/` — the site (static, Cloudflare). `index.html` is BUILT from `app/bet-desk.html` by `tools/standalone_patch.py`; the JSON beside it is copied from `app/data/` by `publish()` in `tools/ci.sh`. Never hand-edit `public/`.
- `worker/` + `wrangler.jsonc` — Cloudflare Worker in front of the site: invite sign-in (`index.js`), one Durable Object per person for bets (`desk.js`), guest pick sheet (`picks.js`), Honcho notes (`honcho.js`). `public/account.js` is the browser side.
- `app/` — the desk source and the exporters.
  - `bet-desk.html` (410 KB, inline JS; owned by the "Private betting app and tracker" thread). UI layer = `redesign.js`, inlined by `redesign_inline.py`. `sim.js` = Watch the sim viewer.
  - `op_live.py` OddsPapi sweep -> `app/data/odds/*.json`. `export_slates.py` -> `game_slates.json` (games + consensus lines, every model joins to it by game id). `line_tools.py` -> `line_tools.json` (fair price, EV, middles, arbs). `export_games.py`, `export_model_a.py`, `export_nba_props.py`, `export_nfl_props.py`, `add_recent.py` (workload rule), `past_props.py`.
  - `routines/*.md` — the step lists the shell scripts mirror (Claude routines used the same lists).
- `models/` — one folder per model.
  - Live game models (Model A sim + Model B tiers): `cbb_ab/`, `nfl_ab/`, `nba_ab/`, `cfb_ab/` (each: `live.py` -> `app/data/<sport>_ab.json` + `history/runs.jsonl`; `grade.py` -> `history/grade.json`). `tennis_ab/` is held.
  - Model A engines: `rebuild/basketball` (NBA stack, CBB v2 in `cbb_live/`), `rebuild/football` (NFL drive model, CFB v4 in `cfb_gap/live/`), `rebuild/hockey`, `rebuild/shadow/shadow.py` (live shadow projections).
  - Shared Model B code: `model-b/` (`devig.py`, `ml_anchor.py`, `unproven.py`, `pregame.py` = keep started games frozen).
  - Player props: `nba-player/`, `nfl-player/`, `nhl-player/`; `injuries/scan.py` (ESPN injury feed).
  - `grading/grade.py` (scorecard.json), `snap_games.js` / `snap_flags.js` (headless desk snapshots), `sims/build.py`, `parlay/`, `promo/`, `oddspapi/`, `team-strength/`, older `cbb/`, `cfb-v2/`, `cfb-v3/`.
- `tools/` — `ci.sh` (step/publish/commit helpers), `morning.sh`, `pull.sh noon|night|nfl`, `cfb_v2.sh`, `check_data.py` (data check before every publish), `sync_from_project.sh` (copies the Claude project folder over the repo; repo-only changes must be re-applied after it).
- `.github/workflows/` — `run.yml` is the shared job; `morning.yml`, `noon.yml`, `night.yml`, `nfl-sunday.yml`, `nfl-tuesday.yml`, `cfb-v2.yml` only set schedules. All share concurrency group `betdesk-data`.
- `docs/audit/`, `docs/research/` — findings from the audit and research rounds.

## Entry points
- Morning: `tools/morning.sh` = odds sweep (background) -> props (NHL, NBA, NFL) -> team models -> slates -> line tools -> A+B live + grade per sport -> sims -> parlays/promo -> snapshots -> grade -> publish + commit to main.
- Pulls: `tools/pull.sh` = odds sweep -> injury scan -> slates -> line tools -> Model B live per sport -> guest picks -> sims -> publish + commit.

## Rules that bite
- Scheduled runs rewrite `app/data/**`, `public/*.json`, `models/**/history/*`, `models/**/data/**` on main. Never put those files in a pull request (it will conflict).
- Applied one-shot patches live in `archive/applied-patches/`; never rerun them.
- The project folder (`/mnt/project-files/sports-analytics`) holds the same code; a change made only in the repo is undone by the next sync unless it is also copied there.

## Known issues
See `docs/audit/betdesk.md`.
