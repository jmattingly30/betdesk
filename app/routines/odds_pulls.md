# Bet Desk odds pulls (J 2026-10-28 schedule; J 2026-10-29: fresh short sessions on a smaller model, each pull publishes itself,
# noon and night pulls skip book group 4). Times America/Denver. The scheduled tasks only say "follow this file";
# change the steps here, not in the tasks.

| task | when | sports | books |
|---|---|---|---|
| Morning chain | 6:30 daily | all, plus NFL props build and the props refresh | all 12. Steps in morning_chain.md |
| Noon pull | 12:00 daily | NFL NBA NHL from OddsPapi; CFB+CBB game lines from The Odds API (3 credits each, CBB from 2026-11-02) | OP_GROUPS=3 |
| Night pull (look ahead) | 22:15 daily | all | OP_GROUPS=3 |
| NFL Sunday night pull | 20:45 Sun | NFL | all 12 |
| NFL Tuesday pull | 15:30 Tue | NFL | all 12 |

OP_GROUPS=3 skips Bally, Circa and BetOnline; op_live.py carries those books' last prices forward from the previous file.

## Steps for every pull except the morning chain (TASK names the row)
Rules: run each step in its own Bash call starting with `cd /mnt/project-files/sports-analytics/app` (never chain later steps
after a cd). Never print or store API keys. Edit no file except what the scripts write; start no other work.
1. Setup if missing: `python3 -m pip install -q pyarrow pandas numpy requests`.
2. Sweep with run_in_background and wait for it to exit (never poll with sleep):
   - Noon: `OP_GROUPS=3 python3 op_live.py NFL NBA NHL`
   - Night: `OP_GROUPS=3 OP_DEADLINE_MIN=110 python3 op_live.py NFL NBA CBB NHL CFB` (timeout 7200000; ~85 min with CBB in season; CFB last and the deadline
     stops it before the 2-hour kill, 2026-11-07)
   - NFL Sunday / Tuesday: `python3 op_live.py NFL`
   It holds data/odds/oddspapi.lock while running (one loop per key).
2b. Night only, when step 2 exits: `python3 op_live.py TENNIS` with run_in_background (timeout 5400000; 45-60+ min, the 11-06 night run passed 60 min) -> data/odds/tennis.json.
   Do steps 3-4a while it runs; wait for it to exit before step 5 (2026-11-02: one combined sweep hit the 2-hour limit in tennis).
2c. Every pull: `python3 /mnt/project-files/sports-analytics/models/injuries/scan.py` (injury news since the morning, J 2026-11-06;
    NBA Model B/Model A in 4d drop players now listed Out or Doubtful). Publish app/data/injuries.json.
3. Slates:
   - Noon: `ODDS_FORCE=CFB,CBB python3 export_slates.py` (college game lines from The Odds API; CBB skipped before 2026-11-02)
   - All others: `env -u ODDS_API_KEY python3 export_slates.py` (no Odds API credits; reuses saved replies)
4. `python3 line_tools.py`
4b. Noon and night: `python3 /mnt/project-files/sports-analytics/models/cbb_ab/live.py` (CBB Model B re-priced on the new lines -> app/data/cbb_ab.json).
4c. Every pull (noon, night, NFL Sunday, NFL Tuesday): `python3 /mnt/project-files/sports-analytics/models/nfl_ab/live.py` (NFL Model B re-priced on the new lines -> app/data/nfl_ab.json; J 2026-11-05).
4d. Noon and night: `python3 /mnt/project-files/sports-analytics/models/nba_ab/live.py` (NBA Model B re-priced on the new lines -> app/data/nba_ab.json; J 2026-11-05; ~1 min, needs scipy. The night pull catches NBA openers, where Model B's tiers were tested).
4e. Noon and night, after step 3 (and 4a at night): `python3 /mnt/project-files/sports-analytics/models/cfb_ab/live.py` (CFB Model B re-priced on the new lines -> app/data/cfb_ab.json; J 2026-11-05; ~15 s).
4f. Noon and night (night: after the 2b tennis sweep exits): `python3 /mnt/project-files/sports-analytics/models/tennis_ab/live.py` (tennis Model A sim + Model B tiers from tennis_model.json + tennis.json -> app/data/tennis_ab.json; J 2026-11-05; ~5 s, paper only). Night only, then: `python3 /mnt/project-files/sports-analytics/models/tennis_ab/grade.py` (grades past matches from tennisexplorer results, ~3 pages 3 s apart).
4g. Every pull, after 4b-4f: `python3 /mnt/project-files/sports-analytics/models/sims/build.py` (Watch the sim, J 2026-11-06; NFL Sunday/Tuesday pulls: add ` NFL`; about 1 min) -> app/data/sims_{nba,cbb,nfl,cfb,tennis}.json.
4a. Night pull only: `cd /mnt/project-files/sports-analytics/models/oddspapi && python3 cfb_usage.py` (rebuilds app/data/cfb_usage.json from the CFB props; noon has no CFB props).
5. Publish (never edit bet-desk.html): copy app/bet-desk.html to <scratchpad>/betdesk/index.html, and app/data/game_slates.json,
   app/data/line_tools.json, app/data/cbb_ab.json (noon and night), app/data/nfl_ab.json (every pull), app/data/nba_ab.json (noon and night), app/data/cfb_ab.json (noon and night), app/data/tennis_ab.json and app/data/tennis_model.json (noon and night, file time in the last 30 hours), app/data/cfb_usage.json (night only), app/data/sims_*.json that 4g rewrote this pull, app/sim.js as sim.js, app/data/odds/tennis.json as tennis.json if its file time is in the last 30 hours and each app/data/odds/props_{NFL,NBA,NHL}.json made in the last 30 hours (its "made" field) to
   <scratchpad>/betdesk/ at the bare name. Publish with the Artifact tool: url https://claude.ai/artifact/Pdq3m2UxoriBf2gBdzgBp4,
   file_path <scratchpad>/betdesk/index.html, root <scratchpad>/betdesk, files map of only those data files. If refused for files you
   have not read, read them with the Artifact read action (paths) and publish again. Do not pass capabilities.
6. All fine: no reply; set_thread_resolved(resolved=true) and end with no_reply_needed. If op_live.py fails or writes no games for a
   sport that has games, or a script or the publish fails: post ONE short message saying exactly what failed (no secrets), leave the
   thread open, and send the same line to the Bet Desk thread (send_message to session_01JsbVsY24TteeBqiGur7frJ).
