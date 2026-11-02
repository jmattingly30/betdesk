# Next step: move the odds pulls (send this to the Sports Analytics project)

```text
The Bet Desk app now lives in my private repo jmattingly30/betdesk (static site in public/). Move the odds pulls there. I'm low on usage: work efficiently, don't re-research, and ask before any step that will run longer than ~10 minutes.

0. Attach the repo with add_repo (owner jmattingly30, repo betdesk, access push) and clone it. If refused, tell me the exact reason and stop.
1. Copy only what the pulls need into pipeline/: app/op_live.py, export_slates.py, line_tools.py, app/routines/ (odds_pulls.md, morning_chain.md), models/oddspapi/cfb_usage.py, the CFB v2 export scripts, and every script/model file they call (trace imports). Add requirements.txt. No API keys, .env or lock files.
2. Make the scripts write their output files into public/ (the same file names the page already loads). Don't change any model math or public/index.html, except copying in a newer app/bet-desk.html if it differs from the artifact (keep the Backup card, manifest and icon lines that were added).
3. Add GitHub Actions workflows replacing these Claude scheduled tasks, same times (America/Denver; GitHub cron is UTC): morning chain daily 6:30 am (morning_chain.md parts A-E), noon pull daily 12:00 pm, night pull daily 10:15 pm, NFL Sunday pull Sun 8:45 pm, NFL Tuesday pull Tue 3:30 pm, CFB v2 refresh Mon + Thu 9:00 am. Each runs the scripts and commits the changed files in public/. Secrets: ODDSPAPI_API_KEY, ODDS_API_KEY, CFBD_API_KEY. Replace Claude-only steps (Artifact publish, hearthbot, PushNotification) with the commit; on failure just fail so GitHub emails me. Add a manual "Run workflow" trigger to each. Estimate monthly Action minutes against the 2,000 free private-repo minutes and suggest a trimmed schedule if it's over.
4. Test one sport only (not a full sweep), push to main.
5. Don't disable my existing Claude scheduled tasks or the artifact. Finish with where to add the 3 secrets.
```
