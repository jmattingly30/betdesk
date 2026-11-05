# J's Bet Desk

Private, standalone version of the Bet Desk (moved out of the Claude artifact on 2026-11-02).

- `public/` is the whole app: `index.html` plus the model and odds data files it loads. It's a static site with no build step.
- Bets and settings save on the device (browser storage). Use **Settings (⚙) → Backup → Export / Import bets** to back them up or move them between devices.
- Add it to an iPhone home screen from Safari: Share → Add to Home Screen.

## Hosting (Cloudflare Pages, free)

1. https://dash.cloudflare.com → Workers & Pages → Create → Pages → **Connect to Git** → pick `jmattingly30/betdesk`.
2. Framework preset: **None**. Build command: *(leave empty)*. Build output directory: `public`. Deploy.
3. Keep it private: Zero Trust → Access → Applications → Add → Self-hosted, enter the `*.pages.dev` domain, and add a policy "Emails: <your email>". Cloudflare emails you a one-time code to sign in.

Every push to `main` redeploys the site.

## Odds pipeline (GitHub Actions)

The pull scripts and model inputs now live here (`app/`, `models/`, `op/`, `archive/`, same layout as the Claude project
folder; the runner links the repo at `/mnt/project-files/sports-analytics` so the routine steps run unchanged).
`tools/morning.sh`, `tools/pull.sh`, `tools/cfb_v2.sh` follow `app/routines/morning_chain.md` and `odds_pulls.md`,
copy changed data into `public/` and commit to `main` (which redeploys the site). A failed step fails the job, so
GitHub emails you, after whatever did refresh is published.

| workflow | Denver time |
|---|---|
| Bet Desk morning chain | daily 6:30 |
| Bet Desk noon pull | daily 12:00 |
| Bet Desk night pull | daily 22:15 |
| Bet Desk NFL Sunday pull | Sun 20:45 |
| Bet Desk NFL Tuesday pull | Tue 15:30 |
| CFB v2 weekly refresh | Mon + Thu 9:00 |

GitHub cron is UTC, so each has two crons (daylight and standard time) and the first step skips the one that is off
by an hour. Each has a Run workflow button on the Actions tab. Secrets (Settings > Secrets and variables > Actions):
`ODDSPAPI_API_KEY`, `ODDS_API_KEY`, `CFBD_API_KEY`.

Not covered: the MLB slate (Google Drive; type runs by hand) and model files other Claude threads change in the
project folder after 2026-11-02 (tennis_model.json, new model code); copy those in by hand.
