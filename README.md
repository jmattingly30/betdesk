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

## Shared desks (sign-in, guests, Honcho)

`wrangler.jsonc` + `worker/` put a Cloudflare Worker in front of every request:

- **Sign-in by invite link.** `/join/<token>` shows an "Open my Bet Desk" button (so link previews in message apps do not use the link up); tapping it sets a signed, HttpOnly cookie (400 days). J's link uses the
  `OWNER_TOKEN` secret; each guest's link is made in J's desk (Settings > People), works once (one device), and only its
  SHA-256 hash is stored. "New link" is for another device; "Sign out everywhere" and "Turn off" block that person.
  A shared link (`/invite/<token>`, Settings > People > "One link for several people") lets up to N people each make
  their own desk by typing a name; making a new one or "Turn off" closes it. Anyone else gets a "this desk is private" page.
- **Separate storage.** Each person's bets and settings live in their own Durable Object (`u:<id>`); the guest list is
  in `directory`. The person always comes from the cookie, never from the request, so bets cannot mix. J's desk
  (`public/index.html` + `public/account.js`) now saves to J's account and copies the bets on a phone up once.
- **What each person can load.** J: everything in `public/`, as before. Guests: only `public/guest/` (a simple pick
  sheet built by `worker/picks.js` from the Model B files), icons, and `/api/*` for their own data. Raw model files,
  J's page and J's bets are never served to guests.
- **Honcho** (`worker/honcho.js`, guests only): bets, results, passes and "tell the desk" notes are sent as the guest's
  messages; once a day (and every 6 actions) Honcho's answer sets their sport order, detail level (1 Simple,
  2 Guided, 3 Full; also grows with use) and the note at the top of their desk.

Secrets (Cloudflare > Workers & Pages > betdesk > Settings > Variables and secrets, type Secret):
`SESSION_SECRET` (any long random string), `OWNER_TOKEN` (long random string; J's link is `https://<site>/join/<OWNER_TOKEN>`),
`HONCHO_API_KEY`. Until the first two are set, the Worker serves `public/` with no sign-in, exactly as before.
If Cloudflare Access protects the site, add each guest's email to its policy, or remove the Access app (the Worker
sign-in replaces it). Local test: `npx wrangler dev --env-file <file with the three secrets>`.

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

Each workflow file only sets its schedule and script; the shared job is `.github/workflows/run.yml`. Before every publish
`tools/check_data.py` checks that games, teams and players line up across the data files (a crossed game fails the job).
GitHub cron is UTC, so each has two crons (daylight and standard time) and the first step skips the one that is off
by an hour. Each has a Run workflow button on the Actions tab. Secrets (Settings > Secrets and variables > Actions):
`ODDSPAPI_API_KEY`, `ODDS_API_KEY`, `CFBD_API_KEY`.

Not covered: the MLB slate (Google Drive; type runs by hand) and model files other Claude threads change in the
project folder after 2026-11-02 (tennis_model.json, new model code); copy those in by hand.
