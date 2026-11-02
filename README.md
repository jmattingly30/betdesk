# J's Bet Desk

Private, standalone version of the Bet Desk (moved out of the Claude artifact on 2026-11-02).

- `public/` is the whole app: `index.html` plus the model and odds data files it loads. It's a static site with no build step.
- Bets and settings save on the device (browser storage). Use **Settings (⚙) → Backup → Export / Import bets** to back them up or move them between devices.
- Add it to an iPhone home screen from Safari: Share → Add to Home Screen.

## Hosting (Cloudflare Pages, free)

1. https://dash.cloudflare.com → Workers & Pages → Create → Pages → **Connect to Git** → pick `jmattingly30/betdesk`.
2. Framework preset: **None**. Build command: *(leave empty)*. Build output directory: `public`. Deploy.
3. Keep it private: Zero Trust → Access → Applications → Add → Self-hosted, enter the `*.pages.dev` domain, and add a policy "Emails: jackmattingly30@gmail.com". Cloudflare emails you a one-time code to sign in.

Every push to `main` redeploys the site.

## Not moved yet: the daily odds pulls

The pull scripts (`op_live.py`, `export_slates.py`, `line_tools.py`, the model builds and routines) live in the Sports Analytics Claude project and still publish to the old artifact. See `docs/NEXT_STEP.md` for the request to send that project to move them here as GitHub Actions. Until then the data in `public/` is a snapshot from 2026-11-02.

The MLB "slate from your MLB app" panel read Google Drive through Claude, so it can't do that here; type the app's runs by hand.
