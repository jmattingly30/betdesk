# OddsPapi (free plan) — what it gives us (checked 2026-10-27)

Key: env var `ODDSPAPI_API_KEY` (query param `apiKey`). Base `https://api.oddspapi.io/v4`. Allowed in the cloud env (J added `api.oddspapi.io` 2026-10-27).

## Limits
- 250 billable calls per month (resets the 1st). Billable: /fixtures, /tournaments, /odds, /odds-by-tournaments, /markets, /scores, /settlements, /sports, /bookmakers, /participants.
- `/v4/historical-odds` is FREE (does not count), but 5 s minimum between calls per key, max 3 bookmakers per call. History starts January 2026 (no older seasons).
- `/v4/account` is free; shows request_count. As of 2026-10-27 15:15 UTC: 48 of 250 used (tennis thread).
- /fixtures: sportId + from/to at most ~9-10 days apart, or tournamentId alone (returns all seasons of that event). One call returns every fixture in the window (tennis 9 days = ~2,500 fixtures).
- All 366 books on the plan incl. pinnacle, draftkings, fanduel, betmgm, caesars, espnbet, fanatics, betrivers, hardrockbet, ballybet, circasports, betonline.ag, bet365, betfair-ex. No live feed on free, but historical includes in-play prices.

## Coverage seen (one finished fixture each, pinnacle + draftkings + fanduel)
| Sport | Markets per book (Pin / DK / FD) | Player prop markets (Pin / DK / FD) | First price, hours before start (Pin / DK / FD) |
|---|---|---|---|
| NFL (PHI v CAR 10-18) | 253 / 794 / 818 | 55 / 435 / 344 | 161 / 927 / 286 |
| CFB (Sam Houston v NMSU 10-22) | 167 / 507 / 478 | 9 / 82 / 183 | 76 / 124 / 147 |
| NBA (UTA v NOP 10-24) | 418 / 887 / 819 | 28 / 175 / 210 | 43 / 43 / 43 |
| CBB (Villanova v Utah St 2026-03-20) | 219 / 54 / 47 | 5 / 5 / 0 | 95 / 117 / 95 |
| NHL (NYR v ANA 10-19) | 118 / 123 / 137 | 6 / 27 / 21 | 29 / 30 / 31 |
| MLB (BAL v MIL 09-19) | 136 / 212 / 123 | 10 / 63 / 14 | 8 / 16 / 22 |
| Tennis (ATP/Challenger) | ~60 / ~90 / ~77 | — | about 1-2 days |

Every price change is timestamped (`createdAt`, plus Pinnacle `limit`), so openers, closers and full line movement are available per book, and in-play prices let you infer the winner of a match. Fixture objects have no scores; /scores and /settlements are billable (1 call per fixture).

## How to use cheaply
1. Spend 1 billable call per 9-day window per sport to list fixtures (a full NFL season = ~25 calls).
2. Loop `/historical-odds?fixtureId=...&bookmakers=pinnacle,draftkings,fanduel` at 5.2 s spacing (free). ~690 fixtures an hour per key; only one loop can run at a time per key, so threads must take turns.
3. Map market ids with one `/v4/markets` call (cached at scratchpad op/markets.json in the tennis thread; 56k markets, `playerProp` flag).

Code example: `models/tennis-golf-research/op_pull.py` (fixture windows + resumable history pull + compact open/close/last extraction).
