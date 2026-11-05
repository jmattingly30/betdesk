# Free odds feeds: how we split The Odds API and OddsPapi (2026-10-27)

J (2026-10-27 18:00): split pulls between The Odds API and OddsPapi to get the most pulls a month, keeping room for
research pulls. No paid feed.

## What each free plan really costs
| | The Odds API (free) | OddsPapi (free) |
|---|---|---|
| Monthly limit | 500 credits | 250 billed calls |
| Current game lines | 1 credit per market per 10 books (3 per sport per pull) | **free** via /historical-odds (3 books per call, 5 s apart per key) |
| Freshness | live at pull time | prices to within minutes for games not yet started (tested NFL 18:02 UTC: DK 4 min old) |
| Props, alt lines | paid only | **free** in the same call (NFL DK ~435 prop markets) |
| Line history / openers / closers | paid only | **free**, every price change timestamped, since Jan 2026 |
| What is billed | every odds call | only fixture lists, /odds, /scores, /markets, /sports |
| Books | 10 named per credit | 366, incl. Pinnacle, Circa, bet365, all Colorado books |

So OddsPapi is the main feed and The Odds API becomes the backup.

## The setup (built)
- `app/op_live.py`: sweeps NFL (next 9 days), CFB (7), NBA and CBB (2) through the free endpoint. 4 calls per game
  for 12 books: Pinnacle, DraftKings, FanDuel / BetMGM, Caesars, ESPN BET / Fanatics, BetRivers, bet365 /
  Bally Bet, Circa, BetOnline. It writes `data/odds/{SPORT}.json` in The Odds API's format.
- Billed calls: one fixture list per sport family (football = NFL + CFB, basketball = NBA + CBB), cached 24 h:
  at most 2 a day, ~60 a month.
- `export_slates.py` uses a sweep under 6 h old. It only spends Odds API credits when the sweep failed or is stale
  (`ODDS_FORCE=1` forces a pull).
- Key sharing: the OddsPapi key allows one loop at a time. A live sweep writes `data/odds/oddspapi.lock`, and the
  tennis and odds-history loops pause while it is held. History loops run in the hours between sweeps.

## Props first (J 2026-10-27 18:22)
Having odds for every market daily matters more to J than refreshing several times a day. The daily sweep therefore
saves every player prop at every book from the same free calls: `data/odds/props_{NFL,NBA,NHL,CFB,CBB}.json`, with
main lines, alt lines, ladders (250+, 275+...) and Yes/No markets like anytime TD. On GB-CAR, 3 books gave 937 prop
prices. The free history has no player names, so names come from one billed `/odds-by-tournaments` call (DraftKings,
whole league, with names and DK's main-line flags). It is cached by player id and only re-asked when 5+ unknown
players show up, at most once a league a day.

## Monthly budget
| Use | OddsPapi billed (250) | OddsPapi free calls (~690/h) | Odds API credits (500) |
|---|---|---|---|
| Daily sweep ~14:30 UTC, all sports, lines + every prop | ~45 (fixture lists every 2 days: football, basketball, hockey) + ~30 (player names) | ~150-500 a day, 15-45 min | 0 |
| Game-day sweeps: Sat ~15:00 UTC (CFB), Sun ~15:30 UTC (NFL after inactives) | 0 (cached lists) | ~300 a week | 0 |
| Weekly accuracy check: one Odds API pull vs our sweep | 0 | 0 | ~50 (12 a week) |
| Fallback when a sweep fails | 0 | 0 | 12 per failed day |
| **Left for research** | **~175** (tennis, golf, season fixture lists for history) | **~20 h a day** | **~400+** |

More sweeps cost no API budget. The real limit is Claude usage, because each scheduled sweep is a cloud session.
Good candidates are NFL Sunday 11:30 ET and 3 pm ET, and CBB evenings from November.

## Limits to know
- The main line per book is picked as the lowest-hold rung near even money. Books post every alt rung, and right after
  a move this can pick the wrong rung (DK 47.5 vs 48.5). The weekly Odds API check catches drift.
- Not truly live: a sweep is a snapshot, and a 4-call game takes ~21 s. Middles that open and close inside an hour can
  still be missed.
- History starts January 2026, so it backtests this season only.

## Next steps this unlocks
1. Prop odds are now saved daily (built). Next, Bet Desk fills each prop's book line and price automatically,
   shows EV at the best book against our model, finds prop middles, and replaces the promo optimizer's estimated
   prices.
2. Alt-line middles and +EV, since every rung is priced.
3. Real openers and closers per book, for automatic CLV on J's bets and true sharp-move flags
   (Pinnacle/Circa moves vs DK/FD).
