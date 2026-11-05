# OddsPapi 2026 odds history -> our models (thread "Use OddsPapi odds history in our models", 2026-10-27)

- `pull.py`: fixture lists (op/us/fx_<lg>.json, 1 billable call per league via tournamentId) + free /historical-odds for
  Pinnacle, DraftKings, FanDuel -> op/us/hist_<lg>.jsonl (main lines + every full-game player prop; open, 24 h, close,
  alive-at-start). One loop per key: book a slot with the tennis thread, respect app/data/odds/oddspapi.lock.
- `props_backtest.py nba | nhl | live [nba,nhl,nfl]`: our UNBLENDED prop chances vs each book's no-vig price at the
  opening and closing lines; bets when our EV beats the price (ROI, CLV); size check (our mean vs close line vs actual).
  Player ids have no names in history: matched per game (Hungarian on line vs our mean; NHL on book chance vs Poisson
  on our mean, which slightly favours agreement) and voted across games. Results in results/.

## Results 2026-10-28 (806 games pulled: NBA/NHL Mar-Jun + Oct, NFL Sep-Oct), after cleaning bad book lines
Cleaning (load_props): drop a book's player-stat when its closing main line is far from the other books' (FanDuel NBA
'points' carried another market for ~9% of players and had made the model look far better than it is).
Out of sample (NBA test_predictions fit to 2023-24; NHL fit 2023-25), Mar-Jun 2026:
- NBA 14.1k lines: log loss model .6963 vs book .6859; 50/50 blend .6863 = book (no added information overall).
  Means unbiased (pts 14.71 vs actual 14.88). Bets: DraftKings/FanDuel OVERS at 12%+ EV at the open 201 bets, ROI
  +11.6% (se 8.7), CLV +3.6%; at 5%+ 494 bets, +9.6% (se 5.0), CLV +1.5%. Pinnacle: none. Unders: CLV negative.
- NHL 7.2k lines: model beats Pinnacle AND DraftKings on assists, points, goals (close .6510 vs .6549; SOG even with
  Pinnacle). Bets profit (Pinnacle unders 5%+: +10%, se 3) but CLV negative everywhere -> outcome accuracy, no
  line-move edge; spring/playoff under drift may explain part of the ROI.
Live Bet Desk snapshots (Oct 16-26, small): NHL SOG beats Pinnacle and DK at the close (.681/.679 vs .685/.686), NFL
receptions beat Pinnacle (.633 vs .668) and DK, lose to FanDuel; NBA desk props run LOW vs actual (pts -5%, reb -12%,
PRA -7%), unlike the backtest model -> an early-season export issue.

## Full Jan-Jun 2026 rerun (2026-10-28 15:00 UTC; history for these books starts Jan 13; DK/FD NBA props only from March)
- NBA 32.3k lines (Pinnacle 16.4k, DK 8.7k, FD 7.3k): model .6952/.6972 vs book .6873/.6864 at open/close, blend = book.
  DK/FD OVERS 12%+ EV at the open: 232 bets, ROI +10.7% (se 8.0), CLV +4.0% (5%+: 574, +5.6%, CLV +1.6%) -> confirms the
  Bet Desk PROVEN rule. Unders: CLV -2.5..-3.9% -> stay held back. Pinnacle overs at close lose (-12.6% at 12%+).
- NHL 16.2k lines: model .6601 vs book .6608 (ties; beats DK on assists/points). DK unders at open 5%+ 305 bets +14.9%
  (se 6.1) but CLV -5.4% -> not adopted (outcome luck/drift, not line edge). Means unbiased (goals .37 = actual .37).
- lines.py -> op/us/lines_{cbb,nba,nhl}.csv: CBB 1,962 D1 games Jan 13-Apr 7 (multi-book openers for the CBB v2 test).
