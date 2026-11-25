// End-to-end test of apps_script/Server.gs + Bundle.gs on the fake Google services (test/gas_stubs.js), using the
// same recorded real data as tests.js (MLB schedule, Savant CSVs, Baseball-Reference page) plus a sample odds
// response. Needs globals: FIXTURES (name -> text). Writes window.GAS_RESULTS.
(function () {
  var results = [];
  function check(name, ok, detail) { results.push({ name: name, ok: !!ok, detail: detail === undefined ? '' : String(detail) }); }
  var D = __require('data'), U = __require('util');

  // A game that is ready to run: the Reds game from the recorded schedule, with both lineups filled from real Savant batters
  var sched = JSON.parse(FIXTURES['schedule_2026-09-18.json']);
  var bats = D.parseSavantPlayers(FIXTURES['savant_batters_2026.csv']).filter(function (b) { return b.pa >= 150 && isFinite(b.woba) && isFinite(b.xwoba); }).slice(0, 18);
  var game = null;
  sched.dates.forEach(function (d) { d.games.forEach(function (g) { if (g.teams.home.team.name === 'Cincinnati Reds' || g.teams.away.team.name === 'Cincinnati Reds') game = g; }); });
  var toP = function (b) { return { id: b.id, fullName: b.name, primaryPosition: { abbreviation: 'DH' } }; };
  game.lineups = { awayPlayers: bats.slice(0, 9).map(toP), homePlayers: bats.slice(9, 18).map(toP) };
  var awayName = game.teams.away.team.name, homeName = game.teams.home.team.name;
  var oddsEvents = [{ id: 'e1', commence_time: game.gameDate, away_team: awayName, home_team: homeName, bookmakers: [
    { key: 'betmgm', title: 'BetMGM', markets: [{ key: 'h2h', outcomes: [{ name: awayName, price: 110 }, { name: homeName, price: -130 }] }] },
    { key: 'draftkings', title: 'DraftKings', markets: [{ key: 'h2h', outcomes: [{ name: awayName, price: 115 }, { name: homeName, price: -135 }] }] },
    { key: 'fanduel', title: 'FanDuel', markets: [{ key: 'h2h', outcomes: [{ name: awayName, price: 108 }, { name: homeName, price: -126 }] }] }] }];
  var finalStatus = { code: 200, text: JSON.stringify({ dates: [{ games: [{ gamePk: game.gamePk, status: { detailedState: 'Final' }, teams: { away: { score: 5 }, home: { score: 3 } } }] }] }) };
  ROUTES = [
    { match: /statsapi\.mlb\.com\/api\/v1\/schedule\?sportId=1&gamePks=/, respond: function () { return finalStatus; } },
    { match: /statsapi\.mlb\.com\/api\/v1\/schedule\?sportId=1&date=2026-09-18/, respond: function () { return { code: 200, text: JSON.stringify(sched) }; } },
    { match: /baseballsavant.*statcast_search/, respond: function () { return { code: 200, text: 'pitch_type,woba_value,woba_denom,fielder_2\nFF,0.9,1,1\nSL,0,1,1\nFF,0,0,1\n' }; } },
    { match: /baseballsavant.*type=batter/, respond: function () { return { code: 200, text: FIXTURES['savant_batters_2026.csv'] }; } },
    { match: /baseballsavant.*type=pitcher/, respond: function () { return { code: 200, text: FIXTURES['savant_pitchers_2026.csv'] }; } },
    { match: /baseballsavant.*catcher_framing/, respond: function () { return { code: 200, text: 'player_id,player_name,n_called_pitches,runs_extra_strikes\n1,Sample C,6000,5\n' }; } },
    { match: /statsapi\.mlb\.com\/api\/v1\/people/, respond: function () { return { code: 200, text: '{"people":[]}' }; } },
    { match: /baseballsavant.*outs_above_average/, respond: function () { return { code: 200, text: 'player_id,last_name,outs_above_average,diff_success_rate_formatted\n10,Sample,4,2%\n' }; } },
    // model v2 also reads last season; the 2026 page stands in for 2025
    { match: /baseball-reference\.com\/leagues\/majors\/202[56]/, respond: function () { return { code: 200, text: FIXTURES['bbref_2026.html'] }; } },
    { match: /api\.the-odds-api\.com/, respond: function () { return { code: 200, text: JSON.stringify(oddsEvents), headers: { 'x-requests-remaining': '481' } }; } },
  ];

  try {
    // ---- setup ----
    setup();
    var ss = SpreadsheetApp.openById(PROPS.LOG_SHEET_ID);
    check('setup: log spreadsheet created with PlayLog + FanGraphs tabs, empty default tab removed',
      ss.getSheets().map(function (s) { return s.getName(); }).join(',') === 'PlayLog,FanGraphs', ss.getSheets().map(function (s) { return s.getName(); }).join(','));
    check('setup: one 5-minute refreshLog timer', TRIGGERS.length === 1 && TRIGGERS[0].fn === 'refreshLog' && TRIGGERS[0].minutes === 5);
    setup();
    check('setup: running it again does not duplicate the timer', TRIGGERS.length === 1);

    var page = doGet({ parameter: {} });
    check('doGet: serves Index.html with the phone viewport tag', page.file === 'Index' && /width=device-width/.test(page.meta.viewport || ''));

    var info = api('info', {});
    check('info: no odds key yet -> oddsFeed false; CSV link is the web app URL', info.oddsFeed === false && info.csvUrl === 'https://script.google.com/macros/s/TEST/exec?export=csv');
    PROPS.ODDS_API_KEY = 'test-key';
    check('info: odds key is only ever read server-side', api('info', {}).oddsFeed === true && JSON.stringify(api('info', {})).indexOf('test-key') < 0);

    // ---- no-FanGraphs run ----
    var s = api('saveSettings', { useFangraphs: false });
    check('settings: no-FanGraphs saved', s.useFangraphs === false && JSON.parse(PROPS.SETTINGS).useFangraphs === false);
    var look = api('lookup', { team: 'Reds', date: '2026-09-18' });
    check('lookup: finds the Reds game, ready to run', look.games && look.games.length === 1 && look.games[0].readiness.ready, JSON.stringify(look).slice(0, 200));
    var run = api('run', { gamePk: game.gamePk, date: '2026-09-18' });
    check('run (no FanGraphs): ok, no FanGraphs data needed', run.ok && run.result.inputs === 'no-fangraphs', run.ok ? '' : JSON.stringify(run.missing));
    if (!run.ok) throw new Error('run failed');
    check('run: all four models with fair moneylines', ['hierarchical', 'pythagorean', 'pythagenpat', 'pythagenpatSeason'].every(function (k) { return isFinite(run.result[k].awayWinProb) && isFinite(run.result[k].awayFairMl); }));
    check('run: every book returned for shopping', run.books.length === 3 && run.books.map(function (b) { return b.book; }).join(',') === 'BetMGM,DraftKings,FanDuel');
    check('run: log row saved with the best price per side (away +115 DraftKings, home -126 FanDuel)',
      run.logRow.mkt_source === 'feed' && run.logRow.mkt_awayMl === 115 && run.logRow.mkt_awayBook === 'DraftKings' && run.logRow.mkt_homeMl === -126 && run.logRow.mkt_homeBook === 'FanDuel' && run.logRow.mkt_nBooks === 3);
    check('run: odds credits left recorded from the response header', PROPS.ODDS_REMAINING === '481');

    // ---- the log sheet keeps types ----
    var log = api('log', {});
    var r0 = log.rows[0];
    check('log: one row read back from the Sheet', log.rows.length === 1 && r0.id === run.logRow.id);
    check('log: dates stay text (not converted to Date), numbers stay numbers',
      r0.gameDate === '2026-09-18' && typeof r0.runAt === 'string' && r0.runAt === run.logRow.runAt && typeof r0.awayLambda === 'number' && r0.gamePk === game.gamePk && r0.oddsTaken === null && r0.played === '');
    check('log: every stored field comes back identical', Object.keys(run.logRow).every(function (k) { var a = run.logRow[k], b = r0[k]; return a === b || (a == null && b == null) || (a === '' && b === '') || (typeof a === 'number' && Math.abs(a - b) < 1e-9); }),
      Object.keys(run.logRow).filter(function (k) { var a = run.logRow[k], b = r0[k]; return !(a === b || (a == null && b == null) || (typeof a === 'number' && Math.abs(a - b) < 1e-9)); }).map(function (k) { return k + ':' + run.logRow[k] + '!=' + r0[k]; }).join(' '));

    // ---- your entry ----
    var up = api('update', { id: r0.id, patch: { played: 'yes', side: 'home', oddsTaken: '-126', stake: '2', book: 'FanDuel', notes: '=HYPERLINK("http://x")' } });
    check('update: play saved', up.row && up.row.played === 'yes' && up.row.oddsTaken === -126 && up.row.stake === 2);
    var notesCol = __require('playlog').CSV_COLUMNS.indexOf('notes');
    var cell = ss.getSheetByName('PlayLog').data[1][notesCol];
    check('update: a note starting with "=" is stored as text, never as a formula', cell === '=HYPERLINK("http://x")', JSON.stringify(cell));
    check('update: bad odds refused, nothing written', !!api('update', { id: r0.id, patch: { oddsTaken: '50' } }).errors && api('log', {}).rows[0].oddsTaken === -126);
    check('update: unknown id -> clear error', api('update', { id: 'nope', patch: { notes: 'x' } }).error === 'No log entry with that id');

    // ---- the timer fills in the final ----
    var ref = api('refresh', {});
    var afterRefresh = api('log', {}).rows[0];
    check('refresh: final score from MLB Stats API, bet graded (home lost 5-3 at 2u -> -2u)',
      afterRefresh.finalAway === 5 && afterRefresh.finalHome === 3 && afterRefresh.result === 'L' && afterRefresh.plUnits === -2, JSON.stringify(ref));
    check('refresh: game already started -> no closing capture attempted', ref.closingChecked === 0);
    check('refresh: nothing left to do on the next run', api('refresh', {}).rowsUpdated === 0);

    // ---- CSV export ----
    var csv = doGet({ parameter: { export: 'csv' } });
    check('CSV export: download with every column and the row', csv.file === 'playlog_2026-09-18.csv' && csv.mime === 'text/csv' && csv.text.split('\r\n')[0].indexOf('id,runAt,gameDate') === 1 && csv.text.indexOf(r0.id) > 0);
    check('CSV export: the "=" note is quote-guarded', csv.text.indexOf("\"'=HYPERLINK(\"\"http://x\"\")\"") > 0);

    // ---- FanGraphs mode ----
    api('saveSettings', { useFangraphs: true });
    var miss = api('run', { gamePk: game.gamePk, date: '2026-09-18' });
    check('FanGraphs on, nothing uploaded -> lists the missing uploads and the no-FanGraphs option',
      miss.ok === false && miss.missing.some(function (m) { return /bullpen_2026/.test(m); }) && miss.missing.some(function (m) { return /switch off/.test(m); }), JSON.stringify(miss.missing));
    var codes = U.TEAMS.map(function (t) { return t.codes[0]; });
    var bullpenCsv = 'Team,IP,TBF,wOBA\n' + codes.map(function (c, i) { return c + ',' + (450 + i) + '.1,' + (1900 + i) + ',0.3' + (10 + i % 9); }).join('\n');
    var wrcCsv = 'Team,wRC+\n' + codes.map(function (c, i) { return c + ',' + (90 + i); }).join('\n');
    var frvCsv = 'Team,FRV\n' + codes.map(function (c, i) { return c + ',' + (i - 15); }).join('\n');
    check('upload: a file with too few teams is refused', /Only 1 of 30/.test(api('uploadFangraphs', { kind: 'bullpen', season: '2026', text: 'Team,IP,TBF,wOBA\nNYY,400,1700,0.31' }).error || ''));
    var u1 = api('uploadFangraphs', { kind: 'bullpen', season: '2026', text: bullpenCsv });
    api('uploadFangraphs', { kind: 'team_wrc', season: '2026', text: wrcCsv });
    api('uploadFangraphs', { kind: 'team_frv', season: '2026', text: frvCsv });
    check('upload: 30-team files accepted', u1.saved === 'bullpen_2026' && u1.teams === 30);
    var fetchesBefore = FETCHES.filter(function (u) { return /baseballsavant/.test(u); }).length;
    var run2 = api('run', { gamePk: game.gamePk, date: '2026-09-18', useFangraphs: true });
    check('run (FanGraphs on): uses the uploaded files', run2.ok && run2.result.inputs === 'fangraphs' && run2.result.cells.H17 > 0, run2.ok ? '' : JSON.stringify(run2.missing));
    check('run: Savant CSVs came from the cache the second time', FETCHES.filter(function (u) { return /baseballsavant/.test(u); }).length === fetchesBefore);
    check('run: FanGraphs changes the projection (bullpen, wRC+, FRV applied)', run2.ok && Math.abs(run2.result.projectedRuns.awayRaw - run.result.projectedRuns.awayRaw) > 1e-6);

    // ---- closing line: the timer captures every book once, within 10 minutes of first pitch ----
    var cols = __require('playlog').CSV_COLUMNS, sheetRows = ss.getSheetByName('PlayLog').data;
    var ri = sheetRows.findIndex(function (r) { return r[0] === run2.logRow.id; });
    var soon = new Date(Date.now() + 5 * 60 * 1000).toISOString();
    sheetRows[ri][cols.indexOf('startUtc')] = soon; oddsEvents[0].commence_time = soon;   // the feed lists the game at its new time too
    var oddsCalls = FETCHES.filter(function (u) { return /the-odds-api/.test(u); }).length;
    oddsMemo_ = null;
    var ref2 = api('refresh', {});
    var closedRow = api('log', {}).rows.filter(function (r) { return r.id === run2.logRow.id; })[0];
    var nv = function (a, h) { var L = __require('playlog'); return L.noVigAway(a, h); };
    check('closing: captured 5 minutes before first pitch, best prices + consensus no-vig of all 3 books',
      ref2.closingChecked === 1 && closedRow.close_source === 'feed' && closedRow.close_awayMl === 115 && closedRow.close_homeMl === -126 &&
      Math.abs(closedRow.close_nvAway - (nv(110, -130) + nv(115, -135) + nv(108, -126)) / 3) < 1e-9 && closedRow.close_book === 'best of 3 books', JSON.stringify(ref2));
    check('closing: one odds request for the capture', FETCHES.filter(function (u) { return /the-odds-api/.test(u); }).length === oddsCalls + 1);
    oddsMemo_ = null;
    check('closing: captured only once per game (saves odds credits)', api('refresh', {}).closingChecked === 0);
    oddsEvents[0].commence_time = game.gameDate; oddsMemo_ = null;   // put the feed back for later tests

    // ---- summary, delete ----
    var log2 = api('log', {});
    check('log: two runs of the same game -> two rows, one game in the summary', log2.rows.length === 2 && log2.summary.games === 1 && log2.summary.played.record === '0-1');
    check('delete: removes the entry', api('remove', { id: run2.logRow.id }).deleted === true && api('log', {}).rows.length === 1);
    // Pregame snapshot: the first pass runs "before first pitch" (clock set to the night before), the finals feed then
    // reports every scheduled game as final, and each final must carry that pregame projection.
    ROUTES.unshift({ match: /statsapi\.mlb\.com\/api\/v1\/schedule\?sportId=1&startDate=/, respond: function () {
      return { code: 200, text: JSON.stringify({ dates: [{ date: '2026-09-18', games: sched.dates[0].games.map(function (g) {
        return { gamePk: g.gamePk, status: { abstractGameState: 'Final' }, teams: { away: { score: 4, team: g.teams.away.team }, home: { score: 2, team: g.teams.home.team } } }; }) }] }) }; } });
    var realNow = Date.now; Date.now = function () { return Date.parse('2026-09-18T06:00:00Z'); };
    var sl;
    try { sl = api('slate', { date: '2026-09-18' }); } finally { Date.now = realNow; }
    var doc = JSON.parse(DRIVE['Bet Desk MLB']['mlb_slate.json']);
    var okg = doc.games.filter(function (g) { return g.status === 'ok'; });
    check('slate: every game on the date is in mlb_slate.json', sl.games === doc.games.length && doc.games.length > 5, sl.games + ' games, ' + okg.length + ' run');
    check('slate: run games carry the efficiency model (primary), v3, v3 + defense and v3 full, and no retired v2', doc.meta.primary === 'eff' && doc.meta.marketBlend.model === 0.0275 && okg.length > 0 && okg.every(function (g) { return !('v2' in g.models) && g.models.eff && g.models.eff.totals.length && g.inputs.home.efficiencySource && g.models.v3 && g.models.v3d && g.models.v3f && g.models.v3.totals.length; }));
    check('slate: inputs include efficiency, framing, handedness', okg.every(function (g) { return g.inputs.home.efficiency > 0 && g.inputs.home.framing != null && g.inputs.home.platoon != null; }));
    var fetches0 = FETCHES.length; api('slate', { date: '2026-09-18' });
    check('slate: second pass reuses cached games', FETCHES.length - fetches0 <= 2, (FETCHES.length - fetches0) + ' fetches');
    var okPks = okg.map(function (g) { return g.gamePk; }), withPre = doc.finals.filter(function (f) { return f.pregame; });
    check('slate: finals carry the last pregame projection of every run game', okPks.length > 0 && okPks.every(function (pk) {
      var f = doc.finals.filter(function (x) { return x.gamePk === pk; })[0], g = okg.filter(function (x) { return x.gamePk === pk; })[0];
      return f && f.pregame && f.pregame.models.v3.runsHome === g.models.v3.runsHome && f.pregame.models.eff.runsHome === g.models.eff.runsHome && !('v2' in f.pregame.models) && f.pregame.models.v3.total === g.models.v3.total &&
        f.pregame.models.v3f && f.pregame.models.v3d && f.awayRuns === 4; }), withPre.length + ' of ' + okPks.length);
    check('slate: games never run have pregame null', doc.finals.filter(function (f) { return okPks.indexOf(f.gamePk) < 0; }).every(function (f) { return f.pregame === null; }));
    var keep = PROPS['slate_pre_' + okPks[0]];
    PROPS['slate_pre_999'] = JSON.stringify({ startUtc: '2020-01-01T00:00:00Z' });
    api('slate', { date: '2026-09-18' });
    check('slate: after first pitch the pregame snapshot is kept, not overwritten', PROPS['slate_pre_' + okPks[0]] === keep);
    check('slate: snapshots older than 7 days are dropped', !('slate_pre_999' in PROPS));
    check('api: unknown call is refused', (function () { try { api('eval', {}); return false; } catch (e) { return /unknown call/.test(e.message); } })());
  } catch (e) {
    check('driver crashed', false, e.message + '\n' + e.stack);
  }
  window.GAS_RESULTS = results;
})();
