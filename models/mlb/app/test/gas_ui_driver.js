// Clicks through the real page (apps_script/Index.html) served by the Apps Script server on fake Google
// services: look up -> confirm -> tap a book's price -> mark played -> save -> open the log.
// Runs after gas_driver.js (which leaves one logged game and the fake data routes in place).
(function () {
  var results = [];
  function check(name, ok, detail) { results.push({ name: name, ok: !!ok, detail: detail === undefined ? '' : String(detail) }); }
  var wait = function (ms) { return new Promise(function (r) { setTimeout(r, ms); }); };
  async function until(fn, ms) { for (var t = 0; t < (ms || 5000); t += 50) { var v = fn(); if (v) return v; await wait(50); } return null; }
  var byText = function (sel, re) { return Array.from(document.querySelectorAll(sel)).filter(function (e) { return re.test(e.textContent); }); };
  (async function () {
    try {
      await wait(200);
      check('page loads; FanGraphs switch drawn from saved settings', byText('#fgSeg button.on', /Use FanGraphs/).length === 1);
      byText('#fgSeg button', /No FanGraphs/)[0].click();
      await wait(100);
      check('switching FanGraphs off saves the setting on the server', JSON.parse(PROPS.SETTINGS).useFangraphs === false);
      document.getElementById('team').value = 'Reds';
      document.getElementById('date').value = '2026-09-18';   // the recorded schedule's date (blank would mean today)
      document.getElementById('lookup').click();
      var confirmBtn = await until(function () { return byText('button', /Yes, confirm/)[0]; });
      check('lookup shows the game with the confirm button', !!confirmBtn);
      confirmBtn.click();
      var booksCard = await until(function () { return byText('.card h3', /Every book \(3\)/)[0]; });
      check('run shows every book with the best prices highlighted', !!booksCard && document.querySelectorAll('td.best').length === 2);
      check('run shows all four win probabilities', byText('td', /^Pythagenpat, season R\/G/).length === 1 && byText('td', /^Hierarchical Monte Carlo$/).length === 1);
      check('run reports no-FanGraphs mode', byText('.msg.warn', /No-FanGraphs mode/).length === 1);
      var dk = byText('td.tap', /^\+115$/)[0];
      dk.click();
      await wait(50);
      var form = byText('.card', /Saved to your play log/).pop();
      var oddsInputs = form.querySelectorAll('input[aria-label="Odds you took"]');
      check('tapping a book fills side, odds and book', oddsInputs[0].value === '115' && form.querySelector('select[aria-label="Sportsbook"]').value === 'DraftKings' &&
        byText('.seg button.on', new RegExp(document.querySelector('.card .big').textContent.split(' @ ')[0])).length >= 1);
      byText('button', /^Played$/).pop().click();
      form.querySelector('textarea').value = 'shopped DK';
      byText('button', /^Save$/).pop().click();
      var saved = await until(function () { return byText('.msg.ok', /^Saved\./)[0]; });
      check('saving the play works', !!saved, saved ? saved.textContent : '');
      var rows = SERVER_API('log', {}).rows;
      var mine = rows[0];
      check('the play is in the Sheet: away at +115, DraftKings, notes', mine.played === 'yes' && mine.side === 'away' && mine.oddsTaken === 115 && mine.book === 'DraftKings' && mine.notes === 'shopped DK', JSON.stringify(mine).slice(0, 300));
      check('market odds from the feed were not overwritten by the form', mine.mkt_source === 'feed' && mine.mkt_awayMl === 115);
      document.getElementById('tabLog').click();
      var cards = await until(function () { var c = document.querySelectorAll('#logView .card .big'); return c.length ? c : null; });
      check('log tab lists the entries with the summary', cards && cards.length === rows.length && byText('#logView .stat', /Games you played/).length === 1);
      check('log tab offers the CSV download at the web app URL', !!document.querySelector('#logView a.btn[href$="?export=csv"]'));
      document.getElementById('tabRun').click();
    } catch (e) { check('UI driver crashed', false, e.message + ' ' + e.stack); }
    window.UI_RESULTS = results;
    document.title = 'UI_DONE';
  })();
})();
