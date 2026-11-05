// Reads every upcoming slate game's market chances from Bet Desk itself ("Our model" mode, not blended with the line),
// so the promo optimizer prices game legs exactly as the app's cards do. Prints JSON to the path given (default stdout).
// node probe_games.js [out.json]      (same headless setup as models/grading/snap_games.js)
const http = require('http'), fs = require('fs'), path = require('path');
let chromium; try { ({ chromium } = require('playwright')); } catch (e) { ({ chromium } = require('/opt/node22/lib/node_modules/playwright')); }
const APP = path.resolve(__dirname, '../../app');
// injected into the app's own scope (its functions are not globals): run each slate game's card and keep every market row
const HOOK = `window.BD_PROMO = () => { BLEND = false; cfVer = 'v3'; let cap = null; const orig = gameRows;
  gameRows = function (o, rows, sport, extra, head) { if (PROBING) cap = rows.map(r => [...r]); return orig.apply(this, arguments); };
  const res = [];
  try { ['NFL', 'CFB', 'NBA', 'CBB'].forEach(sp => gsGames(sp).filter(x => !started(x) && x.ln).forEach(x => { cap = null; Object.keys(PROBE).forEach(k => delete PROBE[k]);
    const r = probeGames(sp, [{g:x.id, d:x.d, k:x.k, home:x.h, away:x.a, n:x.n||0, feed:x.ln}])[0];
    if (r && cap) res.push({sp, id:x.id, d:x.d, k:x.k, h:x.h, a:x.a, n:x.n||0, hs:+r.raw.hs.toFixed(2), as:+r.raw.as.toFixed(2), rows:cap, ln:x.ln}); })); }
  finally { gameRows = orig; }
  return {made: new Date().toISOString(), games: res}; };`;
const srv = http.createServer((q, r) => {
  const u = q.url.split('?')[0], f = u === '/' || u.startsWith('/index.html') ? path.join(APP, 'bet-desk.html') : path.join(APP, 'data', path.basename(u));
  fs.readFile(f, (e, b) => { if (e) { r.writeHead(404); r.end(); return; }
    if (f.endsWith('.html')) b = Buffer.from(String(b).replace('window.BD_SNAP = ', HOOK + '\nwindow.BD_SNAP = '));   // served copy only; bet-desk.html is untouched
    r.writeHead(200, { 'content-type': f.endsWith('.json') ? 'application/json' : 'text/html; charset=utf-8' }); r.end(b); });
});
srv.listen(0, '127.0.0.1', async () => {
  const port = srv.address().port, exe = ['/opt/pw-browsers/chromium'].find(fs.existsSync);
  const b = await chromium.launch(exe ? { executablePath: exe } : {});
  try {
    const p = await (await b.newContext({ timezoneId: 'America/New_York' })).newPage();
    await p.route(u => !u.href.startsWith(`http://127.0.0.1:${port}`), r => r.abort());
    await p.goto(`http://127.0.0.1:${port}/index.html`);
    await p.waitForFunction(() => window.BD_PROMO && document.querySelector('#gameList'), null, { timeout: 30000 });
    await p.waitForTimeout(1500);
    const out = await p.evaluate(() => window.BD_PROMO());
    const s = JSON.stringify(out);
    if (process.argv[2]) fs.writeFileSync(process.argv[2], s); else process.stdout.write(s);
    console.error('probed', out.games.length, 'games');
  } finally { await b.close(); srv.close(); }
});
