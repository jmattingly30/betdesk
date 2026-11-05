// Snapshot every upcoming game's projection from Bet Desk itself, so game grading scores exactly what the app showed.
// node snap_games.js            (after app/export_slates.py; grade.py grades the snapshots once games are final)
// Serves app/bet-desk.html + app/data/*.json locally, opens it in headless Chromium, calls window.BD_SNAP() and merges
// the rows into history/GAMES.json.gz. A game is only (re)written before its kickoff, so a projection is never
// replaced after the fact.
const http = require('http'), fs = require('fs'), path = require('path'), zlib = require('zlib');
let chromium; try { ({ chromium } = require('playwright')); } catch (e) { ({ chromium } = require('/opt/node22/lib/node_modules/playwright')); }
const APP = path.resolve(__dirname, '../../app'), HIST = path.join(__dirname, 'history', 'GAMES.json.gz');
const srv = http.createServer((q, r) => {
  const f = q.url.split('?')[0] === '/' || q.url.startsWith('/index.html') ? path.join(APP, 'bet-desk.html') : path.join(APP, 'data', path.basename(q.url.split('?')[0]));
  fs.readFile(f, (e, b) => { if (e) { r.writeHead(404); r.end(); return; } r.writeHead(200, { 'content-type': f.endsWith('.json') ? 'application/json' : 'text/html; charset=utf-8' }); r.end(b); });
});
srv.listen(0, '127.0.0.1', async () => {
  const port = srv.address().port, exe = ['/opt/pw-browsers/chromium'].find(fs.existsSync);
  const b = await chromium.launch(exe ? { executablePath: exe } : {});
  try {
    const p = await (await b.newContext({ timezoneId: 'America/New_York' })).newPage();
    await p.route(u => !u.href.startsWith(`http://127.0.0.1:${port}`), r => r.abort());
    await p.goto(`http://127.0.0.1:${port}/index.html`);
    await p.waitForFunction(() => window.BD_SNAP && document.querySelector('#gameList'), null, { timeout: 30000 });
    await p.waitForTimeout(1500);
    const snap = await p.evaluate(() => window.BD_SNAP());
    const H = fs.existsSync(HIST) ? JSON.parse(zlib.gunzipSync(fs.readFileSync(HIST))) : {};
    const now = Date.now(); let n = 0;
    snap.rows.forEach(r => { if (!r.k || Date.parse(r.k) > now) { H[r.sp + '|' + r.id] = { ...r, made: snap.made, ratings: { cfb: snap.cfb, games: snap.games, cbb: snap.cbb } }; n++; } });
    fs.mkdirSync(path.dirname(HIST), { recursive: true });
    fs.writeFileSync(HIST, zlib.gzipSync(JSON.stringify(H)));
    const by = {}; snap.rows.forEach(r => by[r.sp] = (by[r.sp] || 0) + 1);
    console.log('game snapshot', n, 'saved', JSON.stringify(by), 'history', Object.keys(H).length);
  } finally { await b.close(); srv.close(); }
});
