// Snapshot Bet Desk's line-based prop flags (J 2026-10-28) so they can be graded: did the flagged side win, and did the
// price beat the close. node snap_flags.js   (after the props files and app/data/odds/props_*.json are fresh)
// Serves app/bet-desk.html + app/data/*.json (+ app/data/odds/props_*.json) headless, calls window.BD_FLAGS() and merges the
// rows into history/FLAGS.json.gz keyed sp|game|player|stat|side. A flag is only (re)written before its kickoff, so the
// stored price is the last pregame flag the desk showed. Rows: sp, n, t, o, g, d, k, side, line, book, price, proj, ours
// (our chance), blend (50/50 with the books), market (books' no-vig), ev, evb, kind (proven | flag | check), kick, sweep.
const http = require('http'), fs = require('fs'), path = require('path'), zlib = require('zlib');
let chromium; try { ({ chromium } = require('playwright')); } catch (e) { ({ chromium } = require('/opt/node22/lib/node_modules/playwright')); }
const APP = path.resolve(__dirname, '../../app'), HIST = path.join(__dirname, 'history', 'FLAGS.json.gz');
const srv = http.createServer((q, r) => {
  const u = q.url.split('?')[0], bn = path.basename(u);
  const f = u === '/' || u.startsWith('/index.html') ? path.join(APP, 'bet-desk.html') : path.join(APP, 'data', bn.startsWith('props_') ? 'odds' : '', bn);
  fs.readFile(f, (e, b) => { if (e) { r.writeHead(404); r.end(); return; } r.writeHead(200, { 'content-type': f.endsWith('.json') ? 'application/json' : 'text/html; charset=utf-8' }); r.end(b); });
});
srv.listen(0, '127.0.0.1', async () => {
  const port = srv.address().port, exe = ['/opt/pw-browsers/chromium'].find(fs.existsSync);
  const b = await chromium.launch(exe ? { executablePath: exe } : {});
  try {
    const p = await (await b.newContext({ timezoneId: 'America/New_York' })).newPage();
    await p.route(u => !u.href.startsWith(`http://127.0.0.1:${port}`), r => r.abort());
    await p.goto(`http://127.0.0.1:${port}/index.html`);
    await p.waitForFunction(() => window.BD_FLAGS && document.querySelector('#flagOut'), null, { timeout: 30000 });
    await p.waitForTimeout(3000);
    const snap = await p.evaluate(() => window.BD_FLAGS());
    const H = fs.existsSync(HIST) ? JSON.parse(zlib.gunzipSync(fs.readFileSync(HIST))) : {};
    const now = Date.now(); let n = 0;
    snap.rows.forEach(r => { if (!r.kick || Date.parse(r.kick) > now) { H[[r.sp, r.g, r.n, r.k, r.side].join('|')] = { ...r, made: snap.made }; n++; } });
    fs.mkdirSync(path.dirname(HIST), { recursive: true });
    fs.writeFileSync(HIST, zlib.gzipSync(JSON.stringify(H)));
    const by = {}; snap.rows.forEach(r => { const k = r.sp + ' ' + r.kind; by[k] = (by[k] || 0) + 1; });
    console.log('flag snapshot', n, 'saved', JSON.stringify(by), 'history', Object.keys(H).length);
  } finally { await b.close(); srv.close(); }
});
