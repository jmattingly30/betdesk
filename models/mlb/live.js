#!/usr/bin/env node
// MLB on the desk (2026-11-25, J: "move MLB fully onto the desk"). Runs J's live Apps Script code UNCHANGED
// (models/mlb/app/apps_script/Bundle.gs + Server.gs, byte-identical to J's deployed "MLB Win Probability" project)
// under Node, with Google's services replaced by plain equivalents:
//   UrlFetchApp        -> curl (same User-Agent the script sends)
//   CacheService       -> memory for this run
//   PropertiesService  -> models/mlb/state.json (pregame projections for grading, cached past-season Baseball-Reference)
//   DriveApp file      -> app/data/mlb_slate.json (what the Drive folder "Bet Desk MLB" held)
// Bullpens (J 2026-11-25: "if we can get the bullpen stats in without fangraphs that would be terrific"): J's app has a
// "use FanGraphs data" switch (bullpen IP/TBF/wOBA against, team wRC+, team FRV, uploaded by hand). FanGraphs blocks
// automated requests, so this runner turns the switch ON and fills the bullpen file from MLB Stats API's official
// reliever split for each team (see bullpenCsv). wRC+ is filled with 100 and FRV with 0 for every team, which J's code
// treats as exact no-ops, so the only new input is the bullpen. A FanGraphs export J saves as
// models/mlb/uploads/{bullpen|team_wrc|team_frv}_{season}.csv replaces the stand-in for that kind and season.
// MLB_BULLPEN=off reproduces J's live app exactly (no-FanGraphs mode, league-average bullpens); the parity check uses it.
// exportSlate() stops starting new games after 4.5 minutes (Apps Script limit) and marks the rest pending; this runner
// calls it again (the in-memory cache keeps finished games) until nothing is pending, like J's 30-minute timer did.
// Usage: node models/mlb/live.js [YYYY-MM-DD]   (default: today in the script's time zone, America/Denver)
'use strict';
const fs = require('fs'), path = require('path'), vm = require('vm'), crypto = require('crypto');
const { execFileSync } = require('child_process');

const HERE = __dirname, REPO = path.join(HERE, '..', '..');
const APP = path.join(HERE, 'app', 'apps_script');
const STATE = path.join(HERE, 'state.json');
const OUT = process.env.MLB_SLATE_OUT || path.join(REPO, 'app', 'data', 'mlb_slate.json');
const TZ = JSON.parse(fs.readFileSync(path.join(APP, 'appsscript.json'), 'utf8')).timeZone || 'America/Denver';
const BULLPEN_ON = process.env.MLB_BULLPEN !== 'off';
const UPLOADS = path.join(HERE, 'uploads');

let PROPS = {};
if (fs.existsSync(STATE)) PROPS = JSON.parse(fs.readFileSync(STATE, 'utf8'));
delete PROPS.ODDS_API_KEY;   // never stored here
PROPS.SETTINGS = JSON.stringify({ useFangraphs: BULLPEN_ON });
const CACHE = {}, LOG = [];
let fetches = 0, fetchFails = 0;

function curl(url, headers) {
  const args = ['-sS', '-L', '--compressed', '--max-time', '90', '--retry', '3', '--retry-delay', '4', '-w', '\n%{http_code}'];
  for (const [k, v] of Object.entries(headers || {})) args.push('-H', `${k}: ${v}`);
  args.push(url);
  fetches++;
  try {
    const out = execFileSync('curl', args, { maxBuffer: 512 * 1024 * 1024, encoding: 'utf8', stdio: ['ignore', 'pipe', 'pipe'] });
    const i = out.lastIndexOf('\n');
    return { code: Number(out.slice(i + 1)) || 0, text: out.slice(0, i) };
  } catch (e) { return { code: 0, text: String(e.stderr || e.message || e) }; }
}

const ctx = {
  console, JSON, Math, Date, TextDecoder,
  Logger: { log: m => { LOG.push(String(m)); console.log('[apps script] ' + m); } },
  UrlFetchApp: { fetch(url, opt) {
    const r = curl(url, opt && opt.headers);
    if (r.code !== 200) fetchFails++;
    return { getResponseCode: () => r.code, getContentText: () => r.text, getHeaders: () => ({}) };
  } },
  CacheService: { getScriptCache: () => ({
    get: k => (Object.prototype.hasOwnProperty.call(CACHE, k) ? CACHE[k] : null),
    put: (k, v) => { if (String(v).length > 100000) throw new Error('cache value too large'); CACHE[k] = String(v); } }) },
  PropertiesService: { getScriptProperties: () => ({
    getProperty: k => (Object.prototype.hasOwnProperty.call(PROPS, k) ? PROPS[k] : null),
    setProperty: (k, v) => { PROPS[k] = String(v); },
    getProperties: () => Object.assign({}, PROPS),
    deleteProperty: k => { delete PROPS[k]; } }) },
  Session: { getScriptTimeZone: () => TZ },
  Utilities: {
    sleep: ms => { execFileSync('sleep', [String(ms / 1000)]); },
    formatDate(d, tz, fmt) {
      if (fmt !== 'yyyy-MM-dd') throw new Error('formatDate: unsupported format ' + fmt);
      return new Intl.DateTimeFormat('en-CA', { timeZone: tz, year: 'numeric', month: '2-digit', day: '2-digit' }).format(d);
    },
    DigestAlgorithm: { MD5: 'md5' },
    computeDigest: (alg, s) => Array.from(crypto.createHash(alg).update(String(s)).digest()),
    base64EncodeWebSafe: b => Buffer.from(b.map(x => x & 255)).toString('base64').replace(/\+/g, '-').replace(/\//g, '_'),
  },
  DriveApp: (() => {
    const file = { setContent: t => { fs.mkdirSync(path.dirname(OUT), { recursive: true }); fs.writeFileSync(OUT, t); } };
    const folder = { getFilesByName: () => { let used = false; return { hasNext: () => !used, next: () => { used = true; return file; } }; },
      createFile: (n, t) => file.setContent(t) };
    return { getFoldersByName: () => { let used = false; return { hasNext: () => !used, next: () => { used = true; return folder; } }; },
      createFolder: () => folder };
  })(),
};
ctx.window = ctx; vm.createContext(ctx);
for (const f of ['Bundle.gs', 'Server.gs']) vm.runInContext(fs.readFileSync(path.join(APP, f), 'utf8'), ctx, { filename: f });

// ---------- bullpens from MLB Stats API (stand-in for the FanGraphs bullpen upload) ----------
// Each team's reliever split (sitCodes=rp: every PA a pitcher faced in relief) -> IP, batters faced, wOBA against.
// wOBA uses standard linear weights, rescaled each season so the league's wOBA from these counts equals Savant's league
// wOBA (the scale every other wOBA in J's model is on). No FanGraphs constants needed.
const W = { ubb: 0.689, hbp: 0.720, b1: 0.882, b2: 1.254, b3: 1.590, hr: 2.050 };
const json = url => { const r = curl(url, { 'User-Agent': ctx.UA_ || 'Mozilla/5.0' }); if (r.code !== 200) throw new Error(`HTTP ${r.code} for ${url}`); return JSON.parse(r.text); };
const raw = st => {
  const ubb = st.baseOnBalls - (st.intentionalWalks || 0), b1 = st.hits - st.doubles - st.triples - st.homeRuns;
  const den = st.atBats + ubb + (st.sacFlies || 0) + (st.hitByPitch || 0);
  return { num: W.ubb * ubb + W.hbp * (st.hitByPitch || 0) + W.b1 * b1 + W.b2 * st.doubles + W.b3 * st.triples + W.hr * st.homeRuns, den };
};
const bullpenMemo = {}, bullpenNotes = [];
function bullpenCsv(season) {
  if (bullpenMemo[season]) return bullpenMemo[season];
  const base = 'https://statsapi.mlb.com/api/v1/teams/stats?group=pitching&sportIds=1&gameType=R&season=' + season;
  const rp = json(base + '&stats=statSplits&sitCodes=rp').stats[0].splits;
  const all = json(base + '&stats=season').stats[0].splits;
  if (rp.length < 30 || all.length < 30) throw new Error(`MLB Stats API has ${rp.length} reliever / ${all.length} team rows for ${season}`);
  let n = 0, d = 0; for (const t of all) { const x = raw(t.stat); n += x.num; d += x.den; }
  const sv = curl(`https://baseballsavant.mlb.com/leaderboard/custom?year=${season}&type=pitcher&filter=&min=1&selections=pa,woba&chart=false&x=pa&y=pa&r=no&chartType=beeswarm&csv=true`);
  let lgSavant = null;
  if (sv.code === 200) {
    const rows = sv.text.replace(/^\ufeff/, '').trim().split('\n').map(l => l.match(/("[^"]*"|[^,]+)/g) || []);
    const h = rows[0].map(x => x.replace(/"/g, '')), ip = h.indexOf('pa'), iw = h.indexOf('woba');
    let pa = 0, w = 0; for (const r of rows.slice(1)) { const a = +String(r[ip]).replace(/"/g, ''), b = +String(r[iw]).replace(/"/g, ''); if (a > 0 && b >= 0) { pa += a; w += a * b; } }
    if (pa > 10000) lgSavant = w / pa;
  }
  const scale = lgSavant ? lgSavant / (n / d) : 1;
  const lines = ['Team,IP,TBF,wOBA'];
  for (const t of rp) { const x = raw(t.stat); lines.push(`"${t.team.name}",${t.stat.inningsPitched},${t.stat.battersFaced},${(scale * x.num / x.den).toFixed(4)}`); }
  bullpenNotes.push(`${season}: ${rp.length} bullpens from MLB Stats API, league wOBA ${(n / d).toFixed(3)} raw -> ${lgSavant ? lgSavant.toFixed(3) + ' Savant' : 'unscaled (Savant league wOBA unavailable)'}`);
  if (process.env.MLB_SHOW_BULLPEN) console.log(lines.join('\n'));
  return (bullpenMemo[season] = lines.join('\n'));
}
const teamConst = (col, v) => ['Team,' + col].concat(vm.runInContext('R_("util").TEAMS', ctx).map(t => `"${t.name}",${v}`)).join('\n');
ctx.__fgGet = (kind, season) => {
  const up = path.join(UPLOADS, `${kind}_${season}.csv`);
  if (fs.existsSync(up)) return { uploadedAt: fs.statSync(up).mtimeMs, text: fs.readFileSync(up, 'utf8') };
  const text = kind === 'bullpen' ? bullpenCsv(season) : kind === 'team_wrc' ? teamConst('wRC+', 100) : kind === 'team_frv' ? teamConst('FRV', 0) : null;
  return text == null ? null : { uploadedAt: Date.now(), text };
};
if (BULLPEN_ON) vm.runInContext('FgStore_.get = function (kind, season) { return __fgGet(kind, String(season)); };', ctx);

const arg = process.argv[2];
if (arg && !/^\d{4}-\d{2}-\d{2}$/.test(arg)) { console.error('usage: node models/mlb/live.js [YYYY-MM-DD]'); process.exit(2); }
let doc, pass = 0;
do { pass++; ctx.__date = arg || ''; doc = vm.runInContext('exportSlate(__date)', ctx); }
while (doc.pending > 0 && pass < 6);

fs.writeFileSync(STATE, JSON.stringify(PROPS, Object.keys(PROPS).sort(), 0) + '\n');
const st = {}; for (const g of doc.games) st[g.status] = (st[g.status] || 0) + 1;
console.log(`MLB slate ${doc.date}: ${doc.games.length} games ${JSON.stringify(st)}, ${doc.finals.length} finals, ${pass} pass(es), ${fetches} fetches (${fetchFails} not 200)`);
console.log('bullpens: ' + (BULLPEN_ON ? (bullpenNotes.join('; ') || 'no games needed them') : 'off (league average, as J\'s live app)'));
for (const g of doc.games) if (g.status !== 'ok') console.log(`  ${g.away} @ ${g.home}: ${g.status} ${(g.notes || []).slice(0, 2).join(' | ')}`);
const warn = new Set(); for (const g of doc.games) for (const w of g.warnings || []) warn.add(w.replace(/\d+(\.\d+)?/g, '#'));
for (const w of warn) console.log('  warning: ' + w);
if (doc.games.some(g => g.status === 'error')) process.exitCode = 1;
