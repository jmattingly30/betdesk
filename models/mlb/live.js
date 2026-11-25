#!/usr/bin/env node
// MLB on the desk (2026-11-25, J: "move MLB fully onto the desk"). Runs J's live Apps Script code UNCHANGED
// (models/mlb/app/apps_script/Bundle.gs + Server.gs, byte-identical to J's deployed "MLB Win Probability" project)
// under Node, with Google's services replaced by plain equivalents:
//   UrlFetchApp        -> curl (same User-Agent the script sends)
//   CacheService       -> memory for this run
//   PropertiesService  -> models/mlb/state.json (pregame projections for grading, cached past-season Baseball-Reference)
//   DriveApp file      -> app/data/mlb_slate.json (what the Drive folder "Bet Desk MLB" held)
// Settings match J's live app: SETTINGS.useFangraphs = false (his play log shows every game ran "no-fangraphs").
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
const DEFAULT_PROPS = { SETTINGS: JSON.stringify({ useFangraphs: false }) };

let PROPS = Object.assign({}, DEFAULT_PROPS);
if (fs.existsSync(STATE)) PROPS = Object.assign(PROPS, JSON.parse(fs.readFileSync(STATE, 'utf8')));
delete PROPS.ODDS_API_KEY;   // never stored here
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

const arg = process.argv[2];
if (arg && !/^\d{4}-\d{2}-\d{2}$/.test(arg)) { console.error('usage: node models/mlb/live.js [YYYY-MM-DD]'); process.exit(2); }
let doc, pass = 0;
do { pass++; ctx.__date = arg || ''; doc = vm.runInContext('exportSlate(__date)', ctx); }
while (doc.pending > 0 && pass < 6);

fs.writeFileSync(STATE, JSON.stringify(PROPS, Object.keys(PROPS).sort(), 0) + '\n');
const st = {}; for (const g of doc.games) st[g.status] = (st[g.status] || 0) + 1;
console.log(`MLB slate ${doc.date}: ${doc.games.length} games ${JSON.stringify(st)}, ${doc.finals.length} finals, ${pass} pass(es), ${fetches} fetches (${fetchFails} not 200)`);
for (const g of doc.games) if (g.status !== 'ok') console.log(`  ${g.away} @ ${g.home}: ${g.status} ${(g.notes || []).slice(0, 2).join(' | ')}`);
const warn = new Set(); for (const g of doc.games) for (const w of g.warnings || []) warn.add(w.replace(/\d+(\.\d+)?/g, '#'));
for (const w of warn) console.log('  warning: ' + w);
if (doc.games.some(g => g.status === 'error')) process.exitCode = 1;
