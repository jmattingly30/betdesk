'use strict';
// Shared helpers: team lookup, CSV parsing, number/IP parsing.

// [mlbId, full name, nickname, source codes (FanGraphs/BBRef/Savant), extra aliases]
const RAW_TEAMS = [
  [108, 'Los Angeles Angels', 'Angels', ['LAA', 'ANA'], ['la angels', 'anaheim angels']],
  [109, 'Arizona Diamondbacks', 'Diamondbacks', ['ARI', 'AZ'], ['dbacks', 'd-backs', 'snakes']],
  [110, 'Baltimore Orioles', 'Orioles', ['BAL'], ['os', "o's", 'birds']],
  [111, 'Boston Red Sox', 'Red Sox', ['BOS'], ['redsox']],
  [112, 'Chicago Cubs', 'Cubs', ['CHC'], []],
  [113, 'Cincinnati Reds', 'Reds', ['CIN'], []],
  [114, 'Cleveland Guardians', 'Guardians', ['CLE'], ['indians']],
  [115, 'Colorado Rockies', 'Rockies', ['COL'], []],
  [116, 'Detroit Tigers', 'Tigers', ['DET'], []],
  [117, 'Houston Astros', 'Astros', ['HOU'], ['stros']],
  [118, 'Kansas City Royals', 'Royals', ['KCR', 'KC', 'KCA'], []],
  [119, 'Los Angeles Dodgers', 'Dodgers', ['LAD'], ['la dodgers']],
  [120, 'Washington Nationals', 'Nationals', ['WSN', 'WSH', 'WAS'], ['nats']],
  [121, 'New York Mets', 'Mets', ['NYM'], []],
  [133, 'Athletics', 'Athletics', ['ATH', 'OAK'], ['as', "a's", 'oakland athletics', 'oakland']],
  [134, 'Pittsburgh Pirates', 'Pirates', ['PIT'], []],
  [135, 'San Diego Padres', 'Padres', ['SDP', 'SD'], []],
  [136, 'Seattle Mariners', 'Mariners', ['SEA'], []],
  [137, 'San Francisco Giants', 'Giants', ['SFG', 'SF'], []],
  [138, 'St. Louis Cardinals', 'Cardinals', ['STL'], ['cards', 'st louis']],
  [139, 'Tampa Bay Rays', 'Rays', ['TBR', 'TB'], []],
  [140, 'Texas Rangers', 'Rangers', ['TEX'], []],
  [141, 'Toronto Blue Jays', 'Blue Jays', ['TOR'], ['jays', 'bluejays']],
  [142, 'Minnesota Twins', 'Twins', ['MIN'], []],
  [143, 'Philadelphia Phillies', 'Phillies', ['PHI'], ['phils']],
  [144, 'Atlanta Braves', 'Braves', ['ATL'], []],
  [145, 'Chicago White Sox', 'White Sox', ['CHW', 'CWS', 'CHA'], ['whitesox']],
  [146, 'Miami Marlins', 'Marlins', ['MIA', 'FLA'], []],
  [147, 'New York Yankees', 'Yankees', ['NYY'], ['yanks']],
  [158, 'Milwaukee Brewers', 'Brewers', ['MIL'], []],
];

function normKey(s) {
  return String(s == null ? '' : s)
    .normalize('NFD').replace(/[̀-ͯ]/g, '')
    .toLowerCase()
    .replace(/[‘’'`.]/g, '')
    .replace(/[^a-z0-9 -]/g, ' ')
    .replace(/\s+/g, ' ')
    .trim();
}

const TEAMS = RAW_TEAMS.map(([id, name, nick, codes, aliases]) => ({ id, name, nick, codes, aliases }));
const TEAM_BY_ID = new Map(TEAMS.map(t => [t.id, t]));
const TEAM_KEYS = new Map();
for (const t of TEAMS) {
  for (const k of [t.name, t.nick, ...t.codes, ...t.aliases]) TEAM_KEYS.set(normKey(k), t.id);
}

// Returns a team object, or null if the text doesn't identify exactly one franchise.
function findTeam(query) {
  const k = normKey(query);
  if (!k) return null;
  if (TEAM_KEYS.has(k)) return TEAM_BY_ID.get(TEAM_KEYS.get(k));
  const hits = TEAMS.filter(t => normKey(t.name).includes(k) || normKey(t.nick).includes(k));
  return hits.length === 1 ? hits[0] : null;
}

function normName(s) {
  return normKey(s).replace(/\b(jr|sr|ii|iii|iv)\b/g, '').replace(/-/g, ' ').replace(/\s+/g, ' ').trim();
}

function parseCsv(text) {
  text = String(text).replace(/^﻿/, '');
  const rows = [];
  let row = [], field = '', inQ = false;
  for (let i = 0; i < text.length; i++) {
    const c = text[i];
    if (inQ) {
      if (c === '"') { if (text[i + 1] === '"') { field += '"'; i++; } else inQ = false; }
      else field += c;
    } else if (c === '"') inQ = true;
    else if (c === ',') { row.push(field); field = ''; }
    else if (c === '\n' || c === '\r') {
      if (c === '\r' && text[i + 1] === '\n') i++;
      row.push(field); field = '';
      if (row.length > 1 || row[0] !== '') rows.push(row);
      row = [];
    } else field += c;
  }
  if (field !== '' || row.length) { row.push(field); if (row.length > 1 || row[0] !== '') rows.push(row); }
  return rows;
}

function csvToObjects(text) {
  const rows = parseCsv(text);
  if (!rows.length) return { headers: [], rows: [] };
  const headers = rows[0].map(h => h.trim());
  return {
    headers,
    rows: rows.slice(1).map(r => { const o = {}; headers.forEach((h, i) => { o[h] = r[i] === undefined ? '' : r[i]; }); return o; }),
  };
}

const hkey = s => String(s).toLowerCase().replace(/[^a-z0-9+]/g, '');

// Find the first header (case/punctuation-insensitive) matching any alias; returns the header text or null.
function findHeader(headers, aliases) {
  const want = aliases.map(hkey);
  return headers.find(h => want.includes(hkey(h))) || null;
}

function num(x) {
  if (typeof x === 'number') return x;
  const s = String(x == null ? '' : x).trim().replace(/,/g, '').replace(/%$/, '');
  if (s === '' || s === '-' || s === '--') return NaN;
  const v = parseFloat(s);
  return Number.isFinite(v) ? v : NaN;
}

// Baseball notation ("397.2" = 397 2/3 innings) -> true decimal. Values already in decimal
// form (e.g. "98.6666") pass through unchanged.
function ipToDecimal(x) {
  const s = String(x == null ? '' : x).trim();
  const m = s.match(/^(\d+)\.(\d+)$/);
  if (m && m[2].length === 1 && '012'.includes(m[2])) return parseInt(m[1], 10) + parseInt(m[2], 10) / 3;
  return num(s);
}

function wavg(items, valKey, wKey) {
  let sw = 0, sv = 0;
  for (const it of items) {
    const w = it[wKey], v = it[valKey];
    if (Number.isFinite(w) && Number.isFinite(v)) { sw += w; sv += v * w; }
  }
  return sw > 0 ? sv / sw : NaN;
}

// No-vig American moneyline for a win probability - the sheet's "Predicted Odds" cell (Tester H53/J53).
function fairMoneyline(p) {
  if (p > 0.5) return -100 * p / (1 - p);
  if (p < 0.5) return 100 * (1 - p) / p;
  return 100;
}

module.exports = { TEAMS, TEAM_BY_ID, findTeam, normKey, normName, parseCsv, csvToObjects, findHeader, num, ipToDecimal, wavg, fairMoneyline };
