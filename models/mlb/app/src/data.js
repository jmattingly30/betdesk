'use strict';
// Parsers for every data source, plus the season-window rule and season blending.
// All functions here are pure (text in, objects out) so they can be tested without a network.

const { csvToObjects, findHeader, num, ipToDecimal, wavg, findTeam, normName } = require('./util');

// ---------- season window rule ----------
// Returns the seasons of stats to use, oldest first.
function seasonsNeeded(currentSeason, currentGames, cfg) {
  return currentGames >= cfg.sampleWindow.minCurrentSeasonGames ? [currentSeason] : [currentSeason - 1, currentSeason];
}

// ---------- Savant (automatic CSV download) ----------
function parseSavantPlayers(csvText) {
  const { headers, rows } = csvToObjects(csvText);
  const nameH = findHeader(headers, ['last_name, first_name']);
  const idH = findHeader(headers, ['player_id']);
  const ipH = findHeader(headers, ['p_formatted_ip']);
  if (!nameH || !idH) throw new Error('Savant CSV is missing the name/player_id columns');
  return rows.map(r => {
    const [last, first] = String(r[nameH]).split(',').map(s => s.trim());
    return { id: num(r[idH]), name: first ? first + ' ' + last : last, pa: num(r.pa), woba: num(r.woba), xwoba: num(r.xwoba), ip: ipH ? ipToDecimal(r[ipH]) : NaN,
      k: 'k_percent' in r ? num(r.k_percent) : NaN, bb: 'bb_percent' in r ? num(r.bb_percent) : NaN,
      // Model v3 hitter power: ISO (Savant isolated_power, else SLG - AVG) over at-bats (else ~0.89 x PA)
      iso: 'isolated_power' in r ? num(r.isolated_power) : ('slg_percent' in r && 'batting_avg' in r ? num(r.slg_percent) - num(r.batting_avg) : NaN),
      ab: 'ab' in r ? num(r.ab) : NaN };
  }).filter(p => Number.isFinite(p.id));
}

// ---------- Baseball-Reference team batting table (games, PA, runs) ----------
function parseBbrefTeams(html) {
  const start = html.indexOf('id="teams_standard_batting"');
  if (start < 0) throw new Error('Baseball-Reference page has no teams_standard_batting table (layout changed or blocked)');
  const table = html.slice(start, html.indexOf('</table>', start));
  const out = new Map();
  const rowRe = /<tr[^>]*>\s*<th[^>]*data-stat="team_name"[^>]*>([\s\S]*?)<\/th>([\s\S]*?)<\/tr>/g;
  let m;
  while ((m = rowRe.exec(table))) {
    const team = findTeam(m[1].replace(/<[^>]*>/g, '').trim());
    if (!team) continue; // "League Average" row etc.
    const cells = {};
    const cellRe = /<td[^>]*data-stat="([^"]+)"[^>]*>([\s\S]*?)<\/td>/g;
    let c;
    while ((c = cellRe.exec(m[2]))) cells[c[1]] = c[2].replace(/<[^>]*>/g, '').trim();
    // J's run-efficiency rule (2026-10-16) also needs the team's wOBA, built from the same table
    const n = k => num(cells[k]), h1 = n('H') - n('2B') - n('3B') - n('HR'), ibb = Number.isFinite(n('IBB')) ? n('IBB') : 0;
    const wobaDen = n('AB') + n('BB') - ibb + n('SF') + n('HBP');
    const wobaNum = 0.69 * (n('BB') - ibb) + 0.72 * n('HBP') + 0.89 * h1 + 1.27 * n('2B') + 1.62 * n('3B') + 2.10 * n('HR');
    out.set(team.id, { games: num(cells.G), pa: num(cells.PA), runs: num(cells.R),
      wobaNum: Number.isFinite(wobaNum) ? wobaNum : NaN, wobaDen: Number.isFinite(wobaDen) && wobaDen > 0 ? wobaDen : NaN });
  }
  return out;
}

// ---------- FanGraphs hand-exported CSVs ----------
function parseFgBatters(csvText) {
  const { headers, rows } = csvToObjects(csvText);
  const h = {
    name: findHeader(headers, ['Name', 'NameASCII', 'Player']), team: findHeader(headers, ['Team', 'Tm']),
    pa: findHeader(headers, ['PA']), woba: findHeader(headers, ['wOBA']), xwoba: findHeader(headers, ['xwOBA']),
    id: findHeader(headers, ['MLBAMID', 'xMLBAMID', 'mlbam_id', 'mlbamid']),
  };
  const missing = ['name', 'pa', 'woba', 'xwoba'].filter(k => !h[k]);
  if (missing.length) throw new Error('FanGraphs batters CSV is missing column(s): ' + missing.join(', ') + ' (found: ' + headers.join(', ') + ')');
  return rows.map(r => ({ id: h.id ? num(r[h.id]) : NaN, name: String(r[h.name]).trim(), team: h.team ? r[h.team] : '', pa: num(r[h.pa]), woba: num(r[h.woba]), xwoba: num(r[h.xwoba]) }))
    .filter(p => p.name);
}

// Team-level tables (bullpen IP/BF/wOBA, team wRC+, team FRV). `spec` maps output keys to header aliases.
// Several rows for the same team are summed (counting stats) or weighted (rates) as noted by spec.kind.
function parseTeamTable(csvText, spec) {
  const { headers, rows } = csvToObjects(csvText);
  const teamH = findHeader(headers, ['Team', 'Tm', 'Name']);
  if (!teamH) throw new Error('Team CSV has no Team column (found: ' + headers.join(', ') + ')');
  const cols = {};
  for (const [key, aliases] of Object.entries(spec)) cols[key] = findHeader(headers, aliases);
  const out = new Map(), unknown = [];
  for (const r of rows) {
    const t = findTeam(r[teamH]);
    if (!t) { if (String(r[teamH]).trim()) unknown.push(String(r[teamH]).trim()); continue; }
    const rec = out.get(t.id) || {};
    for (const key of Object.keys(spec)) {
      if (!cols[key]) continue;
      const v = key === 'ip' ? ipToDecimal(r[cols[key]]) : num(r[cols[key]]);
      rec[key] = key === 'ip' || key === 'bf' ? (rec[key] || 0) + v : v; // counts add up if a team appears twice
    }
    out.set(t.id, rec);
  }
  return { teams: out, columnsFound: cols, unknownTeams: unknown };
}

const BULLPEN_SPEC = { ip: ['IP'], bf: ['TBF', 'BF', 'PA'], woba: ['wOBA'] };
const WRC_SPEC = { wrcPlus: ['wRC+', 'wRCPlus'] };
const FRV_SPEC = { frv: ['FRV', 'Fielding Run Value'] };

// ---------- blending seasons ----------
function blendPlayers(perSeason, keyFn) {
  const groups = new Map();
  for (const rows of perSeason) for (const p of rows) {
    const k = keyFn(p); if (k == null) continue;
    (groups.get(k) || groups.set(k, []).get(k)).push(p);
  }
  const out = new Map();
  for (const [k, list] of groups) {
    const pa = list.reduce((s, p) => s + (Number.isFinite(p.pa) ? p.pa : 0), 0);
    const rec = { id: list[0].id, name: list[0].name, pa, woba: wavg(list, 'woba', 'pa'), xwoba: wavg(list, 'xwoba', 'pa') };
    const kb = list.filter(p => Number.isFinite(p.k) && Number.isFinite(p.bb) && p.pa > 0);
    if (kb.length) { const n = kb.reduce((s, p) => s + p.pa, 0); rec.k = kb.reduce((s, p) => s + p.k * p.pa, 0) / n; rec.bb = kb.reduce((s, p) => s + p.bb * p.pa, 0) / n; }
    const is = list.filter(p => Number.isFinite(p.iso) && p.pa > 0);
    if (is.length) { const abOf = p => (Number.isFinite(p.ab) && p.ab > 0 ? p.ab : 0.89 * p.pa), n = is.reduce((s, p) => s + abOf(p), 0); rec.ab = n; rec.iso = is.reduce((s, p) => s + p.iso * abOf(p), 0) / n; }
    if (list.some(p => Number.isFinite(p.ip))) rec.ip = list.reduce((s, p) => s + (Number.isFinite(p.ip) ? p.ip : 0), 0);
    out.set(k, rec);
  }
  return out;
}

function indexPlayers(perSeasonRows) {
  const byId = blendPlayers(perSeasonRows, p => (Number.isFinite(p.id) ? p.id : null));
  const nameGroups = new Map();
  for (const rows of perSeasonRows) for (const p of rows) { const k = normName(p.name); (nameGroups.get(k) || nameGroups.set(k, []).get(k)).push(p); }
  const byName = new Map();
  for (const [k, list] of nameGroups) {
    const ids = new Set(list.map(p => (Number.isFinite(p.id) ? p.id : k)));
    if (ids.size === 1) byName.set(k, blendPlayers([list], () => 0).get(0)); // unique name only
  }
  return { byId, byName };
}

function lookupPlayer(index, person) {
  const byId = index.byId.get(person.id);
  if (byId) return { rec: byId, how: 'id' };
  const byName = index.byName.get(normName(person.name));
  return byName ? { rec: byName, how: 'name' } : { rec: null, how: null };
}

// Full-season team QPA: PA-weighted wOBA and xWOBA over every batter on the team, averaged
// (same definition as the lineup QPA in Tester E11), pooled across the seasons in the window.
function teamSeasonQpa(perSeasonRows) {
  const acc = new Map();
  for (const rows of perSeasonRows) for (const p of rows) {
    const t = findTeam(p.team);
    if (!t || !(p.pa > 0) || !Number.isFinite(p.woba) || !Number.isFinite(p.xwoba)) continue;
    const a = acc.get(t.id) || { pa: 0, w: 0, x: 0 };
    a.pa += p.pa; a.w += p.woba * p.pa; a.x += p.xwoba * p.pa; acc.set(t.id, a);
  }
  const out = new Map();
  for (const [id, a] of acc) out.set(id, { pa: a.pa, qpa: (a.w / a.pa + a.x / a.pa) / 2 });
  return out;
}

function blendTeamTables(perSeasonMaps) {
  const out = new Map();
  for (const m of perSeasonMaps) for (const [id, v] of m) {
    const cur = out.get(id) || { games: 0, pa: 0, runs: 0 };
    cur.games += v.games; cur.pa += v.pa; cur.runs += v.runs; out.set(id, cur);
  }
  return out;
}

function blendBullpens(perSeasonMaps) {
  const out = new Map();
  for (const m of perSeasonMaps) for (const [id, v] of m) {
    const cur = out.get(id) || { ip: 0, bf: 0, wSum: 0 };
    cur.ip += v.ip; cur.bf += v.bf; cur.wSum += v.woba * v.bf; out.set(id, cur);
  }
  for (const v of out.values()) { v.woba = v.wSum / v.bf; delete v.wSum; }
  return out;
}

// ---------- model v2: league context and per-season sources ----------
// League wOBA/QPA, starter K-BB and runs per team-game for the current season, padded with last season early on.
function leagueContext(o, v2) {
  const pad = v2.leaguePad, d = v2.defaults;
  const qOf = p => (Number.isFinite(p.woba) && Number.isFinite(p.xwoba) ? (p.woba + p.xwoba) / 2 : p.woba);
  const avg = (rows, f) => { let n = 0, s = 0; for (const p of rows || []) { const x = f(p); if (p.pa > 0 && Number.isFinite(x)) { n += p.pa; s += p.pa * x; } } return { n, s }; };
  const padded = (cur, prev, padN, dflt) => { const pv = prev.n ? prev.s / prev.n : dflt; return (cur.s + padN * pv) / (cur.n + padN); };
  const woba = padded(avg(o.batCur, qOf), avg(o.batPrev, qOf), pad.wobaPa, d.woba);
  const kbbF = p => (Number.isFinite(p.k) && Number.isFinite(p.bb) ? (p.k - p.bb) / 100 : NaN);
  const kbb = padded(avg(o.pitCur, kbbF), avg(o.pitPrev, kbbF), pad.wobaPa, d.kbb);
  const tot = m => { let r = 0, g = 0; for (const v of (m ? m.values() : [])) if (Number.isFinite(v.runs) && Number.isFinite(v.games)) { r += v.runs; g += v.games; } return { s: r, n: g }; };
  const runs = padded(tot(o.teamsCur), tot(o.teamsPrev), pad.teamGames, d.runs);
  // Model v3: league hitter K%, BB% (as fractions) and ISO, same padding as wOBA
  const pct = f => p => (Number.isFinite(p[f]) ? p[f] / 100 : NaN), isoF = p => p.iso;
  const hit = { k: padded(avg(o.batCur, pct('k')), avg(o.batPrev, pct('k')), pad.wobaPa, d.k),
    bb: padded(avg(o.batCur, pct('bb')), avg(o.batPrev, pct('bb')), pad.wobaPa, d.bb),
    iso: padded(avg(o.batCur, isoF), avg(o.batPrev, isoF), pad.wobaPa, d.iso) };
  return { woba, kbb, runs, hit };
}

// J's run-efficiency rule (2026-10-16): a lineup is only as good as the runs it actually puts across. Efficiency =
// the team's runs / the runs its wOBA would produce at the league's conversion rate, this season plus last season at
// 0.6 weight (Baseball-Reference team batting). Model v2/v3 multiply projected runs by efficiency^c.eff; the backtest
// fit c.eff (about 0.2: roughly a fifth of a team's efficiency gap repeats; models/mlb-app/backtest_efficiency.py).
function seasonEfficiencyTotals(map, teamId, slope) {
  if (!map) return null;
  let r = 0, wn = 0, wd = 0;
  for (const v of map.values()) if ([v.runs, v.wobaNum, v.wobaDen].every(Number.isFinite)) { r += v.runs; wn += v.wobaNum; wd += v.wobaDen; }
  const t = map.get(teamId);
  if (!wd || !t || ![t.runs, t.wobaNum, t.wobaDen].every(Number.isFinite)) return null;
  const lgRpa = r / wd, lgW = wn / wd;
  return { runs: t.runs, expected: t.wobaDen * (lgRpa + slope * (t.wobaNum / t.wobaDen - lgW)), woba: t.wobaNum / t.wobaDen, games: t.games };
}

function teamEfficiency(teamId, teams, e) {
  if (!e || !teams) return null;
  const cur = seasonEfficiencyTotals(teams.cur, teamId, e.slope), prev = seasonEfficiencyTotals(teams.prev, teamId, e.slope);
  let r = e.K || 0, x = e.K || 0;
  if (cur) { r += cur.runs; x += cur.expected; }
  if (prev) { r += e.prevSeasonWeight * prev.runs; x += e.prevSeasonWeight * prev.expected; }
  if (!(x > 0) || (!cur && !prev)) return null;
  const raw = Math.min(Math.max(r / x, e.clip[0]), e.clip[1]);
  return { efficiency: raw, lnEff: Math.log(raw), cur, prev };
}

// Team R/G anchor (2026-10-16, J): against an average pitcher a team should project near its own season runs per game.
// anchor = ln(((R + 0.6 R_prev + 20 x league) / (G + 0.6 G_prev + 20)) / league runs per team-game)
function teamAnchor(teamId, teams, lgRuns, prevWeight) {
  if (!teams || !(lgRuns > 0)) return null;
  let r = 20 * lgRuns, g = 20, any = false;
  for (const [m, w] of [[teams.cur, 1], [teams.prev, prevWeight]]) {
    const t = m && m.get(teamId);
    if (t && Number.isFinite(t.runs) && t.games > 0) { r += w * t.runs; g += w * t.games; any = true; }
  }
  return any ? { anchor: Math.log((r / g) / lgRuns), rpg: (r - 20 * lgRuns) / (g - 20) } : null;
}

// ---------- catcher framing and fielding (Savant leaderboards, 2026-10-16) ----------
// Framing: runs from extra strikes and called pitches per catcher. The backtest measure is extra called strikes per taken
// pitch (percentage points), regressed with 3000 takes; Savant's runs convert at ~0.125 runs per strike.
function parseSavantFraming(csvText) {
  const { headers, rows } = csvToObjects(csvText);
  const id = findHeader(headers, ['player_id', 'id', 'catcher_id']);
  const runs = findHeader(headers, ['runs_extra_strikes', 'rv_tot', 'framing_runs', 'catcher_framing_runs']);
  const n = findHeader(headers, ['n_called_pitches', 'called_pitches', 'pitches', 'n_pitches']);
  if (!id || !runs) throw new Error('Savant framing CSV is missing player_id / framing-runs columns (found: ' + headers.join(', ') + ')');
  const out = new Map();
  for (const r of rows) { const k = num(r[id]); if (Number.isFinite(k)) out.set(k, { runs: num(r[runs]), takes: n ? num(r[n]) : NaN }); }
  return out;
}
// Outs above average per fielder; attempts recovered from OAA / success-rate difference when no attempts column.
function parseSavantOaa(csvText) {
  const { headers, rows } = csvToObjects(csvText);
  const id = findHeader(headers, ['player_id', 'id']), oaa = findHeader(headers, ['outs_above_average', 'oaa']);
  const att = findHeader(headers, ['attempts', 'n_attempts', 'opportunities']);
  const diff = findHeader(headers, ['diff_success_rate_formatted', 'diff_success_rate', 'success_rate_diff']);
  if (!id || !oaa) throw new Error('Savant OAA CSV is missing player_id / outs_above_average columns (found: ' + headers.join(', ') + ')');
  const out = new Map();
  for (const r of rows) {
    const k = num(r[id]), o = num(r[oaa]); if (!Number.isFinite(k) || !Number.isFinite(o)) continue;
    let a = att ? num(r[att]) : NaN;
    if (!(a > 0) && diff) { const d = num(r[diff]) / 100; if (Math.abs(d) > 0.001) a = o / d; }
    const cur = out.get(k);                       // a player listed once per position: add them up
    out.set(k, cur ? { oaa: cur.oaa + o, attempts: (cur.attempts || 0) + (a > 0 ? a : 0) } : { oaa: o, attempts: a > 0 ? a : NaN });
  }
  return out;
}
const POS_NUM = { P: 1, C: 2, '1B': 3, '2B': 4, '3B': 5, SS: 6, LF: 7, CF: 8, RF: 9 };
// share of balls in play each position handles (Retrosheet 2015-25)
const POS_SHARE = { 3: 0.07, 4: 0.12, 5: 0.11, 6: 0.13, 7: 0.15, 8: 0.17, 9: 0.16 };
// lineup: [{id, name, pos}] of the FIELDING team. Returns { framing (pp), fielding (OAA per chance x100, share-weighted), catcher, fielders }
function teamDefense(lineup, d, prevWeight) {
  if (!d) return null;
  const pw = prevWeight, out = { framing: 0, fielding: 0, catcher: null, fielders: [] };
  const catcher = (lineup || []).find(p => p.pos === 'C');
  if (catcher && d.framing) {
    let runs = 0, takes = 3000;
    for (const [m, w] of [[d.framing.cur, 1], [d.framing.prev, pw]]) {
      const r = m && m.get(catcher.id);
      if (r && Number.isFinite(r.runs)) { runs += w * r.runs; takes += w * (r.takes > 0 ? r.takes : 4000); }
    }
    out.framing = runs / 0.125 / takes * 100; out.catcher = catcher.name;
  }
  if (d.oaa) for (const p of lineup || []) {
    const pos = POS_NUM[p.pos]; if (!POS_SHARE[pos]) continue;
    let o = 0, a = 250;
    for (const [m, w] of [[d.oaa.cur, 1], [d.oaa.prev, pw]]) {
      const r = m && m.get(p.id);
      if (r) { o += w * r.oaa; a += w * (r.attempts > 0 ? r.attempts : 250); }
    }
    const v = o / a * 100 * POS_SHARE[pos]; out.fielding += v; out.fielders.push({ name: p.name, pos: p.pos, value: v });
  }
  return out;
}

// ---------- handedness and pitcher-catcher pairing (2026-10-16, "Model v3 full", logged only) ----------
// League platoon gap (Retrosheet 2014-25): a hitter's wOBA vs a hand minus his overall wOBA, by how he bats.
const PLATOON_GAP = { L: { L: -0.0254, R: 0.0078 }, R: { L: 0.0122, R: -0.0051 }, S: { L: 0.0016, R: -0.0006 } };
// MLB Stats API /people -> Map id -> { bats: 'L'|'R'|'S', throws: 'L'|'R' }
function parseHands(json) {
  const out = new Map();
  for (const p of (json && json.people) || []) out.set(p.id, { bats: (p.batSide && p.batSide.code) || 'R', throws: (p.pitchHand && p.pitchHand.code) || 'R' });
  return out;
}
// mean league platoon gap of tonight's nine vs the opposing starter's hand, in wOBA points x starter share (backtest units)
function lineupPlatoon(lineup, starterThrows, hands, starterShare) {
  if (!hands || !lineup || !lineup.length) return null;
  const th = starterThrows === 'L' ? 'L' : 'R';
  const g = lineup.map(p => { const h = hands.get(p.id); return (PLATOON_GAP[h ? h.bats : 'R'] || PLATOON_GAP.R)[th]; });
  return g.reduce((a, b) => a + b, 0) / g.length * 100 * starterShare;
}
// Savant pitch-by-pitch CSV for one pitcher-season -> wOBA allowed overall and with each catcher (fielder_2)
function parseStatcastPairs(csvText) {
  const { headers, rows } = csvToObjects(csvText);
  const wv = findHeader(headers, ['woba_value']), wd = findHeader(headers, ['woba_denom']), c = findHeader(headers, ['fielder_2', 'fielder_2_1']);
  if (!wv || !wd || !c) throw new Error('Savant pitch CSV is missing woba_value / woba_denom / fielder_2');
  const all = { w: 0, n: 0 }, by = {};
  for (const r of rows) {
    const d = num(r[wd]); if (!(d > 0)) continue;
    const w = num(r[wv]) || 0, k = String(num(r[c]));
    all.w += w; all.n += d; by[k] = by[k] || { w: 0, n: 0 }; by[k].w += w; by[k].n += d;
  }
  return { all, by };
}
// pair residual: this catcher's wOBA allowed with this pitcher minus the pitcher's own rate, regressed with 400 PA,
// in wOBA points x starter share (positive = the pair allows more)
function pairTandem(pairs, catcherId, prevWeight, starterShare) {
  if (!pairs || catcherId == null) return null;
  let res = 0, n = 400;
  for (const [d, w] of [[pairs.cur, 1], [pairs.prev, prevWeight]]) {
    if (!d || !(d.all.n > 0)) continue;
    const pr = d.by[String(catcherId)]; if (!pr) continue;
    res += w * (pr.w - pr.n * d.all.w / d.all.n); n += w * pr.n;
  }
  return res / n * 100 * starterShare;
}

function v2Sources(o, v2) {
  return { lg: leagueContext(o, v2), teams: { cur: o.teamsCur || null, prev: o.teamsPrev || null }, defense: o.defense || null, hands: o.hands || null, pairs: o.pairs || null, batters: { cur: indexPlayers([o.batCur || []]), prev: indexPlayers([o.batPrev || []]) },
    pitchers: { cur: indexPlayers([o.pitCur || []]), prev: indexPlayers([o.pitPrev || []]) }, seasons: o.seasons };
}

module.exports = { leagueContext, v2Sources, teamEfficiency, seasonEfficiencyTotals, teamAnchor, teamDefense, parseHands, lineupPlatoon, parseStatcastPairs, pairTandem, PLATOON_GAP, parseSavantFraming, parseSavantOaa, seasonsNeeded, parseSavantPlayers, parseBbrefTeams, parseFgBatters, parseTeamTable, BULLPEN_SPEC, WRC_SPEC, FRV_SPEC,
  indexPlayers, lookupPlayer, teamSeasonQpa, blendTeamTables, blendBullpens };
