'use strict';
// Network + disk layer. Everything with real logic lives in the pure modules; this file only
// fetches, caches, reads your exported CSVs, and hands text to the parsers.

const fs = require('fs');
const path = require('path');
const D = require('./data');
const { TEAMS } = require('./util');

const ROOT = path.join(__dirname, '..');
const CACHE = path.join(ROOT, 'data', 'cache');
const UA = 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0 Safari/537.36';
const HOUR = 3600 * 1000;

class MissingDataError extends Error {
  constructor(message, items) { super(message); this.items = items || []; }
}

async function fetchText(url, cacheName, ttlMs) {
  fs.mkdirSync(CACHE, { recursive: true });
  const file = path.join(CACHE, cacheName);
  if (ttlMs && fs.existsSync(file) && Date.now() - fs.statSync(file).mtimeMs < ttlMs) return fs.readFileSync(file, 'utf8');
  const res = await fetch(url, { headers: { 'User-Agent': UA } });
  if (!res.ok) throw new MissingDataError(`${new URL(url).host} returned HTTP ${res.status} for ${url}`);
  const text = await res.text();
  if (cacheName) fs.writeFileSync(file, text);
  return text;
}

async function fetchSchedule(date) {
  const url = `https://statsapi.mlb.com/api/v1/schedule?sportId=1&date=${date}&hydrate=team,probablePitcher,lineups,venue,weather`;
  const res = await fetch(url, { headers: { 'User-Agent': UA } });
  if (!res.ok) throw new MissingDataError(`MLB Stats API returned HTTP ${res.status}`);
  return res.json();
}

const savantUrl = (season, type, sel) =>
  `https://baseballsavant.mlb.com/leaderboard/custom?year=${season}&type=${type}&filter=&min=1&selections=${sel}&chart=false&x=pa&y=pa&r=no&chartType=beeswarm&csv=true`;

const PITCHER_SEL = 'p_formatted_ip,pa,woba,xwoba,k_percent,bb_percent';
// Catcher framing and fielding leaderboards (2026-10-16, Model v3 + defense)
const framingUrl = season => `https://baseballsavant.mlb.com/catcher_framing?year=${season}&team=&min=1&type=catcher&sort=4,1&csv=true`;
const pitchFileUrl = (season, pid) => `https://baseballsavant.mlb.com/statcast_search/csv?all=true&hfSea=${season}%7C&hfGT=R%7C&player_type=pitcher&pitchers_lookup%5B%5D=${pid}&type=details&min_pitches=0&min_results=0&min_pas=0`;
const oaaUrl = season => `https://baseballsavant.mlb.com/leaderboard/outs_above_average?type=Fielder&startYear=${season}&endYear=${season}&split=no&team=&range=year&min=1&pos=&roles=&viz=hide&csv=true`;
// K%, BB% and ISO added 2026-10-10 for model v3 (new cache file name so old 3-column files aren't reused)
const BATTER_SEL = 'pa,ab,woba,xwoba,k_percent,bb_percent,isolated_power';

const fgPath = (kind, season) => path.join(ROOT, 'data', 'fangraphs', `${kind}_${season}.csv`);

const FILE_HELP = {
  batters: 'FanGraphs batters leaderboard (all qualified=0), columns: Name, Team, PA, wOBA, xwOBA (+ MLBAMID if you can add it) -> Export Data',
  bullpen: 'FanGraphs team pitching, split = Reliever, columns: Team, IP, TBF, wOBA -> Export Data',
  team_wrc: 'FanGraphs team batting, columns: Team, wRC+ -> Export Data',
  team_frv: 'FanGraphs team fielding, columns: Team, FRV -> Export Data',
};

function readExport(kind, season, warnings, cfg, isCurrent) {
  const file = fgPath(kind, season);
  if (!fs.existsSync(file)) return null;
  const ageDays = (Date.now() - fs.statSync(file).mtimeMs) / (24 * HOUR);
  if (isCurrent && ageDays > cfg.csvMaxAgeDays) warnings.push(`${path.basename(file)} is ${ageDays.toFixed(1)} days old - re-export it for today's numbers`);
  return fs.readFileSync(file, 'utf8');
}

async function loadSources(game, cfg) {
  const warnings = [];
  const season = game.season;

  // 1. Baseball-Reference current season -> games played decides the sample window
  const bbUrl = y => `https://www.baseball-reference.com/leagues/majors/${y}.shtml`;
  const currentTeams = D.parseBbrefTeams(await fetchText(bbUrl(season), `bbref_${season}.html`, 6 * HOUR));
  const gp = [game.away.teamId, game.home.teamId].map(id => (currentTeams.get(id) || {}).games);
  if (gp.some(g => !Number.isFinite(g))) throw new MissingDataError('Baseball-Reference has no games-played for one of the teams');
  const currentGames = Math.min(...gp);
  const seasons = D.seasonsNeeded(season, currentGames, cfg);
  if (seasons.length > 1) warnings.push(`Only ${currentGames} games played in ${season}; blending ${seasons.join(' + ')} (rule threshold: ${cfg.sampleWindow.minCurrentSeasonGames}).`);

  // 2. Which hand-exported files do we need? Report ALL missing ones at once.
  const fg = cfg.useFangraphs !== false;
  const kinds = (fg ? ['bullpen', 'team_wrc', 'team_frv'] : []).concat(cfg.batterSource === 'fangraphs-csv' ? ['batters'] : []);
  const texts = {}, absent = [];
  for (const s of seasons) for (const k of kinds) {
    const t = readExport(k, s, warnings, cfg, s === season);
    if (t == null) absent.push(`${path.relative(ROOT, fgPath(k, s))}  <-  ${FILE_HELP[k]}`); else texts[`${k}_${s}`] = t;
  }
  if (absent.length) throw new MissingDataError('Missing hand-exported data file(s):', absent);

  // 3. Per-season parsing
  const teamOff = [], batterRows = [], pitcherRows = [], bullpens = [], wrcs = [], frvs = [];
  for (const s of seasons) {
    if (s === season) teamOff.push(currentTeams);
    else { await new Promise(r => setTimeout(r, 3500)); teamOff.push(D.parseBbrefTeams(await fetchText(bbUrl(s), `bbref_${s}.html`, 30 * 24 * HOUR))); }

    batterRows.push(cfg.batterSource === 'savant'
      ? D.parseSavantPlayers(await fetchText(savantUrl(s, 'batter', BATTER_SEL), `savant_batter_${s}_v3.csv`, 6 * HOUR))
      : D.parseFgBatters(texts[`batters_${s}`]));
    pitcherRows.push(D.parseSavantPlayers(await fetchText(savantUrl(s, 'pitcher', PITCHER_SEL), `savant_pitcher_${s}_v2.csv`, (s === season ? 6 : 30 * 24) * HOUR)));

    if (!fg) continue;
    const bp = D.parseTeamTable(texts[`bullpen_${s}`], D.BULLPEN_SPEC).teams;
    if ([...bp.values()].some(v => !Number.isFinite(v.woba) || !Number.isFinite(v.ip) || !Number.isFinite(v.bf)))
      throw new MissingDataError(`bullpen_${s}.csv must have numeric IP, TBF and wOBA for every team:`, [FILE_HELP.bullpen]);
    bullpens.push(bp);
    wrcs.push(D.parseTeamTable(texts[`team_wrc_${s}`], D.WRC_SPEC).teams);
    frvs.push(D.parseTeamTable(texts[`team_frv_${s}`], D.FRV_SPEC).teams);
  }

  // 4. Blend seasons (weights: PA for wRC+, sums for FRV)
  const wrc = new Map(), frv = new Map();
  for (const t of fg ? TEAMS : []) {
    let sw = 0, sv = 0, fs_ = 0, any = false;
    seasons.forEach((s, i) => {
      const w = (teamOff[i].get(t.id) || {}).pa, r = wrcs[i].get(t.id), f = frvs[i].get(t.id);
      if (r && Number.isFinite(r.wrcPlus) && Number.isFinite(w)) { sw += w; sv += r.wrcPlus * w; any = true; }
      if (f && Number.isFinite(f.frv)) fs_ += f.frv;
    });
    if (any) wrc.set(t.id, { wrcPlus: sv / sw });
    if (frvs.some(m => m.has(t.id))) frv.set(t.id, { frv: fs_ });
  }

  let teamQpa;
  if (cfg.rqpaBasis === 'season') {
    if (cfg.batterSource !== 'fangraphs-csv') throw new MissingDataError("rqpaBasis 'season' needs the FanGraphs batters export (it has the Team column); set batterSource to 'fangraphs-csv'.");
    teamQpa = D.teamSeasonQpa(batterRows);
  }

  // 5. Model v2 always uses this season and last season (last season at 0.6 weight), whatever the legacy rule chose
  const v2 = { teamsCur: currentTeams, seasons: [season - 1, season] };
  for (const s of [season - 1, season]) {
    const i = seasons.indexOf(s), cur = s === season, tag = cur ? 'Cur' : 'Prev', ttl = (cur ? 6 : 30 * 24) * HOUR;
    v2['bat' + tag] = i >= 0 && cfg.batterSource === 'savant' ? batterRows[i] : D.parseSavantPlayers(await fetchText(savantUrl(s, 'batter', BATTER_SEL), `savant_batter_${s}_v3.csv`, ttl));
    v2['pit' + tag] = i >= 0 ? pitcherRows[i] : D.parseSavantPlayers(await fetchText(savantUrl(s, 'pitcher', PITCHER_SEL), `savant_pitcher_${s}_v2.csv`, ttl));
    if (!cur) v2.teamsPrev = i >= 0 ? teamOff[i] : D.parseBbrefTeams(await fetchText(bbUrl(s), `bbref_${s}.html`, 30 * 24 * HOUR));
  }
  // 6. Catcher framing + fielders for "Model v3 + defense" (2026-10-16). Optional: a failure only skips that model.
  try {
    const d = { framing: {}, oaa: {} };
    for (const s of [season - 1, season]) {
      const tag = s === season ? 'cur' : 'prev', ttl = (s === season ? 6 : 30 * 24) * HOUR;
      d.framing[tag] = D.parseSavantFraming(await fetchText(framingUrl(s), `savant_framing_${s}.csv`, ttl));
      d.oaa[tag] = D.parseSavantOaa(await fetchText(oaaUrl(s), `savant_oaa_${s}.csv`, ttl));
    }
    v2.defense = d;
  } catch (e) { warnings.push('Savant catcher framing / fielding not loaded (' + e.message + '); Model v3 + defense skipped'); }
  // 7. Hands (MLB Stats API) and each starter's pitch file (Savant) for "Model v3 full". Optional.
  try {
    const ids = [game.away, game.home].flatMap(sd => sd.lineup.map(x => x.id).concat(sd.probable ? [sd.probable.id] : []));
    const res = await fetch(`https://statsapi.mlb.com/api/v1/people?personIds=${ids.join(',')}`, { headers: { 'User-Agent': UA } });
    if (!res.ok) throw new MissingDataError(`MLB Stats API people returned HTTP ${res.status}`);
    v2.hands = D.parseHands(await res.json());
  } catch (e) { warnings.push('Batter/pitcher hands not loaded (' + e.message + '); Model v3 full skipped'); }
  try {
    const pairs = {};
    for (const sd of [game.away, game.home]) {
      if (!sd.probable) continue;
      const pid = sd.probable.id, d = {};
      for (const s of [season - 1, season]) {
        const tag = s === season ? 'cur' : 'prev', ttl = (s === season ? 6 : 30 * 24) * HOUR;
        d[tag] = D.parseStatcastPairs(await fetchText(pitchFileUrl(s, pid), `savant_pitches_${pid}_${s}.csv`, ttl));
      }
      pairs[pid] = d;
    }
    v2.pairs = pairs;
  } catch (e) { warnings.push('Savant pitch file for a starter not loaded (' + e.message + '); pairing left at average'); }

  return {
    warnings,
    sources: {
      v2: D.v2Sources(v2, cfg.v2),
      teamQpa, batterIndex: D.indexPlayers(batterRows), pitcherIndex: D.indexPlayers(pitcherRows),
      teamOffense: D.blendTeamTables(teamOff), bullpen: fg ? D.blendBullpens(bullpens) : new Map(), wrc, frv, seasonsUsed: seasons,
    },
  };
}

function saveResult(result) {
  const dir = path.join(ROOT, 'output');
  fs.mkdirSync(dir, { recursive: true });
  const slug = s => String(s).replace(/[^A-Za-z0-9]+/g, '_');
  fs.writeFileSync(path.join(dir, `${result.date}_${slug(result.away)}_at_${slug(result.home)}.json`), JSON.stringify(result, null, 2));
  const log = path.join(dir, 'results_log_v2.csv');   // v2 columns added 2026-10-09; the older results_log.csv is left as it was
  const head = 'run_time,date,away,home,venue,park_factor,away_raw_runs,home_raw_runs,away_scaled_lambda,home_scaled_lambda,away_win_prob_hierarchical,home_win_prob_hierarchical,std_error,error_mean,error_std,iterations,away_win_prob_plugin_poisson,away_win_prob_pythagorean,seasons_used,v2_away_runs,v2_home_runs,v2_away_win_prob\n';
  if (!fs.existsSync(log)) fs.writeFileSync(log, head);
  const r = result, h = r.hierarchical, p = r.projectedRuns, v = r.v3, f = (x, d) => (Number.isFinite(x) ? x.toFixed(d) : '');
  fs.appendFileSync(log, [new Date().toISOString(), r.date, r.away, r.home, r.venue, r.parkFactor, f(p.awayRaw, 4), f(p.homeRaw, 4), f(p.awayScaled, 4), f(p.homeScaled, 4),
    h ? f(h.awayWinProb, 5) : '', h ? f(h.homeWinProb, 5) : '', h ? f(h.standardError, 5) : '', h ? h.errorMean : '', h ? h.errorStd : '', h ? h.iterations : '',
    r.comparison ? f(r.comparison.plugInPoissonAway, 5) : '', r.pythagorean ? f(r.pythagorean.awayWinProb, 5) : '', r.seasonsUsed.join('+'),
    v ? f(v.runsAway, 4) : '', v ? f(v.runsHome, 4) : '', v ? f(v.awayWinProb, 5) : ''].map(v_ => `"${v_}"`).join(',') + '\n');
}

// ---------- play log storage (local version): data/playlog.json ----------
// Every write goes to a temp file first and is then renamed over the log, and the previous version is kept
// as data/playlog.previous.json, so a crash mid-write can't lose the log. Export to CSV from the Log page.
const LOG_FILE = path.join(ROOT, 'data', 'playlog.json');
const logStore = {
  list() { return fs.existsSync(LOG_FILE) ? JSON.parse(fs.readFileSync(LOG_FILE, 'utf8')) : []; },
  saveAll(rows) {
    fs.mkdirSync(path.dirname(LOG_FILE), { recursive: true });
    if (fs.existsSync(LOG_FILE)) fs.copyFileSync(LOG_FILE, LOG_FILE.replace(/\.json$/, '.previous.json'));
    const tmp = LOG_FILE + '.tmp';
    fs.writeFileSync(tmp, JSON.stringify(rows, null, 1));
    fs.renameSync(tmp, LOG_FILE);
  },
  insert(row) { const rows = this.list(); rows.push(row); this.saveAll(rows); return row; },
  update(id, fn) { const rows = this.list(); const i = rows.findIndex(r => r.id === id); if (i < 0) return null; rows[i] = fn(rows[i]); this.saveAll(rows); return rows[i]; },
  remove(id) { const rows = this.list(); const keep = rows.filter(r => r.id !== id); if (keep.length === rows.length) return false; this.saveAll(keep); return true; },
};

// ---------- market odds (optional; needs ODDS_API_KEY in the server's environment) ----------
let oddsCache = { at: 0, events: null };
async function fetchOddsEvents(regions, maxAgeMs = 5 * 60 * 1000) {
  const key = process.env.ODDS_API_KEY;
  if (!key) return null;
  if (oddsCache.events && Date.now() - oddsCache.at < maxAgeMs) return oddsCache.events;
  const res = await fetch(require('./odds').ODDS_URL(key, regions));
  if (!res.ok) throw new Error(`The Odds API returned HTTP ${res.status}`);
  oddsCache = { at: Date.now(), events: await res.json() };
  return oddsCache.events;
}

// ---------- final scores (MLB Stats API) ----------
async function fetchGameStatuses(gamePks) {
  const out = new Map();
  if (!gamePks.length) return out;
  const url = `https://statsapi.mlb.com/api/v1/schedule?sportId=1&gamePks=${gamePks.join(',')}`;
  const res = await fetch(url, { headers: { 'User-Agent': UA } });
  if (!res.ok) throw new MissingDataError(`MLB Stats API returned HTTP ${res.status}`);
  const j = await res.json();
  for (const d of j.dates || []) for (const g of d.games || []) {
    out.set(g.gamePk, { status: g.status && g.status.detailedState, awayScore: g.teams.away.score, homeScore: g.teams.home.score });
  }
  return out;
}

module.exports = { fetchSchedule, loadSources, saveResult, MissingDataError, logStore, fetchOddsEvents, fetchGameStatuses };
