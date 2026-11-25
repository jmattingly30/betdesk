'use strict';
// Game lookup (from MLB Stats API schedule JSON) and input assembly. Pure functions.

const { findTeam, TEAM_BY_ID, fairMoneyline } = require('./util');
const { lookupPlayer, teamEfficiency, teamAnchor, teamDefense, lineupPlatoon, pairTandem } = require('./data');
const { project } = require('./projection');
const { projectV2, totalsProb } = require('./projection_v2');
const { projectEff } = require('./projection_eff');
const { hierarchical, exactPoisson, pythagorean, pythagenpat } = require('./montecarlo');

// ---------- schedule -> games ----------
function parseSide(side, players, lineupSize) {
  const team = side.team;
  const lineup = (players || []).map(p => ({ id: p.id, name: p.fullName, pos: p.primaryPosition && p.primaryPosition.abbreviation }));
  return {
    teamId: team.id, name: team.name, defaultVenueId: team.venue && team.venue.id,
    probable: side.probablePitcher ? { id: side.probablePitcher.id, name: side.probablePitcher.fullName } : null,
    lineup, lineupPosted: lineup.length === lineupSize,
  };
}

// MLB Stats API game types: R regular season; F Wild Card, D Division Series, L LCS, W World Series.
// Spring training (S), exhibitions (E) and the All-Star Game (A) are never run.
const GAME_TYPE_LABELS = { R: 'Regular season', F: 'Wild Card', D: 'Division Series', L: 'League Championship Series', W: 'World Series' };

function parseGames(scheduleJson, cfg) {
  const games = [], allowed = cfg.gameTypes || ['R'];
  for (const d of scheduleJson.dates || []) for (const g of d.games || []) {
    const type = g.gameType || 'R';
    if (!allowed.includes(type)) continue;
    games.push({
      gameType: type, seriesDescription: g.seriesDescription || GAME_TYPE_LABELS[type] || type,
      gamePk: g.gamePk, date: g.officialDate, startUtc: g.gameDate, status: g.status && g.status.detailedState,
      gameNumber: g.gameNumber, doubleHeader: g.doubleHeader, season: Number(g.season),
      venue: g.venue ? { id: g.venue.id, name: g.venue.name } : null,
      // game-time forecast (MLB posts it a few hours before first pitch; model v3 adds 0 without it)
      weather: g.weather && g.weather.temp != null && g.weather.temp !== '' ? { tempF: Number(g.weather.temp), condition: g.weather.condition || '', wind: g.weather.wind || '' } : null,
      away: parseSide(g.teams.away, g.lineups && g.lineups.awayPlayers, cfg.lineupSize),
      home: parseSide(g.teams.home, g.lineups && g.lineups.homePlayers, cfg.lineupSize),
    });
  }
  return games;
}

// Returns { team, games } or { error }
function gamesForTeam(scheduleJson, teamQuery, cfg) {
  const team = findTeam(teamQuery);
  if (!team) return { error: `"${teamQuery}" doesn't match exactly one MLB team. Try the nickname (e.g. "Yankees") or abbreviation.` };
  const games = parseGames(scheduleJson, cfg).filter(g => g.away.teamId === team.id || g.home.teamId === team.id);
  if (!games.length) return { team, games: [], error: `No regular-season or postseason game found for the ${team.name} on this date.` };
  return { team, games };
}

// What is and isn't ready, in plain words (used before the user confirms).
function readiness(game) {
  const problems = [];
  for (const [label, s] of [['Away', game.away], ['Home', game.home]]) {
    if (!s.probable) problems.push(`${label} (${s.name}): no probable starting pitcher listed yet.`);
    if (!s.lineupPosted) problems.push(`${label} (${s.name}): lineup not posted yet` + (s.lineup.length ? ` (only ${s.lineup.length} names so far).` : '.'));
  }
  if (game.venue && game.home.defaultVenueId && game.venue.id !== game.home.defaultVenueId)
    problems.push(`Note: game venue (${game.venue.name}) is not ${game.home.name}'s usual park; the park factor used is the home team's normal one.`);
  if (game.gameType && game.gameType !== 'R')
    problems.push(`Note: postseason game (${game.seriesDescription}). The projection uses regular-season stats.`);
  const hardStop = problems.filter(p => !p.startsWith('Note:'));
  return { ready: hardStop.length === 0, problems };
}

// ---------- assembling model inputs ----------
// sources: { batterIndex, pitcherIndex, teamOffense: Map, bullpen: Map, wrc: Map, frv: Map, seasonsUsed: [..] }
function buildSide(sideGame, sources, missing, warnings, cfg) {
  const tid = sideGame.teamId, name = sideGame.name;
  const lineup = [];
  for (const p of sideGame.lineup) {
    const { rec, how } = lookupPlayer(sources.batterIndex, p);
    if (!rec || !Number.isFinite(rec.woba) || !Number.isFinite(rec.xwoba) || !(rec.pa > 0)) missing.push(`${name}: no batter stats found for ${p.name} (MLBAM ${p.id})`);
    else { lineup.push({ woba: rec.woba, xwoba: rec.xwoba, pa: rec.pa, name: p.name }); if (how === 'name') warnings.push(`${p.name}: matched to batter stats by NAME (no MLBAM id in the file) - check it's the right player`); }
  }
  let starter = null;
  if (sideGame.probable) {
    const r = sources.pitcherIndex.byId.get(sideGame.probable.id);
    if (!r || !Number.isFinite(r.woba) || !Number.isFinite(r.xwoba) || !(r.pa > 0) || !(r.ip > 0)) missing.push(`${name}: no Savant starter stats for probable pitcher ${sideGame.probable.name} (MLBAM ${sideGame.probable.id})`);
    else starter = { woba: r.woba, xwoba: r.xwoba, tbf: r.pa, ip: r.ip, name: sideGame.probable.name };
  }
  const off = sources.teamOffense.get(tid);
  if (!off) missing.push(`${name}: no Baseball-Reference team games/PA/runs`);
  // No-FanGraphs mode: no bullpen term, wRC+ 100 and FRV 0 (both exact no-ops in the formula)
  const fg = cfg.useFangraphs !== false;
  const bp = fg ? sources.bullpen.get(tid) : null;
  if (fg && (!bp || !Number.isFinite(bp.ip) || !Number.isFinite(bp.bf) || !Number.isFinite(bp.woba))) missing.push(`${name}: bullpen IP/BF/wOBA not available (need FanGraphs bullpen IP+TBF and bullpen wOBA against)`);
  const wrc = fg ? sources.wrc.get(tid) : { wrcPlus: 100 }, frv = fg ? sources.frv.get(tid) : { frv: 0 };
  if (fg && (!wrc || !Number.isFinite(wrc.wrcPlus))) missing.push(`${name}: team wRC+ not available (FanGraphs)`);
  if (fg && (!frv || !Number.isFinite(frv.frv))) missing.push(`${name}: team FRV not available (FanGraphs)`);
  let seasonQpa;
  if (cfg.rqpaBasis === 'season') {
    const q = sources.teamQpa && sources.teamQpa.get(tid);
    if (!q) missing.push(`${name}: no season-long QPA (no batters with team "${name}" in the FanGraphs batters export)`);
    else seasonQpa = q.qpa;
  }
  return { lineup, starter, bullpen: fg ? bp : null, team: off && wrc && frv ? { games: off.games, pa: off.pa, runs: off.runs, wrcPlus: wrc.wrcPlus, frv: frv.frv, seasonQpa } : null };
}

// Model v3 inputs: each hitter's and the starter's line for this season and last season (either may be missing:
// v3 regresses toward league average instead of refusing), plus the FanGraphs bullpen when loaded.
function buildSideV2(sideGame, sources, warnings, cfg) {
  const v = sources.v2, name = sideGame.name;
  const lineup = sideGame.lineup.map(p => {
    const cur = lookupPlayer(v.batters.cur, p).rec, prev = lookupPlayer(v.batters.prev, p).rec;
    if (!cur && !prev) warnings.push(`Model v3: no MLB batting stats for ${p.name} (${name}) in ${v.seasons.join(' or ')}; using a below-average rookie prior`);
    return { cur, prev, name: p.name };
  });
  const id = sideGame.probable.id;
  const starter = { cur: v.pitchers.cur.byId.get(id) || null, prev: v.pitchers.prev.byId.get(id) || null };
  if (!starter.cur && !starter.prev) warnings.push(`Model v3: no MLB pitching stats for ${sideGame.probable.name} (${name}); using a replacement-level prior`);
  const bp = cfg.useFangraphs !== false ? sources.bullpen.get(sideGame.teamId) : null;
  const efficiency = teamEfficiency(sideGame.teamId, v.teams, cfg.v2 && cfg.v2.efficiency);
  if (cfg.v2 && cfg.v2.efficiency && !efficiency) warnings.push(`Model v3: no Baseball-Reference runs/wOBA for ${name}; run efficiency left at 1.00`);
  const anchor = teamAnchor(sideGame.teamId, v.teams, v.lg.runs, cfg.v2 ? cfg.v2.prevSeasonWeight : 0.6);
  const defense = teamDefense(sideGame.lineup, v.defense, cfg.v2 ? cfg.v2.prevSeasonWeight : 0.6);
  return { teamId: sideGame.teamId, lineup, starter, bullpen: bp || null, efficiency, anchor, defense };
}

// Returns { ok:false, missing, warnings } or { ok:true, result }
// J's efficiency model is the primary model (2026-10-19) beside J's original Tester-sheet models (Monte Carlo, Pythagorean,
// Pythagenpat), which run when their inputs are all there; if not, they are skipped with a note instead of blocking the
// efficiency model. v3, v3 + defense and v3 full are logged for comparison only. Model v2 was retired 2026-10-19 (J).
function runGame(game, sources, cfg, rng) {
  const ready = readiness(game);
  if (!ready.ready) return { ok: false, missing: ready.problems.filter(p => !p.startsWith('Note:')), warnings: [] };
  const missing = [], warnings = ready.problems.filter(p => p.startsWith('Note:'));
  const park = cfg.parkFactors[game.home.teamId];
  if (!park) return { ok: false, missing: [`No park factor for ${game.home.name}`], warnings };

  // J's efficiency model (primary): season R/G at full strength x tonight's matchup on the runs scale (src/projection_eff.js)
  let eff = null;
  if (sources.v2 && cfg.eff) {
    const a = buildSideV2(game.away, sources, warnings, cfg), h = buildSideV2(game.home, sources, warnings, cfg);
    const p = projectEff(a, h, { lg: sources.v2.lg, teams: sources.v2.teams, parkFactors: cfg.parkFactors, homeTeamId: game.home.teamId }, cfg.eff, cfg.v3);
    if (p) {
      eff = { ...p, awayFairMl: fairMoneyline(p.awayWinProb), homeFairMl: fairMoneyline(p.homeWinProb),
        bullpen: a.bullpen && h.bullpen ? 'FanGraphs bullpens' : 'league-average bullpens' };
      for (const [k, s] of [['away', p.sides.away], ['home', p.sides.home]]) if (s.source.startsWith('none')) warnings.push(`Efficiency model: no Baseball-Reference efficiency for ${game[k].name} yet; using 1.00`);
    } else warnings.push('Efficiency model skipped: the Baseball-Reference team table could not be loaded');
  }

  // Model v3 (comparison only): regressed hitter/starter wOBA-xwOBA matchup + lineup K%/BB%/ISO + temperature + run efficiency + R/G anchor.
  // (It shares its player inputs with the retired v2: buildSideV2 / projectV2 are just the function names.)
  // Pythagorean (1.83) and a Monte Carlo run on v3's projected runs too, like the sheet models do on theirs.
  let v3 = null;
  if (sources.v2 && cfg.v3 && sources.v2.lg.hit) {
    const a = buildSideV2(game.away, sources, eff ? [] : warnings, cfg), h = buildSideV2(game.home, sources, eff ? [] : warnings, cfg);
    const tempF = game.weather && Number.isFinite(game.weather.tempF) ? game.weather.tempF : null;
    const p = projectV2(a, h, { lg: sources.v2.lg, parkFactors: cfg.parkFactors, homeTeamId: game.home.teamId, tempF }, cfg.v3);
    const v = cfg.v3, tp = totalsProb(p.runsHome, p.runsAway, v.nbAlpha, v.totalLines, v.totalsCalSlope);
    const pyth = pythagorean(p.runsAway, p.runsHome, cfg.pythagoreanExponent);
    // stage-1 spread = the negative binomial's own extra spread (sd of a team's true rate = sqrt(alpha) x runs),
    // so this Monte Carlo is the simulation version of v3's win probability, before calibration
    const mcStd = Math.sqrt(v.nbAlpha) * (p.runsAway + p.runsHome) / 2;
    const mc = hierarchical({ lambdaA: p.runsAway, lambdaB: p.runsHome, errorMean: 0, errorStd: mcStd, iterations: v.mcIterations, floorLambda: cfg.floorLambda, rng });
    const isoMissing = p.detail.away.skills.hittersWithIso + p.detail.home.skills.hittersWithIso === 0;
    if (isoMissing) warnings.push('Model v3: the Savant batter file has no ISO column, so v3 ran without its power term');
    if (tempF == null) warnings.push('Model v3: no game-time temperature posted yet; v3 used an average night (73.5F)');
    v3 = { awayWinProb: p.awayWinProb, homeWinProb: p.homeWinProb, awayFairMl: fairMoneyline(p.awayWinProb), homeFairMl: fairMoneyline(p.homeWinProb),
      runsAway: p.runsAway, runsHome: p.runsHome, rawHomeWinProb: p.rawHomeWinProb, tiePct: p.tiePct, total: tp.mean, medianTotal: tp.median, totals: tp.lines, tempF,
      bullpen: a.bullpen && h.bullpen ? 'FanGraphs bullpens' : 'league-average bullpens',
      sides: { away: { efficiency: p.detail.away.efficiency, seasonRpg: p.detail.away.seasonRpg }, home: { efficiency: p.detail.home.efficiency, seasonRpg: p.detail.home.seasonRpg } },
      inputs: p.detail,
      pythagorean: { awayWinProb: pyth.pA, homeWinProb: pyth.pB, exponent: cfg.pythagoreanExponent },
      monteCarlo: { awayWinProb: mc.pA, homeWinProb: mc.pB, errorStd: mcStd, iterations: mc.iterations },
      detail: { league: sources.v2.lg, away: p.detail.away.skills, home: p.detail.home.skills } };
  }

  // Model v3 + catcher framing + fielders (2026-10-16). Shown and logged beside v3; not used for picks until its Savant
  // inputs are checked against the backtest's scale on live games.
  let v3d = null;
  if (sources.v2 && cfg.v3d && sources.v2.lg.hit && sources.v2.defense) {
    const a = buildSideV2(game.away, sources, [], cfg), h = buildSideV2(game.home, sources, [], cfg);
    const tempF = game.weather && Number.isFinite(game.weather.tempF) ? game.weather.tempF : null;
    const p = projectV2(a, h, { lg: sources.v2.lg, parkFactors: cfg.parkFactors, homeTeamId: game.home.teamId, tempF }, cfg.v3d);
    if (!game.away.lineup.some(x => x.pos === 'C') || !game.home.lineup.some(x => x.pos === 'C')) warnings.push('Model v3 + defense: a starting catcher is not marked in the lineup; framing left at average');
    v3d = { awayWinProb: p.awayWinProb, homeWinProb: p.homeWinProb, awayFairMl: fairMoneyline(p.awayWinProb), homeFairMl: fairMoneyline(p.homeWinProb),
      runsAway: p.runsAway, runsHome: p.runsHome, defense: { away: a.defense, home: h.defense } };
  } else if (sources.v2 && cfg.v3d && !sources.v2.defense) warnings.push('Model v3 + defense skipped: Savant catcher framing / fielding leaderboards could not be loaded');

  // "Model v3 full" (2026-10-16, J): v3 + defense + hitters' platoon vs the starter's hand + the starter-catcher pairing.
  // Logged beside the others; not used for picks until graded live.
  let v3f = null;
  if (sources.v2 && cfg.v3f && sources.v2.lg.hit && sources.v2.defense && sources.v2.hands) {
    const a = buildSideV2(game.away, sources, [], cfg), h = buildSideV2(game.home, sources, [], cfg), V = sources.v2, sh = cfg.v3f.starterShare, pw = cfg.v3f.prevSeasonWeight;
    const hand = id => (V.hands.get(id) || {}).throws || 'R', catcherOf = sg => (sg.lineup.find(x => x.pos === 'C') || {}).id;
    a.platoon = lineupPlatoon(game.away.lineup, hand(game.home.probable.id), V.hands, sh);
    h.platoon = lineupPlatoon(game.home.lineup, hand(game.away.probable.id), V.hands, sh);
    a.tandem = pairTandem(V.pairs && V.pairs[game.away.probable.id], catcherOf(game.away), pw, sh);
    h.tandem = pairTandem(V.pairs && V.pairs[game.home.probable.id], catcherOf(game.home), pw, sh);
    const tempF = game.weather && Number.isFinite(game.weather.tempF) ? game.weather.tempF : null;
    const p = projectV2(a, h, { lg: V.lg, parkFactors: cfg.parkFactors, homeTeamId: game.home.teamId, tempF }, cfg.v3f);
    if (!V.pairs) warnings.push('Model v3 full: Savant pitch file for a starter not loaded; pairing left at average');
    v3f = { awayWinProb: p.awayWinProb, homeWinProb: p.homeWinProb, awayFairMl: fairMoneyline(p.awayWinProb), homeFairMl: fairMoneyline(p.homeWinProb),
      runsAway: p.runsAway, runsHome: p.runsHome,
      extra: { away: { platoon: a.platoon, tandem: a.tandem, starterThrows: hand(game.away.probable.id) }, home: { platoon: h.platoon, tandem: h.tandem, starterThrows: hand(game.home.probable.id) } } };
  }

  const t1 = buildSide(game.away, sources, missing, warnings, cfg);
  const t2 = buildSide(game.home, sources, missing, warnings, cfg);
  let legacy = null;
  if (missing.length) {
    if (!eff && !v3) return { ok: false, missing, warnings };
    warnings.push(`Original sheet models skipped (${eff ? 'the efficiency model' : 'model v3'} still ran): ` + missing.join('; '));
  } else legacy = legacyModels(t1, t2, park, sources, cfg, rng);

  const runs = legacy ? legacy.projectedRuns : (m => ({ awayRaw: m.runsAway, homeRaw: m.runsHome, awayScaled: m.runsAway, homeScaled: m.runsHome }))(eff || v3);
  return {
    ok: true, warnings,
    result: {
      gamePk: game.gamePk, date: game.date, startUtc: game.startUtc, gameType: game.gameType || 'R', seriesDescription: game.seriesDescription || '',
      away: game.away.name, home: game.home.name, venue: game.venue && game.venue.name, parkFactor: park,
      starters: { away: game.away.probable.name, home: game.home.probable.name },
      seasonsUsed: sources.seasonsUsed, rqpaBasis: cfg.rqpaBasis || 'tonight',
      inputs: cfg.useFangraphs === false ? 'no-fangraphs' : 'fangraphs',
      eff, v3, v3d, v3f, weather: game.weather || null,
      projectedRuns: runs,
      ...(legacy ? legacy.models : { hierarchical: null, sensitivity: [], pythagorean: null, pythagenpat: null, pythagenpatSeason: null, comparison: null, cells: null }),
    },
  };
}

function legacyModels(t1, t2, park, sources, cfg, rng) {
  const proj = project(t1, t2, cfg);
  const lambdaA = proj.runs1 * park, lambdaB = proj.runs2 * park;
  const hier = hierarchical({ lambdaA, lambdaB, errorMean: cfg.errorMean, errorStd: cfg.errorStd, iterations: cfg.iterations, floorLambda: cfg.floorLambda, rng });
  const sens = cfg.sensitivityStds.map(std => ({ errorStd: std, ...hierarchical({ lambdaA, lambdaB, errorMean: cfg.errorMean, errorStd: std, iterations: cfg.iterations, floorLambda: cfg.floorLambda, rng }) }));
  // The sheet's "Probability of Win" (Tester H51/J51): 162-game scores C36/K36 with a fixed exponent.
  // No park factor - with a fixed exponent it would cancel anyway.
  const pyth = pythagorean(proj.score162.team1, proj.score162.team2, cfg.pythagoreanExponent);
  // Pythagenpat, two run environments. The win ratio is scale-free, so only the exponent differs.
  const ppat = pythagenpat(lambdaA, lambdaB, cfg.pythagenpatConstant);
  const seasonRpg = proj.cells.C26 + proj.cells.K26;
  const ppatSeason = pythagenpat(lambdaA, lambdaB, cfg.pythagenpatConstant, seasonRpg);
  const withFair = (r, extra) => ({ awayWinProb: r.pA, homeWinProb: r.pB, exponent: r.exponent, constant: cfg.pythagenpatConstant, awayFairMl: fairMoneyline(r.pA), homeFairMl: fairMoneyline(r.pB), ...extra });
  return {
    projectedRuns: { awayRaw: proj.runs1, homeRaw: proj.runs2, awayScaled: lambdaA, homeScaled: lambdaB },
    models: {
      hierarchical: { awayWinProb: hier.pA, homeWinProb: hier.pB, awayFairMl: fairMoneyline(hier.pA), homeFairMl: fairMoneyline(hier.pB), tiePct: hier.tiePct, standardError: hier.standardError, iterations: hier.iterations, errorMean: cfg.errorMean, errorStd: cfg.errorStd, flooredPct: hier.flooredPct },
      sensitivity: sens.map(s => ({ errorStd: s.errorStd, awayWinProb: s.pA, homeWinProb: s.pB })),
      pythagorean: { awayWinProb: pyth.pA, homeWinProb: pyth.pB, exponent: cfg.pythagoreanExponent, awayFairMl: fairMoneyline(pyth.pA), homeFairMl: fairMoneyline(pyth.pB) },
      pythagenpat: withFair(ppat, { runsPerGame: lambdaA + lambdaB, basis: 'projected (park-adjusted)' }),
      pythagenpatSeason: withFair(ppatSeason, { runsPerGame: seasonRpg, basis: 'season R/G' }),
      comparison: { plugInPoissonAway: exactPoisson(lambdaA, lambdaB).pA },
      cells: proj.cells,
    },
  };
}

module.exports = { GAME_TYPE_LABELS, parseGames, gamesForTeam, readiness, runGame };
