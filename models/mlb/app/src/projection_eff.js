'use strict';
// J's efficiency model (2026-10-19): J's original sheet idea, season runs per game read against the lineup's quality
// (wOBA/xwOBA) at FULL strength, with tonight's matchup moving it on the runs scale. Pure functions.
//
//   runs = efficiency x PA/G x (league R/PA + slope x (matchup QPA - league QPA)) x park factor
//   matchup QPA = tonight's lineup QPA + (0.581 x starter + 0.419 x bullpen - league QPA), every player regressed toward
//                 league average by sample size (same rates as model v3, park-neutral)
//   efficiency  = the team's actual runs / the runs its wOBA should have scored at league conversion (Baseball-Reference)
//                 last season's number for games 1-29 (J's rule), then 1 + (games / 162) x (this season's - 1), so a team
//                 converting 5% better all season sits at 1.05 by game 162
//   win chance  = Pythagorean 1.83 on those runs, calibrated; totals = negative binomial, calibrated vs openers
// Backtest (Retrosheet 2015-25, wOBA only): models/mlb-app/backtest_efficiency_schedule.py, settings fit_efficiency_app.py.
const { sideRates, totalsProb } = require('./projection_v2');
const { seasonEfficiencyTotals } = require('./data');

const clip = (x, lo, hi) => Math.min(Math.max(x, lo), hi);

// { efficiency, source, games, thisSeason, lastSeason } for one team
function efficiencyFor(teamId, teams, e) {
  const cur = seasonEfficiencyTotals(teams && teams.cur, teamId, e.slope), prev = seasonEfficiencyTotals(teams && teams.prev, teamId, e.slope);
  const lastSeason = prev && prev.expected > 0 ? clip(prev.runs / prev.expected, e.prevClip[0], e.prevClip[1]) : null;
  const thisSeason = cur && cur.expected > 0 && cur.games > 0 ? clip(cur.runs / cur.expected, e.curClip[0], e.curClip[1]) : null;
  const games = cur && cur.games > 0 ? cur.games : 0;
  if (games < e.priorGames || thisSeason == null)
    return { efficiency: lastSeason != null ? lastSeason : 1, source: lastSeason != null ? 'last season' : 'none (1.00)', games, thisSeason, lastSeason };
  const w = Math.min(games / e.fullGames, 1);
  return { efficiency: 1 + w * (thisSeason - 1), source: `this season x ${games}/${e.fullGames}`, games, thisSeason, lastSeason };
}

// league runs per PA and each team's PA per game, this season (last season before any games)
function teamPaPerGame(teamId, teams) {
  for (const m of [teams && teams.cur, teams && teams.prev]) { const t = m && m.get(teamId); if (t && t.games > 0 && t.pa > 0) return t.pa / t.games; }
  return null;
}
function leagueRunsPerPa(teams) {
  for (const m of [teams && teams.cur, teams && teams.prev]) {
    if (!m) continue; let r = 0, pa = 0;
    for (const t of m.values()) if (Number.isFinite(t.runs) && Number.isFinite(t.pa)) { r += t.runs; pa += t.pa; }
    if (pa > 0) return r / pa;
  }
  return null;
}

// away/home: buildSideV2 output; ctx: { lg, teams, parkFactors, homeTeamId }; e: cfg.eff; v: cfg.v3 (player regression settings)
function projectEff(away, home, ctx, e, v) {
  const lg = ctx.lg, pf = ctx.parkFactors[ctx.homeTeamId] || 1, lgRpa = leagueRunsPerPa(ctx.teams);
  if (!(lgRpa > 0)) return null;
  const A = sideRates(away, ctx, v), H = sideRates(home, ctx, v), sh = v.starterShare;
  const side = (bat, pit, sideIn) => {
    const q = bat.lineup + (sh * pit.starter + (1 - sh) * pit.bullpen - lg.woba);
    const paG = teamPaPerGame(sideIn.teamId, ctx.teams) || e.defaultPaPerGame;
    const base = paG * (lgRpa + e.slope * (q - lg.woba));
    const ef = efficiencyFor(sideIn.teamId, ctx.teams, e);
    return { runs: Math.max(ef.efficiency * base * pf, 0.5), matchupQpa: q, paPerGame: paG, runsAtLeagueConversion: base * pf, ...ef };
  };
  const a = side(A, H, away), h = side(H, A, home), x = e.pythExponent;
  const raw = Math.pow(h.runs, x) / (Math.pow(h.runs, x) + Math.pow(a.runs, x));
  const lo = Math.log(raw / (1 - raw)), pHome = 1 / (1 + Math.exp(-(e.calibration.intercept + e.calibration.slope * lo)));
  const tp = totalsProb(h.runs, a.runs, e.nbAlpha, v.totalLines, e.totalsCalSlope);
  return { runsAway: a.runs, runsHome: h.runs, pythHomeWinProb: raw, homeWinProb: pHome, awayWinProb: 1 - pHome,
    total: tp.mean, medianTotal: tp.median, totals: tp.lines, sides: { away: a, home: h }, leagueRunsPerPa: lgRpa };
}

module.exports = { projectEff, efficiencyFor };
