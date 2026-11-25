'use strict';
// The Tester sheet's projection chain, cell for cell. `cells` in the result uses the sheet's cell
// names (Team 1 = away, Team 2 = home) so any number can be traced back to the spreadsheet.
//
// Formula corrections vs. the workbook as received (agreed with the owner):
//   1. M2, K19, H18 weighted Team 1's bullpen by raw season BF (E17); now BF-per-reliever (F17).
//   2. P18 weighted by IP; now BF-weighted exactly like H18.
//   3. C33/K33 subtracted the opponent's to-date FRV as a full-season number; now FRV/G x 162.
//   4. C11/D11/J11/K11 are computed live as the average of the 9 hitters above.
// `opts.legacy` re-creates the old (uncorrected) formulas; it exists only for regression tests.

const { wavg } = require('./util');

function need(v, label) {
  if (!Number.isFinite(v)) throw new Error('Missing or non-numeric input: ' + label);
  return v;
}

function lineupStats(hitters) {
  const pa = hitters.reduce((s, h) => s + need(h.pa, 'hitter PA'), 0);
  const woba = wavg(hitters, 'woba', 'pa');   // E3 / L3
  const xwoba = wavg(hitters, 'xwoba', 'pa'); // E7 / L7
  return { woba, xwoba, qpa: (woba + xwoba) / 2, avgPa: pa / hitters.length };
}

function starterStats(s) {
  need(s.woba, 'starter wOBA'); need(s.xwoba, 'starter xwOBA'); need(s.tbf, 'starter TBF'); need(s.ip, 'starter IP');
  return { avg: (s.woba + s.xwoba) / 2, tbf: s.tbf, ip: s.ip, paPerOut: s.tbf / (s.ip * 3) };
}

function bullpenStats(b, relievers) {
  need(b.woba, 'bullpen wOBA'); need(b.ip, 'bullpen IP'); need(b.bf, 'bullpen BF');
  const bfPer = b.bf / relievers, ipPer = b.ip / relievers;
  return { woba: b.woba, bf: b.bf, bfPer, ipPer, paPerOut: bfPer / (ipPer * 3) };
}

function pitchingPaPerOut(S, B, mode) {
  if (mode === 'bf-raw') return (B.paPerOut * B.bf + S.paPerOut * S.tbf) / (B.bf + S.tbf);        // old H18
  if (mode === 'ip') return (S.paPerOut * S.ip + B.paPerOut * B.ipPer) / (S.ip + B.ipPer);         // old P18
  return (B.paPerOut * B.bfPer + S.paPerOut * S.tbf) / (B.bfPer + S.tbf);                          // corrected H18 / P18
}

// No-FanGraphs mode: no bullpen term anywhere. A zero-weight bullpen drops out of every blend exactly
// (H18/P18 become the starter's PA/out, F2/M2 and C19/K19 weight only lineup and starter), the same
// "starter pitches the whole game" form the full-season backtest used.
const NO_BULLPEN = { woba: 0, bf: 0, bfPer: 0, ipPer: 0, paPerOut: 0 };

// t1 / t2: { lineup:[{woba,xwoba,pa}x9], starter:{woba,xwoba,tbf,ip}, bullpen:{woba,ip,bf} or null (no-FanGraphs mode),
//            team:{games,pa,runs,frv,wrcPlus} }
function project(t1, t2, cfg, opts) {
  const legacy = (opts && opts.legacy) || {};
  const n = cfg.relieversPerBullpen, outs = cfg.outsPerGame, G = cfg.seasonGames;

  const L1 = lineupStats(t1.lineup), L2 = lineupStats(t2.lineup);
  const S1 = starterStats(t1.starter), S2 = starterStats(t2.starter);
  if (!t1.bullpen !== !t2.bullpen) throw new Error('Either both bullpens or neither');
  const B1 = t1.bullpen ? bullpenStats(t1.bullpen, n) : NO_BULLPEN, B2 = t2.bullpen ? bullpenStats(t2.bullpen, n) : NO_BULLPEN;

  const paG1 = need(t1.team.pa / t1.team.games, 'Team 1 PA/G'), paG2 = need(t2.team.pa / t2.team.games, 'Team 2 PA/G');
  const rG1 = need(t1.team.runs / t1.team.games, 'Team 1 R/G'), rG2 = need(t2.team.runs / t2.team.games, 'Team 2 R/G');
  const frv1 = need(t1.team.frv, 'Team 1 FRV'), frv2 = need(t2.team.frv, 'Team 2 FRV');
  const wrc1 = need(t1.team.wrcPlus, 'Team 1 wRC+'), wrc2 = need(t2.team.wrcPlus, 'Team 2 wRC+');

  const H18 = pitchingPaPerOut(S1, B1, legacy.h18Raw ? 'bf-raw' : 'bf-per');
  const P18 = pitchingPaPerOut(S2, B2, legacy.p18Ip ? 'ip' : 'bf-per');
  const H19 = H18 * outs, P19 = P18 * outs;

  // Team 1's bullpen weight when it is the OPPONENT of Team 2's hitters (M2, K19)
  const wB1 = legacy.m2k19Raw ? B1.bf : B1.bfPer;

  const F2 = (L1.qpa * L1.avgPa + S2.avg * S2.tbf + B2.woba * B2.bfPer) / (L1.avgPa + S2.tbf + B2.bfPer);
  const M2 = (L2.qpa * L2.avgPa + S1.avg * S1.tbf + B1.woba * wB1) / (L2.avgPa + S1.tbf + wB1);

  const C19 = (P19 * (B2.bfPer + S2.tbf) + paG1 * L1.avgPa) / (B2.bfPer + S2.tbf + L1.avgPa);
  const K19 = (H19 * (S1.tbf + wB1) + paG2 * L2.avgPa) / (S1.tbf + wB1 + L2.avgPa);

  const C20 = F2 * C19, K20 = M2 * K19;
  const season = cfg.rqpaBasis === 'season';
  const q1 = season ? need(t1.team.seasonQpa, 'Team 1 season QPA') : L1.qpa;
  const q2 = season ? need(t2.team.seasonQpa, 'Team 2 season QPA') : L2.qpa;
  const C25 = q1 * paG1, K25 = q2 * paG2;              // QPA per game
  const C28 = rG1 / C25, K28 = rG2 / K25;              // runs per QPA
  const C21 = C20 * C28, K21 = K20 * K28;
  const C29 = C21 * G, K29 = K21 * G;

  const C31 = cfg.frvWeight * frv1 / t1.team.games, K31 = cfg.frvWeight * frv2 / t2.team.games;
  const C33 = legacy.frv ? C29 - frv2 * cfg.frvWeight : C29 - K31 * G;
  const K33 = legacy.frv ? K29 - frv1 * cfg.frvWeight : K29 - C31 * G;
  const C36 = (wrc1 - 100) + C33, K36 = K33 + (wrc2 - 100);
  const C37 = C36 / G, K37 = K36 / G;

  return {
    runs1: C37, runs2: K37, score162: { team1: C36, team2: K36 },
    cells: { E3: L1.woba, E7: L1.xwoba, E11: L1.qpa, D11: L1.avgPa, L3: L2.woba, L7: L2.xwoba, L11: L2.qpa, K11: L2.avgPa,
      E15: S1.avg, M15: S2.avg, H15: S1.paPerOut, P15: S2.paPerOut, H17: B1.paPerOut, P17: B2.paPerOut,
      F17: B1.bfPer, N17: B2.bfPer, H18, P18, H19, P19, F2, M2, C19, K19, C20, K20, C25, K25, C26: rG1, K26: rG2, C28, K28,
      C21, K21, C29, K29, C31, K31, C33, K33, C36, K36, C37, K37 },
  };
}

module.exports = { project, lineupStats, starterStats, bullpenStats };
