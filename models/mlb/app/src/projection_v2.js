'use strict';
// Model v2 (2026-10-09 audit). A rate-based run model shown beside the Tester chain. The 2026-10-09 audit wrongly
// called the chain's R/QPA divisor a sign bug; it is J's run-efficiency rule (wOBA/xwOBA read together with the runs a
// team actually scores), and v2 now carries that rule too (see "Run efficiency" below). Fitted and backtested on Retrosheet
// 2015-2025: models/mlb-app/backtest_chain.py in the Sports Analytics project.
//
//   player rate  = (cur PA x cur QPA + 0.6 x prev PA x prev QPA + N x league) / (cur PA + 0.6 x prev PA + N)
//                  N = 220 PA for hitters, 250 for starters, 600 for a bullpen (regression to the mean),
//                  then divided by sqrt((1 + home park of the player's team) / 2) so Coors hitters aren't counted twice
//   lineup       = mean of the 9 hitters
//   pitching     = 0.581 x starter + 0.419 x bullpen   (starters face 58% of batters; league average bullpen
//                  when no FanGraphs bullpen file is loaded)
//   runs         = league runs per team-game x exp(c0 + c1*(lineup-lg)*100 + c2*(pitching-lg)*100
//                  + c3*(starter K-BB% - lg)*100*0.581 + c4*home + c5*ln(park))
//   win prob     = negative-binomial runs for each team, ties (extra innings) 52% home, then a fitted calibration
//   with market  = logistic(a + b*logit(no-vig market) + c*logit(model)) - see marketBlend()
//
// Model v3 (2026-10-10) = the same formula with four more terms in the exponent, coefficients refit together:
//   + L_k*(lineup K% - lg)*100 + L_bb*(lineup BB% - lg)*100 + L_iso*(lineup ISO - lg)*100 + temp*(temp F - 73.5)/10
//   each hitter's K%/BB%/ISO regressed like QPA but with smaller N (they settle faster); a missing temperature
//   or skill adds 0. totalsProb() turns the two run means into over/under probabilities for a total.
//
// Run efficiency (2026-10-16, J's design rule): + eff * ln(team efficiency) in the exponent, i.e. runs x efficiency^eff.
//   efficiency = team runs / runs its wOBA should produce at the league conversion rate (data.js teamEfficiency), so a
//   lineup with a great wOBA that keeps scoring few runs is not projected to score a lot. eff is fitted (v2 0.20, v3
//   0.15): the share of a team's efficiency gap that has carried forward 2015-2025. A missing team table adds 0.
// R/G anchor (2026-10-16, J): + anchor * ln(team R/G regressed / league), so against an average pitcher the projection
//   leans toward the team's own season scoring. Catcher framing + fielders (model "v3d", shown beside v3, not used for
//   picks until checked live): + framing * catcher extra-strike pp + fld_def * fielders' OAA per chance, fielding side.

function qpaOf(rec) {
  if (!rec) return null;
  const q = Number.isFinite(rec.woba) && Number.isFinite(rec.xwoba) ? (rec.woba + rec.xwoba) / 2 : rec.woba;
  return Number.isFinite(q) && rec.pa > 0 ? q : null;
}

// cur, prev: {pa, woba, xwoba} or null. Returns { rate, pa (effective sample), fromPrior (true if no data) }
function shrunkRate(cur, prev, prior, priorN, prevWeight) {
  let n = 0, s = 0;
  const qc = qpaOf(cur), qp = qpaOf(prev);
  if (qc != null) { n += cur.pa; s += cur.pa * qc; }
  if (qp != null) { n += prevWeight * prev.pa; s += prevWeight * prev.pa * qp; }
  return { rate: (s + priorN * prior) / (n + priorN), pa: n, fromPrior: n === 0 };
}

function shrunkKbb(cur, prev, prior, priorN, prevWeight) {
  let n = 0, s = 0;
  for (const [r, w] of [[cur, 1], [prev, prevWeight]]) {
    if (r && r.pa > 0 && Number.isFinite(r.k) && Number.isFinite(r.bb)) { n += w * r.pa; s += w * r.pa * (r.k - r.bb) / 100; }
  }
  return n ? (s + priorN * prior) / (n + priorN) : null;
}

// One fast-settling hitter skill (v3). field 'k'/'bb' are percents per PA, 'iso' is a rate per AB.
function shrunkSkill(cur, prev, field, prior, priorN, prevWeight) {
  let n = 0, s = 0;
  for (const [r, w] of [[cur, 1], [prev, prevWeight]]) {
    if (!r || !(r.pa > 0) || !Number.isFinite(r[field])) continue;
    const d = field === 'iso' ? (r.ab > 0 ? r.ab : 0.89 * r.pa) : r.pa, x = field === 'iso' ? r[field] : r[field] / 100;
    n += w * d; s += w * d * x;
  }
  return (s + priorN * prior) / (n + priorN);
}

const parkNeutral = pf => Math.sqrt((1 + (Number.isFinite(pf) ? pf : 1)) / 2);

// side: { teamId, lineup: [{cur, prev}] x9, starter: {cur, prev}, bullpen: {bf, woba} | null }
// ctx:  { lg: { woba, kbb, runs }, parkFactors: {teamId: pf}, homeTeamId }
function sideRates(side, ctx, v) {
  const lg = ctx.lg, pn = parkNeutral(ctx.parkFactors[side.teamId]);
  const hitters = side.lineup.map(h => shrunkRate(h.cur, h.prev, lg.woba, v.shrinkPa.hitter, v.prevSeasonWeight));
  // a hitter with no MLB stats at all: a below-average rookie prior
  const rates = hitters.map(h => (h.fromPrior ? lg.woba * v.unknownHitter : h.rate) / pn);
  const sp = shrunkRate(side.starter.cur, side.starter.prev, lg.woba * v.starterPriorScale, v.shrinkPa.starter, v.prevSeasonWeight);
  const spRate = (sp.fromPrior ? lg.woba * v.unknownStarter : sp.rate) / pn;
  const kbbRaw = shrunkKbb(side.starter.cur, side.starter.prev, lg.kbb * v.starterKbbPriorScale, v.shrinkPa.starter, v.prevSeasonWeight);
  const kbb = kbbRaw == null ? (sp.fromPrior ? lg.kbb * v.unknownStarterKbb : lg.kbb) : kbbRaw;
  const bp = side.bullpen && side.bullpen.bf > 0 && Number.isFinite(side.bullpen.woba)
    ? (side.bullpen.bf * side.bullpen.woba + v.shrinkPa.bullpen * lg.woba) / (side.bullpen.bf + v.shrinkPa.bullpen) / pn : lg.woba;
  let skills = null;
  if (v.skillShrink && lg.hit) {
    const mean = f => side.lineup.reduce((a, h) => a + shrunkSkill(h.cur, h.prev, f, lg.hit[f], v.skillShrink[f], v.prevSeasonWeight), 0) / side.lineup.length;
    skills = { k: mean('k'), bb: mean('bb'), iso: mean('iso'), hittersWithIso: side.lineup.filter(h => [h.cur, h.prev].some(r => r && Number.isFinite(r.iso))).length };
  }
  const eff = side.efficiency || null, anc = side.anchor || null;
  return { efficiency: eff ? eff.efficiency : null, lnEff: eff ? eff.lnEff : 0, anchor: anc ? anc.anchor : 0, seasonRpg: anc ? anc.rpg : null, defense: side.defense || null, platoon: Number.isFinite(side.platoon) ? side.platoon : null, tandem: Number.isFinite(side.tandem) ? side.tandem : null, lineup: rates.reduce((a, b) => a + b, 0) / rates.length, hitters: rates, hittersFromPrior: hitters.filter(h => h.fromPrior).length, skills,
    starter: spRate, starterPa: sp.pa, starterKbb: kbb, starterKbbKnown: kbbRaw != null, bullpen: bp, bullpenKnown: bp !== lg.woba };
}

function expectedRuns(bat, pit, home, ctx, v) {
  const c = v.coefficients, lg = ctx.lg, sh = v.starterShare;
  const pitching = sh * pit.starter + (1 - sh) * pit.bullpen;
  const lin = c.const + c.lineup * (bat.lineup - lg.woba) * 100 + c.pitch * (pitching - lg.woba) * 100
    + c.sp_kbb * (pit.starterKbb - lg.kbb) * 100 * sh + c.home * (home ? 1 : 0) + c.ln_pf * Math.log(ctx.parkFactors[ctx.homeTeamId] || 1)
    + (bat.skills && c.L_k != null ? c.L_k * (bat.skills.k - lg.hit.k) * 100 + c.L_bb * (bat.skills.bb - lg.hit.bb) * 100 + c.L_iso * (bat.skills.iso - lg.hit.iso) * 100 : 0)
    + (c.temp != null && Number.isFinite(ctx.tempF) ? c.temp * (ctx.tempF - v.tempCenterF) / 10 : 0)
    + (c.eff != null && Number.isFinite(bat.lnEff) ? c.eff * bat.lnEff : 0)
    + (c.anchor != null && Number.isFinite(bat.anchor) ? c.anchor * bat.anchor : 0)
    + (c.framing != null && pit.defense ? c.framing * pit.defense.framing + c.fld_def * pit.defense.fielding : 0)
    + (c.platoon != null && Number.isFinite(bat.platoon) ? c.platoon * bat.platoon : 0)
    + (c.tandem != null && Number.isFinite(pit.tandem) ? c.tandem * pit.tandem : 0);
  return lg.runs * Math.exp(lin);
}

function nbPmf(mean, alpha, maxK) {
  const r = 1 / alpha, p = r / (r + mean), out = [Math.pow(p, r)];
  for (let k = 1; k <= maxK; k++) out.push(out[k - 1] * (k - 1 + r) / k * (1 - p));
  return out;
}

function winProbHome(runsHome, runsAway, v) {
  const K = 40, ph = nbPmf(runsHome, v.nbAlpha, K), pa = nbPmf(runsAway, v.nbAlpha, K);
  let win = 0, tie = 0, cdfA = 0;
  for (let k = 0; k <= K; k++) { win += ph[k] * cdfA; tie += ph[k] * pa[k]; cdfA += pa[k]; }
  const raw = win + v.tieHomeWin * tie;
  const lo = Math.log(raw / (1 - raw)), cal = v.calibration.intercept + v.calibration.slope * lo;
  return { raw, calibrated: 1 / (1 + Math.exp(-cal)), tie };
}

function projectV2(away, home, ctx, v) {
  const A = sideRates(away, ctx, v), H = sideRates(home, ctx, v);
  const runsAway = expectedRuns(A, H, false, ctx, v), runsHome = expectedRuns(H, A, true, ctx, v);
  const wp = winProbHome(runsHome, runsAway, v);
  return { runsAway, runsHome, homeWinProb: wp.calibrated, awayWinProb: 1 - wp.calibrated, rawHomeWinProb: wp.raw, tiePct: wp.tie,
    detail: { league: ctx.lg, away: A, home: H } };
}

// Over/under for total runs: the two teams' negative-binomial scores added up. Returns, per line,
// { line, over, under, push, overNoPush } (overNoPush = over / (over + under), what a bet that refunds a push wins)
// and pOver / pUnder: overNoPush shrunk toward 50% by calSlope (the raw distribution is overconfident against a real
// betting line; slope fit on 2021-22 opening totals, see v3_coefficients.json). These are the numbers to bet from.
function totalsProb(runsHome, runsAway, alpha, lines, calSlope) {
  const K = 40, ph = nbPmf(runsHome, alpha, K), pa = nbPmf(runsAway, alpha, K), tot = new Array(2 * K + 1).fill(0);
  for (let i = 0; i <= K; i++) for (let j = 0; j <= K; j++) tot[i + j] += ph[i] * pa[j];
  let cum = 0, median = null;
  for (let t = 0; t < tot.length; t++) { cum += tot[t]; if (median == null && cum >= 0.5) median = t; }
  const rows = lines.map(line => {
    let over = 0, under = 0, push = 0;
    tot.forEach((p, t) => { if (t > line) over += p; else if (t < line) under += p; else push += p; });
    const raw = over / (over + under), b = Number.isFinite(calSlope) ? calSlope : 1;
    const pOver = 1 / (1 + Math.exp(-b * Math.log(raw / (1 - raw))));
    return { line, over, under, push, overNoPush: raw, pOver, pUnder: 1 - pOver };
  });
  return { mean: runsHome + runsAway, median, lines: rows };
}

// Model blended with the no-vig market probability (both for the HOME team). Returns the home probability.
function marketBlend(pHomeModel, pHomeMarket, b) {
  const lo = p => Math.log(p / (1 - p));
  const z = b.const + b.mkt * lo(pHomeMarket) + b.model * lo(pHomeModel);
  return 1 / (1 + Math.exp(-z));
}

module.exports = { projectV2, sideRates, marketBlend, shrunkRate, shrunkKbb, shrunkSkill, winProbHome, nbPmf, totalsProb, qpaOf };
