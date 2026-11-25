'use strict';
// Run with:  node test/tests.js
// Uses real payloads saved in test/fixtures (MLB schedule, Savant CSVs, Baseball-Reference page).

const IS_NODE = typeof process !== 'undefined' && process.versions && process.versions.node;
const req = IS_NODE ? (n => require(n)) : require; // Edge harness injects its own require
const U = req('../src/util'), D = req('../src/data'), P = req('../src/projection'), M = req('../src/montecarlo');
const G = req('../src/game'), cfg = req('../src/config'), V2 = req('../src/projection_v2');
const L = req('../src/playlog'), O = req('../src/odds');

function fixture(name) {
  if (IS_NODE) return require('fs').readFileSync(require('path').join(__dirname, 'fixtures', name), 'utf8').replace(/^\uFEFF+/, '');
  return FIXTURES[name];
}

const results = [];
function check(name, ok, detail) { results.push({ name, ok: !!ok, detail: detail === undefined ? '' : String(detail) }); }
function near(name, actual, expected, tol) {
  check(name, Math.abs(actual - expected) <= tol, `actual ${actual} expected ${expected} (tol ${tol})`);
}

// ---------- util ----------
{
  const t = q => { const x = U.findTeam(q); return x ? x.nick : null; };
  check('team aliases', t('DBacks') === 'Diamondbacks' && t('D-Backs') === 'Diamondbacks' && t('As') === 'Athletics' && t("A's") === 'Athletics' &&
    t('Os') === 'Orioles' && t("O's") === 'Orioles' && t('NYY') === 'Yankees' && t('KC') === 'Royals' && t('yank') === 'Yankees' && t('white sox') === 'White Sox' && t('KCR') === 'Royals' && t('WSH') === 'Nationals',
    ['DBacks', 'D-Backs', 'As', "A's", 'Os', 'NYY', 'KC', 'yank', 'white sox', 'KCR', 'WSH'].map(t).join(','));
  check('ambiguous team names rejected', U.findTeam('Sox') === null && U.findTeam('Chicago') === null && U.findTeam('zzz') === null);
  check('30 teams', U.TEAMS.length === 30 && Object.keys(cfg.parkFactors).length === 30 && U.TEAMS.every(t => cfg.parkFactors[t.id]));
  near('ipToDecimal baseball notation 397.2', U.ipToDecimal('397.2'), 397 + 2 / 3, 1e-9);
  near('ipToDecimal already decimal 98.6666', U.ipToDecimal('98.6666'), 98.6666, 1e-9);
  near('ipToDecimal 48.0', U.ipToDecimal('48.0'), 48, 1e-9);
  check('name normalization', U.normName('Heriberto Hernández') === U.normName('Heriberto Hernandez') && U.normName('Ronald Acuña Jr.') === U.normName('Ronald Acuna'));
}

// ---------- parsers on real fixtures ----------
const savantP = D.parseSavantPlayers(fixture('savant_pitchers_2026.csv'));
const savantB = D.parseSavantPlayers(fixture('savant_batters_2026.csv'));
{
  const b = savantP.find(p => p.id === 690925);
  check('Savant pitchers parse', savantP.length > 300 && b && b.name === 'Clayton Beeter' && b.pa === 209 && Math.abs(b.woba - 0.314) < 1e-9 && Math.abs(b.ip - 48) < 1e-9, `${savantP.length} rows`);
  check('Savant batters parse', savantB.length > 400 && savantB[0].id > 0 && Number.isFinite(savantB[0].woba), `${savantB.length} rows`);
  const bb = D.parseBbrefTeams(fixture('bbref_2026.html'));
  const ari = bb.get(109);
  check('BBRef 30 teams', bb.size === 30, bb.size);
  check('BBRef ARI row', ari && ari.games === 153 && ari.pa === 5794 && ari.runs === 678, JSON.stringify(ari));
}

// ---------- FanGraphs-style CSVs ----------
{
  const fg = D.parseFgBatters('﻿"Name","Team","PA","wOBA","xwOBA","MLBAMID"\n"Elly De La Cruz","CIN",600,".370",".360",682829\n"Ronald Acuña Jr.","ATL",500,0.400,0.390,660670\n');
  check('FG batters parse', fg.length === 2 && fg[0].id === 682829 && Math.abs(fg[1].woba - 0.4) < 1e-9);
  let err = ''; try { D.parseFgBatters('Name,PA,wOBA\nA,1,.3\n'); } catch (e) { err = e.message; }
  check('FG batters missing xwOBA column is reported', /xwoba/i.test(err), err);
  const bp = D.parseTeamTable('Team,IP,TBF,wOBA\nYankees,397.2,1682,.300\nKCR,400,1700,.310\nAthletics,390.1,1650,.320\nLg Avg,1,1,.3\n', D.BULLPEN_SPEC);
  check('team table parse', bp.teams.size === 3 && Math.abs(bp.teams.get(147).ip - (397 + 2 / 3)) < 1e-9 && bp.teams.get(118).bf === 1700 && bp.unknownTeams.join() === 'Lg Avg', JSON.stringify([...bp.teams]));
  const w = D.parseTeamTable('Team,wRC+\nDodgers,112\n', D.WRC_SPEC), f = D.parseTeamTable('Tm,FRV\nDodgers,-3\n', D.FRV_SPEC);
  check('wRC+ / FRV parse', w.teams.get(119).wrcPlus === 112 && f.teams.get(119).frv === -3);
}

// ---------- window rule + blending ----------
{
  const S = (yr, g, c) => D.seasonsNeeded(yr, g, c || cfg).join('+');
  check('window: 2026 with 153 games -> 2026 only', S(2026, 153) === '2026');
  check('window: 2027 with 40 games -> 2026+2027', S(2027, 40) === '2026+2027');
  check('window: exactly 100 -> current only; 99 -> blend', S(2027, 100) === '2027' && S(2027, 99) === '2026+2027');
  check('window threshold is configurable', S(2027, 60, { sampleWindow: { minCurrentSeasonGames: 50 } }) === '2027' && S(2027, 60, { sampleWindow: { minCurrentSeasonGames: 120 } }) === '2026+2027');
  const idx = D.indexPlayers([[{ id: 1, name: 'A B', pa: 100, woba: 0.3, xwoba: 0.3 }], [{ id: 1, name: 'A B', pa: 300, woba: 0.4, xwoba: 0.2 }]]);
  const a = idx.byId.get(1);
  check('player blend is PA-weighted', a.pa === 400 && Math.abs(a.woba - 0.375) < 1e-9 && Math.abs(a.xwoba - 0.225) < 1e-9);
  const tb = D.blendTeamTables([new Map([[1, { games: 100, pa: 3800, runs: 450 }]]), new Map([[1, { games: 60, pa: 2300, runs: 250 }]])]).get(1);
  check('team offense blend sums', tb.games === 160 && tb.pa === 6100 && tb.runs === 700);
  const bb = D.blendBullpens([new Map([[1, { ip: 100, bf: 400, woba: 0.3 }]]), new Map([[1, { ip: 50, bf: 200, woba: 0.36 }]])]).get(1);
  check('bullpen blend: sums + BF-weighted wOBA', bb.ip === 150 && bb.bf === 600 && Math.abs(bb.woba - 0.32) < 1e-9);
}

// ---------- projection chain vs. the actual spreadsheet ----------
const H = (a) => a.map(([woba, xwoba, pa]) => ({ woba, xwoba, pa }));
const T1 = {
  lineup: H([[.3404007816, .3520676272, 279], [.295441783, .2818495929, 129], [.3669968249, .3314325388, 455], [.3050660625, .2695567634, 286], [.3415840564, .3210886252, 459], [.3303666696, .3239778355, 357], [.2809449154, .3291277363, 138], [.3666030603, .3094336005, 142], [.3155092105, .3174821085, 205]]),
  starter: { woba: .298, xwoba: .309, tbf: 399, ip: 98.6666 }, bullpen: { woba: .2919963509, ip: 397.2, bf: 1682 },
  team: { games: 109, pa: 4100, runs: 481, frv: 7, wrcPlus: 100 },
};
const T2 = {
  lineup: H([[.3930843708, .3518325668, 449], [.6295998096, .4275000095, 2], [.3136014994, .3292001527, 235], [.3041511998, .3130198, 448], [.3028755892, .3241142812, 188], [.2973191994, .2960008157, 229], [.3293345173, .2199625015, 24], [.357774851, .3985821156, 89], [.3028638939, .3178075874, 325]]),
  starter: { woba: .388, xwoba: .372, tbf: 61, ip: 12.333333 }, bullpen: { woba: .3412239246, ip: 506, bf: 2251 },
  team: { games: 109, pa: 4221, runs: 596, frv: -1, wrcPlus: 106 },
};
{
  const old = P.project(T1, T2, cfg, { legacy: { h18Raw: true, p18Ip: true, m2k19Raw: true, frv: true } });
  const c = old.cells;
  // Cached values from the workbook's Tester sheet (as received)
  near('legacy F2', c.F2, 0.33833981925828005, 1e-9); near('legacy M2', c.M2, 0.2973335488127466, 1e-9);
  near('legacy H18', c.H18, 1.399358228357108, 1e-9); near('legacy P18', c.P18, 1.509922829150707, 1e-9);
  near('legacy C19', c.C19, 39.371259706665576, 1e-9); near('legacy K19', c.K19, 37.87311689340896, 1e-9);
  near('legacy C21', c.C21, 4.793412327, 1e-8); near('legacy K21', c.K21, 4.865144661, 1e-8);
  near('legacy C37', old.runs1, 4.797116031, 1e-8); near('legacy K37', old.runs2, 4.876255772, 1e-8);
  near('legacy Pythagorean H51', M.pythagorean(old.score162.team1, old.score162.team2, 1.83).pA, 0.4925145988, 1e-9);
  near('legacy C11 (was hardcoded)', c.D11, 272.2222222, 1e-6); near('legacy K11 (was hardcoded)', c.K11, 221, 1e-9);

  const e17 = P.project(T1, T2, cfg, { legacy: { p18Ip: true, frv: true } }); // only fix #1 applied
  near('fix#1 only: M2 (independent recompute)', e17.cells.M2, 0.306794579930533, 1e-9);
  near('fix#1 only: H18', e17.cells.H18, 1.36991291148337, 1e-9);
  near('fix#1 only: K19', e17.cells.K19, 37.4500442362103, 1e-9);
  near('fix#1 only: K37', e17.runs2, 4.97498584467111, 1e-9);
  near('fix#1 only: Pythagorean', M.pythagorean(e17.score162.team1, e17.score162.team2, 1.83).pA, 0.48334967505511, 1e-9);

  const fixed = P.project(T1, T2, cfg);
  check('fully corrected chain returns finite runs', Number.isFinite(fixed.runs1) && Number.isFinite(fixed.runs2), `${fixed.runs1} / ${fixed.runs2}`);
  // FRV fix makes the 162-game score consistent with the per-game row: C33/162 == C21 - K31
  near('FRV fix: C33/162 equals C21 - K31 (row 34 == row 32)', fixed.cells.C33 / 162, fixed.cells.C21 - fixed.cells.K31, 1e-12);
  near('FRV fix: K33/162 equals K21 - C31', fixed.cells.K33 / 162, fixed.cells.K21 - fixed.cells.C31, 1e-12);
  let threw = '';
  try { P.project({ ...T1, starter: { ...T1.starter, woba: NaN } }, T2, cfg); } catch (e) { threw = e.message; }
  check('missing input throws (never guesses)', /starter wOBA/.test(threw), threw);

  // No-FanGraphs mode vs. the full-season backtest's no-bullpen formula (compute_projections.ps1), computed independently
  const nf1 = { ...T1, bullpen: null, team: { ...T1.team, wrcPlus: 100, frv: 0 } }, nf2 = { ...T2, bullpen: null, team: { ...T2.team, wrcPlus: 100, frv: 0 } };
  const nf = P.project(nf1, nf2, cfg);
  const l1 = P.lineupStats(T1.lineup), l2 = P.lineupStats(T2.lineup), s1 = P.starterStats(T1.starter), s2 = P.starterStats(T2.starter);
  const paG1 = 4100 / 109, paG2 = 4221 / 109;
  const eF2 = (l1.qpa * l1.avgPa + s2.avg * s2.tbf) / (l1.avgPa + s2.tbf), eM2 = (l2.qpa * l2.avgPa + s1.avg * s1.tbf) / (l2.avgPa + s1.tbf);
  const eC19 = (s2.paPerOut * 27 * s2.tbf + paG1 * l1.avgPa) / (s2.tbf + l1.avgPa), eK19 = (s1.paPerOut * 27 * s1.tbf + paG2 * l2.avgPa) / (s1.tbf + l2.avgPa);
  near('no-FanGraphs: away runs = backtest no-bullpen formula', nf.runs1, eF2 * eC19 * (481 / 109) / (l1.qpa * paG1), 1e-12);
  near('no-FanGraphs: home runs = backtest no-bullpen formula', nf.runs2, eM2 * eK19 * (596 / 109) / (l2.qpa * paG2), 1e-12);
  near('no-FanGraphs: wRC+ 100 and FRV 0 leave the 162-game score unchanged (C36 = C29)', nf.cells.C36, nf.cells.C29, 1e-9);
  let mixed = ''; try { P.project(nf1, T2, cfg); } catch (e) { mixed = e.message; }
  check('no-FanGraphs: a bullpen on one side only is refused', /both bullpens or neither/.test(mixed), mixed);
}

// ---------- season-long QPA option ----------
{
  const q = D.teamSeasonQpa([[
    { team: 'NYY', pa: 100, woba: 0.30, xwoba: 0.34 }, { team: 'Yankees', pa: 300, woba: 0.40, xwoba: 0.30 },
    { team: '- - -', pa: 500, woba: 0.5, xwoba: 0.5 }, { team: 'BOS', pa: 200, woba: 0.32, xwoba: 0.32 }]]).get(147);
  near('season QPA = avg of PA-weighted wOBA and xwOBA over the team\'s batters (traded "- - -" rows skipped)', q.qpa, ((0.30 * 100 + 0.40 * 300) / 400 + (0.34 * 100 + 0.30 * 300) / 400) / 2, 1e-12);
  check('season QPA counts PA', q.pa === 400);

  const seasonCfg = { ...cfg, rqpaBasis: 'season' };
  const base = P.project(T1, T2, cfg);
  const L1q = P.lineupStats(T1.lineup).qpa, L2q = P.lineupStats(T2.lineup).qpa;
  const same = P.project({ ...T1, team: { ...T1.team, seasonQpa: L1q } }, { ...T2, team: { ...T2.team, seasonQpa: L2q } }, seasonCfg);
  near('season basis == tonight basis when season QPA equals tonight QPA (runs1)', same.runs1, base.runs1, 1e-12);
  near('season basis == tonight basis when season QPA equals tonight QPA (runs2)', same.runs2, base.runs2, 1e-12);

  const better = { ...T1, lineup: T1.lineup.map((h, i) => (i === 0 ? { ...h, woba: h.woba + 0.05, xwoba: h.xwoba + 0.05 } : h)) };
  const tonightUp = P.project(better, T2, cfg).runs1, seasonUp = P.project({ ...better, team: { ...T1.team, seasonQpa: L1q } }, { ...T2, team: { ...T2.team, seasonQpa: L2q } }, seasonCfg).runs1;
  check('tonight basis: better lineup lowers the projection (efficiency cancels it)', tonightUp < base.runs1, `${base.runs1.toFixed(4)} -> ${tonightUp.toFixed(4)}`);
  check('season basis: better lineup raises the projection', seasonUp > base.runs1, `${base.runs1.toFixed(4)} -> ${seasonUp.toFixed(4)}`);
  let threw = ''; try { P.project(T1, T2, seasonCfg); } catch (e) { threw = e.message; }
  check('season basis without season QPA throws, never guesses', /season QPA/.test(threw), threw);
}

// ---------- Monte Carlo ----------
{
  near('exact Poisson equals sheet Poisson!E6', M.exactPoisson(4.65320255, 4.729968099).pA, 0.48863901050906616, 1e-9);
  const rng = M.seeded(12345), n = 200000; let s = 0, s2 = 0;
  for (let i = 0; i < n; i++) { const v = M.poissonRandom(4.554, rng); s += v; s2 += v * v; }
  near('Knuth Poisson mean', s / n, 4.554, 0.03); near('Knuth Poisson variance', s2 / n - (s / n) ** 2, 4.554, 0.06);
  let ns = 0, ns2 = 0; for (let i = 0; i < n; i++) { const v = M.normalRandom(-0.58, 3.03, rng); ns += v; ns2 += v * v; }
  near('Box-Muller mean', ns / n, -0.58, 0.04); near('Box-Muller std', Math.sqrt(ns2 / n - (ns / n) ** 2), 3.03, 0.04);
  const plug = M.exactPoisson(4.65, 4.73).pA;
  const h0 = M.hierarchical({ lambdaA: 4.65, lambdaB: 4.73, errorMean: 0, errorStd: 0, iterations: 400000, rng: M.seeded(7) });
  check('hierarchical with std=0, mean=0 converges to plug-in Poisson', Math.abs(h0.pA - plug) < 4 * h0.standardError, `${h0.pA.toFixed(5)} vs ${plug.toFixed(5)} (SE ${h0.standardError.toFixed(5)})`);
  const pf = M.exactPoisson(5.5, 3.8).pA;
  const h1 = M.hierarchical({ lambdaA: 5.5, lambdaB: 3.8, errorMean: -0.58, errorStd: 3.03, iterations: 300000, rng: M.seeded(9) });
  check('hierarchical pulls a favorite toward 50%', h1.pA > 0.5 && h1.pA < pf, `${h1.pA.toFixed(4)} vs plug-in ${pf.toFixed(4)}`);
  near('win probs sum to 1', h1.pA + h1.pB, 1, 1e-12);
  near('Pythagorean symmetric', M.pythagorean(5, 5, 1.83).pA, 0.5, 1e-12);
  // "Probability of Win" (G48) + "Predicted Odds" (G50): cached values from real 2026 game sheets and the live Tester
  near('Pythagorean = "Giants @ Phillies 430" G48', M.pythagorean(654.3570223, 737.1961631, 1.83).pA, 0.4456808056, 1e-9);
  near('Pythagorean = "Tigers @ Padres 327" G48', M.pythagorean(756.1968468, 678.5625169, 1.83).pA, 0.5493970787, 1e-9);
  near('Pythagorean = "Pirates @ Mets 326" G48', M.pythagorean(552.1971875, 665.7576378, 1.83).pA, 0.4152636364, 1e-9);
  near('Pythagorean = live Tester H51', M.pythagorean(729.8894896, 789.9534351, 1.83).pA, 0.4638835167, 1e-9);
  near('fair moneyline = "Giants @ Phillies 430" G50 (underdog)', U.fairMoneyline(0.4456808056), 124.3758285, 1e-6);
  near('fair moneyline = "Tigers @ Padres 327" G50 (favorite)', U.fairMoneyline(0.5493970787), -121.9248817, 1e-6);
  near('fair moneyline = live Tester H53 / J53', U.fairMoneyline(0.4638835167) + U.fairMoneyline(0.5361164833), 0, 1e-6);
  near('fair moneyline = live Tester H53', U.fairMoneyline(0.4638835167), 115.5713587, 1e-6);
  check('fair moneyline at exactly 50% is +100', U.fairMoneyline(0.5) === 100);
  // Pythagenpat - worked example from the live Tester (MIA @ WSN, park 0.97)
  const lamA = 4.505490677 * 0.97, lamB = 4.876255772 * 0.97;
  const pp = M.pythagenpat(lamA, lamB, 0.287);
  near('Pythagenpat projected: exponent = (lambdaA + lambdaB)^0.287', pp.exponent, Math.pow((4.505490677 + 4.876255772) * 0.97, 0.287), 1e-12);
  near('Pythagenpat projected: worked example exponent 1.8847', pp.exponent, 1.8847, 5e-5);
  near('Pythagenpat projected: worked example P(MIA)', pp.pA, 0.46281, 5e-6);
  near('Pythagenpat sums to 1', pp.pA + pp.pB, 1, 1e-12);
  const ps = M.pythagenpat(lamA, lamB, 0.287, 4.412844037 + 5.467889908);
  near('Pythagenpat season R/G: exponent = (C26 + K26)^0.287', ps.exponent, 1.9298, 5e-5);
  near('Pythagenpat season R/G: worked example P(MIA)', ps.pA, 0.46192, 5e-6);
  near('Pythagenpat reproduces the live Tester K51 (0.285, season R/G, C36/K36)',
    M.pythagenpat(789.9534351, 729.8894896, 0.285, 4.412844037 + 5.467889908).pA, 0.5379046315, 1e-9);
  check('Pythagenpat: win ratio is scale-free (only the exponent depends on the scale)',
    Math.abs(M.pythagenpat(4.1, 5.2, 0.287, 9).pA - M.pythagenpat(4.1 * 162, 5.2 * 162, 0.287, 9).pA) < 1e-12);
  check('Pythagenpat: a higher run environment pushes the favorite further from 50%',
    M.pythagenpat(5.8, 5.0, 0.287).pA > M.pythagenpat(4.64, 4.0, 0.287).pA && M.pythagenpat(4.64, 4.0, 0.287).pA > 0.5);
}


// ---------- model v2 (2026-10-09 audit) ----------
{
  const lg = { woba: 0.315, kbb: 0.13, runs: 4.4 };
  const hit = w => ({ cur: { pa: 300, woba: w, xwoba: w }, prev: { pa: 500, woba: w, xwoba: w } });
  const side = (id, w, sp) => ({ teamId: id, lineup: Array(9).fill(hit(w)), starter: { cur: { pa: 400, woba: sp, xwoba: sp, k: 24, bb: 7 }, prev: null }, bullpen: null });
  const ctx = { lg, parkFactors: cfg.parkFactors, homeTeamId: 147 };
  const runs = [0.29, 0.315, 0.34].map(w => V2.projectV2(side(111, w, 0.31), side(147, 0.315, 0.31), ctx, cfg.v2).runsAway);
  check('v2: a better lineup projects MORE runs at the same run efficiency', runs[0] < runs[1] && runs[1] < runs[2], runs.map(x => x.toFixed(3)).join(' < '));
  const better = V2.projectV2(side(111, 0.315, 0.28), side(147, 0.315, 0.31), ctx, cfg.v2), worse = V2.projectV2(side(111, 0.315, 0.34), side(147, 0.315, 0.31), ctx, cfg.v2);
  check('v2: a better opposing starter projects fewer runs', better.runsHome < worse.runsHome);
  // same numbers as the Python backtest's formula (models/mlb-app/backtest_chain.py), checked 2026-10-09
  const even = V2.projectV2(side(110, 0.315, 0.315), side(110, 0.315, 0.315), { lg, parkFactors: { 110: 1.0 }, homeTeamId: 110 }, cfg.v2);
  near('v2: away runs match the Python fit', even.runsAway, 4.14811, 1e-4);
  near('v2: home runs match the Python fit (home edge)', even.runsHome, 4.24165, 1e-4);
  near('v2: evenly matched teams -> home wins 53.1% (calibrated)', even.homeWinProb, 0.53085, 1e-4);
  // J's run-efficiency rule (2026-10-16): same wOBA, fewer actual runs -> fewer projected runs
  const teamRow = (runs, woba) => ({ games: 120, pa: 4600, runs, wobaDen: 4500, wobaNum: woba * 4500 });
  const league = new Map([[1, teamRow(540, 0.315)], [2, teamRow(540, 0.315)], [3, teamRow(540, 0.315)]]);
  const withTeam = (runs, woba) => { const m = new Map(league); m.set(111, teamRow(runs, woba)); return { cur: m, prev: null }; };
  const effOf = (runs, woba) => D.teamEfficiency(111, withTeam(runs, woba), cfg.v2.efficiency);
  const lgRuns = 4500 * (540 * 3 / 13500), eAvg = effOf(lgRuns, 0.315);
  near('efficiency: a league-average team converts at 1.000', eAvg.efficiency, 1.0, 0.01);
  const great = effOf(480, 0.350);
  check('efficiency: .350 wOBA team scoring only 480 runs is below 1', great.efficiency < 0.9, great.efficiency.toFixed(3));
  const sideE = (w, e) => Object.assign(side(111, w, 0.31), { efficiency: e });
  const r_eff = V2.projectV2(sideE(0.35, great), side(147, 0.315, 0.31), ctx, cfg.v2).runsAway, r_plain = V2.projectV2(side(111, 0.35, 0.31), side(147, 0.315, 0.31), ctx, cfg.v2).runsAway;
  near('efficiency: runs x efficiency^eff (v2)', r_eff / r_plain, Math.pow(great.efficiency, cfg.v2.coefficients.eff), 1e-9);
  check('efficiency: inefficient lineup projects fewer runs than its wOBA alone', r_eff < r_plain, r_eff.toFixed(3) + ' < ' + r_plain.toFixed(3));
  check('efficiency: a better lineup with the same efficiency still projects more runs', V2.projectV2(sideE(0.35, eAvg), side(147, 0.315, 0.31), ctx, cfg.v2).runsAway > V2.projectV2(sideE(0.32, eAvg), side(147, 0.315, 0.31), ctx, cfg.v2).runsAway);
  const bb = D.parseBbrefTeams(fixture('bbref_2026.html')), ari = bb.get(109);
  near('bbref: team wOBA built from the table (ARI 2026)', ari.wobaNum / ari.wobaDen, 0.3127, 0.002);
  const effAri = D.teamEfficiency(109, { cur: bb, prev: null }, cfg.v2.efficiency);
  check('bbref: every team gets an efficiency near 1', Array.from(bb.keys()).every(id => { const e = D.teamEfficiency(id, { cur: bb, prev: null }, cfg.v2.efficiency); return e && e.efficiency > 0.85 && e.efficiency < 1.15; }), effAri && effAri.efficiency.toFixed(3));
  // R/G anchor (2026-10-16): same lineup and same efficiency, a team that has scored more projects more
  const anc = rpg => D.teamAnchor(111, { cur: new Map([[111, { runs: rpg * 100, games: 100 }]]), prev: null }, 4.4, 0.6);
  near('anchor: a league-average R/G team has anchor 0', anc(4.4).anchor, 0, 1e-12);
  near('anchor: 20 games of league average mixed in', Math.exp(anc(5.5).anchor) * 4.4, (550 + 88) / 120, 1e-9);
  const sideA = (r) => Object.assign(side(111, 0.315, 0.31), { anchor: anc(r) });
  const hi = V2.projectV2(sideA(5.5), side(147, 0.315, 0.31), ctx, cfg.v2).runsAway, lo = V2.projectV2(sideA(3.5), side(147, 0.315, 0.31), ctx, cfg.v2).runsAway;
  check('anchor: 5.5 R/G team projects more than a 3.5 R/G team with the same lineup', hi > lo, hi.toFixed(3) + ' > ' + lo.toFixed(3));
  // catcher framing and fielders (v3d)
  const fr = D.parseSavantFraming('player_id,player_name,n_called_pitches,runs_extra_strikes\n1,Good C,6000,12\n2,Bad C,5000,-8\n');
  const oa = D.parseSavantOaa('player_id,last_name,outs_above_average,diff_success_rate_formatted\n10,SS Glove,10,4%\n11,LF Bad,-6,-3%\n');
  near('framing CSV parsed', fr.get(1).runs, 12, 1e-9); near('OAA attempts from success-rate gap', oa.get(10).attempts, 250, 1e-9);
  const lu = cId => [{ id: cId, name: 'C', pos: 'C' }, { id: 10, name: 'SS', pos: 'SS' }, { id: 11, name: 'LF', pos: 'LF' }, { id: 99, name: 'DH', pos: 'DH' }];
  const dd = { framing: { cur: fr, prev: null }, oaa: { cur: oa, prev: null } };
  const good = D.teamDefense(lu(1), dd, 0.6), bad = D.teamDefense(lu(2), dd, 0.6);
  near('framing: 12 runs on 6000 takes -> 96 strikes / 9000 regressed takes', good.framing, 12 / 0.125 / 9000 * 100, 1e-9);
  check('framing: the bad framer is below zero', bad.framing < 0 && good.framing > 0);
  near('fielding: SS share x OAA per regressed chance + LF', good.fielding, (10 / 500 * 0.13 + (-6) / 450 * 0.15) * 100, 1e-9);
  const sideD = (d) => Object.assign(side(147, 0.315, 0.31), { defense: d });
  const vsGood = V2.projectV2(side(111, 0.315, 0.31), sideD(good), ctx, cfg.v3d).runsAway, vsBad = V2.projectV2(side(111, 0.315, 0.31), sideD(bad), ctx, cfg.v3d).runsAway;
  check('v3d: hitters project fewer runs against the good framer', vsGood < vsBad, vsGood.toFixed(3) + ' < ' + vsBad.toFixed(3));
  near('v3d: no defense data = no change', V2.projectV2(side(111, 0.315, 0.31), side(147, 0.315, 0.31), ctx, cfg.v3d).runsAway,
    V2.projectV2(side(111, 0.315, 0.31), sideD({ framing: 0, fielding: 0 }), ctx, cfg.v3d).runsAway, 1e-12);
  // handedness + pairing (v3f)
  const hands = D.parseHands({ people: [{ id: 1, batSide: { code: 'L' }, pitchHand: { code: 'R' } }, { id: 2, batSide: { code: 'R' }, pitchHand: { code: 'R' } }, { id: 9, batSide: { code: 'R' }, pitchHand: { code: 'L' } }] });
  const lefties = Array(9).fill({ id: 1 }), righties = Array(9).fill({ id: 2 });
  check('platoon: lefty lineup is better vs a RHP than vs a LHP', D.lineupPlatoon(lefties, 'R', hands, 0.581) > D.lineupPlatoon(lefties, 'L', hands, 0.581));
  near('platoon: units = league gap x 100 x starter share', D.lineupPlatoon(righties, 'L', hands, 0.581), 0.0122 * 100 * 0.581, 1e-9);
  const pf = D.parseStatcastPairs('woba_value,woba_denom,fielder_2\n0.9,1,5\n0,1,5\n0,1,6\n0,1,6\n0,0,6\n');
  near('pairs: wOBA per catcher', pf.by['5'].w / pf.by['5'].n, 0.45, 1e-9);
  check('pairing: the catcher he allows more with is positive', D.pairTandem({ cur: pf, prev: null }, 5, 0.6, 0.581) > 0 && D.pairTandem({ cur: pf, prev: null }, 6, 0.6, 0.581) < 0);
  check('pairing: unknown catcher = 0', D.pairTandem({ cur: pf, prev: null }, 77, 0.6, 0.581) === 0);
  const sideP = (pl) => Object.assign(side(111, 0.315, 0.31), { platoon: pl });
  check('v3f: a platoon edge adds runs', V2.projectV2(sideP(1), side(147, 0.315, 0.31), ctx, cfg.v3f).runsAway > V2.projectV2(sideP(-1), side(147, 0.315, 0.31), ctx, cfg.v3f).runsAway);
  const shrink = V2.shrunkRate({ pa: 40, woba: 0.45, xwoba: 0.45 }, null, 0.315, 220, 0.6);
  check('v2: a 40-PA hot start is pulled most of the way to league average', shrink.rate > 0.33 && shrink.rate < 0.34, shrink.rate.toFixed(4));
  const nb = V2.nbPmf(4.5, cfg.v2.nbAlpha, 60);
  near('v2: negative-binomial runs sum to 1 and keep the mean', nb.reduce((a, b) => a + b, 0), 1, 1e-6);
  near('v2: negative-binomial mean', nb.reduce((a, p, k) => a + p * k, 0), 4.5, 1e-3);
  const lc = D.leagueContext({ batCur: [{ pa: 100, woba: 0.30, xwoba: 0.32 }], batPrev: [{ pa: 1000, woba: 0.32, xwoba: 0.32 }], pitCur: [], pitPrev: [],
    teamsCur: new Map([[1, { games: 10, runs: 50 }]]), teamsPrev: new Map([[1, { games: 162, runs: 729 }]]) }, cfg.v2);
  check('v2: league context pads a young season with last season', Math.abs(lc.woba - (100 * 0.31 + 20000 * 0.32) / 20100) < 1e-9 && Math.abs(lc.runs - (50 + 600 * 4.5) / 610) < 1e-9 && lc.kbb === cfg.v2.defaults.kbb,
    JSON.stringify(lc));
  const blendHome = V2.marketBlend(0.60, 0.50, cfg.v2.marketBlend);
  check('v2 + market: the blend sits between market and model, close to the market', blendHome > 0.5 && blendHome < 0.53, blendHome.toFixed(4));
}

// ---------- model v3 (2026-10-10: v2 + lineup K%/BB%/ISO + temperature) ----------
{
  const lg = { woba: 0.315, kbb: 0.13, runs: 4.4, hit: { k: 0.22, bb: 0.085, iso: 0.16 } };
  const hit = (k, bb, iso) => ({ cur: { pa: 300, ab: 270, woba: 0.315, xwoba: 0.315, k, bb, iso }, prev: { pa: 500, ab: 450, woba: 0.315, xwoba: 0.315, k, bb, iso } });
  const side = (id, h) => ({ teamId: id, lineup: Array(9).fill(h), starter: { cur: { pa: 400, woba: 0.315, xwoba: 0.315, k: 23, bb: 10 }, prev: null }, bullpen: null });
  const avg = hit(22, 8.5, 0.16), ctx = t => ({ lg, parkFactors: { 110: 1.0 }, homeTeamId: 110, tempF: t });
  const even = V2.projectV2(side(110, avg), side(110, avg), ctx(73.5), cfg.v3);
  // league-average everything at 73.5F: runs = league x exp(const) away, x exp(const + home) home (backtest_v3.py fit)
  const noNew = { ...cfg.v3, coefficients: { ...cfg.v3.coefficients, L_k: 0, L_bb: 0, L_iso: 0, temp: 0 } };
  const base = V2.projectV2(side(110, avg), side(110, avg), ctx(73.5), noNew);
  near('v3: league-average K/BB/ISO at 73.5F add nothing (same as v3 without the new terms)', even.runsAway, base.runsAway, 1e-12);
  near('v3: home edge = exp(home coefficient)', even.runsHome / even.runsAway, Math.exp(cfg.v3.coefficients.home), 1e-9);
  const hot = V2.projectV2(side(110, avg), side(110, avg), ctx(83.5), cfg.v3);
  near('v3: 10 degrees warmer = +3.6% runs', hot.runsAway / even.runsAway, Math.exp(cfg.v3.coefficients.temp), 1e-9);
  near('v3: no temperature posted = average night', V2.projectV2(side(110, avg), side(110, avg), ctx(null), cfg.v3).runsAway, even.runsAway, 1e-12);
  const power = V2.projectV2(side(110, hit(22, 8.5, 0.22)), side(110, avg), ctx(73.5), cfg.v3).runsAway;
  const whiff = V2.projectV2(side(110, hit(28, 8.5, 0.16)), side(110, avg), ctx(73.5), cfg.v3).runsAway;
  const walks = V2.projectV2(side(110, hit(22, 11, 0.16)), side(110, avg), ctx(73.5), cfg.v3).runsAway;
  check('v3: more power and more walks -> more runs, more strikeouts -> fewer', power > even.runsAway && walks > even.runsAway && whiff < even.runsAway, [whiff, even.runsAway, walks, power].map(x => x.toFixed(3)).join(' '));
  check('v2 ignores the v3 inputs (unchanged)', Math.abs(V2.projectV2(side(110, hit(28, 11, 0.22)), side(110, avg), ctx(95), cfg.v2).runsAway - V2.projectV2(side(110, avg), side(110, avg), ctx(60), cfg.v2).runsAway) < 1e-12);
  near('v3: a 30-PA 40% strikeout start is mostly regressed (60 PA prior)', V2.shrunkSkill({ pa: 30, k: 40 }, null, 'k', 0.22, 60, 0.6), (30 * 0.4 + 60 * 0.22) / 90, 1e-12);
  near('v3: ISO is weighted by at-bats (0.89 x PA when AB is missing)', V2.shrunkSkill({ pa: 100, iso: 0.3 }, null, 'iso', 0.16, 160, 0.6), (89 * 0.3 + 160 * 0.16) / 249, 1e-12);
  const tp = V2.totalsProb(4.5, 4.3, cfg.v3.nbAlpha, [8, 8.5, 9], 1);
  check('totals: over + under + push = 1 on every line, no push on a half line', tp.lines.every(t => Math.abs(t.over + t.under + t.push - 1) < 1e-6) && tp.lines[1].push === 0, JSON.stringify(tp.lines.map(t => [t.line, t.over.toFixed(3), t.push.toFixed(3)])));
  near('totals: mean of the summed distribution', tp.mean, 8.8, 1e-12);
  // runs are skewed right, so a projected 8.8 has its median at 8 and over 8.5 is just under 50%
  check('totals: right skew (median below the mean), over 8 > over 8.5 > over 9', tp.median === 8 && tp.lines[0].pOver > tp.lines[1].pOver && tp.lines[1].pOver > tp.lines[2].pOver && tp.lines[1].pOver > 0.45 && tp.lines[1].pOver < 0.5);
  const tc = V2.totalsProb(4.5, 4.3, cfg.v3.nbAlpha, [8.5], cfg.v3.totalsCalSlope).lines[0];
  check('totals: calibrated chance is pulled toward 50% (raw overstates it vs a real line)', Math.abs(tc.pOver - 0.5) < Math.abs(tc.overNoPush - 0.5) && Math.abs(tc.pOver + tc.pUnder - 1) < 1e-12, [tc.overNoPush, tc.pOver].map(x => x.toFixed(4)).join(' -> '));
  const lc = D.leagueContext({ batCur: [{ pa: 100, woba: 0.3, xwoba: 0.3, k: 30, bb: 10, iso: 0.2 }], batPrev: [], pitCur: [], pitPrev: [], teamsCur: new Map(), teamsPrev: new Map() }, cfg.v2);
  near('v3: league K% pads a young season with the default', lc.hit.k, (100 * 0.3 + 20000 * cfg.v2.defaults.k) / 20100, 1e-12);
  const sv = D.parseSavantPlayers('"last_name, first_name",player_id,pa,ab,woba,xwoba,k_percent,bb_percent,isolated_power\n"Doe, Jo",5,100,88,0.33,0.34,21.5,9.1,0.201\n');
  check('Savant batter file: K%, BB%, ISO and AB parsed', sv[0].k === 21.5 && sv[0].bb === 9.1 && sv[0].iso === 0.201 && sv[0].ab === 88, JSON.stringify(sv[0]));
  const blended = D.indexPlayers([[{ id: 5, name: 'Jo Doe', pa: 100, ab: 90, woba: 0.3, xwoba: 0.3, k: 20, bb: 8, iso: 0.2 }, { id: 5, name: 'Jo Doe', pa: 100, ab: 90, woba: 0.3, xwoba: 0.3, k: 30, bb: 8, iso: 0.1 }]]).byId.get(5);
  check('traded player: ISO and AB combine across teams', Math.abs(blended.iso - 0.15) < 1e-12 && blended.ab === 180 && blended.k === 25, JSON.stringify(blended));
}

// ---------- MLB schedule fixture + end-to-end assembly ----------
{
  const sched = JSON.parse(fixture('schedule_2026-09-18.json'));
  const games = G.parseGames(sched, cfg);
  check('schedule parses 15 regular-season games', games.length === 15, games.length);
  // Postseason: F/D/L/W run, spring training / exhibition / All-Star never do
  const mixed = JSON.parse(JSON.stringify(sched));
  const tmpl = mixed.dates[0].games[0];
  mixed.dates[0].games = ['R', 'F', 'D', 'L', 'W', 'S', 'E', 'A'].map((t, i) => ({ ...JSON.parse(JSON.stringify(tmpl)), gamePk: 900 + i, gameType: t, seriesDescription: t === 'F' ? 'Wild Card Game' : undefined }));
  const types = G.parseGames(mixed, cfg).map(x => x.gameType).join('');
  check('game types: regular season + Wild Card/Division/LCS/World Series run; spring, exhibition, All-Star do not', types === 'RFDLW', types);
  const wc = G.parseGames(mixed, cfg).find(x => x.gameType === 'F');
  check('postseason label from the schedule ("Wild Card Game"), fallback label otherwise',
    wc.seriesDescription === 'Wild Card Game' && G.parseGames(mixed, cfg).find(x => x.gameType === 'W').seriesDescription === 'World Series');
  check('postseason games are flagged before you confirm (regular-season stats)', G.readiness(wc).problems.some(p => /^Note: postseason game \(Wild Card Game\)/.test(p)));
  check('regular-season-only setting still possible', G.parseGames(mixed, { ...cfg, gameTypes: ['R'] }).length === 1);
  const reds = G.gamesForTeam(sched, 'Reds', cfg);
  const g = reds.games[0];
  check('Reds game found, home lineup posted / away not', reds.games.length === 1 && g.home.name === 'Cincinnati Reds' && g.home.lineupPosted && !g.away.lineupPosted, JSON.stringify([g.away.lineup.length, g.home.lineup.length]));
  const r = G.readiness(g);
  check('readiness refuses when a lineup is missing and says so', !r.ready && r.problems.some(p => /lineup not posted/.test(p)), r.problems.join(' | '));
  check('unknown team gives a clear error', /doesn't match exactly one MLB team/.test(G.gamesForTeam(sched, 'zzz', cfg).error));
  const notRun = G.runGame(g, {}, cfg);
  check('runGame does not run without lineups', notRun.ok === false && notRun.missing.length > 0);

  // Build a fully-populated game from real Savant fixtures so the whole pipeline runs
  const bats = savantB.filter(b => b.pa >= 150 && Number.isFinite(b.woba) && Number.isFinite(b.xwoba)).slice(0, 18);
  const full = JSON.parse(JSON.stringify(g));
  full.away.lineup = bats.slice(0, 9).map(b => ({ id: b.id, name: b.name })); full.away.lineupPosted = true;
  full.home.lineup = bats.slice(9, 18).map(b => ({ id: b.id, name: b.name })); full.home.lineupPosted = true;
  const bbTeams = D.parseBbrefTeams(fixture('bbref_2026.html'));
  const bullpen = new Map(), wrc = new Map(), frv = new Map();
  for (const t of U.TEAMS) { bullpen.set(t.id, { ip: 450, bf: 1900, woba: 0.31 }); wrc.set(t.id, { wrcPlus: 100 }); frv.set(t.id, { frv: 0 }); }
  const sources = { batterIndex: D.indexPlayers([bats.concat(savantB)]), pitcherIndex: D.indexPlayers([savantP]), teamOffense: bbTeams, bullpen, wrc, frv, seasonsUsed: [2026] };
  const small = { ...cfg, iterations: 60000 };
  sources.v2 = D.v2Sources({ batCur: savantB, batPrev: [], pitCur: savantP, pitPrev: [], teamsCur: bbTeams, teamsPrev: null, seasons: [2025, 2026] }, cfg.v2);
  const out = G.runGame(full, sources, small, M.seeded(1));
  check('end-to-end: model v3 runs on real data next to the sheet models; retired v2 is gone', out.ok && out.result.v3 && out.result.v3.homeWinProb > 0.25 && out.result.v3.homeWinProb < 0.75 &&
    out.result.v3.runsAway > 2 && out.result.v3.runsAway < 7 && out.result.hierarchical && !('v2' in out.result) && Number.isFinite(out.result.v3.sides.home.efficiency),
    out.ok ? JSON.stringify({ away: out.result.v3.awayWinProb, runs: [out.result.v3.runsAway, out.result.v3.runsHome] }) : out.missing.join(' | '));
  check('end-to-end: model v3 runs with totals, Pythagorean and Monte Carlo on its runs', out.result.v3 && out.result.v3.totals.length === cfg.v3.totalLines.length &&
    Math.abs(out.result.v3.awayWinProb + out.result.v3.homeWinProb - 1) < 1e-12 && Math.abs(out.result.v3.total - out.result.v3.runsAway - out.result.v3.runsHome) < 1e-12 &&
    out.result.v3.pythagorean.homeWinProb > 0.2 && out.result.v3.monteCarlo.homeWinProb > 0.2 && Math.abs(out.result.v3.monteCarlo.homeWinProb - out.result.v3.pythagorean.homeWinProb) < 0.08,
    JSON.stringify(out.result.v3 && { runs: [out.result.v3.runsAway, out.result.v3.runsHome], wp: out.result.v3.homeWinProb, py: out.result.v3.pythagorean.homeWinProb, mc: out.result.v3.monteCarlo.homeWinProb }));
  {
    const ef = out.result.eff, sd = ef && ef.sides;
    check('end-to-end: efficiency model runs on real data (primary), win chance and totals consistent', ef && ef.runsAway > 2 && ef.runsAway < 7 && ef.runsHome > 2 && ef.runsHome < 7 &&
      Math.abs(ef.awayWinProb + ef.homeWinProb - 1) < 1e-12 && Math.abs(ef.total - ef.runsAway - ef.runsHome) < 1e-12 && ef.totals.length === cfg.v3.totalLines.length &&
      Math.abs(sd.home.runs - sd.home.efficiency * sd.home.runsAtLeagueConversion) < 1e-9, JSON.stringify(ef && { runs: [ef.runsAway, ef.runsHome], wp: ef.homeWinProb, eff: [sd.away.efficiency, sd.home.efficiency], src: sd.home.source }));
    const PE = require('../src/projection_eff'), e = cfg.eff;
    const tm = (g, r, x) => new Map([[1, { games: g, runs: r, pa: g * 38, wobaNum: x * 0.32 * g * 34, wobaDen: g * 34 }], [2, { games: g, runs: g * 4.4, pa: g * 38, wobaNum: 0.32 * g * 34, wobaDen: g * 34 }]]);
    // team 1 has the league's wOBA; its runs set its efficiency directly against team 2 (also league wOBA)
    const full = PE.efficiencyFor(1, { cur: tm(162, 162 * 4.4 * 1.05 / 1.025, 1), prev: tm(162, 162 * 4.4, 1) }, e);
    const r29 = PE.efficiencyFor(1, { cur: tm(29, 29 * 6, 1), prev: tm(162, 162 * 4.4 * 1.1 / 1.05, 1) }, e);
    const r81 = PE.efficiencyFor(1, { cur: tm(81, 81 * 4.4 * 1.05 / 1.025, 1), prev: tm(162, 162 * 4.4, 1) }, e);
    const none = PE.efficiencyFor(1, { cur: null, prev: null }, e);
    check('efficiency schedule: full this-season number at 162 games, half the gap at 81', Math.abs(full.efficiency - full.thisSeason) < 1e-12 && full.thisSeason > 1.01 &&
      Math.abs((r81.efficiency - 1) - (r81.thisSeason - 1) * 0.5) < 1e-12, JSON.stringify([full, r81]));
    check('efficiency schedule: last season used before game 30, 1.00 with no data', r29.source === 'last season' && Math.abs(r29.efficiency - r29.lastSeason) < 1e-12 && r29.lastSeason > 1.02 &&
      none.efficiency === 1, JSON.stringify([r29, none]));
  }
  const wx = JSON.parse(JSON.stringify(sched)); wx.dates[0].games[0].weather = { condition: 'Clear', temp: '88', wind: '10 mph, Out To CF' };
  check('schedule weather -> temperature for v3', G.parseGames(wx, cfg)[0].weather.tempF === 88 && G.parseGames(sched, cfg)[0].weather === null);
  check('log row carries v3 win prob, runs and total', (() => { const row = L.newRow(out.result, null, { primaryModel: 'v3mkt', v3: cfg.v3 }); return Number.isFinite(row.v3_away) && Number.isFinite(row.v3_total) && L.CSV_COLUMNS.includes('v3_total'); })());
  const rookie = JSON.parse(JSON.stringify(full)); rookie.away.lineup[0] = { id: 1, name: 'Brand New Rookie' };
  const rk = G.runGame(rookie, sources, small, M.seeded(1));
  check('a hitter with no stats: v3 still runs (rookie prior), the sheet models are skipped with a note', rk.ok && rk.result.v3 && rk.result.hierarchical === null &&
    rk.warnings.some(w => /Brand New Rookie/.test(w)) && rk.warnings.some(w => /Original sheet models skipped/.test(w)), rk.ok ? rk.warnings.join(' | ') : rk.missing.join(' | '));
  if (rk.ok) {
    const rr = L.newRow(rk.result, { awayMl: 110, homeMl: -130, nBooks: 1, books: [] }, cfg);
    const lo = x => Math.log(x / (1 - x)), b = cfg.v3.marketBlend, nvA = L.noVigAway(110, -130);
    const ph = 1 / (1 + Math.exp(-(b.const + b.mkt * lo(1 - nvA) + b.model * lo(1 - rr.v3_away))));
    check('log row without the sheet models: efficiency model (picks) and v3 + market filled, sheet models and v2 empty',
      Number.isFinite(rr.v3_away) && Math.abs(rr.v3mkt_away - (1 - ph)) < 1e-12 && rr.mc_away === null && rr.pick_model === 'eff' && rr.pick_side === rr.eff_pick &&
      Number.isFinite(rr.eff_away) && Number.isFinite(rr.effmkt_away) && Number.isFinite(rr.eff_awayRuns) && Number.isFinite(rr.awayEfficiency) && rr.v2_away == null && rr.v2_awayRuns == null &&
      L.CSV_COLUMNS.includes('v2mkt_pick') && L.CSV_COLUMNS.includes('v3mkt_pick'), JSON.stringify({ v3: rr.v3_away, mkt: rr.v3mkt_away, pick: rr.pick_side }));
    const old = L.derive({ ...rr, pick_model: 'v2mkt', v2mkt_pick: 'home', v2mkt_edgePp: 3.1 });
    check('an old row whose pick came from v2 + market keeps that pick', old.pick_side === 'home' && old.pick_edgePp === 3.1);
  }
  check('end-to-end run succeeds on real data', out.ok, out.ok ? '' : out.missing.join(' | '));
  if (out.ok) {
    const x = out.result;
    check('end-to-end: probabilities valid', x.hierarchical.awayWinProb > 0.25 && x.hierarchical.awayWinProb < 0.75 && Math.abs(x.hierarchical.awayWinProb + x.hierarchical.homeWinProb - 1) < 1e-12,
      `${x.away} ${x.hierarchical.awayWinProb.toFixed(4)} / ${x.home} ${x.hierarchical.homeWinProb.toFixed(4)}; runs ${x.projectedRuns.awayScaled.toFixed(2)}/${x.projectedRuns.homeScaled.toFixed(2)}; park ${x.parkFactor}`);
    check('end-to-end: park factor is the HOME team\'s', x.parkFactor === cfg.parkFactors[full.home.teamId]);
    check('end-to-end: sensitivity + comparison present', x.sensitivity.length === 1 && Number.isFinite(x.comparison.plugInPoissonAway));
    const py = x.pythagorean;
    check('end-to-end: Pythagorean = H51/J51 on the 162-game scores C36/K36, both teams, sums to 1',
      Math.abs(py.awayWinProb + py.homeWinProb - 1) < 1e-12 && py.exponent === cfg.pythagoreanExponent &&
      Math.abs(py.awayWinProb - M.pythagorean(x.cells.C36, x.cells.K36, cfg.pythagoreanExponent).pA) < 1e-12,
      `${py.awayWinProb.toFixed(4)} / ${py.homeWinProb.toFixed(4)}`);
    const pz = x.pythagenpat, pzs = x.pythagenpatSeason, lam = x.projectedRuns;
    check('end-to-end: Pythagenpat (projected) uses the park-adjusted runs the Monte Carlo uses',
      Math.abs(pz.exponent - Math.pow(lam.awayScaled + lam.homeScaled, cfg.pythagenpatConstant)) < 1e-12 &&
      Math.abs(pz.awayWinProb - M.pythagenpat(lam.awayScaled, lam.homeScaled, cfg.pythagenpatConstant).pA) < 1e-12,
      `${pz.awayWinProb.toFixed(4)}, exponent ${pz.exponent.toFixed(3)}`);
    check('end-to-end: Pythagenpat (season) exponent comes from season R/G C26 + K26',
      Math.abs(pzs.exponent - Math.pow(x.cells.C26 + x.cells.K26, cfg.pythagenpatConstant)) < 1e-12 &&
      Math.abs(x.cells.C26 - x.cells.C28 * x.cells.C25) < 1e-9 && Math.abs(pzs.awayWinProb + pzs.homeWinProb - 1) < 1e-12,
      `${pzs.awayWinProb.toFixed(4)}, exponent ${pzs.exponent.toFixed(3)}`);
    check('end-to-end: Pythagorean, Monte Carlo and both Pythagenpats all carry fair moneylines that mirror each other',
      [py, x.hierarchical, pz, pzs].every(m => Number.isFinite(m.awayFairMl) && (Math.abs(m.awayFairMl + m.homeFairMl) < 1e-9 || (m.awayFairMl === 100 && m.homeFairMl === 100))),
      `${py.awayFairMl.toFixed(1)} / ${py.homeFairMl.toFixed(1)}`);
  }
  // Sheet models alone (no v2 sources): missing data must be reported, not guessed
  const legacyOnly = { ...sources, v2: undefined };
  const gone = JSON.parse(JSON.stringify(full)); gone.away.lineup[0] = { id: 1, name: 'Nobody Real' };
  const miss = G.runGame(gone, legacyOnly, small, M.seeded(1));
  check('sheet models only: unknown batter -> reported as missing, no result', !miss.ok && miss.missing.some(m => /Nobody Real/.test(m)), miss.missing && miss.missing.join(' | '));
  const seasonCfg = { ...small, rqpaBasis: 'season' };
  const teamQpa = new Map(U.TEAMS.map(t => [t.id, { qpa: 0.32, pa: 5000 }]));
  const sOut = G.runGame(full, { ...sources, teamQpa }, seasonCfg, M.seeded(1));
  check('end-to-end with rqpaBasis=season runs and reports the basis', sOut.ok && sOut.result.rqpaBasis === 'season', sOut.ok ? '' : sOut.missing.join(' | '));
  const sMiss = G.runGame(full, { ...legacyOnly, teamQpa: new Map() }, seasonCfg, M.seeded(1));
  check('season basis: missing team QPA reported', !sMiss.ok && sMiss.missing.some(m => /season-long QPA/.test(m)));
  const nobp = { ...legacyOnly, bullpen: new Map() };
  const miss2 = G.runGame(full, nobp, small, M.seeded(1));
  check('missing bullpen -> reported as missing', !miss2.ok && miss2.missing.some(m => /bullpen/.test(m)));
  // A postseason game runs the same way and is marked as one in the result and the log
  const post = G.runGame({ ...full, gameType: 'D', seriesDescription: 'NL Division Series' }, sources, small, M.seeded(1));
  check('postseason game runs; result and log row carry the game type',
    post.ok && post.result.gameType === 'D' && post.result.seriesDescription === 'NL Division Series' && L.newRow(post.result, null, {}).gameType === 'D' &&
    post.warnings.some(w => /postseason/.test(w)), post.ok ? '' : post.missing.join(' | '));
  // No-FanGraphs mode runs with no bullpen, wRC+ or FRV data at all
  const noFg = G.runGame(full, { ...sources, bullpen: new Map(), wrc: new Map(), frv: new Map() }, { ...small, useFangraphs: false }, M.seeded(1));
  check('no-FanGraphs mode: runs without any FanGraphs data', noFg.ok && noFg.result.inputs === 'no-fangraphs', noFg.ok ? '' : noFg.missing.join(' | '));
  if (noFg.ok) check('no-FanGraphs mode: no bullpen term (H17 = 0) and C36 = C29', noFg.result.cells.H17 === 0 && Math.abs(noFg.result.cells.C36 - noFg.result.cells.C29) < 1e-9);
}

// ---------- play log ----------
{
  near('implied: -150 = 60%', L.implied(-150), 0.6, 1e-12);
  near('implied: +150 = 40%', L.implied(150), 0.4, 1e-12);
  near('payout: -150 pays 0.667u, +120 pays 1.2u', L.payout(-150) + L.payout(120), 100 / 150 + 1.2, 1e-12);
  near('no-vig: -110/-110 = 50%', L.noVigAway(-110, -110), 0.5, 1e-12);
  check('cents: -150 -> -50, +110 -> +10, fair -150 vs -120 = 30c, -110 vs +110 = 20c',
    L.cents(-150) === -50 && L.cents(110) === 10 && L.cents(-120) - L.cents(-150) === 30 && L.cents(110) - L.cents(-110) === 20);
  check('parseMl accepts "-135", "+120", 120; rejects 50 and junk', L.parseMl('-135') === -135 && L.parseMl('+120') === 120 && L.parseMl(120) === 120 && L.parseMl(50) === null && L.parseMl('abc') === null);

  const pk = L.modelPick(0.55, 100, -120, 2);
  near('model pick: 55% vs +100 -> away, edge 5pp', pk.edgePp, 5, 1e-9);
  near('model pick: edge in cents (fair -122.2 vs +100 = 22.2c)', pk.edgeCents, 22.2222222, 1e-6);
  check('model pick: side is away', pk.side === 'away');
  const none = L.modelPick(0.51, -110, -110, 2);
  check('model pick: below the 2pp threshold -> no side, edge still reported', none.side === '' && Math.abs(none.edgePp - (51 - 100 * 110 / 210)) < 1e-9, none.edgePp);
  check('model pick: no market odds -> nothing', L.modelPick(0.6, null, null, 2).side === '' && L.modelPick(0.6, null, null, 2).edgePp === null);

  const res = { gamePk: 777, date: '2026-09-18', startUtc: '2026-09-18T23:10:00Z', away: 'Chicago Cubs', home: 'Cincinnati Reds', venue: 'Great American Ball Park',
    starters: { away: 'A Starter', home: 'H Starter' }, projectedRuns: { awayRaw: 4.6, homeRaw: 4.2, awayScaled: 4.922, homeScaled: 4.494 },
    hierarchical: { awayWinProb: 0.53, awayFairMl: -112.8, homeFairMl: 112.8 }, pythagorean: { awayWinProb: 0.55, awayFairMl: -122.2, homeFairMl: 122.2 },
    pythagenpat: { awayWinProb: 0.551, awayFairMl: -122.7, homeFairMl: 122.7 }, pythagenpatSeason: { awayWinProb: 0.552, awayFairMl: -123.2, homeFairMl: 123.2 } };
  const MKT = { awayMl: 100, awayBook: 'BetMGM', homeMl: -120, homeBook: 'BetMGM', nBooks: 1, books: [{ book: 'BetMGM', awayMl: 100, homeMl: -120 }], at: '2026-09-18T19:59:00Z' };
  const row = L.newRow(res, MKT, { pickThresholdPp: 2, primaryModel: 'pyth', now: new Date('2026-09-18T20:00:00Z') });
  check('log row: auto fields filled (teams, SPs, runs, all 4 models, market, primary pick)',
    row.away === 'Chicago Cubs' && row.awaySP === 'A Starter' && row.awayLambda === 4.922 && row.ppatSeason_away === 0.552 && row.mc_homeFair === 112.8 &&
    row.mkt_awayBook === 'BetMGM' && row.mkt_source === 'feed' && row.mkt_books === 'BetMGM +100/-120' && row.pick_model === 'pyth' && row.pick_side === 'away' &&
    Math.abs(row.pick_edgePp - 5) < 1e-9 && row.played === '' && row.result === '');
  const bare = L.newRow(res, null, {});
  check('log row: no market -> no picks, row still saved', bare.pick_side === '' && bare.mkt_awayMl === null && bare.mkt_source === '');
  // odds typed in by hand after shopping: every model's pick and edge follow them
  const manual = L.applyEntry(bare, { mkt_awayMl: '+100', mkt_homeMl: '-120', mkt_awayBook: 'Bet365' }).row;
  check('manual market odds -> picks and edges for every model', manual.mkt_source === 'you' && manual.pyth_pick === 'away' && Math.abs(manual.pyth_edgePp - 5) < 1e-9 &&
    manual.mc_pick === 'away' && Math.abs(manual.mc_edgePp - 3) < 1e-9 && manual.pick_side === 'away');
  check('manual market odds: one side only is refused', !!L.applyEntry(bare, { mkt_awayMl: '+100' }).errors);
  check('manual market odds: clearing them removes the picks', L.applyEntry(manual, { mkt_awayMl: '', mkt_homeMl: '' }).row.pick_side === '');
  check('editing your notes keeps the feed\'s odds details', L.applyEntry(row, { notes: 'x' }).row.mkt_source === 'feed');

  check('entry: "played" without side/odds is refused', !!L.applyEntry(row, { played: 'yes' }).errors);
  check('entry: bad odds refused', !!L.applyEntry(row, { played: 'yes', side: 'away', oddsTaken: '50' }).errors);
  check('entry: non-editable field refused', !!L.applyEntry(row, { away: 'Yankees' }).errors);
  const e1 = L.applyEntry(row, { played: 'yes', side: 'away', oddsTaken: '+120', stake: '2', book: 'BetMGM', notes: 'lean' });
  check('entry: valid play saved', e1.row && e1.row.oddsTaken === 120 && e1.row.stake === 2 && e1.row.result === '' && e1.row.plUnits === null);

  const won = L.applyFinal(e1.row, { status: 'Final', awayScore: 5, homeScore: 3 });
  check('final: away 5-3 -> W, P/L = 2 x 1.2 = +2.4u', won.result === 'W' && Math.abs(won.plUnits - 2.4) < 1e-12 && won.finalAway === 5);
  const lost = L.applyFinal(e1.row, { status: 'Final', awayScore: 1, homeScore: 3 });
  check('final: loss -> L, -2u', lost.result === 'L' && lost.plUnits === -2);
  const pp = L.applyFinal(e1.row, { status: 'Postponed' });
  check('final: postponed -> void, 0u', pp.result === 'void' && pp.plUnits === 0);
  check('final: game not over yet -> still pending', L.applyFinal(e1.row, { status: 'In Progress', awayScore: 2, homeScore: 1 }).result === '');

  const closed = L.applyClosing(won, { awayMl: 100, homeMl: -120, awayBook: 'BetMGM', at: '2026-09-18T22:55:00Z' });
  const nvA = 0.5 / (0.5 + 120 / 220);
  near('CLV: close no-vig (47.83%) minus implied of +120 (45.45%) = +2.37pp', closed.clvPp, 100 * (nvA - 100 / 220), 1e-9);
  near('CLV as EV%: 47.83% x 2.2 - 1 = +5.2%', closed.clvEvPct, 100 * (nvA * 2.2 - 1), 1e-9);
  check('model pick graded at the run-time price: away +100 won -> +1u', closed.pick_result === 'W' && closed.pick_plUnits === 1);
  const cons = L.applyClosing(won, { awayMl: 105, homeMl: -115, nvAway: 0.47, nBooks: 8 });
  near('CLV uses the feed\'s consensus no-vig across books when there is one', cons.clvPp, 100 * (0.47 - 100 / 220), 1e-9);
  check('closing line from the feed is labeled', cons.close_source === 'feed' && cons.close_book === 'best of 8 books');
  const typedClose = L.applyEntry(won, { close_awayMl: '-110', close_homeMl: '-110' }).row;
  near('closing line typed in by hand -> CLV from it (50% - 45.45%)', typedClose.clvPp, 100 * (0.5 - 100 / 220), 1e-9);
  check('the feed never overwrites a closing line you typed', L.applyClosing(typedClose, { awayMl: 150, homeMl: -170, nvAway: 0.4 }) === typedClose);

  const passed = L.applyFinal(L.applyEntry(L.newRow({ ...res, gamePk: 778 }, MKT, { now: new Date('2026-09-18T20:00:00Z') }), { played: 'no' }).row, { status: 'Final', awayScore: 0, homeScore: 2 });
  const rerun = L.newRow(res, null, { now: new Date('2026-09-18T19:00:00Z') });  // an earlier run of game 777
  const s = L.summarize([closed, lost.id ? { ...lost, id: 'x2', gamePk: 779 } : lost, passed, rerun]);
  check('summary: played record 1-1, units +0.4', s.played.record === '1-1' && Math.abs(s.played.units - 0.4) < 1e-12, JSON.stringify(s.played));
  check('summary: passed game graded by the model pick (away lost) -> 0-1', s.passedCount === 1 && s.passedModelPick.record === '0-1');
  check('summary: re-runs of a game count once in model stats', s.games === 3 && s.byModel.find(m => m.model === 'pyth').picks.n === 3);
  check('summary: average CLV over plays with a closing line', Math.abs(s.avgClvPp - closed.clvPp) < 1e-12);
  check('summary: edge buckets present', s.byEdge.length === 4 && s.byEdge.reduce((a, b) => a + b.played.n, 0) === 2);

  const csv = L.toCsv([{ ...closed, notes: '=HYPERLINK("x")' }]);
  const lines = csv.trim().split('\r\n');
  check('CSV: header + one row, every column', lines.length === 2 && lines[0].split(',').length === L.CSV_COLUMNS.length);
  check('CSV: a note starting with "=" can\'t run as a spreadsheet formula', lines[1].includes(`"'=HYPERLINK(""x"")"`), lines[1].slice(-60));
  check('CSV: negative odds stay numbers', lines[1].split(',').includes('-120'));

  // The Odds API v4 /odds response shape (sample built from the API's documentation, not a live response)
  const events = [
    { id: 'e1', sport_key: 'baseball_mlb', commence_time: '2026-09-18T17:10:00Z', home_team: 'Cincinnati Reds', away_team: 'Chicago Cubs',
      bookmakers: [{ key: 'betmgm', title: 'BetMGM', last_update: '2026-09-18T16:00:00Z', markets: [{ key: 'h2h', outcomes: [{ name: 'Chicago Cubs', price: -105 }, { name: 'Cincinnati Reds', price: -115 }] }] }] },
    { id: 'e2', sport_key: 'baseball_mlb', commence_time: '2026-09-18T23:10:00Z', home_team: 'Cincinnati Reds', away_team: 'Chicago Cubs',
      bookmakers: [{ key: 'draftkings', title: 'DraftKings', markets: [{ key: 'h2h', outcomes: [{ name: 'Cincinnati Reds', price: -118 }, { name: 'Chicago Cubs', price: 100 }] }] },
                   { key: 'betmgm', title: 'BetMGM', last_update: '2026-09-18T22:00:00Z', markets: [{ key: 'h2h', outcomes: [{ name: 'Chicago Cubs', price: 102 }, { name: 'Cincinnati Reds', price: -122 }] }] }] },
    { id: 'e3', sport_key: 'baseball_mlb', commence_time: '2026-09-18T20:05:00Z', home_team: 'Athletics', away_team: 'Seattle Mariners',
      bookmakers: [{ key: 'fanduel', title: 'FanDuel', markets: [{ key: 'h2h', outcomes: [{ name: 'Athletics', price: 130 }, { name: 'Seattle Mariners', price: -154 }] }] }] }];
  const b1 = O.gameBooks(events, { awayId: 112, homeId: 113, startUtc: '2026-09-18T23:10:00Z' });
  check('odds: doubleheader -> the event nearest this game\'s start; every book, prices by team', b1.length === 2 && b1[0].book === 'BetMGM' && b1[0].awayMl === 102 && b1[0].homeMl === -122 && b1[1].book === 'DraftKings' && b1[1].homeMl === -118, JSON.stringify(b1));
  const s1 = O.summarizeBooks(b1);
  check('odds: best price per side across books (away +102 BetMGM, home -118 DraftKings)', s1.awayMl === 102 && s1.awayBook === 'BetMGM' && s1.homeMl === -118 && s1.homeBook === 'DraftKings' && s1.nBooks === 2);
  near('odds: consensus no-vig = average of each book\'s no-vig', s1.nvAway, (L.noVigAway(102, -122) + L.noVigAway(100, -118)) / 2, 1e-12);
  const b2 = O.gameBooks(events, { awayId: 136, homeId: 133, startUtc: '2026-09-18T20:05:00Z' });
  check('odds: "Athletics" maps to the A\'s', b2.length === 1 && b2[0].book === 'FanDuel' && b2[0].awayMl === -154 && b2[0].homeMl === 130, JSON.stringify(b2));
  check('odds: game not listed -> no books, no summary', O.gameBooks(events, { awayId: 147, homeId: 111, startUtc: '2026-09-18T23:10:00Z' }).length === 0 && O.summarizeBooks([]) === null);
  check('odds: request URL built server-side with the key and regions encoded', /apiKey=abc%26x/.test(O.ODDS_URL('abc&x', 'us,us2')) && /regions=us%2Cus2/.test(O.ODDS_URL('k', 'us,us2')));
}

// ---------- report ----------
const failed = results.filter(r => !r.ok);
if (IS_NODE) {
  for (const r of results) console.log((r.ok ? 'PASS ' : 'FAIL ') + r.name + (r.ok ? '' : '  -> ' + r.detail));
  console.log(`\n${results.length - failed.length}/${results.length} passed`);
  process.exit(failed.length ? 1 : 0);
} else {
  globalThis.TEST_RESULTS = results;
}
