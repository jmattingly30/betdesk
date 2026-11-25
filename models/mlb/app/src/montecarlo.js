'use strict';
// Win-probability engines: hierarchical Monte Carlo, exact plug-in Poisson, Pythagorean.

// Seedable generator for tests (mulberry32). Production uses Math.random.
function seeded(seed) {
  let a = seed >>> 0;
  return function () {
    a = (a + 0x6D2B79F5) >>> 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

// Knuth's algorithm (same as the Apps Script version)
function poissonRandom(lambda, rng) {
  const L = Math.exp(-lambda);
  let k = 0, p = 1;
  do { k++; p *= rng(); } while (p > L);
  return k - 1;
}

// Box-Muller (same as the Apps Script version)
function normalRandom(mean, std, rng) {
  let u1 = rng();
  const u2 = rng();
  if (u1 === 0) u1 = 0.0001;
  return mean + std * Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2);
}

// Two-stage simulation. Win prob is tie-adjusted: winA / (winA + winB).
function hierarchical({ lambdaA, lambdaB, errorMean, errorStd, iterations, floorLambda = 0.05, rng = Math.random }) {
  let winA = 0, winB = 0, tie = 0, floored = 0, sumLamA = 0, sumLamB = 0;
  for (let i = 0; i < iterations; i++) {
    let la = lambdaA + normalRandom(errorMean, errorStd, rng);
    let lb = lambdaB + normalRandom(errorMean, errorStd, rng);
    if (la < floorLambda) { la = floorLambda; floored++; }
    if (lb < floorLambda) { lb = floorLambda; floored++; }
    sumLamA += la; sumLamB += lb;
    const ra = poissonRandom(la, rng), rb = poissonRandom(lb, rng);
    if (ra > rb) winA++; else if (rb > ra) winB++; else tie++;
  }
  const resolved = winA + winB;
  const pA = winA / resolved;
  return {
    pA, pB: 1 - pA,
    tiePct: tie / iterations,
    standardError: Math.sqrt(pA * (1 - pA) / resolved),
    flooredPct: floored / (2 * iterations),
    avgSampledLambdaA: sumLamA / iterations,
    avgSampledLambdaB: sumLamB / iterations,
    iterations,
  };
}

// Exact discrete Poisson with no projection uncertainty (what Poisson!E6 shows today), tie-adjusted.
function exactPoisson(lambdaA, lambdaB, maxRuns = 40) {
  const pmf = (lam) => {
    const out = [Math.exp(-lam)];
    for (let k = 1; k <= maxRuns; k++) out.push(out[k - 1] * lam / k);
    return out;
  };
  const a = pmf(lambdaA), b = pmf(lambdaB);
  let winA = 0, winB = 0, tie = 0;
  for (let i = 0; i <= maxRuns; i++) for (let j = 0; j <= maxRuns; j++) {
    const p = a[i] * b[j];
    if (i > j) winA += p; else if (j > i) winB += p; else tie += p;
  }
  return { pA: winA / (winA + winB), pB: winB / (winA + winB), tiePct: tie };
}

function pythagorean(scoreA, scoreB, exponent) {
  const a = Math.pow(scoreA, exponent), b = Math.pow(scoreB, exponent);
  return { pA: a / (a + b), pB: b / (a + b) };
}

// Pythagenpat for one game: exponent = (runs per game)^constant. Runs per game defaults to A + B
// (in one game, A's runs allowed are B's runs); pass envRunsPerGame to set the environment another way.
function pythagenpat(runsA, runsB, constant, envRunsPerGame = runsA + runsB) {
  const exponent = Math.pow(envRunsPerGame, constant);
  return { ...pythagorean(runsA, runsB, exponent), exponent };
}

module.exports = { seeded, poissonRandom, normalRandom, hierarchical, exactPoisson, pythagorean, pythagenpat };
