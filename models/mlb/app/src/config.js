'use strict';
// Everything you might want to tweak lives here.

module.exports = {
  // Shared base settings. Model v2 itself was retired 2026-10-19 (J) and no longer runs; v3 and the models built on it
  // inherit these settings (shrinkage, starter share, run-efficiency settings, tie rule) and override the coefficients.
  // Originally fitted on Retrosheet 2015-2025 (models/mlb-app/backtest_chain.py). See src/projection_v2.js.
  v2: {
    "coefficients": {
      "const": -0.02967,
      "lineup": 0.05827,
      "pitch": 0.0346,
      "sp_kbb": -0.02658,
      "home": 0.0223,
      "ln_pf": 1.41616,
      "eff": 0.11001,
      "anchor": 0.21258
    },
    "nbAlpha": 0.2576,
    "starterShare": 0.581,
    "shrinkPa": {
      "hitter": 220.0,
      "starter": 250.0,
      "bullpen": 600.0
    },
    "prevSeasonWeight": 0.6,
    "tieHomeWin": 0.52,
    "calibration": {
      "intercept": 0.0666,
      "slope": 1.3276
    },
    "marketBlend": {
      "const": 0.0159,
      "mkt": 0.8501,
      "model": 0.1511
    },
    // anchor (2026-10-16, J): team R/G regressed 20 games toward league, last season at 0.6 weight (data.js teamAnchor);
    // refit together with eff: models/mlb-app/backtest_pitching.py -> pitching_app_coefficients.json
    // J's run-efficiency rule (2026-10-16): wOBA/xwOBA are read together with the runs a team actually scores.
    // efficiency = (R + 0.6 x last season R) / (same for the runs the team's wOBA should produce), from the
    // Baseball-Reference team batting table; projected runs x efficiency^eff (eff above). slope = league runs per PA
    // per point of wOBA (Retrosheet 2015-2025). clip keeps a broken table from doing damage.
    // models/mlb-app/backtest_efficiency.py -> efficiency_coefficients.json
    "efficiency": { "slope": 0.7063, "K": 0, "prevSeasonWeight": 0.6, "clip": [0.7, 1.4] },
    "unknownHitter": 0.92,
    "unknownStarter": 1.05,
    "starterPriorScale": 1.02,
    "starterKbbPriorScale": 0.95,
    "unknownStarterKbb": 0.8,
    "leaguePad": {
      "wobaPa": 20000,
      "teamGames": 600
    },
    "defaults": {
      "woba": 0.315,
      "kbb": 0.135,
      "runs": 4.4,
      "k": 0.225,
      "bb": 0.084,
      "iso": 0.160
    }
  },

  // Hierarchical Monte Carlo (backtested projection error, runs). Stage 1 samples the game's
  // true scoring rate as projection + Normal(errorMean, errorStd); stage 2 draws runs ~ Poisson.
  errorMean: -0.58,
  errorStd: 3.03,
  iterations: 200000,
  floorLambda: 0.05,
  // Extra stage-1 std values to print as a sensitivity check (2.2 = std after removing the
  // Poisson noise that is already inside the 3.03 backtest error). Empty array = off.
  sensitivityStds: [2.2],

  // Which seasons of stats to use. Rule: use current-season-to-date alone once it has at least
  // `minCurrentSeasonGames` games; before that, blend current season-to-date with the full prior season.
  sampleWindow: {
    minCurrentSeasonGames: 100,
    gamesBasis: 'min-of-two-teams', // games played by the two teams in this matchup: use the smaller
  },

  // Projection constants (from the Tester sheet)
  relieversPerBullpen: 8,   // "/8" in Tester rows 17
  outsPerGame: 27,
  seasonGames: 162,
  frvWeight: 0.6,
  pythagoreanExponent: 1.83,
  // Pythagenpat: exponent = (runs per game)^constant (David Smyth's 0.287). Two versions are shown:
  //   projected - runs per game = both teams' park-adjusted projected runs (the Monte Carlo's inputs)
  //   season    - runs per game = both teams' season R/G (sheet C26 + K26, like Tester K44)
  pythagenpatConstant: 0.287,
  lineupSize: 9,

  // What lineup QPA goes in R/QPA (Tester C25/K25 = QPA x PA/G, then C28 = R/G / that):
  //   'tonight' = tonight's nine hitters (the sheet as built)
  //   'season'  = the team's full-season QPA: every batter on the team in the FanGraphs batters
  //               export, PA-weighted (needs the Team column; players with no single team are skipped)
  // Only the R/QPA divisor changes; tonight's lineup still drives F2/M2.
  rqpaBasis: 'tonight',

  // Batter stats source: 'savant' (automatic, no export needed) or 'fangraphs-csv' (your original
  // method; slightly different wOBA/xwOBA methodology, needs a daily export)
  batterSource: 'savant',

  // Games that can be run: R regular season, F Wild Card, D Division Series, L LCS, W World Series.
  // Postseason games use regular-season stats (the full season is then always >= 100 games).
  gameTypes: ['R', 'F', 'D', 'L', 'W'],

  // false = no-FanGraphs mode: no bullpen term, wRC+ 100, FRV 0 (the setup the full-season backtest
  // validated). The Run screen can switch this per run.
  useFangraphs: true,

  // Bullpen IP/TBF/wOBA against, team wRC+ and team FRV all come from FanGraphs (owner's choice -
  // FanGraphs splits each pitcher's innings by role, so a swingman's starts never leak into the
  // bullpen number the way an automatic role-classification would). See README for export format.

  // Play log. A model "picks" a side when its win probability beats the market's vig-included implied
  // probability by at least pickThresholdPp percentage points (same rule as the backtest).
  pickThresholdPp: 2,
  primaryModel: 'eff',             // whose pick/edge fills the log's pick_* columns: eff (J's efficiency model, 2026-10-19), effmkt, v3mkt, v3, mc, pyth, ppat or ppatSeason
  // Optional market odds from The Odds API (a licensed feed; set ODDS_API_KEY on the server, free plan =
  // 500 credits/month). Every book in these regions is shown so you can shop; each region costs 1 credit per
  // request ('us' = the main US books; 'us2' adds more US books). Without a key you type odds in yourself.
  oddsRegions: 'us,us2',
  closingCaptureMinutes: 10,       // the closing line = one capture of every book within this many minutes of first pitch

  // Warn when a hand-exported CSV is older than this many days
  csvMaxAgeDays: 2,

  // Local web server (127.0.0.1 = only reachable from this computer)
  host: '127.0.0.1',
  port: 3000,

  // 3-Year Statcast park factors, copied from the ParkFactors sheet, keyed by the HOME team's MLB id
  parkFactors: {
    115: 1.12, 133: 1.08, 111: 1.09, 113: 1.07, 112: 1.04, 141: 1.02, 109: 1.01, 142: 1.01,
    110: 1.00, 119: 1.00, 147: 0.99, 143: 0.99, 145: 0.99, 144: 0.99, 118: 0.99, 146: 0.98,
    158: 0.98, 114: 0.97, 120: 0.97, 140: 0.97, 116: 0.97, 137: 0.96, 117: 0.96, 134: 0.96,
    138: 0.95, 108: 0.95, 135: 0.94, 136: 0.92, 139: 0.92, 121: 0.98,
  },
};

// Model v3 (2026-10-10, MLB-1): model v2 plus tonight's lineup K%, BB% and ISO (each hitter regressed toward league
// average: 60 PA for K%, 120 for BB%, 160 AB for ISO, last season at 0.6 weight) and game-time temperature. Same
// wOBA/xwOBA lineup, starter, bullpen, park and home terms as v2, refit together. Retrosheet 2015-2025:
// models/mlb-app/backtest_v3.py -> v3_coefficients.json. Moneyline accuracy equals v2; total runs improve.
// 2026-10-16: v3 also carries J's run-efficiency term (same efficiency as v2, its own fitted weight).
module.exports.v3 = Object.assign({}, module.exports.v2, {
  "coefficients": {
    "const": -0.02694,
    "lineup": 0.03066,
    "pitch": 0.03434,
    "sp_kbb": -0.02659,
    "home": 0.0222,
    "ln_pf": 1.2628,
    "L_k": -0.00912,
    "L_bb": 0.01765,
    "L_iso": 0.01597,
    "temp": 0.03523,
    "eff": 0.09674,
    "anchor": 0.14786
  },
  "nbAlpha": 0.2556,
  "calibration": {
    "intercept": 0.0671,
    "slope": 1.32
  },
  // v3 + market (2026-10-19, sets picks now that v2 is retired): fit vs 2021-22 openers, models/mlb-app/pitching_app_coefficients.json
  "marketBlend": { "const": 0.0154, "mkt": 0.8546, "model": 0.1506 },
  "skillShrink": { "k": 60, "bb": 120, "iso": 160 },
  "tempCenterF": 73.5,
  // over/under probabilities are shown for these totals (plus the market total when you enter one)
  "totalLines": [6.5, 7, 7.5, 8, 8.5, 9, 9.5, 10, 10.5, 11, 11.5],
  "totalsCalSlope": 0.498,
  "mcIterations": 100000
});

// Model v3 + defense (2026-10-16, J): v3 + the catcher's framing (extra called strikes per taken pitch, percentage points)
// + tonight's fielders' outs above average per chance, both from Savant leaderboards. Backtest 2021-25: moneyline log loss
// 0.6756 vs 0.6761 for v3 alone. Shown and logged beside v3; it does not set picks until its live inputs are checked.
module.exports.v3d = Object.assign({}, module.exports.v3, {
  "coefficients": {
    "const": -0.02661,
    "lineup": 0.0281,
    "pitch": 0.03009,
    "sp_kbb": -0.02639,
    "home": 0.02235,
    "ln_pf": 1.2019,
    "L_k": -0.00954,
    "L_bb": 0.01746,
    "L_iso": 0.01713,
    "temp": 0.03504,
    "eff": 0.10008,
    "anchor": 0.15555,
    "framing": -0.03442,
    "fld_def": -0.0222
  },
  "nbAlpha": 0.2551,
  "calibration": {
    "intercept": 0.0676,
    "slope": 1.3027
  }
});

// Model v3 full (2026-10-16, J): v3 + defense + each lineup's handedness vs the starter (league platoon gap by how each
// hitter bats) + the starter-catcher pairing (Savant pitch file: wOBA allowed with tonight's catcher vs his own rate,
// regressed with 400 PA). Backtest 2021-25: moneyline 0.6760 (v3 + defense 0.6756), best closing-line value of any
// version (+1.11 pts on 2+ pt picks), totals 54.1% vs the opener. Logged only; not used for picks.
module.exports.v3f = Object.assign({}, module.exports.v3, {
  "coefficients": {
    "const": -0.02968,
    "lineup": 0.0285,
    "pitch": 0.02747,
    "sp_kbb": -0.02646,
    "home": 0.02256,
    "ln_pf": 1.2016,
    "L_k": -0.00978,
    "L_bb": 0.01776,
    "L_iso": 0.01707,
    "temp": 0.03512,
    "eff": 0.10368,
    "anchor": 0.15664,
    "framing": -0.03285,
    "fld_def": -0.02189,
    "platoon": 0.05329,
    "tandem": 0.0138
  },
  "nbAlpha": 0.255,
  "calibration": {
    "intercept": 0.0672,
    "slope": 1.3111
  }
});

// J's efficiency model (2026-10-19, primary): J's season runs per game read against the lineup's wOBA/xwOBA at full
// strength, tonight's matchup moving it on the runs scale; last season's efficiency for games 1-29, then this season's
// counted at games/162. Backtest 2021-25 (wOBA only): calibrated log loss 0.6765 vs the sheet's 0.6809, CLV +0.0069 on
// 2+ pt picks. See src/projection_eff.js; settings from models/mlb-app/fit_efficiency_app.py (efficiency_app_settings.json).
module.exports.eff = {
  "slope": 0.717,                 // league runs per PA gained per point of wOBA (team-seasons 2015-19)
  "priorGames": 30,               // before this many games, use last season's efficiency (J's rule)
  "fullGames": 162,               // this season's efficiency counts in full at this many games
  "prevClip": [0.7, 1.4], "curClip": [0.5, 2.0],
  "defaultPaPerGame": 38,
  "pythExponent": 1.83,
  "calibration": { "intercept": 0.1419, "slope": 1.0974 },
  "nbAlpha": 0.2429,
  "totalsCalSlope": 0.1776,
  // efficiency model + market, fit vs 2021-22 openers: the market dominates (model weight 0.03), so picks come from the
  // efficiency model itself (primaryModel 'eff') and this blend is logged beside it
  "marketBlend": { "const": 0.0375, "mkt": 0.9805, "model": 0.0275 }
};
