'use strict';
// Play log: one row per game run, whether or not it was played. Pure functions only (no file or network
// access), so the same code runs in the local server and in the Apps Script version. Storage lives elsewhere.
//
// Row fields
//   auto at run time : id, runAt, gameDate, gamePk, gameType (R, or F/D/L/W for the postseason), startUtc, away, home, venue, awaySP, homeSP, inputs,
//                      awayProjRuns/homeProjRuns (sheet C37/K37), awayLambda/homeLambda (after park factor),
//                      <model>_away (win prob), <model>_awayFair/<model>_homeFair (no-vig fair moneylines)
//   market odds      : mkt_awayMl/mkt_homeMl (+ books) - the best price on each side from the odds feed, or what
//                      you type in after shopping; mkt_books lists every book the feed returned
//   model picks      : <model>_pick ('away'/'home'/''), <model>_edgePp, <model>_edgeCents, recomputed whenever
//                      the market odds change; pick_* = the primary model's
//   entered by you   : played ('yes'/'no'/''), side ('away'/'home'), oddsTaken (American), stake (units, optional -
//                      P/L assumes 1 when blank), book, notes
//   filled in later  : close_* (closing line: feed, or typed in), finalAway/finalHome/gameStatus, result
//                      ('W'/'L'/'void'/''), plUnits, clvPp, clvEvPct, pick_result/pick_plUnits/pick_clvPp

const MODELS = [
  // Model v2 retired 2026-10-19 (J); its old columns stay in the log as history (LEGACY_MODEL_KEYS) but are no longer filled.
  // J's efficiency model (2026-10-19, primary) and its blend with the market (logged; the market dominates that blend)
  { key: 'eff', label: 'Efficiency model', from: r => r.eff },
  { key: 'effmkt', label: 'Efficiency model + market', from: () => null },
  { key: 'v3', label: 'Model v3', from: r => r.v3 },
  // v3 blended with the no-vig market odds on the entry (sets the pick); filled in by derive() once market odds are on the row
  { key: 'v3mkt', label: 'Model v3 + market', from: () => null },
  // v3 + R/G anchor + catcher framing + fielders (2026-10-16); logged so its live Savant inputs can be checked before it leads
  { key: 'v3d', label: 'Model v3 + defense', from: r => r.v3d },
  // v3 + defense + handedness + starter-catcher pairing (2026-10-16, J asked for it; logged only)
  { key: 'v3f', label: 'Model v3 full', from: r => r.v3f },
  { key: 'mc', label: 'Monte Carlo', from: r => r.hierarchical },
  { key: 'pyth', label: 'Pythagorean', from: r => r.pythagorean },
  { key: 'ppat', label: 'Pythagenpat (projected)', from: r => r.pythagenpat },
  { key: 'ppatSeason', label: 'Pythagenpat (season R/G)', from: r => r.pythagenpatSeason },
];
const LEGACY_MODEL_KEYS = ['v2', 'v2mkt'];
const BOOKS = ['BetMGM', 'Bet365', 'DraftKings', 'FanDuel', 'Caesars', 'Other'];

// ---------- odds math ----------
const isMl = v => typeof v === 'number' && Number.isFinite(v) && Math.abs(v) >= 100;
function implied(ml) { return ml < 0 ? -ml / (-ml + 100) : 100 / (ml + 100); }            // vig included
function payout(ml) { return ml > 0 ? ml / 100 : 100 / -ml; }                              // profit per 1 unit
function noVigAway(mlAway, mlHome) { const a = implied(mlAway), h = implied(mlHome); return a / (a + h); }
function fairMl(p) { if (p > 0.5) return -100 * p / (1 - p); if (p < 0.5) return 100 * (1 - p) / p; return 100; }
// "Cents": American odds on one straight line where -100 and +100 meet (-150 -> -50, +110 -> +10), so the
// gap between a fair line and an offered line reads the way bettors say it (fair -150 vs -120 = 30 cents).
function cents(ml) { return ml > 0 ? ml - 100 : ml + 100; }
function parseMl(v) {
  if (v === null || v === undefined || v === '') return null;
  const n = Number(String(v).replace(/[^\d.+-]/g, ''));
  return isMl(n) ? Math.round(n) : null;
}

// Best side for one model against market odds. Edge = model prob - vig-included implied prob of that price.
function modelPick(pAway, mlAway, mlHome, thresholdPp) {
  if (!Number.isFinite(pAway) || !isMl(mlAway) || !isMl(mlHome)) return { side: '', edgePp: null, edgeCents: null };
  const eA = pAway - implied(mlAway), eH = (1 - pAway) - implied(mlHome);
  const side = eA >= eH ? 'away' : 'home';
  const edge = Math.max(eA, eH);
  const p = side === 'away' ? pAway : 1 - pAway, ml = side === 'away' ? mlAway : mlHome;
  return { side: edge * 100 >= thresholdPp ? side : '', edgePp: edge * 100, edgeCents: cents(ml) - cents(fairMl(p)) };
}

const booksText = books => (books || []).map(b => `${b.book} ${b.awayMl > 0 ? '+' : ''}${b.awayMl}/${b.homeMl > 0 ? '+' : ''}${b.homeMl}`).join('; ');

// ---------- row creation (auto-filled part) ----------
// market: summarizeBooks() output from the odds feed, or null.
function newRow(result, market, opts) {
  const o = opts || {}, now = o.now || new Date();
  const p = result.projectedRuns;
  const row = {
    id: `${result.gamePk}-${now.getTime().toString(36)}`, runAt: now.toISOString(), gameDate: result.date, gamePk: result.gamePk,
    gameType: result.gameType || 'R',
    startUtc: result.startUtc || '', away: result.away, home: result.home, venue: result.venue || '',
    awaySP: result.starters.away, homeSP: result.starters.home, inputs: result.inputs || 'fangraphs',
    awayProjRuns: p.awayRaw, homeProjRuns: p.homeRaw, awayLambda: p.awayScaled, homeLambda: p.homeScaled,
    pick_model: o.primaryModel || 'pyth', pick_threshold: o.pickThresholdPp == null ? 2 : o.pickThresholdPp,
    eff_awayRuns: result.eff ? result.eff.runsAway : null, eff_homeRuns: result.eff ? result.eff.runsHome : null, eff_total: result.eff ? result.eff.total : null,
    awayEfficiency: result.eff ? result.eff.sides.away.efficiency : null, homeEfficiency: result.eff ? result.eff.sides.home.efficiency : null,
    blendE_const: o.eff && o.eff.marketBlend ? o.eff.marketBlend.const : null, blendE_mkt: o.eff && o.eff.marketBlend ? o.eff.marketBlend.mkt : null,
    blendE_model: o.eff && o.eff.marketBlend ? o.eff.marketBlend.model : null,
    v3_awayRuns: result.v3 ? result.v3.runsAway : null, v3_homeRuns: result.v3 ? result.v3.runsHome : null,
    v3_total: result.v3 ? result.v3.total : null, v3_tempF: result.v3 ? result.v3.tempF : null,
    v3d_awayRuns: result.v3d ? result.v3d.runsAway : null, v3d_homeRuns: result.v3d ? result.v3d.runsHome : null,
    awayFraming: result.v3d && result.v3d.defense.away ? result.v3d.defense.away.framing : null, homeFraming: result.v3d && result.v3d.defense.home ? result.v3d.defense.home.framing : null,
    awayFielding: result.v3d && result.v3d.defense.away ? result.v3d.defense.away.fielding : null, homeFielding: result.v3d && result.v3d.defense.home ? result.v3d.defense.home.fielding : null,
    v3f_awayRuns: result.v3f ? result.v3f.runsAway : null, v3f_homeRuns: result.v3f ? result.v3f.runsHome : null,
    awayPlatoon: result.v3f ? result.v3f.extra.away.platoon : null, homePlatoon: result.v3f ? result.v3f.extra.home.platoon : null,
    awayPairing: result.v3f ? result.v3f.extra.away.tandem : null, homePairing: result.v3f ? result.v3f.extra.home.tandem : null,
    blend3_const: o.v3 && o.v3.marketBlend ? o.v3.marketBlend.const : null, blend3_mkt: o.v3 && o.v3.marketBlend ? o.v3.marketBlend.mkt : null,
    blend3_model: o.v3 && o.v3.marketBlend ? o.v3.marketBlend.model : null,
  };
  for (const m of MODELS) {
    const r = m.from(result);
    row[`${m.key}_away`] = r ? r.awayWinProb : null;
    row[`${m.key}_awayFair`] = r ? r.awayFairMl : null;
    row[`${m.key}_homeFair`] = r ? r.homeFairMl : null;
  }
  Object.assign(row, { mkt_source: '', mkt_awayMl: null, mkt_awayBook: '', mkt_homeMl: null, mkt_homeBook: '', mkt_nBooks: null, mkt_books: '', mkt_at: '' });
  if (market && isMl(market.awayMl) && isMl(market.homeMl)) {
    Object.assign(row, { mkt_source: 'feed', mkt_awayMl: market.awayMl, mkt_awayBook: market.awayBook || '', mkt_homeMl: market.homeMl, mkt_homeBook: market.homeBook || '',
      mkt_nBooks: market.nBooks || null, mkt_books: booksText(market.books), mkt_at: market.at || row.runAt });
  }
  Object.assign(row, { played: '', side: '', oddsTaken: null, stake: null, book: '', notes: '',
    close_source: '', close_awayMl: null, close_homeMl: null, close_book: '', close_nvAway: null, close_at: '',
    finalAway: null, finalHome: null, gameStatus: '', result: '', plUnits: null, clvPp: null, clvEvPct: null,
    pick_result: '', pick_plUnits: null, pick_clvPp: null });
  return derive(row);
}

// ---------- your entries ----------
const EDITABLE = ['played', 'side', 'oddsTaken', 'stake', 'book', 'notes', 'mkt_awayMl', 'mkt_homeMl', 'mkt_awayBook', 'mkt_homeBook', 'close_awayMl', 'close_homeMl', 'close_book'];
// Returns { row } or { errors: [...] }. Only the editable fields can change; derived fields are recomputed.
function applyEntry(row, patch) {
  const out = { ...row }, errors = [];
  patch = patch || {};
  for (const k of Object.keys(patch)) if (!EDITABLE.includes(k)) errors.push(`"${k}" can't be edited`);
  if ('played' in patch) { const v = String(patch.played || '').toLowerCase(); if (!['yes', 'no', ''].includes(v)) errors.push('played must be yes or no'); else out.played = v; }
  if ('side' in patch) { const v = String(patch.side || '').toLowerCase(); if (!['away', 'home', ''].includes(v)) errors.push('side must be away or home'); else out.side = v; }
  const mlField = (k, label) => {
    if (!(k in patch)) return;
    const blank = patch[k] === '' || patch[k] == null, v = blank ? null : parseMl(patch[k]);
    if (!blank && v === null) errors.push(`${label} must be American odds like -135 or +120`); else out[k] = v;
  };
  mlField('oddsTaken', 'the odds you took');
  mlField('mkt_awayMl', 'the away market price'); mlField('mkt_homeMl', 'the home market price');
  mlField('close_awayMl', 'the away closing price'); mlField('close_homeMl', 'the home closing price');
  if ('stake' in patch) { const v = patch.stake === '' || patch.stake == null ? null : Number(patch.stake); if (v !== null && !(v > 0)) errors.push('stake must be a positive number of units'); else out.stake = v; }
  for (const k of ['book', 'notes', 'close_book', 'mkt_awayBook', 'mkt_homeBook']) if (k in patch) out[k] = String(patch[k] == null ? '' : patch[k]).slice(0, k === 'notes' ? 2000 : 60);
  // odds you type in replace the feed's, so the feed-only details no longer apply
  if (out.mkt_awayMl !== row.mkt_awayMl || out.mkt_homeMl !== row.mkt_homeMl) Object.assign(out, { mkt_source: out.mkt_awayMl == null ? '' : 'you', mkt_nBooks: null, mkt_books: '', mkt_at: '' });
  if (out.close_awayMl !== row.close_awayMl || out.close_homeMl !== row.close_homeMl) Object.assign(out, { close_source: out.close_awayMl == null ? '' : 'you', close_nvAway: null, close_at: '' });
  if ((out.mkt_awayMl == null) !== (out.mkt_homeMl == null)) errors.push('enter the market price for both sides (or neither)');
  if ((out.close_awayMl == null) !== (out.close_homeMl == null)) errors.push('enter the closing price for both sides (or neither)');
  if (out.played === 'yes' && (!out.side || out.oddsTaken == null)) errors.push('a played game needs the side and the odds you took');
  if (errors.length) return { errors };
  return { row: derive(out) };
}

// ---------- later: final score, closing odds ----------
function applyFinal(row, game) {
  // game: { status, awayScore, homeScore } from the MLB Stats API
  const out = { ...row, gameStatus: game.status || row.gameStatus };
  if (/^(Final|Game Over|Completed Early)/i.test(game.status || '') && Number.isFinite(game.awayScore) && Number.isFinite(game.homeScore)) {
    out.finalAway = game.awayScore; out.finalHome = game.homeScore;
  } else if (/postponed|cancel/i.test(game.status || '')) { out.finalAway = null; out.finalHome = null; }
  return derive(out);
}

// close: summarizeBooks() output from the odds feed just before first pitch. Doesn't overwrite a closing line you typed.
function applyClosing(row, close) {
  if (!close || !isMl(close.awayMl) || !isMl(close.homeMl) || row.close_source === 'you') return row;
  return derive({ ...row, close_source: 'feed', close_awayMl: close.awayMl, close_homeMl: close.homeMl,
    close_book: close.nBooks ? `best of ${close.nBooks} books` : (close.awayBook || ''), close_nvAway: Number.isFinite(close.nvAway) ? close.nvAway : null, close_at: close.at || '' });
}

// Recompute everything that follows from the other fields.
function derive(row) {
  const r = { ...row };
  // efficiency model + market, same form
  r.effmkt_away = null; r.effmkt_awayFair = null; r.effmkt_homeFair = null;
  if (Number.isFinite(r.eff_away) && isMl(r.mkt_awayMl) && isMl(r.mkt_homeMl) && Number.isFinite(r.blendE_model)) {
    const lo = p => Math.log(p / (1 - p));
    const ph = 1 / (1 + Math.exp(-(r.blendE_const + r.blendE_mkt * lo(1 - noVigAway(r.mkt_awayMl, r.mkt_homeMl)) + r.blendE_model * lo(1 - r.eff_away))));
    r.effmkt_away = 1 - ph; r.effmkt_awayFair = fairMl(1 - ph); r.effmkt_homeFair = fairMl(ph);
  }
  // v3 + market: logistic(const + mkt*logit(no-vig market) + model*logit(v3)), home-team probabilities
  r.v3mkt_away = null; r.v3mkt_awayFair = null; r.v3mkt_homeFair = null;
  if (Number.isFinite(r.v3_away) && isMl(r.mkt_awayMl) && isMl(r.mkt_homeMl) && Number.isFinite(r.blend3_model)) {
    const lo = p => Math.log(p / (1 - p));
    const ph = 1 / (1 + Math.exp(-(r.blend3_const + r.blend3_mkt * lo(1 - noVigAway(r.mkt_awayMl, r.mkt_homeMl)) + r.blend3_model * lo(1 - r.v3_away))));
    r.v3mkt_away = 1 - ph; r.v3mkt_awayFair = fairMl(1 - ph); r.v3mkt_homeFair = fairMl(ph);
  }
  // model picks against the market odds on the row
  for (const m of MODELS) {
    const pk = modelPick(r[`${m.key}_away`], r.mkt_awayMl, r.mkt_homeMl, r.pick_threshold == null ? 2 : r.pick_threshold);
    r[`${m.key}_pick`] = pk.side; r[`${m.key}_edgePp`] = pk.edgePp; r[`${m.key}_edgeCents`] = pk.edgeCents;
  }
  const pm = r.pick_model || 'pyth';
  r.pick_side = r[`${pm}_pick`]; r.pick_edgePp = r[`${pm}_edgePp`]; r.pick_edgeCents = r[`${pm}_edgeCents`];

  const voided = /postponed|cancel/i.test(String(r.gameStatus || ''));
  const final = Number.isFinite(r.finalAway) && Number.isFinite(r.finalHome);
  const winner = final ? (r.finalAway > r.finalHome ? 'away' : r.finalHome > r.finalAway ? 'home' : '') : '';
  // closing no-vig probability: the feed's consensus across books, else the closing prices on the row
  const nvA = Number.isFinite(r.close_nvAway) ? r.close_nvAway : (isMl(r.close_awayMl) && isMl(r.close_homeMl) ? noVigAway(r.close_awayMl, r.close_homeMl) : null);
  const closeNv = side => (nvA == null ? null : side === 'away' ? nvA : 1 - nvA);
  // your bet
  if (r.played === 'yes' && r.side && isMl(r.oddsTaken)) {
    const stake = r.stake > 0 ? r.stake : 1;
    r.result = voided ? 'void' : winner ? (winner === r.side ? 'W' : 'L') : '';
    r.plUnits = r.result === 'W' ? stake * payout(r.oddsTaken) : r.result === 'L' ? -stake : r.result === 'void' ? 0 : null;
    r.clvPp = nvA != null ? 100 * (closeNv(r.side) - implied(r.oddsTaken)) : null;
    r.clvEvPct = nvA != null ? 100 * (closeNv(r.side) * (1 + payout(r.oddsTaken)) - 1) : null;
  } else { r.result = ''; r.plUnits = null; r.clvPp = null; r.clvEvPct = null; }
  // the primary model's pick, graded hypothetically at the market price on the row, 1 unit
  if (r.pick_side) {
    const ml = r.pick_side === 'away' ? r.mkt_awayMl : r.mkt_homeMl;
    r.pick_result = voided ? 'void' : winner ? (winner === r.pick_side ? 'W' : 'L') : '';
    r.pick_plUnits = r.pick_result === 'W' ? payout(ml) : r.pick_result === 'L' ? -1 : r.pick_result === 'void' ? 0 : null;
    r.pick_clvPp = nvA != null ? 100 * (closeNv(r.pick_side) - implied(ml)) : null;
  } else { r.pick_result = ''; r.pick_plUnits = null; r.pick_clvPp = null; }
  return r;
}

// ---------- summaries ----------
function tally(items) {
  // items: [{ result, pl, clv, stake }]
  const s = { n: items.length, w: 0, l: 0, void: 0, pending: 0, units: 0, staked: 0, clvSum: 0, clvN: 0 };
  for (const it of items) {
    if (it.result === 'W') s.w++; else if (it.result === 'L') s.l++; else if (it.result === 'void') s.void++; else s.pending++;
    if (Number.isFinite(it.pl)) { s.units += it.pl; if (it.result !== 'void') s.staked += it.stake || 1; }
    if (Number.isFinite(it.clv)) { s.clvSum += it.clv; s.clvN++; }
  }
  return { n: s.n, record: `${s.w}-${s.l}` + (s.void ? `-${s.void}` : ''), pending: s.pending, winPct: s.w + s.l ? s.w / (s.w + s.l) : null,
    units: s.units, roi: s.staked ? s.units / s.staked : null, avgClvPp: s.clvN ? s.clvSum / s.clvN : null, clvN: s.clvN };
}

// Latest run of each game (re-running a game adds a row; model/passed stats use the last one).
function latestPerGame(rows) {
  const m = new Map();
  for (const r of rows) { const k = String(r.gamePk); if (!m.has(k) || String(r.runAt) > String(m.get(k).runAt)) m.set(k, r); }
  return [...m.values()];
}

const EDGE_BUCKETS = [[-Infinity, 20, 'under 20c'], [20, 30, '20-29c'], [30, 40, '30-39c'], [40, Infinity, '40c+']];

function summarize(rows) {
  const played = rows.filter(r => r.played === 'yes');
  const latest = latestPerGame(rows);
  const passed = latest.filter(r => r.played === 'no' && !played.some(p => String(p.gamePk) === String(r.gamePk)));
  const betItem = r => ({ result: r.result, pl: r.plUnits, clv: r.clvPp, stake: r.stake });
  const pickItem = (r, side) => {
    const ml = side === 'away' ? r.mkt_awayMl : r.mkt_homeMl, final = Number.isFinite(r.finalAway) && Number.isFinite(r.finalHome);
    const voided = /postponed|cancel/i.test(String(r.gameStatus || ''));
    const winner = final ? (r.finalAway > r.finalHome ? 'away' : r.finalHome > r.finalAway ? 'home' : '') : '';
    const result = voided ? 'void' : winner ? (winner === side ? 'W' : 'L') : '';
    const nvA = Number.isFinite(r.close_nvAway) ? r.close_nvAway : (isMl(r.close_awayMl) && isMl(r.close_homeMl) ? noVigAway(r.close_awayMl, r.close_homeMl) : null);
    return { result, pl: result === 'W' ? payout(ml) : result === 'L' ? -1 : result === 'void' ? 0 : null,
      clv: nvA != null ? 100 * ((side === 'away' ? nvA : 1 - nvA) - implied(ml)) : null, stake: 1 };
  };
  // your plays, by the primary model's edge on the side you took, at the odds you took
  const playedEdge = r => { const p = r[`${r.pick_model || 'pyth'}_away`]; if (!Number.isFinite(p) || !isMl(r.oddsTaken)) return null;
    const q = r.side === 'away' ? p : 1 - p; return cents(r.oddsTaken) - cents(fairMl(q)); };
  const playedT = tally(played.map(betItem));
  return {
    rows: rows.length, games: latest.length,
    played: playedT,
    passedModelPick: tally(passed.filter(r => r.pick_side).map(r => pickItem(r, r.pick_side))),
    passedCount: passed.length,
    byModel: MODELS.map(m => ({ model: m.key, label: m.label,
      picks: tally(latest.filter(r => r[`${m.key}_pick`]).map(r => pickItem(r, r[`${m.key}_pick`]))),
      yourPlaysAgreeing: tally(played.filter(r => r.side && r[`${m.key}_pick`] === r.side).map(betItem)) })),
    byEdge: EDGE_BUCKETS.map(([lo, hi, label]) => ({ bucket: label, played: tally(played.filter(r => { const e = playedEdge(r); return e !== null && e >= lo && e < hi; }).map(betItem)) })),
    avgClvPp: playedT.avgClvPp,
  };
}

// ---------- columns / CSV ----------
const CSV_COLUMNS = ['id', 'runAt', 'gameDate', 'gamePk', 'gameType', 'startUtc', 'away', 'home', 'venue', 'awaySP', 'homeSP', 'inputs',
  'awayProjRuns', 'homeProjRuns', 'awayLambda', 'homeLambda',
  ...MODELS.flatMap(m => [`${m.key}_away`, `${m.key}_awayFair`, `${m.key}_homeFair`, `${m.key}_pick`, `${m.key}_edgePp`, `${m.key}_edgeCents`]),
  'mkt_source', 'mkt_awayMl', 'mkt_awayBook', 'mkt_homeMl', 'mkt_homeBook', 'mkt_nBooks', 'mkt_books', 'mkt_at',
  'pick_model', 'pick_threshold', 'pick_side', 'pick_edgePp', 'pick_edgeCents',
  'played', 'side', 'oddsTaken', 'stake', 'book', 'notes',
  'close_source', 'close_awayMl', 'close_homeMl', 'close_book', 'close_nvAway', 'close_at',
  'gameStatus', 'finalAway', 'finalHome', 'result', 'plUnits', 'clvPp', 'clvEvPct',
  'pick_result', 'pick_plUnits', 'pick_clvPp',
  'v2_awayRuns', 'v2_homeRuns', 'blend_const', 'blend_mkt', 'blend_model', 'v3_awayRuns', 'v3_homeRuns', 'v3_total', 'v3_tempF', 'v3d_awayRuns', 'v3d_homeRuns', 'awayFraming', 'homeFraming', 'awayFielding', 'homeFielding', 'v3f_awayRuns', 'v3f_homeRuns', 'awayPlatoon', 'homePlatoon', 'awayPairing', 'homePairing',
  'blend3_const', 'blend3_mkt', 'blend3_model', 'eff_awayRuns', 'eff_homeRuns', 'eff_total', 'awayEfficiency', 'homeEfficiency', 'blendE_const', 'blendE_mkt', 'blendE_model',
  ...LEGACY_MODEL_KEYS.flatMap(k => [`${k}_away`, `${k}_awayFair`, `${k}_homeFair`, `${k}_pick`, `${k}_edgePp`, `${k}_edgeCents`])];
// Columns holding numbers (everything else is text) - used by storage that has to re-type values.
const NUMERIC_COLUMNS = new Set(['gamePk', 'awayProjRuns', 'homeProjRuns', 'awayLambda', 'homeLambda',
  ...MODELS.flatMap(m => [`${m.key}_away`, `${m.key}_awayFair`, `${m.key}_homeFair`, `${m.key}_edgePp`, `${m.key}_edgeCents`]),
  'mkt_awayMl', 'mkt_homeMl', 'mkt_nBooks', 'pick_threshold', 'pick_edgePp', 'pick_edgeCents', 'oddsTaken', 'stake',
  'close_awayMl', 'close_homeMl', 'close_nvAway', 'finalAway', 'finalHome', 'plUnits', 'clvPp', 'clvEvPct', 'pick_plUnits', 'pick_clvPp',
  'v2_awayRuns', 'v2_homeRuns', 'blend_const', 'blend_mkt', 'blend_model', 'v3_awayRuns', 'v3_homeRuns', 'v3_total', 'v3_tempF', 'v3d_awayRuns', 'v3d_homeRuns', 'awayFraming', 'homeFraming', 'awayFielding', 'homeFielding', 'v3f_awayRuns', 'v3f_homeRuns', 'awayPlatoon', 'homePlatoon', 'awayPairing', 'homePairing',
  'blend3_const', 'blend3_mkt', 'blend3_model', 'eff_awayRuns', 'eff_homeRuns', 'eff_total', 'awayEfficiency', 'homeEfficiency', 'blendE_const', 'blendE_mkt', 'blendE_model',
  ...LEGACY_MODEL_KEYS.flatMap(k => [`${k}_away`, `${k}_awayFair`, `${k}_homeFair`, `${k}_edgePp`, `${k}_edgeCents`])]);

function toCsv(rows) {
  const esc = v => { if (v === null || v === undefined) return ''; const s = typeof v === 'number' ? String(+v.toFixed(6)) : String(v);
    // a leading = + - @ would run as a formula in a spreadsheet; numbers are safe, text gets a quote prefix
    const safe = typeof v !== 'number' && /^[=+\-@]/.test(s) ? "'" + s : s;
    return /[",\n\r]/.test(safe) ? `"${safe.replace(/"/g, '""')}"` : safe; };
  return [CSV_COLUMNS.join(','), ...rows.map(r => CSV_COLUMNS.map(c => esc(r[c])).join(','))].join('\r\n') + '\r\n';
}

module.exports = { MODELS, BOOKS, CSV_COLUMNS, NUMERIC_COLUMNS, EDITABLE, implied, payout, noVigAway, fairMl, cents, parseMl, modelPick,
  newRow, applyEntry, applyFinal, applyClosing, derive, summarize, latestPerGame, toCsv };
