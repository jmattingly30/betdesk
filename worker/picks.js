// Guest pick sheet: turns J's models into a short, plain-language list. Guests get this list, never the raw model files.
// Sources: game Model B files (cbb_ab, nfl_ab, nba_ab, cfb_ab), tennis_ab.json, and prop_picks.json (the desk's own
// prop flags with their Model B tiers, written by models/grading/snap_flags.js from the headless desk).
//   tier A -> "Strong pick" (3 stars, tested)   tier B -> "Lean" (2 stars)
//   tier C rated "play" by the unproven rating (z.u, models/model-b/unproven.py) -> "Long shot" (1 star, less
//   proven: shown from level 2). Moneylines are never long shots: their chance is converted from the spread and
//   that conversion makes big underdogs look +EV on its own. Files from before z.u existed fall back to 3%+ EV.

const FILES = { CBB: 'cbb_ab.json', NFL: 'nfl_ab.json', NBA: 'nba_ab.json', CFB: 'cfb_ab.json' };
const MARKETS = { spread: 'spread', total: 'total', ml: 'moneyline' };
const LONG_SHOT_EV = 0.03;
let memo = { at: 0, picks: null };
const LABEL = { A: 'Strong pick', B: 'Lean', C: 'Long shot' }, STARS = { A: 3, B: 2, C: 1 };
const BOOKS = { draftkings: 'DraftKings', fanduel: 'FanDuel', betmgm: 'BetMGM', williamhill_us: 'Caesars', caesars: 'Caesars', espnbet: 'ESPN BET',
  bet365: 'bet365', betrivers: 'BetRivers', fanatics: 'Fanatics', ballybet: 'Bally Bet', pinnacle: 'Pinnacle', hardrockbet: 'Hard Rock', circasports: 'Circa', betonlineag: 'BetOnline' };
const bookName = b => BOOKS[b] || b;
const amer = dec => (dec >= 2 ? Math.round((dec - 1) * 100) : Math.round(-100 / (dec - 1)));
const STAT = { pass_att: 'pass attempts', cmp: 'completions', pass_yds: 'passing yards', pass_td: 'passing TDs', carries: 'rush attempts',
  rush_yds: 'rushing yards', targets: 'targets', rec: 'receptions', rec_yds: 'receiving yards', rush_rec_yds: 'rush + rec yards', td: 'anytime TD',
  pts: 'points', reb: 'rebounds', ast: 'assists', fg3m: 'threes', stl: 'steals', blk: 'blocks', tov: 'turnovers', pra: 'pts + reb + ast',
  sog: 'shots on goal', points: 'points', goals: 'goals', assists: 'assists', hits: 'hits', saves: 'saves', goal: 'anytime goal', assist: 'assist', ga: 'goal or assist' };
// Desk prop kinds -> guest tiers. NHL flags are a lean at most: that model has not beaten the closing line yet.
const PROP_TIER = (sp, kind) => (kind === 'proven' || kind === 'flag' ? (sp === 'NHL' ? 'B' : 'A') : kind === 'lean' ? 'B' : kind === 'unproven' ? 'C' : null);

function buildTennis(d, now) {
  const out = [];
  for (const m of d.matches || []) {
    const B = m.B || {}, A = m.A || {};
    if (!B.price || !B.pick || (m.start && Date.parse(m.start) <= now)) continue;
    const tier = B.tier === 'A' || B.tier === 'B' ? B.tier : B.u && B.u.rec === 'play' ? 'C' : null;
    if (!tier) continue;
    const pA = A.win_p1 == null ? null : B.side === 1 ? A.win_p1 : 1 - A.win_p1;
    const opp = B.side === 1 ? m.p2 : m.p1;
    out.push({
      id: `TENNIS:${m.tex_mid || m.fid || m.p1 + m.p2}:ml`, sport: 'TENNIS', date: (m.start || '').slice(0, 10) || null, kickoff: m.start || null,
      home: m.p1, away: m.p2, matchup: `${m.p1} vs ${m.p2}`, event: m.tournament || m.cat || '', market: 'moneyline', pick: `${B.pick} to win`,
      tier, label: LABEL[tier], stars: STARS[tier], proven: tier !== 'C', book: bookName(B.book), price: amer(B.price),
      chance: pA, ev: Math.round(((B.u && B.u.evU) ?? B.edge ?? 0) * 1000) / 10,
      why: pA != null ? `Our model gives ${B.pick} a ${Math.round(pA * 100)}% chance to beat ${opp}.` : `Our model likes ${B.pick} here.`,
      reasons: [...(tier === 'C' && B.u && B.u.why ? [{ s: '·', t: `Less proven: ${B.u.why}` }] : []), ...(B.note ? [{ s: '·', t: B.note }] : [])],
      score: tier === 'C' && B.u ? B.u.score : null, model: null,
    });
  }
  return out;
}

// Plain reason for a prop. When our projection sits on the bet's side of the line, say that; otherwise (an under on a
// player we project just over a low line, a long-priced over) the edge is the price, so compare our chance with the book's.
const implied = am => (am > 0 ? 100 / (am + 100) : -am / (-am + 100));
function propWhy(r, stat, yes) {
  const ours = Math.round((r.ours || 0) * 100), book = Math.round(implied(r.price) * 100);
  const agrees = !yes && r.proj != null && (r.side === 'Over' ? r.proj > r.line : r.proj < r.line);
  if (agrees) return `Our model projects ${r.n} for about ${r.proj} ${stat}; the line is ${r.line}.`;
  return `Our model gives this a ${ours}% chance; at ${r.price > 0 ? '+' : ''}${r.price} the book pays like it's only ${book}%.`;
}
function buildProps(d, today, now) {
  const out = [];
  for (const r of d.rows || []) {
    const tier = PROP_TIER(r.sp, r.kind);
    if (!tier || (r.kick && Date.parse(r.kick) <= now) || (!r.kick && r.d && r.d < today)) continue;
    const stat = STAT[r.k] || r.k, yes = ['td', 'goal', 'assist', 'ga'].includes(r.k) && r.line <= 0.5;
    out.push({
      id: `${r.sp}:prop:${r.g}:${r.n}:${r.k}:${r.side}`, sport: r.sp, date: r.d || null, kickoff: r.kick || null,
      home: r.t, away: r.o, matchup: `${r.n} (${r.t} vs ${r.o})`, market: 'prop', player: r.n,
      pick: yes ? (r.side === 'Under' ? `${r.n} no ${stat}` : `${r.n} ${stat}`) : `${r.n} ${r.side} ${r.line} ${stat}`,
      tier, label: LABEL[tier], stars: STARS[tier], proven: tier !== 'C', book: bookName(r.book), price: r.price,
      chance: r.ours ?? null, ev: Math.round((r.ev ?? 0) * 1000) / 10,
      why: propWhy(r, stat, yes),
      reasons: tier === 'C' ? [{ s: '·', t: 'Less proven: this kind of prop has not made money in our testing yet, so bet small or just track it.' }] : [],
      score: null, model: null,
    });
  }
  return out;
}

export const denverDate = (t = new Date()) =>
  new Intl.DateTimeFormat('en-CA', { timeZone: 'America/Denver', year: 'numeric', month: '2-digit', day: '2-digit' }).format(t);

const r1 = x => Math.round(x * 10) / 10;
const sgn = x => (x > 0 ? '+' : '') + x;

function why(market, z, g) {
  const side = z.side;
  if (market === 'spread') {
    const m = r1(side === g.h ? z.ours : -z.ours);
    return m > 0 ? `Our model has ${side} winning by ${m}.` : m === 0 ? `Our model calls this game a coin flip.` : `Our model has ${side} losing by only ${-m}, and they get ${z.bet.pt} points.`;
  }
  if (market === 'total') return `Our model expects about ${Math.round(z.ours)} total points; the line is ${z.bet.pt}.`;
  return `Our model gives ${side} a ${Math.round(z.bet.p * 100)}% chance to win.`;
}

function build(sport, d, today, now) {
  const out = [];
  for (const [gid, g] of Object.entries(d.games || {})) {
    if (!g.B || (g.d && g.d < today)) continue;
    if (g.k && Date.parse(g.k) <= now) continue;
    for (const [k, market] of Object.entries(MARKETS)) {
      const z = g.B[k];
      if (!z || !z.bet || !z.side) continue;
      const ev = z.bet.ev ?? 0;
      const longShot = k !== 'ml' && (z.u ? z.u.rec === 'play' : ev >= LONG_SHOT_EV);
      const tier = z.tier === 'A' || z.tier === 'B' ? z.tier : longShot ? 'C' : null;
      if (!tier) continue;
      const pick = market === 'moneyline' ? `${z.side} to win` : market === 'total' ? `${z.side} ${z.bet.pt}` : `${z.side} ${sgn(z.bet.pt)}`;
      out.push({
        id: `${sport}:${gid}:${k}`,
        sport, date: g.d || null, kickoff: g.k || null, home: g.h, away: g.a, market, pick,
        tier, label: { A: 'Strong pick', B: 'Lean', C: 'Long shot' }[tier], stars: { A: 3, B: 2, C: 1 }[tier], proven: tier !== 'C',
        book: z.bet.book_name || z.bet.book, price: z.bet.price, chance: z.bet.p ?? null, ev: Math.round(ev * 1000) / 10,
        why: why(market, z, g),
        reasons: [...(tier === 'C' && z.u && z.u.why ? [{ s: '·', t: `Less proven: ${z.u.why}` }] : []), ...(z.reasons || []).map(([s, t]) => ({ s, t }))],
        score: tier === 'C' && z.u ? z.u.score : null,
        model: g.A ? { home: r1(g.A.hs), away: r1(g.A.as) } : null,
      });
    }
  }
  return out;
}

export async function loadPicks(env, origin) {
  const now = Date.now();
  if (memo.picks && now - memo.at < 5 * 60e3) return memo.picks;
  const today = denverDate();
  const all = [];
  const get = async f => { try { const r = await env.ASSETS.fetch(new Request(`${origin}/${f}`)); return r.ok ? await r.json() : null; } catch (e) { return null; } };
  // one bad or missing file should not empty the sheet
  for (const [sport, f] of Object.entries(FILES)) { const d = await get(f); if (d) all.push(...build(sport, d, today, now)); }
  const t = await get('tennis_ab.json'); if (t) all.push(...buildTennis(t, now));
  const pp = await get('prop_picks.json'); if (pp) all.push(...buildProps(pp, today, now));
  all.sort((a, b) => b.stars - a.stars || (a.kickoff || a.date || '').localeCompare(b.kickoff || b.date || '') || b.ev - a.ev);
  memo = { at: now, picks: all };
  return all;
}
