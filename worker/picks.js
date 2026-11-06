// Guest pick sheet: turns the Model B files (cbb_ab.json, nfl_ab.json, nba_ab.json, cfb_ab.json) into a short,
// plain-language list. Guests get this list, never the raw model files.
//   tier A -> "Strong pick" (3 stars, tested)   tier B -> "Lean" (2 stars)
//   tier C with 3%+ expected value -> "Long shot" (1 star, less proven: shown from level 2)

const FILES = { CBB: 'cbb_ab.json', NFL: 'nfl_ab.json', NBA: 'nba_ab.json', CFB: 'cfb_ab.json' };
const MARKETS = { spread: 'spread', total: 'total', ml: 'moneyline' };
const LONG_SHOT_EV = 0.03;
let memo = { at: 0, picks: null };

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
      const tier = z.tier === 'A' || z.tier === 'B' ? z.tier : ev >= LONG_SHOT_EV ? 'C' : null;
      if (!tier) continue;
      const pick = market === 'moneyline' ? `${z.side} to win` : market === 'total' ? `${z.side} ${z.bet.pt}` : `${z.side} ${sgn(z.bet.pt)}`;
      out.push({
        id: `${sport}:${gid}:${k}`,
        sport, date: g.d || null, kickoff: g.k || null, home: g.h, away: g.a, market, pick,
        tier, label: { A: 'Strong pick', B: 'Lean', C: 'Long shot' }[tier], stars: { A: 3, B: 2, C: 1 }[tier], proven: tier !== 'C',
        book: z.bet.book_name || z.bet.book, price: z.bet.price, chance: z.bet.p ?? null, ev: Math.round(ev * 1000) / 10,
        why: why(market, z, g),
        reasons: (z.reasons || []).map(([s, t]) => ({ s, t })),
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
  for (const [sport, f] of Object.entries(FILES)) {
    try {
      const r = await env.ASSETS.fetch(new Request(`${origin}/${f}`));
      if (r.ok) all.push(...build(sport, await r.json(), today, now));
    } catch (e) { /* one bad file should not empty the sheet */ }
  }
  all.sort((a, b) => b.stars - a.stars || (a.kickoff || a.date || '').localeCompare(b.kickoff || b.date || '') || b.ev - a.ev);
  memo = { at: now, picks: all };
  return all;
}
