// Guest pick sheet (desk simplification step 5, J 2026-11-25): the same price check as J's desk, in plain words.
// Source: priced_bets.json (models/pricing/price.py), one file for all six sports, game lines and pro props.
//   Bet  -> "Bet"  (3 stars; shown at every level)
//   Lean -> "Lean" (2 stars; shown from the Guided level, as in the approved mockup)
//   Pass -> never shown to guests.
// The old tiers (Strong pick / Lean / Long shot from the *_ab files) are retired; bets guests logged under those names are
// shown with the new names on their page.
const SPORTS = ['NFL', 'NBA', 'CBB', 'CFB', 'NHL', 'MLB'];
const LABEL = { Bet: 'Bet', Lean: 'Lean' }, TIER = { Bet: 'A', Lean: 'B' }, STARS = { Bet: 3, Lean: 2 };
let memo = { at: 0, picks: null, games: null };
const STAT = { pass_att: 'pass attempts', cmp: 'completions', pass_yds: 'passing yards', pass_td: 'passing TDs', carries: 'rush attempts',
  rush_yds: 'rushing yards', targets: 'targets', rec: 'catches', rec_yds: 'receiving yards', rush_rec_yds: 'rush + rec yards', td: 'touchdowns',
  pts: 'points', reb: 'rebounds', ast: 'assists', fg3m: 'threes', stl: 'steals', blk: 'blocks', tov: 'turnovers', pra: 'points + rebounds + assists',
  sog: 'shots on goal', points: 'points', goals: 'goals', assists: 'assists', hits: 'hits', saves: 'saves' };
const UNIT = { NHL: 'goals', MLB: 'runs' };

export const denverDate = (t = new Date()) =>
  new Intl.DateTimeFormat('en-CA', { timeZone: 'America/Denver', year: 'numeric', month: '2-digit', day: '2-digit' }).format(t);

const r1 = x => Math.round(x * 10) / 10;
const won10 = am => (am > 0 ? am / 10 : 1000 / -am).toFixed(2);   // dollars won on a $10 bet
const half = l => l % 1 !== 0;

// "Nuggets to win by 4 or more", "Kansas and Duke score 149 or more points", "J. Smith 4 or fewer catches"
function plain(r, g) {
  const L = r.line, u = UNIT[r.sp] || 'points';
  if (r.market === 'ml') return `${r.pick} to win`;
  if (r.market === 'spread') {
    if (Math.abs(L) === 0.5) return `${r.pick} to win (${L > 0 ? '+' : ''}${L})`;
    if (L < 0) return half(L) ? `${r.pick} to win by ${Math.ceil(-L)} or more` : `${r.pick} to win by ${-L + 1} or more (by exactly ${-L} refunds)`;
    if (L === 0) return `${r.pick} to win (a tie refunds)`;
    return half(L) ? `${r.pick} to win, or lose by ${Math.floor(L)} or fewer` : `${r.pick} to win, or lose by ${L - 1} or fewer (by exactly ${L} refunds)`;
  }
  if (r.market === 'total') {
    const who = g ? `${g.a} and ${g.h}` : r.matchup;
    if (r.side === 'Over') return half(L) ? `${who} score ${Math.ceil(L)} or more ${u}` : `${who} score more than ${L} ${u}`;
    return half(L) ? `${who} score ${Math.floor(L)} or fewer ${u}` : `${who} score fewer than ${L} ${u}`;
  }
  const st = STAT[r.stat] || r.stat;
  if (r.stat === 'td' && L <= 0.5) return r.side === 'Over' ? `${r.player} scores a touchdown` : `${r.player} does not score a touchdown`;
  if (r.side === 'Over') return half(L) ? `${r.player} ${Math.ceil(L)} or more ${st}` : `${r.player} more than ${L} ${st}`;
  return half(L) ? `${r.player} ${Math.floor(L)} or fewer ${st}` : `${r.player} fewer than ${L} ${st}`;
}
// One plain reason. When our number sits on the bet's side of the line, say it; when it does not (a long price on a
// player we project under the line), the edge is the price, so compare our chance with what the price pays.
const implied = am => (am > 0 ? 100 / (am + 100) : -am / (-am + 100));
function plainWhy(r, g) {
  const price = `Best price: ${r.book_name}, bet $10 to win $${won10(r.price)}.`;
  const byChance = `We give this a ${Math.round((r.trusted ?? r.ours) * 100)}% chance; the price pays like it's ${Math.round(implied(r.price) * 100)}%. ${price}`;
  const over = r.side === 'Over';
  if (r.market === 'prop') {
    if (r.proj == null || (over ? r.proj <= r.line : r.proj >= r.line)) return byChance;
    return `We expect about ${r1(r.proj)} ${STAT[r.stat] || r.stat}. ${price}`;
  }
  if (r.market === 'ml') return `We give ${r.pick} a ${Math.round(r.ours * 100)}% chance to win. ${price}`;
  if (!g || g.hs == null) return byChance;
  if (r.market === 'total') { const t = g.hs + g.as_; return (over ? t > r.line : t < r.line) ? `We expect about ${Math.round(t)} ${UNIT[r.sp] || 'points'}. ${price}` : byChance; }
  const m = r1(r.pick === g.h ? g.hs - g.as_ : g.as_ - g.hs);
  if (m + r.line <= 0) return byChance;
  return (m > 0 ? `We think ${r.pick} wins by about ${m}.` : m === 0 ? 'We think this game is a coin flip.' : `We think ${r.pick} loses by only about ${-m}.`) + ' ' + price;
}

function build(P, now) {
  const games = [], picks = [], G = {};
  for (const g of P.games || []) {
    if (!SPORTS.includes(g.sp) || (g.k && Date.parse(g.k) <= now)) continue;
    const key = `${g.sp}:${g.id}`; G[key] = g;
    games.push({ key, sport: g.sp, home: g.h, away: g.a, date: (g.k || '').slice(0, 10) || null, kickoff: g.k || null,
      hs: g.hs == null ? null : r1(g.hs), as: g.as_ == null ? null : r1(g.as_), p_home: g.p_home ?? null, spread: g.spread ?? null, total: g.total ?? null });
  }
  for (const r of P.bets || []) {
    if (!LABEL[r.label] || !SPORTS.includes(r.sp) || (r.k && Date.parse(r.k) <= now)) continue;
    const key = `${r.sp}:${r.gid || r.game}`, g = G[key];   // gid: the priced game a prop belongs to (NBA prop keys are slate indices)
    picks.push({
      id: `${r.sp}:${r.game}:${r.market}:${r.stat || ''}:${r.player || ''}:${r.side}`, game: g ? key : null, sport: r.sp,
      date: (r.k || '').slice(0, 10) || r.d || null, kickoff: r.k || null, home: g ? g.h : null, away: g ? g.a : null,
      matchup: r.matchup, market: r.market === 'ml' ? 'moneyline' : r.market, player: r.player || null,
      pick: plain(r, g), tier: TIER[r.label], label: LABEL[r.label], stars: STARS[r.label], proven: true,
      book: r.book_name, price: r.price, chance: r.trusted ?? r.ours ?? null, ev: Math.round((r.ev ?? 0) * 1000) / 10, units: r.label === 'Bet' ? r.stake_u : null,
      why: plainWhy(r, g), reasons: r.reason ? [{ s: '·', t: r.reason }] : [], score: null,
      model: g && g.hs != null ? { home: r1(g.hs), away: r1(g.as_) } : null,
    });
  }
  return { games, picks };
}

export async function loadPicks(env, origin) {
  const now = Date.now();
  if (memo.picks && now - memo.at < 5 * 60e3) return memo;
  let P = null;
  try { const r = await env.ASSETS.fetch(new Request(`${origin}/priced_bets.json`)); P = r.ok ? await r.json() : null; } catch (e) { P = null; }
  const { games, picks } = P ? build(P, now) : { games: [], picks: [] };
  games.sort((a, b) => (a.kickoff || a.date || '').localeCompare(b.kickoff || b.date || ''));
  picks.sort((a, b) => b.stars - a.stars || (a.kickoff || a.date || '').localeCompare(b.kickoff || b.date || '') || b.ev - a.ev);
  memo = { at: now, picks, games };
  return memo;
}
