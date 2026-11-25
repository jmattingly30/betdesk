'use strict';
// Market odds from The Odds API (https://the-odds-api.com), optional - a licensed odds feed, not scraping.
// Pure parsing only; the request is made server-side with your key (ODDS_API_KEY / Script Properties),
// never from the page. One request returns every MLB game and every book in the chosen regions; it costs
// 1 credit per region (free plan: 500 credits/month).

const { findTeam } = require('./util');

const ODDS_URL = (key, regions) => `https://api.the-odds-api.com/v4/sports/baseball_mlb/odds?regions=${encodeURIComponent(regions || 'us')}&markets=h2h&oddsFormat=american&apiKey=${encodeURIComponent(key)}`;

const implied = ml => (ml < 0 ? -ml / (-ml + 100) : 100 / (ml + 100));

// Every book's moneyline for one game. game: { awayId, homeId, startUtc }.
// Doubleheaders: uses the listed event whose start time is closest to this game's (within 6 hours).
function gameBooks(events, game) {
  if (!Array.isArray(events)) return [];
  const start = Date.parse(game.startUtc);
  const cands = events.filter(e => {
    const a = findTeam(e.away_team), h = findTeam(e.home_team);
    return a && h && a.id === game.awayId && h.id === game.homeId;
  });
  if (!cands.length) return [];
  cands.sort((x, y) => Math.abs(Date.parse(x.commence_time) - start) - Math.abs(Date.parse(y.commence_time) - start));
  const ev = cands[0];
  if (Number.isFinite(start) && Math.abs(Date.parse(ev.commence_time) - start) > 6 * 3600 * 1000) return [];
  const out = [];
  for (const bk of ev.bookmakers || []) {
    const mk = (bk.markets || []).find(m => m.key === 'h2h');
    if (!mk) continue;
    const price = id => { const o = (mk.outcomes || []).find(x => { const t = findTeam(x.name); return t && t.id === id; }); return o ? o.price : null; };
    const awayMl = price(game.awayId), homeMl = price(game.homeId);
    if (Number.isFinite(awayMl) && Number.isFinite(homeMl) && Math.abs(awayMl) >= 100 && Math.abs(homeMl) >= 100) {
      out.push({ key: bk.key, book: bk.title || bk.key, awayMl, homeMl, at: mk.last_update || bk.last_update || '' });
    }
  }
  return out.sort((a, b) => a.book.localeCompare(b.book));
}

// Summary for the log: best price on each side (for book shopping) and the consensus no-vig probability
// (average of each book's own no-vig away probability) as the market's fair line.
function summarizeBooks(books) {
  if (!books || !books.length) return null;
  const best = side => books.reduce((b, x) => (b == null || implied(x[side]) < implied(b[side]) ? x : b), null);
  const bA = best('awayMl'), bH = best('homeMl');
  const nv = books.reduce((s, x) => s + implied(x.awayMl) / (implied(x.awayMl) + implied(x.homeMl)), 0) / books.length;
  return { awayMl: bA.awayMl, awayBook: bA.book, homeMl: bH.homeMl, homeBook: bH.book, nvAway: nv, nBooks: books.length,
    at: books.reduce((m, x) => (x.at > m ? x.at : m), ''), books };
}

module.exports = { ODDS_URL, gameBooks, summarizeBooks };
