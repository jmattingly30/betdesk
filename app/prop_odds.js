// ---------- prop odds (J 2026-10-27: book prices on every prop, EV vs our model, props +EV list, prop middles) ----------
// props_{NFL,NBA,NHL}.json from op_live.py (OddsPapi, free): games[{id, k, h, a, players: {name: {stat: {line: {book: {Over|Under|Yes|No|"250+": American}}}}}}]
const PO = {}, PO_SP = ['NFL', 'NBA', 'NHL'];
const PO_MAP = {
  NFL:{'Pass Yards':'pass_yds', 'Pass Attempts':'pass_att', 'Pass Completions':'cmp', 'TD Passes':'pass_td', 'Rush Yards':'rush_yds',
    'Rush Attempts':'carries', 'Receiving Yards':'rec_yds', 'Receptions':'rec', 'To Score TD':'td'},
  NBA:{'Points':'pts', 'Rebounds':'reb', 'Assists':'ast', '3 Point FG':'fg3m', 'Points + Assists + Rebounds':'pra', 'Steals':'stl', 'Blocks':'blk', 'Turnovers':'tov'},
  NHL:{'Shots On Goal':'sog', 'Points':'points', 'Goals':'goals', 'Anytime Goal Scorer':'goals', 'Assists':'assists', 'Blocks':'blk', 'Saves':'saves'}};
const PO_BOOK = {bet365:'bet365', ballybet:'Bally Bet', williamhill_us:'Caesars', betrivers:'BetRivers', fanatics:'Fanatics', pinnacle:'Pinnacle'};
const poBk = b => (LT && LT.books && LT.books[b]) || PO_BOOK[b] || b;
const poMine = b => typeof myBook === 'function' ? myBook(b) : b !== 'pinnacle';
const poNorm = n => String(n).toLowerCase().normalize('NFD').replace(/[̀-ͯ]/g, '').replace(/[.'’,-]/g, ' ')
  .replace(/\b(jr|sr|ii|iii|iv|v)\b/g, '').replace(/\s+/g, ' ').trim();
const poInit = n => { const w = poNorm(n).split(' '); return w.length > 1 ? w[0][0] + ' ' + w.slice(1).join(' ') : w[0]; };
// does a team abbreviation (GB, LAR, PHX, VGK) name this full team name? prefix, initials, or letters in order in the first word
const PO_TEAM = {JAX:'JACKSONVILLE', WSH:'WASHINGTON', LA:'LOS ANGELES', NJ:'NEW JERSEY'};
function poTeam(ab, full){
  const A = String(ab || '').toUpperCase(), F = String(full || '').toUpperCase().replace(/[^A-Z ]/g, ''), w = F.split(' ').filter(Boolean);
  if (!A || !w.length) return false;
  if (PO_TEAM[A] && F.startsWith(PO_TEAM[A])) return true;
  if (F.startsWith(A) || w.map(x => x[0]).join('').startsWith(A)) return true;
  let i = 0; for (const ch of w[0]) if (ch === A[i]) i++; return i === A.length;
}
const POI = {};
function poIndex(sp){
  const d = PO[sp]; if (!d) return null; if (POI[sp] && POI[sp].d === d) return POI[sp].m;
  const m = new Map(), add = (k, x) => { if (!m.has(k)) m.set(k, []); m.get(k).push(x); };
  d.games.forEach(g => Object.entries(g.players).forEach(([n, P]) => { const x = {g, n, P}; add('f:' + poNorm(n), x); add('i:' + poInit(n), x); }));
  POI[sp] = {d, m}; return m;
}
const poDay = k => { const t = new Date(k); return `${t.getFullYear()}-${String(t.getMonth() + 1).padStart(2, '0')}-${String(t.getDate()).padStart(2, '0')}`; };
// this player's OddsPapi entry: same name (full, or first initial + last) and both teams in the game; then the same day if we know it
function poFind(sp, p){
  const m = poIndex(sp); if (!m || !p) return null;
  const c = [...(m.get('f:' + poNorm(p.n)) || []), ...(m.get('i:' + poInit(p.n)) || [])];
  const ok = c.filter(x => (poTeam(p.t, x.g.h) && poTeam(p.o, x.g.a)) || (poTeam(p.t, x.g.a) && poTeam(p.o, x.g.h)));
  return (p.d && ok.find(x => poDay(x.g.k) === p.d)) || ok[0] || null;
}
// every book price for one of our stats: [{L, b, o, u, main}] at your books (Settings > My books), with our model's EV on each side
function poRows(sp, p, stat, all = false){   // all: every book in the sweep (market price), not only yours
  const s = p && p.s[stat], x = s && PO_SP.includes(sp) && poFind(sp, p); if (!x) return [];
  const map = PO_MAP[sp] || {}, at = new Map();
  const put = (L, b, side, pr) => { const k = L + '|' + b; if (!at.has(k)) at.set(k, {L, b, o:null, u:null});
    const r = at.get(k); if (side === 'o' && r.o == null) r.o = pr; if (side === 'u' && r.u == null) r.u = pr; };
  Object.entries(x.P).forEach(([name, lines]) => {
    if (map[name.replace(/ ladder$/, '')] !== stat) return;   // a ladder ("250+") is the same stat's alt overs
    Object.entries(lines).forEach(([l0, B]) => Object.entries(B).forEach(([b, S]) => { if (!all && !poMine(b)) return;
      Object.entries(S).forEach(([sd, pr]) => { if (!Number.isFinite(+pr)) return;
        const n = /^(\d+)\+$/.exec(sd);
        if (n) put(+n[1] - 0.5, b, 'o', +pr);
        else if (sd === 'Over' || sd === 'Under') put(+l0, b, sd === 'Over' ? 'o' : 'u', +pr);
        else if (sd === 'Yes' || sd === 'No') put(0.5, b, sd === 'Yes' ? 'o' : 'u', +pr);
      }); })); });
  // a two-sided line priced like a main line (both sides -140 to +120) far from the other books' main line is a bad feed
  // row (bet365 posts some at -120/-120 25 yards off): shown with a ?, left out of +EV and middles
  // bet365 also posts -120/-120 on several lines at once for one player (2026-10-27 sweep: 211 of its 384 two-sided rows): placeholders
  const ph = r => r.b === 'bet365' && r.o === -120 && r.u === -120;
  const mains = [...at.values()].filter(r => r.o != null && r.u != null && !ph(r)).map(r => r.L).sort((a, b) => a - b), med = mains.length ? mains[mains.length >> 1] : null;
  const odd = r => ph(r) || med != null && r.o != null && r.u != null && Math.abs(r.L - med) > Math.max(1, 0.08*Math.abs(med)) && [r.o, r.u].every(v => v >= -140 && v <= 120);
  return [...at.values()].map(r => { const mp = modelProbs(stat, s, r.L);
    return {...r, main:r.o != null && r.u != null, odd:odd(r), po:mp.po, pu:mp.pu, pp:mp.pp,
      eo:r.o == null ? null : mp.po*(dec(r.o) - 1) - mp.pu, eu:r.u == null ? null : mp.pu*(dec(r.u) - 1) - mp.po}; })
    .sort((a, b) => a.L - b.L || (b.o ?? -1e9) - (a.o ?? -1e9));
}
const poBestEv = r => Math.max(r.eo ?? -9, r.eu ?? -9);
// fill the card's line, prices and book from the best-EV main line once per player and stat (J types nothing)
let PO_FILLED = '', PO_ALT = false;
function poAutofill(p, stat, rows){
  const key = S.sport + '|' + p.n + '|' + (p.d || '') + '|' + stat; if (PO_FILLED === key) return null; PO_FILLED = key;
  // main lines (both sides posted); with none (anytime TD/goal), the line most books post, never a long-shot ladder rung
  let pool = rows.filter(r => r.main && !r.odd);
  if (!pool.length && rows.length){ const n = {}; rows.forEach(r => n[r.L] = (n[r.L] || 0) + 1);
    const top = Object.keys(n).reduce((a, k) => n[k] > n[a] ? k : a); pool = rows.filter(r => String(r.L) === top); }
  if (!pool.length) return null;
  const f = pfFor(p, stat); if (f && f.kind !== 'check') return poSet(f.r);   // a line flag: open on its book
  return poSet(pool.reduce((a, r) => poBestEv(r) > poBestEv(a) ? r : a));
}
function poSet(r){
  $('#pLine').value = r.L; $('#pOver').value = r.o ?? ''; $('#pUnder').value = r.u ?? ''; $('#pBook').value = poBk(r.b); return r.L;
}
function poTable(sp, rows, L){
  if (!PO_SP.includes(sp)) return '';
  if (!PO[sp]) return '';
  if (!rows.length) return `<div class="card"><h3>Book prices</h3><p class="small muted">No book has this prop at your books in the ${esc(PO[sp].made.slice(0, 10))} odds sweep.</p></div>`;
  const alt = rows.filter(r => !r.main).length, show = PO_ALT ? rows : rows.filter(r => r.main).length ? rows.filter(r => r.main) : rows;
  const bestO = {}, bestU = {}; show.forEach(r => { if (r.o != null && !(bestO[r.L] >= r.o)) bestO[r.L] = r.o; if (r.u != null && !(bestU[r.L] >= r.u)) bestU[r.L] = r.u; });
  const ev = v => v == null ? '' : `<span class="${v >= 0 ? 'pos' : 'neg'}">${sgn(v)}</span>`;
  return `<div class="card"><h3>Book prices</h3><p class="small muted">Your books, ${esc(PO[sp].made.slice(0, 16).replace('T', ' '))} UTC sweep. EV is against our model's chance at each line. Best over and under price per line in bold. Tap a row to price it.${show.some(r => r.odd) ? ' ? = a bet365 -120/-120 placeholder or a line far from the other books\', left out of +EV and middles.' : ''}</p>
    <div class="tablewrap"><table class="tight"><thead><tr><th>Line</th><th>Book</th><th>Over</th><th>Under</th><th>EV over</th><th>EV under</th></tr></thead><tbody>${show.map(r => { const i = rows.indexOf(r);
      return `<tr class="click${r.L === L && poBk(r.b) === $('#pBook').value ? ' sel' : ''}" data-po="${i}"><td class="num">${r.L}${r.odd ? ' ?' : ''}</td><td>${esc(poBk(r.b))}</td><td class="num">${r.o == null ? '–' : r.o === bestO[r.L] ? '<b>' + amS(r.o) + '</b>' : amS(r.o)}</td><td class="num">${r.u == null ? '–' : r.u === bestU[r.L] ? '<b>' + amS(r.u) + '</b>' : amS(r.u)}</td><td class="num">${ev(r.eo)}</td><td class="num">${ev(r.eu)}</td></tr>`; }).join('')}</tbody></table></div>
    ${alt && rows.some(r => r.main) ? `<div class="row"><button class="btn ghost sm" type="button" id="poAlt">${PO_ALT ? 'Main lines only' : `Show alt lines (${alt})`}</button></div>` : ''}</div>`;
}
function poWire(out, rows){
  out.querySelectorAll('tr[data-po]').forEach(tr => tr.onclick = () => { poSet(rows[+tr.dataset.po]); renderProp(); });
  const a = out.querySelector('#poAlt'); if (a) a.onclick = () => { PO_ALT = !PO_ALT; renderProp(); };
}
// Flags view modes: line flags (J 2026-10-28, from the real price at your books) and prop middles
let PO_MODE = lsGet('betdesk.flagmode2', 'line');
function poAll(sp){
  const sl = S.slates[sp], st = $('#fStat').value, out = []; if (!sl || !PO[sp]) return out;
  sl.players.forEach(p => { if (!upcoming(p) || !dayOk(p)) return;
    Object.keys(p.s).forEach(k => { if (st && k !== st) return; const rows = poRows(sp, p, k); if (rows.length) out.push({p, k, rows}); }); });
  return out;
}
// over at a lower line at one book + under at a higher line at another; equal returns either way (as the game middle calculator)
function poMidOf(s, k, o, u){
  const d1 = dec(o.o), d2 = dec(u.u), x1 = 100*d2/(d1 + d2), x2 = 100 - x1, w1 = x1*(d1 - 1), w2 = x2*(d2 - 1);
  const m1 = modelProbs(k, s, o.L), m2 = modelProbs(k, s, u.L);
  const P = [[1 - m1.po - m1.pp, -x1 + w2], [m1.pp, w2], [Math.max(0, m1.po - m2.po - m2.pp), w1 + w2], [m2.pp, w1], [m2.po, w1 - x2]].filter(q => q[0] > 1e-9);
  const ev = P.reduce((a, q) => a + q[0]*q[1], 0)/100;
  return {both:Math.max(0, m1.po - m2.po - m2.pp), ev, worst:Math.min(...P.map(q => q[1]))/100, best:Math.max(...P.map(q => q[1]))/100, stake:[+x1.toFixed(1), +x2.toFixed(1)]};
}
function poMidList(sp){
  const L = [];
  poAll(sp).forEach(({p, k, rows}) => { const s = p.s[k], ov = rows.filter(r => r.o != null && !r.odd), un = rows.filter(r => r.u != null && !r.odd);
    ov.forEach(o => un.forEach(u => { if (!(u.L > o.L)) return; const m = poMidOf(s, k, o, u); if (m.ev > 0) L.push({p, k, o, u, ...m}); })); });
  return L.sort((a, b) => b.ev - a.ev);
}
function poOpen(p, k, r){ $('#pPlayer').value = p.n; onPlayer(); clearLine(); $('#pStat').value = k; PO_FILLED = S.sport + '|' + p.n + '|' + (p.d || '') + '|' + k; poSet(r); renderProp(); showSub('props'); }
function poModes(){
  const sp = S.sport, has = !!PO[sp];
  const el = $('#flagMode'); if (!el) return;
  if (PO_MODE !== 'mid') PO_MODE = 'line';
  el.innerHTML = has ? [['line', 'Flags'], ['mid', 'Prop middles']].map(([m, l]) => `<button type="button" data-fm="${m}" aria-pressed="${PO_MODE === m}">${l}</button>`).join('') : '';
  el.querySelectorAll('[data-fm]').forEach(b => b.onclick = () => { PO_MODE = b.dataset.fm; lsSet('betdesk.flagmode2', PO_MODE); renderFlags(); });
  $('#fThrWrap').hidden = has;
  const intro = $('#flagIntro'); if (intro) intro.hidden = has;
}

// ---------- line flags (J 2026-10-28: flag from the actual line and price in our odds pull, not past averages) ----------
// Flag = the best price at your books on a side of the main line (near the books' consensus line), where our chance blended
// 50/50 with the books' no-vig chance, and our chance alone, both give EV at or above the flag minimum. The blend leans half on the books because our
// NBA model alone is less accurate than they are (2026 odds-history test, models/oddspapi/README.md, corrected 10-28).
// Proven = NBA overs at DraftKings/FanDuel with 12%+ EV on our model alone (that test: 201 bets, +11.6% ROI, +3.6% CLV,
// Mar-Jun 2026). Check = our number is far from the books' (projection 25%+ off a line of 3+, or our
// chance 15+ points from theirs): usually a minutes, role or injury miss, never flagged. A price 25%+ more generous than the
// other books' median at the same line (3+ books) is left out as stale or a different market, and prices longer than +300
// are not flagged (our long-tail chances are untested). NBA unders are held back: no edge
// in the test (CLV negative) and October NBA projections run 5-12% low. Model math is unchanged.
const PF_NOTE = {NBA:'Tested: NBA overs at DraftKings/FanDuel made money in the 2026 odds-history test. Unders are held back.',
  NHL:'Our NHL model out-predicts the books, but its bets have not beaten the closing line yet.', NFL:'Not tested against odds history yet.'};
const PF_PROVEN = {sp:'NBA', books:['draftkings', 'fanduel'], ev:0.12};
let PF_THR = +lsGet('betdesk.pfthr', '0.05') || 0.05;
const pfCond = (k, s, L) => { const m = modelProbs(k, s, L); return {...m, c:m.po/Math.max(m.po + m.pu, 1e-9)}; };
// the books' no-vig chance of the over at line L (no push): every two-sided price in the sweep, each moved to L along our
// curve's slope, median; with none two-sided (anytime TD or goal), the median implied over price less a 4.5% margin
function pfMarket(k, s, all, L){
  const c = pfCond(k, s, L).c, est = [];
  all.filter(r => r.main && !r.odd).forEach(r => { const q = (1/dec(r.o))/(1/dec(r.o) + 1/dec(r.u)); est.push(r.L === L ? q : il(lg(q) + lg(c) - lg(pfCond(k, s, r.L).c))); });
  if (!est.length) all.filter(r => r.L === L && r.o != null).forEach(r => est.push(Math.min(0.99, (1/dec(r.o))/1.07)));   // one-sided props hold about 7%
  if (!est.length) return null;
  est.sort((a, b) => a - b); const n = est.length; return n % 2 ? est[n >> 1] : (est[n/2 - 1] + est[n/2])/2;
}
const PFC = new Map();
// every line flag for one sport: [{p, k, side, r, b, pr, L, ch, chb, ev, evb, mk, kind}] (kind proven | flag | check)
function pfAll(sp, everyDay = false){
  const sl = S.slates[sp]; if (!sl || !PO[sp]) return {rows:[], held:0};
  const key = [sp, PO[sp].made, sl.made, everyDay ? '*' : S.day, [...booksOff()].sort().join(','), PF_THR].join('|');
  if (PFC.has(key)) return PFC.get(key);
  const rows = []; let held = 0;
  sl.players.forEach(p => { if (!upcoming(p) || (!everyDay && !dayOk(p))) return;
    Object.entries(p.s).forEach(([k, s]) => { const mine = poRows(sp, p, k); if (!mine.length) return;
      const all = poRows(sp, p, k, true);
      // only lines near the books' consensus main line (a line far from it is usually stale or a different market)
      const ml = all.filter(r => r.main && !r.odd).map(r => r.L).sort((a, b) => a - b), med = ml.length ? ml[ml.length >> 1] : null;
      let pool = mine.filter(r => r.main && !r.odd && (med == null || Math.abs(r.L - med) <= Math.max(0.5, 0.08*Math.abs(med))));
      if (!pool.length){ const n = {}; mine.forEach(r => n[r.L] = (n[r.L] || 0) + 1); const top = Object.keys(n).reduce((a, x) => n[x] > n[a] ? x : a, Object.keys(n)[0]); pool = mine.filter(r => String(r.L) === top && !r.odd); }
      ['Over', 'Under'].forEach(side => { const O = side === 'Over', best = {};
        pool.forEach(r => { const pr = O ? r.o : r.u; if (pr == null || pr > 300) return;
          const imp = all.filter(q => q.L === r.L && !q.odd && (O ? q.o : q.u) != null).map(q => 1/dec(O ? q.o : q.u)).sort((a, b) => a - b);
          if (imp.length >= 3 && 1/dec(pr) < 0.8*imp[imp.length >> 1]) return;
          const m = pfCond(k, s, r.L), mk = pfMarket(k, s, all, r.L), bl = mk == null ? m.c : il(0.5*lg(m.c) + 0.5*lg(mk)), nb = 1 - m.pp;
          const ch = O ? m.po : m.pu, chb = (O ? bl : 1 - bl)*nb, d = dec(pr), x = {p, k, side, r, b:r.b, pr, L:r.L, ch, chb, mk, mc:m.c,
            ev:ch*(d - 1) - (nb - ch), evb:chb*(d - 1) - (nb - chb)};
          if (!best.flag || x.evb > best.flag.evb) best.flag = x;
          if (sp === PF_PROVEN.sp && O && PF_PROVEN.books.includes(r.b) && (!best.pv || x.ev > best.pv.ev)) best.pv = x; });
        const pick = best.pv && best.pv.ev >= PF_PROVEN.ev ? best.pv : best.flag; if (!pick) return;
        const far = (pick.L >= 3 && Math.abs(s.m - pick.L)/pick.L > 0.25) || (pick.mk != null && Math.abs(pick.mc - pick.mk) > 0.15);
        if (far){ if (pick.ev >= PF_THR) rows.push({...pick, kind:'check'}); return; }
        if (sp === 'NBA' && !O){ if (pick.evb >= PF_THR) held++; return; }
        if (pick === best.pv && pick.ev >= PF_PROVEN.ev) rows.push({...pick, kind:'proven'});
        else if (pick.evb >= PF_THR && pick.ev >= PF_THR) rows.push({...pick, kind:'flag'}); }); }); });   // our model alone must like it too
  const rank = {proven:0, flag:1, check:2};
  rows.sort((a, b) => rank[a.kind] - rank[b.kind] || (a.kind === 'check' ? b.ev - a.ev : b.evb - a.evb));
  const out = {rows, held}; PFC.set(key, out); return out;
}
const pfFor = (p, k) => PO[S.sport] ? pfAll(S.sport).rows.filter(x => x.p === p && x.k === k).sort((a, b) => (a.kind === 'check') - (b.kind === 'check'))[0] || null : null;
const PF_TAG = {proven:'<span class="tag t-proven">Proven</span>', flag:'<span class="tag t-flag">Flag</span>', check:'<span class="tag t-check">Check</span>'};
const pfLine = x => x.kind === 'check'
  ? `We project ${x.p.s[x.k].m}; the books' line is ${x.L} (their chance of the over ${pct(x.mk ?? 0)}, ours ${pct(x.mc)}). A gap this size usually means news we're missing, so it isn't flagged.`
  : `${x.side} ${x.L} at ${poBk(x.b)} ${amS(x.pr)}: ${x.kind === 'proven' ? 'our model alone ' + sgn(x.ev) + ' EV (NBA overs at DraftKings/FanDuel, 12%+, made money in the 2026 test)' : 'blended EV ' + sgn(x.evb) + ' (ours ' + sgn(x.ev) + ')'}.`;
// the flags list (Flags view, line mode); returns true when it drew the view
function poRenderFlags(){
  poModes(); const sp = S.sport; if (!PO[sp]) return false;
  const out = $('#flagOut'), when = `${esc(PO[sp].made.slice(0, 16).replace('T', ' '))} UTC`;
  if (PO_MODE === 'line'){
    const st = $('#fStat').value, dv = $('#fDir').value, F = pfAll(sp);
    const L = F.rows.filter(x => (!st || x.k === st) && (dv === 'both' || x.side.toLowerCase() === dv));
    const good = L.filter(x => x.kind !== 'check'), chk = L.filter(x => x.kind === 'check'), show = flagAll ? good : good.slice(0, 40);
    const row = (x, i) => `<button type="button" class="grow" data-i="${i}"><span><b>${tLogo(sp, x.p.t)}${esc(x.p.n)}</b> ${PF_TAG[x.kind]}<br><span class="sm">${esc(x.side)} ${x.L} ${esc(STAT[x.k] || x.k)} · ${esc(poBk(x.b))} <b class="num">${amS(x.pr)}</b> · ${esc(x.p.t)} ${x.p.h ? 'vs' : '@'} ${esc(x.p.o)}${x.p.d && !S.day ? ' · ' + esc(dayName(x.p.d)) : ''}</span></span>
      <span>${x.kind === 'check' ? `<span class="big wrn">check</span><br><span class="sm">we say ${x.p.s[x.k].m}</span>` : `<span class="big pos">${sgn(x.kind === 'proven' ? x.ev : x.evb)}</span><br><span class="sm">${x.kind === 'proven' ? 'our model' : 'blended · ours ' + sgn(x.ev)}</span>`}</span></button>`;
    out.innerHTML = `<p class="small muted">A flag is a real price at your books (${when} sweep) where our chance alone and our chance blended 50/50 with the books' no-vig chance both give EV at or above the flag minimum. <b>Proven</b> = NBA overs at DraftKings or FanDuel with 12%+ EV on our model alone, the bet that made money in the 2026 odds-history test. ${esc(PF_NOTE[sp] || '')}</p>
      <div class="row"><label class="small" style="display:flex;gap:8px;align-items:center">Flag at<select id="pfThr">${[0.03, 0.05, 0.08, 0.12].map(v => `<option value="${v}"${v === PF_THR ? ' selected' : ''}>${Math.round(v*100)}%+ EV</option>`).join('')}</select></label></div>`
      + (show.length ? `<p class="small muted">${good.length} flagged${F.held && dv !== 'over' ? ` · ${F.held} NBA unders held back` : ''}.</p><div class="glist">${show.map((x, i) => row(x, i)).join('')}</div>${good.length > show.length ? `<div class="row"><button class="btn ghost" type="button" id="flagMore">Show all ${good.length}</button></div>` : ''}`
        : `<div class="empty">No ${sp} prop at your books clears ${Math.round(PF_THR*100)}% EV${S.day ? ' on ' + esc(dayName(S.day)) : ''}${F.held && dv !== 'over' ? ` (${F.held} NBA unders held back)` : ''}.</div>`)
      + (chk.length ? `<h3 style="margin-top:14px">Check the news (${chk.length})</h3><p class="small muted">Our projection is far from the books' line. That is usually a minutes, role or injury change we haven't caught, so these are not flags.</p><div class="glist">${chk.slice(0, 15).map((x, i) => row(x, 'c' + i)).join('')}</div>` : '');
    const more = $('#flagMore'); if (more) more.onclick = () => { flagAll = true; renderFlags(); };
    $('#pfThr').onchange = e => { PF_THR = +e.target.value; lsSet('betdesk.pfthr', String(PF_THR)); renderFlags(); renderSubBar(); };
    out.querySelectorAll('.grow[data-i]').forEach(b => b.onclick = () => { const i = b.dataset.i, x = i[0] === 'c' ? chk[+i.slice(1)] : show[+i]; poOpen(x.p, x.k, x.r); });
    return true;
  }
  const L = poMidList(sp), show = L.slice(0, 40);
  out.innerHTML = `<p class="small muted">Over a lower line at one book and under a higher line at another (${when} sweep). Middle chance = both win, from our model. Stakes per 100 return the same either way. A positive worst case (arb) usually means one book's price moved after the sweep: check both books.</p>`
    + (show.length ? show.map((m, i) => `<div class="mkt click" data-i="${i}" style="margin:8px 0"><div class="mh"><b>${tLogo(sp, m.p.t)}${esc(m.p.n)} · ${esc(STAT[m.k] || m.k)}</b><span class="small">${m.worst > 0 ? 'arb · ' : ''}middle ${pct(m.both)} · EV <b class="${m.ev >= 0 ? 'pos' : 'neg'}">${sgn(m.ev, 2)}</b></span></div>
        <div class="kv"><span>Over ${m.o.L} <b class="num">${amS(m.o.o)}</b> ${esc(poBk(m.o.b))}</span><b class="num">${m.stake[0]}</b></div>
        <div class="kv"><span>Under ${m.u.L} <b class="num">${amS(m.u.u)}</b> ${esc(poBk(m.u.b))}</span><b class="num">${m.stake[1]}</b></div>
        <p class="small muted">Worst case ${sgn(m.worst, 2)}, best ${sgn(m.best, 1)} of the total staked · ${esc(m.p.t)} ${m.p.h ? 'vs' : '@'} ${esc(m.p.o)}</p></div>`).join('')
      : `<div class="empty">No ${sp} prop middles with positive EV across your books right now.</div>`);
  out.querySelectorAll('.mkt[data-i]').forEach(b => b.onclick = () => { const m = show[+b.dataset.i]; poOpen(m.p, m.k, m.o); });
  return true;
}
// line flags for grading (models/grading/snap_flags.js): every sport and day, our chance, blended chance and price
window.BD_FLAGS = () => ({made:new Date().toISOString(), rows:PO_SP.flatMap(sp => pfAll(sp, true).rows.map(x => ({sp, n:x.p.n, t:x.p.t, o:x.p.o, g:x.p.g, d:x.p.d || '', k:x.k, side:x.side,
  line:x.L, book:x.b, price:x.pr, proj:x.p.s[x.k].m, ours:+x.ch.toFixed(4), blend:+x.chb.toFixed(4), market:x.mk == null ? null : +x.mk.toFixed(4), ev:+x.ev.toFixed(4), evb:+x.evb.toFixed(4), kind:x.kind, kick:(poFind(sp, x.p) || {g:{}}).g.k || null, sweep:PO[sp].made})))});
