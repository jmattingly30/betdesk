// ---------- Line shop (J 2026-10-27: OddsJam-style book shopping, middles, arbs, low holds, bonus hedges, line moves) ----------
// line_tools.json (app/line_tools.py, daily right after the odds pull) gives every book's line per game, the fair price
// (Pinnacle, else BetOnline/LowVig, else all books) and our model's expected score; landing.json says how often games land
// on each margin and total. Everything below is recomputed here for the books ticked in Settings > My books, with the same
// math as line_tools.py (spec: app/line-tools.md). Prices are from the morning pull: check the book before betting.
let LT = null, LAND = null, SHOP_TAB = lsGet('betdesk.shoptab', 'ev'), SHOP_MODE = lsGet('betdesk.shopmode', 'fair'), SHOP_BONUS = lsGet('betdesk.shopbonus', '');
const LT_SP = ['NFL','CFB','NBA','CBB'], LT_MOVE = {NFL:1, CFB:1.5, NBA:1, CBB:1}, LT_MIN = -0.005;
const bkName = b => (LT && LT.books[b]) || b;
const amS = a => a == null || !Number.isFinite(+a) ? '–' : (a > 0 ? '+' : '') + Math.round(a);
const amOf = d => d >= 2 ? Math.round((d - 1) * 100) : Math.round(-100 / (d - 1));
const ltBooksAll = () => LT ? Object.keys(LT.books) : [];
// default books (J 2026-10-27: Colorado, any legal book there): line_tools.json my_books; Pinnacle and BetOnline are fair-price references only
const booksOff = () => new Set(Array.isArray(S.settings.booksOff) ? S.settings.booksOff
  : LT && Array.isArray(LT.my_books) ? ltBooksAll().filter(b => !LT.my_books.includes(b)) : ['pinnacle']);
const myBook = b => !booksOff().has(b);
function landPmf(sp, kind, mu){
  const L = LAND && LAND[sp] && LAND[sp][kind]; if (!L || mu == null) return null;
  const sd = Math.sqrt(Math.max(L.sd[0] + L.sd[1]*Math.abs(mu), 25)), out = new Map(); let s = 0;
  for (let k = Math.trunc(mu - 6*sd); k <= Math.trunc(mu + 6*sd) + 1; k++){
    const i = (kind === 'm' ? k*(mu < 0 ? -1 : 1) : k) - L.lo, m = i >= 0 && i < L.mult.length ? L.mult[i] : 1;
    const p = (ncdf((k + .5 - mu)/sd) - ncdf((k - .5 - mu)/sd))*m; out.set(k, p); s += p; }
  out.forEach((p, k) => out.set(k, p/s)); return out;
}
// win, push chance: H home margin + a > 0; A a - margin > 0 (a = away point); O total > a; U total < a
const ltRes = (s, a, k) => { const x = {H:k + a, A:a - k, O:k - a, U:a - k}[s]; return x > 0 ? 1 : x === 0 ? 0 : -1; };
function ltWP(P, a, s){ let w = 0, p = 0; P.forEach((q, k) => { const r = ltRes(s, a, k); if (r > 0) w += q; else if (r === 0) p += q; }); return [w, p]; }
function midCalc(o1, o2, P){
  const [s1, a1, p1] = o1, [s2, a2, p2] = o2, d1 = dec(p1), d2 = dec(p2), x1 = 100*d2/(d1 + d2), x2 = 100 - x1;
  let ev = 0, both = 0, lo = Infinity, hi = -Infinity;
  P.forEach((q, k) => { const r1 = ltRes(s1, a1, k), r2 = ltRes(s2, a2, k);
    const g = (r1 > 0 ? x1*(d1 - 1) : r1 === 0 ? 0 : -x1) + (r2 > 0 ? x2*(d2 - 1) : r2 === 0 ? 0 : -x2);
    ev += q*g; lo = Math.min(lo, g); hi = Math.max(hi, g); if (r1 > 0 && r2 > 0) both += q; });
  return {both, ev:ev/100, worst:lo/100, best:hi/100, stake:[+x1.toFixed(1), +x2.toFixed(1)]};
}
const lgt = p => { p = Math.min(Math.max(p, 1e-6), 1 - 1e-6); return Math.log(p/(1 - p)); };
const ltSides = r => { const o = [];
  if (r[0] != null){ if (r[1]) o.push(['spread','H',r[0],r[1]]); if (r[2]) o.push(['spread','A',-r[0],r[2]]); }
  if (r[3] != null){ if (r[4]) o.push(['total','O',r[3],r[4]]); if (r[5]) o.push(['total','U',r[3],r[5]]); }
  if (r[6]) o.push(['ml','H',null,r[6]]); if (r[7]) o.push(['ml','A',null,r[7]]); return o; };
const sideLbl = (g, mk, s, pt) => mk === 'total' ? `${s === 'O' ? 'Over' : 'Under'} ${pt}` : `${s === 'H' ? g.h : g.a}${mk === 'ml' ? ' ML' : ' ' + sgnPts(pt)}`;
// every list for one game at the ticked books (cached until the books or the model weight change)
const LTC = new Map();
function ltGame(sp, g){
  const key = sp + '|' + g.id + '|' + [...booksOff()].sort().join(',') + '|' + S.settings.trustTeam;
  if (LTC.has(key)) return LTC.get(key);
  const f = g.fair || {}, md = g.model, t = (S.settings.trustTeam ?? 30)/100;
  const Pm = landPmf(sp, 'm', f.mu), Pt = landPmf(sp, 't', f.mt), MPm = md ? landPmf(sp, 'm', md.mu) : null, MPt = md ? landPmf(sp, 't', md.mt) : null;
  const ch = (mk, s, pt, P, pml) => mk === 'ml' ? (pml == null ? null : [s === 'H' ? pml : 1 - pml, 0]) : P ? ltWP(P, pt, s) : null;
  const rows = [];
  Object.entries(g.bk || {}).forEach(([b, r]) => { if (!r || !myBook(b)) return;
    ltSides(r).forEach(([mk, s, pt, pr]) => {
      const c = ch(mk, s, pt, mk === 'spread' ? Pm : Pt, f.pml); if (!c) return;
      const [w, p] = c, l = 1 - w - p, x = {b, mk, s, pt, pr, w, p, ev:w*dec(pr) + p - 1, fair:w > 0 && l > 0 ? amOf((w + l)/w) : null};
      const cm = md ? ch(mk, s, pt, mk === 'spread' ? MPm : MPt, md.pml) : null;
      if (cm){ const [wm, pmu] = cm, lm = 1 - wm - pmu; x.mev = wm*dec(pr) + pmu - 1; x.wm = wm/Math.max(wm + lm, 1e-9);
        const nb = 1/(1 + Math.exp(-(t*lgt(wm/Math.max(wm + lm, 1e-9)) + (1 - t)*lgt(w/Math.max(w + l, 1e-9))))); x.bev = nb*(1 - p)*dec(pr) + p - 1; }
      rows.push(x); }); });
  const best = new Map(); rows.forEach(x => { const k = x.mk + x.s + x.pt; if (!best.has(k) || dec(x.pr) > dec(best.get(k).pr)) best.set(k, x); });
  const hold = []; best.forEach(x => { if (x.s !== 'H' && x.s !== 'O') return;
    const o = best.get(x.mk + (x.s === 'H' ? 'A' : 'U') + (x.mk === 'spread' ? -x.pt : x.pt)); if (o) hold.push({mk:x.mk, pt:x.pt, h:1/dec(x.pr) + 1/dec(o.pr) - 1, a:x, b:o}); });
  const mid = [];
  [['spread','H','A',Pm], ['total','O','U',Pt], ['ml','H','A',null]].forEach(([mk, S1, S2, P]) => {
    const A = rows.filter(x => x.mk === mk && x.s === S1), B = rows.filter(x => x.mk === mk && x.s === S2);
    A.forEach(x => B.forEach(y => { if (x.b === y.b) return;
      if (mk === 'ml'){ const s = 1/dec(x.pr) + 1/dec(y.pr); if (s < 1) mid.push({mk, kind:'arb', a:x, b:y, both:0, ev:1/s - 1, worst:1/s - 1, stake:[+(100/dec(x.pr)/s).toFixed(1), +(100/dec(y.pr)/s).toFixed(1)]}); return; }
      const gap = mk === 'spread' ? x.pt + y.pt : y.pt - x.pt; if (gap < 0 || !P) return;
      const m = midCalc([S1, x.pt, x.pr], [S2, y.pt, y.pr], P), kind = m.worst >= 0 ? 'arb' : m.both > 0 ? 'middle' : gap > 0 ? 'push' : 'same';
      if ((kind === 'same' || kind === 'push') && m.ev < LT_MIN) return;
      if (kind === 'middle' && m.ev < LT_MIN && m.both < 0.08) return;
      mid.push({mk, kind, gap, a:x, b:y, ...m}); })); });
  // a leg whose point sits far from the other books' median is more often a stale or mislabelled price than a gift (line_tools.py `check`)
  const far = {NFL:2, CFB:2.5, NBA:3, CBB:3}[sp] ?? 3, med = c => { const v = Object.values(g.bk || {}).map(r => r && r[c]).filter(x => x != null).sort((p, q) => p - q); return v.length >= 3 ? (v.length % 2 ? v[v.length >> 1] : (v[v.length/2 - 1] + v[v.length/2])/2) : null; };
  const mdS = med(0), mdT = med(3);
  mid.forEach(z => { if (z.mk === 'ml'){ if (z.kind === 'arb') z.check = 1; return; } const md = z.mk === 'spread' ? mdS : mdT; if (md == null) return;
    if ([z.a, z.b].some(x => Math.abs((z.mk === 'total' || x.s === 'H' ? x.pt : -x.pt) - md) >= far)) z.check = 1; });
  mid.sort((p, q) => (p.check || 0) - (q.check || 0) || q.ev - p.ev);
  const conv = []; const ml = rows.filter(x => x.mk === 'ml');
  ml.forEach(x => ml.forEach(y => { if (y.s === x.s || y.b === x.b) return; const dA = dec(x.pr), dB = dec(y.pr);
    conv.push({bonus:x, cash:y, conv:(dA - 1)*(1 - 1/dB), hedge:(dA - 1)/dB}); }));
  conv.sort((p, q) => q.conv - p.conv);
  const out = {rows, best:[...best.values()], hold:hold.sort((p, q) => p.h - q.h), mid, conv};
  LTC.set(key, out); return out;
}
const ltStarted = g => g.k && Date.parse(g.k) < Date.now();
function ltList(sp){
  let gs = ((LT && LT.sports[sp]) || []).filter(g => !ltStarted(g));
  if (sp === 'NFL'){ const w = nflWk(), wk = new Map(gsGames('NFL').map(x => [x.id, +x.w])); if (w != null) gs = gs.filter(g => !wk.has(g.id) || wk.get(g.id) === w); }
  else if (S.day && sp !== 'CFB') gs = gs.filter(g => !g.d || g.d === S.day);   // CFB: the whole week
  return gs;
}
// opens the game card for a line-shop row
function ltOpen(sp, g){
  if (sp === 'NFL'){ const x = gsGames('NFL').find(y => y.id === g.id); if (x && +x.w !== nflWk()) NFL_W = +x.w; }
  if (sp !== 'NFL' && S.day && g.d && g.d !== S.day) S.day = g.d;
  S.sub[sp] = 'games'; lsSet('betdesk.sub', JSON.stringify(S.sub)); refreshSport(false);
  const r = [...document.querySelectorAll('#gameList .grow')].find(b => b.innerText.split('\n')[0].trim() === `${g.a} @ ${g.h}`);
  if (r) r.click(); else toast('That game is not on the slate any more');
}
const LT_TABS = [['ev','+EV'], ['mid','Middles'], ['arb','Arbs'], ['hold','Low hold'], ['conv','Bonus hedge'], ['move','Line moves']];
function renderShop(){
  const out = $('#shopOut'); if (!out) return;
  const sp = S.sport;
  if (!LT_SP.includes(sp)){ out.innerHTML = ''; return; }
  if (!LT || !LAND){ out.innerHTML = '<p class="small muted">Book lines are not loaded yet.</p>'; return; }
  const gs = ltList(sp), tab = LT_TABS.some(t => t[0] === SHOP_TAB) ? SHOP_TAB : 'ev';
  const tabs = `<div class="chips" role="group" aria-label="Tool">${LT_TABS.map(([k, l]) => `<button type="button" data-t="${k}" aria-pressed="${k === tab}">${l}</button>`).join('')}</div>`;
  const nb = ltBooksAll().filter(myBook).length;
  const head = `<p class="small muted">Lines from ${esc(LT.lines_from || LT.made)} at your ${nb} books (Settings &gt; My books). Fair price from ${esc((gs[0] && gs[0].fair && gs[0].fair.src) || 'Pinnacle')}. Prices move: check the book before betting. Tap a row to open the game.</p>`;
  const G = gs.map(g => [g, ltGame(sp, g)]), rowsH = [];
  const gm = g => `<span class="small muted">${esc(g.a)} @ ${esc(g.h)}${g.d ? ' · ' + esc(dayName(g.d)) : ''}</span>`;
  const leg = x => `${esc(sideLbl(x.g, x.mk, x.s, x.pt))} <b class="num">${amS(x.pr)}</b> ${esc(bkName(x.b))}`;
  let body = '';
  if (tab === 'ev'){
    const k = SHOP_MODE === 'blend' ? 'bev' : 'ev';
    const L = G.flatMap(([g, t]) => t.best.filter(x => (x[k] ?? -1) > 0).map(x => ({...x, g}))).sort((p, q) => q[k] - p[k]);
    body = `<div class="chips seg-mode" role="group" aria-label="EV against"><button type="button" data-m="fair" aria-pressed="${k === 'ev'}">vs fair price</button><button type="button" data-m="blend" aria-pressed="${k === 'bev'}">vs blended model (${S.settings.trustTeam}%)</button></div>`
      + (L.length ? `<div class="tablewrap"><table class="tight"><thead><tr><th>Bet</th><th>Book</th><th>Price</th><th>Fair</th><th>EV</th><th>Ours</th><th>Blend</th><th></th></tr></thead><tbody>${L.slice(0, 60).map((x, i) => { rowsH.push(x.g);
          return `<tr data-g="${rowsH.length - 1}" class="click"><td class="t">${esc(sideLbl(x.g, x.mk, x.s, x.pt))}<br>${gm(x.g)}</td><td>${esc(bkName(x.b))}</td><td class="num"><b>${amS(x.pr)}</b></td><td class="num">${amS(x.fair)}</td><td class="num ${x.ev > 0 ? 'pos' : 'neg'}">${sgn(x.ev)}</td><td class="num">${sgn(x.mev)}</td><td class="num">${sgn(x.bev)}</td><td><button class="btn ghost sm" type="button" data-log="${i}">Log</button></td></tr>`; }).join('')}</tbody></table></div>`
        : `<p class="small muted">No price at your books beats the ${k === 'ev' ? 'fair price' : 'blended model'} right now.</p>`);
    out.innerHTML = head + tabs + body; wireShop(out, sp, rowsH);
    out.querySelectorAll('[data-m]').forEach(b => b.onclick = e => { e.stopPropagation(); SHOP_MODE = b.dataset.m; lsSet('betdesk.shopmode', SHOP_MODE); renderShop(); });
    out.querySelectorAll('[data-log]').forEach(b => b.onclick = e => { e.stopPropagation(); const x = L[+b.dataset.log];
      openBet({sport:sp, market:{ml:'Moneyline', spread:'Spread', total:'Total'}[x.mk], model:'Own read', desc:sideLbl(x.g, x.mk, x.s, x.pt),
        side:x.mk === 'total' ? (x.s === 'O' ? 'Over' : 'Under') : 'Side', line:x.pt ?? '', odds:x.pr, push:x.p || 0, units:'', book:bkName(x.b),
        pm:x.wm ?? '', pf:x.w/((1 - x.p) || 1), pb:x.w/((1 - x.p) || 1), gm:{h:x.g.h, a:x.g.a}, why:'Stale book (others moved)'}); });
    return;
  }
  if (tab === 'mid' || tab === 'arb'){
    const L = G.flatMap(([g, t]) => t.mid.filter(m => tab === 'arb' ? m.kind === 'arb' : m.kind === 'middle' || m.kind === 'push').map(m => ({...m, g, a:{...m.a, g}, b:{...m.b, g}})));
    L.sort((p, q) => (p.check || 0) - (q.check || 0) || q.ev - p.ev);
    body = (tab === 'arb' ? '<p class="small"><b>Arbs are rare and short-lived.</b> These prices are from the morning pull: check both books before betting, and expect one side to have moved.</p>' : '<p class="small muted">Bet both sides at two books. Middle chance = both bets win; a push window means one side pushes and the other wins. Stakes per 100 give the same return either way.</p>')
      + (L.length ? L.slice(0, 40).map(m => { rowsH.push(m.g);
          return `<div class="mkt click" data-g="${rowsH.length - 1}" style="margin:8px 0"><div class="mh"><b>${esc(m.g.a)} @ ${esc(m.g.h)}</b><span class="small">${m.check ? '<b class="neg">verify at the book</b> · ' : ''}${m.kind === 'arb' ? 'arb' : m.kind === 'push' ? 'push window' : 'middle ' + pct(m.both)} · EV <b class="${m.ev >= 0 ? 'pos' : 'neg'}">${sgn(m.ev, 2)}</b></span></div>
            <div class="kv"><span>${leg(m.a)}</span><b class="num">${m.stake[0]}</b></div><div class="kv"><span>${leg(m.b)}</span><b class="num">${m.stake[1]}</b></div>
            <p class="small muted">${m.kind === 'arb' ? 'Every result returns at least ' + sgn(m.worst, 2) : `Worst case ${sgn(m.worst, 2)}, best ${sgn(m.best, 1)} of the total staked`}${m.g.d ? ' · ' + esc(dayName(m.g.d)) : ''}</p></div>`; }).join('')
        : `<p class="small muted">No ${tab === 'arb' ? 'arbs' : 'middles'} across your books right now.</p>`);
  }
  else if (tab === 'hold'){
    const L = G.flatMap(([g, t]) => t.hold.filter(h => h.h < 0.025).map(h => ({...h, g}))).sort((p, q) => p.h - q.h);
    body = '<p class="small muted">Best price on each side at your books, where the combined hold is under 2.5% (a normal line holds about 4.5%).</p>'
      + (L.length ? `<div class="tablewrap"><table class="tight"><thead><tr><th>Game</th><th>Side 1</th><th>Side 2</th><th>Hold</th></tr></thead><tbody>${L.slice(0, 60).map(h => { rowsH.push(h.g);
          return `<tr class="click" data-g="${rowsH.length - 1}"><td class="t">${esc(h.g.a)} @ ${esc(h.g.h)}</td><td>${leg({...h.a, g:h.g})}</td><td>${leg({...h.b, g:h.g})}</td><td class="num ${h.h < 0 ? 'pos' : ''}">${pct(h.h, 2)}</td></tr>`; }).join('')}</tbody></table></div>`
        : '<p class="small muted">No market at your books holds under 2.5% right now.</p>');
  }
  else if (tab === 'conv'){
    const mine = ltBooksAll().filter(myBook), bon = mine.includes(SHOP_BONUS) ? SHOP_BONUS : '';
    const L = G.flatMap(([g, t]) => t.conv.filter(c => !bon || c.bonus.b === bon).slice(0, 3).map(c => ({...c, g}))).sort((p, q) => q.conv - p.conv);
    body = `<label class="small">Bonus bet at<select id="shopBonus"><option value="">any of my books</option>${mine.map(b => `<option value="${b}"${b === bon ? ' selected' : ''}>${esc(bkName(b))}</option>`).join('')}</select></label>
      <p class="small muted">Bet the bonus on the long side, hedge the other side with cash at another book. Kept = the share of the bonus you lock in whoever wins. Moneylines only.</p>`
      + (L.length ? `<div class="tablewrap"><table class="tight"><thead><tr><th>Bonus bet</th><th>Cash hedge</th><th>Kept</th><th>Hedge per $1</th></tr></thead><tbody>${L.slice(0, 40).map(c => { rowsH.push(c.g);
          return `<tr class="click" data-g="${rowsH.length - 1}"><td class="t">${leg({...c.bonus, g:c.g})}<br>${gm(c.g)}</td><td>${leg({...c.cash, g:c.g})}</td><td class="num"><b>${pct(c.conv)}</b></td><td class="num">$${c.hedge.toFixed(2)}</td></tr>`; }).join('')}</tbody></table></div>`
        : '<p class="small muted">No moneyline hedges at your books.</p>');
  }
  else if (tab === 'move'){
    const L = gs.filter(g => (g.move && g.move.flag) || Math.abs((g.fair && g.fair.sharp_vs_books) || 0) >= 0.5)
      .sort((p, q) => Math.abs((q.move && q.move.dmu) || 0) - Math.abs((p.move && p.move.dmu) || 0));
    body = `<p class="small muted">Where the sharp book's number has moved since the first pull (${LT_MOVE[sp]}+ pts on the margin, ${LT_MOVE[sp]*1.5}+ on the total), or sits 0.5+ pts away from the other books. A move toward a team is usually sharp money on it. One pull a day, so moves inside a day are not seen.</p>`
      + (L.length ? `<div class="tablewrap"><table class="tight"><thead><tr><th>Game</th><th>Margin move</th><th>Total move</th><th>Sharp vs books</th></tr></thead><tbody>${L.map(g => { rowsH.push(g);
          const mv = g.move || {}, sv = g.fair && g.fair.sharp_vs_books, tw = (d, h, a) => d == null || Math.abs(d) < 0.05 ? '–' : `${Math.abs(d).toFixed(1)} toward ${esc(d > 0 ? h : a)}`;
          return `<tr class="click" data-g="${rowsH.length - 1}"><td class="t">${esc(g.a)} @ ${esc(g.h)}${mv.since ? `<br><span class="small muted">since ${esc(String(mv.since).slice(0, 10))}</span>` : ''}</td><td>${tw(mv.dmu, g.h, g.a)}</td><td>${mv.dmt == null || Math.abs(mv.dmt) < 0.05 ? '–' : (mv.dmt > 0 ? 'up ' : 'down ') + Math.abs(mv.dmt).toFixed(1)}</td><td>${tw(sv, g.h, g.a)}</td></tr>`; }).join('')}</tbody></table></div>`
        : '<p class="small muted">No big sharp moves or splits right now. Moves build up as each morning pull is saved.</p>');
  }
  out.innerHTML = head + tabs + body; wireShop(out, sp, rowsH);
  const sb = $('#shopBonus'); if (sb) sb.onchange = () => { SHOP_BONUS = sb.value; lsSet('betdesk.shopbonus', SHOP_BONUS); renderShop(); };
}
function wireShop(out, sp, rowsH){
  out.querySelectorAll('[data-t]').forEach(b => b.onclick = () => { SHOP_TAB = b.dataset.t; lsSet('betdesk.shoptab', SHOP_TAB); renderShop(); });
  out.querySelectorAll('[data-g]').forEach(r => r.onclick = () => ltOpen(sp, rowsH[+r.dataset.g]));
}
const shopCount = () => { if (!LT || !LAND || !LT_SP.includes(S.sport)) return 0;
  return ltList(S.sport).reduce((n, g) => n + ltGame(S.sport, g).mid.filter(m => m.kind !== 'same' && m.ev > 0).length, 0); };
// the game card's "Every book" table: each book's spread, total and moneyline, best price at your books in bold
function everyBook(sp){
  if (!LT || !LT_SP.includes(sp) || !GCFG[sp]) return '';
  const h = $(GCFG[sp].ids[0]).value, a = $(GCFG[sp].ids[1]).value, g = ((LT.sports[sp]) || []).find(x => x.h === h && x.a === a);
  if (!g || !g.bk) return '';
  const bk = Object.entries(g.bk).filter(([, r]) => r), f = g.fair || {};
  const bestOf = (i, ok) => { let v = null; bk.forEach(([b, r]) => { if (myBook(b) && r[i] && ok(r) && (v == null || dec(r[i]) > dec(v))) v = r[i]; }); return v; };
  const mode = (i) => { const c = {}; bk.forEach(([, r]) => { if (r[i] != null) c[r[i]] = (c[r[i]] || 0) + 1; }); return +Object.entries(c).sort((p, q) => q[1] - p[1])[0]?.[0]; };
  const hp = mode(0), tp = mode(3);
  const bH = bestOf(1, r => r[0] === hp), bA = bestOf(2, r => r[0] === hp), bO = bestOf(4, r => r[3] === tp), bU = bestOf(5, r => r[3] === tp), bmH = bestOf(6, () => true), bmA = bestOf(7, () => true);
  const cell = (pt, pr, best) => pr ? `${pt != null ? `${pt} ` : ''}<span class="num">${best != null && pr === best ? `<b>${amS(pr)}</b>` : amS(pr)}</span>` : '–';
  return `<details class="card" style="margin-top:10px"><summary><h3 style="display:inline">Every book</h3> <span class="small muted">${bk.length} books</span></summary>
    <div class="tablewrap"><table class="tight"><thead><tr><th>Book</th><th>${esc(g.h)}</th><th>${esc(g.a)}</th><th>Over</th><th>Under</th><th>${esc(g.h)} ML</th><th>${esc(g.a)} ML</th></tr></thead><tbody>
    <tr><td class="t"><b>Fair</b> <span class="small muted">${esc(f.src || '')}</span></td><td>${f.L != null ? sgnPts(f.L) : '–'}</td><td>${f.L != null ? sgnPts(-f.L) : '–'}</td><td>${f.T ?? '–'}</td><td>${f.T ?? '–'}</td><td class="num">${f.pml != null ? am(f.pml) : '–'}</td><td class="num">${f.pml != null ? am(1 - f.pml) : '–'}</td></tr>
    ${bk.map(([b, r]) => `<tr${myBook(b) ? '' : ' class="muted"'}><td class="t">${esc(bkName(b))}${myBook(b) ? '' : ' <span class="small">(not yours)</span>'}</td><td>${cell(r[0] != null ? sgnPts(r[0]) : null, r[1], r[0] === hp ? bH : null)}</td><td>${cell(r[0] != null ? sgnPts(-r[0]) : null, r[2], r[0] === hp ? bA : null)}</td><td>${cell(r[3], r[4], r[3] === tp ? bO : null)}</td><td>${cell(r[3], r[5], r[3] === tp ? bU : null)}</td><td>${cell(null, r[6], bmH)}</td><td>${cell(null, r[7], bmA)}</td></tr>`).join('')}
    </tbody></table></div><p class="small muted">Bold = best price at your books on the most common number. Morning pull (${esc(LT.lines_from || LT.made)}).</p></details>`;
}
// Settings > My books: every book on by default except Pinnacle (the fair-price reference)
function renderMyBooks(){
  const el = $('#myBooks'); if (!el) return;
  if (!LT){ el.innerHTML = '<p class="small muted">Book list loads with the line data.</p>'; return; }
  const off = booksOff();
  el.innerHTML = `<div class="chips">${ltBooksAll().map(b => `<label class="chk"><input type="checkbox" data-b="${b}"${off.has(b) ? '' : ' checked'}> ${esc(bkName(b))}</label>`).join('')}</div>`;
  el.querySelectorAll('input').forEach(i => i.onchange = async () => {
    const o = [...el.querySelectorAll('input')].filter(x => !x.checked).map(x => x.dataset.b);
    S.settings = {...S.settings, booksOff:o}; LTC.clear();
    try { if (S.db) await S.db.doc('settings/main').set(S.settings); else lsSave(); toast('Books saved'); } catch(e){ toast('Could not save books'); }
    renderShop(); renderSubBar(); renderMidWatch(); });
}
// Bets > Middle watch: for each pending spread or total bet, the best opposite price at your books, priced on the landing curve
function betGame(b){
  const sp = b.sport; if (!LT || !LT_SP.includes(sp)) return null;
  const gs = (LT.sports[sp] || []).filter(g => !ltStarted(g) && (!b.date || !g.d || g.d >= b.date));
  if (b.gm) return gs.find(g => g.h === b.gm.h && g.a === b.gm.a) || null;
  if (b.market !== 'Spread') return null;
  const t = String(b.desc || '').replace(/\s*[+-]?\d+(\.\d+)?\s*$/, '').trim();
  return gs.find(g => g.h === t || g.a === t) || null;
}
function midWatchRows(){
  if (!LT || !LAND) return [];
  return S.bets.filter(b => b.result === 'pending' && (b.market === 'Spread' || b.market === 'Total') && odds(b.odds) != null && num(b.line) != null).map(b => {
    const g = betGame(b); if (!g) return null;
    const sp = b.sport, ln = num(b.line), o1 = odds(b.odds); let s1, P;
    if (b.market === 'Total'){ s1 = b.side === 'Under' ? 'U' : 'O'; P = landPmf(sp, 't', g.fair && g.fair.mt); }
    else { const t = b.gm ? null : String(b.desc || '').replace(/\s*[+-]?\d+(\.\d+)?\s*$/, '').trim();
      const home = b.gm ? String(b.desc || '').startsWith(g.h) : t === g.h; s1 = home ? 'H' : 'A'; P = landPmf(sp, 'm', g.fair && g.fair.mu); }
    if (!P) return null;
    let best = null;
    Object.entries(g.bk || {}).forEach(([bk, r]) => { if (!r || !myBook(bk)) return;
      const opp = s1 === 'H' ? (r[0] != null && r[2] ? ['A', -r[0], r[2]] : null) : s1 === 'A' ? (r[0] != null && r[1] ? ['H', r[0], r[1]] : null)
        : s1 === 'O' ? (r[3] != null && r[5] ? ['U', r[3], r[5]] : null) : (r[3] != null && r[4] ? ['O', r[3], r[4]] : null);
      if (!opp) return; const m = midCalc([s1, ln, o1], opp, P);
      if (!best || m.ev > best.m.ev) best = {bk, opp, m}; });
    return best ? {b, g, s1, ln, o1, ...best} : null;
  }).filter(Boolean);
}
function renderMidWatch(){
  const el = $('#midWatch'); if (!el) return;
  const R = midWatchRows(); if (!R.length){ el.hidden = true; el.innerHTML = ''; return; }
  const live = R.filter(r => r.m.both > 0 || r.m.ev > 0);
  el.hidden = false;
  el.innerHTML = `<h3>Middle watch</h3><p class="small muted">Watching ${R.length} pending spread and total bet${R.length > 1 ? 's' : ''} for the other side at your books (morning lines).</p>`
    + (live.length ? live.map(r => { const u = num(r.b.units), hedge = u ? u*dec(r.o1)/dec(r.opp[2]) : null, oppL = sideLbl(r.g, r.b.market === 'Total' ? 'total' : 'spread', r.opp[0], r.opp[1]);
        return `<div class="mkt" style="margin:8px 0"><div class="mh"><b>${esc(r.b.desc)} ${amS(r.o1)}</b><span class="small">${r.m.both > 0 ? 'middle ' + pct(r.m.both) : 'push window'} · EV <b class="${r.m.ev >= 0 ? 'pos' : 'neg'}">${sgn(r.m.ev, 2)}</b></span></div>
          <div class="kv"><span>Other side: ${esc(oppL)} <b class="num">${amS(r.opp[2])}</b> ${esc(bkName(r.bk))}</span><b class="num">${hedge ? hedge.toFixed(2) + 'u' : ''}</b></div>
          <p class="small muted">${hedge ? `Bet ${hedge.toFixed(2)}u to return the same either way; ` : ''}worst case ${sgn(r.m.worst, 2)}, best ${sgn(r.m.best, 1)} of both stakes.</p></div>`; }).join('')
      : '<p class="small muted">No middle or +EV hedge open yet. This fills in when the other side moves past your number.</p>');
}
// Settings > Calculators: middle, arb and bonus hedge
function renderLtCalc(){
  const sp = $('#lcSp') && $('#lcSp').value, mk = $('#lcMk') && $('#lcMk').value; if (!sp) return;
  const a1 = num($('#lcL1').value), p1 = odds($('#lcP1').value), a2 = num($('#lcL2').value), p2 = odds($('#lcP2').value);
  $('#lcL1').previousSibling.textContent = mk === 'total' ? 'Over line' : 'Team 1 spread';
  $('#lcL2').previousSibling.textContent = mk === 'total' ? 'Under line' : 'Team 2 spread';
  let h = '';
  if (a1 != null && a2 != null && p1 != null && p2 != null && LAND){
    // team 1 = home side of the curve: its spread a1, team 2's spread a2; expected margin halfway between the two lines
    const P = mk === 'total' ? landPmf(sp, 't', (a1 + a2)/2) : landPmf(sp, 'm', (a2 - a1)/2);
    const m = P && midCalc(mk === 'total' ? ['O', a1, p1] : ['H', a1, p1], mk === 'total' ? ['U', a2, p2] : ['A', a2, p2], P);
    if (m) h = `<div class="kv"><span>Middle chance (both win)</span><b>${pct(m.both)}</b></div><div class="kv"><span>EV on the total staked</span><b class="${m.ev >= 0 ? 'pos' : 'neg'}">${sgn(m.ev, 2)}</b></div>
      <div class="kv"><span>Stakes per 100</span><b>${m.stake[0]} / ${m.stake[1]}</b></div><div class="kv"><span>Worst · best case</span><b>${sgn(m.worst, 2)} · ${sgn(m.best, 1)}</b></div>`;
  }
  $('#lcMid').innerHTML = h;
  const x1 = odds($('#lcA1').value), x2 = odds($('#lcA2').value);
  if (x1 != null && x2 != null){ const s = 1/dec(x1) + 1/dec(x2);
    $('#lcArb').innerHTML = `<div class="kv"><span>${s < 1 ? 'Arb: locked return' : 'No arb: hold'}</span><b class="${s < 1 ? 'pos' : ''}">${s < 1 ? sgn(1/s - 1, 2) : pct(s - 1, 2)}</b></div><div class="kv"><span>Stakes per 100</span><b>${(100/dec(x1)/s).toFixed(1)} / ${(100/dec(x2)/s).toFixed(1)}</b></div>`; }
  else $('#lcArb').innerHTML = '';
  const bp = odds($('#lcBp').value), hp = odds($('#lcHp').value), ba = num($('#lcBa').value) ?? 100;
  if (bp != null && hp != null){ const dA = dec(bp), dB = dec(hp), c = (dA - 1)*(1 - 1/dB);
    $('#lcHedge').innerHTML = `<div class="kv"><span>Kept (conversion)</span><b>${pct(c)}</b></div><div class="kv"><span>Hedge with cash</span><b>${(ba*(dA - 1)/dB).toFixed(2)}</b></div><div class="kv"><span>Locked profit</span><b>${(ba*c).toFixed(2)}</b></div>`; }
  else $('#lcHedge').innerHTML = '';
}
function initLtCalc(){
  if (!$('#lcSp')) return;
  ['#lcSp','#lcMk','#lcL1','#lcP1','#lcL2','#lcP2','#lcA1','#lcA2','#lcBp','#lcHp','#lcBa'].forEach(q => $(q).addEventListener('input', renderLtCalc));
  addSignButtons(['#lcP1','#lcP2','#lcA1','#lcA2','#lcBp','#lcHp'].map(x => $(x))); renderLtCalc();
}
