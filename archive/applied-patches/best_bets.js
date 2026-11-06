// ---------- Best bets (J 2026-10-28 rework): every bet worth making at your books, across sports, in one list ----------
// Sources, all at your ticked books (Settings > My books): line-based prop flags (pfAll), game sides where our model beats
// the price on our model alone and blended with the line (ltGame mev/bev, same EV minimum as prop flags), prices that beat
// the sharp fair price (ltGame ev, 2%+), game and prop middles, arbs, and promo picks. Check rows go last.
let BB_SP = lsGet('betdesk.bbsp', 'All'), BB_KIND = lsGet('betdesk.bbkind', 'all'), BB_SORT = lsGet('betdesk.bbsort', 'time'), BB_ALL = false;
const BB_KINDS = [['all', 'All'], ['props', 'Props'], ['games', 'Games'], ['mid', 'Middles'], ['promo', 'Promo']];
const BB_TAG = {...PF_TAG, game:'<span class="tag t-flag">Game</span>', price:'<span class="tag t-flag">Beats fair</span>', mid:'<span class="tag t-flag">Middle</span>',
  arb:'<span class="tag t-check">Arb</span>', verify:'<span class="tag t-check">Verify</span>', promo:'<span class="tag t-proven">Promo</span>'};
const BB_RANK = {proven:0, flag:1, game:1, price:2, mid:3, arb:3, promo:4, verify:5, check:6};
function bbGo(sp){ if (S.sport !== sp) setSport(sp); show('slate'); }
function bbRows(){
  const R = [], now = Date.now(), live = k => !k || Date.parse(k) > now;
  PO_SP.forEach(sp => { if (!PO[sp] || !S.slates[sp]) return;
    pfAll(sp, true).rows.forEach(x => { const f = poFind(sp, x.p), k = f && f.g.k; if (!live(k)) return;
      R.push({type:'props', sp, kind:x.kind, k, ev:x.kind === 'proven' ? x.ev : x.evb, title:`${x.p.n} ${x.side.toLowerCase()} ${x.L} ${STAT[x.k] || x.k}`,
        sub:`${poBk(x.b)} ${amS(x.pr)}`, note:x.kind === 'check' ? `we say ${x.p.s[x.k].m}` : x.kind === 'proven' ? `ours ${pct(x.ch)}` : `ours ${sgn(x.ev)}`,
        go:() => { bbGo(sp); poOpen(x.p, x.k, x.r); }}); });
    // prop middles: over a lower line at one of your books, under a higher line at another
    S.slates[sp].players.forEach(p => { if (!upcoming(p)) return;
      Object.entries(p.s).forEach(([k, s]) => { const rows = poRows(sp, p, k); if (rows.length < 2) return;
        // main lines only, both prices -200 or longer (an alt rung at -2500 is not a middle worth making)
        const ok = r => r.main && !r.odd && r.o >= -200 && r.u >= -200, ov = rows.filter(ok), un = rows.filter(ok), f = poFind(sp, p), kk = f && f.g.k; if (!live(kk)) return;
        ov.forEach(o => un.forEach(u => { if (!(u.L > o.L) || o.b === u.b) return; const m = poMidOf(s, k, o, u); if (!(m.ev >= 0.01 && m.both > 0)) return;
          R.push({type:'mid', sp, kind:m.worst > 0 ? 'arb' : 'mid', k:kk, ev:m.ev, title:`${p.n} ${STAT[k] || k} middle ${o.L} / ${u.L}`,
            sub:`${poBk(o.b)} o${o.L} ${amS(o.o)} · ${poBk(u.b)} u${u.L} ${amS(u.u)}`, note:`${pct(m.both)} both`, go:() => { bbGo(sp); poOpen(p, k, o); }}); })); }); }); });
  if (LT && LAND) LT_SP.forEach(sp => ((LT.sports || {})[sp] || []).filter(g => live(g.k)).forEach(g => {
    const t = ltGame(sp, g), thr = PF_THR, go = () => { bbGo(sp); ltOpen(sp, g); }, gm = `${g.a} @ ${g.h}`;
    t.best.forEach(x => {
      if (x.mev != null && x.mev >= thr && x.bev >= thr) R.push({type:'games', sp, kind:'game', k:g.k, ev:x.bev, title:sideLbl(g, x.mk, x.s, x.pt), sub:`${gm} · ${bkName(x.b)} ${amS(x.pr)}`, note:`ours ${sgn(x.mev)}`, go});
      else if (x.ev >= 0.02) R.push({type:'games', sp, kind:'price', k:g.k, ev:x.ev, title:sideLbl(g, x.mk, x.s, x.pt), sub:`${gm} · ${bkName(x.b)} ${amS(x.pr)}`, note:`fair ${amS(x.fair)}`, go}); });
    t.mid.filter(m => (m.kind === 'middle' || m.kind === 'push' || m.kind === 'arb') && m.ev > 0).forEach(m => R.push({type:'mid', sp, kind:m.check ? 'verify' : m.kind === 'arb' ? 'arb' : 'mid', k:g.k, ev:m.ev,
      title:`${gm} ${m.kind === 'arb' ? 'arb' : m.mk === 'total' ? 'total middle' : 'middle'}`, sub:`${bkName(m.a.b)} ${sideLbl(g, m.a.mk, m.a.s, m.a.pt)} ${amS(m.a.pr)} · ${bkName(m.b.b)} ${sideLbl(g, m.b.mk, m.b.s, m.b.pt)} ${amS(m.b.pr)}`,
      note:m.kind === 'arb' ? 'arb' : `${pct(m.both)} both`, go})); }));
  if (PROMO && PROMO.days){ const today = localDay();
    PROMO_GROUPS.flatMap(gk => Object.keys(PROMO.days).filter(d => d >= today).flatMap(d => PROMO.days[d][gk] || [])
      .filter(c => c.priced && !c.legs.some(l => !live(l.k))).sort((a, b) => b.conversion - a.conversion).slice(0, 3))   // the top 3 of each kind
      .forEach(c => { const sp = (c.sport || [])[0] || 'Other';
        R.push({type:'promo', sp:(c.sport || []).length === 1 ? sp : 'Mixed', kind:'promo', k:c.legs.map(l => l.k).filter(Boolean).sort()[0], ev:c.conversion,
          title:c.legs.map(l => l.pick).join(' + '), sub:`Bonus bet · ${c.book || ''} ${c.odds}`, note:`converts ${pct(c.conversion)}`,
          go:() => { if (SPORT_LIST.includes(sp)) { bbGo(sp); PROMO_BAND = c.band; showSub('promo'); renderPromo(); } }}); }); }
  return R;
}
const bbWhen = k => { if (!k) return ''; const t = new Date(k), d = new Date(t - t.getTimezoneOffset()*6e4).toISOString().slice(0, 10);
  return (d === localDay() ? 'Today' : dayName(d)) + ' ' + t.toLocaleTimeString([], {hour:'numeric', minute:'2-digit'}); };
function renderBest(){
  const out = $('#bestOut'); if (!out) return;
  if (!LT && !PO_SP.some(sp => PO[sp])){ out.innerHTML = '<p class="small muted">Loading odds…</p>'; return; }
  const all = bbRows(), sps = ['All', ...SPORT_LIST.filter(sp => all.some(r => r.sp === sp))];
  if (!sps.includes(BB_SP)) BB_SP = 'All';
  let L = all.filter(r => (BB_SP === 'All' || r.sp === BB_SP) && (BB_KIND === 'all' || r.type === BB_KIND));
  const ck = r => r.kind === 'check' || r.kind === 'verify';
  L.sort(BB_SORT === 'ev' ? (a, b) => ck(a) - ck(b) || b.ev - a.ev
    : (a, b) => ck(a) - ck(b) || String(a.k || '9').localeCompare(String(b.k || '9')) || BB_RANK[a.kind] - BB_RANK[b.kind] || b.ev - a.ev);
  const good = L.filter(r => !ck(r)).length, vis = BB_ALL ? L : L.slice(0, 80);
  const swept = [...PO_SP.filter(sp => PO[sp]).map(sp => PO[sp].made), LT && LT.made].filter(Boolean).map(x => Date.parse(String(x).replace(' UTC', 'Z').replace(' ', 'T'))).filter(isFinite);
  const newest = swept.length ? Math.max(...swept) : null, stale = newest && Date.now() - newest > 30*36e5;
  const chips = (arr, cur, attr) => `<div class="chips" role="group">${arr.map(([v, l]) => `<button type="button" data-${attr}="${v}" aria-pressed="${v === cur}">${esc(l)}</button>`).join('')}</div>`;
  let lastHead = '';
  out.innerHTML = chips(sps.map(s => [s, s === 'All' ? 'All sports' : s]), BB_SP, 'bs') + chips(BB_KINDS, BB_KIND, 'bk')
    + `<div class="row" style="justify-content:space-between"><p class="small ${stale ? 'wrn' : 'muted'}">${good} bet${good === 1 ? '' : 's'} at your books${newest ? ' · odds from ' + esc(new Date(newest).toLocaleString([], {weekday:'short', hour:'numeric', minute:'2-digit'})) : ''}${stale ? ' (over a day old)' : ''}</p>${chips([['time', 'By time'], ['ev', 'By EV']], BB_SORT, 'bo')}</div>`
    + (vis.length ? `<div class="glist">${vis.map((r, i) => { const h = BB_SORT === 'time' && !ck(r) ? bbWhen(r.k) : ck(r) ? 'Check before betting' : '';
        const head = h && h !== lastHead ? `<h3 style="margin-top:10px">${esc(h)}</h3>` : ''; if (h) lastHead = h;
        return head + `<button type="button" class="grow" data-i="${i}"><span><b>${esc(r.title)}</b> ${BB_TAG[r.kind] || ''}<br><span class="sm">${esc(r.sp)} · ${esc(r.sub)}${BB_SORT === 'ev' && r.k ? ' · ' + esc(bbWhen(r.k)) : ''}</span></span>
          <span>${r.kind === 'check' ? '<span class="big wrn">check</span>' : r.type === 'promo' ? `<span class="big pos">${pct(r.ev)}</span>` : `<span class="big pos">${sgn(r.ev)}</span>`}<br><span class="sm">${esc(r.note)}</span></span></button>`; }).join('')}</div>
        ${L.length > vis.length ? `<div class="row"><button class="btn ghost" type="button" id="bbMore">Show all ${L.length}</button></div>` : ''}`
      : `<div class="empty">Nothing at your books clears the EV minimum${BB_SP !== 'All' ? ' for ' + esc(BB_SP) : ''} right now. Odds refresh each morning.</div>`)
    + `<p class="small muted">Props: Proven and Flag need ${Math.round(PF_THR*100)}%+ EV (change it on any sport's Flags tab). Games: our model and the blend both beat the price, or the price beats the sharp fair line by 2%+. Prices move: check the book before betting. Tap a bet to open its card and log it.</p>`;
  out.querySelectorAll('[data-bs]').forEach(b => b.onclick = () => { BB_SP = b.dataset.bs; lsSet('betdesk.bbsp', BB_SP); renderBest(); });
  out.querySelectorAll('[data-bk]').forEach(b => b.onclick = () => { BB_KIND = b.dataset.bk; lsSet('betdesk.bbkind', BB_KIND); renderBest(); });
  out.querySelectorAll('[data-bo]').forEach(b => b.onclick = () => { BB_SORT = b.dataset.bo; lsSet('betdesk.bbsort', BB_SORT); renderBest(); });
  const more = $('#bbMore'); if (more) more.onclick = () => { BB_ALL = true; renderBest(); };
  out.querySelectorAll('.grow[data-i]').forEach(b => b.onclick = () => vis[+b.dataset.i].go());
}
