/* Watch the sim (J 2026-11-06). One viewer for every sport: replays simulated games that models/sims/build.py wrote to
   sims_{sport}.json (possessions for NBA/CBB, drives for NFL/CFB, points for tennis). Loaded only when a game tile's
   "Watch the sim" button is tapped, so the desk page itself does not grow. Uses the desk's own theme tokens and classes.
   Honesty: every replay is one game the simulator played from Model A, never a forecast of the real game.
   Structure: a sport adapter turns one stored sample into a common event list; one renderer draws every sport. */
(function(){
'use strict';
const pct = p => `${Math.round(p * 100)}%`;
const esc = s => String(s ?? '').replace(/[&<>"']/g, c => ({'&':'&amp;', '<':'&lt;', '>':'&gt;', '"':'&quot;', "'":'&#39;'}[c]));
const mmss = s => { s = Math.max(0, Math.round(s)); return `${Math.floor(s / 60)}:${String(s % 60).padStart(2, '0')}`; };

// ---------- adapters: (file meta, sample, [home, away]) -> {ev:[{t, h, a, per, clk, txt, hit}], T, fin, flowLab}
// ev.t = game time (minutes, or points played for tennis); h/a = score after the event; hit = a scoring event (shown in the log)
function possessions(m, s, N, A){
  const ev = [], reg = m.reg_min, per = m.periods, plen = reg / per; let h = 0, a = 0;
  const lab = q => q < per ? (per === 4 ? `Q${q + 1}` : `H${q + 1}`) : `OT${q - per + 1 > 1 ? q - per + 1 : ''}`;
  const push = (side, pts, t, q, left) => { side ? a += pts : h += pts;
    ev.push({t, h, a, per:lab(q), clk:mmss(left * 60), hit:pts > 0, txt:pts ? `${N[side]} score ${pts}` : `${N[side]}: no score`}); };
  const order = s.o ? [1, 0] : [0, 1], seq = [s.h, s.a];
  for (let k = 0; k < s.n; k++) order.forEach((side, j) => { const t = reg * (k + j / 2 + 0.5) / s.n, q = Math.min(per - 1, Math.floor(t / plen));
    push(side, +seq[side][k], t, q, (q + 1) * plen - t); });
  for (let o = 0; o < s.ot; o++) for (let k = 0; k < s.mo; k++) order.forEach((side, j) => { const t = reg + 5 * o + 5 * (k + j / 2 + 0.5) / s.mo;
    push(side, +seq[side][s.n + o * s.mo + k], t, per + o, reg + 5 * (o + 1) - t); });
  return {ev, T:reg + 5 * s.ot, fin:[h, a]};
}
function drives(m, s, N){
  const ev = [], L = m.labels || []; let h = 0, a = 0, el = 0, half = 0;
  const spot = y => y > 50 ? `own ${100 - y}` : y < 50 ? `opp ${y}` : 'midfield';
  for (const [hf, off, y, c, pts, who, sec] of s.d){
    if (hf !== half){ el = (hf - 1) * 1800; half = hf; }
    el += sec; who ? a += pts : h += pts;
    const q = hf > 2 ? 4 : Math.min(3, Math.floor((el - 1e-6) / 900)), sc = pts ? ` · ${who === off ? '+' : `${N[who]} +`}${pts}` : '';
    ev.push({t:el / 60, h, a, per:hf > 2 ? 'OT' : `Q${q + 1}`, clk:hf > 2 ? '' : mmss(Math.max(0, (q + 1) * 900 - el)), hit:pts > 0 || c === 7,
             txt:`${N[off]} ball at ${spot(y)}: ${L[c] || c}${sc} · ${mmss(sec)} drive`});
  }
  return {ev, T:Math.max(60, el / 60), fin:[h, a]};
}
function points(m, s, N){
  // tennis scoring rules exactly as models/sims/build.py plays them: games to 4 by 2, tiebreak to 7 by 2 at 6-6 in every set
  const ev = [], P = s.p, sets = []; let i = 0, sv = s.f, sw = [0, 0], gd = 0;   // the point string ends with the match
  const call = (w, l) => w > 3 && l > 2 ? 'after deuce' : ['at love', 'to 15', 'to 30', 'to 40'][l];
  while (i < P.length){
    const g = [0, 0];
    for (;;){
      if (g[0] === 6 && g[1] === 6){
        const t = [0, 0]; while (!((t[0] >= 7 || t[1] >= 7) && Math.abs(t[0] - t[1]) >= 2) && i < P.length) t[P[i++] === '1' ? 0 : 1]++;
        const w = t[0] > t[1] ? 0 : 1; g[w]++; gd += w ? -1 : 1; sv = 1 - sv;
        ev.push({t:i, h:sw[0], a:sw[1], g:[...g], d:gd, per:`Set ${sets.length + 1}`, clk:`${g[0]}-${g[1]}`, hit:true, txt:`${N[w]} win the tiebreak ${Math.max(...t)}-${Math.min(...t)}`});
        break;
      }
      const p = [0, 0]; while (!((p[0] >= 4 || p[1] >= 4) && Math.abs(p[0] - p[1]) >= 2) && i < P.length) p[P[i++] === '1' ? 0 : 1]++;
      const w = p[0] > p[1] ? 0 : 1; g[w]++; gd += w ? -1 : 1;
      ev.push({t:i, h:sw[0], a:sw[1], g:[...g], d:gd, per:`Set ${sets.length + 1}`, clk:`${g[0]}-${g[1]}`, hit:w !== sv,
               txt:`${N[sv]} serving: ${w === sv ? 'holds' : `${N[w]} breaks`} ${call(p[w], p[1 - w])}`});
      sv = 1 - sv;
      if ((g[0] >= 6 || g[1] >= 6) && Math.abs(g[0] - g[1]) >= 2) break;
    }
    sets.push(g); sw[g[0] > g[1] ? 0 : 1]++;
    const e = ev[ev.length - 1]; e.h = sw[0]; e.a = sw[1]; e.txt += ` · set ${sets.length} to ${N[g[0] > g[1] ? 0 : 1]} ${Math.max(...g)}-${Math.min(...g)}`;
  }
  return {ev, T:Math.max(1, i), fin:sw, sets};
}
const ADAPT = {possession:possessions, drive:drives, point:points};

// ---------- viewer
const CACHE = {}, load = sp => CACHE[sp] || (CACHE[sp] = fetch(`sims_${sp.toLowerCase()}.json`, {cache:'no-cache'})
  .then(r => r.ok ? r.json() : null).catch(() => null).then(j => (j || (delete CACHE[sp]), j)));
const findGame = (F, x) => { const G = F.games || {}, id = String(x.g ?? x.id ?? '');
  return G[id] || Object.values(G).find(g => (g.h === x.home && g.a === x.away) || (g.h === x.p1 && g.a === x.p2)) || null; };
let V = null;   // {F, g, N, L, i, run, pos, timer, speed, stale}

function dialog(){
  let d = document.getElementById('simDlg'); if (d) return d;
  const st = document.createElement('style');
  st.textContent = `#simDlg{width:min(680px,calc(100vw - 16px))}#simDlg .in{padding:14px;display:flex;flex-direction:column;gap:10px}
#simDlg .hd{display:flex;justify-content:space-between;align-items:center;gap:8px}#simDlg h2{margin:0;font:400 22px var(--f-display)}
#simDlg .sb{display:grid;grid-template-columns:1fr auto 1fr;align-items:center;gap:8px;background:var(--sunk);border-radius:10px;padding:10px}
#simDlg .tm{min-width:0}#simDlg .tm span{display:block;font-size:12px;color:var(--muted);overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
#simDlg .tm b{font:400 34px/1.1 var(--f-display)}#simDlg .tm.r{text-align:right}#simDlg .mid{text-align:center;font:11px var(--f-num);color:var(--muted)}
#simDlg .mid b{display:block;font:600 13px var(--f-num);color:var(--ink)}#simDlg svg{width:100%;height:96px;display:block}
#simDlg .ctl{display:flex;flex-wrap:wrap;gap:6px;align-items:center}#simDlg .ctl .btn{min-height:40px}
#simDlg ol{list-style:none;margin:0;padding:0;max-height:180px;overflow:auto;border-top:1px solid var(--line)}
#simDlg li{display:grid;grid-template-columns:64px 1fr auto;gap:8px;padding:6px 0;border-bottom:1px solid var(--line);font-size:12px}
#simDlg li .w{color:var(--muted);font-family:var(--f-num)}#simDlg li b{font-family:var(--f-num)}#simDlg li.dim{color:var(--muted)}
#simDlg .seg button{min-height:36px}`;
  document.head.appendChild(st);
  d = document.createElement('dialog'); d.id = 'simDlg'; d.setAttribute('aria-label', 'Watch the sim'); document.body.appendChild(d);
  d.addEventListener('close', stop); d.addEventListener('click', onClick);
  return d;
}
function stop(){ if (V && V.timer){ clearInterval(V.timer); V.timer = null; } }
// clicks: one delegated handler for the whole dialog (bound once), so a redraw never re-binds anything
function onClick(e){
  const d = dialog(), b = e.target.closest('button'); if (e.target === d) return d.close();   // tap outside the box closes
  if (!b) return; const k = b.dataset.k;
  if (k === 'x') return d.close();
  if (!V || !V.run) return; const E = V.run.ev;
  if (k === 'p') return play();
  if (k === 's'){ stop(); const n = E.findIndex((x, i) => i >= V.pos && x.hit); V.pos = n < 0 ? E.length : n + 1; return update(); }
  if (k === 'f'){ stop(); V.pos = E.length; return update(); }
  if (b.dataset.sp){ V.speed = +b.dataset.sp; if (V.timer){ stop(); play(); } else update(); return; }
  if (b.dataset.i) pick(+b.dataset.i);
}
function pick(i){
  stop(); const run = ADAPT[V.F.unit](V.F, V.g.s[i], V.N), tennis = V.F.unit === 'point', val = e => tennis ? e.d : e.h - e.a;
  const top = Math.max(tennis ? 4 : 8, ...run.ev.map(e => Math.abs(val(e))));
  run.xy = [[0, 48], ...run.ev.map(e => [+(e.t / run.T * 300).toFixed(1), +(48 - val(e) / top * 42).toFixed(1)])];   // flow chart points, once per replay
  V.i = i; V.run = run; V.pos = 0; shell(); update();
}
function play(){
  if (V.timer){ stop(); return update(); }
  if (V.pos >= V.run.ev.length) V.pos = 0;
  const ms = Math.max(30, 24000 / V.run.ev.length / V.speed);       // a full game takes ~24 s at 1x
  V.timer = setInterval(() => { V.pos++; if (V.pos >= V.run.ev.length) stop(); update(); }, ms); update();
}
// the parts that never change during a replay
function shell(){
  const {F, g, N} = V, A = g.A || {}, a = g.audit || {}, tennis = F.unit === 'point';
  const model = tennis ? `Model A: ${N[0]} wins ${pct(A.win_p1)} · about ${Math.round(A.tot_mean)} games`
    : `Model A: ${N[1]} ${A.as.toFixed(1)}, ${N[0]} ${A.hs.toFixed(1)} · ${N[0]} wins ${pct(A.p_home)}`;
  const check = tennis ? `Across ${a.n} simulated matches ${N[0]} won ${pct(a.win_pool)} (Model A ${pct(A.win_p1)}).`
    : `Across ${a.n.toLocaleString()} simulated games weighted to Model A's range, ${N[0]} won ${pct(a.p_home_pool)} (Model A ${pct(A.p_home)}), average margin ${a.margin_pool > 0 ? '+' : ''}${a.margin_pool}, total ${a.total_pool}.`;
  const L = tennis ? [0, 1] : [1, 0], lab = tennis ? 'games ahead' : 'leads';   // tennis: player 1 on the left; team sports: away @ home
  dialog().innerHTML = `<div class="in">
    <div class="hd"><h2>Watch the sim</h2><button class="btn ghost sm" type="button" data-k="x">Close</button></div>
    <p class="small"><span class="chip y">Simulation</span> One game our simulator played from Model A, not a prediction of the real game. ${esc(model)}.${V.stale ? ' <span class="neg">Model A has moved since these sims were built; they refresh with the next update.</span>' : ''}</p>
    <div class="sb" aria-live="polite"><div class="tm"><span>${esc(N[L[0]])}</span><b data-v="l"></b></div>
      <div class="mid"><b data-v="st"></b><span data-v="md"></span></div><div class="tm r"><span>${esc(N[L[1]])}</span><b data-v="r"></b></div></div>
    <svg viewBox="0 0 300 96" role="img" aria-label="Game flow: ${esc(N[0])} ${lab} above the line, ${esc(N[1])} below">
      <line x1="0" x2="300" y1="48" y2="48" stroke="var(--line)"/><text x="2" y="10" font-size="8" fill="var(--muted)" font-family="var(--f-num)">${esc(N[0])} ${lab}</text>
      <text x="2" y="93" font-size="8" fill="var(--muted)" font-family="var(--f-num)">${esc(N[1])} ${lab}</text>
      <path data-v="path" fill="none" stroke="var(--accent)" stroke-width="2" stroke-linejoin="round"/><circle data-v="dot" r="3.5" fill="var(--accent)"/></svg>
    <div class="ctl"><button class="btn sm" type="button" data-k="p"></button><button class="btn ghost sm" type="button" data-k="s">Step</button><button class="btn ghost sm" type="button" data-k="f">Final</button>
      <div class="seg" role="group" aria-label="Speed">${[1, 3, 8].map(v => `<button type="button" data-sp="${v}">${v}x</button>`).join('')}</div></div>
    <div class="ctl"><span class="small muted">${g.s.length} sims:</span><div class="seg" role="group" aria-label="Pick a sim">${g.s.map((_, k) => `<button type="button" data-i="${k}" aria-pressed="${k === V.i}">${k + 1}</button>`).join('')}</div></div>
    <ol data-v="log"></ol>
    <details><summary class="small"><b>How this sim works</b></summary><p class="small">${esc((F.how || '').replace('{poss}', A.poss ? Math.round(A.poss) : 'the projected number of'))}</p>
      <p class="small muted">${esc(check)} Each of the ${g.s.length} replays was drawn at random from those games, so a replay can be a blowout or an upset; over many replays the ${tennis ? 'match results' : 'scores'} line up with Model A. Built ${esc(F.made)}.</p></details></div>`;
  V.L = L;
}
// the parts that move: scoreboard, flow line, log, button labels
function update(){
  const d = dialog(), {F, run, pos, L} = V, e = run.ev[pos - 1], tennis = F.unit === 'point', done = pos >= run.ev.length, q = k => d.querySelector(`[data-v="${k}"]`);
  const sc = [e ? e.h : 0, e ? e.a : 0];
  q('l').textContent = sc[L[0]]; q('r').textContent = sc[L[1]];
  q('st').textContent = !e ? (tennis ? 'First serve' : F.unit === 'drive' ? 'Kickoff' : 'Tip-off') : done ? `Final${!tennis && /OT/.test(e.per) ? ' (OT)' : ''}` : e.per;
  q('md').textContent = tennis ? (done ? run.sets.map(s => s.join('-')).join(' ') : e ? `games ${e.g.join('-')}` : 'sets') : (e && !done ? e.clk : '');
  const xy = run.xy.slice(0, pos + 1), [cx, cy] = xy[xy.length - 1];
  q('path').setAttribute('d', 'M' + xy.join('L').replace(/,/g, ' ')); q('dot').setAttribute('cx', cx); q('dot').setAttribute('cy', cy);
  const log = run.ev.slice(0, pos).filter((x, k) => x.hit || k === pos - 1 || F.unit === 'drive').slice(-40).reverse();   // every drive; scoring trips / breaks only
  q('log').innerHTML = log.map(x => `<li class="${x.hit ? '' : 'dim'}"><span class="w">${esc(x.per)}${x.clk && !tennis ? ` ${esc(x.clk)}` : ''}</span><span>${esc(x.txt)}</span><b>${tennis ? esc(x.g.join('-')) : `${x.a}-${x.h}`}</b></li>`).join('')
    || `<li class="dim"><span class="w"></span><span>Press Play to watch, Step to jump to the next score, or Final for the result.</span><b></b></li>`;
  d.querySelector('[data-k="p"]').textContent = V.timer ? 'Pause' : done ? 'Replay' : pos ? 'Resume' : 'Play';
  d.querySelector('[data-k="s"]').hidden = d.querySelector('[data-k="f"]').hidden = done;
  d.querySelectorAll('[data-sp]').forEach(b => b.setAttribute('aria-pressed', String(V.speed === +b.dataset.sp)));
}
async function open(sp, x){
  const d = dialog(); stop(); if (V) V.run = null;
  d.innerHTML = `<div class="in"><div class="hd"><h2>Watch the sim</h2><button class="btn ghost sm" type="button" data-k="x">Close</button></div><p class="small muted">Loading the simulated games…</p></div>`;
  if (!d.open) d.showModal();
  const F = await load(sp), g = F && findGame(F, x || {});
  if (!g || !g.s || !g.s.length){ d.querySelector('p').textContent = F ? 'No simulated games for this one yet. Sims are built each morning for today\'s games (NFL and CFB: this week).' : 'Could not load the sims file. Try again in a moment.'; return; }
  const cur = x && x._A, A = g.A || {};   // the desk's Model A now: flag sims built before Model A last moved
  const stale = !!(cur && cur.hs != null && A.hs != null && Math.abs(cur.hs - A.hs) + Math.abs(cur.as - A.as) > 1);
  V = {F, g, stale, N:[x && (x.home || x.p1) || g.h, x && (x.away || x.p2) || g.a], speed:(V && V.speed) || 1};
  pick(0);
}
if (typeof module !== 'undefined') module.exports = {ADAPT};
else window.BDSim = {open};
})();
