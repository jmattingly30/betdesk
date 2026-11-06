import sys
p=sys.argv[1] if len(sys.argv)>1 else 'bet-desk.html'; s=open(p).read()
def rep(old,new,cnt=1):
    global s
    n=s.count(old)
    if n!=cnt: sys.exit(f'count {n} for: {old[:80]}')
    s=s.replace(old,new)

# 1. priceSides: Mathax edge on our model, Kelly, book implied, theoretical hold
rep("""  const fp = fairPair(oa, ob); if (!fp) return null;
  const w = trust/100, pbA = il(w*lg(pmA) + (1-w)*lg(fp.fa));
  const st = S.settings;
  const mk = (i, pm, pf, pb, o) => {
    if (o == null) return {name:names[i], pm, pf, pb, odds:null};
    const d = dec(o), ev = (1-pushP)*(pb*d-1), be = 1/d;
    const kel = Math.max(0,(pb*d-1)/(d-1));""",
"""  const fp = fairPair(oa, ob); if (!fp) return null;
  const w = trust/100, pbA = il(w*lg(pmA) + (1-w)*lg(fp.fa));
  const st = S.settings, hold = oa!=null && ob!=null ? mxHold([oa, ob]) : null;
  const mk = (i, pm, pf, pb, o) => {
    if (o == null) return {name:names[i], pm, pf, pb, odds:null, hold};
    const d = dec(o), ev = (1-pushP)*(pb*d-1), be = 1/d, evm = (1-pushP)*mxEdge(pm, o);
    const kel = Math.max(0,(pb*d-1)/(d-1));""")
rep("""    return {name:names[i], pm, pf, pb, odds:o, ev, be, edge:pb-be, stake, need, isOver:i===overIdx};
  };
  const A = mk(0,pmA,fp.fa,pbA,oa), B = mk(1,1-pmA,1-fp.fa,1-pbA,ob);
  return {A, B, assumed:fp.assumed, pushP};
}""","""    return {name:names[i], pm, pf, pb, odds:o, ev, evm, be, edge:pb-be, kel, stake, need, isOver:i===overIdx, hold};
  };
  const A = mk(0,pmA,fp.fa,pbA,oa), B = mk(1,1-pmA,1-fp.fa,1-pbA,ob);
  return {A, B, assumed:fp.assumed, pushP, hold};
}
// ---------- Mathax (J's sheet "Copy of Mathax!", Odds + Futures tabs) ----------
// Edge = chance x decimal odds - 1 (EV per unit). Kelly = (b*p - q)/b, b = decimal - 1, times multiplier and bankroll.
// Theoretical hold = 1 - 1/(sum of every outcome's implied chance). Parlay edge = legs' chances multiplied, priced the same way.
const mxEdge = (p, a) => p*dec(a) - 1;
const mxKelly = (p, a) => { const b = dec(a)-1; return Math.max(0, (b*p - (1-p))/b); };
const mxHold = list => { const t = list.reduce((s,a) => s + imp(a), 0); return t > 0 ? 1 - 1/t : null; };
// Parlay hold, the same definition carried to a parlay: 1 - (no-vig chance of every leg multiplied) x the parlay's decimal odds
const mxParlayHold = (fairs, a) => 1 - fairs.reduce((x,y) => x*y, 1)*dec(a);""")

# 2. side cards (props + manual pricer)
rep("""    <div class="kv"><span>Blended</span><b>${pct(x.pb)}</b></div>
    <div class="kv"><span>Break-even</span><b>${pct(x.be)}</b></div>
    <div class="kv"><span>Edge</span><b class="${x.edge>=0?'pos':'neg'}">${sgn(x.edge)}</b></div>
    <div class="kv"><span>EV per unit</span><b class="${x.ev>=0?'pos':'neg'}">${sgn(x.ev)}</b></div>${extra}</div>`;""",
"""    <div class="kv"><span>Blended</span><b>${pct(x.pb)}</b></div>
    <div class="kv"><span>Book implied (break-even)</span><b>${pct(x.be)}</b></div>
    <div class="kv"><span>Edge, our model</span><b class="${x.evm>=0?'pos':'neg'}">${sgn(x.evm)}</b></div>
    <div class="kv"><span>Edge, blended</span><b class="${x.ev>=0?'pos':'neg'}">${sgn(x.ev)}</b></div>
    <div class="kv"><span>Kelly, blended</span><b>${pct(x.kel)} full · ${x.stake>0?x.stake.toFixed(1)+'u':'none'}</b></div>${extra}</div>`;""")
rep("""        <div class="kv"><span>Book, no vig</span><b>${pct(x.pf)}</b></div>
        <div class="kv"><span>Edge on ${esc(x.name)}</span><b class="${x.edge>=0?'pos':'neg'}">${sgn(x.edge)}</b></div>""",
"""        <div class="kv"><span>Book, no vig · book implied</span><b>${pct(x.pf)} · ${pct(x.be)}</b></div>
        <div class="kv"><span>Edge on ${esc(x.name)}, our model · blended</span><b><span class="${x.evm>=0?'pos':'neg'}">${sgn(x.evm)}</span> · <span class="${x.ev>=0?'pos':'neg'}">${sgn(x.ev)}</span></b></div>
        ${res.hold!=null && !assumed ? `<div class="kv"><span>Book hold</span><b>${pct(res.hold)}</b></div>` : ''}""")
# 3. hold notes on prop + manual pricer
rep("""    ${assumed110?'<p class="note">Odds assumed -110 on both sides. Type the book\\'s over and under odds for the real edge.</p>':''}""",
"""    ${assumed110?'<p class="note">Odds assumed -110 on both sides. Type the book\\'s over and under odds for the real edge.</p>':r.hold!=null?`<p class="small muted">Book hold on this prop: ${pct(r.hold)}.</p>`:''}""")
rep("""    <p class="note">Model weight ${trust}% (set in Models).""","""    ${r.hold!=null?`<p class="small muted">Book hold: ${pct(r.hold)}.</p>`:''}<p class="note">Model weight ${trust}% (set in Models).""")
# 4. parlay legs carry the no-vig chance and hold when both sides were priced
rep("""    side:x.name, odds:x.odds ?? null, p:+x.pb.toFixed(4), pm:+x.pm.toFixed(4), role:'G', model:r[3],""",
"""    side:x.name, odds:x.odds ?? null, p:+x.pb.toFixed(4), pm:+x.pm.toFixed(4), role:'G', model:r[3],
    ...(x.odds!=null && x.hold!=null ? {pf:+x.pf.toFixed(4), hold:+x.hold.toFixed(4)} : {}),""")
rep("""    key: role && st ? role+'_'+st : null, role, label:`${x.p.n} ${dir==='over'?'over':'under'} ${x.L} ${(STAT[x.stat]||x.stat).toLowerCase()}`};
  PL = PL.filter""","""    key: role && st ? role+'_'+st : null, role, label:`${x.p.n} ${dir==='over'?'over':'under'} ${x.L} ${(STAT[x.stat]||x.stat).toLowerCase()}`};
  const sd = dir==='over' ? x.r.A : x.r.B;   // the book's prices, only when both were typed
  if (x.r.hold!=null && !x.assumed110 && sd.odds!=null){ leg.odds = sd.odds; leg.pf = +sd.pf.toFixed(4); leg.hold = +x.r.hold.toFixed(4); }
  PL = PL.filter""")
rep("""  lastProp = {p, stat, L, r, best:v.best||r.A, book:$('#pBook').value.trim()};""",
"""  lastProp = {p, stat, L, r, assumed110, best:v.best||r.A, book:$('#pBook').value.trim()};""")
# 5. parlay builder: Mathax parlay edge + book hold
rep("""  } else h += '<p class="small muted">Type the book\\'s parlay (SGP) odds to see EV and stake.</p>';
  h += (plCache.warn||[]).map""","""  } else h += '<p class="small muted">Type the book\\'s parlay (SGP) odds to see EV and stake.</p>';
  h += `<h3 style="margin-top:12px">Mathax parlay edge</h3>
    <div class="kv"><span>Legs' chances multiplied</span><b>${pct(ind)}</b></div><div class="kv"><span>True odds</span><b>${am(ind)}</b></div>`;
  if (bo != null){
    const fairs = PL.map(l => l.pf), hd = fairs.every(f => f != null) ? mxParlayHold(fairs, bo) : null, legH = PL.filter(l => l.hold != null);
    h += `<div class="kv"><span>Book odds</span><b>${sgnPts(bo)}</b></div>
      <div class="kv"><span>Edge (legs independent) · with our correlations</span><b><span class="${ind*dec(bo)-1>=0?'pos':'neg'}">${sgn(ind*dec(bo)-1)}</span> · <span class="${ev>=0?'pos':'neg'}">${sgn(ev)}</span></b></div>
      <div class="kv"><span>Kelly (legs independent), full · your ${S.settings.kelly ?? 0.25}x</span><b>${pct(mxKelly(ind, bo))} · ${Math.min(+(mxKelly(ind, bo)*(S.settings.kelly ?? 0.25)*(S.settings.bank ?? 100)).toFixed(2), S.settings.max ?? 3)}u</b></div>
      <div class="kv"><span>Book's theoretical hold on this parlay</span><b>${hd != null ? pct(hd) : '–'}</b></div>
      <p class="small muted">${hd != null ? `Hold = 1 minus (every leg's no-vig chance multiplied) times the parlay's decimal odds${legH.length ? `; the legs alone hold ${legH.map(l => pct(l.hold)).join(', ')}` : ''}.` : 'Hold needs both sides of every leg: add game legs from a card with both prices typed, or type the over and under odds before adding a prop.'}</p>`;
  }
  h += (plCache.warn||[]).map""")
# 6. Settings: Mathax calculators
rep("""<section class="view" id="v-settings" hidden>
  <h2>Settings</h2>""","""<section class="view" id="v-settings" hidden>
  <h2>Settings</h2>
  <details class="card" id="mxCard"><summary><h3 style="display:inline">Calculators (Mathax)</h3></summary>
    <div class="grid">
      <label>Moneyline<input id="mxA" class="num" inputmode="numeric" placeholder="-310"></label>
      <label>Probability %<input id="mxP" class="num" inputmode="decimal" placeholder="17"></label>
      <label>Decimal odds<input id="mxD" class="num" inputmode="decimal" placeholder="1.91"></label>
    </div><div id="mxConv"></div>
    <h3 style="margin-top:12px">Edge and Kelly betsizer</h3>
    <div class="grid">
      <label>Win probability %<input id="mxKp" class="num" inputmode="decimal" placeholder="53.14"></label>
      <label>Odds<input id="mxKo" class="num" inputmode="numeric" placeholder="+118"></label>
      <label>Kelly multiplier<input id="mxKm" class="num" inputmode="decimal"></label>
      <label>Bankroll<input id="mxKb" class="num" inputmode="decimal"></label>
    </div><div id="mxKelly"></div>
    <h3 style="margin-top:12px">Theoretical hold</h3>
    <label>Every outcome's odds (2 for a game, up to 40 for a future), separated by spaces or commas<textarea id="mxH" rows="2" placeholder="-1667, +3000"></textarea></label>
    <div id="mxHold"></div>
  </details>""")
rep("""$('#saveSettings').onclick = async () => {""","""function renderMx(){
  const a = odds($('#mxA').value), p = num($('#mxP').value), d = num($('#mxD').value), out = [];
  if (a != null) out.push(`${sgnPts(a)} = ${pct(imp(a),2)} implied · decimal ${dec(a).toFixed(2)}`);
  if (p != null && p > 0 && p < 100) out.push(`${p}% = ${am(p/100)}`);
  if (d != null && d > 1) out.push(`decimal ${d} = ${amDec(d)}`);
  $('#mxConv').innerHTML = out.map(t => `<p class="small">${esc(t)}</p>`).join('');
  const kp = num($('#mxKp').value), ko = odds($('#mxKo').value), km = num($('#mxKm').value) ?? S.settings.kelly, kb = num($('#mxKb').value) ?? S.settings.bank;
  $('#mxKelly').innerHTML = kp != null && kp > 0 && kp < 100 && ko != null ? `<div class="kv"><span>Edge</span><b class="${mxEdge(kp/100,ko)>=0?'pos':'neg'}">${sgn(mxEdge(kp/100,ko),2)}</b></div>
    <div class="kv"><span>Full Kelly</span><b>${pct(mxKelly(kp/100,ko),2)} of bankroll</b></div>
    <div class="kv"><span>Bet (${km}x Kelly on ${kb})</span><b>${(mxKelly(kp/100,ko)*km*kb).toFixed(2)}</b></div>` : '';
  const hs = $('#mxH').value.split(/[\\s,]+/).map(odds).filter(x => x != null), hd = hs.length >= 2 ? mxHold(hs) : null;
  $('#mxHold').innerHTML = hd != null ? `<div class="kv"><span>Theoretical hold (${hs.length} outcomes)</span><b>${pct(hd,2)}</b></div>` : '';
}
function initMx(){
  $('#mxKm').placeholder = S.settings.kelly; $('#mxKb').placeholder = S.settings.bank;
  ['#mxA','#mxP','#mxD','#mxKp','#mxKo','#mxKm','#mxKb','#mxH'].forEach(q => $(q).addEventListener('input', renderMx));
  addSignButtons([$('#mxA'), $('#mxKo')]); renderMx();
}
$('#saveSettings').onclick = async () => {""")
rep("""  addSignButtons(); renderSportBar(); initPrice(); renderAll();""","""  addSignButtons(); renderSportBar(); initPrice(); initMx(); renderAll();""")
open(p,'w').write(s); print('ok')
