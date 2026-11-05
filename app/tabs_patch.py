# Bet Desk rework step 3 (J 2026-10-28): Sports sub-tabs become Games / Props / Tools, parlay slip moves under Bets,
# calculators move from Settings to Tools. Panels and their code are unchanged; only where they sit.
import re
p = '/mnt/project-files/sports-analytics/app/bet-desk.html'; s = open(p).read()
def cut(a, b):
    global s; i = s.index(a); j = s.index(b, i) + len(b); blk = s[i:j]; s = s[:i] + s[j:]; return blk
def rep(a, b):
    global s; assert s.count(a) == 1, a[:60]; s = s.replace(a, b)
# calculators -> Tools panel
calc = cut('  <details class="card" id="mxCard">', '<div id="lcHedge"></div>\n  </details>\n')
# parlays -> Bets
par = cut('  <div class="panel" id="sv-parlays" hidden>', '      <div id="sgOut"></div>\n    </div>\n  </div>\n')
par = par.replace('<div class="panel" id="sv-parlays" hidden>', '<div id="sv-parlays">')
par = par.replace('<div class="row" style="justify-content:space-between"><h3>Suggested parlays</h3>\n        <select id="sgKind" style="width:auto">',
                  '<details id="sgFold"><summary><b>Suggested parlays</b></summary><div class="row" style="justify-content:flex-end">\n        <select id="sgKind" style="width:auto">')
par = par.replace('      <div id="sgOut"></div>\n    </div>\n  </div>\n', '      <div id="sgOut"></div></details>\n    </div>\n  </div>\n')
assert 'sgFold' in par and '</details>' in par
rep('  <div class="panel" id="sv-shop" hidden>\n    <div class="card"><h3>Line shop</h3><div id="shopOut"></div></div>\n  </div>\n',
    '  <div class="panel" id="sv-shop" hidden>\n    <div class="card"><h3>Line shop</h3><div id="shopOut"></div></div>\n  </div>\n  <div class="panel" id="sv-calc" hidden>\n' + calc + '  </div>\n')
rep('  <div class="card" id="midWatch" hidden></div>\n', '  <div class="card" id="midWatch" hidden></div>\n' + par)
rep('<div class="chips" id="flagMode" role="group" aria-label="Flags view"></div>', '<h3>Flagged at your books</h3>\n      <div class="chips" id="flagMode" role="group" aria-label="Flags view"></div>')
# sub-tabs
rep("const VIEW_NAME = {", "// sub-tabs (rework step 3): each shows a set of the panels above\nconst TABS = {games:['games'], props:['props','flags'], tools:['shop','promo','calc']}, TAB_OF = {games:'games', props:'props', flags:'props', shop:'tools', promo:'tools', calc:'tools'};\nconst tabsFor = sp => Object.keys(TABS).filter(t => sp !== 'MLB' && sp !== 'Soccer' || t !== 'tools').filter(t => TABS[t].some(k => k === 'calc' || VIEWS[sp].includes(k)));\nconst VIEW_NAME = {tools:'Tools', ")
old_sub = s[s.index('function renderSubBar(){'):s.index('// new sport: nothing typed')]
new_sub = '''function renderSubBar(){
  const v = tabsFor(S.sport), want = TAB_OF[S.sub[S.sport]] || S.sub[S.sport], cur = v.includes(want) ? want : v[0];
  S.sub[S.sport] = cur;
  const cnt = {props: flagCount(), games: edgeCount(), tools: shopCount()};
  $('#subBar').hidden = v.length < 2;
  $('#subBar').innerHTML = v.map(k => `<button type="button" aria-pressed="${k===cur}" data-k="${k}">${VIEW_NAME[k]}${cnt[k]?`<span class="badge">${cnt[k]}</span>`:''}</button>`).join('');
  $('#subBar').querySelectorAll('button').forEach(b => b.onclick = () => showSub(b.dataset.k));
  const on = TABS[cur].filter(k => k === 'calc' || VIEWS[S.sport].includes(k));
  ['games','shop','props','flags','promo','calc'].forEach(k => $('#sv-'+k).hidden = !on.includes(k));
  $('#sv-parlays').hidden = !VIEWS[S.sport].includes('parlays');
  $('#plCount').previousElementSibling.textContent = `Your ${S.sport} parlay`;
}
// old panel names still work: flags open Props, line shop and promo open Tools (scrolled to that panel), parlays open Bets
function showSub(k){
  if (k === 'parlays'){ show('bets'); const el = $('#sv-parlays'); if (el) el.scrollIntoView(); return; }
  const t = TAB_OF[k] || k; S.sub[S.sport] = t; lsSet('betdesk.sub', JSON.stringify(S.sub)); renderSubBar(); if (t==='games') showGameCard();
  if (k !== t && k !== 'flags' && $('#sv-'+k)) $('#sv-'+k).scrollIntoView(); else window.scrollTo(0,0); }
'''
s = s.replace(old_sub, new_sub)
open(p, 'w').write(s)
print('ok')
