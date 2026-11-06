"""Bet Desk prop odds (J 2026-10-27: odds for every prop daily). Book prices + our model's EV on every prop card (line and
prices default to the best-EV book), Flags view modes Props +EV and Prop middles. Code in prop_odds.js, inlined here;
data props_{NFL,NBA,NHL}.json from op_live.py published at the artifact root. python3 prop_odds_patch.py [bet-desk.html];
safe to run twice, stops if an anchor moved."""
import sys, os
here = os.path.dirname(os.path.abspath(__file__))
path = sys.argv[1] if len(sys.argv) > 1 else f'{here}/bet-desk.html'
s = open(path, encoding='utf-8').read()
JS = open(f'{here}/prop_odds.js', encoding='utf-8').read()
EDITS = [
 ('css', '</style>', 'tr.sel td{background:rgba(46,160,67,.14)}\n</style>'),
 ('flag modes', '''      <p class="small muted">Players whose projection is well above or below their own games at a similar workload. Tap one to price it.</p>
      <div class="grid three">''', '''      <div class="chips" id="flagMode" role="group" aria-label="Flags view"></div>
      <p class="small muted">Players whose projection is well above or below their own games at a similar workload. Tap one to price it.</p>
      <div class="grid three">'''),
 ('flag thr', '<label>Flag at<select id="fThr">', '<label id="fThrWrap">Flag at<select id="fThr">'),
 ('render flags', """function renderFlags(){
  const sp = S.sport, sl = S.slates[sp]; if (!sl || !VIEWS[sp].includes('flags')){ $('#flagOut').innerHTML=''; return; }""",
  """function renderFlags(){
  const sp = S.sport, sl = S.slates[sp]; if (!sl || !VIEWS[sp].includes('flags')){ $('#flagOut').innerHTML=''; return; }
  if (poRenderFlags()) return;"""),
 ('prop line', "const s = p.s[stat], L = num($('#pLine').value);",
  "const s = p.s[stat], pob = poRows(S.sport, p, stat); let L = num($('#pLine').value);\n  if (L == null){ const f = poAutofill(p, stat, pob); if (f != null) L = f; }\n  const bt = poTable(S.sport, pob, L);"),
 ('prop noline', """out.innerHTML = head + `<p class="small muted">Enter the book's line to get the chance of over and under.</p></div>`; return; }""",
  """out.innerHTML = head + `<p class="small muted">Enter the book's line to get the chance of over and under.</p></div>` + bt; poWire(out, pob); return; }"""),
 ('prop card', 'out.innerHTML = head + `<div class="card">${v.html}<div class="sides">', 'out.innerHTML = head + bt + `<div class="card">${v.html}<div class="sides">'),
 ('prop wire', "  $('#plOver').onclick = () => plAddProp('over');", "  poWire(out, pob); $('#plOver').onclick = () => plAddProp('over');"),
 ('files', "'cfb_model.json','cfb_v2.json'];", "'cfb_model.json','cfb_v2.json','props_NFL.json','props_NBA.json','props_NHL.json'];"),
 ('load', "  renderMyBooks(); renderShop(); renderMidWatch();\n",
  "  renderMyBooks(); renderShop(); renderMidWatch();\n  for (const sp of PO_SP){ const j = await getJ('props_' + sp + '.json'); if (j && j.games) PO[sp] = j; }\n  renderFlags(); renderProp();\n"),
 ('js', '// ---------- boot ----------', JS + '\n// ---------- boot ----------'),
]
if 'const PO_MAP' in s: sys.exit('already patched')
for name, a, b in EDITS:
    if s.count(a) != 1: sys.exit(f'anchor {name!r} found {s.count(a)} times')
    s = s.replace(a, b)
open(path, 'w', encoding='utf-8').write(s)
print('patched', path)
