"""Bet Desk rework step 2 (J 2026-10-28, plan https://claude.ai/artifact/XvkVjmErmjZqUPQGvzmLS2): Best bets home screen
(best_bets.js, inlined) as the first tab; Slate renamed Sports; Settings moves to a gear in the header. Opens on Best bets.
python3 best_bets_patch.py [bet-desk.html]; safe to run twice, stops if an anchor moved."""
import sys, os
here = os.path.dirname(os.path.abspath(__file__))
path = sys.argv[1] if len(sys.argv) > 1 else f'{here}/bet-desk.html'
s = open(path, encoding='utf-8').read()
JS = open(f'{here}/best_bets.js', encoding='utf-8').read()
EDITS = [
 ('css', '</style>', 'header.top.bbmode #sportBar,header.top.bbmode #dayBar{display:none}\n.gear{background:none;border:1px solid var(--line);border-radius:999px;color:var(--muted);font-size:16px;line-height:1;padding:6px 10px;cursor:pointer}.gear[aria-pressed="true"]{color:var(--accent);border-color:var(--accent)}\n.brand .right{display:flex;gap:10px;align-items:center}\n</style>'),
 ('gear', '<div class="brand"><h1>Bet Desk</h1><span id="saveState" class="save">Connecting…</span></div>',
  '<div class="brand"><h1>Bet Desk</h1><span class="right"><span id="saveState" class="save">Connecting…</span><button class="gear" type="button" id="gearBtn" aria-label="Settings" aria-pressed="false">⚙</button></span></div>'),
 ('view', '<section class="view" id="v-slate">', '<section class="view" id="v-best" hidden>\n  <h2>Best bets</h2>\n  <div id="bestOut"><p class="small muted">Loading odds…</p></div>\n</section>\n<section class="view" id="v-slate">'),
 ('nav', '''  <button role="tab" data-v="slate" aria-selected="true">Slate</button>
  <button role="tab" data-v="bets" aria-selected="false">Bets</button>
  <button role="tab" data-v="results" aria-selected="false">Results</button>
  <button role="tab" data-v="settings" aria-selected="false">Settings</button>''', '''  <button role="tab" data-v="best" aria-selected="true">Best bets</button>
  <button role="tab" data-v="slate" aria-selected="false">Sports</button>
  <button role="tab" data-v="bets" aria-selected="false">Bets</button>
  <button role="tab" data-v="results" aria-selected="false">Results</button>'''),
 ('show', "  try { localStorage.setItem('betdesk.tab', v); } catch(e){}\n  if (v==='results') renderCharts();",
  "  try { localStorage.setItem('betdesk.tab2', v); } catch(e){}\n  $('header.top').classList.toggle('bbmode', v==='best' || v==='settings'); $('#gearBtn').setAttribute('aria-pressed', v==='settings');\n  if (v==='results') renderCharts();\n  if (v==='best') renderBest();"),
 ('nav click', "document.querySelectorAll('nav.tabs button').forEach(b => b.addEventListener('click', () => show(b.dataset.v)));",
  "document.querySelectorAll('nav.tabs button').forEach(b => b.addEventListener('click', () => show(b.dataset.v)));\n$('#gearBtn').addEventListener('click', () => show($('#v-settings').hidden ? 'settings' : 'best'));"),
 ('start tab', "  try { const t = localStorage.getItem('betdesk.tab'); if (t) show(t); } catch(e){}", "  try { show(localStorage.getItem('betdesk.tab2') || 'best'); } catch(e){ show('best'); }   // J 2026-10-28: opens on Best bets"),
 ('load', "  initGames();\n  refreshSport(true); renderModels();\n}", "  initGames();\n  refreshSport(true); renderModels(); renderBest();\n}"),
 ('stakes', "function redrawStakes(){ renderMyBooks(); LTC.clear(); renderShop();", "function redrawStakes(){ renderMyBooks(); LTC.clear(); PFC.clear(); renderShop(); renderBest();"),
 ('js', '// ---------- boot ----------', JS + '\n// ---------- boot ----------'),
]
if 'function renderBest' in s: sys.exit('already patched')
for name, a, b in EDITS:
    if s.count(a) != 1: sys.exit(f'anchor {name!r} found {s.count(a)} times')
    s = s.replace(a, b)
open(path, 'w', encoding='utf-8').write(s)
print('patched', path)
