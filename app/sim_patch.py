"""Watch the sim hook for the Bet Desk (J 2026-11-06): adds a 'Watch the sim' button to every game tile (NBA, CBB, NFL, CFB) and the
tennis match tile. The button loads app/sim.js on the first tap (published next to the page as sim.js) and opens its viewer on
sims_{sport}.json (models/sims/build.py). The button stays hidden until that sport's sims file answers (one GET per sport
per visit), so a sport whose sims were not built shows no dead button. Nothing else on the desk changes (about 1.3 KB).
Usage: python3 app/sim_patch.py [app/redesign.js]   then python3 app/redesign_inline.py, check bet-desk-v2.html, copy over.
On a desk that already has the first hook (v126) it only upgrades it to the hidden-until-available button. Refuses to run twice."""
import sys, os
f = sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.dirname(os.path.abspath(__file__)), 'redesign.js')
s = open(f, encoding='utf-8').read()
assert 'simShow(' not in s, 'sim hook already applied (with the availability check)'
def sub(old, new):
    global s
    assert s.count(old) == 1, f'anchor not found once: {old[:70]!r}'
    s = s.replace(old, new)
V1_BTN = "const simBtn = sp => SIM_SP.includes(sp) ? `<div class=\"row\"><button class=\"btn ghost sm\" type=\"button\" data-sim>&#9654; Watch the sim</button><span class=\"small muted\">a simulated game from Model A, not a prediction</span></div>` : '';\n"
BTN = """const simBtn = sp => SIM_SP.includes(sp) ? `<div class="row" data-simrow hidden><button class="btn ghost sm" type="button" data-sim>&#9654; Watch the sim</button><span class="small muted">a simulated game from Model A, not a prediction</span></div>` : '';
const SIMOK = {};   // sport -> promise: does sims_{sport}.json exist (asked once per visit; GET, since the artifact host may not answer HEAD)
function simShow(sp, root){ const k = String(sp).toLowerCase();
  (SIMOK[k] = SIMOK[k] || fetch(`sims_${k}.json`, {cache:'no-cache'}).then(r => r.ok).catch(() => false))
    .then(ok => { if (ok) root.querySelectorAll('[data-simrow]').forEach(r => r.hidden = false); }); }
"""
GAME_BIND = "  el.querySelectorAll('[data-sim]').forEach(b => b.onclick = () => simOpen(sp, {...x, _A:(AB2(sp) && cabGame(x, sp) || {}).A}));\n"
TEN_BIND = "  out.querySelectorAll('[data-sim]').forEach(b => b.onclick = () => simOpen('TENNIS', {id:q.fid, p1:q.p1, p2:q.p2, home:N[0], away:N[1]}));\n"
if 'simOpen(' not in s:
    # 1. helper, placed just before the game tile code
    sub("// the game screen: tapping a row fills the desk card (old row click) and shows the sheet above it\n",
    """// Watch the sim (J 2026-11-06): sim.js + sims_{sport}.json (models/sims/build.py), loaded on the first tap so the page stays the same size
const SIM_SP = ['NBA', 'CBB', 'NFL', 'CFB', 'Tennis'];
""" + V1_BTN + """function simOpen(sp, x){
  const go = () => window.BDSim.open(sp, x);
  if (window.BDSim) return go();
  const s = document.createElement('script'); s.src = 'sim.js'; s.onload = go;
  s.onerror = () => { s.remove(); toast('Could not load the sim viewer'); }; document.head.appendChild(s);
}
// the game screen: tapping a row fills the desk card (old row click) and shows the sheet above it
""")
    # 2. game tile: button under the Model A / Model B boxes
    sub("""    <div class="abrow">${AB2(sp) ? abBox(sp, x, 'A', ab.A) + cabBox(sp, x) : ['A', 'B'].map(k => abBox(sp, x, k, ab[k])).join('')}</div>\n""",
        """    <div class="abrow">${AB2(sp) ? abBox(sp, x, 'A', ab.A) + cabBox(sp, x) : ['A', 'B'].map(k => abBox(sp, x, k, ab[k])).join('')}</div>${simBtn(sp)}\n""")
    sub("  $('#shBack').onclick = closeGame;\n", "  $('#shBack').onclick = closeGame;\n" + GAME_BIND)
    # 3. tennis match tile
    sub("""    <div class="abrow">${side(0)}${side(1)}</div>\n""", """    <div class="abrow">${side(0)}${side(1)}</div>${q ? simBtn('Tennis') : ''}\n""")
    sub("  $('#tnBack').onclick = () => { TSHEET = null; renderTennis(); };\n", "  $('#tnBack').onclick = () => { TSHEET = null; renderTennis(); };\n" + TEN_BIND)
# availability check (new desks and the v126 upgrade)
sub(V1_BTN, BTN)
sub(GAME_BIND, GAME_BIND + "  simShow(sp, el);\n")
sub(TEN_BIND, TEN_BIND + "  simShow('TENNIS', out);\n")
open(f, 'w', encoding='utf-8').write(s)
print('sim hook (with availability check) in', f)
