"""Bet Desk line shop (J 2026-10-27, thread "Build line shopping and middling tools"): adds Slate > Line shop, the game
card's Every book table, Settings > My books, Bets > Middle watch and the middle/arb/hedge calculators. The code lives in
line_shop.js and is inlined here. python3 line_shop_patch.py [bet-desk.html]; safe to run twice, stops if an anchor moved."""
import sys, os
here = os.path.dirname(os.path.abspath(__file__))
path = sys.argv[1] if len(sys.argv) > 1 else f'{here}/bet-desk.html'
s = open(path, encoding='utf-8').read()
JS = open(f'{here}/line_shop.js', encoding='utf-8').read()
CALC = '''    <h3 style="margin-top:12px">Middle calculator</h3>
    <div class="grid">
      <label>Sport<select id="lcSp"><option>NFL</option><option>CFB</option><option>NBA</option><option>CBB</option></select></label>
      <label>Market<select id="lcMk"><option value="spread">Spread</option><option value="total">Total</option></select></label>
      <label>Team 1 spread<input id="lcL1" class="num" inputmode="decimal" placeholder="-2.5"></label>
      <label>Price<input id="lcP1" class="num" inputmode="numeric" placeholder="-110"></label>
      <label>Team 2 spread<input id="lcL2" class="num" inputmode="decimal" placeholder="+3.5"></label>
      <label>Price<input id="lcP2" class="num" inputmode="numeric" placeholder="-110"></label>
    </div><div id="lcMid"></div>
    <h3 style="margin-top:12px">Arb calculator</h3>
    <div class="grid">
      <label>Side 1 price<input id="lcA1" class="num" inputmode="numeric" placeholder="+105"></label>
      <label>Side 2 price<input id="lcA2" class="num" inputmode="numeric" placeholder="+102"></label>
    </div><div id="lcArb"></div>
    <h3 style="margin-top:12px">Bonus bet hedge</h3>
    <div class="grid">
      <label>Bonus bet price<input id="lcBp" class="num" inputmode="numeric" placeholder="+250"></label>
      <label>Hedge price (cash)<input id="lcHp" class="num" inputmode="numeric" placeholder="-280"></label>
      <label>Bonus amount<input id="lcBa" class="num" inputmode="decimal" placeholder="100"></label>
    </div><div id="lcHedge"></div>
'''
EDITS = [
 ('css', '</style>', '.click{cursor:pointer}.chk{display:inline-flex;gap:6px;align-items:center;min-height:40px;padding:4px 12px;border:1px solid var(--line);border-radius:999px;font-size:14px}.btn.sm{padding:4px 8px;font-size:12px}\n</style>'),
 ('shop panel', '''      <div id="promoOut"></div>
    </div>
  </div>''', '''      <div id="promoOut"></div>
    </div>
  </div>
  <div class="panel" id="sv-shop" hidden>
    <div class="card"><h3>Line shop</h3><div id="shopOut"></div></div>
  </div>'''),
 ('middle watch card', '  <div class="betlist" id="betList"></div>', '  <div class="card" id="midWatch" hidden></div>\n  <div class="betlist" id="betList"></div>'),
 ('calculators', '''    <div id="mxHold"></div>
  </details>''', '''    <div id="mxHold"></div>
''' + CALC + '''  </details>
  <div class="card"><h3>My books</h3><p class="small muted">Tick the books you can bet. Line shop, Middle watch and Every book use only these. Starts with the Colorado books; Pinnacle and BetOnline are fair-price references only.</p><div id="myBooks"></div></div>'''),
 ('views', "const VIEWS = {NFL:['games','props','flags','parlays','promo'], NBA:['games','props','flags','parlays','promo'], NHL:['props','flags','parlays','promo'],\n  CFB:['games','parlays','promo'], CBB:['games','parlays','promo'],",
           "const VIEWS = {NFL:['games','shop','props','flags','parlays','promo'], NBA:['games','shop','props','flags','parlays','promo'], NHL:['props','flags','parlays','promo'],\n  CFB:['games','shop','parlays','promo'], CBB:['games','shop','parlays','promo'],"),
 ('view name', "const VIEW_NAME = {games:'Games', props:'Props',", "const VIEW_NAME = {games:'Games', shop:'Line shop', props:'Props',"),
 ('subbar count', "const cnt = {flags: flagCount(), parlays: PL.length, games: edgeCount()};", "const cnt = {flags: flagCount(), parlays: PL.length, games: edgeCount(), shop: shopCount()};"),
 ('subbar panels', "['games','props','flags','parlays','promo'].forEach(k => $('#sv-'+k).hidden = k !== cur);", "['games','shop','props','flags','parlays','promo'].forEach(k => $('#sv-'+k).hidden = k !== cur);"),
 ('refresh', "renderParlay(); renderGameList(); renderPromo();\n", "renderParlay(); renderGameList(); renderPromo(); renderShop();\n"),
 ('every book', "  }).join('')}</div>` + extraHtml;", "  }).join('')}</div>` + extraHtml + everyBook(sport);"),
 ('card log game', "pm:x.pm, pf:x.pf, pb:x.pb, ...(GEXTRA[sport]||{})});", "pm:x.pm, pf:x.pf, pb:x.pb, ...(GCFG[sport] ? {gm:{h:$(GCFG[sport].ids[0]).value, a:$(GCFG[sport].ids[1]).value}} : {}), ...(GEXTRA[sport]||{})});"),
 ('keep gm', "...(b.push?{push:b.push}:{})});", "...(b.push?{push:b.push}:{}), ...(b.gm?{gm:b.gm}:{})});"),
 ('tracker', "function renderTracker(){\n", "function renderTracker(){\n  renderMidWatch();\n"),
 ('settings keep books', "trustTeam:Math.min(100,Math.max(0,g('#sTrustTeam',DEF.trustTeam)))};", "trustTeam:Math.min(100,Math.max(0,g('#sTrustTeam',DEF.trustTeam))), ...(S.settings.booksOff ? {booksOff:S.settings.booksOff} : {})};"),
 ('init calc', "  addSignButtons([$('#mxA'), $('#mxKo')]); renderMx();\n}", "  addSignButtons([$('#mxA'), $('#mxKo')]); renderMx(); initLtCalc();\n}"),
 ('files', "const FILES = ['game_slates.json',", "const FILES = ['line_tools.json','landing.json','game_slates.json',"),
 ('load', "  { const j = await getJ('promo_picks.json'); if (j) PROMO = j; }\n", "  { const j = await getJ('promo_picks.json'); if (j) PROMO = j; }\n  { const j = await getJ('line_tools.json'); if (j) LT = j; }\n  { const j = await getJ('landing.json'); if (j) LAND = j; }\n  renderMyBooks(); renderShop(); renderMidWatch();\n"),
 ('redraw', "function redrawStakes(){ renderProp();", "function redrawStakes(){ renderMyBooks(); LTC.clear(); renderShop(); renderProp();"),
 ('code', "// ---------- boot ----------", JS + "\n// ---------- boot ----------"),
]
done = 0
for name, old, new in EDITS:
    if new in s: print('already applied:', name); continue
    n = s.count(old)
    if n != 1: sys.exit(f'STOP: anchor for "{name}" found {n} times; nothing written')
    s = s.replace(old, new, 1); done += 1
open(path, 'w', encoding='utf-8').write(s)
print(done, 'edits written to', path)
