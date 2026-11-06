"""Concept E redesign (J 2026-10-29: Screener look + Scoreboard information, Shrikhand + Michroma; built 2026-10-30).
Makes app/bet-desk-v2.html from app/bet-desk.html: E palette and fonts, Home / Sports / Bets / Data nav, game rows with
Model A vs B, the game screen with "Why the models disagree", Look ahead, Tennis tab, Data tab views, the avg tag and the
CFB usage prior. The new code lives in app/redesign.js and is inlined before the boot section.
Run: python3 app/redesign_patch.py   (then test bet-desk-v2.html and copy it over bet-desk.html)"""
import os, re
HERE = os.path.dirname(os.path.abspath(__file__))
src = open(f'{HERE}/bet-desk.html').read()
if 'Concept E redesign' in src: raise SystemExit('bet-desk.html already has the redesign')

CSS = r"""
/* ---------- Concept E (J 2026-10-29): dark screener, Shrikhand display + Michroma details at 0.8 scale ---------- */
:root,:root:not([data-theme="light"]),:root[data-theme="dark"]{
  --bg:#0f1115; --surface:#161a21; --sunk:#1d222b; --ink:#d9dee7; --muted:#7e8796; --line:#232934;
  --accent:#3ddc84; --accent-ink:#062b1e; --good:#3ddc84; --bad:#ff6b6b; --warn:#ffc94a; --chip:#1f2630; --mb:#7aa7ff;
  --f-display:"Shrikhand",Georgia,serif; --f-body:"Michroma",system-ui,sans-serif; --f-num:"Michroma",ui-monospace,monospace; color-scheme:dark}
:root[data-theme="light"]{--bg:#f4f5f7; --surface:#fff; --sunk:#eceef2; --ink:#161a21; --muted:#5b6473; --line:#d9dde4;
  --accent:#138a4b; --accent-ink:#fff; --good:#138a4b; --bad:#c33; --warn:#9a6a00; --chip:#e7eaef; --mb:#2f5fc4; color-scheme:light}
body{font-size:12px;line-height:1.5;letter-spacing:.01em}
input,select,textarea{font:16px system-ui,-apple-system,"Segoe UI",sans-serif}
.brand h1{font:400 26px var(--f-display);text-transform:none;letter-spacing:0;color:var(--accent)}
.brand h1{white-space:nowrap;font-size:24px}.brand .save{font-size:10px;text-align:right}
#gameSheet,#v-slate .panel{scroll-margin-top:190px}
h2{font:400 21px var(--f-display);letter-spacing:0}
h3{font:600 10px var(--f-body);letter-spacing:.1em}
nav.tabs button{font:400 16px var(--f-display);text-transform:none;letter-spacing:.01em;padding:12px 4px}
.btn{font:600 11px var(--f-body);letter-spacing:.06em}
.btn.sm{padding:5px 9px;font-size:10px}
.chips button{font:600 11px var(--f-body);letter-spacing:.03em;min-height:36px}
.chips.sports{display:flex;overflow-x:auto}
.chips.sports button{flex:0 0 auto;padding:8px 13px;font-size:11px}
.seg.sub button{font:600 10px var(--f-body);letter-spacing:.06em}
.seg button{font:600 11px var(--f-body)}
.grow b,.side .name,.mkt .mh b{font:400 15px var(--f-display);letter-spacing:0;text-transform:none}
.score b{font:400 26px var(--f-display)}
.kpi b{font:600 15px var(--f-num)}
.kpi span,label{font-size:10px;letter-spacing:.06em}
th{font:600 10px var(--f-body)}
td{font-size:11px}
.pill{font:600 10px var(--f-body)}
.tag{font:600 9px var(--f-body);letter-spacing:.05em;padding:2px 7px}
.t-avg{border:1px solid var(--mb);color:var(--mb);text-transform:none}
.small{font-size:11px}
.note{font-size:11px}
.card,.panel{min-width:0}
.panel{background:var(--surface);border:1px solid var(--line);border-radius:10px;padding:12px;gap:8px}
#v-slate .panel{background:none;border:0;padding:0;gap:14px}
#v-slate .panel .panel,#gameSheet .panel{background:var(--surface);border:1px solid var(--line);border-radius:10px;padding:12px;gap:8px}
.panel.ah{border-style:dashed;border-color:var(--warn)}
.elist{display:flex;flex-direction:column;gap:6px}
.erow{background:var(--surface);border:1px solid var(--line);border-radius:10px;padding:8px 10px;display:grid;grid-template-columns:minmax(0,1fr) auto 2.6em;gap:1px 10px;align-items:center;text-align:left;color:var(--ink);font:inherit;width:100%;cursor:pointer}
.erow .t{font:400 16px var(--f-display);line-height:1.15;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
.erow .sc{font:400 19px var(--f-display);text-align:right;color:var(--ink)}
.erow .sb{color:var(--mb);font-size:11px;text-align:right}
.erow .meta{grid-column:1/-1;display:flex;justify-content:space-between;gap:8px;flex-wrap:wrap;font-size:10px;color:var(--muted);margin-top:3px}
.erow.tn{cursor:default}
.chip{display:inline-block;font:700 9.5px var(--f-body);padding:2px 7px;border-radius:999px;background:var(--accent);color:var(--accent-ink);white-space:nowrap}
.chip.y{background:var(--warn);color:#2b1d00}
.chip.b{background:var(--mb);color:#0b1a3a}
.ablg{display:flex;gap:6px;align-items:center;flex-wrap:wrap}
.ablg b{font-family:var(--f-display);font-weight:400}.ka{color:var(--ink)}.kb{color:var(--mb)}
.hsp{display:flex;flex-direction:column;gap:6px}.sph{font:400 17px var(--f-display);color:var(--accent)}
.gt{font-size:24px;line-height:1.15}
.abrow{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:8px}
.abx{background:var(--surface);border:1px solid var(--line);border-radius:10px;padding:10px;display:flex;flex-direction:column;gap:4px;min-width:0}
.abx.b{border-color:color-mix(in srgb,var(--mb) 45%,var(--line))}
.abx .lab{font-size:9.5px;letter-spacing:.08em;text-transform:uppercase;color:var(--muted)}
.abx .abs{display:flex;justify-content:space-between;align-items:baseline;gap:6px}
.abx .abs span{overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
.abx .abs b{font:400 24px var(--f-display)}
.abx.b .abs b{color:var(--mb)}
.abx .bar{height:6px;border-radius:999px;background:var(--chip);overflow:hidden}.abx .bar i{display:block;height:100%;background:var(--accent)}.abx.b .bar i{background:var(--mb)}
details.dis{border:1px solid var(--warn)!important}
details.dis>summary{cursor:pointer;color:var(--warn)}
details.dis>summary b{font:400 16px var(--f-display);color:var(--warn)}
details.dis p{border-left:2px solid var(--line);padding-left:8px}
table.et{font-size:10.5px}
table.et td,table.et th{padding:6px 5px;white-space:normal}
table.et td.t{font-family:var(--f-body)}
table.et td.pos,table.et td.neg{text-align:right;font-weight:700}
table.et .muted{font-size:9.5px}
table.et td.best{background:color-mix(in srgb,var(--accent) 16%,transparent);color:var(--accent);font-weight:700;border-radius:4px}
table.grid7 td,table.grid7 th{white-space:nowrap}
tr.click{cursor:pointer}
@media (max-width:420px){.erow .t{font-size:15px}.abx .abs b{font-size:21px}}
"""

HOME = """<section class="view" id="v-home" hidden>
  <div id="homeOut"><p class="small muted">Loading…</p></div>
</section>
"""

R = [  # (old, new, count)
  ('family=Barlow+Semi+Condensed:wght@500;600;700&family=Barlow:wght@400;500;600&family=JetBrains+Mono:wght@400;600&display=swap',
   'family=Shrikhand&family=Michroma&display=swap', 1),
  ('</style>', CSS + '</style>', 1),
  ('<section class="view" id="v-best" hidden>\n  <h2>Best bets</h2>',
   HOME + '<section class="view" id="v-best" hidden>\n  <div class="row" style="justify-content:space-between"><h2>All best bets</h2><button class="btn ghost sm" type="button" data-go="home">‹ Home</button></div>', 1),
  ('<div class="panel" id="sv-games" hidden>\n    <div id="gameList"></div>',
   '<div class="panel" id="sv-tennis" hidden><div id="tennisOut"></div></div>\n  <div class="panel" id="sv-games" hidden>\n    <div id="gameList"></div><div id="gameListE"></div><div id="gameSheet" hidden></div>', 1),
  ('<h2>Results</h2>', '<h2>Data</h2>', 1),
  ('''  <button role="tab" data-v="best" aria-selected="true">Best bets</button>
  <button role="tab" data-v="slate" aria-selected="false">Sports</button>
  <button role="tab" data-v="bets" aria-selected="false">Bets</button>
  <button role="tab" data-v="results" aria-selected="false">Results</button>''',
   '''  <button role="tab" data-v="home" aria-selected="true">Home</button>
  <button role="tab" data-v="slate" aria-selected="false">Sports</button>
  <button role="tab" data-v="bets" aria-selected="false">Bets</button>
  <button role="tab" data-v="results" aria-selected="false">Data</button>''', 1),
  ("const SPORTS = ['NFL','NBA','CBB','CFB','MLB','NHL','Soccer','Other'];", "const SPORTS = ['NFL','NBA','CBB','CFB','MLB','NHL','Soccer','Tennis','Other'];", 1),
  ("show($('#v-settings').hidden ? 'settings' : 'best')", "show($('#v-settings').hidden ? 'settings' : 'home')", 1),
  ("'betdesk.tab2'", "'betdesk.tab3'", 2),   # J opens on Home once, whatever tab was last saved
  ("show(localStorage.getItem('betdesk.tab3') || 'best'); } catch(e){ show('best'); }", "show(localStorage.getItem('betdesk.tab3') || 'home'); } catch(e){ show('home'); }", 1),
  ("const SPORT_LIST = ['NFL','NBA','NHL','CFB','CBB','MLB','Soccer'];", "const SPORT_LIST = ['NFL','NBA','NHL','CFB','CBB','MLB','Soccer','Tennis'];", 1),
  ("MLB:['games'], Soccer:['games','props']};", "MLB:['games'], Soccer:['games','props'], Tennis:['tennis']};", 1),
  ("const TABS = {games:['games'], props:['props','flags'], tools:['shop','promo','calc']}, TAB_OF = {games:'games', props:'props', flags:'props', shop:'tools', promo:'tools', calc:'tools'};",
   "const TABS = {games:['games'], props:['props','flags'], tools:['shop','promo','calc'], matches:['tennis']}, TAB_OF = {games:'games', props:'props', flags:'props', shop:'tools', promo:'tools', calc:'tools', tennis:'matches'};", 1),
  ("sp !== 'MLB' && sp !== 'Soccer' || t !== 'tools'", "sp !== 'MLB' && sp !== 'Soccer' && sp !== 'Tennis' || t !== 'tools'", 1),
  ("const VIEW_NAME = {tools:'Tools',", "const VIEW_NAME = {matches:'Matches', tools:'Tools',", 1),
  ("MLB:'MLB app', Soccer:'Soccer model'};", "MLB:'MLB app', Soccer:'Soccer model', Tennis:'Own read'};", 1),
  ("['games','shop','props','flags','promo','calc'].forEach(k => $('#sv-'+k).hidden = !on.includes(k));",
   "['games','shop','props','flags','promo','calc','tennis'].forEach(k => $('#sv-'+k).hidden = !on.includes(k)); if (on.includes('tennis')) renderTennis();", 1),
  ("let LASTHEAD = null, LASTMKTS = [];", "let LASTHEAD = null, LASTMKTS = [], LASTROWS = null;", 1),
  ("function gameRows(out, rows, sport, extraHtml, head){\n  LASTHEAD = head || null; LASTMKTS = [];", "function gameRows(out, rows, sport, extraHtml, head){\n  LASTHEAD = head || null; LASTMKTS = []; LASTROWS = rows;", 1),
  ("blend:hd.blend || {hs:hd.hs, as:hd.as}, edges:[]};",
   "blend:hd.blend || {hs:hd.hs, as:hd.as}, edges:[]};\n          { const mr = (LASTROWS || []).find(q => mktKind(q) === 'ml'); r.pml = mr ? mr[2] : null; }   // home win chance from the card (Model A/B rows)", 1),
  ("const m = (neu ? (cfbMargin(C,h,a,0) - cfbMargin(C,a,h,0))/2 : cfbMargin(C,h,a,1)) + adj;",
   "const m = (neu ? (cfbMargin(C,h,a,0) - cfbMargin(C,a,h,0))/2 : cfbMargin(C,h,a,1)) + adj + cfbUse(h[0], a[0]);   // + usage prior (J 2026-10-29)", 1),
  ("gameRows(out, R, 'CFB', txt\n", "gameRows(out, R, 'CFB', txt + cfbUseTxt(h[0], a[0])\n", 1),
  ("'Old CBB model (shadow)'", "'Model B: previous CBB model'", 1),
  ("The previous model is shown under it and graded as a shadow.", "The previous model is shown beside it as Model B.", 1),
  ("<b>${tLogo(sp, x.p.t)}${esc(x.p.n)}</b> ${PF_TAG[x.kind]}", "<b>${tLogo(sp, x.p.t)}${esc(x.p.n)}</b> ${PF_TAG[x.kind]}${avgTag(sp, x.p, x.k)}", 1),
  ("R.push({type:'props', sp, kind:x.kind, k, ev:", "R.push({type:'props', avg:avgTag(sp, x.p, x.k), sp, kind:x.kind, k, ev:", 1),
  ("<b>${esc(r.title)}</b> ${BB_TAG[r.kind] || ''}", "<b>${esc(r.title)}</b> ${BB_TAG[r.kind] || ''}${r.avg || ''}", 1),
  ("b.innerText.split('\\n')[0].trim() === `${g.a} @ ${g.h}`", "b.querySelector('b') && b.querySelector('b').textContent.trim() === `${g.a} @ ${g.h}`", 1),
  ("'props_NHL.json','cbb_v2.json'];", "'props_NHL.json','cbb_v2.json','model_a.json','cfb_usage.json','tennis.json','tennis_model.json'];", 1),
  ("  initGames();\n  refreshSport(true); renderModels(); renderBest();",
   "  { const j = await getJ('cfb_usage.json'); if (j) S.cfbUse = j; }\n  { const j = await getJ('model_a.json'); if (j) MA = j; }\n  { const j = await getJ('tennis.json'); if (Array.isArray(j)) TEN = j; }\n  { const j = await getJ('tennis_model.json'); if (j) TENM = j; }\n  initGames();\n  refreshSport(true); renderModels(); renderBest(); renderHome();", 1),
]
out = src
for old, new, n in R:
    c = out.count(old)
    if c != n: raise SystemExit(f'expected {n} of {old[:80]!r}, found {c}')
    out = out.replace(old, new)
js = open(f'{HERE}/redesign.js').read()
mark = '// ---------- boot ----------'
assert out.count(mark) == 1
out = out.replace(mark, js + '\n' + mark)
open(f'{HERE}/bet-desk-v2.html', 'w').write(out)
print('bet-desk-v2.html', len(out), 'bytes,', len(R), 'replacements + redesign.js')
