"""Bet Desk CBB v2 as the main CBB model (J 2026-10-27, thread "Rebuild score projections for every sport": the model that won
the backtests leads; the old models/cbb ratings become the shadow). Games in cbb_v2.json (models/rebuild/basketball/
live_cbb.py) use its hs/as/poss as "Our model"; other games and two teams typed by hand use cbb_ratings.json, labeled
previous model. Sigma, blend weights and opener tiers are unchanged. python3 cbb_v2_patch.py [bet-desk.html]."""
import sys, os
here = os.path.dirname(os.path.abspath(__file__))
path = sys.argv[1] if len(sys.argv) > 1 else f'{here}/bet-desk.html'
s = open(path, encoding='utf-8').read()
EDITS = [
 ('calc', """  const hca = neutral ? 0 : C.hca, poss = h[3] + a[3] - C.T0;
  const ph = (C.mu + h[1] - a[2] + hca) * poss / 100, pa = (C.mu + a[1] - h[2] - hca) * poss / 100, m = ph - pa, tot = ph + pa;""",
  """  const hca = neutral ? 0 : C.hca, possO = h[3] + a[3] - C.T0;
  const phO = (C.mu + h[1] - a[2] + hca) * possO / 100, paO = (C.mu + a[1] - h[2] - hca) * possO / 100;
  const V = cbbV2(h[0], a[0], neutral), poss = V ? V.poss : possO, ph = V ? V.hs : phO, pa = V ? V.as : paO, m = ph - pa, tot = ph + pa;
  const v2Txt = V ? `<p class="small">Our model: CBB v2 (the previous CBB model stacked with the rebuild; won the 2020-26 backtests). Previous model: ${esc(h[0])} ${phO.toFixed(1)}, ${esc(a[0])} ${paO.toFixed(1)}.</p>`
    : `<p class="small muted">Previous CBB model: this game is not in the CBB v2 file${S.cbb2b ? '' : ' (not loaded)'}.</p>`;"""),
 ('note', "  gameRows(out, R, 'CBB', gapTxt + `<p class=\"note\">Spread and total chances", "  gameRows(out, R, 'CBB', v2Txt + gapTxt + `<p class=\"note\">Spread and total chances"),
 ('fn', "function renderCbbPrice(){", """// CBB v2 game by teams and site (cbb_v2.json games keyed by slate id: {h, a, d, n, hs, as, poss, today_hs, today_as})
function cbbV2(h, a, neutral){
  const G = S.cbb2b && S.cbb2b.games; if (!G) return null;
  const k = x => String(x).toLowerCase(), day = localDay();
  const c = Object.values(G).filter(g => k(g.h) === k(h) && k(g.a) === k(a) && !!g.n === !!neutral && g.hs != null);
  return c.sort((p, q) => (p.d < day) - (q.d < day) || String(p.d).localeCompare(String(q.d)))[0] || null;
}
function renderCbbPrice(){"""),
 ('files', "'props_NFL.json','props_NBA.json','props_NHL.json'];", "'props_NFL.json','props_NBA.json','props_NHL.json','cbb_v2.json'];"),
 ('load', "  { const j = await getJ('cbb_ratings.json'); if (j) S.cbb = j; }", "  { const j = await getJ('cbb_ratings.json'); if (j) S.cbb = j; }\n  { const j = await getJ('cbb_v2.json'); if (j) S.cbb2b = j; }"),
 ('explainer', "    <li><b>CBB card</b> (Games tab): preseason ratings built from returning and transfer production.",
  "    <li><b>CBB card</b> (Games tab): CBB v2${S.cbb2b ? ` (${Object.keys(S.cbb2b.games).length} games, made ${esc(S.cbb2b.made)})` : ''} is the main model: the previous CBB ratings stacked with the rebuild (2020-26 backtest margin error 8.87 vs 8.91, totals 13.24 vs 13.34; 2026 DraftKings openers with a 4+ point gap 63.9% ATS, totals 4+ 54.5% with CLV +1.78). The previous model is shown under it and graded as a shadow. The previous model: preseason ratings built from returning and transfer production."),
]
if 'function cbbV2' in s: sys.exit('already patched')
for name, a, b in EDITS:
    if s.count(a) != 1: sys.exit(f'anchor {name!r} found {s.count(a)} times')
    s = s.replace(a, b)
open(path, 'w', encoding='utf-8').write(s)
print('patched', path)
