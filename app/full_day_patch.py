"""J 2026-11-06: list the whole day's games after they start, in every sport (NBA already did), so a missed bet can still be
logged and the data reviewed. Asserted string swaps on app/redesign.js (refuses to run twice); then
python3 app/redesign_inline.py builds app/bet-desk-v2.html. The data side (games kept after kickoff, frozen at their last
pre-game record) is models/model-b/pregame.py, called by export_slates.py, op_live.py (tennis) and models/*_ab/live.py.

Desk changes:
- Sports > game list: upcoming games first, then "Started and final today" (kickoff within 30 h): status chip (Live / Final
  with the score when the schedule has it), the pre-game Model B pick and desk edge labelled "pre-game". Tapping one opens the
  same tile, with a note that every number is as of the last pre-game update; Log buttons work as before.
- NFL: today's started games stay listed even when the desk's week moves on (Monday night after kickoff).
- Tennis: matches that started in the last 30 h stay listed under the upcoming ones, marked Started.
Home, Best bets, Look ahead and Unproven still list only games not started (those are bets you can still make)."""
import os, sys
HERE = os.path.dirname(os.path.abspath(__file__)); P = f'{HERE}/redesign.js'
s = open(P).read()
if 'J 2026-11-06 full day' in s: sys.exit('full_day_patch already applied')

def sw(a, b):
    global s
    assert s.count(a) == 1, f'expected once: {a[:90]}'
    s = s.replace(a, b)

# 1. helpers (after whenOf)
sw("""const whenOf = x => [x.d && x.d !== localDay() ? dayName(x.d) : '', kick(x)].filter(Boolean).join(' ');""",
"""const whenOf = x => [x.d && x.d !== localDay() ? dayName(x.d) : '', kick(x)].filter(Boolean).join(' ');
// J 2026-11-06 full day: started games stay listed (kickoff within DAY_H) so a missed bet can be logged and the data reviewed
const DAY_H = 30, DUR_H = {NFL:3.5, CFB:4, NBA:2.75, CBB:2.5, Tennis:2.5};
const gsOf = (sp, x) => gsGames(sp).find(y => String(y.id) === String(x.g ?? x.id)) || gsGames(sp).find(y => y.h === x.home && y.a === x.away) || null;
const recentStart = x => started(x) && Date.now() - Date.parse(x.k) <= DAY_H*36e5;
function gStat(sp, x){   // null = not started; else {t: chip text, fin}
  if (!started(x)) return null;
  const f = gsOf(sp, x), fs = f && f.fs;
  if (fs) return {t:`Final ${fs[1]}-${fs[0]}`, fin:true};
  return Date.now() - Date.parse(x.k) > (DUR_H[sp] || 3)*36e5 ? {t:'Ended', fin:true} : {t:'Live', fin:false};
}
const statChip = st => st ? `<span class="chip st${st.fin ? '' : ' lv'}">${esc(st.t)}</span> ` : '';
document.head.insertAdjacentHTML('beforeend', '<style>.chip.st{background:transparent;color:var(--muted);border:1px solid currentColor}.chip.st.lv{color:var(--good)}.erow.past{opacity:.82}.elh{margin:14px 0 6px}</style>');
// NFL: the desk's week moves on once every game of it has kicked off (nflWk), which would drop tonight's game; keep today's started ones
const _slateGamesFD = slateGames;
slateGames = function(sp){
  const gs = _slateGamesFD(sp); if (sp !== 'NFL' || S.day) return gs;
  gsGames('NFL').filter(f => recentStart(f) && !gs.some(y => String(y.g) === String(f.id) || (y.home === f.h && y.away === f.a)))
    .forEach(f => gs.push({g:f.id, d:f.d, k:f.k, home:f.h, away:f.a, n:f.n || 0, feed:f.ln || null}));
  return gs.sort((a, b) => (a.k || a.d || '').localeCompare(b.k || b.d || '') || a.home.localeCompare(b.home));
};""")

# 2. eRow: status chip, pre-game picks on started games
sw("""  const e = r && r.edges && r.edges[0] && !started(x) ? {t:r.edges[0].name, ev:r.edges[0].ev} : null, g = ltFor(sp, x), b = !e && g ? ltTop(sp, g) : null;
  const bet = e || (b && b.mev >= PF_THR && b.bev >= PF_THR ? {t:sideLbl(g, b.mk, b.s, b.pt), ev:b.bev} : null);""",
"""  const st = gStat(sp, x), pre = st ? 'pre-game ' : '';
  const e = r && r.edges && r.edges[0] ? {t:r.edges[0].name, ev:r.edges[0].ev} : null, g = !st && ltFor(sp, x), b = !e && g ? ltTop(sp, g) : null;
  const bet = e || (b && b.mev >= PF_THR && b.bev >= PF_THR ? {t:sideLbl(g, b.mk, b.s, b.pt), ev:b.bev} : null);""")
sw("""  return `<button type="button" class="erow" data-i="${i}">""", """  return `<button type="button" class="erow${st ? ' past' : ''}" data-i="${i}">""")
sw("""    <span class="meta"><span>${esc(whenOf(x))}${fav ? ` · ${esc(fav[0])} ${pct(fav[1], 0)}` : ''}</span><span>${ab.big ? `<span class="chip y">A/B gap</span> ` : ''}${AB2(sp) ? (cbt ? `<span class="chip${cbt[1].tier === 'A' ? '' : ' y'}">${CAB_T[cbt[1].tier]}: """,
"""    <span class="meta"><span>${statChip(st)}${esc(whenOf(x))}${fav ? ` · ${esc(fav[0])} ${pct(fav[1], 0)}` : ''}</span><span>${ab.big ? `<span class="chip y">A/B gap</span> ` : ''}${AB2(sp) ? (cbt ? `<span class="chip${cbt[1].tier === 'A' ? '' : ' y'}">${pre}${CAB_T[cbt[1].tier]}: """)
sw("""(cu => cu ? `<span class="chip u">Unproven ${cu[1].u.score}: """, """(cu => cu ? `<span class="chip u">${pre}Unproven ${cu[1].u.score}: """)
sw("""bet ? `<span class="chip">${esc(bet.t)} ${sgn(bet.ev)}</span>` : ''}${star}""", """bet ? `<span class="chip">${pre}${esc(bet.t)} ${sgn(bet.ev)}</span>` : ''}${star}""")

# 3. game list: upcoming first, then today's started and final games
sw("""  const gs = C ? slateGames(sp).filter(x => !started(x)) : [];
  if (!gs.length){ box.innerHTML = ''; return; }""",
"""  const all = C ? slateGames(sp).filter(x => !started(x) || recentStart(x)) : [];
  const gs = [...all.filter(x => !started(x)), ...all.filter(x => started(x))];   // upcoming first, then the day's started games
  if (!gs.length){ box.innerHTML = ''; return; }""")
sw("""  box.innerHTML = abLegend(sp) + `<div class="elist">${gk.map((x, i) => eRow(sp, x, pr[i], i)).join('')}</div>` + (hid ?""",
"""  const nUp = gk.filter(x => !started(x)).length, rows = (a, z) => gk.slice(a, z).map((x, i) => eRow(sp, x, pr[a + i], a + i)).join('');
  box.innerHTML = abLegend(sp) + (nUp ? `<div class="elist">${rows(0, nUp)}</div>` : '<p class="small muted">No games left to start today.</p>')
    + (gk.length > nUp ? `<p class="small muted elh"><b>Started and final</b> · picks as of the last pre-game update. Tap one to review it or log a bet you missed.</p><div class="elist">${rows(nUp, gk.length)}</div>` : '') + (hid ?""")

# 4. tile: say the numbers are pre-game
sw("""    <h2 class="gt">${tLogo(sp, x.away)}${esc(x.away)} <span class="muted">@</span> ${tLogo(sp, x.home)}${esc(x.home)}</h2>${mk ? `<p class="small muted">Market ${esc(mk)}</p>` : ''}""",
"""    <h2 class="gt">${tLogo(sp, x.away)}${esc(x.away)} <span class="muted">@</span> ${tLogo(sp, x.home)}${esc(x.home)}</h2>${mk ? `<p class="small muted">${started(x) ? 'Last pre-game market' : 'Market'} ${esc(mk)}</p>` : ''}
    ${(st => st ? `<p class="small">${statChip(st)}Started ${esc(kick(x))}. Model A, Model B and the market below are as of the last update before it started${(cg => cg && cg.pre ? ` (${esc(String(cg.pre).slice(0, 16).replace('T', ' '))}${/UTC|Z$/.test(String(cg.pre)) ? '' : ' UTC'})` : '')(AB2(sp) ? cabGame(x, sp) : null)}. You can still log a bet you placed.</p>` : '')(gStat(sp, x))}""")

# 5. tennis: matches that started in the last 30 h stay listed under the upcoming ones
sw("""  const now = Date.now(), all = TEN.filter(m => Date.parse(m.start) > now).sort((a, b) => a.start.localeCompare(b.start));""",
"""  const now = Date.now(), all = [...TEN.filter(m => Date.parse(m.start) > now).sort((a, b) => a.start.localeCompare(b.start)),
    ...TEN.filter(m => Date.parse(m.start) <= now && now - Date.parse(m.start) <= DAY_H*36e5).sort((a, b) => b.start.localeCompare(a.start))];   // J 2026-11-06 full day""")
sw("""    return `<button type="button" class="erow tn" data-i="${i}">${side(0)}${side(1)}<span class="meta"><span>${esc(kick({k:m.start}))}""",
"""    const st = Date.parse(m.start) <= now ? gStat('Tennis', {k:m.start}) : null;
    return `<button type="button" class="erow tn${st ? ' past' : ''}" data-i="${i}">${side(0)}${side(1)}<span class="meta"><span>${statChip(st && {t:st.fin ? 'Started' : 'Live', fin:st.fin})}${esc(kick({k:m.start}))}""")
sw("""'<div class="empty">No upcoming matches in this category.</div>'}`;""", """'<div class="empty">No matches today in this category.</div>'}`;""")

open(P, 'w').write(s)
print('full_day_patch applied to', P)
