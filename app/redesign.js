// ---------- Concept E redesign (J 2026-10-29: Screener look + Scoreboard information; Friday build 2026-10-30) ----------
// Home feed (best bets, Look ahead, today's games), game rows with Model A vs Model B, a game screen with "Why the models
// disagree" (open by default on big gaps), this game's best bets and a book grid, a Tennis tab, the Data tab's Model A vs B
// and not-placed views, the "well above his average" tag and the CFB usage prior. Inlined by redesign_patch.py.
// Model A = the score rebuild (app/data/model_a.json from export_model_a.py) for NFL, NBA, NHL and CFB (v4). For CBB the
// desk card runs CBB v2, which is Model A there, and model_a.json carries the previous CBB model as Model B.
let MA = null, TEN = null, TENM = null, SHEET = null;
const AB_DESK = sp => sp === 'CBB' ? 'A' : 'B';
const AB_SD = {NFL:13.5, CFB:15.5, NBA:12, CBB:10.5};
const AB_GAP = {NFL:[3, 4], CFB:[4, 6], NBA:[4, 6], CBB:[3, 5], NHL:[0.6, 0.8]};   // margin, total: a gap this big opens the why box
const AB_LAB = sp => sp === 'CBB' ? ['Scores · v2 + sim', 'Bets · tier'] : sp === 'NFL' ? ['Scores · rebuild + key numbers', 'Bets · tier'] : sp === 'NBA' && cabOn('NBA') ? ['Scores · rebuild + sim', 'Bets · tier'] : sp === 'CFB' && cabOn('CFB') ? ['Scores · v4 + drive sim', 'Bets · tier'] : sp === 'CFB' ? ['Rebuild v4', cfVer === 'v2' ? 'CFB v2 (yours)' : 'Desk v3'] : ['Rebuild', 'Desk model'];
function maGame(sp, x){
  const M = MA && MA.sports && MA.sports[sp]; if (!M) return null;
  const id = String(x.g ?? x.id ?? ''), h = x.home ?? x.h, a = x.away ?? x.a;
  return M.games.find(g => String(g.id) === id) || M.games.find(g => g.h === h && g.a === a) || null;
}
function poisWin(lh, la){   // NHL: P(home wins), ties split (overtime)
  const f = l => { const p = [Math.exp(-l)]; for (let k = 1; k <= 14; k++) p.push(p[k-1]*l/k); return p; };
  const H = f(lh), A = f(la); let w = 0, t = 0;
  H.forEach((ph, i) => A.forEach((pa, j) => { if (i > j) w += ph*pa; else if (i === j) t += ph*pa; })); return w + t/2;
}
const abWin = (sp, hs, as) => sp === 'NHL' ? poisWin(hs, as) : ncdf((hs - as)/(AB_SD[sp] || 12));
// both models for one game: {A, B, gapM, gapT, big}; each model {hs, as, p (home win), est (chance from the score, not the card)}
function abOf(sp, x, r){
  const desk = r ? {hs:r.raw.hs, as:r.raw.as, p:r.pml ?? abWin(sp, r.raw.hs, r.raw.as), est:r.pml == null} : null;
  if ((sp === 'NFL' || sp === 'NBA' || sp === 'CFB') && cabOn(sp)){ const G = cabGame(x, sp), m = maGame(sp, x);   // NFL, NBA, CFB (J 2026-11-05): Model A = {nfl,nba,cfb}_ab.json scores, Model B = its bet engine; the desk model stays in the full card below
    return {A:G && G.A ? {hs:G.A.hs, as:G.A.as, p:G.A.p_home, est:false, g:G} : m ? {hs:m.hs, as:m.as, p:abWin(sp, m.hs, m.as), est:true, g:m} : null, B:null}; }
  const m = sp === 'CBB' ? null : maGame(sp, x), other = m ? {hs:m.hs, as:m.as, p:abWin(sp, m.hs, m.as), est:true, g:m} : null;   // CBB: Model B is the bet engine (cbb_ab.json), not a second score
  const [A, B] = AB_DESK(sp) === 'A' ? [desk, other] : [other, desk], o = {A, B};
  if (A && B){ o.gapM = (A.hs - A.as) - (B.hs - B.as); o.gapT = (A.hs + A.as) - (B.hs + B.as);
    const g = AB_GAP[sp] || [3, 5]; o.big = Math.abs(o.gapM) >= g[0] || Math.abs(o.gapT) >= g[1]; }
  return o;
}
const abF = (sp, v) => v == null ? '–' : sp === 'NHL' ? v.toFixed(1) : String(Math.round(v));
const abTm = (x, m) => Math.abs(m) < 0.05 ? 'even' : m > 0 ? `${x.home} by ${m.toFixed(1)}` : `${x.away} by ${(-m).toFixed(1)}`;
// ---------- CBB Model A + Model B (J 2026-11-02: CBB is first on the two-model system; models/cbb_ab/live.py -> cbb_ab.json) ----------
// A = projected scores (CBB v2 means + possession sim ranges). B = bets built on A: devigged sharp-weighted market number, the share of
// our gap that has been real (confidence), line age, EV at Colorado books, tier A = bet, B = lean (tracked, not a bet), C = pass.
// NFL joined 2026-11-05 (models/nfl_ab/live.py -> nfl_ab.json): same boxes and tiers; NFL game tiers max B (lean) until live tracking proves them.
// NBA joined 2026-11-05 (models/nba_ab/live.py -> nba_ab.json): rebuild + component fix + possession sim; tiers from the opener tests.
// CFB joined 2026-11-05 (models/cfb_ab/live.py -> cfb_ab.json): v4 + usage prior + missing starters in the score, drive-sim ranges, key-number curve; opener-tested tiers.
const CAB_F = {CBB:'cbb_ab.json', NFL:'nfl_ab.json', NBA:'nba_ab.json', CFB:'cfb_ab.json'}, CABS = {};
const AB2 = sp => sp in CAB_F, cabOn = sp => !!(CABS[sp] && CABS[sp].games);
function cabLoad(sp){ (sp ? [sp] : Object.keys(CAB_F)).forEach(s => { if (CABS[s] !== undefined) return; CABS[s] = false;
  fetch(CAB_F[s], {cache:'no-cache'}).then(r => r.ok ? r.json() : null).then(j => { if (j && j.games){ CABS[s] = j; try { renderGameList(); renderHome(); renderBest(); } catch(e){} } }).catch(() => {}); }); }
const cabGame = (x, sp = 'CBB') => { const C = CABS[sp]; if (!C || !C.games) return null; const id = String(x.g ?? x.id ?? '');
  return C.games[id] || Object.values(C.games).find(g => g.h === (x.home ?? x.h) && g.a === (x.away ?? x.a)) || null; };
const CAB_T = {A:'Bet', B:'Lean', C:'Pass'};
// UNPROVEN rating on Model B passes (J 2026-11-06; models/model-b/unproven.py adds z.u to tier C picks): score 0-100 (50 = no edge),
// rec play (small) / watch / skip, the reason it is unproven and the sample behind it. Never a tier A/B bet: dashed grey chip, kept off
// Best bets and Look ahead, graded apart in each sport's grade.json (spread_U_play, ...). Props dropped by the tier rules show as Unproven.
const U_T = {play:'Small play', watch:'Watch', skip:'Skip'};
document.head.insertAdjacentHTML('beforeend', '<style>.chip.u,.tag.t-un{background:transparent;color:var(--muted);border:1px dashed currentColor}</style>');
const uChip = u => u ? `<span class="chip u" title="Unproven: not a tested bet">Unproven · ${U_T[u.rec]} ${u.score}</span>` : '';
const uLine = u => !u ? '' : `Unproven ${u.score}/100 · ${U_T[u.rec]}. Model A alone: ${pct(u.pA, 0)} chance, EV ${sgn(u.evA)}; halfway between A and B ${sgn(u.evU)}; Model B ${sgn(u.evB)}. Why it is unproven: ${u.why}.`;
const cabU = G => G && G.B ? ['spread', 'total', 'ml'].map(k => [k, G.B[k]]).filter(([k, z]) => z && z.bet && z.tier === 'C' && z.u && z.u.rec !== 'skip') : [];
const uShow = (k, z) => z && (k !== 'ml' || z.tier !== 'C' || (z.u && z.u.rec !== 'skip'));
const cabPick = (k, z) => !z || !z.bet ? '' : k === 'ml' ? `${z.side} ML` : k === 'total' ? `${z.side} ${z.bet.pt}` : `${z.side} ${sgnPts(z.bet.pt)}`;
const cabBets = G => G && G.B ? ['spread', 'total', 'ml'].map(k => [k, G.B[k]]).filter(([k, z]) => z && z.bet && z.tier !== 'C') : [];
function cabBox(sp, x){
  const G = cabGame(x, sp), B = G && G.B;
  if (!B) return `<div class="abx b"><span class="lab">Model B · bets</span><p class="small muted">No lines posted for this game yet.</p></div>`;
  const row = (k, z) => !z ? '' : `<div class="small">${{spread:'Spread', total:'Total', ml:'Moneyline'}[k]} <span class="chip ${z.tier === 'A' ? '' : z.tier === 'B' ? 'y' : 'b'}">${z.tier} · ${CAB_T[z.tier]}</span> ${z.tier === 'C' ? uChip(z.u) : ''}</div><b style="font:400 19px var(--f-display)">${esc(cabPick(k, z) || '–')}</b>
    <p class="small">${z.bet ? `${esc(z.bet.book_name)} ${amS(z.bet.price)} · our chance ${pct(z.bet.p, 0)} · EV ${sgn(z.bet.ev)}` : 'No price posted yet'}</p>
    <p class="small muted">Market ${k !== 'total' ? esc(abTm(x, z.market)) : z.market.toFixed(1)} · gap ${Math.abs(z.gap).toFixed(1)}, ${Math.abs(z.real).toFixed(1)} of it trusted · ${sp === 'NFL' || sp === 'CFB' ? (z.fresh ? 'look-ahead line' : 'game-week line') : z.fresh ? 'new line' : 'older line'}</p>`;
  return `<div class="abx b"><span class="lab">Model B · bets</span>${row('spread', B.spread)}${row('total', B.total)}${uShow('ml', B.ml) ? row('ml', B.ml) : ''}</div>`;
}
function cabRange(sp, x){
  const G = cabGame(x, sp), A = G && G.A; if (!A) return '';
  if (sp === 'CFB') return `<p class="small muted">80% of drive sims: ${esc(x.away)} ${A.a10}–${A.a90}, ${esc(x.home)} ${A.h10}–${A.h90} · total ${A.t10}–${A.t90} · lands on 3: ${pct(A.p3, 0)}, on 7: ${pct(A.p7, 0)}</p>`;
  if (sp === 'NFL') return `<p class="small muted">80% range: ${esc(x.away)} ${A.a10}–${A.a90}, ${esc(x.home)} ${A.h10}–${A.h90} · total ${A.t10}–${A.t90} · lands on 3: ${pct(A.p3, 0)}, on 7: ${pct(A.p7, 0)}</p>`;
  return `<p class="small muted">80% of sims: ${esc(x.away)} ${A.a10}–${A.a90}, ${esc(x.home)} ${A.h10}–${A.h90} · total ${A.t10}–${A.t90} · OT ${pct(A.p_ot, 0)}</p>`;
}
function cabWhy(sp, x){
  const G = cabGame(x, sp), B = G && G.B; if (!B) return '';
  const L = [], gd = {'+':'pos', '-':'neg', '·':'muted'};
  [['spread', 'Spread'], ['total', 'Total'], ['ml', 'Moneyline']].forEach(([k, n]) => { const z = B[k]; if (!uShow(k, z)) return;
    L.push(`<p class="small"><b>${n}: tier ${z.tier} (${CAB_T[z.tier]})</b> · Model A ${k !== 'total' ? esc(abTm(x, z.ours)) : z.ours.toFixed(1)}, market ${k !== 'total' ? esc(abTm(x, z.market)) : z.market.toFixed(1)} (devigged, sharp books weighted up to 1.5x).${z.share != null ? ` Trusted share of the gap here ${Math.round(z.share*100)}%.` : ''}</p>`);
    (z.reasons || []).forEach(r => L.push(`<p class="small"><span class="${gd[r[0]] || 'muted'}">${esc(r[0] === '·' ? '•' : r[0])}</span> ${esc(r[1])}</p>`));
    if (z.tier === 'C' && z.u) L.push(`<p class="small muted">${uChip(z.u)} ${esc(uLine(z.u))}</p>`); });
  const A = G.A; if (A && A.today && A.today[0] != null) L.push(`<p class="small muted">Shadow: previous CBB model ${esc(x.home)} ${(+A.today[0]).toFixed(1)}, ${esc(x.away)} ${(+A.today[1]).toFixed(1)}; rebuild ${(+A.rb[0]).toFixed(1)}–${(+A.rb[1]).toFixed(1)}. Model A blends both.</p>`);
  if (sp === 'NFL'){ const N = S.games && S.games.NFL, QG = N && N.qb && N.qb.games && N.qb.games[x.home + '@' + x.away];
    if (G.desk_m != null) L.push(`<p class="small muted">Desk model (the full card below): ${esc(abTm(x, G.desk_m))}${QG ? `; its starting QBs ${esc(x.home)} ${esc(QG.hq)}, ${esc(x.away)} ${esc(QG.aq)}` : ''}. Model A prices the depth-chart QB and missing regulars with the drive engine's own sizes.</p>`);
    L.push(`<p class="small muted">Tested: Model A team scores miss by 7.4 pts vs the closing line's 7.3 (2018-26). 2026 openers (106 games): our spread gaps of 3+ saw the line move our way 82-86% of the time but covered only 49-56%, so NFL game bets stay leans until live tracking proves them. Player props are where NFL bets come from (unders and receptions at 5%+ EV).</p>`); }
  else if (sp === 'CFB'){ const A = G.A;
    if (A && A.v4) L.push(`<p class="small muted">Model A: rebuild v4 ${esc(x.home)} ${(+A.v4[0]).toFixed(1)}, ${esc(x.away)} ${(+A.v4[1]).toFixed(1)}${A.usage ? `, player usage prior ${sgn1(A.usage)} to ${esc(x.home)}` : ''}${(A.absences || []).length ? ', missing starters ' + A.absences.map(z => `${esc(z.player)} (${esc(z.team)}${z.elite ? ', elite' : ''}) ${sgn1(z.pts)}`).join(', ') : ''}.</p>`);
    (A && A.absences || []).forEach(z => L.push(`<p class="small wrn">Out: ${esc(z.player)} (${esc(z.team)}, ${z.kind === 'qb' ? 'starting QB' : 'top skill player'}${z.elite ? ', elite' : ''}) · ${esc(z.src)}. Priced into Model A at ${sgn1(z.pts)} team pts.</p>`));
    if (G.desk_m != null) L.push(`<p class="small muted">Desk model v3 (the full card below): ${esc(abTm(x, G.desk_m))}.</p>`);
    L.push(`<p class="small muted">Tested 2022-25: Model A team scores miss by 8.9 pts vs the closing line's 8.7 (v3 9.2). Openers with a spread gap of 4+ covered 58% (866 bets), 6+ 59%, game-week gaps of 4+ 55%. Look-ahead unders of 4+ hit 58%; overs and late totals lost. Missing star sizes come from 2022-25 play-by-play (elite QB out: team scored 2.1 under, the close already 1.9 under).</p>`); }
  else if (sp === 'NBA'){ const A = G.A;
    if (A && A.base) L.push(`<p class="small muted">Model A: rebuild ${esc(x.home)} ${(+A.base[0]).toFixed(1)}, ${esc(x.away)} ${(+A.base[1]).toFixed(1)} (${A.src === 'stack' ? "ratings plus tonight's projected lineup" : 'ratings only, no projected lineup yet'}), then the shooting-mix fix ${sgn1(A.fix[0])} / ${sgn1(A.fix[1])} (our 3-point % and free-throw terms were overweighted). Ratings alone: ${(+A.ratings[0]).toFixed(1)}–${(+A.ratings[1]).toFixed(1)}.</p>`);
    if (G.desk_m != null) L.push(`<p class="small muted">Desk team model (the full card below, full rosters): ${esc(abTm(x, G.desk_m))}.</p>`);
    L.push(`<p class="small muted">Tested on openers: spread tier A won 57% (SBR 2021-22, 255) and 62% (2026 DraftKings, 42); tier B 48-52%. Totals with a gap of 4+ at the opener hit 56% (2021-22) and 63% (2026 DraftKings). Close-like lines lost at every gap size, so they are passes.</p>`); }
  else L.push(`<p class="small muted">Tested on the 2026 DK opener: tier A won 60% against the spread (556 bets), tier B 51%, C 51%. Older lines are judged on closing-line results, where only big gaps held up.</p>`);
  const open = cabBets(G).some(([k, z]) => z.tier === 'A');
  return `<details class="dis panel"${open ? ' open' : ''}><summary><b>Why Model B likes it</b> <span class="small muted">${cabBets(G).map(([k, z]) => `${z.tier} ${esc(cabPick(k, z))}`).join(' · ') || 'no bet'}${cabU(G).length ? ` · unproven ${cabU(G).map(([k, z]) => esc(cabPick(k, z))).join(', ')}` : ''}</span></summary>${L.join('')}</details>`;
}
const ltFor = (sp, x) => ((LT && LT.sports && LT.sports[sp]) || []).find(g => String(g.id) === String(x.g ?? x.id) || (g.h === (x.home ?? x.h) && g.a === (x.away ?? x.a))) || null;
const ltTop = (sp, g) => g ? ltGame(sp, g).best.filter(b => b.mev != null).sort((p, q) => q.mev - p.mev)[0] || null : null;
// CFB player-usage prior (J 2026-10-29: implement the usage priors; cfb_usage.json from models/oddspapi/cfb_usage.py)
const cfbUseG = (h, a) => (S.cfbUse && S.cfbUse.games || []).find(g => g.h === h && g.a === a) || null;
function cfbUse(h, a){ const g = cfbUseG(h, a); if (g) return +g.shift || 0; const r = cfbUseG(a, h); return r ? -(+r.shift || 0) : 0; }
const cfbUseTxt = (h, a) => { const s = cfbUse(h, a); return s ? `<p class="small">Player usage prior (offense implied by player prop lines): ${sgn1(s)} pts to ${esc(h)}, included. Totals unchanged.</p>` : ''; };
// elite missing star: tracked, shown only on bets (J 2026-10-29)
const cfbStars = (h, a) => { const g = cfbUseG(h, a) || cfbUseG(a, h); return g ? (g.stars_out || []).filter(s => s.elite) : []; };
const starNote = (h, a) => cfbStars(h, a).map(s => `<p class="small wrn">Elite player out: ${esc(s.name || 'unnamed starter')} (${esc(s.team)}, ${esc(s.kind)} line about ${s.avg_line}). ${cabOn('CFB') ? 'Priced into Model A.' : 'Tracked, not in the price.'}</p>`).join('');
// "well above his average" tag (J 2026-10-28): projection 20%+ above both his recent and season averages at a comparable workload
function avgTag(sp, p, k){
  const s = p && p.s && p.s[k], f = s && flagOf(sp, k, s, 0.2, 'over');
  return f && flagRole(sp, p) ? ` <span class="tag t-avg" title="Our projection vs his recent and season averages">↑ avg +${Math.round(f.over*100)}%</span>` : '';
}
const whenOf = x => [x.d && x.d !== localDay() ? dayName(x.d) : '', kick(x)].filter(Boolean).join(' ');
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
};

// ---------- game rows (Games tab and Home) ----------
function abLegend(sp){
  const [la, lb] = AB_LAB(sp);
  if (AB2(sp)) return `<p class="small muted ablg"><b class="ka">A</b> ${esc(la)} <b class="kb">B</b> ${esc(lb)} (A bet, B lean)${!cabOn(sp) ? ' · Model B file not loaded yet' : ''}</p>`;
  return `<p class="small muted ablg"><b class="ka">A</b> ${esc(la)} <b class="kb">B</b> ${esc(lb)}${!(MA && MA.sports && MA.sports[sp]) ? ` · Model ${AB_DESK(sp) === 'A' ? 'B' : 'A'} has no ${esc(sp)} file yet` : ''}</p>`;
}
function eRow(sp, x, r, i){
  const ab = abOf(sp, x, r), A = ab.A || {}, B = ab.B || {}, M = ab.A || ab.B;
  const fav = M ? (M.p >= 0.5 ? [x.home, M.p] : [x.away, 1 - M.p]) : null;
  const st = gStat(sp, x), pre = st ? 'pre-game ' : '';
  const e = r && r.edges && r.edges[0] ? {t:r.edges[0].name, ev:r.edges[0].ev} : null, g = !st && ltFor(sp, x), b = !e && g ? ltTop(sp, g) : null;
  const bet = e || (b && b.mev >= PF_THR && b.bev >= PF_THR ? {t:sideLbl(g, b.mk, b.s, b.pt), ev:b.bev} : null);
  const cb = AB2(sp) ? cabGame(x, sp) : null, cs = cb && cb.B && cb.B.spread && cb.B.spread.tier !== 'C' ? cb.B.spread : null, cbt = cabBets(cb).filter(([k, z]) => z.tier === 'A')[0] || cabBets(cb)[0];
  if (AB2(sp)){ B.as = undefined; B.hs = undefined; }
  const star = bet && sp === 'CFB' && cfbStars(x.home, x.away).length ? ' <span class="chip y">star out</span>' : '';
  return `<button type="button" class="erow${st ? ' past' : ''}" data-i="${i}">
    <span class="t">${tLogo(sp, x.away)}${esc(x.away)}</span><span class="sc">${abF(sp, A.as)}</span><span class="sb">${AB2(sp) ? (cs && cs.side === x.away ? cs.tier : '') : abF(sp, B.as)}</span>
    <span class="t">${tLogo(sp, x.home)}${esc(x.home)}</span><span class="sc">${abF(sp, A.hs)}</span><span class="sb">${AB2(sp) ? (cs && cs.side === x.home ? cs.tier : '') : abF(sp, B.hs)}</span>
    <span class="meta"><span>${statChip(st)}${esc(whenOf(x))}${fav ? ` · ${esc(fav[0])} ${pct(fav[1], 0)}` : ''}</span><span>${ab.big ? `<span class="chip y">A/B gap</span> ` : ''}${AB2(sp) ? (cbt ? `<span class="chip${cbt[1].tier === 'A' ? '' : ' y'}">${pre}${CAB_T[cbt[1].tier]}: ${esc(cabPick(cbt[0], cbt[1]))}${cbt[1].tier === 'A' ? ' ' + sgn(cbt[1].bet.ev) : ''}</span>` : (cu => cu ? `<span class="chip u">${pre}Unproven ${cu[1].u.score}: ${esc(cabPick(cu[0], cu[1]))}</span>` : '')(cabU(cb).sort((p, q) => q[1].u.score - p[1].u.score)[0])) : bet ? `<span class="chip">${pre}${esc(bet.t)} ${sgn(bet.ev)}</span>` : ''}${star}</span></span></button>`;
}
function renderGameListE(){
  const sp = S.sport, box = $('#gameListE'), C = GCFG[sp]; if (!box) return; if (AB2(sp)) cabLoad(sp);
  const all = C ? slateGames(sp).filter(x => !started(x) || recentStart(x)) : [];
  const gs = [...all.filter(x => !started(x)), ...all.filter(x => started(x))];   // upcoming first, then the day's started games
  if (!gs.length){ box.innerHTML = ''; return; }
  const pr0 = probeGames(sp, gs), keep = gs.map((x, i) => { const ab = abOf(sp, x, pr0[i]); return !!(ab.A || ab.B); });
  const gk = gs.filter((x, i) => keep[i]), pr = pr0.filter((x, i) => keep[i]), hid = gs.length - gk.length;   // J 2026-11-02 cleanup: games neither model projects (non-D1 opponents) are left out
  const nUp = gk.filter(x => !started(x)).length, rows = (a, z) => gk.slice(a, z).map((x, i) => eRow(sp, x, pr[a + i], a + i)).join('');
  box.innerHTML = abLegend(sp) + (nUp ? `<div class="elist">${rows(0, nUp)}</div>` : '<p class="small muted">No games left to start today.</p>')
    + (gk.length > nUp ? `<p class="small muted elh"><b>Started and final</b> · picks as of the last pre-game update. Tap one to review it or log a bet you missed.</p><div class="elist">${rows(nUp, gk.length)}</div>` : '') + (hid ? `<p class="small muted">${hid} game${hid > 1 ? 's' : ''} with no projection (non-Division I or unrated teams) not shown.</p>` : '');
  box.querySelectorAll('.erow').forEach(b => b.onclick = () => openGame(sp, gk[+b.dataset.i]));
}
const _renderGameList = renderGameList;
renderGameList = function(){ _renderGameList(); renderGameListE(); drawSheet(); };
// Watch the sim (J 2026-11-06): sim.js + sims_{sport}.json (models/sims/build.py), loaded on the first tap so the page stays the same size
const SIM_SP = ['NBA', 'CBB', 'NFL', 'CFB', 'Tennis'];
const simBtn = sp => SIM_SP.includes(sp) ? `<div class="row" data-simrow hidden><button class="btn ghost sm" type="button" data-sim>&#9654; Watch the sim</button><span class="small muted">a simulated game from Model A, not a prediction</span></div>` : '';
const SIMOK = {};   // sport -> promise: does sims_{sport}.json exist (asked once per visit)
function simShow(sp, root){ const k = String(sp).toLowerCase();
  (SIMOK[k] = SIMOK[k] || fetch(`sims_${k}.json`, {cache:'no-cache'}).then(r => r.ok).catch(() => false))
    .then(ok => { if (ok) root.querySelectorAll('[data-simrow]').forEach(r => r.hidden = false); }); }
function simOpen(sp, x){
  const go = () => window.BDSim.open(sp, x);
  if (window.BDSim) return go();
  const s = document.createElement('script'); s.src = 'sim.js'; s.onload = go;
  s.onerror = () => { s.remove(); toast('Could not load the sim viewer'); }; document.head.appendChild(s);
}
// the game screen: tapping a row fills the desk card (old row click) and shows the sheet above it
function openGame(sp, x){
  SHEET = {sp, x};
  const old = [...document.querySelectorAll('#gameList .grow')].find(b => b.querySelector('b') && b.querySelector('b').textContent.trim() === `${x.away} @ ${x.home}`);
  if (old) old.click(); drawSheet();
  setTimeout(() => { const el = $('#gameSheet'); if (el && !el.hidden) el.scrollIntoView({block:'start'}); }, 0);
}
function closeGame(){ SHEET = null; drawSheet(); window.scrollTo(0, 0); }
// J 2026-10-30: one game list (the A/B score rows); the old list stays hidden (its rows still fill the desk card), and the desk
// card (explanation, edges, your own prices) moves inside the game tile under the A/B boxes, prop leans and book prices
function drawSheet(){
  const el = $('#gameSheet'), gc = $('#gameCards'); if (!el) return;
  const eOn = !!$('#gameListE').innerHTML.trim(), on = !!(SHEET && SHEET.sp === S.sport && !$('#sv-games').hidden);
  el.hidden = !on; $('#gameListE').hidden = on; $('#gameList').hidden = eOn || on;
  if (gc && gc.parentNode === el) el.after(gc);   // take the card out before the tile is redrawn
  if (gc) gc.hidden = eOn && !on;
  if (!on) return;
  const {sp} = SHEET, x = slateGames(sp).find(y => y.home === SHEET.x.home && y.away === SHEET.x.away) || SHEET.x;
  const r = GCFG[sp] ? probeGames(sp, [x])[0] : null, ab = abOf(sp, x, r), g = ltFor(sp, x), ln = x.ln || lineFor(sp, x);
  const mk = ln ? [ln.L != null ? `${x.home} ${sgnPts(ln.L)}` : '', ln.T != null ? `o/u ${ln.T}` : ''].filter(Boolean).join(' · ') : '';
  el.innerHTML = `<div class="row" style="justify-content:space-between"><button class="btn ghost sm" type="button" id="shBack">‹ All ${esc(sp)} games</button><span class="small muted">${esc(whenOf(x))}${x.n ? ' · neutral' : ''}</span></div>
    <h2 class="gt">${tLogo(sp, x.away)}${esc(x.away)} <span class="muted">@</span> ${tLogo(sp, x.home)}${esc(x.home)}</h2>${mk ? `<p class="small muted">${started(x) ? 'Last pre-game market' : 'Market'} ${esc(mk)}</p>` : ''}
    ${(st => st ? `<p class="small">${statChip(st)}Started ${esc(kick(x))}. Model A, Model B and the market below are as of the last update before it started${(cg => cg && cg.pre ? ` (${esc(String(cg.pre).slice(0, 16).replace('T', ' '))}${/UTC|Z$/.test(String(cg.pre)) ? '' : ' UTC'})` : '')(AB2(sp) ? cabGame(x, sp) : null)}. You can still log a bet you placed.</p>` : '')(gStat(sp, x))}
    <div class="abrow">${AB2(sp) ? abBox(sp, x, 'A', ab.A) + cabBox(sp, x) : ['A', 'B'].map(k => abBox(sp, x, k, ab[k])).join('')}</div>${simBtn(sp)}
    ${AB2(sp) ? cabWhy(sp, x) : whyBox(sp, x, ab, ln)}${gameBets(sp, x, r, g)}${propLeans(sp, x)}${bookGrid(sp, g)}${cfbV2Line(sp, x)}
    <h3 class="cardh">Full model card · explanation, edges and your own prices</h3>`;
  $('#shBack').onclick = closeGame;
  el.querySelectorAll('[data-sim]').forEach(b => b.onclick = () => simOpen(sp, {...x, _A:(AB2(sp) && cabGame(x, sp) || {}).A}));
  simShow(sp, el);
  el.querySelectorAll('[data-lean]').forEach(t => t.onclick = () => { const z = LEANS[+t.dataset.lean]; if (z) poOpen(z.p, z.k, z.r); });
  if (gc){ el.appendChild(gc); gc.hidden = false; }
}
function abBox(sp, x, k, M){
  const lab = AB_LAB(sp)[k === 'A' ? 0 : 1];
  if (!M) return `<div class="abx ${k.toLowerCase()}"><span class="lab">Model ${k} · ${esc(lab)}</span><p class="small muted">No projection for this game yet.</p></div>`;
  const m = M.hs - M.as;
  return `<div class="abx ${k.toLowerCase()}"><span class="lab">Model ${k} · ${esc(lab)}</span>
    <div class="abs"><span>${esc(x.away)}</span><b>${abF(sp, M.as)}</b></div><div class="abs"><span>${esc(x.home)}</span><b>${abF(sp, M.hs)}</b></div>
    <div class="bar"><i style="width:${(M.p*100).toFixed(0)}%"></i></div>
    <p class="small">${esc(x.home)} ${pct(M.p, 0)} <span class="muted">${amS(+am(M.p))}</span> · ${esc(x.away)} ${pct(1 - M.p, 0)} <span class="muted">${amS(+am(1 - M.p))}</span>${M.est ? ' <span class="muted" title="Win chance worked out from the projected score">≈</span>' : ''}</p>
    <p class="small muted">${esc(abTm(x, m))} · total ${(M.hs + M.as).toFixed(1)}</p>${AB2(sp) && k === 'A' ? cabRange(sp, x) : ''}</div>`;
}
// "Why the models disagree" (J 2026-10-29: the information J wanted most from concept A; open by default on a big gap)
function whyBox(sp, x, ab, ln){
  if (!ab.A || !ab.B) return '';
  const A = ab.A, B = ab.B, mA = A.hs - A.as, mB = B.hs - B.as, L = [];
  L.push(`Model A has ${abTm(x, mA)} with ${(A.hs + A.as).toFixed(1)} total points. Model B has ${abTm(x, mB)} with ${(B.hs + B.as).toFixed(1)}.`);
  const dh = A.hs - B.hs, da = A.as - B.as, w = Math.abs(dh) >= Math.abs(da) ? [x.home, dh] : [x.away, da];
  if (Math.abs(w[1]) >= 0.05) L.push(`Most of the difference is ${w[0]}'s score: A has it ${Math.abs(w[1]).toFixed(1)} ${w[1] > 0 ? 'higher' : 'lower'} than B.`);
  if (ln && ln.L != null){ const lm = -ln.L, cm = Math.abs(mA - lm) < Math.abs(mB - lm) ? 'A' : 'B';
    L.push(`The market has ${abTm(x, lm)}${ln.T != null ? ` and ${ln.T} total` : ''}. ${cm} is closer on the margin${ln.T != null ? `, ${Math.abs(A.hs + A.as - ln.T) < Math.abs(B.hs + B.as - ln.T) ? 'A' : 'B'} on the total` : ''}.`); }
  L.push(...abNotes(sp, x, ab));
  const gr = abGrade(sp); if (gr) L.push(gr);
  return `<details class="dis panel"${ab.big ? ' open' : ''}><summary><b>Why the models disagree</b> <span class="small muted">margin gap ${Math.abs(ab.gapM).toFixed(1)} · total gap ${Math.abs(ab.gapT).toFixed(1)}</span></summary>${L.map(t => `<p class="small">${esc(t)}</p>`).join('')}</details>`;
}
function abNotes(sp, x, ab){
  const out = [], M = MA && MA.sports && MA.sports[sp], m = maGame(sp, x);
  if (sp === 'NFL'){ const N = S.games && S.games.NFL, QG = N && N.qb && N.qb.games && N.qb.games[x.home + '@' + x.away], IJ = N && N.inj && N.inj.teams;
    if (QG) out.push(`B's starting QBs: ${x.home} ${QG.hq}, ${x.away} ${QG.aq}${QG.pts ? `; the QB term moves B ${sgn1(QG.pts)} pts to ${x.home}` : ''}.`);
    const inj = [x.home, x.away].map(t => IJ && IJ[t] && IJ[t][1].length ? `${t}: ${IJ[t][1].join(', ')}` : '').filter(Boolean);
    if (inj.length) out.push(`Regulars out or doubtful in B: ${inj.join('; ')}.`);
    out.push('A prices the depth-chart QB and missing regulars with its own drive engine sizes, so a QB change or a long injury list is the usual reason the two split.'); }
  else if (sp === 'NBA'){ out.push(m && m.src === 'stack' ? "A plays out tonight's projected lineup (who plays and their minutes) on its team ratings; B uses the desk's team ratings and injury minutes." : 'A had no projected lineup for this game, so it is ratings only; B uses the desk team ratings and injury minutes.'); }
  else if (sp === 'CBB'){ const rb = m && m.rb; out.push(`A (CBB v2) is a stack: about 56% of B plus 38% of the rebuild, minus players out.${rb && rb[0] != null ? ` The rebuild alone says ${x.home} ${(+rb[0]).toFixed(1)}, ${x.away} ${(+rb[1]).toFixed(1)}.` : ''}`); }
  else if (sp === 'CFB'){ const v = m && m.v3, s = cfbUse(x.home, x.away);
    if (v && v[0] != null && ab.A){ const v3m = v[0] - v[1], mA = ab.A.hs - ab.A.as; out.push(`A starts from the desk's v3 (${abTm(x, v3m)}) and adds program strength, coach career win rate and crowd size, which moved it ${sgn1(mA - v3m)} pts toward ${x.home}; its total uses this season's tempo.`); }
    else out.push('A is v3 plus program strength, coach career win rate and crowd size, with this season\'s tempo for the total.');
    if (s) out.push(`B includes the player usage prior from prop lines: ${sgn1(s)} pts to ${x.home}.`); }
  if (M && M.made && !out.length) out.push(`A: ${M.how}.`);
  return out;
}
function abGrade(sp){
  const G = MA && MA.grade && MA.grade.sports && MA.grade.sports[sp]; if (!G || sp === 'CBB') return '';
  const a = G.rebuild_same_games, b = G.today_same_games, l = G.line_same_games; if (!a || !b || !(a.n >= 5)) return '';
  return `Graded so far (${a.n} games): A missed the margin by ${a.margin_mae} pts on average, B by ${b.margin_mae}${l && l.n ? `; the line missed by ${l.margin_mae} on its ${l.n} games` : ''}.`;
}
// this game's bets: every market's best price at Colorado books by our model's EV (even when nothing clears the minimum), and prop flags
function gameBets(sp, x, r, g){
  if (sp === 'CBB' && cabOn('CBB')) return '';   // CBB: Model B's box and 'Why Model B likes it' carry the bets (shrunk EV); the unshrunk table would contradict them
  const it = [], mb = AB2(sp) && cabOn(sp);      // NFL: same for the game lines, but the prop flags stay
  if (mb);
  else if (g) ltGame(sp, g).best.filter(b => b.mev != null).sort((p, q) => q.mev - p.mev).slice(0, 4).forEach(b =>
    it.push({t:sideLbl(g, b.mk, b.s, b.pt), s:`${bkName(b.b)} ${amS(b.pr)}`, ev:b.mev, ev2:b.bev, n:'ours · blended'}));
  else (r && r.edges || []).forEach(e => it.push({t:e.name, s:e.assumed ? 'at -110' : amS(e.odds), ev:e.ev, n:'blended'}));
  if (PO_SP.includes(sp) && PO[sp]) pfAll(sp, true).rows.filter(z => z.kind !== 'check' && String(z.p.g) === String(x.g)).slice(0, 6).forEach(z =>
    it.push({t:`${z.p.n} ${z.side.toLowerCase()} ${z.L} ${STAT[z.k] || z.k}`, tag:PF_TAG[z.kind] + avgTag(sp, z.p, z.k), s:`${poBk(z.b)} ${amS(z.pr)}`, ev:z.kind === 'proven' ? z.ev : z.evb, n:z.kind === 'proven' ? 'ours' : 'blended'}));
  if (!it.length) return '';
  const good = it.some(z => (z.ev2 ?? z.ev) >= PF_THR && z.ev >= PF_THR);
  return `<div class="panel ebox"><h3>Best bets in this game</h3>${sp === 'CFB' ? starNote(x.home, x.away) : ''}
    <table class="et"><tbody>${it.map(z => `<tr><td class="t">${esc(z.t)}${z.tag || ''}<br><span class="muted">${esc(z.s)}</span></td><td class="${z.ev >= 0 ? 'pos' : 'neg'}">${sgn(z.ev)}${z.ev2 != null ? `<br><span class="muted">${sgn(z.ev2)}</span>` : ''}</td></tr>`).join('')}</tbody></table>
    <p class="small muted">${good ? `Green at ${Math.round(PF_THR*100)}%+ on our model and the blend is a bet.` : 'Nothing here clears the EV minimum; these are the best prices available.'} EV is our model${g ? ', then blended with the market' : ''}.</p></div>`;
}
const _everyBook = everyBook;
everyBook = function(sp){ return SHEET && SHEET.sp === sp ? '' : _everyBook(sp); };   // inside the tile the book grid above already shows it
// prop leans (J 2026-10-30): this game's props where our chance of the over at the books' main line is furthest from 50%,
// as over and under leans; not tested bets, a read on which way each player's number points
let LEANS = [];
function propLeans(sp, x){
  LEANS = [];
  if (!PO_SP.includes(sp) || !PO[sp] || !S.slates[sp]) return '';
  const T = [x.home, x.away], L = [];
  S.slates[sp].players.filter(p => upcoming(p) && (x.d == null || p.d == null || p.d === x.d) && ((x.g != null && String(p.g) === String(x.g)) || (T.includes(p.t) && T.includes(p.o)))).forEach(p =>
    Object.entries(p.s).forEach(([k, s]) => { if (sp === 'NBA' && NBA_PC.includes(k)) return;   // NBA blocks / steals: lost as bets, left out of the leans too
      const all = poRows(sp, p, k, true), ml = all.filter(r => r.main && !r.odd).map(r => r.L).sort((a, b) => a - b); if (!ml.length) return;
      const ln = ml[ml.length >> 1], c = pfCond(k, s, ln).c, mk = pfMarket(k, s, all, ln), O = c >= 0.5;
      const mine = poRows(sp, p, k).filter(r => r.L === ln && (O ? r.o : r.u) != null && !r.odd);
      const r = mine.reduce((a, q) => !a || dec(O ? q.o : q.u) > dec(O ? a.o : a.u) ? q : a, null);
      const pr = r ? (O ? r.o : r.u) : null, ch = O ? c : 1 - c;
      L.push({p, k, s, L:ln, O, ch, mk:mk == null ? null : (O ? mk : 1 - mk), r, pr, ev:pr == null ? null : ch*dec(pr) - 1}); }));
  if (!L.length) return '';
  const top = O => L.filter(z => z.O === O).sort((a, b) => b.ch - a.ch).slice(0, 5);
  const row = z => { LEANS.push(z); return `<tr class="click" data-lean="${LEANS.length - 1}"><td class="t">${esc(z.p.n)} ${z.O ? 'over' : 'under'} ${z.L} ${esc(STAT[z.k] || z.k)}${z.O ? avgTag(sp, z.p, z.k) : ''}<br><span class="muted">we project ${z.s.m} · ${z.r ? `${esc(poBk(z.r.b))} ${amS(z.pr)}` : 'not at Colorado books'}${z.mk != null ? ` · books ${pct(z.mk, 0)}` : ''}</span></td><td><b>${pct(z.ch, 0)}</b>${z.ev != null ? `<br><span class="${z.ev >= 0 ? 'pos' : 'neg'}">${sgn(z.ev)}</span>` : ''}</td></tr>`; };
  const ov = top(true), un = top(false);
  return `<div class="panel ebox"><h3>Prop leans in this game</h3><p class="small muted">Where our projection sits furthest above or below the books' main line. Big % = our chance of that side; under it, EV at the best price at Colorado books. Leans to read the game's overs and unders, not tested bets${sp === 'NBA' ? ' (NBA unders have not tested well)' : ''}. Tap one to open the prop.</p>
    ${ov.length ? `<h4>Overs</h4><table class="et"><tbody>${ov.map(row).join('')}</tbody></table>` : ''}${un.length ? `<h4>Unders</h4><table class="et"><tbody>${un.map(row).join('')}</tbody></table>` : ''}</div>`;
}
function bookGrid(sp, g){
  if (!g || !g.bk) return '';
  const books = Object.keys(g.bk).filter(b => g.bk[b] && (myBook(b) || b === 'pinnacle')); if (!books.length) return '';
  const top = new Map(); ltGame(sp, g).rows.forEach(x => { const k = x.mk + x.s; if (!top.has(k) || x.ev > top.get(k).ev) top.set(k, x); });   // one lit cell per column: the best value vs the sharp fair price
  const c = (b, mk, s, pt, pr, lab) => { const t = top.get(mk + s); return pr == null ? '<td>–</td>' : `<td class="${t && t.b === b && t.pt === pt ? 'best' : ''}">${lab}${amS(pr)}</td>`; };
  return `<div class="panel"><h3>Every book</h3><div class="tablewrap"><table class="et grid7"><thead><tr><th>Book</th><th>${esc(g.h)}</th><th>${esc(g.a)}</th><th>Over</th><th>Under</th><th>${esc(g.h)} ML</th><th>${esc(g.a)} ML</th></tr></thead><tbody>
    ${books.map(b => { const r = g.bk[b]; return `<tr><td class="t">${esc(bkName(b))}${myBook(b) ? '' : ' <span class="muted">ref</span>'}</td>${c(b, 'spread', 'H', r[0], r[1], r[0] != null ? sgnPts(r[0]) + ' ' : '')}${c(b, 'spread', 'A', r[0] != null ? -r[0] : null, r[2], r[0] != null ? sgnPts(-r[0]) + ' ' : '')}${c(b, 'total', 'O', r[3], r[4], r[3] != null ? 'o' + r[3] + ' ' : '')}${c(b, 'total', 'U', r[3], r[5], r[3] != null ? 'u' + r[3] + ' ' : '')}${c(b, 'ml', 'H', null, r[6], '')}${c(b, 'ml', 'A', null, r[7], '')}</tr>`; }).join('')}</tbody></table></div>
    <p class="small muted">Lit = the best value at Colorado books in each column, against the sharp fair price. Pinnacle is a reference only.</p></div>`;
}
function cfbV2Line(sp, x){
  const V = sp === 'CFB' && S.cfb2 && S.cfb2.slate; if (!V) return '';
  const r = V.find(z => z.home === x.home && z.away === x.away); if (!r) return '';
  return `<p class="small">Your CFB v2 (week ${esc(S.cfb2.week)} ratings): ${esc(abTm(x, r.proj_margin))}, total ${(+r.proj_total).toFixed(1)}${r.spread_bet ? ` · v2 bet: ${esc(r.spread_side)} (edge ${r.spread_edge})` : ''}.</p>`;
}

// ---------- Home ----------
function lookAhead(){
  const R = new Map(), now = Date.now(), H = 24*36e5, put = (k, x) => { if (!R.has(k) || R.get(k).ev < x.ev) R.set(k, x); };
  if (LT) LT_SP.filter(sp => !AB2(sp) || !cabOn(sp)).forEach(sp => ((LT.sports || {})[sp] || []).forEach(g => { if (ltStarted(g)) return;
    (g.la || []).forEach(x => { const t = Date.parse(x.po); if (!(now - t <= H) || !myBook(x.b) || !(x.mev >= PF_THR) || !(x.bev >= PF_THR)) return;   // our model and the blend both beat it, like Best bets
      put(sp + g.id + x.mk + x.s, {sp, ev:x.bev, t, b:x.b, title:sideLbl(g, x.mk, x.s, x.pt), sub:`${g.a} @ ${g.h} · ${amS(x.pr)}`, go:() => { bbGo(sp); ltOpen(sp, g); }}); }); }));
  PO_SP.forEach(sp => { if (!PO[sp] || !S.slates[sp]) return;
    pfAll(sp, true).rows.forEach(z => { if (z.kind === 'check' || z.kind === 'lean' || z.kind === 'unproven') return; const f = poFind(sp, z.p); if (!f || ltStarted(f.g)) return;
      const post = f.g.posted && f.g.posted[f.n]; if (!post) return; let ts = null;
      Object.keys(post).forEach(n => { if ((PO_MAP[sp] || {})[n] !== z.k) return; const q = post[n][String(z.L)] || post[n][(+z.L).toFixed(1)]; if (q && q[z.b]) ts = q[z.b]; });
      const t = Date.parse(ts); if (!(now - t <= H)) return;
      put(sp + z.p.n + z.k + z.side, {sp, ev:z.kind === 'proven' ? z.ev : z.evb, t, b:z.b, title:`${z.p.n} ${z.side.toLowerCase()} ${z.L} ${STAT[z.k] || z.k}`, avg:avgTag(sp, z.p, z.k), sub:amS(z.pr), go:() => { bbGo(sp); poOpen(z.p, z.k, z.r); }}); }); });
  Object.keys(CAB_F).forEach(s => cabRows(s).filter(r => r.fresh).forEach(r => put(s + r.key, r)));
  return [...R.values()].sort((a, b) => b.ev - a.ev);
}
// Model B tier A CBB bets as Best bets / Look ahead rows (they replace the old CBB game-line flags once cbb_ab.json is loaded)
function cabRows(sp = 'CBB'){
  if (!cabOn(sp)) return [];
  const now = Date.now(), R = [], ls = ((LT && LT.sports) || {})[sp] || [];
  Object.entries(CABS[sp].games).forEach(([id, G]) => { const sl = (gsGames(sp) || []).find(y => String(y.id) === id); if (sl && sl.k && Date.parse(sl.k) <= now) return;
    cabBets(G).filter(([k, z]) => z.tier === 'A' && z.bet.ev >= PF_THR).forEach(([k, z]) => { const lg = ls.find(g => String(g.id) === id);
      R.push({type:'games', sp, kind:'mb', key:id + k, k:sl && sl.k, ev:z.bet.ev, fresh:z.fresh, t:z.bet.age_h != null ? now - z.bet.age_h*36e5 : now, b:z.bet.book,
        title:cabPick(k, z), sub:`${G.a} @ ${G.h} · ${z.bet.book_name} ${amS(z.bet.price)}`, note:`Model B tier A`,
        go:() => { if (sl){ homeOpen(sp, {g:sl.id, d:sl.d, k:sl.k, w:sl.w, home:sl.h, away:sl.a, n:sl.n || 0}); } else if (lg){ bbGo(sp); ltOpen(sp, lg); } }}); }); });
  return R;
}
BB_TAG.mb = '<span class="tag t-flag">Model B · A</span>'; BB_RANK.mb = 1;
const _bbRows = bbRows;
bbRows = function(){ const R = _bbRows().filter(r => r.kind !== 'lean' && r.kind !== 'unproven'), on = Object.keys(CAB_F).filter(cabOn);   // NFL prop leans (Model B tier B) are not bets
  return R.filter(r => !(on.includes(r.sp) && r.kind === 'game')).concat(...on.map(s => cabRows(s))); };
// NFL player props, Model B tiers (J 2026-11-05; rules in models/nfl_ab/live.py PROPS, from model-b-confidence props.json, desk projections vs
// OddsPapi Oct 2026): A = 5%+ EV under, or receptions either side (unders 61.8%, receptions 66.6% at 5%+ EV) -> stays a Flag;
// B = an over of another stat (53.5% at the opener, 43% at the close) -> Lean, not a bet; C = pass attempts / rushing yards / carries (lost) -> dropped.
const NFL_PC = ['pass_att', 'rush_yds', 'carries'];
PF_TAG.lean = '<span class="tag t-check">Lean</span>'; BB_TAG.lean = PF_TAG.lean; BB_RANK.lean = 3;
PF_TAG.unproven = '<span class="tag t-un" title="Tier C: this stat lost in testing; shown so it can be tracked">Unproven</span>'; BB_TAG.unproven = PF_TAG.unproven; BB_RANK.unproven = 4;
const _pfAll = pfAll;
pfAll = function(sp, everyDay = false){ const o = _pfAll(sp, everyDay); if (sp !== 'NFL' || o.nflB) return o;
  o.rows = o.rows.map(x => { if (x.kind === 'check') return x;
    if (NFL_PC.includes(x.k)){ x.kind = 'unproven'; x.tier = 'C'; return x; }   // J 2026-11-06: shown as Unproven (lost in testing), not dropped
    const t = x.side === 'Under' || x.k === 'rec' ? 'A' : 'B'; if (t === 'B') x.kind = 'lean'; x.tier = t; return x; });
  o.nflB = true; return o; };
// NBA player props, Model B tiers (J 2026-11-05; rules in models/nba_ab/live.py PROPS, model-b-confidence props.json NBA_live_open = desk
// projections vs OddsPapi 2026 openers at 5%+ EV): A = Proven (DK/FD overs 12%+ on our model alone) and other overs on points, threes,
// assists, PRA (threes 58.6%, points 55.8%, assists 55.1%) -> stay Proven / Flag; B = overs on rebounds or turnovers (rebounds 48.8%, -7.9%)
// -> Lean, off Best bets and Look ahead; C = blocks and steals (41% / -11%) -> dropped. Unders stay held back (prop_odds.js).
const NBA_PB = ['reb', 'tov'], NBA_PC = ['blk', 'stl'];
const _pfAllN = pfAll;
pfAll = function(sp, everyDay = false){ const o = _pfAllN(sp, everyDay); if (sp !== 'NBA' || o.nbaB) return o;
  o.rows = o.rows.map(x => { if (x.kind === 'check') return x;
    if (NBA_PC.includes(x.k)){ x.kind = 'unproven'; x.tier = 'C'; return x; }   // J 2026-11-06: shown as Unproven, not dropped
    const t = x.kind === 'proven' || !NBA_PB.includes(x.k) ? 'A' : 'B'; if (t === 'B') x.kind = 'lean'; x.tier = t; return x; });
  o.nbaB = true; return o; };
const agoTxt = t => { const h = (Date.now() - t)/36e5; return h < 1 ? `${Math.max(1, Math.round(h*60))}m ago` : `${Math.round(h)}h ago`; };
function homeGames(){
  const today = localDay(), sps = ['NFL', 'NBA', 'CFB', 'CBB'].filter(sp => GCFG[sp] && gReady(sp));
  const up = sp => gsGames(sp).filter(x => !started(x) && x.d >= today);
  const days = [...new Set(sps.flatMap(sp => up(sp).map(x => x.d)))].sort(), d = days[0];
  if (!d) return {d:null, list:[]};
  return {d, list:sps.map(sp => { const gs = up(sp).filter(x => x.d === d).map(x => ({g:x.id, d:x.d, k:x.k, w:x.w, home:x.h, away:x.a, n:x.n || 0, feed:x.ln || null}));
    return {sp, gs, pr:probeGames(sp, gs)}; }).filter(z => z.gs.length)};
}
function homeOpen(sp, x){
  if (sp === 'NFL' && x.w) NFL_W = +x.w;
  if (S.sport !== sp){ S.sport = sp; lsSet('betdesk.sport', sp); }
  if (sp !== 'NFL' && sp !== 'CFB' && slateDays(sp).includes(x.d)) S.day = x.d;
  S.sub[sp] = 'games'; lsSet('betdesk.sub', JSON.stringify(S.sub)); show('slate'); refreshSport(true);
  openGame(sp, slateGames(sp).find(y => y.home === x.home && y.away === x.away) || x);
}
// Home: unproven picks across the two-model sports (play / watch, best score first), collapsed and apart from Best bets (J 2026-11-06)
function uRows(){
  const now = Date.now(), R = [];
  Object.keys(CAB_F).filter(cabOn).forEach(sp => Object.entries(CABS[sp].games).forEach(([id, G]) => {
    const sl = (gsGames(sp) || []).find(y => String(y.id) === id); if (!sl || (sl.k && Date.parse(sl.k) <= now)) return;
    cabU(G).forEach(([k, z]) => R.push({sp, u:z.u, title:cabPick(k, z), sub:`${G.a} @ ${G.h} · ${z.bet.book_name} ${amS(z.bet.price)}`,
      go:() => homeOpen(sp, {g:sl.id, d:sl.d, k:sl.k, w:sl.w, home:sl.h, away:sl.a, n:sl.n || 0})})); }));
  ((SPORT_LIST.includes('Tennis') && TENAB && TENAB.matches) || []).forEach(m => { const u = m.B && m.B.u;   // J 2026-11-11: tennis off the desk if (!u || u.rec === 'skip' || Date.parse(m.start) <= now) return;
    R.push({sp:'Tennis', u, title:`${tenName(m.B.pick)} ML`, sub:`${tenName(m.p1)} v ${tenName(m.p2)} · ${bkName(m.B.book)} ${amS(amOf(m.B.price))} · paper`, go:() => { TSHEET = m.fid; if (S.sport !== 'Tennis'){ S.sport = 'Tennis'; lsSet('betdesk.sport', 'Tennis'); } S.sub.Tennis = 'matches'; lsSet('betdesk.sub', JSON.stringify(S.sub)); show('slate'); refreshSport(true); renderTennis(); }}); });
  return R.sort((a, b) => b.u.score - a.u.score);
}
const uPanel = U => !U.length ? '' : `<details class="panel"><summary><b>Unproven picks</b> <span class="small muted">${U.length} · not tested bets, for tracking</span></summary>
  <p class="small muted">Model B passes these. The rating says how strongly Model A alone disagrees with the market and how this kind of bet did in testing. Small play = Model A 4%+ halfway EV, Model B not against it, and this kind of bet did not lose in testing. Log them to track them.</p>
  <table class="et"><tbody>${U.map((r, i) => `<tr class="click" data-u="${i}"><td class="t">${esc(r.title)} ${uChip(r.u)}<br><span class="muted">${esc(r.sp)} · ${esc(r.sub)}</span><br><span class="muted">${esc(r.u.why)}</span></td><td class="muted">${sgn(r.u.evU)}</td></tr>`).join('')}</tbody></table></details>`;
function renderHome(){
  const out = $('#homeOut'); if (!out) return; cabLoad();   // every two-model sport (CBB, NFL)
  if (!LT && !PO_SP.some(sp => PO[sp])){ out.innerHTML = '<p class="small muted">Loading odds…</p>'; return; }
  const bb = bbRows().filter(r => r.kind !== 'check' && r.kind !== 'verify' && r.type !== 'promo' && r.type !== 'mid').sort((a, b) => b.ev - a.ev).slice(0, 6);
  const la = lookAhead(), LA = la.slice(0, 8), G = homeGames(), UN = uRows().slice(0, 10);
  const swept = [...PO_SP.filter(sp => PO[sp]).map(sp => PO[sp].made), LT && LT.made].filter(Boolean).map(x => Date.parse(String(x).replace(' UTC', 'Z').replace(' ', 'T'))).filter(isFinite);
  const newest = swept.length ? Math.max(...swept) : null;
  out.innerHTML = `<div class="row" style="justify-content:space-between;align-items:baseline"><h2>${esc(new Date().toLocaleDateString([], {weekday:'long', month:'short', day:'numeric'}))}</h2>${newest ? `<span class="small muted">odds ${esc(agoTxt(newest))}</span>` : ''}</div>
    <div class="panel"><div class="row" style="justify-content:space-between"><h3>Best bets</h3><button class="btn ghost sm" type="button" id="hmAll">All best bets ›</button></div>
      ${bb.length ? `<table class="et"><tbody>${bb.map((r, i) => `<tr class="click" data-b="${i}"><td class="t">${esc(r.title)} ${BB_TAG[r.kind] || ''}${r.avg || ''}<br><span class="muted">${esc(r.sp)} · ${esc(r.sub)}</span></td><td class="pos">${sgn(r.ev)}</td></tr>`).join('')}</tbody></table>` : '<p class="small muted">Nothing at Colorado books clears the EV minimum right now.</p>'}</div>
    <div class="panel ah"><h3>Look ahead</h3><p class="small muted">Lines posted in the last 24 hours that our model and the blend both beat by ${Math.round(PF_THR*100)}%+, biggest blended edge first.</p>
      ${LA.length ? `<table class="et"><tbody>${LA.map((r, i) => `<tr class="click" data-l="${i}"><td class="t">${esc(r.title)}${r.avg || ''}<br><span class="muted">${esc(r.sp)} · ${esc(r.sub)} · opened ${esc(agoTxt(r.t))} at ${esc(bkName(r.b))}</span></td><td class="pos">${sgn(r.ev)}</td></tr>`).join('')}</tbody></table>${la.length > LA.length ? `<p class="small muted">${la.length - LA.length} more in All best bets.</p>` : ''}` : '<p class="small muted">No new line in the last day beats our model by the minimum.</p>'}</div>
    ${uPanel(UN)}
    ${G.d ? `<h3>${esc(G.d === localDay() ? "Today's games" : dayName(G.d) + ' games')}</h3>` + G.list.map((z, j) => `<div class="hsp"><div class="row" style="justify-content:space-between"><b class="sph">${esc(z.sp)}</b>${abLegend(z.sp)}</div><div class="elist">${z.gs.map((x, i) => eRow(z.sp, x, z.pr[i], j + ':' + i)).join('')}</div></div>`).join('') : ''}`;
  $('#hmAll').onclick = () => show('best');
  out.querySelectorAll('[data-b]').forEach(t => t.onclick = () => bb[+t.dataset.b].go());
  out.querySelectorAll('[data-l]').forEach(t => t.onclick = () => LA[+t.dataset.l].go());
  out.querySelectorAll('[data-u]').forEach(t => t.onclick = () => UN[+t.dataset.u].go());
  out.querySelectorAll('.erow').forEach(b => b.onclick = () => { const [j, i] = b.dataset.i.split(':').map(Number), z = G.list[j]; homeOpen(z.sp, z.gs[i]); });
}

// ---------- Tennis (tennis.json from op_live.py; tennis_model.json from the tennis thread, paper only) ----------
let TEN_CAT = lsGet('betdesk.tencat', 'All');
const tenName = n => String(n || '').includes(',') ? String(n).split(',').map(s => s.trim()).reverse().join(' ') : String(n || '');
// J 2026-10-30: tap a match for its tile (model, every book, both players' model inputs) and log bets from it
let TSHEET = null;
if (!MODELS.includes('Tennis model')) MODELS.push('Tennis model');
const tenCur = v => v && (v.now || v.open);
function tenPrice(m){
  const best = [null, null], pn = m.books && m.books.pinnacle, pc = tenCur(pn);
  Object.entries(m.books || {}).forEach(([b, v]) => { const c = tenCur(v); if (!c || !myBook(b)) return; [1, 2].forEach(i => { if (c[i] && (!best[i - 1] || c[i] > best[i - 1][1])) best[i - 1] = [b, c[i]]; }); });
  const nv = pc && pc[1] && pc[2] ? (1/pc[1])/(1/pc[1] + 1/pc[2]) : null;
  return {best, nv, pn};
}
const tenModel = m => ((TENM && TENM.matches) || []).find(z => z.fid && z.fid === m.fid) || null;
// Tennis Model A + B (models/tennis_ab/live.py -> tennis_ab.json, 2026-11-05): A = point sim on our win % (total games, each
// player's games, set scores, fair lines); B = blend with the market, tier B lean (women, 6-10% edge, both 30+ rated) or C. No tier A.
let TENAB = null;
if (SPORT_LIST.includes('Tennis')) fetch('tennis_ab.json', {cache:'no-cache'}).then(r => r.ok ? r.json() : null).then(j => { if (j && j.matches){ TENAB = j; try { renderTennis(); } catch(e){} } }).catch(() => {});
const tenAB = m => ((TENAB && TENAB.matches) || []).find(z => z.fid && z.fid === m.fid) || null;
function renderTennis(){
  const out = $('#tennisOut'); if (!out) return;
  if (!TEN || !TEN.length){ out.innerHTML = '<div class="empty">No tennis odds loaded yet. They come with the morning odds pull.</div>'; return; }
  if (TSHEET){ const m = TEN.find(y => y.fid === TSHEET); if (m) return tenTile(out, m); TSHEET = null; }
  const now = Date.now(), all = [...TEN.filter(m => Date.parse(m.start) > now).sort((a, b) => a.start.localeCompare(b.start)),
    ...TEN.filter(m => Date.parse(m.start) <= now && now - Date.parse(m.start) <= DAY_H*36e5).sort((a, b) => b.start.localeCompare(a.start))];   // J 2026-11-06 full day
  const cats = ['All', ...new Set(all.map(m => m.cat))]; if (!cats.includes(TEN_CAT)) TEN_CAT = 'All';
  const L = all.filter(m => TEN_CAT === 'All' || m.cat === TEN_CAT);
  const row = (m, i) => {
    const {best, nv, pn} = tenPrice(m), z = tenModel(m), pm = z && z.p_model;
    const side = k => { const b = best[k]; return `<span class="t">${esc(tenName(k ? m.p2 : m.p1))}</span><span class="sc">${pm != null ? pct(k ? 1 - pm : pm, 0) : '–'}</span><span class="sb">${b ? amS(amOf(b[1])) : '–'}</span>`; };
    const ev = k => pm == null || !best[k] ? null : (k ? 1 - pm : pm)*best[k][1] - 1, eb = Math.max(ev(0) ?? -1, ev(1) ?? -1);
    const mv = pn && pn.open && pn.now && pn.open[1] && pn.now[1] && pn.open[1] !== pn.now[1] ? ` · Pinnacle ${amS(amOf(pn.open[1]))} → ${amS(amOf(pn.now[1]))}` : '';
    const st = Date.parse(m.start) <= now ? gStat('Tennis', {k:m.start}) : null;
    return `<button type="button" class="erow tn${st ? ' past' : ''}" data-i="${i}">${side(0)}${side(1)}<span class="meta"><span>${statChip(st && {t:st.fin ? 'Started' : 'Live', fin:st.fin})}${esc(kick({k:m.start}))} · ${esc(m.tournament.replace(/ (Women|Men) Singles$/, ''))}${z && z.surface ? ' · ' + esc(z.surface) : ''}${mv}</span><span>${nv != null ? `<span class="chip b">Pinnacle ${esc(tenName(m.p1).split(' ').slice(-1)[0])} ${pct(nv, 0)}</span> ` : ''}${z && z.flag ? '<span class="chip y">paper flag</span> ' : ''}${(() => { const q = tenAB(m); return q ? `<span class="chip">${q.A.tot_med} games</span> ` + (q.B.tier === 'B' ? `<span class="chip y">B lean ${esc(tenName(q.B.pick).split(' ').slice(-1)[0])} · paper</span> ` : q.B.u && q.B.u.rec !== 'skip' ? `<span class="chip u">Unproven ${q.B.u.score}: ${esc(tenName(q.B.pick).split(' ').slice(-1)[0])}</span> ` : '') : ''; })()}${eb >= 0.03 ? `<span class="chip b">model ${sgn(eb)} · paper</span>` : ''}</span></span></button>`; };
  out.innerHTML = `<p class="small muted">Paper only: the tennis model has not beaten closing prices in 2026, so nothing here is a tested bet. Columns: our model's win chance, then the best price at Colorado books. Tap a match for both players' ratings and form, every book, and to log a bet. Model ${TENM ? esc(String(TENM.made).slice(0, 16).replace('T', ' ')) + ' UTC' : 'not loaded'}.</p>
    <div class="chips" role="group">${cats.map(c => `<button type="button" data-c="${esc(c)}" aria-pressed="${c === TEN_CAT}">${esc(c)}</button>`).join('')}</div>
    ${L.length ? `<p class="small muted ablg"><b class="ka">%</b> our model <b class="kb">$</b> best price</p><div class="elist">${L.slice(0, 120).map(row).join('')}</div>` : '<div class="empty">No matches today in this category.</div>'}`;
  out.querySelectorAll('[data-c]').forEach(b => b.onclick = () => { TEN_CAT = b.dataset.c; lsSet('betdesk.tencat', TEN_CAT); renderTennis(); });
  out.querySelectorAll('.erow[data-i]').forEach(b => b.onclick = () => { TSHEET = L[+b.dataset.i].fid; renderTennis(); window.scrollTo(0, 0); });
}
function tenTile(out, m){
  const {best, nv} = tenPrice(m), z = tenModel(m) || {}, pm = z.p_model, pb = z.p_blend, N = [tenName(m.p1), tenName(m.p2)];
  const I = z.info || {}, P = [I.p1 || {}, I.p2 || {}];
  const f1 = v => v == null || v === '' ? '–' : typeof v === 'number' ? (Math.abs(v) >= 100 ? Math.round(v) : Math.abs(v) < 1 ? (v*100).toFixed(1) + '%' : (+v).toFixed(2)) : esc(String(v));
  const wl = v => Array.isArray(v) && v.length >= 2 ? `${v[0]}-${v[1]}` : '–';
  let stat = [['Elo', q => f1(q.elo)], [`${esc(z.surface || 'Surface')} Elo`, q => f1(q.elo_surf)], ['Games-won rating', q => f1(q.gw)],
    ['Serve rating', q => q.srv ?? q.srv_pct ?? '<span class="muted">no serve data</span>', 1], ['Return rating', q => q.ret ?? q.ret_pct ?? '<span class="muted">no serve data</span>', 1],
    ['Style', q => q.style ? esc(q.style) : '<span class="muted">under 15 sets</span>'], ['Games per set', q => f1(q.games_per_set)], ['Tiebreak rate', q => f1(q.tiebreak_rate)],
    ['Form, last 10', q => wl(q.form) + (q.form_str ? ` <span class="muted">${esc(q.form_str)}</span>` : '')], ['Form trend (Elo pts)', q => q.form_rating == null ? '–' : (q.form_rating > 0 ? '+' : '') + (+q.form_rating).toFixed(1)],
    [`Form on ${esc(z.surface || 'this surface')}`, q => wl(q.surf_form)], ['Ranking', q => f1(q.rank)], ['Matches rated', q => f1(q.n)]]
    .filter(([l, f, keep]) => keep || l !== 'Ranking' || P.some(q => q.rank != null));
  stat.forEach(r => { if (r[2]){ const g = r[1]; r[1] = q => { const v = g(q); return typeof v === 'number' ? f1(v) : v; }; } });
  const hasInfo = P.some(q => Object.keys(q).length);
  const ev = (k, p) => p == null || !best[k] ? null : (k ? 1 - p : p)*best[k][1] - 1;
  const side = k => { const b = best[k], e = ev(k, pm), e2 = ev(k, pb);
    return `<div class="abx ${k ? 'b' : 'a'}"><span class="lab">${esc(N[k])}</span>
      <div class="abs"><span>Our model</span><b>${pm != null ? pct(k ? 1 - pm : pm, 0) : '–'}</b></div>
      <p class="small">${pb != null ? `Blended with Pinnacle ${pct(k ? 1 - pb : pb, 0)} · ` : ''}Pinnacle no-vig ${nv != null ? pct(k ? 1 - nv : nv, 0) : '–'}</p>
      <p class="small">Best: <b>${b ? amS(amOf(b[1])) : '–'}</b> ${b ? esc(bkName(b[0])) : ''}${e != null ? ` · EV ours <span class="${e >= 0 ? 'pos' : 'neg'}">${sgn(e)}</span>` : ''}${e2 != null ? ` · blend <span class="${e2 >= 0 ? 'pos' : 'neg'}">${sgn(e2)}</span>` : ''}</p>
      ${pm != null ? `<p class="small muted">Fair price ${amS(+am(k ? 1 - pm : pm))}</p>` : ''}
      <button class="btn sm" type="button" data-log="${k}"${b ? '' : ' disabled'}>Log ${esc(N[k].split(' ').slice(-1)[0])}</button></div>`; };
  const bk = Object.entries(m.books || {}).filter(([, v]) => tenCur(v));
  const cell = (k, v, b) => { const o = v.open && v.open[k], c = v.now && v.now[k]; const lit = best[k - 1] && best[k - 1][0] === b;
    return `<td class="${lit ? 'best' : ''}">${c ? amS(amOf(c)) : o ? amS(amOf(o)) : '–'}${o && c && o !== c ? `<br><span class="muted">open ${amS(amOf(o))}</span>` : ''}</td>`; };
  const why = [];
  if (z.why) why.push(z.why);
  if (pm != null && nv != null){ const d = pm - nv, w = d >= 0 ? 0 : 1;
    why.push(Math.abs(d) < 0.02 ? 'Our model and Pinnacle agree on this match.' : `Our model is ${pct(Math.abs(d), 0)} higher on ${N[w]} than Pinnacle's no-vig price.`); }
  if (z.flag) why.push('Paper flag: the edge clears the tennis thread\'s 6% bar on a rated ITF women or Challenger match. Tracked on paper, not a tested bet.');
  if (pm == null) why.push('This match is not in the model file yet (it prices today and tomorrow each morning).');
  const q = tenAB(m), QA = q && q.A, QB = q && q.B;
  if (QB && QB.tier === 'B') why.push(`Model B lean (paper): ${tenName(QB.pick)} at ${amS(amOf(QB.price))} ${bkName(QB.book)}, blended edge ${sgn(QB.edge)}. ${QB.note}`);
  else if (QB && QB.edge != null) why.push(`Model B: pass (tier C). ${QB.note}${QB.u ? ' ' + uLine(QB.u) : ''}`);
  const pA = QA ? (() => { const Sx = Object.entries(QA.sets || {}).sort((a, b) => b[1] - a[1]);
    const hp = v => v == null ? '–' : (v > 0 ? '+' : '') + v;
    return `<div class="panel"><h3>Model A projection</h3><div class="tablewrap"><table class="et"><tbody>
      <tr><td class="t">Total games</td><td><b>${QA.tot_med}</b> <span class="muted">(80% range ${QA.tot_range[0]}-${QA.tot_range[1]}, fair line ${QA.tot_line_fair})</span></td></tr>
      <tr><td class="t">Games won</td><td>${esc(N[0])} ${QA.games_p1} · ${esc(N[1])} ${QA.games_p2}</td></tr>
      <tr><td class="t">Games handicap</td><td>${esc(N[0])} ${hp(QA.hcp_p1_fair)} <span class="muted">(fair)</span></td></tr>
      <tr><td class="t">Set score</td><td>${Sx.slice(0, 4).map(([k, v]) => `${esc(k)} ${pct(v, 0)}`).join(' · ')} <span class="muted">(first player's sets first)</span></td></tr>
      <tr><td class="t">Straight sets</td><td>${pct(QA.p_straight, 0)}${QA.best_of === 5 ? ' · best of 5' : ''}</td></tr>
      <tr><td class="t">Serve points won</td><td>${pct(QA.serve_pts[0], 0)} · ${pct(QA.serve_pts[1], 0)} <span class="muted">(sim inputs)</span></td></tr></tbody></table></div>
      <p class="small muted">Point-by-point sim on our win chance. 2021-25 test: total games about as close as bet365's closing line (miss 4.86 vs 4.84 games), so these are numbers, not bets.</p></div>`; })() : '';
  out.innerHTML = `<div class="row" style="justify-content:space-between"><button class="btn ghost sm" type="button" id="tnBack">‹ All matches</button><span class="small muted">${esc(kick({k:m.start}))}</span></div>
    <h2 class="gt">${esc(N[0])} <span class="muted">vs</span> ${esc(N[1])}</h2>
    <p class="small muted">${esc(m.tournament.replace(/ (Women|Men) Singles$/, ''))} · ${esc(m.cat)}${z.surface ? ' · ' + esc(z.surface) : ''}</p>
    <div class="abrow">${side(0)}${side(1)}</div>${q ? simBtn('Tennis') : ''}
    <details class="dis panel" open><summary><b>Why</b></summary>${why.map(t => `<p class="small">${esc(t)}</p>`).join('')}<p class="small muted">Paper only: the tennis model has not beaten closing prices in 2026.</p></details>
    ${pA}
    <div class="panel"><h3>Player inputs</h3>${hasInfo ? `<div class="tablewrap"><table class="et"><thead><tr><th></th><th>${esc(N[0])}</th><th>${esc(N[1])}</th></tr></thead><tbody>${stat.map(([l, f]) => `<tr><td class="t">${l}</td><td>${f(P[0])}</td><td>${f(P[1])}</td></tr>`).join('')}</tbody></table></div>`
      : '<p class="small muted">The ratings, serve numbers, style and form for each player are not in the model file yet. They show here once the tennis model writes them.</p>'}</div>
    <div class="panel"><h3>Every book</h3><div class="tablewrap"><table class="et"><thead><tr><th>Book</th><th>${esc(N[0])}</th><th>${esc(N[1])}</th></tr></thead><tbody>${bk.map(([b, v]) => `<tr><td class="t">${esc(bkName(b))}${myBook(b) ? '' : ' <span class="muted">ref</span>'}</td>${cell(1, v, b)}${cell(2, v, b)}</tr>`).join('')}</tbody></table></div>
      <p class="small muted">Lit = best price at Colorado books. Prices from the ${esc(String((bk[0] && bk[0][1].now || [''])[0]).slice(11, 16))} UTC pull; check the book before betting.</p></div>`;
  $('#tnBack').onclick = () => { TSHEET = null; renderTennis(); };
  out.querySelectorAll('[data-sim]').forEach(b => b.onclick = () => simOpen('TENNIS', {id:q.fid, p1:q.p1, p2:q.p2, home:N[0], away:N[1]}));
  simShow('TENNIS', out);
  out.querySelectorAll('[data-log]').forEach(b => b.onclick = () => { const k = +b.dataset.log, x = best[k]; if (!x) return;
    openBet({sport:'Tennis', market:'Moneyline', model:'Tennis model', desc:`${N[k]} ML vs ${N[1 - k]}`, side:'Side', line:'', odds:amOf(x[1]), push:0, units:'', book:bkName(x[0]),
      pm:pm != null ? (k ? 1 - pm : pm) : '', pf:nv != null ? (k ? 1 - nv : nv) : '', pb:pb != null ? (k ? 1 - pb : pb) : '', gm:{h:N[0], a:N[1]}, why:'Own read'}); });
}

// ---------- Data tab: Model A vs B, recommended but not placed ----------
const _renderCharts = renderCharts;
renderCharts = function(){
  if (S.resMode === 'ab' || S.resMode === 'missed'){
    charts.splice(0).forEach(c => c.destroy()); $('#chartSeg').innerHTML = ''; $('#chartKpis').innerHTML = '';
    $('#chartsBody').innerHTML = S.resMode === 'ab' ? abData() : missedData();
  } else _renderCharts();
  $('#resMode').innerHTML = [['bets', 'My bets'], ['ab', 'Model A vs B'], ['missed', 'Not placed'], ['models', 'Report card']].map(([k, l]) => `<button type="button" aria-pressed="${S.resMode === k}" data-k="${k}">${l}</button>`).join('');
  $('#resMode').querySelectorAll('button').forEach(b => b.onclick = () => { S.resMode = b.dataset.k; renderCharts(); });
};
function abData(){
  const G = MA && MA.grade && MA.grade.sports;
  if (!G || !Object.keys(G).length) return '<div class="empty">Model A vs B grading shows here once both models have graded games.</div>';
  const f = x => x == null ? '–' : (+x).toFixed(2);
  const rows = Object.entries(G).filter(([sp, g]) => g.rebuild_same_games && g.rebuild_same_games.n && g.today_same_games).map(([sp, g]) => { const a = g.rebuild_same_games || {}, b = g.today_same_games || {}, l = g.line_same_games || {};
    const wm = a.margin_mae != null && b.margin_mae != null ? (a.margin_mae < b.margin_mae ? 'A' : 'B') : '', wt = a.total_mae != null && b.total_mae != null ? (a.total_mae < b.total_mae ? 'A' : 'B') : '';
    return `<tr><td class="t">${esc(sp)}</td><td>${a.n ?? '–'}</td><td class="${wm === 'A' ? 'best' : ''}">${f(a.margin_mae)}</td><td class="${wm === 'B' ? 'best' : ''}">${f(b.margin_mae)}</td><td>${f(l.margin_mae)}${l.n ? ` <span class="muted">(${l.n})</span>` : ''}</td><td class="${wt === 'A' ? 'best' : ''}">${f(a.total_mae)}</td><td class="${wt === 'B' ? 'best' : ''}">${f(b.total_mae)}</td><td>${f(l.total_mae)}</td></tr>`; }).join('');
  return `<div class="card"><h3>Model A vs Model B</h3><p class="small muted">Average miss in points on the same finished games; lower is better and the better model is lit. A = the rebuild, B = the desk model as it stood that morning. The line column covers only games with a saved line (count in brackets). For CBB (from Nov 2) A is the desk's CBB v2 and B the previous model. Graded ${esc(MA.grade.made || '')}.</p>
    <div class="tablewrap"><table class="et"><thead><tr><th>Sport</th><th>Games</th><th>Margin A</th><th>B</th><th>Line</th><th>Total A</th><th>B</th><th>Line</th></tr></thead><tbody>${rows}</tbody></table></div>
    <p class="small muted">Bets and closing line value by model are on Report card.</p></div>`;
}
function missedData(){
  const LF = SCORE && SCORE.lineflags; if (!LF || !LF.rows) return '<div class="empty">Flag grading loads with the scorecard each morning.</div>';
  const last = n => { const w = String(n).toLowerCase().split(' ').filter(x => !/^(jr\.?|sr\.?|ii|iii|iv)$/.test(x)); return w[w.length - 1] || ''; };
  const placed = r => S.bets.some(b => b.date === r.d && String(b.desc || '').toLowerCase().includes(last(r.n)));
  const rows = LF.rows.filter(r => r.kind !== 'check' && !placed(r)), done = rows.filter(r => r.res === 'W' || r.res === 'L' || r.res === 'P');
  const w = done.filter(r => r.res === 'W').length, l = done.filter(r => r.res === 'L').length, u = done.reduce((s, r) => s + (+r.u || 0), 0);
  const cl = done.filter(r => r.clv != null), clv = cl.length ? cl.reduce((s, r) => s + r.clv, 0)/cl.length : null;
  const by = k => { const d = done.filter(r => r.kind === k); return d.length ? `${k === 'proven' ? 'Proven' : 'Flag'} ${d.filter(r => r.res === 'W').length}-${d.filter(r => r.res === 'L').length}, ${sgnPts(+d.reduce((s, r) => s + (+r.u || 0), 0).toFixed(2))}u` : ''; };
  const recent = done.slice().sort((a, b) => b.d.localeCompare(a.d)).slice(0, 15);
  return `<div class="kpis"><div class="kpi"><span>Not placed</span><b>${done.length}</b></div><div class="kpi"><span>Record</span><b>${w}-${l}</b></div><div class="kpi"><span>Units</span><b class="${u >= 0 ? 'pos' : 'neg'}">${sgnPts(+u.toFixed(2))}</b></div>${clv != null ? `<div class="kpi"><span>Avg CLV</span><b>${sgn(clv)}</b></div>` : ''}</div>
    <div class="card"><h3>Recommended, not placed</h3><p class="small muted">Every prop the desk flagged (Proven or Flag, not Check) that you did not log, graded at the flagged price for 1 unit. ${esc([by('proven'), by('flag')].filter(Boolean).join(' · '))}. ${LF.pending != null ? LF.pending + ' flags are still waiting on results.' : ''} Game-side recommendations join this list once their grading file exists.</p>
      ${recent.length ? `<div class="tablewrap"><table class="et"><thead><tr><th>Bet</th><th>Price</th><th>Result</th><th>Units</th></tr></thead><tbody>${recent.map(r => `<tr><td class="t">${esc(r.n)} ${esc(r.side.toLowerCase())} ${r.line} ${esc(STAT[r.k] || r.k)}<br><span class="muted">${esc(r.sp)} · ${esc(dayName(r.d))} · ${esc(poBk(r.book))}</span></td><td>${amS(r.price)}</td><td>${esc(r.res)}${r.act != null ? ` <span class="muted">(${r.act})</span>` : ''}</td><td class="${r.u >= 0 ? 'pos' : 'neg'}">${sgnPts(r.u)}</td></tr>`).join('')}</tbody></table></div>` : ''}</div>`;
}

// ---------- navigation: Home / Sports / Bets / Data ----------
const _show = show;
show = function(v){
  _show(v);
  const cur = ['home', 'best', 'slate', 'bets', 'results', 'settings'].find(k => !$('#v-' + k).hidden);
  $('header.top').classList.toggle('bbmode', cur === 'home' || cur === 'best' || cur === 'settings');
  if (cur === 'best') document.querySelectorAll('nav.tabs button').forEach(b => b.setAttribute('aria-selected', b.dataset.v === 'home'));
  if (cur === 'home') renderHome();
};
document.querySelectorAll('[data-go]').forEach(b => b.addEventListener('click', () => show(b.dataset.go)));
