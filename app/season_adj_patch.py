"""Bet Desk changes from J's 2026-10-26 season adjustments (thread "Build this season's model adjustments").

python3 season_adj_patch.py [bet-desk.html]     # edits the file in place; safe to run twice (skips applied edits)

1. CFB totals (card 5): v3's total stack now has a mismatch term (am = |margin|, am2 = margin^2 / 100), fit in
   models/cfb-v3/build.py and exported in cfb_model.json stack_total. Older cfb_model.json files without it price as before.
2. Big-disagreement badge (card 3): on NFL and CFB cards (and NBA/CBB when a band is set) a warning when our model's
   margin is far from the line (NFL 3+, CFB 7+, NBA 6+, CBB 6+ pts), with this season's record of our side at that gap
   size from scorecard.json `gapsize` (grade.py). Display only: no number changes.
3. NFL home-field tracker (card 10, watch only): Results > Model report card > Summary, NFL: home teams' margin vs the
   line and vs our model this season, by week, with the alert rule, from scorecard.json `nfl_home` (grade.py)."""
import sys, os
here = os.path.dirname(os.path.abspath(__file__))
path = sys.argv[1] if len(sys.argv) > 1 else f'{here}/bet-desk.html'
s = open(path, encoding='utf-8').read()

EDITS = [
 ('CFB total mismatch term',
  "t = st.t_p*tp + st.o_sum*os + st.const;",
  "t = st.t_p*tp + st.o_sum*os + st.const + (st.am||0)*Math.abs(m) + (st.am2||0)*m*m/100;"),

 ('badge helper',
  "function cfbMargin(C, h, a, hf){",
  """// Big-disagreement badge (J 2026-10-26, season adjustments card 3): display only. In 2026 the bigger our model's gap to
// the line, the worse it did (CFB 10+ pt gaps 28-41), so a large gap is flagged with this season's record at that size.
const GAP_WARN = {NFL:3, CFB:7, NBA:6, CBB:6};
function gapBadge(sport, gap){
  const lim = GAP_WARN[sport], g = Math.abs(gap);
  if (lim == null || !(g >= lim)) return '';
  const rows = ((SCORE && SCORE.gapsize) || []).filter(r => r[0]===sport && r[1] <= g).sort((x,y) => y[1]-x[1]);
  const r = rows[0], rec = r && (r[2]+r[3]) ? ` This season, graded ${esc(sport)} games where our model sat ${r[1]}+ pts from the line: our side went <b>${r[2]}-${r[3]}${r[4]?'-'+r[4]:''}</b>.` : '';
  return `<p class="small" style="border-left:3px solid var(--neg,#b3261e);padding:4px 8px"><b>Big disagreement:</b> our model is ${g.toFixed(1)} pts from the line. Big gaps have mostly been model misses this season, not edges.${rec}</p>`;
}
function cfbMargin(C, h, a, hf){"""),

 ('CFB badge',
  "rr.push([`${h[0]} ${sgnPts(L)}`, `${a[0]} ${sgnPts(-L)}`, ncdf(gap/C.sd_margin), 'CFB team', L]);",
  "rr.push([`${h[0]} ${sgnPts(L)}`, `${a[0]} ${sgnPts(-L)}`, ncdf(gap/C.sd_margin), 'CFB team', L]); var cfGapB = gapBadge('CFB', gap);"),
 ('CFB badge text',
  "${Math.abs(gap).toFixed(1)} pts toward ${gap>0?esc(h[0]):esc(a[0])} from the line.</p>`; }",
  "${Math.abs(gap).toFixed(1)} pts toward ${gap>0?esc(h[0]):esc(a[0])} from the line.</p>`; txt += cfGapB; }"),

 ('NFL badge',
  "    bctr = anc; if (!BLEND) ctr = est;\n",
  "    txt += gapBadge('NFL', gap);\n    bctr = anc; if (!BLEND) ctr = est;\n"),

 ('CFB note text',
  "with roster and transfer inputs added 2026-10-10 (returning QB quality, new or missing starting QB, returning and transferred-in skill production). 2022-25 holdout: margin error 12.6 vs 12.2 for the opener and 12.0 for the close. Bets on 2+ point gaps at the opener won 52.6% (2,197 bets) with +0.3 pts of closing line value,",
  "with roster and transfer inputs added 2026-10-10 (returning QB quality, new or missing starting QB, returning and transferred-in skill production), and since 2026-10-26 a market rating (each team's rating learned from its past closing lines) and a mismatch term in the total. 2022-25 holdout: margin error 12.3 vs 12.2 for the opener and 12.0 for the close. Bets on 2+ point gaps at the opener won 53.4% (1,914 bets) with +0.3 pts of closing line value,"),

 ('NFL home tracker',
  "  const rg = (C.recent||[]).filter(r => r.sp===sp).slice(0, 25);",
  """  const HF = sp==='NFL' && C.nfl_home;   // J 2026-10-26 (season adjustments card 10): watch NFL home field, don't change it yet
  if (HF) h += `<div class="card"><h3>NFL home field this season</h3><div class="tablewrap"><table class="tight"><thead><tr><th></th><th>This season</th><th>2021-25</th></tr></thead><tbody>
      <tr><td class="t">Home teams' average margin</td><td class="${HF.home_margin < HF.line - 1 ? 'neg' : ''}">${sgn1(HF.home_margin)}</td><td>${sgn1(HF.normal.home_margin)}</td></tr>
      <tr><td class="t">Closing line, home side</td><td>${sgn1(HF.line)}</td><td>${sgn1(HF.normal.line)}</td></tr>
      <tr><td class="t">Home teams vs the line</td><td>${sgn1(HF.vs_line)} (${HF.ats_w}-${HF.ats_l} ATS)</td><td>about even</td></tr>
      ${HF.model!=null ? `<tr><td class="t">Our model's home lean beyond the line</td><td>${sgn1(HF.lean)} (${HF.n_model} games)</td><td></td></tr>` : ''}
    </tbody></table></div>
    <p class="small">${HF.alert ? '<b>Alert:</b> home teams are more than 2 pts worse than the line and it is unlikely to be luck (t ' + HF.t + '). Time to re-estimate home field.' : `Watching, no change yet: home teams are ${Math.abs(HF.vs_line).toFixed(1)} pts ${HF.vs_line < 0 ? 'worse' : 'better'} than the line over ${HF.n} games (t ${HF.t}). The alert turns on when that passes 2 pts with t below -2; otherwise revisit at week 12. A home edge learned from recent seasons gave no gain in 2018-2026 tests.`}</p>
    <p class="small muted">By week (home margin / line): ${HF.weeks.map(w => `wk ${w[0]} ${sgn1(w[1])} / ${sgn1(w[2])}`).join(' · ')}</p></div>`;
  const rg = (C.recent||[]).filter(r => r.sp===sp).slice(0, 25);"""),
]

done = 0
for name, old, new in EDITS:
    if new in s: print('already applied:', name); continue
    n = s.count(old)
    if n != 1: sys.exit(f'STOP: anchor for "{name}" found {n} times; nothing written')
    s = s.replace(old, new, 1); done += 1; print('applied:', name)
open(path, 'w', encoding='utf-8').write(s)
print(done, 'edits written to', path)
