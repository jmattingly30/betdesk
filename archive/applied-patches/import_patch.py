"""Adds an Import / export bets card to Settings (2026-11-08). Run once on bet-desk-v2.html; refuses to run twice."""
import sys, pathlib
p = pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else 'bet-desk-v2.html')
h = p.read_text()
if 'id="ioCard"' in h: sys.exit('already patched')
CARD = '''  <div class="card" id="ioCard">
    <h3>Import / export bets</h3>
    <p class="small muted">Bring bets over from the GitHub site (or any Bet Desk backup). Pick the backup file, or paste its text. New bets are added; a bet already here is only changed if the backup's copy is newer and different. Nothing is deleted.</p>
    <label>Backup file (.json)<input id="ioFile" type="file" accept=".json,application/json,text/plain"></label>
    <label>Or paste the backup text<textarea id="ioText" rows="3" placeholder='{"app":"betdesk","bets":[...]}'></textarea></label>
    <div class="row"><button class="btn ghost" type="button" id="ioCheck">Check backup</button><button class="btn" type="button" id="ioGo" hidden>Import</button><button class="btn ghost" type="button" id="ioExport">Copy my bets</button></div>
    <div id="ioOut" class="small"></div>
  </div>
'''
anchor = '  <div class="card" id="dataPanel"></div>\n'
assert h.count(anchor) == 1
h = h.replace(anchor, CARD + anchor)
JS = r'''
// ---------- import / export bets (2026-11-08): same format as the GitHub site's Backup card ----------
(() => {
  const STAMP = new Set(['updated']);
  const same = (a, b) => { const k = new Set([...Object.keys(a), ...Object.keys(b)]); for (const x of k){ if (STAMP.has(x)) continue; if (JSON.stringify(a[x] ?? '') !== JSON.stringify(b[x] ?? '')) return false; } return true; };
  const tms = v => { const t = Date.parse(v || ''); return Number.isFinite(t) ? t : 0; };
  let plan = null;
  const out = html => { $('#ioOut').innerHTML = html; };
  function parse(txt){
    let d; try { d = JSON.parse(txt); } catch(e){ throw new Error('That is not valid JSON. Paste the whole backup, from the first { to the last }.'); }
    const arr = Array.isArray(d) ? d : Array.isArray(d && d.bets) ? d.bets : null;
    if (!arr) throw new Error('No bets list found in that backup.');
    const bets = arr.filter(b => b && typeof b === 'object' && (b.desc || b.player || b.market)).map(b => {
      const c = JSON.parse(JSON.stringify(b));
      if (!c.id || !/^[A-Za-z0-9_\-.~:@+]{1,200}$/.test(String(c.id))) c.id = 'b' + Date.now().toString(36) + Math.random().toString(36).slice(2, 8);
      c.id = String(c.id); if (!c.result) c.result = 'pending';
      return c;
    });
    return {bets, skipped: arr.length - bets.length, exported: d && d.exported};
  }
  function check(txt){
    $('#ioGo').hidden = true; plan = null;
    if (!txt.trim()){ out('<span class="wrn">Pick a file or paste the backup first.</span>'); return; }
    let r; try { r = parse(txt); } catch(e){ out(`<span class="wrn">${esc(e.message)}</span>`); return; }
    const have = new Map(S.bets.map(b => [b.id, b]));
    // same bet logged on two desks gets two ids: match on what was bet too
    const key = b => [b.sport, b.date, String(b.desc || b.player || '').toLowerCase().replace(/\s+/g, ' ').trim(), num(b.line), odds(b.odds)].join('|');
    const keys = new Set(S.bets.map(key));
    const add = [], upd = []; let keep = 0, older = 0;
    r.bets.forEach(b => {
      const o = have.get(b.id);
      if (!o){ const k = key(b); if (keys.has(k)) keep++; else { keys.add(k); add.push(b); } }
      else if (same(o, b)) keep++;
      else if (tms(b.updated) > tms(o.updated)) upd.push(b);
      else older++;
    });
    plan = [...add, ...upd];
    const when = r.exported ? ` (saved ${esc(String(r.exported).slice(0, 10))})` : '';
    const list = add.slice(0, 8).map(b => `<li>${esc(b.date || '')} ${esc(b.sport || '')} ${esc(b.desc || b.player || '')} ${esc(b.odds || '')} · ${esc(b.result)}</li>`).join('') + (add.length > 8 ? `<li>…and ${add.length - 8} more</li>` : '');
    out(`<p>Backup has <b>${r.bets.length}</b> bets${when}: <b class="pos">${add.length} new</b>, <b>${upd.length}</b> newer copies of bets here, ${keep} already here, ${older} older than what's here (left alone)${r.skipped ? `, ${r.skipped} unreadable rows skipped` : ''}.</p>${list ? `<ul>${list}</ul>` : ''}${plan.length ? '' : '<p class="muted">Nothing to import: this desk already has everything in that backup.</p>'}`);
    $('#ioGo').hidden = !plan.length; $('#ioGo').textContent = `Import ${plan.length} bet${plan.length === 1 ? '' : 's'}`;
  }
  $('#ioFile').addEventListener('change', e => {
    const f = e.target.files && e.target.files[0]; if (!f) return;
    const rd = new FileReader(); rd.onload = () => { $('#ioText').value = rd.result; check(String(rd.result)); }; rd.onerror = () => out('<span class="wrn">Could not read that file.</span>'); rd.readAsText(f);
  });
  $('#ioCheck').addEventListener('click', () => check($('#ioText').value));
  $('#ioGo').addEventListener('click', async () => {
    if (!plan || !plan.length || notReady()) return;
    const go = $('#ioGo'); go.disabled = true; let ok = 0, bad = 0;
    for (const b of plan){
      try {
        if (S.db) await S.db.collection('bets').doc(b.id).set(b);
        else { const i = S.bets.findIndex(x => x.id === b.id); i >= 0 ? S.bets[i] = b : S.bets.push(b); }
        ok++; go.textContent = `Importing ${ok}/${plan.length}…`;
      } catch(e){ bad++; }
    }
    if (!S.db){ lsSave(); renderAll(); }
    go.disabled = false; go.hidden = true; plan = null; $('#ioText').value = ''; $('#ioFile').value = '';
    out(`<p class="${bad ? 'wrn' : 'pos'}">Imported ${ok} bet${ok === 1 ? '' : 's'}${bad ? `; ${bad} failed, try Check backup again` : ''}. They're on the Bets tab.</p>`);
    toast(`Imported ${ok} bets`);
  });
  $('#ioExport').addEventListener('click', () => {
    const txt = JSON.stringify({app:'betdesk', version:1, exported:new Date().toISOString(), bets:S.bets.map(b => ({...b}))}, null, 1);
    $('#ioText').value = txt;
    const done = ok => out(`<p>${ok ? 'Copied to your clipboard. ' : ''}The backup (${S.bets.length} bets) is in the box above${ok ? '' : ': tap it, select all, copy'}. Paste it into a note or file to keep it.</p>`);
    try { navigator.clipboard.writeText(txt).then(() => done(true), () => done(false)); } catch(e){ done(false); }
  });
})();
'''
anchor2 = '\nboot();\n})();\n</script>'
assert h.count(anchor2) == 1
h = h.replace(anchor2, '\n' + JS + anchor2)
p.write_text(h)
print('patched', p)
