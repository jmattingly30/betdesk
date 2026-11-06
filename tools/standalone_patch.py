"""Build the standalone page: python3 tools/standalone_patch.py app/bet-desk.html public/index.html
Adds only what running outside Claude needs (same as the 2026-11-02 standalone page): doctype, home-screen
manifest/icon tags, account.js (bets saved to J's account on the shared site + People card), Settings > Backup card (Export / Import bets), 'Saved on this phone' label. No model or design change."""
import sys
src, dst = sys.argv[1], sys.argv[2]
h = open(src, encoding='utf-8').read()
def sub(old, new):
    global h
    assert h.count(old) == 1, f'anchor not found once: {old[:60]!r}'
    h = h.replace(old, new)
HEAD = '<!doctype html><html><head><meta charset=utf8><meta name=viewport content="width=device-width,initial-scale=1"><style>:root{color-scheme:light}body{margin:0;padding:0;font:14px -apple-system,BlinkMacSystemFont,sans-serif;background:#fff;color:#000}img{max-width:100%}[hidden]:not([hidden=until-found i]){display:none!important}</style></head><body>\n'
sub("<title>J's Bet Desk</title>\n", '''<title>J's Bet Desk</title>
<script src="account.js"></script>
<link rel="manifest" href="manifest.webmanifest">
<link rel="apple-touch-icon" href="icon-180.png">
<meta name="apple-mobile-web-app-capable" content="yes">
<meta name="mobile-web-app-capable" content="yes">
<meta name="apple-mobile-web-app-status-bar-style" content="black-translucent">
<meta name="apple-mobile-web-app-title" content="Bet Desk">
<meta name="theme-color" content="#0f1115">
''')
sub('  <div class="card" id="dataPanel"></div>\n', '''  <div class="card" id="backupCard">
    <h3>Backup</h3>
    <p class="small muted">Your bets and settings save on this phone. Export a backup now and then, and use Import to move them to another device (or to bring them over from the old Bet Desk).</p>
    <div class="row"><button class="btn ghost" type="button" id="exportBets">Export bets</button><button class="btn ghost" type="button" id="importBets">Import bets</button><input type="file" id="importFile" accept="application/json,.json" hidden></div>
  </div>
  <div class="card" id="dataPanel"></div>
''')
sub("function toast(t){", r"""// Backup: export / import bets + settings as a JSON file (standalone app keeps them on the device)
document.getElementById('exportBets').addEventListener('click', () => {
  const blob = new Blob([JSON.stringify({bets:S.bets, settings:S.settings, lines:S.lines, exported:new Date().toISOString()}, null, 1)], {type:'application/json'});
  const a = document.createElement('a'); a.href = URL.createObjectURL(blob); a.download = 'betdesk-backup-' + new Date().toISOString().slice(0,10) + '.json';
  document.body.appendChild(a); a.click(); setTimeout(() => { URL.revokeObjectURL(a.href); a.remove(); }, 1000);
});
document.getElementById('importBets').addEventListener('click', () => document.getElementById('importFile').click());
document.getElementById('importFile').addEventListener('change', async e => {
  const f = e.target.files && e.target.files[0]; e.target.value = ''; if (!f) return;
  try {
    const d = JSON.parse(await f.text()); const inc = Array.isArray(d) ? d : (d.bets || []);
    if (!Array.isArray(inc)) throw new Error('no bets');
    const byId = new Map(S.bets.map(b => [b.id, b])); let added = 0, updated = 0;
    inc.forEach(b => { if (!b || !b.id) return; byId.has(b.id) ? updated++ : added++; byId.set(b.id, b); });
    if (!confirm(`Import ${added} new and update ${updated} existing bets?`)) return;
    S.bets = [...byId.values()];
    if (d.settings && Object.keys(d.settings).length) S.settings = {...DEF, ...S.settings, ...d.settings};
    if (S.db){ for (const b of inc) if (b && b.id) await S.db.collection('bets').doc(b.id).set(b); }
    else lsSave();
    renderAll(); toast(`Imported ${added + updated} bets`);
  } catch(err){ toast('Could not read that backup file'); }
});
function toast(t){""")
sub("$('#saveState').textContent = 'Saving on this device only'", "$('#saveState').textContent = 'Saved on this phone'")
open(dst, 'w', encoding='utf-8').write(HEAD + h.rstrip('\n') + '\n\n</body></html>')
print('built', dst)
