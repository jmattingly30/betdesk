"""Re-inline app/redesign.js into the Bet Desk after the 2026-10-30 Concept E patch (redesign_patch.py) is already applied.
Replaces the block from the redesign header up to '// ---------- boot ----------' in app/bet-desk.html and writes
app/bet-desk-v2.html; copy it over bet-desk.html only after checking it. Usage: python3 app/redesign_inline.py"""
import os
HERE = os.path.dirname(os.path.abspath(__file__))
src = open(f'{HERE}/bet-desk.html').read(); js = open(f'{HERE}/redesign.js').read()
head, mark = '// ---------- Concept E redesign', '// ---------- boot ----------'
i, j = src.index(head), src.index(mark)
assert src.count(head) == 1 and i < j
out = src[:i] + js.rstrip() + '\n' + src[j:]
open(f'{HERE}/bet-desk-v2.html', 'w').write(out)
print('bet-desk-v2.html', len(out), 'bytes')
