"""Men's Futures (ITF) matches from cached tennisexplorer player pages -> results_all-style CSV (winner first) with surface.
Usage: fut_rows.py OUT_CSV YEAR [YEAR...]"""
import glob, gzip, re, sys, html
import pandas as pd
import tex_players
out = {}
for y in sys.argv[2:]:
    for f in glob.glob(f'{tex_players.PL}/*_{y}.html.gz'):
        s = gzip.open(f, 'rt', errors='ignore').read()
        i = s.find(f'id="matches-{y}-1-data"')
        if i < 0: continue
        j = s.find(f'id="matches-{y}-2-data"', i); blk = s[i:j if j > 0 else None]
        sec = ''
        for tr in re.split(r'<tr class="', blk)[1:]:
            if tr.startswith('head'):
                m = re.search(r'</span>([^<]+)</(?:a|span)>', tr); sec = m.group(1).strip() if m else ''; continue
            if not sec.startswith('Futures'): continue
            day = re.search(r'<td class="first time">(\d+)\.(\d+)\.</td>', tr)
            tn = tr.split('class="t-name"', 1)[-1].split('</td>', 1)[0]
            pl = re.findall(r'<a href="(/player/[^"]+/)"[^>]*>(?:<strong>)?([^<]+)', tn)
            mid = re.search(r'match-detail/\?id=(\d+)"[^>]*>([^<]*)</a>', tr)
            if not (day and len(pl) == 2 and mid) or mid.group(1) in out: continue
            sets = [(int(a), int(b)) for a, b in re.findall(r'(\d+)-(\d+)', mid.group(2))]
            surf = re.search(r'<span title="([a-z ]+)"', tr)
            od = [o.strip() for o in re.findall(r'<td class="course">([^<]*)</td>', tr)] + ['', '']
            sets = [(1, 0) if a > b else (0, 1) if k == 2 and max(a, b) >= 10 else (a, b) for k, (a, b) in enumerate(sets)]  # match tiebreak = 1 game
            sa = sum(a > b for a, b in sets); sb = sum(b > a for a, b in sets)
            if sets and sa <= sb: sa = sb + 1  # retirement: winner is listed first
            out[mid.group(1)] = dict(day=f'{y}-{day.group(2)}-{day.group(1)}', ev=f'/futures-itf/{y}/atp-men/', mid=int(mid.group(1)),
                                     pa=html.unescape(pl[0][1]).strip(), pb=html.unescape(pl[1][1]).strip(), pid_a=pl[0][0], pid_b=pl[1][0],
                                     sets_a=sa, sets_b=sb, games_a=sum(a for a, b in sets), games_b=sum(b for a, b in sets), nsets=len(sets),
                                     score=mid.group(2).replace(',', ''), odds_a=pd.to_numeric(od[0], errors='coerce'), odds_b=pd.to_numeric(od[1], errors='coerce'),
                                     surf=surf.group(1) if surf else '')
d = pd.DataFrame(out.values()).sort_values('day'); d.to_csv(sys.argv[1], index=False)
print(len(d), d.day.min(), d.day.max(), (d.sets_a > d.sets_b).mean(), d.surf.value_counts().to_dict())
