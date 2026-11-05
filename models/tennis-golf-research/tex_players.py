"""Men's singles match lists from tennisexplorer player pages, incl. Futures (men's ITF) which the daily results pages omit.
Usage: tex_players.py PIDS_JSON YEAR OUT_CSV   (pages cached in ../../tex/pl/, 3 s apart; resumable)
Rows: year, day(dd.mm.), section, ev, mid, pid_a, pid_b, surface, round, score, odds_a, odds_b, page_pid"""
import gzip, os, re, sys, json, time, html, csv, urllib.request
PL = '../../tex/pl'

def fetch(pid, year):
    slug = pid.strip('/').split('/')[-1]
    p = f'{PL}/{slug}_{year}.html.gz'
    if os.path.exists(p) and os.path.getsize(p) > 2000:
        return gzip.open(p, 'rt', encoding='utf-8', errors='ignore').read()
    req = urllib.request.Request(f'https://www.tennisexplorer.com/player/{slug}/?annual={year}', headers={'User-Agent': 'Mozilla/5.0'})
    s = ''
    for t in range(3):
        try:
            s = urllib.request.urlopen(req, timeout=60).read().decode('utf-8', 'ignore'); break
        except Exception:
            time.sleep(10)
    gzip.open(p, 'wt', encoding='utf-8').write(s); time.sleep(3)
    return s

def parse(s, year, page_pid):
    i = s.find(f'id="matches-{year}-1-data"')
    if i < 0: return []
    j = s.find(f'id="matches-{year}-2-data"', i)
    blk = s[i:j if j > 0 else None]
    out, sec, ev = [], '', ''
    for tr in re.split(r'<tr class="', blk)[1:]:
        if tr.startswith('head'):
            m = re.search(r'</span>([^<]+)</(?:a|span)>', tr)
            sec = html.unescape(m.group(1)).strip() if m else ''
            e = re.search(r'<a href="(/[^"]+/\d{4}/atp-men/)"', tr)
            ev = e.group(1) if e else ''
            continue
        day = re.search(r'<td class="first time">([\d.]+)</td>', tr)
        pl = re.findall(r'<a href="(/player/[^"]+/)"', tr.split('class="t-name"', 1)[-1].split('</td>', 1)[0])
        mid = re.search(r'match-detail/\?id=(\d+)"[^>]*>([^<]*)</a>', tr)
        if not (day and len(pl) == 2 and mid): continue
        surf = re.search(r'<span title="([a-z ]+)"', tr)
        rnd = re.search(r'<td class="round"[^>]*>([^<]*)</td>', tr)
        od = re.findall(r'<td class="course">([^<]*)</td>', tr)
        od = [o.strip() if re.match(r'^[\d.]+$', o.strip()) else '' for o in od] + ['', '']
        out.append([year, day.group(1), sec, ev, mid.group(1), pl[0], pl[1], surf.group(1) if surf else '',
                    html.unescape(rnd.group(1)).strip() if rnd else '', mid.group(2).strip(), od[0], od[1], page_pid])
    return out

if __name__ == '__main__':
    pids = json.load(open(sys.argv[1])); year = int(sys.argv[2]); outp = sys.argv[3]
    done = set()
    if os.path.exists(outp):
        for r in csv.reader(open(outp)): done.add(r[-1])
    new = not os.path.exists(outp)
    with open(outp, 'a', newline='') as f:
        w = csv.writer(f)
        if new: w.writerow(['year', 'day', 'section', 'ev', 'mid', 'pid_a', 'pid_b', 'surface', 'round', 'score', 'odds_a', 'odds_b', 'page_pid'])
        n = 0
        for pid in pids:
            if pid in done: continue
            rows = parse(fetch(pid, year), year, pid)
            w.writerows(rows or [[year, '', 'NONE', '', '', '', '', '', '', '', '', '', pid]]); f.flush()
            n += 1
            if n % 100 == 0: print(n, flush=True)
