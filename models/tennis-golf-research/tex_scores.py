"""2026 results with set scores and closing odds from tennisexplorer daily results pages (robots.txt allows /results/).
One page per day, 3 s apart. Singles only. Writes ../../tex/day_YYYYMMDD.html cache and ../../tex/results_2026.csv."""
import os, re, sys, time, html, datetime as dt, urllib.request
import pandas as pd
OUT = '../../tex'


def fetch(d):
    p = f'{OUT}/day_{d:%Y%m%d}.html'
    if os.path.exists(p) and os.path.getsize(p) > 10000:
        return open(p, encoding='utf-8', errors='ignore').read()
    url = f'https://www.tennisexplorer.com/results/?type=all&year={d.year}&month={d.month:02d}&day={d.day:02d}'
    for t in range(3):
        try:
            s = urllib.request.urlopen(url, timeout=60).read().decode('utf-8', 'ignore'); break
        except Exception as e:
            time.sleep(10); s = ''
    open(p, 'w').write(s); time.sleep(3)
    return s


def cell(x):
    x = re.sub(r'<sup>.*?</sup>', '', x); x = re.sub(r'<[^>]+>', '', x)
    return html.unescape(x).strip().replace('\xa0', '')


def parse(s, d):
    rows = []
    ev = None
    for tr in re.findall(r'<tr[^>]*>.*?</tr>', s, flags=re.S):
        if 'class="head flags"' in tr:
            m = re.search(r'<td class="t-name" colspan="2"><a href="([^"]+)"', tr)
            ev = m.group(1) if m else None
            continue
        if ev is None or 'type=double' in ev: continue
        tid = re.search(r'id="(r\d+)(b?)"', tr)
        if not tid: continue
        name = re.search(r'<td class="t-name"><a href="(/player/[^"]+)">([^<]+)</a>', tr)
        if not name: continue
        res = re.search(r'<td class="result">([^<]*)</td>', tr)
        sc = [cell(x) for x in re.findall(r'<td class="score">(.*?)</td>', tr, flags=re.S)]
        odds = re.findall(r'<td class="course[w]?"[^>]*>([^<]*)</td>', tr)
        mid = re.search(r'match-detail/\?id=(\d+)', tr)
        rows.append(dict(day=d, ev=ev, rid=tid.group(1), second=tid.group(2) == 'b', pid=name.group(1), name=html.unescape(name.group(2)),
                         sets=cell(res.group(1)) if res else '', s=sc[:5], odds=odds, mid=mid.group(1) if mid else None))
    out = []
    for i in range(len(rows) - 1):
        a, b = rows[i], rows[i + 1]
        if a['second'] or not b['second'] or a['rid'] != b['rid']: continue
        ga = [int(x) for x in a['s'] if x.isdigit()]; gb = [int(x) for x in b['s'] if x.isdigit()]
        n = min(len(ga), len(gb))
        o = [float(x) for x in a['odds'] if re.match(r'^\d+\.\d+$', x.strip())]
        out.append(dict(day=a['day'], ev=a['ev'], mid=a['mid'], pa=a['name'], pb=b['name'], pid_a=a['pid'], pid_b=b['pid'],
                        sets_a=a['sets'], sets_b=b['sets'], games_a=sum(ga[:n]), games_b=sum(gb[:n]), nsets=n,
                        score=' '.join(f'{x}-{y}' for x, y in zip(ga[:n], gb[:n])), odds_a=o[0] if len(o) > 0 else None, odds_b=o[1] if len(o) > 1 else None))
    return out


if __name__ == '__main__':
    d0 = dt.date.fromisoformat(sys.argv[1]); d1 = dt.date.fromisoformat(sys.argv[2])
    allr = []
    d = d0
    while d <= d1:
        r = parse(fetch(d), d); allr += r
        print(d, len(r), flush=True)
        d += dt.timedelta(days=1)
    pd.DataFrame(allr).to_csv(f'{OUT}/results_{d0:%Y%m%d}_{d1:%Y%m%d}.csv', index=False)
