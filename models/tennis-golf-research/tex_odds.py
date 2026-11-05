"""Opening and closing moneyline odds per bookmaker from tennisexplorer match-detail pages (robots.txt allows them).
Each book cell holds the current price, its change history with times, and the 'Opening odds' row.
Usage: tex_odds.py MIDS_FILE OUT_JSONL   (pages cached in ../../tex/md/, 3 s apart; resumable)"""
import gzip, os, re, sys, json, time, html, urllib.request
MD = '../../tex/md'

def fetch(mid):
    p = f'{MD}/{mid}.html.gz'
    if os.path.exists(p) and os.path.getsize(p) > 2000:
        return gzip.open(p, 'rt', encoding='utf-8', errors='ignore').read()
    req = urllib.request.Request(f'https://www.tennisexplorer.com/match-detail/?id={mid}', headers={'User-Agent': 'Mozilla/5.0'})
    s = ''
    for t in range(3):
        try:
            s = urllib.request.urlopen(req, timeout=60).read().decode('utf-8', 'ignore'); break
        except Exception:
            time.sleep(10)
    gzip.open(p, 'wt', encoding='utf-8').write(s); time.sleep(3)
    return s

def cell(td):
    """-> dict(now, open, open_t, hist=[(t, price)...] newest first)"""
    now = re.search(r'<div class="odds-in[^"]*">([\d.]+)', td)
    rows = re.findall(r'<tr><td>([\d. :]+)</td><td class="bold">([\d.]+)</td>', td)
    op = None
    if 'Opening odds' in td:
        tail = td.split('Opening odds', 1)[1]
        m = re.search(r'<td>([\d. :]+)</td><td class="bold">([\d.]+)</td>', tail)
        if m: op = (m.group(1).strip(), float(m.group(2)))
        hist = re.findall(r'<tr><td>([\d. :]+)</td><td class="bold">([\d.]+)</td>', td.split('Opening odds', 1)[0])
    else:
        hist = rows
    nowp = float(now.group(1)) if now else None
    if op is None and nowp:  # never changed: the current price is the opener (time unknown)
        op = (None, nowp)
    return dict(now=nowp, open=op[1] if op else None, open_t=op[0] if op else None, hist=[(t.strip(), float(p)) for t, p in hist])

def players(s):
    """page-order player ids (first two distinct /player/ links above the odds tables)"""
    ids = []
    for m in re.findall(r'/player/[^"]+/', s[:max(s.find('oddsMenu-1-data'), 0)]):
        if m not in ids: ids.append(m)
    return ids[:2]

def parse(s):
    i = s.find('oddsMenu-1-data'); j = s.find('oddsMenu-2-data')
    if i < 0: return {}
    blk = s[i:j if j > 0 else None]
    out = {}
    for tr in blk.split('<td class="first tl">')[1:]:
        name = re.search(r'<span class="t">([^<]+)</span>', tr)
        k1 = re.search(r'<td class="k1[^"]*">(.*?)</td>\s*<td class="k2[^"]*">', tr, flags=re.S)
        k2 = re.search(r'<td class="k2[^"]*">(.*)$', tr, flags=re.S)
        if not (name and k1 and k2): continue
        out[html.unescape(name.group(1))] = dict(a=cell(k1.group(1)), b=cell(k2.group(1)))
    return out

if __name__ == '__main__':
    mids = [str(m) for m in json.load(open(sys.argv[1]))]
    outp = sys.argv[2]
    done = set()
    if os.path.exists(outp):
        for l in open(outp):
            try: done.add(json.loads(l)['mid'])
            except Exception: pass
    n = 0
    with open(outp, 'a') as f:
        for m in mids:
            if m in done: continue
            books = parse(fetch(m))
            f.write(json.dumps(dict(mid=m, books=books)) + '\n'); f.flush()
            n += 1
            if n % 100 == 0: print(n, flush=True)
