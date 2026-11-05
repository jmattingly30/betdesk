import json, os, time, urllib.request, urllib.error, datetime as dt, sys
KEY=os.environ['ODDSPAPI_API_KEY']; OP=sys.argv[1]
M={str(m['marketId']):m for m in json.load(open(OP+'/markets.json'))}
def get(fid, books):
    url=f'https://api.oddspapi.io/v4/historical-odds?fixtureId={fid}&bookmakers={books}&apiKey={KEY}'
    for _ in range(8):
        try:
            with urllib.request.urlopen(url, timeout=120) as r: return json.load(r)
        except urllib.error.HTTPError as e:
            b=e.read().decode()
            if e.code==429: time.sleep(5.5); continue
            return {'_error':e.code,'b':b[:200]}
    return {'_error':'rl'}
ts=lambda s: dt.datetime.fromisoformat(s.replace('Z','+00:00'))
picks=[('af','NFL'),('af','NCAA, Regular Season'),('bb','NBA'),('hk','NHL'),('mlb','MLB'),('bbm','NCAA Division I, National Championship')]
out={}
for fx,tn in picks:
    F=[f for f in json.load(open(f'{OP}/fx_{fx}.json')) if f['tournamentName']==tn and f.get('statusId')==2]
    f=F[len(F)//2]
    h=get(f['fixtureId'],'pinnacle,draftkings,fanduel'); time.sleep(5.5)
    st=ts(f['startTime']); r={'match':f"{f['participant1Name']} v {f['participant2Name']} {f['startTime'][:10]}"}
    if '_error' in h: r['err']=h; out[tn]=r; print(tn,r); continue
    for bk,v in (h.get('bookmakers') or {}).items():
        ms=v.get('markets',{}); props=0; first=None; pts=0; names=set()
        for mid,m in ms.items():
            meta=M.get(mid,{}); 
            if meta.get('playerProp'): props+=1
            else: names.add(meta.get('marketName','?'))
            for o in m['outcomes'].values():
                for pl,hl in o['players'].items():
                    if pl!='0': props_pl=True
                    for x in hl:
                        pts+=1; t=ts(x['createdAt'])
                        if first is None or t<first: first=t
        r[bk]={'markets':len(ms),'prop_markets':props,'opened_hours_before':round((st-first).total_seconds()/3600,1) if first else None,'price_points':pts,'examples':sorted(names)[:6]}
    out[tn]=r; print(tn, json.dumps(r)[:900], flush=True)
json.dump(out,open(OP+'/probe_other_sports.json','w'),indent=1)
