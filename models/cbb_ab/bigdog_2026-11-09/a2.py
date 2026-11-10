import json, numpy as np, collections
H='/mnt/project-files/sports-analytics/models/cbb_ab/history/'
runs=[json.loads(l) for l in open(H+'runs.jsonl')]; fin=json.load(open(H+'finals.json'))
first={}; last={}; allfirst={}
for r in runs:
    last[r['id']]=r
    z=r['B'].get('spread')
    if not z or not z.get('bet'): continue
    if r['id'] not in allfirst: allfirst[r['id']]=(r,z)
    if z['tier'] in 'AB' and r['id'] not in first: first[r['id']]=(r,z)
def res(r,z):
    f=fin.get(r['id']); 
    if not f: return None
    m=f['h']-f['a']; pt=z['bet']['pt']
    x=(m+pt) if z['side']==r['h'] else (-m+pt); return x
def show(sel,label):
    B=collections.defaultdict(lambda:[0,0])
    for gid,(r,z) in sel.items():
        x=res(r,z)
        if x is None or x==0: continue
        dog=(z['market']>0)!=(z['side']==r['h']); fs=abs(z['market'])
        b='15+' if fs>=15 else '7-15' if fs>=7 else '<7'
        k=(b,'dog' if dog else 'fav'); B[k][0]+=1; B[k][1]+=x>0
    print(label); 
    for k in sorted(B): n,w=B[k]; print(' ',k,n,round(w/n,3))
show(first,'LIVE tier A/B first-seen spread picks by line size')
show({k:v for k,v in first.items() if v[1]['tier']=='A'},'LIVE tier A only')
# all games (tier ABC) on our side, gap3+ at first sight
show({k:v for k,v in allfirst.items() if abs(v[1]['gap'])>=3},'LIVE all games gap3+ first seen')
# model calibration live: our margin vs market vs actual by fav size
rows=[]
for gid,r in last.items():
    f=fin.get(gid); z=r['B'].get('spread')
    if not f or not z: continue
    s=1 if z['market']>=0 else -1
    rows.append((abs(z['market']),(r['A'][0]-r['A'][1])*s,(f['h']-f['a'])*s,z['market']*s))
rows=np.array(rows)
for lo,hi in [(0,7),(7,15),(15,25),(25,60)]:
    k=(rows[:,0]>=lo)&(rows[:,0]<hi)
    print(f'live last-seen {lo}-{hi} n={k.sum()} market={rows[k,3].mean():.2f} ours={rows[k,1].mean():.2f} actual={rows[k,2].mean():.2f}')
print('--- split by game date')
for lab,cond in [('Nov3-5',lambda d:d<'2026-11-06'),('Nov6+',lambda d:d>='2026-11-06')]:
    show({k:v for k,v in first.items() if cond(v[0]['d'])},'tier A/B '+lab)
    rr=[]
    for gid,r in last.items():
        f=fin.get(gid); z=r['B'].get('spread')
        if not f or not z or not cond(r['d']): continue
        s=1 if z['market']>=0 else -1; rr.append((abs(z['market']),(r['A'][0]-r['A'][1])*s,(f['h']-f['a'])*s,z['market']*s))
    rr=np.array(rr)
    for lo,hi in [(0,7),(7,15),(15,25),(25,60)]:
        k=(rr[:,0]>=lo)&(rr[:,0]<hi)
        if k.sum(): print(f'  {lo}-{hi} n={k.sum()} market={rr[k,3].mean():.2f} ours={rr[k,1].mean():.2f} actual={rr[k,2].mean():.2f}')
