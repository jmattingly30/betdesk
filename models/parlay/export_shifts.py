"""Write condition shifts into app/data/parlay_corr.json -> 'shifts'. dz = latent (probit) shift to a leg's OVER chance, shrunk for regression seen out of sample."""
import pandas as pd, numpy as np, json
P='/mnt/project-files/sports-analytics/app/data/parlay_corr.json'
R=pd.read_csv('shifts_all.csv')
SHRINK={'NFL':0.5,'NBA':0.5,'MLB':0.65,'NHL':0.7}
# rules the app can compute from the props files (NFL sp = team spread, negative = favourite; NBA sp = expected margin, POSITIVE = favourite; tot = team implied + opponent implied; xg = team expected goals from summed skater goal projections)
RULES={'NFL':{'fav 7+':[['sp','<=',-7]],'dog 7+':[['sp','>=',7]],'total <=40':[['tot','<=',40]],'total 50+':[['tot','>=',50]],'home':[['h','==',1]],'weeks 1-3':[['week','<=',3]],'weeks 15+':[['week','>=',15]],
          'wind 15+':[['wind','>=',15]],'wind 10-14':[['wind','>=',10],['wind','<=',14]],'cold <=35F':[['temp','<=',35]],'dome/closed':[['dome','==',1]]},
       'NBA':{'dog 10+':[['sp','<=',-10]],'dog 6-9.5':[['sp','>=',-9.5],['sp','<=',-6]],'fav 6-9.5':[['sp','>=',6],['sp','<=',9.5]],'fav 10+':[['sp','>=',10]],
          'total top 20%':[['tot','>=',236]],'total bottom 20%':[['tot','<=',221.5]],'home':[['h','==',1]],'Oct-Nov':[['month','in',[10,11]]],'Mar-Apr':[['month','in',[3,4]]],'back-to-back':[['rest','<=',1]],'3+ days rest':[['rest','>=',3]]},
       'NHL':{'home':[['h','==',1]],'big favourite (gf-ga exp top 15%)':[['xg_diff','>=',0.63]],'big underdog':[['xg_diff','<=',-0.62]],
          'high expected goals (top 20%)':[['xg_tot','>=',6.48]],'low expected goals (bottom 20%)':[['xg_tot','<=',5.73]],'back-to-back':[['rest','<=',1]],'opp back-to-back':[['o_rest','<=',1]]},
       'MLB':None}
sel=R[R.stable&(R.z_all.abs()>=3)&~R.leg.str.startswith('OPP_')&~R.leg.str.startswith('G_')].copy()
sel['dz']=sel.apply(lambda r:round(float(r.dz*SHRINK[r.sport]),3),axis=1)
T=json.load(open(P)); out={}
for sp,x in sel.groupby('sport'):
    out[sp]={'rules':RULES[sp] or sorted(x.condition.unique()),'effects':[{'cond':r.condition,'leg':r.leg,'dz':r.dz,'n':int(r.n),'over_shift_pts':round(float(r.sh_all*100),1)} for r in x.itertuples()]}
T['shifts']=out
T['shifts_note']=('Pre-game conditions that move how often a leg beats our projection (2016-25 NFL, 2025-26 NBA, NHL, MLB 2021-25). '
  'Apply: p_over_adj = Phi(Phi^-1(p_over) + sum of dz for active conditions), capped at +/-0.3. Kept only effects with the same sign in both halves and |z|>=3; '
  'dz shrunk by the share that survived out of sample. Conditions change hit rates but NOT how legs move together (tested: about as many correlation changes as random labels give).')
T['tail_factor']={'note':'multiply the copula joint for 3+ leg stacks: symmetric copula slightly overstates all-over stacks; all-under stacks run at or a bit above it',
  'over':{'NHL':{'3':0.98,'4':0.93},'MLB':{'3':0.99,'4':0.99},'NBA':{'3':0.996},'NFL':{'3':0.998,'4':0.997}},'under':{'NHL':{'3':1.005,'4':1.027},'MLB':{'3':1.003,'4':1.003},'NBA':{'3':1.005},'NFL':{'3':1.001,'4':1.002}}}
json.dump(T,open(P,'w'),separators=(',',':'))
print({k:len(v['effects']) for k,v in out.items()})
