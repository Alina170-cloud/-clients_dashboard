import numpy as np, json, datetime as dt
rng=np.random.default_rng(7)
W=52; LAST=dt.date(2026,9,28)
wk=lambda w: LAST-dt.timedelta(weeks=W-1-w)
N0=1_150_000; NJ=int(N0*0.0042*W); N=N0+NJ
join=np.concatenate([np.full(N0,-1),rng.integers(0,W,NJ)]).astype(np.int16)
dl=join+rng.geometric(0.0011,N); dl=np.where(dl>W+5,999,dl).astype(np.int32)
CH=['sms','email','push']; C=3
pv=[.93,.62,.41]; pc=[.52,.71,.86]; pu=[.0011,.0016,.0024]
use=[.2,.65,.75]; cap=[1,2,3]; lam=[0,.5,.9]; dv=[.96,.97,.82]
valid=[rng.random(N)<pv[c] for c in range(C)]
cons0=[rng.random(N)<pc[c] for c in range(C)]
uns=[]
for c in range(C):
    u=np.maximum(join,0)+rng.geometric(pu[c],N); uns.append(np.where(u>W+5,999,u).astype(np.int32))
never=rng.random(N)<.12
prop=np.where(never,0,rng.beta(1.2,2.5,N)).astype(np.float32)
block=rng.random(N)<.05
rng2=np.random.default_rng(11)
fr=np.clip((join-20)/31,0,1)
valid[1]=valid[1]&~(rng2.random(N)<0.5*fr)
for c in range(C): cons0[c]=cons0[c]&~(rng2.random(N)<0.35*fr)
PER={'4w':4,'12w':12,'26w':26}
wins={}
for p,n in PER.items():
    wins[(p,'cur')]=(W-n,W-1); wins[(p,'prev')]=(W-2*n,W-1-n)
acc={k:dict(tg=[np.zeros(N,bool) for _ in CH],pol=[np.zeros(N,bool) for _ in CH],
            msg=[np.zeros(N,np.int16) for _ in CH],dl=[np.zeros(N,bool) for _ in CH],
            polok=np.zeros(N,bool)) for k in wins}
ex=lambda w:(join<=w)&(dl>w)
av=lambda c,w:ex(w)&valid[c]&cons0[c]&(uns[c]>w)
dyn=[]
for w in range(W):
    e=ex(w); mult=0.78+0.5*w/51+0.1*np.sin(w/3.1)-(0.22 if w in(44,45) else 0); intg=e&(rng.random(N)<prop*mult); pok=(~block)&(rng.random(N)<.88)
    a=[av(c,w) for c in range(C)]
    tg=[intg&a[c]&(rng.random(N)<use[c]) for c in range(C)]
    pol=[tg[c]&pok&(rng.random(N)<.9) for c in range(C)]
    m=[(pol[c]*np.minimum(cap[c],1+rng.poisson(lam[c],N))).astype(np.int16) for c in range(C)]
    m=[np.where(rng.random(N)<.965,x,0).astype(np.int16) for x in m]
    exs=np.maximum(0,m[0]+m[1]+m[2]-4); m[2]=m[2]-np.minimum(m[2],exs)
    snt=[x>0 for x in m]
    d=[snt[c]&(rng.random(N)<dv[c]) for c in range(C)]
    for k,(s,t) in wins.items():
        if s<=w<=t:
            A=acc[k]
            for c in range(C):
                A['tg'][c]|=tg[c]; A['pol'][c]|=pol[c]; A['msg'][c]+=m[c]; A['dl'][c]|=d[c]
            A['polok']|=pok
    if w>=0:
        anyv=e&(valid[0]|valid[1]|valid[2]); anya=a[0]|a[1]|a[2]
        tga=tg[0]|tg[1]|tg[2]; pola=pol[0]|pol[1]|pol[2]; sa=snt[0]|snt[1]|snt[2]; da=d[0]|d[1]|d[2]
        vals=dict(total=e.sum(),valid=anyv.sum(),consent=anya.sum(),tg=tga.sum(),policy=pola.sum(),
                  sent=sa.sum(),delivered=da.sum(),unutilized=(anya&pok&~sa).sum(),rest=(anya&~tga).sum())
        for s,v in vals.items(): dyn.append(dict(week=wk(w).isoformat(),channel='all',series=s,clients=int(v)))
        for c in range(C):
            vc=dict(total=e.sum(),valid=(e&valid[c]).sum(),consent=a[c].sum(),tg=tg[c].sum(),policy=pol[c].sum(),sent=snt[c].sum(),delivered=d[c].sum(),unutilized=(a[c]&pok&~snt[c]).sum(),rest=(a[c]&~tg[c]).sum())
            for s,v in vc.items(): dyn.append(dict(week=wk(w).isoformat(),channel=CH[c],series=s,clients=int(v)))
pct=lambda a,b: round(100*a/b,1) if b else 0.0
def calc(p,which):
    s,t=wins[(p,which)]; n=t-s+1; A=acc[(p,which)]; e=ex(t)
    a=[av(c,t) for c in range(C)]
    R={}
    def fun(vm,am,tgm,polm,sm,dm,k=None):
        k=e if k is None else e&k
        return [int(k.sum()),int((vm&k).sum()),int((am&k).sum()),int((tgm&k).sum()),int((polm&k).sum()),int((sm&k).sum()),int((dm&k).sum())]
    tgc=[A['tg'][c]&a[c] for c in range(C)]; polc=[A['pol'][c]&a[c] for c in range(C)]
    msg=[np.where(a[c],A['msg'][c],0) for c in range(C)]; sc=[x>0 for x in msg]; dc=[A['dl'][c]&a[c] for c in range(C)]
    U=lambda L:L[0]|L[1]|L[2]
    anya=U(a); mt=msg[0]+msg[1]+msg[2]
    R['fun']={'all':fun(e&U(valid),anya,U(tgc),U(polc),U(sc),U(dc))}
    for c in range(C): R['fun'][CH[c]]=fun(e&valid[c],a[c],tgc[c],polc[c],sc[c],dc[c])
    R['func']={}
    for coh,km in [('new',join>=s),('existing',join<s)]:
        R['func'][(coh,'all')]=fun(e&U(valid),anya,U(tgc),U(polc),U(sc),U(dc),km)
        for c in range(C): R['func'][(coh,CH[c])]=fun(e&valid[c],a[c],tgc[c],polc[c],sc[c],dc[c],km)
    R['unutil']=int((anya&A['polok']&(mt==0)).sum())
    R['unutil_c']={CH[c]:int((a[c]&A['polok']&(msg[c]==0)).sum()) for c in range(C)}
    capi=np.minimum(4,sum(a[c]*cap[c] for c in range(C)))
    R['msgs']={'total':int(mt.sum())}; R['max']={'total':int(capi.sum())*n}
    R['avail']={'total':int(anya.sum())}; R['tgn']={'total':int(U(tgc).sum())}; R['reach']={'total':int(U(sc).sum())}
    R['maxpc']={'total':4*n}
    for c in range(C):
        R['msgs'][CH[c]]=int(msg[c].sum()); R['max'][CH[c]]=int(a[c].sum())*cap[c]*n
        R['avail'][CH[c]]=int(a[c].sum()); R['tgn'][CH[c]]=int(tgc[c].sum()); R['reach'][CH[c]]=int(sc[c].sum()); R['maxpc'][CH[c]]=cap[c]*n
    R['util']=pct(R['msgs']['total'],R['max']['total'])
    combo={}
    code=a[0]*1+a[1]*2+a[2]*4
    for k in range(8): combo[k]=int((e&(code==k)).sum())
    R['combo']=combo
    R['nocontact']=int((e&~U(valid)).sum()); R['noconsent']=combo[0]-R['nocontact']
    B=[0,1,3,6,11,21,10**6]; R['dist']={}
    def dist(x,mask): 
        h=np.histogram(x[mask],bins=B)[0]; return [int(v) for v in h]
    R['dist']['total']=dist(mt,anya)
    for c in range(C): R['dist'][CH[c]]=dist(msg[c],a[c])
    R['bc']={}
    for hn,hx,hm in [('total',mt,anya)]+[(CH[c],msg[c],a[c]) for c in range(C)]:
        idx=np.digitize(hx,B)-1
        R['bc'][hn]=[]
        for bi in range(len(B)-1):
            mk=hm&(idx==bi)
            R['bc'][hn].append([(int(msg[m][mk].sum()),int((msg[m][mk]>0).sum())) for m in range(C)])
    ch={}
    exat=lambda wkarr:(join<=wkarr)
    for c in range(C):
        un=valid[c]&cons0[c]&(uns[c]>=s)&(uns[c]<=t)&(dl>uns[c]); de=valid[c]&(dl>=s)&(dl<=t)
        ch[CH[c]]=dict(m=[e&valid[c],a[c],tgc[c]&~polc[c],sc[c],dc[c],un,de],tg=tgc[c])
    tot=[U([ch[x]['m'][i] for x in CH]) for i in range(7)]
    tot[2]=U(tgc)&~U(polc)
    ch['total']=dict(m=tot,tg=U(tgc))
    R['ch']={k:dict(v=[int(x.sum()) for x in d['m']],tg=int(d['tg'].sum())) for k,d in ch.items()}
    return R
res={k:calc(*k) for k in wins}
out={}
fd=lambda d:d.isoformat()
out['periods']=[dict(period=p,weeks=n,start=fd(wk(W-n)),end=fd(wk(W-1)+dt.timedelta(6)),prev_start=fd(wk(W-2*n)),prev_end=fd(wk(W-1-n)+dt.timedelta(6))) for p,n in PER.items()]
kpi=[];funnel=[];capn=[];inten=[];dist=[];chan=[]
SK=['total','valid','consent','tg','policy','sent','delivered']
CN={0:'none',1:'sms',2:'email',3:'sms+email',4:'push',5:'sms+push',6:'email+push',7:'sms+email+push'}
BK=['0','1–2','3–5','6–10','11–20','21+']
for p in PER:
    c,pr=res[(p,'cur')],res[(p,'prev')]
    def k(m,v,pv,unit,ch='all'): kpi.append(dict(period=p,channel=ch,metric=m,value=v,prev=pv,delta_pct=round(v-pv,1) if unit=='pct' else pct(v-pv,pv),unit=unit))
    k('total_base',c['fun']['all'][0],pr['fun']['all'][0],'clients')
    k('available_base',c['fun']['all'][2],pr['fun']['all'][2],'clients')
    k('tg_clients',c['fun']['all'][3],pr['fun']['all'][3],'clients')
    k('unutilized_base',c['unutil'],pr['unutil'],'clients')
    k('utilization_index',c['util'],pr['util'],'pct')
    for x in CH:
        k('total_base',c['fun'][x][0],pr['fun'][x][0],'clients',x)
        k('available_base',c['fun'][x][2],pr['fun'][x][2],'clients',x)
        k('tg_clients',c['fun'][x][3],pr['fun'][x][3],'clients',x)
        k('unutilized_base',c['unutil_c'][x],pr['unutil_c'][x],'clients',x)
        k('utilization_index',pct(c['msgs'][x],c['max'][x]),pct(pr['msgs'][x],pr['max'][x]),'pct',x)
    items=[(('all',chn),f,pr['fun'][chn]) for chn,f in c['fun'].items()]+[(kk,f,pr['func'][kk]) for kk,f in c['func'].items()]
    for (coh,chn),f,pf in items:
        for i in range(7):
            funnel.append(dict(period=p,cohort=coh,channel=chn,step=i+1,step_key=SK[i],clients=f[i],prev_clients=pf[i],
              conv_prev=pct(f[i],f[i-1]) if i else 100.0,conv_first=pct(f[i],f[0]),drop_abs=(f[i-1]-f[i]) if i else 0,
              prev_conv_first=pct(pf[i],pf[0]),delta_clients_pct=pct(f[i]-pf[i],pf[i]),delta_pp=round(pct(f[i],f[0])-pct(pf[i],pf[0]),1)))
    capn.append(dict(period=p,channel='all',combo='total',channels=-1,clients=c['fun']['all'][0],share_pct=100.0,prev_share_pct=100.0,consent_clients=c['fun']['all'][2],consent_share_pct=100.0))
    for nm in ['no_consent','no_contact']:
        kq='noconsent' if nm=='no_consent' else 'nocontact'
        capn.append(dict(period=p,channel='all',combo=nm,channels=0,clients=c[kq],share_pct=pct(c[kq],c['fun']['all'][0]),prev_share_pct=pct(pr[kq],pr['fun']['all'][0]),consent_clients=None,consent_share_pct=None))
    for kk in [7,6,3,5,4,2,1]:
        capn.append(dict(period=p,channel='all',combo=CN[kk],channels=bin(kk).count('1'),clients=c['combo'][kk],share_pct=pct(c['combo'][kk],c['fun']['all'][0]),prev_share_pct=pct(pr['combo'][kk],pr['fun']['all'][0]),consent_clients=c['combo'][kk],consent_share_pct=pct(c['combo'][kk],c['fun']['all'][2])))
    for ci,x in enumerate(CH):
        f,pf=c['fun'][x],pr['fun'][x]
        capn.append(dict(period=p,channel=x,combo='total',channels=-1,clients=f[0],share_pct=100.0,prev_share_pct=100.0,consent_clients=f[2],consent_share_pct=100.0))
        for nm,v,pv in [('no_consent',f[1]-f[2],pf[1]-pf[2]),('no_contact',f[0]-f[1],pf[0]-pf[1])]:
            capn.append(dict(period=p,channel=x,combo=nm,channels=0,clients=v,share_pct=pct(v,f[0]),prev_share_pct=pct(pv,pf[0]),consent_clients=None,consent_share_pct=None))
        for kk in [7,6,3,5,4,2,1]:
            if kk>>ci&1:
                capn.append(dict(period=p,channel=x,combo=CN[kk],channels=bin(kk).count('1'),clients=c['combo'][kk],share_pct=pct(c['combo'][kk],f[0]),prev_share_pct=pct(pr['combo'][kk],pf[0]),consent_clients=c['combo'][kk],consent_share_pct=pct(c['combo'][kk],f[2])))
    for chn in CH+['total']:
        inten.append(dict(period=p,channel=chn,available_clients=c['avail'][chn],sent_messages=c['msgs'][chn],max_messages=c['max'][chn],
          utilization_pct=pct(c['msgs'][chn],c['max'][chn]),avg_per_available=round(c['msgs'][chn]/c['avail'][chn],2),
          avg_per_tg_client=round(c['msgs'][chn]/c['tgn'][chn],2),avg_per_reached=round(c['msgs'][chn]/c['reach'][chn],2),max_per_client=c['maxpc'][chn],reached_clients=c['reach'][chn]))
        for i,b in enumerate(BK):
            dist.append(dict(period=p,channel=chn,bucket=b,bucket_order=i,clients=c['dist'][chn][i],share_pct=pct(c['dist'][chn][i],c['avail'][chn])))
        v=c['ch'][chn]['v']; tgn=c['ch'][chn]['tg']
        chan.append(dict(period=p,channel=chn,clients=v[0],available=v[1],available_pct=pct(v[1],v[0]),policy_filtered=v[2],policy_filtered_pct=pct(v[2],tgn),
          sent=v[3],sent_pct=pct(v[3],v[1]),delivered=v[4],delivered_pct=pct(v[4],v[3]),unsubscribed=v[5],unsubscribed_pct=pct(v[5],v[1]+v[5]),deleted=v[6],deleted_pct=pct(v[6],v[0]+v[6])))
bch=[]
for p in PER:
    c=res[(p,'cur')]
    for hn,L in c['bc'].items():
        for bi,trip in enumerate(L):
            tot=sum(x[0] for x in trip)
            for m,(ms,cl) in enumerate(trip):
                bch.append(dict(period=p,channel=hn,bucket_order=bi,msg_channel=CH[m],messages=ms,clients=cl,messages_share_pct=pct(ms,tot)))
out['bucket_channels']=bch
out.update(kpi=kpi,funnel=funnel,capacity=capn,intensity=inten,msg_distribution=dist,channels=chan,dynamics=dyn)
out['refs']=[
 dict(source='Профиль клиента (CDP)',content='Клиенты, контакты, корректность контактов',snapshot='2026-10-04'),
 dict(source='Реестр согласий',content='Согласия и отписки по каналам',snapshot='2026-10-04'),
 dict(source='Кампании и сегменты',content='Целевые группы запущенных кампаний',snapshot='2026-10-04'),
 dict(source='Журнал контактной политики',content='Проверки и отсев по правилам политики',snapshot='2026-10-04'),
 dict(source='Журнал отправок',content='Отправки и доставки по каналам',snapshot='2026-10-04')]
for k,v in out.items(): json.dump(v,open(f'data/{k}.json','w'),ensure_ascii=False)
for p in PER:
    print(p,[r for r in kpi if r['period']==p]); print(res[(p,'cur')]['fun']); 
print([r for r in inten if r['period']=='4w']); print([r for r in chan if r['period']=='4w']); print([r for r in capn if r['period']=='4w']); print([r for r in dist if r['period']=='4w' and r['channel']=='total'])
