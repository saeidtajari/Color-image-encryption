import numpy as np, json, time, hashlib
from PIL import Image
import cipher as C
from metrics import *
NAMES=['Lena','Baboon','Peppers','Fruits']
IM5={n:np.array(Image.open(f'images512/{n}.png')) for n in NAMES}
K=C.KEY
VAR={'Full cipher':dict(),
 'No DNA (addition mod 256)':dict(dna=False),
 'No substitution box':dict(sbox=False),
 'No DNA, no S-box':dict(dna=False,sbox=False),
 'No output filter':dict(filt=False),
 'No ChaCha20 anchor (chaotic key material only)':dict(anchor=False),
 'No anchor, no filter (key material of version 1)':dict(anchor=False,filt=False),
 'No anchor, no DNA, no S-box':dict(anchor=False,dna=False,sbox=False),
 'No anchor: uncoupled Logistic maps':dict(anchor=False,mapmode=1),
 'No anchor: hybrid branches, no Arnold coupling':dict(anchor=False,mapmode=2),
 'No anchor: Arnold coupling, Logistic only':dict(anchor=False,mapmode=3)}
rng=np.random.default_rng(2024)
crops=[]
for n in NAMES:
    for _ in range(25):
        i,j=rng.integers(0,257,2); crops.append(IM5[n][i:i+256,j:j+256].copy())
Hfix=C.image_digest(crops[0]); nonce=b'A'*16
R={}
def run(name,cfg,rounds=C.ROUNDS):
    cf={**C.DEFAULT_CFG,**cfg}; out={}
    t0=time.time()
    # (a) structure test: fixed streams, difference of cipher images for a 1-value change at (128,128,0)
    D=[]
    for im in crops:
        im2=im.copy(); im2[128,128,0]=(int(im2[128,128,0])+1)%256
        a,_=C.encrypt(im,K,rounds=rounds,nonce=nonce,digest=Hfix,cfg=cf); b,_=C.encrypt(im2,K,rounds=rounds,nonce=nonce,digest=Hfix,cfg=cf)
        D.append(a^b)
    D=np.array(D); r2=np.random.default_rng(1); m=[]
    for _ in range(1500):
        i,j=r2.choice(len(D),2,replace=False); m.append(np.mean(D[i]==D[j]))
    out['diff_match']=float(np.mean(m))                 # ideal ~ 1/256 = 0.0039
    out['diff_zero_frac']=float(np.mean(D==0))          # unchanged bytes (ideal 1/256)
    # (b) NPCR/UACI with fixed digest, Lena and Baboon (30 trials each)
    res=[]
    for n in ('Lena','Baboon'):
        im=crops[NAMES.index(n)*25]; c0,_=C.encrypt(im,K,rounds=rounds,nonce=nonce,digest=Hfix,cfg=cf); r3=np.random.default_rng(9)
        for t in range(30):
            im2=im.copy(); i,j,c=r3.integers(0,256),r3.integers(0,256),r3.integers(0,3); im2[i,j,c]=(int(im2[i,j,c])+1)%256
            c2,_=C.encrypt(im2,K,rounds=rounds,nonce=nonce,digest=Hfix,cfg=cf); res.append(np.concatenate(npcr_uaci(c0,c2)))
    res=np.array(res); out['npcr']=float(res[:,:3].mean()); out['uaci']=float(res[:,3:].mean())
    cr=crit(65536,0.05); out['npcr_pass']=float(np.mean(res[:,:3]>=cr['npcr_min'])); out['uaci_pass']=float(np.mean((res[:,3:]>=cr['uaci_lo'])&(res[:,3:]<=cr['uaci_hi'])))
    # (c) statistics of the cipher image of every crop (first 20 crops)
    ent=[];chi=[];cor=[]
    for im in crops[:40:2]:
        c,_=C.encrypt(im,K,rounds=rounds,nonce=nonce,cfg=cf)
        ent.append(entropy(c)); chi+= [chi2_p(c[:,:,k])[1]>0.05 for k in range(3)]; cor.append(np.mean(np.abs([corr_full(c[:,:,k]) for k in range(3)])))
    out['entropy']=float(np.mean(ent)); out['chi_pass']=float(np.mean(chi)); out['mean_abs_corr']=float(np.mean(cor))
    out['sec']=time.time()-t0; return out
for name,cfg in VAR.items():
    R[name]=run(name,cfg); print(name,{k:round(v,4) for k,v in R[name].items()},flush=True)
R['One round (full cipher)']=run('1r',{},rounds=1); print('1 round',R['One round (full cipher)'],flush=True)
# key-stream quality of the map variants
KS={}
for name,mm in (('proposed',0),('logistic',1),('hybrid_nocoupling',2),('arnold_logistic',3)):
    X=np.empty(1000000);Y=np.empty_like(X);Z=np.empty_like(X)
    st=C.hybrid_steps(0.31,0.27,0.58,3.99,0.999,1.99,1000,np.empty(1000),np.empty(1000),np.empty(1000),mm)
    C.hybrid_steps(*st,3.99,0.999,1.99,1000000,X,Y,Z,mm)
    from scipy.stats import chisquare
    row={}
    Q=C.quantize(X,Y,Z,True)%256                     # chaotic layer (Eq. (10)), no anchor
    for t,V,b in zip('xyz',(X,Y,Z),Q):
        h=np.bincount(b,minlength=256)
        row[t]={'chi2_p':float(chisquare(h).pvalue),'lag1':float(np.corrcoef(b[:-1],b[1:])[0,1]),'mean_state':float(V.mean())}
    row['xy_corr']=float(np.corrcoef(Q[0],Q[1])[0,1])
    KS[name]=row
R['keystream']=KS; print(KS)
json.dump(R,open('results/ablation.json','w'),indent=1)
