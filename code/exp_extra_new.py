import numpy as np, json, math, os
from PIL import Image
from scipy.stats import chi2 as chi2d
import cipher as C
from metrics import *
R={}
img=np.array(Image.open('images/Lena.png')); H=C.image_digest(img); NON=b'N'*16
rng=np.random.default_rng(99)
def rkey(): return (rng.uniform(0.05,0.95),rng.uniform(0.05,0.95),rng.uniform(0.05,0.95),rng.uniform(3.7,4.0),rng.uniform(0.9,1.0),rng.uniform(1.8,2.0))
ps=[];ent=[]
for t in range(200):
    c,_=C.encrypt(img,rkey())
    for ch in range(3):
        x,p=chi2_p(c[:,:,ch]); ps.append(p); ent.append(entropy(c[:,:,ch]))
ps=np.array(ps); R['chi2_keys']={'n':len(ps),'pass_005':float(np.mean(ps>0.05)),'pass_001':float(np.mean(ps>0.01)),'mean_entropy':float(np.mean(ent)),'std_entropy':float(np.std(ent)),'p_hist':np.histogram(ps,bins=10,range=(0,1))[0].tolist()}
print(R['chi2_keys'],flush=True)
# absorption of small key perturbations (key-stream level)
rng2=np.random.default_rng(5); n=3000
def traj(i,p):
    # bytes of the chaotic layer (Eq. (10)), coordinate x; the anchor is left out because every change of the key
    # bytes changes K_s, so that the resolution is a property of the chaotic layer
    X=np.empty(n);Y=np.empty(n);Z=np.empty(n); C.hybrid_steps(i[0],i[1],i[2],p[0],p[1],p[2],n,X,Y,Z); return C.quantize(X,Y,Z,True)[0]
R['absorb']={}
for delta in (1e-15,1e-14,1e-13,1e-12):
    fails=0;T=3000; per=[0]*6
    for t in range(T):
        d=rng2.bytes(32)
        k=[rng2.uniform(0.05,0.95),rng2.uniform(0.05,0.95),rng2.uniform(0.05,0.95),rng2.uniform(3.7,4.0),rng2.uniform(0.9,1.0),rng2.uniform(1.8,2.0)]
        idx=int(rng2.integers(0,6)); k2=k.copy(); k2[idx]+=delta
        a=traj(C.derive_initial(k,d),k[3:]); b=traj(C.derive_initial(k2,d),k2[3:])
        if np.mean((a[1000:]%256)==(b[1000:]%256))>0.05: fails+=1; per[idx]+=1
    R['absorb'][f'{delta:g}']={'trials':T,'unresolved':fails,'per_param':per}; print(delta,fails,per,flush=True)
rng3=np.random.default_rng(31); res=[]
for t in range(100):
    k=[rng3.uniform(0.05,0.95),rng3.uniform(0.05,0.95),rng3.uniform(0.05,0.95),rng3.uniform(3.7,4.0),rng3.uniform(0.9,1.0),rng3.uniform(1.8,2.0)]
    c1,_=C.encrypt(img,tuple(k),nonce=NON,digest=H); k2=k.copy(); k2[int(rng3.integers(0,6))]+=1e-13; c2,_=C.encrypt(img,tuple(k2),nonce=NON,digest=H)
    res.append(100*np.mean(c1!=c2))
res=np.array(res); R['keysens_random13']={'n':100,'mean':float(res.mean()),'min':float(res.min()),'max':float(res.max())}
rs,_=C.gen_round((0.31,0.27,0.57),C.KEY[3:],65536,3,1,C.stream_key(C.KEY,NON,H),C.DEFAULT_CFG)
K3=rs.K3.astype(float); R['k3_crosscorr']=[float(np.corrcoef(K3[0],K3[1])[0,1]),float(np.corrcoef(K3[0],K3[2])[0,1]),float(np.corrcoef(K3[1],K3[2])[0,1])]
R['k3_autocorr']=[float(np.corrcoef(K3[0][:-l],K3[0][l:])[0,1]) for l in (1,2,3)]
g=[('x0',1.0),('y0',1.0),('z0',1.0),('gamma',0.3),('r',0.1),('mu',0.2)]
logs=[math.log10(w/1e-13) for _,w in g]; R['keyspace']={'per_param_log10':logs,'total_log10':sum(logs),'total_log2':sum(logs)/math.log10(2)}
R['perm_valid']=bool(all(len(np.unique(rs.TP[c]))==65536 for c in range(3)))
json.dump(R,open('results/extra_new.json','w'),indent=1); print('done')
