import numpy as np
from scipy.stats import norm, chi2 as chi2d
def entropy(a):
    h=np.bincount(np.asarray(a).reshape(-1),minlength=256).astype(float); p=h/h.sum(); p=p[p>0]; return float(-(p*np.log2(p)).sum())
def corr_full(ch):
    ch=ch.astype(np.float64); out=[]
    for dx,dy in ((0,1),(1,0),(1,1)):
        a=ch[:ch.shape[0]-dx,:ch.shape[1]-dy].ravel(); b=ch[dx:,dy:].ravel()
        out.append(float(np.corrcoef(a,b)[0,1]))
    return out   # horizontal, vertical, diagonal
def corr_pairs(ch,rng,N=2000):
    out=[]; H,W=ch.shape
    for dx,dy in ((0,1),(1,0),(1,1)):
        i=rng.integers(0,H-1,N); j=rng.integers(0,W-1,N)
        x=ch[i,j].astype(float); y=ch[i+dx,j+dy].astype(float)
        out.append((float(np.corrcoef(x,y)[0,1]),x,y))
    return out
def npcr_uaci(a,b):
    a=a.astype(np.int64);b=b.astype(np.int64)
    return 100*np.mean(a!=b,axis=(0,1)), 100*np.mean(np.abs(a-b)/255.0,axis=(0,1))
def chi2_p(ch):
    h=np.bincount(ch.reshape(-1),minlength=256); e=ch.size/256
    x=float(((h-e)**2/e).sum()); return x,float(chi2d.sf(x,255))
def local_entropy(ch,rng,nb=30,bs=1936):
    """Local Shannon entropy (Wu et al. 2013): mean entropy of nb random blocks of bs pixels
    (blocks taken as bs consecutive pixels in row-major order at random positions)."""
    f=ch.reshape(-1); n=len(f); vals=[]
    for _ in range(nb):
        s=int(rng.integers(0,n-bs)); vals.append(entropy(f[s:s+bs]))
    return float(np.mean(vals))
def local_entropy_null(reps=4000,seed=3,nb=30,bs=1936):
    rng=np.random.default_rng(seed); out=[]
    for _ in range(reps):
        out.append(np.mean([entropy(rng.integers(0,256,bs)) for _ in range(nb)]))
    out=np.array(out); return float(out.mean()),float(np.percentile(out,2.5)),float(np.percentile(out,97.5))
v=np.arange(256); _d=np.abs(v[:,None]-v[None,:])/255.0; muU=_d.mean(); sdU=_d.std()
def crit(L,alpha):
    pN=255/256; sN=np.sqrt(pN*(1-pN)/L)
    return {'npcr_min':100*(pN-norm.ppf(1-alpha)*sN),'uaci_lo':100*(muU-norm.ppf(1-alpha/2)*sdU/np.sqrt(L)),'uaci_hi':100*(muU+norm.ppf(1-alpha/2)*sdU/np.sqrt(L))}
