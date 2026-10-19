"""E1: pair statistics and linear biases of the bytes of the chaotic layer (version 2: low byte of floor(1e14 g(s)),
Eq. (10), output filter on, anchor off).  Usage: python3 sa/e1_keystream.py seed nkeys N."""
import sys, json, time, numpy as np
sys.path.insert(0,__import__('os').path.dirname(__import__('os').path.abspath(__file__))+'/..')
import cipher as C
from numba import njit
from scipy.linalg import hadamard
from scipy.stats import chi2
CH=1<<20; LAG=8
@njit(cache=True)
def acc(B, H, n, lag0):
    # B: (3, n+LAG) bytes; H: (78, 65536)
    k=0
    for c in range(3):
        for l in range(1,LAG+1):
            for i in range(n):
                H[k, B[c,i]*256+B[c,i+l]]+=1
            k+=1
    for a in range(3):
        for b in range(3):
            if a==b: continue
            for l in range(0,LAG+1):
                for i in range(n):
                    H[k, B[a,i]*256+B[b,i+l]]+=1
                k+=1
def names():
    out=[]
    for c in 'xyz':
        for l in range(1,LAG+1): out.append(f'{c}_i,{c}_i+{l}')
    for a in 'xyz':
        for b in 'xyz':
            if a==b: continue
            for l in range(0,LAG+1): out.append(f'{a}_i,{b}_i+{l}')
    return out
NM=names()
def run(key,N):
    X=np.empty(CH+LAG);Y=np.empty(CH+LAG);Z=np.empty(CH+LAG)
    st=C.hybrid_steps(key[0],key[1],key[2],key[3],key[4],key[5],1000,np.empty(1000),np.empty(1000),np.empty(1000),0)
    H=np.zeros((len(NM),65536),dtype=np.int64)
    tail=None; done=0
    while done<N:
        n=CH
        Xc=np.empty(n);Yc=np.empty(n);Zc=np.empty(n)
        st=C.hybrid_steps(st[0],st[1],st[2],key[3],key[4],key[5],n,Xc,Yc,Zc,0)
        B=C.quantize(Xc,Yc,Zc,True)%256
        if tail is not None: B=np.concatenate([tail,B],axis=1)
        m=B.shape[1]-LAG
        acc(np.ascontiguousarray(B),H,m,0)
        tail=B[:,-LAG:].copy(); done+=m
    return H,done
Hd=hadamard(256).astype(np.float64)
def analyse(H,N):
    res=[]
    zs=[]
    for k,nm in enumerate(NM):
        h=H[k].astype(np.float64); tot=h.sum()
        e=tot/65536; chi=((h-e)**2/e).sum(); p=float(chi2.sf(chi,65535))
        W=Hd@(h.reshape(256,256)/tot)@Hd
        W[0,0]=0; z=np.abs(W)*np.sqrt(tot)
        res.append(dict(pair=nm,chi_p=p,max_z=float(z.max()),n=int(tot)))
    return res
if __name__=='__main__':
    seed=int(sys.argv[1]); nkeys=int(sys.argv[2]); N=int(float(sys.argv[3]))
    rng=np.random.default_rng(seed); out=[]
    lo=np.array([a for a,b in C.KEY_RANGES]);hi=np.array([b for a,b in C.KEY_RANGES])
    for i in range(nkeys):
        key=tuple(lo+(hi-lo)*rng.random(6)) if not(seed==0 and i==0) else C.KEY
        t=time.time(); H,done=run(key,N); r=analyse(H,done)
        mz=max(x['max_z'] for x in r); ps=np.array([x['chi_p'] for x in r])
        print(i,'min p',ps.min(),'max z',mz,'n',done,'t',round(time.time()-t),flush=True)
        out.append(dict(key=list(key),res=r))
        json.dump(out,open(f'results/sa_e1_{seed}.json','w'))
