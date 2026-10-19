"""E5: end-to-end black-box test on very short messages (L = 1 and 2 pixels): plaintext/cipher byte pairs and cipher byte pairs,
fixed key, random plaintext and random nonce; chi-square of 65536-cell joint histograms and maximal Walsh (linear) bias."""
import sys, json, time, numpy as np, os
sys.path.insert(0,__import__('os').path.dirname(__import__('os').path.abspath(__file__))+'/..')
import cipher as C
from scipy.linalg import hadamard
from scipy.stats import chi2
Hd=hadamard(256).astype(float)
def analyse(h):
    tot=h.sum(); e=tot/65536; chi=((h-e)**2/e).sum()
    W=Hd@(h.reshape(256,256)/tot)@Hd; W[0,0]=0
    return float(chi2.sf(chi,65535)), float(np.abs(W).max()*np.sqrt(tot))
def run(M,Nn,N,seed):
    rng=np.random.default_rng(seed); K=C.KEY; L=M*Nn; nb=3*L
    hists={}
    def H(a): return hists.setdefault(a,np.zeros(65536,dtype=np.int64))
    t=time.time()
    for s in range(N):
        img=rng.integers(0,256,(M,Nn,3),dtype=np.uint8)
        c,_=C.encrypt(img,K,nonce=rng.bytes(16))
        P=np.transpose(img,(2,0,1)).reshape(-1); Cc=np.transpose(c,(2,0,1)).reshape(-1)
        for i in range(nb):
            for j in range(nb):
                H(('P%d'%i,'C%d'%j))[int(P[i])*256+int(Cc[j])]+=1
            for j in range(i+1,nb):
                H(('C%d'%i,'C%d'%j))[int(Cc[i])*256+int(Cc[j])]+=1
        if s%100000==0: print(M,Nn,s,round(time.time()-t),flush=True)
    res={f'{a}-{b}':analyse(h) for (a,b),h in hists.items()}
    return res
if __name__=='__main__':
    M,Nn,N=int(sys.argv[1]),int(sys.argv[2]),int(float(sys.argv[3]))
    r=run(M,Nn,N,5)
    ps=np.array([v[0] for v in r.values()]); zs=np.array([v[1] for v in r.values()])
    out=dict(M=M,N=Nn,samples=N,n_hist=len(r),min_p=float(ps.min()),max_z=float(zs.max()),res=r)
    json.dump(out,open(f'results/sa_e5_{M}x{Nn}.json','w')); print({k:v for k,v in out.items() if k!='res'})
