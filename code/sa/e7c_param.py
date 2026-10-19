"""E7c: sensitivity of the bytes of the chaotic layer (Eq. (10)) to the parameters: steps until the bytes differ after a 1-ulp change."""
import sys, json, numpy as np
sys.path.insert(0,__import__('os').path.dirname(__import__('os').path.abspath(__file__))+'/..'); sys.path.insert(0,__import__('os').path.dirname(__import__('os').path.abspath(__file__)))
import cipher as C
rng=np.random.default_rng(7); res={}
def bytes_(x,y,z,g,r,m,n):
    X=np.empty(n);Y=np.empty(n);Z=np.empty(n)
    C.hybrid_steps(x,y,z,g,r,m,n,X,Y,Z,0)
    return C.quantize(X,Y,Z,True)%256
for name,idx in (('gamma',0),('r',1),('mu',2),('state',3)):
    firsts=[]
    for t in range(300):
        g=rng.uniform(3.7,4.0); r=rng.uniform(0.9,1.0); m=rng.uniform(1.8,2.0)
        x,y,z=rng.random(3)
        X=np.empty(1000);Y=np.empty(1000);Z=np.empty(1000)
        st=C.hybrid_steps(x,y,z,g,r,m,1000,X,Y,Z,0)
        b0=bytes_(*st,g,r,m,200)
        p=[g,r,m]; s=list(st)
        if idx<3: p[idx]=np.nextafter(p[idx],2*p[idx])
        else: s[0]=np.nextafter(s[0],2.0)
        b1=bytes_(*s,*p,200)
        d=np.nonzero((b0!=b1).any(axis=0))[0]
        firsts.append(int(d[0]) if len(d) else 200)
    res[name]=dict(median=float(np.median(firsts)),q90=float(np.quantile(firsts,.9)),max=int(max(firsts)))
    print(name,res[name])
json.dump(res,open('results/sa_e7c.json','w'))
