"""E2 predictability horizon, E3 weak keys (lambda1 scan) and cycle detection (Brent)."""
import sys, json, time, numpy as np
sys.path.insert(0,__import__('os').path.dirname(__import__('os').path.abspath(__file__))+'/..')
import cipher as C
from numba import njit
import math
A=np.array([[3.,1.,4.],[8.,3.,11.],[6.,2.,9.]])
@njit(cache=True)
def step(x,y,z,g,r,mu):
    xx = r*C.psin(x) if x<0.5 else g*x*(1.0-x)
    yy = mu*y if y<0.5 else r*C.psin(1.0-y)
    zz = g*z*(1.0-z) if z<0.5 else mu*(1.0-z)
    return (3.0*xx+1.0*yy+4.0*zz)%1.0,(8.0*xx+3.0*yy+11.0*zz)%1.0,(6.0*xx+2.0*yy+9.0*zz)%1.0
@njit(cache=True)
def lam1(x,y,z,g,r,mu,n,ntr):
    for _ in range(ntr): x,y,z=step(x,y,z,g,r,mu)
    v=np.array([1.0,0.7,0.3]); v/=np.sqrt((v*v).sum()); s=0.0
    for _ in range(n):
        if x<0.5: d1=r*math.pi*math.cos(math.pi*x)
        else: d1=g*(1-2*x)
        if y<0.5: d2=mu
        else: d2=-r*math.pi*math.cos(math.pi*(1-y))
        if z<0.5: d3=g*(1-2*z)
        else: d3=-mu
        a=d1*v[0];b=d2*v[1];c=d3*v[2]
        w0=3*a+1*b+4*c; w1=8*a+3*b+11*c; w2=6*a+2*b+9*c
        nv=math.sqrt(w0*w0+w1*w1+w2*w2)
        if nv==0.0: return -1e9
        s+=math.log(nv); v[0]=w0/nv;v[1]=w1/nv;v[2]=w2/nv
        x,y,z=step(x,y,z,g,r,mu)
    return s/n
@njit(cache=True)
def horizon(x,y,z,g,r,mu,dx,dy,dz,thr,maxn):
    x2=(x+dx)%1.0;y2=(y+dy)%1.0;z2=(z+dz)%1.0
    for i in range(maxn):
        x,y,z=step(x,y,z,g,r,mu); x2,y2,z2=step(x2,y2,z2,g,r,mu)
        if abs(x-x2)>thr or abs(y-y2)>thr or abs(z-z2)>thr: return i+1
    return maxn
@njit(cache=True)
def brent(x,y,z,g,r,mu,maxn):
    # returns (cycle length found or 0, steps used)
    power=1; lam=1
    tx,ty,tz=x,y,z
    hx,hy,hz=step(x,y,z,g,r,mu)
    used=1
    while (tx!=hx or ty!=hy or tz!=hz):
        if power==lam:
            tx,ty,tz=hx,hy,hz; power*=2; lam=0
        hx,hy,hz=step(hx,hy,hz,g,r,mu); lam+=1; used+=1
        if used>maxn: return 0,used
    return lam,used
lo=np.array([a for a,b in C.KEY_RANGES]);hi=np.array([b for a,b in C.KEY_RANGES])
if __name__=='__main__':
    which=sys.argv[1]; rng=np.random.default_rng(int(sys.argv[2]))
    R={}
    if which=='lam':
        nk=int(sys.argv[3]); L=[]; K=[]
        for i in range(nk):
            k=lo+(hi-lo)*rng.random(6)
            L.append(lam1(k[0],k[1],k[2],k[3],k[4],k[5],20000,1000)); K.append(k.tolist())
        L=np.array(L); K=np.array(K)
        R=dict(n=nk,min=float(L.min()),q001=float(np.quantile(L,0.001)),q01=float(np.quantile(L,0.01)),median=float(np.median(L)),
               frac_below_1=float(np.mean(L<1)),frac_below_2=float(np.mean(L<2)),frac_nonpos=float(np.mean(L<=0)),
               argmin=K[L.argmin()].tolist(),lams=L.tolist()[:2000])
        # boundary corners of the parameter box
        cor=[]
        for g in (3.7,3.99999):
            for r in (0.9,0.99999):
                for mu in (1.8,1.99999):
                    cor.append((g,r,mu,float(lam1(0.31,0.27,0.57,g,r,mu,20000,1000))))
        R['corners']=cor
        # parameter grid scan (fixed initial state) for periodic windows
        g_=np.linspace(3.7,4.0,31); r_=np.linspace(0.9,1.0,11); m_=np.linspace(1.8,2.0,11)
        G=np.array([[[lam1(0.31,0.27,0.57,g,r,mu,5000,500) for mu in m_] for r in r_] for g in g_])
        R['grid_min']=float(G.min()); R['grid_frac_below_1']=float(np.mean(G<1)); R['grid_n']=int(G.size)
    elif which=='hor':
        nk=int(sys.argv[3]); out={}
        for d in (1e-10,1e-12,1e-14,1e-15):
            H=[]
            for i in range(nk):
                k=lo+(hi-lo)*rng.random(6)
                s=np.array(C.hybrid_steps(k[0],k[1],k[2],k[3],k[4],k[5],1000,np.empty(1000),np.empty(1000),np.empty(1000),0))
                dd=rng.normal(size=3); dd=dd/np.linalg.norm(dd)*d
                H.append(horizon(s[0],s[1],s[2],k[3],k[4],k[5],dd[0],dd[1],dd[2],0.1,2000))
            out[str(d)]=dict(median=float(np.median(H)),min=int(np.min(H)),max=int(np.max(H)),mean=float(np.mean(H)))
        R=out
    elif which=='cyc':
        nk=int(sys.argv[3]); maxn=int(float(sys.argv[4])); out=[]
        for i in range(nk):
            k=lo+(hi-lo)*rng.random(6); t=time.time()
            s=C.hybrid_steps(k[0],k[1],k[2],k[3],k[4],k[5],1000,np.empty(1000),np.empty(1000),np.empty(1000),0)
            lam,used=brent(s[0],s[1],s[2],k[3],k[4],k[5],maxn)
            out.append(dict(key=k.tolist(),cycle=int(lam),steps=int(used),sec=time.time()-t)); print(i,lam,used,round(time.time()-t),flush=True)
        R=dict(runs=out,maxn=maxn)
    json.dump(R,open(f'results/sa_e23_{which}_{sys.argv[2]}.json','w'))
    print({k:v for k,v in R.items() if k not in('lams','runs')})
