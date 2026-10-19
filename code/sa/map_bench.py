"""Item 2: benchmark of the proposed 3D map against reference maps (independent of the cipher)."""
import sys, json, math, time, numpy as np
sys.path.insert(0,__import__('os').path.dirname(__import__('os').path.abspath(__file__))+'/..')
import cipher as C
from numba import njit
G,R,MU=3.99,0.999,1.99
MODES={'Proposed hybrid 3D map':0,'Logistic (1D, gamma=3.99)':1,'Sine (1D, r=0.999)':4,'Tent (1D, mu=1.99)':5,'Henon (2D, a=1.4, b=0.3)':6,
       'Hybrid branches, no coupling':2,'Arnold coupling of Logistic maps':3}
@njit(cache=True)
def step(mode,x,y,z):
    if mode==0 or mode==2:
        xx = R*C.psin(x) if x<0.5 else G*x*(1.0-x)
        yy = MU*y if y<0.5 else R*C.psin(1.0-y)
        zz = G*z*(1.0-z) if z<0.5 else MU*(1.0-z)
    elif mode==1 or mode==3:
        xx=G*x*(1-x); yy=G*y*(1-y); zz=G*z*(1-z)
    elif mode==4:
        xx=R*math.sin(math.pi*x); yy=R*math.sin(math.pi*y); zz=R*math.sin(math.pi*z)
    elif mode==5:
        xx=MU*x if x<0.5 else MU*(1-x); yy=MU*y if y<0.5 else MU*(1-y); zz=MU*z if z<0.5 else MU*(1-z)
    else:
        return 1.0-1.4*x*x+y, 0.3*x, 0.0
    if mode==0 or mode==3:
        return (3.0*xx+1.0*yy+4.0*zz)%1.0,(8.0*xx+3.0*yy+11.0*zz)%1.0,(6.0*xx+2.0*yy+9.0*zz)%1.0
    return xx,yy,zz
@njit(cache=True)
def series(mode,x,y,z,n,ntr):
    for _ in range(ntr): x,y,z=step(mode,x,y,z)
    out=np.empty(n)
    for i in range(n):
        x,y,z=step(mode,x,y,z); out[i]=x
    return out
@njit(cache=True)
def lyap(mode,x,y,z,n,ntr):
    for _ in range(ntr): x,y,z=step(mode,x,y,z)
    d0=1e-9; wrap = mode!=6
    x2,y2,z2=x+d0,y+d0*0.7,z+d0*0.3
    if mode==6: z2=z
    s=0.0
    for _ in range(n):
        x,y,z=step(mode,x,y,z); x2,y2,z2=step(mode,x2,y2,z2)
        dx=x2-x; dy=y2-y; dz=z2-z
        if wrap:
            dx-=round(dx); dy-=round(dy); dz-=round(dz)
        d=math.sqrt(dx*dx+dy*dy+dz*dz)
        if d==0.0: d=1e-300
        s+=math.log(d/d0)
        f=d0/d; x2=x+dx*f; y2=y+dy*f; z2=z+dz*f
        if mode==6: z2=z
    return s/n
@njit(cache=True)
def perm_entropy(v,m):
    cnt=np.zeros(math.factorial(m) if False else 5040)
    n=v.shape[0]-m+1
    fac=np.ones(m+1,dtype=np.int64)
    for i in range(1,m+1): fac[i]=fac[i-1]*i
    for i in range(n):
        code=0
        for a in range(m):
            c=0
            for b in range(a):
                if v[i+b]>v[i+a]: c+=1
            code+=c*fac[a]
        cnt[code]+=1
    h=0.0
    for k in range(fac[m]):
        if cnt[k]>0:
            p=cnt[k]/n; h-=p*math.log(p)
    return h/math.log(fac[m])
@njit(cache=True)
def zero_one(x,c,nmax):
    N=x.shape[0]; p=np.zeros(N); q=np.zeros(N); sp=0.0; sq=0.0; mean=0.0
    for j in range(N): mean+=x[j]
    mean/=N
    for j in range(N):
        sp+=x[j]*math.cos((j+1)*c); sq+=x[j]*math.sin((j+1)*c); p[j]=sp; q[j]=sq
    M=np.zeros(nmax); ns=np.arange(1,nmax+1)*1.0
    for n in range(1,nmax+1):
        s=0.0
        for j in range(N-n):
            s+=(p[j+n]-p[j])**2+(q[j+n]-q[j])**2
        M[n-1]=s/(N-n)
    Dn=M-mean*mean*(1-np.cos(ns*c))/(1-math.cos(c))
    # correlation of Dn with n
    a=Dn-Dn.mean(); b=ns-ns.mean()
    return (a*b).sum()/math.sqrt((a*a).sum()*(b*b).sum())
@njit(cache=True)
def qstep(mode,x,y,z,sc,dim):
    x,y,z=step(mode,x,y,z)
    x=math.floor(x*sc+0.5)/sc
    if dim>1: y=math.floor(y*sc+0.5)/sc
    else: y=0.0
    if dim>2: z=math.floor(z*sc+0.5)/sc
    else: z=0.0
    return x,y,z
@njit(cache=True)
def brent_q(mode,x,y,z,sc,dim,maxn):
    power=1; lam=1; tx,ty,tz=x,y,z
    hx,hy,hz=qstep(mode,x,y,z,sc,dim); used=1
    while tx!=hx or ty!=hy or tz!=hz:
        if power==lam:
            tx,ty,tz=hx,hy,hz; power*=2; lam=0
        hx,hy,hz=qstep(mode,hx,hy,hz,sc,dim); lam+=1; used+=1
        if used>maxn: return -1,used
    return lam,used
@njit(cache=True)
def splitmix(s):
    s=(s+np.uint64(0x9E3779B97F4A7C15))
    z=s
    z=(z^(z>>np.uint64(30)))*np.uint64(0xBF58476D1CE4E5B9)
    z=(z^(z>>np.uint64(27)))*np.uint64(0x94D049BB133111EB)
    return z^(z>>np.uint64(31))
@njit(cache=True)
def brent_rand(bits,seed,maxn):
    mask=(np.uint64(1)<<np.uint64(bits))-np.uint64(1)
    t=np.uint64(seed)&mask; h=splitmix(t)&mask; power=1; lam=1; used=1
    while t!=h:
        if power==lam:
            t=h; power*=2; lam=0
        h=splitmix(h)&mask; lam+=1; used+=1
        if used>maxn: return -1,used
    return lam,used
if __name__=='__main__':
    rng=np.random.default_rng(11); R_={}
    for name,m in MODES.items():
        t=time.time(); dim=3 if m in (0,2,3) else (2 if m==6 else 1)
        lam=[]; 
        for _ in range(20):
            s=rng.random(3)*0.8+0.1
            if m==6: s=np.array([0.1,0.1,0.0])+rng.random(3)*0.05
            lam.append(lyap(m,s[0],s[1],s[2],200000,1000))
        s=rng.random(3)*0.8+0.1
        if m==6: s=np.array([0.1,0.1,0.0])
        x=series(m,s[0],s[1],s[2],1000000,1000)
        pe6=perm_entropy(x,6)
        # 0-1 test on 10^4 points, 30 values of c
        xs=series(m,s[0],s[1],s[2],10000,1000); cs=rng.uniform(math.pi/5,4*math.pi/5,30)
        K=[zero_one(xs,c,1000) for c in cs]
        R_[name]=dict(lam_mean=float(np.mean(lam)),lam_min=float(np.min(lam)),lam_max=float(np.max(lam)),pe6=float(pe6),K_median=float(np.median(K)),K_min=float(np.min(K)),dim=dim)
        print(name,{k:round(v,4) for k,v in R_[name].items()},round(time.time()-t),flush=True)
    json.dump(R_,open('results/map_bench_a.json','w'),indent=1)
    json.dump(R_,open('results/map_bench_a.json','w'),indent=1)
