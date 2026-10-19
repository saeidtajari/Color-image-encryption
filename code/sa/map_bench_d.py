import numpy as np, json, math, sys
sys.path.insert(0,__import__('os').path.dirname(__import__('os').path.abspath(__file__))+'/..')
import cipher as C
from numba import njit
@njit(cache=True)
def lam(mode,g,r,mu,x,y,z,n):
    for _ in range(1000):
        xx = r*C.psin(x) if (x<0.5 and mode==2) else g*x*(1-x)
        if mode==2:
            yy = mu*y if y<0.5 else r*C.psin(1.0-y); zz = g*z*(1-z) if z<0.5 else mu*(1-z)
            x,y,z=xx%1.0,yy%1.0,zz%1.0
        else:
            yy=g*y*(1-y); zz=g*z*(1-z); x,y,z=(3*xx+yy+4*zz)%1.0,(8*xx+3*yy+11*zz)%1.0,(6*xx+2*yy+9*zz)%1.0
    x2,y2,z2=x+1e-9,y,z; s=0.0
    for _ in range(n):
        for k in range(2):
            if k==0: a,b,c=x,y,z
            else: a,b,c=x2,y2,z2
            if mode==2:
                p1 = r*C.psin(a) if a<0.5 else g*a*(1-a); p2 = mu*b if b<0.5 else r*C.psin(1.0-b); p3 = g*c*(1-c) if c<0.5 else mu*(1-c)
                a,b,c=p1%1.0,p2%1.0,p3%1.0
            else:
                p1=g*a*(1-a); p2=g*b*(1-b); p3=g*c*(1-c); a,b,c=(3*p1+p2+4*p3)%1.0,(8*p1+3*p2+11*p3)%1.0,(6*p1+2*p2+9*p3)%1.0
            if k==0: x,y,z=a,b,c
            else: x2,y2,z2=a,b,c
        dx=x2-x; dy=y2-y; dz=z2-z; dx-=round(dx); dy-=round(dy); dz-=round(dz)
        d=math.sqrt(dx*dx+dy*dy+dz*dz)
        if d==0.0: d=1e-300
        s+=math.log(d/1e-9); f=1e-9/d; x2=x+dx*f; y2=y+dy*f; z2=z+dz*f
    return s/n
res={}
gs=np.linspace(3.7,3.9999,3000)
L=np.array([lam(3,g,0.999,1.99,0.31,0.27,0.57,20000) for g in gs]); res['Arnold coupling of Logistic maps, gamma in [3.7,4)']=dict(frac_pos=float(np.mean(L>0.05)),min=float(L.min()))
rng=np.random.default_rng(3); L2=[]
for _ in range(3000):
    g=rng.uniform(3.7,4.0); r=rng.uniform(0.9,1.0); mu=rng.uniform(1.8,2.0); L2.append(lam(2,g,r,mu,0.31,0.27,0.57,20000))
L2=np.array(L2); res['Hybrid branches without coupling, random (gamma,r,mu) in the key ranges (Lyapunov exponent of the combined 3D map)']=dict(frac_pos=float(np.mean(L2>0.05)),min=float(L2.min()),median=float(np.median(L2)))
print(res); json.dump(res,open('results/map_bench_d.json','w'),indent=1)
