import numpy as np, json, math
from numba import njit
@njit(cache=True)
def lam_1d(kind,p,x,n):
    for _ in range(1000):
        if kind==0: x=p*x*(1-x)
        elif kind==1: x=p*math.sin(math.pi*x)
        else: x=p*x if x<0.5 else p*(1-x)
    s=0.0
    for _ in range(n):
        if kind==0: d=abs(p*(1-2*x)); x=p*x*(1-x)
        elif kind==1: d=abs(p*math.pi*math.cos(math.pi*x)); x=p*math.sin(math.pi*x)
        else: d=p; x=p*x if x<0.5 else p*(1-x)
        if d<1e-300: d=1e-300
        s+=math.log(d)
    return s/n
res={}
for name,kind,lo,hi in (('Logistic, gamma in [3.7,4)',0,3.7,4.0),('Sine, r in [0.9,1]',1,0.9,1.0),('Tent, mu in [1.8,2)',2,1.8,2.0)):
    ps=np.linspace(lo,hi-1e-9,3000); L=np.array([lam_1d(kind,p,0.3141592,20000) for p in ps])
    res[name]=dict(frac_pos=float(np.mean(L>0.05)),frac_nonpos=float(np.mean(L<=0.05)),min=float(L.min()))
    print(name,res[name])
json.dump(res,open('results/map_bench_c.json','w'),indent=1)
