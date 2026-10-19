"""Finite-precision degradation (random-map calibration) and fraction of chaotic parameters."""
import sys, json, math, time, numpy as np
sys.path.insert(0,__import__('os').path.dirname(__import__('os').path.abspath(__file__))+'/..'); sys.path.insert(0,__import__('os').path.dirname(__import__('os').path.abspath(__file__)))
import map_bench as mb
rng=np.random.default_rng(5); out={}
CAP=40_000_000
cases=[('Proposed hybrid 3D map',0,3),('Hybrid branches, no coupling',2,3),('Arnold coupling of Logistic maps',3,3),('Logistic (1D, gamma=3.99)',1,1),('Sine (1D, r=0.999)',4,1),('Tent (1D, mu=1.99)',5,1)]
for bits in (10,12,14):
    sc=float(2**bits)
    for name,m,dim in cases:
        S=2.0**(bits*dim); ref=math.sqrt(math.pi*S/2); used=[]; cyc=[]; t=time.time()
        ntr=300 if dim==3 else 2000
        for _ in range(ntr):
            s=np.floor(rng.random(3)*sc)/sc
            s=np.maximum(s,2.0/sc)
            lam,u=mb.brent_q(m,s[0],s[1],s[2],sc,dim,CAP)
            used.append(u); cyc.append(lam)
        used=np.array(used,float); cap=int(np.sum(used>CAP))
        out[f'{name}|{bits}']=dict(bits=bits,dim=dim,S=S,expected_rho=ref,median_steps=float(np.median(used)),p25=float(np.quantile(used,0.25)),p75=float(np.quantile(used,0.75)),ratio=float(np.median(used)/ref),capped=cap,median_cycle=float(np.median([c for c in cyc if c>0])) if any(c>0 for c in cyc) else -1,trials=ntr)
        print(name,bits,out[f'{name}|{bits}'],round(time.time()-t),flush=True)
    # calibration with an ideal random mapping on S states
    for dim in (1,3):
        b=bits*dim; S=2.0**b; ref=math.sqrt(math.pi*S/2); used=[]
        for i in range(300 if b<=36 else 150):
            lam,u=mb.brent_rand(b,int(rng.integers(1,2**62)),CAP); used.append(u)
        used=np.array(used,float)
        out[f'random mapping|{b}']=dict(bits=b,S=S,expected_rho=ref,median_steps=float(np.median(used)),p25=float(np.quantile(used,0.25)),p75=float(np.quantile(used,0.75)),ratio=float(np.median(used)/ref),trials=300 if b<=36 else 150)
        print('random',b,out[f'random mapping|{b}'],flush=True)
json.dump(out,open('results/map_bench_b.json','w'),indent=1)
