import subprocess, numpy as np, json, os, sys
sys.path.insert(0,os.path.dirname(os.path.abspath(__file__)))
import cipher as C
N=6500000
X=np.empty(N);Y=np.empty(N);Z=np.empty(N)
C.hybrid_steps(0.3,0.2,0.6,3.99,0.999,1.99,N,X,Y,Z,0)
num=np.stack([X,Y,Z],1)
py=np.array(C.hybrid_steps_py(0.3,0.2,0.6,3.99,0.999,1.99,N))
R={'numba_vs_python_bit_equal':bool(np.array_equal(num,py))}
cfgs={'gcc -O2 -ffp-contract=off':('gcc','-O2 -ffp-contract=off',''),
      'gcc -O3 -march=native (FMA contraction on)':('gcc','-O3 -march=native -ffp-contract=fast',''),
      'clang -O2':('clang','-O2',''),
      'gcc -O3 -ffast-math':('gcc','-O3 -ffast-math','')}
def first_div(a,b):
    d=np.nonzero(np.any(a!=b,axis=1))[0]; return int(d[0]) if len(d) else None
for name,(cc,fl,_) in cfgs.items():
    row={}
    for tag,defs in (('polynomial sine (proposed)',''),('libm sin (conventional)','-DOLDSIN')):
        exe='port/m.out'; subprocess.run(f"{cc} {fl} {defs} port/map.c -o {exe} -lm",shell=True,check=True)
        subprocess.run(f"{exe} {N} port/o.bin",shell=True,check=True)
        a=np.fromfile('port/o.bin').reshape(-1,3)
        ref=py if not defs else None
        if defs:   # reference for conventional sine: numba with math.sin
            from numba import njit
            import math
            @njit
            def old(x,y,z,gam,r,mu,n,O):
                for i in range(n):
                    xx=r*math.sin(math.pi*x) if x<0.5 else gam*x*(1-x)
                    yy=mu*y if y<0.5 else r*math.sin(math.pi*y)
                    zz=gam*z*(1-z) if z<0.5 else mu*(1-z)
                    x=(3*xx+1*yy+4*zz)%1.0; y=(8*xx+3*yy+11*zz)%1.0; z=(6*xx+2*yy+9*zz)%1.0
                    O[i,0]=x;O[i,1]=y;O[i,2]=z
            O=np.empty((N,3)); old(0.3,0.2,0.6,3.99,0.999,1.99,N,O); ref=O
        row[tag]={'bit_equal':bool(np.array_equal(a,ref)),'first_divergence':first_div(a,ref)}
    R[name]=row; print(name,row,flush=True)
json.dump(R,open('results/port.json','w'),indent=1)
