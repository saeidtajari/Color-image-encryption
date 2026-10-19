import numpy as np
from numba import njit
import cipher as C
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size':9,'figure.dpi':200})
@njit(cache=True)
def lyap(x,y,z,gam,r,mu,n,ntr):
    A=np.array([[3.,1.,4.],[8.,3.,11.],[6.,2.,9.]])
    Q=np.eye(3); s=np.zeros(3); pi=np.pi
    for i in range(n+ntr):
        if x<0.5:
            xx=r*np.sin(pi*x); dx=r*pi*np.cos(pi*x)
        else:
            xx=gam*x*(1-x); dx=gam*(1-2*x)
        if y<0.5:
            yy=mu*y; dy=mu
        else:
            yy=r*np.sin(pi*y); dy=r*pi*np.cos(pi*y)
        if z<0.5:
            zz=gam*z*(1-z); dz=gam*(1-2*z)
        else:
            zz=mu*(1-z); dz=-mu
        J=A@np.diag(np.array([dx,dy,dz]))
        x=(A[0,0]*xx+A[0,1]*yy+A[0,2]*zz)%1.0
        y=(A[1,0]*xx+A[1,1]*yy+A[1,2]*zz)%1.0
        z=(A[2,0]*xx+A[2,1]*yy+A[2,2]*zz)%1.0
        M=J@Q
        # Gram-Schmidt
        for j in range(3):
            v=M[:,j].copy()
            for k in range(j):
                v-=np.dot(v,Q[:,k])*Q[:,k]
            nv=np.sqrt(np.dot(v,v))
            Q[:,j]=v/nv
            if i>=ntr: s[j]+=np.log(nv)
    return s/n


k=C.KEY
def sweep(idx,vals):
    out=[]
    for v in vals:
        p=[k[3],k[4],k[5]]; p[idx]=v
        out.append(lyap(k[0],k[1],k[2],p[0],p[1],p[2],20000,1000)[0])
    return np.array(out)
g=np.linspace(3.5,3.9999,300); rr=np.linspace(0.5,1.0,300); mm=np.linspace(1.5,1.9999,300)
fig,ax=plt.subplots(1,3,figsize=(9,2.7))
for a,x,y,l,c,t in zip(ax,(g,rr,mm),(sweep(0,g),sweep(1,rr),sweep(2,mm)),('$\\gamma$','$r$','$\\mu$'),('#d62728','#2ca02c','#1f77b4'),'abc'):
    a.plot(x,y,lw=1.2,color=c); a.fill_between(x,0,y,color=c,alpha=.18); a.axhline(0,c='k',lw=.6,ls='--')
    a.set_xlabel(l); a.set_ylabel('$\\lambda_1$'); a.set_title(f'({t})',loc='left'); a.spines[['top','right']].set_visible(False)
plt.tight_layout(); plt.savefig('figs/fig_lyap.png'); plt.close(); print('ok')
