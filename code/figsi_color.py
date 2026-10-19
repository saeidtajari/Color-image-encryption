import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as _plt
_plt.rcParams.update({'font.family':'serif','font.serif':['Liberation Serif'],'mathtext.fontset':'stix'})
_o=_plt.savefig
def _sv(fn,*a,**k):
    heavy=any(k_ in fn for k_ in ('results','special','bif','phase','corr'))
    _o(fn,dpi=200); _o(fn.rsplit('.',1)[0]+'.eps',dpi=300 if heavy else 600)
_plt.savefig=_sv
import numpy as np, sys
from PIL import Image
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from scipy.stats import gaussian_kde
import cipher as C
plt.rcParams.update({'font.size':8,'figure.dpi':200,'axes.grid':False})
K=C.KEY
NAMES=['Lena','Baboon','Peppers','Fruits']
import hashlib
ENCC=np.load('results/enc_cache.npz')
NON={n:hashlib.sha256(n.encode()).digest()[:16] for n in NAMES}
IM={n:np.array(Image.open(f'images/{n}.png')) for n in NAMES}
def corr_pairs(ch,rng,N=2000):
    out=[]
    for dx,dy in ((0,1),(1,0),(1,1)):
        i=rng.integers(0,255,N); j=rng.integers(0,255,N)
        x=ch[i,j].astype(float); y=ch[i+dx,j+dy].astype(float)
        out.append((float(np.corrcoef(x,y)[0,1]),x,y))
    return out
def dens(x,y):
    xy=np.vstack([x,y]); 
    try: d=gaussian_kde(xy+np.random.default_rng(0).normal(0,.5,xy.shape))(xy)
    except Exception: d=np.ones(len(x))
    o=np.argsort(d); return o,d
def corrfig(n,fn):
    ENC=ENCC[n]
    rng=np.random.default_rng(100+NAMES.index(n))
    st=rng.bit_generator.state
    rp=corr_pairs(IM[n][:,:,0],rng); rng.bit_generator.state=st
    rc=corr_pairs(ENC[:,:,0],rng)
    fig,ax=plt.subplots(2,3,figsize=(6.8,4.6))
    for j,dn in enumerate(['Horizontal','Vertical','Diagonal']):
        for i,(rr,lab,cm) in enumerate(((rp,'Plain','inferno'),(rc,'Encrypted','viridis'))):
            x,y=rr[j][1],rr[j][2]; o,d=dens(x,y)
            sc=ax[i,j].scatter(x[o],y[o],c=d[o],s=3,cmap=cm,alpha=.9,linewidths=0)
            ax[i,j].set_xlim(0,255);ax[i,j].set_ylim(0,255);ax[i,j].set_aspect('equal')
            ax[i,j].set_title(f'{lab}, {dn}\n$\\rho$ = {rr[j][0]:.4f}',fontsize=8)
            ax[i,j].set_xlabel('Pixel value $(x,y)$'); ax[i,j].set_ylabel('Adjacent pixel')
            cb=plt.colorbar(sc,ax=ax[i,j],fraction=.046,pad=.03); cb.set_label('density',fontsize=6); cb.ax.tick_params(labelsize=5); cb.set_ticks([])
    plt.tight_layout(); plt.savefig(fn); plt.close()
def histfig(n,fn):
    ENC=ENCC[n]
    fig,ax=plt.subplots(3,2,figsize=(6.2,4.8))
    cols=['#d62728','#2ca02c','#1f77b4']
    for c,cn in enumerate('RGB'):
        ax[c,0].hist(IM[n][:,:,c].reshape(-1),bins=256,range=(0,256),color=cols[c],alpha=.9)
        ax[c,1].hist(ENC[:,:,c].reshape(-1),bins=256,range=(0,256),color=cols[c],alpha=.9)
        ax[c,0].set_ylabel(f'{cn} count'); ax[c,1].set_ylim(0,600)
        ax[c,1].axhline(256,color='k',lw=.7,ls='--')
        for a in ax[c]: a.spines[['top','right']].set_visible(False)
    ax[0,0].set_title('Plain'); ax[0,1].set_title('Encrypted (dashed: ideal 256)')
    ax[2,0].set_xlabel('Pixel value'); ax[2,1].set_xlabel('Pixel value')
    plt.tight_layout(); plt.savefig(fn); plt.close()
for n in ('Lena','Baboon'):
    corrfig(n,f'figsi/fig_corr_{n.lower()}.png'); histfig(n,f'figsi/fig_hist_{n.lower()}.png')
# special
fig,ax=plt.subplots(1,4,figsize=(7.0,2.1))
Eb=ENCC['black']; Ew=ENCC['white']
for a,(im,t) in zip(ax,[(np.zeros((256,256,3),np.uint8),'(a1) all-black'),(np.full((256,256,3),255,np.uint8),'(a2) all-white'),(Eb,'(b1) cipher of (a1)'),(Ew,'(b2) cipher of (a2)')]):
    a.imshow(im); a.set_xticks([]);a.set_yticks([]); a.set_title(t,fontsize=8)
plt.tight_layout(); plt.savefig('figsi/fig_special.png'); plt.close()
# phase
k=K
st=C.hybrid_steps(k[0],k[1],k[2],k[3],k[4],k[5],1000,np.empty(1000),np.empty(1000),np.empty(1000))
n=6000; X=np.empty(n);Y=np.empty(n);Z=np.empty(n)
C.hybrid_steps(st[0],st[1],st[2],k[3],k[4],k[5],n,X,Y,Z)
t=np.arange(n)
fig=plt.figure(figsize=(7,6.2))
for i,(u,v,nu,nv) in enumerate([(X,Y,'x','y'),(X,Z,'x','z'),(Y,Z,'y','z')]):
    a=fig.add_subplot(2,2,i+1); a.scatter(u,v,s=1.2,c=t,cmap='turbo',linewidths=0,alpha=.8); a.set_xlabel(nu);a.set_ylabel(nv);a.set_title(f'({"abc"[i]})',loc='left')
a=fig.add_subplot(2,2,4,projection='3d'); a.scatter(X,Y,Z,s=1,c=t,cmap='turbo',alpha=.7,linewidths=0); a.set_xlabel('x');a.set_ylabel('y');a.set_zlabel('z');a.set_title('(d)',loc='left')
cax=fig.add_axes([0.25,0.012,0.5,0.014]); sm=plt.cm.ScalarMappable(cmap='turbo',norm=plt.Normalize(0,n)); cb=plt.colorbar(sm,cax=cax,orientation='horizontal'); cb.set_label('iteration index',fontsize=7); cb.ax.tick_params(labelsize=6)
plt.tight_layout(rect=(0,.04,1,1)); plt.savefig('figsi/fig_phase.png'); plt.close()
# bifurcation
def bif(idx,vals,var,n=60,ntr=500):
    P=[];V=[]
    for v in vals:
        p=[k[3],k[4],k[5]]; p[idx]=v
        X=np.empty(ntr+n);Y=np.empty_like(X);Z=np.empty_like(X)
        C.hybrid_steps(k[0],k[1],k[2],p[0],p[1],p[2],ntr+n,X,Y,Z)
        W=(X,Y,Z)[var][ntr:]; P.extend([v]*n); V.extend(W)
    return np.array(P),np.array(V)
fig,ax=plt.subplots(1,3,figsize=(9.5,2.9))
for a,(idx,vals,var,xl,yl,t_,cm) in zip(ax,[(0,np.linspace(3.5,4,400),0,'$\\gamma$','x','(a)','plasma'),(1,np.linspace(0.5,1,400),1,'$r$','y','(b)','viridis'),(2,np.linspace(1.5,2,400),2,'$\\mu$','z','(c)','cool')]):
    P,V=bif(idx,vals,var); a.scatter(P,V,s=0.5,c=V,cmap=cm,alpha=.55,rasterized=True,linewidths=0); a.set_xlabel(xl);a.set_ylabel(yl);a.set_title(t_,loc='left')
plt.tight_layout(); plt.savefig('figsi/fig_bif.png'); plt.close()
print('ok')

# results figure
rounds={n:[C.encrypt(IM[n],K,rounds=t,nonce=NON[n])[0] for t in (1,2,3,4)] for n in NAMES}
fig,ax=plt.subplots(6,4,figsize=(7.2,10.6)); rl=['Plain','Encrypted (t=1)','Encrypted (t=2)','Encrypted (t=3)','Encrypted (t=4)','Decrypted']
for j,n in enumerate(NAMES):
    c4,md=C.encrypt(IM[n],K,nonce=NON[n]); assert np.array_equal(c4,rounds[n][3]) and np.array_equal(c4,ENCC[n])
    ims=[IM[n]]+rounds[n]+[C.decrypt(c4,md,K)[0]]
    for i in range(6):
        ax[i,j].imshow(ims[i]); ax[i,j].set_xticks([]);ax[i,j].set_yticks([])
        if i==0: ax[i,j].set_title(n,fontsize=9,fontweight='bold')
        if j==0: ax[i,j].set_ylabel(rl[i])
plt.tight_layout(); plt.savefig('figsi/fig_results.png'); plt.close()
