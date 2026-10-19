"""E4: security margin in the number of rounds (digest and nonce held fixed = strongest-attacker model for the diffusion)."""
import sys, json, time, numpy as np
sys.path.insert(0,__import__('os').path.dirname(__import__('os').path.abspath(__file__))+'/..')
from PIL import Image
import cipher as C
from metrics import npcr_uaci, crit
K=C.KEY; nonce=b'A'*16
IM={n:np.array(Image.open(f'images512/{n}.png')) for n in ('Lena','Baboon','Peppers','Fruits')}
rng=np.random.default_rng(77)
crops=[]
for n in IM:
    for _ in range(25):
        i,j=rng.integers(0,257,2); crops.append(IM[n][i:i+256,j:j+256].copy())
H=C.image_digest(crops[0]); L=256*256
cr=crit(65536,0.05)
R={}
for rounds in (1,2,3,4,5):
    t=time.time(); out={}
    # structure test
    D=[]
    for im in crops:
        im2=im.copy(); im2[128,128,0]=(int(im2[128,128,0])+1)%256
        a,_=C.encrypt(im,K,rounds=rounds,nonce=nonce,digest=H); b,_=C.encrypt(im2,K,rounds=rounds,nonce=nonce,digest=H)
        D.append(a^b)
    D=np.array(D); r2=np.random.default_rng(1); m=[]
    for _ in range(1500):
        i,j=r2.choice(len(D),2,replace=False); m.append(np.mean(D[i]==D[j]))
    out['diff_match']=float(np.mean(m))
    # position-resolved NPCR / UACI: change at the given sequence position (R,G,B channels flattened) 
    im=crops[0]; c0,_=C.encrypt(im,K,rounds=rounds,nonce=nonce,digest=H)
    pos={}
    for name,frac in (('first',0.0),('quarter',0.25),('middle',0.5),('three_quarter',0.75),('last',1.0)):
        vals=[]
        for ch in range(3):
            for tr in range(7):
                p=min(L-1,int(frac*L)+(0 if frac in(0.0,1.0) else tr)) if frac<1 else L-1-tr
                if frac==0.0: p=tr
                i,j=divmod(p,256); im2=im.copy(); im2[i,j,ch]=(int(im2[i,j,ch])+1+tr)%256
                c2,_=C.encrypt(im2,K,rounds=rounds,nonce=nonce,digest=H); vals.append(np.concatenate(npcr_uaci(c0,c2)))
        v=np.array(vals); pos[name]=dict(npcr=float(v[:,:3].mean()),uaci=float(v[:,3:].mean()),npcr_min=float(v[:,:3].min()))
    out['positions']=pos; out['sec']=time.time()-t; R[str(rounds)]=out
    print(rounds,out['diff_match'],{k:round(v['npcr'],3) for k,v in pos.items()},flush=True)
    json.dump(R,open('results/sa_e4.json','w'),indent=1)
