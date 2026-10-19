"""Distribution of the NPCR of a one-value change after 1 to 4 rounds (digest and nonce held fixed)."""
import numpy as np, json, sys
sys.path.insert(0,__import__('os').path.dirname(__import__('os').path.abspath(__file__))+'/..')
import cipher as C
from PIL import Image
im=np.array(Image.open('images/Lena.png')); M,N,_=im.shape; L=M*N; K=C.KEY; nonce=bytes(range(16))
H=C.image_digest(im)
S=C._setup(K,nonce,H,L,3,4)
cur=np.ascontiguousarray(np.transpose(im,(2,0,1)).reshape(3,L)).astype(np.uint8)
def enc(x,r):
    for rs in S[:r]: x=C._round_forward(x,rs,L,C.DEFAULT_CFG)
    return x.reshape(-1)
base={r:enc(cur,r) for r in (1,2,3,4)}
rng=np.random.default_rng(7); out={r:[] for r in (1,2,3,4)}
T=4000
for _ in range(T):
    c=rng.integers(3); q=rng.integers(L); x=cur.copy(); x[c,q]=(int(x[c,q])+1)%256
    for r in (1,2,3,4): out[r].append(100*float(np.mean(enc(x,r)!=base[r])))
res={}
for r in (1,2,3,4):
    a=np.array(out[r]); res[str(r)]={'mean':float(a.mean()),'sd':float(a.std()),'min':float(a.min()),'n_below_99':int((a<99).sum()),'n_below_99_5':int((a<99.5).sum()),'frac_below_99_5':float((a<99.5).mean()),'trials':T}
json.dump(res,open('results/sa_e6b.json','w'),indent=1); print(json.dumps(res,indent=1))
