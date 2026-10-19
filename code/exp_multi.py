"""Extended image set (10 images, 256x256) with independent random keys and nonces."""
import numpy as np, json, time, sys
from PIL import Image
import cipher as C
from metrics import *
NAMES=['Astronaut','Cat','Coffee','Rocket','Colorwheel','Histology','Retina','Hubble','Brick','Checkerboard']
NK=10; NT=10
cr=crit(65536,0.05)
rng=np.random.default_rng(20261012)
res={}
t0=time.time()
for n in NAMES:
    im=np.array(Image.open(f'images_ext/{n}.png').convert('RGB'))
    r={'plain_entropy':float(entropy(im)),'plain_corr':float(np.mean(np.abs([corr_full(im[:,:,k]) for k in range(3)]))),
       'ent':[],'corr':[],'chi':[],'loc':[],'npcr':[],'uaci':[],'np_pass':[],'ua_pass':[],'ok':True}
    for k in range(NK):
        key=(*rng.random(3),rng.uniform(3.7,4.0),rng.uniform(0.9,1.0),rng.uniform(1.8,2.0))
        c,meta=C.encrypt(im,key)
        p,tk,hk=C.decrypt(c,meta,key); r['ok']&=bool(np.array_equal(p,im) and tk and hk)
        r['ent'].append(float(entropy(c))); r['corr'].append(float(np.mean(np.abs([corr_full(c[:,:,j]) for j in range(3)]))))
        r['chi']+= [float(chi2_p(c[:,:,j])[1]>0.05) for j in range(3)]
        rl=np.random.default_rng(5); r['loc'].append(float(np.mean([local_entropy(c[:,:,j],rl) for j in range(3)])))
        nonce=bytes(rng.integers(0,256,16,dtype=np.uint8)); Hf=C.image_digest(im); c0,_=C.encrypt(im,key,nonce=nonce,digest=Hf)
        for t in range(NT):                    # digest of the original image held fixed for both encryptions
            im2=im.copy(); i,j,ch=rng.integers(0,256),rng.integers(0,256),rng.integers(0,3); im2[i,j,ch]=(int(im2[i,j,ch])+1)%256
            c2,_=C.encrypt(im2,key,nonce=nonce,digest=Hf)
            a,b=npcr_uaci(c0,c2)
            r['npcr']+=list(map(float,a)); r['uaci']+=list(map(float,b))
            r['np_pass']+=[float(v>=cr['npcr_min']) for v in a]; r['ua_pass']+=[float(cr['uaci_lo']<=v<=cr['uaci_hi']) for v in b]
    res[n]=r; print(n,round(time.time()-t0),np.mean(r['ent']),np.mean(r['npcr']),np.mean(r['uaci']),r['ok'],flush=True)
    json.dump(res,open('results/multi.json','w'))
