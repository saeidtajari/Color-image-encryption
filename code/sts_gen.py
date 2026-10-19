# Writes binary files for the official NIST STS.
#   x, y, z : bytes of the chaotic layer alone (output filter, quantization, Eq. (10); no anchor), one coordinate each
#   a       : bytes of the anchored key material (Eq. (11)), coordinate x
#   c, c2, c3 : cipher images (40 per test image and 40 of the all-black image, random nonces)
import numpy as np, sys, os, hashlib
from PIL import Image
import cipher as C
which=sys.argv[1]; out=sys.argv[2]
if which in ('x','y','z','a'):
    NSEQ=500; NBYTES=125000
    rng=np.random.default_rng(20261011 if which!='a' else 20261012)
    idx='xyza'.index(which)%3
    with open(out,'wb') as f:
        for s in range(NSEQ):
            x0,y0,z0=rng.random(3); gam=rng.uniform(3.7,4.0); r=rng.uniform(0.9,1.0); mu=rng.uniform(1.8,2.0)
            X=np.empty(C.N_TRANSIENT);Y=np.empty_like(X);Z=np.empty_like(X)
            st=C.hybrid_steps(x0,y0,z0,gam,r,mu,C.N_TRANSIENT,X,Y,Z)
            X=np.empty(NBYTES);Y=np.empty_like(X);Z=np.empty_like(X)
            C.hybrid_steps(st[0],st[1],st[2],gam,r,mu,NBYTES,X,Y,Z)
            Q=C.quantize(X,Y,Z,True)
            if which=='a':
                key=(x0,y0,z0,gam,r,mu); ks=C.stream_key(key,bytes(rng.integers(0,256,16,dtype=np.uint8)),b'')
                Q=Q^C.anchor_words(ks,1,NBYTES)
            f.write((Q[idx]%256).astype(np.uint8).tobytes())
else:
    NAMES=['Lena','Baboon','Peppers','Fruits']
    IM={n:np.array(Image.open(f'images/{n}.png')) for n in NAMES}; IM['Black']=np.zeros((256,256,3),np.uint8)
    with open(out,'wb') as f:
        for n in IM:
            for s in range(40):
                c,_=C.encrypt(IM[n],C.KEY); f.write(c.reshape(-1).tobytes())
