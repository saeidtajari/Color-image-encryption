"""E9: further tests of 1..5 rounds: bit-level avalanche and uniformity of the difference bytes (digest and nonce fixed)."""
import sys, json, numpy as np
sys.path.insert(0,__import__('os').path.dirname(__import__('os').path.abspath(__file__))+'/..')
from PIL import Image
from scipy.stats import chi2
import cipher as C
K=C.KEY; nonce=b'A'*16
IM={n:np.array(Image.open(f'images512/{n}.png').convert('RGB')) for n in ('Lena','Baboon','Peppers','Fruits')}
rng=np.random.default_rng(99); crops=[]
for n in IM:
    for _ in range(25):
        i,j=rng.integers(0,257,2); crops.append(IM[n][i:i+256,j:j+256].copy())
H=C.image_digest(crops[0]); R={}
pc=np.array([bin(i).count('1') for i in range(256)])
for rounds in (1,2,3,4,5):
    hist=np.zeros(256,np.int64); bits=np.zeros(8,np.int64); nb=0; hd=[]
    for im in crops:
        i,j,ch,b=int(rng.integers(0,256)),int(rng.integers(0,256)),int(rng.integers(0,3)),int(rng.integers(0,8))
        im2=im.copy(); im2[i,j,ch]^=(1<<b)
        a,_=C.encrypt(im,K,rounds=rounds,nonce=nonce,digest=H); c,_=C.encrypt(im2,K,rounds=rounds,nonce=nonce,digest=H)
        D=(a^c).reshape(-1); hist+=np.bincount(D,minlength=256)
        for k in range(8): bits[k]+=int(((D>>k)&1).sum())
        nb+=D.size; hd.append(float(pc[D].sum())/(8*D.size))
    n=nb; f=bits/n; z=(f-0.5)/np.sqrt(0.25/n)
    e=n/256; x=float(((hist-e)**2/e).sum()); p=float(chi2.sf(x,255))
    # per-trial Hamming fraction vs random 0.5
    hd=np.array(hd)
    R[str(rounds)]=dict(trials=len(crops),n_bytes=int(n),bit_frac_min=float(f.min()),bit_frac_max=float(f.max()),z_max=float(np.abs(z).max()),
        chi2=x,chi2_p=p,zero_frac=float(hist[0]/n),ham_mean=float(hd.mean()),ham_sd=float(hd.std()),ham_sd_random=float(np.sqrt(0.25/(8*65536*3))))
    print(rounds,R[str(rounds)],flush=True)
json.dump(R,open('results/sa_e9.json','w'),indent=1)
