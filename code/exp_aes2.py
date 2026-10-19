import numpy as np, json, time, os, hashlib
from PIL import Image
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
import cipher as C
from metrics import *
NAMES=['Lena','Baboon','Peppers','Fruits']
IM={n:np.array(Image.open(f'images/{n}.png')) for n in NAMES}
key=bytes.fromhex('000102030405060708090a0b0c0d0e0f'); key32=key+key
IV=bytes.fromhex('f0e1d2c3b4a5968778695a4b3c2d1e0f')
def enc_bytes(mode,data,iv):
    if mode=='ECB': c=Cipher(algorithms.AES(key),modes.ECB())
    elif mode=='CBC': c=Cipher(algorithms.AES(key),modes.CBC(iv))
    elif mode=='CTR': c=Cipher(algorithms.AES(key),modes.CTR(iv))
    else: c=Cipher(algorithms.ChaCha20(key32,iv),mode=None)
    e=c.encryptor(); return e.update(data)+e.finalize()
def E(mode,im,iv): return np.frombuffer(enc_bytes(mode,im.tobytes(),iv),np.uint8).reshape(im.shape)
SCHEMES=[('AES-128 ECB','ECB','fixed'),('AES-128 CBC, fixed IV','CBC','fixed'),('AES-128 CTR, fixed nonce','CTR','fixed'),
         ('AES-128 CBC, fresh IV','CBC','fresh'),('AES-128 CTR, fresh nonce','CTR','fresh'),('ChaCha20, fresh nonce','CHACHA','fresh')]
R={}
def evaluate(label,encf):
    r={'entropy':[],'corr':[],'chi_pass':[],'npcr':[],'uaci':[],'time':[],'local':[]}
    for n in NAMES:
        im=IM[n]; c=encf(im)
        r['entropy'].append(entropy(c)); r['corr'].append(float(np.mean(np.abs([corr_full(c[:,:,k]) for k in range(3)]))))
        r['chi_pass'].append(int(sum(chi2_p(c[:,:,k])[1]>0.05 for k in range(3))))
        rl=np.random.default_rng(5); r['local'].append(float(np.mean([local_entropy(c[:,:,k],rl) for k in range(3)])))
        rng=np.random.default_rng(555+NAMES.index(n)); a=[];u=[]
        for t in range(100):
            im2=im.copy(); i,j,k=rng.integers(0,256),rng.integers(0,256),rng.integers(0,3); im2[i,j,k]=(int(im2[i,j,k])+1)%256
            x,y=npcr_uaci(encf(im),encf(im2)) if label[1]=='fresh' else npcr_uaci(c,encf(im2)); a.append(x.mean());u.append(y.mean())
        r['npcr'].append(float(np.mean(a))); r['uaci'].append(float(np.mean(u)))
        t0=time.perf_counter()
        for _ in range(100): encf(im)
        r['time'].append((time.perf_counter()-t0)/100)
    out={k:float(np.mean(v)) for k,v in r.items()}; out['chi_total']=int(sum(r['chi_pass'])); return out
for lab,mode,iv in SCHEMES:
    if mode=='CHACHA': ivlen=16
    else: ivlen=16
    f=(lambda im,mode=mode,iv=iv: E(mode,im,IV if iv=='fixed' else os.urandom(16)))
    R[lab]=evaluate((lab,iv),f); print(lab,R[lab],flush=True)
# proposed cipher, same measurement
NON=lambda n: hashlib.sha256(n.encode()).digest()[:16]
for lab,fresh in (('Proposed, same nonce',False),('Proposed, fresh nonce',True)):
    r={'entropy':[],'corr':[],'chi_pass':[],'npcr':[],'uaci':[],'time':[],'local':[]}
    for n in NAMES:
        im=IM[n]; nn=NON(n)
        c,_=C.encrypt(im,C.KEY,nonce=nn)
        r['entropy'].append(entropy(c)); r['corr'].append(float(np.mean(np.abs([corr_full(c[:,:,k]) for k in range(3)]))))
        r['chi_pass'].append(int(sum(chi2_p(c[:,:,k])[1]>0.05 for k in range(3))))
        rl=np.random.default_rng(5); r['local'].append(float(np.mean([local_entropy(c[:,:,k],rl) for k in range(3)])))
        rng=np.random.default_rng(555+NAMES.index(n)); a=[];u=[]
        for t in range(30):
            im2=im.copy(); i,j,k=rng.integers(0,256),rng.integers(0,256),rng.integers(0,3); im2[i,j,k]=(int(im2[i,j,k])+1)%256
            c1=C.encrypt(im,C.KEY)[0] if fresh else c
            c2=C.encrypt(im2,C.KEY)[0] if fresh else C.encrypt(im2,C.KEY,nonce=nn)[0]
            x,y=npcr_uaci(c1,c2); a.append(x.mean());u.append(y.mean())
        r['npcr'].append(float(np.mean(a))); r['uaci'].append(float(np.mean(u)))
        t0=time.perf_counter()
        for _ in range(10): C.encrypt(im,C.KEY,nonce=nn)
        r['time'].append((time.perf_counter()-t0)/10)
    R[lab]={k:float(np.mean(v)) for k,v in r.items()}; R[lab]['chi_total']=int(sum(r['chi_pass'])); print(lab,R[lab],flush=True)
R['plain']={'entropy':float(np.mean([entropy(IM[n]) for n in NAMES])),'corr':float(np.mean([np.mean(np.abs([corr_full(IM[n][:,:,k]) for k in range(3)])) for n in NAMES]))}
json.dump(R,open('results/aes2.json','w'),indent=1)
