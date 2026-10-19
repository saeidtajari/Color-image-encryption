"""Quiet timing (run on an idle machine, after all other experiments): Supplementary Table S13, scaling
(Section 3.14), standard ciphers and the proposed cipher entries of Table 12. Updates results/main.json and results/aes2.json."""
import numpy as np, json, time, hashlib, cv2
from PIL import Image
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
import cipher as C
NAMES=['Lena','Baboon','Peppers','Fruits']
R=json.load(open('results/main.json')); A2=json.load(open('results/aes2.json'))
IM={n:np.array(Image.open(f'images/{n}.png')) for n in NAMES}; K=C.KEY
NON={n:hashlib.sha256(n.encode()).digest()[:16] for n in NAMES}
C.encrypt(IM['Lena'],K)
te={};td={}
for n in NAMES:
    a=[];b=[]
    for _ in range(20):
        t=time.perf_counter(); c,md=C.encrypt(IM[n],K,nonce=NON[n]); a.append(time.perf_counter()-t)
        t=time.perf_counter(); C.decrypt(c,md,K); b.append(time.perf_counter()-t)
    te[n]=(float(np.mean(a)),float(np.std(a))); td[n]=(float(np.mean(b)),float(np.std(b)))
R['time_256']={'enc':te,'dec':td}
IM5=np.array(Image.open('images512/Lena.png')); sc={}
for s in (512,1024):
    im=IM5 if s==512 else cv2.resize(IM5,(s,s),interpolation=cv2.INTER_CUBIC); C.encrypt(im,K); a=[];b=[]
    for _ in range(5):
        t=time.perf_counter(); c,md=C.encrypt(im,K); a.append(time.perf_counter()-t)
        t=time.perf_counter(); C.decrypt(c,md,K); b.append(time.perf_counter()-t)
    sc[s]=(float(np.mean(a)),float(np.mean(b)))
R['time_scaling']=sc; R['timing_quiet']=True; json.dump(R,open('results/main.json','w'))
key=bytes.fromhex('000102030405060708090a0b0c0d0e0f'); iv=bytes(16)
def t_aes(mode):
    ts=[]
    for n in NAMES:
        d=IM[n].tobytes(); t0=time.perf_counter()
        for _ in range(200):
            m={'ECB':modes.ECB(),'CBC':modes.CBC(iv),'CTR':modes.CTR(iv)}[mode]; e=Cipher(algorithms.AES(key),m).encryptor(); e.update(d)+e.finalize()
        ts.append((time.perf_counter()-t0)/200)
    return float(np.mean(ts))
for lab,mode in (('AES-128 ECB','ECB'),('AES-128 CBC, fixed IV','CBC'),('AES-128 CTR, fixed nonce','CTR'),('AES-128 CBC, fresh IV','CBC'),('AES-128 CTR, fresh nonce','CTR')): A2[lab]['time']=t_aes(mode)
ts=[]
for n in NAMES:
    d=IM[n].tobytes(); t0=time.perf_counter()
    for _ in range(200): e=Cipher(algorithms.ChaCha20(key*2,iv),mode=None).encryptor(); e.update(d)
    ts.append((time.perf_counter()-t0)/200)
A2['ChaCha20, fresh nonce']['time']=float(np.mean(ts))
tp=[te[n][0] for n in NAMES]
for lab in ('Proposed, same nonce','Proposed, fresh nonce'): A2[lab]['time']=float(np.mean(tp))
json.dump(A2,open('results/aes2.json','w'),indent=1); print({k:v['time'] for k,v in A2.items() if k!='plain'})
