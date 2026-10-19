import numpy as np, json, time, platform, hashlib, sys
from PIL import Image
import cv2
import cipher as C
from metrics import *
NAMES=['Lena','Baboon','Peppers','Fruits']
IM={n:np.array(Image.open(f'images/{n}.png')) for n in NAMES}
IM5={n:np.array(Image.open(f'images512/{n}.png')) for n in NAMES}
K=C.KEY
NON={n:hashlib.sha256(n.encode()).digest()[:16] for n in NAMES}
R={}; L1=65536
R['crit']={f'{a}':crit(L1,a) for a in (0.05,0.01,0.001)}; R['crit512']={f'{a}':crit(512*512,a) for a in (0.05,)}
R['crit_mean_npcr']=100*255/256; R['crit_mean_uaci']=100*muU
rng0=np.random.default_rng(1)
R['ideal_entropy_65536']=float(np.mean([entropy(rng0.integers(0,256,65536)) for _ in range(300)]))
R['ideal_entropy_196608']=float(np.mean([entropy(rng0.integers(0,256,196608)) for _ in range(300)]))
R['ideal_entropy_262144']=float(np.mean([entropy(rng0.integers(0,256,262144)) for _ in range(200)]))
R['local_null']=local_entropy_null(); print('local null',R['local_null'],flush=True)
ENC={};MD={}
C.encrypt(IM['Lena'],K)
t_enc={};t_dec={}
for n in NAMES:
    te=[];td=[]
    for _ in range(20):
        t=time.time(); c,md=C.encrypt(IM[n],K,nonce=NON[n]); te.append(time.time()-t)
        t=time.time(); p,tk,hk=C.decrypt(c,md,K); td.append(time.time()-t)
    assert np.array_equal(p,IM[n]) and tk and hk
    ENC[n]=c;MD[n]=md; t_enc[n]=(float(np.mean(te)),float(np.std(te)));t_dec[n]=(float(np.mean(td)),float(np.std(td)))
R['time_256']={'enc':t_enc,'dec':t_dec}
R['lossless']={n:int(np.count_nonzero(C.decrypt(ENC[n],MD[n],K)[0]!=IM[n])) for n in NAMES}
import numba
R['hw']={'cpu':'Intel Xeon @ 2.80 GHz (single core used)','python':platform.python_version(),'numba':numba.__version__,'numpy':np.__version__}
sc={}
big={s:(IM5['Lena'] if s==512 else cv2.resize(IM5['Lena'],(s,s),interpolation=cv2.INTER_CUBIC)) for s in (512,1024)}
for s,im in big.items():
    C.encrypt(im,K); tt=[];td=[]
    for _ in range(5):
        t=time.time(); c,md=C.encrypt(im,K); tt.append(time.time()-t)
        t=time.time(); C.decrypt(c,md,K); td.append(time.time()-t)
    sc[s]=(float(np.mean(tt)),float(np.mean(td)))
R['time_scaling']=sc; print('timing',R['time_256'],sc,flush=True)
# 512 images: lossless + metrics
ENC5={};MD5={}
for n in NAMES:
    c,md=C.encrypt(IM5[n],K,nonce=NON[n]); p,tk,hk=C.decrypt(c,md,K); assert np.array_equal(p,IM5[n]) and tk and hk
    ENC5[n]=c;MD5[n]=md
# ---------- entropy / local entropy / chi2 / correlation (both sizes)
def basic(IMS,ENCS,tag):
    out={}
    for n in NAMES:
        a=IMS[n]; e=ENCS[n]; rl=np.random.default_rng(77+NAMES.index(n))
        out[n]={'plain_ch':[entropy(a[:,:,c]) for c in range(3)],'cipher_ch':[entropy(e[:,:,c]) for c in range(3)],
                'plain_all':entropy(a),'cipher_all':entropy(e),
                'local_cipher':[local_entropy(e[:,:,c],rl) for c in range(3)],'local_plain':[local_entropy(a[:,:,c],rl) for c in range(3)],
                'chi2':[chi2_p(e[:,:,c]) for c in range(3)],
                'corr_plain':[corr_full(a[:,:,c]) for c in range(3)],'corr_cipher':[corr_full(e[:,:,c]) for c in range(3)]}
    return out
R['b256']=basic(IM,ENC,'256'); R['b512']=basic(IM5,ENC5,'512')
# 2000-pair samples for the scatter plots (R channel) are drawn in the figure script
# ---------- NPCR / UACI
def diff_test(IMS,ENCS,L,TR,seed0,special_pos=True):
    out={}
    for n in NAMES:
        for mode in ('full','fixed'):
            rng=np.random.default_rng(seed0+NAMES.index(n)); res=[];lab=[]
            H0=C.image_digest(IMS[n]); M=IMS[n].shape[0]
            for t in range(TR):
                im2=IMS[n].copy()
                if special_pos and t==0: i,j,c=0,0,0
                elif special_pos and t==1: i,j,c=M-1,M-1,2
                else: i,j,c=rng.integers(0,M),rng.integers(0,M),rng.integers(0,3)
                im2[i,j,c]=(int(im2[i,j,c])+1)%256
                c2,_=C.encrypt(im2,K,nonce=NON[n],digest=(H0 if mode=='fixed' else None))
                res.append(np.concatenate(npcr_uaci(ENCS[n],c2)))
            res=np.array(res); cr=crit(L,0.05)
            out[f'{n}|{mode}']={'mean':res.mean(0).tolist(),'std':res.std(0).tolist(),'first':res[0].tolist(),'last':res[1].tolist(),
              'pass_npcr':float(np.mean(res[:,:3]>=cr['npcr_min'])),'pass_uaci':float(np.mean((res[:,3:]>=cr['uaci_lo'])&(res[:,3:]<=cr['uaci_hi'])))}
        print('diff',n,out[f'{n}|full']['mean'][:3],out[f'{n}|fixed']['mean'][:3],flush=True)
    return out
R['npcr']=diff_test(IM,ENC,L1,100,555)
R['npcr512']=diff_test(IM5,ENC5,512*512,30,777)
# fresh-nonce variant (two encryptions of the modified/unmodified image with independent nonces) and equal images
R['fresh']={}
for n in NAMES:
    rng=np.random.default_rng(31+NAMES.index(n)); res=[]
    for t in range(20):
        im2=IM[n].copy(); i,j,c=rng.integers(0,256),rng.integers(0,256),rng.integers(0,3); im2[i,j,c]=(int(im2[i,j,c])+1)%256
        a,_=C.encrypt(IM[n],K); b,_=C.encrypt(im2,K); res.append(np.concatenate(npcr_uaci(a,b)))
    same=[]
    for t in range(20):
        a,_=C.encrypt(IM[n],K); b,_=C.encrypt(IM[n],K); same.append(np.concatenate(npcr_uaci(a,b)))
    R['fresh'][n]={'modified':np.mean(res,0).tolist(),'identical_image':np.mean(same,0).tolist()}
# ---------- key sensitivity
names6=['x0','y0','z0','gamma','r','mu']
R['keysens']={}
for n in NAMES:
    rows={}; H=C.image_digest(IM[n])
    for delta in (1e-13,1e-14):
        for idx,nm in enumerate(names6+['all']):
            k2=[v+delta for v in K] if nm=='all' else [v+(delta if i==idx else 0) for i,v in enumerate(K)]
            c2,md2=C.encrypt(IM[n],tuple(k2),nonce=NON[n],digest=H)
            a,b=npcr_uaci(ENC[n],c2)
            p,tk,hk=C.decrypt(ENC[n],MD[n],tuple(k2),check_tag=False)
            dn,du=npcr_uaci(IM[n],p)
            _,tk2,_=C.decrypt(ENC[n],MD[n],tuple(k2))
            rows[f'{nm}|{delta:g}']={'npcr':a.tolist(),'uaci':b.tolist(),'dec_npcr':dn.tolist(),'dec_hash_ok':bool(hk),'tag_ok':bool(tk2),
               'dec_corr':float(np.corrcoef(IM[n].reshape(-1),p.reshape(-1))[0,1])}
    R['keysens'][n]=rows
print('keysens done',flush=True)
# ---------- special images
R['special']={}
for nm,val in (('black',0),('white',255)):
    im=np.full((256,256,3),val,np.uint8); c,md=C.encrypt(im,K); p,tk,hk=C.decrypt(c,md,K)
    R['special'][nm]={'entropy':entropy(c),'entropy_ch':[entropy(c[:,:,i]) for i in range(3)],'ok':bool(np.array_equal(p,im) and tk and hk),'chi2':[chi2_p(c[:,:,i]) for i in range(3)]}
    ENC[nm]=c
c1,_=C.encrypt(np.zeros((256,256,3),np.uint8),K,nonce=b'a'*16); c2,_=C.encrypt(np.full((256,256,3),255,np.uint8),K,nonce=b'a'*16)
a,u=npcr_uaci(c1,c2); R['special']['black_vs_white_same_nonce']={'npcr':a.tolist(),'uaci':u.tolist()}
# ---------- attacks with deliberately reused nonce (worst case)
R['attack']={}
nz=b'z'*16; Z0=np.zeros((256,256,3),np.uint8); CZ,_=C.encrypt(Z0,K,nonce=nz); HZ=C.image_digest(Z0)
for n in NAMES:
    A_=IM[n]; cA,mdA=C.encrypt(A_,K,nonce=nz); est=cA^CZ
    a,_u=npcr_uaci(A_,est); cc=float(np.corrcoef(A_.reshape(-1),est.reshape(-1))[0,1])
    streams=C._setup(K,nz,HZ,L1,3); cur=np.ascontiguousarray(np.transpose(cA,(2,0,1)).reshape(3,L1))
    for rs in reversed(streams): cur=C._round_inverse(cur,rs,L1)
    rec=np.transpose(cur.reshape(3,256,256),(1,2,0)); a2,_u2=npcr_uaci(A_,rec); cc2=float(np.corrcoef(A_.reshape(-1),rec.reshape(-1))[0,1])
    R['attack'][n]={'cancel_npcr':a.tolist(),'cancel_corr':cc,'reuse_npcr':a2.tolist(),'reuse_corr':cc2}
# ---------- mask / nonce behaviour
hs=[C.encrypt(IM['Lena'],K)[1][1] for _ in range(200)]
hd=[sum(bin(x^y).count('1') for x,y in zip(hs[i],hs[i+1])) for i in range(199)]
R['hstar']={'distinct':len(set(hs)),'mean_hamming_between_consecutive':float(np.mean(hd)),'bit_freq_mean':float(np.mean(np.unpackbits(np.frombuffer(b''.join(hs),np.uint8)))) }
# ---------- tampering / damage / robustness (tag disabled to measure error propagation)
R['damage']={}
for n in NAMES:
    c=ENC[n].copy(); c[100,100,0]^=1
    p0,tk,hk=C.decrypt(c,MD[n],K); 
    p,tk2,hk2=C.decrypt(c,MD[n],K,check_tag=False); a,_u=npcr_uaci(IM[n],p)
    R['damage'][n]={'rejected_with_tag':p0 is None,'npcr_unchecked':a.tolist(),'hash_ok_unchecked':bool(hk2)}
def psnr(a,b):
    m=np.mean((a.astype(float)-b.astype(float))**2); return float(10*np.log10(255**2/m)) if m>0 else 99.0
R['robust']={}
rng=np.random.default_rng(5)
for n in ('Lena','Baboon'):
    out={}
    for tag,fn in (('salt_pepper_0.1%',lambda c: sp(c,0.001)),('salt_pepper_1%',lambda c: sp(c,0.01)),('crop_1/16_zero',lambda c: cr_(c))):
        pass
    def sp(c,pr):
        c=c.copy(); m=rng.random(c.shape[:2])<pr; c[m]=rng.choice([0,255],size=(int(m.sum()),3)); return c
    def cr_(c):
        c=c.copy(); c[:64,:64]=0; return c
    for tag,c in (('salt_pepper_0.1%',sp(ENC[n],0.001)),('salt_pepper_1%',sp(ENC[n],0.01)),('crop_1/16_zero',cr_(ENC[n]))):
        p,tk,hk=C.decrypt(c,MD[n],K,check_tag=False); a,_u=npcr_uaci(IM[n],p)
        out[tag]={'psnr':psnr(IM[n],p),'npcr':float(np.mean(a)),'tag_ok':bool(tk)}
    R['robust'][n]=out
json.dump(R,open('results/main.json','w'),indent=1)
# save encrypted images for figure script
np.savez_compressed('results/enc_cache.npz',**{n:ENC[n] for n in ENC})
print('done',flush=True)
