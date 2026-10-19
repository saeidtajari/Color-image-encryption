"""E8: summary over 50 images of different content and size, 3 random keys and nonces each."""
import sys, json, time, numpy as np
sys.path.insert(0,__import__('os').path.dirname(__import__('os').path.abspath(__file__))+'/..')
from PIL import Image, ImageDraw
import skimage.data as sd
import cipher as C
from metrics import entropy, corr_full, npcr_uaci, chi2_p, crit
rng=np.random.default_rng(2026)
def rgb(a):
    a=np.asarray(a)
    if a.ndim==2: a=np.stack([a]*3,-1)
    if a.shape[2]==4: a=a[:,:,:3]
    return a.astype(np.uint8)
def fit(a,s):
    a=rgb(a); h,w=a.shape[:2]; m=min(h,w); i=(h-m)//2; j=(w-m)//2
    return np.array(Image.fromarray(a[i:i+m,j:j+m]).resize((s,s),Image.LANCZOS))
imgs=[]   # (name, group, array)
for n in ('Lena','Baboon','Peppers','Fruits'):
    imgs.append((n,'standard 256',fit(np.array(Image.open(f'images512/{n}.png').convert('RGB')),256)))
nat=['astronaut','camera','coffee','chelsea','rocket','brick','grass','gravel','retina','hubble_deep_field','immunohistochemistry','colorwheel','text','moon','page','horse','clock','coins','cell','checkerboard','logo']
for n in nat:
    imgs.append((n,'other 256',fit(getattr(sd,n)(),256)))
# size variants
for n,s in (('Lena',64),('Baboon',64),('astronaut',64),('Lena',128),('Baboon',128),('astronaut',128),('coffee',128),('Peppers',512),('Baboon',512),('Fruits',512),('astronaut',512),('rocket',512),('coffee',512),('colorwheel',512),('retina',1024),('hubble_deep_field',1024)):
    base=np.array(Image.open(f'images512/{n}.png').convert('RGB')) if n in ('Lena','Baboon','Peppers','Fruits') else getattr(sd,n)()
    imgs.append((f'{n}_{s}','size 64 to 1024',fit(base,s)))
# synthetic
S=256
imgs.append(('all black','synthetic',np.zeros((S,S,3),np.uint8)))
imgs.append(('all white','synthetic',np.full((S,S,3),255,np.uint8)))
imgs.append(('constant colour','synthetic',np.full((S,S,3),(128,64,200),np.uint8)))
imgs.append(('uniform noise','synthetic',rng.integers(0,256,(S,S,3),dtype=np.uint8)))
g=np.tile(np.arange(S,dtype=np.uint8)[None,:,None],(S,1,3)); imgs.append(('horizontal gradient','synthetic',g))
st=np.zeros((S,S,3),np.uint8); st[:,::8]=255; imgs.append(('vertical stripes','synthetic',st))
imgs.append(('binary random','synthetic',np.repeat((rng.integers(0,2,(S,S,1))*255).astype(np.uint8),3,2)))
sp=np.zeros((S,S,3),np.uint8); sp[100,100]=255; imgs.append(('single white pixel','synthetic',sp))
im=Image.new('RGB',(S,S),(255,255,255)); d=ImageDraw.Draw(im)
for k in range(12): d.text((8,8+k*20),'Scientia Iranica chaotic cipher test line %d'%k,fill=(0,0,0))
imgs.append(('rendered text','synthetic',np.array(im)))
print(len(imgs),'images',flush=True)
def rkey():
    lo_hi=C.KEY_RANGES
    k=[float(rng.uniform(0.01,0.99)) for _ in range(3)]+[float(rng.uniform(a,b)) for a,b in lo_hi[3:]]
    return tuple(k)
rows=[]
t0=time.time()
for name,grp,im in imgs:
    M,N=im.shape[:2]; L=M*N; cr=crit(L,0.05)
    for kk in range(3):
        key=rkey(); nonce=bytes(rng.integers(0,256,16,dtype=np.uint8))
        c,meta=C.encrypt(im,key,nonce=nonce)
        dec,tag_ok,hash_ok=C.decrypt(c,meta,key)
        exact=bool(tag_ok and hash_ok and dec is not None and np.array_equal(dec,im))
        im2=im.copy(); i,j,ch=int(rng.integers(0,M)),int(rng.integers(0,N)),int(rng.integers(0,3))
        im2[i,j,ch]=(int(im2[i,j,ch])+int(rng.integers(1,256)))%256
        c2,_=C.encrypt(im2,key,nonce=nonce)
        npcr,uaci=npcr_uaci(c,c2)
        cc=[abs(x) for k_ in range(3) for x in corr_full(c[:,:,k_])]
        chip=[chi2_p(c[:,:,k_])[1] for k_ in range(3)]
        rows.append(dict(name=name,group=grp,M=M,N=N,key=kk,ent=entropy(c),ent_ch=[entropy(c[:,:,k_]) for k_ in range(3)],
            rho=float(np.mean(cc)),rho_max=float(max(cc)),chi_p=chip,npcr=[float(x) for x in npcr],uaci=[float(x) for x in uaci],
            npcr_ok=[bool(x>=cr['npcr_min']) for x in npcr],uaci_ok=[bool(cr['uaci_lo']<=x<=cr['uaci_hi']) for x in uaci],exact=exact))
    print(name,round(time.time()-t0),flush=True)
json.dump(rows,open('results/sa_e8.json','w'))
