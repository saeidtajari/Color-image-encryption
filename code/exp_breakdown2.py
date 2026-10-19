"""Interleaved time breakdown: all steps and the total are timed in the same loop (60 runs), residual reported.
The ChaCha20 anchor words are timed separately; this time is part of map_and_key_material_4_rounds."""
import numpy as np, json, time
from PIL import Image
import cipher as C
im=np.array(Image.open('images/Lena.png')); K=C.KEY; nonce=bytes(range(16)); M,N,_=im.shape; L=M*N
C.encrypt(im,K,nonce=nonce)
pc=time.perf_counter
keys=['digest_sha256','hkdf_init_mask_mac_stream','map_and_key_material_4_rounds','confusion_diffusion_cipher_4_rounds','hmac_tag','encrypt_total','decrypt_total','decrypt_tag_reject','anchor_words_4_rounds']
S={k:[] for k in keys}
c0,meta=C.encrypt(im,K,nonce=nonce)
cur=np.ascontiguousarray(np.transpose(im,(2,0,1)).reshape(3,L)).astype(np.uint8)
for it in range(60):
    t=pc();H=C.image_digest(im);S['digest_sha256'].append(pc()-t)
    t=pc();C.seed_digest(H,nonce);C.key_mask(K,nonce);C.hkdf_expand(C._prk(K,nonce),b'mac',32);ks=C.stream_key(K,nonce,H);S['hkdf_init_mask_mac_stream'].append(pc()-t)
    t=pc();st=C._setup(K,nonce,H,L,3);S['map_and_key_material_4_rounds'].append(pc()-t)
    t=pc()
    for rd in range(1,C.ROUNDS+1): C.anchor_words(ks,rd,8*L+514)
    S['anchor_words_4_rounds'].append(pc()-t)
    t=pc();x=cur
    for rs in st: x=C._round_forward(x,rs,L,C.DEFAULT_CFG)
    S['confusion_diffusion_cipher_4_rounds'].append(pc()-t)
    t=pc();C.mac_tag(K,nonce,meta[1],(M,N),3,c0.tobytes());S['hmac_tag'].append(pc()-t)
    t=pc();c,m2=C.encrypt(im,K,nonce=nonce);S['encrypt_total'].append(pc()-t)
    t=pc();C.decrypt(c,m2,K);S['decrypt_total'].append(pc()-t)
    t=pc();C.decrypt(c,(m2[0],m2[1],bytes(32)),K);S['decrypt_tag_reject'].append(pc()-t)
R={k:[float(np.mean(v)),float(np.std(v))] for k,v in S.items()}
st_sum=sum(np.array(S[k]) for k in keys[:5])
oth=np.array(S['encrypt_total'])-st_sum
R['other']=[float(oth.mean()),float(oth.std())]; R['n_runs']=60
json.dump(R,open('results/breakdown.json','w'),indent=1); print(json.dumps(R,indent=1))
