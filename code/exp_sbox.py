import numpy as np, json
import cipher as C
def nl_du(S):
    x=np.arange(256); pc=np.array([bin(i).count('1')&1 for i in range(256)])
    nl=256
    for b in range(1,256):
        f=1-2*np.array([pc[b&int(s)] for s in S]).astype(np.int64)
        w=f.copy(); h=1
        while h<256:
            w=w.reshape(-1,2,h); w=np.concatenate([w[:,0:1,:]+w[:,1:2,:],w[:,0:1,:]-w[:,1:2,:]],axis=1).reshape(-1); h*=2
        nl=min(nl,128-int(np.max(np.abs(w)))//2)
    du=0
    for a in range(1,256):
        d=S[x^a]^S[x]; du=max(du,int(np.bincount(d,minlength=256).max()))
    return nl,du
rng=np.random.default_rng(11); R={}
ours=[];rand=[];chaos=[]
for i in range(300):
    k=(rng.uniform(.05,.95),rng.uniform(.05,.95),rng.uniform(.05,.95),rng.uniform(3.7,4),rng.uniform(.9,1),rng.uniform(1.8,2))
    nonce=bytes(rng.bytes(16)); Hd=bytes(rng.bytes(32))
    rs,_=C.gen_round(C.derive_initial(k,C.seed_digest(Hd,nonce)),k[3:],64,3,1,C.stream_key(k,nonce,Hd),C.DEFAULT_CFG)
    ours.append(nl_du(rs.S1)); rand.append(nl_du(rng.permutation(256)))
    rc,_=C.gen_round(C.derive_initial(k,C.seed_digest(Hd,nonce)),k[3:],64,3,1,None,dict(C.DEFAULT_CFG,anchor=False)); chaos.append(nl_du(rc.S1))
o=np.array(ours); r=np.array(rand); q=np.array(chaos)
R={'ours':{'nl_mean':float(o[:,0].mean()),'nl_min':int(o[:,0].min()),'nl_max':int(o[:,0].max()),'du_mean':float(o[:,1].mean()),'du_min':int(o[:,1].min()),'du_max':int(o[:,1].max())},
   'random_perm':{'nl_mean':float(r[:,0].mean()),'nl_min':int(r[:,0].min()),'nl_max':int(r[:,0].max()),'du_mean':float(r[:,1].mean()),'du_min':int(r[:,1].min()),'du_max':int(r[:,1].max())},'chaos_only':{'nl_mean':float(q[:,0].mean()),'nl_min':int(q[:,0].min()),'nl_max':int(q[:,0].max()),'du_mean':float(q[:,1].mean()),'du_min':int(q[:,1].min()),'du_max':int(q[:,1].max())},'n':300,'aes':{'nl':112,'du':4}}
print(R); json.dump(R,open('results/sbox.json','w'),indent=1)
