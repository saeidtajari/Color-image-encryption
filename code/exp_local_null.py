"""Null distribution of the local Shannon entropy: the same procedure (30 blocks of 1936 pixels at random
positions of one channel) applied to 6000 random 256x256 channels. Updates results/main.json."""
import numpy as np, json
from metrics import local_entropy
rng=np.random.default_rng(12345); v=[]
for _ in range(6000):
    ch=rng.integers(0,256,(256,256),dtype=np.uint8); v.append(local_entropy(ch,rng))
v=np.array(v); R=json.load(open('results/main.json'))
R['local_null_old']=R['local_null']; R['local_null']=[float(v.mean()),float(np.percentile(v,2.5)),float(np.percentile(v,97.5))]; R['local_null_sd']=float(v.std())
json.dump(R,open('results/main.json','w')); print(R['local_null'],R['local_null_sd'])
