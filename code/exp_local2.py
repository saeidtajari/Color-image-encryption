import numpy as np, json
from PIL import Image
import cipher as C
from metrics import *
NAMES=['Lena','Baboon','Peppers','Fruits']
R=json.load(open('results/main.json')); mu,lo,hi=R['local_null']
out={}
for n in NAMES:
    im=np.array(Image.open(f'images/{n}.png')); rng=np.random.default_rng(1000+NAMES.index(n)); vals=[[],[],[]]
    for t in range(100):
        c,_=C.encrypt(im,C.KEY)
        for k in range(3): vals[k].append(local_entropy(c[:,:,k],rng))
    out[n]=[{'mean':float(np.mean(v)),'sd':float(np.std(v)),'pass':float(np.mean([(lo<=x<=hi) for x in v]))} for v in vals]
    print(n,out[n],flush=True)
json.dump(out,open('results/local.json','w'),indent=1)
