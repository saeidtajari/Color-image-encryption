"""Cause of the NPCR of 99.8% for one round: predicted from the bijectivity of the chained passes."""
import numpy as np, json, sys
sys.path.insert(0,__import__('os').path.dirname(__import__('os').path.abspath(__file__))+'/..')
import cipher as C
from PIL import Image
im=np.array(Image.open('images/Lena.png')); M,N,_=im.shape; L=M*N; K=C.KEY; nonce=bytes(range(16))
H=C.image_digest(im)
rs=C._setup(K,nonce,H,L,3)[0]
cur=np.ascontiguousarray(np.transpose(im,(2,0,1)).reshape(3,L)).astype(np.uint8)
base=C._round_forward(cur,rs,L,C.DEFAULT_CFG).reshape(-1)
rng=np.random.default_rng(1); P=[];Q=[]
for _ in range(4000):
    c=rng.integers(3); q=rng.integers(L); x=cur.copy(); x[c,q]=(int(x[c,q])+1)%256
    y=C._round_forward(x,rs,L,C.DEFAULT_CFG).reshape(-1)
    P.append(int(rs.TP[c][q])+c*L); Q.append(float(np.mean(y!=base)))
P=np.array(P);Q=np.array(Q); n=3*L
# prediction: positions below p are certain to differ; positions from p on differ with prob. ~255/256
pred=(P+(n-P)*255/256)/n
res={'mean_obs':100*float(Q.mean()),'mean_pred':100*float(pred.mean()),'rmse_pts':100*float(np.sqrt(np.mean((Q-pred)**2))),
     'corr':float(np.corrcoef(Q,pred)[0,1]),'ideal_random':100*255/256,
     'low_third_obs':100*float(Q[P<n/3].mean()),'low_third_pred':100*float(pred[P<n/3].mean()),
     'high_third_obs':100*float(Q[P>2*n/3].mean()),'high_third_pred':100*float(pred[P>2*n/3].mean()),'ntrials':4000,
     'min_obs':100*float(Q.min()),'max_obs':100*float(Q.max())}
healed=Q<0.5*pred
res['healed_fraction']=float(healed.mean()); res['healed_expected']=1/256
res['mean_obs_not_healed']=100*float(Q[~healed].mean()); res['mean_pred_not_healed']=100*float(pred[~healed].mean())
res['mean_unchanged_prefix_healed']=float(np.mean(1-Q[healed])) if healed.any() else None
res['n_healed']=int(healed.sum())
json.dump(res,open('results/sa_e6.json','w'),indent=1); print(json.dumps(res,indent=1))
