"""E7b: pass rate of the first-step relation for (a) the true state, (b) random wrong states."""
import sys, json, math, numpy as np, numba as nb
sys.path.insert(0,__import__('os').path.dirname(__import__('os').path.abspath(__file__))+'/..'); sys.path.insert(0,__import__('os').path.dirname(__import__('os').path.abspath(__file__)))
import cipher as C
from e7_attack import fx,fy,fz,lob,stepn
@nb.njit(cache=True)
def trial(n, seed, gam, r, mu):
    np.random.seed(seed)
    ok_true=0; ok_wrong=0
    for i in range(n):
        x=np.random.random();y=np.random.random();z=np.random.random()
        # true: observed next-step bytes from the true state
        xn,yn,zn=stepn(x,y,z,gam,r,mu)
        b=(lob(xn),lob(yn),lob(zn))
        lo=(int(math.floor(fx(x,gam,r)*1e14))%256,int(math.floor(fy(y,r,mu)*1e14))%256,int(math.floor(fz(z,gam,mu)*1e14))%256)
        A=((3,1,4),(8,3,11),(6,2,9)); R=(7,21,16)
        good=True
        for k in range(3):
            s=(b[k]-(A[k][0]*lo[0]+A[k][1]*lo[1]+A[k][2]*lo[2]))%256
            if s>R[k]: good=False
        if good: ok_true+=1
        # wrong: another random state's low bytes against the same observation
        x2=np.random.random();y2=np.random.random();z2=np.random.random()
        lo2=(int(math.floor(fx(x2,gam,r)*1e14))%256,int(math.floor(fy(y2,r,mu)*1e14))%256,int(math.floor(fz(z2,gam,mu)*1e14))%256)
        good=True
        for k in range(3):
            s=(b[k]-(A[k][0]*lo2[0]+A[k][1]*lo2[1]+A[k][2]*lo2[2]))%256
            if s>R[k]: good=False
        if good: ok_wrong+=1
    return ok_true, ok_wrong
gam,r,mu=C.KEY[3:6]
n=200_000_000
t,w=trial(n,1,gam,r,mu)
res=dict(n=n,true_pass=t/n,wrong_pass=w/n,wrong_hits=w,pred_wrong=2992/2**24)
print(res); json.dump(res,open('results/sa_e7b.json','w'))
