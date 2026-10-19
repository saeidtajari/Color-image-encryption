const SC=[1.0,-0.16666666666666666,0.008333333333333333,-0.0001984126984126984,2.7557319223985893e-06,-2.505210838544172e-08,1.6059043836821613e-10,-7.647163731819816e-13,2.8114572543455206e-15,-8.22063524662433e-18,1.9572941063391263e-20,-3.868170170630684e-23,6.446950284384474e-26,-9.183689863795546e-29];
function ps(t){const u=Math.PI*t,u2=u*u;let acc=SC[13];for(let k=12;k>=0;k--)acc=acc*u2+SC[k];return u*acc;}
const n=parseInt(process.argv[2]);let x=0.3,y=0.2,z=0.6;const gam=3.99,r=0.999,mu=1.99;
const fs=require('fs');const CH=100000;let buf=new BigInt64Array(3*CH);let m=0;
for(let i=0;i<n;i++){const xx=x<0.5?r*ps(x):gam*x*(1.0-x);const yy=y<0.5?mu*y:r*ps(1.0-y);const zz=z<0.5?gam*z*(1.0-z):mu*(1.0-z);
 const A=3*xx+1*yy+4*zz,B=8*xx+3*yy+11*zz,C=6*xx+2*yy+9*zz;x=A%1.0;y=B%1.0;z=C%1.0;
 const gx=(x+y*z)%1.0,gy=(y+z*x)%1.0,gz=(z+x*y)%1.0;
 buf[m++]=BigInt(Math.floor(gx*1e14));buf[m++]=BigInt(Math.floor(gy*1e14));buf[m++]=BigInt(Math.floor(gz*1e14));
 if(m==3*CH){fs.writeSync(1,Buffer.from(buf.buffer));m=0;}}
if(m>0)fs.writeSync(1,Buffer.from(buf.buffer,0,8*m));
