/* map (Eqs. 5-8) + output filter and quantization (Eq. 10); writes q as int64 little endian */
#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <stdint.h>
static const double SC[14]={1.0,-0.16666666666666666,0.008333333333333333,-0.0001984126984126984,2.7557319223985893e-06,-2.505210838544172e-08,1.6059043836821613e-10,-7.647163731819816e-13,2.8114572543455206e-15,-8.22063524662433e-18,1.9572941063391263e-20,-3.868170170630684e-23,6.446950284384474e-26,-9.183689863795546e-29};
static double ps(double t){double u=M_PI*t,u2=u*u,acc=SC[13];for(int k=12;k>=0;k--)acc=acc*u2+SC[k];return u*acc;}
int main(int argc,char**argv){
 long n=atol(argv[1]); double x=0.3,y=0.2,z=0.6,gam=3.99,r=0.999,mu=1.99;
 FILE*f=fopen(argv[2],"wb"); int64_t o[3];
 for(long i=0;i<n;i++){
  double xx= x<0.5? r*ps(x): gam*x*(1.0-x);
  double yy= y<0.5? mu*y : r*ps(1.0-y);
  double zz= z<0.5? gam*z*(1.0-z): mu*(1.0-z);
  double a=3*xx+1*yy+4*zz, b=8*xx+3*yy+11*zz, c=6*xx+2*yy+9*zz;
  x=fmod(a,1.0);y=fmod(b,1.0);z=fmod(c,1.0);
  double gx=fmod(x+y*z,1.0), gy=fmod(y+z*x,1.0), gz=fmod(z+x*y,1.0);
  o[0]=(int64_t)floor(gx*1e14); o[1]=(int64_t)floor(gy*1e14); o[2]=(int64_t)floor(gz*1e14);
  fwrite(o,8,3,f);
 }
 fclose(f);return 0;}
