import java.io.*;import java.nio.*;
public class M{
 static final double[] SC={1.0,-0.16666666666666666,0.008333333333333333,-0.0001984126984126984,2.7557319223985893e-06,-2.505210838544172e-08,1.6059043836821613e-10,-7.647163731819816e-13,2.8114572543455206e-15,-8.22063524662433e-18,1.9572941063391263e-20,-3.868170170630684e-23,6.446950284384474e-26,-9.183689863795546e-29};
 static double ps(double t){double u=Math.PI*t,u2=u*u,acc=SC[13];for(int k=12;k>=0;k--)acc=acc*u2+SC[k];return u*acc;}
 public static void main(String[] a)throws Exception{int n=Integer.parseInt(a[0]);double x=0.3,y=0.2,z=0.6,gam=3.99,r=0.999,mu=1.99;
  OutputStream o=new BufferedOutputStream(System.out,1<<20);ByteBuffer bb=ByteBuffer.allocate(24).order(ByteOrder.LITTLE_ENDIAN);
  for(int i=0;i<n;i++){double xx=x<0.5?r*ps(x):gam*x*(1.0-x);double yy=y<0.5?mu*y:r*ps(1.0-y);double zz=z<0.5?gam*z*(1.0-z):mu*(1.0-z);
   double A=3*xx+1*yy+4*zz,B=8*xx+3*yy+11*zz,C=6*xx+2*yy+9*zz;x=A%1.0;y=B%1.0;z=C%1.0;bb.clear();bb.putDouble(x).putDouble(y).putDouble(z);o.write(bb.array());}
  o.flush();}}
