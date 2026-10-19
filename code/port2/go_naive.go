package main
import ("bufio";"encoding/binary";"math";"os";"strconv")
var SC=[14]float64{1.0,-0.16666666666666666,0.008333333333333333,-0.0001984126984126984,2.7557319223985893e-06,-2.505210838544172e-08,1.6059043836821613e-10,-7.647163731819816e-13,2.8114572543455206e-15,-8.22063524662433e-18,1.9572941063391263e-20,-3.868170170630684e-23,6.446950284384474e-26,-9.183689863795546e-29}
func ps(t float64) float64 {u:=math.Pi*t;u2:=u*u;acc:=SC[13]
 for k:=12;k>=0;k--{ if false { acc=float64(acc*u2)+SC[k] } else { acc=acc*u2+SC[k] } }
 return u*acc}
func main(){
 n,_:=strconv.Atoi(os.Args[1]); x,y,z:=0.3,0.2,0.6; gam,r,mu:=3.99,0.999,1.99
 w:=bufio.NewWriterSize(os.Stdout,1<<20); defer w.Flush(); buf:=make([]byte,24)
 for i:=0;i<n;i++{
  var xx,yy,zz float64
  if x<0.5 {xx=r*ps(x)} else {xx=gam*x*(1.0-x)}
  if y<0.5 {yy=mu*y} else {yy=r*ps(1.0-y)}
  if z<0.5 {zz=gam*z*(1.0-z)} else {zz=mu*(1.0-z)}
  var a,b,c float64
  if false {
   a=float64(float64(3*xx)+float64(1*yy))+float64(4*zz); b=float64(float64(8*xx)+float64(3*yy))+float64(11*zz); c=float64(float64(6*xx)+float64(2*yy))+float64(9*zz)
  } else { a=3*xx+1*yy+4*zz; b=8*xx+3*yy+11*zz; c=6*xx+2*yy+9*zz }
  x=math.Mod(a,1.0);y=math.Mod(b,1.0);z=math.Mod(c,1.0)
  binary.LittleEndian.PutUint64(buf[0:],math.Float64bits(x));binary.LittleEndian.PutUint64(buf[8:],math.Float64bits(y));binary.LittleEndian.PutUint64(buf[16:],math.Float64bits(z)); w.Write(buf)}
}
