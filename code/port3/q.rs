use std::io::Write;
const SC:[f64;14]=[1.0,-0.16666666666666666,0.008333333333333333,-0.0001984126984126984,2.7557319223985893e-06,-2.505210838544172e-08,1.6059043836821613e-10,-7.647163731819816e-13,2.8114572543455206e-15,-8.22063524662433e-18,1.9572941063391263e-20,-3.868170170630684e-23,6.446950284384474e-26,-9.183689863795546e-29];
fn ps(t:f64)->f64{let u=std::f64::consts::PI*t;let u2=u*u;let mut acc=SC[13];for k in (0..13).rev(){acc=acc*u2+SC[k];}u*acc}
fn main(){let n:usize=std::env::args().nth(1).unwrap().parse().unwrap();let(mut x,mut y,mut z)=(0.3f64,0.2f64,0.6f64);let(gam,r,mu)=(3.99f64,0.999f64,1.99f64);
 let so=std::io::stdout();let mut w=std::io::BufWriter::with_capacity(1<<20,so.lock());
 for _ in 0..n{let xx=if x<0.5{r*ps(x)}else{gam*x*(1.0-x)};let yy=if y<0.5{mu*y}else{r*ps(1.0-y)};let zz=if z<0.5{gam*z*(1.0-z)}else{mu*(1.0-z)};
  let a=3.0*xx+1.0*yy+4.0*zz;let b=8.0*xx+3.0*yy+11.0*zz;let c=6.0*xx+2.0*yy+9.0*zz;x=a%1.0;y=b%1.0;z=c%1.0;
  let gx=(x+y*z)%1.0;let gy=(y+z*x)%1.0;let gz=(z+x*y)%1.0;
  for g in [gx,gy,gz]{let q=(g*1e14).floor() as i64;w.write_all(&q.to_le_bytes()).unwrap();}}}
