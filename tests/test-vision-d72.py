import ctypes as C
import os
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
LIB=Path(os.environ.get('D72_LIB',str(ROOT/'build-vision-d72/libvision_d72.so')))
assert LIB.exists(), 'D=72 intrinsic attention candidate has not been implemented'
lib=C.CDLL(str(LIB)); fn=lib.vision_d72
fn.argtypes=[C.c_void_p]*4+[C.c_int]*4+[C.POINTER(C.c_float)]
fn.restype=C.c_int

def check(n,h,seed,amplitude=1):
 rng=np.random.default_rng(seed)
 q=(rng.normal(size=(h,n,72))*amplitude).astype(np.float32)
 k=(rng.normal(size=(h,n,72))*amplitude).astype(np.float16)
 v=rng.normal(size=(h,n,72)).astype(np.float16)
 s=(q@k.astype(np.float32).transpose(0,2,1))/np.sqrt(72)
 s-=s.max(axis=-1,keepdims=True); p=np.exp(s); p/=p.sum(axis=-1,keepdims=True)
 ref=p@v.astype(np.float32)
 for pipeline in (0,1):
  out=np.full(q.shape,np.nan,np.float32); ms=C.c_float()
  rc=fn(q.ctypes.data,k.ctypes.data,v.ctypes.data,out.ctypes.data,n,h,pipeline,1,C.byref(ms))
  assert rc==0, f'HIP failure {rc}'
  assert np.isfinite(out).all(), (n,h,pipeline,'nonfinite')
  err=float(abs(out-ref).max()); cosine=float(np.vdot(out.ravel(),ref.ravel())/(np.linalg.norm(out)*np.linalg.norm(ref)))
  assert err<.01 and cosine>.9999,(n,h,pipeline,err,cosine)
  print(f'PASS N={n} H={h} pipeline={pipeline} max_abs={err:.6g} cosine={cosine:.9f} ms={ms.value:.3f}',flush=True)
for case in [(1,1,1),(7,2,2),(16,1,3),(17,2,4),(31,3,5),(64,2,6),(79,2,7),(129,1,8),(257,2,9)]: check(*case)
check(79,2,10,3)
