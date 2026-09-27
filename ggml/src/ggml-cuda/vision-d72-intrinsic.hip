// Experimental standalone D=72 vision Flash Attention. Not production dispatch.
#include <hip/hip_runtime.h>
#include <cmath>
#include <algorithm>
using h16 = _Float16 __attribute__((ext_vector_type(16)));
using f8 = float __attribute__((ext_vector_type(8)));

// Four wave32 query tiles reuse one 16-key tile. No N*N materialization.
// Pipeline variant prefetches next K/V into registers during current tile work.
template<bool PIPE>
__global__ __launch_bounds__(128) void vision_d72_intrinsic(
 const float* __restrict__ q, const _Float16* __restrict__ k,
 const _Float16* __restrict__ v, float* __restrict__ out, int n) {
 const int tid=threadIdx.x, lane=tid%32, wave=tid/32, col=lane%16;
 const int q0=blockIdx.x*64+wave*16, head=blockIdx.y;
 const size_t base=size_t(head)*n*72;
 __shared__ _Float16 ks[16][80];
 __shared__ _Float16 vs[80][18];
 __shared__ _Float16 ps[4][16][18];
 h16 qr[5]; f8 acc[5]; float rm[8],rs[8];
 #pragma unroll
 for(int s=0;s<5;s++) {
  #pragma unroll
  for(int j=0;j<16;j++) {
   const int d=s*16+j;
   qr[s][j]=(q0+col<n && d<72)?(_Float16)(q[base+size_t(q0+col)*72+d]*0.11785113019775793f):(_Float16)0;
  }
  acc[s]={};
 }
 #pragma unroll
 for(int i=0;i<8;i++){rm[i]=-INFINITY;rs[i]=0;}
 // Initial tile load, and register staging for the next tile.
 _Float16 nk[10],nv[10];
 #pragma unroll
 for(int j=0;j<10;j++) {
  int idx=tid+j*128, row=idx/80,d=idx%80;
  ks[row][d]=(row<n && d<72)?k[base+row*72+d]:(_Float16)0;
  vs[d][row]=(row<n && d<72)?v[base+row*72+d]:(_Float16)0;
 }
 __syncthreads();
 for(int key0=0;key0<n;key0+=16) {
  if constexpr(PIPE) {
   #pragma unroll
   for(int j=0;j<10;j++) {
    int idx=tid+j*128,row=idx/80,d=idx%80,g=key0+16+row;
    nk[j]=(g<n && d<72)?k[base+size_t(g)*72+d]:(_Float16)0;
    nv[j]=(g<n && d<72)?v[base+size_t(g)*72+d]:(_Float16)0;
   }
   // Prevent moving global prefetch after WMMA/softmax in the scheduling region.
   __builtin_amdgcn_sched_barrier(0);
  }
  f8 score={};
  #pragma unroll
  for(int s=0;s<5;s++) {
   h16 kr;
   #pragma unroll
   for(int j=0;j<16;j++) kr[j]=ks[col][s*16+j];
   score=__builtin_amdgcn_wmma_f32_16x16x16_f16_w32(qr[s],kr,score);
  }
  #pragma unroll
  for(int i=0;i<8;i++) {
   int row=2*i+lane/16;
   float val=(key0+col<n)?score[i]:-INFINITY;
   float mx=val;
   #pragma unroll
   for(int delta=8;delta;delta>>=1) mx=fmaxf(mx,__shfl_xor(mx,delta));
   float nm=fmaxf(rm[i],mx);
   float alpha=__expf(rm[i]-nm),prob=__expf(val-nm),sum=prob;
   #pragma unroll
   for(int delta=8;delta;delta>>=1) sum+=__shfl_xor(sum,delta);
   rs[i]=rs[i]*alpha+sum; rm[i]=nm;
   #pragma unroll
   for(int s=0;s<5;s++) acc[s][i]*=alpha;
   ps[wave][row][col]=(_Float16)prob;
  }
  __syncthreads();
  h16 pr;
  #pragma unroll
  for(int j=0;j<16;j++) pr[j]=ps[wave][col][j];
  #pragma unroll
  for(int s=0;s<5;s++) {
   h16 vr;
   #pragma unroll
   for(int j=0;j<16;j++) vr[j]=vs[s*16+col][j];
   acc[s]=__builtin_amdgcn_wmma_f32_16x16x16_f16_w32(pr,vr,acc[s]);
  }
  __syncthreads();
  if(key0+16<n) {
   #pragma unroll
   for(int j=0;j<10;j++) {
    int idx=tid+j*128,row=idx/80,d=idx%80,g=key0+16+row;
    if constexpr(PIPE) {ks[row][d]=nk[j];vs[d][row]=nv[j];}
    else {
     ks[row][d]=(g<n && d<72)?k[base+size_t(g)*72+d]:(_Float16)0;
     vs[d][row]=(g<n && d<72)?v[base+size_t(g)*72+d]:(_Float16)0;
    }
   }
  }
  __syncthreads();
 }
 #pragma unroll
 for(int i=0;i<8;i++) {
  int row=q0+2*i+lane/16;
  #pragma unroll
  for(int s=0;s<5;s++) {
   int d=s*16+col;
   if(row<n && d<72) out[base+size_t(row)*72+d]=acc[s][i]/rs[i];
  }
 }
}

extern "C" int vision_d72(const float* q,const _Float16* k,const _Float16* v,
 float* out,int n,int heads,int pipeline,int repeats,float* ms) {
 if(n<=0 || heads<=0 || repeats<=0 || (pipeline!=0 && pipeline!=1)) return -1;
 float *dq=nullptr,*dout=nullptr; _Float16 *dk=nullptr,*dv=nullptr;
 hipEvent_t start=nullptr,end=nullptr; hipError_t err=hipSuccess;
 const size_t count=size_t(n)*heads*72;
 #define TRY(x) do {err=(x);if(err!=hipSuccess) goto cleanup;} while(0)
 TRY(hipMalloc(&dq,count*4));TRY(hipMalloc(&dout,count*4));
 TRY(hipMalloc(&dk,count*2));TRY(hipMalloc(&dv,count*2));
 TRY(hipMemcpy(dq,q,count*4,hipMemcpyHostToDevice));
 TRY(hipMemcpy(dk,k,count*2,hipMemcpyHostToDevice));
 TRY(hipMemcpy(dv,v,count*2,hipMemcpyHostToDevice));
 TRY(hipEventCreate(&start));TRY(hipEventCreate(&end));
 for(int i=-1;i<repeats;i++) {
  if(i==0) {TRY(hipEventRecord(start));}
  if(pipeline) vision_d72_intrinsic<true><<<dim3((n+63)/64,heads),128>>>(dq,dk,dv,dout,n);
  else vision_d72_intrinsic<false><<<dim3((n+63)/64,heads),128>>>(dq,dk,dv,dout,n);
  TRY(hipGetLastError());
  if(i==-1) {TRY(hipDeviceSynchronize());}
 }
 TRY(hipEventRecord(end));TRY(hipEventSynchronize(end));
 TRY(hipEventElapsedTime(ms,start,end)); *ms/=repeats;
 TRY(hipMemcpy(out,dout,count*4,hipMemcpyDeviceToHost));
cleanup:
 if(start) (void)hipEventDestroy(start); if(end) (void)hipEventDestroy(end);
 if(dq)(void)hipFree(dq);if(dout)(void)hipFree(dout);if(dk)(void)hipFree(dk);if(dv)(void)hipFree(dv);
 return int(err);
}
