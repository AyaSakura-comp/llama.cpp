// Bounded-memory FP32 reference for validation only, no N*N allocation.
#include <rocblas/rocblas.h>
__global__ void half_to_float(const _Float16* src,float* dst,size_t n){size_t i=size_t(blockIdx.x)*256+threadIdx.x;if(i<n)dst[i]=(float)src[i];}
__global__ void reference_softmax(float* s,int n,int queries){
 int row=blockIdx.x;float* p=s+size_t(row)*n;int t=threadIdx.x;
 __shared__ float buf[256];float mx=-INFINITY;
 for(int i=t;i<n;i+=256)mx=fmaxf(mx,p[i]);buf[t]=mx;__syncthreads();
 for(int d=128;d;d>>=1){if(t<d)buf[t]=fmaxf(buf[t],buf[t+d]);__syncthreads();}mx=buf[0];
 float sum=0;for(int i=t;i<n;i+=256){float x=expf(p[i]-mx);p[i]=x;sum+=x;}buf[t]=sum;__syncthreads();
 for(int d=128;d;d>>=1){if(t<d)buf[t]+=buf[t+d];__syncthreads();}sum=buf[0];
 for(int i=t;i<n;i+=256)p[i]/=sum;
}
static void reference_attention(const float* q,const _Float16* k,const _Float16* v,float* out,int n,int heads){
 float *kf,*vf,*scores;size_t count=size_t(n)*heads*72;
 checked(hipMalloc(&kf,count*4));checked(hipMalloc(&vf,count*4));checked(hipMalloc(&scores,size_t(n)*128*heads*4));
 half_to_float<<<(count+255)/256,256>>>(k,kf,count);half_to_float<<<(count+255)/256,256>>>(v,vf,count);
 rocblas_handle h;if(rocblas_create_handle(&h)!=rocblas_status_success)abort();
 float scale=.11785113019775793f,zero=0,one=1;
 for(int i=0;i<n;i+=128){
  int m=std::min(128,n-i);
  auto a=rocblas_sgemm_strided_batched(h,rocblas_operation_transpose,rocblas_operation_none,n,m,72,&scale,kf,72,size_t(n)*72,q+i*72,72,size_t(n)*72,&zero,scores,n,size_t(n)*m,heads);
  if(a!=rocblas_status_success)abort();
  reference_softmax<<<m*heads,256>>>(scores,n,m);
  auto b=rocblas_sgemm_strided_batched(h,rocblas_operation_none,rocblas_operation_none,72,m,n,&one,vf,72,size_t(n)*72,scores,n,size_t(n)*m,&zero,out+i*72,72,size_t(n)*72,heads);
  if(b!=rocblas_status_success)abort();
 }
 checked(hipDeviceSynchronize());rocblas_destroy_handle(h);checked(hipFree(kf));checked(hipFree(vf));checked(hipFree(scores));
}
