// EXPERIMENTAL diagnostic interposer, not production integration.
// Requires graphs disabled; synchronizes explicitly to avoid assuming ctx ABI.
#include "../../ggml/src/ggml-cuda/vision-d72-intrinsic.hip"
#include "ggml.h"
#include <dlfcn.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>
struct ggml_backend_cuda_context;
template<typename T> __global__ void pack72(const char* src,T* dst,int n,int heads,size_t b1,size_t b2){
 size_t i=size_t(blockIdx.x)*256+threadIdx.x;
 if(i>=size_t(n)*heads*72)return;
 int d=i%72,pos=(i/72)%n,h=i/(size_t(n)*72);
 dst[i]=*(const T*)(src+h*b2+pos*b1+d*sizeof(T));
}
__global__ void scatter72(const float* src,float* dst,int n,int heads){
 size_t i=size_t(blockIdx.x)*256+threadIdx.x;
 if(i>=size_t(n)*heads*72)return;
 int d=i%72,pos=(i/72)%n,h=i/(size_t(n)*72);
 dst[(size_t(pos)*heads+h)*72+d]=src[i];
}
static void checked(hipError_t e){if(e!=hipSuccess){fprintf(stderr,"D72 HIP error: %s\n",hipGetErrorString(e));abort();}}
#include "reference.hip"
void ggml_cuda_flash_attn_ext(ggml_backend_cuda_context& ctx,ggml_tensor* dst){
 using Fn=void(*)(ggml_backend_cuda_context&,ggml_tensor*);
 static Fn original=(Fn)dlsym(RTLD_NEXT,"_Z24ggml_cuda_flash_attn_extR25ggml_backend_cuda_contextP11ggml_tensor");
 auto *q=dst->src[0],*k=dst->src[1],*v=dst->src[2];
 float p[3];memcpy(p,dst->op_params,sizeof(p));
 bool ok=getenv("D72_EXPERIMENTAL") && getenv("GGML_CUDA_DISABLE_GRAPHS") && q && k && v &&
 !dst->src[3] && !dst->src[4] && q->type==GGML_TYPE_F32 && k->type==GGML_TYPE_F16 && v->type==GGML_TYPE_F16 &&
 dst->type==GGML_TYPE_F32 && q->ne[0]==72 && k->ne[0]==72 && v->ne[0]==72 &&
 q->ne[1]==k->ne[1] && q->ne[1]==v->ne[1] && q->ne[2]==16 && k->ne[2]==16 && v->ne[2]==16 &&
 q->ne[3]==1 && k->ne[3]==1 && v->ne[3]==1 && q->nb[0]==4 && k->nb[0]==2 && v->nb[0]==2 &&
 fabsf(p[0]-0.11785113019775793f)<1e-7f && p[1]==0 && p[2]==0;
 if(!ok){if(!original)abort();original(ctx,dst);return;}
 fprintf(stderr,"D72_INTRINSIC_DISPATCH n=%ld heads=16\n",q->ne[1]);
 int n=q->ne[1],heads=16; size_t count=size_t(n)*heads*72;
 checked(hipDeviceSynchronize());
 float *dq,*out;_Float16 *dk,*dv;
 checked(hipMalloc(&dq,count*4));checked(hipMalloc(&dk,count*2));checked(hipMalloc(&dv,count*2));checked(hipMalloc(&out,count*4));
 pack72<float><<<(count+255)/256,256>>>((const char*)q->data,dq,n,heads,q->nb[1],q->nb[2]);
 pack72<_Float16><<<(count+255)/256,256>>>((const char*)k->data,dk,n,heads,k->nb[1],k->nb[2]);
 pack72<_Float16><<<(count+255)/256,256>>>((const char*)v->data,dv,n,heads,v->nb[1],v->nb[2]);
 if(getenv("D72_REFERENCE")) reference_attention(dq,dk,dv,out,n,heads);
 else vision_d72_intrinsic<true><<<dim3((n+63)/64,heads),128>>>(dq,dk,dv,out,n);
 scatter72<<<(count+255)/256,256>>>(out,(float*)dst->data,n,heads);
 checked(hipGetLastError());checked(hipDeviceSynchronize());
 static bool captured=false;
 if(getenv("D72_CAPTURE") && !captured) {
  captured=true;
  auto dump=[&](const char* name,void* ptr,size_t bytes){
   std::vector<char> host(bytes);checked(hipMemcpy(host.data(),ptr,bytes,hipMemcpyDeviceToHost));
   std::string path=std::string(getenv("D72_CAPTURE"))+"/"+name;
   FILE* f=fopen(path.c_str(),"wb");if(!f)abort();fwrite(host.data(),1,bytes,f);fclose(f);
  };
  dump("q.bin",dq,count*4);dump("k.bin",dk,count*2);dump("v.bin",dv,count*2);dump("operator-candidate.bin",out,count*4);
  original(ctx,dst);checked(hipDeviceSynchronize());
  pack72<float><<<(count+255)/256,256>>>((const char*)dst->data,dq,n,heads,size_t(heads)*72*4,72*4);
  checked(hipDeviceSynchronize());dump("operator-reference.bin",dq,count*4);
  scatter72<<<(count+255)/256,256>>>(out,(float*)dst->data,n,heads);checked(hipDeviceSynchronize());
 }
 checked(hipFree(dq));checked(hipFree(dk));checked(hipFree(dv));checked(hipFree(out));
}
