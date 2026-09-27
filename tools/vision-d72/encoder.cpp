// Measurement harness only; dynamically links the deployed encoder and HIP backend.
#include "clip.h"
#include "mtmd-image.h"
#include "ggml-backend.h"
#include <rocprofiler-sdk-roctx/roctx.h>
#include <chrono>
#include <fstream>
#include <iostream>
#include <vector>
#include <stdexcept>
int main(int argc,char **argv) {
 if(argc!=6) { std::cerr<<"encoder RGBfile width height repeats label\n"; return 2; }
 int w=std::stoi(argv[2]),h=std::stoi(argv[3]),n=std::stoi(argv[4]);
 std::vector<unsigned char> rgb(size_t(w)*h*3);
 std::ifstream f(argv[1],std::ios::binary); f.read((char*)rgb.data(),rgb.size());
 if(f.gcount()!=long(rgb.size())) throw std::runtime_error("RGB length mismatch");
 ggml_backend_load_all();
 clip_context_params p{}; p.use_gpu=true; p.flash_attn_type=std::getenv("ENCODER_NO_FA") ? CLIP_FLASH_ATTN_TYPE_DISABLED : CLIP_FLASH_ATTN_TYPE_ENABLED;
 p.image_min_tokens=-1; p.image_max_tokens=-1; p.warmup=false;
 const char* model=std::getenv("D72_MMPROJ");
 if(!model || !*model) throw std::runtime_error("Set D72_MMPROJ to the vision projector GGUF");
 auto init=clip_init(model,p);
 if(!init.ctx_v) throw std::runtime_error("no vision context");
 auto *ctx=init.ctx_v; auto *img=clip_image_u8_init();
 clip_build_img_from_pixels(rgb.data(),w,h,img);
 clip_image_f32_batch batch;
 mtmd_image_preprocessor_dyn_size pre(ctx);
 if(!pre.preprocess(*img,batch)) throw std::runtime_error("preprocessing failed");
 if(clip_image_f32_batch_n_images(&batch)!=1) throw std::runtime_error("expected one image");
 auto *im=clip_image_f32_get_img(&batch,0);
 int tokens=clip_n_output_tokens(ctx,im);
 std::vector<float> output(size_t(tokens)*clip_n_mmproj_embd(ctx));
 std::cout<<"SHAPE "<<argv[5]<<" "<<clip_image_f32_batch_nx(&batch,0)<<"x"<<clip_image_f32_batch_ny(&batch,0)<<" tokens="<<tokens<<std::endl;
 for(int i=-1;i<n;i++) {
  std::string label=std::string(i<0?"warmup_":"encode_")+argv[5]+"_"+std::to_string(i);
  roctxRangePushA(label.c_str());
  auto start=std::chrono::steady_clock::now();
  bool ok=clip_image_batch_encode(ctx,16,&batch,output.data());
  double ms=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-start).count();
  roctxRangePop();
  if(!ok) throw std::runtime_error("encode failed");
  std::cout<<"RESULT "<<label<<" "<<ms<<" ms first="<<output[0]<<std::endl;
 }
 if(const char* path=std::getenv("ENCODER_DUMP")) {
  std::ofstream dump(path,std::ios::binary); dump.write((const char*)output.data(),output.size()*sizeof(float));
 }
 clip_image_u8_free(img); clip_free(ctx);
}
