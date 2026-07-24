# llama.cpp

![llama](https://user-images.githubusercontent.com/1991296/230134379-7181e485-c521-4d23-a0d6-f7b3b61ba524.png)

[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![Release](https://img.shields.io/github/v/release/ggml-org/llama.cpp)](https://github.com/ggml-org/llama.cpp/releases)
[![Server](https://github.com/ggml-org/llama.cpp/actions/workflows/server.yml/badge.svg)](https://github.com/ggml-org/llama.cpp/actions/workflows/server.yml)

[Manifesto](https://github.com/ggml-org/llama.cpp/discussions/205) / [ggml](https://github.com/ggml-org/ggml) / [ops](https://github.com/ggml-org/llama.cpp/blob/master/docs/ops.md)

LLM inference in C/C++

> [Qwen 3.6 35B-A3B / gfx1151 optimization and profiling record](#qwen-36-35b-a3b--gfx1151-optimization-and-profiling-record): MMQ, UMA, sparse MTP logits, and real Pi Agent benchmarks.

## Recent API changes

- [Changelog for `libllama` API](https://github.com/ggml-org/llama.cpp/issues/9289)
- [Changelog for `llama-server` REST API](https://github.com/ggml-org/llama.cpp/issues/9291)

## Hot topics

- **Hugging Face cache migration: models downloaded with `-hf` are now stored in the standard Hugging Face cache directory, enabling sharing with other HF tools.**
- **[guide : using the new WebUI of llama.cpp](https://github.com/ggml-org/llama.cpp/discussions/16938)**
- [guide : running gpt-oss with llama.cpp](https://github.com/ggml-org/llama.cpp/discussions/15396)
- [[FEEDBACK] Better packaging for llama.cpp to support downstream consumers 🤗](https://github.com/ggml-org/llama.cpp/discussions/15313)
- Support for the `gpt-oss` model with native MXFP4 format has been added | [PR](https://github.com/ggml-org/llama.cpp/pull/15091) | [Collaboration with NVIDIA](https://blogs.nvidia.com/blog/rtx-ai-garage-openai-oss) | [Comment](https://github.com/ggml-org/llama.cpp/discussions/15095)
- Multimodal support arrived in `llama-server`: [#12898](https://github.com/ggml-org/llama.cpp/pull/12898) | [documentation](./docs/multimodal.md)
- VS Code extension for FIM completions: https://github.com/ggml-org/llama.vscode
- Vim/Neovim plugin for FIM completions: https://github.com/ggml-org/llama.vim
- Hugging Face Inference Endpoints now support GGUF out of the box! https://github.com/ggml-org/llama.cpp/discussions/9669
- Hugging Face GGUF editor: [discussion](https://github.com/ggml-org/llama.cpp/discussions/9268) | [tool](https://huggingface.co/spaces/CISCai/gguf-editor)

----

## Quick start

Getting started with llama.cpp is straightforward. Here are several ways to install it on your machine:

- Install `llama.cpp` using [brew, nix or winget](docs/install.md)
- Run with Docker - see our [Docker documentation](docs/docker.md)
- Download pre-built binaries from the [releases page](https://github.com/ggml-org/llama.cpp/releases)
- Build from source by cloning this repository - check out [our build guide](docs/build.md)

Once installed, you'll need a model to work with. Head to the [Obtaining and quantizing models](#obtaining-and-quantizing-models) section to learn more.

Example command:

```sh
# Use a local model file
llama-cli -m my_model.gguf

# Or download and run a model directly from Hugging Face
llama-cli -hf ggml-org/gemma-3-1b-it-GGUF

# Launch OpenAI-compatible API server
llama-server -hf ggml-org/gemma-3-1b-it-GGUF
```

## Description

The main goal of `llama.cpp` is to enable LLM inference with minimal setup and state-of-the-art performance on a wide
range of hardware - locally and in the cloud.

- Plain C/C++ implementation without any dependencies
- Apple silicon is a first-class citizen - optimized via ARM NEON, Accelerate and Metal frameworks
- AVX, AVX2, AVX512 and AMX support for x86 architectures
- RVV, ZVFH, ZFH, ZICBOP and ZIHINTPAUSE support for RISC-V architectures
- 1.5-bit, 2-bit, 3-bit, 4-bit, 5-bit, 6-bit, and 8-bit integer quantization for faster inference and reduced memory use
- Custom CUDA kernels for running LLMs on NVIDIA GPUs (support for AMD GPUs via HIP and Moore Threads GPUs via MUSA)
- Vulkan and SYCL backend support
- CPU+GPU hybrid inference to partially accelerate models larger than the total VRAM capacity

The `llama.cpp` project is the main playground for developing new features for the [ggml](https://github.com/ggml-org/ggml) library.

<details>
<summary>Models</summary>

Typically finetunes of the base models below are supported as well.

Instructions for adding support for new models: [HOWTO-add-model.md](docs/development/HOWTO-add-model.md)

#### Text-only

- [X] LLaMA 🦙
- [x] LLaMA 2 🦙🦙
- [x] LLaMA 3 🦙🦙🦙
- [X] [Mistral 7B](https://huggingface.co/mistralai/Mistral-7B-v0.1)
- [x] [Mixtral MoE](https://huggingface.co/models?search=mistral-ai/Mixtral)
- [x] [DBRX](https://huggingface.co/databricks/dbrx-instruct)
- [x] [Jamba](https://huggingface.co/ai21labs)
- [X] [Falcon](https://huggingface.co/models?search=tiiuae/falcon)
- [X] [Chinese LLaMA / Alpaca](https://github.com/ymcui/Chinese-LLaMA-Alpaca) and [Chinese LLaMA-2 / Alpaca-2](https://github.com/ymcui/Chinese-LLaMA-Alpaca-2)
- [X] [Vigogne (French)](https://github.com/bofenghuang/vigogne)
- [X] [BERT](https://github.com/ggml-org/llama.cpp/pull/5423)
- [X] [Koala](https://bair.berkeley.edu/blog/2023/04/03/koala/)
- [X] [Baichuan 1 & 2](https://huggingface.co/models?search=baichuan-inc/Baichuan) + [derivations](https://huggingface.co/hiyouga/baichuan-7b-sft)
- [X] [Aquila 1 & 2](https://huggingface.co/models?search=BAAI/Aquila)
- [X] [Starcoder models](https://github.com/ggml-org/llama.cpp/pull/3187)
- [X] [Refact](https://huggingface.co/smallcloudai/Refact-1_6B-fim)
- [X] [MPT](https://github.com/ggml-org/llama.cpp/pull/3417)
- [X] [Bloom](https://github.com/ggml-org/llama.cpp/pull/3553)
- [x] [Yi models](https://huggingface.co/models?search=01-ai/Yi)
- [X] [StableLM models](https://huggingface.co/stabilityai)
- [x] [Deepseek models](https://huggingface.co/models?search=deepseek-ai/deepseek)
- [x] [Qwen models](https://huggingface.co/models?search=Qwen/Qwen)
- [x] [PLaMo-13B](https://github.com/ggml-org/llama.cpp/pull/3557)
- [x] [Phi models](https://huggingface.co/models?search=microsoft/phi)
- [x] [PhiMoE](https://github.com/ggml-org/llama.cpp/pull/11003)
- [x] [GPT-2](https://huggingface.co/gpt2)
- [x] [Orion 14B](https://github.com/ggml-org/llama.cpp/pull/5118)
- [x] [InternLM2](https://huggingface.co/models?search=internlm2)
- [x] [CodeShell](https://github.com/WisdomShell/codeshell)
- [x] [Gemma](https://ai.google.dev/gemma)
- [x] [Mamba](https://github.com/state-spaces/mamba)
- [x] [Grok-1](https://huggingface.co/keyfan/grok-1-hf)
- [x] [Xverse](https://huggingface.co/models?search=xverse)
- [x] [Command-R models](https://huggingface.co/models?search=CohereForAI/c4ai-command-r)
- [x] [SEA-LION](https://huggingface.co/models?search=sea-lion)
- [x] [GritLM-7B](https://huggingface.co/GritLM/GritLM-7B) + [GritLM-8x7B](https://huggingface.co/GritLM/GritLM-8x7B)
- [x] [OLMo](https://allenai.org/olmo)
- [x] [OLMo 2](https://allenai.org/olmo)
- [x] [OLMoE](https://huggingface.co/allenai/OLMoE-1B-7B-0924)
- [x] [Granite models](https://huggingface.co/collections/ibm-granite/granite-code-models-6624c5cec322e4c148c8b330)
- [x] [GPT-NeoX](https://github.com/EleutherAI/gpt-neox) + [Pythia](https://github.com/EleutherAI/pythia)
- [x] [Snowflake-Arctic MoE](https://huggingface.co/collections/Snowflake/arctic-66290090abe542894a5ac520)
- [x] [Smaug](https://huggingface.co/models?search=Smaug)
- [x] [Poro 34B](https://huggingface.co/LumiOpen/Poro-34B)
- [x] [Bitnet b1.58 models](https://huggingface.co/1bitLLM)
- [x] [Flan T5](https://huggingface.co/models?search=flan-t5)
- [x] [Open Elm models](https://huggingface.co/collections/apple/openelm-instruct-models-6619ad295d7ae9f868b759ca)
- [x] [ChatGLM3-6b](https://huggingface.co/THUDM/chatglm3-6b) + [ChatGLM4-9b](https://huggingface.co/THUDM/glm-4-9b) + [GLMEdge-1.5b](https://huggingface.co/THUDM/glm-edge-1.5b-chat) + [GLMEdge-4b](https://huggingface.co/THUDM/glm-edge-4b-chat)
- [x] [GLM-4-0414](https://huggingface.co/collections/THUDM/glm-4-0414-67f3cbcb34dd9d252707cb2e)
- [x] [SmolLM](https://huggingface.co/collections/HuggingFaceTB/smollm-6695016cad7167254ce15966)
- [x] [EXAONE-3.0-7.8B-Instruct](https://huggingface.co/LGAI-EXAONE/EXAONE-3.0-7.8B-Instruct)
- [x] [FalconMamba Models](https://huggingface.co/collections/tiiuae/falconmamba-7b-66b9a580324dd1598b0f6d4a)
- [x] [Jais](https://huggingface.co/inceptionai/jais-13b-chat)
- [x] [Bielik-11B-v2.3](https://huggingface.co/collections/speakleash/bielik-11b-v23-66ee813238d9b526a072408a)
- [x] [RWKV-7](https://huggingface.co/collections/shoumenchougou/rwkv7-gxx-gguf)
- [x] [RWKV-6](https://github.com/BlinkDL/RWKV-LM)
- [x] [QRWKV-6](https://huggingface.co/recursal/QRWKV6-32B-Instruct-Preview-v0.1)
- [x] [GigaChat-20B-A3B](https://huggingface.co/ai-sage/GigaChat-20B-A3B-instruct)
- [X] [Trillion-7B-preview](https://huggingface.co/trillionlabs/Trillion-7B-preview)
- [x] [Ling models](https://huggingface.co/collections/inclusionAI/ling-67c51c85b34a7ea0aba94c32)
- [x] [LFM2 models](https://huggingface.co/collections/LiquidAI/lfm2-686d721927015b2ad73eaa38)
- [x] [Hunyuan models](https://huggingface.co/collections/tencent/hunyuan-dense-model-6890632cda26b19119c9c5e7)
- [x] [BailingMoeV2 (Ring/Ling 2.0) models](https://huggingface.co/collections/inclusionAI/ling-v2-68bf1dd2fc34c306c1fa6f86)

#### Multimodal

- [x] [LLaVA 1.5 models](https://huggingface.co/collections/liuhaotian/llava-15-653aac15d994e992e2677a7e), [LLaVA 1.6 models](https://huggingface.co/collections/liuhaotian/llava-16-65b9e40155f60fd046a5ccf2)
- [x] [BakLLaVA](https://huggingface.co/models?search=SkunkworksAI/Bakllava)
- [x] [Obsidian](https://huggingface.co/NousResearch/Obsidian-3B-V0.5)
- [x] [ShareGPT4V](https://huggingface.co/models?search=Lin-Chen/ShareGPT4V)
- [x] [MobileVLM 1.7B/3B models](https://huggingface.co/models?search=mobileVLM)
- [x] [Yi-VL](https://huggingface.co/models?search=Yi-VL)
- [x] [Mini CPM](https://huggingface.co/models?search=MiniCPM)
- [x] [Moondream](https://huggingface.co/vikhyatk/moondream2)
- [x] [Bunny](https://github.com/BAAI-DCAI/Bunny)
- [x] [GLM-EDGE](https://huggingface.co/models?search=glm-edge)
- [x] [Qwen2-VL](https://huggingface.co/collections/Qwen/qwen2-vl-66cee7455501d7126940800d)
- [x] [LFM2-VL](https://huggingface.co/collections/LiquidAI/lfm2-vl-68963bbc84a610f7638d5ffa)

</details>

<details>
<summary>Bindings</summary>

- Python: [ddh0/easy-llama](https://github.com/ddh0/easy-llama)
- Python: [abetlen/llama-cpp-python](https://github.com/abetlen/llama-cpp-python)
- Go: [go-skynet/go-llama.cpp](https://github.com/go-skynet/go-llama.cpp)
- Node.js: [withcatai/node-llama-cpp](https://github.com/withcatai/node-llama-cpp)
- JS/TS (llama.cpp server client): [lgrammel/modelfusion](https://modelfusion.dev/integration/model-provider/llamacpp)
- JS/TS (Programmable Prompt Engine CLI): [offline-ai/cli](https://github.com/offline-ai/cli)
- JavaScript/Wasm (works in browser): [tangledgroup/llama-cpp-wasm](https://github.com/tangledgroup/llama-cpp-wasm)
- Typescript/Wasm (nicer API, available on npm): [ngxson/wllama](https://github.com/ngxson/wllama)
- Ruby: [yoshoku/llama_cpp.rb](https://github.com/yoshoku/llama_cpp.rb)
- Rust (more features): [edgenai/llama_cpp-rs](https://github.com/edgenai/llama_cpp-rs)
- Rust (nicer API): [mdrokz/rust-llama.cpp](https://github.com/mdrokz/rust-llama.cpp)
- Rust (more direct bindings): [utilityai/llama-cpp-rs](https://github.com/utilityai/llama-cpp-rs)
- Rust (automated build from crates.io): [ShelbyJenkins/llm_client](https://github.com/ShelbyJenkins/llm_client)
- C#/.NET: [SciSharp/LLamaSharp](https://github.com/SciSharp/LLamaSharp)
- C#/VB.NET (more features - community license): [LM-Kit.NET](https://docs.lm-kit.com/lm-kit-net/index.html)
- Scala 3: [donderom/llm4s](https://github.com/donderom/llm4s)
- Clojure: [phronmophobic/llama.clj](https://github.com/phronmophobic/llama.clj)
- React Native: [mybigday/llama.rn](https://github.com/mybigday/llama.rn)
- Java: [kherud/java-llama.cpp](https://github.com/kherud/java-llama.cpp)
- Java: [QuasarByte/llama-cpp-jna](https://github.com/QuasarByte/llama-cpp-jna)
- Zig: [deins/llama.cpp.zig](https://github.com/Deins/llama.cpp.zig)
- Flutter/Dart: [netdur/llama_cpp_dart](https://github.com/netdur/llama_cpp_dart)
- Flutter: [xuegao-tzx/Fllama](https://github.com/xuegao-tzx/Fllama)
- PHP (API bindings and features built on top of llama.cpp): [distantmagic/resonance](https://github.com/distantmagic/resonance) [(more info)](https://github.com/ggml-org/llama.cpp/pull/6326)
- Guile Scheme: [guile_llama_cpp](https://savannah.nongnu.org/projects/guile-llama-cpp)
- Swift [srgtuszy/llama-cpp-swift](https://github.com/srgtuszy/llama-cpp-swift)
- Swift [ShenghaiWang/SwiftLlama](https://github.com/ShenghaiWang/SwiftLlama)
- Delphi [Embarcadero/llama-cpp-delphi](https://github.com/Embarcadero/llama-cpp-delphi)
- Go (no CGo needed): [hybridgroup/yzma](https://github.com/hybridgroup/yzma)
- Android: [llama.android](/examples/llama.android)

</details>

<details>
<summary>UIs</summary>

*(to have a project listed here, it should clearly state that it depends on `llama.cpp`)*

- [AI Sublime Text plugin](https://github.com/yaroslavyaroslav/OpenAI-sublime-text) (MIT)
- [BonzAI App](https://apps.apple.com/us/app/bonzai-your-local-ai-agent/id6752847988) (proprietary)
- [cztomsik/ava](https://github.com/cztomsik/ava) (MIT)
- [Dot](https://github.com/alexpinel/Dot) (GPL)
- [eva](https://github.com/ylsdamxssjxxdd/eva) (MIT)
- [iohub/collama](https://github.com/iohub/coLLaMA) (Apache-2.0)
- [janhq/jan](https://github.com/janhq/jan) (AGPL)
- [johnbean393/Sidekick](https://github.com/johnbean393/Sidekick) (MIT)
- [KanTV](https://github.com/zhouwg/kantv?tab=readme-ov-file) (Apache-2.0)
- [KodiBot](https://github.com/firatkiral/kodibot) (GPL)
- [llama.vim](https://github.com/ggml-org/llama.vim) (MIT)
- [LARS](https://github.com/abgulati/LARS) (AGPL)
- [Llama Assistant](https://github.com/vietanhdev/llama-assistant) (GPL)
- [LlamaLib](https://github.com/undreamai/LlamaLib) (Apache-2.0)
- [LLMFarm](https://github.com/guinmoon/LLMFarm?tab=readme-ov-file) (MIT)
- [LLMUnity](https://github.com/undreamai/LLMUnity) (MIT)
- [LMStudio](https://lmstudio.ai/) (proprietary)
- [LocalAI](https://github.com/mudler/LocalAI) (MIT)
- [LostRuins/koboldcpp](https://github.com/LostRuins/koboldcpp) (AGPL)
- [MindMac](https://mindmac.app) (proprietary)
- [MindWorkAI/AI-Studio](https://github.com/MindWorkAI/AI-Studio) (FSL-1.1-MIT)
- [Mobile-Artificial-Intelligence/maid](https://github.com/Mobile-Artificial-Intelligence/maid) (MIT)
- [Mozilla-Ocho/llamafile](https://github.com/Mozilla-Ocho/llamafile) (Apache-2.0)
- [nat/openplayground](https://github.com/nat/openplayground) (MIT)
- [nomic-ai/gpt4all](https://github.com/nomic-ai/gpt4all) (MIT)
- [ollama/ollama](https://github.com/ollama/ollama) (MIT)
- [oobabooga/text-generation-webui](https://github.com/oobabooga/text-generation-webui) (AGPL)
- [PocketPal AI](https://github.com/a-ghorbani/pocketpal-ai) (MIT)
- [psugihara/FreeChat](https://github.com/psugihara/FreeChat) (MIT)
- [ptsochantaris/emeltal](https://github.com/ptsochantaris/emeltal) (MIT)
- [pythops/tenere](https://github.com/pythops/tenere) (AGPL)
- [ramalama](https://github.com/containers/ramalama) (MIT)
- [semperai/amica](https://github.com/semperai/amica) (MIT)
- [withcatai/catai](https://github.com/withcatai/catai) (MIT)
- [Autopen](https://github.com/blackhole89/autopen) (GPL)

</details>

<details>
<summary>Tools</summary>

- [akx/ggify](https://github.com/akx/ggify) – download PyTorch models from Hugging Face Hub and convert them to GGML
- [akx/ollama-dl](https://github.com/akx/ollama-dl) – download models from the Ollama library to be used directly with llama.cpp
- [crashr/gppm](https://github.com/crashr/gppm) – launch llama.cpp instances utilizing NVIDIA Tesla P40 or P100 GPUs with reduced idle power consumption
- [gpustack/gguf-parser](https://github.com/gpustack/gguf-parser-go/tree/main/cmd/gguf-parser) - review/check the GGUF file and estimate the memory usage
- [Styled Lines](https://marketplace.unity.com/packages/tools/generative-ai/styled-lines-llama-cpp-model-292902) (proprietary licensed, async wrapper of inference part for game development in Unity3d with pre-built Mobile and Web platform wrappers and a model example)
- [unslothai/unsloth](https://github.com/unslothai/unsloth) – 🦥 exports/saves fine-tuned and trained models to GGUF (Apache-2.0)

</details>

<details>
<summary>Infrastructure</summary>

- [Paddler](https://github.com/intentee/paddler) - Open-source LLMOps platform for hosting and scaling AI in your own infrastructure
- [GPUStack](https://github.com/gpustack/gpustack) - Manage GPU clusters for running LLMs
- [llama_cpp_canister](https://github.com/onicai/llama_cpp_canister) - llama.cpp as a smart contract on the Internet Computer, using WebAssembly
- [llama-swap](https://github.com/mostlygeek/llama-swap) - transparent proxy that adds automatic model switching with llama-server
- [Kalavai](https://github.com/kalavai-net/kalavai-client) - Crowdsource end to end LLM deployment at any scale
- [llmaz](https://github.com/InftyAI/llmaz) - ☸️ Easy, advanced inference platform for large language models on Kubernetes.
- [LLMKube](https://github.com/defilantech/llmkube) - Kubernetes operator for llama.cpp with multi-GPU and Apple Silicon Metal
  support"
</details>

<details>
<summary>Games</summary>

- [Lucy's Labyrinth](https://github.com/MorganRO8/Lucys_Labyrinth) - A simple maze game where agents controlled by an AI model will try to trick you.

</details>


## Supported backends

| Backend | Target devices |
| --- | --- |
| [Metal](docs/build.md#metal-build) | Apple Silicon |
| [BLAS](docs/build.md#blas-build) | All |
| [BLIS](docs/backend/BLIS.md) | All |
| [SYCL](docs/backend/SYCL.md) | Intel and Nvidia GPU |
| [OpenVINO [In Progress]](docs/backend/OPENVINO.md) | Intel CPUs, GPUs, and NPUs |
| [MUSA](docs/build.md#musa) | Moore Threads GPU |
| [CUDA](docs/build.md#cuda) | Nvidia GPU |
| [HIP](docs/build.md#hip) | AMD GPU |
| [ZenDNN](docs/build.md#zendnn) | AMD CPU |
| [Vulkan](docs/build.md#vulkan) | GPU |
| [CANN](docs/build.md#cann) | Ascend NPU |
| [OpenCL](docs/backend/OPENCL.md) | Adreno GPU |
| [IBM zDNN](docs/backend/zDNN.md) | IBM Z & LinuxONE |
| [WebGPU [In Progress]](docs/build.md#webgpu) | All |
| [RPC](https://github.com/ggml-org/llama.cpp/tree/master/tools/rpc) | All |
| [Hexagon [In Progress]](docs/backend/snapdragon/README.md) | Snapdragon |
| [VirtGPU](docs/backend/VirtGPU.md) | VirtGPU APIR |

## Obtaining and quantizing models

The [Hugging Face](https://huggingface.co) platform hosts a [number of LLMs](https://huggingface.co/models?library=gguf&sort=trending) compatible with `llama.cpp`:

- [Trending](https://huggingface.co/models?library=gguf&sort=trending)
- [LLaMA](https://huggingface.co/models?sort=trending&search=llama+gguf)

You can either manually download the GGUF file or directly use any `llama.cpp`-compatible models from [Hugging Face](https://huggingface.co/) or other model hosting sites, by using this CLI argument: `-hf <user>/<model>[:quant]`. For example:

```sh
llama-cli -hf ggml-org/gemma-3-1b-it-GGUF
```

By default, the CLI would download from Hugging Face, you can switch to other options with the environment variable `MODEL_ENDPOINT`. The `MODEL_ENDPOINT` must point to a Hugging Face compatible API endpoint.

After downloading a model, use the CLI tools to run it locally - see below.

`llama.cpp` requires the model to be stored in the [GGUF](https://github.com/ggml-org/ggml/blob/master/docs/gguf.md) file format. Models in other data formats can be converted to GGUF using the `convert_*.py` Python scripts in this repo.

The Hugging Face platform provides a variety of online tools for converting, quantizing and hosting models with `llama.cpp`:

- Use the [GGUF-my-repo space](https://huggingface.co/spaces/ggml-org/gguf-my-repo) to convert to GGUF format and quantize model weights to smaller sizes
- Use the [GGUF-my-LoRA space](https://huggingface.co/spaces/ggml-org/gguf-my-lora) to convert LoRA adapters to GGUF format (more info: https://github.com/ggml-org/llama.cpp/discussions/10123)
- Use the [GGUF-editor space](https://huggingface.co/spaces/CISCai/gguf-editor) to edit GGUF meta data in the browser (more info: https://github.com/ggml-org/llama.cpp/discussions/9268)
- Use the [Inference Endpoints](https://ui.endpoints.huggingface.co/) to directly host `llama.cpp` in the cloud (more info: https://github.com/ggml-org/llama.cpp/discussions/9669)

To learn more about model quantization, [read this documentation](tools/quantize/README.md)

## [`llama-cli`](tools/cli)

#### A CLI tool for accessing and experimenting with most of `llama.cpp`'s functionality.

- <details open>
    <summary>Run in conversation mode</summary>

    Models with a built-in chat template will automatically activate conversation mode. If this doesn't occur, you can manually enable it by adding `-cnv` and specifying a suitable chat template with `--chat-template NAME`

    ```bash
    llama-cli -m model.gguf

    # > hi, who are you?
    # Hi there! I'm your helpful assistant! I'm an AI-powered chatbot designed to assist and provide information to users like you. I'm here to help answer your questions, provide guidance, and offer support on a wide range of topics. I'm a friendly and knowledgeable AI, and I'm always happy to help with anything you need. What's on your mind, and how can I assist you today?
    #
    # > what is 1+1?
    # Easy peasy! The answer to 1+1 is... 2!
    ```

    </details>

- <details>
    <summary>Run in conversation mode with custom chat template</summary>

    ```bash
    # use the "chatml" template (use -h to see the list of supported templates)
    llama-cli -m model.gguf -cnv --chat-template chatml

    # use a custom template
    llama-cli -m model.gguf -cnv --in-prefix 'User: ' --reverse-prompt 'User:'
    ```

    </details>

- <details>
    <summary>Constrain the output with a custom grammar</summary>

    ```bash
    llama-cli -m model.gguf -n 256 --grammar-file grammars/json.gbnf -p 'Request: schedule a call at 8pm; Command:'

    # {"appointmentTime": "8pm", "appointmentDetails": "schedule a a call"}
    ```

    The [grammars/](grammars/) folder contains a handful of sample grammars. To write your own, check out the [GBNF Guide](grammars/README.md).

    For authoring more complex JSON grammars, check out https://grammar.intrinsiclabs.ai/

    </details>


## [`llama-server`](tools/server)

#### A lightweight, [OpenAI API](https://github.com/openai/openai-openapi) compatible, HTTP server for serving LLMs.

- <details open>
    <summary>Start a local HTTP server with default configuration on port 8080</summary>

    ```bash
    llama-server -m model.gguf --port 8080

    # Basic web UI can be accessed via browser: http://localhost:8080
    # Chat completion endpoint: http://localhost:8080/v1/chat/completions
    ```

    </details>

- <details>
    <summary>Support multiple-users and parallel decoding</summary>

    ```bash
    # up to 4 concurrent requests, each with 4096 max context
    llama-server -m model.gguf -c 16384 -np 4
    ```

    </details>

- <details>
    <summary>Enable speculative decoding</summary>

    ```bash
    # the draft.gguf model should be a small variant of the target model.gguf
    llama-server -m model.gguf -md draft.gguf
    ```

    </details>

- <details>
    <summary>Serve an embedding model</summary>

    ```bash
    # use the /embedding endpoint
    llama-server -m model.gguf --embedding --pooling cls -ub 8192
    ```

    </details>

- <details>
    <summary>Serve a reranking model</summary>

    ```bash
    # use the /reranking endpoint
    llama-server -m model.gguf --reranking
    ```

    </details>

- <details>
    <summary>Constrain all outputs with a grammar</summary>

    ```bash
    # custom grammar
    llama-server -m model.gguf --grammar-file grammar.gbnf

    # JSON
    llama-server -m model.gguf --grammar-file grammars/json.gbnf
    ```

    </details>


## [`llama-perplexity`](tools/perplexity)

#### A tool for measuring the [perplexity](tools/perplexity/README.md) [^1] (and other quality metrics) of a model over a given text.

- <details open>
    <summary>Measure the perplexity over a text file</summary>

    ```bash
    llama-perplexity -m model.gguf -f file.txt

    # [1]15.2701,[2]5.4007,[3]5.3073,[4]6.2965,[5]5.8940,[6]5.6096,[7]5.7942,[8]4.9297, ...
    # Final estimate: PPL = 5.4007 +/- 0.67339
    ```

    </details>

- <details>
    <summary>Measure KL divergence</summary>

    ```bash
    # TODO
    ```

    </details>

[^1]: [https://huggingface.co/docs/transformers/perplexity](https://huggingface.co/docs/transformers/perplexity)

## [`llama-bench`](tools/llama-bench)

#### Benchmark the performance of the inference for various parameters.

- <details open>
    <summary>Run default benchmark</summary>

    ```bash
    llama-bench -m model.gguf

    # Output:
    # | model               |       size |     params | backend    | threads |          test |                  t/s |
    # | ------------------- | ---------: | ---------: | ---------- | ------: | ------------: | -------------------: |
    # | qwen2 1.5B Q4_0     | 885.97 MiB |     1.54 B | Metal,BLAS |      16 |         pp512 |      5765.41 ± 20.55 |
    # | qwen2 1.5B Q4_0     | 885.97 MiB |     1.54 B | Metal,BLAS |      16 |         tg128 |        197.71 ± 0.81 |
    #
    # build: 3e0ba0e60 (4229)
    ```

    </details>

## [`llama-simple`](examples/simple)

#### A minimal example for implementing apps with `llama.cpp`. Useful for developers.

- <details>
    <summary>Basic text completion</summary>

    ```bash
    llama-simple -m model.gguf

    # Hello my name is Kaitlyn and I am a 16 year old girl. I am a junior in high school and I am currently taking a class called "The Art of
    ```

    </details>


## Contributing

- Contributors can open PRs
- Collaborators will be invited based on contributions
- Maintainers can push to branches in the `llama.cpp` repo and merge PRs into the `master` branch
- Any help with managing issues, PRs and projects is very appreciated!
- See [good first issues](https://github.com/ggml-org/llama.cpp/issues?q=is%3Aissue+is%3Aopen+label%3A%22good+first+issue%22) for tasks suitable for first contributions
- Read the [CONTRIBUTING.md](CONTRIBUTING.md) for more information
- Make sure to read this: [Inference at the edge](https://github.com/ggml-org/llama.cpp/discussions/205)
- A bit of backstory for those who are interested: [Changelog podcast](https://changelog.com/podcast/532)

## Other documentation

- [cli](tools/cli/README.md)
- [completion](tools/completion/README.md)
- [server](tools/server/README.md)
- [GBNF grammars](grammars/README.md)

#### Development documentation

- [How to build](docs/build.md)
- [Running on Docker](docs/docker.md)
- [Build on Android](docs/android.md)
- [Multi-GPU usage](docs/multi-gpu.md)
- [Performance troubleshooting](docs/development/token_generation_performance_tips.md)
- [GGML tips & tricks](https://github.com/ggml-org/llama.cpp/wiki/GGML-Tips-&-Tricks)

#### Seminal papers and background on the models

If your issue is with model generation quality, then please at least scan the following links and papers to understand the limitations of LLaMA models. This is especially important when choosing an appropriate model size and appreciating both the significant and subtle differences between LLaMA models and ChatGPT:
- LLaMA:
    - [Introducing LLaMA: A foundational, 65-billion-parameter large language model](https://ai.facebook.com/blog/large-language-model-llama-meta-ai/)
    - [LLaMA: Open and Efficient Foundation Language Models](https://arxiv.org/abs/2302.13971)
- GPT-3
    - [Language Models are Few-Shot Learners](https://arxiv.org/abs/2005.14165)
- GPT-3.5 / InstructGPT / ChatGPT:
    - [Aligning language models to follow instructions](https://openai.com/research/instruction-following)
    - [Training language models to follow instructions with human feedback](https://arxiv.org/abs/2203.02155)

## XCFramework
The XCFramework is a precompiled version of the library for iOS, visionOS, tvOS,
and macOS. It can be used in Swift projects without the need to compile the
library from source. For example:
```swift
// swift-tools-version: 5.10
// The swift-tools-version declares the minimum version of Swift required to build this package.

import PackageDescription

let package = Package(
    name: "MyLlamaPackage",
    targets: [
        .executableTarget(
            name: "MyLlamaPackage",
            dependencies: [
                "LlamaFramework"
            ]),
        .binaryTarget(
            name: "LlamaFramework",
            url: "https://github.com/ggml-org/llama.cpp/releases/download/b5046/llama-b5046-xcframework.zip",
            checksum: "c19be78b5f00d8d29a25da41042cb7afa094cbf6280a225abe614b03b20029ab"
        )
    ]
)
```
The above example is using an intermediate build `b5046` of the library. This can be modified
to use a different version by changing the URL and checksum.

## Completions
Command-line completion is available for some environments.

#### Bash Completion
```bash
$ build/bin/llama-cli --completion-bash > ~/.llama-completion.bash
$ source ~/.llama-completion.bash
```
Optionally this can be added to your `.bashrc` or `.bash_profile` to load it
automatically. For example:
```console
$ echo "source ~/.llama-completion.bash" >> ~/.bashrc
```

## Dependencies

- [yhirose/cpp-httplib](https://github.com/yhirose/cpp-httplib) - Single-header HTTP server, used by `llama-server` - MIT license
- [stb-image](https://github.com/nothings/stb) - Single-header image format decoder, used by multimodal subsystem - Public domain
- [nlohmann/json](https://github.com/nlohmann/json) - Single-header JSON library, used by various tools/examples - MIT License
- [miniaudio.h](https://github.com/mackron/miniaudio) - Single-header audio format decoder, used by multimodal subsystem - Public domain
- [subprocess.h](https://github.com/sheredom/subprocess.h) - Single-header process launching solution for C and C++ - Public domain

---

<!-- QWEN-GFX1151-INVESTIGATION-START -->
## Qwen 3.6 35B-A3B / gfx1151 optimization and profiling record

> Status: production deployment `2d3f15e`; ROCm 7.2.2; Qwen port 8001; full public snapshot: https://gist.github.com/AyaSakura-comp/9ea877b7256f6f8344dd079aa833055f
>
> This repository carries a hardware-specific RDNA 3.5 investigation. The record below intentionally includes failed experiments, correctness gates, raw evidence locations, UMA caveats, deployment/rollback details, and real Pi Agent measurements so performance claims remain reproducible.

### Investigation map

1. **MMQ kernel investigation** — profiling, ROCm A/B, Q4_K/Q5_K tile selection, numeric correctness, deployment, and sparse MTP logits.
2. **UMA memory investigation** — mmap/page cache/GTT/HIP copies, direct I/O, prompt-logits traffic, target→draft bounce, and recurrent decode copies.
3. **Latest real Pi benchmarks** — controlled end-to-end latency, merge-sort decode throughput, output-quality caveat, and reusable benchmark skill.

### Qwen 3.6 35B-A3B on gfx1151：MMQ Prefill Optimization Investigation

#### Executive summary

Qwen 的長 prompt latency 主要不是 HTTP、CPU、取樣或 MTP，而是 GPU prefill 中的量化矩陣乘法。gfx1151 在 llama.cpp 中屬於 **RDNA 3.5**；Q4_K/Q5_K 的 128-column MMQ tile 造成極高 VGPR 壓力，其中 Q4_K 甚至 spill。將 **RDNA 3.5 上 Q4_K/Q5_K 的最大 MMQ tile width 從 128 限制為 64**，不改變量化格式、權重、數學公式或其他 GPU/type，即可提高 occupancy 並消除 Q4_K spill。

最終結果：

- Exact 20K cold prefill，相對 fresh ROCm 7.2.2 baseline：**+17.97% TPS**、**-15.23% time**。
- Exact 20K interleaved production A/B：**+21.46% TPS**、**-17.67% time**。
- 真實 Pi Agent `hello`（完整 tools/skills prompt，28,028 tokens）：**+24.59% prompt TPS**，prompt/TTFT 主耗時 **-19.74%**。
- HIP backend correctness、64-token model output、token IDs 與 Top-5 logprobs 全部一致。
- 後續 profiling 將 RDNA 3.5 上 Q4_K 改用 32-column maximum、Q5_K 保持 64-column；Q4_K profiled kernel time 降低約 17%，但最終與 64/64 deployment 的 interleaved end-to-end A/B 只有 **+0.10%**，屬測量噪音。完整 backend 與 64-token numeric comparison 仍完全一致。

---

#### 1. Workload observation

##### 1.1 Qwen 的 workload 結構

測試模型：

- Target：`Qwen3.6-35B-A3B-UD-Q4_K_M.gguf`
- llama.cpp MTP server
- Flash Attention enabled
- Single slot，260K context
- gfx1151 / ROCm 7.2.2

對長 prompt 而言，工作可分為：

1. Prompt tokenize 與 HTTP request。
2. Prefill：大量 token 一次通過 transformer。
3. Decode：逐 token 生成，並由 MTP speculative decoding 加速。

20K exact prompt profile 顯示：

- GPU kernel dispatch：**98.89%**。
- Quantized matmul：**51.11%**。
- Flash Attention：**17.83%**。
- rocBLAS/Tensile GEMM：**9.85%**。
- 其中量化 MMQ：
  - Q4_K：**26.35%**。
  - Q5_K：**12.96%**。
  - Q8_0：**10.14%**。

因此主瓶頸是 GPU 上的 quantized matrix multiplication，不是 CPU、API、host/device transfer、sampling 或 MTP。

##### 1.2 為什麼 MoE 讓 Q4_K/Q5_K 特別重要

Qwen 35B-A3B 是 MoE workload。每個 token 只使用部分 experts，但 prefill 會反覆 dispatch expert weight matmul。Shape trace 中最熱的 MMQ dispatch 為：

- Q4_K，`mmq_x=128`：3,198 calls，約 **6,784.5 ms**。
- Q5_K，`mmq_x=128`：1,482 calls，約 **3,336.2 ms**。

兩者合計已經是非常大的 prefill wall-time component，因此即使不改 attention 或 GEMM，只改善這兩類 kernel，也能明顯改變整體 TTFT。

##### 1.3 Cache 對 benchmark 的影響

llama-server 會保存 KV/prompt checkpoints。`/new`、Pi `--no-session` 或清除 Pi conversation 並不等於清除 server cache。

所以 cold-prefill A/B 必須：

- restart server，或
- request 明確使用 `cache_prompt=false`。

本次 isolated A/B 每輪重啟 server 並等待穩定 idle；Pi `hello` A/B 也每次重啟 Qwen，因此比較的是第一輪真實冷啟動 TTFT，而不是 prompt-cache restore latency。

---

#### 2. Profiling investigation

##### 2.1 先排除 ROCm runtime 升級

使用相同 source、model、prompt、server args、power profile 與 cache 條件，比較 ROCm 7.2.2 和 7.14：

- ROCm 7.14 prefill regression：大約 **3.3–3.6%**。
- Prefill energy：大約增加 **2.05%**。

所以 production 保留 ROCm 7.2.2。這也說明問題不是靠 runtime 升級即可解決。

##### 2.2 正確辨識 GPU architecture

gfx1151 在這份 llama.cpp codebase 中不是 RDNA4，而是：

```cpp
GGML_CUDA_CC_IS_RDNA3_5(cc)
```

該 predicate 包含 gfx1150/gfx1151，並在 RDNA4 之前結束。早期 RDNA4-specific knobs 實際上沒有命中 gfx1151；其 sub-percent 差異只是 benchmark noise。

##### 2.3 Kernel metadata

原始 `mmq_x=128` kernel 使用 `(32, 8)` workgroup。

Q4_K：

- 256 VGPR。
- 12 bytes private scratch。
- 2 VGPR spills。

Q5_K：

- 240 VGPR。
- 無 spill，但 register pressure 仍非常高。

高 VGPR 使用會降低一個 CU 同時 resident 的 waves/workgroups；Q4_K spill 又增加 private-memory traffic。這使 GPU 即使顯示接近 100% busy，也不代表執行效率高：它可能是被 register occupancy 和 spill 限制。

##### 2.4 被否決的方向

以下調整沒有穩定、可重現的 end-to-end 收益：

- RDNA4 `nwarps=4`：實際未命中 gfx1151。
- Global tile 64：範圍太廣，約 0.5%，屬 noise。
- RDNA4 Stream-K：實際未命中，約 0.16%。
- RDNA3.5 launch bounds：未實質改善 VGPR/spill 或整體效能。
- RDNA3.5 `nwarps=16`：與現有 MMA tile layout 不相容，compile-time assertion 顯示 layout 固定需要八個 warps。
- 對 Q8_0 也套用 tile 64：相對 Q4_K/Q5_K-only 略微 regression，因此移除。

這些結果讓 optimization 收斂到「只改真正有資源壓力且佔比最高的 Q4_K/Q5_K」。

---

#### 3. Optimization

最終 patch 位於：

```text
ggml/src/ggml-cuda/mmq.cuh
```

內容：

```cpp
// Q4_K/Q5_K with mmq_x=128 use 240-256 VGPRs per wave on RDNA 3.5; Q4_K also spills.
// The 64-column tile removes the Q4_K spill and increases occupancy for both hot MoE kernels.
const bool rdna35_high_vgpr = type == GGML_TYPE_Q4_K || type == GGML_TYPE_Q5_K;
const int mmq_x_max = GGML_CUDA_CC_IS_RDNA3_5(cc) && rdna35_high_vgpr ? 64 : get_mmq_x_max_host(cc);
```

Commit：

```text
8a10ea9 [verified] hip: tune K-quant MMQ tiles for RDNA 3.5
```

##### 3.1 它實際改了什麼

沒有修改 MMQ kernel 的數學運算，只修改 host-side dispatch 上限：

- RDNA 3.5 + Q4_K/Q5_K：最多選 `mmq_x=64` template。
- 其他 quant types：維持原 dispatch。
- NVIDIA、CDNA、RDNA1/2/3.0、RDNA4：維持原 dispatch。

128-column output 現在由較多個 64-column tiles 完成。雖然 tile/dispatch 數量可能增加，但每個 workgroup 的 register footprint 顯著降低，因此能讓更多 waves resident，並避免 Q4_K spill；收益遠大於額外 tile overhead。

##### 3.2 Resource 變化

Q4_K tile 64 metadata：

- VGPR：`256 → 156–160`。
- Private scratch：`12 → 0 bytes`。
- VGPR spills：`2 → 0`。

Q4_K 熱 kernel median：

- `2.063 ms → 約 1.241 ms`
- 約 **40% kernel-level improvement**。

Q4_K-only 已將 20K prefill 提升到約 897.87 TPS；加入 Q5_K selective cap 後提升到約 941.29 TPS，證明 Q5_K 是第二個重要 hotspot。加入 Q8_0 則略降至約 939.05 TPS，因此最終只保留 Q4_K/Q5_K。

---

#### 4. Correctness gates

##### 4.1 HIP backend vs CPU

Optimized build 通過：

- `MUL_MAT`：**1094 / 1094**。
- `MUL_MAT_ID`：**764 / 764**。
- ROCm backend groups：`OK`。

測試包含一般 MMQ、routed MoE，以及跨 tile 的 shape（包括 64/129-column cases）。

##### 4.2 Model-level numerical comparison

20K prompt、64 generated tokens：

- Generated content：一致。
- Reasoning content：一致。
- 64 個 token IDs：一致。
- 每個 token Top-5 logprobs：完全一致。
- 最大 selected-token logprob difference：**0.0**。

因此目前 evidence 顯示它只是等價的 dispatch tiling，不是近似或精度交換。

##### 4.3 Independent review

獨立 Codex review 結果：

- 無 logic error。
- 無 security concern。
- Architecture predicate 正確包含 gfx1151。
- Template dispatch 安全。
- 原本 128-column Q8_1 temporary allocation 在改用 64-column kernel 後仍是保守且安全的容量。

尚未驗證的硬體範圍是 gfx1150；predicate 有意涵蓋整個 RDNA 3.5，但本機只有 gfx1151。

---

#### 5. End-to-end results

##### 5.1 Fresh build 與 patch effect 必須分開

舊 production binary：

- 773.26 TPS，25.865 s。

相同 ROCm 7.2.2 的 fresh rebuild：

- 797.49 TPS，25.079 s。
- 相對舊 binary 已快 **3.13%**。

因此不能把 production-to-optimized 的全部差異都算在四行 patch 上。最保守的 patch comparison 是 fresh baseline 對 optimized：

- 940.81 TPS，21.258 s。
- TPS：**+17.97%**。
- Time：**-15.23%**。

相對舊 production binary 則是：

- TPS：**+21.67%**。
- Time：**-17.81%**。

##### 5.2 Interleaved exact-20K A/B

為排除 thermal、clock 和 run-order bias，使用：

```text
old → optimized → optimized → old → old → optimized
```

結果：

- Old：768.90 / 769.34 / 768.78 TPS；median **768.90**。
- Optimized：933.88 / 933.11 / 933.87 TPS；median **933.87**。
- TPS：**+21.46%**。
- Prefill time：`26.011 s → 21.416 s`，**-17.67%**。

每組內部變異極小，遠小於 optimization gain。

##### 5.3 Pi Agent real workload

執行：

```bash
pi --offline \
  --provider local-llama \
  --model qwen3.6-35b-q4 \
  --no-session \
  --mode json \
  -p hello
```

工具與完整 skills/system prompt 都保留。雖然使用者只輸入 `hello`，實際 server prompt 是 **28,028 tokens**，這正是 Pi Agent 第一輪 latency 很高的原因。

Cold-start 三次結果：

- Old：640.46 / 642.19 / 641.49 TPS；median **641.49**。
- Optimized：799.76 / 799.26 / 798.91 TPS；median **799.26**。
- Prompt TPS：**+24.59%**。
- Prompt eval：`43.692 s → 35.068 s`，**-19.74%**。
- Pi CLI median wall time：`45.21 s → 36.40 s`，約 **-19.49%**。

Wall time 含 stochastic decode，輸出長度每次不同，所以 prompt eval/TTFT component 是更可靠的指標。實際效果可解讀為：Pi 第一輪冷啟動約少等 **8.6 秒**。

##### 5.4 Decode 與 MTP

此 patch 針對 prefill MMQ tile。Decode 階段通常是單 token、小矩陣 workload，主要走 MMVQ/不同 dispatch path，不會使用本次修改的高 `mmq_x` prefill tile，因此理論上不應期待 decode 加速。

為確認實際效果，另外讓 Pi Agent 回答一題複雜 GPU/MoE 技術問題，要求約 900–1100 English words；old 與 optimized 各跑兩次。實際模型因 reasoning 與格式產生 2,188–2,871 output tokens：

- Old decode：`48.54 / 46.52 TPS`，mean **47.53 TPS**。
- Optimized decode：`47.13 / 45.53 TPS`，mean **46.33 TPS**。
- 表面差異：**-2.52%**。
- Old MTP acceptance mean：**95.96%**。
- Optimized MTP acceptance mean：**94.71%**。

這不是 optimization 導致 decode regression 的可靠證據：四次生成的內容、token 數與 MTP acceptance 不同，optimized runs 的 acceptance 平均低約 1.25 percentage points，而 patch 沒有修改 decode/MMVQ 或 MTP。合理結論是 **decode 沒有可測得的 improvement，約維持原速，觀察到的 -2.5% 落在 stochastic generation/MTP variance 內**。

同一組測試仍再次確認 prefill：

- Old prompt TPS mean：**641.18**。
- Optimized prompt TPS mean：**799.67**。
- Prefill TPS：**+24.72%**。

完整結果：`/tmp/pi-complex-decode-ab-final/summary.json`。

---

#### 6. Energy interpretation

因 optimized prefill 的平均功率沒有大幅上升，而執行時間下降，rough estimate：

- Prefill energy：約 `3503 J → 2952 J`。
- 約 **-15.8%**。

這是由 power samples × elapsed time 得到的 workload estimate，不是外接 calibrated power meter，因此適合看相對變化，不應當成絕對整機能耗認證。

另外，Qwen 與 Gemma 兩個 ROCm runtime 同時 idle 時仍有另一個獨立問題：GPU 100%、約 32–35 W。這是 dual-context/queue/MES 層面的現象，與本次 MMQ prefill optimization 無關。

---

#### 7. Deployment and rollback

Production deployment：

```text
/home/chihmin/llama-mtp-deploy/gfx1151-q4-32-q5-64-5c39e48/bin/llama-server
```

目前 commit：

```text
5c39e48 [verified] hip: lower RDNA 3.5 Q4_K MMQ tile cap
```

Deployment ELF RUNPATH 已改為 `$ORIGIN:/opt/rocm-7.2.2/lib`，確認 runtime 實際載入 deployment 目錄內的 `libggml-hip.so.0.11.1`，不依賴 mutable build worktree。

Systemd drop-in：

```text
/etc/systemd/system/qwen-mtp.service.d/optimized.conf
```

立即 rollback 至前一版可還原 `/etc/systemd/system/qwen-mtp.service.d/optimized.conf.pre-5c39e48`；完全移除 drop-in、`systemctl daemon-reload`、restart `qwen-mtp.service` 則會回到原本 `/home/chihmin/llama-mtp/build/bin/llama-server`。

目前 Qwen optimized 與 Gemma 皆 health OK。依使用者要求，系統 power profile 已設為 `performance`；這是全域平台設定，會同時套用兩個服務。

#### 8. Evidence locations

- Final grouped benchmark：`/tmp/qwen-opt-final/`
- Interleaved A/B：`/tmp/qwen-alternating-ab-final/summary.json`
- Pi `hello` A/B：`/tmp/pi-hello-ab/summary.json`
- Pi complex-answer decode A/B：`/tmp/pi-complex-decode-ab-final/summary.json`
- 64-token numeric compare：`/tmp/qwen-numeric64-old-vs-opt.json`
- Original shape trace：`/tmp/qwen-mmq-shapes-20260724-164843/`
- Baseline profile：`/tmp/qwen-prefill-profile-20260724-145331/`
- Q4 tile-64 profile：`/tmp/qwen-q4-tile64-profile-20260724-171842/`
- ROCm A/B：`/tmp/qwen-rocm-ab-20260724-153155/`
- Follow-up final-build profile：`/tmp/qwen-q45-final-profile-20260724-191751/`
- Q4/Q5 tile-32 profile：`/tmp/qwen-q45-tile32-profile-20260724-192731/`
- Q4=32/Q5=64 first-token numeric compare：`/tmp/qwen-numeric-old-vs-q4-32-q5-64.json`
- Q4=32/Q5=64 64-token numeric compare：`/tmp/qwen-numeric64-old-vs-q4-32-q5-64.json`
- Q4=32/Q5=64 versus previous 64/64 interleaved A/B：`/tmp/qwen-q45-vs-q4-32-ab-final/summary.json`

---

#### 9. Follow-up：Q4_K 32-column candidate 與新瓶頸

##### 9.1 最終部署版本重新 profiling

已部署的 Q4_K/Q5_K 64-column 版本重新 profile 後，主要 GPU kernel time 分布為：

1. Q4_K MMQ-64：**21.37%**。
2. Flash Attention：**18.85%**。
3. Q8_0 MMQ-128：**14.02%**。
4. Q5_K MMQ-64：**12.39%**。
5. Gated Delta Net：**7.23%**。

因此 Q4_K 仍是單一最大 kernel family，Flash Attention 已非常接近第一名。

##### 9.2 Q4_K tile 32 實驗

將 Q4_K/Q5_K 都改為 32-column 時：

- Q4_K kernel total：`3.796 s → 3.151 s`，約 **-17%**。
- Q5_K kernel total：`2.202 s → 2.206 s`，沒有改善。
- Exact 20K cold-prefill median：`940.81 → 973.02 TPS`，約 **+3.42%**。
- Prefill time：`21.258 s → 20.553 s`。

因此採用並部署的 selective configuration 是：

```text
RDNA 3.5 Q4_K = 32-column tile
RDNA 3.5 Q5_K = 64-column tile
```

早期非交錯三次 exact 20K 結果為 `963.74 / 972.26 / 972.57 TPS`，median **972.26 TPS**；但這個約 +3.34% 的表面增益沒有通過最終 interleaved A/B。相同 performance profile 下，前一版 64/64 與新 32/64 按 `old → new → new → old → old → new` 交錯：

- 64/64：`962.51 / 963.73 / 965.64 TPS`，median **963.73**。
- 32/64：`964.66 / 964.68 / 965.19 TPS`，median **964.68**。
- Median gain：**+0.10%**。
- Prefill time reduction：**0.10%**。

因此可確認 Q4_K kernel 本身變快，但目前不能宣稱 end-to-end prefill 有可測得的額外提升。

完整 backend correctness 已通過：

- `MUL_MAT`：1094/1094。
- `MUL_MAT_ID`：764/764。

20K first-token numeric A/B 也完全一致：

- Token ID：`8160`。
- Token：`Here`。
- Selected-token logprob：兩邊都是 `-0.01808076538145542`，delta **0.0**。
- Top-10 token ordering 與 logprobs：完全一致。
- Content、reasoning、usage：完全一致。

64-token model comparison亦通過：content、reasoning、全部 64 個 token IDs、每 token Top-5 logprobs 完全相同，maximum selected-token logprob delta 為 **0.0**。經獨立 Codex review 無 logic/security error 後，已提交為 `5c39e48` 並依使用者要求部署。

##### 9.3 Flash Attention 成為下一個主要瓶頸

Q4_K 改成 tile 32 後，主要占比約為：

1. Flash Attention：**19.5%**。
2. Q4_K：**18.4%**。
3. Q8_0：**14.5%**。
4. Q5_K：**12.9%**。
5. Gated Delta Net：**7.5%**。

手動 Flash Attention 調整均 regression：

- 256 → 128 threads：median **917.99 TPS**，相對 selective baseline 約 **-5.6%**。
- FA batch 64 → 32：median **927.88 TPS**，約 **-4.6%**。

兩項 source config 都已還原。官方 `GGML_HIP_ROCWMMA_FATTN=ON` 方向目前被 ROCm 7.2.2 rocWMMA 2.2 與現有 clang/HIP vector implementation 的編譯相容性問題阻擋；build option 已恢復為 OFF。

#### 10. MTP prefill sparse logits（`2d3f15e`）

UMA profiling發現 MTP prompt會為每個位置計算並D2H複製完整FP32 vocabulary logits，但streaming hook實際只使用 `h_pre_norm`。新版本保留完整hidden rows與recurrent/output-all語意，僅在**單一sequence的prompt-prefill batch**對target LM head投影最後一列；decode、多slot及coupled-sequence batch維持完整logits。

Interleaved exact-20K、`performance`結果：

- `5c39e48`：median **964.75 TPS**，**20.731 s**。
- `2d3f15e`：median **1026.54 TPS**，**19.483 s**。
- 提升：**+6.40% TPS**，**-6.02% prefill time**。

驗證：

- Instrumented 20K prefill沒有≥400 MiB D2H copy。
- 64-token IDs與完整message一致。
- MTP acceptance維持約98.1%；decode使用未修改的full-logits path。
- `-np 2` concurrent 20K/16-token測試取得各自正確且不同的logprob。
- 1-token prompt edge case通過。
- Sparse validity tracking禁止getter暴露未寫入的stale logits rows。
- Qwen35/Qwen35MoE architecture tests與`MUL_MAT` 1094/1094通過。
- Independent Codex review：PASS，無blocking issues。

Live deployment：

```text
/home/chihmin/llama-mtp-deploy/gfx1151-q4-32-q5-64-mtp-sparse-2d3f15e/bin/llama-server
```

Rollback drop-in：

```text
/etc/systemd/system/qwen-mtp.service.d/optimized.conf.pre-2d3f15e
```

### Qwen 3.6 35B-A3B on gfx1151：UMA Memory Profiling

#### Executive summary

這次 profiling 同時觀察 llama.cpp model loading、Linux `mmap`/page cache、AMDGPU GTT、HIP allocation/copy、完整 Pi Agent prefill，以及長輸出 MTP decode。

結論：

1. **目前不是 zero-copy model loading。** GPU weights 先由 GGUF file-backed pages/讀取緩衝區進入 CPU address space，再透過 `hipMemcpyAsync(H2D)` 複製到獨立 `hipMalloc` GTT allocation。冷啟動共傳送 **21.434 GiB H2D**。
2. Production 預設 `mmap=true, direct_io=false` 在冷載入期間造成明顯的暫時雙份 residency：process file-backed PSS 峰值 **21.190 GiB**，同時 process GTT 已成長到 **28.912 GiB**。載入完成後 full-file mapping 會解除，只留下約 515 MiB CPU-mapped tensor range，因此不是永久 21 GiB duplication。
3. `--no-mmap` **不會消除 model→GTT copy**：仍是相同 **21.434 GiB H2D**，而且從 1,106 個 H2D calls 增加到 21,923 個；冷載入時間也沒有改善。
4. `--direct-io` 同樣不會 zero-copy，但會使用 Linux `O_DIRECT`、自動停用 mmap，避免把整個 GGUF 放進 page cache。冷載入 A/B 中，load-to-health 從 **10.656 s 降至 6.042 s**，process file-backed PSS 峰值從 **21.190 GiB 降至 0.098 GiB**。它仍保留相同 21.434 GiB H2D copy。
5. 真正最可疑的 runtime copy 不是 mmap，而是 custom MTP prefill：每個 512-token ubatch 都 D2H 複製完整 FP32 vocabulary logits。55 次 × 508,559,360 bytes = **26.05 GiB**。MTP implementation 實際只讀 `h_pre_norm`，沒有使用這些 prompt-position logits。
6. 完整 prefill 的 H2D/D2H physical copy engine 時間約 **418.8 ms / 31.89 s（約 1.31%）**。它不是目前最大的 latency bottleneck，但移除無用 logits copy 可減少約 26 GiB traffic、約 2 GiB host output allocation，並降低 UMA memory pressure。
7. Decode 有約 **198.18 GiB logical D2D traffic**，主要是 2 MiB recurrent/checkpoint state copies。rocprof 顯示 `__amd_rocclr_copyBuffer` 在 decode 有 208,567 calls、共 **1.421 s**。這不是 mmap 或 discrete-GPU host bounce，而是 Qwen hybrid recurrent state 與 speculative/checkpoint save/restore。

沒有修改 production code 或啟動參數；Qwen、Gemma 已恢復 active，power profile 保持 `performance`。

---

#### 1. Test setup

##### Hardware/runtime

- AMD Ryzen AI MAX+ 395 / Radeon 8060S
- GPU target：gfx1151 / RDNA 3.5
- Physical RAM：約 128 GB
- ROCm：7.2.2
- GPU agent：40 CU、wave32
- Process GPU allocations由 DRM fdinfo 的 `drm-memory-gtt` / `drm-resident-gtt` 驗證

##### Production binary

```text
/home/chihmin/llama-mtp-deploy/gfx1151-q4-32-q5-64-5c39e48/bin/llama-server
```

##### Workload

使用完整 Pi Agent tools/skills system prompt，cold server，complex technical question：

- Prompt：28,295 tokens
- Prefill：31.780 s，890.34 TPS
- Decode：4,343 tokens，91.085 s，47.68 server eval TPS
- MTP enabled，target 與 draft context 都在同一 gfx1151 GPU

所有 attribution profile 都停止 systemd Qwen/Gemma，僅啟動一個 instrumented standalone Qwen，避免另一個 ROCm runtime 干擾記憶體統計。測試後恢復兩個 production services。

---

#### 2. Instrumentation

使用四層 evidence：

1. **rocprofiler-sdk 1.1 / ROCm 7.2.2**
   - kernel trace
   - memory-copy trace
   - memory-allocation trace
   - scratch-memory trace
   - HIP runtime trace
2. **HIP LD_PRELOAD shim**
   - intercept `hipMemcpyAsync`, `hipMemcpy`, `hipMalloc`, `hipHostMalloc`, free calls
   - 記錄 bytes、direction、source/destination address、timestamp
   - 217,231 events
3. **Linux process/system sampling，每 250 ms 或 load A/B 每 50 ms**
   - `/proc/<pid>/smaps_rollup`
   - `/proc/<pid>/numa_maps`
   - `/proc/<pid>/maps`
   - `/proc/<pid>/fdinfo/*` AMDGPU memory accounting
   - `/sys/class/drm/card1/device/mem_info_gtt_used`
   - minor/major faults
4. **Cold page-cache A/B**
   - 每輪使用 `POSIX_FADV_DONTNEED`
   - `mincore()` 確認主 GGUF 在每輪前為 0 resident pages
   - 比較 mmap、no-mmap、direct-io

rocprof 全 trace 對 decode 有很高 instrumentation overhead，因此 latency 使用無 rocprof 的 LD_PRELOAD run；rocprof 主要用於確認 physical copy duration 與 kernel dispatch。

---

#### 3. Persistent memory layout

Server load 完成、尚未收到 request 時：

- DRM process GTT：**29.606 GiB**
- Process RSS：**約 2.01 GiB**
- Process anonymous PSS：**約 1.39 GiB**
- Process file PSS：**約 0.61 GiB**

llama-server log 的主要 backend buffers：

- ROCm0 model：21,087.70 MiB
- CPU_Mapped model：515.31 MiB（mmap mode）
- Target KV：5,080 MiB
- Draft KV：508 MiB
- Recurrent state：62.81 MiB
- Target ROCm0 compute：798.02 MiB
- Draft ROCm0 compute：794.02 MiB
- Two ROCm_Host compute buffers：約 516 MiB each

DRM fdinfo 顯示這些大 allocations 位於 **GTT**，不是 512 MiB visible VRAM carve-out：

```text
drm-memory-gtt: 30316052 KiB
drm-memory-vram: 約 12–19 MiB
```

在 UMA 上 GTT 最終仍是 system DRAM，但它是與 file page cache/CPU staging 分開的 allocation；「同一組實體 RAM」不代表 API 自動 zero-copy。

---

#### 4. mmap behavior

##### 4.1 Production mmap load

Log：

```text
load_tensors: loading model tensors (mmap = true, direct_io = false)
CPU_Mapped model buffer size = 515.31 MiB
ROCm0 model buffer size = 21087.70 MiB
```

冷載入期間：

- Process RSS peak：**21.363 GiB**
- File-backed PSS peak：**21.190 GiB**
- Anonymous PSS peak：1.357 GiB
- Process GTT peak/load-complete：28.912 GiB
- H2D：1,106 calls、**21.434 GiB**
- Load-to-health：**10.656 s**

這證明 loading window 中 GGUF file pages 與 GTT destination 同時 resident。載入後 `/proc/<pid>/maps` 只保留約 515 MiB model range及 metadata page，不再保留 full 21 GiB mapping。主 GGUF 在 server shutdown 後 `mincore()` 顯示約 1.107 GiB resident；這些是 reclaimable file-cache pages，不是 locked GTT。

##### 4.2 `--no-mmap`

- Process RSS peak：1.967 GiB
- File PSS peak：0.098 GiB
- Anonymous PSS peak：1.865 GiB
- H2D bytes：仍為 **21.434 GiB**
- H2D calls：**21,923**
- Load-to-health：10.148 s
- Server log：CPU model buffer變成 `ROCm_Host 515.31 MiB`

`--no-mmap` 只改變 CPU-side loading/staging；它沒有讓 ROCm backend直接使用 GGUF pages，也沒有消除 GPU allocation/copy。

##### 4.3 `--direct-io`

Source implementation 在 Linux 使用：

```text
open(..., O_RDONLY | O_DIRECT)
```

而且 direct I/O available 時會自動停用 mmap。實測：

- Load-to-health：**6.042 s**（相對 mmap -43.3%）
- Process RSS peak：1.967 GiB
- File PSS peak：0.098 GiB
- Main GGUF resident after run：0.010 GiB
- H2D：1,361 calls、**21.434 GiB**
- GTT allocation：完全相同

因此 direct I/O 是「避免 page-cache duplication」而不是 zero-copy。Implementation 仍會先 read 到 aligned CPU buffer、`memcpy` 到 destination staging，再由 HIP copy 到 GTT。它適合作為後續獨立 deployment A/B 候選，但不能宣稱會改善已載入模型的 prefill/decode。

---

#### 5. Whole-workload copy traffic

HIP byte-count shim 的 logical API traffic：

| Phase | H2D | D2H | D2D | Total |
|---|---:|---:|---:|---:|
| Model startup | 21.434 GiB | 0.003 GiB | 0.117 GiB | 21.554 GiB |
| Prefill | 3.690 GiB | 27.133 GiB | 3.881 GiB | 34.703 GiB |
| Decode | 1.738 GiB | 8.354 GiB | 198.179 GiB | 208.271 GiB |

Loaded idle 10 seconds沒有任何 HIP memcpy。

rocprof physical H2D/D2H operations：

| Phase | Calls | Total duration |
|---|---:|---:|
| Startup H2D | 1,203 | 303.86 ms |
| Prefill H2D | 294 | 68.41 ms |
| Prefill D2H | 487 | 350.43 ms |
| Decode H2D | 10,842 | 70.25 ms |
| Decode D2H | 7,722 | 126.10 ms |

Prefill physical copy total約 418.84 ms，占 31.89 s 的 1.31%。Copy engines/queues可能與 kernels overlap，所以不能直接把全部 duration 視為可回收 wall time，但它提供上限與 bandwidth-pressure evidence。

---

#### 6. High-confidence unnecessary copy：full prompt logits

Prefill 中最突出的是：

```text
55 calls × 508,559,360 bytes D2H = 26.05 GiB
```

單次大小恰好等於：

```text
512 ubatch tokens × 248,320 vocabulary × 4-byte FP32 = 508,559,360 bytes
```

也就是每個 target prefill ubatch 把**所有 token 的完整 FP32 logits**複製回 host。

Source data flow：

1. `server_slot::need_embd()` 對 MTP 回傳 true：`tools/server/server-context.cpp`。
2. Prompt filling 將每個 MTP prompt position標記為 output。
3. `llama_set_embeddings(ctx_tgt, slot_batched->need_embd())` 同時打開一般 embeddings output path。
4. `llama_context::decode()` 對每個 output row執行 `ggml_backend_tensor_get_async(... n_outputs*n_vocab*sizeof(float))`。
5. 但 `common_speculative_state_mtp::process()` 實際讀的是 `llama_get_embeddings_pre_norm()` / `h_pre_norm`，沒有使用 prefill position logits。

因此 full logits D2H 是目前最明確的無用 traffic。預期修正方向是分離：

- MTP 所需的 all-row `h_pre_norm`
- 普通 embeddings
- raw logits extraction

不能簡單把所有 output flags 關掉，否則 MTP draft context會失去每個 prompt position的 hidden state。應新增精確 regression test，確認只抑制 MTP-only prefill raw logits，同時保留 h_pre_norm、draft state、token IDs、logprobs 與 acceptance。

此外 prefill 還有約：

- 110 × 4 MiB D2H hidden-state copies：約 0.43 GiB
- 167 × 4 MiB H2D copies：約 0.65 GiB

這是 target `h_pre_norm` 先回 host，再餵給 draft context。對同一 UMA GPU，長期更好的設計是 device-resident target→draft handoff；目前 public MTP code使用 CPU pointer和 `std::memcpy`，仍沿用 discrete-GPU 式 staging。

---

#### 7. Decode D2D traffic

Decode logical D2D：

- 139,440 `hipMemcpyAsync` calls
- **198.179 GiB**
- 主要 sizes：2 MiB與96 KiB

rocprof kernel trace：

```text
__amd_rocclr_copyBuffer
208,567 decode calls
1.421 s total
```

Address routing顯示大量 copies在 target/draft compute buffers、62.8 MiB recurrent state buffer與on-device checkpoint buffers之間。Source中的 `llama_io_write_device` / `llama_io_read_device` 使用 `ggml_backend_tensor_copy` 保存與還原 state；server也明確註記 standard MTP目前總是 re-evaluate draft tokens。

這些 copies：

- 不是 model mmap造成。
- 不是 GTT↔dedicated VRAM transfer；兩端都在 UMA/GTT。
- 仍會消耗實際 DRAM bandwidth並建立第二份 recurrent/checkpoint state。
- 對 speculative rollback、hybrid recurrent state與prompt checkpoints可能是必要的，不能直接刪除。

下一步若優化 decode，應先把 copies按「MTP draft re-eval」「recurrent state」「prompt checkpoint」分開，再測試 `TAG_SPEC_AVOID_DRAFT_REEVAL` 所描述的 TODO，而不是套用通用 zero-copy 假設。

---

#### 8. Workload memory growth

Loaded idle → workload completed：

- Process anonymous PSS：約 1.39 GiB → 3.98–4.64 GiB
- Process GTT：29.606 GiB → 約 32.326 GiB
- Workload新增：
  - 約 2.6–3.2 GiB anonymous host/pinned memory
  - 約 2.72 GiB GTT
- Major faults：0–1
- Minor faults：約 670k–833k

主要新增 allocations包括約 1.97 GiB `hipHostMalloc` output/staging buffer與兩組約 1.02 GiB GPU buffers。這與 MTP all-position outputs及長 context compute/output reservation一致。

---

#### 9. Recommendations

##### Priority 1：修正 MTP prefill raw-logits extraction（已完成）

Commit `2d3f15e [verified] mtp: skip unused Qwen prompt logits` 已部署。它保留完整 `h_pre_norm` 與 recurrent/output-all 語意，但在**單一 sequence 的 prompt-prefill batch**只對最後一列執行 target LM head；speculative decode、多-slot及coupled-sequence batch維持完整 logits。

驗證結果：

- 20K interleaved median：`964.75 → 1026.54 TPS`，**+6.40%**
- Prefill time：`20.731 → 19.483 s`，**-6.02%**
- 485 MiB full-vocabulary D2H copies：`55 → 0`（原28K profile）
- 20K/64-token輸出：token IDs與message完全相同
- MTP acceptance：維持約98.1%；decode路徑未套用此優化
- `-np 2` concurrent 20K + 16-token functional test通過，兩個slot取得各自正確且不同的logprob
- 1-token prompt edge case通過

為避免未寫入rows暴露stale logits，context會追蹤sparse logits validity；`llama_get_logits_ith()`拒絕未計算row，`llama_get_logits()`只回傳有效row。

##### Priority 2：獨立驗證 `--direct-io`

Cold-load evidence非常正面：

- 10.656 s → 6.042 s
- 避免 full-file process RSS/page-cache residency

但 deployment 前仍需：

- warm restart A/B
- repeated interleaved load A/B
- exact 20K numeric與prefill確認
- 確認長期 515 MiB ROCm_Host CPU model buffer的影響

##### Priority 3：device-resident MTP hidden-state handoff

目前 target h_pre_norm走 D2H，再由 CPU staging H2D至 draft。UMA可考慮同 device buffer view/copy，至少避免 host round-trip。但這需要修改 MTP API ownership/synchronization，不是 launch flag。

##### Priority 4：decode recurrent/checkpoint copies

先標記與分類 2 MiB/96 KiB copies，再評估避免 draft re-evaluation或減少 state snapshots；不要直接移除，因為可能破壞 recurrent state rollback correctness。

---

#### 10. Evidence

- Full rocprof run：`/tmp/qwen-uma-memory-profile-20260724-233435/`
- Full byte-count workload：`/tmp/qwen-uma-copy-bytes-20260725-000131/`
- Copy summary：`/tmp/qwen-uma-copy-bytes-20260725-000131/copy_summary.json`
- Cold mmap/no-mmap/direct-io A/B：`/tmp/qwen-uma-load-ab-20260725-000829/`
- Load A/B summary：`/tmp/qwen-uma-load-ab-20260725-000829/summary.json`
- HIP byte-count shim source：`/tmp/hip_copy_trace.cpp`
- Workload profiler wrapper：`/tmp/profile_qwen_uma_full.sh`

### 11. Real Pi Agent benchmarks after sparse-logits deployment

#### 11.1 Controlled end-to-end Pi Agent A/B

The production-like comparison used the complete Pi system prompt, tool schemas, discovered skills, and chat template. Only the user message was constrained to a short no-tool response so stochastic decode would not dominate the result:

```bash
pi --offline \
  --provider local-llama \
  --model qwen3.6-35b-q4 \
  --no-session \
  --mode json \
  -p 'Do not use any tools. Reply with exactly the single word OK and nothing else. Do not explain or think aloud.'
```

Each sample stopped Gemma, selected the `performance` power profile, restarted Qwen to clear server KV/prompt cache, waited for ten consecutive idle samples, and measured both llama-server timing and `/usr/bin/time` around the complete Pi process. Variants were interleaved as `old1 → new1 → new2 → old2 → old3 → new3`.

All valid runs had exactly **28,194 input tokens**:

| Metric | `5c39e48` | `2d3f15e` | Change |
|---|---:|---:|---:|
| Prefill TPS, median | 893.09 | 944.08 | **+5.71%** |
| Prompt eval, median | 31.569 s | 29.864 s | **-5.40%** |
| Complete Pi wall time, median | 32.47 s | 30.75 s | **-5.30%** |
| Decode TPS, median | 76.31 | 76.17 | effectively unchanged |

The first unconstrained `hello` trial was discarded because one run invoked a tool and produced a second server request. The controlled rerun avoided this confound. Evidence: `/tmp/pi-sparse-logits-controlled-ab-20260725-020047/summary.json`.

#### 11.2 Merge-sort sustained decode workload

The deployed `2d3f15e` server was also exercised through real Pi Agent with a no-tool request for a typed, documented, stable Python merge sort plus explanation and five tests. The complete Pi prompt was **28,219 tokens**.

| Run | Output tokens | llama-server decode TPS | `(N-1)/eval_time` | MTP acceptance |
|---:|---:|---:|---:|---:|
| 1 | 1,186 | 60.98 | 60.93 | 95.95% |
| 2 | 1,855 | 66.33 | 66.29 | 97.83% |
| 3 | 952 | 59.89 | 59.83 | 97.99% |

- llama-server decode TPS median: **60.98 TPS**; mean: **62.40 TPS**.
- Standards-style `(N-1)/eval_time` median: **60.93 TPS**.
- Prefill median: **944.22 TPS**, **29.886 s**.
- Complete Pi wall-time median: **49.97 s**.

Decode varies with generated token path and MTP acceptance. The sparse-logits patch is prefill-only, so this workload characterizes production decode rather than claiming a decode gain.

The first generated merge-sort implementation used a stable `<=` merge, but one stability assertion expected equal items in reversed order and failed with `AssertionError`. This is a useful reminder that generation throughput does not establish generated-code correctness. Evidence: `/tmp/pi-mergesort-decode-20260725-022600/summary.json`.

#### 11.3 Reusable benchmark skill

The measurement protocol is packaged as:

```text
/home/chihmin/.pi/agent/skills/benchmark-qwen/
```

Commands:

```bash
## Controlled real-Pi first-response latency
~/.pi/agent/skills/benchmark-qwen/scripts/benchmark-pi-agent.sh latency 3

## Sustained real-Pi merge-sort decode
~/.pi/agent/skills/benchmark-qwen/scripts/benchmark-pi-agent.sh decode 3
```

The skill distinguishes llama-server prefill/decode timing from Pi wall time, reports both server `N/time` and `(N-1)/time`, rejects multi-request/tool-follow-up logs, enforces cold server KV through restart, isolates Gemma, waits for stable idle, records MTP acceptance, and verifies production restoration after completion or interruption.

<!-- QWEN-GFX1151-INVESTIGATION-END -->
