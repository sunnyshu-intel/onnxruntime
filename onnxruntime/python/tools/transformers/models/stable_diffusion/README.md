# Stable Diffusion GPU Optimization

ONNX Runtime uses the following optimizations to speed up Stable Diffusion in CUDA:

* [Flash Attention](https://arxiv.org/abs/2205.14135) for float16 precision. Flash Attention uses tiling to reduce number of GPU memory reads/writes, and improves performance with less memory for long sequence length. The kernel requires GPUs of Compute Capability >= 7.5 (like T4, A100, and RTX 2060~4090).
* [Memory Efficient Attention](https://arxiv.org/abs/2112.05682v2) for float32 precision or older GPUs (like V100). We used the fused multi-head attention kernel in CUTLASS, and the kernel was contributed by xFormers.
* Channel-last (NHWC) convolution. For NVidia GPU with Tensor Cores support, NHWC tensor layout is recommended for convolution. See [Tensor Layouts In Memory: NCHW vs NHWC](https://docs.nvidia.com/deeplearning/performance/dl-performance-convolutional/index.html#tensor-layout).
* GroupNorm for NHWC tensor layout, and SkipGroupNorm fusion which fuses GroupNorm with Add bias and residual inputs
* SkipLayerNormalization which fuses LayerNormalization with Add bias and residual inputs.
* BiasSplitGelu is a fusion of Add bias with SplitGelu activation.
* BiasAdd fuses Add bias and residual.
* Reduce Transpose nodes by graph transformation.

These optimizations are firstly carried out on CUDA EP. They may not work on other EP.

## Scripts:

| Script                                         | Description                                                                               |
| ---------------------------------------------- | ----------------------------------------------------------------------------------------- |
| [demo_txt2img_xl.py](./demo_txt2img_xl.py)     | Demo of text to image generation using Stable Diffusion XL model.                         |
| [demo_txt2img.py](./demo_txt2img.py)           | Demo of text to image generation using Stable Diffusion models except XL.                 |
| [optimize_pipeline.py](./optimize_pipeline.py) | Optimize Stable Diffusion ONNX models exported from Huggingface diffusers or optimum      |
| [benchmark.py](./benchmark.py)                 | Benchmark latency and memory of OnnxRuntime, xFormers or PyTorch 2.0 on stable diffusion. |
| [benchmark_controlnet.py](./benchmark_controlnet.py)| Benchmark latency of canny control net.                                              |

## Run demo with docker

#### Clone the onnxruntime repository
```
git clone https://github.com/microsoft/onnxruntime
cd onnxruntime
```

#### Launch NVIDIA pytorch container

Install nvidia-docker using [these instructions](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html#docker).

```
docker run --rm -it --gpus all -v $PWD:/workspace nvcr.io/nvidia/pytorch:24.04-py3 /bin/bash
```

#### Build onnxruntime from source
This step is optional. Please look at [install onnxruntime-gpu](https://onnxruntime.ai/docs/install/#python-installs) if you do not want to build from source.

```
export CUDACXX=/usr/local/cuda/bin/nvcc
git config --global --add safe.directory '*'
sh build.sh --config Release  --build_shared_lib --parallel --use_cuda --cuda_version 12.4 \
            --cuda_home /usr/local/cuda --cudnn_home /usr/lib/x86_64-linux-gnu/ --build_wheel --skip_tests \
            --use_tensorrt --tensorrt_home /usr/src/tensorrt \
            --cmake_extra_defines onnxruntime_BUILD_UNIT_TESTS=OFF \
            --cmake_extra_defines CMAKE_CUDA_ARCHITECTURES=80 \
            --allow_running_as_root
python3 -m pip install --upgrade pip
python3 -m pip install build/Linux/Release/dist/onnxruntime_gpu-*.whl --force-reinstall
```

If the GPU is not A100, change `CMAKE_CUDA_ARCHITECTURES=80` in the command line according to the GPU compute capacity (like 89 for RTX 4090, or 86 for RTX 3090).
If your machine has less than 64GB memory, replace `--parallel` by `--parallel 4 --nvcc_threads 1 ` to avoid out of memory.

#### Install required packages
First, remove older version of opencv to avoid error like `module 'cv2.dnn' has no attribute 'DictValue'`:
```
pip uninstall -y $(pip list --format=freeze | grep opencv)
rm -rf /usr/local/lib/python3.10/dist-packages/cv2/
apt-get update
DEBIAN_FRONTEND="noninteractive" apt-get install --yes python3-opencv
```

```
cd /workspace/onnxruntime/python/tools/transformers/models/stable_diffusion
python3 -m pip install -r requirements/cuda12/requirements.txt
python3 -m pip install --upgrade polygraphy onnx-graphsurgeon --extra-index-url https://pypi.ngc.nvidia.com
```

### Run Demo

You can review the usage of supported pipelines like the following:
```
python3 demo_txt2img.py --help
python3 demo_txt2img_xl.py --help
```

For example:
`--engine {ORT_CUDA,ORT_TRT,TRT}` can be used to choose different backend engines including CUDA or TensorRT execution provider of ONNX Runtime, or TensorRT.
`--work-dir WORK_DIR` can be used to load or save models under the given directory. You can download the [optimized ONNX models of Stable Diffusion XL 1.0](https://huggingface.co/tlwu/stable-diffusion-xl-1.0-onnxruntime#usage-example) to save time in running the XL demo.

#### Generate an image guided by a text prompt
```
python3 demo_txt2img.py "astronaut riding a horse on mars"
```

#### Generate an image with Stable Diffusion XL guided by a text prompt
```
python3 demo_txt2img_xl.py "starry night over Golden Gate Bridge by van gogh"

python3 demo_txt2img_xl.py --enable-refiner "starry night over Golden Gate Bridge by van gogh"
```

If you do not provide prompt, the script will generate different image sizes for a list of prompts for demonstration.

### Generate an image guided by a text prompt using LCM LoRA
```
python3 demo_txt2img_xl.py --scheduler LCM --lora-weights latent-consistency/lcm-lora-sdxl --denoising-steps 4 "Self-portrait oil painting, a beautiful cyborg with golden hair, 8k"
```

#### Generate an image with SDXL LCM model guided by a text prompt
```
python3 demo_txt2img_xl.py --lcm "an astronaut riding a rainbow unicorn, cinematic, dramatic"
```

#### Generate an image with SD-Turbo or SDXL-Turbo model guided by a text prompt
It is recommended to use LCM or EulerA scheduler to run SD-Turbo or SDXL-Turbo model.
```
python3 demo_txt2img.py --version sd-turbo "little cute gremlin wearing a jacket, cinematic, vivid colors, intricate masterpiece, golden ratio, highly detailed"

python3 demo_txt2img_xl.py --version xl-turbo "little cute gremlin wearing a jacket, cinematic, vivid colors, intricate masterpiece, golden ratio, highly detailed"
```

#### Generate an image with a text prompt using a control net
Control Net is supported for 1.5, SDXL base and SDXL-Turbo models in this demo.

```
wget https://huggingface.co/lllyasviel/sd-controlnet-depth/resolve/main/images/stormtrooper.png
python3 demo_txt2img_xl.py --controlnet-image stormtrooper.png --controlnet-type depth --controlnet-scale 0.5 --version xl-turbo "Stormtrooper's lecture in beautiful lecture hall"

wget https://hf.co/datasets/huggingface/documentation-images/resolve/main/diffusers/input_image_vermeer.png
python3 demo_txt2img_xl.py --controlnet-type canny --controlnet-scale 0.5 --controlnet-image input_image_vermeer.png --version xl-turbo --height 1024 --width 1024 "portrait of young Mona Lisa with mountain, river and forest in the background"
```

## Optimize Stable Diffusion ONNX models for Hugging Face Diffusers or Optimum

If you are able to run the above demo with docker, you can use the docker and skip the following setup and fast forward to [Export ONNX pipeline](#export-onnx-pipeline).

Below setup does not use docker. We'll use the environment to optimize ONNX models of Stable Diffusion exported by huggingface diffusers or optimum.
For Windows OS, please change the format of path to be like `.\sd` instead of `./sd`.

It is recommended to create a Conda environment with Python 3.10 for the following setup:
```
conda create -n py310 python=3.10
conda activate py310
```

### Setup Environment (CUDA) without docker

First, we need install CUDA 12.x, [cuDNN](https://docs.nvidia.com/deeplearning/cudnn/install-guide/index.html), and [TensorRT](https://docs.nvidia.com/deeplearning/tensorrt/install-guide/index.html) in the machine.

The verison of CuDNN can be found in https://onnxruntime.ai/docs/execution-providers/CUDA-ExecutionProvider.html#requirements.
The version of TensorRT can be found in https://onnxruntime.ai/docs/execution-providers/TensorRT-ExecutionProvider.html#requirements.

#### CUDA 12.*:
The official package of onnxruntime-gpu 1.19.x is built for CUDA 12.x. You can install it and other python packages like the following:
```
pip install onnxruntime-gpu
pip install torch --index-url https://download.pytorch.org/whl/cu124
pip install --upgrade polygraphy onnx-graphsurgeon --extra-index-url https://pypi.ngc.nvidia.com
pip install -r requirements/cuda12/requirements.txt
```
Finally, `pip install tensorrt` for Linux. For Windows, pip install the tensorrt wheel in the downloaded TensorRT zip file instead.

### Setup Environment (ROCm)

It is recommended that the users run the model with ROCm 6.2 or newer and Python 3.10. You can follow the following to install ROCm 6.x: https://rocmdocs.amd.com/projects/install-on-linux/en/latest/install/quick-start.html
Note that Windows is not supported for ROCm at the moment.

```
pip install -r requirements/rocm/requirements.txt
```

AMD GPU version of PyTorch can be installed from [pytorch.org](https://pytorch.org/get-started/locally/) or [AMD Radeon repo](https://repo.radeon.com/rocm/manylinux/rocm-rel-6.2.3/).

#### Install onnxruntime-rocm

One option is to install prebuilt wheel from https://repo.radeon.com/rocm/manylinux like:
```
wget https://repo.radeon.com/rocm/manylinux/rocm-rel-6.2.3/onnxruntime_rocm-1.18.0-cp310-cp310-linux_x86_64.whl
pip install onnxruntime_rocm-1.18.0-cp310-cp310-linux_x86_64.whl
```

If you want to use latest version of onnxruntime, you can build from source with Rocm 6.x following https://onnxruntime.ai/docs/build/eps.html#amd-rocm.
When the build is finished, you can install the wheel:`pip install build/Linux/Release/dist/*.whl`.

### Export ONNX pipeline
This step will export stable diffusion 1.5 to ONNX model in float32 using script from diffusers.

```
curl https://raw.githubusercontent.com/huggingface/diffusers/v0.15.1/scripts/convert_stable_diffusion_checkpoint_to_onnx.py > convert_sd_onnx.py
python convert_sd_onnx.py --model_path runwayml/stable-diffusion-v1-5  --output_path  ./sd1.5_onnx/fp32
```

For SDXL, use optimum to export the model:
```
pip install optimum diffusers onnx onnxruntime-gpu
optimum-cli export onnx --model stabilityai/stable-diffusion-xl-base-1.0 --task stable-diffusion-xl ./sdxl_onnx/fp32
```

#### Stable Diffusion 3.x and Flux 1.0

Stable Diffusion 3.x and Flux 1.0 requires transformers >= 4.45, and optimum > 1.23.3.
The default opset version for T5 is 12, which does not support bfloat16. To support bfloat16, please set opset version explicitly like below example.

```
git clone https://github.com/huggingface/optimum
cd optimum
pip install -e .

optimum-cli export onnx --model stabilityai/stable-diffusion-3-medium-diffusers ./sd3_onnx/fp32 --opset 15
optimum-cli export onnx --model stabilityai/stable-diffusion-3.5-medium ./sd3.5_medium_onnx/fp32 --opset 15
optimum-cli export onnx --model stabilityai/stable-diffusion-3.5-large ./sd3.5_large_onnx/fp32 --opset 15
optimum-cli export onnx --model black-forest-labs/FLUX.1-schnell ./flux1_schnell_onnx/fp32 --opset 15
optimum-cli export onnx --model black-forest-labs/FLUX.1-dev ./flux1_dev_onnx/fp32 --opset 15
```

### Optimize ONNX Pipeline

Example to optimize the exported float32 ONNX models, then save to float16 models:
```
python -m onnxruntime.transformers.models.stable_diffusion.optimize_pipeline -i ./sd1.5_onnx/fp32 -o ./sd1.5_onnx/fp16 --float16
```

You can also run the script in source code directory like the following:
```
git clone https://github.com/microsoft/onnxruntime
cd onnxruntime/onnxruntime/python/tools/transformers/models/stable_diffusion

python optimize_pipeline.py -i ./sdxl_onnx/fp32 -o ./sdxl_onnx/fp16 --float16
python optimize_pipeline.py -i ./sd3_onnx/fp32 -o ./sd3_onnx/fp16 --float16
python optimize_pipeline.py -i ./sd3.5_medium_onnx/fp32 -o ./sd3.5_medium_onnx/fp16 --float16
python optimize_pipeline.py -i ./sd3.5_large_onnx/fp32 -o ./sd3.5_large_onnx/fp16 --float16
python optimize_pipeline.py -i ./flux1_schnell_onnx/fp32 -o ./flux1_schnell_onnx/fp16 --float16 --bfloat16
python optimize_pipeline.py -i ./flux1_dev_onnx/fp32 -o ./flux1_dev_onnx/fp16 --float16 --bfloat16
```
When converting model to float16, some nodes has overflow risk and we can force those nodes to run in either float32 or bfloat16.
Option `--bfloat16` enables the later. If an operator does not support bfloat16, it will fallback to float32.

For SDXL model, it is recommended to use a machine with 48 GB or more memory to optimize.

### Run Benchmark

#### Run Benchmark with Optimum

The benchmark.py script will run a warm-up prompt twice, and measure the peak GPU memory usage in these two runs, then record them as first_run_memory_MB and second_run_memory_MB. Then it will run 5 runs to get average latency (in seconds), and output the results to benchmark_result.csv.

Note that the first run might need more time and memory: For example, cuDNN convolution algorithm search or model compile happens in the first run.

To avoid black image output for Stable Diffusion 2.1 with CUDA EP, we can set an environment variable before inferencing:
```
export ORT_DISABLE_TRT_FLASH_ATTENTION=1
```

Before running benchmark on PyTorch, you need to be logged in via `huggingface-cli login` once.

Example to benchmark the optimized pipeline of stable diffusion 1.5 with batch size 1 on CUDA EP:
```
python benchmark.py -p ./sd1.5_onnx/fp16 -b 1 -v 1.5
python benchmark.py -b 1 -v 1.5
```
For the first command, '-p' specifies a directory of optimized ONNX pipeline as generated by optimize_pipeline.py.
For the second command without '-p', we will use ORTPipelineForText2Image to export and optimize ONNX models for clip, unet and vae decoder.

On ROCm EP, use the following command instead:
```
python benchmark.py -p ./sd1.5_onnx/fp16 -b 1 --tuning --provider rocm -v 1.5
```

For ROCm EP, you can substitute `python benchmark.py` with `python -m onnxruntime.transformers.models.stable_diffusion.benchmark` since
the installed package is built from source. For CUDA, it is recommended to run `python benchmark.py` with the latest benchmark script.

For ROCm EP, the `--tuning` is mandatory because we heavily rely on tuning to find the runable kernels for ORT `OpKernel`s.

The default parameters are stable diffusion version=1.5, height=512, width=512, steps=50, batch_count=5. Run `python benchmark.py --help` for more information.

#### Stable Diffusion 3.x and Flux 1.0
Example of benchmark with optimum using CUDA provider on stable diffusion 3.5 medium and Flux 1.0:
```
python benchmark.py -e optimum --height 1024 --width 1024 --steps 30 -b 1 -v 3.0M -p sd3_onnx/fp32
python benchmark.py -e optimum --height 1024 --width 1024 --steps 30 -b 1 -v 3.5M -p sd3.5_medium_onnx/fp16
python benchmark.py -e optimum --height 1024 --width 1024 --steps 30 -b 1 -v 3.5L -p sd3.5_large_onnx/fp16
python benchmark.py -e optimum --height 1024 --width 1024 --steps 4 -b 1 -v Flux.1S -p flux1_schnell_onnx/fp16
python benchmark.py -e optimum --height 1024 --width 1024 --steps 30 -b 1 -v Flux.1D -p flux1_dev_onnx/fp16
```

Benchmark PyTorch eager mode performance:
```
python benchmark.py -e torch --height 1024 --width 1024 --steps 30 -b 1 -v 3.5L
python benchmark.py -e torch --height 1024 --width 1024 --steps 30 -b 1 -v Flux.1D
```

### Run Benchmark with xFormers

Run PyTorch 1.13.1+cu117 with xFormers like the following

```
pip install xformers==0.0.16
python benchmark.py -e torch -b 1 --use_xformers -v 1.5
```

### Run Benchmark with PyTorch 2.0 with torch.compile

For CUDA:
```
pip install torch --upgrade --index-url https://download.pytorch.org/whl/cu117
python benchmark.py -e torch -b 1 --enable_torch_compile -v 1.5
```

For ROCm:
```
pip install torch --upgrade --index-url https://download.pytorch.org/whl/rocm5.4.2
python benchmark.py -e torch -b 1 --enable_torch_compile --provider rocm  -v 1.5
```

Sometime, it complains ptxas not found when there are multiple CUDA versions installed. It can be fixed like `export TRITON_PTXAS_PATH=/usr/local/cuda-11.7/bin/ptxas` before running benchmark.

Note that torch.compile is not supported in Windows: we encountered error `Windows not yet supported for torch.compile`. So it is excluded from RTX 3060 results of Windows.


### Run Benchmark with TensorRT or TensorRT execution provider

For TensorRT installation, follow https://docs.nvidia.com/deeplearning/tensorrt/install-guide/index.html.

```
pip install torch==1.13.1+cu117 --extra-index-url https://download.pytorch.org/whl/cu117
pip install --upgrade polygraphy onnx-graphsurgeon --extra-index-url https://pypi.ngc.nvidia.com
pip install -r requirements-tensorrt.txt
export CUDA_MODULE_LOADING=LAZY
python benchmark.py -e tensorrt -b 1 -v 1.5
python benchmark.py -e onnxruntime -r tensorrt -b 1 -v 1.5
python benchmark.py -e onnxruntime -r tensorrt -b 1 -v 1.5 --enable_cuda_graph

python benchmark.py -e tensorrt --height 1024 --width 1024 -s 30  -b 1 -v xl-1.0 --enable_cuda_graph
python benchmark.py -e onnxruntime -r tensorrt --height 1024 --width 1024 -s 30  -b 1 -v xl-1.0 --enable_cuda_graph
```

### Results on RTX 3060 GPU:

To show the impact of each optimization on latency and GPU memory, we did some experiments:

| Optimizations                                                                      | Average Latency (batch_size=1) | Memory in MB (batch_size=1) | Average Latency (batch_size=8) | Memory in MB (batch_size=8) |
| ---------------------------------------------------------------------------------- | ------------------------------ | --------------------------- | ------------------------------ | --------------------------- |
| Raw FP32 models                                                                    | 25.6                           | 10,667                      | OOM                            | OOM                         |
| FP16 baseline                                                                      | 10.2                           | 10,709                      | OOM                            | OOM                         |
| FP16 baseline + FMHA                                                               | 6.1                            | 7,719                       | 39.1                           | 10,821                      |
| FP16 baseline + FMHA + NhwcConv                                                    | 5.5                            | 7,656                       | 38.8                           | 11,615                      |
| FP16 baseline + FMHA + NhwcConv + GroupNorm                                        | 5.1                            | 6,673                       | 35.8                           | 10,763                      |
| FP16 baseline + FMHA + NhwcConv + GroupNorm + BiasSplitGelu                        | 4.9                            | 4,447                       | 33.7                           | 6,669                       |
| FP16 baseline + FMHA + NhwcConv + GroupNorm + BiasSplitGelu + Packed QKV           | 4.8                            | 4,625                       | 33.5                           | 6,663                       |
| FP16 baseline + FMHA + NhwcConv + GroupNorm + BiasSplitGelu + Packed QKV + BiasAdd | 4.7                            | 4,480                       | 33.3                           | 6,499                       |

FP16 baseline contains optimizations available in ONNX Runtime 1.13 including LayerNormalization, SkipLayerNormalization, Gelu and float16 conversion.

Here FMHA means Attention and MultiHeadAttention operators with Flash Attention and Memory Efficient Attention kernels but inputs are not packed. Packed QKV means the inputs are packed.

The last two optimizations (Packed QKV and BiasAdd) are only available in nightly package. Compared to 1.14.1, nightly package has slight improvement in performance.

### Results on MI250X with 1 GCD

With runtime tuning enabled, we get following performance number on one GCD of a MI250X GPU:

| Optimizations                                                         | Average Latency (batch_size=1) | Memory in MB (batch_size=1) | Average Latency (batch_size=8) | Memory in MB (batch_size=8) |
| --------------------------------------------------------------------- | ------------------------------ | --------------------------- | ------------------------------ | --------------------------- |
| Raw FP32 models                                                       | 6.7                            | 17,319                      | 36.4 *                         | 33,787                      |
| FP16 baseline                                                         | 4.1                            | 8,945                       | 24.0 *                         | 34,493                      |
| FP16 baseline + FMHA                                                  | 2.6                            | 4,886                       | 15.0                           | 10,146                      |
| FP16 baseline + FMHA + NhwcConv                                       | 2.4                            | 4,952                       | 14.8                           | 9,632                       |
| FP16 baseline + FMHA + NhwcConv + GroupNorm                           | 2.3                            | 4,906                       | 13.6                           | 9,774                       |
| FP16 baseline + FMHA + NhwcConv + GroupNorm + BiasSplitGelu           | 2.2                            | 4,910                       | 12.5                           | 9,646                       |
| FP16 baseline + FMHA + NhwcConv + GroupNorm + BiasSplitGelu + BiasAdd | 2.2                            | 4,910                       | 12.5                           | 9,778                       |

The entries marked with `*` produce suspicious output images. The might be numerical stability or correctness issue for the pipeline. The performance number is for reference only.


### Example Benchmark output

Common settings for below test results:

| model_name                     | disable_safety_checker | height | width | steps | batch_count | num_prompts |
| ------------------------------ | ---------------------- | ------ | ----- | ----- | ----------- | ----------- |
| runwayml/stable-diffusion-v1-5 | TRUE                   | 512    | 512   | 50    | 5           | 1           |

#### Results of MI250X, 1 GCD (Ubuntu 20.04)

| engine      | version                 | provider              | batch size | average latency | first run memory MB | second run memory MB |
| ----------- | ----------------------- | --------------------- | ---------- | --------------- | ------------------- | -------------------- |
| onnxruntime | 1.15.0+rocm5.4.2        | ROCM                  | 1          | 2.2             | 5,548               | 4,908                |
| torch       | 1.12.1+rocm5.4          | -                     | 1          | 3.4             | 6,653               | 4,613                |
| torch       | 2.0.0+rocm5.4.2         | default               | 1          | 3.2             | 5,977               | 4,368                |
| torch       | 2.0.0+rocm5.4.2         | compile               | 1          | 3.0             | 5,869               | 4,266                |
| onnxruntime | 1.15.0+rocm5.4.2        | ROCM                  | 4          | 6.6             | 5,546               | 4,906                |
| torch       | 1.12.1+rocm5.4          | -                     | 4          | 10.1            | 19,477              | 11,325               |
| torch       | 2.0.0+rocm5.4.2         | default               | 4          | 10.5            | 13,051              | 7,300                |
| torch       | 2.0.0+rocm5.4.2         | compile               | 4          | 9.2             | 12,879              | 7,190                |
| onnxruntime | 1.15.0+rocm5.4.2        | ROCM                  | 8          | 12.5            | 9,778               | 9,006                |
| torch       | 1.12.1+rocm5.4          | -                     | 8          | 19.3            | 55,851              | 20,014               |
| torch       | 2.0.0+rocm5.4.2         | default               | 8          | 20.3            | 23,551              | 11,930               |
| torch       | 2.0.0+rocm5.4.2         | compile               | 8          | 17.8            | 23,303              | 11,800               |

#### Results of MI100 (Ubuntu 20.04)

| engine      | version                 | provider              | batch size | average latency | first run memory MB | second run memory MB |
| ----------- | ----------------------- | --------------------- | ---------- | --------------- | ------------------- | -------------------- |
| onnxruntime | 1.15.0+rocm5.4.2        | ROCM                  | 1          | 2.4             | 5,254               | 4,614                |
| torch       | 1.12.1+rocm5.4          | -                     | 1          | 3.5             | 5,771               | 4,672                |
| torch       | 2.0.0+rocm5.4.2         | default               | 1          | 3.5             | 5,811               | 4,206                |
| torch       | 2.0.0+rocm5.4.2         | compile               | 1          | 3.1             | 5,774               | 4,168                |
| onnxruntime | 1.15.0+rocm5.4.2        | ROCM                  | 4          | 7.5             | 7,290               | 6,646                |
| torch       | 1.12.1+rocm5.4          | -                     | 4          | 10.7            | 19,334              | 11,181               |
| torch       | 2.0.0+rocm5.4.2         | default               | 4          | 11.5            | 12,881              | 7,151                |
| torch       | 2.0.0+rocm5.4.2         | compile               | 4          | 10.0            | 12,740              | 7,073                |
| onnxruntime | 1.15.0+rocm5.4.2        | ROCM                  | 8          | 14.4            | 7,320               | 6,676                |
| torch       | 1.12.1+rocm5.4          | -                     | 8          | 20.2            | 31,820              | 19,908               |
| torch       | 2.0.0+rocm5.4.2         | default               | 8          | 22.2            | 23,415              | 11,815               |
| torch       | 2.0.0+rocm5.4.2         | compile               | 8          | 19.3            | 23,154              | 11,667               |

### Credits

Some CUDA kernels (TensorRT Fused Attention, GroupNorm, SplitGelu and BiasAdd etc.) and demo diffusion were originally implemented in [TensorRT](https://github.com/nviDIA/TensorRT) by Nvidia.
We use [Flash Attention v2](https://github.com/Dao-AILab/flash-attention) in Linux.
We use Memory efficient attention from [CUTLASS](https://github.com/NVIDIA/cutlass). The kernels were developed by Meta xFormers.
The ONNX export script and pipeline for stable diffusion was developed by Huggingface [diffusers](https://github.com/huggingface/diffusers) library.

Most ROCm kernel optimizations are from [composable kernel](https://github.com/ROCmSoftwarePlatform/composable_kernel).
Some kernels are enabled by MIOpen. We hereby thank for the AMD developers' collaboration.

### Future Works
* Update demo to support inpainting.
* Support flash attention in Windows.
* Integration with UI.
* Optimization for H100 GPU.
* Export the whole pipeline into a single ONNX model. This senario is mainly for mobile device.
