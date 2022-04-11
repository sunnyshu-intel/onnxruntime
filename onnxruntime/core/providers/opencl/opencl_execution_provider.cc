// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "opencl_execution_provider.h"
#include "opencl_allocator.h"
#include "opencl_program_manager.h"
#include "opencl_data_transfer.h"
#include "core/common/logging/logging.h"
#include "core/framework/kernel_registry.h"
#include "core/framework/op_kernel.h"

#include <array>
#include <utility>

// Add includes of kernel implementations
#include "memcpy_kernel.h"
#include "core/providers/opencl/math/clip.h"
#include "core/providers/opencl/math/elementwise.h"
#include "core/providers/opencl/nn/conv.h"
#include "core/providers/opencl/nn/global_average_pool.h"
#include "core/providers/opencl/nn/relu.h"
#include "core/providers/opencl/nn/max_pool.h"
#include "core/providers/opencl/nn/concat.h"
#include "core/providers/opencl/tensor/shape.h"
#include "core/providers/opencl/tensor/resize.h"

namespace onnxruntime {
namespace opencl {

Status RegisterOpenCLKernels(KernelRegistry& kernel_registry) {
  static const BuildKernelCreateInfoFn function_table[] = {
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kOpenCLExecutionProvider, kOnnxDomain, 1, 12, Shape)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kOpenCLExecutionProvider, kOnnxDomain, 13, 14, Shape)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_KERNEL_CLASS_NAME(kOpenCLExecutionProvider, kOnnxDomain, 15, Shape)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_KERNEL_CLASS_NAME(kOpenCLExecutionProvider, kOnnxDomain, 1, MemcpyFromHost)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_KERNEL_CLASS_NAME(kOpenCLExecutionProvider, kOnnxDomain, 1, MemcpyToHost)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_KERNEL_CLASS_NAME(kOpenCLExecutionProvider, kOnnxDomain, 7, Add)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_KERNEL_CLASS_NAME(kOpenCLExecutionProvider, kOnnxDomain, 7, Sub)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_KERNEL_CLASS_NAME(kOpenCLExecutionProvider, kOnnxDomain, 7, Mul)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_KERNEL_CLASS_NAME(kOpenCLExecutionProvider, kOnnxDomain, 7, Div)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_KERNEL_CLASS_NAME(kOpenCLExecutionProvider, kOnnxDomain, 6, Clip)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_KERNEL_CLASS_NAME(kOpenCLExecutionProvider, kOnnxDomain, 12, Clip)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kOpenCLExecutionProvider, kOnnxDomain, 6, 12, Relu)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_KERNEL_CLASS_NAME(kOpenCLExecutionProvider, kOnnxDomain, 13, Relu)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_KERNEL_CLASS_NAME(kOpenCLExecutionProvider, kOnnxDomain, 14, Relu)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_KERNEL_CLASS_NAME(kOpenCLExecutionProvider, kOnnxDomain, 1, GlobalAveragePool)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kOpenCLExecutionProvider, kOnnxDomain, 1, 10, Conv)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_KERNEL_CLASS_NAME(kOpenCLExecutionProvider, kOnnxDomain, 11, Conv)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kOpenCLExecutionProvider, kOnnxDomain, 8, 11, MaxPool)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_KERNEL_CLASS_NAME(kOpenCLExecutionProvider, kOnnxDomain, 12, MaxPool)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kOpenCLExecutionProvider, kOnnxDomain, 4, 10, Concat)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kOpenCLExecutionProvider, kOnnxDomain, 11, 12, Concat)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_KERNEL_CLASS_NAME(kOpenCLExecutionProvider, kOnnxDomain, 13, Concat)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_KERNEL_CLASS_NAME(kOpenCLExecutionProvider, kMSDomain, 1, FusedConv)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kOpenCLExecutionProvider, kOnnxDomain, 11, 12, Resize)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_KERNEL_CLASS_NAME(kOpenCLExecutionProvider, kOnnxDomain, 13, Resize)>,
  };

  for (auto& function_table_entry : function_table) {
    KernelCreateInfo info = function_table_entry();
    if (info.kernel_def != nullptr) {  // filter disabled entries where type is void
      VLOGS_DEFAULT(V_GENERIC) << "Register kernel name: " << info.kernel_def->OpName()
                               << ", domain: " << info.kernel_def->Domain();
      ORT_RETURN_IF_ERROR(kernel_registry.Register(std::move(info)));
    }
  }

  return Status::OK();
}

std::shared_ptr<KernelRegistry> GetOpenCLKernelRegistry() {
  std::shared_ptr<KernelRegistry> kernel_registry = std::make_shared<KernelRegistry>();
  ORT_THROW_IF_ERROR(RegisterOpenCLKernels(*kernel_registry));
  return kernel_registry;
}

bool ShouldFlushAfterLaunch(const std::string& device_name) {
  return device_name.find("Mali") != std::string::npos;
}

}  // namespace opencl

OpenCLExecutionProvider::OpenCLExecutionProvider(const OpenCLExecutionProviderInfo& info)
    : IExecutionProvider(kOpenCLExecutionProvider), use_fp16_(info.use_fp16) {
  Status status;
#ifdef CL3W_ENABLE
  if (cl3wInit() != CL3W_OK) {
    ORT_THROW("cl3w initialization failure.");
  }
#endif
  ORT_THROW_IF_ERROR(InitOpenCLContext());
  program_manager_ = std::make_unique<opencl::OpenCLProgramManager>(*this);
  InitCopyKernels();

#ifdef TRACY_ENABLE
  tracy_cl_ctx_ = TracyCLContext(ctx_, dev_);
#endif
}

OpenCLExecutionProvider::~OpenCLExecutionProvider() {
  // FIXME: kernel manager should release all managed kernels and programs

#ifdef TRACY_ENABLE
  TracyCLCollect(tracy_cl_ctx_);
  TracyCLDestroy(tracy_cl_ctx_);
#endif

  clReleaseCommandQueue(cmd_queue_);
  clReleaseDevice(dev_);
  clReleaseContext(ctx_);
}

std::shared_ptr<KernelRegistry> OpenCLExecutionProvider::GetKernelRegistry() const {
  static std::shared_ptr<KernelRegistry> kernel_registry = opencl::GetOpenCLKernelRegistry();
  return kernel_registry;
}
template <class T>
static void GetCLDevInfo(cl_device_id  device ,
                  cl_device_info param_name, T* param) {
  static_assert(std::is_same<T, size_t>::value || std::is_same<T, uint64_t>::value 
      || std::is_same<T, uint32_t>::value || std::is_same<T, size_t[3]>::value);
  ORT_THROW_IF_CL_ERROR(clGetDeviceInfo(device, param_name, sizeof(T), param, nullptr));
}
// Gpu SubGroup, referenced to TNN, GPU-model->a exprienced magic number, as CL_KERNEL_MAX_SUB_GROUP_SIZE_FOR_NDRANGE is not implemented util opencl 2.1
static std::map<int, int> AdrenoSubGroup{
    {640, 128},
    {630, 128},
    {616, 128},
    {612, 64},
    {610, 64},
    {540, 32},
    {530, 32},
    {512, 32},
    {510, 32},
    {509, 32},
    {506, 32},
    {505, 32},
    {405, 32},
    {330, 16},
};
Status OpenCLExecutionProvider::InitOpenCLContext() {
  cl_uint num_platforms;
  ORT_RETURN_IF_CL_ERROR(clGetPlatformIDs(0, nullptr, &num_platforms));
  // NOTE: the EP is in construction, the logger_ is not registered
  LOGS_DEFAULT(VERBOSE) << "[CL] num platforms: " << num_platforms;
  ORT_RETURN_IF_NOT(num_platforms > 0, "Cannot find OpenCL platform.");

  std::vector<cl_platform_id> platforms(num_platforms);
  ORT_RETURN_IF_CL_ERROR(clGetPlatformIDs(static_cast<cl_uint>(platforms.size()), platforms.data(), nullptr));
  int selected_platform_idx = -1;
  // FIXME: add platform selection logic
  for (auto& platform : platforms) {
    size_t ret_size;
    ORT_RETURN_IF_CL_ERROR(clGetPlatformInfo(platform, CL_PLATFORM_VENDOR, 0, nullptr, &ret_size));
    std::string vendor(ret_size, '\0');
    ORT_RETURN_IF_CL_ERROR(clGetPlatformInfo(platform, CL_PLATFORM_VENDOR, vendor.size(), vendor.data(), nullptr));
    LOGS_DEFAULT(VERBOSE) << "[CL] platform vendor: " << vendor;
    if (vendor == "Oclgrind") {
      LOGS_DEFAULT(INFO) << "[CL] platform " << vendor << " selected";
      selected_platform_idx = 1;
      break;
    }
  }
  if (selected_platform_idx == -1) {
    LOGS_DEFAULT(INFO) << "[CL] default platform selected";
    selected_platform_idx = 0;
  }
  auto* selected_platform = platforms[selected_platform_idx];

  cl_int err{};
  cl_context_properties properties[] = {CL_CONTEXT_PLATFORM, (cl_context_properties)selected_platform, 0};
  ctx_ = clCreateContextFromType(properties, CL_DEVICE_TYPE_GPU, /*pfn_notify=*/nullptr, /*user_data=*/nullptr, &err);
  ORT_RETURN_IF_CL_ERROR(err);

  size_t ret_size;
  ORT_RETURN_IF_CL_ERROR(clGetContextInfo(ctx_, CL_CONTEXT_DEVICES, 0, nullptr, &ret_size));
  std::vector<cl_device_id> devices(ret_size);
  ORT_RETURN_IF_CL_ERROR(clGetContextInfo(ctx_, CL_CONTEXT_DEVICES, devices.size(), devices.data(), nullptr));
  LOGS_DEFAULT(VERBOSE) << "[CL] num devices: " << devices.size();
  ORT_RETURN_IF(devices.empty(), "Cannot find OpenCL device.");
  dev_ = devices[0];

  auto GetDeviceInfo = [](cl_device_id dev, cl_device_info info_name) -> std::string {
    size_t ret_size;
    ORT_THROW_IF_CL_ERROR(clGetDeviceInfo(dev, info_name, 0, nullptr, &ret_size));
    std::string ret(ret_size, '\0');
    ORT_THROW_IF_CL_ERROR(clGetDeviceInfo(dev, info_name, ret.size(), ret.data(), nullptr));
    return ret;
  };

  dev_info_.device_name = GetDeviceInfo(dev_, CL_DEVICE_NAME);
  auto device_version = GetDeviceInfo(dev_, CL_DEVICE_VERSION);
  LOGS_DEFAULT(INFO) << "[CL] device name: " << dev_info_.device_name;
  LOGS_DEFAULT(VERBOSE) << "[CL] device vendor: " << device_version;
  LOGS_DEFAULT(VERBOSE) << "[CL] device version: " << GetDeviceInfo(dev_, CL_DEVICE_VERSION);
  auto exts = GetDeviceInfo(dev_, CL_DEVICE_EXTENSIONS);
  LOGS_DEFAULT(VERBOSE) << "[CL] device extensions: " << exts << std::endl;
  dev_info_.has_fp16 = exts.find("cl_khr_fp16") != std::string::npos;
  if (!dev_info_.has_fp16 && UseFp16()) {
    LOGS_DEFAULT(WARNING) << "[CL] FP16 is requested, but is not supported by the device!";
    DisableFp16();
  }
  flush_after_launch_ = opencl::ShouldFlushAfterLaunch(dev_info_.device_name);
  LOGS_DEFAULT(INFO) << "[CL] FP16: " << UseFp16();
  LOGS_DEFAULT(INFO) << "[CL] clFlush after launch: " << flush_after_launch_;
  if (dev_info_.device_name == "QUALCOMM Adreno(TM)") {
    dev_info_.gpu_type = opencl::GpuType::ADRENO;
    //windows will report a warning os sscanf_s
#if !(defined(WIN32) || defined(_WIN32) || defined(_WIN32_) || \
    defined(WIN64) || defined(_WIN64) || defined(_WIN64_))
    sscanf(device_version.c_str(), "%*s%*f%*s%d", &dev_info_.gpu_model);
#endif
#if CL_HPP_TARGET_OPENCL_VERSION >= 200 && CL_TARGET_OPENCL_VERSION >= 210 && defined(CL_HPP_USE_CL_SUB_GROUPS_KHR)
    cl_int cl_ret;
    sub_group_size = kernel.getSubGroupInfo<CL_KERNEL_MAX_SUB_GROUP_SIZE_FOR_NDRANGE>(*device_, range, &cl_ret);
    if (cl_ret != CL_SUCCESS) {
      CHECK_CL_SUCCESS(cl_ret)
      sub_group_size = 0;
    }
#else
    dev_info_.sub_group_size = AdrenoSubGroup.count(dev_info_.gpu_model) ?
        AdrenoSubGroup[dev_info_.gpu_model] : 0;
#endif
  }
  GetCLDevInfo(dev_, CL_DEVICE_GLOBAL_MEM_CACHE_SIZE, &dev_info_.global_memery_cachesize_);
  GetCLDevInfo(dev_, CL_DEVICE_MAX_COMPUTE_UNITS, &dev_info_.compute_units_);
  GetCLDevInfo(dev_, CL_DEVICE_MAX_CLOCK_FREQUENCY, &dev_info_.max_freq_);
  GetCLDevInfo(dev_, CL_DEVICE_LOCAL_MEM_SIZE, &dev_info_.local_memory_size_);
  GetCLDevInfo(dev_, CL_DEVICE_MAX_WORK_GROUP_SIZE, &dev_info_.max_work_group_size);
  GetCLDevInfo(dev_, CL_DEVICE_MAX_WORK_ITEM_SIZES, &(dev_info_.max_work_item_size));
  GetCLDevInfo(dev_, CL_DEVICE_IMAGE2D_MAX_WIDTH, &dev_info_.image_2d_max_size[0]);
  GetCLDevInfo(dev_, CL_DEVICE_IMAGE2D_MAX_HEIGHT, &dev_info_.image_2d_max_size[1]);

#ifdef TRACY_ENABLE
  cmd_queue_ = clCreateCommandQueue(ctx_, dev_, CL_QUEUE_PROFILING_ENABLE, &err);
#else
  cmd_queue_ = clCreateCommandQueue(ctx_, dev_, /*properties=*/0, &err);
#endif
  ORT_RETURN_IF_CL_ERROR(err);

  return Status::OK();
}

// adreno local size calculate //reference to TNN
static std::vector<size_t> AdrenoLocalSize2D(const opencl::NDRange& gws, const opencl::OpenCLDeviceInfo& gpu_info) {
  std::vector<size_t> lws;
  const size_t max_workgroup_size = gpu_info.max_work_group_size;
  const size_t subgroup_size = gpu_info.sub_group_size;
  size_t min_workgroup_count = gpu_info.compute_units_;
  // for the later verion gpu 1 SP can process more than one workgroup
  if (gpu_info.gpu_model >= 540)
    min_workgroup_count = 2 * gpu_info.compute_units_;

  // judge gws[1] fisrt
  if (gws[1] % min_workgroup_count == 0) {
    lws.resize(2);
    lws[1] = std::min<size_t>(gws[1] / min_workgroup_count, max_workgroup_size);
    auto AdrenoLocalSizeValid = [](const opencl::NDRange& gws, std::vector<size_t>& lws,
                                   const size_t subgroup_size)->bool {
      return 0 == (lws[0] * lws[1]) % subgroup_size && 0 == gws[0] % lws[0] && 0 == gws[1] % lws[1] &&
             ((lws[0] < lws[1]) == (gws[0] < gws[1]));
    };
    // if subgroup size is got, then use it
    if (0 != subgroup_size) {
      size_t min_workgroup_size = subgroup_size * 2;
      size_t max_val = std::max<size_t>(max_workgroup_size / lws[1], 1);
      size_t min_val = std::max<size_t>(min_workgroup_size / lws[1], 1);
      lws[0] = std::min<size_t>(gws[0], max_val);
      for (; lws[0] >= min_val; lws[0]--) {
        if (AdrenoLocalSizeValid(gws, lws, subgroup_size)) {
          return lws;
        }
      }
    }

    // another way to calculate lws[0]
    lws[0] = max_workgroup_size / lws[1];
    lws[0] = std::max<size_t>(std::min<size_t>(gws[0], lws[0]), 1);
    if (0 == gws[0] % lws[0] && 0 == gws[1] % lws[1] && ((lws[0] < lws[1]) == (gws[0] < gws[1]))) {
      return lws;
    }
  }

  // judge gws[0] later
  if (gws[0] % min_workgroup_count == 0) {
    lws.resize(2);
    lws[0] = std::min<size_t>(gws[0] / min_workgroup_count, max_workgroup_size);

    // if subgroup size is got, then use it
    if (0 != subgroup_size) {
      size_t min_workgroup_size = subgroup_size * 2;
      size_t max_val = std::max<size_t>(max_workgroup_size / lws[0], 1);
      size_t min_val = std::max<size_t>(min_workgroup_size / lws[0], 1);
      lws[1] = std::min<size_t>(gws[1], max_val);
      for (; lws[1] >= min_val; lws[1]--) {
        if (0 == (lws[0] * lws[1]) % subgroup_size && 0 == gws[0] % lws[0] && 0 == gws[1] % lws[1] &&
            ((lws[0] < lws[1]) == (gws[0] < gws[1]))) {
          return lws;
        }
      }
    }

    // another way to calculate lws[1]
    lws[1] = max_workgroup_size / lws[0];
    lws[1] = std::max<size_t>(std::min<size_t>(gws[1], lws[1]), 1);
    if (0 == gws[0] % lws[0] && 0 == gws[1] % lws[1] && ((lws[0] < lws[1]) == (gws[0] < gws[1]))) {
      return lws;
    }
  }
  lws.clear();
  return lws;
}

opencl::NDRange OpenCLExecutionProvider::DefaultLocalWG2DWithoutTune(const opencl::NDRange& gws) const {
  if (dev_info_.gpu_type != opencl::GpuType::ADRENO) {
    return {};
  }
  std::vector<size_t> lwgs(2);
  if (dev_info_.max_work_group_size == 0) {
    lwgs[0] = lwgs[1] = 1;
  } else {
    lwgs = AdrenoLocalSize2D(gws, dev_info_);
  }
  if (lwgs.size() == 0) {
    return {};
  }
  return {lwgs[0], lwgs[1]};
}

void OpenCLExecutionProvider::RegisterAllocator(std::shared_ptr<AllocatorManager> /*allocator_manager*/) {
  // FIXME: Is it possible to use arena on OpenCL? cl_mem is opaque pointer in
  // OpenCL 1.2 and Shared Virtual Memory (SVM) is only available in OpenCL
  // 2.0, which still have limited support on a wide range of devices. Without
  // SVM we are unable to slice pre-allocated buffer, thus, unable to use it as
  // an arena.
  //
  // See https://stackoverflow.com/a/40951614
  InsertAllocator(CreateAllocator(AllocatorCreationInfo{
      [this](int) {
        return std::make_unique<opencl::OpenCLBufferAllocator>(ctx_);
      },
      0,
      /*use_arena=*/false,
  }));

  InsertAllocator(CreateAllocator(AllocatorCreationInfo{
      [this](int) {
        return std::make_unique<opencl::OpenCLImage2DAllocator>(ctx_, UseFp16(), dev_info_.image_2d_max_size);
      },
      0,
      /*use_arena=*/false,
  }));

  InsertAllocator(CreateAllocator(AllocatorCreationInfo{
      [](int) {
        return std::make_unique<CPUAllocator>(
            OrtMemoryInfo(opencl::CPUAllocatorName, OrtAllocatorType::OrtDeviceAllocator, OrtDevice(), 0, OrtMemTypeCPUOutput));
      }}));

  InsertAllocator(CreateAllocator(AllocatorCreationInfo{
      [](int) {
        return std::make_unique<CPUAllocator>(
            OrtMemoryInfo(opencl::CPUInputAllocatorName, OrtAllocatorType::OrtDeviceAllocator, OrtDevice(), 0, OrtMemTypeCPUInput));
      }}));
}

IAllocatorUniquePtrToClMem OpenCLExecutionProvider::GetScratchBuffer(size_t nbytes) const {
  ZoneScopedN("OpenCLExecutionProvider::GetScratchBuffer");
  auto alloc = GetAllocator(0, (OrtMemType)opencl::CLMemType::OPENCL_BUFFER);
  return {
      static_cast<cl_mem>(alloc->Alloc(nbytes)),
      [=](void* ptr) {
        alloc->Free(ptr);
      }};
}

IAllocatorUniquePtrToClMem OpenCLExecutionProvider::GetScratchImage2D(const opencl::Image2DDesc& desc) const {
  ZoneScopedN("OpenCLExecutionProvider::GetScratchImage2D");
  auto base_alloc = GetAllocator(0, (OrtMemType)opencl::CLMemType::OPENCL_IMAGE_2D);
  auto* alloc = static_cast<opencl::OpenCLImage2DAllocator*>(base_alloc.get());
  return IAllocatorUniquePtr<std::remove_pointer_t<cl_mem>>{
      static_cast<cl_mem>(alloc->Alloc(desc)),
      [=](void* ptr) {
        alloc->Free(ptr);
      }};
}

Status OpenCLExecutionProvider::AfterCLLaunch() const {
  // We observe some performance boost on some devices. If you observe bubble
  // in command queue while host kernel launch is fast enough to throttle the
  // GPU during profiling, then you should consider flush the queue immediately
  // after the kernel launch.
  if (flush_after_launch_) {
    ORT_RETURN_IF_CL_ERROR(clFlush(cmd_queue_), "command queue flush failure.");
  }
  return Status::OK();
}

const opencl::OpenCLProgramManager& OpenCLExecutionProvider::GetProgramManager() const {
  return *program_manager_;
}

opencl::OpenCLProgramManager& OpenCLExecutionProvider::GetProgramManager() {
  return *program_manager_;
}

/*
#pragma region IDataTransfer related code
*/
std::unique_ptr<onnxruntime::IDataTransfer> OpenCLExecutionProvider::GetDataTransfer() const {
  return std::make_unique<opencl::OpenCLDataTransfer>(this, copy_kernels_.get());
}

namespace {
#define CONTENT_NAME copy_tensors_src
#include "opencl_generated/kernels/copy_tensors.cl.inc"
}  // namespace

void OpenCLExecutionProvider::InitCopyKernels() {
  ZoneScopedN("OpenCLExecutionProvider::InitCopyKernels");
  copy_kernels_ = std::make_unique<opencl::OpenCLKernelHolder>(GetProgramManager());
  copy_kernels_->LoadProgram(copy_tensors_src, copy_tensors_src_len);
  copy_kernels_->LoadKernel("CopyBuffer1DToImage2D");
  copy_kernels_->LoadKernel("CopyBuffer2DToImage2D");
  copy_kernels_->LoadKernel("CopyImage2DToBuffer1D");
  copy_kernels_->LoadKernel("CopyBufferNCHWToImage2D");
  copy_kernels_->LoadKernel("CopyImage2DToBufferNCHW");
}
/*
#pragma endregion
*/

}  // namespace onnxruntime
