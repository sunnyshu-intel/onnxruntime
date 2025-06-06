// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "gtest/gtest.h"
#include "core/mlas/inc/mlas.h"
#include "core/framework/run_options.h"
#include "test/common/cuda_op_test_utils.h"
#include "test/providers/provider_test_utils.h"
#include "test/common/dnnl_op_test_utils.h"
#include "test/providers/run_options_config_keys.h"
#include "test/util/include/default_providers.h"

namespace onnxruntime {
namespace test {

namespace {

const onnxruntime::RunOptions run_options = []() {
  onnxruntime::RunOptions options{};
  ORT_THROW_IF_ERROR(options.config_options.AddConfigEntry(kOpTesterRunOptionsConfigTestTunableOp, "true"));
  return options;
}();

const constexpr auto run_with_tunable_op = &run_options;

// Helper function to initialize input matrices
auto initialize_matrix = [](int64_t rows, int64_t cols) {
  std::vector<float> data;
  data.reserve(rows * cols);
  for (int64_t i = 0; i < rows * cols; ++i) {
    data.push_back(static_cast<float>((i % 7) + 1));
  }
  return data;
};

enum class BiasType {
  noBias,      // No bias input
  MBias,       // C shape is {M,1}
  ScalarBias,  // C shape is {1,1}
  MNBias,      // C shape is {M,N}
  NBias        // C shape is {N}
};

// Helper function to initialize bias data for Gemm tests
auto initialize_bias = [](BiasType bias_type, int64_t M, int64_t N) {
  std::pair<std::vector<float>, std::vector<int64_t>> result;
  auto& [data, shape] = result;

  switch (bias_type) {
    case BiasType::noBias:
      break;
    case BiasType::MBias:
      shape = {M, 1};
      for (int64_t i = 0; i < M; ++i) {
        data.push_back(static_cast<float>((i % 7) + 1));
      }
      break;
    case BiasType::ScalarBias:
      shape = {1, 1};
      data.push_back(1.0f);
      break;
    case BiasType::MNBias:
      shape = {M, N};
      for (int64_t i = 0; i < M * N; ++i) {
        data.push_back(static_cast<float>((i % 7) + 1));
      }
      break;
    case BiasType::NBias:
      shape = {N};
      for (int64_t i = 0; i < N; ++i) {
        data.push_back(static_cast<float>((i % 7) + 1));
      }
      break;
  }
  return result;
};

// Helper function to get bias value for Gemm tests
auto get_bias_value = [](const std::vector<float>& bias_data, BiasType bias_type, int64_t i, int64_t j, int64_t N) {
  if (bias_data.empty()) return 0.0f;

  switch (bias_type) {
    case BiasType::noBias:
      return 0.0f;
    case BiasType::MBias:
      return bias_data[i];
    case BiasType::ScalarBias:
      return bias_data[0];
    case BiasType::MNBias:
      return bias_data[i * N + j];
    case BiasType::NBias:
      return bias_data[j];
    default:
      return 0.0f;
  }
};

}  // namespace

// Only CUDA, ROCM, CoreML and XNNPack kernels have float 16 support
TEST(GemmOpTest, GemmNoTrans_f16) {
#ifdef USE_CUDA
  int min_cuda_architecture = 530;
  if (!HasCudaEnvironment(min_cuda_architecture)) {
    LOGS_DEFAULT(WARNING) << "Hardware NOT support FP16";
    return;
  }
#endif

  std::vector<float> A{1.0f, 2.0f, 3.0f, 4.0f,
                       -1.0f, -2.0f, -3.0f, -4.0f};
  std::vector<float> B = {0.5f, 2.1f, 1.2f, -0.3f,
                          -1.2f, 0.2f, 1.0f, -2.1f,
                          1.3f, 4.1f, 1.3f, -8.1f};
  std::vector<float> C = {0.5f, 2.1f, 1.2f,
                          -0.3f, -1.2f, 0.2f};

  std::vector<MLFloat16> f_A(8);
  std::vector<MLFloat16> f_B(12);
  ConvertFloatToMLFloat16(A.data(), f_A.data(), 8);
  ConvertFloatToMLFloat16(B.data(), f_B.data(), 12);

  {
    // bias has same shape as output
    std::vector<MLFloat16> f_Y(6);
    std::vector<float> Y{19.8f, 0.7f, -25.7f,
                         -19.6f, 0.2f, 27.1f};
    ConvertFloatToMLFloat16(Y.data(), f_Y.data(), 6);

    std::vector<MLFloat16> f_C(6);
    ConvertFloatToMLFloat16(C.data(), f_C.data(), 6);

    OpTester test("Gemm", 13);

    test.AddAttribute("transA", (int64_t)0);
    test.AddAttribute("transB", (int64_t)0);
    test.AddAttribute("alpha", 1.0f);
    test.AddAttribute("beta", 1.0f);
    test.AddInput<MLFloat16>("A", {2, 4}, f_A);
    test.AddInput<MLFloat16>("B", {4, 3}, f_B);
    test.AddInput<MLFloat16>("C", {2, 3}, f_C);
    test.AddOutput<MLFloat16>("Y", {2, 3}, f_Y);
    // we used float data with decimal instead of only integer, increase Tolerance to make test pass
    test.SetOutputTolerance(0.005f);
    test.ConfigExcludeEps({kTensorrtExecutionProvider})  // TensorRT: fp16 is not supported
        .Config(run_with_tunable_op)
        .RunWithConfig();
  }
  {
    // bias has  shape {1,  output_features}
    std::vector<MLFloat16> f_Y(6);
    std::vector<float> Y{19.8f, 0.7f, -25.7f,
                         -18.8f, 3.5f, 28.1f};
    ConvertFloatToMLFloat16(Y.data(), f_Y.data(), 6);

    std::vector<MLFloat16> f_C(3);
    ConvertFloatToMLFloat16(C.data(), f_C.data(), 3);
    // CoreML program require B/C are constant
    OpTester test("Gemm", 13);

    test.AddAttribute("transA", (int64_t)0);
    test.AddAttribute("transB", (int64_t)0);
    test.AddAttribute("alpha", 1.0f);
    test.AddAttribute("beta", 1.0f);
    test.AddInput<MLFloat16>("A", {2, 4}, f_A);
    test.AddInput<MLFloat16>("B", {4, 3}, f_B, true);
    test.AddInput<MLFloat16>("C", {3}, f_C, true);
    test.AddOutput<MLFloat16>("Y", {2, 3}, f_Y);
    test.SetOutputTolerance(0.005f);
    test.ConfigExcludeEps({kTensorrtExecutionProvider})  // TensorRT: fp16 is not supported
        .Config(run_with_tunable_op)
        .RunWithConfig();
  }
  {
    // bias is a scalar
    std::vector<MLFloat16> f_Y(6);
    std::vector<float> Y{19.8f, -0.9f, -26.4f,
                         -18.8f, 1.9f, 27.4f};
    ConvertFloatToMLFloat16(Y.data(), f_Y.data(), 6);

    std::vector<MLFloat16> f_C(1);
    ConvertFloatToMLFloat16(C.data(), f_C.data(), 1);
    OpTester test("Gemm", 13);

    test.AddAttribute("transA", (int64_t)0);
    test.AddAttribute("transB", (int64_t)0);
    test.AddAttribute("alpha", 1.0f);
    test.AddAttribute("beta", 1.0f);
    test.AddInput<MLFloat16>("A", {2, 4}, f_A);
    test.AddInput<MLFloat16>("B", {4, 3}, f_B, true);
    test.AddInput<MLFloat16>("C", {1}, f_C, true);
    test.AddOutput<MLFloat16>("Y", {2, 3}, f_Y);
    test.SetOutputTolerance(0.005f);
    test.ConfigExcludeEps({kTensorrtExecutionProvider})  // TensorRT: fp16 is not supported
        .Config(run_with_tunable_op)
        .RunWithConfig();
  }
}

// Only CUDA, ROCM and CoreML kernels have float 16 support
TEST(GemmOpTest, GemmTransB_f16) {
#ifdef USE_CUDA
  int min_cuda_architecture = 530;
  if (!HasCudaEnvironment(min_cuda_architecture)) {
    LOGS_DEFAULT(WARNING) << "Hardware NOT support FP16";
    return;
  }
#endif

  std::vector<float> A{1.0f, 2.0f, 3.0f, 4.0f,
                       -1.0f, -2.0f, -3.0f, -4.0f};
  std::vector<float> B = {0.5f, 2.1f, 1.2f, -0.3f,
                          -1.2f, 0.2f, 1.0f, -2.1f,
                          1.3f, 4.1f, 1.3f, -8.1f};
  std::vector<float> C = {0.5f, 2.1f, 1.2f,
                          -0.3f, -1.2f, 0.2f};

  std::vector<MLFloat16> f_A(8);
  std::vector<MLFloat16> f_B(12);
  ConvertFloatToMLFloat16(A.data(), f_A.data(), 8);
  ConvertFloatToMLFloat16(B.data(), f_B.data(), 12);
  {
    // bias is a scalar and  transB is True
    std::vector<MLFloat16> f_Y(6);
    std::vector<float> Y{7.6f, -5.7f, -18.5f, -6.6f, 6.7f, 19.5f};
    ConvertFloatToMLFloat16(Y.data(), f_Y.data(), 6);

    std::vector<MLFloat16> f_C(1);
    ConvertFloatToMLFloat16(C.data(), f_C.data(), 1);
    OpTester test("Gemm", 13);

    test.AddAttribute("transA", (int64_t)0);
    test.AddAttribute("transB", (int64_t)1);
    test.AddAttribute("alpha", 1.0f);
    test.AddAttribute("beta", 1.0f);
    test.AddInput<MLFloat16>("A", {2, 4}, f_A);
    test.AddInput<MLFloat16>("B", {3, 4}, f_B, true);
    test.AddInput<MLFloat16>("C", {1}, f_C, true);
    test.AddOutput<MLFloat16>("Y", {2, 3}, f_Y);
    test.SetOutputTolerance(0.005f);
    test.ConfigExcludeEps({kTensorrtExecutionProvider})  // TensorRT: fp16 is not supported
        .Config(run_with_tunable_op)
        .RunWithConfig();
  }
}

#if defined(USE_CUDA) || defined(USE_ROCM) || defined(USE_DNNL)
TEST(GemmOpTest, GemmNoTrans_bfloat16) {
#ifdef USE_CUDA
  int min_cuda_architecture = 530;
  if (!HasCudaEnvironment(min_cuda_architecture)) {
    LOGS_DEFAULT(WARNING) << "Hardware NOT support BFP16";
    return;
  }
#endif
#ifdef USE_DNNL
  if (!DnnlHasBF16Support()) {
    LOGS_DEFAULT(WARNING) << "Hardware does NOT support BF16";
    return;
  }
#endif
  OpTester test("Gemm", 14);
  test.AddAttribute("transA", (int64_t)0);
  test.AddAttribute("transB", (int64_t)0);
  test.AddAttribute("alpha", 1.0f);
  test.AddAttribute("beta", 1.0f);
  test.AddInput<BFloat16>("A", {2, 4}, MakeBFloat16({1.0f, 2.0f, 3.0f, 4.0f, -1.0f, -2.0f, -3.0f, -4.0f}));
  test.AddInput<BFloat16>("B", {4, 3}, MakeBFloat16({1.f, 1.f, 1.f, 1.f, 1.f, 1.f, 1.f, 1.f, 1.f, 1.f, 1.f, 1.f}));
  test.AddInput<BFloat16>("C", {2, 3}, MakeBFloat16({1.f, 1.f, 1.f, 1.f, 1.f, 1.f}));
  test.AddOutput<BFloat16>("Y", {2, 3}, MakeBFloat16({11.0f, 11.0f, 11.0f, -9.0f, -9.0f, -9.0f}));
  std::vector<std::unique_ptr<IExecutionProvider>> execution_providers;
  test.Config(run_with_tunable_op);
#ifdef USE_CUDA
  execution_providers.emplace_back(DefaultCudaExecutionProvider());
#elif USE_ROCM
  execution_providers.emplace_back(DefaultRocmExecutionProvider(/*test_tunable_op=*/true));
  test.ConfigEps(std::move(execution_providers))
      .RunWithConfig();

  execution_providers.clear();
  execution_providers.emplace_back(DefaultRocmExecutionProvider(/*test_tunable_op=*/false));
#elif USE_DNNL
  execution_providers.emplace_back(DefaultDnnlExecutionProvider());
#endif
  test.ConfigEps(std::move(execution_providers))
      .RunWithConfig();
}
#endif  // USE_CUDA USE_RCOM USE_DNNL

#if defined(USE_DNNL)
TEST(GemmOpTest, GemmNaN_bfloat16) {
#ifdef USE_DNNL
  if (!DnnlHasBF16Support()) {
    LOGS_DEFAULT(WARNING) << "Hardware does NOT support BF16";
    return;
  }
#endif
  OpTester test("Gemm", 14);

  test.AddAttribute("transA", (int64_t)0);
  test.AddAttribute("transB", (int64_t)0);
  test.AddAttribute("alpha", 1.0f);
  test.AddAttribute("beta", 0.0f);
  test.AddInput<BFloat16>("A", {2, 4},
                          MakeBFloat16({1.0f, 2.0f, 3.0f, 4.0f,
                                        -1.0f, -2.0f, -3.0f, -4.0f}));
  test.AddInput<BFloat16>("B", {4, 3},
                          MakeBFloat16({1.0f, 1.0f, 1.0f, 1.0f,
                                        1.0f, 1.0f, 1.0f, 1.0f,
                                        1.0f, 1.0f, 1.0f, 1.0f}));
  test.AddInput<BFloat16>("C", {2, 3}, MakeBFloat16({1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f}));
  test.AddOutput<BFloat16>("Y", {2, 3},
                           MakeBFloat16({10.0f, 10.0f, 10.0f,
                                         -10.0f, -10.0f, -10.0f}));
  std::vector<std::unique_ptr<IExecutionProvider>> execution_providers;
#if defined(USE_DNNL)
  execution_providers.push_back(DefaultDnnlExecutionProvider());
#endif                                                                                                                //  USE_DNNL
  test.Run(OpTester::ExpectResult::kExpectSuccess, "", {kTensorrtExecutionProvider}, nullptr, &execution_providers);  // TensorRT: Seg fault in parser
}
#endif  //  USE_DNNL

#if defined(USE_DNNL)
TEST(GemmOpTest, GemmScalarBroadcast_bfloat16) {
#ifdef USE_DNNL
  if (!DnnlHasBF16Support()) {
    LOGS_DEFAULT(WARNING) << "Hardware does NOT support BF16";
    return;
  }
#endif
  OpTester test("Gemm", 14);

  test.AddAttribute("transA", (int64_t)0);
  test.AddAttribute("transB", (int64_t)0);
  test.AddAttribute("alpha", 1.0f);
  test.AddInput<BFloat16>("A", {2, 4},
                          MakeBFloat16({1.0f, 2.0f, 3.0f, 4.0f,
                                        -1.0f, -2.0f, -3.0f, -4.0f}));
  test.AddInput<BFloat16>("B", {4, 3},
                          MakeBFloat16({1.0f, 1.0f, 1.0f, 1.0f,
                                        1.0f, 1.0f, 1.0f, 1.0f,
                                        1.0f, 1.0f, 1.0f, 1.0f}));
  test.AddInput<BFloat16>("C", {1}, MakeBFloat16({1.0f}));
  test.AddOutput<BFloat16>("Y", {2, 3},
                           MakeBFloat16({11.0f, 11.0f, 11.0f,
                                         -9.0f, -9.0f, -9.0f}));
  std::vector<std::unique_ptr<IExecutionProvider>> execution_providers;
#if defined(USE_DNNL)
  execution_providers.push_back(DefaultDnnlExecutionProvider());
#endif                                                                                                                //  USE_DNNL
  test.Run(OpTester::ExpectResult::kExpectSuccess, "", {kTensorrtExecutionProvider}, nullptr, &execution_providers);  // TensorRT: Seg fault in parser
}
#endif  //  USE_DNNL

#if defined(USE_DNNL)
TEST(GemmOpTest, Gemm2DBroadcast_1_bfloat16) {
#ifdef USE_DNNL
  if (!DnnlHasBF16Support()) {
    LOGS_DEFAULT(WARNING) << "Hardware does NOT support BF16";
    return;
  }
#endif
  OpTester test("Gemm", 14);

  test.AddAttribute("transA", (int64_t)0);
  test.AddAttribute("transB", (int64_t)0);
  test.AddAttribute("alpha", 1.0f);
  test.AddInput<BFloat16>("A", {2, 4},
                          MakeBFloat16({1.0f, 2.0f, 3.0f, 4.0f,
                                        -1.0f, -2.0f, -3.0f, -4.0f}));
  test.AddInput<BFloat16>("B", {4, 3},
                          MakeBFloat16({1.0f, 1.0f, 1.0f, 1.0f,
                                        1.0f, 1.0f, 1.0f, 1.0f,
                                        1.0f, 1.0f, 1.0f, 1.0f}));
  test.AddInput<BFloat16>("C", {2, 1}, MakeBFloat16({1.0f, 2.0f}));
  test.AddOutput<BFloat16>("Y", {2, 3},
                           MakeBFloat16({11.0f, 11.0f, 11.0f,
                                         -8.0f, -8.0f, -8.0f}));
  std::vector<std::unique_ptr<IExecutionProvider>> execution_providers;
#if defined(USE_DNNL)
  execution_providers.push_back(DefaultDnnlExecutionProvider());
#endif                                                                                                                //  USE_DNNL
  test.Run(OpTester::ExpectResult::kExpectSuccess, "", {kTensorrtExecutionProvider}, nullptr, &execution_providers);  // TensorRT: Seg fault in parser
}
#endif  //  USE_DNNL

#if defined(USE_DNNL)
TEST(GemmOpTest, Gemm2DBroadcast_2_bfloat16) {
#ifdef USE_DNNL
  if (!DnnlHasBF16Support()) {
    LOGS_DEFAULT(WARNING) << "Hardware does NOT support BF16";
    return;
  }
#endif
  OpTester test("Gemm", 14);

  test.AddAttribute("transA", (int64_t)0);
  test.AddAttribute("transB", (int64_t)0);
  test.AddAttribute("alpha", 1.0f);
  test.AddInput<BFloat16>("A", {2, 4},
                          MakeBFloat16({1.0f, 2.0f, 3.0f, 4.0f,
                                        -1.0f, -2.0f, -3.0f, -4.0f}));
  test.AddInput<BFloat16>("B", {4, 3},
                          MakeBFloat16({1.0f, 1.0f, 1.0f, 1.0f,
                                        1.0f, 1.0f, 1.0f, 1.0f,
                                        1.0f, 1.0f, 1.0f, 1.0f}));
  test.AddInput<BFloat16>("C", {1, 3}, MakeBFloat16({1.0f, 2.0f, 3.0f}));
  test.AddOutput<BFloat16>("Y", {2, 3},
                           MakeBFloat16({11.0f, 12.0f, 13.0f,
                                         -9.0f, -8.0f, -7.0f}));
  std::vector<std::unique_ptr<IExecutionProvider>> execution_providers;
#if defined(USE_DNNL)
  execution_providers.push_back(DefaultDnnlExecutionProvider());
#endif                                                                                                                //  USE_DNNL
  test.Run(OpTester::ExpectResult::kExpectSuccess, "", {kTensorrtExecutionProvider}, nullptr, &execution_providers);  // TensorRT: Seg fault in parser
}
#endif  //  USE_DNNL

#if defined(USE_DNNL)
TEST(GemmOpTest, GemmFalseBroadcast_2_bfloat16) {
#ifdef USE_DNNL
  if (!DnnlHasBF16Support()) {
    LOGS_DEFAULT(WARNING) << "Hardware does NOT support BF16";
    return;
  }
#endif
  OpTester test("Gemm", 14);

  test.AddAttribute("transA", (int64_t)0);
  test.AddAttribute("transB", (int64_t)0);
  test.AddAttribute("alpha", 1.0f);
  test.AddInput<BFloat16>("A", {2, 4},
                          MakeBFloat16({1.0f, 2.0f, 3.0f, 4.0f,
                                        -1.0f, -2.0f, -3.0f, -4.0f}));
  test.AddInput<BFloat16>("B", {4, 3},
                          MakeBFloat16({1.0f, 1.0f, 1.0f, 1.0f,
                                        1.0f, 1.0f, 1.0f, 1.0f,
                                        1.0f, 1.0f, 1.0f, 1.0f}));
  test.AddInput<BFloat16>("C", {2, 3}, MakeBFloat16({1.0f, 1.0f, 1.0f, 2.0f, 2.0f, 2.0f}));
  test.AddOutput<BFloat16>("Y", {2, 3},
                           MakeBFloat16({11.0f, 11.0f, 11.0f,
                                         -8.0f, -8.0f, -8.0f}));
  std::vector<std::unique_ptr<IExecutionProvider>> execution_providers;
#if defined(USE_DNNL)
  execution_providers.push_back(DefaultDnnlExecutionProvider());
#endif  //  USE_DNNL
  test.Run(OpTester::ExpectResult::kExpectSuccess, "", {}, nullptr, &execution_providers);
}
#endif  //  USE_DNNL

template <typename T>
class GemmOpTypedTests : public ::testing::Test {
};

// On CPUs without fp16 instructions the tests will output a warning:
// "registered execution providers CPUExecutionProvider were unable to run the model"
// , then they will still pass.
using GemmOpTypedTestsTypes = ::testing::Types<float, double>;
TYPED_TEST_SUITE(GemmOpTypedTests, GemmOpTypedTestsTypes);

TYPED_TEST(GemmOpTypedTests, TestGemmScalarBroadcast) {
  auto run_test = [](bool b_is_initializer, bool c_is_initializer) {
    OpTester test("Gemm");

    test.AddAttribute("transA", (int64_t)0);
    test.AddAttribute("transB", (int64_t)0);
    test.AddAttribute("alpha", 1.0f);
    test.AddAttribute("beta", 1.0f);

    test.AddInput<TypeParam>("A", {2, 4},
                             {static_cast<TypeParam>(1.0f), static_cast<TypeParam>(2.0f), static_cast<TypeParam>(3.0f), static_cast<TypeParam>(4.0f),
                              static_cast<TypeParam>(-1.0f), static_cast<TypeParam>(-2.0f), static_cast<TypeParam>(-3.0f), static_cast<TypeParam>(-4.0f)});
    test.AddInput<TypeParam>("B", {4, 3}, std::vector<TypeParam>(12, static_cast<TypeParam>(1.0f)), b_is_initializer);
    test.AddInput<TypeParam>("C", {1}, std::vector<TypeParam>{static_cast<TypeParam>(1.0f)}, c_is_initializer);
    test.AddOutput<TypeParam>("Y", {2, 3},
                              {static_cast<TypeParam>(11.0f), static_cast<TypeParam>(11.0f), static_cast<TypeParam>(11.0f),
                               static_cast<TypeParam>(-9.0f), static_cast<TypeParam>(-9.0f), static_cast<TypeParam>(-9.0f)});
    test.Config(run_with_tunable_op)
        .RunWithConfig();
  };

  run_test(false, false);
  // CoreML EP requires weight and bias to be initializers
  run_test(true, true);
}

TYPED_TEST(GemmOpTypedTests, TestGemm2DBroadcast_2) {
  OpTester test("Gemm");

  test.AddAttribute("transA", (int64_t)0);
  test.AddAttribute("transB", (int64_t)0);
  test.AddAttribute("alpha", 1.0f);
  test.AddAttribute("beta", 1.0f);

  // Same as GemmBroadcast, but adding the unnecessary second dimension.
  test.AddInput<TypeParam>("A", {2, 4},
                           {static_cast<TypeParam>(1.0f), static_cast<TypeParam>(2.0f), static_cast<TypeParam>(3.0f), static_cast<TypeParam>(4.0f),
                            static_cast<TypeParam>(-1.0f), static_cast<TypeParam>(-2.0f), static_cast<TypeParam>(-3.0f), static_cast<TypeParam>(-4.0f)});
  test.AddInput<TypeParam>("B", {4, 3}, std::vector<TypeParam>(12, static_cast<TypeParam>(1.0f)));
  test.AddInput<TypeParam>("C", {1, 3}, std::vector<TypeParam>{static_cast<TypeParam>(1.0f), static_cast<TypeParam>(2.0f), static_cast<TypeParam>(3.0f)});
  test.AddOutput<TypeParam>("Y", {2, 3},
                            {static_cast<TypeParam>(11.0f), static_cast<TypeParam>(12.0f), static_cast<TypeParam>(13.0f),
                             static_cast<TypeParam>(-9.0f), static_cast<TypeParam>(-8.0f), static_cast<TypeParam>(-7.0f)});
  test.Config(run_with_tunable_op)
      .ConfigExcludeEps({kQnnExecutionProvider})  // Accuracy issues with QNN CPU backend since QNN 2.34
      .RunWithConfig();
}

TYPED_TEST(GemmOpTypedTests, TestGemm2DBroadcast_3) {
  OpTester test("Gemm");

  test.AddAttribute("transA", (int64_t)0);
  test.AddAttribute("transB", (int64_t)0);
  test.AddAttribute("alpha", 1.0f);
  test.AddAttribute("beta", 1.0f);

  // Same as GemmBroadcast, but adding the unnecessary first dimension.
  test.AddInput<TypeParam>("A", {3, 4},
                           std::vector<TypeParam>(12, static_cast<TypeParam>(1.0f)));
  test.AddInput<TypeParam>("B", {4, 4}, std::vector<TypeParam>(16, static_cast<TypeParam>(1.0f)));
  test.AddInput<TypeParam>("C", {1, 1}, std::vector<TypeParam>{static_cast<TypeParam>(1.0f)});
  test.AddOutput<TypeParam>("Y", {3, 4},
                            std::vector<TypeParam>{static_cast<TypeParam>(5.0f), static_cast<TypeParam>(5.0f), static_cast<TypeParam>(5.0f), static_cast<TypeParam>(5.0f),
                                                   static_cast<TypeParam>(5.0f), static_cast<TypeParam>(5.0f), static_cast<TypeParam>(5.0f), static_cast<TypeParam>(5.0f),
                                                   static_cast<TypeParam>(5.0f), static_cast<TypeParam>(5.0f), static_cast<TypeParam>(5.0f), static_cast<TypeParam>(5.0f)});
  test.Config(run_with_tunable_op)
      .RunWithConfig();
}

TYPED_TEST(GemmOpTypedTests, TestGemm2DBroadcast_4) {
  OpTester test("Gemm");

  test.AddAttribute("transA", (int64_t)0);
  test.AddAttribute("transB", (int64_t)0);
  test.AddAttribute("alpha", 1.0f);
  test.AddAttribute("beta", 1.0f);

  // Same as GemmBroadcast, but adding the unnecessary first dimension.
  test.AddInput<TypeParam>("A", {3, 4},
                           std::vector<TypeParam>(12, static_cast<TypeParam>(1.0f)));
  test.AddInput<TypeParam>("B", {4, 4}, std::vector<TypeParam>(16, static_cast<TypeParam>(1.0f)));
  test.AddInput<TypeParam>("C", {3, 1}, std::vector<TypeParam>{static_cast<TypeParam>(1.0f), static_cast<TypeParam>(2.0f), static_cast<TypeParam>(3.0f)});
  test.AddOutput<TypeParam>("Y", {3, 4},
                            std::vector<TypeParam>{static_cast<TypeParam>(5.0f), static_cast<TypeParam>(5.0f), static_cast<TypeParam>(5.0f), static_cast<TypeParam>(5.0f),
                                                   static_cast<TypeParam>(6.0f), static_cast<TypeParam>(6.0f), static_cast<TypeParam>(6.0f), static_cast<TypeParam>(6.0f),
                                                   static_cast<TypeParam>(7.0f), static_cast<TypeParam>(7.0f), static_cast<TypeParam>(7.0f), static_cast<TypeParam>(7.0f)});
  test.Config(run_with_tunable_op)
      .RunWithConfig();
}

TYPED_TEST(GemmOpTypedTests, TestGemmFalseBroadcast) {
  OpTester test("Gemm");

  test.AddAttribute("transA", (int64_t)0);
  test.AddAttribute("transB", (int64_t)0);
  test.AddAttribute("alpha", 1.0f);
  test.AddAttribute("beta", 1.0f);

  test.AddInput<TypeParam>("A", {2, 4},
                           {static_cast<TypeParam>(1.0f), static_cast<TypeParam>(2.0f), static_cast<TypeParam>(3.0f), static_cast<TypeParam>(4.0f),
                            static_cast<TypeParam>(-1.0f), static_cast<TypeParam>(-2.0f), static_cast<TypeParam>(-3.0f), static_cast<TypeParam>(-4.0f)});
  test.AddInput<TypeParam>("B", {4, 3}, std::vector<TypeParam>(12, static_cast<TypeParam>(1.0f)));
  test.AddInput<TypeParam>("C", {2, 3}, std::vector<TypeParam>{static_cast<TypeParam>(1.0f), static_cast<TypeParam>(1.0f), static_cast<TypeParam>(1.0f), static_cast<TypeParam>(2.0f), static_cast<TypeParam>(2.0f), static_cast<TypeParam>(2.0f)});
  test.AddOutput<TypeParam>("Y", {2, 3},
                            {static_cast<TypeParam>(11.0f), static_cast<TypeParam>(11.0f), static_cast<TypeParam>(11.0f),
                             static_cast<TypeParam>(-8.0f), static_cast<TypeParam>(-8.0f), static_cast<TypeParam>(-8.0f)});
  test.Config(run_with_tunable_op)
      .RunWithConfig();
}

TYPED_TEST(GemmOpTypedTests, TestGemmBroadcast) {
  auto run_test = [](bool b_is_initializer, bool c_is_initializer) {
    OpTester test("Gemm");

    test.AddAttribute("transA", (int64_t)0);
    test.AddAttribute("transB", (int64_t)0);
    test.AddAttribute("alpha", 1.0f);
    test.AddAttribute("beta", 1.0f);

    test.AddInput<TypeParam>("A", {2, 4},
                             {static_cast<TypeParam>(1.0f), static_cast<TypeParam>(2.0f), static_cast<TypeParam>(3.0f), static_cast<TypeParam>(4.0f),
                              static_cast<TypeParam>(-1.0f), static_cast<TypeParam>(-2.0f), static_cast<TypeParam>(-3.0f), static_cast<TypeParam>(-4.0f)});
    test.AddInput<TypeParam>("B", {4, 3}, std::vector<TypeParam>(12, static_cast<TypeParam>(1.0f)), b_is_initializer);
    test.AddInput<TypeParam>("C", {3}, std::vector<TypeParam>{static_cast<TypeParam>(1.0f), static_cast<TypeParam>(2.0f), static_cast<TypeParam>(3.0f)}, c_is_initializer);
    test.AddOutput<TypeParam>("Y", {2, 3},
                              {static_cast<TypeParam>(11.0f), static_cast<TypeParam>(12.0f), static_cast<TypeParam>(13.0f),
                               static_cast<TypeParam>(-9.0f), static_cast<TypeParam>(-8.0f), static_cast<TypeParam>(-7.0f)});

    std::unordered_set<std::string> excluded_providers;
#if defined(OPENVINO_CONFIG_GPU)
    excluded_providers.insert(kOpenVINOExecutionProvider);  // OpenVINO: Temporarily disabled due to accuracy issues
#endif

    // Accuracy issues with QNN CPU backend since QNN 2.34
    excluded_providers.insert(kQnnExecutionProvider);

    test.ConfigExcludeEps(excluded_providers)
        .Config(run_with_tunable_op)
        .RunWithConfig();
  };

  run_test(false, false);
  // NNAPI EP requires weight to be an initializer
  run_test(true, false);
  // CoreML EP requires weight and bias both to be initializers
  run_test(true, true);
}

TYPED_TEST(GemmOpTypedTests, TestGemmTrans) {
  OpTester test("Gemm");

  test.AddAttribute("transA", (int64_t)1);
  test.AddAttribute("transB", (int64_t)1);
  test.AddAttribute("alpha", 1.0f);
  test.AddAttribute("beta", 1.0f);

  test.AddInput<TypeParam>("A", {4, 2},
                           {static_cast<TypeParam>(1.0f), static_cast<TypeParam>(-1.0f),
                            static_cast<TypeParam>(2.0f), static_cast<TypeParam>(-2.0f),
                            static_cast<TypeParam>(3.0f), static_cast<TypeParam>(-3.0f),
                            static_cast<TypeParam>(4.0f), static_cast<TypeParam>(-4.0f)});
  test.AddInput<TypeParam>("B", {3, 4}, std::vector<TypeParam>(12, static_cast<TypeParam>(1.0f)));
  test.AddInput<TypeParam>("C", {3}, std::vector<TypeParam>(3, static_cast<TypeParam>(1.0f)));
  test.AddOutput<TypeParam>("Y", {2, 3},
                            {static_cast<TypeParam>(11.0f), static_cast<TypeParam>(11.0f), static_cast<TypeParam>(11.0f),
                             static_cast<TypeParam>(-9.0f), static_cast<TypeParam>(-9.0f), static_cast<TypeParam>(-9.0f)});

  std::unordered_set<std::string> excluded_providers;
#if defined(OPENVINO_CONFIG_GPU)
  excluded_providers.insert(kOpenVINOExecutionProvider);  // OpenVINO: Temporarily disabled due to accuracy issues
#endif
  // Accuracy issues with QNN CPU backend since QNN 2.34
  excluded_providers.insert(kQnnExecutionProvider);

  test.ConfigExcludeEps(excluded_providers)
      .Config(run_with_tunable_op)
      .RunWithConfig();
}

// NNAPI EP's GEMM only works as A*B', add case only B is transposed
// Also test NNAPI EP's handling of non-1D bias (C of Gemm)
TYPED_TEST(GemmOpTypedTests, TestGemmTransB) {
  auto run_test = [](bool b_is_initializer, bool c_is_initializer = false) {
    OpTester test("Gemm");

    test.AddAttribute("transA", (int64_t)0);
    test.AddAttribute("transB", (int64_t)1);
    test.AddAttribute("alpha", 1.0f);
    test.AddAttribute("beta", 1.0f);

    test.AddInput<TypeParam>("A", {2, 4},
                             {static_cast<TypeParam>(1.0f), static_cast<TypeParam>(2.0f), static_cast<TypeParam>(3.0f), static_cast<TypeParam>(4.0f),
                              static_cast<TypeParam>(-1.0f), static_cast<TypeParam>(-2.0f), static_cast<TypeParam>(-3.0f), static_cast<TypeParam>(-4.0f)});
    test.AddInput<TypeParam>("B", {3, 4}, std::vector<TypeParam>(12, static_cast<TypeParam>(1.0f)), b_is_initializer);
    test.AddInput<TypeParam>("C", {1, 3}, std::vector<TypeParam>(3, static_cast<TypeParam>(1.0f)), c_is_initializer);
    test.AddOutput<TypeParam>("Y", {2, 3},
                              {static_cast<TypeParam>(11.0f), static_cast<TypeParam>(11.0f), static_cast<TypeParam>(11.0f),
                               static_cast<TypeParam>(-9.0f), static_cast<TypeParam>(-9.0f), static_cast<TypeParam>(-9.0f)});

    std::unordered_set<std::string> excluded_providers;
#if defined(OPENVINO_CONFIG_GPU)
    excluded_providers.insert(kOpenVINOExecutionProvider);  // OpenVINO: Temporarily disabled due to accuracy issues
#endif
    excluded_providers.insert(kQnnExecutionProvider);  // Accuracy issues with QNN CPU backend since QNN 2.34

    test.ConfigExcludeEps(excluded_providers)
        .Config(run_with_tunable_op)
        .RunWithConfig();
  };
  run_test(false, false);
  // CoreML EP requires weight and bias both to be initializers
  run_test(true, true);
}

// NNAPI EP's GEMM only works as A*B', add case only B is transposed
// Also test NNAPI EP's handling of non-1D bias (C of Gemm) which is broadcastable but not valid for NNAPI
TYPED_TEST(GemmOpTypedTests, TestGemmTransB_1) {
  auto run_test = [](bool b_is_initializer, bool c_is_initializer = false) {
    OpTester test("Gemm");

    test.AddAttribute("transA", (int64_t)0);
    test.AddAttribute("transB", (int64_t)1);
    test.AddAttribute("alpha", 1.0f);
    test.AddAttribute("beta", 1.0f);

    test.AddInput<TypeParam>("A", {2, 4},
                             {static_cast<TypeParam>(1.0f), static_cast<TypeParam>(2.0f), static_cast<TypeParam>(3.0f), static_cast<TypeParam>(4.0f),
                              static_cast<TypeParam>(-1.0f), static_cast<TypeParam>(-2.0f), static_cast<TypeParam>(-3.0f), static_cast<TypeParam>(-4.0f)});
    test.AddInput<TypeParam>("B", {3, 4}, std::vector<TypeParam>(12, static_cast<TypeParam>(1.0f)), b_is_initializer);
    test.AddInput<TypeParam>("C", {2, 1}, std::vector<TypeParam>(2, static_cast<TypeParam>(1.0f)), c_is_initializer);
    test.AddOutput<TypeParam>("Y", {2, 3},
                              {static_cast<TypeParam>(11.0f), static_cast<TypeParam>(11.0f), static_cast<TypeParam>(11.0f),
                               static_cast<TypeParam>(-9.0f), static_cast<TypeParam>(-9.0f), static_cast<TypeParam>(-9.0f)});
#if defined(OPENVINO_CONFIG_GPU)
    test.ConfigExcludeEps({kOpenVINOExecutionProvider});  // OpenVINO: Temporarily disabled due to accuracy issues
#endif
    test.Config(run_with_tunable_op)
        .RunWithConfig();
  };
  run_test(false, false);
  // CoreML EP requires weight and bias both to be initializers
  run_test(true, true);
}

TYPED_TEST(GemmOpTypedTests, TestGemmAlpha) {
  OpTester test("Gemm");

  test.AddAttribute("transA", (int64_t)0);
  test.AddAttribute("transB", (int64_t)0);
  test.AddAttribute("alpha", 0.5f);
  test.AddAttribute("beta", 1.0f);

  test.AddInput<TypeParam>("A", {2, 4},
                           {static_cast<TypeParam>(1.0f), static_cast<TypeParam>(2.0f), static_cast<TypeParam>(3.0f), static_cast<TypeParam>(4.0f),
                            static_cast<TypeParam>(-1.0f), static_cast<TypeParam>(-2.0f), static_cast<TypeParam>(-3.0f), static_cast<TypeParam>(-4.0f)});
  test.AddInput<TypeParam>("B", {4, 3}, std::vector<TypeParam>(12, static_cast<TypeParam>(1.0f)));
  test.AddInput<TypeParam>("C", {3}, std::vector<TypeParam>(3, static_cast<TypeParam>(1.0f)));
  test.AddOutput<TypeParam>("Y", {2, 3},
                            {static_cast<TypeParam>(6.0f), static_cast<TypeParam>(6.0f), static_cast<TypeParam>(6.0f),
                             static_cast<TypeParam>(-4.0f), static_cast<TypeParam>(-4.0f), static_cast<TypeParam>(-4.0f)});
  // test.AddOutput<TypeParam>("Y", {2, 3},
  //                   {5.0f, 5.0f, 5.0f,
  //                    -5.0f, -5.0f, -5.0f});
#if defined(OPENVINO_CONFIG_GPU)
  test.ConfigExcludeEps({kOpenVINOExecutionProvider});  // OpenVINO: Temporarily disabled due to accuracy issues
#else
  test.ConfigExcludeEps({kTensorrtExecutionProvider});  // TensorRT: Seg fault in parser
#endif
  test.Config(run_with_tunable_op)
      .RunWithConfig();
}

TYPED_TEST(GemmOpTypedTests, TestGemmBeta) {
  OpTester test("Gemm");

  test.AddAttribute("transA", (int64_t)0);
  test.AddAttribute("transB", (int64_t)0);
  test.AddAttribute("alpha", 1.0f);
  test.AddAttribute("beta", 2.0f);

  test.AddInput<TypeParam>("A", {2, 4},
                           {static_cast<TypeParam>(1.0f), static_cast<TypeParam>(2.0f), static_cast<TypeParam>(3.0f), static_cast<TypeParam>(4.0f),
                            static_cast<TypeParam>(-1.0f), static_cast<TypeParam>(-2.0f), static_cast<TypeParam>(-3.0f), static_cast<TypeParam>(-4.0f)});
  test.AddInput<TypeParam>("B", {4, 3}, std::vector<TypeParam>(12, static_cast<TypeParam>(1.0f)));
  test.AddInput<TypeParam>("C", {3}, std::vector<TypeParam>(3, static_cast<TypeParam>(1.0f)));
  test.AddOutput<TypeParam>("Y", {2, 3},
                            {static_cast<TypeParam>(12.0f), static_cast<TypeParam>(12.0f), static_cast<TypeParam>(12.0f),
                             static_cast<TypeParam>(-8.0f), static_cast<TypeParam>(-8.0f), static_cast<TypeParam>(-8.0f)});
#if defined(OPENVINO_CONFIG_GPU)
  test.ConfigExcludeEps({kOpenVINOExecutionProvider});  // OpenVINO: Temporarily disabled due to accuracy issues
#else
  test.ConfigExcludeEps({kTensorrtExecutionProvider});  // TensorRT: Seg fault in parser
#endif
  test.Config(run_with_tunable_op)
      .RunWithConfig();
}

TYPED_TEST(GemmOpTypedTests, TestGemmNaN) {
  OpTester test("Gemm");

  test.AddAttribute("transA", (int64_t)0);
  test.AddAttribute("transB", (int64_t)0);
  test.AddAttribute("alpha", 1.0f);
  test.AddAttribute("beta", 0.0f);

  test.AddInput<TypeParam>("A", {2, 4},
                           {static_cast<TypeParam>(1.0f), static_cast<TypeParam>(2.0f), static_cast<TypeParam>(3.0f), static_cast<TypeParam>(4.0f),
                            static_cast<TypeParam>(-1.0f), static_cast<TypeParam>(-2.0f), static_cast<TypeParam>(-3.0f), static_cast<TypeParam>(-4.0f)});
  test.AddInput<TypeParam>("B", {4, 3}, std::vector<TypeParam>(12, static_cast<TypeParam>(1.0f)));
  test.AddInput<TypeParam>("C", {2, 3}, std::vector<TypeParam>(6, static_cast<TypeParam>(1.0f)));
  test.AddOutput<TypeParam>("Y", {2, 3},
                            {static_cast<TypeParam>(10.0f), static_cast<TypeParam>(10.0f), static_cast<TypeParam>(10.0f),
                             static_cast<TypeParam>(-10.0f), static_cast<TypeParam>(-10.0f), static_cast<TypeParam>(-10.0f)});

  // TensorRT: Seg fault in parser
  test.ConfigExcludeEps({kTensorrtExecutionProvider})
      .Config(run_with_tunable_op)
      .RunWithConfig();
}

TYPED_TEST(GemmOpTypedTests, TestGemmAlphaBeta) {
  OpTester test("Gemm", 13);

  test.AddAttribute("transA", (int64_t)0);
  test.AddAttribute("transB", (int64_t)0);
  test.AddAttribute("alpha", 0.5f);
  test.AddAttribute("beta", 2.0f);

  test.AddInput<TypeParam>("A", {2, 4},
                           {static_cast<TypeParam>(1.0f), static_cast<TypeParam>(2.0f), static_cast<TypeParam>(3.0f), static_cast<TypeParam>(4.0f),
                            static_cast<TypeParam>(-1.0f), static_cast<TypeParam>(-2.0f), static_cast<TypeParam>(-3.0f), static_cast<TypeParam>(-4.0f)});
  test.AddInput<TypeParam>("B", {4, 3}, std::vector<TypeParam>(12, static_cast<TypeParam>(1.0f)));
  test.AddInput<TypeParam>("C", {3}, std::vector<TypeParam>(3, static_cast<TypeParam>(1.0f)));
  test.AddOutput<TypeParam>("Y", {2, 3},
                            {static_cast<TypeParam>(7.0f), static_cast<TypeParam>(7.0f), static_cast<TypeParam>(7.0f),
                             static_cast<TypeParam>(-3.0f), static_cast<TypeParam>(-3.0f), static_cast<TypeParam>(-3.0f)});
#if defined(OPENVINO_CONFIG_GPU)
  test.ConfigExcludeEps({kOpenVINOExecutionProvider});  // OpenVINO: Temporarily disabled due to accuracy issues
#else
  test.ConfigExcludeEps({kTensorrtExecutionProvider});  // TensorRT: Seg fault in parser
#endif
  test.Config(run_with_tunable_op)
      .RunWithConfig();
}

// C is 1D
TYPED_TEST(GemmOpTypedTests, TestGemm2DBroadcast_1) {
  OpTester test("Gemm");

  test.AddAttribute("transA", (int64_t)0);
  test.AddAttribute("transB", (int64_t)0);
  test.AddAttribute("alpha", 1.0f);
  test.AddAttribute("beta", 1.0f);
  std::array<TypeParam, 8> a_data{static_cast<TypeParam>(1.0f), static_cast<TypeParam>(2.0f), static_cast<TypeParam>(3.0f), static_cast<TypeParam>(4.0f),
                                  static_cast<TypeParam>(-1.0f), static_cast<TypeParam>(-2.0f), static_cast<TypeParam>(-3.0f), static_cast<TypeParam>(-4.0f)};

  test.AddInput<TypeParam>("A", {2, 4}, a_data.data(), a_data.size());
  test.AddInput<TypeParam>("B", {4, 3}, std::vector<TypeParam>(12, static_cast<TypeParam>(1.0f)));
  test.AddInput<TypeParam>("C", {2, 1}, std::vector<TypeParam>{static_cast<TypeParam>(1.0f), static_cast<TypeParam>(2.0f)});
  test.AddOutput<TypeParam>("Y", {2, 3},
                            {static_cast<TypeParam>(11.0f), static_cast<TypeParam>(11.0f), static_cast<TypeParam>(11.0f),
                             static_cast<TypeParam>(-8.0f), static_cast<TypeParam>(-8.0f), static_cast<TypeParam>(-8.0f)});
  test.Config(run_with_tunable_op)
      .RunWithConfig();
}

TYPED_TEST(GemmOpTypedTests, TestGemmNoTrans) {
  auto run_test = [](bool b_is_initializer, bool c_is_initializer = false) {
    OpTester test("Gemm", 13);

    test.AddAttribute("transA", (int64_t)0);
    test.AddAttribute("transB", (int64_t)0);
    test.AddAttribute("alpha", 1.0f);
    test.AddAttribute("beta", 1.0f);

    std::array<TypeParam, 8> a_data{static_cast<TypeParam>(1.0f), static_cast<TypeParam>(2.0f), static_cast<TypeParam>(3.0f), static_cast<TypeParam>(4.0f),
                                    static_cast<TypeParam>(-1.0f), static_cast<TypeParam>(-2.0f), static_cast<TypeParam>(-3.0f), static_cast<TypeParam>(-4.0f)};

    test.AddInput<TypeParam>("A", {2, 4}, a_data.data(), a_data.size());
    test.AddInput<TypeParam>("B", {4, 3}, std::vector<TypeParam>(12, static_cast<TypeParam>(1.0f)), b_is_initializer);
    test.AddInput<TypeParam>("C", {2, 3}, std::vector<TypeParam>(6, static_cast<TypeParam>(1.0f)), c_is_initializer);
    test.AddOutput<TypeParam>("Y", {2, 3},
                              {static_cast<TypeParam>(11.0f), static_cast<TypeParam>(11.0f), static_cast<TypeParam>(11.0f),
                               static_cast<TypeParam>(-9.0f), static_cast<TypeParam>(-9.0f), static_cast<TypeParam>(-9.0f)});
    test.Config(run_with_tunable_op)
        .RunWithConfig();
  };

  run_test(false, false);
  // NNAPI EP requires weight to be an initializer
  run_test(true, false);
  // CoreML EP requires weight and bias both to be initializers
  run_test(true, true);
}
TYPED_TEST(GemmOpTypedTests, GemmEmptyTensor) {
  OpTester test("Gemm", 13);

  test.AddAttribute("transA", static_cast<int64_t>(0));
  test.AddAttribute("transB", static_cast<int64_t>(0));
  test.AddAttribute("alpha", 1.0f);
  test.AddAttribute("beta", 1.0f);

  test.AddInput<TypeParam>("A", {0, 4},
                           {});
  test.AddInput<TypeParam>("B", {4, 3}, std::vector<TypeParam>(12, static_cast<TypeParam>(1.0f)));
  test.AddInput<TypeParam>("C", {3}, std::vector<TypeParam>(3, static_cast<TypeParam>(1.0f)));
  test.AddOutput<TypeParam>("Y", {0, 3},
                            {});
  // TensorRT: doesn't support dynamic shape yet
  test.ConfigExcludeEps({kTensorrtExecutionProvider, kDnnlExecutionProvider, kQnnExecutionProvider})
      .Config(run_with_tunable_op)
      .RunWithConfig();
}

TYPED_TEST(GemmOpTypedTests, ZeroKWithBias) {
  OpTester test("Gemm", 13);

  test.AddAttribute("transA", static_cast<int64_t>(0));
  test.AddAttribute("transB", static_cast<int64_t>(0));
  test.AddAttribute("alpha", 1.0f);
  test.AddAttribute("beta", 1.0f);

  test.AddInput<TypeParam>("A", {4, 0}, {});
  test.AddInput<TypeParam>("B", {0, 4}, {});
  test.AddInput<TypeParam>("C", {4}, std::vector<TypeParam>(4, static_cast<TypeParam>(1.0f)));
  test.AddOutput<TypeParam>("Y", {4, 4}, std::vector<TypeParam>(16, static_cast<TypeParam>(1.0f)));

  test.ConfigExcludeEps({kCoreMLExecutionProvider, kNnapiExecutionProvider,
                         kDmlExecutionProvider, kDnnlExecutionProvider, kQnnExecutionProvider,
                         kOpenVINOExecutionProvider})
      .Config(run_with_tunable_op)
      .RunWithConfig();
}

TYPED_TEST(GemmOpTypedTests, ZeroKWithNoBias) {
  OpTester test("Gemm", 13);

  test.AddAttribute("transA", static_cast<int64_t>(0));
  test.AddAttribute("transB", static_cast<int64_t>(0));
  test.AddAttribute("alpha", 1.0f);
  test.AddAttribute("beta", .0f);

  test.AddInput<TypeParam>("A", {4, 0}, {});
  test.AddInput<TypeParam>("B", {0, 4}, {});
  test.AddOutput<TypeParam>("Y", {4, 4}, std::vector<TypeParam>(16, static_cast<TypeParam>(0.0f)));

  test.ConfigExcludeEps({kCoreMLExecutionProvider, kNnapiExecutionProvider,
                         kDmlExecutionProvider, kDnnlExecutionProvider, kQnnExecutionProvider,
                         kOpenVINOExecutionProvider})
      .Config(run_with_tunable_op)
      .RunWithConfig();
}

TYPED_TEST(GemmOpTypedTests, MissingBias) {
  OpTester test("Gemm", 11);

  test.AddAttribute("transA", static_cast<int64_t>(0));
  test.AddAttribute("transB", static_cast<int64_t>(0));
  test.AddAttribute("alpha", 1.0f);
  test.AddAttribute("beta", 1.0f);

  test.AddInput<TypeParam>("A", {2, 4},
                           {static_cast<TypeParam>(1.0f), static_cast<TypeParam>(2.0f), static_cast<TypeParam>(3.0f), static_cast<TypeParam>(4.0f),
                            static_cast<TypeParam>(-1.0f), static_cast<TypeParam>(-2.0f), static_cast<TypeParam>(-3.0f), static_cast<TypeParam>(-4.0f)});
  test.AddInput<TypeParam>("B", {4, 3}, std::vector<TypeParam>(12, static_cast<TypeParam>(1.0f)));
  test.AddOutput<TypeParam>("Y", {2, 3},
                            {static_cast<TypeParam>(10.0f), static_cast<TypeParam>(10.0f), static_cast<TypeParam>(10.0f),
                             static_cast<TypeParam>(-10.0f), static_cast<TypeParam>(-10.0f), static_cast<TypeParam>(-10.0f)});
  // tensorRT don't seem to support missing bias
  std::unordered_set<std::string> excluded_provider_types{kTensorrtExecutionProvider};
  // QNN Linux result diff 0.011714935302734375 exceed the threshold
#ifndef _WIN32
  excluded_provider_types.insert(kQnnExecutionProvider);
#endif
  test.ConfigExcludeEps(excluded_provider_types)
      .Config(run_with_tunable_op)
      .RunWithConfig();
}

TYPED_TEST(GemmOpTypedTests, TestGemmWithAlphaOpset11) {
  OpTester test("Gemm", 11);

  test.AddAttribute("alpha", 2.0f);

  test.AddInput<TypeParam>("A", {2, 2},
                           {static_cast<TypeParam>(1.0f), static_cast<TypeParam>(2.0f), static_cast<TypeParam>(3.0f), static_cast<TypeParam>(4.0f)});
  test.AddInput<TypeParam>("B", {2, 2}, std::vector<TypeParam>(4, static_cast<TypeParam>(1.0f)));
  test.AddOutput<TypeParam>("Y", {2, 2},
                            {static_cast<TypeParam>(6.0f), static_cast<TypeParam>(6.0f), static_cast<TypeParam>(14.0f), static_cast<TypeParam>(14.0f)});
  // tensorRT don't seem to support missing bias
  test.ConfigExcludeEps({kTensorrtExecutionProvider})
      .Config(run_with_tunable_op)
      .RunWithConfig();
}

#ifndef ENABLE_TRAINING
// Prepacking is disabled in training builds so no need to test the feature in a training build.
TEST(GemmOpTest, SharedPrepackedWeights) {
  OpTester test("Gemm");

  test.AddAttribute("transA", (int64_t)0);
  test.AddAttribute("transB", (int64_t)0);
  test.AddAttribute("alpha", 1.0f);
  test.AddAttribute("beta", 1.0f);

  std::vector<float> b_init_values(12, 1.0f);
  test.AddInput<float>("A", {2, 4},
                       {1.0f, 2.0f, 3.0f, 4.0f,
                        -1.0f, -2.0f, -3.0f, -4.0f});
  // B is to be an initializer for triggering pre-packing
  test.AddInput<float>("B", {4, 3}, b_init_values, true);
  test.AddInput<float>("C", {2, 3}, std::vector<float>(6, 1.0f));
  test.AddOutput<float>("Y", {2, 3},
                        {11.0f, 11.0f, 11.0f,
                         -9.0f, -9.0f, -9.0f});

  OrtValue b;
  Tensor::InitOrtValue(DataTypeImpl::GetType<float>(), TensorShape({4, 3}),
                       b_init_values.data(), OrtMemoryInfo(CPU, OrtAllocatorType::OrtDeviceAllocator), b);

  SessionOptions so;

  // Set up B as a shared initializer to be shared between sessions
  ASSERT_EQ(so.AddInitializer("B", &b), Status::OK());

  // We want all sessions running using this OpTester to be able to share pre-packed weights if applicable
  test.EnableSharingOfPrePackedWeightsAcrossSessions();

  // Pre-packing is limited just to the CPU EP for now and we will only test the CPU EP
  // and we want to ensure that it is available in this build
  auto cpu_ep = []() -> std::vector<std::unique_ptr<IExecutionProvider>> {
    std::vector<std::unique_ptr<IExecutionProvider>> execution_providers;
    execution_providers.push_back(DefaultCpuExecutionProvider());
    return execution_providers;
  };

  size_t number_of_pre_packed_weights_counter_session_1 = 0;
  size_t number_of_shared_pre_packed_weights_counter = 0;

  // Session 1
  {
    test.Config(so)
        .ConfigEps(cpu_ep())
        .Config(run_with_tunable_op)
        .RunWithConfig(&number_of_pre_packed_weights_counter_session_1, &number_of_shared_pre_packed_weights_counter);
    // Assert that no pre-packed weights have been shared thus far
    ASSERT_EQ(number_of_shared_pre_packed_weights_counter, static_cast<size_t>(0));
  }

  auto number_of_elements_in_shared_prepacked_buffers_container =
      test.GetNumPrePackedWeightsShared();
  // Assert that the number of elements in the shared container
  // is the same as the number of weights that have been pre-packed
  ASSERT_EQ(number_of_pre_packed_weights_counter_session_1, number_of_elements_in_shared_prepacked_buffers_container);

  // On some platforms/architectures MLAS may choose to not do any pre-packing and the number of elements
  // that have been pre-packed will be zero in which case we do not continue with the testing
  // of "sharing" of pre-packed weights as there are no pre-packed weights to be shared at all.
  if (number_of_pre_packed_weights_counter_session_1 == 0)
    return;

  // Session 2
  {
    size_t number_of_pre_packed_weights_counter_session_2 = 0;
    test.Config(so)
        .ConfigEps(cpu_ep())
        .Config(run_with_tunable_op)
        .RunWithConfig(&number_of_pre_packed_weights_counter_session_2, &number_of_shared_pre_packed_weights_counter);

    // Assert that the same number of weights were pre-packed in both sessions
    ASSERT_EQ(number_of_pre_packed_weights_counter_session_1, number_of_pre_packed_weights_counter_session_2);

    // Assert that the number of pre-packed weights that were shared equals
    // the number of pre-packed weights in the second session
    ASSERT_EQ(number_of_pre_packed_weights_counter_session_2,
              static_cast<size_t>(number_of_shared_pre_packed_weights_counter));
  }
}
#endif

TEST(GemmOpTest, GemmOptimizePacked) {
  auto run_test = [](int64_t M, int64_t K, int64_t N, BiasType bias_type) {
    OpTester test("Gemm", 13);

    test.AddAttribute("transA", (int64_t)0);
    test.AddAttribute("transB", (int64_t)0);
    test.AddAttribute("alpha", 1.0f);
    test.AddAttribute("beta", 1.0f);

    std::vector<float> a_data = initialize_matrix(M, K);
    std::vector<float> b_data = initialize_matrix(K, N);

    auto [c_data, c_shape] = initialize_bias(bias_type, M, N);
    bool has_bias = !c_data.empty();

    test.AddInput<float>("A", {M, K}, a_data);
    test.AddInput<float>("B", {K, N}, b_data);
    if (has_bias) {
      test.AddInput<float>("C", c_shape, c_data);
    }

    // Calculate expected output
    std::vector<float> expected_data(M * N, 0.0f);
    for (int64_t i = 0; i < M; ++i) {
      for (int64_t j = 0; j < N; ++j) {
        float sum = 0.0f;
        for (int64_t k = 0; k < K; ++k) {
          sum += a_data[i * K + k] * b_data[k * N + j];
        }
        expected_data[i * N + j] = sum + get_bias_value(c_data, bias_type, i, j, N);
      }
    }

    test.AddOutput<float>("Y", {M, N}, expected_data);
    test.ConfigExcludeEps({kQnnExecutionProvider})
        .Config(run_with_tunable_op)
        .RunWithConfig();
  };

  // Test different matrix sizes with all bias types
  std::vector<std::tuple<int64_t, int64_t, int64_t>> test_sizes = {
      {32, 32, 32}, {64, 64, 64}, {60, 16, 92}, {8, 8, 8}, {128, 128, 128}, {128, 32, 64}, {96, 24, 48}, {48, 48, 120}, {72, 80, 84}, {33, 67, 99}, {1, 1, 1}, {31, 31, 31}};

  std::vector<BiasType> bias_types = {
      BiasType::noBias, BiasType::MBias, BiasType::ScalarBias,
      BiasType::MNBias, BiasType::NBias};

  // Run tests with different combinations of matrix sizes and bias types
  for (const auto& size : test_sizes) {
    for (const auto& bias_type : bias_types) {
      run_test(std::get<0>(size), std::get<1>(size), std::get<2>(size), bias_type);
    }
  }
}

TEST(GemmOpTest, GemmOptimizePackedTransA) {
  auto run_test = [](int64_t M, int64_t K, int64_t N, BiasType bias_type) {
    OpTester test("Gemm", 13);

    test.AddAttribute("transA", (int64_t)1);  // A is transposed
    test.AddAttribute("transB", (int64_t)0);
    test.AddAttribute("alpha", 1.0f);
    test.AddAttribute("beta", 1.0f);

    std::vector<float> a_data = initialize_matrix(K, M);
    std::vector<float> b_data = initialize_matrix(K, N);

    // Initialize bias with appropriate shape
    auto [c_data, c_shape] = initialize_bias(bias_type, M, N);
    bool has_bias = !c_data.empty();

    test.AddInput<float>("A", {K, M}, a_data);
    test.AddInput<float>("B", {K, N}, b_data);
    if (has_bias) {
      test.AddInput<float>("C", c_shape, c_data);
    }

    // Calculate expected output for transposed A
    std::vector<float> expected_data(M * N, 0.0f);
    for (int64_t i = 0; i < M; ++i) {
      for (int64_t j = 0; j < N; ++j) {
        float sum = 0.0f;
        for (int64_t k = 0; k < K; ++k) {
          sum += a_data[k * M + i] * b_data[k * N + j];
        }
        expected_data[i * N + j] = sum + get_bias_value(c_data, bias_type, i, j, N);
      }
    }

    test.AddOutput<float>("Y", {M, N}, expected_data);
    test.ConfigExcludeEps({kQnnExecutionProvider})
        .Config(run_with_tunable_op)
        .RunWithConfig();
  };

  std::vector<std::tuple<int64_t, int64_t, int64_t>> test_sizes = {
      {32, 32, 32}, {64, 64, 64}, {60, 16, 92}, {8, 8, 8}, {128, 128, 128}, {128, 32, 64}, {96, 24, 48}, {48, 48, 120}, {72, 80, 84}, {33, 67, 99}, {1, 1, 1}, {31, 31, 31}, {2, 3, 4}, {63, 64, 65}, {129, 129, 129}};

  std::vector<BiasType> bias_types = {
      BiasType::noBias, BiasType::MBias, BiasType::ScalarBias,
      BiasType::MNBias, BiasType::NBias};

  // Run tests with different combinations
  for (const auto& size : test_sizes) {
    for (const auto& bias_type : bias_types) {
      run_test(std::get<0>(size), std::get<1>(size), std::get<2>(size), bias_type);
    }
  }
}

TEST(GemmOpTest, GemmOptimizePackedTransB) {
  auto run_test = [](int64_t M, int64_t K, int64_t N, BiasType bias_type) {
    OpTester test("Gemm", 13);

    test.AddAttribute("transA", (int64_t)0);
    test.AddAttribute("transB", (int64_t)1);
    test.AddAttribute("alpha", 1.0f);
    test.AddAttribute("beta", 1.0f);

    std::vector<float> a_data = initialize_matrix(M, K);
    std::vector<float> b_data = initialize_matrix(N, K);

    // Initialize bias with appropriate shape
    auto [c_data, c_shape] = initialize_bias(bias_type, M, N);
    bool has_bias = !c_data.empty();

    test.AddInput<float>("A", {M, K}, a_data);
    test.AddInput<float>("B", {N, K}, b_data);
    if (has_bias) {
      test.AddInput<float>("C", c_shape, c_data);
    }

    // Calculate expected output
    std::vector<float> expected_data(M * N, 0.0f);
    for (int64_t i = 0; i < M; ++i) {
      for (int64_t j = 0; j < N; ++j) {
        float sum = 0.0f;
        for (int64_t k = 0; k < K; ++k) {
          sum += a_data[i * K + k] * b_data[j * K + k];
        }
        expected_data[i * N + j] = sum + get_bias_value(c_data, bias_type, i, j, N);
      }
    }

    test.AddOutput<float>("Y", {M, N}, expected_data);
    test.ConfigExcludeEps({kQnnExecutionProvider})
        .Config(run_with_tunable_op)
        .RunWithConfig();
  };

  std::vector<std::tuple<int64_t, int64_t, int64_t>> test_sizes = {
      {32, 32, 32}, {64, 64, 64}, {60, 16, 92}, {8, 8, 8}, {128, 128, 128}, {128, 32, 64}, {96, 24, 48}, {48, 48, 120}, {72, 80, 84}, {33, 67, 99}, {1, 1, 1}, {31, 31, 31}, {2, 3, 4}, {63, 64, 65}, {129, 129, 129}};

  std::vector<BiasType> bias_types = {
      BiasType::noBias, BiasType::MBias, BiasType::ScalarBias,
      BiasType::MNBias, BiasType::NBias};

  // Run tests with different combinations
  for (const auto& size : test_sizes) {
    for (const auto& bias_type : bias_types) {
      run_test(std::get<0>(size), std::get<1>(size), std::get<2>(size), bias_type);
    }
  }
}

TEST(GemmOpTest, GemmOptimizePackedTransAB) {
  auto run_test = [](int64_t M, int64_t K, int64_t N, BiasType bias_type) {
    OpTester test("Gemm", 13);

    test.AddAttribute("transA", (int64_t)1);
    test.AddAttribute("transB", (int64_t)1);
    test.AddAttribute("alpha", 1.0f);
    test.AddAttribute("beta", 1.0f);

    std::vector<float> a_data = initialize_matrix(M, K);
    std::vector<float> b_data = initialize_matrix(N, K);

    // Initialize bias with appropriate shape
    auto [c_data, c_shape] = initialize_bias(bias_type, M, N);
    bool has_bias = !c_data.empty();

    test.AddInput<float>("A", {K, M}, a_data);
    test.AddInput<float>("B", {N, K}, b_data);
    if (has_bias) {
      test.AddInput<float>("C", c_shape, c_data);
    }

    // Calculate expected output for both matrices transposed
    std::vector<float> expected_data(M * N, 0.0f);
    for (int64_t i = 0; i < M; ++i) {
      for (int64_t j = 0; j < N; ++j) {
        float sum = 0.0f;
        for (int64_t k = 0; k < K; ++k) {
          sum += a_data[k * M + i] * b_data[j * K + k];
        }
        expected_data[i * N + j] = sum + get_bias_value(c_data, bias_type, i, j, N);
      }
    }

    test.AddOutput<float>("Y", {M, N}, expected_data);
    test.ConfigExcludeEps({kQnnExecutionProvider})
        .Config(run_with_tunable_op)
        .RunWithConfig();
  };

  std::vector<std::tuple<int64_t, int64_t, int64_t>> test_sizes = {
      {32, 32, 32}, {64, 64, 64}, {60, 16, 92}, {8, 8, 8}, {128, 128, 128}, {128, 32, 64}, {96, 24, 48}, {48, 48, 120}, {72, 80, 84}, {33, 67, 99}, {1, 1, 1}, {31, 31, 31}, {2, 3, 4}, {63, 64, 65}, {64, 64, 65}, {129, 129, 129}};

  std::vector<BiasType> bias_types = {
      BiasType::noBias, BiasType::MBias, BiasType::ScalarBias,
      BiasType::MNBias, BiasType::NBias};

  // Run tests with different combinations
  for (const auto& size : test_sizes) {
    for (const auto& bias_type : bias_types) {
      run_test(std::get<0>(size), std::get<1>(size), std::get<2>(size), bias_type);
    }
  }
}

}  // namespace test
}  // namespace onnxruntime
