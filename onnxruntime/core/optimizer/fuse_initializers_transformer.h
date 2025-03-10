/*
 * Copyright (c) Intel Corporation. All rights reserved.
 * Licensed under the MIT License.
 */

#pragma once

#include "core/optimizer/graph_transformer.h"
#include "core/framework/kernel_registry_manager.h"
#include "core/framework/kernel_registry.h"

namespace onnxruntime {

    /**
     * @class FuseInitializersTransformer
     *
     * A Transformer to fuse cast node that casts init_type to cvt_type for cpu nodes back to their parent nodes. Below
     * is the explanation on how this transforms works. It depends on "InsertCastTransforms" to produce the intermediate
     * representation from which it fuses the initializers (which are the cast node with one single input) back to the
     * parent node.
     *
     * ```
     *
     *         "Input Graph"                       "Intermediate Representation"               "Fusion Transforms"
     *
     *           --------                   --------        --------        --------                 --------
     *          | X_Fp16 |                 | X_Fp16 |      | W_Fp16 |      | B_Fp16 |               | X_Fp16 |
     *           --------                   --------        --------        --------                 --------
     *              |                          |               |               |                        |
     *              |                          |               |               |                        |
     *              |                          V               V               V                        V
     *              |                       | Cast |        | Cast |        | Cast |                 | Cast |
     *              |                       | Fp16 |        | Fp16 |        | Fp16 |                 | Fp16 |
     *              |                       |  To  |        |  To  |        |  To  |                 |  To  |
     *              |                       | Fp32 |        | Fp32 |        | Fp32 |                 | Fp32 |
     *              |                          |               |               |                        |
     *              |                          |               |               |                        |
     *              V                          V               V               V                        V
     *  ----------------------------       -----------------------------------------       ----------------------------
     * |        Conv_Fp16           |     |                                         |     |         Conv_Fp32          |
     * |        --W_Fp16--          | ==> |                Conv_Fp32                | ==> |         --W_Fp32--         |
     * |        --B_Fp16--          |     |                                         |     |         --B_Fp32--         |
     *  ----------------------------       -----------------------------------------       ----------------------------
     *              |                                          |                                        |
     *              |                                          |                                        |
     *              |                                          V                                        V
     *              |                                       | Cast |                                 | Cast |
     *              |                                       | Fp32 |                                 | Fp32 |
     *              |                                       |  To  |                                 |  To  |
     *              |                                       | Fp16 |                                 | Fp16 |
     *              |                                          |                                        |
     *              |                                          |                                        |
     *              V                                          V                                        V
     *           --------                                   --------                                 --------
     *          | Y_Fp16 |                                 | Y_Fp16 |                               | Y_Fp16 |
     *           --------                                   --------                                 --------
     *
     * ```
     *
     */
    class FuseInitializersTransformer : public GraphTransformer {

    public:
        /**
         * @brief   Fuses Initializers to child node after conversion to child node kernel type
         *          to save on Cast at each inference.
         *
         * This function must happen after InsertCastTransformer. Currently only FP16 Initializers are fused with
         * nodes supporting FP32 Kernel, however, the code is designed to apply for any supported conversion/s.
         *
         * @param name                      Name of the transforms, just for logging purpose
         * @param init_type                 The unsupported type for which cast nodes are inserted to convert to
         *                                  supported type for which a kernel exist
         * @param cvt_type                  The supported type for which a kernel exist
         * @param cpu_kernel_registry       Registry of all CPU supported Kernels, which is used to query whether
         *                                  an op node can be safely created
         * @param thread_pool               A pointer to thread pool to support conversion from init_type to cvt_type
         *                                  with multithreading
         */
        FuseInitializersTransformer(const std::string& name,
                                    const onnxruntime::MLDataType init_type,
                                    const onnxruntime::MLDataType cvt_type,
                                    std::shared_ptr<KernelRegistry> cpu_kernel_registry,
                                    onnxruntime::concurrency::ThreadPool *thread_pool = nullptr) :
                                    GraphTransformer(name, {}),
                                    init_type_(init_type),
                                    cvt_type_(cvt_type),
                                    cpu_kernel_registry_(cpu_kernel_registry),
                                    thread_pool_(thread_pool) {}

    private:

        const onnxruntime::MLDataType init_type_;
        const onnxruntime::MLDataType cvt_type_;
        std::shared_ptr<KernelRegistry> cpu_kernel_registry_;
        onnxruntime::concurrency::ThreadPool* thread_pool_;
        Status ApplyImpl(
             Graph& graph,
             bool& modified,
             int graph_level,
             const logging::Logger& logger) const override;
    };
}  // namespace onnxruntime
