/*
 * Copyright (c) Intel Corporation. All rights reserved.
 * Licensed under the MIT License.
 */

#include <string>
#include <set>
#include <algorithm>
#include <utility>
#include "core/graph/graph_utils.h"
#include "core/graph/graph_viewer.h"
#include "core/optimizer/initializer.h"
#include "core/optimizer/fuse_initializers_transformer.h"

namespace onnxruntime {

/**
 * @brief   This function checks if the next-node to the current node supports a
 *          kernel with specific type of inputs/outputs
 *
 * @param graph                Graph object
 * @param node                 Node whose next-node to be checked if it supports a particular Kernel type
 * @param cpu_kernel_registry  Registry of all CPU supported Kernels
 * @param kernel_type          The type of inputs/outputs the Kernel should support
 * @param logger               Logging object
 *
 * @return True if Kernel which supports a specific type of inputs/outputs is found, else returns False.
 */
static bool IsNextNodeSupportedWithGivenKernelType(const Graph& graph, const Node& node,
                                                   std::shared_ptr<KernelRegistry> cpu_kernel_registry,
                                                   const onnxruntime::MLDataType kernel_type,
                                                   const logging::Logger& logger) {

    // Get next node
    const Node& next_node = *graph.GetNode(node.OutputNodesBegin()->Index());

    // If Node EP is not CPU no need to check further as this optimization is as of now only supported on CPU
    if (!(kCpuExecutionProvider == next_node.GetExecutionProviderType())) return false;

    // If schema is not available, there is no way to know whether it is safe to convert this to type T, give up
    const auto* schema = next_node.Op();
    if (!schema) return false;

    // Init type constraint map
    const ONNX_NAMESPACE::TypeConstraintMap& type_schema = schema->typeConstraintMap();
    InlinedHashMap<std::string, MLDataType> type_constraint_map;
    type_constraint_map.reserve(type_schema.size());

    // check if inputs of the node has type T support

    const auto& input_arg_counts = next_node.InputArgCount();
    const auto& input_defs = next_node.InputDefs();
    const auto& formal_inputs = schema->inputs();
    const size_t min_no_of_inputs = std::min(formal_inputs.size(), input_arg_counts.size());

    size_t input_arg_idx_start = 0;
    for (size_t formal_idx = 0; formal_idx < min_no_of_inputs; input_arg_idx_start += static_cast<size_t>(input_arg_counts[formal_idx]), ++formal_idx) {

        const auto& type_str = formal_inputs[formal_idx].GetTypeStr();

        // Don't care about parameter that does not have a type constraint.
        if (type_schema.end() == type_schema.find(type_str)) continue;

        // check if the type constraint is already assigned
        if (type_constraint_map.end() != type_constraint_map.find(type_str)) continue;

        // type_str is like T, T1 or T2 ...
        for (int input_arg_idx = 0; input_arg_idx < input_arg_counts[formal_idx]; ++input_arg_idx) {

            // calc index of input arg in input defs
            auto input_def_idx = input_arg_idx_start + input_arg_idx;

            // Check if the input def exist
            if (!((input_def_idx < input_defs.size()) && input_defs[input_def_idx] && input_defs[input_def_idx]->Exists())) continue;

            // Enforcing the type str to support type T
            type_constraint_map[type_str] = kernel_type;

            break;  // we don't have multiple tensors feeding into one input
        }
    }

    // check if output/s of the node has type T support

    const auto& output_defs = next_node.OutputDefs();
    const auto& formal_outputs = schema->outputs();
    const size_t min_no_of_outputs = std::min(formal_outputs.size(), output_defs.size());

    for (size_t output_def_idx = 0; output_def_idx < min_no_of_outputs; ++output_def_idx) {

        const auto& type_str = formal_outputs[output_def_idx].GetTypeStr();

        // Don't care about parameter that does not have a type constraint.
        if (type_schema.end() == type_schema.find(type_str)) continue;

        // check if the type constraint is already assigned
        if (type_constraint_map.end() != type_constraint_map.find(type_str)) continue;

        // Check if the output def exist
        if (!(output_defs[output_def_idx] && output_defs[output_def_idx]->Exists())) continue;

        // Enforcing the type str to support type T
        type_constraint_map[type_str] = kernel_type;
    }

    // Check if type T version of the kernel is available for this node.
    // The TryFindKernel should return an OK status, with supported kernel info,
    // otherwise it is concluded that the supported Kernel for specific Type T
    // inputs/outputs is not found.
    const KernelCreateInfo* kernel_create_info{};
    const auto lookup_status = cpu_kernel_registry->TryFindKernel(kCpuExecutionProvider,
                                                                  next_node.OpType(),
                                                                  next_node.Domain(),
                                                                  next_node.SinceVersion(),
                                                                  type_constraint_map,
                                                                  logger,
                                                                  &kernel_create_info);
    auto is_next_node_supports_type_T_kernel = (lookup_status.IsOK() && kernel_create_info != nullptr);

    return is_next_node_supports_type_T_kernel;
}

static bool IsCastNodeWithConstraints(const Node& node) {

    // Node must be cast node
    if (!("Cast" == node.OpType())) return false;

    // Node must have no input edges
    if (!(0 == node.GetInputEdgesCount())) return false;

    // Node must have only one output edge
    if (!(1 == node.GetOutputEdgesCount())) return false;

    return true;
}

/**
 * @brief   Check if for the current node at a given arg index, it is an initialized tensor of a specific type.
 *
 * @param graph            Graph Object
 * @param node             Node object
 * @param node_arg_index   Argument index of Node Object
 * @param tensor_type      The type of initialized tensor to be found in the given node at the given arg index
 *
 * @return  True, if at the given arg index of the given node an initialized tensor of "tensor_type" is found, else,
 *          False.
 */
static bool IsNodeInitializedWithGivenTensorTypeAtGivenIndex(const Graph& graph, const Node& node, NodeIndex node_arg_index, const onnxruntime::MLDataType tensor_type) {

    // Node must have initialized tensor
    if (!(graph.IsInitializedTensor(node.InputDefs()[node_arg_index]->Name()))) return false;

    // Node initialzed tensor must be an type T Tensor
    if (!(DataTypeImpl::TypeFromProto(*(node.InputDefs()[node_arg_index]->TypeAsProto())) == tensor_type)) return false;

    return true;
}

/**
 * @brief   Find if fusing the given node to it's next node is valid, that is, if fusing the initializer
 *          to it's parent node is valid.
 *
 * @param graph                Graph Object
 * @param node                 Node object
 * @param node_arg_index       Argument index of Node Object
 * @param cpu_kernel_registry  Registry of all CPU supported Kernels
 * @param init_type            Initialized Type / Source Type / Unsupported Kernel Type
 * @param cvt_type             Conversion Type / Destination Type / Supported Kernel Type
 * @param logger               Logging object
 *
 * @return  True, if the current node is able to fuse to its next node, else, False.
 */
static bool IsFusingInitializerToNodeValid(const Graph& graph, const Node& node, NodeIndex node_arg_index,
                             std::shared_ptr<KernelRegistry> cpu_kernel_registry,
                             const onnxruntime::MLDataType init_type, const onnxruntime::MLDataType cvt_type,
                             const logging::Logger& logger) {

    // Check if current node have "Initialization" type initialized tensor which can be fused into next node,
    // if not available, return false.
    if (!(IsNodeInitializedWithGivenTensorTypeAtGivenIndex(graph, node, node_arg_index, init_type))) return false;

    // Check if "Initialization" type version of the kernel is NOT available for next node,
    // if available, return false.
    if (IsNextNodeSupportedWithGivenKernelType(graph, node, cpu_kernel_registry, init_type, logger)) return false;

    // Check if "Conversion" type version of the kernel is available for next node,
    // if not available, return false.
    if (!(IsNextNodeSupportedWithGivenKernelType(graph, node, cpu_kernel_registry, cvt_type, logger))) return false;

    return true;
}

/**
 * @brief Make a new name from the old node arg name.
 *
 * It replaces "InsertedPrecisionFreeCast_" prefix in a node name with "FusedBack_" prefix.
 *
 * @param old_node_arg_name Old arg name
 *
 * @return New arg name
 */
static const std::string NewNodeArgName(const std::string& old_node_arg_name) {
    static thread_local const std::string pattern_to_be_replaced = "InsertedPrecisionFreeCast_";
    std::string new_node_arg_name = old_node_arg_name;
    auto pos = new_node_arg_name.find(pattern_to_be_replaced);
    if(std::string::npos != pos) new_node_arg_name.replace(pos, pattern_to_be_replaced.size(), "");
    new_node_arg_name = "FusedBack_" + new_node_arg_name;
    return new_node_arg_name;
}

/**
 * @brief   It fuses the initializer in the current node to its next node.
 *
 * The node_arg_index should be usually 0, as the node which just encapsulates Initializer/s have just one input.
 *
 * @param graph            Graph Object
 * @param node             Current Node to be fused with its next node
 * @param node_arg_index   The arg index of the current Node, at which the tensor to be fused to next node is found.
 * @param node_type        The node type of the next-node to the current node.
 * @param thread_pool      Thread pool for multi-threaded conversion of the initializer
 *                          from an unsupported to supported type tensor.
 */
static void FuseInitializerWithNode(Graph& graph, Node& node, NodeIndex node_arg_index,
                                    const onnxruntime::MLDataType node_type,
                                    onnxruntime::concurrency::ThreadPool* thread_pool) {

    // Get next node
    Node& next_node = *graph.GetNode(node.OutputNodesBegin()->Index());

    // Get the index in next node at which the initializer must be replaced
    NodeIndex next_node_arg_index = 0;
    for (; next_node_arg_index < next_node.InputDefs().size(); ++next_node_arg_index) {
        if (node.Name() == next_node.InputDefs()[next_node_arg_index]->Name()) {
            break;
        }
    }

    // Get the src initialized tensor
    auto constant_initializer_tensor = graph_utils::GetConstantInitializer(graph, node.InputDefs()[node_arg_index]->Name());
    ONNX_NAMESPACE::TensorProto src_tensor(*constant_initializer_tensor);
    Initializer src_init{*constant_initializer_tensor, graph.ModelPath()};
    src_init.ToProto(src_tensor);

    // Convert to dst
    ONNX_NAMESPACE::TensorProto dst_tensor;
    if (node_type == DataTypeImpl::GetTensorType<float>())
      dst_tensor = src_init.ToFloat32(graph.GenerateNodeArgName(NewNodeArgName(next_node.InputDefs()[next_node_arg_index]->Name())), thread_pool);
    else if (node_type == DataTypeImpl::GetTensorType<MLFloat16>())
      dst_tensor = src_init.ToFP16(graph.GenerateNodeArgName(NewNodeArgName(next_node.InputDefs()[next_node_arg_index]->Name())));
    else if (node_type == DataTypeImpl::GetTensorType<BFloat16>())
      dst_tensor = src_init.ToBFloat16(graph.GenerateNodeArgName(NewNodeArgName(next_node.InputDefs()[next_node_arg_index]->Name())));
    else
      return;

    // Remove the edge
    graph.RemoveEdge(node.Index(), next_node.Index(), static_cast<int>(node_arg_index), static_cast<int>(next_node_arg_index));

    // Replace in next node
    graph_utils::ReplaceNodeInput(next_node, static_cast<int>(next_node_arg_index), graph_utils::AddInitializer(graph, dst_tensor));
}

Status FuseInitializersTransformer::ApplyImpl(Graph& graph, bool& modified, int /*graph_level*/, const logging::Logger& logger) const {

    // Do nothing if kernel registry is not available
    if (nullptr != cpu_kernel_registry_) {

        // Init
        std::set<std::pair<NodeIndex, NodeIndex>> nodes_to_be_fused_and_removed_from_graph;

        // Get nodes in topological order
        const GraphViewer graph_viewer(graph);
        auto nodes_indexes_in_topological_order = graph_viewer.GetNodesInTopologicalOrder();

        // For each Node
        for (auto node_index : nodes_indexes_in_topological_order) {

            // Get Node
            auto node = graph.GetNode(node_index);

            // Check if the current node is cast node
            if (!(node && IsCastNodeWithConstraints(*node))) continue;

            // For each Node Args
            for (NodeIndex node_arg_index = 0; node_arg_index < node->InputDefs().size(); ++node_arg_index) {

                // Check if fusing initializers to node is valid for the given node and the arg index
                if (IsFusingInitializerToNodeValid(graph, *node, node_arg_index, cpu_kernel_registry_, init_type_, cvt_type_, logger)) {

                    // Add node to the set of nodes to be fused and removed
                    nodes_to_be_fused_and_removed_from_graph.insert(std::make_pair(node_index, node_arg_index));
                }
            }
        }

        // Fuse all Cast Node (with src type Initializer) to Next Node if dst type Kernel is supported
        for(auto node_index_arg_index_pair : nodes_to_be_fused_and_removed_from_graph) {
            auto node = graph.GetNode(node_index_arg_index_pair.first);
            auto node_arg_index = node_index_arg_index_pair.second;
            FuseInitializerWithNode(graph, *node, node_arg_index, cvt_type_, thread_pool_);
        }

        // Remove all nodes considered during replacement
        for(auto node_index_arg_index_pair : nodes_to_be_fused_and_removed_from_graph) {
            graph.RemoveNode(node_index_arg_index_pair.first);
        }

        // set flag to true indicating the graph is changed
        if(!nodes_to_be_fused_and_removed_from_graph.empty()) modified = true;
    }

    return Status::OK();
}

}  // namespace onnxruntime
