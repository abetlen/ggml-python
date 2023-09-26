import io

import numpy as np
import onnx
from onnx import helper
from onnx.onnx_pb import TensorProto

import onnx.backend.test

from onnxruntime import InferenceSession  # type: ignore

from ggml.contrib.onnx import GgmlRuntimeBackend


def test_ggml_onnx_runtime_basic():
    # The name of the input tensor
    input_name = "X"

    # The name of the weights tensor
    weight_name_a = "A"
    weight_name_b = "B"

    # The name of the output
    output_name = "Y"

    # Create the nodes (operations) in our graph
    node1 = helper.make_node(
        "Mul", [input_name, input_name], ["X_squared"], name="node1"
    )  # X^2
    node2 = helper.make_node(
        "Mul", ["X_squared", weight_name_a], ["X_squared_times_a"], name="node2"
    )  # X^2 * A
    node3 = helper.make_node(
        "Add", ["X_squared_times_a", weight_name_b], [output_name], name="node3"
    )  # X^2 * A + B

    # Define the tensors (values) in our graph
    X_value_info = helper.make_tensor_value_info(
        input_name, TensorProto.FLOAT, [None, 1]
    )

    output_value_info = helper.make_tensor_value_info(
        output_name, TensorProto.FLOAT, [None, 1]
    )

    # Set A and B as parameters/weights
    weights_a = np.ones(1, dtype=float).astype(np.float32)
    weights_b = np.ones(1, dtype=float).astype(np.float32)

    A_init = helper.make_tensor(
        weight_name_a,
        TensorProto.FLOAT,
        [
            1,
        ],
        weights_a,
    )
    B_init = helper.make_tensor(
        weight_name_b,
        TensorProto.FLOAT,
        [
            1,
        ],
        weights_b,
    )

    # Create the graph (model).
    graph_def = helper.make_graph(
        [node1, node2, node3],
        "simple_expression_model",
        [X_value_info],
        [output_value_info],
        [A_init, B_init],
    )

    model_def = helper.make_model(graph_def, producer_name="onnx-simple-expression")

    input_data = {"X": np.array([[6.0]], dtype=np.float32)}

    f = io.BytesIO()
    onnx.save(model_def, f)

    runtime_result = InferenceSession(f.getvalue()).run(None, input_data)

    ggml_dummy_model = GgmlRuntimeBackend.prepare(model_def)
    ggml_result = ggml_dummy_model.run(input_data)
    assert ggml_result == runtime_result


def test_ggml_onnx_graph_optimization():
    # Construct an onnx graph and optimize it
    # The graph is of the form y = (A^T)^T * x + b
    # the optimization should remove the transpose operations

    # The name of the input tensor
    input_name = "x"

    # The name of the weights tensor
    weight_name_a = "A"
    weight_name_b = "b"

    # The name of the output
    output_name = "y"

    # Create the nodes (operations) in our graph
    node1 = helper.make_node(
        "Transpose", [weight_name_a], ["A_transposed"], name="node1"
    )  # A^T
    node2 = helper.make_node(
        "Transpose", ["A_transposed"], ["A_transposed_transposed"], name="node2"
    )  # (A^T)^T
    node3 = helper.make_node(
        "MatMul", [input_name, "A_transposed_transposed"], ["x_times_A"], name="node3"
    )  # x * (A^T)^T
    node4 = helper.make_node(
        "Add", ["x_times_A", weight_name_b], [output_name], name="node4"
    )  # x * (A^T)^T + b

    # Define the tensors (values) in our graph
    X_value_info = helper.make_tensor_value_info(
        input_name, TensorProto.FLOAT, [None, 32]
    )

    output_value_info = helper.make_tensor_value_info(
        output_name, TensorProto.FLOAT, [None, 32]
    )

    # Set A and b as parameters/weights
    weights_a = np.random.randn(32, 32).astype(np.float32)

    weights_b = np.random.randn(32).astype(np.float32)

    A_init = helper.make_tensor(
        weight_name_a,
        TensorProto.FLOAT,
        [
            32,
            32,
        ],
        weights_a,
    )
    B_init = helper.make_tensor(
        weight_name_b,
        TensorProto.FLOAT,
        [
            32,
        ],
        weights_b,
    )

    # Create the graph (model).
    graph_def = helper.make_graph(
        [node1, node2, node3, node4],
        "simple_expression_model",
        [X_value_info],
        [output_value_info],
        [A_init, B_init],
    )

    model_def = helper.make_model(graph_def, producer_name="onnx-simple-expression")

    from typing import Optional, List
    from ggml.contrib.onnx import GgmlOnnxGraphOptimizer, GgmlOnnxGraphOptimizerRule
    from onnx.onnx_ml_pb2 import ModelProto, NodeProto

    class TransposeIdentityRule(GgmlOnnxGraphOptimizerRule):
        def __init__(self):
            super().__init__()

        def apply(self, model: ModelProto) -> Optional[ModelProto]:
            # find first transpose node
            transpose_node: Optional[NodeProto] = None
            for node in model.graph.node:
                if node.op_type == "Transpose":
                    transpose_node = node
                    break
            else:
                return None

            # find a transpose node that transposes the output of the first transpose node
            transpose_transpose_node: Optional[NodeProto] = None
            for node in model.graph.node:
                if (
                    node.op_type == "Transpose"
                    and node.input[0] == transpose_node.output[0]
                ):
                    transpose_transpose_node = node
                    break
            else:
                return None

            # Create a new node list without the two transpose nodes
            new_nodes: List[NodeProto] = []
            for node in model.graph.node:
                if node not in [transpose_node, transpose_transpose_node]:
                    new_node = NodeProto()
                    new_node.CopyFrom(node)
                    new_node.input[:] = [transpose_node.input[0] if inp == transpose_transpose_node.output[0] else inp for inp in node.input]
                    new_nodes.append(new_node)
            
            # Create the new graph
            new_graph = helper.make_graph(
                new_nodes,
                model.graph.name,
                model.graph.input,
                model.graph.output,
                model.graph.initializer,
            )

            # create a new model
            new_model = helper.make_model(
                new_graph, producer_name=model.producer_name
            )

            return new_model


    input_data = {"x": np.random.randn(1, 32).astype(np.float32)}

    f = io.BytesIO()
    onnx.save(model_def, f)

    runtime_result = InferenceSession(f.getvalue()).run(None, input_data)

    ggml_dummy_model = GgmlRuntimeBackend.prepare(model_def)
    ggml_result = ggml_dummy_model.run(input_data)
    assert np.allclose(ggml_result[0], runtime_result[0], rtol=1e-03, atol=1e-05)

    optimizer = GgmlOnnxGraphOptimizer(
        model=model_def, rules=[TransposeIdentityRule()]
    )
    new_model = optimizer.optimize()
    assert new_model is not None
    ggml_dummy_model_new = GgmlRuntimeBackend.prepare(new_model)
    assert ggml_dummy_model_new is not None
    ggml_result_new = ggml_dummy_model_new.run(input_data)
    assert np.allclose(ggml_result_new[0], runtime_result[0], rtol=1e-03, atol=1e-05)
    assert sum([node.op_type == "Transpose" for node in new_model.graph.node]) == 0


def test_ggml_onnx_runtime_quantized():
    # Construct an onnx graph of the form y = Ax + b
    # where A and b are weights, x is the input, and y is the output
    # A is a 32x32 matrix of normally distributed random numbers
    # b is a vector of 32 normally distributed random numbers
    # x is a vector of 32 normally distributed random numbers
    # y is the output

    # The name of the input tensor
    input_name = "x"

    # The name of the weights tensor
    weight_name_a = "A"
    weight_name_b = "b"

    # The name of the output
    output_name = "y"

    # Create the nodes (operations) in our graph
    node1 = helper.make_node(
        "MatMul", [input_name, weight_name_a], ["x_times_A"], name="node1"
    )  # x * A
    node2 = helper.make_node(
        "Add", ["x_times_A", weight_name_b], [output_name], name="node2"
    )  # x * A + b

    # Define the tensors (values) in our graph
    X_value_info = helper.make_tensor_value_info(
        input_name, TensorProto.FLOAT, [None, 32]
    )

    output_value_info = helper.make_tensor_value_info(
        output_name, TensorProto.FLOAT, [None, 32]
    )

    # Set A and b as parameters/weights
    weights_a = np.random.randn(32, 32).astype(np.float32)

    weights_b = np.random.randn(32).astype(np.float32)

    A_init = helper.make_tensor(
        weight_name_a,
        TensorProto.FLOAT,
        [
            32,
            32,
        ],
        weights_a,
    )
    B_init = helper.make_tensor(
        weight_name_b,
        TensorProto.FLOAT,
        [
            32,
        ],
        weights_b,
    )

    # Create the graph (model).
    graph_def = helper.make_graph(
        [node1, node2],
        "simple_expression_model",
        [X_value_info],
        [output_value_info],
        [A_init, B_init],
    )

    model_def = helper.make_model(graph_def, producer_name="onnx-simple-expression")

    input_data = {"x": np.random.randn(1, 32).astype(np.float32)}

    f = io.BytesIO()
    onnx.save(model_def, f)

    runtime_result = InferenceSession(f.getvalue()).run(None, input_data)

    ggml_dummy_model = GgmlRuntimeBackend.prepare(model_def)
    ggml_result = ggml_dummy_model.run(input_data)

    assert np.allclose(ggml_result[0], runtime_result[0], rtol=1e-03, atol=1e-05)


backend_test = onnx.backend.test.BackendTest(GgmlRuntimeBackend, __name__)

backend_test.include("test_abs_")

backend_test.include("test_add_")
backend_test.exclude("test_add_uint8_")  # not supported

backend_test.include("test_and_")

backend_test.include("test_argmax_")
backend_test.include("test_argmin_")

backend_test.include("test_operator_basic_")

backend_test.include("test_cast_")

backend_test.include("test_ceil_")

backend_test.include("test_concat_")
backend_test.include("test_operator_concat")

backend_test.include("test_constant_")

backend_test.include("test_constantofshape")

# backend_test.include("_conv_")
# backend_test.exclude("_deform_conv")
# backend_test.exclude("test_operator_conv")


# backend_test.include("_convtranspose_")
# backend_test.exclude("_deform_convtranspose")
# backend_test.exclude("test_operator_convtranspose")

backend_test.include("test_operator_chunk")

backend_test.include("test_depthtospace")

backend_test.include("test_div_")
backend_test.exclude("test_div_uint8_")  # not supported

backend_test.include("test_elu_")
backend_test.include("test_ELU_")
backend_test.include("test_elu_example")

backend_test.include("test_eq_")

backend_test.include("test_equal_")
backend_test.exclude(".*equal.*.*string.*")

backend_test.include("test_exp_")
backend_test.include("test_operator_exp_")

backend_test.include("test_expand_")

backend_test.include("test_flatten_")
backend_test.include("test_operator_flatten_")


backend_test.include("test_floor_")

backend_test.include("test_greater_")

backend_test.include("test_gather_")
backend_test.exclude("test_gather_elements")  # not supported

backend_test.include("test_gemm")
backend_test.exclude("test_gemm_default_scalar_bias")

backend_test.include("test_greater_")

backend_test.include("test_hardsigmoid_")

backend_test.include("test_hardmax_")

backend_test.include("test_identity_")
backend_test.exclude("test_identity_opt")  # test case not correct: ONNX issue
backend_test.exclude("test_identity_sequence")  # test case not correct: ONNX issue

backend_test.include("test_instancenorm")

# backend_test.include("test_leakyrelu")

backend_test.include("test_less_")

backend_test.include("test_log_")

backend_test.include("test_LogSoftmax_")

backend_test.include("test_lrn")

backend_test.include("test_matmul_")
backend_test.include("test_operator_mm")

backend_test.include("test_max_")
backend_test.exclude("test_max_float16")  # not supported
backend_test.exclude("test_max_float64")  # not supported
backend_test.exclude("test_max_int64")  # not supported
backend_test.exclude("test_max_uint")  # not supported
backend_test.include("test_operator_max_")


backend_test.include("test_mean_")

backend_test.include("test_min_")
backend_test.exclude("test_min_float16")  # not supported
backend_test.exclude("test_min_float64")  # not supported
backend_test.exclude("test_min_int64")  # not supported
backend_test.exclude("test_min_uint")  # not supported
backend_test.include("test_operator_min_")


backend_test.include("test_mul_")
backend_test.exclude("test_mul_uint8")  # not supported

backend_test.include("test_neg_")

backend_test.include("test_not_")

backend_test.include("test_or_")

backend_test.include("test_prelu")
backend_test.include("test_PRelu_")
backend_test.include("test_prelu_example")

backend_test.include("test_pow_")
backend_test.exclude("test_pow_bcast")  # not supported
backend_test.exclude("test_pow_types_int64")  # not supported
backend_test.include("test_operator_pow_")

backend_test.include("test_range_")
backend_test.exclude("test_range_float")  # segfault
backend_test.exclude("test_range_int32")  # segfault

backend_test.include("test_reciprocal")

backend_test.include("test_reduce_max_")
backend_test.include("test_reduce_mean_")
backend_test.include("test_operator_reduced_mean_")
backend_test.include("test_reduce_min_")
backend_test.include("test_reduce_prod_")
backend_test.include("test_reduce_sum_")
backend_test.include("test_operator_reduced_sum_")
backend_test.include("test_reduce_log_sum_")
backend_test.exclude("test_reduce_log_sum_exp")


backend_test.include("test_reduce_l1_")
backend_test.include("test_reduce_l2_")

backend_test.include("test_relu_")
backend_test.include("test_relu_example")
backend_test.include("test_ReLU_")

backend_test.include("test_reshape_")
backend_test.exclude("test_reshape_allowzero")  # not supported

backend_test.include("test_selu_")
backend_test.include("test_selu_example")
backend_test.include("test_SELU_")
backend_test.include("test_operator_selu_")

backend_test.include("test_shape_")

backend_test.include("test_sigmoid_")
backend_test.include("test_Sigmoid_")

backend_test.include("test_size_")

backend_test.include("test_slice_")

backend_test.include("test_softmax_")
backend_test.exclude("test_softmax_axis_0")  # not supported
backend_test.exclude("test_softmax_axis_1")  # not supported
backend_test.exclude("test_softmax_large_number")  # not supported
backend_test.exclude("test_softmax_lastdim")  # Out of tolerance
# backend_test.include("test_Softmax")

backend_test.include("test_softplus_")
backend_test.include("test_softsign_")
backend_test.include("test_Softplus")

backend_test.include("test_spacetodepth")

backend_test.include("test_split_")
backend_test.exclude(".*split.*.*to.*.*sequence.*")

backend_test.include("test_sqrt_")
backend_test.include("test_operator_sqrt_")

backend_test.include("test_sub_")
backend_test.exclude("test_sub_bcast_")  # not supported
backend_test.exclude("test_sub_uint8_")  # not supported

backend_test.include("test_sum_")

backend_test.include("test_tanh_")
backend_test.include("test_Tanh_")

backend_test.include("test_tile_")

backend_test.include("test_top_k")

backend_test.include("test_transpose_")

backend_test.include("test_unsqueeze_")
backend_test.exclude("test_unsqueeze_negative_axes")  # 5D Array not supported
backend_test.exclude("test_unsqueeze_three_axes")  # 6D Array not supported
backend_test.exclude("test_unsqueeze_two_axes")  # 5D Array not supported
backend_test.exclude("test_unsqueeze_unsorted_axes")  # 5D Array not supported

backend_test.include("test_where_")
backend_test.exclude("test_where_long")  # not supported

backend_test.include("test_xor_")

backend_test.exclude(".*FLOAT*E*M*.*")
backend_test.exclude(".*ver18.*")

# This is a pytest magic variable to load extra plugins
pytest_plugins = ("onnx.backend.test.report",)

# import all test cases at global scope to make them visible to python.unittest
globals().update(backend_test.enable_report().test_cases)
