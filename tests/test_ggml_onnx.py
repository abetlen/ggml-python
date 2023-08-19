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


# This is a pytest magic variable to load extra plugins
pytest_plugins = ("onnx.backend.test.report",)

backend_test = onnx.backend.test.BackendTest(GgmlRuntimeBackend, __name__)
backend_test.include("test_abs_")
# backend_test.include("test_add_")
# backend_test.include("test_cast")
backend_test.include("test_concat_")
# backend_test.include("test_constant_")
# backend_test.include("test_div_")
# backend_test.include("test_gather_")
# backend_test.include("test_greater_")
# backend_test.include("test_less_")
# backend_test.include("test_log_")
# backend_test.include("test_matmul_")
# backend_test.include("test_max_")
# backend_test.include("test_min_")
# backend_test.include("test_mul_")
# backend_test.include("test_pow_")
# backend_test.include("test_range_")
# backend_test.include("test_reduce_mean_")
# backend_test.include("test_relu_")
backend_test.include("test_reshape_")
# backend_test.include("test_shape")
# backend_test.include("test_softmax_")
backend_test.include("test_sqrt_")
# backend_test.include("test_sub_")
# backend_test.include("test_transpose_")
# backend_test.include("test_unsqueeze_")
# backend_test.include("test_where_")

# backend_test.exclude(".*cuda.*")

# import all test cases at global scope to make them visible to python.unittest
globals().update(backend_test.enable_report().test_cases)
