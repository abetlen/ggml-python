import onnx
from ggml.contrib.onnx import GgmlRuntimeBackend
import numpy as np
import onnx
import numpy as np
from onnx import helper


def test_onnx_run():
    x = helper.make_tensor_value_info("x", onnx.TensorProto.FLOAT, [1])
    a = helper.make_tensor_value_info("a", onnx.TensorProto.FLOAT, [1])
    x_constant = helper.make_tensor_value_info(
        "x_constant", onnx.TensorProto.FLOAT, [1]
    )
    b_constant = helper.make_tensor_value_info(
        "b_constant", onnx.TensorProto.FLOAT, [1]
    )

    f_output = helper.make_tensor_value_info("f", onnx.TensorProto.FLOAT, [1])

    square_node = helper.make_node("Mul", ["x", "x"], ["squared"], name="square_node")
    first_mul_node = helper.make_node(
        "Mul", ["squared", "a"], ["first_mul"], name="first_mul_node"
    )
    second_mul_node = helper.make_node(
        "Add", ["first_mul", "b_constant"], ["f"], name="second_mul_node"
    )

    graph_def = helper.make_graph(
        [square_node, first_mul_node, second_mul_node],
        "expression_graph",
        [x, a, x_constant, b_constant],
        [f_output],
    )

    onnx_model = helper.make_model(graph_def, producer_name="ONNX_expression_model")

    output = GgmlRuntimeBackend.prepare(onnx_model)

    x_val = np.array([2.0], dtype=np.float32)
    a_val = np.array([3.0], dtype=np.float32)
    x_constant_val = np.array([1.0], dtype=np.float32)
    b_constant_val = np.array([4.0], dtype=np.float32)

    input_data = {
        "x": x_val,
        "a": a_val,
        "x_constant": x_constant_val,
        "b_constant": b_constant_val,
    }

    print()
    print()
    print(output.run(input_data))
