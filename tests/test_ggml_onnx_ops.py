import io
from io import BytesIO

import numpy as np
import onnx
import onnxruntime as ort
import pytest
import torch
import torch.onnx

from onnx import TensorProto, helper
from onnxruntime import InferenceSession

import ggml
import ggml.utils
from ggml.contrib.onnx import GgmlRuntimeBackend, ggml_operators


def test_ggml_onnx_runtime_shape_operator():
    tensors_dict = {}

    params = ggml.ggml_init_params(mem_size=16 * 1024 * 1024, mem_buffer=None)
    context = ggml.ggml_init(params=params)

    test_list = [0, 1, 2, 3, 5, 6, 7, 8, 9, 10]

    input_data1 = np.array(test_list, dtype=np.int32)

    tensors_dict["input_tensor"] = ggml.utils.from_numpy(input_data1, context)

    tensors_dict["start1"] = ggml.utils.from_numpy(
        np.array([], dtype=np.int32), context
    )
    tensors_dict["end1"] = ggml.utils.from_numpy(np.array([], dtype=np.int32), context)

    tensors_dict["start2"] = ggml.utils.from_numpy(
        np.array([], dtype=np.int32), context
    )
    tensors_dict["end2"] = ggml.utils.from_numpy(np.array([6], dtype=np.int32), context)

    tensors_dict["start3"] = ggml.utils.from_numpy(
        np.array([2], dtype=np.int32), context
    )
    tensors_dict["end3"] = ggml.utils.from_numpy(np.array([6], dtype=np.int32), context)

    shape_node1 = onnx.NodeProto()
    shape_node1.op_type = "Shape"
    shape_node1.input.extend(["input_tensor"])
    shape_node1.output.extend(["output_tensor1"])

    shape_node2 = onnx.NodeProto()
    shape_node2.op_type = "Shape"
    shape_node2.input.extend(["input_tensor", "start1", "end1"])
    shape_node2.output.extend(["output_tensor2"])

    shape_node3 = onnx.NodeProto()
    shape_node3.op_type = "Shape"
    shape_node3.input.extend(["input_tensor", "start2", "end2"])
    shape_node3.output.extend(["output_tensor3"])

    shape_node4 = onnx.NodeProto()
    shape_node4.op_type = "Shape"
    shape_node4.input.extend(["input_tensor", "start3", "end3"])
    shape_node4.output.extend(["output_tensor4"])

    result1 = ggml_operators["Shape"](shape_node1, tensors_dict, context)
    result2 = ggml_operators["Shape"](shape_node2, tensors_dict, context)
    result3 = ggml_operators["Shape"](shape_node3, tensors_dict, context)
    result4 = ggml_operators["Shape"](shape_node4, tensors_dict, context)

    assert list(ggml.utils.to_numpy(result1) == test_list)
    assert list(ggml.utils.to_numpy(result2) == test_list)
    assert list(ggml.utils.to_numpy(result3) == test_list[:6])
    assert list(ggml.utils.to_numpy(result4) == test_list[2:6])

    ggml.ggml_free(context)


def test_ggml_onnx_runtime_unsqueeze_operator():
    return

    def onnx_unsqueeze(x, axes):
        # Create a simple PyTorch model
        class UnsqueezeModel(torch.nn.Module):
            def forward(self, input):
                for axis in axes:
                    input = torch.unsqueeze(input, dim=axis)
                return input

        model = UnsqueezeModel()

        # Create a sample input tensor
        x_tensor = torch.tensor(x, dtype=torch.int32)

        # Export the PyTorch model to ONNX
        f = BytesIO()
        torch.onnx.export(
            model,
            x_tensor,
            f,
            input_names=["data"],
            output_names=["output"],
            verbose=False,
        )

        # Save the ONNX model to BytesIO object
        onnx_model_bytes = BytesIO(f.getvalue())

        # Load the ONNX model from BytesIO
        onnx_model_bytes.seek(0)
        sess = ort.InferenceSession(onnx_model_bytes.read())

        # Convert the input array to ONNX format (numpy to list)
        x_list = x.tolist()
        input_feed = {"data": x_list}

        # Execute the ONNX model
        output = sess.run(None, input_feed)

        return np.array(output)

    tensors_dict = {}

    params = ggml.ggml_init_params(mem_size=16 * 1024 * 1024, mem_buffer=None)
    context = ggml.ggml_init(params=params)

    test_x = [0, 1, 2, 3, 5, 6]
    test_axes1 = np.array([1], dtype=np.int32)
    test_axes2 = np.array([0], dtype=np.int32)
    test_axes3 = np.array([1, 2], dtype=np.int32)

    input_data1 = np.array(test_x, dtype=np.int32)

    tensors_dict["input_tensor"] = ggml.utils.from_numpy(input_data1, context)

    tensors_dict["axes1"] = ggml.utils.from_numpy(test_axes1, context)
    tensors_dict["axes2"] = ggml.utils.from_numpy(test_axes2, context)
    tensors_dict["axes3"] = ggml.utils.from_numpy(test_axes3, context)

    unsqueeze_node1 = onnx.NodeProto()
    unsqueeze_node1.name = "Input error Test"
    unsqueeze_node1.op_type = "Unsqueeze"
    unsqueeze_node1.input.extend(["input_tensor"])
    unsqueeze_node1.output.extend(["output_tensor1"])

    unsqueeze_node2 = onnx.NodeProto()
    unsqueeze_node2.op_type = "Unsqueeze"
    unsqueeze_node2.input.extend(["input_tensor", "axes1"])
    unsqueeze_node2.output.extend(["output_tensor2"])

    unsqueeze_node3 = onnx.NodeProto()
    unsqueeze_node3.op_type = "Unsqueeze"
    unsqueeze_node3.input.extend(["input_tensor", "axes2"])
    unsqueeze_node3.output.extend(["output_tensor3"])

    unsqueeze_node4 = onnx.NodeProto()
    unsqueeze_node4.op_type = "Unsqueeze"
    unsqueeze_node4.input.extend(["input_tensor", "axes3"])
    unsqueeze_node4.output.extend(["output_tensor4"])

    with pytest.raises(ValueError) as ex_input_error:
        ggml_operators["Unsqueeze"](unsqueeze_node1, tensors_dict, context)
    result2 = ggml_operators["Unsqueeze"](unsqueeze_node2, tensors_dict, context)
    result3 = ggml_operators["Unsqueeze"](unsqueeze_node3, tensors_dict, context)
    result4 = ggml_operators["Unsqueeze"](unsqueeze_node4, tensors_dict, context)

    assert (
        str(ex_input_error.value)
        == 'Error for node "Input error Test": Operation "Unsqueeze" requires exactly two inputs, data and axes. Actual number of inputs: 1'
    )

    print(ggml.utils.to_numpy(result2), onnx_unsqueeze(input_data1, test_axes1))
    print(ggml.utils.to_numpy(result3), onnx_unsqueeze(input_data1, test_axes2))
    print(ggml.utils.to_numpy(result4), onnx_unsqueeze(input_data1, test_axes3))

    assert np.array_equal(
        ggml.utils.to_numpy(result2), onnx_unsqueeze(input_data1, test_axes1)
    )
    assert np.array_equal(
        ggml.utils.to_numpy(result3), onnx_unsqueeze(input_data1, test_axes2)
    )
    assert np.array_equal(
        ggml.utils.to_numpy(result4), onnx_unsqueeze(input_data1, test_axes3)
    )

    ggml.ggml_free(context)


def test_ggml_onnx_runtime_gather_operator():
    def onnx_gather(x, indices, axis):
        # Adjust the axis value to handle negative axis
        if axis < 0:
            axis += len(x.shape)

        # Create ONNX model for Gather operation with specified axis
        node_def = onnx.helper.make_node(
            "Gather", inputs=["data", "indices"], outputs=["output"], axis=axis
        )
        graph_def = onnx.helper.make_graph(
            [node_def],
            "gather_model",
            inputs=[
                onnx.helper.make_tensor_value_info(
                    "data", onnx.TensorProto.INT32, list(x.shape)
                ),
                onnx.helper.make_tensor_value_info(
                    "indices", onnx.TensorProto.INT32, list(indices.shape)
                ),
            ],
            outputs=[
                onnx.helper.make_tensor_value_info(
                    "output", onnx.TensorProto.INT32, list(x.shape)
                )
            ],
        )
        model_def = onnx.helper.make_model(
            graph_def, producer_name="onnx_gather_example"
        )

        # Save the ONNX model to BytesIO object
        onnx_model_bytes = BytesIO()
        onnx.save_model(model_def, onnx_model_bytes)

        # Load the ONNX model from BytesIO
        onnx_model_bytes.seek(0)
        sess = ort.InferenceSession(onnx_model_bytes.read())

        # Convert the input arrays to ONNX format (numpy to list)
        x_list = x.tolist()
        indices_list = indices.tolist()

        # Prepare the input feeds with the two arrays
        input_feed = {"data": x_list, "indices": indices_list}

        # Execute the ONNX model
        output = sess.run(None, input_feed)

        return np.array(output)

    tensors_dict = {}

    params = ggml.ggml_init_params(mem_size=16 * 1024 * 1024, mem_buffer=None)
    context = ggml.ggml_init(params=params)

    test_x = [
        [
            1046676483,
            -1102854076,
            -1089318038,
            1023432841,
            1041114519,
            -1099187814,
            1040889675,
            -1088007423,
            -1096868517,
            -1131772615,
            -1103856891,
            -1097108246,
            -1098364964,
            1024061975,
            -1102637477,
        ]
    ]
    test_indices1 = np.array([1], dtype=np.int32)

    input_data1 = np.array(test_x, dtype=np.int32)

    tensors_dict["input_tensor"] = ggml.utils.from_numpy(input_data1, context)
    tensors_dict["indices"] = ggml.utils.from_numpy(test_indices1, context)

    gather_node2 = onnx.helper.make_node(
        "Gather",
        name="/Gather",
        inputs=["input_tensor", "indices"],
        outputs=["output_tensor2"],
        axis=0,
    )

    result4 = ggml_operators["Gather"](gather_node2, tensors_dict, context)

    print(ggml.utils.to_numpy(result4), onnx_gather(input_data1, test_indices1, 0))

    assert np.array_equal(
        ggml.utils.to_numpy(result4), onnx_gather(input_data1, test_indices1, 0)
    )

    ggml.ggml_free(context)


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
