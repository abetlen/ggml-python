import ctypes
import io
from io import BytesIO

import numpy as np
import onnx
import onnxruntime as ort
import pytest
import torch
import torch.onnx
from onnx import TensorProto, helper, numpy_helper
from onnxruntime import InferenceSession

import ggml
import ggml.utils
from ggml.contrib.onnx import GgmlRuntimeBackend, ggml_operators


def test_ggml_onnx_runtime_shape_operator():
    # return

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

    shape_node1 = onnx.helper.make_node(
        "Shape",
        name="Shape1",
        inputs=["input_tensor"],
        outputs=["output_tensor1"],
    )

    shape_node2 = onnx.helper.make_node(
        "Shape",
        name="Shape2",
        inputs=["input_tensor", "start1", "end1"],
        outputs=["output_tensor2"],
    )

    shape_node3 = onnx.helper.make_node(
        "Shape",
        name="Shape3",
        inputs=["input_tensor", "start2", "end2"],
        outputs=["output_tensor3"],
    )

    shape_node4 = onnx.helper.make_node(
        "Shape",
        name="Shape4",
        inputs=["input_tensor", "start3", "end3"],
        outputs=["output_tensor4"],
    )

    nodes = [shape_node1, shape_node2, shape_node3, shape_node4]
    results = []
    refs = []

    for shape_node in nodes:
        output_tensor = ggml_operators["Shape"](shape_node, tensors_dict, context, refs)
        gf = ggml.ggml_build_forward(output_tensor)
        ggml.ggml_graph_compute_with_ctx(context, ctypes.pointer(gf), 1)
        results.append(ggml.utils.to_numpy(output_tensor))

    assert results[0] == list(input_data1.shape)
    assert results[1] == list(input_data1.shape)
    assert results[2] == list(input_data1[:6].shape)
    assert results[3] == list(input_data1[2:6].shape)

    ggml.ggml_free(context)


def test_ggml_onnx_runtime_unsqueeze_operator():
    # return

    def onnx_unsqueeze(x, axes):
        class UnsqueezeModel(torch.nn.Module):
            def forward(self, input):
                for axis in axes:
                    input = torch.unsqueeze(input, dim=axis)
                return input

        model = UnsqueezeModel()

        x_tensor = torch.tensor(x, dtype=torch.int32)

        f = BytesIO()
        torch.onnx.export(
            model,
            x_tensor,
            f,
            input_names=["data"],
            output_names=["output"],
            verbose=False,
        )

        onnx_model_bytes = BytesIO(f.getvalue())

        onnx_model_bytes.seek(0)
        sess = ort.InferenceSession(onnx_model_bytes.read())

        x_list = x.tolist()
        input_feed = {"data": x_list}

        output = sess.run(None, input_feed)

        return output[0]

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

    refs = []
    nodes = [unsqueeze_node2, unsqueeze_node3, unsqueeze_node4]
    results = []

    with pytest.raises(ValueError) as ex_input_error:
        ggml_operators["Unsqueeze"](unsqueeze_node1, tensors_dict, context, refs)

    for shape_node in nodes:
        output_tensor = ggml_operators["Unsqueeze"](
            shape_node, tensors_dict, context, refs
        )

        gf = ggml.ggml_build_forward(output_tensor)

        ggml.ggml_graph_compute_with_ctx(context, ctypes.pointer(gf), 1)

        results.append(ggml.utils.to_numpy(output_tensor))

    assert (
        str(ex_input_error.value)
        == 'Error for node "Input error Test": Operation "Unsqueeze" requires exactly two inputs, data and axes. Actual number of inputs: 1'
    )

    assert np.array_equal(results[0], onnx_unsqueeze(input_data1, test_axes1))
    assert np.array_equal(results[1], onnx_unsqueeze(input_data1, test_axes2))
    assert np.array_equal(results[2], onnx_unsqueeze(input_data1, test_axes3))

    ggml.ggml_free(context)


def test_ggml_onnx_runtime_gather_operator():
    # return

    def onnx_gather(x, indices, axis):
        if axis < 0:
            axis += len(x.shape)

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

        onnx_model_bytes = BytesIO()
        onnx.save_model(model_def, onnx_model_bytes)

        onnx_model_bytes.seek(0)
        sess = ort.InferenceSession(onnx_model_bytes.read())

        x_list = x.tolist()
        indices_list = indices.tolist()

        input_feed = {"data": x_list, "indices": indices_list}

        output = sess.run(None, input_feed)

        return output[0]

    tensors_dict = {}

    params = ggml.ggml_init_params(mem_size=16 * 1024 * 1024, mem_buffer=None)
    context = ggml.ggml_init(params=params)

    test_x = [
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
    test_indices1 = np.array([1], dtype=np.int32)

    input_data1 = np.array(test_x, dtype=np.int32)

    input_tensor = ggml.utils.from_numpy(input_data1, context)
    indices_tensor = ggml.utils.from_numpy(test_indices1, context)

    tensors_dict["input_tensor"] = input_tensor
    tensors_dict["indices"] = indices_tensor

    gather_node2 = onnx.helper.make_node(
        "Gather",
        name="/Gather",
        inputs=["input_tensor", "indices"],
        outputs=["output_tensor2"],
        axis=0,
    )

    refs = []

    output_tensor = ggml_operators["Gather"](gather_node2, tensors_dict, context, refs)

    gf = ggml.ggml_build_forward(output_tensor)

    ggml.ggml_graph_compute_with_ctx(context, ctypes.pointer(gf), 1)
    output_tensor = ggml.ggml_get_tensor(context, ggml.ggml_get_name(output_tensor))

    assert np.array_equal(
        ggml.utils.to_numpy(output_tensor), onnx_gather(input_data1, test_indices1, 0)
    )

    ggml.ggml_free(context)


def test_ggml_onnx_constant_operator():
    # return

    def onnx_constant(value, dtype, shape):
        tensor = numpy_helper.from_array(value)
        constant_node = onnx.helper.make_node(
            "Constant", inputs=[], outputs=["constant_output"], value=tensor
        )
        graph = onnx.helper.make_graph(
            [constant_node],
            "constant_graph",
            inputs=[],
            outputs=[
                onnx.helper.make_tensor_value_info("constant_output", dtype, shape)
            ],
        )
        model = onnx.helper.make_model(graph)

        return numpy_helper.to_array(model.graph.node[0].attribute[0].t)

    params = ggml.ggml_init_params(mem_size=16 * 1024 * 1024, mem_buffer=None)
    context = ggml.ggml_init(params=params)
    tensors_dict = {}

    constant1 = np.array([1], dtype=np.int32)
    constant2 = np.array([[1]], dtype=np.int32)
    constant3 = np.array([[1, 2], [3, 4], [6, 6]], dtype=np.int32)
    constant4 = np.array(6, dtype=np.int64)

    dtype = onnx.TensorProto.INT32

    constant_numpy1 = onnx_constant(constant1, dtype, constant1.shape)
    constant_numpy2 = onnx_constant(constant2, dtype, constant2.shape)
    constant_numpy3 = onnx_constant(constant3, dtype, constant3.shape)
    constant_numpy4 = onnx_constant(constant4, dtype, constant4.shape)

    constant_node1 = onnx.helper.make_node(
        "Constant",
        inputs=[],
        name="constant_node1",
        outputs=["constant_output1"],
        value=numpy_helper.from_array(constant1),
    )
    constant_node2 = onnx.helper.make_node(
        "Constant",
        name="constant_node2",
        inputs=[],
        outputs=["constant_output2"],
        value=numpy_helper.from_array(constant2),
    )
    constant_node3 = onnx.helper.make_node(
        "Constant",
        name="constant_node3",
        inputs=[],
        outputs=["constant_output3"],
        value=numpy_helper.from_array(constant3),
    )

    constant_node4 = onnx.helper.make_node(
        "Constant",
        name="constant_node3",
        inputs=[],
        outputs=["constant_output3"],
        value=numpy_helper.from_array(constant4),
    )

    nodes = [constant_node1, constant_node2, constant_node3, constant_node4]
    results = []
    refs = []

    for shape_node in nodes:
        output_tensor = ggml_operators["Constant"](
            shape_node, tensors_dict, context, refs
        )
        gf = ggml.ggml_build_forward(output_tensor)
        ggml.ggml_graph_compute_with_ctx(context, ctypes.pointer(gf), 1)
        results.append(ggml.utils.to_numpy(output_tensor))

    assert np.array_equal(results[0], constant_numpy1)
    assert np.array_equal(results[1], constant_numpy2)
    assert np.array_equal(results[2], constant_numpy3)
    assert results[3] == constant_numpy4

    ggml.ggml_free(context)


def test_ggml_onnx_constant_of_shape_operator():
    # return

    def onnx_constant(value, other):
        shape = list(other.shape)
        value = numpy_helper.from_array(value)
        constant_node = onnx.helper.make_node(
            "ConstantOfShape", inputs=["data"], outputs=["constant_output"], value=value
        )
        graph = onnx.helper.make_graph(
            [constant_node],
            "constant_graph",
            inputs=[
                onnx.helper.make_tensor_value_info(
                    "data", onnx.TensorProto.INT64, shape
                )
            ],
            outputs=[
                onnx.helper.make_tensor_value_info(
                    "constant_output", onnx.TensorProto.FLOAT, shape
                )
            ],
        )
        model = onnx.helper.make_model(graph)

        onnx_model_bytes = BytesIO()
        onnx.save_model(model, onnx_model_bytes)

        onnx_model_bytes.seek(0)
        sess = ort.InferenceSession(onnx_model_bytes.read())

        x_list = other.tolist()
        input_feed = {"data": x_list}

        output = sess.run(None, input_feed)

        return output[0]

    params = ggml.ggml_init_params(mem_size=16 * 1024 * 1024, mem_buffer=None)
    context = ggml.ggml_init(params=params)
    tensors_dict = {}

    shape1 = np.array([2, 3, 4], dtype=np.int32)
    value_tensor = np.array([15], dtype=np.float32)

    cof_node1 = onnx.helper.make_node(
        "ConstantOfShape",
        inputs=["shape1"],
        name="cof_node1",
        outputs=["cof_output"],
        value=numpy_helper.from_array(value_tensor),
    )

    tensors_dict["shape1"] = ggml.utils.from_numpy(shape1, context)

    constant_onnx = onnx_constant(value_tensor, shape1)

    nodes = [cof_node1]
    results = []
    refs = []

    for shape_node in nodes:
        output_tensor = ggml_operators["ConstantOfShape"](
            shape_node, tensors_dict, context, refs
        )
        gf = ggml.ggml_build_forward(output_tensor)
        ggml.ggml_graph_compute_with_ctx(context, ctypes.pointer(gf), 1)
        results.append(ggml.utils.to_numpy(output_tensor))
    assert np.array_equal(results[0], constant_onnx)

    ggml.ggml_free(context)


def test_ggml_onnx_concat_operator():
    return

    def onnx_concat(inputs, axis):
        input_data_type = inputs[0].dtype

        graph = onnx.GraphProto()

        input_names = []
        for i, input_array in enumerate(inputs):
            input_name = f"input{i}"
            input_names.append(input_name)

            input_value_info = onnx.helper.make_tensor_value_info(
                input_name,
                onnx.TensorProto.FLOAT
                if input_data_type == np.float32
                else onnx.TensorProto.INT32,
                input_array.shape,
            )
            graph.input.extend([input_value_info])

        concat_node = onnx.NodeProto()
        concat_node.op_type = "Concat"
        concat_node.name = "concat_node"
        concat_node.output.extend(["output"])
        concat_node.attribute.extend([onnx.helper.make_attribute("axis", axis)])
        concat_node.input.extend(input_names)

        output_value_info = onnx.helper.make_tensor_value_info(
            "output",
            onnx.TensorProto.FLOAT
            if input_data_type == np.float32
            else onnx.TensorProto.INT32,
            None,
        )
        graph.output.extend([output_value_info])

        graph.node.extend([concat_node])
        model = onnx.helper.make_model(graph)

        onnx_model_bytes = BytesIO()
        onnx.save_model(model, onnx_model_bytes)

        # Load the ONNX model from BytesIO
        onnx_model_bytes.seek(0)
        sess = ort.InferenceSession(onnx_model_bytes.read())

        input_feed = {
            input_name: input_array
            for input_name, input_array in zip(input_names, inputs)
        }

        output = sess.run(["output"], input_feed)

        return output[0]

    params = ggml.ggml_init_params(mem_size=16 * 1024 * 1024, mem_buffer=None)
    context = ggml.ggml_init(params=params)
    tensors_dict = {}

    array1 = np.array([1], dtype=np.int32)
    array2 = np.array([2, 3, 4, 5], dtype=np.int32)
    array3 = np.array([6], dtype=np.int32)
    array4 = np.array([7, 8, 9, 10], dtype=np.int32)

    tensors_dict["array1"] = ggml.utils.from_numpy(array1, context)
    tensors_dict["array2"] = ggml.utils.from_numpy(array2, context)
    tensors_dict["array3"] = ggml.utils.from_numpy(array3, context)
    tensors_dict["array4"] = ggml.utils.from_numpy(array4, context)

    test1 = ["array1", "array2"]
    inputs1 = [array1, array2]
    test2 = ["array1", "array2", "array3", "array4"]
    inputs2 = [array1, array2, array3, array4]
    axis = 0

    concat_node1 = onnx.helper.make_node(
        "Concat",
        inputs=test1,
        name="concat_node1",
        outputs=["concat_output1"],
        axis=axis,
    )
    concat_node2 = onnx.helper.make_node(
        "Concat",
        inputs=test2,
        name="concat_node2",
        outputs=["concat_output2"],
        axis=axis,
    )

    concat_onnx_result1 = onnx_concat(inputs1, axis)
    concat_onnx_result2 = onnx_concat(inputs2, axis)

    nodes = [concat_node1, concat_node2]
    results = []
    refs = []

    for concat_node in nodes:
        output_tensor = ggml_operators["Concat"](
            concat_node, tensors_dict, context, refs
        )
        gf = ggml.ggml_build_forward(output_tensor)
        ggml.ggml_graph_compute_with_ctx(context, ctypes.pointer(gf), 1)
        results.append(ggml.utils.to_numpy(output_tensor))

    assert np.array_equal(results[0], concat_onnx_result1)
    assert np.array_equal(results[1], concat_onnx_result2)


def test_ggml_onnx_reshape_operation():
    return

    def onnx_reshape(input_tensor, shape):
        class DynamicReshapeModel(torch.nn.Module):
            def __init__(self, shape):
                super(DynamicReshapeModel, self).__init__()
                self.shape = tuple(shape)

            def forward(self, x):
                reshaped = torch.reshape(x, self.shape)
                return reshaped

        if not isinstance(input_tensor, np.ndarray):
            raise ValueError("Input tensor must be a NumPy array")

        if not isinstance(shape, np.ndarray):
            shape = np.array(shape)

        if len(shape) != len(input_tensor.shape):
            raise ValueError(
                "Input shape must have the same number of dimensions as the input tensor"
            )

        model = DynamicReshapeModel(shape)

        input_tensor = torch.tensor(input_tensor, dtype=torch.int32)

        f = BytesIO()
        torch.onnx.export(
            model, input_tensor, f, opset_version=12, do_constant_folding=True
        )
        f.seek(0)

        sess = ort.InferenceSession(f.getvalue())
        input_name = sess.get_inputs()[0].name
        output_name = sess.get_outputs()[0].name

        result = sess.run([output_name], {input_name: input_tensor.numpy()})

        return result[0]

    input_tensor = np.array([[1, 2, 3, 4, 5, 6]], dtype=np.int32)
    new_shape = np.array([2, 3], dtype=np.int32)

    params = ggml.ggml_init_params(mem_size=16 * 1024 * 1024, mem_buffer=None)
    context = ggml.ggml_init(params=params)
    tensors_dict = {}

    tensors_dict["input_tensor"] = ggml.utils.from_numpy(input_tensor, context)
    tensors_dict["new_shape"] = ggml.utils.from_numpy(new_shape, context)

    reshape_node1 = onnx.helper.make_node(
        "Reshape",
        inputs=["input_tensor", "new_shape"],
        name="reshape_node1",
        outputs=["reshape_output1"],
    )

    nodes = [reshape_node1]
    results = []
    refs = []

    for reshape_node in nodes:
        output_tensor = ggml_operators["Reshape"](
            reshape_node, tensors_dict, context, refs
        )
        gf = ggml.ggml_build_forward(output_tensor)
        ggml.ggml_graph_compute_with_ctx(context, ctypes.pointer(gf), 1)
        results.append(ggml.utils.to_numpy(output_tensor))

    assert np.array_equal(results[0], onnx_reshape(input_tensor, new_shape))


def test_ggml_onnx_softmax_operator():
    # return

    input_name = "X"

    output_name = "Softmax_Output"

    input_data = {
        input_name: np.array([[-1.5, 0.001, 3.73, 5.1, 6, 6.0001]], dtype=np.float32)
    }

    node1 = helper.make_node(
        "Softmax", [input_name], [output_name], name="softmax_node"
    )

    X_value_info = helper.make_tensor_value_info(
        input_name, TensorProto.FLOAT, [None, 6]
    )

    softmax_value_info = helper.make_tensor_value_info(
        output_name, TensorProto.FLOAT, [None, 6]
    )

    graph_def = helper.make_graph(
        [node1], "softmax_model", [X_value_info], [softmax_value_info]
    )

    model_def = helper.make_model(graph_def, producer_name="onnx-softmax")

    f = io.BytesIO()
    onnx.save(model_def, f)

    runtime_result = InferenceSession(f.getvalue()).run(None, input_data)

    ggml_dummy_model = GgmlRuntimeBackend.prepare(model_def)

    ggml_result = ggml_dummy_model.run(input_data)

    assert np.allclose(ggml_result, runtime_result, rtol=0.001)


def test_ggml_onnx_reducemean_operator():
    # return

    def onnx_reducemean(x, axes):
        class ReduceMeanModel(torch.nn.Module):
            def forward(self, input):
                return torch.mean(input, dim=axes.tolist(), keepdim=False)

        model = ReduceMeanModel()

        x_tensor = torch.tensor(x, dtype=torch.float32)

        f = BytesIO()
        torch.onnx.export(
            model,
            x_tensor,
            f,
            input_names=["data"],
            output_names=["output"],
            verbose=False,
        )

        onnx_model_bytes = BytesIO(f.getvalue())

        onnx_model_bytes.seek(0)
        sess = ort.InferenceSession(onnx_model_bytes.read())

        x_list = x.tolist()
        input_feed = {"data": x_list}

        output = sess.run(None, input_feed)

        return output[0]

    params = ggml.ggml_init_params(mem_size=16 * 1024 * 1024, mem_buffer=None)
    context = ggml.ggml_init(params=params)
    tensors_dict = {}
    refs = []

    input_array1 = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=np.float32)
    axes1 = np.array([0, 1], dtype=np.int32)

    input_array2 = np.array([[1, 2, 3, 4]], dtype=np.float32)
    axes2 = np.array([1], dtype=np.int32)

    reducemean_numpy1 = onnx_reducemean(input_array1, axes1)
    reducemean_numpy2 = onnx_reducemean(input_array2, axes2)

    tensors_dict["input_array1"] = ggml.utils.from_numpy(input_array1, context)
    tensors_dict["axes1"] = ggml.utils.from_numpy(axes1, context)
    tensors_dict["input_array2"] = ggml.utils.from_numpy(input_array2, context)
    tensors_dict["axes2"] = ggml.utils.from_numpy(axes2, context)

    reducemean_node1 = onnx.helper.make_node(
        "ReduceMean",
        inputs=["input_array1"],
        outputs=["reducemean_output1"],
        axes=axes1,
        keepdims=0,
    )

    reducemean_node2 = onnx.helper.make_node(
        "ReduceMean",
        inputs=["input_array2"],
        outputs=["reducemean_output2"],
        axes=axes2,
        keepdims=0,
    )

    nodes = [reducemean_node1, reducemean_node2]
    results = []

    for reducemean_node in nodes:
        output_tensor = ggml_operators["ReduceMean"](
            reducemean_node, tensors_dict, context, refs
        )
        gf = ggml.ggml_build_forward(output_tensor)
        ggml.ggml_graph_compute_with_ctx(context, ctypes.pointer(gf), 1)
        results.append(ggml.utils.to_numpy(output_tensor))

    assert np.allclose(results[0], reducemean_numpy1)
    assert np.allclose(results[1], reducemean_numpy2)

    ggml.ggml_free(context)


def test_ggml_onnx_less_operator():
    # return

    def onnx_less(x, y):
        class LessModel(torch.nn.Module):
            def forward(self, input1, input2):
                return torch.less(input1, input2)

        model = LessModel()

        x_tensor = torch.tensor(x, dtype=torch.float32)
        y_tensor = torch.tensor(y, dtype=torch.float32)

        f = BytesIO()
        torch.onnx.export(
            model,
            (x_tensor, y_tensor),
            f,
            input_names=["input1", "input2"],
            output_names=["output"],
            verbose=False,
        )

        onnx_model_bytes = BytesIO(f.getvalue())

        onnx_model_bytes.seek(0)
        sess = ort.InferenceSession(onnx_model_bytes.read())

        x_list = x.tolist()
        y_list = y.tolist()
        input_feed = {"input1": x_list, "input2": y_list}

        output = sess.run(None, input_feed)

        return output[0]

    params = ggml.ggml_init_params(mem_size=16 * 1024 * 1024, mem_buffer=None)
    context = ggml.ggml_init(params=params)
    tensors_dict = {}
    refs = []

    input_array1 = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=np.float32)
    input_array2 = np.array([[2, 2, 2], [5, 5, 5], [8, 8, 8]], dtype=np.float32)

    less_numpy = onnx_less(input_array1, input_array2)

    tensors_dict["input_array1"] = ggml.utils.from_numpy(input_array1, context)
    tensors_dict["input_array2"] = ggml.utils.from_numpy(input_array2, context)

    less_node = onnx.helper.make_node(
        "Less",
        inputs=["input_array1", "input_array2"],
        outputs=["less_output"],
    )

    nodes = [less_node]
    results = []

    for less_node in nodes:
        output_tensor = ggml_operators["Less"](less_node, tensors_dict, context, refs)
        gf = ggml.ggml_build_forward(output_tensor)
        ggml.ggml_graph_compute_with_ctx(context, ctypes.pointer(gf), 1)
        results.append(ggml.utils.to_numpy(output_tensor))

    assert np.allclose(results[0], less_numpy)

    ggml.ggml_free(context)


def test_ggml_onnx_greater_operator():
    # return

    def onnx_greater(x, y):
        class GreaterModel(torch.nn.Module):
            def forward(self, input1, input2):
                return torch.gt(input1, input2)

        model = GreaterModel()

        x_tensor = torch.tensor(x, dtype=torch.float32)
        y_tensor = torch.tensor(y, dtype=torch.float32)

        f = BytesIO()
        torch.onnx.export(
            model,
            (x_tensor, y_tensor),
            f,
            input_names=["input1", "input2"],
            output_names=["output"],
            verbose=False,
        )

        onnx_model_bytes = BytesIO(f.getvalue())

        onnx_model_bytes.seek(0)
        sess = ort.InferenceSession(onnx_model_bytes.read())

        x_list = x.tolist()
        y_list = y.tolist()
        input_feed = {"input1": x_list, "input2": y_list}

        output = sess.run(None, input_feed)

        return output[0]

    params = ggml.ggml_init_params(mem_size=16 * 1024 * 1024, mem_buffer=None)
    context = ggml.ggml_init(params=params)
    tensors_dict = {}
    refs = []

    input_array1 = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=np.float32)
    input_array2 = np.array([[2, 2, 2], [5, 5, 5], [8, 8, 8]], dtype=np.float32)

    greater_numpy = onnx_greater(input_array1, input_array2)

    tensors_dict["input_array1"] = ggml.utils.from_numpy(input_array1, context)
    tensors_dict["input_array2"] = ggml.utils.from_numpy(input_array2, context)

    greater_node = onnx.helper.make_node(
        "Greater",
        inputs=["input_array1", "input_array2"],
        outputs=["greater_output"],
    )

    output_tensor = ggml_operators["Greater"](greater_node, tensors_dict, context, refs)
    gf = ggml.ggml_build_forward(output_tensor)
    ggml.ggml_graph_compute_with_ctx(context, ctypes.pointer(gf), 1)
    result = ggml.utils.to_numpy(output_tensor)

    assert np.allclose(result, greater_numpy)

    ggml.ggml_free(context)


def test_ggml_onnx_min_operator():
    # return

    def onnx_min(x):
        class MinModel(torch.nn.Module):
            def forward(self, input1):
                return torch.min(input1)

        model = MinModel()

        x_tensor = torch.tensor(x, dtype=torch.float32)

        f = BytesIO()
        torch.onnx.export(
            model,
            (x_tensor),
            f,
            input_names=["input1"],
            output_names=["output"],
            verbose=False,
        )

        onnx_model_bytes = BytesIO(f.getvalue())

        onnx_model_bytes.seek(0)
        sess = ort.InferenceSession(onnx_model_bytes.read())

        x_list = x.tolist()
        input_feed = {"input1": x_list}

        output = sess.run(None, input_feed)

        return output[0]

    params = ggml.ggml_init_params(mem_size=16 * 1024 * 1024, mem_buffer=None)
    context = ggml.ggml_init(params=params)
    tensors_dict = {}
    refs = []

    input_array1 = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=np.float32)

    min_numpy = onnx_min(input_array1)

    tensors_dict["input_array1"] = ggml.utils.from_numpy(input_array1, context)

    min_node = onnx.helper.make_node(
        "Min",
        inputs=["input_array1"],
        outputs=["min_output"],
    )

    output_tensor = ggml_operators["Min"](min_node, tensors_dict, context, refs)
    gf = ggml.ggml_build_forward(output_tensor)
    ggml.ggml_graph_compute_with_ctx(context, ctypes.pointer(gf), 1)
    result = ggml.utils.to_numpy(output_tensor)

    assert np.allclose(result, min_numpy)

    ggml.ggml_free(context)


def test_ggml_onnx_max_operator():
    # return

    def onnx_max(x):
        class MaxModel(torch.nn.Module):
            def forward(self, input1):
                return torch.max(input1)

        model = MaxModel()

        x_tensor = torch.tensor(x, dtype=torch.float32)

        f = BytesIO()
        torch.onnx.export(
            model,
            (x_tensor),
            f,
            input_names=["input1"],
            output_names=["output"],
            verbose=False,
        )

        onnx_model_bytes = BytesIO(f.getvalue())

        onnx_model_bytes.seek(0)
        sess = ort.InferenceSession(onnx_model_bytes.read())

        x_list = x.tolist()
        input_feed = {"input1": x_list}

        output = sess.run(None, input_feed)

        return output[0]

    params = ggml.ggml_init_params(mem_size=16 * 1024 * 1024, mem_buffer=None)
    context = ggml.ggml_init(params=params)
    tensors_dict = {}
    refs = []

    input_array1 = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=np.float32)
    min_numpy = onnx_max(input_array1)
    tensors_dict["input_array1"] = ggml.utils.from_numpy(input_array1, context)

    min_node = onnx.helper.make_node(
        "Min",
        inputs=["input_array1"],
        outputs=["min_output"],
    )

    output_tensor = ggml_operators["Max"](min_node, tensors_dict, context, refs)
    gf = ggml.ggml_build_forward(output_tensor)
    ggml.ggml_graph_compute_with_ctx(context, ctypes.pointer(gf), 1)
    result = ggml.utils.to_numpy(output_tensor)

    assert np.allclose(result, min_numpy)

    ggml.ggml_free(context)


def test_ggml_onnx_matmul_operator():
    return

    def onnx_matmul(x1, x2):
        class MatMulModel(torch.nn.Module):
            def forward(self, input1, input2):
                return torch.matmul(input1, input2)

        model = MatMulModel()

        x1_tensor = torch.tensor(x1, dtype=torch.float32)
        x2_tensor = torch.tensor(x2, dtype=torch.float32)

        f = BytesIO()
        torch.onnx.export(
            model,
            (x1_tensor, x2_tensor),
            f,
            input_names=["input1", "input2"],
            output_names=["output"],
            verbose=False,
        )

        onnx_model_bytes = BytesIO(f.getvalue())

        onnx_model_bytes.seek(0)
        sess = ort.InferenceSession(onnx_model_bytes.read())

        x1_list = x1.tolist()
        x2_list = x2.tolist()
        input_feed = {"input1": x1_list, "input2": x2_list}

        output = sess.run(None, input_feed)

        return output[0]

    params = ggml.ggml_init_params(mem_size=16 * 1024 * 1024, mem_buffer=None)
    context = ggml.ggml_init(params=params)
    tensors_dict = {}
    refs = []

    # Define input arrays
    input_array1 = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=np.float32)
    input_array2 = np.array([[9, 8, 7], [6, 5, 4], [3, 2, 1]], dtype=np.float32)

    # Compute ONNX MatMul using GGML
    matmul_numpy = onnx_matmul(input_array1, input_array2)

    tensors_dict["input_array1"] = ggml.utils.from_numpy(input_array1, context)
    tensors_dict["input_array2"] = ggml.utils.from_numpy(input_array2, context)

    matmul_node = onnx.helper.make_node(
        "MatMul",
        inputs=["input_array1", "input_array2"],
        outputs=["matmul_output"],
    )

    output_tensor = ggml_operators["MatMul"](matmul_node, tensors_dict, context, refs)
    gf = ggml.ggml_build_forward(output_tensor)
    ggml.ggml_graph_compute_with_ctx(context, ctypes.pointer(gf), 1)
    result = ggml.utils.to_numpy(output_tensor)

    assert np.allclose(result, matmul_numpy)

    ggml.ggml_free(context)


def test_ggml_onnx_pow_operator():
    return

    def onnx_pow(x, y):
        class PowModel(torch.nn.Module):
            def forward(self, input1, input2):
                return torch.pow(input1, input2)

        model = PowModel()

        x_tensor = torch.tensor(x, dtype=torch.float32)
        y_tensor = torch.tensor(y, dtype=torch.float32)

        f = BytesIO()
        torch.onnx.export(
            model,
            (x_tensor, y_tensor),
            f,
            input_names=["input1", "input2"],
            output_names=["output"],
            verbose=False,
        )

        onnx_model_bytes = BytesIO(f.getvalue())

        onnx_model_bytes.seek(0)
        sess = ort.InferenceSession(onnx_model_bytes.read())

        x_list = x.tolist()
        y_list = y.tolist()
        input_feed = {"input1": x_list, "input2": y_list}

        output = sess.run(None, input_feed)

        return output[0]

    params = ggml.ggml_init_params(mem_size=16 * 1024 * 1024, mem_buffer=None)
    context = ggml.ggml_init(params=params)
    tensors_dict = {}
    refs = []

    input_array1 = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=np.float32)
    input_array2 = np.array([[2, 2, 2], [3, 3, 3], [4, 4, 4]], dtype=np.float32)

    pow_numpy = onnx_pow(input_array1, input_array2)

    tensors_dict["input_array1"] = ggml.utils.from_numpy(input_array1, context)
    tensors_dict["input_array2"] = ggml.utils.from_numpy(input_array2, context)

    pow_node = onnx.helper.make_node(
        "Pow",
        inputs=["input_array1", "input_array2"],
        outputs=["pow_output"],
    )

    nodes = [pow_node]
    results = []

    for pow_node in nodes:
        output_tensor = ggml_operators["Pow"](pow_node, tensors_dict, context, refs)
        gf = ggml.ggml_build_forward(output_tensor)
        ggml.ggml_graph_compute_with_ctx(context, ctypes.pointer(gf), 1)
        results.append(ggml.utils.to_numpy(output_tensor))

    assert np.allclose(results[0], pow_numpy)

    ggml.ggml_free(context)


def test_ggml_onnx_relu_operator():
    # return

    def onnx_relu(x):
        class ReluModel(torch.nn.Module):
            def forward(self, input):
                return torch.relu(input)

        model = ReluModel()

        x_tensor = torch.tensor(x, dtype=torch.float32)

        f = BytesIO()
        torch.onnx.export(
            model,
            (x_tensor,),
            f,
            input_names=["input"],
            output_names=["output"],
            verbose=False,
        )

        onnx_model_bytes = BytesIO(f.getvalue())

        onnx_model_bytes.seek(0)
        sess = ort.InferenceSession(onnx_model_bytes.read())

        x_list = x.tolist()
        input_feed = {"input": x_list}

        output = sess.run(None, input_feed)

        return output[0]

    params = ggml.ggml_init_params(mem_size=16 * 1024 * 1024, mem_buffer=None)
    context = ggml.ggml_init(params=params)
    tensors_dict = {}
    refs = []

    input_array = np.array([[1, -2, 3], [-4, 5, -6], [7, -8, 9]], dtype=np.float32)

    relu_numpy = onnx_relu(input_array)

    tensors_dict["input_array"] = ggml.utils.from_numpy(input_array, context)

    relu_node = onnx.helper.make_node(
        "Relu",
        inputs=["input_array"],
        outputs=["relu_output"],
    )

    nodes = [relu_node]
    results = []

    for relu_node in nodes:
        output_tensor = ggml_operators["Relu"](relu_node, tensors_dict, context, refs)
        gf = ggml.ggml_build_forward(output_tensor)
        ggml.ggml_graph_compute_with_ctx(context, ctypes.pointer(gf), 1)
        results.append(ggml.utils.to_numpy(output_tensor))

    assert np.allclose(results[0], relu_numpy)

    ggml.ggml_free(context)


def test_ggml_onnx_transpose_operator():
    # return

    def onnx_transpose(x, dim0=1, dim1=0):
        transpose_node = onnx.helper.make_node(
            "Transpose", inputs=["input"], outputs=["output"], perm=[dim0, dim1]
        )

        graph = onnx.helper.make_graph(
            [transpose_node],
            "transpose_graph",
            inputs=[
                onnx.helper.make_tensor_value_info(
                    "input", onnx.TensorProto.FLOAT, list(x.shape)
                )
            ],
            outputs=[
                onnx.helper.make_tensor_value_info(
                    "output", onnx.TensorProto.FLOAT, list(x.shape)
                )
            ],
        )

        model = onnx.helper.make_model(graph)

        f = BytesIO()
        onnx.save_model(model, f)

        onnx_model_bytes = BytesIO(f.getvalue())

        onnx_model_bytes.seek(0)
        sess = ort.InferenceSession(onnx_model_bytes.read())

        input_feed = {"input": x}

        output = sess.run(None, input_feed)

        return output[0]

    params = ggml.ggml_init_params(mem_size=16 * 1024 * 1024, mem_buffer=None)
    context = ggml.ggml_init(params=params)
    tensors_dict = {}
    refs = []

    input_array = np.array([[1, -2, 3], [-4, 5, -6], [7, -8, 9]], dtype=np.float32)

    transpose_numpy = onnx_transpose(input_array, 1, 0)

    tensors_dict["input_array"] = ggml.utils.from_numpy(input_array, context)

    transpose_node = onnx.helper.make_node(
        "Transpose",
        inputs=["input_array"],
        outputs=["transpose_output"],
        perm=[1, 0],
    )

    nodes = [transpose_node]
    results = []

    for node in nodes:
        output_tensor = ggml_operators["Transpose"](node, tensors_dict, context, refs)
        gf = ggml.ggml_build_forward(output_tensor)
        ggml.ggml_graph_compute_with_ctx(context, ctypes.pointer(gf), 1)
        results.append(ggml.utils.to_numpy(output_tensor))

    assert np.allclose(results[0], transpose_numpy)

    ggml.ggml_free(context)


def test_ggml_onnx_runtime_basic():
    # return

    input_name = "X"

    weight_name_a = "A"
    weight_name_b = "B"
    weight_name_c = "C"
    weight_name_d = "D"

    intermediate_name1 = "intermediate1"
    intermediate_name2 = "intermediate2"
    intermediate_name3 = "intermediate3"
    intermediate_name4 = "intermediate4"
    intermediate_name5 = "intermediate5"
    intermediate_name6 = "intermediate6"

    output_name = "Y"

    node1 = helper.make_node(
        "Mul", [input_name, weight_name_a], [intermediate_name1], name="node1"
    )  # X * A
    node2 = helper.make_node(
        "Div", [intermediate_name1, weight_name_b], [intermediate_name2], name="node2"
    )  # (X * A) / B
    node3 = helper.make_node(
        "Add", [intermediate_name2, weight_name_c], [intermediate_name3], name="node3"
    )  # (X * A / B) + C
    node4 = helper.make_node(
        "Sub", [intermediate_name3, weight_name_d], [intermediate_name4], name="node4"
    )  # (X * A / B) + C - D
    node5 = helper.make_node(
        "Sqrt", [intermediate_name4], [intermediate_name5], name="node5"
    )  # Sqrt((X * A / B) + C - D)
    node6 = helper.make_node(
        "Log", [intermediate_name5], [intermediate_name6], name="node6"
    )  # Log(Sqrt((X * A / B) + C - D))
    node7 = helper.make_node(
        "Abs", [intermediate_name6], [output_name], name="node7"
    )  # Abs(Log(Sqrt((X * A / B) + C - D)))

    X_value_info = helper.make_tensor_value_info(
        input_name, TensorProto.FLOAT, [None, 1]
    )

    output_value_info = helper.make_tensor_value_info(
        output_name, TensorProto.FLOAT, [None, 1]
    )

    weights_a = np.array([50.6], dtype=float).astype(np.float32)
    weights_b = np.array([0.0013], dtype=float).astype(np.float32)
    weights_c = np.array([8.1], dtype=float).astype(np.float32)
    weights_d = np.array([13.22], dtype=float).astype(np.float32)

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
    C_init = helper.make_tensor(
        weight_name_c,
        TensorProto.FLOAT,
        [
            1,
        ],
        weights_c,
    )
    D_init = helper.make_tensor(
        weight_name_d,
        TensorProto.FLOAT,
        [
            1,
        ],
        weights_d,
    )

    graph_def = helper.make_graph(
        [node1, node2, node3, node4, node5, node6, node7],
        "complex_expression_model_with_log",
        [X_value_info],
        [output_value_info],
        [A_init, B_init, C_init, D_init],
    )

    model_def = helper.make_model(graph_def, producer_name="onnx-complex-expression")

    input_data = {"X": np.array([[6.0]], dtype=np.float32)}

    f = io.BytesIO()
    onnx.save(model_def, f)

    runtime_result = InferenceSession(f.getvalue()).run(None, input_data)

    ggml_dummy_model = GgmlRuntimeBackend.prepare(model_def)
    ggml_result = ggml_dummy_model.run(input_data)

    assert np.allclose(ggml_result, runtime_result)
