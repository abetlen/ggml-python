import io
import math
import typing

import numpy as np
import numpy.typing as npt

import pytest

onnx = pytest.importorskip("onnx")
from onnx import helper  # noqa: E402
from onnx.onnx_pb import TensorProto  # noqa: E402

import onnx.backend.test  # noqa: E402,F811

onnxruntime = pytest.importorskip("onnxruntime")
from onnxruntime import InferenceSession  # type: ignore  # noqa: E402

from ggml.contrib.onnx import (  # noqa: E402
    GgmlRuntimeBackend,
    onnx_operators,
)

import onnx.onnx_pb as onnx_pb  # noqa: E402


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


class OnnxModelBuilder:
    """Helper class to build ONNX models."""

    def __init__(self, name: str):
        self.name = name
        self.nodes: typing.List[onnx_pb.NodeProto] = []
        self.inputs: typing.List[onnx_pb.ValueInfoProto] = []
        self.outputs: typing.List[onnx_pb.ValueInfoProto] = []
        self.initializers: typing.List[onnx_pb.TensorProto] = []
        self.counter = 0  # Counter for unique names

    def add_input(
        self,
        name: str,
        elem_type: int,
        shape: typing.Optional[typing.List[typing.Union[str, int, None]]],
    ):
        self.inputs.append(helper.make_tensor_value_info(name, elem_type, shape))
        return self

    def add_node(
        self,
        op_type: str,
        inputs: typing.List[str],
        outputs: typing.List[str],
        name: typing.Optional[str] = None,
    ):
        if name is None:
            name = f"node{self.counter}"
            self.counter += 1
        self.nodes.append(helper.make_node(op_type, inputs, outputs, name=name))
        return self

    def add_initializer(
        self, name: str, data_type: int, dims: typing.Sequence[int], vals: typing.Any
    ):
        self.initializers.append(helper.make_tensor(name, data_type, dims, vals))
        return self

    def add_output(
        self,
        name: str,
        elem_type: int,
        shape: typing.Optional[typing.List[typing.Union[str, int, None]]],
    ):
        self.outputs.append(helper.make_tensor_value_info(name, elem_type, shape))
        return self

    def build_graph(self):
        return helper.make_graph(
            self.nodes, self.name, self.inputs, self.outputs, self.initializers
        )

    def build_model(self, name: typing.Optional[str] = None):
        return helper.make_model(self.build_graph(), producer_name=name or self.name)

    @staticmethod
    def model_bytes(model: onnx_pb.ModelProto) -> bytes:
        f = io.BytesIO()
        onnx.save(model, f)
        return f.getvalue()


def build_simple_graph():
    builder = OnnxModelBuilder("simple_expression_model")
    builder.add_input("X", TensorProto.FLOAT, [None, 1])
    builder.add_initializer(
        "A", TensorProto.FLOAT, [1], np.ones(1, dtype=float).astype(np.float32)
    )
    builder.add_initializer(
        "B", TensorProto.FLOAT, [1], np.ones(1, dtype=float).astype(np.float32)
    )
    builder.add_node("Mul", ["X", "X"], ["X_squared"])
    builder.add_node("Mul", ["X_squared", "A"], ["X_squared_times_a"])
    builder.add_node("Add", ["X_squared_times_a", "B"], ["Y"])
    builder.add_output("Y", TensorProto.FLOAT, [None, 1])
    return builder.build_model()


def build_2d_graph():
    builder = OnnxModelBuilder("simple_expression_model")
    builder.add_input("X", TensorProto.FLOAT, [None, 2])
    builder.add_initializer(
        "A", TensorProto.FLOAT, [2], np.ones(2, dtype=float).astype(np.float32)
    )
    builder.add_initializer(
        "B", TensorProto.FLOAT, [2], np.ones(2, dtype=float).astype(np.float32)
    )
    builder.add_node("Mul", ["X", "X"], ["X_squared"])
    builder.add_node("Mul", ["X_squared", "A"], ["X_squared_times_a"])
    builder.add_node("Add", ["X_squared_times_a", "B"], ["Y"])
    builder.add_output("Y", TensorProto.FLOAT, [None, 2])
    return builder.build_model()


def build_matmul_graph():
    builder = OnnxModelBuilder("simple_expression_model")
    builder.add_input("X", TensorProto.FLOAT, [None, 2])
    builder.add_initializer(
        "A", TensorProto.FLOAT, [2, 3], np.ones((2, 3), dtype=float).astype(np.float32)
    )
    builder.add_initializer(
        "B", TensorProto.FLOAT, [3, 4], np.ones((3, 4), dtype=float).astype(np.float32)
    )
    builder.add_node("MatMul", ["X", "A"], ["X_times_A"])
    builder.add_node("MatMul", ["X_times_A", "B"], ["Y"])
    builder.add_output("Y", TensorProto.FLOAT, [None, 4])
    return builder.build_model()


@pytest.mark.parametrize(
    "model, input_data",
    [
        # (build_simple_graph(), {"X": np.array([[6.0]], dtype=np.float32)}),
        # (build_2d_graph(), {"X": np.array([[6.0, 7.0]], dtype=np.float32)}),
        (build_matmul_graph(), {"X": np.array([[6.0, 7.0]], dtype=np.float32)}),
    ],
)
def test_compare_runtimes(
    model: onnx_pb.ModelProto, input_data: typing.Dict[str, npt.NDArray[typing.Any]]
):
    runtime_result = InferenceSession(OnnxModelBuilder.model_bytes(model)).run(
        None, input_data
    )  # type: ignore
    ggml_dummy_model = GgmlRuntimeBackend.prepare(model)
    ggml_result = ggml_dummy_model.run(input_data)
    assert np.array_equal(ggml_result, runtime_result)


@pytest.mark.parametrize("op_type", ["Abs", "Neg"])
def test_ggml_onnx_native_unary_integer_fallback(op_type: str):
    input_data = np.array([-(2**40), -7, 0, 6], dtype=np.int64)
    model_input = helper.make_tensor_value_info("X", TensorProto.INT64, [4])
    model_output = helper.make_tensor_value_info("Y", TensorProto.INT64, [4])
    node = helper.make_node(op_type, ["X"], ["Y"])
    graph = helper.make_graph(
        [node], f"{op_type.lower()}_int64_model", [model_input], [model_output]
    )
    model = helper.make_model(graph, producer_name=f"{op_type.lower()}-int64")

    input_dict = {"X": input_data}
    runtime_result = InferenceSession(OnnxModelBuilder.model_bytes(model)).run(
        None, input_dict
    )
    ggml_model = GgmlRuntimeBackend.prepare(model)
    ggml_result = ggml_model.run(input_dict)

    assert ggml_result[0].dtype == runtime_result[0].dtype
    np.testing.assert_array_equal(ggml_result[0], runtime_result[0])


@pytest.mark.parametrize(
    "case_name, node, output_type, output_shape",
    [
        (
            "tensor_int64",
            helper.make_node(
                "Constant",
                [],
                ["Y"],
                value=helper.make_tensor(
                    "const",
                    TensorProto.INT64,
                    [2],
                    np.array([2**40, -(2**40)], dtype=np.int64),
                ),
            ),
            TensorProto.INT64,
            [2],
        ),
        (
            "value_int",
            helper.make_node("Constant", [], ["Y"], value_int=2**40),
            TensorProto.INT64,
            [],
        ),
        (
            "value_ints",
            helper.make_node("Constant", [], ["Y"], value_ints=[2**40, -(2**40)]),
            TensorProto.INT64,
            [2],
        ),
        (
            "value_floats",
            helper.make_node("Constant", [], ["Y"], value_floats=[1.25, -2.5]),
            TensorProto.FLOAT,
            [2],
        ),
    ],
)
def test_ggml_onnx_constant_preserves_attribute_dtype_and_values(
    case_name: str,
    node: onnx_pb.NodeProto,
    output_type: int,
    output_shape: typing.List[int],
):
    model_output = helper.make_tensor_value_info("Y", output_type, output_shape)
    graph = helper.make_graph([node], case_name, [], [model_output])
    model = helper.make_model(graph, producer_name=case_name)

    runtime_result = InferenceSession(OnnxModelBuilder.model_bytes(model)).run(None, {})
    ggml_model = GgmlRuntimeBackend.prepare(model)
    ggml_result = ggml_model.run({})

    assert ggml_result[0].dtype == runtime_result[0].dtype
    np.testing.assert_array_equal(ggml_result[0], runtime_result[0])


@pytest.mark.parametrize(
    "case_name, node, output_type",
    [
        (
            "default_float32_zeros",
            helper.make_node("ConstantOfShape", ["shape"], ["Y"]),
            TensorProto.FLOAT,
        ),
        (
            "int64_fill",
            helper.make_node(
                "ConstantOfShape",
                ["shape"],
                ["Y"],
                value=helper.make_tensor(
                    "value",
                    TensorProto.INT64,
                    [1],
                    np.array([2**40], dtype=np.int64),
                ),
            ),
            TensorProto.INT64,
        ),
    ],
)
def test_ggml_onnx_constant_of_shape_default_and_typed_value(
    case_name: str,
    node: onnx_pb.NodeProto,
    output_type: int,
):
    model_input = helper.make_tensor_value_info("shape", TensorProto.INT64, [2])
    model_output = helper.make_tensor_value_info("Y", output_type, [2, 3])
    graph = helper.make_graph([node], case_name, [model_input], [model_output])
    model = helper.make_model(graph, producer_name=case_name)
    input_data = {"shape": np.array([2, 3], dtype=np.int64)}

    runtime_result = InferenceSession(OnnxModelBuilder.model_bytes(model)).run(
        None, input_data
    )
    ggml_model = GgmlRuntimeBackend.prepare(model)
    ggml_result = ggml_model.run(input_data)

    assert ggml_result[0].dtype == runtime_result[0].dtype
    np.testing.assert_array_equal(ggml_result[0], runtime_result[0])


def test_ggml_onnx_comparison_uses_logical_reshape_shape():
    model_input = helper.make_tensor_value_info("X", TensorProto.FLOAT, [6])
    model_output = helper.make_tensor_value_info("Y", TensorProto.BOOL, [2, 3])
    shape_initializer = helper.make_tensor(
        "shape", TensorProto.INT64, [2], np.array([2, 3], dtype=np.int64)
    )
    threshold_initializer = helper.make_tensor(
        "threshold",
        TensorProto.FLOAT,
        [2, 3],
        np.array([[1, 1, 3], [3, 5, 5]], dtype=np.float32),
    )
    nodes = [
        helper.make_node("Reshape", ["X", "shape"], ["reshaped"]),
        helper.make_node("Greater", ["reshaped", "threshold"], ["Y"]),
    ]
    graph = helper.make_graph(
        nodes,
        "reshape_greater_logical_shape",
        [model_input],
        [model_output],
        [shape_initializer, threshold_initializer],
    )
    model = helper.make_model(
        graph,
        producer_name="reshape-greater-logical-shape",
        opset_imports=[helper.make_opsetid("", 18)],
    )
    input_data = {"X": np.array([0, 2, 4, 3, 4, 6], dtype=np.float32)}

    runtime_result = InferenceSession(OnnxModelBuilder.model_bytes(model)).run(
        None, input_data
    )
    ggml_model = GgmlRuntimeBackend.prepare(model)
    ggml_result = ggml_model.run(input_data)

    assert ggml_result[0].shape == (2, 3)
    assert ggml_result[0].dtype == runtime_result[0].dtype
    np.testing.assert_array_equal(ggml_result[0], runtime_result[0])


def test_ggml_onnx_random_normal_like_sets_intermediate_metadata():
    model_input = helper.make_tensor_value_info("X", TensorProto.FLOAT, [2, 3])
    model_output = helper.make_tensor_value_info("Y", TensorProto.DOUBLE, [2, 3])
    zero_initializer = helper.make_tensor(
        "zero", TensorProto.DOUBLE, [2, 3], np.zeros((2, 3), dtype=np.float64)
    )
    nodes = [
        helper.make_node(
            "RandomNormalLike",
            ["X"],
            ["random"],
            dtype=TensorProto.DOUBLE,
            mean=2.5,
            scale=0.0,
            seed=7.0,
        ),
        helper.make_node("Add", ["random", "zero"], ["Y"]),
    ]
    graph = helper.make_graph(
        nodes,
        "random_normal_like_intermediate_metadata",
        [model_input],
        [model_output],
        [zero_initializer],
    )
    model = helper.make_model(
        graph,
        producer_name="random-normal-like-intermediate-metadata",
        opset_imports=[helper.make_opsetid("", 18)],
    )
    input_data = {"X": np.zeros((2, 3), dtype=np.float32)}

    ggml_model = GgmlRuntimeBackend.prepare(model)
    ggml_result = ggml_model.run(input_data)

    assert ggml_result[0].shape == (2, 3)
    assert ggml_result[0].dtype == np.dtype(np.float64)
    np.testing.assert_array_equal(
        ggml_result[0], np.full((2, 3), 2.5, dtype=np.float64)
    )


def test_ggml_onnx_rejects_mismatched_input_dtype():
    model_input = helper.make_tensor_value_info("X", TensorProto.FLOAT, [2])
    model_output = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [2])
    node = helper.make_node("Identity", ["X"], ["Y"])
    graph = helper.make_graph(
        [node], "reject_mismatched_input_dtype", [model_input], [model_output]
    )
    model = helper.make_model(
        graph,
        producer_name="reject-mismatched-input-dtype",
        opset_imports=[helper.make_opsetid("", 18)],
    )
    ggml_model = GgmlRuntimeBackend.prepare(model)

    with pytest.raises(ValueError, match="Unexpected input data type"):
        ggml_model.run({"X": np.array([1.0, 2.0], dtype=np.float64)})


def test_ggml_onnx_lrn_uses_full_channel_dimension():
    model_input = helper.make_tensor_value_info("X", TensorProto.FLOAT, [1, 7, 2, 2])
    model_output = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [1, 7, 2, 2])
    node = helper.make_node(
        "LRN",
        ["X"],
        ["Y"],
        alpha=0.0002,
        beta=0.75,
        bias=1.0,
        size=5,
    )
    graph = helper.make_graph(
        [node], "lrn_full_channel_dimension", [model_input], [model_output]
    )
    model = helper.make_model(
        graph,
        producer_name="lrn-full-channel-dimension",
        opset_imports=[helper.make_opsetid("", 18)],
    )
    input_data = {"X": (np.arange(1, 29, dtype=np.float32).reshape(1, 7, 2, 2) / 10)}

    runtime_result = InferenceSession(OnnxModelBuilder.model_bytes(model)).run(
        None, input_data
    )
    ggml_model = GgmlRuntimeBackend.prepare(model)
    ggml_result = ggml_model.run(input_data)

    assert ggml_result[0].shape == runtime_result[0].shape
    assert ggml_result[0].dtype == runtime_result[0].dtype
    np.testing.assert_allclose(ggml_result[0], runtime_result[0], rtol=1e-6, atol=1e-6)


@pytest.mark.parametrize(
    "case_name, node, inputs, outputs, input_data",
    [
        (
            "where_uint8_logical_dtype",
            helper.make_node("Where", ["condition", "X", "Y"], ["Z"]),
            [
                helper.make_tensor_value_info("condition", TensorProto.BOOL, [2, 3]),
                helper.make_tensor_value_info("X", TensorProto.UINT8, [2, 3]),
                helper.make_tensor_value_info("Y", TensorProto.UINT8, [2, 3]),
            ],
            [helper.make_tensor_value_info("Z", TensorProto.UINT8, [2, 3])],
            {
                "condition": np.array(
                    [[True, False, True], [False, True, False]], dtype=np.bool_
                ),
                "X": np.array([[1, 2, 3], [4, 5, 6]], dtype=np.uint8),
                "Y": np.array([[10, 20, 30], [40, 50, 60]], dtype=np.uint8),
            },
        ),
        (
            "compress_uint8_logical_dtype",
            helper.make_node("Compress", ["X", "condition"], ["Z"], axis=1),
            [
                helper.make_tensor_value_info("X", TensorProto.UINT8, [2, 3]),
                helper.make_tensor_value_info("condition", TensorProto.BOOL, [3]),
            ],
            [helper.make_tensor_value_info("Z", TensorProto.UINT8, [2, 2])],
            {
                "X": np.array([[1, 2, 3], [4, 5, 6]], dtype=np.uint8),
                "condition": np.array([True, False, True], dtype=np.bool_),
            },
        ),
        (
            "gathernd_uint8_logical_dtype",
            helper.make_node("GatherND", ["X", "indices"], ["Z"]),
            [
                helper.make_tensor_value_info("X", TensorProto.UINT8, [2, 2]),
                helper.make_tensor_value_info("indices", TensorProto.INT64, [2, 1]),
            ],
            [helper.make_tensor_value_info("Z", TensorProto.UINT8, [2, 2])],
            {
                "X": np.array([[1, 2], [3, 4]], dtype=np.uint8),
                "indices": np.array([[0], [1]], dtype=np.int64),
            },
        ),
        (
            "center_crop_pad_uint8_logical_dtype",
            helper.make_node("CenterCropPad", ["X", "shape"], ["Z"]),
            [
                helper.make_tensor_value_info("X", TensorProto.UINT8, [2, 3]),
                helper.make_tensor_value_info("shape", TensorProto.INT64, [2]),
            ],
            [helper.make_tensor_value_info("Z", TensorProto.UINT8, [4, 2])],
            {
                "X": np.array([[1, 2, 3], [4, 5, 6]], dtype=np.uint8),
                "shape": np.array([4, 2], dtype=np.int64),
            },
        ),
    ],
)
def test_ggml_onnx_eager_fallback_preserves_uint8_logical_dtype(
    case_name: str,
    node: onnx_pb.NodeProto,
    inputs: typing.List[onnx_pb.ValueInfoProto],
    outputs: typing.List[onnx_pb.ValueInfoProto],
    input_data: typing.Dict[str, npt.NDArray[typing.Any]],
):
    graph = helper.make_graph([node], case_name, inputs, outputs)
    model = helper.make_model(
        graph,
        producer_name=case_name,
        opset_imports=[helper.make_opsetid("", 18)],
    )

    runtime_result = InferenceSession(OnnxModelBuilder.model_bytes(model)).run(
        None, input_data
    )
    ggml_model = GgmlRuntimeBackend.prepare(model)
    ggml_result = ggml_model.run(input_data)

    assert ggml_result[0].shape == runtime_result[0].shape
    assert ggml_result[0].dtype == runtime_result[0].dtype
    np.testing.assert_array_equal(ggml_result[0], runtime_result[0])


def test_ggml_onnx_run_cleans_resources_on_exception(monkeypatch: pytest.MonkeyPatch):
    import ggml.contrib.onnx as ggml_onnx

    calls = []
    original_ggml_free = ggml_onnx.ggml.ggml_free
    original_buffer_free = ggml_onnx.ggml.ggml_backend_buffer_free
    original_free_backend_buffers = (
        ggml_onnx.GgmlOnnxExecutionContext.free_backend_buffers
    )

    def tracked_ggml_free(ctx):
        calls.append("ggml_free")
        return original_ggml_free(ctx)

    def tracked_buffer_free(buffer):
        calls.append("buffer_free")
        return original_buffer_free(buffer)

    def tracked_free_backend_buffers(ctx):
        calls.append("ctx_buffers")
        return original_free_backend_buffers(ctx)

    monkeypatch.setattr(ggml_onnx.ggml, "ggml_free", tracked_ggml_free)
    monkeypatch.setattr(ggml_onnx.ggml, "ggml_backend_buffer_free", tracked_buffer_free)
    monkeypatch.setattr(
        ggml_onnx.GgmlOnnxExecutionContext,
        "free_backend_buffers",
        tracked_free_backend_buffers,
    )

    model_input = helper.make_tensor_value_info("X", TensorProto.FLOAT, [2])
    model_output = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [2])
    node = helper.make_node("UnsupportedTestOp", ["X"], ["Y"])
    graph = helper.make_graph(
        [node], "cleanup_on_exception", [model_input], [model_output]
    )
    model = helper.make_model(
        graph,
        producer_name="cleanup-on-exception",
        opset_imports=[helper.make_opsetid("", 18)],
    )
    ggml_model = GgmlRuntimeBackend.prepare(model)

    with pytest.raises(NotImplementedError, match='Operator "UnsupportedTestOp"'):
        ggml_model.run({"X": np.array([1.0, 2.0], dtype=np.float32)})

    assert "ctx_buffers" in calls
    assert "buffer_free" in calls
    assert calls.count("ggml_free") >= 2


def test_ggml_onnx_is_opset_supported_reports_unsupported_ops():
    model_input = helper.make_tensor_value_info("X", TensorProto.FLOAT, [2])
    model_output = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [2])
    nodes = [
        helper.make_node("UnsupportedTestOp", ["X"], ["hidden"]),
        helper.make_node("AnotherUnsupportedTestOp", ["hidden"], ["Y"]),
        helper.make_node("UnsupportedTestOp", ["Y"], ["Y2"]),
    ]
    graph = helper.make_graph(
        nodes, "unsupported_op_reporting", [model_input], [model_output]
    )
    model = helper.make_model(
        graph,
        producer_name="unsupported-op-reporting",
        opset_imports=[helper.make_opsetid("", 18)],
    )

    plan = GgmlRuntimeBackend.analyze(model)
    supported, message = GgmlRuntimeBackend.is_opset_supported(model)

    assert not supported
    assert {node.operator_class for node in plan.unsupported_nodes} == {"unsupported"}
    assert plan.coverage_report().operator_class_count("unsupported") == 3
    assert message == (
        "Unsupported operators: AnotherUnsupportedTestOp, UnsupportedTestOp"
    )


def test_ggml_onnx_registered_ops_have_operator_objects():
    assert onnx_operators.operators
    assert {
        op_name
        for op_name, spec in onnx_operators.operators.items()
        if spec.op_type != op_name
    } == set()


def test_ggml_onnx_runtime_dispatches_registered_operator_strategy(monkeypatch):
    model_input = helper.make_tensor_value_info("X", TensorProto.FLOAT, [2])
    model_output = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [2])
    node = helper.make_node("Relu", ["X"], ["Y"])
    graph = helper.make_graph(
        [node], "operator_strategy_runtime_dispatch", [model_input], [model_output]
    )
    model = helper.make_model(
        graph,
        producer_name="operator-strategy-runtime-dispatch",
        opset_imports=[helper.make_opsetid("", 18)],
    )
    operator = onnx_operators.operators["Relu"]
    calls = []

    assert operator is not None
    original_lower = operator.lower

    def lower(ctx, node):
        calls.append(node.op_type)
        return original_lower(ctx, node)

    monkeypatch.setattr(operator, "lower", lower)

    ggml_model = GgmlRuntimeBackend.prepare(model, fallback_policy="strict")
    result = ggml_model.run({"X": np.array([-1.0, 2.0], dtype=np.float32)})

    assert calls == ["Relu"]
    np.testing.assert_array_equal(result[0], np.array([0.0, 2.0], dtype=np.float32))


def test_ggml_onnx_rejects_non_cpu_device():
    model_input = helper.make_tensor_value_info("X", TensorProto.FLOAT, [2])
    model_output = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [2])
    node = helper.make_node("Identity", ["X"], ["Y"])
    graph = helper.make_graph(
        [node], "reject_non_cpu_device", [model_input], [model_output]
    )
    model = helper.make_model(
        graph,
        producer_name="reject-non-cpu-device",
        opset_imports=[helper.make_opsetid("", 18)],
    )

    assert GgmlRuntimeBackend.supports_device("CPU")
    assert not GgmlRuntimeBackend.supports_device("CUDA")

    with pytest.raises(ValueError, match="supports CPU only"):
        GgmlRuntimeBackend.prepare(model, device="CUDA")


def test_ggml_onnx_execution_plan_reports_numpy_fallbacks():
    model_input = helper.make_tensor_value_info("X", TensorProto.FLOAT, [2])
    model_output = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [2])
    node = helper.make_node("Celu", ["X"], ["Y"])
    graph = helper.make_graph(
        [node], "execution_plan_numpy_fallback", [model_input], [model_output]
    )
    model = helper.make_model(
        graph,
        producer_name="execution-plan-numpy-fallback",
        opset_imports=[helper.make_opsetid("", 18)],
    )

    plan = GgmlRuntimeBackend.analyze(model)

    assert plan.is_supported
    assert plan.requires_numpy_fallbacks
    assert len(plan.fallback_nodes) == 1
    assert plan.fallback_nodes[0].op_type == "Celu"
    assert plan.fallback_nodes[0].execution == "numpy_runtime"
    assert plan.fallback_nodes[0].operator_class == "numpy_runtime"
    assert plan.fallback_nodes[0].capability == "numpy_runtime"
    assert plan.fallback_nodes[0].reason == "Operator uses NumPy runtime fallback"


def test_ggml_onnx_execution_plan_reports_operator_coverage():
    model_input_x = helper.make_tensor_value_info("X", TensorProto.FLOAT, [2, 2])
    model_input_b = helper.make_tensor_value_info("B", TensorProto.FLOAT, [1, 4])
    reshape_output = helper.make_tensor_value_info("R", TensorProto.FLOAT, [1, 4])
    add_output = helper.make_tensor_value_info("A", TensorProto.FLOAT, [1, 4])
    model_output = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [1, 4])
    shape = helper.make_tensor("shape", TensorProto.INT64, [2], [1, 4])
    graph = helper.make_graph(
        [
            helper.make_node("Reshape", ["X", "shape"], ["R"]),
            helper.make_node("Add", ["R", "B"], ["A"]),
            helper.make_node("Celu", ["A"], ["Y"], alpha=1.0),
        ],
        "execution_plan_operator_coverage",
        [model_input_x, model_input_b],
        [model_output],
        [shape],
        value_info=[reshape_output, add_output],
    )
    model = helper.make_model(
        graph,
        producer_name="execution-plan-operator-coverage",
        opset_imports=[helper.make_opsetid("", 18)],
    )

    plan = GgmlRuntimeBackend.analyze(model)
    report = plan.coverage_report()

    assert report.total_nodes == 3
    assert report.operator_class_count("native_view") == 1
    assert report.operator_class_count("native") == 1
    assert report.operator_class_count("numpy_runtime") == 1
    assert report.known_output_bytes_by_class["native_view"] == 16
    assert report.known_output_bytes_by_class["native"] == 16
    assert report.known_output_bytes_by_class["numpy_runtime"] == 16
    assert report.known_output_bytes == 48
    assert report.known_fallback_output_bytes == 16
    assert report.native_blocking_ops == ("Celu",)
    assert report.native_blocking_summary() == (
        "Celu#2: Operator uses NumPy runtime fallback",
    )
    assert [
        (summary[0], summary[1], summary[4])
        for summary in report.blocking_operator_summaries
    ] == [("Celu", "numpy_runtime", 1)]
    assert report.summary() == (
        "66.7% native, 0.0% decomposed, 33.3% fallback, 0.0% unsupported"
    )


def test_ggml_onnx_strict_policy_rejects_numpy_fallbacks():
    model_input = helper.make_tensor_value_info("X", TensorProto.FLOAT, [2])
    model_output = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [2])
    node = helper.make_node("Celu", ["X"], ["Y"])
    graph = helper.make_graph(
        [node], "strict_policy_rejects_numpy_fallback", [model_input], [model_output]
    )
    model = helper.make_model(
        graph,
        producer_name="strict-policy-rejects-numpy-fallback",
        opset_imports=[helper.make_opsetid("", 18)],
    )

    plan = GgmlRuntimeBackend.analyze(model, fallback_policy="strict")

    assert not plan.is_supported
    assert plan.blocked_nodes[0].op_type == "Celu"
    with pytest.raises(NotImplementedError, match='fallback_policy="strict"'):
        GgmlRuntimeBackend.prepare(model, fallback_policy="strict")


def test_ggml_onnx_strict_policy_runs_native_plan():
    model_input = helper.make_tensor_value_info("X", TensorProto.FLOAT, [2])
    model_output = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [2])
    node = helper.make_node("Identity", ["X"], ["Y"])
    graph = helper.make_graph(
        [node], "strict_policy_native_identity", [model_input], [model_output]
    )
    model = helper.make_model(
        graph,
        producer_name="strict-policy-native-identity",
        opset_imports=[helper.make_opsetid("", 18)],
    )
    input_data = {"X": np.array([1.0, 2.0], dtype=np.float32)}

    ggml_model = GgmlRuntimeBackend.prepare(model, fallback_policy="strict")
    ggml_result = ggml_model.run(input_data)

    assert ggml_model.execution_plan.fallback_policy == "strict"
    assert not ggml_model.fallback_nodes
    assert ggml_model.native_nodes[0].op_type == "Identity"
    assert ggml_model.native_nodes[0].operator_class == "native"
    assert ggml_model.native_nodes[0].reason == "Operator lowers to native ggml"
    np.testing.assert_array_equal(ggml_result[0], input_data["X"])


@pytest.mark.parametrize(
    "op_type, x, expected",
    [
        (
            "Abs",
            np.array([-1.0, 2.0], dtype=np.float32),
            np.array([1.0, 2.0], dtype=np.float32),
        ),
        (
            "Neg",
            np.array([-1.0, 2.0], dtype=np.float32),
            np.array([1.0, -2.0], dtype=np.float32),
        ),
        (
            "LeakyRelu",
            np.array([-2.0, 3.0], dtype=np.float32),
            np.array([-0.02, 3.0], dtype=np.float32),
        ),
        (
            "Log",
            np.array([1.0, np.e], dtype=np.float32),
            np.array([0.0, 1.0], dtype=np.float32),
        ),
        (
            "Elu",
            np.array([-1.0, 2.0], dtype=np.float32),
            np.where(
                np.array([-1.0, 2.0], dtype=np.float32) > 0,
                np.array([-1.0, 2.0], dtype=np.float32),
                np.exp(np.array([-1.0, 2.0], dtype=np.float32)) - 1,
            ),
        ),
        (
            "Relu",
            np.array([-1.0, 2.0], dtype=np.float32),
            np.array([0.0, 2.0], dtype=np.float32),
        ),
        (
            "Sign",
            np.array([-2.0, 0.0], dtype=np.float32),
            np.array([-1.0, 0.0], dtype=np.float32),
        ),
        (
            "Sqrt",
            np.array([1.0, 4.0], dtype=np.float32),
            np.array([1.0, 2.0], dtype=np.float32),
        ),
        (
            "Sigmoid",
            np.array([0.0, 1.0], dtype=np.float32),
            1.0 / (1.0 + np.exp(-np.array([0.0, 1.0], dtype=np.float32))),
        ),
        (
            "Tanh",
            np.array([0.0, 1.0], dtype=np.float32),
            np.tanh(np.array([0.0, 1.0], dtype=np.float32)),
        ),
    ],
)
def test_ggml_onnx_strict_policy_runs_native_float32_unary_ops(
    op_type: str,
    x: npt.NDArray[typing.Any],
    expected: npt.NDArray[typing.Any],
):
    model_input = helper.make_tensor_value_info("X", TensorProto.FLOAT, [2])
    model_output = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [2])
    node = helper.make_node(op_type, ["X"], ["Y"])
    graph = helper.make_graph(
        [node],
        f"strict_native_{op_type.lower()}",
        [model_input],
        [model_output],
    )
    model = helper.make_model(
        graph,
        producer_name=f"strict-native-{op_type.lower()}",
        opset_imports=[helper.make_opsetid("", 18)],
    )

    plan = GgmlRuntimeBackend.analyze(model, fallback_policy="strict")
    ggml_model = GgmlRuntimeBackend.prepare(model, fallback_policy="strict")
    ggml_result = ggml_model.run({"X": x})

    assert plan.is_supported
    assert plan.nodes[0].execution == "native"
    assert not ggml_model.fallback_nodes
    np.testing.assert_allclose(ggml_result[0], expected)


def test_ggml_onnx_strict_policy_rejects_integer_unary_fallback():
    model_input = helper.make_tensor_value_info("X", TensorProto.INT64, [2])
    model_output = helper.make_tensor_value_info("Y", TensorProto.INT64, [2])
    node = helper.make_node("Abs", ["X"], ["Y"])
    graph = helper.make_graph(
        [node], "strict_rejects_integer_abs", [model_input], [model_output]
    )
    model = helper.make_model(
        graph,
        producer_name="strict-rejects-integer-abs",
        opset_imports=[helper.make_opsetid("", 18)],
    )

    plan = GgmlRuntimeBackend.analyze(model, fallback_policy="strict")

    assert not plan.is_supported
    assert plan.blocked_nodes[0].op_type == "Abs"
    assert plan.blocked_nodes[0].execution == "numpy_runtime"
    with pytest.raises(NotImplementedError, match='fallback_policy="strict"'):
        GgmlRuntimeBackend.prepare(model, fallback_policy="strict")


def test_ggml_onnx_strict_policy_rejects_nondefault_elu_alpha_fallback():
    model_input = helper.make_tensor_value_info("X", TensorProto.FLOAT, [2])
    model_output = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [2])
    node = helper.make_node("Elu", ["X"], ["Y"], alpha=0.5)
    graph = helper.make_graph(
        [node], "strict_rejects_nondefault_elu", [model_input], [model_output]
    )
    model = helper.make_model(
        graph,
        producer_name="strict-rejects-nondefault-elu",
        opset_imports=[helper.make_opsetid("", 18)],
    )

    plan = GgmlRuntimeBackend.analyze(model, fallback_policy="strict")

    assert not plan.is_supported
    assert plan.blocked_nodes[0].op_type == "Elu"
    assert plan.blocked_nodes[0].execution == "numpy_runtime"
    with pytest.raises(NotImplementedError, match='fallback_policy="strict"'):
        GgmlRuntimeBackend.prepare(model, fallback_policy="strict")


@pytest.mark.parametrize(
    "op_type, y, expected",
    [
        (
            "Add",
            np.array([3.0, 4.0], dtype=np.float32),
            np.array([4.0, 6.0], dtype=np.float32),
        ),
        (
            "Sub",
            np.array([3.0, 4.0], dtype=np.float32),
            np.array([-2.0, -2.0], dtype=np.float32),
        ),
        (
            "Div",
            np.array([2.0, 4.0], dtype=np.float32),
            np.array([0.5, 0.5], dtype=np.float32),
        ),
        (
            "Mul",
            np.array([3.0, 4.0], dtype=np.float32),
            np.array([3.0, 8.0], dtype=np.float32),
        ),
    ],
)
def test_ggml_onnx_strict_policy_runs_native_float32_binary_ops(
    op_type: str,
    y: npt.NDArray[typing.Any],
    expected: npt.NDArray[typing.Any],
):
    model_input_x = helper.make_tensor_value_info("X", TensorProto.FLOAT, [2])
    model_input_y = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [2])
    model_output = helper.make_tensor_value_info("Z", TensorProto.FLOAT, [2])
    node = helper.make_node(op_type, ["X", "Y"], ["Z"])
    graph = helper.make_graph(
        [node],
        f"strict_native_{op_type.lower()}",
        [model_input_x, model_input_y],
        [model_output],
    )
    model = helper.make_model(
        graph,
        producer_name=f"strict-native-{op_type.lower()}",
        opset_imports=[helper.make_opsetid("", 18)],
    )
    input_data = {"X": np.array([1.0, 2.0], dtype=np.float32), "Y": y}

    plan = GgmlRuntimeBackend.analyze(model, fallback_policy="strict")
    ggml_model = GgmlRuntimeBackend.prepare(model, fallback_policy="strict")
    ggml_result = ggml_model.run(input_data)

    assert plan.is_supported
    assert plan.nodes[0].execution == "native"
    assert not ggml_model.fallback_nodes
    np.testing.assert_array_equal(ggml_result[0], expected)


def test_ggml_onnx_strict_policy_rejects_float32_binary_broadcast_fallback():
    model_input_x = helper.make_tensor_value_info("X", TensorProto.FLOAT, [2, 2])
    model_input_y = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [2])
    model_output = helper.make_tensor_value_info("Z", TensorProto.FLOAT, [2, 2])
    node = helper.make_node("Add", ["X", "Y"], ["Z"])
    graph = helper.make_graph(
        [node],
        "strict_rejects_broadcast_add",
        [model_input_x, model_input_y],
        [model_output],
    )
    model = helper.make_model(
        graph,
        producer_name="strict-rejects-broadcast-add",
        opset_imports=[helper.make_opsetid("", 18)],
    )

    plan = GgmlRuntimeBackend.analyze(model, fallback_policy="strict")

    assert not plan.is_supported
    assert plan.blocked_nodes[0].op_type == "Add"
    assert plan.blocked_nodes[0].execution == "numpy_runtime"
    with pytest.raises(NotImplementedError, match='fallback_policy="strict"'):
        GgmlRuntimeBackend.prepare(model, fallback_policy="strict")


def test_ggml_onnx_pipeline_shape_inference_enables_downstream_native_ops():
    model_input_x = helper.make_tensor_value_info("X", TensorProto.FLOAT, [2])
    model_input_y = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [2])
    model_input_z = helper.make_tensor_value_info("Z", TensorProto.FLOAT, [2])
    model_output = helper.make_tensor_value_info("O", TensorProto.FLOAT, [2])
    graph = helper.make_graph(
        [
            helper.make_node("Add", ["X", "Y"], ["A"]),
            helper.make_node("Add", ["A", "Z"], ["O"]),
        ],
        "shape_inference_enables_downstream_native_ops",
        [model_input_x, model_input_y, model_input_z],
        [model_output],
    )
    model = helper.make_model(
        graph,
        producer_name="shape-inference-enables-downstream-native-ops",
        opset_imports=[helper.make_opsetid("", 18)],
    )
    input_data = {
        "X": np.array([1.0, 2.0], dtype=np.float32),
        "Y": np.array([3.0, 4.0], dtype=np.float32),
        "Z": np.array([5.0, 6.0], dtype=np.float32),
    }

    pipeline = GgmlRuntimeBackend.build_pipeline(model, fallback_policy="strict")
    ggml_model = GgmlRuntimeBackend.prepare(model, fallback_policy="strict")
    ggml_result = ggml_model.run(input_data)

    assert "infer_shapes" in pipeline.optimization_passes
    assert pipeline.model_ir.tensor("A") is not None
    assert [node.execution for node in pipeline.execution_plan.nodes] == [
        "native",
        "native",
    ]
    assert not ggml_model.fallback_nodes
    np.testing.assert_array_equal(
        ggml_result[0], input_data["X"] + input_data["Y"] + input_data["Z"]
    )


def test_ggml_onnx_native_or_numpy_ops_stay_in_fallback_island():
    model_input_x = helper.make_tensor_value_info("X", TensorProto.FLOAT, [2])
    model_input_y = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [2])
    celu_output = helper.make_tensor_value_info("C", TensorProto.FLOAT, [2])
    model_output = helper.make_tensor_value_info("Z", TensorProto.FLOAT, [2])
    graph = helper.make_graph(
        [
            helper.make_node("Celu", ["X"], ["C"], alpha=1.0),
            helper.make_node("Add", ["C", "Y"], ["Z"]),
        ],
        "native_or_numpy_fallback_island",
        [model_input_x, model_input_y],
        [model_output],
        value_info=[celu_output],
    )
    model = helper.make_model(
        graph,
        producer_name="native-or-numpy-fallback-island",
        opset_imports=[helper.make_opsetid("", 18)],
    )
    input_data = {
        "X": np.array([-1.0, 2.0], dtype=np.float32),
        "Y": np.array([3.0, 4.0], dtype=np.float32),
    }

    plan = GgmlRuntimeBackend.analyze(model)
    ggml_model = GgmlRuntimeBackend.prepare(model)
    ggml_result = ggml_model.run(input_data)
    runtime_result = InferenceSession(OnnxModelBuilder.model_bytes(model)).run(
        None, input_data
    )

    assert [node.execution for node in plan.nodes] == [
        "numpy_runtime",
        "numpy_runtime",
    ]
    assert [node.operator_class for node in plan.nodes] == [
        "numpy_runtime",
        "numpy_runtime",
    ]
    assert plan.nodes[1].reason == "Operator depends on a NumPy fallback input"
    assert [node.op_type for node in ggml_model.fallback_nodes] == ["Celu", "Add"]
    assert len(plan.fallback_islands) == 1
    assert plan.fallback_islands[0].operator_types == ("Celu", "Add")
    assert plan.fallback_islands[0].input_names == ("X", "Y")
    assert plan.fallback_islands[0].output_names == ("C", "Z")
    assert [
        execution.operator_types
        for execution in ggml_model.last_numpy_fallback_island_executions
    ] == [("Celu", "Add")]
    np.testing.assert_allclose(ggml_result[0], runtime_result[0])


def test_ggml_onnx_fallback_island_dispatches_operator_numpy_evaluator(monkeypatch):
    model_input_x = helper.make_tensor_value_info("X", TensorProto.FLOAT, [2])
    model_input_y = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [2])
    celu_output = helper.make_tensor_value_info("C", TensorProto.FLOAT, [2])
    model_output = helper.make_tensor_value_info("Z", TensorProto.FLOAT, [2])
    graph = helper.make_graph(
        [
            helper.make_node("Celu", ["X"], ["C"], alpha=1.0),
            helper.make_node("Add", ["C", "Y"], ["Z"]),
        ],
        "fallback_island_operator_numpy_dispatch",
        [model_input_x, model_input_y],
        [model_output],
        value_info=[celu_output],
    )
    model = helper.make_model(
        graph,
        producer_name="fallback-island-operator-numpy-dispatch",
        opset_imports=[helper.make_opsetid("", 18)],
    )
    operator = onnx_operators.operators["Add"]
    calls = []

    assert operator is not None
    original_eval_numpy = operator.eval_numpy

    def eval_numpy(node, inputs):
        calls.append(node.op_type)
        return original_eval_numpy(node, inputs)

    monkeypatch.setattr(operator, "eval_numpy", eval_numpy)

    ggml_model = GgmlRuntimeBackend.prepare(model)
    result = ggml_model.run(
        {
            "X": np.array([-1.0, 2.0], dtype=np.float32),
            "Y": np.array([3.0, 4.0], dtype=np.float32),
        }
    )

    assert calls == ["Add"]
    assert [
        execution.operator_types
        for execution in ggml_model.last_numpy_fallback_island_executions
    ] == [("Celu", "Add")]
    np.testing.assert_allclose(
        result[0],
        np.array([2.3678794, 6.0], dtype=np.float32),
    )


def test_ggml_onnx_fallback_island_dispatches_transpose_operator_numpy_evaluator(
    monkeypatch,
):
    model_input = helper.make_tensor_value_info("X", TensorProto.FLOAT, [2, 3])
    transpose_output = helper.make_tensor_value_info("T", TensorProto.FLOAT, [3, 2])
    model_output = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [3, 2])
    graph = helper.make_graph(
        [
            helper.make_node("Transpose", ["X"], ["T"], perm=[1, 0]),
            helper.make_node("Celu", ["T"], ["Y"], alpha=1.0),
        ],
        "fallback_island_transpose_operator_numpy_dispatch",
        [model_input],
        [model_output],
        value_info=[transpose_output],
    )
    model = helper.make_model(
        graph,
        producer_name="fallback-island-transpose-operator-numpy-dispatch",
        opset_imports=[helper.make_opsetid("", 18)],
    )
    operator = onnx_operators.operators["Transpose"]
    calls = []
    original_eval_numpy = operator.eval_numpy

    def eval_numpy(node, inputs):
        calls.append(node.op_type)
        return original_eval_numpy(node, inputs)

    monkeypatch.setattr(operator, "eval_numpy", eval_numpy)

    input_data = {"X": np.arange(-3, 3, dtype=np.float32).reshape(2, 3)}
    ggml_model = GgmlRuntimeBackend.prepare(model)
    result = ggml_model.run(input_data)

    transposed = np.transpose(input_data["X"], axes=(1, 0))
    expected = np.maximum(0, transposed) + np.minimum(0, np.exp(transposed) - 1)
    assert calls == ["Transpose"]
    assert [
        execution.operator_types
        for execution in ggml_model.last_numpy_fallback_island_executions
    ] == [("Transpose", "Celu")]
    np.testing.assert_allclose(result[0], expected)


def test_ggml_onnx_fallback_island_dispatches_reduce_sum_operator_numpy_evaluator(
    monkeypatch,
):
    model_input = helper.make_tensor_value_info("X", TensorProto.FLOAT, [2, 2])
    celu_output = helper.make_tensor_value_info("C", TensorProto.FLOAT, [2, 2])
    model_output = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [1, 1])
    reduce_sum_node = helper.make_node(
        "ReduceSum",
        ["C"],
        ["Y"],
        keepdims=1,
        noop_with_empty_axes=0,
    )
    axes_attr = onnx_pb.AttributeProto()
    axes_attr.name = "axes"
    axes_attr.type = onnx_pb.AttributeProto.INTS
    reduce_sum_node.attribute.append(axes_attr)
    graph = helper.make_graph(
        [
            helper.make_node("Celu", ["X"], ["C"], alpha=1.0),
            reduce_sum_node,
        ],
        "fallback_island_reduce_sum_operator_numpy_dispatch",
        [model_input],
        [model_output],
        value_info=[celu_output],
    )
    model = helper.make_model(
        graph,
        producer_name="fallback-island-reduce-sum-operator-numpy-dispatch",
        opset_imports=[helper.make_opsetid("", 18)],
    )
    operator = onnx_operators.operators["ReduceSum"]
    calls = []
    original_eval_numpy = operator.eval_numpy

    def eval_numpy(node, inputs):
        calls.append(node.op_type)
        return original_eval_numpy(node, inputs)

    monkeypatch.setattr(operator, "eval_numpy", eval_numpy)

    input_data = {"X": np.array([[-1.0, 2.0], [3.0, -4.0]], dtype=np.float32)}
    ggml_model = GgmlRuntimeBackend.prepare(model)
    result = ggml_model.run(input_data)

    celu = np.maximum(0, input_data["X"]) + np.minimum(0, np.exp(input_data["X"]) - 1)
    expected = np.sum(celu, keepdims=True)
    assert calls == ["ReduceSum"]
    assert [
        execution.operator_types
        for execution in ggml_model.last_numpy_fallback_island_executions
    ] == [("Celu", "ReduceSum")]
    np.testing.assert_allclose(result[0], expected)


def test_ggml_onnx_native_or_numpy_fallback_island_spans_multiple_nodes():
    model_input_x = helper.make_tensor_value_info("X", TensorProto.FLOAT, [2])
    model_input_y = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [2])
    celu_output = helper.make_tensor_value_info("C", TensorProto.FLOAT, [2])
    add_output = helper.make_tensor_value_info("A", TensorProto.FLOAT, [2])
    model_output = helper.make_tensor_value_info("Z", TensorProto.FLOAT, [2])
    graph = helper.make_graph(
        [
            helper.make_node("Celu", ["X"], ["C"], alpha=1.0),
            helper.make_node("Add", ["C", "Y"], ["A"]),
            helper.make_node("Relu", ["A"], ["Z"]),
        ],
        "native_or_numpy_multi_node_fallback_island",
        [model_input_x, model_input_y],
        [model_output],
        value_info=[celu_output, add_output],
    )
    model = helper.make_model(
        graph,
        producer_name="native-or-numpy-multi-node-fallback-island",
        opset_imports=[helper.make_opsetid("", 18)],
    )
    input_data = {
        "X": np.array([-1.0, 2.0], dtype=np.float32),
        "Y": np.array([3.0, -4.0], dtype=np.float32),
    }

    plan = GgmlRuntimeBackend.analyze(model)
    ggml_model = GgmlRuntimeBackend.prepare(model)
    ggml_result = ggml_model.run(input_data)
    runtime_result = InferenceSession(OnnxModelBuilder.model_bytes(model)).run(
        None, input_data
    )

    assert [node.execution for node in plan.nodes] == [
        "numpy_runtime",
        "numpy_runtime",
        "numpy_runtime",
    ]
    assert [node.operator_class for node in plan.nodes] == [
        "numpy_runtime",
        "numpy_runtime",
        "numpy_runtime",
    ]
    assert [node.op_type for node in ggml_model.fallback_nodes] == [
        "Celu",
        "Add",
        "Relu",
    ]
    assert len(plan.fallback_islands) == 1
    assert plan.fallback_islands[0].operator_types == ("Celu", "Add", "Relu")
    assert plan.fallback_islands[0].input_names == ("X", "Y")
    assert plan.fallback_islands[0].output_names == ("C", "A", "Z")
    assert plan.coverage_report().fallback_islands == plan.fallback_islands
    assert [
        execution.operator_types
        for execution in ggml_model.last_numpy_fallback_island_executions
    ] == [("Celu", "Add", "Relu")]
    np.testing.assert_allclose(ggml_result[0], runtime_result[0])


def test_ggml_onnx_strict_policy_runs_native_float32_square_pow():
    model_input = helper.make_tensor_value_info("X", TensorProto.FLOAT, [2])
    model_output = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [2])
    exponent = helper.make_tensor("P", TensorProto.FLOAT, [], [2.0])
    node = helper.make_node("Pow", ["X", "P"], ["Y"])
    graph = helper.make_graph(
        [node],
        "strict_native_square_pow",
        [model_input],
        [model_output],
        [exponent],
    )
    model = helper.make_model(
        graph,
        producer_name="strict-native-square-pow",
        opset_imports=[helper.make_opsetid("", 18)],
    )
    input_data = {"X": np.array([-2.0, 3.0], dtype=np.float32)}

    plan = GgmlRuntimeBackend.analyze(model, fallback_policy="strict")
    ggml_model = GgmlRuntimeBackend.prepare(model, fallback_policy="strict")
    ggml_result = ggml_model.run(input_data)

    assert plan.is_supported
    assert plan.nodes[0].execution == "native"
    assert not ggml_model.fallback_nodes
    np.testing.assert_array_equal(
        ggml_result[0], np.array([4.0, 9.0], dtype=np.float32)
    )


@pytest.mark.parametrize(
    "op_type, inputs, expected",
    [
        (
            "Add",
            ["X", "S"],
            np.array([4.0, 5.0], dtype=np.float32),
        ),
        (
            "Add",
            ["S", "X"],
            np.array([4.0, 5.0], dtype=np.float32),
        ),
        (
            "Mul",
            ["X", "S"],
            np.array([3.0, 6.0], dtype=np.float32),
        ),
        (
            "Mul",
            ["S", "X"],
            np.array([3.0, 6.0], dtype=np.float32),
        ),
    ],
)
def test_ggml_onnx_strict_policy_runs_native_float32_scalar_binary_ops(
    op_type: str,
    inputs: typing.List[str],
    expected: npt.NDArray[typing.Any],
):
    model_input = helper.make_tensor_value_info("X", TensorProto.FLOAT, [2])
    model_output = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [2])
    scalar = helper.make_tensor("S", TensorProto.FLOAT, [], [3.0])
    node = helper.make_node(op_type, inputs, ["Y"])
    graph = helper.make_graph(
        [node],
        f"strict_native_scalar_{op_type.lower()}",
        [model_input],
        [model_output],
        [scalar],
    )
    model = helper.make_model(
        graph,
        producer_name=f"strict-native-scalar-{op_type.lower()}",
        opset_imports=[helper.make_opsetid("", 18)],
    )

    plan = GgmlRuntimeBackend.analyze(model, fallback_policy="strict")
    ggml_model = GgmlRuntimeBackend.prepare(model, fallback_policy="strict")
    ggml_result = ggml_model.run({"X": np.array([1.0, 2.0], dtype=np.float32)})

    assert plan.is_supported
    assert plan.nodes[0].execution == "native"
    assert not ggml_model.fallback_nodes
    np.testing.assert_array_equal(ggml_result[0], expected)


@pytest.mark.parametrize(
    ("op_type", "expected"),
    [
        ("Add", np.array([4.0, 5.0], dtype=np.float32)),
        ("Mul", np.array([3.0, 6.0], dtype=np.float32)),
    ],
)
def test_ggml_onnx_strict_policy_runs_native_dynamic_scalar_binary_ops(
    op_type: str,
    expected: npt.NDArray[typing.Any],
):
    model_input = helper.make_tensor_value_info("X", TensorProto.FLOAT, [2])
    scalar_input = helper.make_tensor_value_info("S", TensorProto.FLOAT, [1])
    model_output = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [2])
    node = helper.make_node(op_type, ["X", "S"], ["Y"])
    graph = helper.make_graph(
        [node],
        f"strict_native_dynamic_scalar_{op_type.lower()}",
        [model_input, scalar_input],
        [model_output],
    )
    model = helper.make_model(
        graph,
        producer_name=f"strict-native-dynamic-scalar-{op_type.lower()}",
        opset_imports=[helper.make_opsetid("", 18)],
    )

    plan = GgmlRuntimeBackend.analyze(model, fallback_policy="strict")
    ggml_model = GgmlRuntimeBackend.prepare(model, fallback_policy="strict")
    ggml_result = ggml_model.run(
        {
            "X": np.array([1.0, 2.0], dtype=np.float32),
            "S": np.array([3.0], dtype=np.float32),
        }
    )

    assert plan.is_supported
    assert plan.nodes[0].execution == "native"
    assert plan.nodes[0].operator_class == "native"
    assert plan.nodes[0].reason == "Operator lowers to native ggml"
    assert not ggml_model.fallback_nodes
    np.testing.assert_array_equal(ggml_result[0], expected)


def test_ggml_onnx_strict_policy_runs_decomposed_same_shape_sum():
    model_input_x = helper.make_tensor_value_info("X", TensorProto.FLOAT, [2])
    model_input_y = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [2])
    model_input_z = helper.make_tensor_value_info("Z", TensorProto.FLOAT, [2])
    model_output = helper.make_tensor_value_info("O", TensorProto.FLOAT, [2])
    node = helper.make_node("Sum", ["X", "Y", "Z"], ["O"])
    graph = helper.make_graph(
        [node],
        "strict_decomposed_same_shape_sum",
        [model_input_x, model_input_y, model_input_z],
        [model_output],
    )
    model = helper.make_model(
        graph,
        producer_name="strict-decomposed-same-shape-sum",
        opset_imports=[helper.make_opsetid("", 18)],
    )
    input_data = {
        "X": np.array([1.0, 2.0], dtype=np.float32),
        "Y": np.array([3.0, 4.0], dtype=np.float32),
        "Z": np.array([5.0, 6.0], dtype=np.float32),
    }

    plan = GgmlRuntimeBackend.analyze(model, fallback_policy="strict")
    ggml_model = GgmlRuntimeBackend.prepare(model, fallback_policy="strict")
    ggml_result = ggml_model.run(input_data)
    report = ggml_model.coverage_report

    assert plan.is_supported
    assert plan.nodes[0].execution == "decomposed"
    assert plan.nodes[0].operator_class == "decomposed"
    assert report.operator_class_count("decomposed") == 1
    assert not ggml_model.fallback_nodes
    np.testing.assert_array_equal(
        ggml_result[0], input_data["X"] + input_data["Y"] + input_data["Z"]
    )


def test_ggml_onnx_strict_policy_runs_decomposed_same_shape_mean():
    model_input_x = helper.make_tensor_value_info("X", TensorProto.FLOAT, [2])
    model_input_y = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [2])
    model_input_z = helper.make_tensor_value_info("Z", TensorProto.FLOAT, [2])
    model_output = helper.make_tensor_value_info("O", TensorProto.FLOAT, [2])
    node = helper.make_node("Mean", ["X", "Y", "Z"], ["O"])
    graph = helper.make_graph(
        [node],
        "strict_decomposed_same_shape_mean",
        [model_input_x, model_input_y, model_input_z],
        [model_output],
    )
    model = helper.make_model(
        graph,
        producer_name="strict-decomposed-same-shape-mean",
        opset_imports=[helper.make_opsetid("", 18)],
    )
    input_data = {
        "X": np.array([1.0, 2.0], dtype=np.float32),
        "Y": np.array([3.0, 4.0], dtype=np.float32),
        "Z": np.array([5.0, 6.0], dtype=np.float32),
    }

    plan = GgmlRuntimeBackend.analyze(model, fallback_policy="strict")
    ggml_model = GgmlRuntimeBackend.prepare(model, fallback_policy="strict")
    ggml_result = ggml_model.run(input_data)

    assert plan.is_supported
    assert plan.nodes[0].execution == "decomposed"
    assert plan.nodes[0].operator_class == "decomposed"
    assert not ggml_model.fallback_nodes
    np.testing.assert_array_equal(
        ggml_result[0],
        (input_data["X"] + input_data["Y"] + input_data["Z"]) / 3,
    )


def test_ggml_onnx_strict_policy_rejects_broadcast_sum_fallback():
    model_input_x = helper.make_tensor_value_info("X", TensorProto.FLOAT, [2])
    model_input_y = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [1])
    model_output = helper.make_tensor_value_info("O", TensorProto.FLOAT, [2])
    node = helper.make_node("Sum", ["X", "Y"], ["O"])
    graph = helper.make_graph(
        [node],
        "strict_rejects_broadcast_sum",
        [model_input_x, model_input_y],
        [model_output],
    )
    model = helper.make_model(
        graph,
        producer_name="strict-rejects-broadcast-sum",
        opset_imports=[helper.make_opsetid("", 18)],
    )
    input_data = {
        "X": np.array([1.0, 2.0], dtype=np.float32),
        "Y": np.array([3.0], dtype=np.float32),
    }

    plan = GgmlRuntimeBackend.analyze(model, fallback_policy="strict")
    compat_model = GgmlRuntimeBackend.prepare(model)
    compat_result = compat_model.run(input_data)

    assert not plan.is_supported
    assert plan.nodes[0].execution == "numpy_runtime"
    assert plan.nodes[0].operator_class == "numpy_runtime"
    np.testing.assert_array_equal(compat_result[0], input_data["X"] + input_data["Y"])


@pytest.mark.parametrize(
    "keepdims, output_shape",
    [
        (0, []),
        (1, [1, 1]),
    ],
)
def test_ggml_onnx_strict_policy_runs_native_full_reduce_sum(
    keepdims: int,
    output_shape: typing.List[int],
):
    model_input = helper.make_tensor_value_info("X", TensorProto.FLOAT, [2, 2])
    model_output = helper.make_tensor_value_info("Y", TensorProto.FLOAT, output_shape)
    node = helper.make_node("ReduceSum", ["X"], ["Y"], keepdims=keepdims)
    graph = helper.make_graph(
        [node],
        "strict_native_full_reduce_sum",
        [model_input],
        [model_output],
    )
    model = helper.make_model(
        graph,
        producer_name="strict-native-full-reduce-sum",
        opset_imports=[helper.make_opsetid("", 18)],
    )
    input_data = {"X": np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)}

    plan = GgmlRuntimeBackend.analyze(model, fallback_policy="strict")
    ggml_model = GgmlRuntimeBackend.prepare(model, fallback_policy="strict")
    ggml_result = ggml_model.run(input_data)

    assert plan.is_supported
    assert plan.nodes[0].execution == "native"
    assert not ggml_model.fallback_nodes
    np.testing.assert_array_equal(
        ggml_result[0], np.sum(input_data["X"], keepdims=bool(keepdims))
    )


def test_ggml_onnx_strict_policy_rejects_axis_reduce_sum_fallback():
    model_input = helper.make_tensor_value_info("X", TensorProto.FLOAT, [2, 2])
    model_output = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [1, 2])
    node = helper.make_node("ReduceSum", ["X"], ["Y"], axes=[0], keepdims=1)
    graph = helper.make_graph(
        [node],
        "strict_rejects_axis_reduce_sum",
        [model_input],
        [model_output],
    )
    model = helper.make_model(
        graph,
        producer_name="strict-rejects-axis-reduce-sum",
        opset_imports=[helper.make_opsetid("", 18)],
    )

    plan = GgmlRuntimeBackend.analyze(model, fallback_policy="strict")

    assert not plan.is_supported
    assert plan.blocked_nodes[0].op_type == "ReduceSum"
    assert plan.blocked_nodes[0].execution == "numpy_runtime"
    with pytest.raises(NotImplementedError, match='fallback_policy="strict"'):
        GgmlRuntimeBackend.prepare(model, fallback_policy="strict")


def test_ggml_onnx_analyze_resolves_native_or_numpy_to_concrete_execution():
    model_input = helper.make_tensor_value_info("X", TensorProto.FLOAT, [2, 2])
    model_output = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [1, 2])
    axes = helper.make_tensor("axes", TensorProto.INT64, [1], [0])
    node = helper.make_node("ReduceSum", ["X", "axes"], ["Y"], keepdims=1)
    graph = helper.make_graph(
        [node],
        "concrete_native_or_numpy_execution",
        [model_input],
        [model_output],
        [axes],
    )
    model = helper.make_model(
        graph,
        producer_name="concrete-native-or-numpy-execution",
        opset_imports=[helper.make_opsetid("", 18)],
    )

    plan = GgmlRuntimeBackend.analyze(model)

    assert plan.nodes[0].execution == "numpy_runtime"
    assert plan.nodes[0].operator_class == "numpy_runtime"


def test_ggml_onnx_strict_policy_runs_native_constant_shape_reshape():
    model_input = helper.make_tensor_value_info("X", TensorProto.FLOAT, [2, 2])
    model_output = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [1, 4])
    shape = helper.make_tensor("shape", TensorProto.INT64, [2], [1, 4])
    node = helper.make_node("Reshape", ["X", "shape"], ["Y"])
    graph = helper.make_graph(
        [node],
        "strict_native_constant_shape_reshape",
        [model_input],
        [model_output],
        [shape],
    )
    model = helper.make_model(
        graph,
        producer_name="strict-native-constant-shape-reshape",
        opset_imports=[helper.make_opsetid("", 18)],
    )
    input_data = {"X": np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)}

    plan = GgmlRuntimeBackend.analyze(model, fallback_policy="strict")
    ggml_model = GgmlRuntimeBackend.prepare(model, fallback_policy="strict")
    ggml_result = ggml_model.run(input_data)

    assert plan.is_supported
    assert plan.nodes[0].execution == "native"
    assert plan.nodes[0].operator_class == "native_view"
    assert plan.nodes[0].output_types[0].shape == (1, 4)
    assert plan.nodes[0].reason == "Operator lowers to a native ggml view"
    assert not ggml_model.fallback_nodes
    np.testing.assert_array_equal(ggml_result[0], input_data["X"].reshape(1, 4))


def test_ggml_onnx_strict_policy_runs_native_flatten_view():
    model_input = helper.make_tensor_value_info("X", TensorProto.FLOAT, [2, 3, 4])
    model_output = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [2, 12])
    node = helper.make_node("Flatten", ["X"], ["Y"], axis=1)
    graph = helper.make_graph(
        [node],
        "strict_native_flatten_view",
        [model_input],
        [model_output],
    )
    model = helper.make_model(
        graph,
        producer_name="strict-native-flatten-view",
        opset_imports=[helper.make_opsetid("", 18)],
    )
    input_data = {
        "X": np.arange(24, dtype=np.float32).reshape(2, 3, 4),
    }

    plan = GgmlRuntimeBackend.analyze(model, fallback_policy="strict")
    ggml_model = GgmlRuntimeBackend.prepare(model, fallback_policy="strict")
    ggml_result = ggml_model.run(input_data)

    assert plan.is_supported
    assert plan.nodes[0].execution == "native"
    assert plan.nodes[0].operator_class == "native_view"
    assert plan.nodes[0].output_types[0].shape == (2, 12)
    assert not ggml_model.fallback_nodes
    np.testing.assert_array_equal(ggml_result[0], input_data["X"].reshape(2, 12))


def test_ggml_onnx_strict_policy_runs_native_transpose_view():
    model_input = helper.make_tensor_value_info("X", TensorProto.FLOAT, [2, 3, 4])
    model_output = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [4, 2, 3])
    node = helper.make_node("Transpose", ["X"], ["Y"], perm=[2, 0, 1])
    graph = helper.make_graph(
        [node],
        "strict_native_transpose_view",
        [model_input],
        [model_output],
    )
    model = helper.make_model(
        graph,
        producer_name="strict-native-transpose-view",
        opset_imports=[helper.make_opsetid("", 18)],
    )
    input_data = {
        "X": np.arange(24, dtype=np.float32).reshape(2, 3, 4),
    }

    plan = GgmlRuntimeBackend.analyze(model, fallback_policy="strict")
    ggml_model = GgmlRuntimeBackend.prepare(model, fallback_policy="strict")
    ggml_result = ggml_model.run(input_data)

    assert plan.is_supported
    assert plan.nodes[0].execution == "native"
    assert plan.nodes[0].operator_class == "native_view"
    assert plan.nodes[0].output_types[0].shape == (4, 2, 3)
    assert not ggml_model.fallback_nodes
    np.testing.assert_array_equal(
        ggml_result[0], np.transpose(input_data["X"], axes=(2, 0, 1))
    )


def test_ggml_onnx_materializes_computed_native_transpose_view_output():
    model_input = helper.make_tensor_value_info("X", TensorProto.FLOAT, [2, 2])
    model_output = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [1, 4])
    multiplier = helper.make_tensor(
        "multiplier",
        TensorProto.FLOAT,
        [2, 2],
        np.ones((2, 2), dtype=np.float32).reshape(-1),
    )
    power = helper.make_tensor(
        "power",
        TensorProto.FLOAT,
        [],
        np.array(2.0, dtype=np.float32).reshape(-1),
    )
    graph = helper.make_graph(
        [
            helper.make_node("Abs", ["X"], ["T0"]),
            helper.make_node("Mul", ["T0", "multiplier"], ["T1"]),
            helper.make_node("Pow", ["T1", "power"], ["T2"]),
            helper.make_node("Flatten", ["T2"], ["T3"], axis=2),
            helper.make_node("Sigmoid", ["T3"], ["T4"]),
            helper.make_node("Transpose", ["T4"], ["Y"], perm=[1, 0]),
        ],
        "computed_native_transpose_view_output",
        [model_input],
        [model_output],
        [multiplier, power],
    )
    model = helper.make_model(
        graph,
        producer_name="computed-native-transpose-view-output",
        opset_imports=[helper.make_opsetid("", 18)],
    )
    input_data = {"X": np.arange(4, dtype=np.float32).reshape(2, 2)}

    plan = GgmlRuntimeBackend.analyze(model, fallback_policy="strict")
    ggml_model = GgmlRuntimeBackend.prepare(model, fallback_policy="strict")
    ggml_result = ggml_model.run(input_data)

    flattened = np.square(np.abs(input_data["X"])).reshape(4, 1)
    expected = 1.0 / (1.0 + np.exp(-flattened))
    assert plan.is_supported
    assert not ggml_model.fallback_nodes
    np.testing.assert_allclose(ggml_result[0], expected.T, rtol=1e-6, atol=1e-6)


def test_ggml_onnx_planner_falls_back_shape_view_after_layout_view():
    model_input = helper.make_tensor_value_info("X", TensorProto.FLOAT, [2, 3])
    transpose_output = helper.make_tensor_value_info("T", TensorProto.FLOAT, [3, 2])
    model_output = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [1, 6])
    shape = helper.make_tensor("shape", TensorProto.INT64, [2], [1, 6])
    graph = helper.make_graph(
        [
            helper.make_node("Transpose", ["X"], ["T"], perm=[1, 0]),
            helper.make_node("Reshape", ["T", "shape"], ["Y"]),
        ],
        "layout_view_to_shape_view_fallback",
        [model_input],
        [model_output],
        [shape],
        value_info=[transpose_output],
    )
    model = helper.make_model(
        graph,
        producer_name="layout-view-to-shape-view-fallback",
        opset_imports=[helper.make_opsetid("", 18)],
    )
    input_data = {"X": np.arange(6, dtype=np.float32).reshape(2, 3)}

    strict_plan = GgmlRuntimeBackend.analyze(model, fallback_policy="strict")
    compat_plan = GgmlRuntimeBackend.analyze(model)
    ggml_model = GgmlRuntimeBackend.prepare(model)
    ggml_result = ggml_model.run(input_data)

    assert not strict_plan.is_supported
    assert [node.execution for node in compat_plan.nodes] == [
        "numpy_runtime",
        "numpy_runtime",
    ]
    assert compat_plan.nodes[1].reason == (
        "Native shape view depends on a native layout view"
    )
    assert [
        execution.operator_types
        for execution in ggml_model.last_numpy_fallback_island_executions
    ] == [("Transpose", "Reshape")]
    np.testing.assert_array_equal(
        ggml_result[0], np.transpose(input_data["X"], axes=(1, 0)).reshape(1, 6)
    )


def test_ggml_onnx_planner_falls_back_native_consumer_after_layout_view():
    model_input = helper.make_tensor_value_info("X", TensorProto.FLOAT, [2, 3])
    transpose_output = helper.make_tensor_value_info("T", TensorProto.FLOAT, [3, 2])
    model_output = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [3, 2])
    divisor = helper.make_tensor(
        "divisor",
        TensorProto.FLOAT,
        [3, 2],
        np.full((3, 2), 2.0, dtype=np.float32).reshape(-1),
    )
    graph = helper.make_graph(
        [
            helper.make_node("Transpose", ["X"], ["T"], perm=[1, 0]),
            helper.make_node("Div", ["T", "divisor"], ["Y"]),
        ],
        "layout_view_to_native_consumer_fallback",
        [model_input],
        [model_output],
        [divisor],
        value_info=[transpose_output],
    )
    model = helper.make_model(
        graph,
        producer_name="layout-view-to-native-consumer-fallback",
        opset_imports=[helper.make_opsetid("", 18)],
    )
    input_data = {"X": np.arange(6, dtype=np.float32).reshape(2, 3)}

    strict_plan = GgmlRuntimeBackend.analyze(model, fallback_policy="strict")
    compat_plan = GgmlRuntimeBackend.analyze(model)
    ggml_model = GgmlRuntimeBackend.prepare(model)
    ggml_result = ggml_model.run(input_data)

    assert not strict_plan.is_supported
    assert [node.execution for node in compat_plan.nodes] == [
        "numpy_runtime",
        "numpy_runtime",
    ]
    assert compat_plan.nodes[1].reason == (
        "Native operator depends on a native layout view"
    )
    assert [
        execution.operator_types
        for execution in ggml_model.last_numpy_fallback_island_executions
    ] == [("Transpose", "Div")]
    np.testing.assert_array_equal(
        ggml_result[0], np.transpose(input_data["X"], axes=(1, 0)) / 2.0
    )


def test_ggml_onnx_strict_policy_rejects_rank5_transpose_fallback():
    model_input = helper.make_tensor_value_info("X", TensorProto.FLOAT, [1, 2, 3, 1, 2])
    model_output = helper.make_tensor_value_info(
        "Y", TensorProto.FLOAT, [1, 3, 2, 1, 2]
    )
    node = helper.make_node("Transpose", ["X"], ["Y"], perm=[0, 2, 1, 3, 4])
    graph = helper.make_graph(
        [node],
        "strict_rejects_rank5_transpose_fallback",
        [model_input],
        [model_output],
    )
    model = helper.make_model(
        graph,
        producer_name="strict-rejects-rank5-transpose-fallback",
        opset_imports=[helper.make_opsetid("", 18)],
    )
    input_data = {
        "X": np.arange(12, dtype=np.float32).reshape(1, 2, 3, 1, 2),
    }

    plan = GgmlRuntimeBackend.analyze(model, fallback_policy="strict")
    compat_plan = GgmlRuntimeBackend.analyze(model)
    compat_model = GgmlRuntimeBackend.prepare(model)
    compat_result = compat_model.run(input_data)

    assert not plan.is_supported
    assert plan.nodes[0].execution == "numpy_runtime"
    assert plan.nodes[0].operator_class == "numpy_runtime"
    assert (
        compat_plan.nodes[0].reason
        == "Transpose rank 5 exceeds ggml native view rank 4"
    )
    assert compat_model.fallback_nodes[0].op_type == "Transpose"
    assert [
        execution.operator_types
        for execution in compat_model.last_numpy_fallback_island_executions
    ] == [("Transpose",)]
    np.testing.assert_array_equal(
        compat_result[0], np.transpose(input_data["X"], axes=(0, 2, 1, 3, 4))
    )


def test_ggml_onnx_planner_isolates_layout_views_from_numpy_fallback_consumers():
    model_input = helper.make_tensor_value_info("X", TensorProto.FLOAT, [2, 3])
    transpose_output = helper.make_tensor_value_info("T", TensorProto.FLOAT, [3, 2])
    model_output = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [3, 2])
    graph = helper.make_graph(
        [
            helper.make_node("Transpose", ["X"], ["T"], perm=[1, 0]),
            helper.make_node("Celu", ["T"], ["Y"], alpha=1.0),
        ],
        "layout_view_to_numpy_fallback_consumer",
        [model_input],
        [model_output],
        value_info=[transpose_output],
    )
    model = helper.make_model(
        graph,
        producer_name="layout-view-to-numpy-fallback-consumer",
        opset_imports=[helper.make_opsetid("", 18)],
    )
    input_data = {
        "X": np.arange(-3, 3, dtype=np.float32).reshape(2, 3),
    }

    plan = GgmlRuntimeBackend.analyze(model)
    ggml_model = GgmlRuntimeBackend.prepare(model)
    ggml_result = ggml_model.run(input_data)

    assert [node.execution for node in plan.nodes] == [
        "numpy_runtime",
        "numpy_runtime",
    ]
    assert plan.nodes[0].reason == "Native layout view feeds a NumPy fallback consumer"
    assert [node.op_type for node in ggml_model.fallback_nodes] == ["Transpose", "Celu"]
    assert [
        execution.operator_types
        for execution in ggml_model.last_numpy_fallback_island_executions
    ] == [("Transpose", "Celu")]
    transposed = np.transpose(input_data["X"], axes=(1, 0))
    expected = np.maximum(0, transposed) + np.minimum(0, np.exp(transposed) - 1)
    np.testing.assert_allclose(
        ggml_result[0],
        expected,
    )


def test_ggml_onnx_planner_propagates_isolated_layout_view_fallbacks():
    model_input_x = helper.make_tensor_value_info("X", TensorProto.FLOAT, [2, 3])
    model_input_b = helper.make_tensor_value_info("B", TensorProto.FLOAT, [3, 2])
    transpose_output = helper.make_tensor_value_info("T", TensorProto.FLOAT, [3, 2])
    celu_output = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [3, 2])
    model_output = helper.make_tensor_value_info("Z", TensorProto.FLOAT, [3, 2])
    graph = helper.make_graph(
        [
            helper.make_node("Transpose", ["X"], ["T"], perm=[1, 0]),
            helper.make_node("Celu", ["T"], ["Y"], alpha=1.0),
            helper.make_node("Add", ["T", "B"], ["Z"]),
        ],
        "layout_view_fallback_propagates_to_native_consumer",
        [model_input_x, model_input_b],
        [model_output],
        value_info=[transpose_output, celu_output],
    )
    model = helper.make_model(
        graph,
        producer_name="layout-view-fallback-propagates-to-native-consumer",
        opset_imports=[helper.make_opsetid("", 18)],
    )
    input_data = {
        "X": np.arange(-3, 3, dtype=np.float32).reshape(2, 3),
        "B": np.ones((3, 2), dtype=np.float32),
    }

    plan = GgmlRuntimeBackend.analyze(model)
    ggml_model = GgmlRuntimeBackend.prepare(model)
    ggml_result = ggml_model.run(input_data)

    assert [node.execution for node in plan.nodes] == [
        "numpy_runtime",
        "numpy_runtime",
        "numpy_runtime",
    ]
    assert plan.nodes[0].reason == "Native layout view feeds a NumPy fallback consumer"
    assert plan.nodes[2].reason == "Operator depends on a NumPy fallback input"
    assert [
        execution.operator_types
        for execution in ggml_model.last_numpy_fallback_island_executions
    ] == [("Transpose", "Celu", "Add")]
    np.testing.assert_allclose(
        ggml_result[0],
        np.transpose(input_data["X"], axes=(1, 0)) + input_data["B"],
    )


def test_ggml_onnx_strict_policy_allows_native_shape_views_as_native_inputs():
    model_input_x = helper.make_tensor_value_info("X", TensorProto.FLOAT, [2, 2])
    model_input_b = helper.make_tensor_value_info("B", TensorProto.FLOAT, [1, 4])
    reshape_output = helper.make_tensor_value_info("R", TensorProto.FLOAT, [1, 4])
    model_output = helper.make_tensor_value_info("Z", TensorProto.FLOAT, [1, 4])
    shape = helper.make_tensor("shape", TensorProto.INT64, [2], [1, 4])
    graph = helper.make_graph(
        [
            helper.make_node("Reshape", ["X", "shape"], ["R"]),
            helper.make_node("Add", ["R", "B"], ["Z"]),
        ],
        "strict_native_shape_view_feeds_add",
        [model_input_x, model_input_b],
        [model_output],
        [shape],
        value_info=[reshape_output],
    )
    model = helper.make_model(
        graph,
        producer_name="strict-native-shape-view-feeds-add",
        opset_imports=[helper.make_opsetid("", 18)],
    )
    input_data = {
        "X": np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32),
        "B": np.array([[10.0, 20.0, 30.0, 40.0]], dtype=np.float32),
    }

    plan = GgmlRuntimeBackend.analyze(model, fallback_policy="strict")
    ggml_model = GgmlRuntimeBackend.prepare(model, fallback_policy="strict")
    ggml_result = ggml_model.run(input_data)

    assert plan.is_supported
    assert [node.execution for node in plan.nodes] == ["native", "native"]
    assert [node.operator_class for node in plan.nodes] == ["native_view", "native"]
    assert not ggml_model.fallback_nodes
    np.testing.assert_array_equal(
        ggml_result[0], input_data["X"].reshape(1, 4) + input_data["B"]
    )


def test_ggml_onnx_strict_policy_preserves_chained_native_view_metadata():
    model_input_x = helper.make_tensor_value_info("X", TensorProto.FLOAT, [1, 2, 1])
    model_input_b = helper.make_tensor_value_info("B", TensorProto.FLOAT, [1, 2])
    squeezed_output = helper.make_tensor_value_info("S", TensorProto.FLOAT, [2])
    unsqueezed_output = helper.make_tensor_value_info("U", TensorProto.FLOAT, [1, 2])
    flattened_output = helper.make_tensor_value_info("F", TensorProto.FLOAT, [1, 2])
    model_output = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [1, 2])
    squeeze_axes = helper.make_tensor("squeeze_axes", TensorProto.INT64, [2], [0, 2])
    unsqueeze_axes = helper.make_tensor("unsqueeze_axes", TensorProto.INT64, [1], [0])
    graph = helper.make_graph(
        [
            helper.make_node("Squeeze", ["X", "squeeze_axes"], ["S"]),
            helper.make_node("Unsqueeze", ["S", "unsqueeze_axes"], ["U"]),
            helper.make_node("Flatten", ["U"], ["F"], axis=1),
            helper.make_node("Add", ["F", "B"], ["Y"]),
        ],
        "strict_chained_native_view_metadata",
        [model_input_x, model_input_b],
        [model_output],
        [squeeze_axes, unsqueeze_axes],
        value_info=[squeezed_output, unsqueezed_output, flattened_output],
    )
    model = helper.make_model(
        graph,
        producer_name="strict-chained-native-view-metadata",
        opset_imports=[helper.make_opsetid("", 18)],
    )
    input_data = {
        "X": np.array([[[1.0], [2.0]]], dtype=np.float32),
        "B": np.array([[10.0, 20.0]], dtype=np.float32),
    }

    plan = GgmlRuntimeBackend.analyze(model, fallback_policy="strict")
    ggml_model = GgmlRuntimeBackend.prepare(model, fallback_policy="strict")
    ggml_result = ggml_model.run(input_data)

    assert plan.is_supported
    assert [node.operator_class for node in plan.nodes] == [
        "native_view",
        "native_view",
        "native_view",
        "native",
    ]
    assert [node.output_types[0].shape for node in plan.nodes] == [
        (2,),
        (1, 2),
        (1, 2),
        (1, 2),
    ]
    assert not ggml_model.fallback_nodes
    expected = np.squeeze(input_data["X"], axis=(0, 2))[None, :] + input_data["B"]
    np.testing.assert_array_equal(ggml_result[0], expected)


def test_ggml_onnx_strict_policy_rejects_dynamic_shape_reshape_fallback():
    model_input = helper.make_tensor_value_info("X", TensorProto.FLOAT, [2, 2])
    shape_input = helper.make_tensor_value_info("shape", TensorProto.INT64, [2])
    model_output = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [1, 4])
    node = helper.make_node("Reshape", ["X", "shape"], ["Y"])
    graph = helper.make_graph(
        [node],
        "strict_rejects_dynamic_shape_reshape",
        [model_input, shape_input],
        [model_output],
    )
    model = helper.make_model(
        graph,
        producer_name="strict-rejects-dynamic-shape-reshape",
        opset_imports=[helper.make_opsetid("", 18)],
    )

    plan = GgmlRuntimeBackend.analyze(model, fallback_policy="strict")
    compat_plan = GgmlRuntimeBackend.analyze(model)

    assert not plan.is_supported
    assert plan.blocked_nodes[0].op_type == "Reshape"
    assert plan.blocked_nodes[0].execution == "numpy_runtime"
    assert compat_plan.nodes[0].reason == "View parameters are dynamic"
    with pytest.raises(NotImplementedError, match='fallback_policy="strict"'):
        GgmlRuntimeBackend.prepare(model, fallback_policy="strict")


def test_ggml_onnx_strict_policy_runs_native_static_axes_squeeze():
    model_input = helper.make_tensor_value_info("X", TensorProto.FLOAT, [1, 2, 1])
    model_output = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [2])
    axes = helper.make_tensor("axes", TensorProto.INT64, [2], [0, 2])
    node = helper.make_node("Squeeze", ["X", "axes"], ["Y"])
    graph = helper.make_graph(
        [node],
        "strict_native_static_axes_squeeze",
        [model_input],
        [model_output],
        [axes],
    )
    model = helper.make_model(
        graph,
        producer_name="strict-native-static-axes-squeeze",
        opset_imports=[helper.make_opsetid("", 18)],
    )
    input_data = {"X": np.array([[[1.0], [2.0]]], dtype=np.float32)}

    plan = GgmlRuntimeBackend.analyze(model, fallback_policy="strict")
    ggml_model = GgmlRuntimeBackend.prepare(model, fallback_policy="strict")
    ggml_result = ggml_model.run(input_data)

    assert plan.is_supported
    assert plan.nodes[0].execution == "native"
    assert not ggml_model.fallback_nodes
    np.testing.assert_array_equal(ggml_result[0], np.squeeze(input_data["X"], (0, 2)))


def test_ggml_onnx_strict_policy_runs_native_no_axes_squeeze():
    model_input = helper.make_tensor_value_info("X", TensorProto.FLOAT, [1, 2, 1])
    model_output = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [2])
    node = helper.make_node("Squeeze", ["X"], ["Y"])
    graph = helper.make_graph(
        [node],
        "strict_native_no_axes_squeeze",
        [model_input],
        [model_output],
    )
    model = helper.make_model(
        graph,
        producer_name="strict-native-no-axes-squeeze",
        opset_imports=[helper.make_opsetid("", 18)],
    )
    input_data = {"X": np.array([[[1.0], [2.0]]], dtype=np.float32)}

    plan = GgmlRuntimeBackend.analyze(model, fallback_policy="strict")
    ggml_model = GgmlRuntimeBackend.prepare(model, fallback_policy="strict")
    ggml_result = ggml_model.run(input_data)

    assert plan.is_supported
    assert plan.nodes[0].execution == "native"
    assert plan.nodes[0].output_types[0].shape == (2,)
    assert not ggml_model.fallback_nodes
    np.testing.assert_array_equal(ggml_result[0], np.squeeze(input_data["X"]))


def test_ggml_onnx_strict_policy_runs_native_static_axes_unsqueeze():
    model_input = helper.make_tensor_value_info("X", TensorProto.FLOAT, [2])
    model_output = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [1, 2, 1])
    axes = helper.make_tensor("axes", TensorProto.INT64, [2], [0, 2])
    node = helper.make_node("Unsqueeze", ["X", "axes"], ["Y"])
    graph = helper.make_graph(
        [node],
        "strict_native_static_axes_unsqueeze",
        [model_input],
        [model_output],
        [axes],
    )
    model = helper.make_model(
        graph,
        producer_name="strict-native-static-axes-unsqueeze",
        opset_imports=[helper.make_opsetid("", 18)],
    )
    input_data = {"X": np.array([1.0, 2.0], dtype=np.float32)}

    plan = GgmlRuntimeBackend.analyze(model, fallback_policy="strict")
    ggml_model = GgmlRuntimeBackend.prepare(model, fallback_policy="strict")
    ggml_result = ggml_model.run(input_data)

    assert plan.is_supported
    assert plan.nodes[0].execution == "native"
    assert not ggml_model.fallback_nodes
    np.testing.assert_array_equal(
        ggml_result[0], np.expand_dims(np.expand_dims(input_data["X"], 0), 2)
    )


def test_ggml_onnx_strict_policy_rejects_dynamic_axes_squeeze_fallback():
    model_input = helper.make_tensor_value_info("X", TensorProto.FLOAT, [1, 2])
    axes_input = helper.make_tensor_value_info("axes", TensorProto.INT64, [1])
    model_output = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [2])
    node = helper.make_node("Squeeze", ["X", "axes"], ["Y"])
    graph = helper.make_graph(
        [node],
        "strict_rejects_dynamic_axes_squeeze",
        [model_input, axes_input],
        [model_output],
    )
    model = helper.make_model(
        graph,
        producer_name="strict-rejects-dynamic-axes-squeeze",
        opset_imports=[helper.make_opsetid("", 18)],
    )

    plan = GgmlRuntimeBackend.analyze(model, fallback_policy="strict")
    compat_plan = GgmlRuntimeBackend.analyze(model)

    assert not plan.is_supported
    assert plan.blocked_nodes[0].op_type == "Squeeze"
    assert plan.blocked_nodes[0].execution == "numpy_runtime"
    assert compat_plan.nodes[0].reason == "View parameters are dynamic"
    with pytest.raises(NotImplementedError, match='fallback_policy="strict"'):
        GgmlRuntimeBackend.prepare(model, fallback_policy="strict")


def test_ggml_onnx_strict_policy_rejects_general_pow_fallback():
    model_input = helper.make_tensor_value_info("X", TensorProto.FLOAT, [2])
    model_output = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [2])
    exponent = helper.make_tensor("P", TensorProto.FLOAT, [], [3.0])
    node = helper.make_node("Pow", ["X", "P"], ["Y"])
    graph = helper.make_graph(
        [node],
        "strict_rejects_general_pow",
        [model_input],
        [model_output],
        [exponent],
    )
    model = helper.make_model(
        graph,
        producer_name="strict-rejects-general-pow",
        opset_imports=[helper.make_opsetid("", 18)],
    )

    plan = GgmlRuntimeBackend.analyze(model, fallback_policy="strict")

    assert not plan.is_supported
    assert plan.blocked_nodes[0].op_type == "Pow"
    assert plan.blocked_nodes[0].execution == "numpy_runtime"
    with pytest.raises(NotImplementedError, match='fallback_policy="strict"'):
        GgmlRuntimeBackend.prepare(model, fallback_policy="strict")


def test_ggml_onnx_build_ir_tracks_tensor_and_node_metadata(
    monkeypatch: pytest.MonkeyPatch,
):
    def fail_to_array(*args, **kwargs):
        raise AssertionError("build_ir should not materialize initializer arrays")

    monkeypatch.setattr(onnx.numpy_helper, "to_array", fail_to_array)

    model_input = helper.make_tensor_value_info("X", TensorProto.FLOAT, ["batch", 3])
    model_output = helper.make_tensor_value_info("Y", TensorProto.FLOAT, ["batch", 2])
    weight = helper.make_tensor(
        "W",
        TensorProto.FLOAT,
        [3, 2],
        np.arange(6, dtype=np.float32).reshape(3, 2),
    )
    node = helper.make_node("Gemm", ["X", "W"], ["Y"], alpha=0.5, transB=0)
    graph = helper.make_graph(
        [node], "typed_model_ir", [model_input], [model_output], [weight]
    )
    model = helper.make_model(
        graph,
        producer_name="typed-model-ir",
        opset_imports=[helper.make_opsetid("", 18)],
    )

    model_ir = GgmlRuntimeBackend.build_ir(model)
    input_info = model_ir.tensor("X")
    weight_info = model_ir.tensor("W")

    assert model_ir.inputs == ("X",)
    assert model_ir.outputs == ("Y",)
    assert model_ir.initializers == ("W",)
    assert input_info is not None
    assert input_info.shape == ("batch", 3)
    assert np.dtype(input_info.dtype) == np.dtype(np.float32)
    assert not input_info.initializer
    assert weight_info is not None
    assert weight_info.shape == (3, 2)
    assert np.dtype(weight_info.dtype) == np.dtype(np.float32)
    assert weight_info.initializer
    assert weight_info.constant
    assert weight_info.scalar_value is None
    assert model_ir.nodes[0].op_type == "Gemm"
    assert model_ir.nodes[0].attributes == ("alpha", "transB")
    assert model_ir.nodes[0].attribute("alpha") == 0.5
    assert model_ir.nodes[0].attribute("transB") == 0


def test_ggml_onnx_build_ir_tracks_scalar_initializer_metadata():
    scalar = helper.make_tensor("P", TensorProto.FLOAT, [], [2.0])
    model_output = helper.make_tensor_value_info("P", TensorProto.FLOAT, [])
    graph = helper.make_graph([], "scalar_initializer_ir", [], [model_output], [scalar])
    model = helper.make_model(
        graph,
        producer_name="scalar-initializer-ir",
        opset_imports=[helper.make_opsetid("", 18)],
    )

    model_ir = GgmlRuntimeBackend.build_ir(model)
    scalar_info = model_ir.tensor("P")

    assert scalar_info is not None
    assert scalar_info.shape == ()
    assert np.dtype(scalar_info.dtype) == np.dtype(np.float32)
    assert scalar_info.initializer
    assert scalar_info.constant
    assert scalar_info.scalar_value == 2.0


def test_ggml_onnx_build_ir_tracks_constant_node_metadata():
    node = helper.make_node("Constant", [], ["C"], value_ints=[1, 2, 3])
    model_output = helper.make_tensor_value_info("C", TensorProto.INT64, [3])
    graph = helper.make_graph([node], "constant_node_ir", [], [model_output])
    model = helper.make_model(
        graph,
        producer_name="constant-node-ir",
        opset_imports=[helper.make_opsetid("", 18)],
    )

    model_ir = GgmlRuntimeBackend.build_ir(model)
    constant_info = model_ir.tensor("C")

    assert constant_info is not None
    assert constant_info.shape == (3,)
    assert np.dtype(constant_info.dtype) == np.dtype(np.int64)
    assert constant_info.constant
    assert not constant_info.initializer


def test_ggml_onnx_build_ir_tracks_node_domain():
    model_input = helper.make_tensor_value_info("X", TensorProto.FLOAT, [2, 3])
    model_output = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [2, 3])
    node = helper.make_node("SiLU", ["X"], ["Y"], domain="com.ggml")
    graph = helper.make_graph([node], "domain_ir", [model_input], [model_output])
    model = helper.make_model(
        graph,
        producer_name="domain-ir",
        opset_imports=[helper.make_opsetid("", 18), helper.make_opsetid("com.ggml", 1)],
    )

    model_ir = GgmlRuntimeBackend.build_ir(model)

    assert model_ir.nodes[0].op_type == "SiLU"
    assert model_ir.nodes[0].domain == "com.ggml"


def test_ggml_onnx_build_pipeline_uses_optimized_typed_ir_and_plan():
    constant_node = helper.make_node("Constant", [], ["C"], value_ints=[1, 2, 3])
    identity_node = helper.make_node("Identity", ["C"], ["Y"])
    model_output = helper.make_tensor_value_info("Y", TensorProto.INT64, [3])
    graph = helper.make_graph(
        [constant_node, identity_node],
        "optimized_typed_ir_pipeline",
        [],
        [model_output],
    )
    model = helper.make_model(
        graph,
        producer_name="optimized-typed-ir-pipeline",
        opset_imports=[helper.make_opsetid("", 18)],
    )

    pipeline = GgmlRuntimeBackend.build_pipeline(model, fallback_policy="strict")
    constant_info = pipeline.model_ir.tensor("C")
    constant_type = pipeline.tensor_types["C"]
    node_output_types = pipeline.execution_plan.nodes[0].output_types

    assert pipeline.original_model is model
    assert pipeline.optimization_passes == ("fold_constant_nodes",)
    assert [node.op_type for node in pipeline.optimized_model.graph.node] == [
        "Identity"
    ]
    assert constant_info is not None
    assert constant_info.initializer
    assert constant_info.constant
    assert constant_type.shape == (3,)
    assert np.dtype(constant_type.dtype) == np.dtype(np.int64)
    assert constant_type.element_count == 3
    assert constant_type.nbytes == 24
    assert node_output_types[0].describe == "int64(3,)"
    assert pipeline.execution_plan.is_supported
    assert pipeline.execution_plan.nodes[0].operator_class == "native"
    assert pipeline.execution_plan.nodes[0].known_output_bytes == 24
    assert pipeline.coverage_report.summary() == (
        "100.0% native, 0.0% decomposed, 0.0% fallback, 0.0% unsupported"
    )


def test_ggml_onnx_fold_constant_nodes_moves_constants_to_initializers():
    constant_node = helper.make_node("Constant", [], ["C"], value_ints=[1, 2, 3])
    identity_node = helper.make_node("Identity", ["C"], ["Y"])
    model_output = helper.make_tensor_value_info("Y", TensorProto.INT64, [3])
    graph = helper.make_graph(
        [constant_node, identity_node], "fold_constant_nodes", [], [model_output]
    )
    model = helper.make_model(
        graph,
        producer_name="fold-constant-nodes",
        opset_imports=[helper.make_opsetid("", 18)],
    )

    folded = GgmlRuntimeBackend.fold_constant_nodes(model)

    assert [node.op_type for node in folded.graph.node] == ["Identity"]
    assert [initializer.name for initializer in folded.graph.initializer] == ["C"]
    np.testing.assert_array_equal(
        onnx.numpy_helper.to_array(folded.graph.initializer[0]),
        np.array([1, 2, 3], dtype=np.int64),
    )


def test_ggml_onnx_prepare_uses_folded_constant_model():
    constant_node = helper.make_node("Constant", [], ["Y"], value_ints=[1, 2, 3])
    model_output = helper.make_tensor_value_info("Y", TensorProto.INT64, [3])
    graph = helper.make_graph(
        [constant_node], "folded_constant_runtime", [], [model_output]
    )
    model = helper.make_model(
        graph,
        producer_name="folded-constant-runtime",
        opset_imports=[helper.make_opsetid("", 18)],
    )

    ggml_model = GgmlRuntimeBackend.prepare(model, fallback_policy="strict")
    ggml_result = ggml_model.run({})

    assert len(ggml_model.graph.node) == 0
    assert ggml_model.execution_plan.nodes == ()
    np.testing.assert_array_equal(ggml_result[0], np.array([1, 2, 3], dtype=np.int64))


@pytest.mark.parametrize(
    "opset,scale,bias,scale_mode",
    [
        (
            18,
            np.asarray([0.5, 1.5], dtype=np.float32),
            np.asarray([0.25, -0.5], dtype=np.float32),
            "group",
        ),
        (
            21,
            np.asarray([0.5, 0.75, 1.25, 1.5], dtype=np.float32),
            np.asarray([0.25, 0.5, -0.25, -0.5], dtype=np.float32),
            "channel",
        ),
    ],
)
def test_ggml_onnx_group_normalization_uses_opset_scale_semantics(
    opset: int,
    scale: npt.NDArray[np.float32],
    bias: npt.NDArray[np.float32],
    scale_mode: str,
):
    x = np.arange(8, dtype=np.float32).reshape(1, 4, 2)
    x_info = helper.make_tensor_value_info("X", TensorProto.FLOAT, [1, 4, 2])
    y_info = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [1, 4, 2])
    scale_tensor = onnx.numpy_helper.from_array(scale, name="scale")
    bias_tensor = onnx.numpy_helper.from_array(bias, name="bias")
    node = helper.make_node(
        "GroupNormalization",
        ["X", "scale", "bias"],
        ["Y"],
        num_groups=2,
        epsilon=1e-5,
    )
    graph = helper.make_graph(
        [node],
        f"group_normalization_opset_{opset}",
        [x_info],
        [y_info],
        [scale_tensor, bias_tensor],
    )
    model = helper.make_model(
        graph,
        producer_name=f"group-normalization-opset-{opset}",
        opset_imports=[helper.make_opsetid("", opset)],
    )

    grouped = x.reshape(1, 2, 2, 2)
    mean = np.mean(grouped, axis=(2, 3), keepdims=True)
    variance = np.var(grouped, axis=(2, 3), keepdims=True)
    normalized_grouped = (grouped - mean) / np.sqrt(variance + 1e-5)
    if scale_mode == "group":
        expected = (
            scale.reshape(1, 2, 1, 1) * normalized_grouped + bias.reshape(1, 2, 1, 1)
        ).reshape(1, 4, 2)
    else:
        expected = normalized_grouped.reshape(1, 4, 2)
        expected = scale.reshape(1, 4, 1) * expected + bias.reshape(1, 4, 1)

    ggml_model = GgmlRuntimeBackend.prepare(model)
    actual = ggml_model.run({"X": x})

    np.testing.assert_allclose(actual[0], expected, rtol=1e-5, atol=1e-5)
    assert [node.op_type for node in ggml_model.fallback_nodes] == [
        "GroupNormalization"
    ]


def run_onnxruntime_model(
    model: onnx.ModelProto,
    inputs: typing.Dict[str, npt.NDArray[typing.Any]],
):
    model.ir_version = min(model.ir_version, 10)
    return InferenceSession(model.SerializeToString()).run(None, inputs)


def run_onnx_operator_numpy_reference(
    node: onnx.NodeProto,
    inputs: typing.Sequence[npt.NDArray[typing.Any]],
):
    operator = onnx_operators.get(node.op_type, node.domain)
    assert operator is not None
    assert operator.has_numpy_evaluator
    return operator.eval_numpy(node, tuple(inputs))


@pytest.mark.parametrize(
    "op_type,opset,attrs,initializers",
    [
        ("Gelu", 20, {}, []),
        ("Gelu", 20, {"approximate": "tanh"}, []),
        ("LpNormalization", 22, {"axis": 1, "p": 2}, []),
        (
            "RMSNormalization",
            23,
            {"axis": 1},
            [
                helper.make_tensor(
                    "scale",
                    TensorProto.FLOAT,
                    [3],
                    np.asarray([0.5, 1.0, 1.5], dtype=np.float32),
                )
            ],
        ),
    ],
)
def test_ggml_onnx_standard_native_ggml_mappings_match_numpy_reference(
    op_type: str,
    opset: int,
    attrs: typing.Dict[str, typing.Any],
    initializers: typing.List[onnx.TensorProto],
):
    model_input = helper.make_tensor_value_info("X", TensorProto.FLOAT, [2, 3])
    model_output = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [2, 3])
    inputs = ["X", *(initializer.name for initializer in initializers)]
    node = helper.make_node(op_type, inputs, ["Y"], **attrs)
    graph = helper.make_graph(
        [node],
        f"standard_native_{op_type.lower()}",
        [model_input],
        [model_output],
        initializers,
    )
    model = helper.make_model(
        graph,
        producer_name=f"standard-native-{op_type.lower()}",
        opset_imports=[helper.make_opsetid("", opset)],
    )
    input_data = {
        "X": np.asarray([[-1.5, -0.5, 0.25], [0.75, 1.5, 2.25]], dtype=np.float32)
    }
    reference_inputs = [
        input_data["X"],
        *(onnx.numpy_helper.to_array(initializer) for initializer in initializers),
    ]

    expected = run_onnx_operator_numpy_reference(node, reference_inputs)
    ggml_model = GgmlRuntimeBackend.prepare(model, fallback_policy="strict")
    actual = ggml_model.run(input_data)

    atol = 2e-4 if op_type == "Gelu" and attrs.get("approximate") == "tanh" else 1e-5
    rtol = 2e-4 if op_type == "Gelu" and attrs.get("approximate") == "tanh" else 1e-5
    np.testing.assert_allclose(actual[0], expected[0], rtol=rtol, atol=atol)
    assert ggml_model.coverage_report.summary() == (
        "100.0% native, 0.0% decomposed, 0.0% fallback, 0.0% unsupported"
    )


def test_ggml_onnx_attention_matches_numpy_reference_with_numpy_fallback():
    q_info = helper.make_tensor_value_info("Q", TensorProto.FLOAT, [1, 2, 4])
    k_info = helper.make_tensor_value_info("K", TensorProto.FLOAT, [1, 2, 4])
    v_info = helper.make_tensor_value_info("V", TensorProto.FLOAT, [1, 2, 4])
    y_info = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [1, 2, 4])
    node = helper.make_node(
        "Attention",
        ["Q", "K", "V"],
        ["Y"],
        q_num_heads=2,
        kv_num_heads=2,
    )
    graph = helper.make_graph([node], "attention", [q_info, k_info, v_info], [y_info])
    model = helper.make_model(
        graph,
        producer_name="attention",
        opset_imports=[helper.make_opsetid("", 24)],
    )
    q = np.arange(8, dtype=np.float32).reshape(1, 2, 4) / 10.0
    inputs = {"Q": q, "K": q + 0.1, "V": q + 0.2}

    expected = run_onnx_operator_numpy_reference(
        node, [inputs["Q"], inputs["K"], inputs["V"]]
    )
    ggml_model = GgmlRuntimeBackend.prepare(model)
    actual = ggml_model.run(inputs)

    np.testing.assert_allclose(actual[0], expected[0], rtol=1e-5, atol=1e-5)
    assert [node.op_type for node in ggml_model.fallback_nodes] == ["Attention"]


def test_ggml_onnx_rotary_embedding_matches_numpy_reference_with_numpy_fallback():
    x_info = helper.make_tensor_value_info("X", TensorProto.FLOAT, [1, 2, 4])
    y_info = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [1, 2, 4])
    cos_values = np.asarray([[[0.8, 0.6], [0.5, 0.25]]], dtype=np.float32)
    sin_values = np.asarray([[[0.2, 0.4], [0.5, 0.75]]], dtype=np.float32)
    cos = helper.make_tensor("cos", TensorProto.FLOAT, [1, 2, 2], cos_values.ravel())
    sin = helper.make_tensor("sin", TensorProto.FLOAT, [1, 2, 2], sin_values.ravel())
    node = helper.make_node(
        "RotaryEmbedding",
        ["X", "cos", "sin"],
        ["Y"],
        num_heads=1,
    )
    graph = helper.make_graph(
        [node], "rotary_embedding", [x_info], [y_info], [cos, sin]
    )
    model = helper.make_model(
        graph,
        producer_name="rotary-embedding",
        opset_imports=[helper.make_opsetid("", 23)],
    )
    inputs = {"X": np.arange(8, dtype=np.float32).reshape(1, 2, 4)}

    expected = run_onnx_operator_numpy_reference(
        node, [inputs["X"], cos_values, sin_values]
    )
    ggml_model = GgmlRuntimeBackend.prepare(model)
    actual = ggml_model.run(inputs)

    np.testing.assert_allclose(actual[0], expected[0], rtol=1e-5, atol=1e-5)
    assert [node.op_type for node in ggml_model.fallback_nodes] == ["RotaryEmbedding"]


def make_ggml_extension_model(
    op_type: str,
    inputs: typing.Sequence[str],
    outputs: typing.Sequence[str],
    input_infos: typing.Sequence[onnx.ValueInfoProto],
    output_infos: typing.Sequence[onnx.ValueInfoProto],
    attrs: typing.Optional[typing.Dict[str, typing.Any]] = None,
):
    node = helper.make_node(
        op_type,
        list(inputs),
        list(outputs),
        domain="com.ggml",
        **(attrs or {}),
    )
    graph = helper.make_graph(
        [node], f"ggml_{op_type.lower()}", list(input_infos), list(output_infos)
    )
    model = helper.make_model(
        graph,
        producer_name=f"ggml-{op_type.lower()}",
        opset_imports=[helper.make_opsetid("", 18), helper.make_opsetid("com.ggml", 1)],
    )
    return model


@pytest.mark.parametrize("op_type", ["SiLU", "QuickGelu"])
def test_ggml_onnx_unary_extension_ops_run_native(op_type: str):
    x_info = helper.make_tensor_value_info("X", TensorProto.FLOAT, [2, 3])
    y_info = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [2, 3])
    model = make_ggml_extension_model(op_type, ["X"], ["Y"], [x_info], [y_info])
    x = np.linspace(-2.0, 2.0, 6, dtype=np.float32).reshape(2, 3)
    if op_type == "SiLU":
        expected = x / (1.0 + np.exp(-x))
    else:
        expected = x / (1.0 + np.exp(-1.702 * x))

    ggml_model = GgmlRuntimeBackend.prepare(model, fallback_policy="strict")
    actual = ggml_model.run({"X": x})

    tolerance = 5e-4 if op_type == "QuickGelu" else 1e-5
    np.testing.assert_allclose(actual[0], expected, rtol=tolerance, atol=tolerance)
    assert not ggml_model.fallback_nodes


@pytest.mark.parametrize(
    "op_type", ["ReGLU", "GeGLU", "SwiGLU", "GeGLUErf", "GeGLUQuick"]
)
def test_ggml_onnx_glu_extension_ops_run_native(op_type: str):
    x_info = helper.make_tensor_value_info("X", TensorProto.FLOAT, [2, 4])
    y_info = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [2, 2])
    model = make_ggml_extension_model(op_type, ["X"], ["Y"], [x_info], [y_info])
    x = np.linspace(-2.0, 2.0, 8, dtype=np.float32).reshape(2, 4)
    gate = x[:, :2]
    values = x[:, 2:]
    if op_type == "ReGLU":
        activation = np.maximum(gate, 0)
    elif op_type == "SwiGLU":
        activation = gate / (1.0 + np.exp(-gate))
    elif op_type == "GeGLUQuick":
        activation = gate / (1.0 + np.exp(-1.702 * gate))
    else:
        erf = np.vectorize(math.erf)
        if op_type == "GeGLUErf":
            activation = 0.5 * gate * (1.0 + erf(gate / np.sqrt(2.0)))
        else:
            inner = np.sqrt(2.0 / np.pi) * (gate + 0.044715 * np.power(gate, 3))
            activation = 0.5 * gate * (1.0 + np.tanh(inner))
    expected = values * activation

    ggml_model = GgmlRuntimeBackend.prepare(model, fallback_policy="strict")
    actual = ggml_model.run({"X": x})

    tolerance = 1e-3 if op_type != "ReGLU" else 1e-5
    np.testing.assert_allclose(actual[0], expected, rtol=tolerance, atol=tolerance)
    assert not ggml_model.fallback_nodes


def test_ggml_onnx_flash_attention_extension_runs_native():
    q_info = helper.make_tensor_value_info("Q", TensorProto.FLOAT, [1, 1, 2, 2])
    k_info = helper.make_tensor_value_info("K", TensorProto.FLOAT, [1, 1, 2, 2])
    v_info = helper.make_tensor_value_info("V", TensorProto.FLOAT, [1, 1, 2, 2])
    y_info = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [1, 1, 2, 2])
    model = make_ggml_extension_model(
        "FlashAttention",
        ["Q", "K", "V"],
        ["Y"],
        [q_info, k_info, v_info],
        [y_info],
    )
    q = np.asarray([[[[0.1, 0.2], [0.3, 0.4]]]], dtype=np.float32)
    k = np.asarray([[[[0.2, 0.1], [0.5, 0.6]]]], dtype=np.float32)
    v = np.asarray([[[[1.0, 2.0], [3.0, 4.0]]]], dtype=np.float32)
    scores = np.matmul(q, np.swapaxes(k, -1, -2)) / np.sqrt(2.0)
    probabilities = np.exp(scores - np.max(scores, axis=-1, keepdims=True))
    probabilities = probabilities / np.sum(probabilities, axis=-1, keepdims=True)
    expected = np.matmul(probabilities, v)

    ggml_model = GgmlRuntimeBackend.prepare(model, fallback_policy="strict")
    actual = ggml_model.run({"Q": q, "K": k, "V": v})

    np.testing.assert_allclose(actual[0], expected, rtol=1e-5, atol=1e-5)
    assert not ggml_model.fallback_nodes


def test_ggml_onnx_misc_extension_ops_run_native():
    x_info = helper.make_tensor_value_info("X", TensorProto.FLOAT, [2, 3])
    roll_output = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [2, 3])
    roll_model = make_ggml_extension_model(
        "Roll", ["X"], ["Y"], [x_info], [roll_output], {"shifts": [1, 0]}
    )
    x = np.arange(6, dtype=np.float32).reshape(2, 3)
    roll_runtime = GgmlRuntimeBackend.prepare(roll_model, fallback_policy="strict")
    roll_actual = roll_runtime.run({"X": x})[0]
    np.testing.assert_allclose(roll_actual, np.roll(x, 1, axis=0))

    argsort_output = helper.make_tensor_value_info("Y", TensorProto.INT32, [2, 3])
    argsort_model = make_ggml_extension_model(
        "ArgSort", ["X"], ["Y"], [x_info], [argsort_output]
    )
    argsort_runtime = GgmlRuntimeBackend.prepare(
        argsort_model, fallback_policy="strict"
    )
    argsort_actual = argsort_runtime.run(
        {"X": np.asarray([[3.0, 1.0, 2.0], [0.0, 2.0, 1.0]], dtype=np.float32)}
    )[0]
    np.testing.assert_array_equal(
        argsort_actual,
        np.asarray([[1, 2, 0], [0, 2, 1]], dtype=np.int32),
    )


def test_ggml_onnx_registers_forward_ggml_extension_ops():
    expected_ops = {
        "AddRelPos",
        "ArgSort",
        "Fill",
        "FlashAttention",
        "GatedDeltaNet",
        "GatedLinearAttention",
        "GeGLU",
        "GeGLUErf",
        "GeGLUQuick",
        "GetRelPos",
        "ReGLU",
        "RWKVWKV6",
        "RWKVWKV7",
        "Rope",
        "Roll",
        "SSMConv",
        "SSMScan",
        "SiLU",
        "SwiGLU",
        "SwiGLUOAI",
        "TimestepEmbedding",
        "WindowPartition",
        "WindowUnpartition",
    }

    assert {
        op_type
        for domain, op_type in onnx_operators.domain_operators
        if domain == "com.ggml"
    }.issuperset(expected_ops)


def test_ggml_onnx_fold_static_shape_nodes_folds_shape_and_size():
    model_input = helper.make_tensor_value_info("X", TensorProto.FLOAT, [2, 3, 4])
    shape_output = helper.make_tensor_value_info("S", TensorProto.INT64, [1])
    size_output = helper.make_tensor_value_info("N", TensorProto.INT64, [])
    shape_node = helper.make_node("Shape", ["X"], ["S"], start=1, end=-1)
    size_node = helper.make_node("Size", ["X"], ["N"])
    graph = helper.make_graph(
        [shape_node, size_node],
        "fold_static_shape_nodes",
        [model_input],
        [shape_output, size_output],
    )
    model = helper.make_model(
        graph,
        producer_name="fold-static-shape-nodes",
        opset_imports=[helper.make_opsetid("", 18)],
    )

    folded = GgmlRuntimeBackend.fold_static_shape_nodes(model)
    initializers = {
        initializer.name: onnx.numpy_helper.to_array(initializer)
        for initializer in folded.graph.initializer
    }

    assert len(folded.graph.node) == 0
    np.testing.assert_array_equal(initializers["S"], np.array([3], dtype=np.int64))
    np.testing.assert_array_equal(initializers["N"], np.array(24, dtype=np.int64))


def test_ggml_onnx_fold_static_shape_nodes_keeps_symbolic_shapes():
    model_input = helper.make_tensor_value_info("X", TensorProto.FLOAT, ["batch", 3])
    model_output = helper.make_tensor_value_info("S", TensorProto.INT64, [2])
    shape_node = helper.make_node("Shape", ["X"], ["S"])
    graph = helper.make_graph(
        [shape_node], "keep_symbolic_shape_node", [model_input], [model_output]
    )
    model = helper.make_model(
        graph,
        producer_name="keep-symbolic-shape-node",
        opset_imports=[helper.make_opsetid("", 18)],
    )

    folded = GgmlRuntimeBackend.fold_static_shape_nodes(model)

    assert [node.op_type for node in folded.graph.node] == ["Shape"]
    assert len(folded.graph.initializer) == 0


# def test_ggml_onnx_graph_optimization():
#     # Construct an onnx graph and optimize it
#     # The graph is of the form y = (A^T)^T * x + b
#     # the optimization should remove the transpose operations

#     # The name of the input tensor
#     input_name = "x"

#     # The name of the weights tensor
#     weight_name_a = "A"
#     weight_name_b = "b"

#     # The name of the output
#     output_name = "y"

#     # Create the nodes (operations) in our graph
#     node1 = helper.make_node(
#         "Transpose", [weight_name_a], ["A_transposed"], name="node1"
#     )  # A^T
#     node2 = helper.make_node(
#         "Transpose", ["A_transposed"], ["A_transposed_transposed"], name="node2"
#     )  # (A^T)^T
#     node3 = helper.make_node(
#         "MatMul", [input_name, "A_transposed_transposed"], ["x_times_A"], name="node3"
#     )  # x * (A^T)^T
#     node4 = helper.make_node(
#         "Add", ["x_times_A", weight_name_b], [output_name], name="node4"
#     )  # x * (A^T)^T + b

#     # Define the tensors (values) in our graph
#     X_value_info = helper.make_tensor_value_info(
#         input_name, TensorProto.FLOAT, [None, 32]
#     )

#     output_value_info = helper.make_tensor_value_info(
#         output_name, TensorProto.FLOAT, [None, 32]
#     )

#     # Set A and b as parameters/weights
#     weights_a = np.random.randn(32, 32).astype(np.float32)

#     weights_b = np.random.randn(32).astype(np.float32)

#     A_init = helper.make_tensor(
#         weight_name_a,
#         TensorProto.FLOAT,
#         [
#             32,
#             32,
#         ],
#         weights_a,
#     )
#     B_init = helper.make_tensor(
#         weight_name_b,
#         TensorProto.FLOAT,
#         [
#             32,
#         ],
#         weights_b,
#     )

#     # Create the graph (model).
#     graph_def = helper.make_graph(
#         [node1, node2, node3, node4],
#         "simple_expression_model",
#         [X_value_info],
#         [output_value_info],
#         [A_init, B_init],
#     )

#     model_def = helper.make_model(graph_def, producer_name="onnx-simple-expression")

#     from typing import Optional, List
#     from ggml.contrib.onnx import OnnxGraphRuleEngine, OnnxGraphRule
#     from onnx.onnx_ml_pb2 import ModelProto, NodeProto

#     class TransposeIdentityRule(OnnxGraphRule):
#         """Transpose Identity Rewrite Rule

#         This rules removes two consecutive transpose nodes that transpose the same tensor.

#         ie Transpose(Transpose(x)) -> x"""

#         def __init__(self):
#             super().__init__()

#         def apply(self, model: ModelProto) -> Optional[ModelProto]:
#             # find first transpose node
#             transpose_node: Optional[NodeProto] = None
#             for node in model.graph.node:
#                 if node.op_type == "Transpose":
#                     transpose_node = node
#                     break
#             else:
#                 return None

#             # find a transpose node that transposes the output of the first transpose node
#             transpose_transpose_node: Optional[NodeProto] = None
#             for node in model.graph.node:
#                 if (
#                     node.op_type == "Transpose"
#                     and node.input[0] == transpose_node.output[0]
#                 ):
#                     transpose_transpose_node = node
#                     break
#             else:
#                 return None

#             # Create a new node list without the two transpose nodes
#             new_nodes: List[NodeProto] = []
#             for node in model.graph.node:
#                 if node not in [transpose_node, transpose_transpose_node]:
#                     new_node = NodeProto()
#                     new_node.CopyFrom(node)
#                     new_node.input[:] = [transpose_node.input[0] if inp == transpose_transpose_node.output[0] else inp for inp in node.input]
#                     new_nodes.append(new_node)

#             # Create the new graph
#             new_graph = helper.make_graph(
#                 new_nodes,
#                 model.graph.name,
#                 model.graph.input,
#                 model.graph.output,
#                 model.graph.initializer,
#             )

#             # create a new model
#             new_model = helper.make_model(
#                 new_graph, producer_name=model.producer_name
#             )

#             return new_model


#     input_data = {"x": np.random.randn(1, 32).astype(np.float32)}

#     f = io.BytesIO()
#     onnx.save(model_def, f)

#     runtime_result = InferenceSession(f.getvalue()).run(None, input_data)

#     ggml_dummy_model = GgmlRuntimeBackend.prepare(model_def)
#     ggml_result = ggml_dummy_model.run(input_data)
#     assert np.allclose(ggml_result[0], runtime_result[0], rtol=1e-03, atol=1e-05)

#     optimizer = OnnxGraphRuleEngine(
#         rules=[TransposeIdentityRule()]
#     )
#     new_model = optimizer.optimize(model=model_def)
#     assert new_model is not None
#     ggml_dummy_model_new = GgmlRuntimeBackend.prepare(new_model)
#     assert ggml_dummy_model_new is not None
#     ggml_result_new = ggml_dummy_model_new.run(input_data)
#     assert np.allclose(ggml_result_new[0], runtime_result[0], rtol=1e-03, atol=1e-05)
#     assert sum([node.op_type == "Transpose" for node in new_model.graph.node]) == 0


# def test_ggml_onnx_runtime_quantized():
#     # Construct an onnx graph of the form Y = X * A + B
#     # and compute the result of the graph with quantized weights
#     # A and B and compare the result with the result of the
#     # unquantized graph

#     # Sizes: X = (32, 32), A = (32, 32), B = (32, 32)

#     # The expressions Y = X * A + B cannot be computed directly with quantized
#     # weights, because ggml expects the quantized weights to appear as the first
#     # input of the MatMul and Add nodes. Therefore, we rewrite the expression
#     # using the following identities:
#     # (AB)^T = B^T A^T
#     # A = (A^T)^T
#     # A + B = B + A
#     # The final expression is Y = B + (A^T X^T)^T

#     from typing import Optional, List, Set
#     from ggml.contrib.onnx import OnnxGraphRuleEngine, OnnxGraphRule
#     from onnx.onnx_ml_pb2 import ModelProto, NodeProto

#     def _depends_on_input(name: str, model: ModelProto) -> bool:
#         # Depth first search to find any node ancestor in model.graph.inputs
#         # that is an ancestor of node
#         initializers = { node.name: node for node in model.graph.initializer }
#         inputs = { node.name: node for node in model.graph.input }
#         outputs = { node.name: node for node in model.graph.output }
#         nodes = { node.name: node for node in model.graph.node }

#         def _dfs(name: str, visited: Set[str]) -> bool:
#             if name in visited:
#                 return False
#             if name in inputs:
#                 return True
#             if name not in nodes:
#                 return False
#             visited.add(name)
#             for inp in nodes[name].input:
#                 if inp in initializers:
#                     continue
#                 if inp in outputs:
#                     continue
#                 if _dfs(nodes[inp].name, visited):
#                     return True
#             return False
#         return _dfs(name, set())

#     class MatMulTransposeRule(OnnxGraphRule):
#         def __init__(self):
#             super().__init__()

#         def apply(self, model: ModelProto) -> Optional[ModelProto]:
#             # find a matmul node
#             matmul_node: Optional[NodeProto] = None
#             for node in model.graph.node:
#                 if node.op_type == "MatMul":
#                     matmul_node = node
#                     break
#             else:
#                 return None

#             # get first and second input of matmul node
#             matmul_input_0 = matmul_node.input[0]
#             matmul_input_1 = matmul_node.input[1]

#             # check that first input is _not_ a weight or constant tensor
#             if _depends_on_input(matmul_input_0, model):
#                 return None

#             # check that second input is a weight or constant tensor
#             if not _depends_on_input(matmul_input_1, model):
#                 return None

#             # replace Matmul(matmul_input_0, matmul_input_1) with Transpose(MatMul(Transpose(matmul_input_1), Transpose(matmul_input_0)))

#             # create new transpose nodes for the inputs
#             transpose_node_0 = NodeProto()
#             transpose_node_0.CopyFrom(matmul_node)
#             transpose_node_0.op_type = "Transpose"
#             transpose_node_0.name = matmul_input_0 + "_transposed"
#             transpose_node_0.input[:] = [matmul_input_0]
#             transpose_node_0.output[:] = [matmul_input_0 + "_transposed"]

#             transpose_node_1 = NodeProto()
#             transpose_node_1.CopyFrom(matmul_node)
#             transpose_node_1.op_type = "Transpose"
#             transpose_node_1.name = matmul_input_1 + "_transposed"
#             transpose_node_1.input[:] = [matmul_input_1]
#             transpose_node_1.output[:] = [matmul_input_1 + "_transposed"]

#             # create new matmul node
#             new_matmul_node = NodeProto()
#             new_matmul_node.CopyFrom(matmul_node)
#             new_matmul_node.op_type = "MatMul"
#             new_matmul_node.name = matmul_node.name + "_inner"
#             new_matmul_node.input[:] = [transpose_node_1.output[0], transpose_node_0.output[0]]
#             new_matmul_node.output[:] = [matmul_node.output[0]]

#             # create final transpose node
#             final_transpose_node = NodeProto()
#             final_transpose_node.CopyFrom(matmul_node)
#             final_transpose_node.op_type = "Transpose"
#             final_transpose_node.name = matmul_node.name # this is the name of the original matmul node
#             final_transpose_node.input[:] = [new_matmul_node.output[0]]
#             final_transpose_node.output[:] = [matmul_node.output[0]]

#             # Create the new node list
#             new_nodes: List[NodeProto] = []
#             for node in model.graph.node:
#                 if node not in [matmul_node]:
#                     new_node = NodeProto()
#                     new_node.CopyFrom(node)
#                     new_nodes.append(new_node)
#                 else:
#                     new_nodes.extend([transpose_node_0, transpose_node_1, new_matmul_node, final_transpose_node])

#             # Create the new graph
#             new_graph = helper.make_graph(
#                 new_nodes,
#                 model.graph.name,
#                 model.graph.input,
#                 model.graph.output,
#                 model.graph.initializer,
#             )

#             # create a new model
#             new_model = helper.make_model(
#                 new_graph, producer_name=model.producer_name
#             )

#             return new_model

#     class AddAssociativityRule(OnnxGraphRule):
#         def __init__(self):
#             super().__init__()

#         def apply(self, model: ModelProto) -> Optional[ModelProto]:
#             # find an add node
#             add_node: Optional[NodeProto] = None
#             for node in model.graph.node:
#                 if node.op_type == "Add":
#                     add_node = node
#                     break
#             else:
#                 return None

#             # get first and second input of add node
#             add_input_0 = add_node.input[0]
#             add_input_1 = add_node.input[1]

#             # check that first input is _not_ a weight or constant tensor
#             if _depends_on_input(add_input_0, model):
#                 return None

#             # check that second input is a weight or constant tensor
#             if not _depends_on_input(add_input_1, model):
#                 return None

#             # replace Add(add_input_0, add_input_1) with Add(add_input_1, add_input_0)

#             # create new add node
#             new_add_node = NodeProto()
#             new_add_node.CopyFrom(add_node)
#             new_add_node.op_type = "Add"
#             new_add_node.name = add_node.name
#             new_add_node.input[:] = [add_input_1, add_input_0]
#             new_add_node.output[:] = [add_node.output[0]]

#             # Create the new node list
#             new_nodes: List[NodeProto] = []
#             for node in model.graph.node:
#                 if node not in [add_node]:
#                     new_node = NodeProto()
#                     new_node.CopyFrom(node)
#                     new_nodes.append(new_node)
#                 else:
#                     new_nodes.extend([new_add_node])

#             # Create the new graph
#             new_graph = helper.make_graph(
#                 new_nodes,
#                 model.graph.name,
#                 model.graph.input,
#                 model.graph.output,
#                 model.graph.initializer,
#             )

#             # create a new model
#             new_model = helper.make_model(
#                 new_graph, producer_name=model.producer_name
#             )

#             return new_model

#     engine = OnnxGraphRuleEngine(
#         rules=[MatMulTransposeRule(), AddAssociativityRule()]
#     )

#     # The name of the input tensor
#     input_name = "X"

#     # The name of the weights tensor
#     weight_name_a = "A"
#     weight_name_b = "B"

#     # The name of the output
#     output_name = "Y"

#     # Create the nodes (operations) in our graph Y = X * A + B

#     # X * A

#     node1 = helper.make_node(
#         "MatMul", [input_name, weight_name_a], ["X_times_A"], name="node1"
#     )  # X * A

#     # X * A + B

#     node2 = helper.make_node(
#         "Add", ["X_times_A", weight_name_b], [output_name], name="node2"
#     )  # X * A + B

#     # Define the tensors (values) in our graph
#     X_value_info = helper.make_tensor_value_info(
#         input_name, TensorProto.FLOAT, [None, 32]
#     )

#     output_value_info = helper.make_tensor_value_info(
#         output_name, TensorProto.FLOAT, [None, 32]
#     )

#     # Set A and B as parameters/weights
#     weights_a = np.random.randn(32, 32).astype(np.float32)

#     weights_b = np.random.randn(32, 32).astype(np.float32)

#     A_init = helper.make_tensor(
#         weight_name_a,
#         TensorProto.FLOAT,
#         [
#             32,
#             32,
#         ],
#         weights_a,
#     )
#     B_init = helper.make_tensor(
#         weight_name_b,
#         TensorProto.FLOAT,
#         [
#             32,
#             32,
#         ],
#         weights_b,
#     )

#     # Create the graph (model).
#     graph_def = helper.make_graph(
#         [node1, node2],
#         "simple_expression_model",
#         [X_value_info],
#         [output_value_info],
#         [A_init, B_init],
#     )

#     model_def = helper.make_model(graph_def, producer_name="onnx-simple-expression")

#     input_data = {"X": np.random.randn(1, 32).astype(np.float32)}

#     f = io.BytesIO()
#     onnx.save(model_def, f)

#     runtime_result = InferenceSession(f.getvalue()).run(None, input_data)

#     # rewrite the graph
#     new_model = engine.optimize(model=model_def)
#     assert new_model is not None

#     ggml_dummy_model = GgmlRuntimeBackend.prepare(new_model)
#     ggml_result = ggml_dummy_model.run(input_data)
#     assert np.allclose(ggml_result[0], runtime_result[0], rtol=1e-03, atol=1e-05)


backend_test = onnx.backend.test.BackendTest(GgmlRuntimeBackend, __name__)

backend_test.include("test_abs_")

backend_test.include("test_acos_")
backend_test.include("test_acosh_")

backend_test.include("test_add_")

backend_test.include("test_and_")
backend_test.include("test_and[234]d_")

backend_test.include("test_ai_onnx_ml_array_feature_extractor")
backend_test.include("test_ai_onnx_ml_binarizer")

backend_test.include("test_asin_")
backend_test.include("test_asinh_")
backend_test.include("test_atan_")
backend_test.include("test_atanh_")

backend_test.include("test_averagepool_1d_default")
backend_test.include("test_averagepool_3d_default")
backend_test.include("test_averagepool_2d_")

backend_test.include("test_batchnorm_")
backend_test.include("test_BatchNorm1d_3d_input_eval")
backend_test.include("test_BatchNorm2d(_momentum)?_eval")
backend_test.include("test_BatchNorm3d(_momentum)?_eval")

backend_test.include("test_blackmanwindow")

backend_test.include("test_bitwise_and_i16_3d")
backend_test.include("test_bitwise_and_i32_2d")
backend_test.include("test_bitwise_and_ui(8_bcast_4v3d|64_bcast_3v1d)")
backend_test.include("test_bitwise_not_2d")
backend_test.include("test_bitwise_not_[34]d")
backend_test.include("test_bitwise_or_i16_4d")
backend_test.include("test_bitwise_or_i32_2d")
backend_test.include("test_bitwise_or_ui(8_bcast_4v3d|64_bcast_3v1d)")
backend_test.include("test_bitwise_xor_i16_3d")
backend_test.include("test_bitwise_xor_i32_2d")
backend_test.include("test_bitwise_xor_ui(8_bcast_4v3d|64_bcast_3v1d)")

backend_test.include("test_bitshift_(left|right)_uint(8|16|32|64)")

backend_test.include("test_argmax_")
backend_test.include("test_argmin_")

backend_test.include("test_operator_basic_")
backend_test.include("test_operator_add")
backend_test.include("test_operator_non_float_params")
backend_test.include("test_operator_params")

backend_test.include("test_cast_DOUBLE_to_FLOAT_")
backend_test.include("test_cast_DOUBLE_to_FLOAT16_")
backend_test.include("test_cast_FLOAT8E4M3FNUZ_to_FLOAT_")
backend_test.include("test_cast_FLOAT8E4M3FN_to_FLOAT_")
backend_test.include("test_cast_FLOAT8E5M2FNUZ_to_FLOAT_")
backend_test.include("test_cast_FLOAT8E5M2_to_FLOAT_")
backend_test.include("test_cast_FLOAT16_to_DOUBLE_")
backend_test.include("test_cast_FLOAT16_to_FLOAT_")
backend_test.include("test_cast_FLOAT_to_DOUBLE_")
backend_test.include("test_cast_FLOAT_to_FLOAT16_")
backend_test.include("test_cast_STRING_to_FLOAT_")
backend_test.include("test_castlike_DOUBLE_to_FLOAT(_expanded)?_")
backend_test.include("test_castlike_DOUBLE_to_FLOAT16(_expanded)?_")
backend_test.include("test_castlike_FLOAT8E4M3FNUZ_to_FLOAT(_expanded)?_")
backend_test.include("test_castlike_FLOAT8E4M3FN_to_FLOAT(_expanded)?_")
backend_test.include("test_castlike_FLOAT8E5M2FNUZ_to_FLOAT(_expanded)?_")
backend_test.include("test_castlike_FLOAT8E5M2_to_FLOAT(_expanded)?_")
backend_test.include("test_castlike_FLOAT16_to_DOUBLE(_expanded)?_")
backend_test.include("test_castlike_FLOAT16_to_FLOAT(_expanded)?_")
backend_test.include("test_castlike_FLOAT_to_DOUBLE(_expanded)?_")
backend_test.include("test_castlike_FLOAT_to_FLOAT16(_expanded)?_")
backend_test.include("test_castlike_STRING_to_FLOAT(_expanded)?_")

backend_test.include("test_ceil_")

backend_test.include("test_celu")

backend_test.include("test_center_crop_pad")

backend_test.include("test_clip_")
backend_test.include("test_operator_clip")

backend_test.include("test_col2im")

backend_test.include("test_concat_")
backend_test.include("test_operator_concat")

backend_test.include("test_compress_")

backend_test.include("test_constant_")

backend_test.include("test_constantofshape")

backend_test.include("test_cos_")

backend_test.include("test_cosh_")

backend_test.include("test_cumsum_")

backend_test.include("test_basic_conv_")
backend_test.include("test_conv_with_")
backend_test.include("test_convinteger_(with|without)_padding_(cpu|cuda)$")
backend_test.include(
    "test_Conv1d(_dilated|_groups|_pad1|_pad1size1|_pad2|_pad2size1|_stride)?_(cpu|cuda)$"
)
backend_test.include("test_Conv2d(_no_bias|_strided|_dilated|_padding)?_(cpu|cuda)$")
backend_test.include(
    "test_Conv2d_depthwise(_strided|_padded|_with_multiplier)?_(cpu|cuda)$"
)
backend_test.include("test_Conv2d_groups(_thnn)?_(cpu|cuda)$")
backend_test.include(
    "test_Conv3d(_no_bias|_stride|_stride_padding|_dilated|_dilated_strided|_groups)?_(cpu|cuda)$"
)
backend_test.include("test_operator_conv_(cpu|cuda)$")


backend_test.include("test_convtranspose")
backend_test.include("test_ConvTranspose2d(_no_bias)?_(cpu|cuda)$")
backend_test.include("test_operator_convtranspose_(cpu|cuda)$")

backend_test.include("test_operator_chunk")

backend_test.include("test_depthtospace")
backend_test.include("test_PixelShuffle")

backend_test.include("test_dft")

backend_test.include("test_det_")

backend_test.include("test_div_")

backend_test.include("test_dropout_")
backend_test.include("test_training_dropout")

backend_test.include("test_elu_")
backend_test.include("test_ELU_")
# backend_test.include("test_elu_example")

backend_test.include("test_einsum_")

backend_test.include("test_erf_")

# backend_test.include("test_eq_")

backend_test.include("test_equal_")
backend_test.exclude(".*equal.*.*string.*")

backend_test.include("test_exp_")
backend_test.include("test_operator_exp_")
backend_test.include("test_operator_index")

backend_test.include("test_eyelike")

backend_test.include("test_expand_")
backend_test.include("test_expand_shape_model")

backend_test.include("test_AvgPool1d")
backend_test.include("test_AvgPool2d")
backend_test.include("test_AvgPool3d(_stride|_stride1_pad0_gpu_input)?_(cpu|cuda)$")

backend_test.include("test_flatten_")
backend_test.include("test_operator_flatten_")


backend_test.include("test_floor_")

backend_test.include("test_greater_")
backend_test.include("test_greater_bcast")
backend_test.include("test_greater_equal")

backend_test.include("test_gridsample")

backend_test.include("test_group_normalization_")

backend_test.include("test_gather_")

backend_test.include("test_gathernd_")

backend_test.include("test_gemm")

backend_test.include("test_globalaveragepool")
backend_test.include("test_globalmaxpool")

backend_test.include("test_hardsigmoid_")

backend_test.include("test_hardswish")

backend_test.include("test_hardmax_")

backend_test.include("test_hammingwindow")

backend_test.include("test_hannwindow")

backend_test.include("test_identity_")
backend_test.exclude("test_identity_opt")  # test case not correct: ONNX issue
backend_test.exclude("test_identity_sequence")  # test case not correct: ONNX issue

backend_test.include("test_instancenorm")

backend_test.include("test_isinf_")
backend_test.include("test_isnan_")

backend_test.include("test_leakyrelu")
backend_test.include("test_LeakyReLU")

backend_test.include("test_layer_normalization_")
backend_test.exclude(
    "test_layer_normalization_.*expanded"
)  # expanded graph form not supported

backend_test.include("test_less_")
backend_test.include("test_less_bcast")
backend_test.include("test_less_equal")

backend_test.include("test_log_")

backend_test.include("test_logsoftmax_")
backend_test.include("test_LogSoftmax")
backend_test.include("test_log_softmax_")

backend_test.include("test_lrn")

backend_test.include("test_lppool_1d_default")
backend_test.include("test_lppool_3d_default")
backend_test.include("test_lppool_2d_")

backend_test.include("test_matmul_")
backend_test.include("test_operator_mm")
backend_test.include("test_Linear")
backend_test.include("test_Embedding")
backend_test.include("test_GLU")

backend_test.include("test_max_")
backend_test.include("test_operator_max_")

backend_test.include("test_maxpool_1d_default")
backend_test.include("test_maxpool_3d_default")
backend_test.include("test_maxpool_2d_")
backend_test.include("test_maxpool_with_argmax_2d")
backend_test.include("test_operator_maxpool")
backend_test.include("test_MaxPool1d")
backend_test.include("test_MaxPool2d")
backend_test.include("test_MaxPool3d(_stride|_stride_padding)?_(cpu|cuda)$")

backend_test.include("test_maxunpool_")


backend_test.include("test_mean_")

backend_test.include("test_mish")

backend_test.include("test_melweightmatrix")

backend_test.include("test_mvn")

backend_test.include("test_mod_broadcast")
backend_test.include("test_mod_mixed_sign_float32")
backend_test.include("test_mod_mixed_sign_float64")
backend_test.include("test_mod_mixed_sign_int8")
backend_test.include("test_mod_mixed_sign_int16")
backend_test.include("test_mod_mixed_sign_int32")
backend_test.include("test_mod_mixed_sign_int64")
backend_test.include("test_mod_mixed_sign_float16")
backend_test.include("test_mod_int64_fmod")
backend_test.include("test_mod_uint(8|16|32|64)")

backend_test.include("test_min_")
backend_test.include("test_operator_min_")


backend_test.include("test_mul_")

backend_test.include("test_neg_")

backend_test.include("test_nllloss_")

backend_test.include("test_nonzero_")

backend_test.include("test_nonmaxsuppression")

backend_test.include("test_not_")

backend_test.include("test_onehot_")

backend_test.include("test_or_")
backend_test.include("test_or[234]d_")

backend_test.include("test_constant_pad")
backend_test.include("test_edge_pad")
backend_test.include("test_reflect_pad")
backend_test.include("test_wrap_pad")
backend_test.include("test_operator_pad")
backend_test.include("test_ConstantPad2d")
backend_test.include("test_ReflectionPad2d")
backend_test.include("test_ReplicationPad2d")
backend_test.include("test_ZeroPad2d")

backend_test.include("test_prelu")
backend_test.include("test_PRelu_")
backend_test.include("test_PReLU_[123]d(_multiparam)?_")
# backend_test.include("test_prelu_example")

backend_test.include("test_PoissonNLLLLoss_no_reduce")

backend_test.include("test_pow_")
backend_test.include("test_operator_pow_")

backend_test.include("test_dequantizelinear(_axis|_e4m3fn|_e5m2)?_(cpu|cuda)$")
backend_test.include(
    "test_dynamicquantizelinear(_(min|max)_adjusted)?(_expanded)?_(cpu|cuda)$"
)
backend_test.include("test_matmulinteger_(cpu|cuda)$")
backend_test.include("test_qlinearconv_(cpu|cuda)$")
backend_test.include("test_qlinearmatmul_(2D|3D)_(cpu|cuda)$")
backend_test.include("test_quantizelinear(_axis)?_(cpu|cuda)$")

backend_test.include("test_range_")
backend_test.exclude("test_range_.*expanded")  # Loop operator not supported

backend_test.include("test_reciprocal")

backend_test.include("test_reduce_max_")
backend_test.include("test_reduce_mean_")
backend_test.include("test_operator_reduced_mean_")
backend_test.include("test_reduce_min_")
backend_test.include("test_reduce_prod_")
backend_test.include("test_reduce_sum_")
backend_test.include("test_operator_reduced_sum_")
backend_test.include("test_reduce_sum_square")
backend_test.include("test_reduce_log_sum_")


backend_test.include("test_reduce_l1_")
backend_test.include("test_reduce_l2_")

backend_test.include("test_relu_")
# backend_test.include("test_relu_example")
backend_test.include("test_ReLU_")

backend_test.include("test_resize_")

backend_test.include("test_reversesequence_")

backend_test.include("test_roialign")

backend_test.include("test_round_")

backend_test.include("test_scatter_")
backend_test.include("test_scatternd")

backend_test.include("test_reshape_")
backend_test.include("test_operator_view")

backend_test.include("test_selu_")
# backend_test.include("test_selu_example")
backend_test.include("test_SELU_")
backend_test.include("test_operator_selu_")

backend_test.include("test_shape_")

backend_test.include("test_shrink")

backend_test.include("test_sigmoid_")
backend_test.include("test_Sigmoid_")

backend_test.include("test_single_relu_model")

backend_test.include("test_sign_")

backend_test.include("test_sin_")

backend_test.include("test_sinh_")

backend_test.include("test_size_")

backend_test.include("test_slice_")

backend_test.include("test_softmax_")
backend_test.include("test_Softmax")
backend_test.include("test_Softmin")
backend_test.include("test_softmax_functional_")
backend_test.include("test_softmax_lastdim")

backend_test.include("test_sce_")

backend_test.include("test_softplus_")
backend_test.include("test_softsign_")
backend_test.include("test_Softplus")
backend_test.include("test_Softsign")

backend_test.include("test_stft")

backend_test.include("test_spacetodepth")

backend_test.include("test_split_")
backend_test.exclude("test_split_to_sequence")  # sequence output not supported

backend_test.include("test_sqrt_")
backend_test.include("test_operator_sqrt_")

backend_test.include("test_squeeze_")

backend_test.include("test_sub_")

backend_test.include("test_sum_")

backend_test.include("test_tan_")

backend_test.include("test_tanh_")
backend_test.include("test_Tanh_")

backend_test.include("test_thresholdedrelu")

backend_test.include("test_tile_")
backend_test.include("test_operator_repeat")

backend_test.include("test_top_k")

backend_test.include("test_tril")
backend_test.include("test_triu")

backend_test.include("test_transpose_")
backend_test.include("test_operator_symbolic_override")
backend_test.include("test_operator_permute2")

backend_test.include("test_unsqueeze_")

backend_test.include("test_upsample_nearest")

backend_test.include("test_unique_")

backend_test.include("test_where_")

backend_test.include("test_xor_")
backend_test.include("test_xor[234]d_")

# backend_test.exclude(".*FLOAT*E*M*.*")
# backend_test.exclude(".*ver18.*")

# This is a pytest magic variable to load extra plugins
pytest_plugins = ("onnx.backend.test.report",)

# import all test cases at global scope to make them visible to python.unittest
globals().update(backend_test.enable_report().test_cases)
