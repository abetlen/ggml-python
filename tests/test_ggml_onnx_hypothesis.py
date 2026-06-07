import io
import enum
import itertools
import typing
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
import pytest

onnx = pytest.importorskip("onnx")
onnxruntime = pytest.importorskip("onnxruntime")
hypothesis = pytest.importorskip("hypothesis")
pytest.importorskip("hypothesis.extra.numpy")

from hypothesis import HealthCheck, event, given, settings, strategies as st  # noqa: E402
from hypothesis.extra import numpy as hnp  # noqa: E402
from onnx import helper  # noqa: E402
from onnx.onnx_pb import TensorProto  # noqa: E402
from onnxruntime import InferenceSession  # type: ignore  # noqa: E402

from ggml.contrib.onnx import (  # noqa: E402
    GgmlRuntimeBackend,
    OnnxOperator,
    ViewTransformSemantics,
    onnx_operators,
)


FLOAT32_VALUES = st.floats(
    min_value=-3.0,
    max_value=3.0,
    allow_nan=False,
    allow_infinity=False,
    width=32,
)
POSITIVE_FLOAT32_VALUES = st.floats(
    min_value=0.25,
    max_value=3.0,
    allow_nan=False,
    allow_infinity=False,
    width=32,
)
NONZERO_FLOAT32_VALUES = st.one_of(
    st.floats(
        min_value=-3.0,
        max_value=-0.25,
        allow_nan=False,
        allow_infinity=False,
        width=32,
    ),
    POSITIVE_FLOAT32_VALUES,
)
SHAPES = (
    (),
    (1,),
    (2,),
    (3,),
    (1, 3),
    (2, 1),
    (2, 2),
    (2, 3),
    (3, 2),
    (1, 2, 3),
    (2, 1, 3),
    (2, 2, 2),
    (1, 2, 1, 3),
    (2, 1, 2, 1),
)
GRAPH_SHAPES = tuple(shape for shape in SHAPES if shape)
ZERO_RESHAPE_SHAPES = (
    (0,),
    (0, 1),
    (1, 0),
    (0, 3),
    (2, 0),
    (0, 2, 1),
    (1, 0, 2, 1),
)
NATIVE_GENERATOR_OP_TYPES = frozenset(
    {
        "Abs",
        "Add",
        "ArgMax",
        "ArgMin",
        "AveragePool",
        "Clip",
        "Concat",
        "Conv",
        "Div",
        "Elu",
        "Expand",
        "Flatten",
        "Gather",
        "GlobalAveragePool",
        "GlobalMaxPool",
        "HardSigmoid",
        "HardSwish",
        "Identity",
        "LeakyRelu",
        "Log",
        "MatMul",
        "MaxPool",
        "Mean",
        "Mul",
        "Neg",
        "Pad",
        "Pow",
        "ReduceL1",
        "ReduceL2",
        "ReduceSum",
        "ReduceSumSquare",
        "Relu",
        "Reshape",
        "Sigmoid",
        "Sign",
        "Slice",
        "Softmax",
        "Sqrt",
        "Split",
        "Squeeze",
        "Sub",
        "Sum",
        "Tanh",
        "Tile",
        "Transpose",
        "Unsqueeze",
    }
)
NATIVE_ACTIVATION_OP_TYPES = frozenset({"HardSigmoid", "HardSwish"})
UNARY_OP_TYPES = frozenset(
    {
        "Abs",
        "Elu",
        "Identity",
        "LeakyRelu",
        "Log",
        "Neg",
        "Relu",
        "Sigmoid",
        "Sign",
        "Sqrt",
        "Tanh",
    }
)
BINARY_OP_TYPES = frozenset({"Add", "Div", "Mul", "Sub"})
VARIADIC_OP_TYPES = frozenset({"Mean", "Sum"})
NATIVE_REDUCE_ALL_OP_TYPES = frozenset(
    {"ReduceL1", "ReduceL2", "ReduceSum", "ReduceSumSquare"}
)
SHAPE_VIEW_OP_TYPES = frozenset({"Flatten", "Reshape", "Squeeze", "Unsqueeze"})
SHAPE_PRESERVING_UNARY_OP_TYPES = frozenset(
    op_type for op_type in UNARY_OP_TYPES if op_type not in {"Log", "Sqrt"}
)
DAG_JOIN_OP_TYPES = frozenset({"Add", "Mean", "Mul", "Sub", "Sum"})
ARITHMETIC_OP_TYPES = frozenset({"Add", "Div", "Mul", "Sub"})
BROADCAST_SHAPE_PAIRS = (
    ((2, 1), (1, 3)),
    ((1, 3), (2, 1)),
    ((2, 1, 3), (1, 2, 1)),
    ((1, 2, 1), (2, 1, 3)),
    ((1, 2, 1, 3), (2, 1, 2, 1)),
    ((2, 1, 2, 1), (1, 2, 1, 3)),
)
SPLIT_INPUT_SHAPES = tuple(
    shape
    for shape in GRAPH_SHAPES
    if len(shape) <= 4 and any(dim >= 2 for dim in shape)
)


@dataclass(frozen=True)
class InitializerSpec:
    name: str
    data_type: int
    value: npt.NDArray[typing.Any]


class ValueDomain(enum.Enum):
    ANY_FLOAT = "any_float"
    NONNEGATIVE = "nonnegative"
    POSITIVE = "positive"


class LayoutKind(enum.Enum):
    CONTIGUOUS = "contiguous"
    SHAPE_VIEW = "shape_view"
    LAYOUT_VIEW = "layout_view"


@dataclass(frozen=True)
class TensorSpec:
    name: str
    shape: typing.Tuple[int, ...]
    dtype: np.dtype[typing.Any]
    domain: ValueDomain = ValueDomain.ANY_FLOAT
    layout: LayoutKind = LayoutKind.CONTIGUOUS


@dataclass(frozen=True)
class GeneratedOpSpec:
    op_type: str
    output_shape: typing.Tuple[int, ...]
    attrs: typing.Dict[str, typing.Any]
    initializers: typing.Tuple[InitializerSpec, ...] = ()
    output_dtype: np.dtype[typing.Any] = np.dtype(np.float32)

    @property
    def is_layout_view(self) -> bool:
        return self.op_type in {"Slice", "Split", "Transpose"}

    @property
    def is_shape_view(self) -> bool:
        return self.op_type in SHAPE_VIEW_OP_TYPES


@dataclass(frozen=True)
class OpCase:
    op_type: str
    inputs: typing.Tuple[str, ...]
    outputs: typing.Tuple[str, ...]
    attrs: typing.Dict[str, typing.Any]
    initializers: typing.Tuple[InitializerSpec, ...]
    output_specs: typing.Tuple[TensorSpec, ...]


@dataclass(frozen=True)
class TestGraphIR:
    __test__: typing.ClassVar[bool] = False

    name: str
    inputs: typing.Tuple[TensorSpec, ...]
    ops: typing.Tuple[OpCase, ...]
    outputs: typing.Tuple[TensorSpec, ...]
    input_values: typing.Dict[str, npt.NDArray[typing.Any]]
    fallback_start: typing.Optional[int] = None
    opset_version: int = 18
    rtol: float = 1e-4
    atol: float = 1e-4


@dataclass(frozen=True)
class GeneratedGraphCase:
    description: str
    ir: TestGraphIR
    model: onnx.ModelProto
    inputs: typing.Dict[str, npt.NDArray[typing.Any]]
    node_count: int
    fallback_start: typing.Optional[int]
    branch_count: int = 0
    runtime_input_count: int = 1
    op_types: typing.Tuple[str, ...] = ()
    expected_fallback_indices: typing.Tuple[int, ...] = ()
    rtol: float = 1e-4
    atol: float = 1e-4

    def __repr__(self) -> str:
        return self.description


def backend_native_op_types() -> typing.FrozenSet[str]:
    native_executions = {
        OnnxOperator.EXECUTION_NATIVE,
        OnnxOperator.EXECUTION_NATIVE_OR_NUMPY_RUNTIME,
        OnnxOperator.EXECUTION_DECOMPOSED,
    }
    return frozenset(
        op_type
        for op_type, operator in onnx_operators.operators.items()
        if operator.execution in native_executions
    )


def model_bytes(model: onnx.ModelProto) -> bytes:
    buffer = io.BytesIO()
    buffer.write(model.SerializeToString())
    return buffer.getvalue()


def float_array_strategy(
    shape: typing.Tuple[int, ...],
    elements: st.SearchStrategy[float] = FLOAT32_VALUES,
) -> st.SearchStrategy[npt.NDArray[np.float32]]:
    return hnp.arrays(np.float32, shape, elements=elements)


def canonical_array(
    shape: typing.Tuple[int, ...],
    positive: bool = False,
) -> npt.NDArray[np.float32]:
    element_count = int(np.prod(shape)) if shape else 1
    if positive:
        values = np.linspace(0.25, 2.5, element_count, dtype=np.float32)
    else:
        values = np.linspace(-1.5, 2.5, element_count, dtype=np.float32)
    if not shape:
        return np.asarray(values[0], dtype=np.float32)
    return values.reshape(shape).astype(np.float32)


def make_float_initializer(name: str, value: npt.ArrayLike) -> InitializerSpec:
    array = np.asarray(value, dtype=np.float32)
    return InitializerSpec(name, TensorProto.FLOAT, array)


def make_int64_initializer(name: str, value: npt.ArrayLike) -> InitializerSpec:
    array = np.asarray(value, dtype=np.int64)
    return InitializerSpec(name, TensorProto.INT64, array)


def initializer_proto(initializer: InitializerSpec) -> onnx.TensorProto:
    return helper.make_tensor(
        initializer.name,
        initializer.data_type,
        list(initializer.value.shape),
        initializer.value.reshape(-1),
    )


def value_info(
    name: str, elem_type: int, shape: typing.Tuple[int, ...]
) -> onnx.ValueInfoProto:
    return helper.make_tensor_value_info(name, elem_type, list(shape))


def tensor_spec_value_info(spec: TensorSpec) -> onnx.ValueInfoProto:
    if spec.dtype == np.dtype(np.float32):
        return value_info(spec.name, TensorProto.FLOAT, spec.shape)
    if spec.dtype == np.dtype(np.int64):
        return value_info(spec.name, TensorProto.INT64, spec.shape)
    raise ValueError(f"Unsupported test tensor dtype: {spec.dtype}")


def alternate_shapes(
    shape: typing.Tuple[int, ...],
) -> typing.Tuple[typing.Tuple[int, ...], ...]:
    element_count = int(np.prod(shape)) if shape else 1
    shapes = [
        candidate
        for candidate in SHAPES
        if int(np.prod(candidate)) == element_count and len(candidate) <= 4
    ]
    return tuple(shapes) or (shape,)


def shape_element_count(shape: typing.Tuple[int, ...]) -> int:
    return int(np.prod(shape, dtype=np.int64)) if shape else 1


def reshape_output_shape(
    input_shape: typing.Tuple[int, ...],
    shape_values: npt.NDArray[np.int64],
    allowzero: bool,
) -> typing.Tuple[int, ...]:
    new_shape = np.asarray(shape_values, dtype=np.int64).ravel().copy()
    if not allowzero:
        keep_indexes = np.where(new_shape == 0)[0]
        new_shape[keep_indexes] = np.asarray(input_shape, dtype=np.int64)[keep_indexes]
    return np.empty(input_shape, dtype=np.float32).reshape(new_shape).shape


def reduce_all_shape(
    shape: typing.Tuple[int, ...], keepdims: bool
) -> typing.Tuple[int, ...]:
    if keepdims:
        return tuple(1 for _ in shape)
    return ()


def normalize_axes(axes: typing.Sequence[int], rank: int) -> typing.Tuple[int, ...]:
    return tuple(axis + rank if axis < 0 else axis for axis in axes)


def reduce_axes_shape(
    shape: typing.Tuple[int, ...],
    axes: typing.Sequence[int],
    keepdims: bool,
    noop: bool = False,
) -> typing.Tuple[int, ...]:
    if noop:
        return shape
    if not axes:
        return reduce_all_shape(shape, keepdims)

    normalized_axes = set(normalize_axes(axes, len(shape)))
    if keepdims:
        return tuple(
            1 if axis in normalized_axes else dim for axis, dim in enumerate(shape)
        )
    return tuple(dim for axis, dim in enumerate(shape) if axis not in normalized_axes)


def flatten_shape(
    shape: typing.Tuple[int, ...],
    axis: int,
) -> typing.Tuple[int, int]:
    if axis < 0:
        axis += len(shape)
    first_dim = int(np.prod(shape[:axis], dtype=np.int64)) if axis else 1
    second_dim = int(np.prod(shape[axis:], dtype=np.int64)) if axis < len(shape) else 1
    return first_dim, second_dim


def squeeze_shape(
    shape: typing.Tuple[int, ...],
    axes: typing.Optional[typing.Tuple[int, ...]],
) -> typing.Tuple[int, ...]:
    if axes is None:
        return np.squeeze(np.empty(shape, dtype=np.float32)).shape
    normalized_axes = tuple(axis + len(shape) if axis < 0 else axis for axis in axes)
    return np.squeeze(np.empty(shape, dtype=np.float32), axis=normalized_axes).shape


def unsqueeze_shape(
    shape: typing.Tuple[int, ...],
    axes: typing.Tuple[int, ...],
) -> typing.Tuple[int, ...]:
    output_rank = len(shape) + len(axes)
    output = np.empty(shape, dtype=np.float32)
    normalized_axes = [axis if axis >= 0 else axis + output_rank for axis in axes]
    for axis in sorted(normalized_axes):
        output = np.expand_dims(output, axis=axis)
    return output.shape


def axes_with_negative_forms(
    axes: typing.Sequence[int],
    rank: int,
) -> typing.Tuple[typing.Tuple[int, ...], ...]:
    variants = []
    for mask in range(1 << len(axes)):
        variant = []
        for index, axis in enumerate(axes):
            if mask & (1 << index):
                variant.append(axis - rank)
            else:
                variant.append(axis)
        variants.append(tuple(variant))
    return tuple(variants)


def native_transpose_permutations(
    rank: int,
) -> typing.Tuple[typing.Tuple[int, ...], ...]:
    if rank == 0:
        return ()
    if rank <= 2:
        return tuple(itertools.permutations(range(rank)))
    return tuple(ViewTransformSemantics.GGML_TRANSPOSE_AXES_BY_RANK.get(rank, ()))


def transpose_shape(
    shape: typing.Tuple[int, ...],
    permutation: typing.Tuple[int, ...],
) -> typing.Tuple[int, ...]:
    return tuple(shape[axis] for axis in permutation)


def broadcast_shape(
    left_shape: typing.Tuple[int, ...],
    right_shape: typing.Tuple[int, ...],
) -> typing.Tuple[int, ...]:
    return tuple(np.broadcast_shapes(left_shape, right_shape))


def normalize_axis(axis: int, rank: int) -> int:
    return axis + rank if axis < 0 else axis


def split_output_shapes(
    shape: typing.Tuple[int, ...],
    axis: int,
    split_values: typing.Sequence[int],
) -> typing.Tuple[typing.Tuple[int, ...], ...]:
    normalized_axis = normalize_axis(axis, len(shape))
    output_shapes = []
    for split_value in split_values:
        output_shape = list(shape)
        output_shape[normalized_axis] = split_value
        output_shapes.append(tuple(output_shape))
    return tuple(output_shapes)


def concat_shape(
    shape: typing.Tuple[int, ...], axis: int, extra_dim: int
) -> typing.Tuple[int, ...]:
    normalized_axis = normalize_axis(axis, len(shape))
    output_shape = list(shape)
    output_shape[normalized_axis] += extra_dim
    return tuple(output_shape)


def conv2d_shape(
    input_shape: typing.Tuple[int, ...],
    output_channels: int,
    kernel_shape: typing.Tuple[int, int],
    pads: typing.Tuple[int, int, int, int],
    strides: typing.Tuple[int, int],
) -> typing.Tuple[int, int, int, int]:
    output_h = (
        input_shape[2] + pads[0] + pads[2] - (kernel_shape[0] - 1) - 1
    ) // strides[0] + 1
    output_w = (
        input_shape[3] + pads[1] + pads[3] - (kernel_shape[1] - 1) - 1
    ) // strides[1] + 1
    return input_shape[0], output_channels, output_h, output_w


def pool2d_shape(
    input_shape: typing.Tuple[int, ...],
    kernel_shape: typing.Tuple[int, int],
    pads: typing.Tuple[int, int, int, int],
    strides: typing.Tuple[int, int],
) -> typing.Tuple[int, int, int, int]:
    output_h = (input_shape[2] + pads[0] + pads[2] - kernel_shape[0]) // strides[0] + 1
    output_w = (input_shape[3] + pads[1] + pads[3] - kernel_shape[1]) // strides[1] + 1
    return input_shape[0], input_shape[1], output_h, output_w


def pad_shape(
    input_shape: typing.Tuple[int, ...], pads: typing.Sequence[int]
) -> typing.Tuple[int, ...]:
    rank = len(input_shape)
    return tuple(
        input_shape[i] + int(pads[i]) + int(pads[i + rank]) for i in range(rank)
    )


def slice_shape(
    input_shape: typing.Tuple[int, ...],
    axis: int,
    start: int,
    end: int,
) -> typing.Tuple[int, ...]:
    normalized_axis = normalize_axis(axis, len(input_shape))
    output_shape = list(input_shape)
    output_shape[normalized_axis] = end - start
    return tuple(output_shape)


def output_domain(op: GeneratedOpSpec, input_spec: TensorSpec) -> ValueDomain:
    if op.op_type in {
        "Abs",
        "Clip",
        "HardSigmoid",
        "Pow",
        "ReduceL1",
        "ReduceL2",
        "ReduceSumSquare",
        "Relu",
        "Sqrt",
    }:
        return ValueDomain.NONNEGATIVE
    if op.op_type in {"Sigmoid", "Softmax"}:
        return ValueDomain.POSITIVE
    if op.op_type in SHAPE_VIEW_OP_TYPES or op.op_type in {
        "Expand",
        "Identity",
        "Slice",
        "Split",
        "Tile",
        "Transpose",
    }:
        return input_spec.domain
    return ValueDomain.ANY_FLOAT


def output_layout(op: GeneratedOpSpec) -> LayoutKind:
    if op.is_layout_view:
        return LayoutKind.LAYOUT_VIEW
    if op.is_shape_view:
        return LayoutKind.SHAPE_VIEW
    return LayoutKind.CONTIGUOUS


def op_case_output_domain(
    op_type: str, input_specs: typing.Sequence[TensorSpec]
) -> ValueDomain:
    if not input_specs:
        return ValueDomain.ANY_FLOAT
    if op_type in {
        "Abs",
        "Clip",
        "HardSigmoid",
        "Pow",
        "ReduceL1",
        "ReduceL2",
        "ReduceSumSquare",
        "Relu",
        "Sqrt",
    }:
        return ValueDomain.NONNEGATIVE
    if op_type in {"Sigmoid", "Softmax"}:
        return ValueDomain.POSITIVE
    if op_type in SHAPE_VIEW_OP_TYPES or op_type in {
        "Expand",
        "Identity",
        "Slice",
        "Split",
        "Tile",
        "Transpose",
    }:
        return input_specs[0].domain
    return ValueDomain.ANY_FLOAT


def op_case_output_layout(op_type: str) -> LayoutKind:
    if op_type in {"Slice", "Split", "Transpose"}:
        return LayoutKind.LAYOUT_VIEW
    if op_type in SHAPE_VIEW_OP_TYPES:
        return LayoutKind.SHAPE_VIEW
    return LayoutKind.CONTIGUOUS


def bind_op_case(
    op: GeneratedOpSpec,
    index: int,
    input_spec: TensorSpec,
    is_final: bool,
    output_name: typing.Optional[str] = None,
) -> OpCase:
    if output_name is None:
        output_name = "Y" if is_final else "T%d" % index
    initializers = tuple(
        InitializerSpec(
            "%s_%d" % (initializer.name, index),
            initializer.data_type,
            initializer.value,
        )
        for initializer in op.initializers
    )
    output_spec = TensorSpec(
        output_name,
        op.output_shape,
        op.output_dtype,
        output_domain(op, input_spec),
        output_layout(op),
    )
    return OpCase(
        op.op_type,
        (input_spec.name, *(initializer.name for initializer in initializers)),
        (output_name,),
        op.attrs,
        initializers,
        (output_spec,),
    )


def build_graph_ir(
    name: str,
    input_shape: typing.Tuple[int, ...],
    input_array: npt.NDArray[np.float32],
    ops: typing.Sequence[GeneratedOpSpec],
    fallback_start: typing.Optional[int],
    input_domain: ValueDomain = ValueDomain.ANY_FLOAT,
    opset_version: int = 18,
) -> TestGraphIR:
    input_spec = TensorSpec(
        "X",
        input_shape,
        np.dtype(np.float32),
        input_domain,
        LayoutKind.CONTIGUOUS,
    )
    current_spec = input_spec
    op_cases = []
    for index, op in enumerate(ops):
        op_case = bind_op_case(
            op,
            index,
            current_spec,
            is_final=index == len(ops) - 1,
        )
        op_cases.append(op_case)
        current_spec = op_case.output_specs[0]

    return TestGraphIR(
        name=name,
        inputs=(input_spec,),
        ops=tuple(op_cases),
        outputs=(current_spec,),
        input_values={"X": np.asarray(input_array, dtype=np.float32)},
        fallback_start=fallback_start,
        opset_version=opset_version,
    )


def build_direct_graph_case(
    description: str,
    inputs: typing.Sequence[TensorSpec],
    ops: typing.Sequence[OpCase],
    outputs: typing.Sequence[TensorSpec],
    input_values: typing.Dict[str, npt.NDArray[typing.Any]],
    fallback_start: typing.Optional[int],
    branch_count: int = 0,
    opset_version: int = 18,
    expected_fallback_indices: typing.Sequence[int] = (),
) -> GeneratedGraphCase:
    ir = TestGraphIR(
        name=description,
        inputs=tuple(inputs),
        ops=tuple(ops),
        outputs=tuple(outputs),
        input_values=input_values,
        fallback_start=fallback_start,
        opset_version=opset_version,
    )
    model = to_onnx_model(ir)
    return GeneratedGraphCase(
        description=description,
        ir=ir,
        model=model,
        inputs=ir.input_values,
        node_count=len(ir.ops),
        fallback_start=ir.fallback_start,
        branch_count=branch_count,
        runtime_input_count=len(ir.inputs),
        op_types=tuple(op.op_type for op in ir.ops),
        expected_fallback_indices=tuple(expected_fallback_indices),
    )


def to_onnx_model(ir: TestGraphIR) -> onnx.ModelProto:
    nodes = []
    initializers = []
    for index, op in enumerate(ir.ops):
        initializers.extend(
            initializer_proto(initializer) for initializer in op.initializers
        )
        nodes.append(
            helper.make_node(
                op.op_type,
                list(op.inputs),
                list(op.outputs),
                name="%s_%d" % (op.op_type.lower(), index),
                **op.attrs,
            )
        )

    graph = helper.make_graph(
        nodes,
        ir.name,
        [tensor_spec_value_info(input_spec) for input_spec in ir.inputs],
        [tensor_spec_value_info(output_spec) for output_spec in ir.outputs],
        initializers,
    )
    return helper.make_model(
        graph,
        producer_name=ir.name,
        opset_imports=[helper.make_opsetid("", ir.opset_version)],
    )


def build_model_case(
    description: str,
    input_shape: typing.Tuple[int, ...],
    input_array: npt.NDArray[np.float32],
    ops: typing.Sequence[GeneratedOpSpec],
    fallback_start: typing.Optional[int],
    input_domain: ValueDomain = ValueDomain.ANY_FLOAT,
    opset_version: int = 18,
    expected_fallback_indices: typing.Sequence[int] = (),
) -> GeneratedGraphCase:
    ir = build_graph_ir(
        description,
        input_shape,
        input_array,
        ops,
        fallback_start,
        input_domain=input_domain,
        opset_version=opset_version,
    )
    model = to_onnx_model(ir)
    return GeneratedGraphCase(
        description=description,
        ir=ir,
        model=model,
        inputs=ir.input_values,
        node_count=len(ir.ops),
        fallback_start=ir.fallback_start,
        op_types=tuple(op.op_type for op in ir.ops),
        expected_fallback_indices=tuple(expected_fallback_indices),
    )


def op_input_shapes(op_type: str) -> typing.Tuple[typing.Tuple[int, ...], ...]:
    if op_type in {"AveragePool", "MaxPool"}:
        return ((1, 1, 3, 3), (1, 2, 4, 4), (2, 1, 5, 4))
    if op_type == "Conv":
        return (
            (1, 1, 3, 3),
            (1, 2, 4, 4),
            (2, 1, 5, 4),
        )
    if op_type in {"ArgMax", "ArgMin", "Gather"}:
        return ((2, 3), (3, 2), (4, 3))
    if op_type in {"Expand", "Tile"}:
        return tuple(shape for shape in GRAPH_SHAPES if len(shape) <= 4)
    if op_type in {"GlobalAveragePool", "GlobalMaxPool"}:
        return ((1, 1, 3, 3), (1, 2, 4, 4), (2, 1, 5, 4))
    if op_type == "MatMul":
        return ((1, 1), (1, 3), (2, 1), (2, 3), (3, 2))
    if op_type in NATIVE_REDUCE_ALL_OP_TYPES:
        return tuple(shape for shape in GRAPH_SHAPES if len(shape) <= 4)
    if op_type == "Flatten":
        return tuple(shape for shape in GRAPH_SHAPES if len(shape) <= 4)
    if op_type == "Squeeze":
        return tuple(shape for shape in GRAPH_SHAPES if 1 in shape and len(shape) <= 4)
    if op_type == "Unsqueeze":
        return tuple(shape for shape in SHAPES if len(shape) <= 3)
    if op_type == "Transpose":
        return tuple(
            shape for shape in GRAPH_SHAPES if native_transpose_permutations(len(shape))
        )
    if op_type in {"Concat", "Split"}:
        return tuple(shape for shape in GRAPH_SHAPES if len(shape) <= 4)
    if op_type == "Slice":
        return tuple(
            shape
            for shape in GRAPH_SHAPES
            if len(shape) <= 4 and any(dim >= 2 for dim in shape)
        )
    if op_type == "Pad":
        return tuple(shape for shape in GRAPH_SHAPES if len(shape) <= 4)
    if op_type == "Softmax":
        return GRAPH_SHAPES
    if op_type == "Reshape":
        return tuple(shape for shape in SHAPES if len(shape) <= 4)
    return tuple(shape for shape in SHAPES if len(shape) <= 4)


def op_needs_positive_input(op_type: str) -> bool:
    return op_type in {"Log", "Sqrt"}


def canonical_input_shape(op_type: str) -> typing.Tuple[int, ...]:
    if op_type in {"AveragePool", "Conv", "MaxPool"}:
        return (1, 1, 4, 4)
    if op_type in {"ArgMax", "ArgMin", "Gather"}:
        return (3, 2)
    if op_type in {"Expand", "Tile"}:
        return (2, 1)
    if op_type in {"GlobalAveragePool", "GlobalMaxPool"}:
        return (1, 2, 4, 3)
    if op_type == "MatMul":
        return (2, 3)
    if op_type in NATIVE_REDUCE_ALL_OP_TYPES:
        return (2, 3)
    if op_type == "Flatten":
        return (2, 3, 4)
    if op_type == "Reshape":
        return (2, 3)
    if op_type == "Squeeze":
        return (1, 2, 1)
    if op_type == "Transpose":
        return (2, 3)
    if op_type == "Unsqueeze":
        return (2, 3)
    if op_type == "Slice":
        return (3, 2)
    return (2, 3)


def canonical_generated_op(
    op_type: str,
    shape: typing.Tuple[int, ...],
) -> GeneratedOpSpec:
    if op_type == "HardSigmoid":
        return GeneratedOpSpec(op_type, shape, {"alpha": 1.0 / 6.0, "beta": 0.5})
    if op_type == "HardSwish":
        return GeneratedOpSpec(op_type, shape, {})
    if op_type in UNARY_OP_TYPES:
        attrs = {"alpha": 0.01} if op_type == "LeakyRelu" else {}
        return GeneratedOpSpec(op_type, shape, attrs)
    if op_type in {"ArgMax", "ArgMin"}:
        return GeneratedOpSpec(
            op_type,
            (shape[0],),
            {"axis": -1, "keepdims": 0, "select_last_index": 1},
            output_dtype=np.dtype(np.int64),
        )

    if op_type in {"GlobalAveragePool", "GlobalMaxPool"}:
        return GeneratedOpSpec(op_type, (shape[0], shape[1], 1, 1), {})
    if op_type == "AveragePool":
        kernel_shape = (2, 2)
        strides = (1, 1)
        pads = (0, 0, 0, 0)
        return GeneratedOpSpec(
            op_type,
            pool2d_shape(shape, kernel_shape, pads, strides),
            {"kernel_shape": list(kernel_shape), "strides": list(strides)},
        )
    if op_type == "MaxPool":
        kernel_shape = (2, 2)
        strides = (1, 1)
        pads = (0, 0, 0, 0)
        return GeneratedOpSpec(
            op_type,
            pool2d_shape(shape, kernel_shape, pads, strides),
            {"kernel_shape": list(kernel_shape), "strides": list(strides)},
        )
    if op_type == "Clip":
        return GeneratedOpSpec(
            op_type,
            shape,
            {},
            (
                make_float_initializer("min", np.asarray(0.0, dtype=np.float32)),
                make_float_initializer("max", np.asarray(2.0, dtype=np.float32)),
            ),
        )
    if op_type == "Concat":
        axis = 0
        constant_shape = (1, *shape[1:])
        return GeneratedOpSpec(
            op_type,
            concat_shape(shape, axis, constant_shape[axis]),
            {"axis": axis},
            (make_float_initializer("C", canonical_array(constant_shape)),),
        )
    if op_type == "Conv":
        output_channels = 2
        kernel_shape = (2, 2)
        pads = (0, 0, 0, 0)
        strides = (1, 1)
        weights = canonical_array((output_channels, shape[1], *kernel_shape))
        bias = np.linspace(0.1, 0.2, output_channels, dtype=np.float32)
        return GeneratedOpSpec(
            op_type,
            conv2d_shape(shape, output_channels, kernel_shape, pads, strides),
            {
                "kernel_shape": list(kernel_shape),
                "pads": list(pads),
                "strides": list(strides),
            },
            (
                make_float_initializer("W", weights),
                make_float_initializer("B", bias),
            ),
        )

    if op_type == "Expand":
        output_shape = (*shape[:-1], 3) if shape[-1] == 1 else shape
        return GeneratedOpSpec(
            op_type,
            output_shape,
            {},
            (
                make_int64_initializer(
                    "shape", np.asarray(output_shape, dtype=np.int64)
                ),
            ),
        )
    if op_type == "Gather":
        indices = np.asarray([0, min(shape[0] - 1, 1)], dtype=np.int64)
        return GeneratedOpSpec(
            op_type,
            (indices.size, shape[1]),
            {"axis": 0},
            (make_int64_initializer("indices", indices),),
        )
    if op_type == "Pad":
        pads = tuple(0 for _ in shape) + tuple(1 for _ in shape)
        return GeneratedOpSpec(
            op_type,
            pad_shape(shape, pads),
            {"mode": b"constant"},
            (make_int64_initializer("pads", np.asarray(pads, dtype=np.int64)),),
        )
    if op_type == "Tile":
        repeats = tuple(
            2 if index == len(shape) - 1 else 1 for index in range(len(shape))
        )
        output_shape = tuple(dim * repeat for dim, repeat in zip(shape, repeats))
        return GeneratedOpSpec(
            op_type,
            output_shape,
            {},
            (make_int64_initializer("repeats", np.asarray(repeats, dtype=np.int64)),),
        )
    if op_type == "Slice":
        axis = 0
        start = 0
        end = max(1, shape[axis] - 1)
        return GeneratedOpSpec(
            op_type,
            slice_shape(shape, axis, start, end),
            {},
            (
                make_int64_initializer("starts", np.asarray([start], dtype=np.int64)),
                make_int64_initializer("ends", np.asarray([end], dtype=np.int64)),
                make_int64_initializer("axes", np.asarray([axis], dtype=np.int64)),
            ),
        )
    if op_type == "Softmax":
        return GeneratedOpSpec(op_type, shape, {"axis": -1})
    if op_type in {"Add", "Mul", "Sub"}:
        return GeneratedOpSpec(
            op_type,
            shape,
            {},
            (make_float_initializer("C", canonical_array(shape)),),
        )
    if op_type == "Div":
        return GeneratedOpSpec(
            op_type,
            shape,
            {},
            (make_float_initializer("C", np.full(shape, 2.0, dtype=np.float32)),),
        )
    if op_type == "Pow":
        return GeneratedOpSpec(
            op_type,
            shape,
            {},
            (make_float_initializer("P", np.array(2.0, dtype=np.float32)),),
        )
    if op_type in VARIADIC_OP_TYPES:
        return GeneratedOpSpec(
            op_type,
            shape,
            {},
            (
                make_float_initializer("A", canonical_array(shape)),
                make_float_initializer("B", canonical_array(shape, positive=True)),
            ),
        )
    if op_type in NATIVE_REDUCE_ALL_OP_TYPES:
        return GeneratedOpSpec(op_type, reduce_all_shape(shape, keepdims=True), {})
    if op_type == "MatMul":
        rhs = canonical_array((shape[-1], 2), positive=True)
        return GeneratedOpSpec(
            op_type,
            (shape[-2], 2),
            {},
            (make_float_initializer("B", rhs),),
        )
    if op_type == "Flatten":
        axis = 1
        return GeneratedOpSpec(op_type, flatten_shape(shape, axis), {"axis": axis})
    if op_type == "Reshape":
        output_shape = (3, 2)
        return GeneratedOpSpec(
            op_type,
            output_shape,
            {},
            (
                make_int64_initializer(
                    "shape", np.asarray(output_shape, dtype=np.int64)
                ),
            ),
        )
    if op_type == "Squeeze":
        axes = (0, 2)
        return GeneratedOpSpec(
            op_type,
            squeeze_shape(shape, axes),
            {},
            (make_int64_initializer("axes", np.asarray(axes, dtype=np.int64)),),
        )
    if op_type == "Transpose":
        permutation = (1, 0)
        return GeneratedOpSpec(
            op_type,
            tuple(shape[axis] for axis in permutation),
            {"perm": list(permutation)},
        )
    if op_type == "Split":
        return GeneratedOpSpec(op_type, shape, {"axis": 0, "num_outputs": 1})
    if op_type == "Unsqueeze":
        axes = (0, 2)
        return GeneratedOpSpec(
            op_type,
            unsqueeze_shape(shape, axes),
            {},
            (make_int64_initializer("axes", np.asarray(axes, dtype=np.int64)),),
        )
    raise AssertionError(f'No canonical generator for native op "{op_type}"')


def canonical_native_single_op_cases() -> typing.Tuple[GeneratedGraphCase, ...]:
    cases = []
    for op_type in sorted(NATIVE_GENERATOR_OP_TYPES):
        shape = canonical_input_shape(op_type)
        positive_input = op_needs_positive_input(op_type)
        input_array = canonical_array(shape, positive=positive_input)
        cases.append(
            build_model_case(
                "hypothesis_canonical_%s" % op_type.lower(),
                shape,
                input_array,
                [canonical_generated_op(op_type, shape)],
                fallback_start=None,
                input_domain=(
                    ValueDomain.POSITIVE if positive_input else ValueDomain.ANY_FLOAT
                ),
            )
        )
    return tuple(cases)


@st.composite
def generated_op_strategy(
    draw: st.DrawFn,
    op_type: str,
    input_spec: TensorSpec,
) -> GeneratedOpSpec:
    shape = input_spec.shape
    if op_type == "HardSigmoid":
        return GeneratedOpSpec(op_type, shape, {"alpha": 1.0 / 6.0, "beta": 0.5})

    if op_type == "HardSwish":
        return GeneratedOpSpec(op_type, shape, {})

    if op_type in UNARY_OP_TYPES:
        attrs = {"alpha": 0.01} if op_type == "LeakyRelu" else {}
        return GeneratedOpSpec(op_type, shape, attrs)

    if op_type in {"ArgMax", "ArgMin"}:
        return GeneratedOpSpec(
            op_type,
            (shape[0],),
            {
                "axis": draw(st.sampled_from([-1, 1])),
                "keepdims": 0,
                "select_last_index": 1,
            },
            output_dtype=np.dtype(np.int64),
        )

    if op_type in {"GlobalAveragePool", "GlobalMaxPool"}:
        return GeneratedOpSpec(op_type, (shape[0], shape[1], 1, 1), {})

    if op_type == "AveragePool":
        kernel_h = draw(st.integers(min_value=1, max_value=min(3, shape[2])))
        kernel_w = draw(st.integers(min_value=1, max_value=min(3, shape[3])))
        kernel_shape = (kernel_h, kernel_w)
        stride_h = draw(st.integers(min_value=1, max_value=2))
        stride_w = draw(st.integers(min_value=1, max_value=2))
        strides = (stride_h, stride_w)
        pads = (0, 0, 0, 0)
        return GeneratedOpSpec(
            op_type,
            pool2d_shape(shape, kernel_shape, pads, strides),
            {
                "kernel_shape": list(kernel_shape),
                "strides": list(strides),
                "pads": list(pads),
            },
        )

    if op_type == "MaxPool":
        kernel_h = draw(st.integers(min_value=1, max_value=min(3, shape[2])))
        kernel_w = draw(st.integers(min_value=1, max_value=min(3, shape[3])))
        kernel_shape = (kernel_h, kernel_w)
        stride_h = draw(st.integers(min_value=1, max_value=2))
        stride_w = draw(st.integers(min_value=1, max_value=2))
        strides = (stride_h, stride_w)
        pads = (0, 0, 0, 0)
        return GeneratedOpSpec(
            op_type,
            pool2d_shape(shape, kernel_shape, pads, strides),
            {
                "kernel_shape": list(kernel_shape),
                "strides": list(strides),
                "pads": list(pads),
            },
        )

    if op_type == "Clip":
        return GeneratedOpSpec(
            op_type,
            shape,
            {},
            (
                make_float_initializer("min", np.asarray(0.0, dtype=np.float32)),
                make_float_initializer("max", np.asarray(2.0, dtype=np.float32)),
            ),
        )

    if op_type == "Concat":
        axis = draw(st.integers(min_value=0, max_value=len(shape) - 1))
        constant_shape = list(shape)
        constant_shape[axis] = draw(st.integers(min_value=1, max_value=3))
        constant = draw(float_array_strategy(tuple(constant_shape)))
        return GeneratedOpSpec(
            op_type,
            concat_shape(shape, axis, constant_shape[axis]),
            {"axis": axis},
            (make_float_initializer("C", constant),),
        )

    if op_type == "Conv":
        output_channels = draw(st.integers(min_value=1, max_value=3))
        pad = draw(st.integers(min_value=0, max_value=1))
        kernel_h = draw(st.integers(min_value=1, max_value=min(3, shape[2])))
        kernel_w = draw(st.integers(min_value=1, max_value=min(3, shape[3])))
        kernel_shape = (kernel_h, kernel_w)
        pads = (pad, pad, pad, pad)
        strides = (1, 1)
        weights = draw(
            float_array_strategy((output_channels, shape[1], kernel_h, kernel_w))
        )
        output_shape = conv2d_shape(shape, output_channels, kernel_shape, pads, strides)
        bias = draw(float_array_strategy((output_channels,)))
        return GeneratedOpSpec(
            op_type,
            output_shape,
            {
                "kernel_shape": list(kernel_shape),
                "pads": list(pads),
                "strides": list(strides),
            },
            (
                make_float_initializer("W", weights),
                make_float_initializer("B", bias),
            ),
        )

    if op_type == "Expand":
        extra_rank = draw(st.integers(min_value=0, max_value=4 - len(shape)))
        leading_shape = tuple(
            draw(st.integers(min_value=1, max_value=3)) for _ in range(extra_rank)
        )
        body_shape = tuple(
            draw(st.integers(min_value=1, max_value=3)) if dim == 1 else dim
            for dim in shape
        )
        output_shape = leading_shape + body_shape
        return GeneratedOpSpec(
            op_type,
            output_shape,
            {},
            (
                make_int64_initializer(
                    "shape", np.asarray(output_shape, dtype=np.int64)
                ),
            ),
        )

    if op_type == "Gather":
        index_count = draw(st.integers(min_value=1, max_value=min(3, shape[0])))
        indices = draw(
            st.lists(
                st.integers(min_value=0, max_value=shape[0] - 1),
                min_size=index_count,
                max_size=index_count,
            ).map(lambda values: np.asarray(values, dtype=np.int64))
        )
        return GeneratedOpSpec(
            op_type,
            (indices.size, shape[1]),
            {"axis": 0},
            (make_int64_initializer("indices", indices),),
        )

    if op_type == "Pad":
        after_pads = draw(
            st.lists(
                st.integers(min_value=0, max_value=2),
                min_size=len(shape),
                max_size=len(shape),
            ).map(tuple)
        )
        pads = tuple(0 for _ in shape) + after_pads
        return GeneratedOpSpec(
            op_type,
            pad_shape(shape, pads),
            {"mode": b"constant"},
            (make_int64_initializer("pads", np.asarray(pads, dtype=np.int64)),),
        )

    if op_type == "Tile":
        repeats = draw(
            st.lists(
                st.integers(min_value=1, max_value=3),
                min_size=len(shape),
                max_size=len(shape),
            ).map(tuple)
        )
        output_shape = tuple(dim * repeat for dim, repeat in zip(shape, repeats))
        return GeneratedOpSpec(
            op_type,
            output_shape,
            {},
            (make_int64_initializer("repeats", np.asarray(repeats, dtype=np.int64)),),
        )

    if op_type == "Slice":
        axis = draw(
            st.sampled_from(tuple(i for i, dim in enumerate(shape) if dim >= 2))
        )
        start = draw(st.integers(min_value=0, max_value=shape[axis] - 1))
        end = draw(st.integers(min_value=start + 1, max_value=shape[axis]))
        return GeneratedOpSpec(
            op_type,
            slice_shape(shape, axis, start, end),
            {},
            (
                make_int64_initializer("starts", np.asarray([start], dtype=np.int64)),
                make_int64_initializer("ends", np.asarray([end], dtype=np.int64)),
                make_int64_initializer("axes", np.asarray([axis], dtype=np.int64)),
            ),
        )

    if op_type == "Softmax":
        return GeneratedOpSpec(op_type, shape, {"axis": -1})

    if op_type in {"Add", "Mul"}:
        use_scalar = draw(st.booleans())
        constant_shape = () if use_scalar else shape
        constant = draw(float_array_strategy(constant_shape))
        return GeneratedOpSpec(
            op_type,
            shape,
            {},
            (make_float_initializer("C", constant),),
        )

    if op_type == "Sub":
        constant = draw(float_array_strategy(shape))
        return GeneratedOpSpec(
            op_type,
            shape,
            {},
            (make_float_initializer("C", constant),),
        )

    if op_type == "Div":
        constant = draw(float_array_strategy(shape, NONZERO_FLOAT32_VALUES))
        return GeneratedOpSpec(
            op_type,
            shape,
            {},
            (make_float_initializer("C", constant),),
        )

    if op_type == "Pow":
        return GeneratedOpSpec(
            op_type,
            shape,
            {},
            (make_float_initializer("P", np.array(2.0, dtype=np.float32)),),
        )

    if op_type in VARIADIC_OP_TYPES:
        left = draw(float_array_strategy(shape))
        right = draw(float_array_strategy(shape))
        return GeneratedOpSpec(
            op_type,
            shape,
            {},
            (
                make_float_initializer("A", left),
                make_float_initializer("B", right),
            ),
        )

    if op_type in NATIVE_REDUCE_ALL_OP_TYPES:
        keepdims = draw(st.booleans())
        return GeneratedOpSpec(
            op_type,
            reduce_all_shape(shape, keepdims),
            {"keepdims": int(keepdims)},
        )

    if op_type == "MatMul":
        output_columns = draw(st.integers(min_value=1, max_value=3))
        rhs = draw(float_array_strategy((shape[-1], output_columns)))
        return GeneratedOpSpec(
            op_type,
            (shape[-2], output_columns),
            {},
            (make_float_initializer("B", rhs),),
        )

    if op_type == "Flatten":
        axis = draw(st.integers(min_value=0, max_value=len(shape)))
        return GeneratedOpSpec(op_type, flatten_shape(shape, axis), {"axis": axis})

    if op_type == "Reshape":
        output_shape = draw(st.sampled_from(alternate_shapes(shape)))
        shape_values: npt.NDArray[np.int64]
        if output_shape and draw(st.booleans()):
            shape_values = np.asarray(output_shape, dtype=np.int64)
            shape_values[-1] = -1
        else:
            shape_values = np.asarray(output_shape, dtype=np.int64)
        return GeneratedOpSpec(
            op_type,
            output_shape,
            {},
            (make_int64_initializer("shape", shape_values),),
        )

    if op_type == "Squeeze":
        singleton_axes = tuple(index for index, dim in enumerate(shape) if dim == 1)
        axes = draw(
            st.lists(
                st.sampled_from(singleton_axes),
                min_size=1,
                max_size=len(singleton_axes),
                unique=True,
            ).map(tuple)
        )
        return GeneratedOpSpec(
            op_type,
            squeeze_shape(shape, axes),
            {},
            (make_int64_initializer("axes", np.asarray(axes, dtype=np.int64)),),
        )

    if op_type == "Transpose":
        permutation = draw(st.sampled_from(native_transpose_permutations(len(shape))))
        return GeneratedOpSpec(
            op_type,
            tuple(shape[axis] for axis in permutation),
            {"perm": list(permutation)},
        )

    if op_type == "Split":
        axis = draw(st.integers(min_value=0, max_value=len(shape) - 1))
        return GeneratedOpSpec(op_type, shape, {"axis": axis, "num_outputs": 1})

    if op_type == "Unsqueeze":
        max_new_axes = min(2, 4 - len(shape))
        axis_count = draw(st.integers(min_value=1, max_value=max_new_axes))
        output_rank = len(shape) + axis_count
        axes = draw(
            st.lists(
                st.integers(min_value=0, max_value=output_rank - 1),
                min_size=axis_count,
                max_size=axis_count,
                unique=True,
            ).map(tuple)
        )
        return GeneratedOpSpec(
            op_type,
            unsqueeze_shape(shape, axes),
            {},
            (make_int64_initializer("axes", np.asarray(axes, dtype=np.int64)),),
        )

    raise AssertionError(f'No generator for native op "{op_type}"')


@st.composite
def reduce_axes_strategy(
    draw: st.DrawFn,
    rank: int,
    allow_empty: bool,
) -> typing.Tuple[int, ...]:
    min_size = 0 if allow_empty else 1
    axis_count = draw(st.integers(min_value=min_size, max_value=rank))
    if axis_count == 0:
        return ()

    normalized_axes = draw(
        st.lists(
            st.integers(min_value=0, max_value=rank - 1),
            min_size=axis_count,
            max_size=axis_count,
            unique=True,
        )
    )
    axes = []
    for axis in normalized_axes:
        if draw(st.booleans()):
            axes.append(axis - rank)
        else:
            axes.append(axis)
    return tuple(axes)


@st.composite
def reduce_sum_variant_cases(draw: st.DrawFn) -> GeneratedGraphCase:
    shape = draw(st.sampled_from(GRAPH_SHAPES))
    input_array = draw(float_array_strategy(shape))
    keepdims = draw(st.booleans())
    variant = draw(st.sampled_from(["implicit_all", "attr_axes", "input_axes"]))

    if variant == "implicit_all":
        op = GeneratedOpSpec(
            "ReduceSum",
            reduce_all_shape(shape, keepdims),
            {"keepdims": int(keepdims)},
        )
        return build_model_case(
            "hypothesis_reducesum_implicit_all",
            shape,
            input_array,
            [op],
            fallback_start=None,
        )

    if variant == "attr_axes":
        axes = draw(reduce_axes_strategy(len(shape), allow_empty=False))
        reduce_all = set(normalize_axes(axes, len(shape))) == set(range(len(shape)))
        expected_fallback_indices = () if reduce_all else (0,)
        op = GeneratedOpSpec(
            "ReduceSum",
            reduce_axes_shape(shape, axes, keepdims),
            {"axes": list(axes), "keepdims": int(keepdims)},
        )
        return build_model_case(
            "hypothesis_reducesum_attr_axes",
            shape,
            input_array,
            [op],
            fallback_start=0 if expected_fallback_indices else None,
            opset_version=11,
            expected_fallback_indices=expected_fallback_indices,
        )

    axes = draw(reduce_axes_strategy(len(shape), allow_empty=True))
    noop = draw(st.booleans()) if not axes else False
    reduce_all = not noop and (
        not axes or set(normalize_axes(axes, len(shape))) == set(range(len(shape)))
    )
    expected_fallback_indices = () if reduce_all else (0,)
    op = GeneratedOpSpec(
        "ReduceSum",
        reduce_axes_shape(shape, axes, keepdims, noop=noop),
        {"keepdims": int(keepdims), "noop_with_empty_axes": int(noop)},
        (make_int64_initializer("axes", np.asarray(axes, dtype=np.int64)),),
    )
    return build_model_case(
        "hypothesis_reducesum_input_axes",
        shape,
        input_array,
        [op],
        fallback_start=0 if expected_fallback_indices else None,
        expected_fallback_indices=expected_fallback_indices,
    )


@st.composite
def squeeze_axes_strategy(
    draw: st.DrawFn,
    shape: typing.Tuple[int, ...],
) -> typing.Tuple[int, ...]:
    singleton_axes = tuple(index for index, dim in enumerate(shape) if dim == 1)
    axis_count = draw(st.integers(min_value=1, max_value=len(singleton_axes)))
    normalized_axes = draw(
        st.lists(
            st.sampled_from(singleton_axes),
            min_size=axis_count,
            max_size=axis_count,
            unique=True,
        )
    )
    shuffled_axes = tuple(draw(st.permutations(tuple(normalized_axes))))
    return draw(st.sampled_from(axes_with_negative_forms(shuffled_axes, len(shape))))


@st.composite
def unsqueeze_axes_strategy(
    draw: st.DrawFn,
    input_rank: int,
) -> typing.Tuple[int, ...]:
    axis_count = draw(
        st.integers(
            min_value=1,
            max_value=ViewTransformSemantics.GGML_MAX_DIMS - input_rank,
        )
    )
    output_rank = input_rank + axis_count
    normalized_axes = draw(
        st.lists(
            st.integers(min_value=0, max_value=output_rank - 1),
            min_size=axis_count,
            max_size=axis_count,
            unique=True,
        )
    )
    shuffled_axes = tuple(draw(st.permutations(tuple(normalized_axes))))
    return draw(st.sampled_from(axes_with_negative_forms(shuffled_axes, output_rank)))


@st.composite
def shape_view_variant_cases(draw: st.DrawFn) -> GeneratedGraphCase:
    variant = draw(
        st.sampled_from(
            [
                "squeeze_no_axes",
                "squeeze_attr_axes",
                "squeeze_input_axes",
                "unsqueeze_attr_axes",
                "unsqueeze_input_axes",
            ]
        )
    )

    if variant == "squeeze_no_axes":
        shape = draw(st.sampled_from(GRAPH_SHAPES))
        input_array = draw(float_array_strategy(shape))
        op = GeneratedOpSpec("Squeeze", squeeze_shape(shape, None), {})
        return build_model_case(
            "hypothesis_shape_view_squeeze_no_axes",
            shape,
            input_array,
            [op],
            fallback_start=None,
        )

    if variant in {"squeeze_attr_axes", "squeeze_input_axes"}:
        shape = draw(
            st.sampled_from(tuple(shape for shape in GRAPH_SHAPES if 1 in shape))
        )
        input_array = draw(float_array_strategy(shape))
        axes = draw(squeeze_axes_strategy(shape))
        op = GeneratedOpSpec(
            "Squeeze",
            squeeze_shape(shape, axes),
            {"axes": list(axes)} if variant == "squeeze_attr_axes" else {},
            ()
            if variant == "squeeze_attr_axes"
            else (make_int64_initializer("axes", np.asarray(axes, dtype=np.int64)),),
        )
        return build_model_case(
            "hypothesis_shape_view_%s" % variant,
            shape,
            input_array,
            [op],
            fallback_start=None,
            opset_version=11 if variant == "squeeze_attr_axes" else 18,
        )

    shape = draw(
        st.sampled_from(
            tuple(
                shape
                for shape in SHAPES
                if len(shape) < ViewTransformSemantics.GGML_MAX_DIMS
            )
        )
    )
    input_array = draw(float_array_strategy(shape))
    axes = draw(unsqueeze_axes_strategy(len(shape)))
    op = GeneratedOpSpec(
        "Unsqueeze",
        unsqueeze_shape(shape, axes),
        {"axes": list(axes)} if variant == "unsqueeze_attr_axes" else {},
        ()
        if variant == "unsqueeze_attr_axes"
        else (make_int64_initializer("axes", np.asarray(axes, dtype=np.int64)),),
    )
    return build_model_case(
        "hypothesis_shape_view_%s" % variant,
        shape,
        input_array,
        [op],
        fallback_start=None,
        opset_version=11 if variant == "unsqueeze_attr_axes" else 18,
    )


@st.composite
def reshape_variant_cases(draw: st.DrawFn) -> GeneratedGraphCase:
    variant = draw(
        st.sampled_from(
            [
                "plain",
                "infer_dim",
                "zero_copy",
                "zero_copy_infer_dim",
                "scalar_output",
                "allowzero",
            ]
        )
    )

    if variant == "allowzero":
        shape = draw(st.sampled_from(ZERO_RESHAPE_SHAPES))
        output_shape = draw(st.sampled_from(ZERO_RESHAPE_SHAPES))
        shape_values = np.asarray(output_shape, dtype=np.int64)
        allowzero = True
    elif variant == "scalar_output":
        shape = draw(
            st.sampled_from(
                tuple(shape for shape in SHAPES if shape_element_count(shape) == 1)
            )
        )
        shape_values = np.asarray([], dtype=np.int64)
        output_shape = ()
        allowzero = False
    elif variant == "infer_dim":
        shape = draw(
            st.sampled_from(
                tuple(shape for shape in SHAPES if shape_element_count(shape) > 0)
            )
        )
        output_shape = draw(
            st.sampled_from(
                tuple(candidate for candidate in alternate_shapes(shape) if candidate)
            )
        )
        infer_axis = draw(st.integers(min_value=0, max_value=len(output_shape) - 1))
        shape_values = np.asarray(output_shape, dtype=np.int64)
        shape_values[infer_axis] = -1
        allowzero = False
    elif variant == "zero_copy":
        shape = draw(st.sampled_from(tuple(shape for shape in SHAPES if shape)))
        axis_count = draw(st.integers(min_value=1, max_value=len(shape)))
        zero_axes = draw(
            st.lists(
                st.integers(min_value=0, max_value=len(shape) - 1),
                min_size=axis_count,
                max_size=axis_count,
                unique=True,
            )
        )
        shape_values = np.asarray(shape, dtype=np.int64)
        for axis in zero_axes:
            shape_values[axis] = 0
        output_shape = reshape_output_shape(shape, shape_values, allowzero=False)
        allowzero = False
    elif variant == "zero_copy_infer_dim":
        shape = draw(
            st.sampled_from(tuple(shape for shape in SHAPES if len(shape) >= 2))
        )
        infer_axis = draw(st.integers(min_value=0, max_value=len(shape) - 1))
        zero_candidates = tuple(
            axis for axis in range(len(shape)) if axis != infer_axis
        )
        axis_count = draw(st.integers(min_value=1, max_value=len(zero_candidates)))
        zero_axes = draw(
            st.lists(
                st.sampled_from(zero_candidates),
                min_size=axis_count,
                max_size=axis_count,
                unique=True,
            )
        )
        shape_values = np.asarray(shape, dtype=np.int64)
        for axis in zero_axes:
            shape_values[axis] = 0
        shape_values[infer_axis] = -1
        output_shape = reshape_output_shape(shape, shape_values, allowzero=False)
        allowzero = False
    else:
        shape = draw(st.sampled_from(SHAPES))
        output_shape = draw(st.sampled_from(alternate_shapes(shape)))
        shape_values = np.asarray(output_shape, dtype=np.int64)
        allowzero = False

    input_array = draw(float_array_strategy(shape))
    op = GeneratedOpSpec(
        "Reshape",
        output_shape,
        {"allowzero": 1} if allowzero else {},
        (make_int64_initializer("shape", shape_values),),
    )
    return build_model_case(
        "hypothesis_reshape_%s" % variant,
        shape,
        input_array,
        [op],
        fallback_start=None,
    )


@st.composite
def flatten_variant_cases(draw: st.DrawFn) -> GeneratedGraphCase:
    shape = draw(st.sampled_from(SHAPES))
    input_array = draw(float_array_strategy(shape))
    axis = draw(st.integers(min_value=-len(shape), max_value=len(shape)))
    op = GeneratedOpSpec(
        "Flatten",
        flatten_shape(shape, axis),
        {"axis": axis},
    )
    return build_model_case(
        "hypothesis_flatten_axis",
        shape,
        input_array,
        [op],
        fallback_start=None,
    )


@st.composite
def transpose_variant_cases(draw: st.DrawFn) -> GeneratedGraphCase:
    shape = draw(
        st.sampled_from(
            tuple(
                shape
                for shape in GRAPH_SHAPES
                if native_transpose_permutations(len(shape))
            )
        )
    )
    input_array = draw(float_array_strategy(shape))
    variant = draw(st.sampled_from(["default", "identity", "reverse", "permutation"]))
    if variant == "default":
        permutation = tuple(reversed(range(len(shape))))
        attrs = {}
    elif variant == "identity":
        permutation = tuple(range(len(shape)))
        attrs = {"perm": list(permutation)}
    elif variant == "reverse":
        permutation = tuple(reversed(range(len(shape))))
        attrs = {"perm": list(permutation)}
    else:
        permutation = draw(st.sampled_from(native_transpose_permutations(len(shape))))
        attrs = {"perm": list(permutation)}

    op = GeneratedOpSpec(
        "Transpose",
        transpose_shape(shape, permutation),
        attrs,
    )
    return build_model_case(
        "hypothesis_transpose_%s" % variant,
        shape,
        input_array,
        [op],
        fallback_start=None,
    )


@st.composite
def arithmetic_broadcast_cases(draw: st.DrawFn) -> GeneratedGraphCase:
    op_type = draw(st.sampled_from(sorted(ARITHMETIC_OP_TYPES)))
    variant = draw(
        st.sampled_from(
            [
                "same_shape",
                "scalar_rhs",
                "scalar_lhs",
                "broadcast",
            ]
        )
    )

    if variant == "same_shape":
        left_shape = right_shape = draw(st.sampled_from(SHAPES))
        expected_fallback_indices: typing.Tuple[int, ...] = ()
    elif variant == "scalar_rhs":
        left_shape = draw(st.sampled_from(GRAPH_SHAPES))
        right_shape = ()
        expected_fallback_indices = () if op_type in {"Add", "Mul"} else (0,)
    elif variant == "scalar_lhs":
        left_shape = ()
        right_shape = draw(st.sampled_from(GRAPH_SHAPES))
        expected_fallback_indices = () if op_type in {"Add", "Mul"} else (0,)
    else:
        left_shape, right_shape = draw(st.sampled_from(BROADCAST_SHAPE_PAIRS))
        expected_fallback_indices = (0,)

    right_elements = NONZERO_FLOAT32_VALUES if op_type == "Div" else FLOAT32_VALUES
    left_array = draw(float_array_strategy(left_shape))
    right_array = draw(float_array_strategy(right_shape, right_elements))
    output_shape = broadcast_shape(left_shape, right_shape)

    left_spec = TensorSpec("A", left_shape, np.dtype(np.float32))
    right_spec = TensorSpec("B", right_shape, np.dtype(np.float32))
    op_case = make_direct_op_case(
        0,
        op_type,
        (left_spec, right_spec),
        output_shape,
        {},
    )

    return build_direct_graph_case(
        "hypothesis_arithmetic_%s_%s" % (op_type.lower(), variant),
        (left_spec, right_spec),
        (op_case,),
        op_case.output_specs,
        {
            "A": np.asarray(left_array, dtype=np.float32),
            "B": np.asarray(right_array, dtype=np.float32),
        },
        fallback_start=0 if expected_fallback_indices else None,
        expected_fallback_indices=expected_fallback_indices,
    )


@st.composite
def split_values_strategy(
    draw: st.DrawFn,
    total: int,
    output_count: int,
) -> typing.Tuple[int, ...]:
    cut_points = draw(
        st.lists(
            st.integers(min_value=1, max_value=total - 1),
            min_size=output_count - 1,
            max_size=output_count - 1,
            unique=True,
        ).map(sorted)
    )
    points = (0, *cut_points, total)
    return tuple(points[index + 1] - points[index] for index in range(output_count))


@st.composite
def split_variant_cases(draw: st.DrawFn) -> GeneratedGraphCase:
    variant = draw(st.sampled_from(["num_outputs", "split_input"]))
    shape = draw(st.sampled_from(SPLIT_INPUT_SHAPES))
    axis = draw(
        st.sampled_from(tuple(index for index, dim in enumerate(shape) if dim >= 2))
    )
    raw_axis = axis - len(shape) if draw(st.booleans()) else axis
    axis_size = shape[axis]

    if variant == "num_outputs":
        output_count = draw(
            st.sampled_from(
                tuple(
                    count
                    for count in range(2, min(4, axis_size) + 1)
                    if axis_size % count == 0
                )
            )
        )
        split_values = tuple(axis_size // output_count for _ in range(output_count))
        attrs = {"axis": raw_axis, "num_outputs": output_count}
        initializers: typing.Tuple[InitializerSpec, ...] = ()
        inputs = ("X",)
    else:
        output_count = draw(st.integers(min_value=2, max_value=min(4, axis_size)))
        split_values = draw(split_values_strategy(axis_size, output_count))
        split_initializer = make_int64_initializer(
            "split", np.asarray(split_values, dtype=np.int64)
        )
        attrs = {"axis": raw_axis}
        initializers = (split_initializer,)
        inputs = ("X", "split")

    output_shapes = split_output_shapes(shape, raw_axis, split_values)
    output_specs = tuple(
        TensorSpec("Y%d" % index, output_shape, np.dtype(np.float32))
        for index, output_shape in enumerate(output_shapes)
    )
    op_case = OpCase(
        "Split",
        inputs,
        tuple(output_spec.name for output_spec in output_specs),
        attrs,
        initializers,
        output_specs,
    )
    input_array = draw(float_array_strategy(shape))

    return build_direct_graph_case(
        "hypothesis_split_%s" % variant,
        (TensorSpec("X", shape, np.dtype(np.float32)),),
        (op_case,),
        output_specs,
        {"X": np.asarray(input_array, dtype=np.float32)},
        fallback_start=None,
        expected_fallback_indices=(),
    )


def available_native_ops_for_spec(
    input_spec: TensorSpec,
    allow_shape_views: bool,
    allow_layout_views: bool,
    allow_domain_sensitive_ops: bool,
) -> typing.Tuple[str, ...]:
    shape = input_spec.shape
    if input_spec.dtype != np.dtype(np.float32):
        return ()
    ops = set(UNARY_OP_TYPES)
    ops.difference_update({"Log", "Sqrt"})
    ops.update(NATIVE_ACTIVATION_OP_TYPES)
    if allow_domain_sensitive_ops:
        if input_spec.domain == ValueDomain.POSITIVE:
            ops.add("Log")
        if input_spec.domain in {ValueDomain.NONNEGATIVE, ValueDomain.POSITIVE}:
            ops.add("Sqrt")
    ops.update(BINARY_OP_TYPES)
    ops.update(VARIADIC_OP_TYPES)
    ops.add("Pow")
    ops.add("Clip")
    if shape:
        ops.update(NATIVE_REDUCE_ALL_OP_TYPES)
        ops.add("Softmax")
    if shape and len(shape) <= 4:
        ops.add("Concat")
        ops.add("Expand")
        ops.add("Pad")
        ops.add("Tile")
    if len(shape) == 2:
        ops.add("Gather")
        ops.add("MatMul")
    if len(shape) == 4:
        ops.add("AveragePool")
        ops.add("Conv")
        ops.add("GlobalAveragePool")
        ops.add("GlobalMaxPool")
        ops.add("MaxPool")
    if allow_shape_views:
        ops.add("Reshape")
        if shape:
            ops.add("Flatten")
        if 1 in shape:
            ops.add("Squeeze")
        if len(shape) <= 3:
            ops.add("Unsqueeze")
    if allow_layout_views and native_transpose_permutations(len(shape)):
        ops.add("Transpose")
    if allow_layout_views and shape and len(shape) <= 4:
        ops.add("Split")
        if any(dim >= 2 for dim in shape):
            ops.add("Slice")
    return tuple(sorted(ops))


@st.composite
def native_op_strategy(
    draw: st.DrawFn,
    input_spec: TensorSpec,
    allow_shape_views: bool = True,
    allow_layout_views: bool = True,
    allow_domain_sensitive_ops: bool = True,
) -> GeneratedOpSpec:
    op_type = draw(
        st.sampled_from(
            available_native_ops_for_spec(
                input_spec,
                allow_shape_views,
                allow_layout_views,
                allow_domain_sensitive_ops,
            )
        )
    )
    return draw(generated_op_strategy(op_type, input_spec))


def non_layout_values(
    values: typing.Sequence[TensorSpec],
) -> typing.Tuple[TensorSpec, ...]:
    return tuple(value for value in values if value.layout != LayoutKind.LAYOUT_VIEW)


def same_shape_value_pairs(
    values: typing.Sequence[TensorSpec],
) -> typing.Tuple[typing.Tuple[TensorSpec, TensorSpec], ...]:
    eligible_values = non_layout_values(values)
    return tuple(
        (left, right)
        for index, left in enumerate(eligible_values)
        for right in eligible_values[index + 1 :]
        if left.shape == right.shape
    )


def make_direct_op_case(
    index: int,
    op_type: str,
    input_specs: typing.Sequence[TensorSpec],
    output_shape: typing.Tuple[int, ...],
    attrs: typing.Dict[str, typing.Any],
    initializers: typing.Sequence[InitializerSpec] = (),
) -> OpCase:
    output_name = "T%d" % index
    renamed_initializers = tuple(
        InitializerSpec(
            "%s_%d" % (initializer.name, index),
            initializer.data_type,
            initializer.value,
        )
        for initializer in initializers
    )
    output_spec = TensorSpec(
        output_name,
        output_shape,
        input_specs[0].dtype,
        op_case_output_domain(op_type, input_specs),
        op_case_output_layout(op_type),
    )
    return OpCase(
        op_type,
        (
            *(input_spec.name for input_spec in input_specs),
            *(initializer.name for initializer in renamed_initializers),
        ),
        (output_name,),
        attrs,
        renamed_initializers,
        (output_spec,),
    )


@st.composite
def direct_shape_preserving_unary_op_case_strategy(
    draw: st.DrawFn,
    index: int,
    input_spec: TensorSpec,
) -> OpCase:
    op_type = draw(st.sampled_from(sorted(SHAPE_PRESERVING_UNARY_OP_TYPES)))
    attrs = {"alpha": 0.01} if op_type == "LeakyRelu" else {}
    return make_direct_op_case(index, op_type, (input_spec,), input_spec.shape, attrs)


@st.composite
def direct_join_op_case_strategy(
    draw: st.DrawFn,
    index: int,
    left_spec: TensorSpec,
    right_spec: TensorSpec,
) -> OpCase:
    op_type = draw(st.sampled_from(sorted(DAG_JOIN_OP_TYPES)))
    return make_direct_op_case(
        index,
        op_type,
        (left_spec, right_spec),
        left_spec.shape,
        {},
    )


@st.composite
def direct_native_op_case_strategy(
    draw: st.DrawFn,
    index: int,
    values: typing.Sequence[TensorSpec],
    allow_layout_views: bool,
) -> OpCase:
    input_spec = draw(st.sampled_from(non_layout_values(values)))
    op = draw(
        native_op_strategy(
            input_spec,
            allow_shape_views=True,
            allow_layout_views=allow_layout_views,
            allow_domain_sensitive_ops=True,
        )
    )
    return bind_op_case(
        op, index, input_spec, is_final=False, output_name="T%d" % index
    )


@st.composite
def native_single_op_cases(draw: st.DrawFn) -> GeneratedGraphCase:
    op_type = draw(st.sampled_from(sorted(NATIVE_GENERATOR_OP_TYPES)))
    shape = draw(st.sampled_from(op_input_shapes(op_type)))
    elements = (
        POSITIVE_FLOAT32_VALUES if op_needs_positive_input(op_type) else FLOAT32_VALUES
    )
    input_array = draw(float_array_strategy(shape, elements))
    input_domain = (
        ValueDomain.POSITIVE
        if op_needs_positive_input(op_type)
        else ValueDomain.ANY_FLOAT
    )
    input_spec = TensorSpec(
        "X",
        shape,
        np.dtype(np.float32),
        input_domain,
        LayoutKind.CONTIGUOUS,
    )
    op = draw(generated_op_strategy(op_type, input_spec))
    return build_model_case(
        "hypothesis_single_%s" % op_type.lower(),
        shape,
        input_array,
        [op],
        fallback_start=None,
        input_domain=input_domain,
    )


@st.composite
def native_dag_graph_cases(draw: st.DrawFn) -> GeneratedGraphCase:
    shape = draw(st.sampled_from(GRAPH_SHAPES))
    input_specs = (
        TensorSpec("X0", shape, np.dtype(np.float32)),
        TensorSpec("X1", shape, np.dtype(np.float32)),
    )
    input_values = {
        "X0": draw(float_array_strategy(shape)),
        "X1": draw(float_array_strategy(shape)),
    }
    node_count = draw(st.integers(min_value=3, max_value=8))
    ops = []
    values = list(input_specs)

    first_op = draw(direct_shape_preserving_unary_op_case_strategy(0, input_specs[0]))
    ops.append(first_op)
    values.append(first_op.output_specs[0])

    second_op = draw(direct_shape_preserving_unary_op_case_strategy(1, input_specs[1]))
    ops.append(second_op)
    values.append(second_op.output_specs[0])

    join_op = draw(
        direct_join_op_case_strategy(
            2,
            first_op.output_specs[0],
            second_op.output_specs[0],
        )
    )
    ops.append(join_op)
    values.append(join_op.output_specs[0])
    branch_count = 1

    for index in range(3, node_count):
        pairs = same_shape_value_pairs(values)
        use_join = bool(pairs) and draw(st.booleans())
        if use_join:
            left_spec, right_spec = draw(st.sampled_from(pairs))
            op_case = draw(direct_join_op_case_strategy(index, left_spec, right_spec))
            branch_count += 1
        else:
            op_case = draw(
                direct_native_op_case_strategy(
                    index,
                    values,
                    allow_layout_views=index == node_count - 1,
                )
            )
        ops.append(op_case)
        values.append(op_case.output_specs[0])

    op_names = "_".join(op.op_type.lower() for op in ops)
    return build_direct_graph_case(
        "hypothesis_native_dag_%s" % op_names,
        input_specs,
        ops,
        (values[-1],),
        input_values,
        fallback_start=None,
        branch_count=branch_count,
    )


@st.composite
def fallback_suffix_op_strategy(
    draw: st.DrawFn, shape: typing.Tuple[int, ...]
) -> GeneratedOpSpec:
    op_type = draw(st.sampled_from(["Add", "Sub", "Mul", "Relu", "Tanh", "Abs"]))
    if op_type in {"Relu", "Tanh", "Abs"}:
        return GeneratedOpSpec(op_type, shape, {})
    constant = draw(float_array_strategy(shape))
    return GeneratedOpSpec(
        op_type,
        shape,
        {},
        (make_float_initializer("F", constant),),
    )


@st.composite
def native_graph_cases(draw: st.DrawFn) -> GeneratedGraphCase:
    shape = draw(st.sampled_from(GRAPH_SHAPES))
    input_array = draw(float_array_strategy(shape))
    node_count = draw(st.integers(min_value=2, max_value=8))
    ops = []
    current_spec = TensorSpec(
        "X",
        shape,
        np.dtype(np.float32),
        ValueDomain.ANY_FLOAT,
        LayoutKind.CONTIGUOUS,
    )
    allow_shape_views = True
    for index in range(node_count):
        op = draw(
            native_op_strategy(
                current_spec,
                allow_shape_views,
                allow_layout_views=index == node_count - 1,
                allow_domain_sensitive_ops=True,
            )
        )
        ops.append(op)
        current_spec = TensorSpec(
            "T%d" % index,
            op.output_shape,
            op.output_dtype,
            output_domain(op, current_spec),
            output_layout(op),
        )
        if op.is_layout_view:
            allow_shape_views = False
    op_names = "_".join(op.op_type.lower() for op in ops)
    return build_model_case(
        "hypothesis_native_%s" % op_names,
        shape,
        input_array,
        ops,
        fallback_start=None,
    )


@st.composite
def mixed_graph_cases(draw: st.DrawFn) -> GeneratedGraphCase:
    shape = draw(st.sampled_from(GRAPH_SHAPES))
    input_array = draw(float_array_strategy(shape))
    prefix_count = draw(st.integers(min_value=1, max_value=5))
    suffix_count = draw(st.integers(min_value=1, max_value=3))

    ops = []
    current_spec = TensorSpec(
        "X",
        shape,
        np.dtype(np.float32),
        ValueDomain.ANY_FLOAT,
        LayoutKind.CONTIGUOUS,
    )
    allow_shape_views = True
    for _ in range(prefix_count):
        op = draw(
            native_op_strategy(
                current_spec,
                allow_shape_views,
                allow_layout_views=False,
                allow_domain_sensitive_ops=True,
            )
        )
        ops.append(op)
        current_spec = TensorSpec(
            "T%d" % (len(ops) - 1),
            op.output_shape,
            op.output_dtype,
            output_domain(op, current_spec),
            output_layout(op),
        )
        if op.is_layout_view:
            allow_shape_views = False

    fallback_op = draw(
        st.sampled_from(
            [
                GeneratedOpSpec("Celu", current_spec.shape, {"alpha": 1.0}),
                GeneratedOpSpec("Elu", current_spec.shape, {"alpha": 0.5}),
            ]
        )
    )
    ops.append(fallback_op)
    current_spec = TensorSpec(
        "T%d" % (len(ops) - 1),
        fallback_op.output_shape,
        fallback_op.output_dtype,
        ValueDomain.ANY_FLOAT,
        LayoutKind.CONTIGUOUS,
    )

    for _ in range(suffix_count):
        op = draw(fallback_suffix_op_strategy(current_spec.shape))
        ops.append(op)
        current_spec = TensorSpec(
            "T%d" % (len(ops) - 1),
            op.output_shape,
            op.output_dtype,
            ValueDomain.ANY_FLOAT,
            LayoutKind.CONTIGUOUS,
        )

    op_names = "_".join(op.op_type.lower() for op in ops)
    return build_model_case(
        "hypothesis_mixed_%s" % op_names,
        shape,
        input_array,
        ops,
        fallback_start=prefix_count,
    )


def run_onnxruntime(
    case: GeneratedGraphCase,
) -> typing.Sequence[npt.NDArray[typing.Any]]:
    return InferenceSession(model_bytes(case.model)).run(None, case.inputs)


def run_ggml(case: GeneratedGraphCase, fallback_policy: str = "compat"):
    ggml_model = GgmlRuntimeBackend.prepare(case.model, fallback_policy=fallback_policy)
    return ggml_model, ggml_model.run(case.inputs)


def assert_outputs_match(
    case: GeneratedGraphCase,
    expected_outputs: typing.Sequence[npt.NDArray[typing.Any]],
    actual_outputs: typing.Sequence[npt.NDArray[typing.Any]],
) -> None:
    assert len(actual_outputs) == len(expected_outputs)
    for actual, expected in zip(actual_outputs, expected_outputs):
        np.testing.assert_allclose(
            actual,
            expected,
            rtol=case.rtol,
            atol=case.atol,
            equal_nan=True,
        )


def ggml_graph_percentage(report: typing.Any) -> float:
    return (
        report.native_percentage
        + report.operator_class_percentage("decomposed")
        + report.operator_class_percentage("constant_fold")
    )


def assert_native_graph_health(
    case: GeneratedGraphCase, ggml_model: typing.Any
) -> None:
    report = ggml_model.coverage_report
    assert ggml_model.execution_plan.is_supported
    assert report.total_nodes == case.node_count
    assert report.fallback_nodes == (), report.summary()
    assert report.operator_class_count("unsupported") == 0
    assert np.isclose(ggml_graph_percentage(report), 1.0), report.summary()
    assert ggml_model.last_numpy_fallback_island_executions == ()


def assert_mixed_graph_health(case: GeneratedGraphCase, ggml_model: typing.Any) -> None:
    assert case.fallback_start is not None
    report = ggml_model.coverage_report
    expected_fallback_indices = tuple(range(case.fallback_start, case.node_count))
    actual_fallback_indices = tuple(node.index for node in report.fallback_nodes)
    prefix_nodes = ggml_model.execution_plan.nodes[: case.fallback_start]

    assert report.total_nodes == case.node_count
    assert report.operator_class_count("unsupported") == 0
    assert actual_fallback_indices == expected_fallback_indices, report.summary()
    assert all(not node.requires_numpy_fallback for node in prefix_nodes)
    assert len(report.fallback_islands) == 1
    assert report.fallback_islands[0].node_indices == expected_fallback_indices
    assert ggml_graph_percentage(report) > 0.0, report.summary()
    assert report.fallback_percentage > 0.0, report.summary()
    assert tuple(
        island.node_indices
        for island in ggml_model.last_numpy_fallback_island_executions
    ) == (expected_fallback_indices,)


def assert_reduce_sum_variant_health(
    case: GeneratedGraphCase, ggml_model: typing.Any
) -> None:
    if not case.expected_fallback_indices:
        assert_native_graph_health(case, ggml_model)
        return

    report = ggml_model.coverage_report
    strict_plan = GgmlRuntimeBackend.analyze(case.model, fallback_policy="strict")
    actual_fallback_indices = tuple(node.index for node in report.fallback_nodes)

    assert not strict_plan.is_supported
    assert report.total_nodes == case.node_count
    assert report.operator_class_count("unsupported") == 0
    assert actual_fallback_indices == case.expected_fallback_indices, report.summary()
    assert tuple(
        island.node_indices
        for island in ggml_model.last_numpy_fallback_island_executions
    ) == (case.expected_fallback_indices,)


def assert_expected_execution_health(
    case: GeneratedGraphCase, ggml_model: typing.Any
) -> None:
    if not case.expected_fallback_indices:
        assert_native_graph_health(case, ggml_model)
        return

    report = ggml_model.coverage_report
    strict_plan = GgmlRuntimeBackend.analyze(case.model, fallback_policy="strict")
    actual_fallback_indices = tuple(node.index for node in report.fallback_nodes)

    assert not strict_plan.is_supported
    assert report.total_nodes == case.node_count
    assert report.operator_class_count("unsupported") == 0
    assert actual_fallback_indices == case.expected_fallback_indices, report.summary()
    assert tuple(
        island.node_indices
        for island in ggml_model.last_numpy_fallback_island_executions
    ) == (case.expected_fallback_indices,)


def assert_split_execution_health(
    case: GeneratedGraphCase, ggml_model: typing.Any
) -> None:
    if not case.expected_fallback_indices:
        assert_native_graph_health(case, ggml_model)
        assert ggml_model.coverage_report.operator_class_count("native_view") == 1
        return

    report = ggml_model.coverage_report
    strict_plan = GgmlRuntimeBackend.analyze(case.model, fallback_policy="strict")
    actual_fallback_indices = tuple(node.index for node in report.fallback_nodes)

    assert not strict_plan.is_supported
    assert report.total_nodes == case.node_count
    assert report.operator_class_count("unsupported") == 0
    assert actual_fallback_indices == case.expected_fallback_indices, report.summary()


def case_axes_values(
    case: GeneratedGraphCase,
) -> typing.Optional[typing.Tuple[int, ...]]:
    op = case.ir.ops[0]
    axes = op.attrs.get("axes")
    if axes is not None:
        return tuple(int(axis) for axis in axes)
    for initializer in op.initializers:
        if initializer.name.startswith("axes"):
            return tuple(int(axis) for axis in initializer.value.flatten())
    return None


def case_initializer_values(
    case: GeneratedGraphCase, prefix: str
) -> typing.Optional[typing.Tuple[int, ...]]:
    op = case.ir.ops[0]
    for initializer in op.initializers:
        if initializer.name.startswith(prefix):
            return tuple(int(value) for value in initializer.value.flatten())
    return None


def test_hypothesis_native_generator_covers_backend_native_ops() -> None:
    assert NATIVE_GENERATOR_OP_TYPES == backend_native_op_types()
    assert {
        case.model.graph.node[0].op_type for case in canonical_native_single_op_cases()
    } == NATIVE_GENERATOR_OP_TYPES


@pytest.mark.parametrize(
    "case",
    canonical_native_single_op_cases(),
    ids=lambda case: case.description,
)
def test_canonical_native_single_op_cases_match_onnxruntime_without_fallback(
    case: GeneratedGraphCase,
) -> None:
    expected_outputs = run_onnxruntime(case)
    ggml_model, actual_outputs = run_ggml(case, fallback_policy="strict")

    assert_outputs_match(case, expected_outputs, actual_outputs)
    assert_native_graph_health(case, ggml_model)


@given(native_single_op_cases())
@settings(
    max_examples=125,
    deadline=None,
    database=None,
    derandomize=True,
    suppress_health_check=[HealthCheck.too_slow],
)
def test_hypothesis_native_single_op_cases_match_onnxruntime_without_fallback(
    case: GeneratedGraphCase,
) -> None:
    expected_outputs = run_onnxruntime(case)
    ggml_model, actual_outputs = run_ggml(case, fallback_policy="strict")

    assert_outputs_match(case, expected_outputs, actual_outputs)
    assert_native_graph_health(case, ggml_model)


@given(reduce_sum_variant_cases())
@settings(
    max_examples=80,
    deadline=None,
    database=None,
    derandomize=True,
    suppress_health_check=[HealthCheck.too_slow],
)
def test_hypothesis_reducesum_variants_match_onnxruntime(
    case: GeneratedGraphCase,
) -> None:
    event("ReduceSum fallback indices: %s" % (case.expected_fallback_indices,))
    event("ReduceSum opset: %d" % case.ir.opset_version)

    expected_outputs = run_onnxruntime(case)
    ggml_model, actual_outputs = run_ggml(case)

    assert_outputs_match(case, expected_outputs, actual_outputs)
    assert_reduce_sum_variant_health(case, ggml_model)


@given(shape_view_variant_cases())
@settings(
    max_examples=100,
    deadline=None,
    database=None,
    derandomize=True,
    suppress_health_check=[HealthCheck.too_slow],
)
def test_hypothesis_shape_view_variants_match_onnxruntime_without_fallback(
    case: GeneratedGraphCase,
) -> None:
    event("shape view case: %s" % case.description)
    event("shape view opset: %d" % case.ir.opset_version)
    axes = case_axes_values(case)
    event("shape view axes count: %s" % ("none" if axes is None else len(axes)))
    if axes is not None:
        event("shape view negative axes: %s" % any(axis < 0 for axis in axes))
        event("shape view raw axes sorted: %s" % (tuple(sorted(axes)) == axes))

    expected_outputs = run_onnxruntime(case)
    ggml_model, actual_outputs = run_ggml(case, fallback_policy="strict")

    assert len(actual_outputs) == len(expected_outputs)
    for actual, expected in zip(actual_outputs, expected_outputs):
        assert actual.shape == expected.shape
    assert_outputs_match(case, expected_outputs, actual_outputs)
    assert_native_graph_health(case, ggml_model)


@given(reshape_variant_cases())
@settings(
    max_examples=100,
    deadline=None,
    database=None,
    derandomize=True,
    suppress_health_check=[HealthCheck.too_slow],
)
def test_hypothesis_reshape_variants_match_onnxruntime_without_fallback(
    case: GeneratedGraphCase,
) -> None:
    shape_values = case_initializer_values(case, "shape")
    input_shape = case.ir.inputs[0].shape
    event("reshape case: %s" % case.description)
    event(
        "reshape input has zero elements: %s" % (shape_element_count(input_shape) == 0)
    )
    event("reshape allowzero: %s" % bool(case.ir.ops[0].attrs.get("allowzero", 0)))
    event(
        "reshape shape rank: %s"
        % ("none" if shape_values is None else len(shape_values))
    )
    if shape_values is not None:
        event(
            "reshape has inferred dim: %s" % any(value == -1 for value in shape_values)
        )
        event("reshape has zero dim: %s" % any(value == 0 for value in shape_values))

    expected_outputs = run_onnxruntime(case)
    ggml_model, actual_outputs = run_ggml(case, fallback_policy="strict")

    assert len(actual_outputs) == len(expected_outputs)
    for actual, expected in zip(actual_outputs, expected_outputs):
        assert actual.shape == expected.shape
    assert_outputs_match(case, expected_outputs, actual_outputs)
    assert_native_graph_health(case, ggml_model)


@given(flatten_variant_cases())
@settings(
    max_examples=80,
    deadline=None,
    database=None,
    derandomize=True,
    suppress_health_check=[HealthCheck.too_slow],
)
def test_hypothesis_flatten_variants_match_onnxruntime_without_fallback(
    case: GeneratedGraphCase,
) -> None:
    axis = int(case.ir.ops[0].attrs["axis"])
    event("flatten rank: %d" % len(case.ir.inputs[0].shape))
    event("flatten axis: %d" % axis)
    event("flatten negative axis: %s" % (axis < 0))

    expected_outputs = run_onnxruntime(case)
    ggml_model, actual_outputs = run_ggml(case, fallback_policy="strict")

    assert len(actual_outputs) == len(expected_outputs)
    for actual, expected in zip(actual_outputs, expected_outputs):
        assert actual.shape == expected.shape
    assert_outputs_match(case, expected_outputs, actual_outputs)
    assert_native_graph_health(case, ggml_model)


@given(transpose_variant_cases())
@settings(
    max_examples=100,
    deadline=None,
    database=None,
    derandomize=True,
    suppress_health_check=[HealthCheck.too_slow],
)
def test_hypothesis_transpose_variants_match_onnxruntime_without_fallback(
    case: GeneratedGraphCase,
) -> None:
    attrs = case.ir.ops[0].attrs
    input_rank = len(case.ir.inputs[0].shape)
    permutation = tuple(attrs.get("perm", tuple(reversed(range(input_rank)))))
    event("transpose rank: %d" % input_rank)
    event("transpose omitted perm: %s" % ("perm" not in attrs))
    event("transpose identity perm: %s" % (permutation == tuple(range(input_rank))))
    event(
        "transpose reverse perm: %s"
        % (permutation == tuple(reversed(range(input_rank))))
    )

    expected_outputs = run_onnxruntime(case)
    ggml_model, actual_outputs = run_ggml(case, fallback_policy="strict")

    assert len(actual_outputs) == len(expected_outputs)
    for actual, expected in zip(actual_outputs, expected_outputs):
        assert actual.shape == expected.shape
    assert_outputs_match(case, expected_outputs, actual_outputs)
    assert_native_graph_health(case, ggml_model)


@given(arithmetic_broadcast_cases())
@settings(
    max_examples=120,
    deadline=None,
    database=None,
    derandomize=True,
    suppress_health_check=[HealthCheck.too_slow],
)
def test_hypothesis_arithmetic_broadcast_variants_match_onnxruntime(
    case: GeneratedGraphCase,
) -> None:
    left_shape = case.ir.inputs[0].shape
    right_shape = case.ir.inputs[1].shape
    op_type = case.ir.ops[0].op_type
    event("arithmetic op: %s" % op_type)
    event("arithmetic fallback expected: %s" % bool(case.expected_fallback_indices))
    event("arithmetic left scalar: %s" % (shape_element_count(left_shape) == 1))
    event("arithmetic right scalar: %s" % (shape_element_count(right_shape) == 1))
    event("arithmetic same shape: %s" % (left_shape == right_shape))

    expected_outputs = run_onnxruntime(case)
    ggml_model, actual_outputs = run_ggml(
        case,
        fallback_policy="compat" if case.expected_fallback_indices else "strict",
    )

    assert len(actual_outputs) == len(expected_outputs)
    for actual, expected in zip(actual_outputs, expected_outputs):
        assert actual.shape == expected.shape
    assert_outputs_match(case, expected_outputs, actual_outputs)
    assert_expected_execution_health(case, ggml_model)


@given(split_variant_cases())
@settings(
    max_examples=100,
    deadline=None,
    database=None,
    derandomize=True,
    suppress_health_check=[HealthCheck.too_slow],
)
def test_hypothesis_split_variants_match_onnxruntime(
    case: GeneratedGraphCase,
) -> None:
    op = case.ir.ops[0]
    axis = int(op.attrs["axis"])
    event("split variant: %s" % case.description)
    event("split rank: %d" % len(case.ir.inputs[0].shape))
    event("split negative axis: %s" % (axis < 0))
    event("split output count: %d" % len(op.outputs))
    event("split has input tensor: %s" % (len(op.inputs) == 2))

    expected_outputs = run_onnxruntime(case)
    ggml_model, actual_outputs = run_ggml(case)

    assert len(actual_outputs) == len(expected_outputs)
    for actual, expected in zip(actual_outputs, expected_outputs):
        assert actual.shape == expected.shape
    assert_outputs_match(case, expected_outputs, actual_outputs)
    assert_split_execution_health(case, ggml_model)


@given(native_graph_cases())
@settings(
    max_examples=80,
    deadline=None,
    database=None,
    derandomize=True,
    suppress_health_check=[HealthCheck.too_slow],
)
def test_hypothesis_native_onnx_graphs_match_onnxruntime_without_fallback(
    case: GeneratedGraphCase,
) -> None:
    expected_outputs = run_onnxruntime(case)
    ggml_model, actual_outputs = run_ggml(case, fallback_policy="strict")

    assert_outputs_match(case, expected_outputs, actual_outputs)
    assert_native_graph_health(case, ggml_model)


@given(native_dag_graph_cases())
@settings(
    max_examples=80,
    deadline=None,
    database=None,
    derandomize=True,
    suppress_health_check=[HealthCheck.too_slow],
)
def test_hypothesis_native_onnx_dags_match_onnxruntime_without_fallback(
    case: GeneratedGraphCase,
) -> None:
    event("native DAG runtime inputs: %d" % case.runtime_input_count)
    event("native DAG branches: %d" % case.branch_count)
    for op_type in sorted(set(case.op_types)):
        event("native DAG op: %s" % op_type)

    expected_outputs = run_onnxruntime(case)
    ggml_model, actual_outputs = run_ggml(case, fallback_policy="strict")

    assert case.runtime_input_count >= 2
    assert case.branch_count >= 1
    assert_outputs_match(case, expected_outputs, actual_outputs)
    assert_native_graph_health(case, ggml_model)


@given(mixed_graph_cases())
@settings(
    max_examples=40,
    deadline=None,
    database=None,
    derandomize=True,
    suppress_health_check=[HealthCheck.too_slow],
)
def test_hypothesis_mixed_onnx_graphs_match_onnxruntime_with_maximal_fallback_island(
    case: GeneratedGraphCase,
) -> None:
    expected_outputs = run_onnxruntime(case)
    ggml_model, actual_outputs = run_ggml(case)

    assert_outputs_match(case, expected_outputs, actual_outputs)
    assert_mixed_graph_health(case, ggml_model)
