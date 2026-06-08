"""GGML ONNX backend.

This module implements a GGML backend for ONNX models and operators.
"""

import math
import ctypes
import contextlib
from dataclasses import dataclass, replace
from typing import (
    Any,
    Callable,
    ClassVar,
    Dict,
    List,
    Optional,
    Sequence,
    Set,
    Tuple,
    Type,
)
from typing_extensions import TypeGuard

import numpy as np
import numpy.typing as npt
import onnx
from onnx.backend.base import Backend, BackendRep
from onnx.helper import tensor_dtype_to_np_dtype
from onnx.onnx_ml_pb2 import (
    GraphProto,
    ModelProto,
    NodeProto,
    ValueInfoProto,
    TensorProto,
)

import ggml
import ggml.utils


def get_tensor_shape(tensor: ggml.ggml_tensor_p) -> Tuple[int, ...]:
    return tuple(reversed(ggml.utils.get_shape(tensor)))


def get_tensor_dtype(tensor: ggml.ggml_tensor_p) -> npt.DTypeLike:
    ggml_type = ggml.utils.GGML_TYPE(tensor.contents.type)
    if ggml_type == ggml.utils.GGML_TYPE.F16:
        ctypes_type = ctypes.c_uint16
    else:
        ctypes_type = np.ctypeslib.as_ctypes_type(
            ggml.utils.GGML_TYPE_TO_NUMPY_DTYPE[ggml_type]
        )
    return np.dtype(ctypes_type)


# ------ Operators ------


GgmlOperator = Callable[["GgmlOnnxExecutionContext", NodeProto], None]


@dataclass(frozen=True)
class TensorType:
    shape: Optional[Tuple[int, ...]]
    dtype: Optional[np.dtype[Any]]
    scalar_value: Optional[Any] = None
    constant_value: Optional[npt.NDArray[Any]] = None
    constant: bool = False

    @staticmethod
    def from_info(tensor_info: Optional["TensorInfo"]) -> "TensorType":
        if tensor_info is None:
            return OnnxOperator.unknown_tensor_type()
        shape = (
            tuple(int(dim) for dim in tensor_info.shape)
            if all(isinstance(dim, int) for dim in tensor_info.shape)
            else None
        )
        dtype = np.dtype(tensor_info.dtype) if tensor_info.dtype is not None else None
        return TensorType(
            shape=shape,
            dtype=dtype,
            scalar_value=tensor_info.scalar_value,
            constant_value=tensor_info.constant_value,
            constant=tensor_info.constant,
        )

    @property
    def is_float32(self) -> bool:
        return self.dtype == np.dtype(np.float32)

    @property
    def is_static_scalar(self) -> bool:
        return self.scalar_value is not None

    @property
    def has_constant_value(self) -> bool:
        return self.constant_value is not None

    @property
    def is_scalar_shape(self) -> bool:
        return self.element_count == 1

    @property
    def rank(self) -> Optional[int]:
        return None if self.shape is None else len(self.shape)

    @property
    def element_count(self) -> Optional[int]:
        if self.shape is None:
            return None
        return int(np.prod(self.shape, dtype=np.int64)) if self.shape else 1

    @property
    def nbytes(self) -> Optional[int]:
        if self.dtype is None or self.element_count is None:
            return None
        return int(self.element_count * self.dtype.itemsize)

    @property
    def describe(self) -> str:
        shape = "?" if self.shape is None else str(self.shape)
        dtype = "?" if self.dtype is None else str(self.dtype)
        return f"{dtype}{shape}"


@dataclass(frozen=True)
class ViewTransformSemantics:
    KIND_SHAPE: ClassVar[str] = "shape"
    KIND_LAYOUT: ClassVar[str] = "layout"
    GGML_MAX_DIMS: ClassVar[int] = 4

    op_type: str
    view_kind: str
    input_type: TensorType
    output_type: TensorType
    has_static_parameters: bool
    permutation: Optional[Tuple[int, ...]]

    # ggml_permute axes are expressed in ggml's evaluated layout, which does
    # not match ONNX logical axes for every rank >= 3 permutation.
    GGML_TRANSPOSE_AXES_BY_RANK: ClassVar[
        Dict[int, Dict[Tuple[int, ...], Tuple[int, int, int, int]]]
    ] = {
        3: {
            (0, 1, 2): (0, 1, 2, 3),
            (0, 2, 1): (1, 0, 2, 3),
            (1, 0, 2): (0, 2, 1, 3),
            (1, 2, 0): (1, 2, 0, 3),
            (2, 0, 1): (2, 0, 1, 3),
            (2, 1, 0): (2, 1, 0, 3),
        },
        4: {
            (0, 1, 2, 3): (0, 1, 2, 3),
            (0, 1, 3, 2): (1, 0, 2, 3),
            (0, 2, 1, 3): (0, 2, 1, 3),
            (0, 2, 3, 1): (1, 2, 0, 3),
            (0, 3, 1, 2): (2, 0, 1, 3),
            (0, 3, 2, 1): (2, 1, 0, 3),
            (1, 0, 2, 3): (0, 1, 3, 2),
            (1, 0, 3, 2): (1, 0, 3, 2),
            (1, 2, 0, 3): (0, 2, 3, 1),
            (1, 2, 3, 0): (1, 2, 3, 0),
            (1, 3, 0, 2): (2, 0, 3, 1),
            (1, 3, 2, 0): (2, 1, 3, 0),
            (2, 0, 1, 3): (0, 3, 1, 2),
            (2, 0, 3, 1): (1, 3, 0, 2),
            (2, 1, 0, 3): (0, 3, 2, 1),
            (2, 1, 3, 0): (1, 3, 2, 0),
            (2, 3, 0, 1): (2, 3, 0, 1),
            (2, 3, 1, 0): (2, 3, 1, 0),
            (3, 0, 1, 2): (3, 0, 1, 2),
            (3, 0, 2, 1): (3, 1, 0, 2),
            (3, 1, 0, 2): (3, 0, 2, 1),
            (3, 1, 2, 0): (3, 1, 2, 0),
            (3, 2, 0, 1): (3, 2, 0, 1),
            (3, 2, 1, 0): (3, 2, 1, 0),
        },
    }

    @classmethod
    def ggml_transpose_axes(
        cls, permutation: Tuple[int, ...]
    ) -> Optional[Tuple[int, int, int, int]]:
        rank = len(permutation)
        if rank <= 2:
            return tuple(permutation) + tuple(
                range(rank, ViewTransformSemantics.GGML_MAX_DIMS)
            )  # type: ignore[return-value]
        return cls.GGML_TRANSPOSE_AXES_BY_RANK.get(rank, {}).get(permutation)

    @property
    def output_shape(self) -> Optional[Tuple[int, ...]]:
        return self.output_type.shape

    @property
    def output_dtype(self) -> Optional[np.dtype[Any]]:
        return self.output_type.dtype or self.input_type.dtype

    @property
    def is_shape_view(self) -> bool:
        return self.view_kind == self.KIND_SHAPE

    @property
    def is_layout_view(self) -> bool:
        return self.view_kind == self.KIND_LAYOUT

    @property
    def native_blocker_reason(self) -> str:
        if self.input_type.shape is None:
            return "View input shape is unknown"
        if self.output_shape is None:
            return "View output shape is unknown"
        if not self.has_static_parameters:
            return "View parameters are dynamic"
        if self.is_shape_view:
            return ""
        if self.is_layout_view:
            if self.permutation is None:
                return "Transpose permutation is invalid or input rank is unknown"
            if len(self.input_type.shape) > ViewTransformSemantics.GGML_MAX_DIMS:
                return (
                    f"Transpose rank {len(self.input_type.shape)} exceeds "
                    f"ggml native view rank {ViewTransformSemantics.GGML_MAX_DIMS}"
                )
            if self.ggml_transpose_axes(self.permutation) is None:
                return (
                    f"Transpose permutation {self.permutation} is not supported "
                    "by ggml native view"
                )
            return ""
        return "Operator is not a view transform"

    @property
    def can_lower_native(self) -> bool:
        return self.native_blocker_reason == ""


class OnnxOperator:
    EXECUTION_NATIVE: ClassVar[str] = "native"
    EXECUTION_DECOMPOSED: ClassVar[str] = "decomposed"
    EXECUTION_CONSTANT_FOLD: ClassVar[str] = "constant_fold"
    EXECUTION_NUMPY_RUNTIME: ClassVar[str] = "numpy_runtime"
    EXECUTION_NUMPY_EAGER: ClassVar[str] = "numpy_eager"
    EXECUTION_NATIVE_OR_NUMPY_RUNTIME: ClassVar[str] = "native_or_numpy_runtime"
    EXECUTION_UNSUPPORTED: ClassVar[str] = "unsupported"
    STRICT_EXECUTIONS: ClassVar[Set[str]] = {
        EXECUTION_NATIVE,
        EXECUTION_DECOMPOSED,
        EXECUTION_CONSTANT_FOLD,
    }
    NATIVE_EXECUTIONS: ClassVar[Set[str]] = {
        EXECUTION_NATIVE,
        EXECUTION_DECOMPOSED,
    }
    NUMPY_FALLBACK_EXECUTIONS: ClassVar[Set[str]] = {
        EXECUTION_NUMPY_RUNTIME,
        EXECUTION_NUMPY_EAGER,
        EXECUTION_NATIVE_OR_NUMPY_RUNTIME,
    }

    CLASS_NATIVE: ClassVar[str] = "native"
    CLASS_NATIVE_VIEW: ClassVar[str] = "native_view"
    CLASS_DECOMPOSED: ClassVar[str] = "decomposed"
    CLASS_CONSTANT_FOLD: ClassVar[str] = "constant_fold"
    CLASS_NUMPY_RUNTIME: ClassVar[str] = "numpy_runtime"
    CLASS_NUMPY_EAGER: ClassVar[str] = "numpy_eager"
    CLASS_CONDITIONAL_NATIVE: ClassVar[str] = "conditional_native"
    CLASS_CONDITIONAL_NATIVE_VIEW: ClassVar[str] = "conditional_native_view"
    CLASS_UNSUPPORTED: ClassVar[str] = "unsupported"
    NUMPY_FALLBACK_CLASSES: ClassVar[Set[str]] = {
        CLASS_NUMPY_RUNTIME,
        CLASS_NUMPY_EAGER,
    }
    NATIVE_CLASSES: ClassVar[Set[str]] = {
        CLASS_NATIVE,
        CLASS_NATIVE_VIEW,
        CLASS_DECOMPOSED,
    }

    def __init__(
        self,
        op_type: str,
        execution: str = "numpy_runtime",
        operator_class: Optional[str] = None,
        devices: Tuple[str, ...] = ("CPU",),
        view_kind: Optional[str] = None,
        domains: Tuple[str, ...] = ("",),
    ):
        self.op_type = op_type
        self.execution = execution
        self.operator_class = operator_class or self.class_for_execution(execution)
        self.devices = devices
        self.view_kind = view_kind
        self.domains = domains
        self.implementation: Optional[GgmlOperator] = None
        self.has_numpy_evaluator = False

    @property
    def is_layout_view(self) -> bool:
        return self.view_kind == ViewTransformSemantics.KIND_LAYOUT

    @property
    def is_shape_view(self) -> bool:
        return self.view_kind == ViewTransformSemantics.KIND_SHAPE

    def bind_implementation(self, implementation: GgmlOperator):
        self.implementation = implementation
        return self

    def native_strategy(self) -> Tuple[Tuple[str, str, str], ...]:
        return (
            (
                OnnxOperator.EXECUTION_NATIVE,
                OnnxOperator.CLASS_NATIVE,
                "Operator lowers to native ggml",
            ),
        )

    def decomposed_strategy(self) -> Tuple[Tuple[str, str, str], ...]:
        return (
            (
                OnnxOperator.EXECUTION_DECOMPOSED,
                OnnxOperator.CLASS_DECOMPOSED,
                "Operator decomposes into native ggml operations",
            ),
        )

    def native_view_strategy(self) -> Tuple[Tuple[str, str, str], ...]:
        return (
            (
                OnnxOperator.EXECUTION_NATIVE,
                OnnxOperator.CLASS_NATIVE_VIEW,
                "Operator lowers to a native ggml view",
            ),
        )

    def numpy_runtime_strategy(
        self,
        reason: str = "Operator shape, dtype, or attributes require NumPy runtime fallback",
    ) -> Tuple[Tuple[str, str, str], ...]:
        return (
            (
                OnnxOperator.EXECUTION_NUMPY_RUNTIME,
                OnnxOperator.CLASS_NUMPY_RUNTIME,
                reason,
            ),
        )

    def strategies(
        self, tensor_types: Dict[str, TensorType], node: "NodeIR"
    ) -> Tuple[Tuple[str, str, str], ...]:
        del tensor_types, node
        return (
            (
                self.execution,
                self.operator_class,
                self.reason_for_execution(self.execution, self.operator_class),
            ),
        )

    def lower_native(self, ctx: "GgmlOnnxExecutionContext", node: NodeProto) -> None:
        del ctx, node
        raise NotImplementedError(f'Operator "{self.op_type}" has no native lowerer')

    def lower(self, ctx: "GgmlOnnxExecutionContext", node: NodeProto) -> None:
        if self.implementation is None:
            raise NotImplementedError(f'Operator "{self.op_type}" not implemented')
        self.implementation(ctx, node)

    def eval_numpy(
        self, node: NodeProto, inputs: Tuple[npt.NDArray[Any], ...]
    ) -> Tuple[npt.NDArray[Any], ...]:
        del node, inputs
        raise NotImplementedError(f'Operator "{self.op_type}" has no NumPy evaluator')

    @staticmethod
    def class_for_execution(execution: str) -> str:
        if execution == OnnxOperator.EXECUTION_NATIVE:
            return OnnxOperator.CLASS_NATIVE
        if execution == OnnxOperator.EXECUTION_DECOMPOSED:
            return OnnxOperator.CLASS_DECOMPOSED
        if execution == OnnxOperator.EXECUTION_CONSTANT_FOLD:
            return OnnxOperator.CLASS_CONSTANT_FOLD
        if execution == OnnxOperator.EXECUTION_NUMPY_EAGER:
            return OnnxOperator.CLASS_NUMPY_EAGER
        if execution == OnnxOperator.EXECUTION_NATIVE_OR_NUMPY_RUNTIME:
            return OnnxOperator.CLASS_CONDITIONAL_NATIVE
        if execution == OnnxOperator.EXECUTION_UNSUPPORTED:
            return OnnxOperator.CLASS_UNSUPPORTED
        return OnnxOperator.CLASS_NUMPY_RUNTIME

    @staticmethod
    def reason_for_execution(
        execution: str,
        operator_class: str,
        fallback_reason: Optional[str] = None,
    ) -> str:
        if execution == OnnxOperator.EXECUTION_NATIVE:
            if operator_class == OnnxOperator.CLASS_NATIVE_VIEW:
                return "Operator lowers to a native ggml view"
            return "Operator lowers to native ggml"
        if execution == OnnxOperator.EXECUTION_DECOMPOSED:
            return "Operator decomposes into native ggml operations"
        if execution == OnnxOperator.EXECUTION_CONSTANT_FOLD:
            return "Operator is constant-folded during graph optimization"
        if execution == OnnxOperator.EXECUTION_NUMPY_EAGER:
            return "Operator is evaluated eagerly with NumPy"
        if execution == OnnxOperator.EXECUTION_UNSUPPORTED:
            return "Operator is not implemented"
        if fallback_reason is not None:
            return fallback_reason
        return "Operator uses NumPy runtime fallback"

    @staticmethod
    def int_attribute(node: NodeProto, name: str, default: int) -> int:
        return int(
            next((attr.i for attr in node.attribute if attr.name == name), default)
        )

    @staticmethod
    def ints_attribute(node: NodeProto, name: str) -> Optional[Tuple[int, ...]]:
        return next(
            (
                tuple(int(value) for value in attr.ints)
                for attr in node.attribute
                if attr.name == name
            ),
            None,
        )

    @staticmethod
    def float_attribute(node: NodeProto, name: str, default: float) -> float:
        return float(
            next((attr.f for attr in node.attribute if attr.name == name), default)
        )

    @staticmethod
    def string_attribute_value(value: Any) -> str:
        if isinstance(value, bytes):
            return value.decode("utf-8")
        return str(value)

    @staticmethod
    def runtime_tensor_type(ctx: "GgmlOnnxExecutionContext", name: str) -> TensorType:
        return TensorType(
            shape=tuple(ctx.shapes[name]),
            dtype=np.dtype(ctx.get_tensor_dtype(name)),
        )

    @staticmethod
    def unknown_tensor_type() -> TensorType:
        return TensorType(shape=None, dtype=None)

    @staticmethod
    def tensor_type(tensor_types: Dict[str, TensorType], name: str) -> TensorType:
        return tensor_types.get(name, OnnxOperator.unknown_tensor_type())

    @staticmethod
    def constant_tensor_value(
        tensor_types: Dict[str, TensorType], name: str
    ) -> Optional[npt.NDArray[Any]]:
        return OnnxOperator.tensor_type(tensor_types, name).constant_value

    @staticmethod
    def constant_int_values(
        tensor_types: Dict[str, TensorType], name: str
    ) -> Optional[Tuple[int, ...]]:
        value = OnnxOperator.constant_tensor_value(tensor_types, name)
        if value is None:
            return None
        return tuple(int(item) for item in np.asarray(value).flatten())

    @staticmethod
    def constant_scalar_value(
        tensor_types: Dict[str, TensorType], name: str
    ) -> Optional[Any]:
        tensor_type = OnnxOperator.tensor_type(tensor_types, name)
        if tensor_type.scalar_value is not None:
            return tensor_type.scalar_value
        value = tensor_type.constant_value
        if value is not None and value.size == 1:
            return value.reshape(()).item()
        return None

    @staticmethod
    def infer_elementwise_output_shape(
        input_types: Sequence[TensorType],
    ) -> Optional[Tuple[int, ...]]:
        input_shapes = [input_type.shape for input_type in input_types]
        if not input_shapes or any(shape is None for shape in input_shapes):
            return None
        try:
            return tuple(
                np.broadcast_shapes(
                    *(shape for shape in input_shapes if shape is not None)
                )
            )
        except ValueError:
            return None

    @staticmethod
    def infer_elementwise_output_dtype(
        input_types: Sequence[TensorType],
    ) -> Optional[np.dtype[Any]]:
        input_dtypes = [input_type.dtype for input_type in input_types]
        if not input_dtypes or any(dtype is None for dtype in input_dtypes):
            return None
        return np.result_type(*(dtype for dtype in input_dtypes if dtype is not None))

    @staticmethod
    def binary_elementwise_types(
        tensor_types: Dict[str, TensorType], node: "NodeIR"
    ) -> Tuple[TensorType, TensorType]:
        return (
            OnnxOperator.tensor_type(tensor_types, node.inputs[0]),
            OnnxOperator.tensor_type(tensor_types, node.inputs[1]),
        )

    @staticmethod
    def variadic_elementwise_types(
        tensor_types: Dict[str, TensorType], node: "NodeIR"
    ) -> Tuple[TensorType, ...]:
        return tuple(
            OnnxOperator.tensor_type(tensor_types, input_name)
            for input_name in node.inputs
            if input_name
        )

    @staticmethod
    def is_same_shape_float32(input_types: Sequence[TensorType]) -> bool:
        if not input_types:
            return False
        first_shape = input_types[0].shape
        return (
            first_shape is not None
            and all(input_type.shape == first_shape for input_type in input_types)
            and all(input_type.is_float32 for input_type in input_types)
        )

    @staticmethod
    def float32_scalar_broadcast_pair(
        input_types: Sequence[TensorType],
        output_shape: Optional[Tuple[int, ...]],
    ) -> Optional[Tuple[int, int]]:
        if len(input_types) != 2:
            return None
        if output_shape is None or not all(
            input_type.is_float32 for input_type in input_types
        ):
            return None
        for tensor_index, scalar_index in ((0, 1), (1, 0)):
            tensor_type = input_types[tensor_index]
            scalar_type = input_types[scalar_index]
            if tensor_type.shape is None or scalar_type.shape is None:
                continue
            if scalar_type.is_scalar_shape and output_shape == tensor_type.shape:
                return tensor_index, scalar_index
        return None

    @staticmethod
    def has_float32_scalar_broadcast(
        input_types: Sequence[TensorType],
        output_shape: Optional[Tuple[int, ...]],
    ) -> bool:
        return (
            OnnxOperator.float32_scalar_broadcast_pair(input_types, output_shape)
            is not None
        )

    @staticmethod
    def can_repeat_to_shape(
        input_shape: Tuple[int, ...], output_shape: Tuple[int, ...]
    ) -> bool:
        if not output_shape:
            return False
        if len(output_shape) > ViewTransformSemantics.GGML_MAX_DIMS:
            return False
        if len(input_shape) > len(output_shape):
            return False
        promoted_shape = (1,) * (len(output_shape) - len(input_shape)) + input_shape
        return all(
            input_dim > 0 and output_dim > 0 and output_dim % input_dim == 0
            for input_dim, output_dim in zip(promoted_shape, output_shape)
        )

    @staticmethod
    def reshape_native_tensor_to_shape(
        ctx: "GgmlOnnxExecutionContext",
        tensor: ggml.ggml_tensor_p,
        shape: Tuple[int, ...],
    ) -> ggml.ggml_tensor_p:
        storage_shape = tuple(reversed(shape))
        if len(storage_shape) == 0:
            return ggml.ggml_reshape_1d(ctx.ggml_eval_context, tensor, 1)
        if len(storage_shape) == 1:
            return ggml.ggml_reshape_1d(
                ctx.ggml_eval_context,
                tensor,
                storage_shape[0],
            )
        if len(storage_shape) == 2:
            return ggml.ggml_reshape_2d(
                ctx.ggml_eval_context,
                tensor,
                storage_shape[0],
                storage_shape[1],
            )
        if len(storage_shape) == 3:
            return ggml.ggml_reshape_3d(
                ctx.ggml_eval_context,
                tensor,
                storage_shape[0],
                storage_shape[1],
                storage_shape[2],
            )
        if len(storage_shape) == 4:
            return ggml.ggml_reshape_4d(
                ctx.ggml_eval_context,
                tensor,
                storage_shape[0],
                storage_shape[1],
                storage_shape[2],
                storage_shape[3],
            )
        raise ValueError(f"ggml native reshape supports at most 4 dims, got {shape}")

    @staticmethod
    def repeat_native_tensor_to_shape(
        ctx: "GgmlOnnxExecutionContext",
        tensor: ggml.ggml_tensor_p,
        input_shape: Tuple[int, ...],
        output_shape: Tuple[int, ...],
        dtype: npt.DTypeLike,
    ) -> ggml.ggml_tensor_p:
        promoted_shape = (1,) * (len(output_shape) - len(input_shape)) + input_shape
        reshaped = OnnxOperator.reshape_native_tensor_to_shape(
            ctx, tensor, promoted_shape
        )
        target = ctx.from_numpy(np.empty(output_shape, dtype=dtype))
        return ggml.ggml_repeat(ctx.ggml_eval_context, reshaped, target)

    @staticmethod
    def eval_numpy_unary_operator(
        node: NodeProto,
        inputs: Tuple[npt.NDArray[Any], ...],
        numpy_func: Callable[[npt.NDArray[Any]], Any],
    ) -> Tuple[npt.NDArray[Any], ...]:
        if len(inputs) != 1:
            raise ValueError(
                f'Operation "{node.op_type}" requires exactly one input. '
                f"Actual number of inputs: {len(inputs)}"
            )
        return (np.asarray(numpy_func(inputs[0]), dtype=inputs[0].dtype),)

    @staticmethod
    def eval_numpy_elu_operator(
        node: NodeProto, inputs: Tuple[npt.NDArray[Any], ...]
    ) -> Tuple[npt.NDArray[Any], ...]:
        if len(inputs) != 1:
            raise ValueError(
                f'Operation "{node.op_type}" requires exactly one input. '
                f"Actual number of inputs: {len(inputs)}"
            )
        x = inputs[0]
        alpha = OnnxOperator.float_attribute(node, "alpha", 1.0)
        result = np.where(x > 0, x, alpha * (np.exp(x) - 1))
        return (np.asarray(result, dtype=x.dtype),)

    @staticmethod
    def eval_numpy_leaky_relu_operator(
        node: NodeProto, inputs: Tuple[npt.NDArray[Any], ...]
    ) -> Tuple[npt.NDArray[Any], ...]:
        if len(inputs) != 1:
            raise ValueError(
                f'Operation "{node.op_type}" requires exactly one input. '
                f"Actual number of inputs: {len(inputs)}"
            )
        x = inputs[0]
        alpha = OnnxOperator.float_attribute(node, "alpha", 0.01)
        result = np.clip(x, 0, np.inf) + np.clip(x, -np.inf, 0) * alpha
        return (np.asarray(result, dtype=x.dtype),)

    @staticmethod
    def eval_numpy_binary_operator(
        node: NodeProto,
        inputs: Tuple[npt.NDArray[Any], ...],
        numpy_func: Callable[[npt.NDArray[Any], npt.NDArray[Any]], Any],
    ) -> Tuple[npt.NDArray[Any], ...]:
        if len(inputs) != 2:
            raise ValueError(
                f'Operation "{node.op_type}" requires exactly two inputs. '
                f"Actual number of inputs: {len(inputs)}"
            )
        dtype = np.result_type(inputs[0].dtype, inputs[1].dtype)
        return (np.asarray(numpy_func(inputs[0], inputs[1]), dtype=dtype),)

    @staticmethod
    def eval_numpy_variadic_operator(
        node: NodeProto,
        inputs: Tuple[npt.NDArray[Any], ...],
        numpy_func: Callable[[npt.NDArray[Any], npt.NDArray[Any]], Any],
        finalize: Optional[
            Callable[[npt.NDArray[Any], Tuple[npt.NDArray[Any], ...]], npt.NDArray[Any]]
        ] = None,
    ) -> Tuple[npt.NDArray[Any], ...]:
        if not inputs:
            raise ValueError(
                f'Operation "{node.op_type}" requires at least one input. '
                f"Actual number of inputs: {len(inputs)}"
            )
        dtype = np.result_type(*(array.dtype for array in inputs))
        result = np.asarray(inputs[0], dtype=dtype)
        for array in inputs[1:]:
            result = np.asarray(numpy_func(result, array), dtype=dtype)
        if finalize is not None:
            result = finalize(result, inputs)
        return (np.asarray(result, dtype=dtype),)

    @staticmethod
    def eval_numpy_pow_operator(
        node: NodeProto, inputs: Tuple[npt.NDArray[Any], ...]
    ) -> Tuple[npt.NDArray[Any], ...]:
        if len(inputs) != 2:
            raise ValueError(
                f'Operation "{node.op_type}" requires exactly two inputs. '
                f"Actual number of inputs: {len(inputs)}"
            )
        return (np.asarray(np.power(inputs[0], inputs[1]), dtype=inputs[0].dtype),)

    @staticmethod
    def eval_numpy_reduce_sum_operator(
        node: NodeProto, inputs: Tuple[npt.NDArray[Any], ...]
    ) -> Tuple[npt.NDArray[Any], ...]:
        return OnnxOperator.eval_numpy_reduce_operator(
            node,
            inputs,
            lambda data, axes, keepdims: np.sum(data, axis=axes, keepdims=keepdims),
        )

    @staticmethod
    def eval_numpy_reduce_operator(
        node: NodeProto,
        inputs: Tuple[npt.NDArray[Any], ...],
        reducer: Callable[[npt.NDArray[Any], Tuple[int, ...], bool], npt.NDArray[Any]],
    ) -> Tuple[npt.NDArray[Any], ...]:
        if not inputs:
            raise ValueError(
                f'Operation "{node.op_type}" requires at least one input. '
                f"Actual number of inputs: {len(inputs)}"
            )
        data = inputs[0]
        keepdims = bool(OnnxOperator.int_attribute(node, "keepdims", 1))
        noop = bool(OnnxOperator.int_attribute(node, "noop_with_empty_axes", 0))
        axes_provided = (
            OnnxOperator.ints_attribute(node, "axes") is not None or len(inputs) >= 2
        )
        if len(inputs) >= 2:
            axes_values = np.asarray(inputs[1], dtype=np.int64).flatten()
            axes = tuple(int(axis) for axis in axes_values)
        else:
            axes = OnnxOperator.ints_attribute(node, "axes")
        if axes is None:
            axes = ()
        if axes_provided and len(axes) == 0 and noop:
            return (np.asarray(data).copy(),)
        if len(axes) == 0:
            axes = tuple(range(data.ndim))
        axes = tuple(axis + data.ndim if axis < 0 else axis for axis in axes)
        for axis in axes:
            if axis < 0 or axis >= data.ndim:
                raise ValueError(
                    f'Operation "{node.op_type}" axis {axis} is out of bounds '
                    f"for rank {data.ndim}"
                )
        return (np.asarray(reducer(data, axes, keepdims), dtype=data.dtype),)

    def lower_numpy_unary(
        self,
        ctx: "GgmlOnnxExecutionContext",
        node: NodeProto,
        numpy_func: Callable[[npt.NDArray[Any]], Any],
        output_dtype: Optional[npt.DTypeLike] = None,
        output_shape: Optional[Tuple[int, ...]] = None,
    ) -> None:
        node_inputs = [ctx.ggml_tensors_dict[inp] for inp in node.input]
        if len(node_inputs) != 1:
            raise ValueError(
                f'Error for node "{node.name}": Operation "{self.op_type}" requires '
                f"exactly one input. Actual number of inputs: {len(node_inputs)}"
            )

        input_name = node.input[0]
        input_tensor = node_inputs[0]
        input_shape = ctx.shapes[input_name]
        result_shape = output_shape or input_shape
        input_dtype = np.dtype(ctx.get_tensor_dtype(input_name))
        result_dtype = (
            np.dtype(output_dtype) if output_dtype is not None else input_dtype
        )
        storage_dtype = ctx.storage_dtype_for_logical_dtype(result_dtype)
        storage_shape = result_shape if result_shape else (1,)
        if len(storage_shape) > ViewTransformSemantics.GGML_MAX_DIMS:
            storage_shape = (int(np.prod(storage_shape)),)
        output_tensor = ctx.from_numpy(np.empty(storage_shape, dtype=storage_dtype))

        @ggml.ggml_custom2_op_t
        def custom_unary(
            tensor_out: ggml.ggml_tensor_p,
            tensor_in_1: ggml.ggml_tensor_p,
            tensor_in_2: ggml.ggml_tensor_p,
            ith: int,
            nth: int,
            userdata: Optional[ctypes.c_void_p],
        ):
            del tensor_in_1, ith, nth, userdata
            data = ctx.logical_tensor_data(input_name, tensor_in_2, input_shape)
            result = np.asarray(numpy_func(data), dtype=result_dtype)
            ctx.set_tensor_data(tensor_out, result)

        new_tensor = ggml.ggml_map_custom2_inplace(
            ctx.ggml_eval_context,
            output_tensor,
            input_tensor,
            custom_unary,
            1,
            None,
        )
        ctx.refs.append(custom_unary)
        ctx.register_numpy_runtime_tensor(
            node.output[0], new_tensor, result_shape, result_dtype
        )

    def lower_numpy_binary(
        self,
        ctx: "GgmlOnnxExecutionContext",
        node: NodeProto,
        numpy_func: Callable[[npt.NDArray[Any], npt.NDArray[Any]], Any],
        output_dtype: Optional[npt.DTypeLike] = None,
    ) -> None:
        node_inputs = [ctx.ggml_tensors_dict[inp] for inp in node.input]
        if len(node_inputs) != 2:
            raise ValueError(
                f'Error for node "{node.name}": Operation "{self.op_type}" requires '
                f"exactly two inputs. Actual number of inputs: {len(node_inputs)}"
            )

        left_name, right_name = node.input
        left_tensor, right_tensor = node_inputs
        left_shape = ctx.shapes[left_name]
        right_shape = ctx.shapes[right_name]
        result_shape = np.broadcast_shapes(left_shape, right_shape)
        result_dtype = (
            np.dtype(output_dtype)
            if output_dtype is not None
            else np.result_type(
                ctx.get_tensor_dtype(left_name), ctx.get_tensor_dtype(right_name)
            )
        )
        storage_dtype = ctx.storage_dtype_for_logical_dtype(result_dtype)
        storage_shape = result_shape if result_shape else (1,)
        if len(storage_shape) > ViewTransformSemantics.GGML_MAX_DIMS:
            storage_shape = (int(np.prod(storage_shape)),)
        output_tensor = ctx.from_numpy(np.empty(storage_shape, dtype=storage_dtype))

        @ggml.ggml_custom3_op_t
        def custom_binary(
            tensor_out: ggml.ggml_tensor_p,
            tensor_in_1: ggml.ggml_tensor_p,
            tensor_in_2: ggml.ggml_tensor_p,
            tensor_in_3: ggml.ggml_tensor_p,
            ith: int,
            nth: int,
            userdata: Optional[ctypes.c_void_p],
        ):
            del tensor_in_1, ith, nth, userdata
            left = ctx.logical_tensor_data(left_name, tensor_in_2, left_shape)
            right = ctx.logical_tensor_data(right_name, tensor_in_3, right_shape)
            result = np.asarray(numpy_func(left, right), dtype=result_dtype)
            ctx.set_tensor_data(tensor_out, result)

        new_tensor = ggml.ggml_map_custom3_inplace(
            ctx.ggml_eval_context,
            output_tensor,
            left_tensor,
            right_tensor,
            custom_binary,
            1,
            None,
        )
        ctx.refs.append(custom_binary)
        ctx.register_numpy_runtime_tensor(
            node.output[0], new_tensor, result_shape, result_dtype
        )

    def lower_numpy_integer_unary(
        self,
        ctx: "GgmlOnnxExecutionContext",
        node: NodeProto,
        numpy_func: Callable[[npt.NDArray[Any]], Any],
    ) -> None:
        node_inputs = [ctx.ggml_tensors_dict[inp] for inp in node.input]
        if len(node_inputs) != 1:
            raise ValueError(
                f'Error for node "{node.name}": Operation "{self.op_type}" requires '
                f"exactly one input. Actual number of inputs: {len(node_inputs)}"
            )

        input_name = node.input[0]
        input_tensor = node_inputs[0]
        input_shape = ctx.shapes[input_name]
        result_dtype = np.dtype(ctx.get_tensor_dtype(input_name))
        storage_dtype = ctx.storage_dtype_for_logical_dtype(result_dtype)
        storage_shape = input_shape if input_shape else (1,)
        if len(storage_shape) > ViewTransformSemantics.GGML_MAX_DIMS:
            storage_shape = (int(np.prod(storage_shape)),)
        output_tensor = ctx.from_numpy(np.empty(storage_shape, dtype=storage_dtype))

        @ggml.ggml_custom2_op_t
        def custom_unary(
            tensor_out: ggml.ggml_tensor_p,
            tensor_in_1: ggml.ggml_tensor_p,
            tensor_in_2: ggml.ggml_tensor_p,
            ith: int,
            nth: int,
            userdata: Optional[ctypes.c_void_p],
        ):
            del tensor_in_1, ith, nth, userdata
            data = ctx.logical_tensor_data(input_name, tensor_in_2, input_shape)
            result = np.asarray(numpy_func(data), dtype=result_dtype)
            ctx.set_tensor_data(tensor_out, result)

        new_tensor = ggml.ggml_map_custom2_inplace(
            ctx.ggml_eval_context,
            output_tensor,
            input_tensor,
            custom_unary,
            1,
            None,
        )
        ctx.refs.append(custom_unary)
        ctx.register_numpy_runtime_tensor(
            node.output[0], new_tensor, input_shape, result_dtype
        )

    def lower_numpy_integer_binary(
        self,
        ctx: "GgmlOnnxExecutionContext",
        node: NodeProto,
        numpy_func: Callable[[npt.NDArray[Any], npt.NDArray[Any]], Any],
    ) -> None:
        node_inputs = [ctx.ggml_tensors_dict[inp] for inp in node.input]
        if len(node_inputs) != 2:
            raise ValueError(
                f'Error for node "{node.name}": Operation "{self.op_type}" requires '
                f"exactly two inputs. Actual number of inputs: {len(node_inputs)}"
            )

        left_name, right_name = node.input
        left_tensor, right_tensor = node_inputs
        left_shape = ctx.shapes[left_name]
        right_shape = ctx.shapes[right_name]
        result_shape = np.broadcast_shapes(left_shape, right_shape)
        result_dtype = np.result_type(
            ctx.get_tensor_dtype(left_name), ctx.get_tensor_dtype(right_name)
        )
        storage_dtype = ctx.storage_dtype_for_logical_dtype(result_dtype)
        storage_shape = result_shape if result_shape else (1,)
        if len(storage_shape) > ViewTransformSemantics.GGML_MAX_DIMS:
            storage_shape = (int(np.prod(storage_shape)),)
        output_tensor = ctx.from_numpy(np.empty(storage_shape, dtype=storage_dtype))

        @ggml.ggml_custom3_op_t
        def custom_binary(
            tensor_out: ggml.ggml_tensor_p,
            tensor_in_1: ggml.ggml_tensor_p,
            tensor_in_2: ggml.ggml_tensor_p,
            tensor_in_3: ggml.ggml_tensor_p,
            ith: int,
            nth: int,
            userdata: Optional[ctypes.c_void_p],
        ):
            del tensor_in_1, ith, nth, userdata
            left = ctx.logical_tensor_data(left_name, tensor_in_2, left_shape)
            right = ctx.logical_tensor_data(right_name, tensor_in_3, right_shape)
            result = np.asarray(numpy_func(left, right), dtype=result_dtype)
            ctx.set_tensor_data(tensor_out, result)

        new_tensor = ggml.ggml_map_custom3_inplace(
            ctx.ggml_eval_context,
            output_tensor,
            left_tensor,
            right_tensor,
            custom_binary,
            1,
            None,
        )
        ctx.refs.append(custom_binary)
        ctx.register_numpy_runtime_tensor(
            node.output[0], new_tensor, result_shape, result_dtype
        )

    def lower_numpy_variadic(
        self,
        ctx: "GgmlOnnxExecutionContext",
        node: NodeProto,
        numpy_func: Callable[[npt.NDArray[Any], npt.NDArray[Any]], Any],
    ) -> None:
        node_inputs = [ctx.ggml_tensors_dict[inp] for inp in node.input]
        if len(node_inputs) < 1:
            raise ValueError(
                f'Error for node "{node.name}": Operation "{self.op_type}" requires '
                f"at least one input. Actual number of inputs: {len(node_inputs)}"
            )

        arrays = [
            ctx.logical_tensor_eval_data(name, tensor, ctx.shapes[name])
            for name, tensor in zip(node.input, node_inputs)
        ]
        result_dtype = np.result_type(
            *(ctx.get_tensor_dtype(name) for name in node.input)
        )
        result = arrays[0]
        for array in arrays[1:]:
            result = numpy_func(result, array)
        result = np.asarray(result, dtype=result_dtype)
        storage_dtype = ctx.storage_dtype_for_logical_dtype(result_dtype)
        storage = np.asarray(result, dtype=storage_dtype)

        new_tensor = ctx.from_numpy(storage)
        ctx.register_numpy_eager_tensor(
            node.output[0], new_tensor, result.shape, result_dtype
        )

    def lower_native_unary_or_numpy(
        self,
        ctx: "GgmlOnnxExecutionContext",
        node: NodeProto,
        ggml_func: Callable[
            [ggml.ggml_context_p, ggml.ggml_tensor_p], ggml.ggml_tensor_p
        ],
        numpy_func: Callable[[npt.NDArray[Any]], Any],
    ) -> None:
        node_inputs = [ctx.ggml_tensors_dict[inp] for inp in node.input]
        if len(node_inputs) != 1:
            raise ValueError(
                f'Error for node "{node.name}": Operation "{self.op_type}" requires '
                f"exactly one input. Actual number of inputs: {len(node_inputs)}"
            )

        input_name = node.input[0]
        output_name = node.output[0]
        input_type = self.runtime_tensor_type(ctx, input_name)
        if (
            not ctx.can_emit_native(output_name)
            or not ctx.can_run_native(node)
            or not input_type.is_float32
        ):
            self.lower_numpy_unary(ctx, node, numpy_func)
            return

        result = ggml_func(ctx.ggml_eval_context, node_inputs[0])
        ctx.register_native_tensor(
            output_name,
            result,
            input_type.shape or ctx.shapes[input_name],
            input_type.dtype or np.dtype(np.float32),
        )

    def lower_native_binary_or_numpy(
        self,
        ctx: "GgmlOnnxExecutionContext",
        node: NodeProto,
        ggml_func: Callable[
            [ggml.ggml_context_p, ggml.ggml_tensor_p, ggml.ggml_tensor_p],
            ggml.ggml_tensor_p,
        ],
        numpy_func: Callable[[npt.NDArray[Any], npt.NDArray[Any]], Any],
        scalar_func: Optional[
            Callable[
                [ggml.ggml_context_p, ggml.ggml_tensor_p, ggml.ggml_tensor_p, float],
                ggml.ggml_tensor_p,
            ]
        ] = None,
    ) -> None:
        node_inputs = [ctx.ggml_tensors_dict[inp] for inp in node.input]
        if len(node_inputs) != 2:
            raise ValueError(
                f'Error for node "{node.name}": Operation "{self.op_type}" requires '
                f"exactly two inputs. Actual number of inputs: {len(node_inputs)}"
            )

        left_name, _right_name = node.input
        output_name = node.output[0]
        input_types = (
            self.runtime_tensor_type(ctx, node.input[0]),
            self.runtime_tensor_type(ctx, node.input[1]),
        )
        output_shape = self.infer_elementwise_output_shape(input_types)
        can_run_native = ctx.can_emit_native(output_name) and ctx.can_run_native(node)

        if can_run_native and self.is_same_shape_float32(input_types):
            result = ggml_func(ctx.ggml_eval_context, node_inputs[0], node_inputs[1])
            ctx.register_native_tensor(
                output_name,
                result,
                output_shape or ctx.shapes[left_name],
                np.dtype(np.float32),
            )
            return

        if can_run_native and scalar_func is not None:
            scalar_pair = self.float32_scalar_broadcast_pair(input_types, output_shape)
            if scalar_pair is not None:
                tensor_index, scalar_index = scalar_pair
                tensor_name = node.input[tensor_index]
                scalar_name = node.input[scalar_index]
                tensor_shape = ctx.shapes[tensor_name]
                scalar_shape = ctx.shapes[scalar_name]
                scalar_data = ctx.logical_tensor_eval_data(
                    scalar_name, node_inputs[scalar_index], scalar_shape
                )
                scalar_value = float(scalar_data.reshape(()).item())
                result = scalar_func(
                    ctx.ggml_eval_context,
                    node_inputs[tensor_index],
                    node_inputs[scalar_index],
                    scalar_value,
                )
                ctx.register_native_tensor(
                    output_name, result, tensor_shape, np.dtype(np.float32)
                )
                return

        self.lower_numpy_binary(ctx, node, numpy_func)

    def pool_parameters_from_node(
        self, node: NodeProto, x_shape: Tuple[int, ...]
    ) -> Tuple[
        Tuple[int, ...],
        Tuple[int, ...],
        Tuple[int, ...],
        Tuple[int, ...],
        Tuple[int, ...],
        Tuple[int, ...],
        int,
        int,
        int,
        int,
    ]:
        kernel_shape = tuple(
            int(v)
            for attr in node.attribute
            if attr.name == "kernel_shape"
            for v in attr.ints
        )
        if not kernel_shape:
            raise ValueError(f'Error for node "{node.name}": kernel_shape is required')
        spatial_rank = len(kernel_shape)
        if len(x_shape) != spatial_rank + 2:
            raise ValueError(
                f'Error for node "{node.name}": input rank does not match kernel rank.'
            )
        strides = (
            tuple(
                int(v)
                for attr in node.attribute
                if attr.name == "strides"
                for v in attr.ints
            )
            or (1,) * spatial_rank
        )
        dilations = (
            tuple(
                int(v)
                for attr in node.attribute
                if attr.name == "dilations"
                for v in attr.ints
            )
            or (1,) * spatial_rank
        )
        pads_attr = tuple(
            int(v) for attr in node.attribute if attr.name == "pads" for v in attr.ints
        )
        auto_pad = next(
            (
                attr.s.decode("utf-8")
                for attr in node.attribute
                if attr.name == "auto_pad"
            ),
            "NOTSET",
        )
        ceil_mode = next(
            (attr.i for attr in node.attribute if attr.name == "ceil_mode"), 0
        )
        count_include_pad = next(
            (attr.i for attr in node.attribute if attr.name == "count_include_pad"), 0
        )
        storage_order = next(
            (attr.i for attr in node.attribute if attr.name == "storage_order"), 0
        )
        p = next((attr.i for attr in node.attribute if attr.name == "p"), 2)

        spatial_shape = x_shape[2:]
        effective_kernel = tuple(
            (kernel_shape[i] - 1) * dilations[i] + 1 for i in range(spatial_rank)
        )

        if auto_pad in {"SAME_UPPER", "SAME_LOWER"}:
            output_spatial = tuple(
                int(math.ceil(spatial_shape[i] / strides[i]))
                for i in range(spatial_rank)
            )
            pad_totals = [
                max(
                    0,
                    (output_spatial[i] - 1) * strides[i]
                    + effective_kernel[i]
                    - spatial_shape[i],
                )
                for i in range(spatial_rank)
            ]
            if auto_pad == "SAME_UPPER":
                pads_begin = tuple(total // 2 for total in pad_totals)
            else:
                pads_begin = tuple(total - total // 2 for total in pad_totals)
            pads_end = tuple(
                total - begin for total, begin in zip(pad_totals, pads_begin)
            )
        elif auto_pad == "VALID":
            pads_begin = (0,) * spatial_rank
            pads_end = (0,) * spatial_rank
            output_spatial = tuple(
                int(
                    math.floor(
                        (spatial_shape[i] - effective_kernel[i]) / strides[i] + 1
                    )
                )
                for i in range(spatial_rank)
            )
        else:
            pads = pads_attr or (0,) * (2 * spatial_rank)
            pads_begin = tuple(pads[:spatial_rank])
            pads_end = tuple(pads[spatial_rank:])
            output_spatial = []
            for i in range(spatial_rank):
                numerator = (
                    spatial_shape[i] + pads_begin[i] + pads_end[i] - effective_kernel[i]
                )
                if ceil_mode:
                    output_dim = int(math.ceil(numerator / strides[i] + 1))
                    if (output_dim - 1) * strides[i] >= spatial_shape[i] + pads_begin[
                        i
                    ]:
                        output_dim -= 1
                else:
                    output_dim = int(math.floor(numerator / strides[i] + 1))
                output_spatial.append(output_dim)
            output_spatial = tuple(output_spatial)

        return (
            kernel_shape,
            strides,
            dilations,
            pads_begin,
            pads_end,
            output_spatial,
            int(ceil_mode),
            int(count_include_pad),
            int(storage_order),
            int(p),
        )

    def pool_parameters_from_ir(
        self, node: "NodeIR", x_shape: Tuple[int, ...]
    ) -> Optional[
        Tuple[
            Tuple[int, ...],
            Tuple[int, ...],
            Tuple[int, ...],
            Tuple[int, ...],
            Tuple[int, ...],
            Tuple[int, ...],
            int,
            int,
            int,
            int,
        ]
    ]:
        kernel_shape = tuple(int(v) for v in node.attribute("kernel_shape", ()))
        if not kernel_shape:
            return None
        spatial_rank = len(kernel_shape)
        if len(x_shape) != spatial_rank + 2:
            return None
        strides = tuple(int(v) for v in node.attribute("strides", (1,) * spatial_rank))
        dilations = tuple(
            int(v) for v in node.attribute("dilations", (1,) * spatial_rank)
        )
        pads_attr = tuple(int(v) for v in node.attribute("pads", ()))
        auto_pad = str(node.attribute("auto_pad", "NOTSET"))
        ceil_mode = int(node.attribute("ceil_mode", 0))
        count_include_pad = int(node.attribute("count_include_pad", 0))
        storage_order = int(node.attribute("storage_order", 0))
        p = int(node.attribute("p", 2))
        spatial_shape = x_shape[2:]
        effective_kernel = tuple(
            (kernel_shape[i] - 1) * dilations[i] + 1 for i in range(spatial_rank)
        )

        if auto_pad in {"SAME_UPPER", "SAME_LOWER"}:
            output_spatial = tuple(
                int(math.ceil(spatial_shape[i] / strides[i]))
                for i in range(spatial_rank)
            )
            pad_totals = [
                max(
                    0,
                    (output_spatial[i] - 1) * strides[i]
                    + effective_kernel[i]
                    - spatial_shape[i],
                )
                for i in range(spatial_rank)
            ]
            if auto_pad == "SAME_UPPER":
                pads_begin = tuple(total // 2 for total in pad_totals)
            else:
                pads_begin = tuple(total - total // 2 for total in pad_totals)
            pads_end = tuple(
                total - begin for total, begin in zip(pad_totals, pads_begin)
            )
        elif auto_pad == "VALID":
            pads_begin = (0,) * spatial_rank
            pads_end = (0,) * spatial_rank
            output_spatial = tuple(
                int(
                    math.floor(
                        (spatial_shape[i] - effective_kernel[i]) / strides[i] + 1
                    )
                )
                for i in range(spatial_rank)
            )
        elif auto_pad == "NOTSET":
            pads = pads_attr or (0,) * (2 * spatial_rank)
            if len(pads) != spatial_rank * 2:
                return None
            pads_begin = tuple(pads[:spatial_rank])
            pads_end = tuple(pads[spatial_rank:])
            output_values = []
            for i in range(spatial_rank):
                numerator = (
                    spatial_shape[i] + pads_begin[i] + pads_end[i] - effective_kernel[i]
                )
                if ceil_mode:
                    output_dim = int(math.ceil(numerator / strides[i] + 1))
                    if (output_dim - 1) * strides[i] >= spatial_shape[i] + pads_begin[
                        i
                    ]:
                        output_dim -= 1
                else:
                    output_dim = int(math.floor(numerator / strides[i] + 1))
                output_values.append(output_dim)
            output_spatial = tuple(output_values)
        else:
            return None

        return (
            kernel_shape,
            strides,
            dilations,
            pads_begin,
            pads_end,
            output_spatial,
            ceil_mode,
            count_include_pad,
            storage_order,
            p,
        )

    @staticmethod
    def can_lower_pool_native(
        mode: str,
        x_shape: Tuple[int, ...],
        x_dtype: np.dtype[Any],
        kernel_shape: Tuple[int, ...],
        strides: Tuple[int, ...],
        dilations: Tuple[int, ...],
        pads_begin: Tuple[int, ...],
        pads_end: Tuple[int, ...],
        output_spatial: Tuple[int, ...],
        ceil_mode: int,
        count_include_pad: int,
        outputs: Sequence[str],
    ) -> bool:
        if mode not in {"Average", "Max"}:
            return False
        if x_dtype != np.dtype(np.float32):
            return False
        if len(x_shape) != 4 or len(kernel_shape) != 2:
            return False
        if len(strides) != 2 or len(dilations) != 2:
            return False
        if any(dilation != 1 for dilation in dilations):
            return False
        if ceil_mode:
            return False
        if pads_begin != pads_end:
            return False
        if any(dim <= 0 for dim in output_spatial):
            return False
        if mode == "Average" and not count_include_pad and any(pads_begin):
            return False
        if mode == "Max" and len(outputs) > 1 and outputs[1]:
            return False
        return True

    def pool_strategies(
        self, tensor_types: Dict[str, TensorType], node: "NodeIR", mode: str
    ) -> Tuple[Tuple[str, str, str], ...]:
        if len(node.inputs) == 1:
            input_type = self.tensor_type(tensor_types, node.inputs[0])
            if input_type.shape is not None and input_type.dtype is not None:
                parameters = self.pool_parameters_from_ir(node, input_type.shape)
                if parameters is not None:
                    (
                        kernel_shape,
                        strides,
                        dilations,
                        pads_begin,
                        pads_end,
                        output_spatial,
                        ceil_mode,
                        count_include_pad,
                        _storage_order,
                        _p,
                    ) = parameters
                    if self.can_lower_pool_native(
                        mode,
                        input_type.shape,
                        input_type.dtype,
                        kernel_shape,
                        strides,
                        dilations,
                        pads_begin,
                        pads_end,
                        output_spatial,
                        ceil_mode,
                        count_include_pad,
                        node.outputs,
                    ):
                        return self.native_strategy()
        return self.numpy_runtime_strategy(
            f"{mode}Pool requires exact 2D float32 ggml_pool_2d semantics to lower native"
        )

    def eval_numpy_pool(
        self,
        node: NodeProto,
        inputs: Tuple[npt.NDArray[Any], ...],
        mode: str,
    ) -> Tuple[npt.NDArray[Any], ...]:
        if len(inputs) != 1:
            raise ValueError(f'Operation "{node.op_type}" requires one input')
        x = inputs[0]
        x_shape = tuple(x.shape)
        (
            kernel_shape,
            strides,
            dilations,
            pads_begin,
            pads_end,
            output_spatial,
            _ceil_mode,
            count_include_pad,
            storage_order,
            p,
        ) = self.pool_parameters_from_node(node, x_shape)
        spatial_rank = len(kernel_shape)
        spatial_shape = x_shape[2:]
        effective_kernel = tuple(
            (kernel_shape[i] - 1) * dilations[i] + 1 for i in range(spatial_rank)
        )
        x_dtype = np.dtype(x.dtype)
        if mode == "Average":
            pad_value = 0 if count_include_pad else np.nan
        elif mode == "Lp":
            pad_value = 0
        elif np.issubdtype(x_dtype, np.integer):
            pad_value = np.iinfo(x_dtype).min
        else:
            pad_value = -np.inf
        padded = np.pad(
            x,
            ((0, 0), (0, 0), *zip(pads_begin, pads_end)),
            mode="constant",
            constant_values=pad_value,
        )

        output_shape = (*x_shape[:2], *output_spatial)
        output = np.empty(output_shape, dtype=x_dtype)
        indices = (
            np.empty(output_shape, dtype=np.int64)
            if mode == "Max" and len(node.output) > 1
            else None
        )
        for n in range(output_shape[0]):
            for c in range(output_shape[1]):
                for out_index in np.ndindex(*output_spatial):
                    slices = tuple(
                        slice(
                            out_index[i] * strides[i],
                            out_index[i] * strides[i] + effective_kernel[i],
                            dilations[i],
                        )
                        for i in range(spatial_rank)
                    )
                    window = padded[(n, c, *slices)]
                    if mode == "Average":
                        value = (
                            np.mean(window) if count_include_pad else np.nanmean(window)
                        )
                    elif mode == "Lp":
                        value = np.power(np.sum(np.power(np.abs(window), p)), 1.0 / p)
                    else:
                        value = np.nanmax(window)
                        if indices is not None:
                            flat_index = int(np.nanargmax(window))
                            local_index = np.unravel_index(flat_index, window.shape)
                            input_index = tuple(
                                out_index[i] * strides[i]
                                + local_index[i] * dilations[i]
                                - pads_begin[i]
                                for i in range(spatial_rank)
                            )
                            if storage_order == 0:
                                index = np.ravel_multi_index(
                                    (n, c, *input_index),
                                    x_shape,
                                )
                            else:
                                index = (n * x_shape[1] + c) * np.prod(
                                    spatial_shape
                                ) + np.ravel_multi_index(
                                    tuple(reversed(input_index)),
                                    tuple(reversed(spatial_shape)),
                                )
                            indices[(n, c, *out_index)] = int(index)
                    output[(n, c, *out_index)] = value

        if indices is not None:
            return output, indices
        return (output,)

    def lower_pool(self, ctx: "GgmlOnnxExecutionContext", node: NodeProto, mode: str):
        node_inputs = [ctx.ggml_tensors_dict[inp] for inp in node.input]

        if len(node_inputs) != 1:
            raise ValueError(
                f'Error for node "{node.name}": Operation "{mode}Pool" requires exactly one input. Actual number of inputs: {len(node_inputs)}'
            )

        x_shape = ctx.shapes[node.input[0]]
        parameters = self.pool_parameters_from_node(node, x_shape)
        (
            kernel_shape,
            strides,
            dilations,
            pads_begin,
            pads_end,
            output_spatial,
            ceil_mode,
            count_include_pad,
            _storage_order,
            _p,
        ) = parameters
        x_dtype = np.dtype(ctx.get_tensor_dtype(node.input[0]))

        if (
            ctx.can_emit_native(node.output[0])
            and ctx.can_run_native(node)
            and self.can_lower_pool_native(
                mode,
                x_shape,
                x_dtype,
                kernel_shape,
                strides,
                dilations,
                pads_begin,
                pads_end,
                output_spatial,
                ceil_mode,
                count_include_pad,
                node.output,
            )
        ):
            op = ggml.GGML_OP_POOL_AVG if mode == "Average" else ggml.GGML_OP_POOL_MAX
            result = ggml.ggml_pool_2d(
                ctx.ggml_eval_context,
                node_inputs[0],
                op,
                kernel_shape[1],
                kernel_shape[0],
                strides[1],
                strides[0],
                ctypes.c_float(float(pads_begin[1])),
                ctypes.c_float(float(pads_begin[0])),
            )
            output_shape = (*x_shape[:2], *output_spatial)
            ctx.register_native_tensor(
                node.output[0], result, output_shape, np.dtype(np.float32)
            )
            return

        x = ctx.logical_tensor_eval_data(node.input[0], node_inputs[0], x_shape)
        outputs = self.eval_numpy_pool(node, (x,), mode)
        ctx.set_logical_output(node.output[0], outputs[0], outputs[0].dtype)
        if len(outputs) > 1:
            ctx.set_logical_output(node.output[1], outputs[1], outputs[1].dtype)

    def lower_window(
        self,
        ctx: "GgmlOnnxExecutionContext",
        node: NodeProto,
        window_func: Callable[[npt.NDArray[Any], float], npt.NDArray[Any]],
    ):
        node_inputs = [ctx.ggml_tensors_dict[inp] for inp in node.input]

        if len(node_inputs) != 1:
            raise ValueError(
                f'Error for node "{node.name}": Operation "{self.op_type}" requires exactly one input. Actual number of inputs: {len(node_inputs)}'
            )

        size_shape = ctx.shapes[node.input[0]]
        size = int(
            ctx.to_numpy(ctx.eval_tensor(node_inputs[0])).reshape(size_shape).item()
        )
        periodic = next(
            (attr.i for attr in node.attribute if attr.name == "periodic"), 1
        )
        output_datatype = next(
            (attr.i for attr in node.attribute if attr.name == "output_datatype"),
            TensorProto.FLOAT,
        )
        output_dtype = np.dtype(tensor_dtype_to_np_dtype(output_datatype))
        denominator = size if periodic else size - 1
        values = np.arange(0, size, 1, dtype=np.float32)
        output = np.asarray(window_func(values, denominator), dtype=output_dtype)

        new_tensor = ctx.ggml_tensors_dict[node.output[0]] = ctx.from_numpy(output)
        ctx.set_tensor_shape(new_tensor, output.shape)
        ctx.shapes[node.output[0]] = output.shape
        ctx.set_tensor_dtype(node.output[0], output_dtype)

    def reduce_axes(
        self,
        ctx: "GgmlOnnxExecutionContext",
        node: NodeProto,
        node_inputs: List[ggml.ggml_tensor_p],
        rank: int,
    ) -> Tuple[Tuple[int, ...], bool]:
        axes_attr = next((attr for attr in node.attribute if attr.name == "axes"), None)
        noop_with_empty_axes = next(
            (attr.i for attr in node.attribute if attr.name == "noop_with_empty_axes"),
            0,
        )

        axes_provided = axes_attr is not None or len(node_inputs) > 1
        if axes_attr is not None:
            axes = [int(axis) for axis in axes_attr.ints]
        elif len(node_inputs) > 1:
            axes_eval = ctx.eval_tensor(node_inputs[1])
            axes = [int(axis) for axis in ctx.to_numpy(axes_eval).flatten()]
        else:
            axes = []

        if axes_provided and len(axes) == 0 and noop_with_empty_axes == 1:
            return (), True

        if len(axes) == 0:
            axes = list(range(rank))

        normalized_axes = []
        for axis in axes:
            axis = axis + rank if axis < 0 else axis
            if axis < 0 or axis >= rank:
                raise ValueError(
                    f'Error for node "{node.name}": Reduce axis {axis} is out of bounds for rank {rank}'
                )
            normalized_axes.append(axis)

        return tuple(normalized_axes), False

    @staticmethod
    def reduce_output_shape(
        tensor_shape: Tuple[int, ...],
        axes: Tuple[int, ...],
        keepdims: bool,
    ) -> Tuple[int, ...]:
        if keepdims:
            output_shape = list(tensor_shape)
            for axis in axes:
                output_shape[axis] = 1
            return tuple(output_shape)

        return tuple(dim for axis, dim in enumerate(tensor_shape) if axis not in axes)

    def static_reduce_all_axes(
        self, tensor_types: Dict[str, TensorType], node: "NodeIR"
    ) -> bool:
        if not node.inputs:
            return False
        input_shape = self.tensor_type(tensor_types, node.inputs[0]).shape
        if input_shape is None:
            return False
        axes = node.attribute("axes")
        axes_provided = axes is not None or len(node.inputs) > 1
        if axes is None and len(node.inputs) > 1 and node.inputs[1]:
            axes = self.constant_int_values(tensor_types, node.inputs[1])
            if axes is None:
                return False
        if axes is None:
            axes = ()
        axes = tuple(int(axis) for axis in axes)
        noop = bool(int(node.attribute("noop_with_empty_axes", 0)))
        if axes_provided and len(axes) == 0 and noop:
            return False
        if len(axes) == 0:
            axes = tuple(range(len(input_shape)))
        normalized_axes = tuple(
            axis + len(input_shape) if axis < 0 else axis for axis in axes
        )
        if any(axis < 0 or axis >= len(input_shape) for axis in normalized_axes):
            return False
        return set(normalized_axes) == set(range(len(input_shape)))

    def reduce_all_native_strategy(
        self, tensor_types: Dict[str, TensorType], node: "NodeIR"
    ) -> Tuple[Tuple[str, str, str], ...]:
        if len(node.inputs) in {1, 2}:
            input_type = self.tensor_type(tensor_types, node.inputs[0])
            if input_type.is_float32 and self.static_reduce_all_axes(
                tensor_types, node
            ):
                return self.native_strategy()
        return self.numpy_runtime_strategy(
            f"{self.op_type} requires float32 reduce-all axes to lower native"
        )

    def lower_native_reduce_all_or_numpy(
        self,
        ctx: "GgmlOnnxExecutionContext",
        node: NodeProto,
        native_reducer: Callable[
            ["GgmlOnnxExecutionContext", ggml.ggml_tensor_p], ggml.ggml_tensor_p
        ],
        numpy_reducer: Callable[
            [npt.NDArray[Any], Tuple[int, ...], bool], npt.NDArray[Any]
        ],
    ) -> None:
        if ctx.can_emit_native(node.output[0]) and ctx.can_run_native(node):
            node_inputs = [ctx.ggml_tensors_dict[inp] for inp in node.input]
            if len(node_inputs) in {1, 2}:
                input_name = node.input[0]
                input_tensor = node_inputs[0]
                tensor_shape = ctx.shapes[input_name]
                tensor_dtype = np.dtype(ctx.get_tensor_dtype(input_name))
                axes, noop = self.reduce_axes(ctx, node, node_inputs, len(tensor_shape))
                keepdims = bool(
                    next(
                        (attr.i for attr in node.attribute if attr.name == "keepdims"),
                        1,
                    )
                )
                if (
                    tensor_dtype == np.dtype(np.float32)
                    and not noop
                    and set(axes) == set(range(len(tensor_shape)))
                ):
                    output_shape = self.reduce_output_shape(
                        tensor_shape, axes, keepdims
                    )
                    result = native_reducer(ctx, input_tensor)
                    ctx.register_native_tensor(
                        node.output[0],
                        result,
                        output_shape,
                        tensor_dtype,
                    )
                    return

        self.lower_reduce(ctx, node, numpy_reducer)

    def lower_reduce(
        self,
        ctx: "GgmlOnnxExecutionContext",
        node: NodeProto,
        reducer: Callable[[npt.NDArray[Any], Tuple[int, ...], bool], npt.NDArray[Any]],
    ):
        node_inputs = [ctx.ggml_tensors_dict[inp] for inp in node.input]

        if len(node_inputs) > 2 or len(node_inputs) < 1:
            raise ValueError(
                f'Error for node "{node.name}": Operation "{self.op_type}" requires at least one input. Actual number of inputs: {len(node_inputs)}'
            )

        output_name = node.output[0]
        input_tensor = node_inputs[0]
        tensor_shape = ctx.shapes[node.input[0]]
        tensor_dtype = np.dtype(ctx.get_tensor_dtype(node.input[0]))
        axes, noop = self.reduce_axes(ctx, node, node_inputs, len(tensor_shape))
        keepdims = bool(
            next((attr.i for attr in node.attribute if attr.name == "keepdims"), 1)
        )

        if noop:
            ctx.alias_numpy_runtime_tensor(output_name, node.input[0], tensor_shape)
            return

        output_shape = self.reduce_output_shape(tensor_shape, axes, keepdims)
        if any(dim == 0 for dim in tensor_shape):
            tensor = np.empty(tensor_shape, dtype=tensor_dtype)
            try:
                result = reducer(tensor, axes, keepdims)
            except ValueError:
                if self.op_type not in {"ReduceMax", "ReduceMin"}:
                    raise
                if tensor_dtype == np.dtype(np.bool_):
                    identity = False if self.op_type == "ReduceMax" else True
                elif np.issubdtype(tensor_dtype, np.floating):
                    identity = -np.inf if self.op_type == "ReduceMax" else np.inf
                elif np.issubdtype(tensor_dtype, np.unsignedinteger):
                    info = np.iinfo(tensor_dtype)
                    identity = info.min if self.op_type == "ReduceMax" else info.max
                elif np.issubdtype(tensor_dtype, np.signedinteger):
                    info = np.iinfo(tensor_dtype)
                    identity = info.min if self.op_type == "ReduceMax" else info.max
                else:
                    raise
                result = np.full(output_shape, identity, dtype=tensor_dtype)
                ctx.set_numpy_runtime_output(output_name, result, result.dtype)
                return
            else:
                ctx.set_numpy_runtime_output(output_name, result, result.dtype)
                return

        x = np.empty(output_shape, dtype=tensor_dtype)
        x_t = ctx.from_numpy(x)

        @ggml.ggml_custom2_op_t
        def custom_reduce(
            tensor_out: ggml.ggml_tensor_p,
            tensor_in_1: ggml.ggml_tensor_p,
            tensor_in_2: ggml.ggml_tensor_p,
            ith: int,
            nth: int,
            userdata: Optional[ctypes.c_void_p],
        ):
            tensor = ctx.to_numpy(tensor_in_2).reshape(tensor_shape)
            result = reducer(tensor, axes, keepdims)

            ctx.set_tensor_data(tensor_out, result)

        new_tensor = ggml.ggml_map_custom2_inplace(
            ctx.ggml_eval_context,
            x_t,
            input_tensor,
            custom_reduce,
            1,
            None,
        )

        ctx.refs.append(custom_reduce)
        ctx.register_numpy_runtime_tensor(
            output_name,
            new_tensor,
            output_shape,
            tensor_dtype,
        )

    def lower_scatter_elements_like(
        self,
        ctx: "GgmlOnnxExecutionContext",
        node: NodeProto,
        reduction: str = "none",
    ):
        node_inputs = [ctx.ggml_tensors_dict[inp] for inp in node.input]

        if len(node_inputs) != 3:
            raise ValueError(
                f'Error for node "{node.name}": Operation "{self.op_type}" requires exactly three inputs. Actual number of inputs: {len(node_inputs)}'
            )

        data_shape = ctx.shapes[node.input[0]]
        indices_shape = ctx.shapes[node.input[1]]
        updates_shape = ctx.shapes[node.input[2]]
        axis = next((attr.i for attr in node.attribute if attr.name == "axis"), 0)
        if axis < 0:
            axis += len(data_shape)

        output = (
            ctx.to_numpy(ctx.eval_tensor(node_inputs[0])).reshape(data_shape).copy()
        )
        indices = ctx.to_numpy(ctx.eval_tensor(node_inputs[1])).reshape(indices_shape)
        updates = ctx.to_numpy(ctx.eval_tensor(node_inputs[2])).reshape(updates_shape)

        for source_index in np.ndindex(*indices.shape):
            target_index = list(source_index)
            target_axis_index = int(indices[source_index])
            if target_axis_index < 0:
                target_axis_index += data_shape[axis]
            target_index[axis] = target_axis_index
            target_index = tuple(target_index)
            update = updates[source_index]
            if reduction in {"none", ""}:
                output[target_index] = update
            elif reduction == "add":
                output[target_index] += update
            elif reduction == "mul":
                output[target_index] *= update
            elif reduction == "max":
                output[target_index] = np.maximum(output[target_index], update)
            elif reduction == "min":
                output[target_index] = np.minimum(output[target_index], update)
            else:
                raise NotImplementedError(
                    f'Scatter reduction "{reduction}" is not supported'
                )

        new_tensor = ctx.ggml_tensors_dict[node.output[0]] = ctx.from_numpy(output)
        ctx.set_tensor_shape(new_tensor, output.shape)
        ctx.shapes[node.output[0]] = output.shape
        ctx.set_tensor_dtype(node.output[0], output.dtype)


class ViewOnnxOperator(OnnxOperator):
    def __init__(self, op_type: str, view_kind: str):
        super().__init__(
            op_type,
            OnnxOperator.EXECUTION_NATIVE_OR_NUMPY_RUNTIME,
            OnnxOperator.CLASS_CONDITIONAL_NATIVE_VIEW,
            view_kind=view_kind,
        )
        self.has_numpy_evaluator = True

    def validate_view_inputs(
        self, node: NodeProto, node_inputs: Sequence[ggml.ggml_tensor_p]
    ) -> None:
        del node, node_inputs
        raise NotImplementedError(
            f'Operator "{self.op_type}" has no view input validator'
        )

    def has_static_parameters(
        self,
        tensor_types: Dict[str, TensorType],
        node: "NodeIR",
        input_type: TensorType,
    ) -> bool:
        del tensor_types, node, input_type
        return False

    def static_permutation(
        self, tensor_types: Dict[str, TensorType], node: "NodeIR"
    ) -> Optional[Tuple[int, ...]]:
        del tensor_types, node
        return None

    def runtime_permutation(
        self,
        ctx: "GgmlOnnxExecutionContext",
        node: NodeProto,
        input_shape: Tuple[int, ...],
    ) -> Optional[Tuple[int, ...]]:
        del ctx, node, input_shape
        return None

    def runtime_output_shape(
        self,
        ctx: "GgmlOnnxExecutionContext",
        node: NodeProto,
        input_shape: Tuple[int, ...],
    ) -> Tuple[int, ...]:
        del ctx, node, input_shape
        raise NotImplementedError(
            f'Operator "{self.op_type}" has no runtime output shape resolver'
        )

    @staticmethod
    def storage_shape(shape: Tuple[int, ...]) -> Tuple[int, ...]:
        if not shape:
            return (1,)
        if len(shape) <= ViewTransformSemantics.GGML_MAX_DIMS:
            return shape
        return (int(np.prod(shape)),)

    @staticmethod
    def reshape_tensor_view(
        ctx: "GgmlOnnxExecutionContext",
        tensor: ggml.ggml_tensor_p,
        shape: Tuple[int, ...],
    ) -> ggml.ggml_tensor_p:
        storage_shape = ViewOnnxOperator.storage_shape(tuple(int(dim) for dim in shape))
        ne = tuple(reversed(storage_shape))
        if len(ne) == 1:
            return ggml.ggml_reshape_1d(ctx.ggml_eval_context, tensor, ne[0])
        if len(ne) == 2:
            return ggml.ggml_reshape_2d(ctx.ggml_eval_context, tensor, ne[0], ne[1])
        if len(ne) == 3:
            return ggml.ggml_reshape_3d(
                ctx.ggml_eval_context, tensor, ne[0], ne[1], ne[2]
            )
        if len(ne) == 4:
            return ggml.ggml_reshape_4d(
                ctx.ggml_eval_context, tensor, ne[0], ne[1], ne[2], ne[3]
            )
        raise ValueError(
            f"GGML tensors support at most {ViewTransformSemantics.GGML_MAX_DIMS} dimensions"
        )

    @staticmethod
    def transpose_tensor_view(
        ctx: "GgmlOnnxExecutionContext",
        tensor: ggml.ggml_tensor_p,
        input_shape: Tuple[int, ...],
        permutation: Tuple[int, ...],
    ) -> ggml.ggml_tensor_p:
        rank = len(input_shape)
        if rank > ViewTransformSemantics.GGML_MAX_DIMS:
            raise ValueError(
                f"GGML tensors support at most {ViewTransformSemantics.GGML_MAX_DIMS} dimensions"
            )

        axes = ViewTransformSemantics.ggml_transpose_axes(permutation)
        if axes is None:
            raise ValueError(f"Unsupported native transpose permutation: {permutation}")

        return ggml.ggml_permute(ctx.ggml_eval_context, tensor, *axes)

    def static_semantics(
        self, tensor_types: Dict[str, TensorType], node: "NodeIR"
    ) -> ViewTransformSemantics:
        input_type = self.tensor_type(tensor_types, node.inputs[0])
        output_type = (
            self.tensor_type(tensor_types, node.outputs[0])
            if node.outputs
            else TensorType(shape=None, dtype=input_type.dtype)
        )
        return ViewTransformSemantics(
            op_type=self.op_type,
            view_kind=self.view_kind or "",
            input_type=input_type,
            output_type=output_type,
            has_static_parameters=self.has_static_parameters(
                tensor_types, node, input_type
            ),
            permutation=self.static_permutation(tensor_types, node),
        )

    def runtime_semantics(
        self, ctx: "GgmlOnnxExecutionContext", node: NodeProto
    ) -> ViewTransformSemantics:
        input_name = node.input[0]
        input_type = self.runtime_tensor_type(ctx, input_name)
        if input_type.shape is None:
            raise ValueError(
                f'Error for node "{node.name}": View input shape is unknown'
            )
        output_shape = self.runtime_output_shape(ctx, node, input_type.shape)
        return ViewTransformSemantics(
            op_type=self.op_type,
            view_kind=self.view_kind or "",
            input_type=input_type,
            output_type=TensorType(shape=output_shape, dtype=input_type.dtype),
            has_static_parameters=True,
            permutation=self.runtime_permutation(ctx, node, input_type.shape),
        )

    def strategies(
        self, tensor_types: Dict[str, TensorType], node: "NodeIR"
    ) -> Tuple[Tuple[str, str, str], ...]:
        semantics = self.static_semantics(tensor_types, node)
        if semantics.can_lower_native:
            return self.native_view_strategy()
        return self.numpy_runtime_strategy(semantics.native_blocker_reason)

    def lower_view_transform(
        self,
        ctx: "GgmlOnnxExecutionContext",
        node: NodeProto,
        tensor: ggml.ggml_tensor_p,
        transform: ViewTransformSemantics,
    ) -> ggml.ggml_tensor_p:
        output_name = node.output[0]
        input_name = node.input[0]
        output_shape = transform.output_shape
        output_dtype = transform.output_dtype
        if output_shape is None or output_dtype is None:
            raise ValueError(
                f'Error for node "{node.name}": View output shape and dtype are required'
            )

        if (
            ctx.can_emit_native(output_name)
            and ctx.can_run_native(node)
            and transform.can_lower_native
        ):
            if self.is_layout_view:
                if transform.input_type.shape is None or transform.permutation is None:
                    raise ValueError(
                        f'Error for node "{node.name}": Native {self.op_type} '
                        "requires input shape and permutation"
                    )
                result = self.transpose_tensor_view(
                    ctx,
                    tensor,
                    transform.input_type.shape,
                    transform.permutation,
                )
            else:
                result = self.reshape_tensor_view(ctx, tensor, output_shape)
            return ctx.register_native_view_tensor(
                output_name,
                result,
                output_shape,
                output_dtype,
                input_name,
            )

        if self.is_layout_view:
            if transform.input_type.shape is None or transform.permutation is None:
                raise ValueError(
                    f'Error for node "{node.name}": {self.op_type} requires input '
                    "shape and permutation"
                )
            input_array = ctx.logical_tensor_eval_data(
                input_name,
                tensor,
                transform.input_type.shape,
            )
            output_array = np.transpose(input_array, axes=transform.permutation).copy()
            return ctx.set_numpy_runtime_output(
                output_name,
                output_array,
                output_dtype,
            )

        return ctx.alias_numpy_runtime_tensor(output_name, input_name, output_shape)

    def lower(self, ctx: "GgmlOnnxExecutionContext", node: NodeProto) -> None:
        node_inputs = [ctx.ggml_tensors_dict[inp] for inp in node.input]
        self.validate_view_inputs(node, node_inputs)
        self.lower_view_transform(
            ctx,
            node,
            node_inputs[0],
            self.runtime_semantics(ctx, node),
        )


@dataclass(frozen=True)
class TensorInfo:
    name: str
    shape: Tuple[Any, ...]
    dtype: Optional[npt.DTypeLike] = None
    initializer: bool = False
    constant: bool = False
    scalar_value: Optional[Any] = None
    constant_value: Optional[npt.NDArray[Any]] = None


@dataclass(frozen=True)
class TensorState:
    NATIVE: ClassVar[str] = "native"
    NUMPY: ClassVar[str] = "numpy"
    VIEW: ClassVar[str] = "view"
    NATIVE_STATES: ClassVar[Set[str]] = {
        NATIVE,
        VIEW,
    }

    name: str
    tensor: ggml.ggml_tensor_p
    shape: Tuple[int, ...]
    dtype: npt.DTypeLike
    storage: str
    producer_execution: str
    source: Optional[str] = None

    @property
    def can_consume_native(self) -> bool:
        return (
            self.storage in self.NATIVE_STATES
            and self.producer_execution not in OnnxOperator.NUMPY_FALLBACK_EXECUTIONS
        )


@dataclass(frozen=True)
class NodeIR:
    index: int
    name: str
    op_type: str
    domain: str
    inputs: Tuple[str, ...]
    outputs: Tuple[str, ...]
    attributes: Tuple[str, ...]
    attribute_values: Dict[str, Any]

    def attribute(self, name: str, default: Any = None) -> Any:
        return self.attribute_values.get(name, default)

    def input_types(
        self, tensor_types: Dict[str, TensorType]
    ) -> Tuple[TensorType, ...]:
        return tuple(
            OnnxOperator.tensor_type(tensor_types, input_name)
            for input_name in self.inputs
            if input_name
        )

    def output_types(
        self, tensor_types: Dict[str, TensorType]
    ) -> Tuple[TensorType, ...]:
        return tuple(
            OnnxOperator.tensor_type(tensor_types, output_name)
            for output_name in self.outputs
            if output_name
        )


@dataclass(frozen=True)
class ModelIR:
    nodes: Tuple[NodeIR, ...]
    inputs: Tuple[str, ...]
    outputs: Tuple[str, ...]
    initializers: Tuple[str, ...]
    tensors: Dict[str, TensorInfo]

    def tensor(self, name: str) -> Optional[TensorInfo]:
        return self.tensors.get(name)

    def build_tensor_types(self) -> Dict[str, TensorType]:
        return {
            name: TensorType.from_info(tensor_info)
            for name, tensor_info in self.tensors.items()
        }


@dataclass(frozen=True)
class ModelOptimizationResult:
    model: ModelProto
    applied_passes: Tuple[str, ...]


@dataclass(frozen=True)
class ExecutionPlanNode:
    index: int
    name: str
    op_type: str
    domain: str
    execution: str
    operator_class: str
    inputs: Tuple[str, ...]
    outputs: Tuple[str, ...]
    allowed: bool
    reason: str = ""
    input_types: Tuple[TensorType, ...] = ()
    output_types: Tuple[TensorType, ...] = ()

    @property
    def requires_numpy_fallback(self) -> bool:
        return (
            self.execution in OnnxOperator.NUMPY_FALLBACK_EXECUTIONS
            or self.operator_class in OnnxOperator.NUMPY_FALLBACK_CLASSES
        )

    @property
    def capability(self) -> str:
        return self.operator_class

    @property
    def known_output_bytes(self) -> int:
        return sum(output.nbytes or 0 for output in self.output_types)

    @property
    def unknown_output_count(self) -> int:
        return sum(1 for output in self.output_types if output.nbytes is None)


@dataclass(frozen=True)
class FallbackIsland:
    nodes: Tuple[ExecutionPlanNode, ...]

    @property
    def operator_types(self) -> Tuple[str, ...]:
        return tuple(node.op_type for node in self.nodes)

    @property
    def node_indices(self) -> Tuple[int, ...]:
        return tuple(node.index for node in self.nodes)

    @property
    def input_names(self) -> Tuple[str, ...]:
        internal_outputs = {
            output_name
            for node in self.nodes
            for output_name in node.outputs
            if output_name
        }
        return tuple(
            sorted(
                {
                    input_name
                    for node in self.nodes
                    for input_name in node.inputs
                    if input_name and input_name not in internal_outputs
                }
            )
        )

    @property
    def output_names(self) -> Tuple[str, ...]:
        return tuple(
            output_name
            for node in self.nodes
            for output_name in node.outputs
            if output_name
        )

    @property
    def known_output_bytes(self) -> int:
        return sum(node.known_output_bytes for node in self.nodes)


@dataclass(frozen=True)
class ExecutionPlan:
    FALLBACK_COMPAT: ClassVar[str] = "compat"
    FALLBACK_STRICT: ClassVar[str] = "strict"
    FALLBACK_NATIVE: ClassVar[str] = "native"
    FALLBACK_POLICIES: ClassVar[Set[str]] = {
        FALLBACK_COMPAT,
        FALLBACK_STRICT,
        FALLBACK_NATIVE,
    }

    nodes: Tuple[ExecutionPlanNode, ...]
    fallback_policy: str = FALLBACK_COMPAT

    @property
    def unsupported_nodes(self) -> Tuple[ExecutionPlanNode, ...]:
        return tuple(
            node
            for node in self.nodes
            if node.operator_class == OnnxOperator.CLASS_UNSUPPORTED
        )

    @property
    def blocked_nodes(self) -> Tuple[ExecutionPlanNode, ...]:
        return tuple(node for node in self.nodes if not node.allowed)

    @property
    def fallback_nodes(self) -> Tuple[ExecutionPlanNode, ...]:
        return tuple(node for node in self.nodes if node.requires_numpy_fallback)

    @property
    def native_nodes(self) -> Tuple[ExecutionPlanNode, ...]:
        return tuple(
            node
            for node in self.nodes
            if node.operator_class in OnnxOperator.NATIVE_CLASSES
        )

    @property
    def requires_numpy_fallbacks(self) -> bool:
        return bool(self.fallback_nodes)

    @property
    def fallback_islands(self) -> Tuple[FallbackIsland, ...]:
        islands: List[FallbackIsland] = []
        current_nodes: List[ExecutionPlanNode] = []
        for node in self.nodes:
            if node.requires_numpy_fallback:
                current_nodes.append(node)
                continue
            if current_nodes:
                islands.append(FallbackIsland(tuple(current_nodes)))
                current_nodes = []
        if current_nodes:
            islands.append(FallbackIsland(tuple(current_nodes)))
        return tuple(islands)

    @property
    def is_supported(self) -> bool:
        return not self.blocked_nodes

    @property
    def unsupported_ops(self) -> Tuple[str, ...]:
        return tuple(sorted({node.op_type for node in self.unsupported_nodes}))

    @property
    def operator_class_counts(self) -> Dict[str, int]:
        counts = {
            OnnxOperator.CLASS_NATIVE: 0,
            OnnxOperator.CLASS_NATIVE_VIEW: 0,
            OnnxOperator.CLASS_DECOMPOSED: 0,
            OnnxOperator.CLASS_CONSTANT_FOLD: 0,
            OnnxOperator.CLASS_NUMPY_RUNTIME: 0,
            OnnxOperator.CLASS_NUMPY_EAGER: 0,
            OnnxOperator.CLASS_UNSUPPORTED: 0,
        }
        for node in self.nodes:
            counts[node.operator_class] = counts.get(node.operator_class, 0) + 1
        return counts

    @property
    def operator_class_percentages(self) -> Dict[str, float]:
        total = len(self.nodes)
        if total == 0:
            return {
                operator_class: 0.0 for operator_class in self.operator_class_counts
            }
        return {
            operator_class: count / total
            for operator_class, count in self.operator_class_counts.items()
        }

    def coverage_report(self) -> "ExecutionCoverageReport":
        return ExecutionCoverageReport(
            nodes=self.nodes,
            total_nodes=len(self.nodes),
            counts=self.operator_class_counts,
            percentages=self.operator_class_percentages,
            fallback_nodes=self.fallback_nodes,
            fallback_islands=self.fallback_islands,
            unsupported_nodes=self.unsupported_nodes,
            blocked_nodes=self.blocked_nodes,
        )

    def failure_message(self) -> str:
        messages = []
        unsupported_ops = self.unsupported_ops
        if unsupported_ops:
            messages.append("Unsupported operators: " + ", ".join(unsupported_ops))
        blocked_fallbacks = [
            node
            for node in self.blocked_nodes
            if node.execution != OnnxOperator.EXECUTION_UNSUPPORTED
        ]
        if blocked_fallbacks:
            blocked = ", ".join(
                f"{node.op_type}"
                + (f'("{node.name}")' if node.name else f"#{node.index}")
                + f" [{node.execution}]"
                for node in blocked_fallbacks
            )
            messages.append(
                f'fallback_policy="{self.fallback_policy}" does not allow: {blocked}'
            )
        return "; ".join(messages)


OperatorSummary = Tuple[
    str,
    str,
    str,
    str,
    int,
    Tuple[int, ...],
    int,
    int,
]
# Layout: op_type, operator_class, execution, reason, count,
# node_indices, known_output_bytes, unknown_output_count.


@dataclass(frozen=True)
class ExecutionCoverageReport:
    nodes: Tuple[ExecutionPlanNode, ...]
    total_nodes: int
    counts: Dict[str, int]
    percentages: Dict[str, float]
    fallback_nodes: Tuple[ExecutionPlanNode, ...]
    fallback_islands: Tuple[FallbackIsland, ...]
    unsupported_nodes: Tuple[ExecutionPlanNode, ...]
    blocked_nodes: Tuple[ExecutionPlanNode, ...]

    def operator_class_percentage(self, operator_class: str) -> float:
        return self.percentages.get(operator_class, 0.0)

    def operator_class_count(self, operator_class: str) -> int:
        return self.counts.get(operator_class, 0)

    @property
    def native_percentage(self) -> float:
        return self.operator_class_percentage(
            OnnxOperator.CLASS_NATIVE
        ) + self.operator_class_percentage(OnnxOperator.CLASS_NATIVE_VIEW)

    @property
    def fallback_percentage(self) -> float:
        return self.operator_class_percentage(
            OnnxOperator.CLASS_NUMPY_RUNTIME
        ) + self.operator_class_percentage(OnnxOperator.CLASS_NUMPY_EAGER)

    @property
    def native_blocking_nodes(self) -> Tuple[ExecutionPlanNode, ...]:
        return self.fallback_nodes + self.unsupported_nodes

    @property
    def native_blocking_ops(self) -> Tuple[str, ...]:
        return tuple(sorted({node.op_type for node in self.native_blocking_nodes}))

    def native_blocking_summary(self, limit: Optional[int] = None) -> Tuple[str, ...]:
        nodes = self.native_blocking_nodes
        if limit is not None:
            nodes = nodes[:limit]
        return tuple(
            f"{node.op_type}"
            + (f'("{node.name}")' if node.name else f"#{node.index}")
            + f": {node.reason}"
            for node in nodes
        )

    @property
    def known_output_bytes_by_class(self) -> Dict[str, int]:
        bytes_by_class = {
            OnnxOperator.CLASS_NATIVE: 0,
            OnnxOperator.CLASS_NATIVE_VIEW: 0,
            OnnxOperator.CLASS_DECOMPOSED: 0,
            OnnxOperator.CLASS_CONSTANT_FOLD: 0,
            OnnxOperator.CLASS_NUMPY_RUNTIME: 0,
            OnnxOperator.CLASS_NUMPY_EAGER: 0,
            OnnxOperator.CLASS_UNSUPPORTED: 0,
        }
        for node in self.nodes:
            bytes_by_class[node.operator_class] = (
                bytes_by_class.get(node.operator_class, 0) + node.known_output_bytes
            )
        return bytes_by_class

    @property
    def known_output_bytes(self) -> int:
        return sum(self.known_output_bytes_by_class.values())

    @property
    def known_fallback_output_bytes(self) -> int:
        return sum(node.known_output_bytes for node in self.fallback_nodes)

    @property
    def operator_summaries(self) -> Tuple[OperatorSummary, ...]:
        grouped: Dict[
            Tuple[str, str, str, str],
            List[ExecutionPlanNode],
        ] = {}
        for node in self.nodes:
            key = (node.op_type, node.operator_class, node.execution, node.reason)
            grouped.setdefault(key, []).append(node)
        return tuple(
            (
                op_type,
                operator_class,
                execution,
                reason,
                len(nodes),
                tuple(node.index for node in nodes),
                sum(node.known_output_bytes for node in nodes),
                sum(node.unknown_output_count for node in nodes),
            )
            for (op_type, operator_class, execution, reason), nodes in sorted(
                grouped.items()
            )
        )

    @property
    def blocking_operator_summaries(self) -> Tuple[OperatorSummary, ...]:
        blocking_keys = {
            (node.op_type, node.operator_class, node.execution, node.reason)
            for node in self.native_blocking_nodes
        }
        return tuple(
            summary
            for summary in self.operator_summaries
            if summary[:4] in blocking_keys
        )

    def summary(self) -> str:
        if self.total_nodes == 0:
            return "0 nodes"
        return (
            f"{self.native_percentage * 100:.1f}% native, "
            f"{self.operator_class_percentage(OnnxOperator.CLASS_DECOMPOSED) * 100:.1f}% decomposed, "
            f"{self.fallback_percentage * 100:.1f}% fallback, "
            f"{self.operator_class_percentage(OnnxOperator.CLASS_UNSUPPORTED) * 100:.1f}% unsupported"
        )


@dataclass(frozen=True)
class OnnxRuntimePipeline:
    original_model: ModelProto
    optimized_model: ModelProto
    model_ir: ModelIR
    tensor_types: Dict[str, TensorType]
    execution_plan: ExecutionPlan
    optimization_passes: Tuple[str, ...]

    @property
    def coverage_report(self) -> ExecutionCoverageReport:
        return self.execution_plan.coverage_report()


NumpyFallbackKernel = Callable[
    [NodeProto, Tuple[npt.NDArray[Any], ...]], Tuple[npt.NDArray[Any], ...]
]


class OnnxOperatorRegistry:
    def __init__(self):
        self.operators: Dict[str, OnnxOperator] = {}
        self.domain_operators: Dict[Tuple[str, str], OnnxOperator] = {}

    def register(self, operator_cls: Type[OnnxOperator]) -> Type[OnnxOperator]:
        operator = operator_cls()
        for domain in operator.domains:
            key = (domain, operator.op_type)
            if key in self.domain_operators:
                raise ValueError(
                    f'Operator "{operator.op_type}" is already registered '
                    f'for domain "{domain}"'
                )
            self.domain_operators[key] = operator
            if domain == "":
                if operator.op_type in self.operators:
                    raise ValueError(
                        f'Operator "{operator.op_type}" is already registered'
                    )
                self.operators[operator.op_type] = operator
        return operator_cls

    def get(self, op_type: str, domain: str = "") -> Optional[OnnxOperator]:
        return self.domain_operators.get((domain, op_type))


onnx_operators = OnnxOperatorRegistry()


class NumpyFallbackExecutor:
    def __init__(self):
        self.executed_islands: List[FallbackIsland] = []

    def can_execute_node(self, node: NodeProto) -> bool:
        operator = onnx_operators.get(node.op_type, node.domain)
        return bool(operator is not None and operator.has_numpy_evaluator)

    def node_kernel(self, node: NodeProto) -> NumpyFallbackKernel:
        operator = onnx_operators.get(node.op_type, node.domain)
        if operator is not None and operator.has_numpy_evaluator:
            return operator.eval_numpy
        raise KeyError(f'Operator "{node.op_type}" has no NumPy fallback kernel')

    def can_execute_island(
        self,
        island: FallbackIsland,
        nodes: Sequence[NodeProto],
    ) -> bool:
        return len(island.nodes) == len(nodes) and all(
            plan_node.op_type == node.op_type and self.can_execute_node(node)
            for plan_node, node in zip(island.nodes, nodes)
        )

    @staticmethod
    def _tensor_array(ctx: "GgmlOnnxExecutionContext", name: str) -> npt.NDArray[Any]:
        state = ctx.tensor_state(name)
        return ctx.logical_tensor_eval_data(name, state.tensor, state.shape)

    def execute_island(
        self,
        ctx: "GgmlOnnxExecutionContext",
        island: FallbackIsland,
        nodes: Sequence[NodeProto],
    ):
        arrays: Dict[str, npt.NDArray[Any]] = {
            input_name: self._tensor_array(ctx, input_name)
            for input_name in island.input_names
        }
        outputs: Dict[str, npt.NDArray[Any]] = {}

        for node in nodes:
            kernel = self.node_kernel(node)
            node_inputs = tuple(
                arrays[input_name]
                if input_name in arrays
                else self._tensor_array(ctx, input_name)
                for input_name in node.input
                if input_name
            )
            node_outputs = kernel(node, node_inputs)
            output_names = tuple(
                output_name for output_name in node.output if output_name
            )
            if len(node_outputs) != len(output_names):
                raise RuntimeError(
                    f'NumPy fallback for "{node.op_type}" returned '
                    f"{len(node_outputs)} outputs, expected {len(output_names)}"
                )
            for output_name, output_array in zip(output_names, node_outputs):
                array = np.asarray(output_array)
                arrays[output_name] = array
                outputs[output_name] = array

        for output_name, output_array in outputs.items():
            ctx.set_numpy_runtime_output(output_name, output_array, output_array.dtype)

        self.executed_islands.append(island)


class ArgOpsUserData(ctypes.Structure):
    _fields_ = [
        ("axis", ctypes.c_int),
        ("keepdims", ctypes.c_int),
        ("select_last_index", ctypes.c_int),
    ]


class DepthToSpaceUserData(ctypes.Structure):
    _fields_ = [
        ("blocksize", ctypes.c_int),
        ("mode", ctypes.c_char_p),
    ]


class DropoutUserData(ctypes.Structure):
    _fields_ = [
        ("seed", ctypes.c_int),
        ("training_mode", ctypes.c_bool),
        ("ratio", ctypes.c_float),
    ]


class LRNUserData(ctypes.Structure):
    _fields_ = [
        ("alpha", ctypes.c_double),
        ("beta", ctypes.c_double),
        ("bias", ctypes.c_double),
        ("size", ctypes.c_int),
    ]


class SeluUserData(ctypes.Structure):
    _fields_ = [
        ("alpha", ctypes.c_double),
        ("gamma", ctypes.c_double),
    ]


class SplitUserData(ctypes.Structure):
    _fields_ = [
        ("axis", ctypes.c_int),
        ("split_index", ctypes.c_int),
    ]


class TopKUserData(ctypes.Structure):
    _fields_ = [
        ("axis", ctypes.c_int),
        ("largest", ctypes.c_int),
        ("sorted", ctypes.c_int),
        ("k", ctypes.c_int),
    ]


@onnx_operators.register
class AcosOperator(OnnxOperator):
    def __init__(self):
        super().__init__("Acos")

    def lower(self, ctx: "GgmlOnnxExecutionContext", node: NodeProto) -> None:
        self.lower_numpy_unary(ctx, node, np.arccos)


@onnx_operators.register
class AcoshOperator(OnnxOperator):
    def __init__(self):
        super().__init__("Acosh")

    def lower(self, ctx: "GgmlOnnxExecutionContext", node: NodeProto) -> None:
        self.lower_numpy_unary(ctx, node, np.arccosh)


@onnx_operators.register
class AbsOperator(OnnxOperator):
    def __init__(self):
        super().__init__(
            "Abs",
            OnnxOperator.EXECUTION_NATIVE_OR_NUMPY_RUNTIME,
            OnnxOperator.CLASS_CONDITIONAL_NATIVE,
        )
        self.has_numpy_evaluator = True

    def strategies(
        self, tensor_types: Dict[str, TensorType], node: "NodeIR"
    ) -> Tuple[Tuple[str, str, str], ...]:
        if (
            len(node.inputs) == 1
            and self.tensor_type(tensor_types, node.inputs[0]).is_float32
        ):
            return self.native_strategy()
        return self.numpy_runtime_strategy()

    def eval_numpy(
        self, node: NodeProto, inputs: Tuple[npt.NDArray[Any], ...]
    ) -> Tuple[npt.NDArray[Any], ...]:
        return self.eval_numpy_unary_operator(node, inputs, np.abs)

    def lower(self, ctx: "GgmlOnnxExecutionContext", node: NodeProto) -> None:
        self.lower_native_unary_or_numpy(ctx, node, ggml.ggml_abs, np.abs)


@onnx_operators.register
class AddOperator(OnnxOperator):
    def __init__(self):
        super().__init__(
            "Add",
            OnnxOperator.EXECUTION_NATIVE_OR_NUMPY_RUNTIME,
            OnnxOperator.CLASS_CONDITIONAL_NATIVE,
        )
        self.has_numpy_evaluator = True

    @staticmethod
    def ggml_add_scalar(
        ggml_context: ggml.ggml_context_p,
        tensor: ggml.ggml_tensor_p,
        scalar_tensor: ggml.ggml_tensor_p,
        scalar_value: float,
    ) -> ggml.ggml_tensor_p:
        del scalar_value
        return ggml.ggml_add1(ggml_context, tensor, scalar_tensor)

    def strategies(
        self, tensor_types: Dict[str, TensorType], node: "NodeIR"
    ) -> Tuple[Tuple[str, str, str], ...]:
        if len(node.inputs) != 2:
            return self.numpy_runtime_strategy()
        input_types = self.binary_elementwise_types(tensor_types, node)
        output_shape = self.infer_elementwise_output_shape(input_types)
        if self.is_same_shape_float32(input_types):
            return self.native_strategy()
        if self.has_float32_scalar_broadcast(input_types, output_shape):
            return self.native_strategy()
        return self.numpy_runtime_strategy()

    def eval_numpy(
        self, node: NodeProto, inputs: Tuple[npt.NDArray[Any], ...]
    ) -> Tuple[npt.NDArray[Any], ...]:
        return self.eval_numpy_binary_operator(node, inputs, np.add)

    def lower(self, ctx: "GgmlOnnxExecutionContext", node: NodeProto) -> None:
        self.lower_native_binary_or_numpy(
            ctx, node, ggml.ggml_add, np.add, scalar_func=self.ggml_add_scalar
        )


@onnx_operators.register
class AndOperator(OnnxOperator):
    def __init__(self):
        super().__init__("And")

    def lower(self, ctx: "GgmlOnnxExecutionContext", node: NodeProto) -> None:
        self.lower_numpy_binary(ctx, node, np.logical_and, np.bool_)


@onnx_operators.register
class AsinOperator(OnnxOperator):
    def __init__(self):
        super().__init__("Asin")

    def lower(self, ctx: "GgmlOnnxExecutionContext", node: NodeProto) -> None:
        self.lower_numpy_unary(ctx, node, np.arcsin)


@onnx_operators.register
class AsinhOperator(OnnxOperator):
    def __init__(self):
        super().__init__("Asinh")

    def lower(self, ctx: "GgmlOnnxExecutionContext", node: NodeProto) -> None:
        self.lower_numpy_unary(ctx, node, np.arcsinh)


@onnx_operators.register
class AtanOperator(OnnxOperator):
    def __init__(self):
        super().__init__("Atan")

    def lower(self, ctx: "GgmlOnnxExecutionContext", node: NodeProto) -> None:
        self.lower_numpy_unary(ctx, node, np.arctan)


@onnx_operators.register
class AtanhOperator(OnnxOperator):
    def __init__(self):
        super().__init__("Atanh")

    def lower(self, ctx: "GgmlOnnxExecutionContext", node: NodeProto) -> None:
        self.lower_numpy_unary(ctx, node, np.arctanh)


@onnx_operators.register
class AttentionOperator(OnnxOperator):
    def __init__(self):
        super().__init__("Attention", domains=("", "com.microsoft"))
        self.has_numpy_evaluator = True

    @staticmethod
    def head_count(node: NodeProto, name: str, default: int) -> int:
        return OnnxOperator.int_attribute(node, name, default)

    @staticmethod
    def as_heads(
        value: npt.NDArray[Any],
        num_heads: int,
    ) -> Tuple[npt.NDArray[Any], str]:
        if value.ndim == 3:
            batch, sequence, hidden = value.shape
            if hidden % num_heads != 0:
                raise ValueError("Attention hidden size must be divisible by heads")
            head_size = hidden // num_heads
            return value.reshape(batch, sequence, num_heads, head_size).transpose(
                0, 2, 1, 3
            ), "BSD"
        if value.ndim == 4:
            if value.shape[1] == num_heads:
                return value, "BHSD"
            if value.shape[2] == num_heads:
                return value.transpose(0, 2, 1, 3), "BSHD"
        raise ValueError("Attention expects rank-3 or rank-4 Q/K/V tensors")

    @staticmethod
    def restore_heads(value: npt.NDArray[Any], layout: str) -> npt.NDArray[Any]:
        if layout == "BSD":
            batch, heads, sequence, head_size = value.shape
            return value.transpose(0, 2, 1, 3).reshape(
                batch, sequence, heads * head_size
            )
        if layout == "BHSD":
            return value
        if layout == "BSHD":
            return value.transpose(0, 2, 1, 3)
        raise ValueError(f'Unsupported Attention layout "{layout}"')

    @staticmethod
    def attention_mask_bias(
        mask: npt.NDArray[Any],
        dtype: npt.DTypeLike,
    ) -> npt.NDArray[Any]:
        if mask.dtype == np.dtype(np.bool_):
            return np.where(mask, 0.0, -np.inf).astype(dtype)
        return mask.astype(dtype)

    def eval_numpy(
        self, node: NodeProto, inputs: Tuple[npt.NDArray[Any], ...]
    ) -> Tuple[npt.NDArray[Any], ...]:
        if len(inputs) < 3:
            raise ValueError(f'Operation "{node.op_type}" requires Q, K, and V inputs')
        q, k, v = inputs[:3]
        q_num_heads = self.head_count(node, "q_num_heads", 1)
        kv_num_heads = self.head_count(node, "kv_num_heads", q_num_heads)
        q_heads, q_layout = self.as_heads(q, q_num_heads)
        k_heads, _ = self.as_heads(k, kv_num_heads)
        v_heads, _ = self.as_heads(v, kv_num_heads)
        if q_num_heads != kv_num_heads:
            if q_num_heads % kv_num_heads != 0:
                raise ValueError("q_num_heads must be divisible by kv_num_heads")
            repeats = q_num_heads // kv_num_heads
            k_heads = np.repeat(k_heads, repeats, axis=1)
            v_heads = np.repeat(v_heads, repeats, axis=1)

        head_size = q_heads.shape[-1]
        scale = self.float_attribute(node, "scale", 1.0 / math.sqrt(head_size))
        scores = np.matmul(q_heads, np.swapaxes(k_heads, -1, -2)) * scale
        if len(inputs) >= 4:
            mask = self.attention_mask_bias(inputs[3], scores.dtype)
            if mask.ndim == 2:
                mask = mask.reshape(1, 1, *mask.shape)
            elif mask.ndim == 3:
                mask = mask.reshape(mask.shape[0], 1, mask.shape[1], mask.shape[2])
            scores = scores + mask
        if self.int_attribute(node, "is_causal", 0):
            q_length = scores.shape[-2]
            kv_length = scores.shape[-1]
            causal = np.triu(np.ones((q_length, kv_length), dtype=np.bool_), k=1)
            scores = np.where(causal, -np.inf, scores)

        scores = scores - np.max(scores, axis=-1, keepdims=True)
        probabilities = np.exp(scores)
        probabilities = probabilities / np.sum(probabilities, axis=-1, keepdims=True)
        output = np.matmul(probabilities, v_heads).astype(q.dtype)
        outputs: Tuple[npt.NDArray[Any], ...] = (self.restore_heads(output, q_layout),)
        if len(node.output) > 1 and node.output[1]:
            outputs = (*outputs, k.astype(k.dtype))
        if len(node.output) > 2 and node.output[2]:
            outputs = (*outputs, v.astype(v.dtype))
        if len(node.output) > 3 and node.output[3]:
            outputs = (*outputs, scores.astype(q.dtype))
        return outputs

    def lower(self, ctx: "GgmlOnnxExecutionContext", node: NodeProto) -> None:
        arrays = tuple(
            ctx.logical_tensor_eval_data(
                name, ctx.ggml_tensors_dict[name], ctx.shapes[name]
            )
            for name in node.input
            if name
        )
        outputs = self.eval_numpy(node, arrays)
        output_names = tuple(name for name in node.output if name)
        for output_name, output in zip(output_names, outputs):
            ctx.set_numpy_runtime_output(output_name, output, output.dtype)


@onnx_operators.register
class AveragePoolOperator(OnnxOperator):
    def __init__(self):
        super().__init__(
            "AveragePool",
            OnnxOperator.EXECUTION_NATIVE_OR_NUMPY_RUNTIME,
            OnnxOperator.CLASS_CONDITIONAL_NATIVE,
        )
        self.has_numpy_evaluator = True

    def strategies(
        self, tensor_types: Dict[str, TensorType], node: "NodeIR"
    ) -> Tuple[Tuple[str, str, str], ...]:
        return self.pool_strategies(tensor_types, node, "Average")

    def eval_numpy(
        self, node: NodeProto, inputs: Tuple[npt.NDArray[Any], ...]
    ) -> Tuple[npt.NDArray[Any], ...]:
        return self.eval_numpy_pool(node, inputs, "Average")

    def lower(self, ctx: "GgmlOnnxExecutionContext", node: NodeProto) -> None:
        self.lower_pool(ctx, node, "Average")


@onnx_operators.register
class BatchNormalizationOperator(OnnxOperator):
    def __init__(self):
        super().__init__("BatchNormalization")

    def lower(self, ctx: "GgmlOnnxExecutionContext", node: NodeProto) -> None:
        node_inputs = [ctx.ggml_tensors_dict[inp] for inp in node.input]

        if len(node_inputs) != 5:
            raise ValueError(
                f'Error for node "{node.name}": Operation "BatchNormalization" requires exactly five inputs. Actual number of inputs: {len(node_inputs)}'
            )

        training_mode = next(
            (attr.i for attr in node.attribute if attr.name == "training_mode"), 0
        )

        x, scale, bias, input_mean, input_var = node_inputs
        x_shape = ctx.shapes[node.input[0]]
        x_dtype = np.dtype(ctx.get_tensor_dtype(node.input[0]))
        epsilon = next(
            (attr.f for attr in node.attribute if attr.name == "epsilon"), 1e-5
        )
        momentum = next(
            (attr.f for attr in node.attribute if attr.name == "momentum"), 0.9
        )

        param_shape = (1, x_shape[1], *((1,) * (len(x_shape) - 2)))
        scale_data = ctx.to_numpy(ctx.eval_tensor(scale)).reshape(param_shape)
        bias_data = ctx.to_numpy(ctx.eval_tensor(bias)).reshape(param_shape)
        mean_data = ctx.to_numpy(ctx.eval_tensor(input_mean)).reshape(param_shape)
        var_data = ctx.to_numpy(ctx.eval_tensor(input_var)).reshape(param_shape)

        if training_mode:
            x_data = ctx.logical_tensor_eval_data(node.input[0], x, x_shape).astype(
                np.float32
            )
            reduction_axes = tuple(axis for axis in range(len(x_shape)) if axis != 1)
            current_mean = np.mean(x_data, axis=reduction_axes)
            current_var = np.var(x_data, axis=reduction_axes)
            current_mean_data = current_mean.reshape(param_shape)
            current_var_data = current_var.reshape(param_shape)
            y = (
                scale_data
                * (x_data - current_mean_data)
                / np.sqrt(current_var_data + epsilon)
                + bias_data
            )
            ctx.set_logical_output(node.output[0], y.astype(x_dtype), x_dtype)
            if len(node.output) > 1 and node.output[1]:
                running_mean = mean_data.reshape(
                    current_mean.shape
                ) * momentum + current_mean * (1 - momentum)
                ctx.set_logical_output(
                    node.output[1], running_mean.astype(x_dtype), x_dtype
                )
            if len(node.output) > 2 and node.output[2]:
                running_var = var_data.reshape(
                    current_var.shape
                ) * momentum + current_var * (1 - momentum)
                ctx.set_logical_output(
                    node.output[2], running_var.astype(x_dtype), x_dtype
                )
            return

        x_t = ctx.from_numpy(np.empty(x_shape, dtype=x_dtype))

        @ggml.ggml_custom2_op_t
        def custom_batchnorm(
            tensor_out: ggml.ggml_tensor_p,
            tensor_in_1: ggml.ggml_tensor_p,
            tensor_in_2: ggml.ggml_tensor_p,
            ith: int,
            nth: int,
            userdata: Optional[ctypes.c_void_p],
        ):
            x = ctx.to_numpy(tensor_in_2).reshape(x_shape)
            y = scale_data * (x - mean_data) / np.sqrt(var_data + epsilon) + bias_data
            ctx.set_tensor_data(tensor_out, np.asarray(y, dtype=x_dtype))

        new_tensor = ctx.ggml_tensors_dict[node.output[0]] = (
            ggml.ggml_map_custom2_inplace(
                ctx.ggml_eval_context,
                x_t,
                x,
                custom_batchnorm,
                1,
                None,
            )
        )
        ctx.refs.append(custom_batchnorm)
        ctx.set_tensor_shape(new_tensor, x_shape)
        ctx.shapes[node.output[0]] = x_shape


@onnx_operators.register
class BitwiseAndOperator(OnnxOperator):
    def __init__(self):
        super().__init__("BitwiseAnd")

    def lower(self, ctx: "GgmlOnnxExecutionContext", node: NodeProto) -> None:
        self.lower_numpy_integer_binary(ctx, node, np.bitwise_and)


@onnx_operators.register
class BitwiseNotOperator(OnnxOperator):
    def __init__(self):
        super().__init__("BitwiseNot")

    def lower(self, ctx: "GgmlOnnxExecutionContext", node: NodeProto) -> None:
        self.lower_numpy_integer_unary(ctx, node, np.bitwise_not)


@onnx_operators.register
class BitwiseOrOperator(OnnxOperator):
    def __init__(self):
        super().__init__("BitwiseOr")

    def lower(self, ctx: "GgmlOnnxExecutionContext", node: NodeProto) -> None:
        self.lower_numpy_integer_binary(ctx, node, np.bitwise_or)


@onnx_operators.register
class BitwiseXorOperator(OnnxOperator):
    def __init__(self):
        super().__init__("BitwiseXor")

    def lower(self, ctx: "GgmlOnnxExecutionContext", node: NodeProto) -> None:
        self.lower_numpy_integer_binary(ctx, node, np.bitwise_xor)


@onnx_operators.register
class BitShiftOperator(OnnxOperator):
    def __init__(self):
        super().__init__("BitShift")

    def lower(self, ctx: "GgmlOnnxExecutionContext", node: NodeProto) -> None:
        direction = next(
            attr.s.decode("utf-8")
            for attr in node.attribute
            if attr.name == "direction"
        )
        if direction == "LEFT":
            bitshift = np.left_shift
        elif direction == "RIGHT":
            bitshift = np.right_shift
        else:
            raise ValueError(
                f'Error for node "{node.name}": BitShift direction must be LEFT or RIGHT.'
            )
        self.lower_numpy_integer_binary(ctx, node, bitshift)


@onnx_operators.register
class BlackmanWindowOperator(OnnxOperator):
    def __init__(self):
        super().__init__("BlackmanWindow")

    def lower(self, ctx: "GgmlOnnxExecutionContext", node: NodeProto) -> None:
        def blackman(values: npt.NDArray[Any], denominator: float) -> npt.NDArray[Any]:
            return (
                0.42
                - 0.5 * np.cos(2 * np.pi * values / denominator)
                + 0.08 * np.cos(4 * np.pi * values / denominator)
            )

        self.lower_window(ctx, node, blackman)


@onnx_operators.register
class ArgMaxOperator(OnnxOperator):
    def __init__(self):
        super().__init__(
            "ArgMax",
            OnnxOperator.EXECUTION_NATIVE_OR_NUMPY_RUNTIME,
            OnnxOperator.CLASS_CONDITIONAL_NATIVE,
        )
        self.has_numpy_evaluator = True

    @staticmethod
    def argmax_output_shape(
        input_shape: Tuple[int, ...], axis: int, keepdims: bool
    ) -> Tuple[int, ...]:
        axis = axis + len(input_shape) if axis < 0 else axis
        if keepdims:
            output_shape = list(input_shape)
            output_shape[axis] = 1
            return tuple(output_shape)
        return input_shape[:axis] + input_shape[axis + 1 :]

    def strategies(
        self, tensor_types: Dict[str, TensorType], node: "NodeIR"
    ) -> Tuple[Tuple[str, str, str], ...]:
        if len(node.inputs) == 1:
            input_type = self.tensor_type(tensor_types, node.inputs[0])
            axis = int(node.attribute("axis", 0))
            keepdims = int(node.attribute("keepdims", 1))
            select_last_index = int(node.attribute("select_last_index", 0))
            if (
                input_type.is_float32
                and input_type.shape is not None
                and len(input_type.shape) == 2
                and axis in {-1, 1}
                and keepdims == 0
                and select_last_index == 1
            ):
                return self.native_strategy()
        return self.numpy_runtime_strategy(
            "ArgMax requires 2D float32 input, axis=-1/1, keepdims=0, "
            "and select_last_index=1 to lower to ggml_argmax"
        )

    def eval_numpy(
        self, node: NodeProto, inputs: Tuple[npt.NDArray[Any], ...]
    ) -> Tuple[npt.NDArray[Any], ...]:
        if len(inputs) != 1:
            raise ValueError(f'Operation "{node.op_type}" requires one input')
        axis = self.int_attribute(node, "axis", 0)
        keepdims = bool(self.int_attribute(node, "keepdims", 1))
        select_last_index = bool(self.int_attribute(node, "select_last_index", 0))
        data = inputs[0]
        if select_last_index:
            data = np.flip(data, axis)
        result = np.argmax(data, axis=axis)
        if select_last_index:
            axis_value = axis + inputs[0].ndim if axis < 0 else axis
            result = inputs[0].shape[axis_value] - result - 1
        if keepdims:
            result = np.expand_dims(result, axis)
        return (np.asarray(result, dtype=np.int64),)

    def lower(self, ctx: "GgmlOnnxExecutionContext", node: NodeProto) -> None:
        node_inputs = [ctx.ggml_tensors_dict[inp] for inp in node.input]

        if len(node_inputs) != 1:
            raise ValueError(
                f'Error for node "{node.name}": Operation "ArgMax" requires exactly two inputs, data and axes. Actual number of inputs: {len(node_inputs)}'
            )

        data = node_inputs[0]
        name = node.output[0]

        axis = next((attr.i for attr in node.attribute if attr.name == "axis"), 0)
        keepdims = next(
            (attr.i for attr in node.attribute if attr.name == "keepdims"), 1
        )
        select_last_index = next(
            (attr.i for attr in node.attribute if attr.name == "select_last_index"), 0
        )

        x_shape = ctx.shapes[node.input[0]]
        x_dtype = np.dtype(ctx.get_tensor_dtype(node.input[0]))

        if (
            ctx.can_emit_native(name)
            and ctx.can_run_native(node)
            and x_dtype == np.dtype(np.float32)
            and len(x_shape) == 2
            and axis in {-1, 1}
            and not keepdims
            and select_last_index
        ):
            result = ggml.ggml_argmax(ctx.ggml_eval_context, data)
            ctx.register_native_tensor(
                name,
                result,
                self.argmax_output_shape(x_shape, axis, bool(keepdims)),
                np.dtype(np.int64),
            )
            return

        dummpy_data = np.empty(x_shape, dtype=np.int32)

        if select_last_index:
            dummpy_data = np.flip(dummpy_data, axis)

        dummy_result = np.argmax(dummpy_data, axis=axis)

        if select_last_index:
            dummy_result = dummpy_data.shape[axis] - dummy_result - 1

        if keepdims:
            dummy_result = np.expand_dims(dummy_result, axis)

        dummy_result = dummy_result.astype(np.int32)

        x_t = ctx.from_numpy(dummy_result)

        argmax_userdata = ArgOpsUserData(axis, keepdims, select_last_index)
        userdata_p = ctypes.cast(ctypes.pointer(argmax_userdata), ctypes.c_void_p)

        @ggml.ggml_custom2_op_t
        def custom_arg_max(
            tensor_out: ggml.ggml_tensor_p,
            tensor_in_1: ggml.ggml_tensor_p,
            tensor_in_2: ggml.ggml_tensor_p,
            ith: int,
            nth: int,
            userdata: Optional[ctypes.c_void_p],
        ):
            x = ctx.to_numpy(tensor_in_2).reshape(ctx.shapes[node.input[0]])
            userdata_data_ptr = ctypes.cast(userdata, ctypes.POINTER(ArgOpsUserData))
            userdata_data = userdata_data_ptr.contents

            axis = userdata_data.axis
            keepdims = userdata_data.keepdims
            select_last_index = userdata_data.select_last_index

            if select_last_index:
                x = np.flip(x, axis)

            y = np.argmax(x, axis=axis)

            if select_last_index:
                y = x.shape[axis] - y - 1

            if keepdims:
                y = np.expand_dims(y, axis)

            y = y.astype(np.int32)

            ctx.set_tensor_data(tensor_out, y)

        new_tensor = ctx.ggml_tensors_dict[name] = ggml.ggml_map_custom2_inplace(
            ctx.ggml_eval_context,
            x_t,
            data,
            custom_arg_max,
            1,
            userdata_p,
        )
        ctx.refs.append(custom_arg_max)

        ctx.set_tensor_dtype(name, np.dtype(np.int64))
        ctx.set_tensor_shape(new_tensor, dummy_result.shape)
        ctx.shapes[name] = dummy_result.shape
        ctx.refs.append(argmax_userdata)


@onnx_operators.register
class ArgMinOperator(OnnxOperator):
    def __init__(self):
        super().__init__(
            "ArgMin",
            OnnxOperator.EXECUTION_NATIVE_OR_NUMPY_RUNTIME,
            OnnxOperator.CLASS_CONDITIONAL_NATIVE,
        )
        self.has_numpy_evaluator = True

    def strategies(
        self, tensor_types: Dict[str, TensorType], node: "NodeIR"
    ) -> Tuple[Tuple[str, str, str], ...]:
        if len(node.inputs) == 1:
            input_type = self.tensor_type(tensor_types, node.inputs[0])
            axis = int(node.attribute("axis", 0))
            keepdims = int(node.attribute("keepdims", 1))
            select_last_index = int(node.attribute("select_last_index", 0))
            if (
                input_type.is_float32
                and input_type.shape is not None
                and len(input_type.shape) == 2
                and axis in {-1, 1}
                and keepdims == 0
                and select_last_index == 1
            ):
                return self.native_strategy()
        return self.numpy_runtime_strategy(
            "ArgMin requires 2D float32 input, axis=-1/1, keepdims=0, "
            "and select_last_index=1 to lower to ggml_neg + ggml_argmax"
        )

    def eval_numpy(
        self, node: NodeProto, inputs: Tuple[npt.NDArray[Any], ...]
    ) -> Tuple[npt.NDArray[Any], ...]:
        if len(inputs) != 1:
            raise ValueError(f'Operation "{node.op_type}" requires one input')
        axis = self.int_attribute(node, "axis", 0)
        keepdims = bool(self.int_attribute(node, "keepdims", 1))
        select_last_index = bool(self.int_attribute(node, "select_last_index", 0))
        data = inputs[0]
        if select_last_index:
            data = np.flip(data, axis)
        result = np.argmin(data, axis=axis)
        if select_last_index:
            axis_value = axis + inputs[0].ndim if axis < 0 else axis
            result = inputs[0].shape[axis_value] - result - 1
        if keepdims:
            result = np.expand_dims(result, axis)
        return (np.asarray(result, dtype=np.int64),)

    def lower(self, ctx: "GgmlOnnxExecutionContext", node: NodeProto) -> None:
        node_inputs = [ctx.ggml_tensors_dict[inp] for inp in node.input]

        if len(node_inputs) != 1:
            raise ValueError(
                f'Error for node "{node.name}": Operation "ArgMin" requires exactly two inputs, data and axes. Actual number of inputs: {len(node_inputs)}'
            )

        data = node_inputs[0]
        name = node.output[0]

        axis = next((attr.i for attr in node.attribute if attr.name == "axis"), 0)
        keepdims = next(
            (attr.i for attr in node.attribute if attr.name == "keepdims"), 1
        )
        select_last_index = next(
            (attr.i for attr in node.attribute if attr.name == "select_last_index"), 0
        )

        x_shape = ctx.shapes[node.input[0]]
        x_dtype = np.dtype(ctx.get_tensor_dtype(node.input[0]))

        if (
            ctx.can_emit_native(name)
            and ctx.can_run_native(node)
            and x_dtype == np.dtype(np.float32)
            and len(x_shape) == 2
            and axis in {-1, 1}
            and not keepdims
            and select_last_index
        ):
            negated = ggml.ggml_neg(ctx.ggml_eval_context, data)
            result = ggml.ggml_argmax(ctx.ggml_eval_context, negated)
            ctx.register_native_tensor(
                name,
                result,
                ArgMaxOperator.argmax_output_shape(x_shape, axis, bool(keepdims)),
                np.dtype(np.int64),
            )
            return

        dummpy_data = np.empty(x_shape, dtype=np.int32)

        if select_last_index:
            dummpy_data = np.flip(dummpy_data, axis)

        dummy_result = np.argmin(dummpy_data, axis=axis)

        if select_last_index:
            dummy_result = dummpy_data.shape[axis] - dummy_result - 1

        if keepdims:
            dummy_result = np.expand_dims(dummy_result, axis)

        dummy_result = dummy_result.astype(np.int32)

        x_t = ctx.from_numpy(dummy_result)

        argmax_userdata = ArgOpsUserData(axis, keepdims, select_last_index)
        userdata_p = ctypes.cast(ctypes.pointer(argmax_userdata), ctypes.c_void_p)

        @ggml.ggml_custom2_op_t
        def custom_arg_min(
            tensor_out: ggml.ggml_tensor_p,
            tensor_in_1: ggml.ggml_tensor_p,
            tensor_in_2: ggml.ggml_tensor_p,
            ith: int,
            nth: int,
            userdata: Optional[ctypes.c_void_p],
        ):
            x = ctx.to_numpy(tensor_in_2).reshape(ctx.shapes[node.input[0]])
            userdata_data_ptr = ctypes.cast(userdata, ctypes.POINTER(ArgOpsUserData))
            userdata_data = userdata_data_ptr.contents

            axis = userdata_data.axis
            keepdims = userdata_data.keepdims
            select_last_index = userdata_data.select_last_index

            if select_last_index:
                x = np.flip(x, axis)

            y = np.argmin(x, axis=axis)

            if select_last_index:
                y = x.shape[axis] - y - 1

            if keepdims:
                y = np.expand_dims(y, axis)

            y = y.astype(np.int32)

            ctx.set_tensor_data(tensor_out, y)

        new_tensor = ctx.ggml_tensors_dict[name] = ggml.ggml_map_custom2_inplace(
            ctx.ggml_eval_context,
            x_t,
            data,
            custom_arg_min,
            1,
            userdata_p,
        )
        ctx.refs.append(custom_arg_min)

        ctx.set_tensor_dtype(name, np.dtype(np.int64))
        ctx.set_tensor_shape(new_tensor, dummy_result.shape)
        ctx.shapes[name] = dummy_result.shape
        ctx.refs.append(argmax_userdata)


@onnx_operators.register
class ArrayFeatureExtractorOperator(OnnxOperator):
    def __init__(self):
        super().__init__("ArrayFeatureExtractor", domains=("", "ai.onnx.ml"))

    def lower(self, ctx: "GgmlOnnxExecutionContext", node: NodeProto) -> None:
        node_inputs = [ctx.ggml_tensors_dict[inp] for inp in node.input]

        if len(node_inputs) != 2:
            raise ValueError(
                f'Error for node "{node.name}": Operation "ArrayFeatureExtractor" requires exactly two inputs. Actual number of inputs: {len(node_inputs)}'
            )

        x, indices = node_inputs
        x_shape = ctx.shapes[node.input[0]]
        indices_shape = ctx.shapes[node.input[1]]
        output_shape = (*x_shape[:-1], *indices_shape)
        x_t = ctx.from_numpy(np.empty(output_shape, dtype=get_tensor_dtype(x)))

        @ggml.ggml_custom3_op_t
        def custom_array_feature_extractor(
            tensor_out: ggml.ggml_tensor_p,
            tensor_in_1: ggml.ggml_tensor_p,
            tensor_in_2: ggml.ggml_tensor_p,
            tensor_in_3: ggml.ggml_tensor_p,
            ith: int,
            nth: int,
            userdata: Optional[ctypes.c_void_p],
        ):
            x = ctx.to_numpy(tensor_in_2).reshape(x_shape)
            indices = ctx.to_numpy(tensor_in_3).reshape(indices_shape).astype(np.int64)
            output = np.take(x, indices, axis=-1)
            ctx.set_tensor_data(tensor_out, output)

        new_tensor = ctx.ggml_tensors_dict[node.output[0]] = (
            ggml.ggml_map_custom3_inplace(
                ctx.ggml_eval_context,
                x_t,
                x,
                indices,
                custom_array_feature_extractor,
                1,
                None,
            )
        )
        ctx.refs.append(custom_array_feature_extractor)
        ctx.set_tensor_shape(new_tensor, output_shape)
        ctx.shapes[node.output[0]] = output_shape


@onnx_operators.register
class BinarizerOperator(OnnxOperator):
    def __init__(self):
        super().__init__("Binarizer", domains=("", "ai.onnx.ml"))

    def lower(self, ctx: "GgmlOnnxExecutionContext", node: NodeProto) -> None:
        node_inputs = [ctx.ggml_tensors_dict[inp] for inp in node.input]

        if len(node_inputs) != 1:
            raise ValueError(
                f'Error for node "{node.name}": Operation "Binarizer" requires exactly one input. Actual number of inputs: {len(node_inputs)}'
            )

        x = node_inputs[0]
        x_shape = ctx.shapes[node.input[0]]
        x_dtype = get_tensor_dtype(x)
        threshold = next(
            (attr.f for attr in node.attribute if attr.name == "threshold"), 0.0
        )
        x_t = ctx.from_numpy(np.empty(x_shape, dtype=x_dtype))

        @ggml.ggml_custom2_op_t
        def custom_binarizer(
            tensor_out: ggml.ggml_tensor_p,
            tensor_in_1: ggml.ggml_tensor_p,
            tensor_in_2: ggml.ggml_tensor_p,
            ith: int,
            nth: int,
            userdata: Optional[ctypes.c_void_p],
        ):
            x = ctx.to_numpy(tensor_in_2).reshape(x_shape)
            output = (x > threshold).astype(x_dtype)
            ctx.set_tensor_data(tensor_out, output)

        new_tensor = ctx.ggml_tensors_dict[node.output[0]] = (
            ggml.ggml_map_custom2_inplace(
                ctx.ggml_eval_context,
                x_t,
                x,
                custom_binarizer,
                1,
                None,
            )
        )
        ctx.refs.append(custom_binarizer)
        ctx.set_tensor_shape(new_tensor, x_shape)
        ctx.shapes[node.output[0]] = x_shape


@onnx_operators.register
class CastOperator(OnnxOperator):
    def __init__(self):
        super().__init__("Cast")

    def lower(self, ctx: "GgmlOnnxExecutionContext", node: NodeProto) -> None:
        node_inputs = [ctx.ggml_tensors_dict[inp] for inp in node.input]

        if len(node_inputs) != 1:
            raise ValueError(
                f'Error for node "{node.name}": Operation "Cast" requires exactly one input and a dtype. Actual number of inputs: {len(node_inputs)}'
            )

        onnx_type = next(attr.i for attr in node.attribute if attr.name == "to")

        np_data_type = np.dtype(tensor_dtype_to_np_dtype(onnx_type))
        input_name = node.input[0]
        input_shape = ctx.shapes[input_name]
        if np.dtype(ctx.get_tensor_dtype(input_name)) == np_data_type:
            ctx.ggml_tensors_dict[node.output[0]] = node_inputs[0]
            ctx.shapes[node.output[0]] = input_shape
            ctx.set_tensor_dtype(node.output[0], np_data_type)
            return
        storage_dtype = ctx.storage_dtype_for_logical_dtype(np_data_type)
        storage_shape = input_shape if input_shape else (1,)
        if len(storage_shape) > ViewTransformSemantics.GGML_MAX_DIMS:
            storage_shape = (int(np.prod(storage_shape)),)
        output_tensor = ctx.from_numpy(np.empty(storage_shape, dtype=storage_dtype))

        @ggml.ggml_custom2_op_t
        def custom_cast(
            tensor_out: ggml.ggml_tensor_p,
            tensor_in_1: ggml.ggml_tensor_p,
            tensor_in_2: ggml.ggml_tensor_p,
            ith: int,
            nth: int,
            userdata: Optional[ctypes.c_void_p],
        ):
            del tensor_in_1, ith, nth, userdata
            tensor = ctx.logical_tensor_data(input_name, tensor_in_2, input_shape)
            ctx.set_tensor_data(tensor_out, tensor.astype(np_data_type))

        new_tensor = ggml.ggml_map_custom2_inplace(
            ctx.ggml_eval_context,
            output_tensor,
            node_inputs[0],
            custom_cast,
            1,
            None,
        )
        ctx.refs.append(custom_cast)
        ctx.register_numpy_runtime_tensor(
            node.output[0], new_tensor, input_shape, np_data_type
        )


@onnx_operators.register
class CastLikeOperator(OnnxOperator):
    def __init__(self):
        super().__init__("CastLike")

    def lower(self, ctx: "GgmlOnnxExecutionContext", node: NodeProto) -> None:
        node_inputs = [ctx.ggml_tensors_dict[inp] for inp in node.input]

        if len(node_inputs) != 2:
            raise ValueError(
                f'Error for node "{node.name}": Operation "CastLike" requires exactly two inputs. Actual number of inputs: {len(node_inputs)}'
            )
        a, _ = node_inputs

        np_data_dtype = np.dtype(ctx.get_tensor_dtype(node.input[1]))
        input_name = node.input[0]
        input_shape = ctx.shapes[input_name]
        if np.dtype(ctx.get_tensor_dtype(input_name)) == np_data_dtype:
            ctx.ggml_tensors_dict[node.output[0]] = a
            ctx.shapes[node.output[0]] = input_shape
            ctx.set_tensor_dtype(node.output[0], np_data_dtype)
            return
        storage_dtype = ctx.storage_dtype_for_logical_dtype(np_data_dtype)
        storage_shape = input_shape if input_shape else (1,)
        if len(storage_shape) > ViewTransformSemantics.GGML_MAX_DIMS:
            storage_shape = (int(np.prod(storage_shape)),)
        output_tensor = ctx.from_numpy(np.empty(storage_shape, dtype=storage_dtype))

        @ggml.ggml_custom2_op_t
        def custom_cast_like(
            tensor_out: ggml.ggml_tensor_p,
            tensor_in_1: ggml.ggml_tensor_p,
            tensor_in_2: ggml.ggml_tensor_p,
            ith: int,
            nth: int,
            userdata: Optional[ctypes.c_void_p],
        ):
            del tensor_in_1, ith, nth, userdata
            tensor = ctx.logical_tensor_data(input_name, tensor_in_2, input_shape)
            ctx.set_tensor_data(tensor_out, tensor.astype(np_data_dtype))

        new_tensor = ggml.ggml_map_custom2_inplace(
            ctx.ggml_eval_context,
            output_tensor,
            a,
            custom_cast_like,
            1,
            None,
        )
        ctx.refs.append(custom_cast_like)
        ctx.register_numpy_runtime_tensor(
            node.output[0], new_tensor, input_shape, np_data_dtype
        )


@onnx_operators.register
class CeilOperator(OnnxOperator):
    def __init__(self):
        super().__init__("Ceil")

    def lower(self, ctx: "GgmlOnnxExecutionContext", node: NodeProto) -> None:
        node_inputs = [ctx.ggml_tensors_dict[inp] for inp in node.input]

        if len(node_inputs) != 1:
            raise ValueError(
                f'Error for node "{node.name}": Operation "Ceil" requires exactly one input. Actual number of inputs: {len(node_inputs)}'
            )
        a = node_inputs[0]
        np_dtype = get_tensor_dtype(a)

        x = np.empty(ctx.shapes[node.input[0]], dtype=np_dtype)
        x_t = ctx.from_numpy(x)

        output_shape = ctx.shapes[node.input[0]]

        @ggml.ggml_custom2_op_t
        def custom_ceil(
            tensor_out: ggml.ggml_tensor_p,
            tensor_in_1: ggml.ggml_tensor_p,
            tensor_in_2: ggml.ggml_tensor_p,
            ith: int,
            nth: int,
            userdata: Optional[ctypes.c_void_p],
        ):
            tensor = ctx.to_numpy(tensor_in_2)
            x = np.ceil(tensor)
            ctx.set_tensor_data(tensor_out, np.array(x))

        new_tensor = ctx.ggml_tensors_dict[node.output[0]] = (
            ggml.ggml_map_custom2_inplace(
                ctx.ggml_eval_context,
                x_t,
                a,
                custom_ceil,
                1,
                None,
            )
        )

        ctx.refs.append(custom_ceil)
        ctx.set_tensor_shape(new_tensor, output_shape)
        ctx.shapes[node.output[0]] = output_shape


@onnx_operators.register
class CeluOperator(OnnxOperator):
    def __init__(self):
        super().__init__("Celu")
        self.has_numpy_evaluator = True

    def eval_numpy(
        self, node: NodeProto, inputs: Tuple[npt.NDArray[Any], ...]
    ) -> Tuple[npt.NDArray[Any], ...]:
        if len(inputs) != 1:
            raise ValueError(
                f'Operation "{node.op_type}" requires exactly one input. '
                f"Actual number of inputs: {len(inputs)}"
            )
        x = inputs[0]
        alpha = self.float_attribute(node, "alpha", 1.0)
        result = np.maximum(0, x) + np.minimum(0, alpha * (np.exp(x / alpha) - 1))
        return (np.asarray(result, dtype=x.dtype),)

    def lower(self, ctx: "GgmlOnnxExecutionContext", node: NodeProto) -> None:
        alpha = self.float_attribute(node, "alpha", 1.0)

        def celu(x: npt.NDArray[Any]) -> npt.NDArray[Any]:
            return np.maximum(0, x) + np.minimum(0, alpha * (np.exp(x / alpha) - 1))

        self.lower_numpy_unary(ctx, node, celu)


@onnx_operators.register
class CenterCropPadOperator(OnnxOperator):
    def __init__(self):
        super().__init__("CenterCropPad")

    def lower(self, ctx: "GgmlOnnxExecutionContext", node: NodeProto) -> None:
        node_inputs = [ctx.ggml_tensors_dict[inp] for inp in node.input]

        if len(node_inputs) not in {2, 3}:
            raise ValueError(
                f'Error for node "{node.name}": Operation "CenterCropPad" requires two or three inputs. Actual number of inputs: {len(node_inputs)}'
            )

        x_shape = ctx.shapes[node.input[0]]
        x = ctx.logical_tensor_eval_data(node.input[0], node_inputs[0], x_shape)
        shape = tuple(
            int(dim)
            for dim in ctx.logical_tensor_eval_data(
                node.input[1], node_inputs[1], ctx.shapes[node.input[1]]
            ).ravel()
        )
        if len(node_inputs) == 3:
            axes = tuple(
                int(axis)
                for axis in ctx.logical_tensor_eval_data(
                    node.input[2], node_inputs[2], ctx.shapes[node.input[2]]
                ).ravel()
            )
        else:
            axes = tuple(
                int(axis)
                for attr in node.attribute
                if attr.name == "axes"
                for axis in attr.ints
            ) or tuple(range(len(x_shape)))

        normalized_axes = tuple(
            axis + len(x_shape) if axis < 0 else axis for axis in axes
        )
        if len(shape) == len(x_shape):
            target_shape = tuple(shape[axis] for axis in normalized_axes)
        elif len(shape) == len(normalized_axes):
            target_shape = shape
        else:
            raise ValueError(
                f'Error for node "{node.name}": CenterCropPad shape length must match axes length.'
            )

        slices: List[slice] = [slice(None)] * len(x_shape)
        pads = [(0, 0)] * len(x_shape)
        output_shape = list(x_shape)
        for axis, target_dim in zip(normalized_axes, target_shape):
            input_dim = x_shape[axis]
            output_shape[axis] = target_dim
            if target_dim < input_dim:
                start = (input_dim - target_dim) // 2
                slices[axis] = slice(start, start + target_dim)
            elif target_dim > input_dim:
                pad_before = (target_dim - input_dim) // 2
                pad_after = target_dim - input_dim - pad_before
                pads[axis] = (pad_before, pad_after)

        output = np.pad(x[tuple(slices)], pads, mode="constant")
        output = output.reshape(tuple(output_shape)).astype(
            ctx.get_tensor_dtype(node.input[0])
        )

        ctx.set_logical_output(node.output[0], output, output.dtype)


@onnx_operators.register
class ClipOperator(OnnxOperator):
    def __init__(self):
        super().__init__(
            "Clip",
            OnnxOperator.EXECUTION_NATIVE_OR_NUMPY_RUNTIME,
            OnnxOperator.CLASS_CONDITIONAL_NATIVE,
        )
        self.has_numpy_evaluator = True

    @staticmethod
    def clip_attribute(node: NodeProto, name: str) -> Optional[float]:
        return next(
            (float(attr.f) for attr in node.attribute if attr.name == name), None
        )

    def static_clip_bounds(
        self, tensor_types: Dict[str, TensorType], node: "NodeIR"
    ) -> Optional[Tuple[float, float]]:
        min_value = self.clip_attribute_value(tensor_types, node, "min", 1)
        max_value = self.clip_attribute_value(tensor_types, node, "max", 2)
        if min_value is None:
            min_value = -np.inf
        if max_value is None:
            max_value = np.inf
        if np.ndim(min_value) != 0 or np.ndim(max_value) != 0:
            return None
        return float(min_value), float(max_value)

    def clip_attribute_value(
        self,
        tensor_types: Dict[str, TensorType],
        node: "NodeIR",
        attr_name: str,
        input_index: int,
    ) -> Optional[Any]:
        attr_value = node.attribute(attr_name)
        if attr_value is not None:
            return attr_value
        if len(node.inputs) > input_index and node.inputs[input_index]:
            scalar_value = self.constant_scalar_value(
                tensor_types, node.inputs[input_index]
            )
            if scalar_value is not None:
                return scalar_value
            tensor_value = self.constant_tensor_value(
                tensor_types, node.inputs[input_index]
            )
            if tensor_value is not None and tensor_value.size == 1:
                return tensor_value.reshape(()).item()
            if tensor_value is not None:
                return tensor_value
            return None
        return None

    def strategies(
        self, tensor_types: Dict[str, TensorType], node: "NodeIR"
    ) -> Tuple[Tuple[str, str, str], ...]:
        if len(node.inputs) >= 1:
            input_type = self.tensor_type(tensor_types, node.inputs[0])
            bounds = self.static_clip_bounds(tensor_types, node)
            if input_type.is_float32 and bounds is not None and bounds[0] <= bounds[1]:
                return self.native_strategy()
        return self.numpy_runtime_strategy(
            "Clip requires float32 input and ordered scalar bounds to lower to ggml_clamp"
        )

    def runtime_clip_bounds(
        self,
        ctx: "GgmlOnnxExecutionContext",
        node: NodeProto,
        node_inputs: Sequence[Optional[ggml.ggml_tensor_p]],
    ) -> Optional[Tuple[float, float]]:
        min_value = self.clip_attribute(node, "min")
        max_value = self.clip_attribute(node, "max")
        if min_value is None and len(node_inputs) > 1 and node_inputs[1] is not None:
            min_array = ctx.logical_tensor_eval_data(
                node.input[1], node_inputs[1], ctx.shapes[node.input[1]]
            )
            if min_array.size != 1:
                return None
            min_value = float(min_array.reshape(()).item())
        if max_value is None and len(node_inputs) > 2 and node_inputs[2] is not None:
            max_array = ctx.logical_tensor_eval_data(
                node.input[2], node_inputs[2], ctx.shapes[node.input[2]]
            )
            if max_array.size != 1:
                return None
            max_value = float(max_array.reshape(()).item())
        if min_value is None:
            min_value = -np.inf
        if max_value is None:
            max_value = np.inf
        return float(min_value), float(max_value)

    def eval_numpy(
        self, node: NodeProto, inputs: Tuple[npt.NDArray[Any], ...]
    ) -> Tuple[npt.NDArray[Any], ...]:
        input_iter = iter(inputs)
        positional_inputs: List[Optional[npt.NDArray[Any]]] = []
        for input_name in node.input:
            positional_inputs.append(next(input_iter) if input_name else None)
        data = positional_inputs[0]
        if data is None:
            raise ValueError(f'Operation "{node.op_type}" requires data input')
        min_value = self.clip_attribute(node, "min")
        max_value = self.clip_attribute(node, "max")
        if (
            min_value is None
            and len(positional_inputs) > 1
            and positional_inputs[1] is not None
        ):
            min_value = positional_inputs[1]
        if (
            max_value is None
            and len(positional_inputs) > 2
            and positional_inputs[2] is not None
        ):
            max_value = positional_inputs[2]
        if min_value is None and max_value is None:
            return (np.asarray(data, dtype=data.dtype),)
        return (np.asarray(np.clip(data, min_value, max_value), dtype=data.dtype),)

    def lower(self, ctx: "GgmlOnnxExecutionContext", node: NodeProto) -> None:
        node_inputs = [
            ctx.ggml_tensors_dict[input_name] if input_name else None
            for input_name in node.input
        ]
        data = node_inputs[0]
        if data is None:
            raise ValueError(f'Error for node "{node.name}": Clip requires data input')
        shape = ctx.shapes[node.input[0]]
        dtype = np.dtype(ctx.get_tensor_dtype(node.input[0]))
        bounds = self.runtime_clip_bounds(ctx, node, node_inputs)

        if (
            bounds is not None
            and bounds[0] <= bounds[1]
            and ctx.can_emit_native(node.output[0])
            and ctx.can_run_native(node)
            and dtype == np.dtype(np.float32)
        ):
            result = ggml.ggml_clamp(ctx.ggml_eval_context, data, bounds[0], bounds[1])
            ctx.register_native_tensor(node.output[0], result, shape, dtype)
            return

        input_array = ctx.logical_tensor_eval_data(node.input[0], data, shape)
        eval_inputs: Tuple[npt.NDArray[Any], ...] = (input_array,)
        for index, tensor in enumerate(node_inputs[1:], start=1):
            if tensor is not None:
                eval_inputs = (
                    *eval_inputs,
                    ctx.logical_tensor_eval_data(
                        node.input[index], tensor, ctx.shapes[node.input[index]]
                    ),
                )
        output = self.eval_numpy(node, eval_inputs)[0]
        ctx.set_numpy_runtime_output(
            node.output[0],
            output,
            output.dtype,
        )


@onnx_operators.register
class Col2ImOperator(OnnxOperator):
    def __init__(self):
        super().__init__("Col2Im")

    def lower(self, ctx: "GgmlOnnxExecutionContext", node: NodeProto) -> None:
        node_inputs = [ctx.ggml_tensors_dict[inp] for inp in node.input]

        if len(node_inputs) != 3:
            raise ValueError(
                f'Error for node "{node.name}": Operation "Col2Im" requires exactly three inputs. Actual number of inputs: {len(node_inputs)}'
            )

        input_shape = ctx.shapes[node.input[0]]
        image_shape_shape = ctx.shapes[node.input[1]]
        block_shape_shape = ctx.shapes[node.input[2]]

        columns = ctx.to_numpy(ctx.eval_tensor(node_inputs[0])).reshape(input_shape)
        image_shape = (
            ctx.to_numpy(ctx.eval_tensor(node_inputs[1]))
            .reshape(image_shape_shape)
            .astype(np.int64)
        )
        block_shape = (
            ctx.to_numpy(ctx.eval_tensor(node_inputs[2]))
            .reshape(block_shape_shape)
            .astype(np.int64)
        )

        spatial_rank = len(image_shape)
        dilations = np.asarray(
            next(
                (attr.ints for attr in node.attribute if attr.name == "dilations"),
                [1] * spatial_rank,
            ),
            dtype=np.int64,
        )
        pads = np.asarray(
            next(
                (attr.ints for attr in node.attribute if attr.name == "pads"),
                [0] * (2 * spatial_rank),
            ),
            dtype=np.int64,
        )
        strides = np.asarray(
            next(
                (attr.ints for attr in node.attribute if attr.name == "strides"),
                [1] * spatial_rank,
            ),
            dtype=np.int64,
        )

        channels = input_shape[1] // math.prod(block_shape)
        output_shape = (input_shape[0], channels, *image_shape)

        pads_start = pads[:spatial_rank]
        pads_end = pads[spatial_rank:]
        kernel_extent = dilations * (block_shape - 1) + 1
        block_counts = (
            image_shape + pads_start + pads_end - kernel_extent
        ) // strides + 1

        if math.prod(block_counts) != input_shape[2]:
            raise ValueError(
                f'Error for node "{node.name}": Col2Im input columns do not match inferred block count.'
            )

        reshaped_columns = columns.reshape(
            input_shape[0], channels, *block_shape, *block_counts
        )
        output = np.zeros(output_shape, dtype=get_tensor_dtype(node_inputs[0]))

        for kernel_index in np.ndindex(*block_shape):
            kernel_offset = (
                np.asarray(kernel_index, dtype=np.int64) * dilations - pads_start
            )
            for block_index in np.ndindex(*block_counts):
                image_index = (
                    kernel_offset + np.asarray(block_index, dtype=np.int64) * strides
                )
                if np.all((image_index >= 0) & (image_index < image_shape)):
                    output[(slice(None), slice(None), *image_index)] += (
                        reshaped_columns[
                            (slice(None), slice(None), *kernel_index, *block_index)
                        ]
                    )

        new_tensor = ctx.ggml_tensors_dict[node.output[0]] = ctx.from_numpy(output)
        ctx.set_tensor_shape(new_tensor, output_shape)
        ctx.shapes[node.output[0]] = output_shape
        ctx.set_tensor_dtype(node.output[0], output.dtype)


@onnx_operators.register
class ConcatOperator(OnnxOperator):
    def __init__(self):
        super().__init__(
            "Concat",
            OnnxOperator.EXECUTION_NATIVE_OR_NUMPY_RUNTIME,
            OnnxOperator.CLASS_CONDITIONAL_NATIVE,
        )
        self.has_numpy_evaluator = True

    @staticmethod
    def concat_axis(axis: int, rank: int) -> Optional[int]:
        axis = axis + rank if axis < 0 else axis
        if axis < 0 or axis >= rank:
            return None
        return axis

    @staticmethod
    def concat_output_shape(
        input_shapes: Sequence[Tuple[int, ...]], axis: int
    ) -> Optional[Tuple[int, ...]]:
        if not input_shapes:
            return None
        rank = len(input_shapes[0])
        if any(len(shape) != rank for shape in input_shapes):
            return None
        output_shape = list(input_shapes[0])
        output_shape[axis] = 0
        for shape in input_shapes:
            for dim_index, dim in enumerate(shape):
                if dim_index == axis:
                    output_shape[axis] += dim
                elif dim != input_shapes[0][dim_index]:
                    return None
        return tuple(output_shape)

    def static_concat_shape(
        self, tensor_types: Dict[str, TensorType], node: "NodeIR"
    ) -> Optional[Tuple[int, Tuple[int, ...]]]:
        if len(node.inputs) < 1:
            return None
        input_types = [self.tensor_type(tensor_types, name) for name in node.inputs]
        if any(input_type.shape is None for input_type in input_types):
            return None
        if any(input_type.dtype != np.dtype(np.float32) for input_type in input_types):
            return None
        input_shapes = tuple(
            input_type.shape
            for input_type in input_types
            if input_type.shape is not None
        )
        axis = self.concat_axis(int(node.attribute("axis", 0)), len(input_shapes[0]))
        if axis is None:
            return None
        output_shape = self.concat_output_shape(input_shapes, axis)
        if output_shape is None:
            return None
        return axis, output_shape

    def strategies(
        self, tensor_types: Dict[str, TensorType], node: "NodeIR"
    ) -> Tuple[Tuple[str, str, str], ...]:
        static_shape = self.static_concat_shape(tensor_types, node)
        if (
            static_shape is not None
            and len(static_shape[1]) <= ViewTransformSemantics.GGML_MAX_DIMS
            and not any(dim == 0 for dim in static_shape[1])
        ):
            return self.native_strategy()
        return self.numpy_runtime_strategy(
            "Concat requires static non-empty float32 inputs with rank <= 4 "
            "to lower to ggml_concat"
        )

    def eval_numpy(
        self, node: NodeProto, inputs: Tuple[npt.NDArray[Any], ...]
    ) -> Tuple[npt.NDArray[Any], ...]:
        if not inputs:
            raise ValueError(f'Operation "{node.op_type}" requires inputs')
        axis = self.int_attribute(node, "axis", 0)
        if axis < 0:
            axis += inputs[0].ndim
        return (np.concatenate(inputs, axis=axis),)

    def lower(self, ctx: "GgmlOnnxExecutionContext", node: NodeProto) -> None:
        node_inputs = [ctx.ggml_tensors_dict[inp] for inp in node.input]

        axis = self.concat_axis(
            self.int_attribute(node, "axis", 0), len(ctx.shapes[node.input[0]])
        )
        if axis is None:
            raise ValueError(f'Error for node "{node.name}": Concat axis is invalid')
        shapes = [tuple(ctx.shapes[input_]) for input_ in node.input]
        output_shape = self.concat_output_shape(shapes, axis)
        if output_shape is None:
            raise ValueError(
                "All tensors must have the same shape except along the specified axis."
            )
        output_dtype = np.result_type(
            *(ctx.get_tensor_dtype(name) for name in node.input)
        )

        if (
            ctx.can_emit_native(node.output[0])
            and ctx.can_run_native(node)
            and output_dtype == np.dtype(np.float32)
            and len(output_shape) <= ViewTransformSemantics.GGML_MAX_DIMS
            and not any(dim == 0 for dim in output_shape)
        ):
            storage_axis = len(output_shape) - axis - 1
            result = node_inputs[0]
            for next_tensor in node_inputs[1:]:
                result = ggml.ggml_concat(
                    ctx.ggml_eval_context, result, next_tensor, storage_axis
                )
            ctx.register_native_tensor(
                node.output[0], result, output_shape, np.dtype(np.float32)
            )
            return

        arrays = [
            ctx.logical_tensor_eval_data(name, tensor, shape)
            for name, tensor, shape in zip(node.input, node_inputs, shapes)
        ]
        output = self.eval_numpy(node, tuple(arrays))[0]
        ctx.set_numpy_runtime_output(node.output[0], output, output.dtype)


@onnx_operators.register
class CompressOperator(OnnxOperator):
    def __init__(self):
        super().__init__("Compress")

    def lower(self, ctx: "GgmlOnnxExecutionContext", node: NodeProto) -> None:
        node_inputs = [ctx.ggml_tensors_dict[inp] for inp in node.input]

        if len(node_inputs) != 2:
            raise ValueError(
                f'Error for node "{node.name}": Operation "Compress" requires exactly two inputs. Actual number of inputs: {len(node_inputs)}'
            )

        data_shape = ctx.shapes[node.input[0]]
        condition_shape = ctx.shapes[node.input[1]]
        axis = next((attr.i for attr in node.attribute if attr.name == "axis"), None)

        data = ctx.logical_tensor_eval_data(node.input[0], node_inputs[0], data_shape)
        condition = ctx.logical_tensor_eval_data(
            node.input[1], node_inputs[1], condition_shape
        )
        output = np.compress(condition.astype(np.bool_), data, axis=axis)

        ctx.set_logical_output(node.output[0], output, output.dtype)


@onnx_operators.register
class ConstantOperator(OnnxOperator):
    def __init__(self):
        super().__init__("Constant", execution=OnnxOperator.EXECUTION_CONSTANT_FOLD)

    @staticmethod
    def tensor_proto_from_node(node: NodeProto) -> Optional[TensorProto]:
        if node.op_type != "Constant" or len(node.output) != 1:
            return None

        value_attrs = [
            attr
            for attr in node.attribute
            if attr.name.startswith("value") or attr.name == "sparse_value"
        ]
        if len(value_attrs) != 1:
            return None

        output_name = node.output[0]
        value_attr = value_attrs[0]
        if value_attr.name == "value":
            tensor = TensorProto()
            tensor.CopyFrom(value_attr.t)
            tensor.name = output_name
            return tensor
        if value_attr.name == "value_float":
            return onnx.helper.make_tensor(
                output_name, TensorProto.FLOAT, [], [value_attr.f]
            )
        if value_attr.name == "value_floats":
            return onnx.helper.make_tensor(
                output_name,
                TensorProto.FLOAT,
                [len(value_attr.floats)],
                list(value_attr.floats),
            )
        if value_attr.name == "value_int":
            return onnx.helper.make_tensor(
                output_name, TensorProto.INT64, [], [value_attr.i]
            )
        if value_attr.name == "value_ints":
            return onnx.helper.make_tensor(
                output_name,
                TensorProto.INT64,
                [len(value_attr.ints)],
                list(value_attr.ints),
            )
        return None

    def lower(self, ctx: "GgmlOnnxExecutionContext", node: NodeProto) -> None:
        tensor = self.tensor_proto_from_node(node)
        if tensor is None:
            value_attrs = [
                attr
                for attr in node.attribute
                if attr.name.startswith("value") or attr.name == "sparse_value"
            ]
            if len(value_attrs) == 1 and value_attrs[0].name == "sparse_value":
                raise NotImplementedError("Sparse constants are not supported")
            if len(value_attrs) == 1 and value_attrs[0].name in (
                "value_string",
                "value_strings",
            ):
                raise NotImplementedError("String constants are not supported")
            raise ValueError(
                f'Error for node "{node.name}": Constant node must have exactly one value attribute.'
            )

        data_value = onnx.numpy_helper.to_array(tensor)
        ctx.set_logical_output(node.output[0], data_value, data_value.dtype)


@onnx_operators.register
class ConstantOfShapeOperator(OnnxOperator):
    def __init__(self):
        super().__init__("ConstantOfShape")

    def lower(self, ctx: "GgmlOnnxExecutionContext", node: NodeProto) -> None:
        node_inputs = [ctx.ggml_tensors_dict[inp] for inp in node.input]

        if len(node_inputs) != 1:
            raise ValueError(
                f'Error for node "{node.name}": Operation "ConstantOfShape" requires exactly one input. Actual number of inputs: {len(node_inputs)}'
            )

        value_attr = next(
            (attr for attr in node.attribute if attr.name == "value"), None
        )
        if value_attr is None:
            data_value = np.asarray([0], dtype=np.float32)
        elif value_attr.HasField("t"):
            data_value = onnx.numpy_helper.to_array(value_attr.t)
            if data_value.size != 1:
                raise ValueError(
                    f'Error for node "{node.name}": ConstantOfShape value must contain exactly one element.'
                )
            if data_value.dtype.kind in ("O", "S", "U"):
                raise NotImplementedError(
                    "String ConstantOfShape values are not supported"
                )
        else:
            raise ValueError(
                f'Error for node "{node.name}": ConstantOfShape value must be a tensor attribute.'
            )

        node_inputs_0 = ctx.eval_tensor(node_inputs[0])
        shape = tuple(
            int(dim)
            for dim in ctx.to_numpy(node_inputs_0).reshape(ctx.shapes[node.input[0]])
        )
        value = data_value.reshape(-1)[0]
        new_tensor = np.full(shape, value, dtype=data_value.dtype)
        ctx.set_logical_output(node.output[0], new_tensor, data_value.dtype)


@onnx_operators.register
class CosOperator(OnnxOperator):
    def __init__(self):
        super().__init__("Cos")

    def lower(self, ctx: "GgmlOnnxExecutionContext", node: NodeProto) -> None:
        node_inputs = [ctx.ggml_tensors_dict[inp] for inp in node.input]

        if len(node_inputs) != 1:
            raise ValueError(
                f'Error for node "{node.name}": Operation "Cos" requires exactly one input. Actual number of inputs: {len(node_inputs)}'
            )

        a = node_inputs[0]
        a_shape = ctx.get_tensor_shape(a)
        a_dtype = get_tensor_dtype(a)

        x = np.empty(a_shape, dtype=a_dtype)
        x_t = ctx.from_numpy(x)

        @ggml.ggml_custom2_op_t
        def custom_cos(
            tensor_out: ggml.ggml_tensor_p,
            tensor_in_1: ggml.ggml_tensor_p,
            tensor_in_2: ggml.ggml_tensor_p,
            ith: int,
            nth: int,
            userdata: Optional[ctypes.c_void_p],
        ):
            a = ctx.to_numpy(tensor_in_2)
            y = np.cos(a)

            ctx.set_tensor_data(tensor_out, y)

        new_tensor = ctx.ggml_tensors_dict[node.output[0]] = (
            ggml.ggml_map_custom2_inplace(
                ctx.ggml_eval_context,
                x_t,
                a,
                custom_cos,
                1,
                None,
            )
        )

        ctx.refs.append(custom_cos)
        ctx.set_tensor_shape(new_tensor, ctx.shapes[node.input[0]])
        ctx.shapes[node.output[0]] = ctx.shapes[node.input[0]]


@onnx_operators.register
class CoshOperator(OnnxOperator):
    def __init__(self):
        super().__init__("Cosh")

    def lower(self, ctx: "GgmlOnnxExecutionContext", node: NodeProto) -> None:
        self.lower_numpy_unary(ctx, node, np.cosh)


@onnx_operators.register
class CumSumOperator(OnnxOperator):
    def __init__(self):
        super().__init__("CumSum")

    def lower(self, ctx: "GgmlOnnxExecutionContext", node: NodeProto) -> None:
        node_inputs = [ctx.ggml_tensors_dict[inp] for inp in node.input]

        if len(node_inputs) != 2:
            raise ValueError(
                f'Error for node "{node.name}": Operation "CumSum" requires exactly two inputs. Actual number of inputs: {len(node_inputs)}'
            )

        axis_tensor = ctx.eval_tensor(node_inputs[1])
        axis = int(ctx.to_numpy(axis_tensor).reshape(ctx.shapes[node.input[1]]).item())
        input_shape = ctx.shapes[node.input[0]]
        if axis < 0:
            axis += len(input_shape)
        input_tensor = node_inputs[0]
        input_dtype = get_tensor_dtype(input_tensor)

        exclusive = next(
            (attr.i for attr in node.attribute if attr.name == "exclusive"), 0
        )
        reverse = next((attr.i for attr in node.attribute if attr.name == "reverse"), 0)

        def cumsum(x: npt.NDArray[Any]) -> npt.NDArray[Any]:
            values = np.flip(x, axis=axis) if reverse else x
            output = np.cumsum(values, axis=axis)
            if exclusive:
                output = np.roll(output, 1, axis=axis)
                index = [slice(None)] * output.ndim
                index[axis] = 0
                output[tuple(index)] = 0
            if reverse:
                output = np.flip(output, axis=axis)
            return output

        x_t = ctx.from_numpy(np.empty(input_shape, dtype=input_dtype))

        @ggml.ggml_custom2_op_t
        def custom_cumsum(
            tensor_out: ggml.ggml_tensor_p,
            tensor_in_1: ggml.ggml_tensor_p,
            tensor_in_2: ggml.ggml_tensor_p,
            ith: int,
            nth: int,
            userdata: Optional[ctypes.c_void_p],
        ):
            x = ctx.to_numpy(tensor_in_2).reshape(input_shape)
            ctx.set_tensor_data(tensor_out, np.asarray(cumsum(x), dtype=input_dtype))

        new_tensor = ctx.ggml_tensors_dict[node.output[0]] = (
            ggml.ggml_map_custom2_inplace(
                ctx.ggml_eval_context,
                x_t,
                input_tensor,
                custom_cumsum,
                1,
                None,
            )
        )
        ctx.refs.append(custom_cumsum)
        ctx.set_tensor_shape(new_tensor, input_shape)
        ctx.shapes[node.output[0]] = input_shape


@onnx_operators.register
class ConvOperator(OnnxOperator):
    def __init__(self):
        super().__init__(
            "Conv",
            OnnxOperator.EXECUTION_NATIVE_OR_NUMPY_RUNTIME,
            OnnxOperator.CLASS_CONDITIONAL_NATIVE,
        )
        self.has_numpy_evaluator = True

    @staticmethod
    def conv_parameters(
        node: NodeProto,
        x_shape: Tuple[int, ...],
        w_shape: Tuple[int, ...],
    ) -> Tuple[List[int], int, Tuple[int, ...], List[int], List[int]]:
        auto_pad = next(
            (attr.s.decode() for attr in node.attribute if attr.name == "auto_pad"),
            "NOTSET",
        )
        dilations = list(
            next(
                (attr.ints for attr in node.attribute if attr.name == "dilations"),
                [1 for _ in x_shape[2:]],
            )
        )
        group = int(
            next((attr.i for attr in node.attribute if attr.name == "group"), 1)
        )
        kernel_shape = tuple(
            int(v)
            for v in next(
                (attr.ints for attr in node.attribute if attr.name == "kernel_shape"),
                w_shape[2:],
            )
        )
        pads = list(
            next(
                (attr.ints for attr in node.attribute if attr.name == "pads"),
                [0 for _ in x_shape[2:]] * 2,
            )
        )
        strides = list(
            next(
                (attr.ints for attr in node.attribute if attr.name == "strides"),
                [1 for _ in x_shape[2:]],
            )
        )

        if auto_pad == "VALID":
            pads = [0 for _ in x_shape[2:]] * 2
        elif auto_pad in {"SAME_LOWER", "SAME_UPPER"}:
            head = []
            tail = []
            for i, d in enumerate(x_shape[2:]):
                target_size = (d + strides[i] - 1) // strides[i]
                pad_needed = (target_size - 1) * strides[i] + kernel_shape[i] - d
                if auto_pad == "SAME_LOWER":
                    pad_head = (pad_needed + 1) // 2
                else:
                    pad_head = pad_needed // 2
                pad_tail = pad_needed - pad_head
                head.append(pad_head)
                tail.append(pad_tail)
            pads = head + tail

        return dilations, group, kernel_shape, pads, strides

    @staticmethod
    def conv_parameters_from_ir(
        node: "NodeIR",
        x_shape: Tuple[int, ...],
        w_shape: Tuple[int, ...],
    ) -> Tuple[List[int], int, Tuple[int, ...], List[int], List[int]]:
        auto_pad = str(node.attribute("auto_pad", "NOTSET"))
        dilations = list(node.attribute("dilations", (1,) * len(x_shape[2:])))
        group = int(node.attribute("group", 1))
        kernel_shape = tuple(
            int(v) for v in node.attribute("kernel_shape", w_shape[2:])
        )
        pads = list(node.attribute("pads", (0,) * (2 * len(x_shape[2:]))))
        strides = list(node.attribute("strides", (1,) * len(x_shape[2:])))

        if auto_pad == "VALID":
            pads = [0 for _ in x_shape[2:]] * 2
        elif auto_pad in {"SAME_LOWER", "SAME_UPPER"}:
            head = []
            tail = []
            for i, d in enumerate(x_shape[2:]):
                target_size = (d + strides[i] - 1) // strides[i]
                pad_needed = (target_size - 1) * strides[i] + kernel_shape[i] - d
                if auto_pad == "SAME_LOWER":
                    pad_head = (pad_needed + 1) // 2
                else:
                    pad_head = pad_needed // 2
                pad_tail = pad_needed - pad_head
                head.append(pad_head)
                tail.append(pad_tail)
            pads = head + tail

        return dilations, group, kernel_shape, pads, strides

    def strategies(
        self, tensor_types: Dict[str, TensorType], node: "NodeIR"
    ) -> Tuple[Tuple[str, str, str], ...]:
        if len(node.inputs) >= 2:
            input_type = self.tensor_type(tensor_types, node.inputs[0])
            weight_type = self.tensor_type(tensor_types, node.inputs[1])
            bias_type = (
                self.tensor_type(tensor_types, node.inputs[2])
                if len(node.inputs) >= 3 and node.inputs[2]
                else TensorType(shape=None, dtype=np.dtype(np.float32))
            )
            if (
                input_type.is_float32
                and weight_type.is_float32
                and bias_type.is_float32
                and input_type.shape is not None
                and weight_type.shape is not None
            ):
                dilations, group, kernel_shape, pads, strides = (
                    self.conv_parameters_from_ir(
                        node, input_type.shape, weight_type.shape
                    )
                )
                spatial_rank = len(input_type.shape[2:])
                if (
                    group == 1
                    and spatial_rank == 2
                    and len(strides) == spatial_rank
                    and len(dilations) == spatial_rank
                    and len(kernel_shape) == spatial_rank
                    and len(pads) == spatial_rank * 2
                    and len({int(pad) for pad in pads}) <= 1
                ):
                    return self.native_strategy()
        return self.numpy_runtime_strategy(
            "Conv requires 2D float32 input/weights/bias, group=1, and uniform "
            "padding to lower to ggml_conv_2d"
        )

    @staticmethod
    def grouped_conv_nd(
        x: npt.NDArray[Any],
        w: npt.NDArray[Any],
        bias: npt.NDArray[Any],
        pads: Sequence[int],
        strides: Sequence[int],
        dilations: Sequence[int],
        group: int,
    ) -> npt.NDArray[Any]:
        if x.ndim < 3 or w.ndim != x.ndim:
            raise NotImplementedError("Grouped convolution requires NCHW-style tensors")

        spatial_rank = x.ndim - 2
        if len(pads) != spatial_rank * 2:
            raise NotImplementedError(
                f"Grouped convolution requires {spatial_rank * 2} pad values"
            )
        if len(strides) != spatial_rank or len(dilations) != spatial_rank:
            raise NotImplementedError(
                f"Grouped convolution requires {spatial_rank} strides and dilations"
            )

        n, c, *spatial_shape = x.shape
        m, c_per_group, *kernel_shape = w.shape
        if group < 1:
            raise ValueError(f"Convolution group must be positive, got {group}")
        if c % group != 0:
            raise ValueError(f"Input channels {c} must be divisible by group {group}")
        if m % group != 0:
            raise ValueError(f"Output channels {m} must be divisible by group {group}")
        if c // group != c_per_group:
            raise ValueError(
                f"Weight channel count {c_per_group} does not match input channels per group {c // group}"
            )
        if bias.shape != (m,):
            raise ValueError(
                f"Convolution bias must have shape ({m},), got {bias.shape}"
            )

        pads = [int(pad) for pad in pads]
        strides = [int(stride) for stride in strides]
        dilations = [int(dilation) for dilation in dilations]
        output_spatial = tuple(
            (
                spatial_shape[i]
                + pads[i]
                + pads[i + spatial_rank]
                - dilations[i] * (kernel_shape[i] - 1)
                - 1
            )
            // strides[i]
            + 1
            for i in range(spatial_rank)
        )
        if any(dim < 0 for dim in output_spatial):
            raise ValueError(f"Invalid convolution output shape {output_spatial}")

        output_dtype = np.result_type(x.dtype, w.dtype, bias.dtype)
        x_padded = np.pad(
            x.astype(output_dtype, copy=False),
            (
                (0, 0),
                (0, 0),
                *((pads[i], pads[i + spatial_rank]) for i in range(spatial_rank)),
            ),
            mode="constant",
        )
        w_data = w.astype(output_dtype, copy=False)
        y = np.empty((n, m, *output_spatial), dtype=output_dtype)
        output_channels_per_group = m // group
        contraction_axes = (
            [1, *range(2, 2 + spatial_rank)],
            [1, *range(2, 2 + spatial_rank)],
        )

        for group_idx in range(group):
            c_start = group_idx * c_per_group
            m_start = group_idx * output_channels_per_group
            m_end = m_start + output_channels_per_group
            x_group = x_padded[:, c_start : c_start + c_per_group]
            w_group = w_data[m_start:m_end]
            for output_index in np.ndindex(*output_spatial):
                input_indices = tuple(
                    output_index[i] * strides[i]
                    + np.arange(kernel_shape[i]) * dilations[i]
                    for i in range(spatial_rank)
                )
                patch = x_group[(slice(None), slice(None), *np.ix_(*input_indices))]
                y[(slice(None), slice(m_start, m_end), *output_index)] = np.tensordot(
                    patch,
                    w_group,
                    axes=contraction_axes,
                )

        y += bias.astype(output_dtype, copy=False).reshape(
            (1, m, *((1,) * spatial_rank))
        )
        return y

    def eval_numpy(
        self, node: NodeProto, inputs: Tuple[npt.NDArray[Any], ...]
    ) -> Tuple[npt.NDArray[Any], ...]:
        if len(inputs) < 2:
            raise ValueError(f'Operation "{node.op_type}" requires at least two inputs')
        x = inputs[0]
        w = inputs[1]
        if len(inputs) >= 3:
            bias = inputs[2]
        else:
            bias = np.zeros((w.shape[0],), dtype=np.result_type(x.dtype, w.dtype))
        dilations, group, _kernel_shape, pads, strides = self.conv_parameters(
            node, tuple(x.shape), tuple(w.shape)
        )
        return (
            self.grouped_conv_nd(
                x,
                w,
                bias,
                pads,
                strides,
                dilations,
                group,
            ),
        )

    def lower(self, ctx: "GgmlOnnxExecutionContext", node: NodeProto) -> None:
        node_inputs = [ctx.ggml_tensors_dict[inp] for inp in node.input]

        if len(node_inputs) < 2:
            raise ValueError(
                f'Error for node "{node.name}": Operation "Conv" requires 2 - 3 inputs. Actual number of inputs: {len(node_inputs)}'
            )

        node_inputs_iter = iter(node_inputs)
        x = next(node_inputs_iter)
        x_shape = ctx.shapes[node.input[0]]
        w = next(node_inputs_iter)
        w_shape = ctx.shapes[node.input[1]]
        m = w_shape[0]
        bias = next(
            node_inputs_iter,
            ctx.from_numpy(np.full(m, 0, dtype=get_tensor_dtype(x))),
        )

        dilations, group, kernel_shape, pads, strides = self.conv_parameters(
            node, x_shape, w_shape
        )

        spatial_rank = len(x_shape[2:])
        if len(strides) != spatial_rank:
            raise NotImplementedError(
                f"Conv expected {spatial_rank} strides, got {len(strides)}"
            )
        if len(dilations) != spatial_rank:
            raise NotImplementedError(
                f"Conv expected {spatial_rank} dilations, got {len(dilations)}"
            )
        uniform_padding = len({int(pad) for pad in pads}) <= 1
        if group != 1 or spatial_rank != 2 or not uniform_padding:
            x_data = ctx.logical_tensor_eval_data(node.input[0], x, x_shape)
            w_data = ctx.logical_tensor_eval_data(node.input[1], w, w_shape)
            if len(node.input) >= 3 and node.input[2]:
                bias_data = ctx.logical_tensor_eval_data(
                    node.input[2], bias, ctx.shapes[node.input[2]]
                )
            else:
                bias_data = np.zeros(
                    (m,), dtype=np.result_type(x_data.dtype, w_data.dtype)
                )

            y = self.grouped_conv_nd(
                x_data,
                w_data,
                bias_data,
                pads,
                strides,
                dilations,
                int(group),
            )
            ctx.set_logical_output(node.output[0], y, y.dtype)
            return

        if ggml.ggml_is_permuted(x):
            x_dtype = get_tensor_dtype(x)
            x_shape = ggml.utils.get_shape(x)

            x = ggml.ggml_cpy(
                ctx.ggml_eval_context,
                x,
                ggml.ggml_new_tensor(
                    ctx.ggml_eval_context,
                    ctx.map_to_ggml_type(x_dtype).value,
                    len(x_shape),
                    (ctypes.c_int64 * len(x_shape))(*x_shape),
                ),
            )

        cur = ggml.ggml_conv_2d(
            ctx.ggml_eval_context,
            w,
            x,
            strides[0],
            strides[1],
            pads[0],
            pads[1],
            dilations[0],
            dilations[1],
        )
        bias_tensor = ggml.ggml_reshape_3d(
            ctx.ggml_eval_context, bias, 1, 1, bias.contents.ne[0]
        )
        result = ggml.ggml_add(
            ctx.ggml_eval_context,
            cur,
            ggml.ggml_repeat(
                ctx.ggml_eval_context,
                bias_tensor,
                cur,
            ),
        )

        output_spatial = tuple(
            int(
                math.floor(
                    (
                        x_shape[2 + i]
                        + pads[i]
                        + pads[i + len(kernel_shape)]
                        - dilations[i] * (kernel_shape[i] - 1)
                        - 1
                    )
                    / strides[i]
                    + 1
                )
            )
            for i in range(len(kernel_shape))
        )
        output_shape = (x_shape[0], w_shape[0], *output_spatial)
        ctx.register_native_tensor(node.output[0], result, output_shape, np.float32)


@onnx_operators.register
class ConvIntegerOperator(OnnxOperator):
    def __init__(self):
        super().__init__("ConvInteger")

    @staticmethod
    def reshape_conv_zero_point(
        zero_point: npt.NDArray[Any],
        channels: int,
    ) -> npt.NDArray[Any]:
        if zero_point.shape == ():
            return zero_point
        if zero_point.shape != (channels,):
            raise NotImplementedError(
                f"Per-channel convolution zero point must have shape ({channels},), got {zero_point.shape}"
            )
        return zero_point.reshape((channels, 1, 1, 1))

    @staticmethod
    def quantized_conv2d(
        x: npt.NDArray[Any],
        w: npt.NDArray[Any],
        x_zero_point: npt.NDArray[Any],
        w_zero_point: npt.NDArray[Any],
        pads: Sequence[int],
        strides: Sequence[int],
        dilations: Sequence[int],
        group: int,
    ) -> npt.NDArray[np.int32]:
        if x.ndim != 4 or w.ndim != 4:
            raise NotImplementedError("Only 2D NCHW quantized convolution is supported")
        if group != 1:
            raise NotImplementedError("Grouped quantized convolution is not supported")
        if len(pads) != 4:
            raise NotImplementedError("Quantized convolution requires four pad values")
        if len(strides) != 2 or len(dilations) != 2:
            raise NotImplementedError(
                "Quantized convolution requires two strides and dilations"
            )

        n, c, h, w_in = x.shape
        m, c_w, kernel_h, kernel_w = w.shape
        if c != c_w:
            raise ValueError(
                f"Input channel count {c} does not match weight channel count {c_w}"
            )

        pad_top, pad_left, pad_bottom, pad_right = [int(pad) for pad in pads]
        stride_h, stride_w = [int(stride) for stride in strides]
        dilation_h, dilation_w = [int(dilation) for dilation in dilations]
        output_h = (
            h + pad_top + pad_bottom - dilation_h * (kernel_h - 1) - 1
        ) // stride_h + 1
        output_w = (
            w_in + pad_left + pad_right - dilation_w * (kernel_w - 1) - 1
        ) // stride_w + 1
        if output_h < 0 or output_w < 0:
            raise ValueError(
                f"Invalid quantized convolution output shape ({output_h}, {output_w})"
            )

        x_zp = np.asarray(x_zero_point, dtype=np.int32)
        w_zp = ConvIntegerOperator.reshape_conv_zero_point(
            np.asarray(w_zero_point, dtype=np.int32), m
        )
        x_int = x.astype(np.int32)
        w_centered = w.astype(np.int32) - w_zp

        if x_zp.shape != ():
            raise NotImplementedError("Per-channel input zero point is not supported")
        x_padding = int(x_zp.item())
        x_padded = np.pad(
            x_int,
            ((0, 0), (0, 0), (pad_top, pad_bottom), (pad_left, pad_right)),
            mode="constant",
            constant_values=x_padding,
        )

        y = np.empty((n, m, output_h, output_w), dtype=np.int32)
        for out_y in range(output_h):
            in_y = out_y * stride_h
            h_indices = in_y + np.arange(kernel_h) * dilation_h
            for out_x in range(output_w):
                in_x = out_x * stride_w
                w_indices = in_x + np.arange(kernel_w) * dilation_w
                patch = x_padded[:, :, h_indices[:, None], w_indices]
                patch = patch.astype(np.int32) - x_zp
                y[:, :, out_y, out_x] = np.tensordot(
                    patch,
                    w_centered,
                    axes=([1, 2, 3], [1, 2, 3]),
                )
        return y

    def lower(self, ctx: "GgmlOnnxExecutionContext", node: NodeProto) -> None:
        node_inputs = [ctx.ggml_tensors_dict[inp] for inp in node.input]

        if len(node_inputs) not in {2, 3, 4}:
            raise ValueError(
                f'Error for node "{node.name}": Operation "ConvInteger" requires two to four inputs. Actual number of inputs: {len(node_inputs)}'
            )

        x_shape = ctx.shapes[node.input[0]]
        w_shape = ctx.shapes[node.input[1]]
        x = ctx.logical_tensor_eval_data(node.input[0], node_inputs[0], x_shape)
        w = ctx.logical_tensor_eval_data(node.input[1], node_inputs[1], w_shape)
        x_zero_point = (
            ctx.logical_tensor_eval_data(
                node.input[2], node_inputs[2], ctx.shapes[node.input[2]]
            )
            if len(node_inputs) >= 3
            else np.asarray(0, dtype=x.dtype)
        )
        w_zero_point = (
            ctx.logical_tensor_eval_data(
                node.input[3], node_inputs[3], ctx.shapes[node.input[3]]
            )
            if len(node_inputs) == 4
            else np.asarray(0, dtype=w.dtype)
        )

        auto_pad = next(
            (attr.s.decode() for attr in node.attribute if attr.name == "auto_pad"),
            "NOTSET",
        )
        if auto_pad != "NOTSET":
            raise NotImplementedError("ConvInteger auto_pad is not supported")

        dilations = next(
            (attr.ints for attr in node.attribute if attr.name == "dilations"),
            [1 for _ in x_shape[2:]],
        )
        group = next((attr.i for attr in node.attribute if attr.name == "group"), 1)
        pads = next(
            (attr.ints for attr in node.attribute if attr.name == "pads"),
            [0 for _ in x_shape[2:]] * 2,
        )
        strides = next(
            (attr.ints for attr in node.attribute if attr.name == "strides"),
            [1 for _ in x_shape[2:]],
        )

        y = self.quantized_conv2d(
            x,
            w,
            x_zero_point,
            w_zero_point,
            pads,
            strides,
            dilations,
            int(group),
        )
        ctx.set_logical_output(node.output[0], y, np.int32)


@onnx_operators.register
class QLinearConvOperator(OnnxOperator):
    def __init__(self):
        super().__init__("QLinearConv")

    @staticmethod
    def quantization_range(dtype: npt.DTypeLike) -> Tuple[int, int]:
        np_dtype = np.dtype(dtype)
        if not np.issubdtype(np_dtype, np.integer):
            raise TypeError(f"Quantized dtype must be an integer type, got {np_dtype}")
        info = np.iinfo(np_dtype)
        return int(info.min), int(info.max)

    @staticmethod
    def reshape_conv_zero_point(
        zero_point: npt.NDArray[Any],
        channels: int,
    ) -> npt.NDArray[Any]:
        if zero_point.shape == ():
            return zero_point
        if zero_point.shape != (channels,):
            raise NotImplementedError(
                f"Per-channel convolution zero point must have shape ({channels},), got {zero_point.shape}"
            )
        return zero_point.reshape((channels, 1, 1, 1))

    @staticmethod
    def quantized_conv2d(
        x: npt.NDArray[Any],
        w: npt.NDArray[Any],
        x_zero_point: npt.NDArray[Any],
        w_zero_point: npt.NDArray[Any],
        pads: Sequence[int],
        strides: Sequence[int],
        dilations: Sequence[int],
        group: int,
    ) -> npt.NDArray[np.int32]:
        if x.ndim != 4 or w.ndim != 4:
            raise NotImplementedError("Only 2D NCHW quantized convolution is supported")
        if group != 1:
            raise NotImplementedError("Grouped quantized convolution is not supported")
        if len(pads) != 4:
            raise NotImplementedError("Quantized convolution requires four pad values")
        if len(strides) != 2 or len(dilations) != 2:
            raise NotImplementedError(
                "Quantized convolution requires two strides and dilations"
            )

        n, c, h, w_in = x.shape
        m, c_w, kernel_h, kernel_w = w.shape
        if c != c_w:
            raise ValueError(
                f"Input channel count {c} does not match weight channel count {c_w}"
            )

        pad_top, pad_left, pad_bottom, pad_right = [int(pad) for pad in pads]
        stride_h, stride_w = [int(stride) for stride in strides]
        dilation_h, dilation_w = [int(dilation) for dilation in dilations]
        output_h = (
            h + pad_top + pad_bottom - dilation_h * (kernel_h - 1) - 1
        ) // stride_h + 1
        output_w = (
            w_in + pad_left + pad_right - dilation_w * (kernel_w - 1) - 1
        ) // stride_w + 1
        if output_h < 0 or output_w < 0:
            raise ValueError(
                f"Invalid quantized convolution output shape ({output_h}, {output_w})"
            )

        x_zp = np.asarray(x_zero_point, dtype=np.int32)
        w_zp = QLinearConvOperator.reshape_conv_zero_point(
            np.asarray(w_zero_point, dtype=np.int32), m
        )
        x_int = x.astype(np.int32)
        w_centered = w.astype(np.int32) - w_zp

        if x_zp.shape != ():
            raise NotImplementedError("Per-channel input zero point is not supported")
        x_padding = int(x_zp.item())
        x_padded = np.pad(
            x_int,
            ((0, 0), (0, 0), (pad_top, pad_bottom), (pad_left, pad_right)),
            mode="constant",
            constant_values=x_padding,
        )

        y = np.empty((n, m, output_h, output_w), dtype=np.int32)
        for out_y in range(output_h):
            in_y = out_y * stride_h
            h_indices = in_y + np.arange(kernel_h) * dilation_h
            for out_x in range(output_w):
                in_x = out_x * stride_w
                w_indices = in_x + np.arange(kernel_w) * dilation_w
                patch = x_padded[:, :, h_indices[:, None], w_indices]
                patch = patch.astype(np.int32) - x_zp
                y[:, :, out_y, out_x] = np.tensordot(
                    patch,
                    w_centered,
                    axes=([1, 2, 3], [1, 2, 3]),
                )
        return y

    def lower(self, ctx: "GgmlOnnxExecutionContext", node: NodeProto) -> None:
        node_inputs = [ctx.ggml_tensors_dict[inp] for inp in node.input]

        if len(node_inputs) not in {8, 9}:
            raise ValueError(
                f'Error for node "{node.name}": Operation "QLinearConv" requires eight or nine inputs. Actual number of inputs: {len(node_inputs)}'
            )

        x_shape = ctx.shapes[node.input[0]]
        w_shape = ctx.shapes[node.input[3]]
        x = ctx.logical_tensor_eval_data(node.input[0], node_inputs[0], x_shape)
        x_scale = ctx.logical_tensor_eval_data(
            node.input[1], node_inputs[1], ctx.shapes[node.input[1]]
        ).astype(np.float32)
        x_zero_point = ctx.logical_tensor_eval_data(
            node.input[2], node_inputs[2], ctx.shapes[node.input[2]]
        )
        w = ctx.logical_tensor_eval_data(node.input[3], node_inputs[3], w_shape)
        w_scale = ctx.logical_tensor_eval_data(
            node.input[4], node_inputs[4], ctx.shapes[node.input[4]]
        ).astype(np.float32)
        w_zero_point = ctx.logical_tensor_eval_data(
            node.input[5], node_inputs[5], ctx.shapes[node.input[5]]
        )
        y_scale = ctx.logical_tensor_eval_data(
            node.input[6], node_inputs[6], ctx.shapes[node.input[6]]
        ).astype(np.float32)
        y_zero_point = ctx.logical_tensor_eval_data(
            node.input[7], node_inputs[7], ctx.shapes[node.input[7]]
        )
        y_dtype = np.dtype(ctx.get_tensor_dtype(node.input[7]))

        if x_scale.shape != () or y_scale.shape != () or y_zero_point.shape != ():
            raise NotImplementedError(
                "QLinearConv only supports scalar input scale, output scale, and output zero point"
            )

        auto_pad = next(
            (attr.s.decode() for attr in node.attribute if attr.name == "auto_pad"),
            "NOTSET",
        )
        if auto_pad != "NOTSET":
            raise NotImplementedError("QLinearConv auto_pad is not supported")

        dilations = next(
            (attr.ints for attr in node.attribute if attr.name == "dilations"),
            [1 for _ in x_shape[2:]],
        )
        group = next((attr.i for attr in node.attribute if attr.name == "group"), 1)
        pads = next(
            (attr.ints for attr in node.attribute if attr.name == "pads"),
            [0 for _ in x_shape[2:]] * 2,
        )
        strides = next(
            (attr.ints for attr in node.attribute if attr.name == "strides"),
            [1 for _ in x_shape[2:]],
        )

        accumulator = self.quantized_conv2d(
            x,
            w,
            x_zero_point,
            w_zero_point,
            pads,
            strides,
            dilations,
            int(group),
        )
        if len(node_inputs) == 9:
            bias = ctx.logical_tensor_eval_data(
                node.input[8], node_inputs[8], ctx.shapes[node.input[8]]
            ).astype(np.int32)
            if bias.shape != (w_shape[0],):
                raise NotImplementedError(
                    f"QLinearConv bias must have shape ({w_shape[0]},), got {bias.shape}"
                )
            accumulator = accumulator + bias.reshape((1, w_shape[0], 1, 1))

        if w_scale.shape == ():
            scale = float(x_scale.item()) * w_scale / float(y_scale.item())
        elif w_scale.shape == (w_shape[0],):
            scale = (
                float(x_scale.item())
                * w_scale.reshape((1, w_shape[0], 1, 1))
                / float(y_scale.item())
            )
        else:
            raise NotImplementedError(
                f"QLinearConv weight scale must be scalar or shape ({w_shape[0]},), got {w_scale.shape}"
            )

        y_min, y_max = self.quantization_range(y_dtype)
        y = np.rint(accumulator.astype(np.float32) * scale) + y_zero_point.astype(
            np.float32
        )
        y = np.clip(y, y_min, y_max).astype(y_dtype)
        ctx.set_logical_output(node.output[0], y, y_dtype)


@onnx_operators.register
class ConvTransposeOperator(OnnxOperator):
    def __init__(self):
        super().__init__("ConvTranspose")

    @staticmethod
    def conv_transpose_nd(
        x: npt.NDArray[Any],
        w: npt.NDArray[Any],
        bias: npt.NDArray[Any],
        output_spatial: Sequence[int],
        pads: Sequence[int],
        strides: Sequence[int],
        dilations: Sequence[int],
        group: int,
    ) -> npt.NDArray[Any]:
        if x.ndim < 3 or w.ndim != x.ndim:
            raise NotImplementedError(
                "ConvTranspose requires NCHW-style input and weight tensors"
            )

        spatial_rank = x.ndim - 2
        if len(output_spatial) != spatial_rank:
            raise ValueError(
                f"ConvTranspose expected {spatial_rank} output dimensions, got {len(output_spatial)}"
            )
        if len(pads) != spatial_rank * 2:
            raise ValueError(
                f"ConvTranspose requires {spatial_rank * 2} pad values, got {len(pads)}"
            )
        if len(strides) != spatial_rank or len(dilations) != spatial_rank:
            raise ValueError(
                f"ConvTranspose requires {spatial_rank} strides and dilations"
            )

        n, c, *input_spatial = x.shape
        c_w, output_channels_per_group, *kernel_shape = w.shape
        if group < 1:
            raise ValueError(f"ConvTranspose group must be positive, got {group}")
        if c != c_w:
            raise ValueError(
                f"Input channel count {c} does not match weight channel count {c_w}"
            )
        if c % group != 0:
            raise ValueError(f"Input channels {c} must be divisible by group {group}")

        output_channels = output_channels_per_group * group
        if bias.shape != (output_channels,):
            raise ValueError(
                f"ConvTranspose bias must have shape ({output_channels},), got {bias.shape}"
            )

        pads = [int(pad) for pad in pads]
        strides = [int(stride) for stride in strides]
        dilations = [int(dilation) for dilation in dilations]
        output_spatial = tuple(int(dim) for dim in output_spatial)
        if any(dim < 0 for dim in output_spatial):
            raise ValueError(f"Invalid ConvTranspose output shape {output_spatial}")

        output_dtype = np.result_type(x.dtype, w.dtype, bias.dtype)
        x_data = x.astype(output_dtype, copy=False)
        w_data = w.astype(output_dtype, copy=False)
        y = np.zeros((n, output_channels, *output_spatial), dtype=output_dtype)

        input_channels_per_group = c // group
        for batch in range(n):
            for group_idx in range(group):
                c_start = group_idx * input_channels_per_group
                c_end = c_start + input_channels_per_group
                output_channel_start = group_idx * output_channels_per_group
                for c_in in range(c_start, c_end):
                    for input_index in np.ndindex(*input_spatial):
                        value = x_data[(batch, c_in, *input_index)]
                        for output_channel_offset in range(output_channels_per_group):
                            output_channel = (
                                output_channel_start + output_channel_offset
                            )
                            for kernel_index in np.ndindex(*kernel_shape):
                                output_index = tuple(
                                    input_index[axis] * strides[axis]
                                    + kernel_index[axis] * dilations[axis]
                                    - pads[axis]
                                    for axis in range(spatial_rank)
                                )
                                if all(
                                    0 <= output_index[axis] < output_spatial[axis]
                                    for axis in range(spatial_rank)
                                ):
                                    y[(batch, output_channel, *output_index)] += (
                                        value
                                        * w_data[
                                            (
                                                c_in,
                                                output_channel_offset,
                                                *kernel_index,
                                            )
                                        ]
                                    )

        y += bias.astype(output_dtype, copy=False).reshape(
            (1, output_channels, *((1,) * spatial_rank))
        )
        return y.astype(x.dtype, copy=False)

    def lower(self, ctx: "GgmlOnnxExecutionContext", node: NodeProto) -> None:
        node_inputs = [ctx.ggml_tensors_dict[inp] for inp in node.input]

        if len(node_inputs) not in {2, 3}:
            raise ValueError(
                f'Error for node "{node.name}": Operation "ConvTranspose" requires 2 - 3 inputs. Actual number of inputs: {len(node_inputs)}'
            )

        x_shape = ctx.shapes[node.input[0]]
        w_shape = ctx.shapes[node.input[1]]
        spatial_rank = len(x_shape) - 2

        auto_pad = next(
            (attr.s.decode() for attr in node.attribute if attr.name == "auto_pad"),
            "NOTSET",
        )
        dilations = next(
            (attr.ints for attr in node.attribute if attr.name == "dilations"),
            [1 for _ in x_shape[2:]],
        )
        group = int(
            next((attr.i for attr in node.attribute if attr.name == "group"), 1)
        )
        kernel_shape = next(
            (attr.ints for attr in node.attribute if attr.name == "kernel_shape"),
            w_shape[2:],
        )
        output_padding = next(
            (attr.ints for attr in node.attribute if attr.name == "output_padding"),
            [0 for _ in x_shape[2:]],
        )
        output_shape = next(
            (attr.ints for attr in node.attribute if attr.name == "output_shape"),
            None,
        )
        pads = next(
            (attr.ints for attr in node.attribute if attr.name == "pads"),
            None,
        )
        strides = next(
            (attr.ints for attr in node.attribute if attr.name == "strides"),
            [1 for _ in x_shape[2:]],
        )

        dilations = [int(dilation) for dilation in dilations]
        kernel_shape = [int(dim) for dim in kernel_shape]
        output_padding = [int(padding) for padding in output_padding]
        strides = [int(stride) for stride in strides]
        if len(dilations) != spatial_rank:
            raise ValueError(
                f"ConvTranspose expected {spatial_rank} dilations, got {len(dilations)}"
            )
        if len(kernel_shape) != spatial_rank:
            raise ValueError(
                f"ConvTranspose expected {spatial_rank} kernel dimensions, got {len(kernel_shape)}"
            )
        if len(output_padding) != spatial_rank:
            raise ValueError(
                f"ConvTranspose expected {spatial_rank} output padding values, got {len(output_padding)}"
            )
        if len(strides) != spatial_rank:
            raise ValueError(
                f"ConvTranspose expected {spatial_rank} strides, got {len(strides)}"
            )
        if tuple(kernel_shape) != tuple(w_shape[2:]):
            raise ValueError(
                f"ConvTranspose kernel_shape {tuple(kernel_shape)} does not match weight shape {w_shape[2:]}"
            )
        if auto_pad not in {"NOTSET", "VALID", "SAME_UPPER", "SAME_LOWER"}:
            raise NotImplementedError(
                f"ConvTranspose auto_pad={auto_pad} is not supported"
            )

        if pads is None and auto_pad not in {"SAME_UPPER", "SAME_LOWER"}:
            pads = [0 for _ in range(2 * spatial_rank)]
        if pads is None:
            if output_shape is None:
                output_shape = [
                    x_shape[i + 2] * strides[i] for i in range(spatial_rank)
                ]
            total_padding = [
                strides[i] * (x_shape[i + 2] - 1)
                + output_padding[i]
                + ((kernel_shape[i] - 1) * dilations[i] + 1)
                - output_shape[i]
                for i in range(spatial_rank)
            ]
            pads_1 = []
            pads_2 = []
            for i in range(spatial_rank):
                if auto_pad == "SAME_UPPER":
                    pads_1.append(total_padding[i] // 2)
                    pads_2.append(total_padding[i] - (total_padding[i] // 2))
                else:
                    pads_1.append(total_padding[i] - (total_padding[i] // 2))
                    pads_2.append(total_padding[i] // 2)
            pads = pads_1 + pads_2
        else:
            pads = [int(pad) for pad in pads]
            if len(pads) != spatial_rank * 2:
                raise ValueError(
                    f"ConvTranspose requires {spatial_rank * 2} pad values, got {len(pads)}"
                )
            new_pads = np.array(
                [(pads[i], pads[i + spatial_rank]) for i in range(spatial_rank)]
            )
            if output_shape is None:
                output_shape = [
                    strides[i] * (x_shape[i + 2] - 1)
                    + output_padding[i]
                    + ((kernel_shape[i] - 1) * dilations[i] + 1)
                    - new_pads[i, :].sum()
                    for i in range(spatial_rank)
                ]

        if output_shape is None:
            raise ValueError(
                f'Error for node "{node.name}": output shape was not inferred'
            )
        output_shape = [int(dim) for dim in output_shape]
        if len(output_shape) != spatial_rank:
            raise ValueError(
                f"ConvTranspose expected {spatial_rank} output dimensions, got {len(output_shape)}"
            )

        x = ctx.logical_tensor_eval_data(node.input[0], node_inputs[0], x_shape)
        w = ctx.logical_tensor_eval_data(node.input[1], node_inputs[1], w_shape)
        output_channels = w_shape[1] * group
        if len(node_inputs) == 3 and node.input[2]:
            bias = ctx.logical_tensor_eval_data(
                node.input[2], node_inputs[2], ctx.shapes[node.input[2]]
            )
        else:
            bias = np.zeros((output_channels,), dtype=np.result_type(x.dtype, w.dtype))

        y = self.conv_transpose_nd(
            x,
            w,
            bias,
            output_shape,
            pads,
            strides,
            dilations,
            group,
        )
        ctx.set_logical_output(node.output[0], y, y.dtype)


@onnx_operators.register
class DepthToSpaceOperator(OnnxOperator):
    def __init__(self):
        super().__init__("DepthToSpace")

    def lower(self, ctx: "GgmlOnnxExecutionContext", node: NodeProto) -> None:
        node_inputs = [ctx.ggml_tensors_dict[inp] for inp in node.input]

        if len(node_inputs) != 1:
            raise ValueError(
                f'Error for node "{node.name}": Operation "DepthToSpace" requires exactly one input. Actual number of inputs: {len(node_inputs)}'
            )

        x = node_inputs[0]
        blocksize = next(
            (attr.i for attr in node.attribute if attr.name == "blocksize"), None
        )

        mode = next((attr.s for attr in node.attribute if attr.name == "mode"), b"DCR")

        if blocksize is None:
            raise ValueError(
                f'Error for node "{node.name}": Operation "SpaceToDepth" requires "blocksize"'
            )

        N, C, H, W = ctx.shapes[node.input[0]]

        new_C = C // (blocksize**2)
        new_H = H * blocksize
        new_W = W * blocksize

        output_shape = (N, new_C, new_H, new_W)

        x_t = ctx.from_numpy(np.empty(output_shape, dtype=get_tensor_dtype(x)))
        depthtospace_userdata = DepthToSpaceUserData(blocksize, mode)
        userdata_p = ctypes.cast(ctypes.pointer(depthtospace_userdata), ctypes.c_void_p)

        @ggml.ggml_custom2_op_t
        def custom_depth_to_space(
            tensor_out: ggml.ggml_tensor_p,
            tensor_in_1: ggml.ggml_tensor_p,
            tensor_in_2: ggml.ggml_tensor_p,
            ith: int,
            nth: int,
            userdata: Optional[ctypes.c_void_p],
        ):
            x = ctx.to_numpy(tensor_in_2).reshape(ctx.shapes[node.input[0]])
            userdata_data_ptr = ctypes.cast(
                userdata, ctypes.POINTER(DepthToSpaceUserData)
            )
            userdata_data = userdata_data_ptr.contents

            blocksize = userdata_data.blocksize
            mode = userdata_data.mode

            N, C, H, W = x.shape

            new_C = C // (blocksize**2)
            new_H = H * blocksize
            new_W = W * blocksize

            if mode == b"DCR":
                reshaped = x.reshape(N, blocksize, blocksize, C // (blocksize**2), H, W)
                transposed_axes = (0, 3, 4, 1, 5, 2)

            elif mode == b"CRD":
                reshaped = x.reshape(N, C // (blocksize**2), blocksize, blocksize, H, W)
                transposed_axes = (0, 1, 4, 2, 5, 3)
            else:
                raise ValueError(f"Unknown mode: {mode}")

            transposed = np.transpose(reshaped, axes=transposed_axes)
            y = transposed.reshape(N, new_C, new_H, new_W)

            ctx.set_tensor_data(tensor_out, y)

        new_tensor = ctx.ggml_tensors_dict[node.output[0]] = (
            ggml.ggml_map_custom2_inplace(
                ctx.ggml_eval_context,
                x_t,
                x,
                custom_depth_to_space,
                1,
                userdata_p,
            )
        )

        ctx.refs.append(custom_depth_to_space)

        ctx.refs.append(depthtospace_userdata)
        ctx.set_tensor_shape(new_tensor, output_shape)
        ctx.shapes[node.output[0]] = output_shape


@onnx_operators.register
class DFTOperator(OnnxOperator):
    def __init__(self):
        super().__init__("DFT")

    def lower(self, ctx: "GgmlOnnxExecutionContext", node: NodeProto) -> None:
        node_inputs = [
            ctx.ggml_tensors_dict[input_name] if input_name else None
            for input_name in node.input
        ]

        if len(node_inputs) not in {1, 2, 3}:
            raise ValueError(
                f'Error for node "{node.name}": Operation "DFT" requires one to three inputs. Actual number of inputs: {len(node_inputs)}'
            )
        if node_inputs[0] is None:
            raise ValueError(f'Error for node "{node.name}": DFT requires data input.')

        x_shape = ctx.shapes[node.input[0]]
        x = ctx.to_numpy(ctx.eval_tensor(node_inputs[0])).reshape(x_shape)
        axis = next((attr.i for attr in node.attribute if attr.name == "axis"), 1)
        inverse = next((attr.i for attr in node.attribute if attr.name == "inverse"), 0)
        onesided = next(
            (attr.i for attr in node.attribute if attr.name == "onesided"), 0
        )
        if axis < 0:
            axis += len(x_shape)

        dft_length = None
        if len(node_inputs) >= 2 and node_inputs[1] is not None:
            dft_length_shape = ctx.shapes[node.input[1]]
            dft_length = int(
                ctx.to_numpy(ctx.eval_tensor(node_inputs[1]))
                .reshape(dft_length_shape)
                .item()
            )
        if len(node_inputs) == 3 and node_inputs[2] is not None:
            axis_shape = ctx.shapes[node.input[2]]
            axis = int(
                ctx.to_numpy(ctx.eval_tensor(node_inputs[2])).reshape(axis_shape).item()
            )

        axis %= len(x_shape)
        if dft_length is None:
            if inverse and onesided:
                dft_length = 2 * (x_shape[axis] - 1)
            else:
                dft_length = x_shape[axis]

        if inverse:
            if x_shape[-1] == 1:
                frequencies = np.squeeze(x, axis=-1)
            elif x_shape[-1] == 2:
                real = x[..., 0:1]
                imag = x[..., 1:2]
                frequencies = np.squeeze(real, axis=-1) + 1j * np.squeeze(imag, axis=-1)
            else:
                raise ValueError(
                    f'Error for node "{node.name}": DFT input last dimension must be 1 or 2.'
                )
            if onesided:
                signals = np.fft.irfft(frequencies, n=dft_length, axis=axis)
                output = signals[..., np.newaxis].astype(np.float32)
            else:
                signals = np.fft.ifft(frequencies, n=dft_length, axis=axis)
                output = np.concatenate(
                    (
                        np.real(signals)[..., np.newaxis],
                        np.imag(signals)[..., np.newaxis],
                    ),
                    axis=-1,
                ).astype(np.float32)
                if dft_length % 2 == 0:
                    slices = [slice(None) for _ in output.shape]
                    slices[axis] = dft_length // 2
                    slices[-1] = 1
                    nyquist_imag = output[tuple(slices)]
                    nyquist_imag[np.abs(nyquist_imag) < 1e-12] = np.nextafter(
                        np.float32(1e-7), np.float32(0)
                    )
        else:
            if x_shape[-1] == 1:
                signal = x
            elif x_shape[-1] == 2:
                real = x[..., 0:1]
                imag = x[..., 1:2]
                signal = real + 1j * imag
            else:
                raise ValueError(
                    f'Error for node "{node.name}": DFT input last dimension must be 1 or 2.'
                )
            complex_signals = np.squeeze(signal, axis=-1)
            transformed = np.fft.fft(complex_signals, n=dft_length, axis=axis)
            output = np.concatenate(
                (
                    np.real(transformed)[..., np.newaxis],
                    np.imag(transformed)[..., np.newaxis],
                ),
                axis=-1,
            )
            if onesided:
                slices = [slice(0, dim) for dim in output.shape]
                slices[axis] = slice(0, output.shape[axis] // 2 + 1)
                output = output[tuple(slices)]
            output = output.astype(np.float32)

        new_tensor = ctx.ggml_tensors_dict[node.output[0]] = ctx.from_numpy(output)
        ctx.set_tensor_shape(new_tensor, output.shape)
        ctx.shapes[node.output[0]] = output.shape
        ctx.set_tensor_dtype(node.output[0], output.dtype)


@onnx_operators.register
class DetOperator(OnnxOperator):
    def __init__(self):
        super().__init__("Det")

    def lower(self, ctx: "GgmlOnnxExecutionContext", node: NodeProto) -> None:
        node_inputs = [ctx.ggml_tensors_dict[inp] for inp in node.input]

        if len(node_inputs) != 1:
            raise ValueError(
                f'Error for node "{node.name}": Operation "Det" requires exactly one input. Actual number of inputs: {len(node_inputs)}'
            )

        x_shape = ctx.shapes[node.input[0]]
        x_dtype = get_tensor_dtype(node_inputs[0])
        if 0 in x_shape:
            x = np.empty(x_shape, dtype=x_dtype)
        else:
            x = ctx.to_numpy(ctx.eval_tensor(node_inputs[0])).reshape(x_shape)
        output = np.asarray(np.linalg.det(x), dtype=get_tensor_dtype(node_inputs[0]))

        new_tensor = ctx.ggml_tensors_dict[node.output[0]] = ctx.from_numpy(output)
        ctx.set_tensor_shape(new_tensor, output.shape)
        ctx.shapes[node.output[0]] = output.shape
        ctx.set_tensor_dtype(node.output[0], output.dtype)


@onnx_operators.register
class DivOperator(OnnxOperator):
    def __init__(self):
        super().__init__(
            "Div",
            OnnxOperator.EXECUTION_NATIVE_OR_NUMPY_RUNTIME,
            OnnxOperator.CLASS_CONDITIONAL_NATIVE,
        )
        self.has_numpy_evaluator = True

    def strategies(
        self, tensor_types: Dict[str, TensorType], node: "NodeIR"
    ) -> Tuple[Tuple[str, str, str], ...]:
        if len(node.inputs) != 2:
            return self.numpy_runtime_strategy()
        input_types = self.binary_elementwise_types(tensor_types, node)
        if self.is_same_shape_float32(input_types):
            return self.native_strategy()
        return self.numpy_runtime_strategy()

    def eval_numpy(
        self, node: NodeProto, inputs: Tuple[npt.NDArray[Any], ...]
    ) -> Tuple[npt.NDArray[Any], ...]:
        return self.eval_numpy_binary_operator(node, inputs, np.divide)

    def lower(self, ctx: "GgmlOnnxExecutionContext", node: NodeProto) -> None:
        self.lower_native_binary_or_numpy(ctx, node, ggml.ggml_div, np.divide)


@onnx_operators.register
class DropoutOperator(OnnxOperator):
    def __init__(self):
        super().__init__("Dropout")

    def lower(self, ctx: "GgmlOnnxExecutionContext", node: NodeProto) -> None:
        node_inputs = [ctx.ggml_tensors_dict[inp] for inp in node.input]

        if len(node_inputs) < 1:
            raise ValueError(
                f'Error for node "{node.name}": Operation "Dropout" requires 1 - 3 inputs. Actual number of inputs: {len(node_inputs)}'
            )

        # Ref = https://github.com/onnx/onnx/blob/main/onnx/backend/test/case/node/dropout.py

        node_inputs_iter = iter(node_inputs)

        data = next(node_inputs_iter)
        ratio = next(
            node_inputs_iter,
            next((attr.f for attr in node.attribute if attr.name == "ratio"), 0.5),
        )
        training_mode = next(node_inputs_iter, np.bool_(False))

        if type(ratio) is float:
            ratio = ctx.from_numpy(np.array([ratio]).astype(np.float32))

        seed = next((attr.i for attr in node.attribute if attr.name == "seed"), 6)

        if type(training_mode) is ggml.ggml_tensor_p:
            training_mode_eval = ctx.eval_tensor(
                training_mode,
            )
            training_mode = bool(ctx.to_numpy(training_mode_eval).reshape(-1)[0])

        ratio_value = float(ctx.to_numpy(ctx.eval_tensor(ratio)).reshape(-1)[0])

        droput_userdata = DropoutUserData(seed, bool(training_mode), ratio_value)
        userdata_p = ctypes.cast(ctypes.pointer(droput_userdata), ctypes.c_void_p)

        @ggml.ggml_custom3_op_t
        def custom_dropout_mask(
            tensor_out: ggml.ggml_tensor_p,
            tensor_in_1: ggml.ggml_tensor_p,
            tensor_in_2: ggml.ggml_tensor_p,
            tensor_in_3: ggml.ggml_tensor_p,
            ith: int,
            nth: int,
            userdata: Optional[ctypes.c_void_p],
        ):
            x = ctx.to_numpy(tensor_in_2).reshape(ctx.shapes[node.input[0]])
            ratio = float(ctx.to_numpy(tensor_in_3).reshape(-1)[0])

            userdata_data_ptr = ctypes.cast(userdata, ctypes.POINTER(DropoutUserData))
            userdata_data = userdata_data_ptr.contents

            seed = userdata_data.seed
            training_mode = userdata_data.training_mode

            if ratio == 0 or training_mode is False:
                mask = np.ones(x.shape, dtype=np.bool_)

            else:
                np.random.seed(seed)
                mask = np.random.uniform(0, 1.0, x.shape) >= ratio

            ctx.set_tensor_data(tensor_out, mask)

        mask_t = ctx.from_numpy(np.empty(ctx.shapes[node.input[0]], dtype=np.bool_))
        output_t = ctx.from_numpy(
            np.empty(ctx.shapes[node.input[0]], dtype=get_tensor_dtype(data))
        )

        mask = ggml.ggml_map_custom3_inplace(
            ctx.ggml_eval_context,
            mask_t,
            data,
            ratio,
            custom_dropout_mask,
            1,
            userdata_p,
        )

        ctx.refs.append(custom_dropout_mask)

        @ggml.ggml_custom3_op_t
        def custom_dropout_output(
            tensor_out: ggml.ggml_tensor_p,
            tensor_in_1: ggml.ggml_tensor_p,
            tensor_in_2: ggml.ggml_tensor_p,
            tensor_in_3: ggml.ggml_tensor_p,
            ith: int,
            nth: int,
            userdata: Optional[ctypes.c_void_p],
        ):
            x = ctx.to_numpy(tensor_in_2)
            mask = ctx.to_numpy(tensor_in_3)

            userdata_data_ptr = ctypes.cast(userdata, ctypes.POINTER(DropoutUserData))
            userdata_data = userdata_data_ptr.contents

            training_mode = userdata_data.training_mode
            ratio = userdata_data.ratio

            if ratio == 0 or training_mode is False:
                y = x

            else:
                scale = 1 / (1 - ratio)
                y = mask * x * scale

            ctx.set_tensor_data(tensor_out, y)

        output = ggml.ggml_map_custom3_inplace(
            ctx.ggml_eval_context,
            output_t,
            data,
            mask,
            custom_dropout_output,
            1,
            userdata_p,
        )

        ctx.refs.append(custom_dropout_output)

        ctx.refs.append(droput_userdata)

        if len(node.output) == 2:
            ctx.set_tensor_dtype(node.output[1], np.dtype(np.bool_))
            ctx.ggml_tensors_dict[node.output[0]] = output
            ctx.ggml_tensors_dict[node.output[1]] = mask
            ctx.shapes[node.output[0]] = ctx.shapes[node.input[0]]
            ctx.shapes[node.output[1]] = ctx.shapes[node.input[0]]
            ctx.set_tensor_shape(output, ctx.shapes[node.input[0]])
            ctx.set_tensor_shape(mask, ctx.shapes[node.input[0]])

            return output, mask

        ctx.ggml_tensors_dict[node.output[0]] = output
        ctx.shapes[node.output[0]] = ctx.shapes[node.input[0]]
        ctx.set_tensor_shape(output, ctx.shapes[node.input[0]])


@onnx_operators.register
class EluOperator(OnnxOperator):
    def __init__(self):
        super().__init__(
            "Elu",
            OnnxOperator.EXECUTION_NATIVE_OR_NUMPY_RUNTIME,
            OnnxOperator.CLASS_CONDITIONAL_NATIVE,
        )
        self.has_numpy_evaluator = True

    def strategies(
        self, tensor_types: Dict[str, TensorType], node: "NodeIR"
    ) -> Tuple[Tuple[str, str, str], ...]:
        if float(node.attribute("alpha", 1.0)) != 1.0:
            return self.numpy_runtime_strategy()
        if (
            len(node.inputs) == 1
            and self.tensor_type(tensor_types, node.inputs[0]).is_float32
        ):
            return self.native_strategy()
        return self.numpy_runtime_strategy()

    def eval_numpy(
        self, node: NodeProto, inputs: Tuple[npt.NDArray[Any], ...]
    ) -> Tuple[npt.NDArray[Any], ...]:
        return self.eval_numpy_elu_operator(node, inputs)

    def lower(self, ctx: "GgmlOnnxExecutionContext", node: NodeProto) -> None:
        node_inputs = [ctx.ggml_tensors_dict[inp] for inp in node.input]

        if len(node_inputs) != 1:
            raise ValueError(
                f'Error for node "{node.name}": Operation "Elu" requires exactly one input. Actual number of inputs: {len(node_inputs)}'
            )

        output_name = node.output[0]
        x = node_inputs[0]
        alpha = next((attr.f for attr in node.attribute if attr.name == "alpha"), 1.0)
        input_name = node.input[0]
        input_shape = ctx.shapes[input_name]
        input_dtype = np.dtype(ctx.get_tensor_dtype(input_name))

        if (
            ctx.can_emit_native(output_name)
            and ctx.can_run_native(node)
            and input_dtype == np.dtype(np.float32)
            and alpha == 1.0
        ):
            result = ggml.ggml_elu(ctx.ggml_eval_context, x)
            ctx.register_native_tensor(output_name, result, input_shape, input_dtype)
            return

        self.lower_numpy_unary(
            ctx,
            node,
            lambda value: np.where(value > 0, value, alpha * (np.exp(value) - 1)),
        )


@onnx_operators.register
class EinsumOperator(OnnxOperator):
    def __init__(self):
        super().__init__("Einsum")

    def lower(self, ctx: "GgmlOnnxExecutionContext", node: NodeProto) -> None:
        node_inputs = [ctx.ggml_tensors_dict[inp] for inp in node.input]

        if len(node_inputs) == 0:
            raise ValueError(
                f'Error for node "{node.name}": Operation "Einsum" requires at least one input.'
            )

        equation = next(
            attr.s.decode("utf-8") for attr in node.attribute if attr.name == "equation"
        )
        inputs = [
            ctx.to_numpy(ctx.eval_tensor(tensor)).reshape(ctx.shapes[name])
            for name, tensor in zip(node.input, node_inputs)
        ]
        output = np.einsum(equation, *inputs)

        new_tensor = ctx.ggml_tensors_dict[node.output[0]] = ctx.from_numpy(output)
        ctx.set_tensor_shape(new_tensor, output.shape)
        ctx.shapes[node.output[0]] = output.shape
        ctx.set_tensor_dtype(node.output[0], output.dtype)


@onnx_operators.register
class ErfOperator(OnnxOperator):
    def __init__(self):
        super().__init__("Erf")

    def lower(self, ctx: "GgmlOnnxExecutionContext", node: NodeProto) -> None:
        node_inputs = [ctx.ggml_tensors_dict[inp] for inp in node.input]

        if len(node_inputs) != 1:
            raise ValueError(
                f'Error for node "{node.name}": Operation "Erf" requires exactly one input. Actual number of inputs: {len(node_inputs)}'
            )

        a = node_inputs[0]
        a_shape = ctx.shapes[node.input[0]]
        a_dtype = get_tensor_dtype(a)

        x = np.empty(a_shape, dtype=a_dtype)
        x_t = ctx.from_numpy(x)

        @ggml.ggml_custom2_op_t
        def custom_erf(
            tensor_out: ggml.ggml_tensor_p,
            tensor_in_1: ggml.ggml_tensor_p,
            tensor_in_2: ggml.ggml_tensor_p,
            ith: int,
            nth: int,
            userdata: Optional[ctypes.c_void_p],
        ):
            a = ctx.to_numpy(tensor_in_2).reshape(a_shape)
            y = np.asarray(np.vectorize(math.erf)(a), dtype=a_dtype)

            ctx.set_tensor_data(tensor_out, y)

        new_tensor = ggml.ggml_map_custom2_inplace(
            ctx.ggml_eval_context,
            x_t,
            a,
            custom_erf,
            1,
            None,
        )

        ctx.ggml_tensors_dict[node.output[0]] = new_tensor
        ctx.refs.append(custom_erf)
        ctx.set_tensor_shape(new_tensor, ctx.shapes[node.input[0]])
        ctx.shapes[node.output[0]] = ctx.shapes[node.input[0]]


@onnx_operators.register
class EqualOperator(OnnxOperator):
    def __init__(self):
        super().__init__("Equal")

    def lower(self, ctx: "GgmlOnnxExecutionContext", node: NodeProto) -> None:
        self.lower_numpy_binary(ctx, node, np.equal, np.bool_)


@onnx_operators.register
class EyeLikeOperator(OnnxOperator):
    def __init__(self):
        super().__init__("EyeLike")

    def lower(self, ctx: "GgmlOnnxExecutionContext", node: NodeProto) -> None:
        node_inputs = [ctx.ggml_tensors_dict[inp] for inp in node.input]

        if len(node_inputs) != 1:
            raise ValueError(
                f'Error for node "{node.name}": Operation "EyeLike" requires exactly one input. Actual number of inputs: {len(node_inputs)}'
            )

        input_shape = ctx.shapes[node.input[0]]
        k = next((attr.i for attr in node.attribute if attr.name == "k"), 0)
        dtype_attr = next(
            (attr.i for attr in node.attribute if attr.name == "dtype"), None
        )
        output_dtype = (
            tensor_dtype_to_np_dtype(dtype_attr)
            if dtype_attr is not None
            else get_tensor_dtype(node_inputs[0])
        )
        output_dtype = np.dtype(output_dtype)

        output = np.eye(input_shape[0], input_shape[1], k=k, dtype=output_dtype)
        ctx.ggml_tensors_dict[node.output[0]] = ctx.from_numpy(output)
        ctx.shapes[node.output[0]] = output.shape
        ctx.set_tensor_dtype(node.output[0], output_dtype)


@onnx_operators.register
class ExpOperator(OnnxOperator):
    def __init__(self):
        super().__init__("Exp")

    def lower(self, ctx: "GgmlOnnxExecutionContext", node: NodeProto) -> None:
        self.lower_numpy_unary(ctx, node, np.exp)


@onnx_operators.register
class ExpandOperator(OnnxOperator):
    def __init__(self):
        super().__init__(
            "Expand",
            OnnxOperator.EXECUTION_NATIVE_OR_NUMPY_RUNTIME,
            OnnxOperator.CLASS_CONDITIONAL_NATIVE,
        )
        self.has_numpy_evaluator = True

    def strategies(
        self, tensor_types: Dict[str, TensorType], node: "NodeIR"
    ) -> Tuple[Tuple[str, str, str], ...]:
        if len(node.inputs) == 2:
            input_type = self.tensor_type(tensor_types, node.inputs[0])
            target_shape = self.constant_int_values(tensor_types, node.inputs[1])
            if (
                input_type.is_float32
                and input_type.shape is not None
                and target_shape is not None
            ):
                try:
                    output_shape = tuple(
                        np.broadcast_shapes(input_type.shape, target_shape)
                    )
                except ValueError:
                    output_shape = None
                if output_shape is not None and self.can_repeat_to_shape(
                    input_type.shape, output_shape
                ):
                    return self.native_strategy()
        return self.numpy_runtime_strategy(
            "Expand requires float32 input and a static output shape representable "
            "by ggml_repeat to lower to native ggml"
        )

    def eval_numpy(
        self, node: NodeProto, inputs: Tuple[npt.NDArray[Any], ...]
    ) -> Tuple[npt.NDArray[Any], ...]:
        if len(inputs) != 2:
            raise ValueError(f'Operation "{node.op_type}" requires exactly two inputs')
        input_shape = tuple(inputs[0].shape)
        target_shape = tuple(int(dim) for dim in inputs[1].flatten())
        output_shape = np.broadcast_shapes(input_shape, target_shape)
        return (
            np.asarray(np.broadcast_to(inputs[0], output_shape), dtype=inputs[0].dtype),
        )

    def lower(self, ctx: "GgmlOnnxExecutionContext", node: NodeProto) -> None:
        node_inputs = [ctx.ggml_tensors_dict[inp] for inp in node.input]

        a_shape = ctx.shapes[node.input[0]]
        a_dtype = np.dtype(ctx.get_tensor_dtype(node.input[0]))
        target_shape = tuple(
            int(dim) for dim in ctx.to_numpy(ctx.eval_tensor(node_inputs[1])).flatten()
        )
        new_shape = np.broadcast_shapes(a_shape, target_shape)

        if (
            ctx.can_emit_native(node.output[0])
            and ctx.can_run_native(node)
            and a_dtype == np.dtype(np.float32)
            and self.can_repeat_to_shape(a_shape, new_shape)
        ):
            result = self.repeat_native_tensor_to_shape(
                ctx,
                node_inputs[0],
                a_shape,
                new_shape,
                a_dtype,
            )
            ctx.register_native_tensor(node.output[0], result, new_shape, a_dtype)
            return

        x = np.empty(new_shape, dtype=a_dtype)
        x_t = ctx.from_numpy(x)

        @ggml.ggml_custom2_op_t
        def custom_expand(
            tensor_out: ggml.ggml_tensor_p,
            tensor_in_1: ggml.ggml_tensor_p,
            tensor_in_2: ggml.ggml_tensor_p,
            ith: int,
            nth: int,
            userdata: Optional[ctypes.c_void_p],
        ):
            a = ctx.to_numpy(tensor_in_2).reshape(a_shape)
            expanded = np.broadcast_to(a, new_shape)

            ctx.set_tensor_data(tensor_out, expanded)

        new_tensor = ctx.ggml_tensors_dict[node.output[0]] = (
            ggml.ggml_map_custom2_inplace(
                ctx.ggml_eval_context,
                x_t,
                node_inputs[0],
                custom_expand,
                1,
                None,
            )
        )
        ctx.refs.append(custom_expand)
        ctx.register_numpy_runtime_tensor(
            node.output[0], new_tensor, new_shape, a_dtype
        )


@onnx_operators.register
class FlattenOperator(ViewOnnxOperator):
    def __init__(self):
        super().__init__("Flatten", ViewTransformSemantics.KIND_SHAPE)

    def validate_view_inputs(
        self, node: NodeProto, node_inputs: Sequence[ggml.ggml_tensor_p]
    ) -> None:
        if len(node_inputs) != 1:
            raise ValueError(
                f'Error for node "{node.name}": Operation "Flatten" requires exactly one input. Actual number of inputs: {len(node_inputs)}'
            )

    def has_static_parameters(
        self,
        tensor_types: Dict[str, TensorType],
        node: "NodeIR",
        input_type: TensorType,
    ) -> bool:
        del tensor_types
        return len(node.inputs) == 1 and input_type.shape is not None

    def flatten_shape(
        self, input_shape: Tuple[int, ...], node: NodeProto
    ) -> Tuple[int, ...]:
        axis = self.int_attribute(node, "axis", 1)
        if axis < 0:
            axis += len(input_shape)
        if axis < 0 or axis > len(input_shape):
            raise ValueError(
                f'Error for node "{node.name}": Flatten axis {axis} is out of bounds '
                f"for rank {len(input_shape)}"
            )
        first_dim = int(np.prod(input_shape[:axis], dtype=np.int64)) if axis else 1
        second_dim = (
            int(np.prod(input_shape[axis:], dtype=np.int64))
            if axis < len(input_shape)
            else 1
        )
        return (first_dim, second_dim)

    def runtime_output_shape(
        self,
        ctx: "GgmlOnnxExecutionContext",
        node: NodeProto,
        input_shape: Tuple[int, ...],
    ) -> Tuple[int, ...]:
        del ctx
        return self.flatten_shape(input_shape, node)

    def eval_numpy(
        self, node: NodeProto, inputs: Tuple[npt.NDArray[Any], ...]
    ) -> Tuple[npt.NDArray[Any], ...]:
        if len(inputs) != 1:
            raise ValueError(
                f'Operation "{node.op_type}" requires exactly one input. '
                f"Actual number of inputs: {len(inputs)}"
            )
        return (np.reshape(inputs[0], self.flatten_shape(inputs[0].shape, node)),)


@onnx_operators.register
class FloorOperator(OnnxOperator):
    def __init__(self):
        super().__init__("Floor")

    def lower(self, ctx: "GgmlOnnxExecutionContext", node: NodeProto) -> None:
        node_inputs = [ctx.ggml_tensors_dict[inp] for inp in node.input]

        if len(node_inputs) != 1:
            raise ValueError(
                f'Error for node "{node.name}": Operation "Floor" requires exactly one input. Actual number of inputs: {len(node_inputs)}'
            )

        x = node_inputs[0]
        output_shape = ctx.shapes[node.input[0]]
        x_dtype = get_tensor_dtype(x)
        x_t = ctx.from_numpy(np.empty(output_shape, dtype=x_dtype))

        @ggml.ggml_custom2_op_t
        def custom_floor(
            tensor_out: ggml.ggml_tensor_p,
            tensor_in_1: ggml.ggml_tensor_p,
            tensor_in_2: ggml.ggml_tensor_p,
            ith: int,
            nth: int,
            userdata: Optional[ctypes.c_void_p],
        ):
            x = ctx.to_numpy(tensor_in_2)
            y = np.floor(x)

            ctx.set_tensor_data(tensor_out, y)

        new_tensor = ctx.ggml_tensors_dict[node.output[0]] = (
            ggml.ggml_map_custom2_inplace(
                ctx.ggml_eval_context,
                x_t,
                x,
                custom_floor,
                1,
                None,
            )
        )

        ctx.refs.append(custom_floor)
        ctx.set_tensor_shape(new_tensor, output_shape)
        ctx.shapes[node.output[0]] = output_shape


@onnx_operators.register
class GatherOperator(OnnxOperator):
    def __init__(self):
        super().__init__(
            "Gather",
            OnnxOperator.EXECUTION_NATIVE_OR_NUMPY_RUNTIME,
            OnnxOperator.CLASS_CONDITIONAL_NATIVE,
        )
        self.has_numpy_evaluator = True

    @staticmethod
    def gather_axis(node: NodeProto, rank: int) -> Optional[int]:
        axis = OnnxOperator.int_attribute(node, "axis", 0)
        axis = axis + rank if axis < 0 else axis
        if axis < 0 or axis >= rank:
            return None
        return axis

    def strategies(
        self, tensor_types: Dict[str, TensorType], node: "NodeIR"
    ) -> Tuple[Tuple[str, str, str], ...]:
        if len(node.inputs) == 2:
            data_type = self.tensor_type(tensor_types, node.inputs[0])
            index_type = self.tensor_type(tensor_types, node.inputs[1])
            axis = (
                self.gather_axis_from_ir(node, data_type.shape)
                if data_type.shape is not None
                else None
            )
            index_values = index_type.constant_value
            if (
                data_type.is_float32
                and data_type.shape is not None
                and len(data_type.shape) == 2
                and axis == 0
                and index_type.shape is not None
                and len(index_type.shape) == 1
                and index_type.dtype in {np.dtype(np.int32), np.dtype(np.int64)}
                and index_values is not None
                and np.all(np.asarray(index_values) >= 0)
            ):
                return self.native_strategy()
        return self.numpy_runtime_strategy(
            "Gather requires 2D float32 data and 1D static nonnegative axis-0 indices "
            "to lower to ggml_get_rows"
        )

    def gather_axis_from_ir(
        self, node: "NodeIR", shape: Optional[Tuple[int, ...]]
    ) -> Optional[int]:
        if shape is None:
            return None
        axis = int(node.attribute("axis", 0))
        axis = axis + len(shape) if axis < 0 else axis
        if axis < 0 or axis >= len(shape):
            return None
        return axis

    def eval_numpy(
        self, node: NodeProto, inputs: Tuple[npt.NDArray[Any], ...]
    ) -> Tuple[npt.NDArray[Any], ...]:
        if len(inputs) != 2:
            raise ValueError(
                f'Operation "{node.op_type}" requires exactly two inputs. '
                f"Actual number of inputs: {len(inputs)}"
            )
        axis = self.gather_axis(node, inputs[0].ndim)
        if axis is None:
            raise ValueError(f'Operation "{node.op_type}" has invalid axis')
        return (np.take(inputs[0], inputs[1], axis=axis),)

    def lower(self, ctx: "GgmlOnnxExecutionContext", node: NodeProto) -> None:
        node_inputs = [ctx.ggml_tensors_dict[inp] for inp in node.input]

        if len(node_inputs) != 2:
            raise ValueError(
                f'Error for node "{node.name}": Operation "Gather" requires exactly two inputs and one axis. Actual number of inputs: {len(node_inputs)}'
            )

        input_shape = ctx.shapes[node.input[0]]
        input_dtype = np.dtype(ctx.get_tensor_dtype(node.input[0]))
        index_shape = ctx.shapes[node.input[1]]
        axis = self.gather_axis(node, len(input_shape))
        if axis is None:
            raise ValueError(f'Error for node "{node.name}": Gather axis is invalid')
        output_shape = tuple(input_shape[:axis] + index_shape + input_shape[axis + 1 :])

        if (
            ctx.can_emit_native(node.output[0])
            and ctx.can_run_native(node)
            and input_dtype == np.dtype(np.float32)
            and len(input_shape) == 2
            and axis == 0
            and len(index_shape) == 1
        ):
            indices = ctx.logical_tensor_eval_data(
                node.input[1], node_inputs[1], index_shape
            )
            if np.all(indices >= 0):
                index_tensor = node_inputs[1]
                if np.dtype(ctx.get_tensor_dtype(node.input[1])) != np.dtype(np.int32):
                    index_tensor = ctx.from_numpy(np.asarray(indices, dtype=np.int32))
                result = ggml.ggml_get_rows(
                    ctx.ggml_eval_context, node_inputs[0], index_tensor
                )
                ctx.register_native_tensor(
                    node.output[0],
                    result,
                    output_shape,
                    input_dtype,
                )
                return

        input_array = ctx.logical_tensor_eval_data(
            node.input[0], node_inputs[0], input_shape
        )
        index_array = ctx.logical_tensor_eval_data(
            node.input[1], node_inputs[1], index_shape
        )
        output = self.eval_numpy(node, (input_array, index_array))[0]
        ctx.set_numpy_runtime_output(node.output[0], output, output.dtype)


@onnx_operators.register
class GatherElementsOperator(OnnxOperator):
    def __init__(self):
        super().__init__("GatherElements")

    def lower(self, ctx: "GgmlOnnxExecutionContext", node: NodeProto) -> None:
        node_inputs = [ctx.ggml_tensors_dict[inp] for inp in node.input]

        if len(node_inputs) != 2:
            raise ValueError(
                f'Error for node "{node.name}": Operation "GatherElements" requires exactly two inputs. Actual number of inputs: {len(node_inputs)}'
            )

        data, indices = node_inputs
        data_shape = ctx.shapes[node.input[0]]
        indices_shape = ctx.shapes[node.input[1]]
        axis = next((attr.i for attr in node.attribute if attr.name == "axis"), 0)
        if axis < 0:
            axis += len(data_shape)

        output_shape = indices_shape
        x_t = ctx.from_numpy(np.empty(output_shape, dtype=get_tensor_dtype(data)))

        @ggml.ggml_custom3_op_t
        def custom_gather_elements(
            tensor_out: ggml.ggml_tensor_p,
            tensor_in_1: ggml.ggml_tensor_p,
            tensor_in_2: ggml.ggml_tensor_p,
            tensor_in_3: ggml.ggml_tensor_p,
            ith: int,
            nth: int,
            userdata: Optional[ctypes.c_void_p],
        ):
            data = ctx.to_numpy(tensor_in_2).reshape(data_shape)
            indices = ctx.to_numpy(tensor_in_3).reshape(indices_shape)
            output = np.take_along_axis(data, indices, axis=axis)
            ctx.set_tensor_data(tensor_out, output)

        new_tensor = ctx.ggml_tensors_dict[node.output[0]] = (
            ggml.ggml_map_custom3_inplace(
                ctx.ggml_eval_context,
                x_t,
                data,
                indices,
                custom_gather_elements,
                1,
                None,
            )
        )
        ctx.refs.append(custom_gather_elements)
        ctx.set_tensor_shape(new_tensor, output_shape)
        ctx.shapes[node.output[0]] = output_shape


@onnx_operators.register
class GatherNDOperator(OnnxOperator):
    def __init__(self):
        super().__init__("GatherND")

    def lower(self, ctx: "GgmlOnnxExecutionContext", node: NodeProto) -> None:
        node_inputs = [ctx.ggml_tensors_dict[inp] for inp in node.input]

        if len(node_inputs) != 2:
            raise ValueError(
                f'Error for node "{node.name}": Operation "GatherND" requires exactly two inputs. Actual number of inputs: {len(node_inputs)}'
            )

        data_shape = ctx.shapes[node.input[0]]
        indices_shape = ctx.shapes[node.input[1]]
        batch_dims = next(
            (attr.i for attr in node.attribute if attr.name == "batch_dims"), 0
        )

        data = ctx.logical_tensor_eval_data(node.input[0], node_inputs[0], data_shape)
        indices = ctx.logical_tensor_eval_data(
            node.input[1], node_inputs[1], indices_shape
        )

        batch_dims_shape = list(indices_shape[:batch_dims])
        batch_dims_size = int(np.prod(batch_dims_shape)) if batch_dims_shape else 1
        index_depth = indices_shape[-1]
        data_rank = len(data_shape)
        if index_depth > data_rank - batch_dims:
            raise ValueError(
                f'Error for node "{node.name}": GatherND index depth exceeds data rank.'
            )
        output_shape = tuple(
            batch_dims_shape
            + list(indices_shape[batch_dims:-1])
            + list(data_shape[batch_dims + index_depth :])
        )

        reshaped_indices = indices.reshape(batch_dims_size, -1, index_depth)
        reshaped_data = data.reshape((batch_dims_size,) + data_shape[batch_dims:])

        output_data = []
        for batch_dim in range(reshaped_indices.shape[0]):
            for outer_dim in range(reshaped_indices.shape[1]):
                gather_index = tuple(
                    reshaped_indices[batch_dim][outer_dim].astype(np.int64)
                )
                output_data.append(reshaped_data[(batch_dim, *gather_index)])

        output = np.asarray(output_data, dtype=data.dtype).reshape(output_shape)
        ctx.set_logical_output(node.output[0], output, output.dtype)


@onnx_operators.register
class GemmOperator(OnnxOperator):
    def __init__(self):
        super().__init__("Gemm")

    @staticmethod
    def broadcast_tensor(
        ctx: "GgmlOnnxExecutionContext",
        tensor: ggml.ggml_tensor_p,
        shape: Tuple[int, ...],
    ):
        ggml_type = ggml.utils.GGML_TYPE(tensor.contents.type)

        if ggml_type == ggml.utils.GGML_TYPE.F32:
            new_tensor = ggml.ggml_new_tensor(
                ctx.ggml_eval_context,
                ggml_type.value,
                len(shape),
                (ctypes.c_int64 * len(shape))(*shape),
            )

            new_tensor = ggml.ggml_repeat(
                ctx.ggml_eval_context,
                tensor,
                new_tensor,
            )
        else:

            @ggml.ggml_custom2_op_t
            def custom_broadcast_to(
                tensor_out: ggml.ggml_tensor_p,
                tensor_in_1: ggml.ggml_tensor_p,
                tensor_in_2: ggml.ggml_tensor_p,
                ith: int,
                nth: int,
                userdata: Optional[ctypes.c_void_p],
            ):
                a = ctx.to_numpy(tensor_in_2)

                x = np.broadcast_to(a, shape)
                ctx.set_tensor_data(tensor_out, x)

            x = np.empty(shape, dtype=ctx.get_raw_tensor_dtype(tensor))
            x_t = ctx.from_numpy(x)
            new_tensor = ggml.ggml_map_custom2_inplace(
                ctx.ggml_eval_context,
                x_t,
                tensor,
                custom_broadcast_to,
                1,
                None,
            )
            ctx.refs.append(custom_broadcast_to)
        return new_tensor

    @classmethod
    def broadcast_shapes(
        cls,
        ctx: "GgmlOnnxExecutionContext",
        a: ggml.ggml_tensor_p,
        b: ggml.ggml_tensor_p,
    ):
        a_shape = ctx.get_tensor_shape(a)
        b_shape = ctx.get_tensor_shape(b)

        output_shape = tuple(
            reversed(np.broadcast(np.empty(a_shape), np.empty(b_shape)).shape)
        )  # TODO: Fix this

        a_shaped = a
        b_shaped = b

        if a_shape != output_shape:
            a_shaped = cls.broadcast_tensor(ctx, a, output_shape)
        if b_shape != output_shape:
            b_shaped = cls.broadcast_tensor(ctx, b, output_shape)

        return a_shaped, b_shaped

    def lower(self, ctx: "GgmlOnnxExecutionContext", node: NodeProto) -> None:
        node_inputs = [ctx.ggml_tensors_dict[inp] for inp in node.input]

        if len(node_inputs) < 2:
            raise ValueError(
                f'Error for node "{node.name}": Operation "Gemm" requires at least two inputs. Actual number of inputs: {len(node_inputs)}'
            )

        node_inputs_iter = iter(node_inputs)

        a = next(node_inputs_iter)
        b = next(node_inputs_iter)
        c = next(node_inputs_iter, None)

        alpha = next((attr.f for attr in node.attribute if attr.name == "alpha"), 1.0)
        beta = next((attr.f for attr in node.attribute if attr.name == "beta"), 1.0)

        transA = next((attr.i for attr in node.attribute if attr.name == "transA"), 0)
        transB = next((attr.i for attr in node.attribute if attr.name == "transB"), 0)

        # TODO: broadcast? Current broadcasting method fails during tests

        a_dtype = ctx.get_tensor_dtype(node.input[0])
        b_dtype = ctx.get_tensor_dtype(node.input[1])

        a_transposed = a
        b_transposed = b

        if transA:
            a_permute = ggml.ggml_transpose(
                ctx.ggml_eval_context,
                a,
            )
            a_shape = ggml.utils.get_shape(a_permute)
            a_transposed = ggml.ggml_cpy(
                ctx.ggml_eval_context,
                a_permute,
                ggml.ggml_new_tensor(
                    ctx.ggml_eval_context,
                    ctx.map_to_ggml_type(a_dtype).value,
                    len(a_shape),
                    (ctypes.c_int64 * len(a_shape))(*a_shape),
                ),
            )

        if not transB:
            b_permute = ggml.ggml_transpose(
                ctx.ggml_eval_context,
                b,
            )
            b_shape = ggml.utils.get_shape(b_permute)
            b_transposed = ggml.ggml_cpy(
                ctx.ggml_eval_context,
                b_permute,
                ggml.ggml_new_tensor(
                    ctx.ggml_eval_context,
                    ctx.map_to_ggml_type(b_dtype).value,
                    len(b_shape),
                    (ctypes.c_int64 * len(b_shape))(*b_shape),
                ),
            )

        # Y = alpha * np.dot(A, B) + beta * C
        # ref: https://github.com/onnx/onnx/blob/main/onnx/backend/test/case/node/gemm.py

        mul_mat_result = ggml.ggml_mul_mat(
            ctx.ggml_eval_context,
            b_transposed,
            a_transposed,
        )

        alpha_t = ctx.from_numpy(
            np.full(
                ctx.get_tensor_shape(mul_mat_result),
                alpha,
                dtype=ctx.get_raw_tensor_dtype(mul_mat_result),
            ),
        )

        mul_mat_result = ggml.ggml_mul_inplace(
            ctx.ggml_eval_context, mul_mat_result, alpha_t
        )

        if c is None:
            c = ctx.from_numpy(
                np.full(
                    ctx.get_tensor_shape(mul_mat_result),
                    0,
                    dtype=ctx.get_raw_tensor_dtype(mul_mat_result),
                ),
            )

        c, mul_mat_result = self.broadcast_shapes(ctx, c, mul_mat_result)

        beta_t = ctx.from_numpy(
            np.full(
                ctx.get_tensor_shape(mul_mat_result),
                beta,
                dtype=ctx.get_raw_tensor_dtype(mul_mat_result),
            ),
        )

        mul_mat_result = ggml.ggml_add_inplace(
            ctx.ggml_eval_context,
            mul_mat_result,
            ggml.ggml_mul_inplace(ctx.ggml_eval_context, c, beta_t),
        )

        ctx.ggml_tensors_dict[node.output[0]] = mul_mat_result
        ctx.set_tensor_shape(mul_mat_result, ctx.get_tensor_shape(mul_mat_result))
        ctx.shapes[node.output[0]] = ctx.get_tensor_shape(mul_mat_result)


@onnx_operators.register
class GeluOperator(OnnxOperator):
    def __init__(self):
        super().__init__(
            "Gelu",
            OnnxOperator.EXECUTION_NATIVE_OR_NUMPY_RUNTIME,
            OnnxOperator.CLASS_CONDITIONAL_NATIVE,
        )
        self.has_numpy_evaluator = True

    @staticmethod
    def approximate_mode(node: NodeProto) -> str:
        return next(
            (
                attr.s.decode("utf-8")
                for attr in node.attribute
                if attr.name == "approximate"
            ),
            "none",
        )

    @staticmethod
    def numpy_gelu(x: npt.NDArray[Any], approximate: str) -> npt.NDArray[Any]:
        if approximate == "none":
            erf = np.vectorize(math.erf)
            return 0.5 * x * (1.0 + erf(x / np.sqrt(2.0)))
        if approximate == "tanh":
            inner = np.sqrt(2.0 / np.pi) * (x + 0.044715 * np.power(x, 3))
            return 0.5 * x * (1.0 + np.tanh(inner))
        raise ValueError(f'Unsupported Gelu approximate mode "{approximate}"')

    def strategies(
        self, tensor_types: Dict[str, TensorType], node: "NodeIR"
    ) -> Tuple[Tuple[str, str, str], ...]:
        approximate = self.string_attribute_value(node.attribute("approximate", "none"))
        if (
            len(node.inputs) == 1
            and self.tensor_type(tensor_types, node.inputs[0]).is_float32
            and approximate in {"none", "tanh"}
        ):
            return self.native_strategy()
        return self.numpy_runtime_strategy(
            "Gelu requires float32 input and approximate=none/tanh to lower native"
        )

    def eval_numpy(
        self, node: NodeProto, inputs: Tuple[npt.NDArray[Any], ...]
    ) -> Tuple[npt.NDArray[Any], ...]:
        if len(inputs) != 1:
            raise ValueError(
                f'Operation "{node.op_type}" requires exactly one input. '
                f"Actual number of inputs: {len(inputs)}"
            )
        return (
            np.asarray(
                self.numpy_gelu(inputs[0], self.approximate_mode(node)),
                dtype=inputs[0].dtype,
            ),
        )

    def lower(self, ctx: "GgmlOnnxExecutionContext", node: NodeProto) -> None:
        approximate = self.approximate_mode(node)
        ggml_func = ggml.ggml_gelu_erf if approximate == "none" else ggml.ggml_gelu

        def gelu(x: npt.NDArray[Any]) -> npt.NDArray[Any]:
            return self.numpy_gelu(x, approximate)

        self.lower_native_unary_or_numpy(ctx, node, ggml_func, gelu)


@onnx_operators.register
class GlobalAveragePoolOperator(OnnxOperator):
    def __init__(self):
        super().__init__(
            "GlobalAveragePool",
            OnnxOperator.EXECUTION_NATIVE_OR_NUMPY_RUNTIME,
            OnnxOperator.CLASS_CONDITIONAL_NATIVE,
        )
        self.has_numpy_evaluator = True

    def strategies(
        self, tensor_types: Dict[str, TensorType], node: "NodeIR"
    ) -> Tuple[Tuple[str, str, str], ...]:
        if len(node.inputs) == 1:
            input_type = self.tensor_type(tensor_types, node.inputs[0])
            if (
                input_type.is_float32
                and input_type.shape is not None
                and len(input_type.shape) == 4
            ):
                return self.native_strategy()
        return self.numpy_runtime_strategy(
            "GlobalAveragePool requires 4D float32 input to lower to ggml_pool_2d"
        )

    def eval_numpy(
        self, node: NodeProto, inputs: Tuple[npt.NDArray[Any], ...]
    ) -> Tuple[npt.NDArray[Any], ...]:
        if len(inputs) != 1:
            raise ValueError(f'Operation "{node.op_type}" requires one input')
        spatial_axes = tuple(range(2, inputs[0].ndim))
        return (np.mean(inputs[0], axis=spatial_axes, keepdims=True),)

    def lower(self, ctx: "GgmlOnnxExecutionContext", node: NodeProto) -> None:
        node_inputs = [ctx.ggml_tensors_dict[inp] for inp in node.input]
        x_shape = ctx.shapes[node.input[0]]
        x_dtype = np.dtype(ctx.get_tensor_dtype(node.input[0]))
        spatial_axes = tuple(range(2, len(x_shape)))
        output_shape = (*x_shape[:2], *((1,) * len(spatial_axes)))

        if (
            ctx.can_emit_native(node.output[0])
            and ctx.can_run_native(node)
            and x_dtype == np.dtype(np.float32)
            and len(x_shape) == 4
        ):
            result = ggml.ggml_pool_2d(
                ctx.ggml_eval_context,
                node_inputs[0],
                ggml.GGML_OP_POOL_AVG,
                x_shape[3],
                x_shape[2],
                1,
                1,
                ctypes.c_float(0.0),
                ctypes.c_float(0.0),
            )
            ctx.register_native_tensor(node.output[0], result, output_shape, x_dtype)
            return

        self.lower_numpy_unary(
            ctx,
            node,
            lambda x: np.mean(x, axis=spatial_axes, keepdims=True),
            output_shape=output_shape,
        )


@onnx_operators.register
class GlobalMaxPoolOperator(OnnxOperator):
    def __init__(self):
        super().__init__(
            "GlobalMaxPool",
            OnnxOperator.EXECUTION_NATIVE_OR_NUMPY_RUNTIME,
            OnnxOperator.CLASS_CONDITIONAL_NATIVE,
        )
        self.has_numpy_evaluator = True

    def strategies(
        self, tensor_types: Dict[str, TensorType], node: "NodeIR"
    ) -> Tuple[Tuple[str, str, str], ...]:
        if len(node.inputs) == 1:
            input_type = self.tensor_type(tensor_types, node.inputs[0])
            if (
                input_type.is_float32
                and input_type.shape is not None
                and len(input_type.shape) == 4
            ):
                return self.native_strategy()
        return self.numpy_runtime_strategy(
            "GlobalMaxPool requires 4D float32 input to lower to ggml_pool_2d"
        )

    def eval_numpy(
        self, node: NodeProto, inputs: Tuple[npt.NDArray[Any], ...]
    ) -> Tuple[npt.NDArray[Any], ...]:
        if len(inputs) != 1:
            raise ValueError(f'Operation "{node.op_type}" requires one input')
        spatial_axes = tuple(range(2, inputs[0].ndim))
        return (np.max(inputs[0], axis=spatial_axes, keepdims=True),)

    def lower(self, ctx: "GgmlOnnxExecutionContext", node: NodeProto) -> None:
        node_inputs = [ctx.ggml_tensors_dict[inp] for inp in node.input]
        x_shape = ctx.shapes[node.input[0]]
        x_dtype = np.dtype(ctx.get_tensor_dtype(node.input[0]))
        spatial_axes = tuple(range(2, len(x_shape)))
        output_shape = (*x_shape[:2], *((1,) * len(spatial_axes)))

        if (
            ctx.can_emit_native(node.output[0])
            and ctx.can_run_native(node)
            and x_dtype == np.dtype(np.float32)
            and len(x_shape) == 4
        ):
            result = ggml.ggml_pool_2d(
                ctx.ggml_eval_context,
                node_inputs[0],
                ggml.GGML_OP_POOL_MAX,
                x_shape[3],
                x_shape[2],
                1,
                1,
                ctypes.c_float(0.0),
                ctypes.c_float(0.0),
            )
            ctx.register_native_tensor(node.output[0], result, output_shape, x_dtype)
            return

        self.lower_numpy_unary(
            ctx,
            node,
            lambda x: np.max(x, axis=spatial_axes, keepdims=True),
            output_shape=output_shape,
        )


@onnx_operators.register
class GreaterOperator(OnnxOperator):
    def __init__(self):
        super().__init__("Greater")

    def lower(self, ctx: "GgmlOnnxExecutionContext", node: NodeProto) -> None:
        self.lower_numpy_binary(ctx, node, np.greater, np.bool_)


@onnx_operators.register
class HardSigmoidOperator(OnnxOperator):
    def __init__(self):
        super().__init__(
            "HardSigmoid",
            OnnxOperator.EXECUTION_NATIVE_OR_NUMPY_RUNTIME,
            OnnxOperator.CLASS_CONDITIONAL_NATIVE,
        )
        self.has_numpy_evaluator = True

    def strategies(
        self, tensor_types: Dict[str, TensorType], node: "NodeIR"
    ) -> Tuple[Tuple[str, str, str], ...]:
        alpha = float(node.attribute("alpha", 0.2))
        beta = float(node.attribute("beta", 0.5))
        if (
            len(node.inputs) == 1
            and self.tensor_type(tensor_types, node.inputs[0]).is_float32
            and math.isclose(alpha, 1.0 / 6.0, rel_tol=1e-6, abs_tol=1e-7)
            and math.isclose(beta, 0.5, rel_tol=1e-6, abs_tol=1e-7)
        ):
            return self.native_strategy()
        return self.numpy_runtime_strategy(
            "HardSigmoid requires float32 input, alpha=1/6, and beta=0.5 "
            "to lower to ggml_hardsigmoid"
        )

    def eval_numpy(
        self, node: NodeProto, inputs: Tuple[npt.NDArray[Any], ...]
    ) -> Tuple[npt.NDArray[Any], ...]:
        alpha = self.float_attribute(node, "alpha", 0.2)
        beta = self.float_attribute(node, "beta", 0.5)
        return self.eval_numpy_unary_operator(
            node, inputs, lambda x: np.clip((x * alpha) + beta, 0, 1)
        )

    def lower(self, ctx: "GgmlOnnxExecutionContext", node: NodeProto) -> None:
        alpha = self.float_attribute(node, "alpha", 0.2)
        beta = self.float_attribute(node, "beta", 0.5)

        def hard_sigmoid(x: npt.NDArray[Any]) -> npt.NDArray[Any]:
            return np.clip((x * alpha) + beta, 0, 1)

        if math.isclose(alpha, 1.0 / 6.0, rel_tol=1e-6, abs_tol=1e-7) and math.isclose(
            beta, 0.5, rel_tol=1e-6, abs_tol=1e-7
        ):
            self.lower_native_unary_or_numpy(
                ctx, node, ggml.ggml_hardsigmoid, hard_sigmoid
            )
            return

        self.lower_numpy_unary(ctx, node, hard_sigmoid)


@onnx_operators.register
class HardSwishOperator(OnnxOperator):
    def __init__(self):
        super().__init__(
            "HardSwish",
            OnnxOperator.EXECUTION_NATIVE_OR_NUMPY_RUNTIME,
            OnnxOperator.CLASS_CONDITIONAL_NATIVE,
        )
        self.has_numpy_evaluator = True

    def strategies(
        self, tensor_types: Dict[str, TensorType], node: "NodeIR"
    ) -> Tuple[Tuple[str, str, str], ...]:
        if (
            len(node.inputs) == 1
            and self.tensor_type(tensor_types, node.inputs[0]).is_float32
        ):
            return self.native_strategy()
        return self.numpy_runtime_strategy(
            "HardSwish requires float32 input to lower to ggml_hardswish"
        )

    def eval_numpy(
        self, node: NodeProto, inputs: Tuple[npt.NDArray[Any], ...]
    ) -> Tuple[npt.NDArray[Any], ...]:
        return self.eval_numpy_unary_operator(
            node, inputs, lambda x: x * np.maximum(0, np.minimum(1, x / 6 + 0.5))
        )

    def lower(self, ctx: "GgmlOnnxExecutionContext", node: NodeProto) -> None:
        def hardswish(x: npt.NDArray[Any]) -> npt.NDArray[Any]:
            return x * np.maximum(0, np.minimum(1, x / 6 + 0.5))

        self.lower_native_unary_or_numpy(ctx, node, ggml.ggml_hardswish, hardswish)


@onnx_operators.register
class HammingWindowOperator(OnnxOperator):
    def __init__(self):
        super().__init__("HammingWindow")

    def lower(self, ctx: "GgmlOnnxExecutionContext", node: NodeProto) -> None:
        def hamming(values: npt.NDArray[Any], denominator: float) -> npt.NDArray[Any]:
            a0 = 25 / 46
            return a0 - (1 - a0) * np.cos(2 * np.pi * values / denominator)

        self.lower_window(ctx, node, hamming)


@onnx_operators.register
class HannWindowOperator(OnnxOperator):
    def __init__(self):
        super().__init__("HannWindow")

    def lower(self, ctx: "GgmlOnnxExecutionContext", node: NodeProto) -> None:
        def hann(values: npt.NDArray[Any], denominator: float) -> npt.NDArray[Any]:
            return 0.5 - 0.5 * np.cos(2 * np.pi * values / denominator)

        self.lower_window(ctx, node, hann)


@onnx_operators.register
class GridSampleOperator(OnnxOperator):
    def __init__(self):
        super().__init__("GridSample")

    def lower(self, ctx: "GgmlOnnxExecutionContext", node: NodeProto) -> None:
        node_inputs = [ctx.ggml_tensors_dict[inp] for inp in node.input]

        if len(node_inputs) != 2:
            raise ValueError(
                f'Error for node "{node.name}": Operation "GridSample" requires exactly two inputs. Actual number of inputs: {len(node_inputs)}'
            )

        x_shape = ctx.shapes[node.input[0]]
        grid_shape = ctx.shapes[node.input[1]]
        x = ctx.to_numpy(ctx.eval_tensor(node_inputs[0])).reshape(x_shape)
        grid = ctx.to_numpy(ctx.eval_tensor(node_inputs[1])).reshape(grid_shape)
        mode = next(
            (attr.s for attr in node.attribute if attr.name == "mode"), b"bilinear"
        )
        padding_mode = next(
            (attr.s for attr in node.attribute if attr.name == "padding_mode"), b"zeros"
        )
        align_corners = next(
            (attr.i for attr in node.attribute if attr.name == "align_corners"), 0
        )

        if mode not in {b"bilinear", b"linear", b"nearest", b"bicubic", b"cubic"}:
            raise NotImplementedError(
                f'Error for node "{node.name}": GridSample mode {mode!r} is not implemented.'
            )
        if padding_mode not in {b"zeros", b"border", b"reflection"}:
            raise ValueError(
                f'Error for node "{node.name}": Unknown GridSample padding mode {padding_mode!r}.'
            )

        mode_name = mode.decode("utf-8") if isinstance(mode, bytes) else str(mode)
        padding_mode_name = (
            padding_mode.decode("utf-8")
            if isinstance(padding_mode, bytes)
            else str(padding_mode)
        )
        if mode_name == "bilinear":
            mode_name = "linear"
        elif mode_name == "bicubic":
            mode_name = "cubic"

        def clamp(value: int, lower: int, upper: int) -> int:
            return max(lower, min(value, upper))

        def denormalize(coord: float, size: int) -> float:
            if align_corners:
                return (coord + 1.0) * (size - 1) / 2.0
            return ((coord + 1.0) * size - 1.0) / 2.0

        def reflect(coord: float, lower: float, upper: float) -> float:
            span = upper - lower
            if span == 0:
                return lower
            if coord < lower:
                delta = lower - coord
                count = int(delta / span)
                remainder = delta - count * span
                return lower + remainder if count % 2 == 0 else upper - remainder
            if coord > upper:
                delta = coord - upper
                count = int(delta / span)
                remainder = delta - count * span
                return upper - remainder if count % 2 == 0 else lower + remainder
            return coord

        def border_for_dims(dims: Tuple[int, ...]) -> npt.NDArray[np.float64]:
            border = np.zeros(len(dims) * 2, dtype=np.float64)
            for index, dim in enumerate(dims):
                if align_corners:
                    border[index] = 0.0
                    border[index + len(dims)] = float(dim - 1)
                else:
                    border[index] = -0.5
                    border[index + len(dims)] = float(dim) - 0.5
            return border

        def pixel_at_array(
            array: npt.NDArray[Any],
            index: int,
            border: Sequence[float],
        ) -> Any:
            size = array.shape[0]
            if padding_mode_name == "zeros":
                if 0 <= index < size:
                    return array[index]
                return array.dtype.type(0)
            if padding_mode_name == "border":
                return array[clamp(index, 0, size - 1)]
            reflected = int(reflect(index, border[0], border[1]))
            return array[reflected]

        def pixel_at_ndarray(
            array: npt.NDArray[Any],
            indices: Sequence[int],
            border: Sequence[float],
        ) -> Any:
            num_dims = array.ndim
            if num_dims == 1:
                return pixel_at_array(array, indices[0], border)
            index = indices[0]
            size = array.shape[0]
            if padding_mode_name == "zeros":
                if 0 <= index < size:
                    next_array = array[index]
                else:
                    next_array = np.zeros_like(array[0])
            elif padding_mode_name == "border":
                next_array = array[clamp(index, 0, size - 1)]
            else:
                next_array = array[int(reflect(index, border[0], border[num_dims]))]
            next_border = list(border[1:num_dims]) + list(
                border[1 + num_dims : 2 * num_dims]
            )
            return pixel_at_ndarray(next_array, indices[1:], next_border)

        def cubic_coefficients(value: float) -> Tuple[float, float, float, float]:
            cubic_alpha = -0.75
            x_abs = abs(value)
            return (
                (
                    (cubic_alpha * (x_abs + 1) - 5 * cubic_alpha) * (x_abs + 1)
                    + 8 * cubic_alpha
                )
                * (x_abs + 1)
                - 4 * cubic_alpha,
                ((cubic_alpha + 2) * x_abs - (cubic_alpha + 3)) * x_abs * x_abs + 1,
                ((cubic_alpha + 2) * (1 - x_abs) - (cubic_alpha + 3))
                * (1 - x_abs)
                * (1 - x_abs)
                + 1,
                (
                    (cubic_alpha * (2 - x_abs) - 5 * cubic_alpha) * (2 - x_abs)
                    + 8 * cubic_alpha
                )
                * (2 - x_abs)
                - 4 * cubic_alpha,
            )

        def linear_interpolate_1d(
            data: npt.NDArray[Any],
            coord: float,
            border: Sequence[float],
        ) -> Any:
            index = int(np.floor(coord))
            weight = abs(coord - index)
            left = pixel_at_array(data, index, border)
            right = pixel_at_array(data, index + 1, border)
            return left * (1.0 - weight) + right * weight

        def cubic_interpolate_1d(
            data: npt.NDArray[Any],
            coord: float,
            border: Sequence[float],
        ) -> Any:
            index = int(np.floor(coord))
            coeffs = cubic_coefficients(coord - index)
            return sum(
                coeffs[offset + 1] * pixel_at_array(data, index + offset, border)
                for offset in (-1, 0, 1, 2)
            )

        def linear_interpolate_nd(
            data: npt.NDArray[Any],
            coords: Sequence[float],
            border: Sequence[float],
        ) -> Any:
            num_dims = data.ndim
            if num_dims == 1:
                return linear_interpolate_1d(data, coords[0], border)
            values = np.asarray(
                [
                    linear_interpolate_nd(
                        data[index],
                        coords[1:],
                        list(border[1:num_dims])
                        + list(border[1 + num_dims : 2 * num_dims]),
                    )
                    for index in range(data.shape[0])
                ],
                dtype=data.dtype,
            )
            return linear_interpolate_1d(
                values, coords[0], [border[0], border[num_dims]]
            )

        def cubic_interpolate_nd(
            data: npt.NDArray[Any],
            coords: Sequence[float],
            border: Sequence[float],
        ) -> Any:
            num_dims = data.ndim
            if num_dims == 1:
                return cubic_interpolate_1d(data, coords[0], border)
            values = np.asarray(
                [
                    cubic_interpolate_nd(
                        data[index],
                        coords[1:],
                        list(border[1:num_dims])
                        + list(border[1 + num_dims : 2 * num_dims]),
                    )
                    for index in range(data.shape[0])
                ],
                dtype=data.dtype,
            )
            return cubic_interpolate_1d(
                values, coords[0], [border[0], border[num_dims]]
            )

        batch_size = x_shape[0]
        channels = x_shape[1]
        spatial_dims = tuple(x_shape[2:])
        output_spatial_shape = tuple(grid_shape[1:-1])
        output = np.empty(
            (batch_size, channels, *output_spatial_shape),
            dtype=get_tensor_dtype(node_inputs[0]),
        )
        border = border_for_dims(spatial_dims)

        for batch in range(batch_size):
            grid_data = grid[batch]
            for channel in range(channels):
                x_data = x[batch, channel]
                for output_index in np.ndindex(*output_spatial_shape):
                    normalized_coords = grid_data[output_index][::-1]
                    coords = np.asarray(
                        [
                            denormalize(float(coord), spatial_dims[index])
                            for index, coord in enumerate(normalized_coords)
                        ],
                        dtype=np.float32,
                    )
                    if mode_name == "nearest":
                        coords = np.rint(coords).astype(np.int32)
                    for index, coord in enumerate(coords):
                        lower = border[index]
                        upper = border[index + len(spatial_dims)]
                        if coord < lower or coord > upper:
                            if padding_mode_name == "border":
                                coords[index] = max(
                                    0.0,
                                    min(float(coord), float(spatial_dims[index] - 1)),
                                )
                            elif padding_mode_name == "reflection":
                                coords[index] = reflect(float(coord), lower, upper)
                    if mode_name == "nearest":
                        output[(batch, channel, *output_index)] = pixel_at_ndarray(
                            x_data,
                            coords,
                            border,
                        )
                    elif mode_name == "linear":
                        output[(batch, channel, *output_index)] = linear_interpolate_nd(
                            x_data, coords, border
                        )
                    elif mode_name == "cubic":
                        output[(batch, channel, *output_index)] = cubic_interpolate_nd(
                            x_data, coords, border
                        )
                    else:
                        raise RuntimeError(
                            f"GridSample interpolation mode {mode_name!r} is not implemented."
                        )

        new_tensor = ctx.ggml_tensors_dict[node.output[0]] = ctx.from_numpy(output)
        ctx.set_tensor_shape(new_tensor, output.shape)
        ctx.shapes[node.output[0]] = output.shape
        ctx.set_tensor_dtype(node.output[0], output.dtype)
        return


@onnx_operators.register
class GroupNormalizationOperator(OnnxOperator):
    def __init__(self):
        super().__init__("GroupNormalization")

    def lower(self, ctx: "GgmlOnnxExecutionContext", node: NodeProto) -> None:
        node_inputs = [ctx.ggml_tensors_dict[inp] for inp in node.input]

        if len(node_inputs) != 3:
            raise ValueError(
                f'Error for node "{node.name}": Operation "GroupNormalization" requires exactly three inputs. Actual number of inputs: {len(node_inputs)}'
            )

        x_shape = ctx.shapes[node.input[0]]
        scale_shape = ctx.shapes[node.input[1]]
        bias_shape = ctx.shapes[node.input[2]]
        num_groups = next(
            (attr.i for attr in node.attribute if attr.name == "num_groups"), None
        )
        if num_groups is None:
            raise ValueError(
                f'Error for node "{node.name}": Operation "GroupNormalization" requires "num_groups" attribute.'
            )
        epsilon = next(
            (attr.f for attr in node.attribute if attr.name == "epsilon"), 1e-5
        )

        x = ctx.to_numpy(ctx.eval_tensor(node_inputs[0])).reshape(x_shape)
        scale = ctx.to_numpy(ctx.eval_tensor(node_inputs[1])).reshape(scale_shape)
        bias = ctx.to_numpy(ctx.eval_tensor(node_inputs[2])).reshape(bias_shape)
        if x_shape[1] % num_groups != 0:
            raise ValueError(
                f'Error for node "{node.name}": channel dimension must be divisible by num_groups.'
            )
        group_size = x_shape[1] // num_groups
        grouped_shape = (x_shape[0], num_groups, group_size, *x_shape[2:])
        x_grouped = x.reshape(grouped_shape)
        axes = tuple(range(2, len(grouped_shape)))
        mean = np.mean(x_grouped, axis=axes, keepdims=True)
        variance = np.var(x_grouped, axis=axes, keepdims=True)
        normalized_grouped = (x_grouped - mean) / np.sqrt(variance + epsilon)
        opset_version = ctx.get_opset_version(node.domain)
        if opset_version is None:
            if scale.size == x_shape[1] and bias.size == x_shape[1]:
                scale_mode = "channel"
            elif scale.size == num_groups and bias.size == num_groups:
                scale_mode = "group"
            else:
                raise ValueError(
                    f'Error for node "{node.name}": scale and bias must have one value per channel or one value per group.'
                )
        elif opset_version >= 21:
            if scale.size != x_shape[1] or bias.size != x_shape[1]:
                raise ValueError(
                    f'Error for node "{node.name}": scale and bias must have one value per channel.'
                )
            scale_mode = "channel"
        else:
            if scale.size != num_groups or bias.size != num_groups:
                raise ValueError(
                    f'Error for node "{node.name}": scale and bias must have one value per group.'
                )
            scale_mode = "group"

        if scale_mode == "channel":
            normalized = normalized_grouped.reshape(x_shape)
            broadcast_shape = (1, x_shape[1], *((1,) * len(x_shape[2:])))
            output = scale.reshape(broadcast_shape) * normalized + bias.reshape(
                broadcast_shape
            )
        else:
            broadcast_shape = (1, num_groups, *((1,) * (len(grouped_shape) - 2)))
            output = (
                scale.reshape(broadcast_shape) * normalized_grouped
                + bias.reshape(broadcast_shape)
            ).reshape(x_shape)
        output = output.astype(get_tensor_dtype(node_inputs[0]))

        new_tensor = ctx.ggml_tensors_dict[node.output[0]] = ctx.from_numpy(output)
        ctx.set_tensor_shape(new_tensor, output.shape)
        ctx.shapes[node.output[0]] = output.shape
        ctx.set_tensor_dtype(node.output[0], output.dtype)


@onnx_operators.register
class HardmaxOperator(OnnxOperator):
    def __init__(self):
        super().__init__("Hardmax")

    def lower(self, ctx: "GgmlOnnxExecutionContext", node: NodeProto) -> None:
        node_inputs = [ctx.ggml_tensors_dict[inp] for inp in node.input]

        if len(node_inputs) != 1:
            raise ValueError(
                f'Error for node "{node.name}": Operation "Hardmax" requires exactly one input. Actual number of inputs: {len(node_inputs)}'
            )

        x = node_inputs[0]
        axis = next((attr.i for attr in node.attribute if attr.name == "axis"), -1)
        axis_c = ctypes.c_int(axis)

        @ggml.ggml_custom1_op_t
        def custom_hardmax(
            tensor_out: ggml.ggml_tensor_p,
            tensor_in_1: ggml.ggml_tensor_p,
            ith: int,
            nth: int,
            userdata: Optional[ctypes.c_void_p],
        ):
            axis = ctypes.cast(userdata, ctypes.POINTER(ctypes.c_int)).contents.value
            x = ctx.to_numpy(tensor_in_1)

            max_indices = np.argmax(x, axis=axis, keepdims=True)
            y = np.zeros_like(x)
            np.put_along_axis(y, max_indices, 1, axis=axis)

            ctx.set_tensor_data(tensor_out, y)

        new_tensor = ctx.ggml_tensors_dict[node.output[0]] = (
            ggml.ggml_map_custom1_inplace(
                ctx.ggml_eval_context,
                x,
                custom_hardmax,
                1,
                ctypes.pointer(axis_c),
            )
        )

        ctx.refs.append(custom_hardmax)

        ctx.refs.append(axis_c)
        ctx.set_tensor_shape(new_tensor, ctx.shapes[node.input[0]])
        ctx.shapes[node.output[0]] = ctx.shapes[node.input[0]]


@onnx_operators.register
class IdentityOperator(OnnxOperator):
    def __init__(self):
        super().__init__("Identity", execution=OnnxOperator.EXECUTION_NATIVE)

    def lower(self, ctx: "GgmlOnnxExecutionContext", node: NodeProto) -> None:
        node_inputs = [ctx.ggml_tensors_dict[inp] for inp in node.input]

        if len(node_inputs) != 1:
            raise ValueError(
                f'Error for node "{node.name}": Operation "Identity" requires exactly one input. Actual number of inputs: {len(node_inputs)}'
            )

        output_name = node.output[0]

        ctx.alias_tensor(output_name, node.input[0], ctx.shapes[node.input[0]])


@onnx_operators.register
class InstanceNormalizationOperator(OnnxOperator):
    def __init__(self):
        super().__init__("InstanceNormalization")

    def lower(self, ctx: "GgmlOnnxExecutionContext", node: NodeProto) -> None:
        node_inputs = [ctx.ggml_tensors_dict[inp] for inp in node.input]

        if len(node_inputs) != 3:
            raise ValueError(
                f'Error for node "{node.name}": Operation "InstanceNormalization" requires exactly three inputs. Actual number of inputs: {len(node_inputs)}'
            )
        input_tensor, scale, B = node_inputs
        input_shape = ctx.shapes[node.input[0]]
        input_dtype = get_tensor_dtype(input_tensor)
        scale_data = ctx.to_numpy(ctx.eval_tensor(scale))
        bias_data = ctx.to_numpy(ctx.eval_tensor(B))
        epsilon = next(
            (attr.f for attr in node.attribute if attr.name == "epsilon"), 1e-05
        )
        epsilon_c = ctypes.c_double(epsilon)
        x_t = ctx.from_numpy(np.empty(input_shape, dtype=input_dtype))

        @ggml.ggml_custom2_op_t
        def custom_instancenorm(
            tensor_out: ggml.ggml_tensor_p,
            tensor_in_1: ggml.ggml_tensor_p,
            tensor_in_2: ggml.ggml_tensor_p,
            ith: int,
            nth: int,
            userdata: Optional[ctypes.c_void_p],
        ):
            x = ctx.to_numpy(tensor_in_2).reshape(input_shape)
            s = scale_data
            bias = bias_data
            epsilon = ctypes.cast(
                userdata, ctypes.POINTER(ctypes.c_double)
            ).contents.value

            dims_x = len(x.shape)
            axis = tuple(range(2, dims_x))
            mean = np.mean(x, axis=axis, keepdims=True)
            var = np.var(x, axis=axis, keepdims=True)

            dim_ones = (1,) * (dims_x - 2)
            s = s.reshape(-1, *dim_ones)
            bias = bias.reshape(-1, *dim_ones)

            y = s * (x - mean) / np.sqrt(var + epsilon) + bias
            ctx.set_tensor_data(tensor_out, y)

        new_tensor = ctx.ggml_tensors_dict[node.output[0]] = (
            ggml.ggml_map_custom2_inplace(
                ctx.ggml_eval_context,
                x_t,
                input_tensor,
                custom_instancenorm,
                1,
                ctypes.pointer(epsilon_c),
            )
        )
        ctx.refs.append(custom_instancenorm)
        ctx.refs.append(epsilon_c)
        ctx.set_tensor_shape(new_tensor, ctx.shapes[node.input[0]])
        ctx.shapes[node.output[0]] = ctx.shapes[node.input[0]]


@onnx_operators.register
class IsInfOperator(OnnxOperator):
    def __init__(self):
        super().__init__("IsInf")

    def lower(self, ctx: "GgmlOnnxExecutionContext", node: NodeProto) -> None:
        detect_negative = next(
            (attr.i for attr in node.attribute if attr.name == "detect_negative"), 1
        )
        detect_positive = next(
            (attr.i for attr in node.attribute if attr.name == "detect_positive"), 1
        )

        def isinf(a: npt.NDArray[Any]) -> npt.NDArray[Any]:
            result = np.zeros(a.shape, dtype=np.bool_)
            if detect_negative:
                result |= np.isneginf(a)
            if detect_positive:
                result |= np.isposinf(a)
            return result

        self.lower_numpy_unary(ctx, node, isinf, np.dtype(np.bool_))


@onnx_operators.register
class IsNaNOperator(OnnxOperator):
    def __init__(self):
        super().__init__("IsNaN")

    def lower(self, ctx: "GgmlOnnxExecutionContext", node: NodeProto) -> None:
        self.lower_numpy_unary(ctx, node, np.isnan, np.dtype(np.bool_))


@onnx_operators.register
class LRNOperator(OnnxOperator):
    def __init__(self):
        super().__init__("LRN")

    def lower(self, ctx: "GgmlOnnxExecutionContext", node: NodeProto) -> None:
        node_inputs = [ctx.ggml_tensors_dict[inp] for inp in node.input]

        if len(node_inputs) != 1:
            raise ValueError(
                f'Error for node "{node.name}": Operation "LRN" requires exactly one input. Actual number of inputs: {len(node_inputs)}'
            )

        x = node_inputs[0]
        alpha = next(
            (attr.f for attr in node.attribute if attr.name == "alpha"), 0.0001
        )
        beta = next((attr.f for attr in node.attribute if attr.name == "beta"), 0.75)
        bias = next((attr.f for attr in node.attribute if attr.name == "bias"), 1.0)
        size = next((attr.i for attr in node.attribute if attr.name == "size"), None)

        if size is None:
            raise ValueError(
                f'Error for node "{node.name}": Operation "LRN" requires "size" attibute.'
            )

        input_shape = ctx.shapes[node.input[0]]
        input_dtype = np.dtype(ctx.get_tensor_dtype(node.input[0]))
        lrn_userdata = LRNUserData(alpha, beta, bias, size)
        userdata_p = ctypes.cast(ctypes.pointer(lrn_userdata), ctypes.c_void_p)

        @ggml.ggml_custom1_op_t
        def custom_lrn(
            tensor_out: ggml.ggml_tensor_p,
            tensor_in_1: ggml.ggml_tensor_p,
            ith: int,
            nth: int,
            userdata: Optional[ctypes.c_void_p],
        ):
            userdata_data_ptr = ctypes.cast(userdata, ctypes.POINTER(LRNUserData))
            userdata_data = userdata_data_ptr.contents

            alpha = userdata_data.alpha
            beta = userdata_data.beta
            bias = userdata_data.bias
            size = userdata_data.size

            x = ctx.logical_tensor_data(node.input[0], tensor_in_1, input_shape)

            channel_count = x.shape[1]
            square_sum = np.zeros(x.shape, dtype=x.dtype)
            for n, c, h, w in np.ndindex(x.shape):
                square_sum[n, c, h, w] = np.sum(
                    x[
                        n,
                        max(0, c - int(math.floor((size - 1) / 2))) : min(
                            channel_count, c + int(math.ceil((size - 1) / 2)) + 1
                        ),
                        h,
                        w,
                    ]
                    ** 2
                )
            y = x / ((bias + (alpha / size) * square_sum) ** beta)

            ctx.set_tensor_data(tensor_out, y)

        new_tensor = ctx.ggml_tensors_dict[node.output[0]] = (
            ggml.ggml_map_custom1_inplace(
                ctx.ggml_eval_context,
                x,
                custom_lrn,
                1,
                userdata_p,
            )
        )

        ctx.refs.append(custom_lrn)
        ctx.refs.append(lrn_userdata)
        ctx.set_tensor_shape(new_tensor, input_shape)
        ctx.shapes[node.output[0]] = input_shape
        ctx.set_tensor_dtype(node.output[0], input_dtype)


@onnx_operators.register
class LpPoolOperator(OnnxOperator):
    def __init__(self):
        super().__init__("LpPool")

    def lower(self, ctx: "GgmlOnnxExecutionContext", node: NodeProto) -> None:
        self.lower_pool(ctx, node, "Lp")


@onnx_operators.register
class LpNormalizationOperator(OnnxOperator):
    def __init__(self):
        super().__init__(
            "LpNormalization",
            OnnxOperator.EXECUTION_NATIVE_OR_NUMPY_RUNTIME,
            OnnxOperator.CLASS_CONDITIONAL_NATIVE,
        )
        self.has_numpy_evaluator = True

    @staticmethod
    def normalized_axis(node: NodeProto, rank: int) -> int:
        axis = OnnxOperator.int_attribute(node, "axis", -1)
        return axis + rank if axis < 0 else axis

    def strategies(
        self, tensor_types: Dict[str, TensorType], node: "NodeIR"
    ) -> Tuple[Tuple[str, str, str], ...]:
        if len(node.inputs) == 1:
            input_type = self.tensor_type(tensor_types, node.inputs[0])
            axis = int(node.attribute("axis", -1))
            p = int(node.attribute("p", 2))
            if input_type.shape is not None:
                axis = axis + len(input_type.shape) if axis < 0 else axis
            if (
                input_type.is_float32
                and input_type.shape is not None
                and p == 2
                and axis == len(input_type.shape) - 1
            ):
                return self.native_strategy()
        return self.numpy_runtime_strategy(
            "LpNormalization requires float32 input, p=2, and last-axis "
            "normalization to lower to ggml_l2_norm"
        )

    def eval_numpy(
        self, node: NodeProto, inputs: Tuple[npt.NDArray[Any], ...]
    ) -> Tuple[npt.NDArray[Any], ...]:
        if len(inputs) != 1:
            raise ValueError(f'Operation "{node.op_type}" requires one input')
        x = inputs[0]
        p = self.int_attribute(node, "p", 2)
        axis = self.normalized_axis(node, x.ndim)
        norm = np.sum(np.abs(x) ** p, axis=axis, keepdims=True) ** (1.0 / p)
        result = np.divide(x, norm, out=np.zeros_like(x), where=norm != 0)
        return (np.asarray(result, dtype=x.dtype),)

    def lower(self, ctx: "GgmlOnnxExecutionContext", node: NodeProto) -> None:
        node_inputs = [ctx.ggml_tensors_dict[inp] for inp in node.input]
        if len(node_inputs) != 1:
            raise ValueError(
                f'Error for node "{node.name}": LpNormalization requires one input'
            )
        input_name = node.input[0]
        input_shape = ctx.shapes[input_name]
        input_dtype = np.dtype(ctx.get_tensor_dtype(input_name))
        p = self.int_attribute(node, "p", 2)
        axis = self.normalized_axis(node, len(input_shape))
        if (
            ctx.can_emit_native(node.output[0])
            and ctx.can_run_native(node)
            and input_dtype == np.dtype(np.float32)
            and p == 2
            and axis == len(input_shape) - 1
        ):
            result = ggml.ggml_l2_norm(ctx.ggml_eval_context, node_inputs[0], 1e-12)
            ctx.register_native_tensor(node.output[0], result, input_shape, input_dtype)
            return

        input_array = ctx.logical_tensor_eval_data(
            input_name, node_inputs[0], input_shape
        )
        output = self.eval_numpy(node, (input_array,))[0]
        ctx.set_numpy_runtime_output(node.output[0], output, output.dtype)


@onnx_operators.register
class LeakyReluOperator(OnnxOperator):
    def __init__(self):
        super().__init__(
            "LeakyRelu",
            OnnxOperator.EXECUTION_NATIVE_OR_NUMPY_RUNTIME,
            OnnxOperator.CLASS_CONDITIONAL_NATIVE,
        )
        self.has_numpy_evaluator = True

    def strategies(
        self, tensor_types: Dict[str, TensorType], node: "NodeIR"
    ) -> Tuple[Tuple[str, str, str], ...]:
        if (
            len(node.inputs) == 1
            and self.tensor_type(tensor_types, node.inputs[0]).is_float32
        ):
            return self.native_strategy()
        return self.numpy_runtime_strategy()

    def eval_numpy(
        self, node: NodeProto, inputs: Tuple[npt.NDArray[Any], ...]
    ) -> Tuple[npt.NDArray[Any], ...]:
        return self.eval_numpy_leaky_relu_operator(node, inputs)

    def lower(self, ctx: "GgmlOnnxExecutionContext", node: NodeProto) -> None:
        node_inputs = [ctx.ggml_tensors_dict[inp] for inp in node.input]

        if len(node_inputs) != 1:
            raise ValueError(
                f'Error for node "{node.name}": Operation "LeakyRelu" requires exactly one input. Actual number of inputs: {len(node_inputs)}'
            )

        x = node_inputs[0]
        alpha = next((attr.f for attr in node.attribute if attr.name == "alpha"), 0.01)
        output_name = node.output[0]
        input_name = node.input[0]
        input_shape = ctx.shapes[input_name]
        input_dtype = np.dtype(ctx.get_tensor_dtype(input_name))

        if (
            ctx.can_emit_native(output_name)
            and ctx.can_run_native(node)
            and input_dtype == np.dtype(np.float32)
        ):
            result = ggml.ggml_leaky_relu(ctx.ggml_eval_context, x, float(alpha), False)
            ctx.register_native_tensor(output_name, result, input_shape, input_dtype)
            return

        axis_c = ctypes.c_double(alpha)

        @ggml.ggml_custom1_op_t
        def custom_leaky_relu(
            tensor_out: ggml.ggml_tensor_p,
            tensor_in_1: ggml.ggml_tensor_p,
            ith: int,
            nth: int,
            userdata: Optional[ctypes.c_void_p],
        ):
            alpha = ctypes.cast(
                userdata, ctypes.POINTER(ctypes.c_double)
            ).contents.value
            x = ctx.to_numpy(tensor_in_1)
            y = np.clip(x, 0, np.inf) + np.clip(x, -np.inf, 0) * alpha

            ctx.set_tensor_data(tensor_out, y)

        new_tensor = ggml.ggml_map_custom1_inplace(
            ctx.ggml_eval_context,
            x,
            custom_leaky_relu,
            1,
            ctypes.pointer(axis_c),
        )

        ctx.refs.append(custom_leaky_relu)
        ctx.refs.append(axis_c)
        ctx.register_numpy_runtime_tensor(
            output_name, new_tensor, input_shape, input_dtype
        )


@onnx_operators.register
class LayerNormalizationOperator(OnnxOperator):
    def __init__(self):
        super().__init__("LayerNormalization")

    def lower(self, ctx: "GgmlOnnxExecutionContext", node: NodeProto) -> None:
        node_inputs = [ctx.ggml_tensors_dict[inp] for inp in node.input]

        if len(node_inputs) not in {2, 3}:
            raise ValueError(
                f'Error for node "{node.name}": Operation "LayerNormalization" requires two or three inputs. Actual number of inputs: {len(node_inputs)}'
            )

        x_shape = ctx.shapes[node.input[0]]
        x_dtype = get_tensor_dtype(node_inputs[0])
        axis = next((attr.i for attr in node.attribute if attr.name == "axis"), -1)
        epsilon = next(
            (attr.f for attr in node.attribute if attr.name == "epsilon"), 1e-5
        )
        stash_type = next(
            (attr.i for attr in node.attribute if attr.name == "stash_type"), 1
        )

        if axis < 0:
            axis += len(x_shape)
        if axis < 0 or axis > len(x_shape):
            raise ValueError(
                f'Error for node "{node.name}": LayerNormalization axis {axis} is out of bounds for rank {len(x_shape)}'
            )

        compute_dtype = np.float32 if stash_type == 1 else x_dtype
        axes = tuple(range(axis, len(x_shape)))
        x = (
            ctx.to_numpy(ctx.eval_tensor(node_inputs[0]))
            .reshape(x_shape)
            .astype(compute_dtype)
        )
        scale = ctx.to_numpy(ctx.eval_tensor(node_inputs[1])).reshape(
            ctx.shapes[node.input[1]]
        )
        bias = (
            ctx.to_numpy(ctx.eval_tensor(node_inputs[2])).reshape(
                ctx.shapes[node.input[2]]
            )
            if len(node_inputs) == 3
            else 0
        )

        mean = np.mean(x, axis=axes, keepdims=True)
        variance = np.mean(np.square(x - mean), axis=axes, keepdims=True)
        inv_std = np.reciprocal(np.sqrt(variance + epsilon))
        output = ((x - mean) * inv_std * scale + bias).astype(x_dtype)

        output_tensor = ctx.ggml_tensors_dict[node.output[0]] = ctx.from_numpy(output)
        ctx.set_tensor_shape(output_tensor, output.shape)
        ctx.shapes[node.output[0]] = output.shape
        ctx.set_tensor_dtype(node.output[0], output.dtype)

        optional_outputs = (
            (node.output[1], mean.astype(compute_dtype)),
            (node.output[2], inv_std.astype(compute_dtype)),
        )
        for output_name, output_value in optional_outputs[
            : max(0, len(node.output) - 1)
        ]:
            if output_name == "":
                continue
            tensor = ctx.ggml_tensors_dict[output_name] = ctx.from_numpy(output_value)
            ctx.set_tensor_shape(tensor, output_value.shape)
            ctx.shapes[output_name] = output_value.shape
            ctx.set_tensor_dtype(output_name, output_value.dtype)


@onnx_operators.register
class GreaterOrEqualOperator(OnnxOperator):
    def __init__(self):
        super().__init__("GreaterOrEqual")

    def lower(self, ctx: "GgmlOnnxExecutionContext", node: NodeProto) -> None:
        self.lower_numpy_binary(ctx, node, np.greater_equal, np.bool_)


@onnx_operators.register
class LessOperator(OnnxOperator):
    def __init__(self):
        super().__init__("Less")

    def lower(self, ctx: "GgmlOnnxExecutionContext", node: NodeProto) -> None:
        self.lower_numpy_binary(ctx, node, np.less, np.bool_)


@onnx_operators.register
class LessOrEqualOperator(OnnxOperator):
    def __init__(self):
        super().__init__("LessOrEqual")

    def lower(self, ctx: "GgmlOnnxExecutionContext", node: NodeProto) -> None:
        self.lower_numpy_binary(ctx, node, np.less_equal, np.bool_)


@onnx_operators.register
class LogOperator(OnnxOperator):
    def __init__(self):
        super().__init__(
            "Log",
            OnnxOperator.EXECUTION_NATIVE_OR_NUMPY_RUNTIME,
            OnnxOperator.CLASS_CONDITIONAL_NATIVE,
        )
        self.has_numpy_evaluator = True

    def strategies(
        self, tensor_types: Dict[str, TensorType], node: "NodeIR"
    ) -> Tuple[Tuple[str, str, str], ...]:
        if (
            len(node.inputs) == 1
            and self.tensor_type(tensor_types, node.inputs[0]).is_float32
        ):
            return self.native_strategy()
        return self.numpy_runtime_strategy()

    def eval_numpy(
        self, node: NodeProto, inputs: Tuple[npt.NDArray[Any], ...]
    ) -> Tuple[npt.NDArray[Any], ...]:
        return self.eval_numpy_unary_operator(node, inputs, np.log)

    def lower(self, ctx: "GgmlOnnxExecutionContext", node: NodeProto) -> None:
        self.lower_native_unary_or_numpy(ctx, node, ggml.ggml_log, np.log)


@onnx_operators.register
class LogSoftmaxOperator(OnnxOperator):
    def __init__(self):
        super().__init__("LogSoftmax")

    def lower(self, ctx: "GgmlOnnxExecutionContext", node: NodeProto) -> None:
        node_inputs = [ctx.ggml_tensors_dict[inp] for inp in node.input]

        if len(node_inputs) != 1:
            raise ValueError(
                f'Error for node "{node.name}": Operation "LogSoftmax" requires exactly one input. Actual number of inputs: {len(node_inputs)}'
            )

        output_name = node.output[0]
        a = node_inputs[0]
        input_shape = ctx.shapes[node.input[0]]
        input_dtype = get_tensor_dtype(a)
        axis = next((attr.i for attr in node.attribute if attr.name == "axis"), -1)
        if axis < 0:
            axis += len(input_shape)

        x_t = ctx.from_numpy(np.empty(input_shape, dtype=input_dtype))

        @ggml.ggml_custom2_op_t
        def custom_log_softmax(
            tensor_out: ggml.ggml_tensor_p,
            tensor_in_1: ggml.ggml_tensor_p,
            tensor_in_2: ggml.ggml_tensor_p,
            ith: int,
            nth: int,
            userdata: Optional[ctypes.c_void_p],
        ):
            x = ctx.to_numpy(tensor_in_2).reshape(input_shape)
            x = x - np.max(x, axis=axis, keepdims=True)
            y = x - np.log(np.sum(np.exp(x), axis=axis, keepdims=True))
            ctx.set_tensor_data(tensor_out, y)

        log_result = ggml.ggml_map_custom2_inplace(
            ctx.ggml_eval_context,
            x_t,
            a,
            custom_log_softmax,
            1,
            None,
        )
        ctx.refs.append(custom_log_softmax)
        ctx.register_numpy_runtime_tensor(
            output_name,
            log_result,
            input_shape,
            input_dtype,
        )


@onnx_operators.register
class MatMulOperator(OnnxOperator):
    def __init__(self):
        super().__init__(
            "MatMul",
            OnnxOperator.EXECUTION_NATIVE_OR_NUMPY_RUNTIME,
            OnnxOperator.CLASS_CONDITIONAL_NATIVE,
        )
        self.has_numpy_evaluator = True

    def strategies(
        self, tensor_types: Dict[str, TensorType], node: "NodeIR"
    ) -> Tuple[Tuple[str, str, str], ...]:
        if len(node.inputs) == 2:
            left = self.tensor_type(tensor_types, node.inputs[0])
            right = self.tensor_type(tensor_types, node.inputs[1])
            if (
                left.is_float32
                and right.is_float32
                and left.shape is not None
                and right.shape is not None
                and len(left.shape) == 2
                and len(right.shape) == 2
            ):
                return self.native_strategy()
        return self.numpy_runtime_strategy(
            "MatMul requires rank 2 float32 inputs to lower to ggml_mul_mat"
        )

    def eval_numpy(
        self, node: NodeProto, inputs: Tuple[npt.NDArray[Any], ...]
    ) -> Tuple[npt.NDArray[Any], ...]:
        if len(inputs) != 2:
            raise ValueError(
                f'Operation "{node.op_type}" requires exactly two inputs. '
                f"Actual number of inputs: {len(inputs)}"
            )
        return (np.matmul(inputs[0], inputs[1]),)

    @staticmethod
    def broadcast_tensor(
        ctx: "GgmlOnnxExecutionContext",
        tensor: ggml.ggml_tensor_p,
        shape: Tuple[int, ...],
    ):
        ggml_type = ggml.utils.GGML_TYPE(tensor.contents.type)

        if ggml_type == ggml.utils.GGML_TYPE.F32:
            new_tensor = ggml.ggml_new_tensor(
                ctx.ggml_eval_context,
                ggml_type.value,
                len(shape),
                (ctypes.c_int64 * len(shape))(*shape),
            )

            new_tensor = ggml.ggml_repeat(
                ctx.ggml_eval_context,
                tensor,
                new_tensor,
            )
        else:

            @ggml.ggml_custom2_op_t
            def custom_broadcast_to(
                tensor_out: ggml.ggml_tensor_p,
                tensor_in_1: ggml.ggml_tensor_p,
                tensor_in_2: ggml.ggml_tensor_p,
                ith: int,
                nth: int,
                userdata: Optional[ctypes.c_void_p],
            ):
                a = ctx.to_numpy(tensor_in_2)

                x = np.broadcast_to(a, shape)
                ctx.set_tensor_data(tensor_out, x)

            x = np.empty(shape, dtype=ctx.get_raw_tensor_dtype(tensor))
            x_t = ctx.from_numpy(x)
            new_tensor = ggml.ggml_map_custom2_inplace(
                ctx.ggml_eval_context,
                x_t,
                tensor,
                custom_broadcast_to,
                1,
                None,
            )
            ctx.refs.append(custom_broadcast_to)
        return new_tensor

    @classmethod
    def broadcast_shapes(
        cls,
        ctx: "GgmlOnnxExecutionContext",
        a: ggml.ggml_tensor_p,
        b: ggml.ggml_tensor_p,
    ):
        a_shape = ctx.get_tensor_shape(a)
        b_shape = ctx.get_tensor_shape(b)

        output_shape = tuple(
            reversed(np.broadcast(np.empty(a_shape), np.empty(b_shape)).shape)
        )  # TODO: Fix this

        a_shaped = a
        b_shaped = b

        if a_shape != output_shape:
            a_shaped = cls.broadcast_tensor(ctx, a, output_shape)
        if b_shape != output_shape:
            b_shaped = cls.broadcast_tensor(ctx, b, output_shape)

        return a_shaped, b_shaped

    def lower(self, ctx: "GgmlOnnxExecutionContext", node: NodeProto) -> None:
        node_inputs = [ctx.ggml_tensors_dict[inp] for inp in node.input]

        if len(node_inputs) != 2:
            raise ValueError(
                f'Error for node "{node.name}": Operation "MatMul" requires exactly two inputs. Actual number of inputs: {len(node_inputs)}'
            )

        a_name, b_name = node.input

        output_name = node.output[0]
        a, b = node_inputs

        a_shape, b_shape = ctx.shapes[a_name], ctx.shapes[b_name]
        a_dtype = np.dtype(ctx.get_tensor_dtype(a_name))
        b_dtype = np.dtype(ctx.get_tensor_dtype(b_name))

        if (
            not ctx.can_emit_native(output_name)
            or not ctx.can_run_native(node)
            or a_dtype != np.dtype(np.float32)
            or b_dtype != np.dtype(np.float32)
            or len(a_shape) != 2
            or len(b_shape) != 2
        ):
            left = ctx.logical_tensor_eval_data(a_name, a, a_shape)
            right = ctx.logical_tensor_eval_data(b_name, b, b_shape)
            output = self.eval_numpy(node, (left, right))[0]
            ctx.set_numpy_runtime_output(output_name, output, output.dtype)
            return

        # TODO: is this check required? broadcast alone wont pass ONNX tests but is broadcasting itself even required or should it fail if a,b are not correct?
        try:
            np.matmul(np.empty(a_shape), np.empty(b_shape))
        except ValueError:
            a, b = self.broadcast_shapes(ctx, a, b)

        if ggml.ggml_is_permuted(a):
            a_dtype = ctx.get_tensor_dtype(a_name)
            a_shape = ggml.utils.get_shape(a)
            a = ggml.ggml_cpy(
                ctx.ggml_eval_context,
                a,
                ggml.ggml_new_tensor(
                    ctx.ggml_eval_context,
                    ctx.map_to_ggml_type(a_dtype).value,
                    len(a_shape),
                    (ctypes.c_int64 * len(a_shape))(*a_shape),
                ),
            )

        b_dtype = ctx.get_tensor_dtype(b_name)

        b_permute = ggml.ggml_transpose(
            ctx.ggml_eval_context,
            b,
        )

        b_shape = ggml.utils.get_shape(b_permute)

        b_transposed = ggml.ggml_cpy(
            ctx.ggml_eval_context,
            b_permute,
            ggml.ggml_new_tensor(
                ctx.ggml_eval_context,
                ctx.map_to_ggml_type(b_dtype).value,
                len(b_shape),
                (ctypes.c_int64 * len(b_shape))(*b_shape),
            ),
        )
        mul_mat_result = ggml.ggml_mul_mat(
            ctx.ggml_eval_context,
            b_transposed,
            a,
        )

        ctx.ggml_tensors_dict[output_name] = mul_mat_result
        ctx.shapes[output_name] = tuple(
            reversed(mul_mat_result.contents.ne[: max(len(a_shape), len(b_shape))])
        )


@onnx_operators.register
class MaxOperator(OnnxOperator):
    def __init__(self):
        super().__init__("Max")
        self.has_numpy_evaluator = True

    def eval_numpy(
        self, node: NodeProto, inputs: Tuple[npt.NDArray[Any], ...]
    ) -> Tuple[npt.NDArray[Any], ...]:
        if not inputs:
            raise ValueError(
                f'Operation "{node.op_type}" requires at least one input. '
                f"Actual number of inputs: {len(inputs)}"
            )
        dtype = np.result_type(*(array.dtype for array in inputs))
        result = np.asarray(inputs[0], dtype=dtype)
        for array in inputs[1:]:
            result = np.asarray(np.maximum(result, array), dtype=dtype)
        return (np.asarray(result, dtype=dtype),)

    def lower(self, ctx: "GgmlOnnxExecutionContext", node: NodeProto) -> None:
        self.lower_numpy_variadic(ctx, node, np.maximum)


@onnx_operators.register
class MaxPoolOperator(OnnxOperator):
    def __init__(self):
        super().__init__(
            "MaxPool",
            OnnxOperator.EXECUTION_NATIVE_OR_NUMPY_RUNTIME,
            OnnxOperator.CLASS_CONDITIONAL_NATIVE,
        )
        self.has_numpy_evaluator = True

    def strategies(
        self, tensor_types: Dict[str, TensorType], node: "NodeIR"
    ) -> Tuple[Tuple[str, str, str], ...]:
        return self.pool_strategies(tensor_types, node, "Max")

    def eval_numpy(
        self, node: NodeProto, inputs: Tuple[npt.NDArray[Any], ...]
    ) -> Tuple[npt.NDArray[Any], ...]:
        return self.eval_numpy_pool(node, inputs, "Max")

    def lower(self, ctx: "GgmlOnnxExecutionContext", node: NodeProto) -> None:
        self.lower_pool(ctx, node, "Max")


@onnx_operators.register
class MaxUnpoolOperator(OnnxOperator):
    def __init__(self):
        super().__init__("MaxUnpool")

    def lower(self, ctx: "GgmlOnnxExecutionContext", node: NodeProto) -> None:
        node_inputs = [ctx.ggml_tensors_dict[inp] for inp in node.input]

        if len(node_inputs) not in {2, 3}:
            raise ValueError(
                f'Error for node "{node.name}": Operation "MaxUnpool" requires 2 - 3 inputs. Actual number of inputs: {len(node_inputs)}'
            )

        x_shape = ctx.shapes[node.input[0]]
        indices_shape = ctx.shapes[node.input[1]]
        if x_shape != indices_shape:
            raise ValueError(
                f"MaxUnpool input shape {x_shape} must match indices shape {indices_shape}"
            )
        if len(x_shape) > 4:
            raise NotImplementedError("MaxUnpool with rank > 4 is not supported")

        kernel_shape = next(
            (attr.ints for attr in node.attribute if attr.name == "kernel_shape"),
            None,
        )
        if kernel_shape is None:
            raise ValueError(f'Error for node "{node.name}": kernel_shape is required')
        kernel_shape = [int(dim) for dim in kernel_shape]
        spatial_rank = len(kernel_shape)
        if len(x_shape) != spatial_rank + 2:
            raise ValueError(
                f"MaxUnpool input rank {len(x_shape)} does not match kernel rank {spatial_rank}"
            )

        pads = next(
            (attr.ints for attr in node.attribute if attr.name == "pads"),
            [0 for _ in range(spatial_rank * 2)],
        )
        strides = next(
            (attr.ints for attr in node.attribute if attr.name == "strides"),
            [1 for _ in range(spatial_rank)],
        )
        pads = [int(pad) for pad in pads]
        strides = [int(stride) for stride in strides]
        if len(pads) != spatial_rank * 2:
            raise ValueError(
                f"MaxUnpool requires {spatial_rank * 2} pad values, got {len(pads)}"
            )
        if len(strides) != spatial_rank:
            raise ValueError(
                f"MaxUnpool requires {spatial_rank} strides, got {len(strides)}"
            )

        inferred_shape = x_shape[:2] + tuple(
            (x_shape[axis + 2] - 1) * strides[axis]
            - (pads[axis] + pads[spatial_rank + axis])
            + kernel_shape[axis]
            for axis in range(spatial_rank)
        )
        if len(node_inputs) == 3 and node.input[2]:
            output_shape_tensor = ctx.eval_tensor(node_inputs[2])
            output_shape = tuple(
                int(dim)
                for dim in ctx.to_numpy(output_shape_tensor)
                .reshape(ctx.shapes[node.input[2]])
                .ravel()
            )
        else:
            output_shape = inferred_shape
        if len(output_shape) > 4:
            raise NotImplementedError("MaxUnpool with rank > 4 is not supported")

        x = ctx.logical_tensor_eval_data(node.input[0], node_inputs[0], x_shape)
        indices = ctx.logical_tensor_eval_data(
            node.input[1], node_inputs[1], indices_shape
        )
        unpooled = np.zeros(int(np.prod(inferred_shape)), dtype=x.dtype)
        unpooled[indices.astype(np.int64).ravel()] = x.ravel()
        unpooled = unpooled.reshape(inferred_shape)

        y = np.zeros(output_shape, dtype=x.dtype)
        slices = tuple(slice(0, dim) for dim in inferred_shape)
        y[slices] = unpooled
        ctx.set_logical_output(node.output[0], y, y.dtype)


@onnx_operators.register
class MeanOperator(OnnxOperator):
    def __init__(self):
        super().__init__(
            "Mean",
            OnnxOperator.EXECUTION_NATIVE_OR_NUMPY_RUNTIME,
            OnnxOperator.CLASS_CONDITIONAL_NATIVE,
        )
        self.has_numpy_evaluator = True

    def strategies(
        self, tensor_types: Dict[str, TensorType], node: "NodeIR"
    ) -> Tuple[Tuple[str, str, str], ...]:
        if self.is_same_shape_float32(
            self.variadic_elementwise_types(tensor_types, node)
        ):
            return self.decomposed_strategy()
        return self.numpy_runtime_strategy()

    def eval_numpy(
        self, node: NodeProto, inputs: Tuple[npt.NDArray[Any], ...]
    ) -> Tuple[npt.NDArray[Any], ...]:
        return self.eval_numpy_variadic_operator(
            node,
            inputs,
            lambda left, right: left + right,
            lambda result, inputs: result / len(inputs),
        )

    def lower(self, ctx: "GgmlOnnxExecutionContext", node: NodeProto) -> None:
        node_inputs = [ctx.ggml_tensors_dict[inp] for inp in node.input]

        if len(node_inputs) < 1:
            raise ValueError(
                f'Error for node "{node.name}": Operation "Mean" requires at least one input. Actual number of inputs: {len(node_inputs)}'
            )

        output_name = node.output[0]
        input_types = tuple(
            self.runtime_tensor_type(ctx, input_name) for input_name in node.input
        )
        first_shape = input_types[0].shape

        if (
            ctx.can_emit_native(output_name)
            and ctx.can_run_native(node)
            and self.is_same_shape_float32(input_types)
            and first_shape is not None
        ):
            sums = node_inputs[0]
            for tensor in node_inputs[1:]:
                sums = ggml.ggml_add(ctx.ggml_eval_context, sums, tensor)

            coef_np = np.full(first_shape, len(node_inputs), dtype=np.float32)
            coef_t = ctx.from_numpy(coef_np)
            mean = ggml.ggml_div(ctx.ggml_eval_context, sums, coef_t)
            ctx.register_decomposed_tensor(
                output_name,
                mean,
                first_shape,
                np.dtype(np.float32),
            )
            return

        def mean_func(
            left: npt.NDArray[Any], right: npt.NDArray[Any]
        ) -> npt.NDArray[Any]:
            return left + right

        self.lower_numpy_variadic(ctx, node, mean_func)
        output_state = ctx.tensor_state(output_name)
        output = ctx.logical_tensor_eval_data(
            output_name, output_state.tensor, output_state.shape
        )
        ctx.set_logical_output(
            output_name,
            output / len(node_inputs),
            output_state.dtype,
        )


@onnx_operators.register
class MeanVarianceNormalizationOperator(OnnxOperator):
    def __init__(self):
        super().__init__("MeanVarianceNormalization")

    def lower(self, ctx: "GgmlOnnxExecutionContext", node: NodeProto) -> None:
        node_inputs = [ctx.ggml_tensors_dict[inp] for inp in node.input]

        if len(node_inputs) != 1:
            raise ValueError(
                f'Error for node "{node.name}": Operation "MeanVarianceNormalization" requires exactly one input. Actual number of inputs: {len(node_inputs)}'
            )

        x_shape = ctx.shapes[node.input[0]]
        x_dtype = get_tensor_dtype(node_inputs[0])
        axes = tuple(
            int(axis)
            for attr in node.attribute
            if attr.name == "axes"
            for axis in attr.ints
        ) or (0, 2, 3)
        axes = tuple(axis + len(x_shape) if axis < 0 else axis for axis in axes)
        epsilon = next(
            (attr.f for attr in node.attribute if attr.name == "epsilon"), 1e-9
        )

        x = ctx.to_numpy(ctx.eval_tensor(node_inputs[0])).reshape(x_shape)
        mean = np.mean(x, axis=axes, keepdims=True)
        variance = np.mean(np.square(x - mean), axis=axes, keepdims=True)
        output = ((x - mean) / np.sqrt(variance + epsilon)).astype(x_dtype)

        new_tensor = ctx.ggml_tensors_dict[node.output[0]] = ctx.from_numpy(output)
        ctx.set_tensor_shape(new_tensor, output.shape)
        ctx.shapes[node.output[0]] = output.shape
        ctx.set_tensor_dtype(node.output[0], output.dtype)


@onnx_operators.register
class MelWeightMatrixOperator(OnnxOperator):
    def __init__(self):
        super().__init__("MelWeightMatrix")

    def lower(self, ctx: "GgmlOnnxExecutionContext", node: NodeProto) -> None:
        node_inputs = [ctx.ggml_tensors_dict[inp] for inp in node.input]

        if len(node_inputs) != 5:
            raise ValueError(
                f'Error for node "{node.name}": Operation "MelWeightMatrix" requires exactly five inputs. Actual number of inputs: {len(node_inputs)}'
            )

        def scalar_input(index: int) -> Any:
            return (
                ctx.to_numpy(ctx.eval_tensor(node_inputs[index]))
                .reshape(ctx.shapes[node.input[index]])
                .item()
            )

        num_mel_bins = int(scalar_input(0))
        dft_length = int(scalar_input(1))
        sample_rate = int(scalar_input(2))
        lower_edge_hertz = float(scalar_input(3))
        upper_edge_hertz = float(scalar_input(4))

        num_spectrogram_bins = dft_length // 2 + 1
        frequency_bins = np.arange(0, num_mel_bins + 2, dtype=np.float32)
        low_frequency_mel = 2595 * np.log10(1 + lower_edge_hertz / 700)
        high_frequency_mel = 2595 * np.log10(1 + upper_edge_hertz / 700)
        mel_step = (high_frequency_mel - low_frequency_mel) / frequency_bins.shape[0]

        frequency_bins = frequency_bins * mel_step + low_frequency_mel
        frequency_bins = 700 * (np.power(10, frequency_bins / 2595) - 1)
        frequency_bins = ((dft_length + 1) * frequency_bins) // sample_rate
        frequency_bins = frequency_bins.astype(int)

        output = np.zeros((num_spectrogram_bins, num_mel_bins), dtype=np.float32)
        for i in range(num_mel_bins):
            lower_frequency_value = frequency_bins[i]
            center_frequency_point = frequency_bins[i + 1]
            higher_frequency_point = frequency_bins[i + 2]
            low_to_center = center_frequency_point - lower_frequency_value
            if low_to_center == 0:
                output[center_frequency_point, i] = 1
            else:
                for j in range(lower_frequency_value, center_frequency_point + 1):
                    output[j, i] = float(j - lower_frequency_value) / float(
                        low_to_center
                    )
            center_to_high = higher_frequency_point - center_frequency_point
            if center_to_high > 0:
                for j in range(center_frequency_point, higher_frequency_point):
                    output[j, i] = float(higher_frequency_point - j) / float(
                        center_to_high
                    )

        new_tensor = ctx.ggml_tensors_dict[node.output[0]] = ctx.from_numpy(output)
        ctx.set_tensor_shape(new_tensor, output.shape)
        ctx.shapes[node.output[0]] = output.shape
        ctx.set_tensor_dtype(node.output[0], output.dtype)


@onnx_operators.register
class MishOperator(OnnxOperator):
    def __init__(self):
        super().__init__("Mish")

    def lower(self, ctx: "GgmlOnnxExecutionContext", node: NodeProto) -> None:
        def mish(x: npt.NDArray[Any]) -> npt.NDArray[Any]:
            return x * np.tanh(np.log1p(np.exp(x)))

        self.lower_numpy_unary(ctx, node, mish)


@onnx_operators.register
class ModOperator(OnnxOperator):
    def __init__(self):
        super().__init__("Mod")

    def lower(self, ctx: "GgmlOnnxExecutionContext", node: NodeProto) -> None:
        fmod = next((attr.i for attr in node.attribute if attr.name == "fmod"), 0)
        mod_func = np.fmod if fmod else np.mod
        input_dtypes = [np.dtype(ctx.get_tensor_dtype(name)) for name in node.input]
        if all(np.issubdtype(dtype, np.integer) for dtype in input_dtypes):
            self.lower_numpy_integer_binary(ctx, node, mod_func)
        else:
            self.lower_numpy_binary(ctx, node, mod_func)


@onnx_operators.register
class MinOperator(OnnxOperator):
    def __init__(self):
        super().__init__("Min")
        self.has_numpy_evaluator = True

    def eval_numpy(
        self, node: NodeProto, inputs: Tuple[npt.NDArray[Any], ...]
    ) -> Tuple[npt.NDArray[Any], ...]:
        if not inputs:
            raise ValueError(
                f'Operation "{node.op_type}" requires at least one input. '
                f"Actual number of inputs: {len(inputs)}"
            )
        dtype = np.result_type(*(array.dtype for array in inputs))
        result = np.asarray(inputs[0], dtype=dtype)
        for array in inputs[1:]:
            result = np.asarray(np.minimum(result, array), dtype=dtype)
        return (np.asarray(result, dtype=dtype),)

    def lower(self, ctx: "GgmlOnnxExecutionContext", node: NodeProto) -> None:
        self.lower_numpy_variadic(ctx, node, np.minimum)


@onnx_operators.register
class MulOperator(OnnxOperator):
    def __init__(self):
        super().__init__(
            "Mul",
            OnnxOperator.EXECUTION_NATIVE_OR_NUMPY_RUNTIME,
            OnnxOperator.CLASS_CONDITIONAL_NATIVE,
        )
        self.has_numpy_evaluator = True

    @staticmethod
    def ggml_mul_scalar(
        ggml_context: ggml.ggml_context_p,
        tensor: ggml.ggml_tensor_p,
        scalar_tensor: ggml.ggml_tensor_p,
        scalar_value: float,
    ) -> ggml.ggml_tensor_p:
        del scalar_tensor
        return ggml.ggml_scale(ggml_context, tensor, scalar_value)

    def strategies(
        self, tensor_types: Dict[str, TensorType], node: "NodeIR"
    ) -> Tuple[Tuple[str, str, str], ...]:
        if len(node.inputs) != 2:
            return self.numpy_runtime_strategy()
        input_types = self.binary_elementwise_types(tensor_types, node)
        output_shape = self.infer_elementwise_output_shape(input_types)
        if self.is_same_shape_float32(input_types):
            return self.native_strategy()
        if self.has_float32_scalar_broadcast(input_types, output_shape):
            return self.native_strategy()
        return self.numpy_runtime_strategy()

    def eval_numpy(
        self, node: NodeProto, inputs: Tuple[npt.NDArray[Any], ...]
    ) -> Tuple[npt.NDArray[Any], ...]:
        return self.eval_numpy_binary_operator(node, inputs, np.multiply)

    def lower(self, ctx: "GgmlOnnxExecutionContext", node: NodeProto) -> None:
        self.lower_native_binary_or_numpy(
            ctx, node, ggml.ggml_mul, np.multiply, scalar_func=self.ggml_mul_scalar
        )


@onnx_operators.register
class NegOperator(OnnxOperator):
    def __init__(self):
        super().__init__(
            "Neg",
            OnnxOperator.EXECUTION_NATIVE_OR_NUMPY_RUNTIME,
            OnnxOperator.CLASS_CONDITIONAL_NATIVE,
        )
        self.has_numpy_evaluator = True

    def strategies(
        self, tensor_types: Dict[str, TensorType], node: "NodeIR"
    ) -> Tuple[Tuple[str, str, str], ...]:
        if (
            len(node.inputs) == 1
            and self.tensor_type(tensor_types, node.inputs[0]).is_float32
        ):
            return self.native_strategy()
        return self.numpy_runtime_strategy()

    def eval_numpy(
        self, node: NodeProto, inputs: Tuple[npt.NDArray[Any], ...]
    ) -> Tuple[npt.NDArray[Any], ...]:
        return self.eval_numpy_unary_operator(node, inputs, np.negative)

    def lower(self, ctx: "GgmlOnnxExecutionContext", node: NodeProto) -> None:
        self.lower_native_unary_or_numpy(ctx, node, ggml.ggml_neg, np.negative)


@onnx_operators.register
class NegativeLogLikelihoodLossOperator(OnnxOperator):
    def __init__(self):
        super().__init__("NegativeLogLikelihoodLoss")

    @staticmethod
    def negative_log_likelihood_loss(
        input_data: npt.NDArray[Any],
        target: npt.NDArray[Any],
        weight: Optional[npt.NDArray[Any]],
        reduction: str,
        ignore_index: Optional[int],
    ) -> npt.NDArray[Any]:
        input_shape = input_data.shape
        target_shape = target.shape
        batch_size = input_shape[0]
        channel_count = input_shape[1]

        gather_weight = None
        if weight is not None:
            gather_weight = np.take(
                weight, np.asarray(target, dtype=np.int32), mode="clip"
            )
            if ignore_index is not None:
                gather_weight = np.where(
                    target == ignore_index, 0, gather_weight
                ).astype(np.float32)
        elif ignore_index is not None:
            gather_weight = np.where(target == ignore_index, 0, 1).astype(np.float32)

        if len(input_shape) != 3:
            input_data = input_data.reshape((batch_size, channel_count, -1))
            target = target.reshape((batch_size, -1))

        sample_count = input_data.shape[2]
        loss = np.zeros((batch_size, sample_count), dtype=np.float32)
        for batch_index in range(batch_size):
            for sample_index in range(sample_count):
                target_index = target[batch_index][sample_index]
                if target_index != ignore_index:
                    loss[batch_index][sample_index] = -input_data[batch_index][
                        target_index
                    ][sample_index]

        if len(input_shape) != 3:
            loss = loss.reshape(target_shape)

        if gather_weight is not None:
            loss = gather_weight * loss
            if reduction == "mean":
                return np.asarray(loss.sum() / gather_weight.sum(), dtype=np.float32)

        if reduction == "mean":
            return np.asarray(np.mean(loss), dtype=np.float32)
        if reduction == "sum":
            return np.asarray(np.sum(loss), dtype=np.float32)
        return loss.astype(np.float32)

    def lower(self, ctx: "GgmlOnnxExecutionContext", node: NodeProto) -> None:
        node_inputs = [ctx.ggml_tensors_dict[inp] for inp in node.input]

        if len(node_inputs) not in {2, 3}:
            raise ValueError(
                f'Error for node "{node.name}": Operation "NegativeLogLikelihoodLoss" requires two or three inputs. Actual number of inputs: {len(node_inputs)}'
            )

        input_shape = ctx.shapes[node.input[0]]
        target_shape = ctx.shapes[node.input[1]]
        input_data = ctx.to_numpy(ctx.eval_tensor(node_inputs[0])).reshape(input_shape)
        target = ctx.to_numpy(ctx.eval_tensor(node_inputs[1])).reshape(target_shape)
        weight = None
        if len(node_inputs) == 3:
            weight = ctx.to_numpy(ctx.eval_tensor(node_inputs[2])).reshape(
                ctx.shapes[node.input[2]]
            )
        reduction = next(
            (
                attr.s.decode("utf-8")
                for attr in node.attribute
                if attr.name == "reduction"
            ),
            "mean",
        )
        ignore_index = next(
            (attr.i for attr in node.attribute if attr.name == "ignore_index"), None
        )

        output = self.negative_log_likelihood_loss(
            input_data, target, weight, reduction, ignore_index
        )
        new_tensor = ctx.ggml_tensors_dict[node.output[0]] = ctx.from_numpy(output)
        ctx.set_tensor_shape(new_tensor, output.shape)
        ctx.shapes[node.output[0]] = output.shape
        ctx.set_tensor_dtype(node.output[0], output.dtype)


@onnx_operators.register
class NotOperator(OnnxOperator):
    def __init__(self):
        super().__init__("Not")

    def lower(self, ctx: "GgmlOnnxExecutionContext", node: NodeProto) -> None:
        node_inputs = [ctx.ggml_tensors_dict[inp] for inp in node.input]

        if len(node_inputs) != 1:
            raise ValueError(
                f'Error for node "{node.name}": Operation "Not" requires exactly one input. Actual number of inputs: {len(node_inputs)}'
            )
        name = node.output[0]
        output_shape = ctx.shapes[node.input[0]]
        x_t = ctx.from_numpy(np.empty(output_shape, dtype=np.bool_))

        @ggml.ggml_custom2_op_t
        def custom_not(
            tensor_out: ggml.ggml_tensor_p,
            tensor_in_1: ggml.ggml_tensor_p,
            tensor_in_2: ggml.ggml_tensor_p,
            ith: int,
            nth: int,
            userdata: Optional[ctypes.c_void_p],
        ):
            a = ctx.to_numpy(tensor_in_2)
            x = np.logical_not(a)

            ctx.set_tensor_data(tensor_out, x)

        new_tensor = ctx.ggml_tensors_dict[name] = ggml.ggml_map_custom2_inplace(
            ctx.ggml_eval_context,
            x_t,
            node_inputs[0],
            custom_not,
            1,
            None,
        )

        ctx.refs.append(custom_not)

        ctx.set_tensor_dtype(name, np.dtype(np.bool_))
        ctx.set_tensor_shape(new_tensor, output_shape)
        ctx.shapes[name] = output_shape


@onnx_operators.register
class NonZeroOperator(OnnxOperator):
    def __init__(self):
        super().__init__("NonZero")

    def lower(self, ctx: "GgmlOnnxExecutionContext", node: NodeProto) -> None:
        node_inputs = [ctx.ggml_tensors_dict[inp] for inp in node.input]

        if len(node_inputs) != 1:
            raise ValueError(
                f'Error for node "{node.name}": Operation "NonZero" requires exactly one input. Actual number of inputs: {len(node_inputs)}'
            )

        input_shape = ctx.shapes[node.input[0]]
        x = ctx.logical_tensor_eval_data(node.input[0], node_inputs[0], input_shape)
        output = np.asarray(np.nonzero(x), dtype=np.int64)
        ctx.set_logical_output(node.output[0], output, output.dtype)


@onnx_operators.register
class NonMaxSuppressionOperator(OnnxOperator):
    def __init__(self):
        super().__init__("NonMaxSuppression")

    def lower(self, ctx: "GgmlOnnxExecutionContext", node: NodeProto) -> None:
        node_inputs = [
            ctx.ggml_tensors_dict[inp] if inp != "" else None for inp in node.input
        ]

        if len(node_inputs) not in {2, 3, 4, 5}:
            raise ValueError(
                f'Error for node "{node.name}": Operation "NonMaxSuppression" requires two to five inputs. Actual number of inputs: {len(node_inputs)}'
            )

        boxes_tensor = node_inputs[0]
        scores_tensor = node_inputs[1]
        if boxes_tensor is None or scores_tensor is None:
            raise ValueError(
                f'Error for node "{node.name}": NonMaxSuppression boxes and scores inputs are required.'
            )

        def optional_scalar(index: int, default: float) -> float:
            if index >= len(node_inputs) or node_inputs[index] is None:
                return default
            tensor = node_inputs[index]
            return float(
                ctx.to_numpy(ctx.eval_tensor(tensor))
                .reshape(ctx.shapes[node.input[index]])
                .item()
            )

        boxes_shape = ctx.shapes[node.input[0]]
        scores_shape = ctx.shapes[node.input[1]]
        boxes = ctx.to_numpy(ctx.eval_tensor(boxes_tensor)).reshape(boxes_shape)
        scores = ctx.to_numpy(ctx.eval_tensor(scores_tensor)).reshape(scores_shape)
        max_output_boxes_per_class = int(optional_scalar(2, 0))
        iou_threshold = optional_scalar(3, 0.0)
        score_threshold = optional_scalar(4, -np.inf)
        center_point_box = next(
            (attr.i for attr in node.attribute if attr.name == "center_point_box"), 0
        )

        def box_corners(box: npt.NDArray[Any]) -> Tuple[float, float, float, float]:
            if center_point_box:
                x_center, y_center, width, height = box
                y1 = y_center - height / 2
                x1 = x_center - width / 2
                y2 = y_center + height / 2
                x2 = x_center + width / 2
            else:
                y1, x1, y2, x2 = box
            return min(y1, y2), min(x1, x2), max(y1, y2), max(x1, x2)

        def intersection_over_union(
            box_a: npt.NDArray[Any], box_b: npt.NDArray[Any]
        ) -> float:
            a_y1, a_x1, a_y2, a_x2 = box_corners(box_a)
            b_y1, b_x1, b_y2, b_x2 = box_corners(box_b)
            intersection_height = max(0.0, min(a_y2, b_y2) - max(a_y1, b_y1))
            intersection_width = max(0.0, min(a_x2, b_x2) - max(a_x1, b_x1))
            intersection = intersection_height * intersection_width
            area_a = max(0.0, a_y2 - a_y1) * max(0.0, a_x2 - a_x1)
            area_b = max(0.0, b_y2 - b_y1) * max(0.0, b_x2 - b_x1)
            union = area_a + area_b - intersection
            if union <= 0:
                return 0.0
            return intersection / union

        selected_indices = []
        for batch_index in range(scores_shape[0]):
            for class_index in range(scores_shape[1]):
                class_scores = scores[batch_index, class_index]
                candidate_indices = np.argsort(-class_scores, kind="stable")
                selected_box_indices: List[int] = []

                for box_index in candidate_indices:
                    if len(selected_box_indices) >= max_output_boxes_per_class:
                        break
                    if class_scores[box_index] <= score_threshold:
                        continue

                    should_select = True
                    for selected_box_index in selected_box_indices:
                        iou = intersection_over_union(
                            boxes[batch_index, box_index],
                            boxes[batch_index, selected_box_index],
                        )
                        if iou > iou_threshold:
                            should_select = False
                            break

                    if should_select:
                        selected_box_indices.append(int(box_index))
                        selected_indices.append(
                            [batch_index, class_index, int(box_index)]
                        )

        output = np.asarray(selected_indices, dtype=np.int64).reshape(-1, 3)
        new_tensor = ctx.ggml_tensors_dict[node.output[0]] = ctx.from_numpy(output)
        ctx.set_tensor_shape(new_tensor, output.shape)
        ctx.shapes[node.output[0]] = output.shape
        ctx.set_tensor_dtype(node.output[0], output.dtype)


@onnx_operators.register
class OrOperator(OnnxOperator):
    def __init__(self):
        super().__init__("Or")

    def lower(self, ctx: "GgmlOnnxExecutionContext", node: NodeProto) -> None:
        self.lower_numpy_binary(ctx, node, np.logical_or, np.bool_)


@onnx_operators.register
class OneHotOperator(OnnxOperator):
    def __init__(self):
        super().__init__("OneHot")

    def lower(self, ctx: "GgmlOnnxExecutionContext", node: NodeProto) -> None:
        node_inputs = [ctx.ggml_tensors_dict[inp] for inp in node.input]

        if len(node_inputs) != 3:
            raise ValueError(
                f'Error for node "{node.name}": Operation "OneHot" requires exactly three inputs. Actual number of inputs: {len(node_inputs)}'
            )

        indices_shape = ctx.shapes[node.input[0]]
        depth_shape = ctx.shapes[node.input[1]]
        values_shape = ctx.shapes[node.input[2]]
        axis = next((attr.i for attr in node.attribute if attr.name == "axis"), -1)

        indices = ctx.to_numpy(ctx.eval_tensor(node_inputs[0])).reshape(indices_shape)
        depth = int(
            ctx.to_numpy(ctx.eval_tensor(node_inputs[1])).reshape(depth_shape).item()
        )
        values = ctx.to_numpy(ctx.eval_tensor(node_inputs[2])).reshape(values_shape)

        rank = len(indices_shape)
        if axis < 0:
            axis += rank + 1

        left_shape = indices_shape[:axis]
        right_shape = indices_shape[axis:]
        depth_range = np.arange(depth)
        targets = depth_range.reshape(
            (1,) * len(left_shape) + (depth,) + (1,) * len(right_shape)
        )
        normalized_indices = np.mod(indices.astype(np.int64), depth).reshape(
            (*left_shape, 1, *right_shape)
        )
        output = np.where(targets == normalized_indices, values[1], values[0]).astype(
            values.dtype
        )

        new_tensor = ctx.ggml_tensors_dict[node.output[0]] = ctx.from_numpy(output)
        ctx.set_tensor_shape(new_tensor, output.shape)
        ctx.shapes[node.output[0]] = output.shape
        ctx.set_tensor_dtype(node.output[0], output.dtype)


@onnx_operators.register
class PadOperator(OnnxOperator):
    def __init__(self):
        super().__init__(
            "Pad",
            OnnxOperator.EXECUTION_NATIVE_OR_NUMPY_RUNTIME,
            OnnxOperator.CLASS_CONDITIONAL_NATIVE,
        )
        self.has_numpy_evaluator = True

    @staticmethod
    def pad_mode(node: NodeProto) -> str:
        mode = next(
            (attr.s for attr in node.attribute if attr.name == "mode"), b"constant"
        )
        return mode.decode("utf-8") if isinstance(mode, bytes) else str(mode)

    @staticmethod
    def pad_attr_values(node: NodeProto) -> Optional[Tuple[int, ...]]:
        values = OnnxOperator.ints_attribute(node, "pads")
        return None if values is None else tuple(int(value) for value in values)

    def pad_axes_from_values(
        self, input_rank: int, axes_values: Optional[Sequence[int]]
    ) -> Optional[Tuple[int, ...]]:
        if axes_values is None:
            return tuple(range(input_rank))
        axes = []
        for axis in axes_values:
            axis = int(axis)
            axis = axis + input_rank if axis < 0 else axis
            if axis < 0 or axis >= input_rank:
                return None
            axes.append(axis)
        return tuple(axes)

    def pad_width_from_values(
        self,
        input_rank: int,
        pads_values: Sequence[int],
        axes_values: Optional[Sequence[int]],
    ) -> Optional[Tuple[Tuple[int, int], ...]]:
        axes = self.pad_axes_from_values(input_rank, axes_values)
        if axes is None:
            return None
        if len(pads_values) != len(axes) * 2:
            return None
        pad_width = [(0, 0) for _ in range(input_rank)]
        for index, axis in enumerate(axes):
            pad_width[axis] = (
                int(pads_values[index]),
                int(pads_values[index + len(axes)]),
            )
        return tuple(pad_width)

    def static_pad_width(
        self, tensor_types: Dict[str, TensorType], node: "NodeIR"
    ) -> Optional[Tuple[Tuple[int, int], ...]]:
        if not node.inputs:
            return None
        input_shape = self.tensor_type(tensor_types, node.inputs[0]).shape
        if input_shape is None:
            return None
        pads_values = self.pad_attr_values_from_ir(tensor_types, node)
        if pads_values is None:
            return None
        axes_values = None
        if len(node.inputs) >= 4 and node.inputs[3]:
            axes_values = self.constant_int_values(tensor_types, node.inputs[3])
            if axes_values is None:
                return None
        return self.pad_width_from_values(len(input_shape), pads_values, axes_values)

    def pad_attr_values_from_ir(
        self, tensor_types: Dict[str, TensorType], node: "NodeIR"
    ) -> Optional[Tuple[int, ...]]:
        pads_attr = node.attribute("pads")
        if pads_attr is not None:
            return tuple(int(value) for value in pads_attr)
        if len(node.inputs) >= 2 and node.inputs[1]:
            return self.constant_int_values(tensor_types, node.inputs[1])
        return None

    def static_pad_value(
        self, tensor_types: Dict[str, TensorType], node: "NodeIR"
    ) -> Optional[Any]:
        value_attr = node.attribute("value")
        if value_attr is not None:
            return value_attr
        if len(node.inputs) >= 3 and node.inputs[2]:
            return self.constant_scalar_value(tensor_types, node.inputs[2])
        return 0

    @staticmethod
    def can_native_pad_width(pad_width: Sequence[Tuple[int, int]]) -> bool:
        return all(before == 0 and after >= 0 for before, after in pad_width)

    def strategies(
        self, tensor_types: Dict[str, TensorType], node: "NodeIR"
    ) -> Tuple[Tuple[str, str, str], ...]:
        if len(node.inputs) >= 1:
            input_type = self.tensor_type(tensor_types, node.inputs[0])
            pad_width = self.static_pad_width(tensor_types, node)
            pad_value = self.static_pad_value(tensor_types, node)
            mode = node.attribute("mode", b"constant")
            mode = mode.decode("utf-8") if isinstance(mode, bytes) else str(mode)
            if (
                input_type.is_float32
                and input_type.shape is not None
                and len(input_type.shape) <= ViewTransformSemantics.GGML_MAX_DIMS
                and mode == "constant"
                and pad_width is not None
                and self.can_native_pad_width(pad_width)
                and pad_value is not None
                and float(pad_value) == 0.0
            ):
                return self.native_strategy()
        return self.numpy_runtime_strategy(
            "Pad requires static float32 constant-zero right-padding with rank <= 4 "
            "to lower to ggml_pad"
        )

    def runtime_pad_width_and_value(
        self,
        ctx: "GgmlOnnxExecutionContext",
        node: NodeProto,
        node_inputs: Sequence[Optional[ggml.ggml_tensor_p]],
    ) -> Tuple[Tuple[Tuple[int, int], ...], npt.NDArray[Any]]:
        input_shape = ctx.shapes[node.input[0]]
        input_rank = len(input_shape)
        pads_tensor = node_inputs[1] if len(node_inputs) > 1 else None
        if pads_tensor is None:
            pads_values = self.pad_attr_values(node)
            if pads_values is None:
                raise ValueError(
                    f'Error for node "{node.name}": Operation "Pad" requires pads'
                )
        else:
            pads_values = tuple(
                int(value)
                for value in ctx.logical_tensor_eval_data(
                    node.input[1], pads_tensor, ctx.shapes[node.input[1]]
                ).flatten()
            )
        axes_values = None
        if len(node_inputs) > 3 and node_inputs[3] is not None:
            axes_values = tuple(
                int(value)
                for value in ctx.logical_tensor_eval_data(
                    node.input[3], node_inputs[3], ctx.shapes[node.input[3]]
                ).flatten()
            )
        pad_width = self.pad_width_from_values(input_rank, pads_values, axes_values)
        if pad_width is None:
            raise ValueError(f'Error for node "{node.name}": Invalid Pad axes or pads')

        value_tensor = node_inputs[2] if len(node_inputs) > 2 else None
        dtype = np.dtype(ctx.get_tensor_dtype(node.input[0]))
        if value_tensor is None:
            value_attr = next(
                (attr.f for attr in node.attribute if attr.name == "value"), None
            )
            constant_values = np.asarray(
                0 if value_attr is None else value_attr, dtype=dtype
            )
        else:
            constant_values = ctx.logical_tensor_eval_data(
                node.input[2], value_tensor, ctx.shapes[node.input[2]]
            )
        return pad_width, constant_values

    def eval_numpy(
        self, node: NodeProto, inputs: Tuple[npt.NDArray[Any], ...]
    ) -> Tuple[npt.NDArray[Any], ...]:
        input_iter = iter(inputs)
        positional_inputs: List[Optional[npt.NDArray[Any]]] = []
        for input_name in node.input:
            positional_inputs.append(next(input_iter) if input_name else None)
        data = positional_inputs[0]
        if data is None:
            raise ValueError(f'Operation "{node.op_type}" requires data input')
        pads = positional_inputs[1] if len(positional_inputs) > 1 else None
        if pads is None:
            pads_values = self.pad_attr_values(node)
            if pads_values is None:
                raise ValueError(f'Operation "{node.op_type}" requires pads')
        else:
            pads_values = tuple(int(value) for value in np.asarray(pads).flatten())
        axes = None
        if len(positional_inputs) > 3 and positional_inputs[3] is not None:
            axes = tuple(
                int(value) for value in np.asarray(positional_inputs[3]).flatten()
            )
        pad_width = self.pad_width_from_values(data.ndim, pads_values, axes)
        if pad_width is None:
            raise ValueError(f'Operation "{node.op_type}" has invalid pads or axes')
        constant_values = np.asarray(0, dtype=data.dtype)
        if len(positional_inputs) > 2 and positional_inputs[2] is not None:
            constant_values = positional_inputs[2]
        else:
            value_attr = next(
                (attr.f for attr in node.attribute if attr.name == "value"), None
            )
            if value_attr is not None:
                constant_values = np.asarray(value_attr, dtype=data.dtype)
        mode = self.pad_mode(node)
        if mode == "constant":
            return (
                np.asarray(
                    np.pad(
                        data,
                        pad_width=pad_width,
                        mode=mode,
                        constant_values=constant_values,
                    ),
                    dtype=data.dtype,
                ),
            )
        return (
            np.asarray(np.pad(data, pad_width=pad_width, mode=mode), dtype=data.dtype),
        )

    def lower(self, ctx: "GgmlOnnxExecutionContext", node: NodeProto) -> None:
        node_inputs = [
            ctx.ggml_tensors_dict[inp] if inp else None for inp in node.input
        ]

        node_inputs += [None] * (4 - len(node_inputs))
        data = node_inputs[0]
        output_name = node.output[0]

        input_shape = ctx.shapes[node.input[0]]
        mode = self.pad_mode(node)
        pad_width, constant_values = self.runtime_pad_width_and_value(
            ctx, node, node_inputs
        )

        expand_by = [sum(pad) for pad in pad_width]
        prev_shape = input_shape

        output_shape = [sum(x) for x in zip(prev_shape, expand_by)]
        assert data is not None
        a_dtype = np.dtype(ctx.get_tensor_dtype(node.input[0]))

        if (
            ctx.can_emit_native(output_name)
            and ctx.can_run_native(node)
            and a_dtype == np.dtype(np.float32)
            and mode == "constant"
            and len(input_shape) <= ViewTransformSemantics.GGML_MAX_DIMS
            and constant_values.size == 1
            and float(np.asarray(constant_values).reshape(()).item()) == 0.0
            and self.can_native_pad_width(pad_width)
        ):
            storage_pads = [0, 0, 0, 0]
            for logical_axis, (_before, after) in enumerate(pad_width):
                storage_axis = len(input_shape) - logical_axis - 1
                storage_pads[storage_axis] = int(after)
            result = ggml.ggml_pad(ctx.ggml_eval_context, data, *storage_pads)
            ctx.register_native_tensor(
                output_name,
                result,
                tuple(output_shape),
                a_dtype,
            )
            return

        storage_dtype = ctx.storage_dtype_for_logical_dtype(a_dtype)
        x = np.empty(output_shape, dtype=storage_dtype)
        output_shape_tracker = ctx.from_numpy(x)

        @ggml.ggml_custom2_op_t
        def custom_pad(
            dst: ggml.ggml_tensor_p,
            a: ggml.ggml_tensor_p,
            b: ggml.ggml_tensor_p,
            ith: int,
            nth: int,
            userdata: Optional[ctypes.c_void_p],
        ):
            b_array = ctx.logical_tensor_data(node.input[0], b, input_shape)
            if mode == "constant":
                padded = np.pad(
                    b_array,
                    pad_width=pad_width,
                    mode=mode,
                    constant_values=constant_values,
                )  # type: ignore
            else:
                padded = np.pad(
                    b_array,
                    pad_width=pad_width,
                    mode=mode,
                )  # type: ignore
            ctx.set_tensor_data(dst, padded)

        ctx.ggml_tensors_dict[output_name] = ggml.ggml_map_custom2(
            ctx.ggml_eval_context,
            output_shape_tracker,
            data,
            custom_pad,
            1,
            None,
        )
        ctx.refs.append(custom_pad)
        ctx.shapes[output_name] = tuple(output_shape)
        ctx.set_tensor_dtype(output_name, a_dtype)


@onnx_operators.register
class PReluOperator(OnnxOperator):
    def __init__(self):
        super().__init__("PRelu")

    def lower(self, ctx: "GgmlOnnxExecutionContext", node: NodeProto) -> None:
        node_inputs = [ctx.ggml_tensors_dict[inp] for inp in node.input]

        if len(node_inputs) != 2:
            raise ValueError(
                f'Error for node "{node.name}": Operation "PRelu" requires exactly two inputs. Actual number of inputs: {len(node_inputs)}'
            )
        x, slope = node_inputs
        x_shape = ctx.shapes[node.input[0]]
        slope_shape = ctx.shapes[node.input[1]]
        x_dtype = get_tensor_dtype(x)
        x_t = ctx.from_numpy(np.empty(x_shape, dtype=x_dtype))

        @ggml.ggml_custom3_op_t
        def custom_prelu(
            tensor_out: ggml.ggml_tensor_p,
            tensor_in_1: ggml.ggml_tensor_p,
            tensor_in_2: ggml.ggml_tensor_p,
            tensor_in_3: ggml.ggml_tensor_p,
            ith: int,
            nth: int,
            userdata: Optional[ctypes.c_void_p],
        ):
            x_data = ctx.to_numpy(tensor_in_2).reshape(x_shape)
            slope_data = ctx.to_numpy(tensor_in_3).reshape(slope_shape)

            if (
                len(slope_shape) == 1
                and len(x_shape) > 1
                and slope_shape[0] == x_shape[1]
            ):
                try:
                    np.broadcast_shapes(x_shape, slope_shape)
                except ValueError:
                    slope_data = slope_data.reshape(
                        (1, slope_shape[0], *((1,) * (len(x_shape) - 2)))
                    )

            y = np.clip(x_data, 0, np.inf) + np.clip(x_data, -np.inf, 0) * slope_data

            ctx.set_tensor_data(tensor_out, y)

        new_tensor = ctx.ggml_tensors_dict[node.output[0]] = (
            ggml.ggml_map_custom3_inplace(
                ctx.ggml_eval_context,
                x_t,
                x,
                slope,
                custom_prelu,
                1,
                None,
            )
        )

        ctx.refs.append(custom_prelu)
        ctx.set_tensor_shape(new_tensor, x_shape)
        ctx.shapes[node.output[0]] = x_shape
        ctx.set_tensor_dtype(node.output[0], x_dtype)


@onnx_operators.register
class PowOperator(OnnxOperator):
    def __init__(self):
        super().__init__(
            "Pow",
            OnnxOperator.EXECUTION_NATIVE_OR_NUMPY_RUNTIME,
            OnnxOperator.CLASS_CONDITIONAL_NATIVE,
        )
        self.has_numpy_evaluator = True

    def strategies(
        self, tensor_types: Dict[str, TensorType], node: "NodeIR"
    ) -> Tuple[Tuple[str, str, str], ...]:
        if len(node.inputs) != 2:
            return self.numpy_runtime_strategy()
        input_dtype = self.tensor_type(tensor_types, node.inputs[0]).dtype
        exponent = self.tensor_type(tensor_types, node.inputs[1]).scalar_value
        if input_dtype == np.dtype(np.float32) and exponent == 2:
            return self.native_strategy()
        return self.numpy_runtime_strategy()

    def eval_numpy(
        self, node: NodeProto, inputs: Tuple[npt.NDArray[Any], ...]
    ) -> Tuple[npt.NDArray[Any], ...]:
        return self.eval_numpy_pow_operator(node, inputs)

    def lower(self, ctx: "GgmlOnnxExecutionContext", node: NodeProto) -> None:
        node_inputs = [ctx.ggml_tensors_dict[inp] for inp in node.input]

        if len(node_inputs) != 2:
            raise ValueError(
                f'Error for node "{node.name}": Operation "Pow" requires exactly two inputs. Actual number of inputs: {len(node_inputs)}'
            )

        x1 = node_inputs[0]
        x2 = node_inputs[1]
        x1_shape = ctx.shapes[node.input[0]]
        x2_shape = ctx.shapes[node.input[1]]
        output_name = node.output[0]
        x1_dtype = np.dtype(ctx.get_tensor_dtype(node.input[0]))

        if (
            ctx.can_emit_native(output_name)
            and ctx.can_run_native(node)
            and x1_dtype == np.dtype(np.float32)
        ):
            exponent = ctx.logical_tensor_eval_data(node.input[1], x2, x2_shape)
            if exponent.size == 1 and exponent.reshape(()).item() == 2:
                result = ggml.ggml_sqr(ctx.ggml_eval_context, x1)
                ctx.register_native_tensor(
                    output_name, result, x1_shape, np.dtype(np.float32)
                )
                return

        output_shape = np.broadcast_shapes(x1_shape, x2_shape)
        x_t = ctx.from_numpy(np.empty(output_shape, dtype=get_tensor_dtype(x1)))

        @ggml.ggml_custom3_op_t
        def custom_pow(
            tensor_out: ggml.ggml_tensor_p,
            tensor_in_1: ggml.ggml_tensor_p,
            tensor_in_2: ggml.ggml_tensor_p,
            tensor_in_3: ggml.ggml_tensor_p,
            ith: int,
            nth: int,
            userdata: Optional[ctypes.c_void_p],
        ):
            x1 = ctx.to_numpy(tensor_in_2).reshape(x1_shape)
            x2 = ctx.to_numpy(tensor_in_3).reshape(x2_shape)

            new_tensor = np.power(x1, x2)

            ctx.set_tensor_data(tensor_out, new_tensor)

        new_tensor = ggml.ggml_map_custom3_inplace(
            ctx.ggml_eval_context,
            x_t,
            x1,
            x2,
            custom_pow,
            1,
            None,
        )

        ctx.refs.append(custom_pow)
        ctx.register_numpy_runtime_tensor(
            output_name, new_tensor, output_shape, get_tensor_dtype(x1)
        )


@onnx_operators.register
class QuantizeLinearOperator(OnnxOperator):
    def __init__(self):
        super().__init__("QuantizeLinear")

    @staticmethod
    def quantization_range(dtype: npt.DTypeLike) -> Tuple[int, int]:
        np_dtype = np.dtype(dtype)
        if not np.issubdtype(np_dtype, np.integer):
            raise TypeError(f"Quantized dtype must be an integer type, got {np_dtype}")
        info = np.iinfo(np_dtype)
        return int(info.min), int(info.max)

    @staticmethod
    def quantization_axis(node: NodeProto, rank: int) -> int:
        axis = next((attr.i for attr in node.attribute if attr.name == "axis"), 1)
        if axis < 0:
            axis += rank
        return int(axis)

    @staticmethod
    def reshape_quantization_parameter(
        value: npt.NDArray[Any],
        input_shape: Tuple[int, ...],
        axis: int,
    ) -> npt.NDArray[Any]:
        if value.shape == ():
            return value
        shape = [1] * len(input_shape)
        shape[axis] = value.shape[0]
        return value.reshape(shape)

    def lower(self, ctx: "GgmlOnnxExecutionContext", node: NodeProto) -> None:
        node_inputs = [ctx.ggml_tensors_dict[inp] for inp in node.input]

        if len(node_inputs) not in {2, 3}:
            raise ValueError(
                f'Error for node "{node.name}": Operation "QuantizeLinear" requires two or three inputs. Actual number of inputs: {len(node_inputs)}'
            )

        x_shape = ctx.shapes[node.input[0]]
        x = ctx.logical_tensor_eval_data(node.input[0], node_inputs[0], x_shape).astype(
            np.float32
        )
        scale = ctx.logical_tensor_eval_data(
            node.input[1], node_inputs[1], ctx.shapes[node.input[1]]
        ).astype(np.float32)
        if len(node_inputs) == 3:
            zero_point_dtype = np.dtype(ctx.get_tensor_dtype(node.input[2]))
            zero_point = ctx.logical_tensor_eval_data(
                node.input[2], node_inputs[2], ctx.shapes[node.input[2]]
            )
        else:
            zero_point_dtype = np.dtype(np.uint8)
            zero_point = np.asarray(0, dtype=zero_point_dtype)

        axis = self.quantization_axis(node, len(x_shape))
        scale = self.reshape_quantization_parameter(scale, x_shape, axis)
        zero_point = self.reshape_quantization_parameter(zero_point, x_shape, axis)

        qmin, qmax = self.quantization_range(zero_point_dtype)
        quantized = np.rint(x / scale) + zero_point.astype(np.float32)
        quantized = np.clip(quantized, qmin, qmax).astype(zero_point_dtype)
        ctx.set_logical_output(node.output[0], quantized, zero_point_dtype)


@onnx_operators.register
class DequantizeLinearOperator(OnnxOperator):
    def __init__(self):
        super().__init__("DequantizeLinear")

    @staticmethod
    def quantization_axis(node: NodeProto, rank: int) -> int:
        axis = next((attr.i for attr in node.attribute if attr.name == "axis"), 1)
        if axis < 0:
            axis += rank
        return int(axis)

    @staticmethod
    def reshape_quantization_parameter(
        value: npt.NDArray[Any],
        input_shape: Tuple[int, ...],
        axis: int,
    ) -> npt.NDArray[Any]:
        if value.shape == ():
            return value
        shape = [1] * len(input_shape)
        shape[axis] = value.shape[0]
        return value.reshape(shape)

    def lower(self, ctx: "GgmlOnnxExecutionContext", node: NodeProto) -> None:
        node_inputs = [ctx.ggml_tensors_dict[inp] for inp in node.input]

        if len(node_inputs) not in {2, 3}:
            raise ValueError(
                f'Error for node "{node.name}": Operation "DequantizeLinear" requires two or three inputs. Actual number of inputs: {len(node_inputs)}'
            )

        x_shape = ctx.shapes[node.input[0]]
        x = ctx.logical_tensor_eval_data(node.input[0], node_inputs[0], x_shape)
        scale = ctx.logical_tensor_eval_data(
            node.input[1], node_inputs[1], ctx.shapes[node.input[1]]
        ).astype(np.float32)
        if len(node_inputs) == 3:
            zero_point = ctx.logical_tensor_eval_data(
                node.input[2], node_inputs[2], ctx.shapes[node.input[2]]
            )
        else:
            zero_point = np.asarray(0, dtype=x.dtype)

        axis = self.quantization_axis(node, len(x_shape))
        scale = self.reshape_quantization_parameter(scale, x_shape, axis)
        zero_point = self.reshape_quantization_parameter(zero_point, x_shape, axis)

        dequantized = (x.astype(np.float32) - zero_point.astype(np.float32)) * scale
        ctx.set_logical_output(node.output[0], dequantized, np.float32)


@onnx_operators.register
class DynamicQuantizeLinearOperator(OnnxOperator):
    def __init__(self):
        super().__init__("DynamicQuantizeLinear")

    def lower(self, ctx: "GgmlOnnxExecutionContext", node: NodeProto) -> None:
        node_inputs = [ctx.ggml_tensors_dict[inp] for inp in node.input]

        if len(node_inputs) != 1:
            raise ValueError(
                f'Error for node "{node.name}": Operation "DynamicQuantizeLinear" requires exactly one input. Actual number of inputs: {len(node_inputs)}'
            )

        x_shape = ctx.shapes[node.input[0]]
        x = ctx.logical_tensor_eval_data(node.input[0], node_inputs[0], x_shape).astype(
            np.float32
        )
        x_min = min(float(np.min(x)), 0.0)
        x_max = max(float(np.max(x)), 0.0)
        scale = np.asarray((x_max - x_min) / 255.0, dtype=np.float32)
        if scale == 0:
            zero_point = np.asarray(0, dtype=np.uint8)
            y = np.zeros(x_shape, dtype=np.uint8)
        else:
            zero_point = np.asarray(
                np.clip(np.rint(-x_min / float(scale)), 0, 255), dtype=np.uint8
            )
            y = np.clip(
                np.rint(x / scale) + zero_point.astype(np.float32), 0, 255
            ).astype(np.uint8)

        ctx.set_logical_output(node.output[0], y, np.uint8)
        ctx.set_logical_output(node.output[1], scale, np.float32)
        ctx.set_logical_output(node.output[2], zero_point, np.uint8)


@onnx_operators.register
class MatMulIntegerOperator(OnnxOperator):
    def __init__(self):
        super().__init__("MatMulInteger")

    def lower(self, ctx: "GgmlOnnxExecutionContext", node: NodeProto) -> None:
        node_inputs = [ctx.ggml_tensors_dict[inp] for inp in node.input]

        if len(node_inputs) not in {2, 3, 4}:
            raise ValueError(
                f'Error for node "{node.name}": Operation "MatMulInteger" requires two to four inputs. Actual number of inputs: {len(node_inputs)}'
            )

        a = ctx.logical_tensor_eval_data(
            node.input[0], node_inputs[0], ctx.shapes[node.input[0]]
        ).astype(np.int32)
        b = ctx.logical_tensor_eval_data(
            node.input[1], node_inputs[1], ctx.shapes[node.input[1]]
        ).astype(np.int32)
        a_zero_point = (
            ctx.logical_tensor_eval_data(
                node.input[2], node_inputs[2], ctx.shapes[node.input[2]]
            ).astype(np.int32)
            if len(node_inputs) >= 3
            else np.asarray(0, dtype=np.int32)
        )
        b_zero_point = (
            ctx.logical_tensor_eval_data(
                node.input[3], node_inputs[3], ctx.shapes[node.input[3]]
            ).astype(np.int32)
            if len(node_inputs) == 4
            else np.asarray(0, dtype=np.int32)
        )

        y = np.matmul(a - a_zero_point, b - b_zero_point).astype(np.int32)
        ctx.set_logical_output(node.output[0], y, np.int32)


@onnx_operators.register
class QLinearMatMulOperator(OnnxOperator):
    def __init__(self):
        super().__init__("QLinearMatMul")

    @staticmethod
    def quantization_range(dtype: npt.DTypeLike) -> Tuple[int, int]:
        np_dtype = np.dtype(dtype)
        if not np.issubdtype(np_dtype, np.integer):
            raise TypeError(f"Quantized dtype must be an integer type, got {np_dtype}")
        info = np.iinfo(np_dtype)
        return int(info.min), int(info.max)

    def lower(self, ctx: "GgmlOnnxExecutionContext", node: NodeProto) -> None:
        node_inputs = [ctx.ggml_tensors_dict[inp] for inp in node.input]

        if len(node_inputs) != 8:
            raise ValueError(
                f'Error for node "{node.name}": Operation "QLinearMatMul" requires exactly eight inputs. Actual number of inputs: {len(node_inputs)}'
            )

        a = ctx.logical_tensor_eval_data(
            node.input[0], node_inputs[0], ctx.shapes[node.input[0]]
        ).astype(np.int32)
        a_scale = ctx.logical_tensor_eval_data(
            node.input[1], node_inputs[1], ctx.shapes[node.input[1]]
        ).astype(np.float32)
        a_zero_point = ctx.logical_tensor_eval_data(
            node.input[2], node_inputs[2], ctx.shapes[node.input[2]]
        ).astype(np.int32)
        b = ctx.logical_tensor_eval_data(
            node.input[3], node_inputs[3], ctx.shapes[node.input[3]]
        ).astype(np.int32)
        b_scale = ctx.logical_tensor_eval_data(
            node.input[4], node_inputs[4], ctx.shapes[node.input[4]]
        ).astype(np.float32)
        b_zero_point = ctx.logical_tensor_eval_data(
            node.input[5], node_inputs[5], ctx.shapes[node.input[5]]
        ).astype(np.int32)
        y_scale = ctx.logical_tensor_eval_data(
            node.input[6], node_inputs[6], ctx.shapes[node.input[6]]
        ).astype(np.float32)
        y_zero_point = ctx.logical_tensor_eval_data(
            node.input[7], node_inputs[7], ctx.shapes[node.input[7]]
        )
        y_dtype = np.dtype(ctx.get_tensor_dtype(node.input[7]))

        y_min, y_max = self.quantization_range(y_dtype)
        y = np.matmul(a - a_zero_point, b - b_zero_point).astype(np.float32)
        y = np.rint(y * a_scale * b_scale / y_scale) + y_zero_point.astype(np.float32)
        y = np.clip(y, y_min, y_max).astype(y_dtype)
        ctx.set_logical_output(node.output[0], y, y_dtype)


@onnx_operators.register
class RandomNormalLikeOperator(OnnxOperator):
    def __init__(self):
        super().__init__("RandomNormalLike")

    def lower(self, ctx: "GgmlOnnxExecutionContext", node: NodeProto) -> None:
        node_inputs = [ctx.ggml_tensors_dict[inp] for inp in node.input]

        if len(node_inputs) != 1:
            raise ValueError(
                f'Error for node "{node.name}": Operation "RandomNormalLike" requires exactly one input. Actual number of inputs: {len(node_inputs)}'
            )

        shape = ctx.shapes[node.input[0]]
        dtype_attr = next(
            (attr.i for attr in node.attribute if attr.name == "dtype"), None
        )
        output_dtype = (
            np.dtype(tensor_dtype_to_np_dtype(dtype_attr))
            if dtype_attr is not None
            else np.dtype(ctx.get_tensor_dtype(node.input[0]))
        )
        mean = next((attr.f for attr in node.attribute if attr.name == "mean"), 0.0)
        scale = next((attr.f for attr in node.attribute if attr.name == "scale"), 1.0)
        seed = next((attr.f for attr in node.attribute if attr.name == "seed"), None)
        rng = np.random.default_rng(None if seed is None else int(seed))
        output = rng.normal(loc=mean, scale=scale, size=shape).astype(output_dtype)
        ctx.set_logical_output(node.output[0], output, output_dtype)


@onnx_operators.register
class RMSNormalizationOperator(OnnxOperator):
    def __init__(self):
        super().__init__(
            "RMSNormalization",
            OnnxOperator.EXECUTION_NATIVE_OR_NUMPY_RUNTIME,
            OnnxOperator.CLASS_CONDITIONAL_NATIVE,
        )
        self.has_numpy_evaluator = True

    @staticmethod
    def normalized_axis(node: NodeProto, rank: int) -> int:
        axis = OnnxOperator.int_attribute(node, "axis", -1)
        return axis + rank if axis < 0 else axis

    def native_parameters(
        self,
        tensor_types: Dict[str, TensorType],
        node: "NodeIR",
    ) -> Optional[Tuple[Tuple[int, ...], Tuple[int, ...]]]:
        if len(node.inputs) != 2:
            return None
        input_type = self.tensor_type(tensor_types, node.inputs[0])
        scale_type = self.tensor_type(tensor_types, node.inputs[1])
        if (
            not input_type.is_float32
            or not scale_type.is_float32
            or input_type.shape is None
            or scale_type.shape is None
        ):
            return None
        axis = int(node.attribute("axis", -1))
        axis = axis + len(input_type.shape) if axis < 0 else axis
        if axis != len(input_type.shape) - len(scale_type.shape):
            return None
        if axis != len(input_type.shape) - 1:
            return None
        if not self.can_repeat_to_shape(scale_type.shape, input_type.shape):
            return None
        return input_type.shape, scale_type.shape

    def strategies(
        self, tensor_types: Dict[str, TensorType], node: "NodeIR"
    ) -> Tuple[Tuple[str, str, str], ...]:
        if self.native_parameters(tensor_types, node) is not None:
            return self.native_strategy()
        return self.numpy_runtime_strategy(
            "RMSNormalization requires float32 input/scale and last-axis "
            "normalization to lower to ggml_rms_norm"
        )

    def eval_numpy(
        self, node: NodeProto, inputs: Tuple[npt.NDArray[Any], ...]
    ) -> Tuple[npt.NDArray[Any], ...]:
        if len(inputs) != 2:
            raise ValueError(f'Operation "{node.op_type}" requires two inputs')
        x, scale = inputs
        epsilon = self.float_attribute(node, "epsilon", 1e-5)
        axis = self.normalized_axis(node, x.ndim)
        axes = tuple(range(axis, x.ndim))
        mean_square = np.mean(np.square(x.astype(np.float32)), axis=axes, keepdims=True)
        normalized = x.astype(np.float32) / np.sqrt(mean_square + epsilon)
        return (np.asarray(normalized * scale, dtype=x.dtype),)

    def lower(self, ctx: "GgmlOnnxExecutionContext", node: NodeProto) -> None:
        node_inputs = [ctx.ggml_tensors_dict[inp] for inp in node.input]
        if len(node_inputs) != 2:
            raise ValueError(
                f'Error for node "{node.name}": RMSNormalization requires two inputs'
            )
        input_name, scale_name = node.input
        input_shape = ctx.shapes[input_name]
        scale_shape = ctx.shapes[scale_name]
        input_dtype = np.dtype(ctx.get_tensor_dtype(input_name))
        scale_dtype = np.dtype(ctx.get_tensor_dtype(scale_name))
        axis = self.normalized_axis(node, len(input_shape))
        epsilon = self.float_attribute(node, "epsilon", 1e-5)

        if (
            ctx.can_emit_native(node.output[0])
            and ctx.can_run_native(node)
            and input_dtype == np.dtype(np.float32)
            and scale_dtype == np.dtype(np.float32)
            and axis == len(input_shape) - len(scale_shape)
            and axis == len(input_shape) - 1
            and self.can_repeat_to_shape(scale_shape, input_shape)
        ):
            normalized = ggml.ggml_rms_norm(
                ctx.ggml_eval_context, node_inputs[0], epsilon
            )
            scale = node_inputs[1]
            if scale_shape != input_shape:
                scale = self.repeat_native_tensor_to_shape(
                    ctx, scale, scale_shape, input_shape, np.dtype(np.float32)
                )
            result = ggml.ggml_mul(ctx.ggml_eval_context, normalized, scale)
            ctx.register_native_tensor(
                node.output[0], result, input_shape, np.dtype(np.float32)
            )
            return

        input_array = ctx.logical_tensor_eval_data(
            input_name, node_inputs[0], input_shape
        )
        scale_array = ctx.logical_tensor_eval_data(
            scale_name, node_inputs[1], scale_shape
        )
        output = self.eval_numpy(node, (input_array, scale_array))[0]
        ctx.set_numpy_runtime_output(node.output[0], output, output.dtype)


@onnx_operators.register
class RangeOperator(OnnxOperator):
    def __init__(self):
        super().__init__("Range")

    def lower(self, ctx: "GgmlOnnxExecutionContext", node: NodeProto) -> None:
        node_inputs = [ctx.ggml_tensors_dict[inp] for inp in node.input]

        if len(node_inputs) != 3:
            raise ValueError(
                f'Error for node "{node.name}": Operation "Range" requires exactly three inputs. Actual number of inputs: {len(node_inputs)}'
            )

        start_tensor, limit_tensor, delta_tensor = (
            ctx.eval_tensor(node_input) for node_input in node_inputs
        )
        start = ctx.to_numpy(start_tensor).reshape(ctx.shapes[node.input[0]]).item()
        limit = ctx.to_numpy(limit_tensor).reshape(ctx.shapes[node.input[1]]).item()
        delta = ctx.to_numpy(delta_tensor).reshape(ctx.shapes[node.input[2]]).item()
        output_dtype = np.dtype(ctx.get_tensor_dtype(node.input[0]))
        output = np.arange(start, limit, delta, dtype=output_dtype)

        new_tensor = ctx.ggml_tensors_dict[node.output[0]] = ctx.from_numpy(output)
        ctx.set_tensor_shape(new_tensor, output.shape)
        ctx.shapes[node.output[0]] = output.shape
        ctx.set_tensor_dtype(node.output[0], output.dtype)


@onnx_operators.register
class ReciprocalOperator(OnnxOperator):
    def __init__(self):
        super().__init__("Reciprocal")

    def lower(self, ctx: "GgmlOnnxExecutionContext", node: NodeProto) -> None:
        node_inputs = [ctx.ggml_tensors_dict[inp] for inp in node.input]

        if len(node_inputs) != 1:
            raise ValueError(
                f'Error for node "{node.name}": Operation "Reciprocal" requires exactly one input. Actual number of inputs: {len(node_inputs)}'
            )

        x = node_inputs[0]

        @ggml.ggml_custom1_op_t
        def custom_reciprocal(
            tensor_out: ggml.ggml_tensor_p,
            tensor_in_1: ggml.ggml_tensor_p,
            ith: int,
            nth: int,
            userdata: Optional[ctypes.c_void_p],
        ):
            x = ctx.to_numpy(tensor_in_1)
            y = np.reciprocal(x)

            ctx.set_tensor_data(tensor_out, y)

        new_tensor = ctx.ggml_tensors_dict[node.output[0]] = (
            ggml.ggml_map_custom1_inplace(
                ctx.ggml_eval_context,
                x,
                custom_reciprocal,
                1,
                None,
            )
        )

        ctx.refs.append(custom_reciprocal)
        ctx.set_tensor_shape(new_tensor, ctx.shapes[node.input[0]])
        ctx.shapes[node.output[0]] = ctx.shapes[node.input[0]]


@onnx_operators.register
class ReduceL1Operator(OnnxOperator):
    def __init__(self):
        super().__init__(
            "ReduceL1",
            OnnxOperator.EXECUTION_NATIVE_OR_NUMPY_RUNTIME,
            OnnxOperator.CLASS_CONDITIONAL_NATIVE,
        )
        self.has_numpy_evaluator = True

    def strategies(
        self, tensor_types: Dict[str, TensorType], node: "NodeIR"
    ) -> Tuple[Tuple[str, str, str], ...]:
        return self.reduce_all_native_strategy(tensor_types, node)

    def eval_numpy(
        self, node: NodeProto, inputs: Tuple[npt.NDArray[Any], ...]
    ) -> Tuple[npt.NDArray[Any], ...]:
        return self.eval_numpy_reduce_operator(
            node,
            inputs,
            lambda tensor, axes, keepdims: np.sum(
                np.abs(tensor), axis=axes, keepdims=keepdims
            ),
        )

    def lower(self, ctx: "GgmlOnnxExecutionContext", node: NodeProto) -> None:
        self.lower_native_reduce_all_or_numpy(
            ctx,
            node,
            lambda reduce_ctx, tensor: ggml.ggml_sum(
                reduce_ctx.ggml_eval_context,
                ggml.ggml_abs(reduce_ctx.ggml_eval_context, tensor),
            ),
            lambda tensor, axes, keepdims: np.sum(
                np.abs(tensor), axis=axes, keepdims=keepdims
            ),
        )


@onnx_operators.register
class ReduceL2Operator(OnnxOperator):
    def __init__(self):
        super().__init__(
            "ReduceL2",
            OnnxOperator.EXECUTION_NATIVE_OR_NUMPY_RUNTIME,
            OnnxOperator.CLASS_CONDITIONAL_NATIVE,
        )
        self.has_numpy_evaluator = True

    def strategies(
        self, tensor_types: Dict[str, TensorType], node: "NodeIR"
    ) -> Tuple[Tuple[str, str, str], ...]:
        return self.reduce_all_native_strategy(tensor_types, node)

    def eval_numpy(
        self, node: NodeProto, inputs: Tuple[npt.NDArray[Any], ...]
    ) -> Tuple[npt.NDArray[Any], ...]:
        return self.eval_numpy_reduce_operator(
            node,
            inputs,
            lambda tensor, axes, keepdims: np.sqrt(
                np.sum(np.square(tensor), axis=axes, keepdims=keepdims)
            ),
        )

    def lower(self, ctx: "GgmlOnnxExecutionContext", node: NodeProto) -> None:
        self.lower_native_reduce_all_or_numpy(
            ctx,
            node,
            lambda reduce_ctx, tensor: ggml.ggml_sqrt(
                reduce_ctx.ggml_eval_context,
                ggml.ggml_sum(
                    reduce_ctx.ggml_eval_context,
                    ggml.ggml_sqr(reduce_ctx.ggml_eval_context, tensor),
                ),
            ),
            lambda tensor, axes, keepdims: np.sqrt(
                np.sum(np.square(tensor), axis=axes, keepdims=keepdims)
            ),
        )


@onnx_operators.register
class ReduceLogSumOperator(OnnxOperator):
    def __init__(self):
        super().__init__("ReduceLogSum")

    def lower(self, ctx: "GgmlOnnxExecutionContext", node: NodeProto) -> None:
        self.lower_reduce(
            ctx,
            node,
            lambda tensor, axes, keepdims: np.log(
                np.sum(tensor, axis=axes, keepdims=keepdims)
            ),
        )


@onnx_operators.register
class ReduceLogSumExpOperator(OnnxOperator):
    def __init__(self):
        super().__init__("ReduceLogSumExp")

    def lower(self, ctx: "GgmlOnnxExecutionContext", node: NodeProto) -> None:
        self.lower_reduce(
            ctx,
            node,
            lambda tensor, axes, keepdims: np.log(
                np.sum(np.exp(tensor), axis=axes, keepdims=keepdims)
            ),
        )


@onnx_operators.register
class ReduceMaxOperator(OnnxOperator):
    def __init__(self):
        super().__init__("ReduceMax")

    def lower(self, ctx: "GgmlOnnxExecutionContext", node: NodeProto) -> None:
        self.lower_reduce(
            ctx,
            node,
            lambda tensor, axes, keepdims: np.max(tensor, axis=axes, keepdims=keepdims),
        )


@onnx_operators.register
class ReduceMeanOperator(OnnxOperator):
    def __init__(self):
        super().__init__("ReduceMean")
        self.has_numpy_evaluator = True

    def strategies(
        self, tensor_types: Dict[str, TensorType], node: "NodeIR"
    ) -> Tuple[Tuple[str, str, str], ...]:
        del tensor_types, node
        return self.numpy_runtime_strategy()

    def eval_numpy(
        self, node: NodeProto, inputs: Tuple[npt.NDArray[Any], ...]
    ) -> Tuple[npt.NDArray[Any], ...]:
        return self.eval_numpy_reduce_operator(
            node,
            inputs,
            lambda tensor, axes, keepdims: np.mean(
                tensor, axis=axes, keepdims=keepdims
            ),
        )

    def lower(self, ctx: "GgmlOnnxExecutionContext", node: NodeProto) -> None:
        self.lower_reduce(
            ctx,
            node,
            lambda tensor, axes, keepdims: np.mean(
                tensor, axis=axes, keepdims=keepdims
            ),
        )


@onnx_operators.register
class ReduceMinOperator(OnnxOperator):
    def __init__(self):
        super().__init__("ReduceMin")

    def lower(self, ctx: "GgmlOnnxExecutionContext", node: NodeProto) -> None:
        self.lower_reduce(
            ctx,
            node,
            lambda tensor, axes, keepdims: np.min(tensor, axis=axes, keepdims=keepdims),
        )


@onnx_operators.register
class ReduceProdOperator(OnnxOperator):
    def __init__(self):
        super().__init__("ReduceProd")

    def lower(self, ctx: "GgmlOnnxExecutionContext", node: NodeProto) -> None:
        self.lower_reduce(
            ctx,
            node,
            lambda tensor, axes, keepdims: np.prod(
                tensor, axis=axes, keepdims=keepdims
            ),
        )


@onnx_operators.register
class ReduceSumOperator(OnnxOperator):
    def __init__(self):
        super().__init__(
            "ReduceSum",
            OnnxOperator.EXECUTION_NATIVE_OR_NUMPY_RUNTIME,
            OnnxOperator.CLASS_CONDITIONAL_NATIVE,
        )
        self.has_numpy_evaluator = True

    def strategies(
        self, tensor_types: Dict[str, TensorType], node: "NodeIR"
    ) -> Tuple[Tuple[str, str, str], ...]:
        return self.reduce_all_native_strategy(tensor_types, node)

    def eval_numpy(
        self, node: NodeProto, inputs: Tuple[npt.NDArray[Any], ...]
    ) -> Tuple[npt.NDArray[Any], ...]:
        return self.eval_numpy_reduce_sum_operator(node, inputs)

    def lower(self, ctx: "GgmlOnnxExecutionContext", node: NodeProto) -> None:
        self.lower_native_reduce_all_or_numpy(
            ctx,
            node,
            lambda reduce_ctx, tensor: ggml.ggml_sum(
                reduce_ctx.ggml_eval_context, tensor
            ),
            lambda tensor, axes, keepdims: np.sum(tensor, axis=axes, keepdims=keepdims),
        )


@onnx_operators.register
class ReduceSumSquareOperator(OnnxOperator):
    def __init__(self):
        super().__init__(
            "ReduceSumSquare",
            OnnxOperator.EXECUTION_NATIVE_OR_NUMPY_RUNTIME,
            OnnxOperator.CLASS_CONDITIONAL_NATIVE,
        )
        self.has_numpy_evaluator = True

    def strategies(
        self, tensor_types: Dict[str, TensorType], node: "NodeIR"
    ) -> Tuple[Tuple[str, str, str], ...]:
        return self.reduce_all_native_strategy(tensor_types, node)

    def eval_numpy(
        self, node: NodeProto, inputs: Tuple[npt.NDArray[Any], ...]
    ) -> Tuple[npt.NDArray[Any], ...]:
        return self.eval_numpy_reduce_operator(
            node,
            inputs,
            lambda tensor, axes, keepdims: np.sum(
                np.square(tensor), axis=axes, keepdims=keepdims
            ),
        )

    def lower(self, ctx: "GgmlOnnxExecutionContext", node: NodeProto) -> None:
        self.lower_native_reduce_all_or_numpy(
            ctx,
            node,
            lambda reduce_ctx, tensor: ggml.ggml_sum(
                reduce_ctx.ggml_eval_context,
                ggml.ggml_sqr(reduce_ctx.ggml_eval_context, tensor),
            ),
            lambda tensor, axes, keepdims: np.sum(
                np.square(tensor), axis=axes, keepdims=keepdims
            ),
        )


@onnx_operators.register
class ReluOperator(OnnxOperator):
    def __init__(self):
        super().__init__(
            "Relu",
            OnnxOperator.EXECUTION_NATIVE_OR_NUMPY_RUNTIME,
            OnnxOperator.CLASS_CONDITIONAL_NATIVE,
        )
        self.has_numpy_evaluator = True

    def strategies(
        self, tensor_types: Dict[str, TensorType], node: "NodeIR"
    ) -> Tuple[Tuple[str, str, str], ...]:
        if (
            len(node.inputs) == 1
            and self.tensor_type(tensor_types, node.inputs[0]).is_float32
        ):
            return self.native_strategy()
        return self.numpy_runtime_strategy()

    def eval_numpy(
        self, node: NodeProto, inputs: Tuple[npt.NDArray[Any], ...]
    ) -> Tuple[npt.NDArray[Any], ...]:
        return self.eval_numpy_unary_operator(node, inputs, lambda x: np.maximum(x, 0))

    def lower(self, ctx: "GgmlOnnxExecutionContext", node: NodeProto) -> None:
        self.lower_native_unary_or_numpy(
            ctx, node, ggml.ggml_relu, lambda x: np.maximum(x, 0)
        )


@onnx_operators.register
class ReshapeOperator(ViewOnnxOperator):
    def __init__(self):
        super().__init__("Reshape", ViewTransformSemantics.KIND_SHAPE)

    def validate_view_inputs(
        self, node: NodeProto, node_inputs: Sequence[ggml.ggml_tensor_p]
    ) -> None:
        if len(node_inputs) != 2:
            raise ValueError(
                f'Error for node "{node.name}": Operation "Reshape" requires exactly two inputs. Actual number of inputs: {len(node_inputs)}'
            )

    def has_static_parameters(
        self,
        tensor_types: Dict[str, TensorType],
        node: "NodeIR",
        input_type: TensorType,
    ) -> bool:
        del input_type
        return (
            len(node.inputs) == 2
            and self.tensor_type(tensor_types, node.inputs[1]).constant
        )

    def reshape_shape(
        self,
        input_shape: Tuple[int, ...],
        shape_values: npt.NDArray[Any],
        node: NodeProto,
    ) -> Tuple[int, ...]:
        new_shape = np.asarray(shape_values, dtype=np.int64).ravel().copy()
        if self.int_attribute(node, "allowzero", 0) != 1:
            keep_idxs = np.where(new_shape == 0)[0]
            new_shape[keep_idxs] = np.array(input_shape)[keep_idxs]
        return np.empty(input_shape, dtype=np.float32).reshape(new_shape).shape

    def runtime_output_shape(
        self,
        ctx: "GgmlOnnxExecutionContext",
        node: NodeProto,
        input_shape: Tuple[int, ...],
    ) -> Tuple[int, ...]:
        shape_input_name = node.input[1]
        shape_tensor = ctx.ggml_tensors_dict[shape_input_name]
        shape_values = ctx.logical_tensor_eval_data(
            shape_input_name,
            shape_tensor,
            ctx.shapes[shape_input_name],
        )
        return self.reshape_shape(input_shape, shape_values, node)

    def eval_numpy(
        self, node: NodeProto, inputs: Tuple[npt.NDArray[Any], ...]
    ) -> Tuple[npt.NDArray[Any], ...]:
        if len(inputs) != 2:
            raise ValueError(
                f'Operation "{node.op_type}" requires exactly two inputs. '
                f"Actual number of inputs: {len(inputs)}"
            )
        output_shape = self.reshape_shape(inputs[0].shape, inputs[1], node)
        return (np.reshape(inputs[0], output_shape),)


@onnx_operators.register
class ResizeOperator(OnnxOperator):
    def __init__(self):
        super().__init__("Resize")

    def lower(self, ctx: "GgmlOnnxExecutionContext", node: NodeProto) -> None:
        node_inputs = [
            ctx.ggml_tensors_dict[inp] if inp != "" else None for inp in node.input
        ]
        node_inputs.extend([None] * (4 - len(node_inputs)))

        if len(node_inputs) > 4:
            raise ValueError(
                f'Error for node "{node.name}": Operation "Resize" requires 1-4 inputs. Actual number of inputs: {len(node_inputs)}'
            )

        a, roi_T, scales_T, sizes_T = node_inputs

        assert a is not None

        a_shape = ctx.shapes[node.input[0]]
        a_dtype = get_tensor_dtype(a)

        coordinate_transformation_mode = next(
            (
                attr.s.decode("utf-8")
                for attr in node.attribute
                if attr.name == "coordinate_transformation_mode"
            ),
            "half_pixel",
        )
        cubic_coeff_a = next(
            (attr.f for attr in node.attribute if attr.name == "cubic_coeff_a"), -0.75
        )
        mode = next(
            (attr.s.decode("utf-8") for attr in node.attribute if attr.name == "mode"),
            "nearest",
        )
        nearest_mode = next(
            (
                attr.s.decode("utf-8")
                for attr in node.attribute
                if attr.name == "nearest_mode"
            ),
            "round_prefer_floor",
        )
        antialias = bool(
            next((attr.i for attr in node.attribute if attr.name == "antialias"), 0)
        )
        exclude_outside = bool(
            next(
                (attr.i for attr in node.attribute if attr.name == "exclude_outside"), 0
            )
        )
        extrapolation_value = next(
            (attr.f for attr in node.attribute if attr.name == "extrapolation_value"),
            0.0,
        )
        keep_aspect_ratio_policy = next(
            (
                attr.s.decode("utf-8")
                for attr in node.attribute
                if attr.name == "keep_aspect_ratio_policy"
            ),
            "stretch",
        )

        if mode not in ["nearest", "linear", "cubic"]:
            raise NotImplementedError(f"mode={mode} is not supported")
        if nearest_mode not in [
            "ceil",
            "floor",
            "round_prefer_ceil",
            "round_prefer_floor",
        ]:
            raise NotImplementedError(f"nearest_mode={nearest_mode} is not supported")
        if keep_aspect_ratio_policy not in ["stretch", "not_larger", "not_smaller"]:
            raise NotImplementedError(
                f"keep_aspect_ratio_policy={keep_aspect_ratio_policy} is not supported"
            )
        if coordinate_transformation_mode not in [
            "align_corners",
            "asymmetric",
            "half_pixel",
            "half_pixel_symmetric",
            "pytorch_half_pixel",
            "tf_crop_and_resize",
        ]:
            raise NotImplementedError(
                f"coordinate_transformation_mode={coordinate_transformation_mode} is not supported"
            )

        axes_attr = next(
            (attr.ints for attr in node.attribute if attr.name == "axes"), None
        )
        axes = (
            tuple(range(len(a_shape)))
            if axes_attr is None
            else tuple(
                int(axis if axis >= 0 else axis + len(a_shape)) for axis in axes_attr
            )
        )

        roi: Optional[np.ndarray] = None
        if roi_T is not None:
            roi_t = ctx.eval_tensor(roi_T)
            roi_shape = ctx.shapes[node.input[1]]
            roi_input = (
                ctx.to_numpy(roi_t).reshape(roi_shape).astype(dtype=np.float64).ravel()
            )
            if roi_input.shape[0] == 2 * len(a_shape):
                roi = roi_input
            elif roi_input.shape[0] == 2 * len(axes):
                roi = np.concatenate(
                    [
                        np.zeros(len(a_shape), dtype=np.float64),
                        np.ones(len(a_shape), dtype=np.float64),
                    ]
                )
                for i, axis in enumerate(axes):
                    roi[axis] = roi_input[i]
                    roi[len(a_shape) + axis] = roi_input[len(axes) + i]
            else:
                raise ValueError(
                    f'Error for node "{node.name}": "roi" parameter must have length 2 * rank(X) or 2 * len(axes)'
                )

        if sizes_T is not None:
            sizes_t = ctx.eval_tensor(sizes_T)
            sizes_shape = ctx.shapes[node.input[3]]
            sizes = tuple(
                int(dim) for dim in ctx.to_numpy(sizes_t).reshape(sizes_shape).ravel()
            )
            if len(sizes) == len(a_shape):
                output_shape = sizes
            elif len(sizes) == len(axes):
                output_shape_list = list(a_shape)
                if keep_aspect_ratio_policy == "stretch":
                    for axis, dim in zip(axes, sizes):
                        output_shape_list[axis] = dim
                else:
                    axis_scales = [
                        target_size / a_shape[axis]
                        for axis, target_size in zip(axes, sizes)
                    ]
                    scale = (
                        min(axis_scales)
                        if keep_aspect_ratio_policy == "not_larger"
                        else max(axis_scales)
                    )
                    resize_dim = (
                        math.floor
                        if keep_aspect_ratio_policy == "not_larger"
                        else math.ceil
                    )
                    for axis in axes:
                        output_shape_list[axis] = resize_dim(a_shape[axis] * scale)
                output_shape = tuple(output_shape_list)
            else:
                raise ValueError(
                    f'Error for node "{node.name}": "sizes" parameter must have the same length as the rank of input "X" or "axes"'
                )
            scales = np.divide(output_shape, a_shape, dtype=np.float64)
        else:
            if scales_T is None:
                raise ValueError(
                    f'Error for node "{node.name}": Operation "Resize" requires scales or sizes'
                )
            scales_t = ctx.eval_tensor(scales_T)
            scales_shape = ctx.shapes[node.input[2]]
            scales_input = (
                ctx.to_numpy(scales_t).reshape(scales_shape).astype(dtype=np.float64)
            )
            if scales_input.shape[0] == len(a_shape):
                scales = scales_input
            elif scales_input.shape[0] == len(axes):
                scales = np.ones(len(a_shape), dtype=np.float64)
                for axis, scale in zip(axes, scales_input):
                    scales[axis] = scale
            else:
                raise ValueError(
                    f'Error for node "{node.name}": "scales" parameter must have the same length as the rank of input "X" or "axes"'
                )
            output_shape = tuple(
                int(dim)
                for dim in np.floor(np.asarray(a_shape) * scales).astype(np.int64)
            )

        x_t = ctx.from_numpy(np.empty(output_shape, dtype=a_dtype))

        def resize_source_coordinate(output_index: int, axis: int) -> Optional[float]:
            input_size = a_shape[axis]
            output_size = output_shape[axis]
            scale = scales[axis]
            output_width = scale * input_size
            if coordinate_transformation_mode == "asymmetric":
                source = output_index / scale
            elif coordinate_transformation_mode == "align_corners":
                source = (
                    0
                    if output_width == 1
                    else output_index * (input_size - 1) / (output_width - 1)
                )
            elif coordinate_transformation_mode == "tf_crop_and_resize":
                if roi is None:
                    raise ValueError(
                        f'Error for node "{node.name}": "roi" is required for tf_crop_and_resize'
                    )
                start = roi[axis]
                end = roi[len(a_shape) + axis]
                if output_width == 1:
                    source = (end - start) * (input_size - 1) / 2
                else:
                    source = (
                        output_index
                        * (end - start)
                        * (input_size - 1)
                        / (output_width - 1)
                    )
                source += start * (input_size - 1)
                if source < 0 or source > input_size - 1:
                    return None
            elif (
                coordinate_transformation_mode == "pytorch_half_pixel"
                and output_width == 1
            ):
                source = -0.5
            elif coordinate_transformation_mode == "half_pixel_symmetric":
                adjustment = output_size / output_width
                center = input_size / 2
                offset = center * (1 - adjustment)
                source = offset + (output_index + 0.5) / scale - 0.5
            else:
                source = (output_index + 0.5) / scale - 0.5
            return source

        def resize_source_index(output_index: int, axis: int) -> int:
            input_size = a_shape[axis]
            source = resize_source_coordinate(output_index, axis)
            if source is None:
                return 0
            if nearest_mode == "ceil":
                nearest = math.ceil(source)
            elif nearest_mode == "floor":
                nearest = math.floor(source)
            elif nearest_mode == "round_prefer_ceil":
                nearest = math.ceil(source) if source % 1 == 0.5 else round(source)
            else:
                nearest = math.floor(source) if source % 1 == 0.5 else round(source)
            return min(max(int(nearest), 0), input_size - 1)

        def resize_linear_coeffs(ratio: float, scale: float) -> np.ndarray:
            if antialias:
                scale = min(scale, 1.0)
                start = int(np.floor(-1 / scale) + 1)
                footprint = 2 - 2 * start
                args = (np.arange(start, start + footprint) - ratio) * scale
                coeffs = np.clip(1 - np.abs(args), 0, 1)
                return np.asarray(coeffs / np.sum(coeffs), dtype=np.float64)
            return np.array([1 - ratio, ratio], dtype=np.float64)

        def resize_cubic_coeffs(ratio: float, scale: float) -> np.ndarray:
            if antialias:
                scale = min(scale, 1.0)

                def compute_coeff(x: float) -> float:
                    x = abs(x)
                    x_2 = x * x
                    x_3 = x * x_2
                    if x <= 1:
                        return (cubic_coeff_a + 2) * x_3 - (cubic_coeff_a + 3) * x_2 + 1
                    if x < 2:
                        return (
                            cubic_coeff_a * x_3
                            - 5 * cubic_coeff_a * x_2
                            + 8 * cubic_coeff_a * x
                            - 4 * cubic_coeff_a
                        )
                    return 0.0

                start = int(np.floor(-2 / scale) + 1)
                end = 2 - start
                coeffs = np.array(
                    [compute_coeff(scale * (i - ratio)) for i in range(start, end)],
                    dtype=np.float64,
                )
                return coeffs / np.sum(coeffs)

            coeffs = [
                (
                    (cubic_coeff_a * (ratio + 1) - 5 * cubic_coeff_a) * (ratio + 1)
                    + 8 * cubic_coeff_a
                )
                * (ratio + 1)
                - 4 * cubic_coeff_a,
                ((cubic_coeff_a + 2) * ratio - (cubic_coeff_a + 3)) * ratio * ratio + 1,
                ((cubic_coeff_a + 2) * (1 - ratio) - (cubic_coeff_a + 3))
                * (1 - ratio)
                * (1 - ratio)
                + 1,
                (
                    (cubic_coeff_a * ((1 - ratio) + 1) - 5 * cubic_coeff_a)
                    * ((1 - ratio) + 1)
                    + 8 * cubic_coeff_a
                )
                * ((1 - ratio) + 1)
                - 4 * cubic_coeff_a,
            ]
            return np.array(coeffs, dtype=np.float64)

        def resize_neighbor_indices(
            source: float, count: int, limit: int
        ) -> Tuple[np.ndarray, np.ndarray]:
            pad_width = int(np.ceil(count / 2))
            padded_source = source + pad_width
            padded_limit = limit + 2 * pad_width
            padded_indices = sorted(
                range(padded_limit), key=lambda idx: (abs(padded_source - idx), idx)
            )[:count]
            source_indices = (
                np.array(sorted(padded_indices), dtype=np.int64) - pad_width
            )
            data_indices = np.clip(source_indices, 0, limit - 1)
            return source_indices, data_indices

        def resize_interpolate(a: np.ndarray, idx: Tuple[int, ...]) -> Any:
            axis_coeffs: List[np.ndarray] = []
            axis_data_indices: List[np.ndarray] = []

            for axis, output_index in enumerate(idx):
                source = resize_source_coordinate(output_index, axis)
                if source is None:
                    return extrapolation_value

                source_floor = math.floor(source)
                ratio = 1 if float(source).is_integer() else source - source_floor
                scale = float(scales[axis])
                coeffs = (
                    resize_linear_coeffs(ratio, scale)
                    if mode == "linear"
                    else resize_cubic_coeffs(ratio, scale)
                )
                source_indices, data_indices = resize_neighbor_indices(
                    source, len(coeffs), a_shape[axis]
                )
                if exclude_outside:
                    coeffs = coeffs.copy()
                    outside = (source_indices < 0) | (source_indices >= a_shape[axis])
                    coeffs[outside] = 0
                    coeffs_sum = np.sum(coeffs)
                    if coeffs_sum != 0:
                        coeffs /= coeffs_sum
                axis_coeffs.append(coeffs)
                axis_data_indices.append(data_indices)

            value = 0.0
            for neighbor_offsets in np.ndindex(
                *(len(coeffs) for coeffs in axis_coeffs)
            ):
                weight = 1.0
                source_idx = []
                for axis, neighbor_offset in enumerate(neighbor_offsets):
                    weight *= axis_coeffs[axis][neighbor_offset]
                    source_idx.append(axis_data_indices[axis][neighbor_offset])
                value += weight * a[tuple(source_idx)]
            return value

        @ggml.ggml_custom2_op_t
        def custom_resize(
            tensor_out: ggml.ggml_tensor_p,
            tensor_in_1: ggml.ggml_tensor_p,
            tensor_in_2: ggml.ggml_tensor_p,
            ith: int,
            nth: int,
            userdata: Optional[ctypes.c_void_p],
        ):
            a = ctx.to_numpy(tensor_in_2).reshape(a_shape)
            y = np.empty(output_shape, dtype=a_dtype)

            for idx in np.ndindex(*output_shape):
                if mode == "nearest":
                    source_idx = tuple(
                        resize_source_index(output_index, axis)
                        for axis, output_index in enumerate(idx)
                    )
                    y[idx] = a[source_idx]
                else:
                    y[idx] = resize_interpolate(a, idx)
            ctx.set_tensor_data(tensor_out, y)

        new_tensor = ctx.ggml_tensors_dict[node.output[0]] = (
            ggml.ggml_map_custom2_inplace(
                ctx.ggml_eval_context,
                x_t,
                a,
                custom_resize,
                1,
                None,
            )
        )
        ctx.refs.append(custom_resize)
        ctx.set_tensor_shape(new_tensor, output_shape)
        ctx.shapes[node.output[0]] = output_shape
        ctx.ggml_tensors_dict[node.output[0]] = new_tensor


@onnx_operators.register
class UpsampleOperator(OnnxOperator):
    def __init__(self):
        super().__init__("Upsample")

    def lower(self, ctx: "GgmlOnnxExecutionContext", node: NodeProto) -> None:
        node_inputs = [ctx.ggml_tensors_dict[inp] for inp in node.input]

        if len(node_inputs) != 2:
            raise ValueError(
                f'Error for node "{node.name}": Operation "Upsample" requires exactly two inputs. Actual number of inputs: {len(node_inputs)}'
            )

        x, scales_tensor = node_inputs
        x_shape = ctx.shapes[node.input[0]]
        x_dtype = get_tensor_dtype(x)
        mode = next(
            (attr.s.decode("utf-8") for attr in node.attribute if attr.name == "mode"),
            "nearest",
        )
        if mode != "nearest":
            raise NotImplementedError(f"Upsample mode={mode} is not supported")

        scales = ctx.to_numpy(ctx.eval_tensor(scales_tensor)).reshape(
            ctx.shapes[node.input[1]]
        )
        output_shape = tuple(
            int(dim) for dim in np.floor(np.asarray(x_shape) * scales).astype(np.int64)
        )
        x_t = ctx.from_numpy(np.empty(output_shape, dtype=x_dtype))

        @ggml.ggml_custom2_op_t
        def custom_upsample(
            tensor_out: ggml.ggml_tensor_p,
            tensor_in_1: ggml.ggml_tensor_p,
            tensor_in_2: ggml.ggml_tensor_p,
            ith: int,
            nth: int,
            userdata: Optional[ctypes.c_void_p],
        ):
            x_array = ctx.to_numpy(tensor_in_2).reshape(x_shape)
            y = np.empty(output_shape, dtype=x_dtype)
            for idx in np.ndindex(*output_shape):
                source_idx = tuple(
                    min(
                        max(int(math.floor(output_index / scales[axis])), 0),
                        x_shape[axis] - 1,
                    )
                    for axis, output_index in enumerate(idx)
                )
                y[idx] = x_array[source_idx]
            ctx.set_tensor_data(tensor_out, y)

        new_tensor = ctx.ggml_tensors_dict[node.output[0]] = (
            ggml.ggml_map_custom2_inplace(
                ctx.ggml_eval_context,
                x_t,
                x,
                custom_upsample,
                1,
                None,
            )
        )
        ctx.refs.append(custom_upsample)
        ctx.set_tensor_shape(new_tensor, output_shape)
        ctx.shapes[node.output[0]] = output_shape


@onnx_operators.register
class ReverseSequenceOperator(OnnxOperator):
    def __init__(self):
        super().__init__("ReverseSequence")

    def lower(self, ctx: "GgmlOnnxExecutionContext", node: NodeProto) -> None:
        node_inputs = [ctx.ggml_tensors_dict[inp] for inp in node.input]

        if len(node_inputs) != 2:
            raise ValueError(
                f'Error for node "{node.name}": Operation "ReverseSequence" requires exactly two inputs. Actual number of inputs: {len(node_inputs)}'
            )

        x_shape = ctx.shapes[node.input[0]]
        sequence_lens_shape = ctx.shapes[node.input[1]]
        time_axis = next(
            (attr.i for attr in node.attribute if attr.name == "time_axis"), 0
        )
        batch_axis = next(
            (attr.i for attr in node.attribute if attr.name == "batch_axis"), 1
        )
        if time_axis < 0:
            time_axis += len(x_shape)
        if batch_axis < 0:
            batch_axis += len(x_shape)

        x = ctx.to_numpy(ctx.eval_tensor(node_inputs[0])).reshape(x_shape)
        sequence_lens = ctx.to_numpy(ctx.eval_tensor(node_inputs[1])).reshape(
            sequence_lens_shape
        )
        output = x.copy()

        for batch_index in range(x_shape[batch_axis]):
            sequence_length = int(sequence_lens[batch_index])
            for time_index in range(sequence_length):
                src_index = [slice(None)] * len(x_shape)
                dst_index = [slice(None)] * len(x_shape)
                src_index[batch_axis] = batch_index
                dst_index[batch_axis] = batch_index
                src_index[time_axis] = sequence_length - 1 - time_index
                dst_index[time_axis] = time_index
                output[tuple(dst_index)] = x[tuple(src_index)]

        new_tensor = ctx.ggml_tensors_dict[node.output[0]] = ctx.from_numpy(output)
        ctx.set_tensor_shape(new_tensor, output.shape)
        ctx.shapes[node.output[0]] = output.shape
        ctx.set_tensor_dtype(node.output[0], output.dtype)


@onnx_operators.register
class RoiAlignOperator(OnnxOperator):
    def __init__(self):
        super().__init__("RoiAlign")

    def lower(self, ctx: "GgmlOnnxExecutionContext", node: NodeProto) -> None:
        node_inputs = [ctx.ggml_tensors_dict[inp] for inp in node.input]

        if len(node_inputs) != 3:
            raise ValueError(
                f'Error for node "{node.name}": Operation "RoiAlign" requires exactly three inputs. Actual number of inputs: {len(node_inputs)}'
            )

        x_shape = ctx.shapes[node.input[0]]
        rois_shape = ctx.shapes[node.input[1]]
        batch_indices_shape = ctx.shapes[node.input[2]]
        x = ctx.to_numpy(ctx.eval_tensor(node_inputs[0])).reshape(x_shape)
        rois = ctx.to_numpy(ctx.eval_tensor(node_inputs[1])).reshape(rois_shape)
        batch_indices = ctx.to_numpy(ctx.eval_tensor(node_inputs[2])).reshape(
            batch_indices_shape
        )

        mode = next((attr.s for attr in node.attribute if attr.name == "mode"), b"avg")
        spatial_scale = next(
            (attr.f for attr in node.attribute if attr.name == "spatial_scale"), 1.0
        )
        output_height = next(
            (attr.i for attr in node.attribute if attr.name == "output_height"), 1
        )
        output_width = next(
            (attr.i for attr in node.attribute if attr.name == "output_width"), 1
        )
        sampling_ratio = next(
            (attr.i for attr in node.attribute if attr.name == "sampling_ratio"), 0
        )
        coordinate_transformation_mode = next(
            (
                attr.s
                for attr in node.attribute
                if attr.name == "coordinate_transformation_mode"
            ),
            b"half_pixel",
        )

        if mode not in {b"avg", b"max"}:
            raise NotImplementedError(
                f'Error for node "{node.name}": RoiAlign mode {mode!r} is not implemented.'
            )
        if coordinate_transformation_mode not in {b"half_pixel", b"output_half_pixel"}:
            raise ValueError(
                f'Error for node "{node.name}": unknown RoiAlign coordinate_transformation_mode {coordinate_transformation_mode!r}.'
            )

        _, channels, height, width = x_shape
        output = np.empty(
            (rois_shape[0], channels, output_height, output_width),
            dtype=get_tensor_dtype(node_inputs[0]),
        )

        def bilinear_sample_contributions(
            batch: int,
            channel: int,
            y: float,
            x_coordinate: float,
        ) -> Tuple[float, float, float, float]:
            if y < -1.0 or y > height or x_coordinate < -1.0 or x_coordinate > width:
                return 0.0, 0.0, 0.0, 0.0

            y = min(max(y, 0.0), height - 1.0)
            x_coordinate = min(max(x_coordinate, 0.0), width - 1.0)
            y_low = int(math.floor(y))
            x_low = int(math.floor(x_coordinate))
            y_high = min(y_low + 1, height - 1)
            x_high = min(x_low + 1, width - 1)

            y_lerp = y - y_low
            x_lerp = x_coordinate - x_low
            top_left = x[batch, channel, y_low, x_low]
            top_right = x[batch, channel, y_low, x_high]
            bottom_left = x[batch, channel, y_high, x_low]
            bottom_right = x[batch, channel, y_high, x_high]

            y_weight_high = 1 - y_lerp
            x_weight_high = 1 - x_lerp
            return (
                float(top_left * y_weight_high * x_weight_high),
                float(top_right * y_weight_high * x_lerp),
                float(bottom_left * y_lerp * x_weight_high),
                float(bottom_right * y_lerp * x_lerp),
            )

        for roi_index, roi in enumerate(rois):
            offset = 0.5 if coordinate_transformation_mode == b"half_pixel" else 0.0
            roi_start_w = roi[0] * spatial_scale - offset
            roi_start_h = roi[1] * spatial_scale - offset
            roi_end_w = roi[2] * spatial_scale - offset
            roi_end_h = roi[3] * spatial_scale - offset

            roi_width = roi_end_w - roi_start_w
            roi_height = roi_end_h - roi_start_h
            if coordinate_transformation_mode == b"output_half_pixel":
                roi_width = max(roi_width, 1.0)
                roi_height = max(roi_height, 1.0)

            bin_width = roi_width / output_width
            bin_height = roi_height / output_height
            sample_count_w = (
                sampling_ratio
                if sampling_ratio > 0
                else max(int(math.ceil(roi_width / output_width)), 1)
            )
            sample_count_h = (
                sampling_ratio
                if sampling_ratio > 0
                else max(int(math.ceil(roi_height / output_height)), 1)
            )
            sample_count = sample_count_h * sample_count_w
            batch_index = int(batch_indices[roi_index])

            for channel in range(channels):
                for out_y in range(output_height):
                    for out_x in range(output_width):
                        result = 0.0
                        max_result = None

                        for sample_y in range(sample_count_h):
                            y = (
                                roi_start_h
                                + (out_y + (sample_y + 0.5) / sample_count_h)
                                * bin_height
                            )
                            for sample_x in range(sample_count_w):
                                x_coordinate = (
                                    roi_start_w
                                    + (out_x + (sample_x + 0.5) / sample_count_w)
                                    * bin_width
                                )
                                sample_contributions = bilinear_sample_contributions(
                                    batch_index, channel, y, x_coordinate
                                )
                                if mode == b"avg":
                                    result += sum(sample_contributions)
                                else:
                                    sample_max = max(sample_contributions)
                                    max_result = (
                                        sample_max
                                        if max_result is None
                                        else max(max_result, sample_max)
                                    )

                        if mode == b"avg":
                            result /= sample_count
                        else:
                            result = 0.0 if max_result is None else max_result
                        output[roi_index, channel, out_y, out_x] = result

        new_tensor = ctx.ggml_tensors_dict[node.output[0]] = ctx.from_numpy(output)
        ctx.set_tensor_shape(new_tensor, output.shape)
        ctx.shapes[node.output[0]] = output.shape
        ctx.set_tensor_dtype(node.output[0], output.dtype)


@onnx_operators.register
class RotaryEmbeddingOperator(OnnxOperator):
    def __init__(self):
        super().__init__(
            "RotaryEmbedding",
            domains=("", "com.microsoft", "com.microsoft.nchwc"),
        )
        self.has_numpy_evaluator = True

    @staticmethod
    def rotate_half(
        value: npt.NDArray[Any],
        cos_cache: npt.NDArray[Any],
        sin_cache: npt.NDArray[Any],
        interleaved: bool,
    ) -> npt.NDArray[Any]:
        if interleaved:
            even = value[..., 0::2]
            odd = value[..., 1::2]
            rotated_even = even * cos_cache - odd * sin_cache
            rotated_odd = even * sin_cache + odd * cos_cache
            output = np.empty_like(value)
            output[..., 0::2] = rotated_even
            output[..., 1::2] = rotated_odd
            return output

        half = value.shape[-1] // 2
        first = value[..., :half]
        second = value[..., half:]
        return np.concatenate(
            (
                first * cos_cache - second * sin_cache,
                first * sin_cache + second * cos_cache,
            ),
            axis=-1,
        )

    @staticmethod
    def normalized_caches(
        cos_cache: npt.NDArray[Any],
        sin_cache: npt.NDArray[Any],
        position_ids: Optional[npt.NDArray[Any]],
        batch_size: int,
        sequence_length: int,
        rotary_half: int,
    ) -> Tuple[npt.NDArray[Any], npt.NDArray[Any]]:
        if position_ids is not None:
            cos_cache = cos_cache[np.asarray(position_ids, dtype=np.int64)]
            sin_cache = sin_cache[np.asarray(position_ids, dtype=np.int64)]
        if cos_cache.ndim == 2:
            cos_cache = cos_cache[:sequence_length].reshape(1, sequence_length, 1, -1)
            sin_cache = sin_cache[:sequence_length].reshape(1, sequence_length, 1, -1)
        elif cos_cache.ndim == 3:
            cos_cache = cos_cache.reshape(batch_size, sequence_length, 1, -1)
            sin_cache = sin_cache.reshape(batch_size, sequence_length, 1, -1)
        elif cos_cache.ndim == 4:
            pass
        else:
            raise ValueError("RotaryEmbedding cos/sin caches must have rank 2, 3, or 4")
        return cos_cache[..., :rotary_half], sin_cache[..., :rotary_half]

    def eval_numpy(
        self, node: NodeProto, inputs: Tuple[npt.NDArray[Any], ...]
    ) -> Tuple[npt.NDArray[Any], ...]:
        if len(inputs) not in {3, 4}:
            raise ValueError(
                f'Operation "{node.op_type}" requires three or four inputs'
            )
        original = inputs[0]
        x = original
        original_rank = x.ndim
        if x.ndim == 4:
            x = x.transpose(0, 2, 1, 3)
        elif x.ndim == 3:
            num_heads = self.int_attribute(node, "num_heads", 0)
            if num_heads <= 0:
                raise ValueError("RotaryEmbedding rank-3 input requires num_heads")
            batch_size, sequence_length, hidden_size = x.shape
            if hidden_size % num_heads != 0:
                raise ValueError("RotaryEmbedding hidden size must divide num_heads")
            x = x.reshape(
                batch_size, sequence_length, num_heads, hidden_size // num_heads
            )
        else:
            raise ValueError("RotaryEmbedding expects rank-3 or rank-4 input")

        batch_size, sequence_length, _num_heads, head_size = x.shape
        rotary_dim = self.int_attribute(node, "rotary_embedding_dim", 0) or head_size
        interleaved = bool(self.int_attribute(node, "interleaved", 0))
        if rotary_dim > head_size or rotary_dim % 2:
            raise ValueError(
                "RotaryEmbedding rotary dimension must be even and <= head size"
            )
        position_ids = inputs[3] if len(inputs) == 4 else None
        cos_cache, sin_cache = self.normalized_caches(
            inputs[1],
            inputs[2],
            position_ids,
            batch_size,
            sequence_length,
            rotary_dim // 2,
        )
        rotated = self.rotate_half(
            x[..., :rotary_dim],
            cos_cache,
            sin_cache,
            interleaved,
        )
        if rotary_dim < head_size:
            x = np.concatenate((rotated, x[..., rotary_dim:]), axis=-1)
        else:
            x = rotated
        if original_rank == 4:
            x = x.transpose(0, 2, 1, 3)
        else:
            x = x.reshape(original.shape)
        return (np.asarray(x, dtype=original.dtype),)

    def lower(self, ctx: "GgmlOnnxExecutionContext", node: NodeProto) -> None:
        arrays = tuple(
            ctx.logical_tensor_eval_data(
                name, ctx.ggml_tensors_dict[name], ctx.shapes[name]
            )
            for name in node.input
            if name
        )
        output = self.eval_numpy(node, arrays)[0]
        ctx.set_numpy_runtime_output(node.output[0], output, output.dtype)


@onnx_operators.register
class RoundOperator(OnnxOperator):
    def __init__(self):
        super().__init__("Round")

    def lower(self, ctx: "GgmlOnnxExecutionContext", node: NodeProto) -> None:
        self.lower_numpy_unary(ctx, node, np.round)


@onnx_operators.register
class ScatterOperator(OnnxOperator):
    def __init__(self):
        super().__init__("Scatter")

    def lower(self, ctx: "GgmlOnnxExecutionContext", node: NodeProto) -> None:
        self.lower_scatter_elements_like(ctx, node)


@onnx_operators.register
class ScatterElementsOperator(OnnxOperator):
    def __init__(self):
        super().__init__("ScatterElements")

    def lower(self, ctx: "GgmlOnnxExecutionContext", node: NodeProto) -> None:
        reduction = next(
            (
                attr.s.decode("utf-8")
                for attr in node.attribute
                if attr.name == "reduction"
            ),
            "none",
        )
        self.lower_scatter_elements_like(ctx, node, reduction)


@onnx_operators.register
class ScatterNDOperator(OnnxOperator):
    def __init__(self):
        super().__init__("ScatterND")

    def lower(self, ctx: "GgmlOnnxExecutionContext", node: NodeProto) -> None:
        node_inputs = [ctx.ggml_tensors_dict[inp] for inp in node.input]

        if len(node_inputs) != 3:
            raise ValueError(
                f'Error for node "{node.name}": Operation "ScatterND" requires exactly three inputs. Actual number of inputs: {len(node_inputs)}'
            )

        data_shape = ctx.shapes[node.input[0]]
        indices_shape = ctx.shapes[node.input[1]]
        updates_shape = ctx.shapes[node.input[2]]
        reduction = next(
            (
                attr.s.decode("utf-8")
                for attr in node.attribute
                if attr.name == "reduction"
            ),
            "none",
        )

        output = (
            ctx.to_numpy(ctx.eval_tensor(node_inputs[0])).reshape(data_shape).copy()
        )
        indices = ctx.to_numpy(ctx.eval_tensor(node_inputs[1])).reshape(indices_shape)
        updates = ctx.to_numpy(ctx.eval_tensor(node_inputs[2])).reshape(updates_shape)

        for source_index in np.ndindex(*indices_shape[:-1]):
            target_index = []
            for axis, target_axis_index in enumerate(indices[source_index]):
                target_axis_index = int(target_axis_index)
                if target_axis_index < 0:
                    target_axis_index += data_shape[axis]
                target_index.append(target_axis_index)
            target_index = tuple(target_index)
            update = updates[source_index]
            if reduction in {"none", ""}:
                output[target_index] = update
            elif reduction == "add":
                output[target_index] += update
            elif reduction == "mul":
                output[target_index] *= update
            elif reduction == "max":
                output[target_index] = np.maximum(output[target_index], update)
            elif reduction == "min":
                output[target_index] = np.minimum(output[target_index], update)
            else:
                raise NotImplementedError(
                    f'Scatter reduction "{reduction}" is not supported'
                )

        new_tensor = ctx.ggml_tensors_dict[node.output[0]] = ctx.from_numpy(output)
        ctx.set_tensor_shape(new_tensor, output.shape)
        ctx.shapes[node.output[0]] = output.shape
        ctx.set_tensor_dtype(node.output[0], output.dtype)


@onnx_operators.register
class SeluOperator(OnnxOperator):
    def __init__(self):
        super().__init__("Selu")

    def lower(self, ctx: "GgmlOnnxExecutionContext", node: NodeProto) -> None:
        node_inputs = [ctx.ggml_tensors_dict[inp] for inp in node.input]

        if len(node_inputs) != 1:
            raise ValueError(
                f'Error for node "{node.name}": Operation "Selu" requires exactly one input. Actual number of inputs: {len(node_inputs)}'
            )

        x = node_inputs[0]

        alpha = next(
            (attr.f for attr in node.attribute if attr.name == "alpha"),
            1.67326319217681884765625,
        )
        gamma = next(
            (attr.f for attr in node.attribute if attr.name == "gamma"),
            1.05070102214813232421875,
        )

        selu_userdata = SeluUserData(alpha, gamma)
        userdata_p = ctypes.cast(ctypes.pointer(selu_userdata), ctypes.c_void_p)

        @ggml.ggml_custom1_op_t
        def custom_selu(
            tensor_out: ggml.ggml_tensor_p,
            tensor_in_1: ggml.ggml_tensor_p,
            ith: int,
            nth: int,
            userdata: Optional[ctypes.c_void_p],
        ):
            userdata_data_ptr = ctypes.cast(userdata, ctypes.POINTER(SeluUserData))
            userdata_data = userdata_data_ptr.contents
            x = ctx.to_numpy(tensor_in_1)

            alpha = userdata_data.alpha
            gamma = userdata_data.gamma

            y = (
                np.clip(x, 0, np.inf) * gamma
                + (np.exp(np.clip(x, -np.inf, 0)) - 1) * alpha * gamma
            )

            ctx.set_tensor_data(tensor_out, y)

        new_tensor = ctx.ggml_tensors_dict[node.output[0]] = (
            ggml.ggml_map_custom1_inplace(
                ctx.ggml_eval_context,
                x,
                custom_selu,
                1,
                userdata_p,
            )
        )

        ctx.refs.append(custom_selu)

        ctx.refs.append(selu_userdata)
        ctx.set_tensor_shape(new_tensor, ctx.shapes[node.input[0]])
        ctx.shapes[node.output[0]] = ctx.shapes[node.input[0]]


@onnx_operators.register
class ShapeOperator(OnnxOperator):
    def __init__(self):
        super().__init__("Shape")

    def lower(self, ctx: "GgmlOnnxExecutionContext", node: NodeProto) -> None:
        node_inputs = [ctx.ggml_tensors_dict[inp] for inp in node.input]

        if len(node_inputs) != 1:
            raise ValueError(
                f'Error for node "{node.name}": Operation "Shape" requires exactly one input. Actual number of inputs: {len(node_inputs)}'
            )

        tensor_shape = np.array(ctx.shapes[node.input[0]], dtype=np.int64)
        name = node.output[0]
        start = next((attr.i for attr in node.attribute if attr.name == "start"), None)
        end = next(
            (attr.i for attr in node.attribute if attr.name == "end"),
            None,
        )
        shape_slice = tensor_shape[start:end]
        ctx.set_logical_output(name, shape_slice, np.int64)


@onnx_operators.register
class SigmoidOperator(OnnxOperator):
    def __init__(self):
        super().__init__(
            "Sigmoid",
            OnnxOperator.EXECUTION_NATIVE_OR_NUMPY_RUNTIME,
            OnnxOperator.CLASS_CONDITIONAL_NATIVE,
        )
        self.has_numpy_evaluator = True

    def strategies(
        self, tensor_types: Dict[str, TensorType], node: "NodeIR"
    ) -> Tuple[Tuple[str, str, str], ...]:
        if (
            len(node.inputs) == 1
            and self.tensor_type(tensor_types, node.inputs[0]).is_float32
        ):
            return self.native_strategy()
        return self.numpy_runtime_strategy()

    def eval_numpy(
        self, node: NodeProto, inputs: Tuple[npt.NDArray[Any], ...]
    ) -> Tuple[npt.NDArray[Any], ...]:
        return self.eval_numpy_unary_operator(
            node, inputs, lambda x: 1.0 / (1.0 + np.exp(-x))
        )

    def lower(self, ctx: "GgmlOnnxExecutionContext", node: NodeProto) -> None:
        self.lower_native_unary_or_numpy(
            ctx, node, ggml.ggml_sigmoid, lambda x: 1.0 / (1.0 + np.exp(-x))
        )


@onnx_operators.register
class SignOperator(OnnxOperator):
    def __init__(self):
        super().__init__(
            "Sign",
            OnnxOperator.EXECUTION_NATIVE_OR_NUMPY_RUNTIME,
            OnnxOperator.CLASS_CONDITIONAL_NATIVE,
        )
        self.has_numpy_evaluator = True

    def strategies(
        self, tensor_types: Dict[str, TensorType], node: "NodeIR"
    ) -> Tuple[Tuple[str, str, str], ...]:
        if (
            len(node.inputs) == 1
            and self.tensor_type(tensor_types, node.inputs[0]).is_float32
        ):
            return self.native_strategy()
        return self.numpy_runtime_strategy()

    def eval_numpy(
        self, node: NodeProto, inputs: Tuple[npt.NDArray[Any], ...]
    ) -> Tuple[npt.NDArray[Any], ...]:
        return self.eval_numpy_unary_operator(node, inputs, np.sign)

    def lower(self, ctx: "GgmlOnnxExecutionContext", node: NodeProto) -> None:
        self.lower_native_unary_or_numpy(ctx, node, ggml.ggml_sgn, np.sign)


@onnx_operators.register
class ShrinkOperator(OnnxOperator):
    def __init__(self):
        super().__init__("Shrink")

    def lower(self, ctx: "GgmlOnnxExecutionContext", node: NodeProto) -> None:
        lambd = next((attr.f for attr in node.attribute if attr.name == "lambd"), 0.5)
        bias = next((attr.f for attr in node.attribute if attr.name == "bias"), 0.0)

        def shrink(x: npt.NDArray[Any]) -> npt.NDArray[Any]:
            return np.where(x < -lambd, x + bias, np.where(x > lambd, x - bias, 0))

        self.lower_numpy_unary(ctx, node, shrink)


@onnx_operators.register
class SinOperator(OnnxOperator):
    def __init__(self):
        super().__init__("Sin")

    def lower(self, ctx: "GgmlOnnxExecutionContext", node: NodeProto) -> None:
        self.lower_numpy_unary(ctx, node, np.sin)


@onnx_operators.register
class SinhOperator(OnnxOperator):
    def __init__(self):
        super().__init__("Sinh")

    def lower(self, ctx: "GgmlOnnxExecutionContext", node: NodeProto) -> None:
        self.lower_numpy_unary(ctx, node, np.sinh)


@onnx_operators.register
class SizeOperator(OnnxOperator):
    def __init__(self):
        super().__init__("Size")

    def lower(self, ctx: "GgmlOnnxExecutionContext", node: NodeProto) -> None:
        node_inputs = [ctx.ggml_tensors_dict[inp] for inp in node.input]

        if len(node_inputs) != 1:
            raise ValueError(
                f'Error for node "{node.name}": Operation "Size" requires exactly one input. Actual number of inputs: {len(node_inputs)}'
            )

        tensor_shape = np.array(ctx.shapes[node.input[0]], dtype=np.int64)
        name = node.output[0]
        tensor_size_np = np.prod(tensor_shape).astype(np.int64)
        ctx.set_logical_output(name, np.array(tensor_size_np), np.int64)


@onnx_operators.register
class SliceOperator(OnnxOperator):
    def __init__(self):
        super().__init__(
            "Slice",
            OnnxOperator.EXECUTION_NATIVE_OR_NUMPY_RUNTIME,
            OnnxOperator.CLASS_CONDITIONAL_NATIVE_VIEW,
            view_kind=ViewTransformSemantics.KIND_LAYOUT,
        )
        self.has_numpy_evaluator = True

    @staticmethod
    def normalize_slice_axis(axis: int, rank: int) -> int:
        axis = axis + rank if axis < 0 else axis
        if axis < 0 or axis >= rank:
            raise ValueError(f"Slice axis {axis} is out of bounds for rank {rank}")
        return axis

    @staticmethod
    def normalize_slice_bound(value: int, dim: int) -> int:
        value = value + dim if value < 0 else value
        return min(max(value, 0), dim)

    def slice_parameters_from_values(
        self,
        input_shape: Tuple[int, ...],
        starts: Sequence[int],
        ends: Sequence[int],
        axes: Optional[Sequence[int]],
        steps: Optional[Sequence[int]],
    ) -> Optional[Tuple[Tuple[slice, ...], Tuple[int, ...]]]:
        rank = len(input_shape)
        axes_values = (
            tuple(range(len(starts)))
            if axes is None
            else tuple(self.normalize_slice_axis(int(axis), rank) for axis in axes)
        )
        steps_values = (
            (1,) * len(starts) if steps is None else tuple(int(v) for v in steps)
        )
        if len(starts) != len(ends) or len(starts) != len(axes_values):
            return None
        if len(starts) != len(steps_values):
            return None

        slices = [slice(None)] * rank
        output_shape = list(input_shape)
        for raw_start, raw_end, axis, step in zip(
            starts, ends, axes_values, steps_values
        ):
            if step != 1:
                return None
            start = self.normalize_slice_bound(int(raw_start), input_shape[axis])
            end = self.normalize_slice_bound(int(raw_end), input_shape[axis])
            if end < start:
                return None
            slices[axis] = slice(start, end, step)
            output_shape[axis] = end - start
        return tuple(slices), tuple(output_shape)

    def static_slice_parameters(
        self, tensor_types: Dict[str, TensorType], node: "NodeIR"
    ) -> Optional[Tuple[Tuple[slice, ...], Tuple[int, ...]]]:
        if len(node.inputs) < 1:
            return None
        input_shape = self.tensor_type(tensor_types, node.inputs[0]).shape
        if input_shape is None:
            return None
        starts = node.attribute("starts")
        ends = node.attribute("ends")
        axes = node.attribute("axes")
        steps = node.attribute("steps")
        if starts is None and len(node.inputs) >= 3:
            starts = self.constant_int_values(tensor_types, node.inputs[1])
            ends = self.constant_int_values(tensor_types, node.inputs[2])
            if len(node.inputs) >= 4 and node.inputs[3]:
                axes = self.constant_int_values(tensor_types, node.inputs[3])
            if len(node.inputs) >= 5 and node.inputs[4]:
                steps = self.constant_int_values(tensor_types, node.inputs[4])
        if starts is None or ends is None:
            return None
        return self.slice_parameters_from_values(input_shape, starts, ends, axes, steps)

    def strategies(
        self, tensor_types: Dict[str, TensorType], node: "NodeIR"
    ) -> Tuple[Tuple[str, str, str], ...]:
        parameters = self.static_slice_parameters(tensor_types, node)
        input_type = (
            self.tensor_type(tensor_types, node.inputs[0]) if node.inputs else None
        )
        output_type = (
            self.tensor_type(tensor_types, node.outputs[0]) if node.outputs else None
        )
        if (
            parameters is not None
            and input_type is not None
            and input_type.is_float32
            and input_type.shape is not None
            and len(input_type.shape) <= ViewTransformSemantics.GGML_MAX_DIMS
            and output_type is not None
            and output_type.shape is not None
            and not any(dim == 0 for dim in output_type.shape)
        ):
            return self.native_view_strategy()
        return self.numpy_runtime_strategy(
            "Slice requires static float32 non-empty step-1 slices with rank <= 4 "
            "to lower as a native ggml view"
        )

    def runtime_slice_parameters(
        self,
        ctx: "GgmlOnnxExecutionContext",
        node: NodeProto,
        node_inputs: Sequence[Optional[ggml.ggml_tensor_p]],
    ) -> Tuple[Tuple[slice, ...], Tuple[int, ...]]:
        input_shape = tuple(ctx.shapes[node.input[0]])
        if len(node_inputs) >= 3:
            starts_tensor = node_inputs[1]
            ends_tensor = node_inputs[2]
            if starts_tensor is None or ends_tensor is None:
                raise ValueError(
                    f'Error for node "{node.name}": Slice requires starts and ends inputs.'
                )
            starts = tuple(
                int(v)
                for v in ctx.logical_tensor_eval_data(
                    node.input[1], starts_tensor, ctx.shapes[node.input[1]]
                ).flatten()
            )
            ends = tuple(
                int(v)
                for v in ctx.logical_tensor_eval_data(
                    node.input[2], ends_tensor, ctx.shapes[node.input[2]]
                ).flatten()
            )
            axes = None
            if len(node_inputs) >= 4 and node_inputs[3] is not None:
                axes = tuple(
                    int(v)
                    for v in ctx.logical_tensor_eval_data(
                        node.input[3], node_inputs[3], ctx.shapes[node.input[3]]
                    ).flatten()
                )
            steps = None
            if len(node_inputs) >= 5 and node_inputs[4] is not None:
                steps = tuple(
                    int(v)
                    for v in ctx.logical_tensor_eval_data(
                        node.input[4], node_inputs[4], ctx.shapes[node.input[4]]
                    ).flatten()
                )
        else:
            starts = tuple(self.ints_attribute(node, "starts") or ())
            ends = tuple(self.ints_attribute(node, "ends") or ())
            axes = self.ints_attribute(node, "axes")
            steps = self.ints_attribute(node, "steps")
        parameters = self.slice_parameters_from_values(
            input_shape, starts, ends, axes, steps
        )
        if parameters is None:
            raise ValueError(
                f'Error for node "{node.name}": Slice parameters cannot lower as a native view'
            )
        return parameters

    @staticmethod
    def native_slice_view(
        ctx: "GgmlOnnxExecutionContext",
        input_tensor: ggml.ggml_tensor_p,
        output_shape: Tuple[int, ...],
        slices: Tuple[slice, ...],
    ) -> ggml.ggml_tensor_p:
        storage_shape = tuple(reversed(output_shape))
        strides = tuple(input_tensor.contents.nb[: len(storage_shape)])
        offset = 0
        rank = len(output_shape)
        for logical_axis, slice_value in enumerate(slices):
            storage_axis = rank - logical_axis - 1
            offset += int(slice_value.start or 0) * int(strides[storage_axis])

        if len(storage_shape) == 1:
            return ggml.ggml_view_1d(
                ctx.ggml_eval_context,
                input_tensor,
                storage_shape[0],
                offset,
            )
        if len(storage_shape) == 2:
            return ggml.ggml_view_2d(
                ctx.ggml_eval_context,
                input_tensor,
                storage_shape[0],
                storage_shape[1],
                strides[1],
                offset,
            )
        if len(storage_shape) == 3:
            return ggml.ggml_view_3d(
                ctx.ggml_eval_context,
                input_tensor,
                storage_shape[0],
                storage_shape[1],
                storage_shape[2],
                strides[1],
                strides[2],
                offset,
            )
        if len(storage_shape) == 4:
            return ggml.ggml_view_4d(
                ctx.ggml_eval_context,
                input_tensor,
                storage_shape[0],
                storage_shape[1],
                storage_shape[2],
                storage_shape[3],
                strides[1],
                strides[2],
                strides[3],
                offset,
            )
        raise ValueError("Slice native view supports rank 1 through 4")

    def eval_numpy(
        self, node: NodeProto, inputs: Tuple[npt.NDArray[Any], ...]
    ) -> Tuple[npt.NDArray[Any], ...]:
        if len(inputs) < 1:
            raise ValueError(f'Operation "{node.op_type}" requires data input')
        input_iter = iter(inputs)
        positional_inputs: List[Optional[npt.NDArray[Any]]] = []
        for input_name in node.input:
            positional_inputs.append(next(input_iter) if input_name else None)
        data = positional_inputs[0]
        if data is None:
            raise ValueError(f'Operation "{node.op_type}" requires data input')
        if len(positional_inputs) >= 3:
            starts_input = positional_inputs[1]
            ends_input = positional_inputs[2]
            if starts_input is None or ends_input is None:
                raise ValueError(f'Operation "{node.op_type}" requires starts and ends')
            starts = tuple(int(v) for v in np.asarray(starts_input).flatten())
            ends = tuple(int(v) for v in np.asarray(ends_input).flatten())
            axes = (
                tuple(int(v) for v in np.asarray(positional_inputs[3]).flatten())
                if len(positional_inputs) >= 4 and positional_inputs[3] is not None
                else None
            )
            steps = (
                tuple(int(v) for v in np.asarray(positional_inputs[4]).flatten())
                if len(positional_inputs) >= 5 and positional_inputs[4] is not None
                else None
            )
        else:
            starts = tuple(self.ints_attribute(node, "starts") or ())
            ends = tuple(self.ints_attribute(node, "ends") or ())
            axes = self.ints_attribute(node, "axes")
            steps = self.ints_attribute(node, "steps")
        rank = data.ndim
        axes_values = (
            tuple(range(len(starts)))
            if axes is None
            else tuple(self.normalize_slice_axis(int(axis), rank) for axis in axes)
        )
        steps_values = (
            (1,) * len(starts) if steps is None else tuple(int(v) for v in steps)
        )
        slices = [slice(None)] * rank
        for raw_start, raw_end, axis, step in zip(
            starts, ends, axes_values, steps_values
        ):
            dim = data.shape[axis]
            start = raw_start + dim if raw_start < 0 else raw_start
            end = raw_end + dim if raw_end < 0 else raw_end
            slices[axis] = slice(start, end, step)
        return (np.asarray(data[tuple(slices)]).copy(),)

    def lower(self, ctx: "GgmlOnnxExecutionContext", node: NodeProto) -> None:
        node_inputs = [
            ctx.ggml_tensors_dict[inp] if inp else None for inp in node.input
        ]
        data_tensor = node_inputs[0]
        if data_tensor is None:
            raise ValueError(
                f'Error for node "{node.name}": Operation "Slice" requires data input.'
            )

        a_shape = tuple(ctx.shapes[node.input[0]])
        a_dtype = np.dtype(ctx.get_tensor_dtype(node.input[0]))
        try:
            slices, output_shape = self.runtime_slice_parameters(ctx, node, node_inputs)
        except ValueError:
            data = ctx.logical_tensor_eval_data(node.input[0], data_tensor, a_shape)
            eval_inputs: Tuple[npt.NDArray[Any], ...] = (data,)
            for index, tensor in enumerate(node_inputs[1:], start=1):
                if tensor is not None:
                    eval_inputs = (
                        *eval_inputs,
                        ctx.logical_tensor_eval_data(
                            node.input[index], tensor, ctx.shapes[node.input[index]]
                        ),
                    )
            output = self.eval_numpy(node, eval_inputs)[0]
            ctx.set_numpy_runtime_output(node.output[0], output, output.dtype)
            return

        if (
            ctx.can_emit_native(node.output[0])
            and ctx.can_run_native(node)
            and ggml.ggml_is_contiguous(data_tensor)
            and a_dtype == np.dtype(np.float32)
            and len(a_shape) <= ViewTransformSemantics.GGML_MAX_DIMS
            and not any(dim == 0 for dim in output_shape)
        ):
            view = self.native_slice_view(ctx, data_tensor, output_shape, slices)
            ctx.register_native_view_tensor(
                node.output[0],
                view,
                output_shape,
                a_dtype,
                node.input[0],
            )
            return

        data = ctx.logical_tensor_eval_data(node.input[0], data_tensor, a_shape)
        output = np.asarray(data[tuple(slices)]).copy()
        ctx.set_numpy_runtime_output(node.output[0], output, output.dtype)


@onnx_operators.register
class SoftmaxOperator(OnnxOperator):
    def __init__(self):
        super().__init__(
            "Softmax",
            OnnxOperator.EXECUTION_NATIVE_OR_NUMPY_RUNTIME,
            OnnxOperator.CLASS_CONDITIONAL_NATIVE,
        )
        self.has_numpy_evaluator = True

    @staticmethod
    def softmax_axis(node: NodeProto, rank: int) -> int:
        axis = OnnxOperator.int_attribute(node, "axis", -1)
        return axis + rank if axis < 0 else axis

    def strategies(
        self, tensor_types: Dict[str, TensorType], node: "NodeIR"
    ) -> Tuple[Tuple[str, str, str], ...]:
        if len(node.inputs) == 1:
            input_type = self.tensor_type(tensor_types, node.inputs[0])
            if input_type.is_float32 and input_type.shape is not None:
                axis = int(node.attribute("axis", -1))
                axis = axis + len(input_type.shape) if axis < 0 else axis
                if axis == len(input_type.shape) - 1:
                    return self.native_strategy()
        return self.numpy_runtime_strategy(
            "Softmax requires float32 input and last-axis reduction to lower to ggml_soft_max"
        )

    def eval_numpy(
        self, node: NodeProto, inputs: Tuple[npt.NDArray[Any], ...]
    ) -> Tuple[npt.NDArray[Any], ...]:
        if len(inputs) != 1:
            raise ValueError(
                f'Operation "{node.op_type}" requires exactly one input. '
                f"Actual number of inputs: {len(inputs)}"
            )
        x = inputs[0]
        axis = self.softmax_axis(node, x.ndim)
        x = x - np.max(x, axis=axis, keepdims=True)
        exp_x = np.exp(x)
        return (
            np.asarray(
                exp_x / np.sum(exp_x, axis=axis, keepdims=True), dtype=inputs[0].dtype
            ),
        )

    def lower(self, ctx: "GgmlOnnxExecutionContext", node: NodeProto) -> None:
        node_inputs = [ctx.ggml_tensors_dict[inp] for inp in node.input]

        if len(node_inputs) != 1:
            raise ValueError(
                f'Error for node "{node.name}": Operation "Softmax" requires exactly one input. Actual number of inputs: {len(node_inputs)}'
            )

        output_name = node.output[0]
        a = node_inputs[0]
        input_shape = ctx.shapes[node.input[0]]
        input_dtype = np.dtype(ctx.get_tensor_dtype(node.input[0]))
        axis = self.softmax_axis(node, len(input_shape))

        if (
            ctx.can_emit_native(output_name)
            and ctx.can_run_native(node)
            and input_dtype == np.dtype(np.float32)
            and axis == len(input_shape) - 1
        ):
            softmax_input = a
            if not ggml.ggml_is_contiguous(softmax_input):
                softmax_input = ggml.ggml_cont(ctx.ggml_eval_context, softmax_input)
            result = ggml.ggml_soft_max(ctx.ggml_eval_context, softmax_input)
            ctx.register_native_tensor(output_name, result, input_shape, input_dtype)
            return

        input_array = ctx.logical_tensor_eval_data(node.input[0], a, input_shape)
        output = self.eval_numpy(node, (input_array,))[0]
        ctx.set_numpy_runtime_output(output_name, output, output.dtype)


@onnx_operators.register
class SoftmaxCrossEntropyLossOperator(OnnxOperator):
    def __init__(self):
        super().__init__("SoftmaxCrossEntropyLoss")

    @staticmethod
    def negative_log_likelihood_loss(
        input_data: npt.NDArray[Any],
        target: npt.NDArray[Any],
        weight: Optional[npt.NDArray[Any]],
        reduction: str,
        ignore_index: Optional[int],
    ) -> npt.NDArray[Any]:
        input_shape = input_data.shape
        target_shape = target.shape
        batch_size = input_shape[0]
        channel_count = input_shape[1]

        gather_weight = None
        if weight is not None:
            gather_weight = np.take(
                weight, np.asarray(target, dtype=np.int32), mode="clip"
            )
            if ignore_index is not None:
                gather_weight = np.where(
                    target == ignore_index, 0, gather_weight
                ).astype(np.float32)
        elif ignore_index is not None:
            gather_weight = np.where(target == ignore_index, 0, 1).astype(np.float32)

        if len(input_shape) != 3:
            input_data = input_data.reshape((batch_size, channel_count, -1))
            target = target.reshape((batch_size, -1))

        sample_count = input_data.shape[2]
        loss = np.zeros((batch_size, sample_count), dtype=np.float32)
        for batch_index in range(batch_size):
            for sample_index in range(sample_count):
                target_index = target[batch_index][sample_index]
                if target_index != ignore_index:
                    loss[batch_index][sample_index] = -input_data[batch_index][
                        target_index
                    ][sample_index]

        if len(input_shape) != 3:
            loss = loss.reshape(target_shape)

        if gather_weight is not None:
            loss = gather_weight * loss
            if reduction == "mean":
                return np.asarray(loss.sum() / gather_weight.sum(), dtype=np.float32)

        if reduction == "mean":
            return np.asarray(np.mean(loss), dtype=np.float32)
        if reduction == "sum":
            return np.asarray(np.sum(loss), dtype=np.float32)
        return loss.astype(np.float32)

    def lower(self, ctx: "GgmlOnnxExecutionContext", node: NodeProto) -> None:
        node_inputs = [ctx.ggml_tensors_dict[inp] for inp in node.input]

        if len(node_inputs) not in {2, 3}:
            raise ValueError(
                f'Error for node "{node.name}": Operation "SoftmaxCrossEntropyLoss" requires two or three inputs. Actual number of inputs: {len(node_inputs)}'
            )

        x_shape = ctx.shapes[node.input[0]]
        target_shape = ctx.shapes[node.input[1]]
        x = ctx.to_numpy(ctx.eval_tensor(node_inputs[0])).reshape(x_shape)
        target = ctx.to_numpy(ctx.eval_tensor(node_inputs[1])).reshape(target_shape)
        weight = None
        if len(node_inputs) == 3:
            weight = ctx.to_numpy(ctx.eval_tensor(node_inputs[2])).reshape(
                ctx.shapes[node.input[2]]
            )
        reduction = next(
            (
                attr.s.decode("utf-8")
                for attr in node.attribute
                if attr.name == "reduction"
            ),
            "mean",
        )
        ignore_index = next(
            (attr.i for attr in node.attribute if attr.name == "ignore_index"), None
        )

        max_x = np.max(x, axis=1, keepdims=True)
        exp_x = np.exp(x - max_x)
        probabilities = exp_x / np.sum(exp_x, axis=1, keepdims=True)
        log_prob = np.log(probabilities).astype(np.float32)
        loss = self.negative_log_likelihood_loss(
            log_prob,
            target,
            weight,
            reduction,
            ignore_index,
        )

        outputs = ((node.output[0], loss),)
        if len(node.output) > 1:
            outputs = (*outputs, (node.output[1], log_prob))

        for output_name, output in outputs:
            new_tensor = ctx.ggml_tensors_dict[output_name] = ctx.from_numpy(output)
            ctx.set_tensor_shape(new_tensor, output.shape)
            ctx.shapes[output_name] = output.shape
            ctx.set_tensor_dtype(output_name, output.dtype)


@onnx_operators.register
class SoftplusOperator(OnnxOperator):
    def __init__(self):
        super().__init__("Softplus")

    def lower(self, ctx: "GgmlOnnxExecutionContext", node: NodeProto) -> None:
        def softplus(x: npt.NDArray[Any]) -> npt.NDArray[Any]:
            return np.log1p(np.exp(x))

        self.lower_numpy_unary(ctx, node, softplus)


@onnx_operators.register
class SoftsignOperator(OnnxOperator):
    def __init__(self):
        super().__init__("Softsign")

    def lower(self, ctx: "GgmlOnnxExecutionContext", node: NodeProto) -> None:
        node_inputs = [ctx.ggml_tensors_dict[inp] for inp in node.input]

        if len(node_inputs) != 1:
            raise ValueError(
                f'Error for node "{node.name}": Operation "Softsign" requires exactly one input. Actual number of inputs: {len(node_inputs)}'
            )

        x = node_inputs[0]
        x_shape = get_tensor_shape(x)
        x_dtype = get_tensor_dtype(x)

        # y = x / (1 + abs(x))
        one_np = np.full(x_shape, 1, dtype=x_dtype)
        one_t = ctx.from_numpy(one_np)
        x_abs = ggml.ggml_abs(ctx.ggml_eval_context, x)
        one_plus_abs = ggml.ggml_add(ctx.ggml_eval_context, one_t, x_abs)
        y = ggml.ggml_div(ctx.ggml_eval_context, x, one_plus_abs)
        ctx.ggml_tensors_dict[node.output[0]] = y
        ctx.shapes[node.output[0]] = ctx.shapes[node.input[0]]


@onnx_operators.register
class QuickGeluOperator(OnnxOperator):
    def __init__(self):
        super().__init__(
            "QuickGelu",
            OnnxOperator.EXECUTION_NATIVE_OR_NUMPY_RUNTIME,
            OnnxOperator.CLASS_CONDITIONAL_NATIVE,
            domains=("com.microsoft", "com.ggml"),
        )
        self.has_numpy_evaluator = True

    def strategies(
        self, tensor_types: Dict[str, TensorType], node: "NodeIR"
    ) -> Tuple[Tuple[str, str, str], ...]:
        if (
            len(node.inputs) == 1
            and self.tensor_type(tensor_types, node.inputs[0]).is_float32
            and float(node.attribute("alpha", 1.702)) == 1.702
        ):
            return self.native_strategy()
        return self.numpy_runtime_strategy(
            "QuickGelu requires float32 input and alpha=1.702 to lower native"
        )

    def eval_numpy(
        self, node: NodeProto, inputs: Tuple[npt.NDArray[Any], ...]
    ) -> Tuple[npt.NDArray[Any], ...]:
        if len(inputs) != 1:
            raise ValueError(f'Operation "{node.op_type}" requires one input')
        alpha = self.float_attribute(node, "alpha", 1.702)
        x = inputs[0]
        return (np.asarray(x / (1.0 + np.exp(-alpha * x)), dtype=x.dtype),)

    def lower(self, ctx: "GgmlOnnxExecutionContext", node: NodeProto) -> None:
        alpha = self.float_attribute(node, "alpha", 1.702)

        def quick_gelu(x: npt.NDArray[Any]) -> npt.NDArray[Any]:
            return x / (1.0 + np.exp(-alpha * x))

        if alpha == 1.702:
            self.lower_native_unary_or_numpy(
                ctx, node, ggml.ggml_gelu_quick, quick_gelu
            )
            return
        self.lower_numpy_unary(ctx, node, quick_gelu)


@onnx_operators.register
class SiLUOperator(OnnxOperator):
    def __init__(self):
        super().__init__(
            "SiLU",
            OnnxOperator.EXECUTION_NATIVE_OR_NUMPY_RUNTIME,
            OnnxOperator.CLASS_CONDITIONAL_NATIVE,
            domains=("com.ggml",),
        )
        self.has_numpy_evaluator = True

    def strategies(
        self, tensor_types: Dict[str, TensorType], node: "NodeIR"
    ) -> Tuple[Tuple[str, str, str], ...]:
        if (
            len(node.inputs) == 1
            and self.tensor_type(tensor_types, node.inputs[0]).is_float32
        ):
            return self.native_strategy()
        return self.numpy_runtime_strategy(
            "SiLU requires float32 input to lower native"
        )

    def eval_numpy(
        self, node: NodeProto, inputs: Tuple[npt.NDArray[Any], ...]
    ) -> Tuple[npt.NDArray[Any], ...]:
        if len(inputs) != 1:
            raise ValueError(f'Operation "{node.op_type}" requires one input')
        x = inputs[0]
        return (np.asarray(x / (1.0 + np.exp(-x)), dtype=x.dtype),)

    def lower(self, ctx: "GgmlOnnxExecutionContext", node: NodeProto) -> None:
        self.lower_native_unary_or_numpy(
            ctx, node, ggml.ggml_silu, lambda x: x / (1.0 + np.exp(-x))
        )


class GgmlGluOperator(OnnxOperator):
    GGML_FUNC: ClassVar[
        Callable[[ggml.ggml_context_p, ggml.ggml_tensor_p], ggml.ggml_tensor_p]
    ]

    def __init__(self, op_type: str):
        super().__init__(
            op_type,
            OnnxOperator.EXECUTION_NATIVE_OR_NUMPY_RUNTIME,
            OnnxOperator.CLASS_CONDITIONAL_NATIVE,
            domains=("com.ggml",),
        )
        self.has_numpy_evaluator = True

    @staticmethod
    def gelu_erf(x: npt.NDArray[Any]) -> npt.NDArray[Any]:
        erf = np.vectorize(math.erf)
        return 0.5 * x * (1.0 + erf(x / np.sqrt(2.0)))

    @staticmethod
    def gelu_quick(x: npt.NDArray[Any]) -> npt.NDArray[Any]:
        return x / (1.0 + np.exp(-1.702 * x))

    def activation(self, gate: npt.NDArray[Any]) -> npt.NDArray[Any]:
        if self.op_type == "ReGLU":
            return np.maximum(gate, 0)
        if self.op_type == "GeGLU":
            return GeluOperator.numpy_gelu(gate, "tanh")
        if self.op_type == "SwiGLU":
            return gate / (1.0 + np.exp(-gate))
        if self.op_type == "GeGLUErf":
            return self.gelu_erf(gate)
        if self.op_type == "GeGLUQuick":
            return self.gelu_quick(gate)
        raise ValueError(f'Unsupported GLU operator "{self.op_type}"')

    def strategies(
        self, tensor_types: Dict[str, TensorType], node: "NodeIR"
    ) -> Tuple[Tuple[str, str, str], ...]:
        if len(node.inputs) == 1:
            input_type = self.tensor_type(tensor_types, node.inputs[0])
            if (
                input_type.is_float32
                and input_type.shape is not None
                and input_type.shape[-1] % 2 == 0
            ):
                return self.native_strategy()
        return self.numpy_runtime_strategy(
            f"{self.op_type} requires float32 input with even last dimension"
        )

    def eval_numpy(
        self, node: NodeProto, inputs: Tuple[npt.NDArray[Any], ...]
    ) -> Tuple[npt.NDArray[Any], ...]:
        if len(inputs) != 1:
            raise ValueError(f'Operation "{node.op_type}" requires one input')
        x = inputs[0]
        split = x.shape[-1] // 2
        gate = x[..., :split]
        values = x[..., split:]
        return (np.asarray(values * self.activation(gate), dtype=x.dtype),)

    def lower(self, ctx: "GgmlOnnxExecutionContext", node: NodeProto) -> None:
        node_inputs = [ctx.ggml_tensors_dict[inp] for inp in node.input]
        input_name = node.input[0]
        input_shape = ctx.shapes[input_name]
        input_dtype = np.dtype(ctx.get_tensor_dtype(input_name))
        output_shape = (*input_shape[:-1], input_shape[-1] // 2)
        if (
            ctx.can_emit_native(node.output[0])
            and ctx.can_run_native(node)
            and input_dtype == np.dtype(np.float32)
            and input_shape[-1] % 2 == 0
        ):
            result = self.GGML_FUNC(ctx.ggml_eval_context, node_inputs[0])
            ctx.register_native_tensor(
                node.output[0], result, output_shape, input_dtype
            )
            return
        input_array = ctx.logical_tensor_eval_data(
            input_name, node_inputs[0], input_shape
        )
        output = self.eval_numpy(node, (input_array,))[0]
        ctx.set_numpy_runtime_output(node.output[0], output, output.dtype)


@onnx_operators.register
class ReGLUOperator(GgmlGluOperator):
    GGML_FUNC = staticmethod(ggml.ggml_reglu)

    def __init__(self):
        super().__init__("ReGLU")


@onnx_operators.register
class GeGLUOperator(GgmlGluOperator):
    GGML_FUNC = staticmethod(ggml.ggml_geglu)

    def __init__(self):
        super().__init__("GeGLU")


@onnx_operators.register
class SwiGLUOperator(GgmlGluOperator):
    GGML_FUNC = staticmethod(ggml.ggml_swiglu)

    def __init__(self):
        super().__init__("SwiGLU")


@onnx_operators.register
class GeGLUErfOperator(GgmlGluOperator):
    GGML_FUNC = staticmethod(ggml.ggml_geglu_erf)

    def __init__(self):
        super().__init__("GeGLUErf")


@onnx_operators.register
class GeGLUQuickOperator(GgmlGluOperator):
    GGML_FUNC = staticmethod(ggml.ggml_geglu_quick)

    def __init__(self):
        super().__init__("GeGLUQuick")


@onnx_operators.register
class SwiGLUOAIOperator(OnnxOperator):
    def __init__(self):
        super().__init__(
            "SwiGLUOAI",
            execution=OnnxOperator.EXECUTION_NATIVE,
            domains=("com.ggml",),
        )

    def lower(self, ctx: "GgmlOnnxExecutionContext", node: NodeProto) -> None:
        node_inputs = [ctx.ggml_tensors_dict[inp] for inp in node.input]
        alpha = self.float_attribute(node, "alpha", 1.702)
        limit = self.float_attribute(node, "limit", 7.0)
        result = ggml.ggml_swiglu_oai(
            ctx.ggml_eval_context, node_inputs[0], node_inputs[1], alpha, limit
        )
        output_shape = ctx.shapes.get(node.output[0], ctx.get_tensor_shape(result))
        ctx.register_native_tensor(
            node.output[0], result, output_shape, ctx.get_tensor_dtype(node.input[0])
        )


@onnx_operators.register
class GgmlRollOperator(OnnxOperator):
    def __init__(self):
        super().__init__(
            "Roll", execution=OnnxOperator.EXECUTION_NATIVE, domains=("com.ggml",)
        )

    def lower(self, ctx: "GgmlOnnxExecutionContext", node: NodeProto) -> None:
        node_inputs = [ctx.ggml_tensors_dict[inp] for inp in node.input]
        input_shape = ctx.shapes[node.input[0]]
        shifts = list(self.ints_attribute(node, "shifts") or ())
        while len(shifts) < 4:
            shifts.append(0)
        storage_shifts = tuple(reversed(shifts[: len(input_shape)]))
        storage_shifts = storage_shifts + (0,) * (4 - len(storage_shifts))
        result = ggml.ggml_roll(ctx.ggml_eval_context, node_inputs[0], *storage_shifts)
        ctx.register_native_tensor(
            node.output[0], result, input_shape, ctx.get_tensor_dtype(node.input[0])
        )


@onnx_operators.register
class GgmlFillOperator(OnnxOperator):
    def __init__(self):
        super().__init__(
            "Fill", execution=OnnxOperator.EXECUTION_NATIVE, domains=("com.ggml",)
        )

    def lower(self, ctx: "GgmlOnnxExecutionContext", node: NodeProto) -> None:
        node_inputs = [ctx.ggml_tensors_dict[inp] for inp in node.input]
        value = self.float_attribute(node, "value", 0.0)
        result = ggml.ggml_fill(ctx.ggml_eval_context, node_inputs[0], value)
        ctx.register_native_tensor(
            node.output[0],
            result,
            ctx.shapes[node.input[0]],
            ctx.get_tensor_dtype(node.input[0]),
        )


@onnx_operators.register
class GgmlArgSortOperator(OnnxOperator):
    def __init__(self):
        super().__init__(
            "ArgSort", execution=OnnxOperator.EXECUTION_NATIVE, domains=("com.ggml",)
        )

    def lower(self, ctx: "GgmlOnnxExecutionContext", node: NodeProto) -> None:
        node_inputs = [ctx.ggml_tensors_dict[inp] for inp in node.input]
        descending = bool(self.int_attribute(node, "descending", 0))
        order = ggml.GGML_SORT_ORDER_DESC if descending else ggml.GGML_SORT_ORDER_ASC
        result = ggml.ggml_argsort(ctx.ggml_eval_context, node_inputs[0], order)
        ctx.register_native_tensor(
            node.output[0], result, ctx.shapes[node.input[0]], np.dtype(np.int32)
        )


@onnx_operators.register
class GgmlRopeOperator(OnnxOperator):
    def __init__(self):
        super().__init__(
            "Rope", execution=OnnxOperator.EXECUTION_NATIVE, domains=("com.ggml",)
        )

    def lower(self, ctx: "GgmlOnnxExecutionContext", node: NodeProto) -> None:
        node_inputs = [ctx.ggml_tensors_dict[inp] for inp in node.input]
        n_dims = self.int_attribute(node, "n_dims", ctx.shapes[node.input[0]][-1])
        mode = self.int_attribute(node, "mode", ggml.GGML_ROPE_TYPE_NORMAL)
        result = ggml.ggml_rope(
            ctx.ggml_eval_context, node_inputs[0], node_inputs[1], n_dims, mode
        )
        ctx.register_native_tensor(
            node.output[0],
            result,
            ctx.shapes[node.input[0]],
            ctx.get_tensor_dtype(node.input[0]),
        )


@onnx_operators.register
class GgmlFlashAttentionOperator(OnnxOperator):
    def __init__(self):
        super().__init__(
            "FlashAttention",
            execution=OnnxOperator.EXECUTION_NATIVE,
            domains=("com.ggml",),
        )

    def lower(self, ctx: "GgmlOnnxExecutionContext", node: NodeProto) -> None:
        node_inputs = [ctx.ggml_tensors_dict[inp] for inp in node.input if inp]
        if len(node_inputs) not in {3, 4}:
            raise ValueError(
                "com.ggml.FlashAttention requires Q, K, V, and optional mask"
            )
        q_shape = ctx.shapes[node.input[0]]
        scale = self.float_attribute(node, "scale", 1.0 / math.sqrt(q_shape[-1]))
        max_bias = self.float_attribute(node, "max_bias", 0.0)
        logit_softcap = self.float_attribute(node, "logit_softcap", 0.0)
        mask = node_inputs[3] if len(node_inputs) == 4 else None
        result = ggml.ggml_flash_attn_ext(
            ctx.ggml_eval_context,
            node_inputs[0],
            node_inputs[1],
            node_inputs[2],
            mask,
            scale,
            max_bias,
            logit_softcap,
        )
        if len(q_shape) == 4:
            result = ggml.ggml_permute(ctx.ggml_eval_context, result, 0, 2, 1, 3)
        ctx.register_native_tensor(
            node.output[0], result, q_shape, ctx.get_tensor_dtype(node.input[0])
        )


@onnx_operators.register
class GgmlTimestepEmbeddingOperator(OnnxOperator):
    def __init__(self):
        super().__init__(
            "TimestepEmbedding",
            execution=OnnxOperator.EXECUTION_NATIVE,
            domains=("com.ggml",),
        )

    def lower(self, ctx: "GgmlOnnxExecutionContext", node: NodeProto) -> None:
        node_inputs = [ctx.ggml_tensors_dict[inp] for inp in node.input]
        dim = self.int_attribute(node, "dim", 0)
        max_period = self.int_attribute(node, "max_period", 10000)
        result = ggml.ggml_timestep_embedding(
            ctx.ggml_eval_context, node_inputs[0], dim, max_period
        )
        output_shape = ctx.shapes.get(node.output[0], ctx.get_tensor_shape(result))
        ctx.register_native_tensor(
            node.output[0], result, output_shape, np.dtype(np.float32)
        )


@onnx_operators.register
class GgmlWindowPartitionOperator(OnnxOperator):
    def __init__(self):
        super().__init__(
            "WindowPartition",
            execution=OnnxOperator.EXECUTION_NATIVE,
            domains=("com.ggml",),
        )

    def lower(self, ctx: "GgmlOnnxExecutionContext", node: NodeProto) -> None:
        node_inputs = [ctx.ggml_tensors_dict[inp] for inp in node.input]
        window = self.int_attribute(node, "window", 1)
        result = ggml.ggml_win_part(ctx.ggml_eval_context, node_inputs[0], window)
        output_shape = ctx.shapes.get(node.output[0], ctx.get_tensor_shape(result))
        ctx.register_native_tensor(
            node.output[0], result, output_shape, ctx.get_tensor_dtype(node.input[0])
        )


@onnx_operators.register
class GgmlWindowUnpartitionOperator(OnnxOperator):
    def __init__(self):
        super().__init__(
            "WindowUnpartition",
            execution=OnnxOperator.EXECUTION_NATIVE,
            domains=("com.ggml",),
        )

    def lower(self, ctx: "GgmlOnnxExecutionContext", node: NodeProto) -> None:
        node_inputs = [ctx.ggml_tensors_dict[inp] for inp in node.input]
        width = self.int_attribute(node, "width", 1)
        height = self.int_attribute(node, "height", 1)
        window = self.int_attribute(node, "window", 1)
        result = ggml.ggml_win_unpart(
            ctx.ggml_eval_context, node_inputs[0], width, height, window
        )
        output_shape = ctx.shapes.get(node.output[0], ctx.get_tensor_shape(result))
        ctx.register_native_tensor(
            node.output[0], result, output_shape, ctx.get_tensor_dtype(node.input[0])
        )


@onnx_operators.register
class GgmlGetRelPosOperator(OnnxOperator):
    def __init__(self):
        super().__init__(
            "GetRelPos",
            execution=OnnxOperator.EXECUTION_NATIVE,
            domains=("com.ggml",),
        )

    def lower(self, ctx: "GgmlOnnxExecutionContext", node: NodeProto) -> None:
        node_inputs = [ctx.ggml_tensors_dict[inp] for inp in node.input]
        qh = self.int_attribute(node, "qh", 1)
        kh = self.int_attribute(node, "kh", 1)
        result = ggml.ggml_get_rel_pos(ctx.ggml_eval_context, node_inputs[0], qh, kh)
        output_shape = ctx.shapes.get(node.output[0], ctx.get_tensor_shape(result))
        ctx.register_native_tensor(
            node.output[0], result, output_shape, ctx.get_tensor_dtype(node.input[0])
        )


@onnx_operators.register
class GgmlAddRelPosOperator(OnnxOperator):
    def __init__(self):
        super().__init__(
            "AddRelPos",
            execution=OnnxOperator.EXECUTION_NATIVE,
            domains=("com.ggml",),
        )

    def lower(self, ctx: "GgmlOnnxExecutionContext", node: NodeProto) -> None:
        node_inputs = [ctx.ggml_tensors_dict[inp] for inp in node.input]
        result = ggml.ggml_add_rel_pos(
            ctx.ggml_eval_context, node_inputs[0], node_inputs[1], node_inputs[2]
        )
        output_shape = ctx.shapes.get(node.output[0], ctx.get_tensor_shape(result))
        ctx.register_native_tensor(
            node.output[0], result, output_shape, ctx.get_tensor_dtype(node.input[0])
        )


@onnx_operators.register
class GgmlSSMConvOperator(OnnxOperator):
    def __init__(self):
        super().__init__(
            "SSMConv", execution=OnnxOperator.EXECUTION_NATIVE, domains=("com.ggml",)
        )

    def lower(self, ctx: "GgmlOnnxExecutionContext", node: NodeProto) -> None:
        node_inputs = [ctx.ggml_tensors_dict[inp] for inp in node.input]
        result = ggml.ggml_ssm_conv(
            ctx.ggml_eval_context, node_inputs[0], node_inputs[1]
        )
        output_shape = ctx.shapes.get(node.output[0], ctx.get_tensor_shape(result))
        ctx.register_native_tensor(
            node.output[0], result, output_shape, ctx.get_tensor_dtype(node.input[0])
        )


@onnx_operators.register
class GgmlSSMScanOperator(OnnxOperator):
    def __init__(self):
        super().__init__(
            "SSMScan", execution=OnnxOperator.EXECUTION_NATIVE, domains=("com.ggml",)
        )

    def lower(self, ctx: "GgmlOnnxExecutionContext", node: NodeProto) -> None:
        node_inputs = [ctx.ggml_tensors_dict[inp] for inp in node.input]
        result = ggml.ggml_ssm_scan(ctx.ggml_eval_context, *node_inputs)
        output_shape = ctx.shapes.get(node.output[0], ctx.get_tensor_shape(result))
        ctx.register_native_tensor(
            node.output[0], result, output_shape, ctx.get_tensor_dtype(node.input[0])
        )


@onnx_operators.register
class GgmlGatedLinearAttentionOperator(OnnxOperator):
    def __init__(self):
        super().__init__(
            "GatedLinearAttention",
            execution=OnnxOperator.EXECUTION_NATIVE,
            domains=("com.ggml",),
        )

    def lower(self, ctx: "GgmlOnnxExecutionContext", node: NodeProto) -> None:
        node_inputs = [ctx.ggml_tensors_dict[inp] for inp in node.input]
        scale = self.float_attribute(node, "scale", 1.0)
        result = ggml.ggml_gated_linear_attn(
            ctx.ggml_eval_context,
            node_inputs[0],
            node_inputs[1],
            node_inputs[2],
            node_inputs[3],
            node_inputs[4],
            scale,
        )
        output_shape = ctx.shapes.get(node.output[0], ctx.get_tensor_shape(result))
        ctx.register_native_tensor(
            node.output[0], result, output_shape, ctx.get_tensor_dtype(node.input[0])
        )


@onnx_operators.register
class GgmlGatedDeltaNetOperator(OnnxOperator):
    def __init__(self):
        super().__init__(
            "GatedDeltaNet",
            execution=OnnxOperator.EXECUTION_NATIVE,
            domains=("com.ggml",),
        )

    def lower(self, ctx: "GgmlOnnxExecutionContext", node: NodeProto) -> None:
        node_inputs = [ctx.ggml_tensors_dict[inp] for inp in node.input]
        result = ggml.ggml_gated_delta_net(ctx.ggml_eval_context, *node_inputs)
        output_shape = ctx.shapes.get(node.output[0], ctx.get_tensor_shape(result))
        ctx.register_native_tensor(
            node.output[0], result, output_shape, ctx.get_tensor_dtype(node.input[0])
        )


@onnx_operators.register
class GgmlRWKVWKV6Operator(OnnxOperator):
    def __init__(self):
        super().__init__(
            "RWKVWKV6", execution=OnnxOperator.EXECUTION_NATIVE, domains=("com.ggml",)
        )

    def lower(self, ctx: "GgmlOnnxExecutionContext", node: NodeProto) -> None:
        node_inputs = [ctx.ggml_tensors_dict[inp] for inp in node.input]
        result = ggml.ggml_rwkv_wkv6(ctx.ggml_eval_context, *node_inputs)
        output_shape = ctx.shapes.get(node.output[0], ctx.get_tensor_shape(result))
        ctx.register_native_tensor(
            node.output[0], result, output_shape, ctx.get_tensor_dtype(node.input[0])
        )


@onnx_operators.register
class GgmlRWKVWKV7Operator(OnnxOperator):
    def __init__(self):
        super().__init__(
            "RWKVWKV7", execution=OnnxOperator.EXECUTION_NATIVE, domains=("com.ggml",)
        )

    def lower(self, ctx: "GgmlOnnxExecutionContext", node: NodeProto) -> None:
        node_inputs = [ctx.ggml_tensors_dict[inp] for inp in node.input]
        result = ggml.ggml_rwkv_wkv7(ctx.ggml_eval_context, *node_inputs)
        output_shape = ctx.shapes.get(node.output[0], ctx.get_tensor_shape(result))
        ctx.register_native_tensor(
            node.output[0], result, output_shape, ctx.get_tensor_dtype(node.input[0])
        )


@onnx_operators.register
class STFTOperator(OnnxOperator):
    def __init__(self):
        super().__init__("STFT")

    def lower(self, ctx: "GgmlOnnxExecutionContext", node: NodeProto) -> None:
        node_inputs = [
            ctx.ggml_tensors_dict[inp] if inp != "" else None for inp in node.input
        ]

        if len(node_inputs) not in {2, 3, 4}:
            raise ValueError(
                f'Error for node "{node.name}": Operation "STFT" requires two to four inputs. Actual number of inputs: {len(node_inputs)}'
            )

        signal_tensor = node_inputs[0]
        frame_step_tensor = node_inputs[1]
        window_tensor = node_inputs[2] if len(node_inputs) >= 3 else None
        frame_length_tensor = node_inputs[3] if len(node_inputs) == 4 else None
        if signal_tensor is None or frame_step_tensor is None:
            raise ValueError(
                f'Error for node "{node.name}": STFT signal and frame_step inputs are required.'
            )

        signal_shape = ctx.shapes[node.input[0]]
        signal = ctx.to_numpy(ctx.eval_tensor(signal_tensor)).reshape(signal_shape)
        frame_step = int(
            ctx.to_numpy(ctx.eval_tensor(frame_step_tensor))
            .reshape(ctx.shapes[node.input[1]])
            .item()
        )

        window = None
        if window_tensor is not None:
            window = ctx.to_numpy(ctx.eval_tensor(window_tensor)).reshape(
                ctx.shapes[node.input[2]]
            )

        if frame_length_tensor is not None:
            frame_length = int(
                ctx.to_numpy(ctx.eval_tensor(frame_length_tensor))
                .reshape(ctx.shapes[node.input[3]])
                .item()
            )
        elif window is not None:
            frame_length = int(window.shape[0])
        else:
            raise ValueError(
                f'Error for node "{node.name}": STFT requires frame_length when window is not provided.'
            )

        onesided = next(
            (attr.i for attr in node.attribute if attr.name == "onesided"), 1
        )
        batch_size = signal_shape[0]
        signal_length = signal_shape[1]
        frame_count = 1 + (signal_length - frame_length) // frame_step
        output_length = frame_length // 2 + 1 if onesided else frame_length
        output = np.empty((batch_size, frame_count, output_length, 2), dtype=np.float32)

        if signal_shape[-1] == 1:
            complex_signal = signal[..., 0].astype(np.complex64)
        elif signal_shape[-1] == 2:
            complex_signal = signal[..., 0].astype(np.complex64) + 1j * signal[
                ..., 1
            ].astype(np.complex64)
        else:
            raise ValueError(
                f'Error for node "{node.name}": STFT signal last dimension must be 1 or 2.'
            )

        for batch in range(batch_size):
            for frame in range(frame_count):
                start = frame * frame_step
                stop = start + frame_length
                frame_signal = complex_signal[batch, start:stop]
                if window is not None:
                    frame_signal = frame_signal * window
                complex_out = np.fft.fft(frame_signal)[:output_length]
                output[batch, frame] = np.stack(
                    (complex_out.real, complex_out.imag), axis=1
                )

        new_tensor = ctx.ggml_tensors_dict[node.output[0]] = ctx.from_numpy(output)
        ctx.set_tensor_shape(new_tensor, output.shape)
        ctx.shapes[node.output[0]] = output.shape
        ctx.set_tensor_dtype(node.output[0], output.dtype)


@onnx_operators.register
class SpaceToDepthOperator(OnnxOperator):
    def __init__(self):
        super().__init__("SpaceToDepth")

    def lower(self, ctx: "GgmlOnnxExecutionContext", node: NodeProto) -> None:
        node_inputs = [ctx.ggml_tensors_dict[inp] for inp in node.input]

        if len(node_inputs) != 1:
            raise ValueError(
                f'Error for node "{node.name}": Operation "SpaceToDepth" requires exactly one input. Actual number of inputs: {len(node_inputs)}'
            )

        x = node_inputs[0]
        blocksize = next(
            (attr.i for attr in node.attribute if attr.name == "blocksize"), None
        )

        if blocksize is None:
            raise ValueError(
                f'Error for node "{node.name}": Operation "SpaceToDepth" requires "blocksize"'
            )

        N, C, H, W = ctx.shapes[node.input[0]]
        new_H = H // blocksize
        new_W = W // blocksize
        output_shape = (N, C * blocksize * blocksize, new_H, new_W)

        x_t = ctx.from_numpy(np.empty(output_shape, dtype=get_tensor_dtype(x)))

        blocksize_c = ctypes.c_int(blocksize)

        @ggml.ggml_custom2_op_t
        def custom_space_to_depth(
            tensor_out: ggml.ggml_tensor_p,
            tensor_in_1: ggml.ggml_tensor_p,
            tensor_in_2: ggml.ggml_tensor_p,
            ith: int,
            nth: int,
            userdata: Optional[ctypes.c_void_p],
        ):
            x = ctx.to_numpy(tensor_in_2).reshape(ctx.shapes[node.input[0]])
            blocksize = ctypes.cast(
                userdata, ctypes.POINTER(ctypes.c_int)
            ).contents.value

            N, C, H, W = x.shape
            new_H = H // blocksize
            new_W = W // blocksize

            reshaped = x.reshape(N, C, new_H, blocksize, new_W, blocksize)
            transposed = reshaped.transpose(
                0, 3, 5, 1, 2, 4
            )  # ONNX specification TODO: Test more examples
            y = transposed.reshape(N, C * (blocksize**2), new_H, new_W)

            ctx.set_tensor_data(tensor_out, y)

        new_tensor = ctx.ggml_tensors_dict[node.output[0]] = (
            ggml.ggml_map_custom2_inplace(
                ctx.ggml_eval_context,
                x_t,
                x,
                custom_space_to_depth,
                1,
                ctypes.pointer(blocksize_c),
            )
        )

        ctx.refs.append(custom_space_to_depth)

        ctx.refs.append(blocksize_c)
        ctx.set_tensor_shape(new_tensor, output_shape)
        ctx.shapes[node.output[0]] = output_shape


@onnx_operators.register
class SplitOperator(OnnxOperator):
    def __init__(self):
        super().__init__(
            "Split",
            OnnxOperator.EXECUTION_NATIVE_OR_NUMPY_RUNTIME,
            OnnxOperator.CLASS_CONDITIONAL_NATIVE_VIEW,
            view_kind=ViewTransformSemantics.KIND_LAYOUT,
        )
        self.has_numpy_evaluator = True

    @staticmethod
    def split_axis(axis: int, rank: int) -> Optional[int]:
        if axis < 0:
            axis += rank
        if axis < 0 or axis >= rank:
            return None
        return axis

    @staticmethod
    def split_shapes(
        input_shape: Tuple[int, ...],
        axis: int,
        split_values: Sequence[int],
    ) -> Tuple[Tuple[int, ...], ...]:
        output_shapes = []
        for split_value in split_values:
            output_shape = list(input_shape)
            output_shape[axis] = int(split_value)
            output_shapes.append(tuple(output_shape))
        return tuple(output_shapes)

    @staticmethod
    def inferred_split_shapes(
        input_shape: Tuple[int, ...],
        axis: int,
        num_outputs: int,
    ) -> Tuple[Tuple[int, ...], ...]:
        split_size = input_shape[axis] // num_outputs
        remainder = input_shape[axis] % num_outputs
        split_values = [
            split_size + (1 if split_index < remainder else 0)
            for split_index in range(num_outputs)
        ]
        return SplitOperator.split_shapes(input_shape, axis, split_values)

    @staticmethod
    def can_native_view_shapes(
        input_shape: Tuple[int, ...],
        output_shapes: Sequence[Tuple[int, ...]],
        axis: int,
    ) -> bool:
        if not input_shape:
            return False
        if len(input_shape) > ViewTransformSemantics.GGML_MAX_DIMS:
            return False
        if not output_shapes:
            return False
        if any(any(dim == 0 for dim in output_shape) for output_shape in output_shapes):
            return False
        if (
            sum(output_shape[axis] for output_shape in output_shapes)
            != input_shape[axis]
        ):
            return False
        for output_shape in output_shapes:
            if len(output_shape) != len(input_shape):
                return False
            for dim_index, dim in enumerate(output_shape):
                if dim_index != axis and dim != input_shape[dim_index]:
                    return False
        return True

    def static_split_shapes(
        self, tensor_types: Dict[str, TensorType], node: "NodeIR"
    ) -> Optional[Tuple[int, Tuple[Tuple[int, ...], ...]]]:
        if len(node.inputs) not in {1, 2}:
            return None
        input_shape = self.tensor_type(tensor_types, node.inputs[0]).shape
        if input_shape is None:
            return None
        axis = self.split_axis(int(node.attribute("axis", 0)), len(input_shape))
        if axis is None:
            return None
        output_shapes = tuple(
            self.tensor_type(tensor_types, output_name).shape
            for output_name in node.outputs
            if output_name
        )
        if any(output_shape is None for output_shape in output_shapes):
            return None
        typed_output_shapes = tuple(
            output_shape for output_shape in output_shapes if output_shape is not None
        )
        if self.can_native_view_shapes(input_shape, typed_output_shapes, axis):
            return axis, typed_output_shapes
        return None

    def strategies(
        self, tensor_types: Dict[str, TensorType], node: "NodeIR"
    ) -> Tuple[Tuple[str, str, str], ...]:
        if self.static_split_shapes(tensor_types, node) is not None:
            return self.native_view_strategy()
        return self.numpy_runtime_strategy(
            "Split requires static non-empty output shapes with rank <= 4 "
            "to lower as native ggml views"
        )

    def runtime_split_shapes(
        self,
        ctx: "GgmlOnnxExecutionContext",
        node: NodeProto,
        node_inputs: Sequence[ggml.ggml_tensor_p],
    ) -> Tuple[int, Tuple[Tuple[int, ...], ...]]:
        input_shape = tuple(ctx.shapes[node.input[0]])
        axis = self.split_axis(self.int_attribute(node, "axis", 0), len(input_shape))
        if axis is None:
            raise ValueError(
                f'Error for node "{node.name}": Split axis is invalid for '
                f"rank {len(input_shape)}"
            )
        num_outputs = self.int_attribute(node, "num_outputs", len(node.output))
        if len(node_inputs) == 1:
            return axis, self.inferred_split_shapes(input_shape, axis, num_outputs)

        split_shape = ctx.shapes[node.input[1]]
        split_values = tuple(
            int(value)
            for value in ctx.logical_tensor_eval_data(
                node.input[1], node_inputs[1], split_shape
            ).flatten()
        )
        return axis, self.split_shapes(input_shape, axis, split_values)

    @staticmethod
    def native_split_view(
        ctx: "GgmlOnnxExecutionContext",
        input_tensor: ggml.ggml_tensor_p,
        output_shape: Tuple[int, ...],
        axis: int,
        split_start: int,
    ) -> ggml.ggml_tensor_p:
        storage_shape = tuple(reversed(output_shape))
        storage_axis = len(output_shape) - axis - 1
        strides = tuple(input_tensor.contents.nb[: len(storage_shape)])
        offset = int(split_start) * int(strides[storage_axis])

        if len(storage_shape) == 1:
            return ggml.ggml_view_1d(
                ctx.ggml_eval_context,
                input_tensor,
                storage_shape[0],
                offset,
            )
        if len(storage_shape) == 2:
            return ggml.ggml_view_2d(
                ctx.ggml_eval_context,
                input_tensor,
                storage_shape[0],
                storage_shape[1],
                strides[1],
                offset,
            )
        if len(storage_shape) == 3:
            return ggml.ggml_view_3d(
                ctx.ggml_eval_context,
                input_tensor,
                storage_shape[0],
                storage_shape[1],
                storage_shape[2],
                strides[1],
                strides[2],
                offset,
            )
        if len(storage_shape) == 4:
            return ggml.ggml_view_4d(
                ctx.ggml_eval_context,
                input_tensor,
                storage_shape[0],
                storage_shape[1],
                storage_shape[2],
                storage_shape[3],
                strides[1],
                strides[2],
                strides[3],
                offset,
            )
        raise ValueError("Split native view supports rank 1 through 4")

    def can_lower_native_runtime(
        self,
        ctx: "GgmlOnnxExecutionContext",
        node: NodeProto,
        input_tensor: ggml.ggml_tensor_p,
        output_shapes: Sequence[Tuple[int, ...]],
        axis: int,
    ) -> bool:
        input_shape = tuple(ctx.shapes[node.input[0]])
        return (
            all(ctx.can_emit_native(output_name) for output_name in node.output)
            and ctx.can_run_native(node)
            and ggml.ggml_is_contiguous(input_tensor)
            and self.can_native_view_shapes(input_shape, output_shapes, axis)
        )

    def eval_numpy(
        self, node: NodeProto, inputs: Tuple[npt.NDArray[Any], ...]
    ) -> Tuple[npt.NDArray[Any], ...]:
        if len(inputs) not in {1, 2}:
            raise ValueError(
                f'Operation "{node.op_type}" requires one or two inputs. '
                f"Actual number of inputs: {len(inputs)}"
            )
        input_array = inputs[0]
        axis = self.split_axis(self.int_attribute(node, "axis", 0), input_array.ndim)
        if axis is None:
            raise ValueError(
                f'Operation "{node.op_type}" has invalid axis for '
                f"rank {input_array.ndim}"
            )
        if len(inputs) == 2:
            split_values = tuple(
                int(value) for value in np.asarray(inputs[1]).flatten()
            )
        else:
            num_outputs = self.int_attribute(node, "num_outputs", len(node.output))
            split_values = tuple(
                shape[axis]
                for shape in self.inferred_split_shapes(
                    input_array.shape, axis, num_outputs
                )
            )
        output_arrays = []
        split_start = 0
        for split_value in split_values:
            split_end = split_start + split_value
            slices = [slice(None)] * input_array.ndim
            slices[axis] = slice(split_start, split_end)
            output_arrays.append(input_array[tuple(slices)].copy())
            split_start = split_end
        return tuple(output_arrays)

    def lower(self, ctx: "GgmlOnnxExecutionContext", node: NodeProto) -> None:
        node_inputs = [ctx.ggml_tensors_dict[inp] for inp in node.input]

        if len(node_inputs) < 1 or len(node_inputs) > 2:
            raise ValueError(
                f'Error for node "{node.name}": Operation "Split" requires 1 - 2 inputs. Actual number of inputs: {len(node_inputs)}'
            )

        input_tensor = node_inputs[0]
        axis, split_shapes = self.runtime_split_shapes(ctx, node, node_inputs)
        input_dtype = np.dtype(ctx.get_tensor_dtype(node.input[0]))

        split_start = 0
        if self.can_lower_native_runtime(ctx, node, input_tensor, split_shapes, axis):
            for output_name, split_shape in zip(node.output, split_shapes):
                split_view = self.native_split_view(
                    ctx,
                    input_tensor,
                    split_shape,
                    axis,
                    split_start,
                )
                ctx.register_native_view_tensor(
                    output_name,
                    split_view,
                    split_shape,
                    input_dtype,
                    node.input[0],
                )
                split_start += split_shape[axis]
            return

        input_array = ctx.logical_tensor_eval_data(
            node.input[0], input_tensor, tuple(ctx.shapes[node.input[0]])
        )
        eval_inputs = (input_array,)
        if len(node_inputs) == 2:
            split_array = ctx.logical_tensor_eval_data(
                node.input[1], node_inputs[1], ctx.shapes[node.input[1]]
            )
            eval_inputs = (input_array, split_array)
        output_arrays = self.eval_numpy(node, eval_inputs)
        for output_name, output_array in zip(node.output, output_arrays):
            ctx.set_numpy_runtime_output(output_name, output_array, output_array.dtype)


@onnx_operators.register
class SqrtOperator(OnnxOperator):
    def __init__(self):
        super().__init__(
            "Sqrt",
            OnnxOperator.EXECUTION_NATIVE_OR_NUMPY_RUNTIME,
            OnnxOperator.CLASS_CONDITIONAL_NATIVE,
        )
        self.has_numpy_evaluator = True

    def strategies(
        self, tensor_types: Dict[str, TensorType], node: "NodeIR"
    ) -> Tuple[Tuple[str, str, str], ...]:
        if (
            len(node.inputs) == 1
            and self.tensor_type(tensor_types, node.inputs[0]).is_float32
        ):
            return self.native_strategy()
        return self.numpy_runtime_strategy()

    def eval_numpy(
        self, node: NodeProto, inputs: Tuple[npt.NDArray[Any], ...]
    ) -> Tuple[npt.NDArray[Any], ...]:
        return self.eval_numpy_unary_operator(node, inputs, np.sqrt)

    def lower(self, ctx: "GgmlOnnxExecutionContext", node: NodeProto) -> None:
        self.lower_native_unary_or_numpy(ctx, node, ggml.ggml_sqrt, np.sqrt)


@onnx_operators.register
class SqueezeOperator(ViewOnnxOperator):
    def __init__(self):
        super().__init__("Squeeze", ViewTransformSemantics.KIND_SHAPE)

    def validate_view_inputs(
        self, node: NodeProto, node_inputs: Sequence[ggml.ggml_tensor_p]
    ) -> None:
        if len(node_inputs) not in {1, 2}:
            raise ValueError(
                f'Error for node "{node.name}": Operation "Squeeze" requires one or two inputs, data and optional axes. Actual number of inputs: {len(node_inputs)}'
            )

    def has_static_parameters(
        self,
        tensor_types: Dict[str, TensorType],
        node: "NodeIR",
        input_type: TensorType,
    ) -> bool:
        return (
            "axes" in node.attributes
            or (len(node.inputs) == 1 and input_type.shape is not None)
            or (
                len(node.inputs) == 2
                and self.tensor_type(tensor_types, node.inputs[1]).constant
            )
        )

    def axes_from_inputs(
        self, node: NodeProto, inputs: Tuple[npt.NDArray[Any], ...]
    ) -> Optional[Tuple[int, ...]]:
        if len(inputs) == 2:
            return tuple(int(axis) for axis in np.asarray(inputs[1]).flatten())
        return self.ints_attribute(node, "axes")

    def axes_from_runtime(
        self, ctx: "GgmlOnnxExecutionContext", node: NodeProto
    ) -> Optional[Tuple[int, ...]]:
        if len(node.input) == 2 and node.input[1]:
            axes_tensor = ctx.ggml_tensors_dict[node.input[1]]
            return tuple(
                int(axis)
                for axis in ctx.logical_tensor_eval_data(
                    node.input[1],
                    axes_tensor,
                    ctx.shapes[node.input[1]],
                ).flatten()
            )
        return self.ints_attribute(node, "axes")

    def squeeze_shape(
        self, input_shape: Tuple[int, ...], axes: Optional[Tuple[int, ...]]
    ) -> Tuple[int, ...]:
        if axes is None:
            return np.squeeze(np.empty(input_shape, dtype=np.float32)).shape
        axes = tuple(axis + len(input_shape) if axis < 0 else axis for axis in axes)
        return np.squeeze(np.empty(input_shape, dtype=np.float32), axis=axes).shape

    def runtime_output_shape(
        self,
        ctx: "GgmlOnnxExecutionContext",
        node: NodeProto,
        input_shape: Tuple[int, ...],
    ) -> Tuple[int, ...]:
        return self.squeeze_shape(input_shape, self.axes_from_runtime(ctx, node))

    def eval_numpy(
        self, node: NodeProto, inputs: Tuple[npt.NDArray[Any], ...]
    ) -> Tuple[npt.NDArray[Any], ...]:
        if len(inputs) not in {1, 2}:
            raise ValueError(
                f'Operation "{node.op_type}" requires one or two inputs. '
                f"Actual number of inputs: {len(inputs)}"
            )
        axes = self.axes_from_inputs(node, inputs)
        if axes is None:
            return (np.squeeze(inputs[0]),)
        axes = tuple(axis + inputs[0].ndim if axis < 0 else axis for axis in axes)
        return (np.squeeze(inputs[0], axis=axes),)


@onnx_operators.register
class SubOperator(OnnxOperator):
    def __init__(self):
        super().__init__(
            "Sub",
            OnnxOperator.EXECUTION_NATIVE_OR_NUMPY_RUNTIME,
            OnnxOperator.CLASS_CONDITIONAL_NATIVE,
        )
        self.has_numpy_evaluator = True

    def strategies(
        self, tensor_types: Dict[str, TensorType], node: "NodeIR"
    ) -> Tuple[Tuple[str, str, str], ...]:
        if len(node.inputs) != 2:
            return self.numpy_runtime_strategy()
        input_types = self.binary_elementwise_types(tensor_types, node)
        if self.is_same_shape_float32(input_types):
            return self.native_strategy()
        return self.numpy_runtime_strategy()

    def eval_numpy(
        self, node: NodeProto, inputs: Tuple[npt.NDArray[Any], ...]
    ) -> Tuple[npt.NDArray[Any], ...]:
        return self.eval_numpy_binary_operator(node, inputs, np.subtract)

    def lower(self, ctx: "GgmlOnnxExecutionContext", node: NodeProto) -> None:
        self.lower_native_binary_or_numpy(ctx, node, ggml.ggml_sub, np.subtract)


@onnx_operators.register
class SumOperator(OnnxOperator):
    def __init__(self):
        super().__init__(
            "Sum",
            OnnxOperator.EXECUTION_NATIVE_OR_NUMPY_RUNTIME,
            OnnxOperator.CLASS_CONDITIONAL_NATIVE,
        )
        self.has_numpy_evaluator = True

    def strategies(
        self, tensor_types: Dict[str, TensorType], node: "NodeIR"
    ) -> Tuple[Tuple[str, str, str], ...]:
        if self.is_same_shape_float32(
            self.variadic_elementwise_types(tensor_types, node)
        ):
            return self.decomposed_strategy()
        return self.numpy_runtime_strategy()

    def eval_numpy(
        self, node: NodeProto, inputs: Tuple[npt.NDArray[Any], ...]
    ) -> Tuple[npt.NDArray[Any], ...]:
        return self.eval_numpy_variadic_operator(node, inputs, np.add)

    def lower(self, ctx: "GgmlOnnxExecutionContext", node: NodeProto) -> None:
        node_inputs = [ctx.ggml_tensors_dict[inp] for inp in node.input]

        if len(node_inputs) < 1:
            raise ValueError(
                f'Error for node "{node.name}": Operation "Sum" requires at least one input. Actual number of inputs: {len(node_inputs)}'
            )

        output_name = node.output[0]
        input_types = tuple(
            self.runtime_tensor_type(ctx, input_name) for input_name in node.input
        )
        first_shape = input_types[0].shape
        if (
            ctx.can_emit_native(output_name)
            and ctx.can_run_native(node)
            and self.is_same_shape_float32(input_types)
            and first_shape is not None
        ):
            result = node_inputs[0]
            for tensor in node_inputs[1:]:
                result = ggml.ggml_add(ctx.ggml_eval_context, result, tensor)
            ctx.register_decomposed_tensor(
                output_name,
                result,
                first_shape,
                np.dtype(np.float32),
            )
            return

        self.lower_numpy_variadic(ctx, node, np.add)


@onnx_operators.register
class TanOperator(OnnxOperator):
    def __init__(self):
        super().__init__("Tan")

    def lower(self, ctx: "GgmlOnnxExecutionContext", node: NodeProto) -> None:
        self.lower_numpy_unary(ctx, node, np.tan)


@onnx_operators.register
class TanhOperator(OnnxOperator):
    def __init__(self):
        super().__init__(
            "Tanh",
            OnnxOperator.EXECUTION_NATIVE_OR_NUMPY_RUNTIME,
            OnnxOperator.CLASS_CONDITIONAL_NATIVE,
        )
        self.has_numpy_evaluator = True

    def strategies(
        self, tensor_types: Dict[str, TensorType], node: "NodeIR"
    ) -> Tuple[Tuple[str, str, str], ...]:
        if (
            len(node.inputs) == 1
            and self.tensor_type(tensor_types, node.inputs[0]).is_float32
        ):
            return self.native_strategy()
        return self.numpy_runtime_strategy()

    def eval_numpy(
        self, node: NodeProto, inputs: Tuple[npt.NDArray[Any], ...]
    ) -> Tuple[npt.NDArray[Any], ...]:
        return self.eval_numpy_unary_operator(node, inputs, np.tanh)

    def lower(self, ctx: "GgmlOnnxExecutionContext", node: NodeProto) -> None:
        self.lower_native_unary_or_numpy(ctx, node, ggml.ggml_tanh, np.tanh)


@onnx_operators.register
class ThresholdedReluOperator(OnnxOperator):
    def __init__(self):
        super().__init__("ThresholdedRelu")

    def lower(self, ctx: "GgmlOnnxExecutionContext", node: NodeProto) -> None:
        alpha = next((attr.f for attr in node.attribute if attr.name == "alpha"), 1.0)

        def thresholded_relu(x: npt.NDArray[Any]) -> npt.NDArray[Any]:
            return np.where(x > alpha, x, 0)

        self.lower_numpy_unary(ctx, node, thresholded_relu)


@onnx_operators.register
class TileOperator(OnnxOperator):
    def __init__(self):
        super().__init__(
            "Tile",
            OnnxOperator.EXECUTION_NATIVE_OR_NUMPY_RUNTIME,
            OnnxOperator.CLASS_CONDITIONAL_NATIVE,
        )
        self.has_numpy_evaluator = True

    @staticmethod
    def tile_shapes(
        input_shape: Tuple[int, ...],
        repeats: Sequence[int],
    ) -> Tuple[Tuple[int, ...], Tuple[int, ...]]:
        repeats_tuple = tuple(int(repeat) for repeat in repeats)
        if len(repeats_tuple) > len(input_shape):
            promoted_shape = (1,) * (
                len(repeats_tuple) - len(input_shape)
            ) + input_shape
        else:
            promoted_shape = input_shape
            repeats_tuple = (1,) * (
                len(input_shape) - len(repeats_tuple)
            ) + repeats_tuple
        output_shape = tuple(
            dim * repeat for dim, repeat in zip(promoted_shape, repeats_tuple)
        )
        return promoted_shape, output_shape

    def strategies(
        self, tensor_types: Dict[str, TensorType], node: "NodeIR"
    ) -> Tuple[Tuple[str, str, str], ...]:
        if len(node.inputs) == 2:
            input_type = self.tensor_type(tensor_types, node.inputs[0])
            repeats = self.constant_int_values(tensor_types, node.inputs[1])
            if (
                input_type.is_float32
                and input_type.shape is not None
                and repeats is not None
                and len(repeats) == len(input_type.shape)
                and all(repeat > 0 for repeat in repeats)
            ):
                promoted_shape, output_shape = self.tile_shapes(
                    input_type.shape, repeats
                )
                if self.can_repeat_to_shape(promoted_shape, output_shape):
                    return self.native_strategy()
        return self.numpy_runtime_strategy(
            "Tile requires float32 input and static positive repeats representable "
            "by ggml_repeat to lower to native ggml"
        )

    def eval_numpy(
        self, node: NodeProto, inputs: Tuple[npt.NDArray[Any], ...]
    ) -> Tuple[npt.NDArray[Any], ...]:
        if len(inputs) != 2:
            raise ValueError(f'Operation "{node.op_type}" requires exactly two inputs')
        repeats = tuple(int(repeat) for repeat in inputs[1].astype(np.int64).ravel())
        return (np.asarray(np.tile(inputs[0], repeats), dtype=inputs[0].dtype),)

    def lower(self, ctx: "GgmlOnnxExecutionContext", node: NodeProto) -> None:
        node_inputs = [ctx.ggml_tensors_dict[inp] for inp in node.input]

        if len(node_inputs) != 2:
            raise ValueError(
                f'Error for node "{node.name}": Operation "Tile" requires exactly two inputs. Actual number of inputs: {len(node_inputs)}'
            )

        x, repeats = node_inputs

        input_shape = ctx.shapes[node.input[0]]
        repeats_eval = ctx.eval_tensor(
            repeats,
        )
        repeats_shape = ctx.shapes[node.input[1]]
        repeats_vals = (
            ctx.to_numpy(repeats_eval).reshape(repeats_shape).astype(dtype=np.int64)
        )
        repeats_tuple = tuple(int(repeat) for repeat in repeats_vals.ravel())
        promoted_shape, output_shape = self.tile_shapes(input_shape, repeats_tuple)
        input_dtype = np.dtype(ctx.get_tensor_dtype(node.input[0]))

        if (
            ctx.can_emit_native(node.output[0])
            and ctx.can_run_native(node)
            and input_dtype == np.dtype(np.float32)
            and len(repeats_tuple) == len(input_shape)
            and all(repeat > 0 for repeat in repeats_tuple)
            and self.can_repeat_to_shape(promoted_shape, output_shape)
        ):
            result = self.repeat_native_tensor_to_shape(
                ctx,
                x,
                promoted_shape,
                output_shape,
                input_dtype,
            )
            ctx.register_native_tensor(
                node.output[0], result, output_shape, input_dtype
            )
            return

        x_t = ctx.from_numpy(
            np.empty(output_shape, dtype=input_dtype),
        )

        @ggml.ggml_custom3_op_t
        def custom_tile(
            tensor_out: ggml.ggml_tensor_p,
            tensor_in_1: ggml.ggml_tensor_p,
            tensor_in_2: ggml.ggml_tensor_p,
            tensor_in_3: ggml.ggml_tensor_p,
            ith: int,
            nth: int,
            userdata: Optional[ctypes.c_void_p],
        ):
            x = ctx.to_numpy(tensor_in_2).reshape(input_shape)

            y = np.tile(x, repeats_tuple)
            ctx.set_tensor_data(tensor_out, y)

        new_tensor = ctx.ggml_tensors_dict[node.output[0]] = (
            ggml.ggml_map_custom3_inplace(
                ctx.ggml_eval_context,
                x_t,
                x,
                repeats,
                custom_tile,
                1,
                None,
            )
        )

        ctx.refs.append(custom_tile)
        ctx.register_numpy_runtime_tensor(
            node.output[0], new_tensor, output_shape, input_dtype
        )


@onnx_operators.register
class TopKOperator(OnnxOperator):
    def __init__(self):
        super().__init__("TopK")

    def lower(self, ctx: "GgmlOnnxExecutionContext", node: NodeProto) -> None:
        node_inputs = [ctx.ggml_tensors_dict[inp] for inp in node.input]

        if len(node_inputs) != 2:
            raise ValueError(
                f'Error for node "{node.name}": Operation "TopK" requires exactly two inputs. Actual number of inputs: {len(node_inputs)}'
            )

        x, k = node_inputs

        input_shape = get_tensor_shape(x)

        axis = next((attr.i for attr in node.attribute if attr.name == "axis"), -1)
        largest = next((attr.i for attr in node.attribute if attr.name == "largest"), 1)
        sorted_flag = next(
            (attr.i for attr in node.attribute if attr.name == "sorted"), 0
        )

        k_eval = ctx.eval_tensor(
            k,
        )
        k_np = ctx.to_numpy(k_eval)[0]

        topk_userdata = TopKUserData(axis, largest, sorted_flag, k_np)
        userdata_p = ctypes.cast(ctypes.pointer(topk_userdata), ctypes.c_void_p)

        output_shape = list(input_shape)
        output_shape[axis] = k_np
        output_shape = tuple(output_shape)

        indices_t = ctx.from_numpy(
            np.empty(output_shape, dtype=np.int32),
        )

        values_t = ctx.from_numpy(
            np.empty(output_shape, dtype=get_tensor_dtype(x)),
        )

        @ggml.ggml_custom2_op_t
        def custom_top_k_indices(
            tensor_out: ggml.ggml_tensor_p,
            tensor_in_1: ggml.ggml_tensor_p,
            tensor_in_2: ggml.ggml_tensor_p,
            ith: int,
            nth: int,
            userdata: Optional[ctypes.c_void_p],
        ):
            x = ctx.to_numpy(tensor_in_2)

            userdata_data_ptr = ctypes.cast(userdata, ctypes.POINTER(TopKUserData))
            userdata_data = userdata_data_ptr.contents

            axis = userdata_data.axis
            largest = bool(userdata_data.largest)

            k = userdata_data.k

            if largest:
                sorted_indices = np.argsort(-x, axis=axis, kind="stable")
            else:
                sorted_indices = np.argsort(x, axis=axis, kind="stable")

            topk_indices = np.take(sorted_indices, np.arange(k), axis=axis)

            ctx.set_tensor_data(tensor_out, topk_indices)

        indices = ggml.ggml_map_custom2_inplace(
            ctx.ggml_eval_context,
            indices_t,
            x,
            custom_top_k_indices,
            1,
            userdata_p,
        )

        ctx.refs.append(custom_top_k_indices)

        @ggml.ggml_custom3_op_t
        def custom_top_k_values(
            tensor_out: ggml.ggml_tensor_p,
            tensor_in_1: ggml.ggml_tensor_p,
            tensor_in_2: ggml.ggml_tensor_p,
            tensor_in_3: ggml.ggml_tensor_p,
            ith: int,
            nth: int,
            userdata: Optional[ctypes.c_void_p],
        ):
            x = ctx.to_numpy(tensor_in_2)
            topk_indices = ctx.to_numpy(tensor_in_3).astype(np.int32)

            userdata_data_ptr = ctypes.cast(userdata, ctypes.POINTER(TopKUserData))
            userdata_data = userdata_data_ptr.contents

            axis = userdata_data.axis

            topk_values = np.take_along_axis(x, topk_indices, axis=axis)

            ctx.set_tensor_data(tensor_out, topk_values)

        values = ggml.ggml_map_custom3_inplace(
            ctx.ggml_eval_context,
            values_t,
            x,
            indices,
            custom_top_k_values,
            1,
            userdata_p,
        )

        ctx.refs.append(custom_top_k_values)

        ctx.ggml_tensors_dict[node.output[0]] = values
        ctx.ggml_tensors_dict[node.output[1]] = indices
        ctx.shapes[node.output[0]] = output_shape
        ctx.shapes[node.output[1]] = output_shape
        ctx.set_tensor_shape(values, output_shape)
        ctx.set_tensor_shape(indices, output_shape)

        ctx.refs.append(topk_userdata)

        ctx.set_tensor_dtype(node.output[0], ctx.get_tensor_dtype(node.input[0]))
        ctx.set_tensor_dtype(node.output[1], np.dtype(np.int64))


@onnx_operators.register
class TriluOperator(OnnxOperator):
    def __init__(self):
        super().__init__("Trilu")

    def lower(self, ctx: "GgmlOnnxExecutionContext", node: NodeProto) -> None:
        node_inputs = [ctx.ggml_tensors_dict[inp] for inp in node.input]

        if len(node_inputs) not in {1, 2}:
            raise ValueError(
                f'Error for node "{node.name}": Operation "Trilu" requires one or two inputs. Actual number of inputs: {len(node_inputs)}'
            )

        x_shape = ctx.shapes[node.input[0]]
        x_dtype = get_tensor_dtype(node_inputs[0])
        if 0 in x_shape:
            x = np.empty(x_shape, dtype=x_dtype)
        else:
            x = ctx.to_numpy(ctx.eval_tensor(node_inputs[0])).reshape(x_shape)
        k = 0
        if len(node_inputs) == 2:
            k_shape = ctx.shapes[node.input[1]]
            k = int(
                ctx.to_numpy(ctx.eval_tensor(node_inputs[1])).reshape(k_shape).item()
            )
        upper = next((attr.i for attr in node.attribute if attr.name == "upper"), 1)

        output = (np.triu(x, k) if upper else np.tril(x, k)).astype(x_dtype)
        new_tensor = ctx.ggml_tensors_dict[node.output[0]] = ctx.from_numpy(output)
        ctx.set_tensor_shape(new_tensor, output.shape)
        ctx.shapes[node.output[0]] = output.shape
        ctx.set_tensor_dtype(node.output[0], output.dtype)


@onnx_operators.register
class TransposeOperator(ViewOnnxOperator):
    def __init__(self):
        super().__init__("Transpose", ViewTransformSemantics.KIND_LAYOUT)

    @staticmethod
    def normalize_permutation(
        permutation: Optional[Sequence[int]], rank: int
    ) -> Optional[Tuple[int, ...]]:
        if permutation is None:
            permutation = tuple(reversed(range(rank)))
        if len(permutation) != rank:
            return None
        normalized = tuple(int(axis) for axis in permutation)
        if set(normalized) != set(range(rank)):
            return None
        return normalized

    def validate_view_inputs(
        self, node: NodeProto, node_inputs: Sequence[ggml.ggml_tensor_p]
    ) -> None:
        if len(node_inputs) != 1:
            raise ValueError(
                f'Error for node "{node.name}": Operation "Transpose" requires exactly one input. Actual number of inputs: {len(node_inputs)}'
            )

    def has_static_parameters(
        self,
        tensor_types: Dict[str, TensorType],
        node: "NodeIR",
        input_type: TensorType,
    ) -> bool:
        del tensor_types, node
        return input_type.shape is not None

    def static_permutation(
        self, tensor_types: Dict[str, TensorType], node: "NodeIR"
    ) -> Optional[Tuple[int, ...]]:
        input_type = self.tensor_type(tensor_types, node.inputs[0])
        if input_type.shape is None:
            return None
        return self.normalize_permutation(
            node.attribute("perm"),
            len(input_type.shape),
        )

    def runtime_permutation(
        self,
        ctx: "GgmlOnnxExecutionContext",
        node: NodeProto,
        input_shape: Tuple[int, ...],
    ) -> Tuple[int, ...]:
        del ctx
        permutation = self.normalize_permutation(
            self.ints_attribute(node, "perm"),
            len(input_shape),
        )
        if permutation is None:
            raise ValueError(
                f'Error for node "{node.name}": Invalid "perm" attribute '
                f"for rank {len(input_shape)}"
            )
        return permutation

    def runtime_output_shape(
        self,
        ctx: "GgmlOnnxExecutionContext",
        node: NodeProto,
        input_shape: Tuple[int, ...],
    ) -> Tuple[int, ...]:
        permutation = self.runtime_permutation(ctx, node, input_shape)
        return tuple(input_shape[axis] for axis in permutation)

    def eval_numpy(
        self, node: NodeProto, inputs: Tuple[npt.NDArray[Any], ...]
    ) -> Tuple[npt.NDArray[Any], ...]:
        if len(inputs) != 1:
            raise ValueError(
                f'Operation "{node.op_type}" requires exactly one input. '
                f"Actual number of inputs: {len(inputs)}"
            )
        permutation = self.normalize_permutation(
            self.ints_attribute(node, "perm"),
            inputs[0].ndim,
        )
        if permutation is None:
            raise ValueError(
                f'Operation "{node.op_type}" has invalid "perm" '
                f"for rank {inputs[0].ndim}"
            )
        return (np.transpose(inputs[0], axes=permutation).copy(),)


@onnx_operators.register
class UnsqueezeOperator(ViewOnnxOperator):
    def __init__(self):
        super().__init__("Unsqueeze", ViewTransformSemantics.KIND_SHAPE)

    def validate_view_inputs(
        self, node: NodeProto, node_inputs: Sequence[ggml.ggml_tensor_p]
    ) -> None:
        if len(node_inputs) not in {1, 2}:
            raise ValueError(
                f'Error for node "{node.name}": Operation "Unsqueeze" requires one or two inputs, data and optional axes. Actual number of inputs: {len(node_inputs)}'
            )

    def has_static_parameters(
        self,
        tensor_types: Dict[str, TensorType],
        node: "NodeIR",
        input_type: TensorType,
    ) -> bool:
        del input_type
        return "axes" in node.attributes or (
            len(node.inputs) == 2
            and self.tensor_type(tensor_types, node.inputs[1]).constant
        )

    def axes_from_inputs(
        self, node: NodeProto, inputs: Tuple[npt.NDArray[Any], ...]
    ) -> Tuple[int, ...]:
        if len(inputs) == 2:
            return tuple(int(axis) for axis in np.asarray(inputs[1]).flatten())
        return tuple(self.ints_attribute(node, "axes") or ())

    def axes_from_runtime(
        self, ctx: "GgmlOnnxExecutionContext", node: NodeProto
    ) -> Tuple[int, ...]:
        if len(node.input) == 2 and node.input[1]:
            axes_tensor = ctx.ggml_tensors_dict[node.input[1]]
            return tuple(
                int(axis)
                for axis in ctx.logical_tensor_eval_data(
                    node.input[1],
                    axes_tensor,
                    ctx.shapes[node.input[1]],
                ).flatten()
            )
        return tuple(self.ints_attribute(node, "axes") or ())

    def unsqueeze_shape(
        self, input_shape: Tuple[int, ...], axes: Tuple[int, ...]
    ) -> Tuple[int, ...]:
        output_rank = len(input_shape) + len(axes)
        axes_values = [axis if axis >= 0 else axis + output_rank for axis in axes]
        axes_values.sort()
        shape_probe = np.empty(input_shape, dtype=np.float32)
        for axis in axes_values:
            shape_probe = np.expand_dims(shape_probe, axis=axis)
        return shape_probe.shape

    def runtime_output_shape(
        self,
        ctx: "GgmlOnnxExecutionContext",
        node: NodeProto,
        input_shape: Tuple[int, ...],
    ) -> Tuple[int, ...]:
        return self.unsqueeze_shape(input_shape, self.axes_from_runtime(ctx, node))

    def eval_numpy(
        self, node: NodeProto, inputs: Tuple[npt.NDArray[Any], ...]
    ) -> Tuple[npt.NDArray[Any], ...]:
        if len(inputs) not in {1, 2}:
            raise ValueError(
                f'Operation "{node.op_type}" requires one or two inputs. '
                f"Actual number of inputs: {len(inputs)}"
            )
        output = inputs[0]
        output_rank = output.ndim + len(self.axes_from_inputs(node, inputs))
        axes_values = [
            axis if axis >= 0 else axis + output_rank
            for axis in self.axes_from_inputs(node, inputs)
        ]
        for axis in sorted(axes_values):
            output = np.expand_dims(output, axis=axis)
        return (output,)


@onnx_operators.register
class UniqueOperator(OnnxOperator):
    def __init__(self):
        super().__init__("Unique")

    def lower(self, ctx: "GgmlOnnxExecutionContext", node: NodeProto) -> None:
        node_inputs = [ctx.ggml_tensors_dict[inp] for inp in node.input]

        if len(node_inputs) != 1:
            raise ValueError(
                f'Error for node "{node.name}": Operation "Unique" requires exactly one input. Actual number of inputs: {len(node_inputs)}'
            )

        x_shape = ctx.shapes[node.input[0]]
        x = ctx.to_numpy(ctx.eval_tensor(node_inputs[0])).reshape(x_shape)
        axis_attr = next(
            (attr.i for attr in node.attribute if attr.name == "axis"), None
        )
        sorted_attr = next(
            (attr.i for attr in node.attribute if attr.name == "sorted"), 1
        )
        axis = None
        if axis_attr is not None:
            axis = axis_attr + len(x_shape) if axis_attr < 0 else axis_attr

        y, indices, inverse_indices, counts = np.unique(
            x,
            return_index=True,
            return_inverse=True,
            return_counts=True,
            axis=axis,
        )

        if sorted_attr == 0:
            order = np.argsort(indices)
            remap = np.empty_like(order)
            remap[order] = np.arange(order.shape[0])
            y = np.take(y, order, axis=axis or 0)
            indices = indices[order]
            inverse_indices = remap[inverse_indices]
            counts = counts[order]

        outputs = (
            (node.output[0], y, y.dtype),
            (node.output[1] if len(node.output) > 1 else "", indices, np.int64),
            (
                node.output[2] if len(node.output) > 2 else "",
                inverse_indices,
                np.int64,
            ),
            (node.output[3] if len(node.output) > 3 else "", counts, np.int64),
        )
        for output_name, output_value, output_dtype in outputs:
            if output_name == "":
                continue
            output = np.asarray(output_value, dtype=output_dtype)
            tensor = ctx.ggml_tensors_dict[output_name] = ctx.from_numpy(output)
            ctx.set_tensor_shape(tensor, output.shape)
            ctx.shapes[output_name] = output.shape
            ctx.set_tensor_dtype(output_name, output.dtype)


@onnx_operators.register
class WhereOperator(OnnxOperator):
    def __init__(self):
        super().__init__("Where")

    def lower(self, ctx: "GgmlOnnxExecutionContext", node: NodeProto) -> None:
        node_inputs = [ctx.ggml_tensors_dict[inp] for inp in node.input]

        if len(node_inputs) != 3:
            raise ValueError(
                f'Error for node "{node.name}": Operation "Where" requires exactly three inputs. Actual number of inputs: {len(node_inputs)}'
            )

        c_shape = ctx.shapes[node.input[0]]
        x_shape = ctx.shapes[node.input[1]]
        y_shape = ctx.shapes[node.input[2]]
        output_shape = np.broadcast_shapes(c_shape, x_shape, y_shape)

        condition = ctx.logical_tensor_eval_data(node.input[0], node_inputs[0], c_shape)
        x = ctx.logical_tensor_eval_data(node.input[1], node_inputs[1], x_shape)
        y = ctx.logical_tensor_eval_data(node.input[2], node_inputs[2], y_shape)
        output = np.asarray(np.where(condition, x, y), dtype=np.result_type(x, y))

        ctx.set_logical_output(
            node.output[0], output.reshape(output_shape), output.dtype
        )


@onnx_operators.register
class XorOperator(OnnxOperator):
    def __init__(self):
        super().__init__("Xor")

    def lower(self, ctx: "GgmlOnnxExecutionContext", node: NodeProto) -> None:
        self.lower_numpy_binary(ctx, node, np.logical_xor, np.bool_)


class GgmlOnnxExecutionContext:
    UNSIGNED_TO_SIGNED_DTYPE: ClassVar[Dict[np.dtype[Any], npt.DTypeLike]] = {
        np.dtype(np.uint8): np.int8,
        np.dtype(np.uint16): np.int16,
        np.dtype(np.uint32): np.int32,
        np.dtype(np.uint64): np.int64,
    }

    def __init__(
        self,
        backend: "GgmlBackendRep",
        ggml_tensors_dict: Dict[str, ggml.ggml_tensor_p],
        ggml_eval_context: ggml.ggml_context_p,
        refs: List[Any],
        max_tensors: int,
        shapes: Dict[str, Tuple[int, ...]],
        dtypes: Optional[Dict[str, npt.DTypeLike]] = None,
        native_outputs: Optional[Set[str]] = None,
        execution_by_output: Optional[Dict[str, str]] = None,
        opset_imports: Optional[Dict[str, int]] = None,
    ):
        self.backend = backend
        self.ggml_tensors_dict = ggml_tensors_dict
        self.ggml_eval_context = ggml_eval_context
        self.refs = refs
        self.ggml_tensor_shapes: Dict[int, Tuple[int, ...]] = {}
        self.dtypes: Dict[str, npt.DTypeLike] = dict(dtypes or {})
        self.max_tensors = max_tensors
        # self.ggml_graph = ggml.ggml_new_graph_custom(self.ggml_eval_context, max_tensors, False)
        self.ggml_graph = None
        self.n_threads = 8
        self.shapes = shapes
        self.native_outputs: Set[str] = set(native_outputs or ())
        self.execution_by_output: Dict[str, str] = dict(execution_by_output or {})
        self.opset_imports: Dict[str, int] = dict(opset_imports or {})
        self.tensor_states: Dict[str, TensorState] = {}
        self.backend_buffers: List[Any] = []
        for name, tensor in self.ggml_tensors_dict.items():
            if name in self.shapes:
                self.register_tensor(
                    name,
                    tensor,
                    self.shapes[name],
                    self.get_tensor_dtype(name),
                    storage=TensorState.NATIVE,
                    producer_execution=self.execution_by_output.get(
                        name, OnnxOperator.EXECUTION_NATIVE
                    ),
                )

    def get_opset_version(self, domain: str = "") -> Optional[int]:
        return self.opset_imports.get(domain)

    @staticmethod
    def storage_dtype_for_logical_dtype(dtype: npt.DTypeLike) -> npt.DTypeLike:
        np_dtype = np.dtype(dtype)
        if np_dtype == np.dtype(np.bool_):
            return np.int32
        if np_dtype in GgmlOnnxExecutionContext.UNSIGNED_TO_SIGNED_DTYPE:
            return GgmlOnnxExecutionContext.UNSIGNED_TO_SIGNED_DTYPE[np_dtype]
        if np_dtype.type in ggml.utils.NUMPY_DTYPE_TO_GGML_TYPE:
            return np_dtype
        return np.float32

    @staticmethod
    def map_to_ggml_type(dtype: npt.DTypeLike):
        np_dtype = np.dtype(dtype)
        if np_dtype == np.dtype(np.bool_):
            return ggml.utils.GGML_TYPE.I32

        storage_dtype = GgmlOnnxExecutionContext.UNSIGNED_TO_SIGNED_DTYPE.get(
            np_dtype, np_dtype
        )
        ggml_type = ggml.utils.NUMPY_DTYPE_TO_GGML_TYPE.get(
            np.dtype(storage_dtype).type,
            ggml.utils.GGML_TYPE.F32,
        )
        return ggml_type

    def set_tensor_shape(self, tensor: ggml.ggml_tensor_p, shape: Tuple[int, ...]):
        key = ctypes.addressof(tensor.contents)
        self.ggml_tensor_shapes[key] = shape

    def get_tensor_shape(self, tensor: ggml.ggml_tensor_p) -> Tuple[int, ...]:
        key = ctypes.addressof(tensor.contents)
        if key not in self.ggml_tensor_shapes:
            self.ggml_tensor_shapes[key] = get_tensor_shape(tensor)
        return self.ggml_tensor_shapes[key]

    def set_tensor_dtype(self, name: str, dtype: npt.DTypeLike):
        self.dtypes[name] = dtype

    def get_tensor_dtype(self, name: str) -> npt.DTypeLike:
        tensor_dtype = get_tensor_dtype(self.ggml_tensors_dict[name])
        return self.dtypes.get(name, tensor_dtype)

    def get_raw_tensor_dtype(self, tensor: ggml.ggml_tensor_p) -> npt.DTypeLike:
        return get_tensor_dtype(tensor)

    def register_tensor(
        self,
        name: str,
        tensor: ggml.ggml_tensor_p,
        shape: Tuple[int, ...],
        dtype: Optional[npt.DTypeLike] = None,
        storage: Optional[str] = None,
        producer_execution: Optional[str] = None,
        source: Optional[str] = None,
    ) -> ggml.ggml_tensor_p:
        producer = producer_execution or self.execution_by_output.get(
            name, OnnxOperator.EXECUTION_NATIVE
        )
        if storage is None:
            storage = (
                TensorState.NUMPY
                if producer in OnnxOperator.NUMPY_FALLBACK_EXECUTIONS
                else TensorState.NATIVE
            )
        self.ggml_tensors_dict[name] = tensor
        self.shapes[name] = tuple(shape)
        self.set_tensor_shape(tensor, tuple(shape))
        if dtype is None:
            dtype = get_tensor_dtype(tensor)
        self.set_tensor_dtype(name, dtype)
        self.tensor_states[name] = TensorState(
            name=name,
            tensor=tensor,
            shape=tuple(shape),
            dtype=np.dtype(dtype),
            storage=storage,
            producer_execution=producer,
            source=source,
        )
        return tensor

    def register_native_tensor(
        self,
        name: str,
        tensor: ggml.ggml_tensor_p,
        shape: Tuple[int, ...],
        dtype: npt.DTypeLike,
    ) -> ggml.ggml_tensor_p:
        return self.register_tensor(
            name,
            tensor,
            shape,
            dtype,
            storage=TensorState.NATIVE,
            producer_execution=OnnxOperator.EXECUTION_NATIVE,
        )

    def register_decomposed_tensor(
        self,
        name: str,
        tensor: ggml.ggml_tensor_p,
        shape: Tuple[int, ...],
        dtype: npt.DTypeLike,
    ) -> ggml.ggml_tensor_p:
        return self.register_tensor(
            name,
            tensor,
            shape,
            dtype,
            storage=TensorState.NATIVE,
            producer_execution=OnnxOperator.EXECUTION_DECOMPOSED,
        )

    def register_native_view_tensor(
        self,
        name: str,
        tensor: ggml.ggml_tensor_p,
        shape: Tuple[int, ...],
        dtype: npt.DTypeLike,
        source: str,
    ) -> ggml.ggml_tensor_p:
        return self.register_tensor(
            name,
            tensor,
            shape,
            dtype,
            storage=TensorState.VIEW,
            producer_execution=OnnxOperator.EXECUTION_NATIVE,
            source=source,
        )

    def register_numpy_runtime_tensor(
        self,
        name: str,
        tensor: ggml.ggml_tensor_p,
        shape: Tuple[int, ...],
        dtype: npt.DTypeLike,
    ) -> ggml.ggml_tensor_p:
        return self.register_tensor(
            name,
            tensor,
            shape,
            dtype,
            storage=TensorState.NUMPY,
            producer_execution=OnnxOperator.EXECUTION_NUMPY_RUNTIME,
        )

    def register_numpy_eager_tensor(
        self,
        name: str,
        tensor: ggml.ggml_tensor_p,
        shape: Tuple[int, ...],
        dtype: npt.DTypeLike,
    ) -> ggml.ggml_tensor_p:
        return self.register_tensor(
            name,
            tensor,
            shape,
            dtype,
            storage=TensorState.NUMPY,
            producer_execution=OnnxOperator.EXECUTION_NUMPY_EAGER,
        )

    def alias_numpy_runtime_tensor(
        self,
        output_name: str,
        input_name: str,
        shape: Tuple[int, ...],
    ) -> ggml.ggml_tensor_p:
        return self.alias_tensor(
            output_name,
            input_name,
            shape,
            storage=TensorState.NUMPY,
            producer_execution=OnnxOperator.EXECUTION_NUMPY_RUNTIME,
        )

    def alias_tensor(
        self,
        output_name: str,
        input_name: str,
        shape: Tuple[int, ...],
        storage: str = TensorState.VIEW,
        producer_execution: Optional[str] = None,
    ) -> ggml.ggml_tensor_p:
        input_state = self.tensor_state(input_name)
        producer = (
            producer_execution
            if producer_execution is not None
            else input_state.producer_execution
        )
        self.ggml_tensors_dict[output_name] = input_state.tensor
        self.shapes[output_name] = tuple(shape)
        self.set_tensor_dtype(output_name, input_state.dtype)
        self.tensor_states[output_name] = TensorState(
            name=output_name,
            tensor=input_state.tensor,
            shape=tuple(shape),
            dtype=np.dtype(input_state.dtype),
            storage=storage,
            producer_execution=producer,
            source=input_name,
        )
        return input_state.tensor

    def logical_tensor_data(
        self,
        name: str,
        tensor: ggml.ggml_tensor_p,
        shape: Tuple[int, ...],
    ) -> npt.NDArray[Any]:
        dtype = np.dtype(self.get_tensor_dtype(name))
        if any(dim == 0 for dim in shape):
            return np.empty(shape, dtype=dtype)
        array = self.to_numpy(tensor).reshape(shape)
        if dtype == np.dtype(np.float16) and array.dtype == np.dtype(np.uint16):
            return array.view(np.float16)
        if array.dtype != dtype:
            return array.astype(dtype)
        return array

    def logical_tensor_eval_data(
        self,
        name: str,
        tensor: ggml.ggml_tensor_p,
        shape: Tuple[int, ...],
    ) -> npt.NDArray[Any]:
        dtype = np.dtype(self.get_tensor_dtype(name))
        if any(dim == 0 for dim in shape):
            return np.empty(shape, dtype=dtype)
        array = self.to_numpy(self.eval_tensor(tensor)).reshape(shape)
        if dtype == np.dtype(np.float16) and array.dtype == np.dtype(np.uint16):
            return array.view(np.float16)
        if array.dtype != dtype:
            return array.astype(dtype)
        return array

    def set_logical_output(
        self,
        name: str,
        array: npt.NDArray[Any],
        dtype: npt.DTypeLike,
    ) -> ggml.ggml_tensor_p:
        logical_dtype = np.dtype(dtype)
        logical_array = np.asarray(array, dtype=logical_dtype)
        storage_dtype = self.storage_dtype_for_logical_dtype(logical_dtype)
        storage_array = np.asarray(logical_array, dtype=storage_dtype)
        tensor = self.from_numpy(storage_array)
        return self.register_numpy_eager_tensor(
            name, tensor, logical_array.shape, logical_dtype
        )

    def set_numpy_runtime_output(
        self,
        name: str,
        array: npt.NDArray[Any],
        dtype: npt.DTypeLike,
    ) -> ggml.ggml_tensor_p:
        logical_dtype = np.dtype(dtype)
        logical_array = np.asarray(array, dtype=logical_dtype)
        storage_dtype = self.storage_dtype_for_logical_dtype(logical_dtype)
        storage_array = np.asarray(logical_array, dtype=storage_dtype)
        tensor = self.from_numpy(storage_array)
        return self.register_numpy_runtime_tensor(
            name, tensor, logical_array.shape, logical_dtype
        )

    def finalize_node_outputs(self, node: NodeProto):
        for output_name in node.output:
            if not output_name or output_name in self.tensor_states:
                continue
            if output_name not in self.ggml_tensors_dict:
                raise RuntimeError(
                    f'Operator "{node.op_type}" did not produce declared output "{output_name}"'
                )
            tensor = self.ggml_tensors_dict[output_name]
            shape = self.shapes.get(output_name, self.get_tensor_shape(tensor))
            dtype = self.dtypes.get(output_name, get_tensor_dtype(tensor))
            producer = self.execution_by_output.get(
                output_name, OnnxOperator.EXECUTION_NATIVE
            )
            storage = (
                TensorState.NUMPY
                if producer in OnnxOperator.NUMPY_FALLBACK_EXECUTIONS
                else TensorState.NATIVE
            )
            self.register_tensor(
                output_name,
                tensor,
                shape,
                dtype,
                storage=storage,
                producer_execution=producer,
            )

    def tensor_state(self, name: str) -> TensorState:
        state = self.tensor_states.get(name)
        if state is not None:
            return state
        tensor = self.ggml_tensors_dict[name]
        shape = self.shapes.get(name, self.get_tensor_shape(tensor))
        producer = self.execution_by_output.get(name, OnnxOperator.EXECUTION_NATIVE)
        storage = (
            TensorState.NUMPY
            if producer in OnnxOperator.NUMPY_FALLBACK_EXECUTIONS
            else TensorState.NATIVE
        )
        self.register_tensor(
            name,
            tensor,
            shape,
            self.get_tensor_dtype(name),
            storage=storage,
            producer_execution=producer,
        )
        return self.tensor_states[name]

    def can_consume_native(self, name: str) -> bool:
        return self.tensor_state(name).can_consume_native

    def can_emit_native(self, name: str) -> bool:
        return name in self.native_outputs

    def can_run_native(self, node: NodeProto) -> bool:
        return all(
            self.can_consume_native(input_name)
            for input_name in node.input
            if input_name
        )

    def to_numpy(self, tensor: ggml.ggml_tensor_p) -> npt.NDArray[Any]:
        shape = self.get_tensor_shape(tensor)
        np_dtype = get_tensor_dtype(tensor)
        storage_shape = ggml.utils.get_shape(tensor)
        array = np.empty(storage_shape, dtype=np_dtype)
        tensor_nbytes = ggml.ggml_nbytes(tensor)
        if tensor_nbytes:
            ggml.ggml_backend_tensor_get(
                tensor,
                array.ctypes.data_as(ctypes.c_void_p),
                0,
                tensor_nbytes,
            )
        return array.reshape(shape)

    @staticmethod
    def storage_array(array: npt.NDArray[Any]) -> npt.NDArray[Any]:
        storage_array = np.asarray(array)
        storage_dtype = np.dtype(storage_array.dtype)
        if storage_dtype == np.dtype(np.bool_):
            storage_dtype = np.dtype(np.int32)
        elif storage_dtype in GgmlOnnxExecutionContext.UNSIGNED_TO_SIGNED_DTYPE:
            storage_dtype = np.dtype(
                GgmlOnnxExecutionContext.UNSIGNED_TO_SIGNED_DTYPE[storage_dtype]
            )
        if storage_dtype != storage_array.dtype:
            storage_array = storage_array.astype(storage_dtype)

        storage_shape = storage_array.shape
        if not storage_shape:
            storage_shape = (1,)
        elif len(storage_shape) > ViewTransformSemantics.GGML_MAX_DIMS:
            storage_shape = (int(np.prod(storage_shape)),)
        if storage_shape != storage_array.shape:
            storage_array = storage_array.reshape(storage_shape)
        return np.ascontiguousarray(storage_array)

    @staticmethod
    def alloc_tensor_in_backend_buffer(
        buffer: ggml.ggml_backend_buffer_t, tensor: ggml.ggml_tensor_p
    ):
        base = ggml.ggml_backend_buffer_get_base(buffer)
        if base is None:
            raise RuntimeError("Failed to get GGML backend buffer base")
        alignment = ggml.ggml_backend_buffer_get_alignment(buffer)
        offset = (alignment - (base % alignment)) % alignment
        ggml.ggml_backend_tensor_alloc(buffer, tensor, ctypes.c_void_p(base + offset))

    def alloc_backend_buffer_for_tensor(
        self, tensor: ggml.ggml_tensor_p
    ) -> ggml.ggml_backend_buffer_t:
        buft = ggml.ggml_backend_get_default_buffer_type(self.backend.ggml_backend)
        if buft is None:
            raise RuntimeError("Failed to get GGML backend buffer type")
        alloc_size = ggml.ggml_backend_buft_get_alloc_size(buft, tensor)
        buffer = ggml.ggml_backend_buft_alloc_buffer(buft, alloc_size)
        if buffer is None:
            raise RuntimeError("Failed to allocate GGML backend buffer")
        self.alloc_tensor_in_backend_buffer(buffer, tensor)
        return buffer

    def from_numpy(self, array: npt.NDArray[Any]) -> ggml.ggml_tensor_p:
        array = np.asarray(array)
        logical_shape = array.shape
        storage_array = self.storage_array(array)
        tensor = ggml.utils.from_numpy(storage_array, self.ggml_eval_context)

        ggml.ggml_set_input(tensor)
        ggml.ggml_set_output(tensor)

        if array.size > 0:
            tensor_buffer = self.alloc_backend_buffer_for_tensor(tensor)
            self.refs.append(tensor)
            self.refs.append(tensor_buffer)
            self.backend_buffers.append(tensor_buffer)
            self.set_tensor_data(tensor, storage_array)

        if storage_array.shape != logical_shape and logical_shape:
            self.set_tensor_shape(tensor, logical_shape)

        return tensor

    def eval_tensor(self, tensor: ggml.ggml_tensor_p):
        if not tensor.contents.src[0]:
            return tensor
        if ggml.ggml_nbytes(tensor) == 0:
            return tensor
        tensor_to_eval = tensor
        if not ggml.ggml_is_contiguous(tensor):
            tensor_to_eval = ggml.ggml_cont(self.ggml_eval_context, tensor)

        self.ggml_graph = ggml.ggml_new_graph_custom(
            self.ggml_eval_context, self.max_tensors, False
        )
        ggml.ggml_set_output(tensor_to_eval)
        ggml.ggml_build_forward_expand(self.ggml_graph, tensor_to_eval)

        default_buffer_type = ggml.ggml_backend_get_default_buffer_type(
            self.backend.ggml_backend
        )
        assert default_buffer_type is not None
        gallocr = ggml.ggml_gallocr_new(default_buffer_type)
        if gallocr is None:
            raise RuntimeError("Failed to create GGML graph allocator")
        try:
            if not ggml.ggml_gallocr_alloc_graph(gallocr, self.ggml_graph):
                raise RuntimeError("Failed to allocate GGML graph")

            if (
                ggml.ggml_backend_graph_compute(
                    self.backend.ggml_backend, self.ggml_graph
                )
                != ggml.GGML_STATUS_SUCCESS
            ):
                raise RuntimeError("Failed to compute GGML graph")

            tensor_copy = ggml.ggml_dup_tensor(self.ggml_eval_context, tensor_to_eval)
            tensor_copy_buffer = self.alloc_backend_buffer_for_tensor(tensor_copy)
            tensor_nbytes = ggml.ggml_nbytes(tensor_to_eval)
            if tensor_nbytes:
                storage_array = np.empty(
                    ggml.utils.get_shape(tensor_to_eval),
                    dtype=get_tensor_dtype(tensor_to_eval),
                )
                ggml.ggml_backend_tensor_get(
                    tensor_to_eval,
                    storage_array.ctypes.data_as(ctypes.c_void_p),
                    0,
                    tensor_nbytes,
                )
                ggml.ggml_backend_tensor_set(
                    tensor_copy,
                    storage_array.ctypes.data_as(ctypes.c_void_p),
                    0,
                    tensor_nbytes,
                )
            self.set_tensor_shape(tensor_copy, self.get_tensor_shape(tensor_to_eval))
            self.refs.append(tensor_copy)
            self.refs.append(tensor_copy_buffer)
            self.backend_buffers.append(tensor_copy_buffer)
            return tensor_copy
        finally:
            ggml.ggml_gallocr_free(gallocr)

    def set_tensor_data(self, tensor: ggml.ggml_tensor_p, array: npt.NDArray[Any]):
        tensor_dtype = get_tensor_dtype(tensor)
        array = np.asarray(array)

        if tensor_dtype == np.dtype(np.uint16) and array.dtype == np.float16:
            array = array.view(np.uint16)
        elif array.dtype != tensor_dtype:
            array = array.astype(tensor_dtype)

        array = np.ascontiguousarray(array)
        tensor_nbytes = ggml.ggml_nbytes(tensor)
        if array.nbytes != tensor_nbytes:
            raise ValueError(
                f"Cannot copy {array.nbytes} bytes into GGML tensor with {tensor_nbytes} bytes"
            )

        ggml.ggml_backend_tensor_set(
            tensor,
            array.ctypes.data_as(ctypes.c_void_p),
            0,
            tensor_nbytes,
        )

    def free_backend_buffers(self):
        while self.backend_buffers:
            ggml.ggml_backend_buffer_free(self.backend_buffers.pop())


class GgmlBackendRep(BackendRep):
    def __init__(
        self,
        graph: GraphProto,
        weights: Dict[str, ggml.ggml_tensor_p],
        inputs: Sequence[ValueInfoProto],
        outputs: Sequence[ValueInfoProto],
        shapes: Dict[str, Tuple[int, ...]],
        dtypes: Dict[str, npt.DTypeLike],
        ggml_context: ggml.ggml_context_p,
        ggml_init_params: ggml.ggml_init_params,
        ggml_backend: ggml.ggml_backend_t,
        ggml_weights_buffer: Any,
        execution_plan: ExecutionPlan,
        ir_pipeline: OnnxRuntimePipeline,
        opset_imports: Dict[str, int],
    ):
        super(GgmlBackendRep, self).__init__()
        self.graph = graph
        self.weights = weights
        self.inputs = inputs
        self.outputs = outputs
        self.shapes = shapes
        self.dtypes = dtypes
        self.ggml_context = ggml_context
        self.ggml_init_params = ggml_init_params
        self.ggml_backend = ggml_backend
        self.ggml_weights_buffer = ggml_weights_buffer
        self.execution_plan = execution_plan
        self.ir_pipeline = ir_pipeline
        self.opset_imports = opset_imports
        self.last_numpy_fallback_island_executions: Tuple[FallbackIsland, ...] = ()

    @property
    def fallback_nodes(self) -> Tuple[ExecutionPlanNode, ...]:
        return self.execution_plan.fallback_nodes

    @property
    def native_nodes(self) -> Tuple[ExecutionPlanNode, ...]:
        return self.execution_plan.native_nodes

    @property
    def coverage_report(self) -> ExecutionCoverageReport:
        return self.execution_plan.coverage_report()

    def __del__(self):
        ggml_weights_buffer = getattr(self, "ggml_weights_buffer", None)
        if ggml_weights_buffer is not None:
            ggml.ggml_backend_buffer_free(ggml_weights_buffer)
            self.ggml_weights_buffer = None
        ggml_context = getattr(self, "ggml_context", None)
        if ggml_context is not None:
            ggml.ggml_free(ggml_context)
            self.ggml_context = None
        ggml_backend = getattr(self, "ggml_backend", None)
        if ggml_backend is not None:
            ggml.ggml_backend_free(ggml_backend)
            self.ggml_backend = None

    @staticmethod
    def _is_list_of_arraylike(x: Any) -> TypeGuard[List[npt.ArrayLike]]:
        return isinstance(x, list) and all(isinstance(y, (np.ndarray, list)) for y in x)

    @staticmethod
    def _is_dict_of_arraylike(x: Any) -> TypeGuard[Dict[str, npt.ArrayLike]]:
        return (
            isinstance(x, dict)
            and all(isinstance(y, (np.ndarray, list)) for y in x.values())
            and all(isinstance(k, str) for k in x.keys())
        )

    def run(self, inputs: Any, **kwargs: Any) -> Tuple[Any, ...]:
        """Run the model with the specified inputs."""

        if self._is_list_of_arraylike(inputs):
            inputs = {k.name: v for k, v in zip(self.inputs, inputs)}

        if not self._is_dict_of_arraylike(inputs):
            raise TypeError("inputs must be a dict of input names to array-like values")

        model_graph = self.graph
        exit_node = None
        ggml_tensors = self.weights.copy()
        shapes = self.shapes.copy()
        dtypes = self.dtypes.copy()
        self.last_numpy_fallback_island_executions = ()

        cleanup = contextlib.ExitStack()
        try:
            ggml_input_context = ggml.ggml_init(
                ggml.ggml_init_params(
                    mem_size=2
                    * ggml.GGML_DEFAULT_GRAPH_SIZE
                    * ggml.ggml_tensor_overhead(),  # FIXME: Reduce to n inputs or combine with tensors context
                    no_alloc=True,
                )
            )
            if ggml_input_context is None:
                raise RuntimeError("Failed to initialize GGML input context")
            cleanup.callback(ggml.ggml_free, ggml_input_context)

            # Create entry inputs
            input_tensors: List[str] = []
            for model_input in model_graph.input:
                input_name = model_input.name

                # Check if the input includes expected values
                if input_name not in inputs:
                    if input_name in ggml_tensors:
                        continue
                    raise KeyError(f'"{input_name}" must be included in the inputs.')

                input_data = np.array(inputs[input_name])
                shapes[input_name] = input_data.shape

                # Check for rank of input
                expected_rank = len(list(model_input.type.tensor_type.shape.dim))
                actual_rank = input_data.ndim

                if expected_rank != actual_rank:
                    raise ValueError(
                        f"INVALID_ARGUMENT : Invalid rank for input: {input_name} Got: {actual_rank} Expected: {expected_rank} Please fix either the inputs or the model."
                    )

                # Check for input types
                expected_dtype = np.dtype(
                    GgmlRuntimeBackend.ONNX_DTYPE_MAP[
                        model_input.type.tensor_type.elem_type
                    ]
                )
                dtypes[input_name] = expected_dtype

                if input_data.dtype != expected_dtype:
                    raise ValueError(
                        f'INVALID_ARGUMENT : Unexpected input data type for "{input_name}". Actual: {input_data.dtype}, expected: {expected_dtype}'
                    )

                # Create the input tensors with the correct type/shape
                ggml_type = GgmlOnnxExecutionContext.map_to_ggml_type(input_data.dtype)
                storage_shape = input_data.shape
                if not storage_shape:
                    storage_shape = (1,)
                elif len(storage_shape) > ViewTransformSemantics.GGML_MAX_DIMS:
                    storage_shape = (int(np.prod(storage_shape)),)
                shape = tuple(reversed(storage_shape))

                tensor = ggml.ggml_new_tensor(
                    ggml_input_context,
                    ggml_type.value,
                    len(shape),
                    (ctypes.c_int64 * len(shape))(*shape),
                )
                ggml.ggml_set_input(tensor)
                ggml.ggml_set_output(tensor)

                ggml_tensors[input_name] = tensor
                input_tensors.append(input_name)

            ggml_input_buffer = None
            if input_tensors:
                ggml_input_buffer = ggml.ggml_backend_alloc_ctx_tensors(
                    ggml_input_context,
                    self.ggml_backend,
                )

            if input_tensors and ggml_input_buffer is None:
                if any(ggml.ggml_nbytes(ggml_tensors[name]) for name in input_tensors):
                    raise RuntimeError("Failed to allocate GGML input buffer")
            if ggml_input_buffer is not None:
                cleanup.callback(ggml.ggml_backend_buffer_free, ggml_input_buffer)

            # Set user inputs
            for key, value in inputs.items():
                tensor = ggml_tensors[key]
                tensor_dtype = get_tensor_dtype(tensor)
                array = np.asarray(np.array(value))

                if tensor_dtype == np.dtype(np.uint16) and array.dtype == np.float16:
                    array = array.view(np.uint16)
                elif array.dtype != tensor_dtype:
                    array = array.astype(tensor_dtype)

                array = np.ascontiguousarray(array)
                tensor_nbytes = ggml.ggml_nbytes(tensor)
                if array.nbytes != tensor_nbytes:
                    raise ValueError(
                        f"Cannot copy {array.nbytes} bytes into GGML tensor with {tensor_nbytes} bytes"
                    )

                if tensor_nbytes:
                    ggml.ggml_backend_tensor_set(
                        tensor,
                        array.ctypes.data_as(ctypes.c_void_p),
                        0,
                        tensor_nbytes,
                    )

            # Define context
            max_tensors = 8192
            max_overhead = (
                ggml.ggml_tensor_overhead() * max_tensors
                + ggml.ggml_graph_overhead_custom(max_tensors, False)
            )
            mem_buffer = (ctypes.c_uint8 * max_overhead)()
            ggml_eval_context = ggml.ggml_init(
                ggml.ggml_init_params(
                    mem_size=max_overhead,
                    mem_buffer=ctypes.cast(mem_buffer, ctypes.c_void_p),
                    no_alloc=True,
                )
            )
            if ggml_eval_context is None:
                raise RuntimeError("Failed to initialize GGML context")
            cleanup.callback(ggml.ggml_free, ggml_eval_context)

            refs: List[Any] = []
            refs.append(mem_buffer)

            ctx = GgmlOnnxExecutionContext(
                self,
                ggml_tensors,
                ggml_eval_context,
                refs,
                max_tensors,
                shapes,
                dtypes,
                {
                    output
                    for plan_node in self.execution_plan.native_nodes
                    for output in plan_node.outputs
                },
                {
                    output: plan_node.execution
                    for plan_node in self.execution_plan.nodes
                    for output in plan_node.outputs
                    if output
                },
                self.opset_imports,
            )
            cleanup.callback(ctx.free_backend_buffers)
            for input_name in input_tensors:
                ctx.set_tensor_shape(ggml_tensors[input_name], shapes[input_name])

            # Build layers
            model_nodes = list(model_graph.node)
            fallback_executor = NumpyFallbackExecutor()
            fallback_islands_by_start = {
                island.node_indices[0]: island
                for island in self.execution_plan.fallback_islands
                if island.node_indices
            }
            node_index = 0
            while node_index < len(model_nodes):
                node = model_nodes[node_index]
                fallback_island = fallback_islands_by_start.get(node_index)
                if fallback_island is not None:
                    island_nodes = model_nodes[
                        node_index : node_index + len(fallback_island.nodes)
                    ]
                    if fallback_executor.can_execute_island(
                        fallback_island, island_nodes
                    ):
                        fallback_executor.execute_island(
                            ctx,
                            fallback_island,
                            island_nodes,
                        )
                        for island_node in island_nodes:
                            ctx.finalize_node_outputs(island_node)
                        node_index += len(island_nodes)
                        continue

                operator_spec = onnx_operators.get(node.op_type, node.domain)
                if operator_spec is None:
                    raise NotImplementedError(
                        f'Operator "{node.op_type}" not implemented'
                    )
                operator_spec.lower(ctx, node)
                ctx.finalize_node_outputs(node)
                node_index += 1

            graph_outputs: List[npt.NDArray[Any]] = []
            for output in self.outputs:
                exit_node = ctx.eval_tensor(ggml_tensors[output.name])
                shape = ctx.shapes.get(output.name, ctx.get_tensor_shape(exit_node))
                # NOTE: 0 dimension in ggml may cause bugs
                max_tensors = np.prod(shape)  # type: ignore
                graph_output: npt.NDArray[Any] = (
                    ctx.logical_tensor_data(output.name, exit_node, shape)
                    if max_tensors > 0
                    else np.empty(shape, dtype=ctx.get_tensor_dtype(output.name))
                )

                graph_outputs.append(graph_output)

            self.last_numpy_fallback_island_executions = tuple(
                fallback_executor.executed_islands
            )
            return graph_outputs
        finally:
            cleanup.close()


class GgmlRuntimeBackend(Backend):
    try:
        ONNX_DTYPE_MAP: ClassVar[Dict[int, npt.DTypeLike]] = {
            elem_type: np_dtype
            for elem_type, np_dtype in onnx.mapping.TENSOR_TYPE_TO_NP_TYPE.items()  # type: ignore
        }
    except AttributeError:
        ONNX_DTYPE_MAP = {
            elem_type: mapping.np_dtype
            for elem_type, mapping in onnx._mapping.TENSOR_TYPE_MAP.items()  # type: ignore[attr-defined]
        }

    @staticmethod
    def _value_info_shape(value_info: ValueInfoProto) -> Tuple[Any, ...]:
        if not value_info.type.HasField("tensor_type"):
            return ()
        shape = value_info.type.tensor_type.shape
        dims = []
        for dim in shape.dim:
            if dim.HasField("dim_value"):
                dims.append(int(dim.dim_value))
            elif dim.HasField("dim_param"):
                dims.append(dim.dim_param)
            else:
                dims.append(None)
        return tuple(dims)

    @staticmethod
    def _value_info_dtype(value_info: ValueInfoProto) -> Optional[npt.DTypeLike]:
        if not value_info.type.HasField("tensor_type"):
            return None
        elem_type = value_info.type.tensor_type.elem_type
        if elem_type == TensorProto.UNDEFINED:
            return None
        return GgmlRuntimeBackend.ONNX_DTYPE_MAP.get(elem_type)

    @staticmethod
    def _tensor_proto_dtype(tensor: TensorProto) -> Optional[npt.DTypeLike]:
        if tensor.data_type == TensorProto.UNDEFINED:
            return None
        return GgmlRuntimeBackend.ONNX_DTYPE_MAP.get(tensor.data_type)

    @staticmethod
    def _tensor_proto_scalar_value(tensor: TensorProto) -> Optional[Any]:
        tensor_size = int(np.prod(tensor.dims)) if tensor.dims else 1
        if tensor_size != 1:
            return None
        try:
            array = onnx.numpy_helper.to_array(tensor)
        except Exception:
            return None
        if array.size != 1:
            return None
        return array.reshape(()).item()

    @staticmethod
    def _tensor_proto_constant_value(
        tensor: TensorProto, max_elements: int = 64
    ) -> Optional[npt.NDArray[Any]]:
        tensor_size = int(np.prod(tensor.dims)) if tensor.dims else 1
        if tensor_size > max_elements:
            return None
        try:
            return np.asarray(onnx.numpy_helper.to_array(tensor))
        except Exception:
            return None

    @classmethod
    def _constant_node_tensor_info(cls, node: NodeProto) -> Optional[TensorInfo]:
        tensor = ConstantOperator.tensor_proto_from_node(node)
        if tensor is None:
            return None
        return TensorInfo(
            name=tensor.name,
            shape=tuple(tensor.dims),
            dtype=cls._tensor_proto_dtype(tensor),
            constant=True,
            scalar_value=cls._tensor_proto_scalar_value(tensor),
            constant_value=cls._tensor_proto_constant_value(tensor),
        )

    @staticmethod
    def _attribute_value(attr: onnx.AttributeProto) -> Any:
        value = onnx.helper.get_attribute_value(attr)
        if isinstance(value, list):
            return tuple(value)
        return value

    @classmethod
    def fold_constant_nodes(cls, model: ModelProto) -> ModelProto:
        graph = model.graph
        initializer_names = {initializer.name for initializer in graph.initializer}
        folded_initializers: List[TensorProto] = []
        remaining_nodes: List[NodeProto] = []

        for node in graph.node:
            tensor = ConstantOperator.tensor_proto_from_node(node)
            if tensor is None or tensor.name in initializer_names:
                remaining_nodes.append(node)
                continue
            folded_initializers.append(tensor)
            initializer_names.add(tensor.name)

        if not folded_initializers:
            return model

        folded_model = ModelProto()
        folded_model.CopyFrom(model)
        del folded_model.graph.node[:]
        folded_model.graph.node.extend(remaining_nodes)
        folded_model.graph.initializer.extend(folded_initializers)
        return folded_model

    @classmethod
    def fold_static_cast_nodes(cls, model: ModelProto) -> ModelProto:
        graph = model.graph
        initializer_by_name = {
            initializer.name: initializer for initializer in graph.initializer
        }
        initializer_names = set(initializer_by_name)
        folded_initializers: List[TensorProto] = []
        remaining_nodes: List[NodeProto] = []

        for node in graph.node:
            if (
                node.op_type != "Cast"
                or node.domain
                or len(node.input) != 1
                or len(node.output) != 1
                or node.input[0] not in initializer_by_name
                or node.output[0] in initializer_names
            ):
                remaining_nodes.append(node)
                continue

            to_attr = next((attr for attr in node.attribute if attr.name == "to"), None)
            if to_attr is None:
                remaining_nodes.append(node)
                continue

            try:
                target_dtype = np.dtype(tensor_dtype_to_np_dtype(to_attr.i))
                casted = onnx.numpy_helper.to_array(
                    initializer_by_name[node.input[0]]
                ).astype(target_dtype)
                tensor = onnx.numpy_helper.from_array(casted, name=node.output[0])
            except (TypeError, ValueError):
                remaining_nodes.append(node)
                continue

            folded_initializers.append(tensor)
            initializer_by_name[node.output[0]] = tensor
            initializer_names.add(node.output[0])

        if not folded_initializers:
            return model

        folded_model = ModelProto()
        folded_model.CopyFrom(model)
        del folded_model.graph.node[:]
        folded_model.graph.node.extend(remaining_nodes)
        folded_model.graph.initializer.extend(folded_initializers)
        return folded_model

    @classmethod
    def _static_tensor_shape(
        cls, model_ir: ModelIR, name: str
    ) -> Optional[Tuple[int, ...]]:
        tensor_info = model_ir.tensor(name)
        if tensor_info is None:
            return None
        if not all(isinstance(dim, int) for dim in tensor_info.shape):
            return None
        return tuple(int(dim) for dim in tensor_info.shape)

    @staticmethod
    def _tensor_dtype(model_ir: ModelIR, name: str) -> Optional[np.dtype[Any]]:
        tensor_info = model_ir.tensor(name)
        if tensor_info is None or tensor_info.dtype is None:
            return None
        return np.dtype(tensor_info.dtype)

    @staticmethod
    def _tensor_scalar_value(model_ir: ModelIR, name: str) -> Optional[Any]:
        tensor_info = model_ir.tensor(name)
        if tensor_info is None:
            return None
        return tensor_info.scalar_value

    @staticmethod
    def _tensor_is_constant(model_ir: ModelIR, name: str) -> bool:
        tensor_info = model_ir.tensor(name)
        return bool(tensor_info is not None and tensor_info.constant)

    @classmethod
    def _resolve_node_operator(
        cls,
        tensor_types: Dict[str, TensorType],
        node: NodeIR,
        operator: OnnxOperator,
    ) -> Tuple[str, str, str]:
        strategies = operator.strategies(tensor_types, node)
        if not strategies:
            return (
                OnnxOperator.EXECUTION_UNSUPPORTED,
                OnnxOperator.CLASS_UNSUPPORTED,
                f'Operator "{node.op_type}" has no available strategies',
            )
        return strategies[0]

    @staticmethod
    def _shape_slice(node: NodeProto, rank: int) -> slice:
        start = next((attr.i for attr in node.attribute if attr.name == "start"), 0)
        end = next((attr.i for attr in node.attribute if attr.name == "end"), rank)
        start = start + rank if start < 0 else start
        end = end + rank if end < 0 else end
        start = min(max(start, 0), rank)
        end = min(max(end, 0), rank)
        return slice(start, end)

    @classmethod
    def fold_static_shape_nodes(cls, model: ModelProto) -> ModelProto:
        graph = model.graph
        model_ir = cls.build_ir(model)
        initializer_names = {initializer.name for initializer in graph.initializer}
        folded_initializers: List[TensorProto] = []
        remaining_nodes: List[NodeProto] = []

        for node in graph.node:
            if len(node.output) != 1 or node.output[0] in initializer_names:
                remaining_nodes.append(node)
                continue

            input_shape = (
                cls._static_tensor_shape(model_ir, node.input[0])
                if len(node.input) >= 1
                else None
            )
            if input_shape is None:
                remaining_nodes.append(node)
                continue

            output_name = node.output[0]
            if node.op_type == "Shape":
                values = np.asarray(
                    input_shape[cls._shape_slice(node, len(input_shape))],
                    dtype=np.int64,
                )
                tensor = onnx.helper.make_tensor(
                    output_name,
                    TensorProto.INT64,
                    [values.shape[0]],
                    values,
                )
            elif node.op_type == "Size":
                size = int(np.prod(input_shape))
                tensor = onnx.helper.make_tensor(
                    output_name,
                    TensorProto.INT64,
                    [],
                    [size],
                )
            else:
                remaining_nodes.append(node)
                continue

            folded_initializers.append(tensor)
            initializer_names.add(output_name)

        if not folded_initializers:
            return model

        folded_model = ModelProto()
        folded_model.CopyFrom(model)
        del folded_model.graph.node[:]
        folded_model.graph.node.extend(remaining_nodes)
        folded_model.graph.initializer.extend(folded_initializers)
        return folded_model

    @classmethod
    def optimize_model(cls, model: ModelProto) -> ModelProto:
        return cls.optimize_model_with_report(model).model

    @staticmethod
    def infer_model_shapes(model: ModelProto) -> ModelProto:
        try:
            return onnx.shape_inference.infer_shapes(model)
        except onnx.shape_inference.InferenceError:
            return model

    @classmethod
    def optimize_model_with_report(cls, model: ModelProto) -> ModelOptimizationResult:
        applied_passes: List[str] = []
        before = model
        model = cls.fold_constant_nodes(model)
        if model is not before:
            applied_passes.append("fold_constant_nodes")

        before = model
        model = cls.fold_static_cast_nodes(model)
        if model is not before:
            applied_passes.append("fold_static_cast_nodes")

        before = model
        model = cls.fold_static_shape_nodes(model)
        if model is not before:
            applied_passes.append("fold_static_shape_nodes")

        before_serialized = model.SerializeToString()
        inferred_model = cls.infer_model_shapes(model)
        if inferred_model.SerializeToString() != before_serialized:
            applied_passes.append("infer_shapes")
            model = inferred_model

        return ModelOptimizationResult(
            model=model, applied_passes=tuple(applied_passes)
        )

    @classmethod
    def build_ir(cls, model: ModelProto) -> ModelIR:
        graph = model.graph
        tensors: Dict[str, TensorInfo] = {}

        for value_info in [*graph.input, *graph.value_info, *graph.output]:
            tensors[value_info.name] = TensorInfo(
                name=value_info.name,
                shape=cls._value_info_shape(value_info),
                dtype=cls._value_info_dtype(value_info),
                initializer=False,
                constant=False,
            )

        for initializer in graph.initializer:
            tensors[initializer.name] = TensorInfo(
                name=initializer.name,
                shape=tuple(initializer.dims),
                dtype=cls._tensor_proto_dtype(initializer),
                initializer=True,
                constant=True,
                scalar_value=cls._tensor_proto_scalar_value(initializer),
                constant_value=cls._tensor_proto_constant_value(initializer),
            )

        nodes = tuple(
            NodeIR(
                index=index,
                name=node.name,
                op_type=node.op_type,
                domain=node.domain,
                inputs=tuple(node.input),
                outputs=tuple(node.output),
                attributes=tuple(attr.name for attr in node.attribute),
                attribute_values={
                    attr.name: cls._attribute_value(attr) for attr in node.attribute
                },
            )
            for index, node in enumerate(graph.node)
        )

        for node in graph.node:
            constant_info = cls._constant_node_tensor_info(node)
            if constant_info is not None:
                tensors[constant_info.name] = constant_info

        return ModelIR(
            nodes=nodes,
            inputs=tuple(value.name for value in graph.input),
            outputs=tuple(value.name for value in graph.output),
            initializers=tuple(initializer.name for initializer in graph.initializer),
            tensors=tensors,
        )

    @staticmethod
    def _normalize_fallback_policy(fallback_policy: str) -> str:
        if fallback_policy not in ExecutionPlan.FALLBACK_POLICIES:
            valid = ", ".join(sorted(ExecutionPlan.FALLBACK_POLICIES))
            raise ValueError(
                f'Unknown fallback_policy "{fallback_policy}". Expected one of: {valid}'
            )
        return fallback_policy

    @staticmethod
    def _execution_allowed(execution: str, fallback_policy: str) -> bool:
        if execution == OnnxOperator.EXECUTION_UNSUPPORTED:
            return False
        if fallback_policy == ExecutionPlan.FALLBACK_COMPAT:
            return True
        if fallback_policy == ExecutionPlan.FALLBACK_STRICT:
            return execution in OnnxOperator.STRICT_EXECUTIONS
        return execution in OnnxOperator.NATIVE_EXECUTIONS

    @staticmethod
    def _depends_on_numpy_fallback(
        node: NodeIR, tensor_executions: Dict[str, str]
    ) -> bool:
        return any(
            tensor_executions.get(input_name) in OnnxOperator.NUMPY_FALLBACK_EXECUTIONS
            for input_name in node.inputs
            if input_name
        )

    @classmethod
    def _isolate_layout_view_fallbacks(
        cls, nodes: List[ExecutionPlanNode], fallback_policy: str
    ) -> Tuple[List[ExecutionPlanNode], bool]:
        numpy_fallback_inputs = {
            input_name
            for node in nodes
            if node.execution in OnnxOperator.NUMPY_FALLBACK_EXECUTIONS
            for input_name in node.inputs
            if input_name
        }
        adjusted_nodes = []
        changed = False
        for node in nodes:
            spec = onnx_operators.get(node.op_type, node.domain)
            if (
                spec is not None
                and spec.is_layout_view
                and node.execution == OnnxOperator.EXECUTION_NATIVE
                and any(output in numpy_fallback_inputs for output in node.outputs)
            ):
                execution = OnnxOperator.EXECUTION_NUMPY_RUNTIME
                reason = "Native layout view feeds a NumPy fallback consumer"
                allowed = cls._execution_allowed(execution, fallback_policy)
                if not allowed:
                    reason = (
                        f'Operator "{node.op_type}" requires {execution} execution, '
                        f'which fallback_policy="{fallback_policy}" disallows'
                    )
                adjusted_nodes.append(
                    replace(
                        node,
                        execution=execution,
                        operator_class=OnnxOperator.CLASS_NUMPY_RUNTIME,
                        allowed=allowed,
                        reason=reason,
                    )
                )
                changed = True
                continue
            adjusted_nodes.append(node)
        return adjusted_nodes, changed

    @classmethod
    def _propagate_numpy_fallback_dependencies(
        cls, nodes: List[ExecutionPlanNode], fallback_policy: str
    ) -> Tuple[List[ExecutionPlanNode], bool]:
        tensor_executions: Dict[str, str] = {}
        tensor_native_layout_views: Dict[str, bool] = {}
        adjusted_nodes = []
        changed = False
        for node in nodes:
            execution = node.execution
            operator_class = node.operator_class
            allowed = node.allowed
            reason = node.reason
            spec = onnx_operators.get(node.op_type, node.domain)
            if (
                spec is not None
                and spec.execution == OnnxOperator.EXECUTION_NATIVE_OR_NUMPY_RUNTIME
                and execution
                in {OnnxOperator.EXECUTION_NATIVE, OnnxOperator.EXECUTION_DECOMPOSED}
                and cls._depends_on_numpy_fallback(node, tensor_executions)
            ):
                execution = OnnxOperator.EXECUTION_NUMPY_RUNTIME
                operator_class = OnnxOperator.CLASS_NUMPY_RUNTIME
                reason = "Operator depends on a NumPy fallback input"
                allowed = cls._execution_allowed(execution, fallback_policy)
                if not allowed:
                    reason = (
                        f'Operator "{node.op_type}" requires {execution} execution, '
                        f'which fallback_policy="{fallback_policy}" disallows'
                    )
                changed = True
            if (
                spec is not None
                and spec.is_shape_view
                and execution == OnnxOperator.EXECUTION_NATIVE
                and any(
                    tensor_native_layout_views.get(input_name, False)
                    for input_name in node.inputs
                    if input_name
                )
            ):
                if spec.has_numpy_evaluator:
                    execution = OnnxOperator.EXECUTION_NUMPY_RUNTIME
                    operator_class = OnnxOperator.CLASS_NUMPY_RUNTIME
                else:
                    execution = OnnxOperator.EXECUTION_UNSUPPORTED
                    operator_class = OnnxOperator.CLASS_UNSUPPORTED
                reason = "Native shape view depends on a native layout view"
                allowed = cls._execution_allowed(execution, fallback_policy)
                if not allowed:
                    reason = (
                        f'Operator "{node.op_type}" requires {execution} execution, '
                        f'which fallback_policy="{fallback_policy}" disallows'
                        if execution != OnnxOperator.EXECUTION_UNSUPPORTED
                        else (
                            "Native shape view depends on a native layout view "
                            "and has no NumPy fallback"
                        )
                    )
                changed = True
            if (
                spec is not None
                and not spec.is_layout_view
                and execution
                in {OnnxOperator.EXECUTION_NATIVE, OnnxOperator.EXECUTION_DECOMPOSED}
                and any(
                    tensor_native_layout_views.get(input_name, False)
                    for input_name in node.inputs
                    if input_name
                )
            ):
                if spec.has_numpy_evaluator:
                    execution = OnnxOperator.EXECUTION_NUMPY_RUNTIME
                    operator_class = OnnxOperator.CLASS_NUMPY_RUNTIME
                else:
                    execution = OnnxOperator.EXECUTION_UNSUPPORTED
                    operator_class = OnnxOperator.CLASS_UNSUPPORTED
                reason = "Native operator depends on a native layout view"
                allowed = cls._execution_allowed(execution, fallback_policy)
                if not allowed:
                    reason = (
                        f'Operator "{node.op_type}" requires {execution} execution, '
                        f'which fallback_policy="{fallback_policy}" disallows'
                        if execution != OnnxOperator.EXECUTION_UNSUPPORTED
                        else (
                            "Native operator depends on a native layout view "
                            "and has no NumPy fallback"
                        )
                    )
                changed = True
            adjusted_node = (
                replace(
                    node,
                    execution=execution,
                    operator_class=operator_class,
                    allowed=allowed,
                    reason=reason,
                )
                if execution != node.execution
                else node
            )
            adjusted_nodes.append(adjusted_node)
            for output_name in node.outputs:
                if output_name:
                    tensor_executions[output_name] = execution
                    tensor_native_layout_views[output_name] = (
                        spec is not None
                        and spec.is_layout_view
                        and execution == OnnxOperator.EXECUTION_NATIVE
                    )
        return adjusted_nodes, changed

    @classmethod
    def _resolve_plan_dependencies(
        cls, nodes: List[ExecutionPlanNode], fallback_policy: str
    ) -> List[ExecutionPlanNode]:
        while True:
            nodes, isolated = cls._isolate_layout_view_fallbacks(nodes, fallback_policy)
            nodes, propagated = cls._propagate_numpy_fallback_dependencies(
                nodes, fallback_policy
            )
            if not isolated and not propagated:
                return nodes

    @classmethod
    def _analyze_ir(
        cls,
        model_ir: ModelIR,
        fallback_policy: str = ExecutionPlan.FALLBACK_COMPAT,
        tensor_types: Optional[Dict[str, TensorType]] = None,
    ) -> ExecutionPlan:
        fallback_policy = cls._normalize_fallback_policy(fallback_policy)
        if tensor_types is None:
            tensor_types = model_ir.build_tensor_types()
        nodes = []
        tensor_executions = {
            name: OnnxOperator.EXECUTION_NATIVE
            for name in (*model_ir.inputs, *model_ir.initializers)
            if name
        }
        for node in model_ir.nodes:
            operator = onnx_operators.get(node.op_type, node.domain)
            if operator is None:
                execution = OnnxOperator.EXECUTION_UNSUPPORTED
                operator_class = OnnxOperator.CLASS_UNSUPPORTED
                reason = f'Operator "{node.op_type}" not implemented'
            else:
                execution, operator_class, reason = cls._resolve_node_operator(
                    tensor_types, node, operator
                )
                if (
                    operator.execution == OnnxOperator.EXECUTION_NATIVE_OR_NUMPY_RUNTIME
                    and execution
                    in {
                        OnnxOperator.EXECUTION_NATIVE,
                        OnnxOperator.EXECUTION_DECOMPOSED,
                    }
                    and cls._depends_on_numpy_fallback(node, tensor_executions)
                ):
                    execution = OnnxOperator.EXECUTION_NUMPY_RUNTIME
                    operator_class = OnnxOperator.CLASS_NUMPY_RUNTIME
                    reason = "Operator depends on a NumPy fallback input"

            allowed = cls._execution_allowed(execution, fallback_policy)
            if not allowed and execution != OnnxOperator.EXECUTION_UNSUPPORTED:
                reason = (
                    f'Operator "{node.op_type}" requires {execution} execution, '
                    f'which fallback_policy="{fallback_policy}" disallows'
                )

            nodes.append(
                ExecutionPlanNode(
                    index=node.index,
                    name=node.name,
                    op_type=node.op_type,
                    domain=node.domain,
                    execution=execution,
                    operator_class=operator_class,
                    inputs=node.inputs,
                    outputs=node.outputs,
                    allowed=allowed,
                    reason=reason,
                    input_types=node.input_types(tensor_types),
                    output_types=node.output_types(tensor_types),
                )
            )
            for output_name in node.outputs:
                if output_name:
                    tensor_executions[output_name] = execution
        nodes = cls._resolve_plan_dependencies(nodes, fallback_policy)
        return ExecutionPlan(tuple(nodes), fallback_policy=fallback_policy)

    @classmethod
    def build_pipeline(
        cls,
        model: ModelProto,
        fallback_policy: str = ExecutionPlan.FALLBACK_COMPAT,
    ) -> OnnxRuntimePipeline:
        optimization = cls.optimize_model_with_report(model)
        model_ir = cls.build_ir(optimization.model)
        tensor_types = model_ir.build_tensor_types()
        execution_plan = cls._analyze_ir(
            model_ir,
            fallback_policy=fallback_policy,
            tensor_types=tensor_types,
        )
        return OnnxRuntimePipeline(
            original_model=model,
            optimized_model=optimization.model,
            model_ir=model_ir,
            tensor_types=tensor_types,
            execution_plan=execution_plan,
            optimization_passes=optimization.applied_passes,
        )

    @classmethod
    def analyze(
        cls,
        model: ModelProto,
        fallback_policy: str = ExecutionPlan.FALLBACK_COMPAT,
    ) -> ExecutionPlan:
        return cls.build_pipeline(model, fallback_policy=fallback_policy).execution_plan

    @classmethod
    def is_opset_supported(cls, model: ModelProto):
        unsupported_ops = cls.analyze(model).unsupported_ops
        if unsupported_ops:
            return False, "Unsupported operators: " + ", ".join(unsupported_ops)
        return True, ""

    @classmethod
    def supports_device(cls, device: str) -> bool:
        return device == "CPU"

    @classmethod
    def prepare(
        cls,
        model: ModelProto,
        device: Optional[str] = "CPU",
        fallback_policy: str = ExecutionPlan.FALLBACK_COMPAT,
        **kwargs: Any,
    ):
        """Load the model and creates the ggml runtime backend representation
        for the onnx graph.

        Parameters:
            model: ModelProto (returned by `onnx.load`),
            device: requested device for the computation

        Returns:
            GGML Backend Representation"""
        # This fails with large models.
        # https://github.com/onnx/onnx/blob/b60f69412abb5393ab819b936b473f83867f6c87/onnx/backend/base.py#L85
        # super(GgmlRuntimeBackend, cls).prepare(model, device, **kwargs)
        if device not in (None, "CPU"):
            raise ValueError(
                f'Unsupported device "{device}". GGML ONNX backend supports CPU only.'
            )

        ir_pipeline = cls.build_pipeline(model, fallback_policy=fallback_policy)
        model = ir_pipeline.optimized_model
        execution_plan = ir_pipeline.execution_plan
        opset_imports = {opset.domain: opset.version for opset in model.opset_import}
        if (
            fallback_policy != ExecutionPlan.FALLBACK_COMPAT
            and not execution_plan.is_supported
        ):
            raise NotImplementedError(execution_plan.failure_message())

        cleanup = contextlib.ExitStack()
        try:
            ggml_backend = ggml.ggml_backend_cpu_init()
            if ggml_backend is None:
                raise RuntimeError("Failed to initialize GGML CPU backend")
            cleanup.callback(ggml.ggml_backend_free, ggml_backend)

            graph = model.graph
            weights: Dict[str, ggml.ggml_tensor_p] = {}
            shapes: Dict[str, Tuple[int, ...]] = {}
            dtypes: Dict[str, npt.DTypeLike] = {}

            n_tensors = len(graph.initializer)
            ggml_init_params = ggml.ggml_init_params(
                mem_size=n_tensors * ggml.ggml_tensor_overhead(),
                no_alloc=True,
            )

            ggml_weights_context = ggml.ggml_init(ggml_init_params)
            if ggml_weights_context is None:
                raise RuntimeError("Failed to initialize GGML context")
            cleanup.callback(ggml.ggml_free, ggml_weights_context)

            pairs: List[Tuple[ggml.ggml_tensor_p, npt.NDArray[Any]]] = []

            for initializer in graph.initializer:
                name = initializer.name
                np_array: npt.NDArray[Any] = onnx.numpy_helper.to_array(initializer)  # type: ignore
                storage_array = np.asarray(np_array)
                storage_dtype = np.dtype(storage_array.dtype)
                if storage_dtype == np.dtype(np.bool_):
                    storage_dtype = np.dtype(np.int32)
                elif storage_dtype in GgmlOnnxExecutionContext.UNSIGNED_TO_SIGNED_DTYPE:
                    storage_dtype = np.dtype(
                        GgmlOnnxExecutionContext.UNSIGNED_TO_SIGNED_DTYPE[storage_dtype]
                    )
                if storage_dtype != storage_array.dtype:
                    storage_array = storage_array.astype(storage_dtype)

                storage_shape = storage_array.shape
                if not storage_shape:
                    storage_shape = (1,)
                elif len(storage_shape) > ViewTransformSemantics.GGML_MAX_DIMS:
                    storage_shape = (int(np.prod(storage_shape)),)
                if storage_shape != storage_array.shape:
                    storage_array = storage_array.reshape(storage_shape)
                storage_array = np.ascontiguousarray(storage_array)

                tensor = ggml.utils.from_numpy(
                    x=storage_array, ctx=ggml_weights_context
                )
                ggml.ggml_set_name(tensor, name.encode())
                weights[name] = tensor
                shapes[name] = np_array.shape
                dtypes[name] = np_array.dtype
                pairs.append((tensor, storage_array))

            ggml_weights_buffer = None
            if pairs:
                ggml_weights_buffer = ggml.ggml_backend_alloc_ctx_tensors(
                    ggml_weights_context, ggml_backend
                )
                if ggml_weights_buffer is None:
                    if any(ggml.ggml_nbytes(tensor) for tensor, _ in pairs):
                        raise RuntimeError("Failed to allocate GGML weights buffer")
                else:
                    cleanup.callback(ggml.ggml_backend_buffer_free, ggml_weights_buffer)

            for tensor, storage_array in pairs:
                tensor_dtype = get_tensor_dtype(tensor)
                array = np.asarray(storage_array)

                if tensor_dtype == np.dtype(np.uint16) and array.dtype == np.float16:
                    array = array.view(np.uint16)
                elif array.dtype != tensor_dtype:
                    array = array.astype(tensor_dtype)

                array = np.ascontiguousarray(array)
                tensor_nbytes = ggml.ggml_nbytes(tensor)
                if array.nbytes != tensor_nbytes:
                    raise ValueError(
                        f"Cannot copy {array.nbytes} bytes into GGML tensor with {tensor_nbytes} bytes"
                    )

                if tensor_nbytes:
                    ggml.ggml_backend_tensor_set(
                        tensor,
                        array.ctypes.data_as(ctypes.c_void_p),
                        0,
                        tensor_nbytes,
                    )

            rep = GgmlBackendRep(
                graph=graph,
                weights=weights,
                inputs=graph.input,
                outputs=graph.output,
                shapes=shapes,
                dtypes=dtypes,
                ggml_context=ggml_weights_context,
                ggml_init_params=ggml_init_params,
                ggml_backend=ggml_backend,
                ggml_weights_buffer=ggml_weights_buffer,
                execution_plan=execution_plan,
                ir_pipeline=ir_pipeline,
                opset_imports=opset_imports,
            )
            cleanup.pop_all()
            return rep
        finally:
            cleanup.close()

    @classmethod
    def run_model(
        cls, model: ModelProto, inputs: Any, device: Optional[str] = None, **kwargs
    ) -> Tuple[Any, ...]:
        """Compute the prediction."""
        rep = cls.prepare(model, device, **kwargs)
        return rep.run(inputs, **kwargs)

    @classmethod
    def run_node(
        cls,
        node: NodeProto,
        inputs: Any,
        device: Optional[str] = None,
        outputs_info=None,  # type: ignore
        **kwargs: Any,
    ) -> Tuple[Any, ...]:
        """
        This method is not implemented as it is much more efficient
        to run a whole model than every node independently.
        """
        raise NotImplementedError(
            "It is much more efficient to run a whole model than every node independently."
        )
