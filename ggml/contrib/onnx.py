"""GGML ONNX backend.

This module implements a GGML backend for ONNX models and operators.
"""
import ctypes
import re
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import onnx
from onnx.backend.base import Backend, BackendRep

from onnx.helper import tensor_dtype_to_np_dtype, np_dtype_to_tensor_dtype
from onnx.onnx_ml_pb2 import GraphProto, ModelProto, NodeProto

import ggml
import ggml.utils

ggml_operators = {}
onnx_dtype_map = {
    elem_type: np_dtype
    for elem_type, np_dtype in onnx.mapping.TENSOR_TYPE_TO_NP_TYPE.items()
}


def ggml_operator(operator):
    def inner(func):
        ggml_operators[operator] = func
        return func

    return inner


def map_to_ggml_type(dtype: np.dtype):
    np_data_type_limit = np.dtype(str(dtype).replace("64", "32"))
    ggml_type = ggml.utils.NUMPY_DTYPE_TO_GGML_TYPE.get(
        np_data_type_limit.type,
        ggml.utils.GGML_TYPE.F32,  # TODO: Add i64 but for now, use i32 if looking for i64 or f64
    )

    return ggml_type


def get_tensor_shape(tensor):
    return tuple(reversed(ggml.utils.get_shape(tensor)))


def set_tensor_out(tensor, ndarray):
    output_shape = get_tensor_shape(tensor)

    if output_shape == ():
        ggml.utils.to_numpy(tensor)[()] = ndarray
    else:
        ggml.utils.to_numpy(tensor)[:] = ndarray


def get_tensor_dtype(tensor):
    ggml_type = ggml.utils.GGML_TYPE(tensor.contents.type)
    if ggml_type == ggml.utils.GGML_TYPE.F16:
        ctypes_type = ctypes.c_uint16
    else:
        ctypes_type = np.ctypeslib.as_ctypes_type(
            ggml.utils.GGML_TYPE_TO_NUMPY_DTYPE[ggml_type]
        )
    return np.dtype(ctypes_type)


def can_quantize(
    np_array: np.ndarray,
    name: str,
    graph_def: GraphProto,
):
    return False

    allowed_op_types = set(["MatMul"])

    is_weight = is_2d = is_f32 = is_op_supported = False

    is_weight = name in [initializer.name for initializer in graph_def.initializer]
    is_2d = np_array.ndim == 2
    is_f32 = np_array.dtype == np.float32
    is_op_supported = any(
        [
            node
            for node in graph_def.node
            if node.op_type in allowed_op_types
            and name in node.input
            and node.input[0] == name
        ]
    )

    return all([is_weight, is_2d, is_f32, is_op_supported])


def broadcast_tensor(
    ctx: ggml.ggml_context_p, tensor: ggml.ggml_tensor_p, shape: Tuple
):
    ggml_type = ggml.utils.GGML_TYPE(tensor.contents.type)

    new_tensor = ggml.ggml_new_tensor(
        ctx,
        ggml_type.value,
        len(shape),
        (ctypes.c_int64 * len(shape))(*shape),
    )

    new_tensor = ggml.ggml_repeat(
        ctx,
        tensor,
        new_tensor,
    )

    # if ggml.utils.get_shape(tensor) == ():
    #     ggml.utils.to_numpy(new_tensor)[()] = ggml.utils.to_numpy(tensor)
    # else:
    #     ggml.utils.to_numpy(new_tensor)[:] = ggml.utils.to_numpy(tensor)

    return new_tensor


def broadcast_shapes(
    ctx: ggml.ggml_context_p,
    a: ggml.ggml_tensor_p,
    b: ggml.ggml_tensor_p,
):
    a_shape = get_tensor_shape(a)
    b_shape = get_tensor_shape(b)

    output_shape = tuple(
        reversed(np.broadcast(np.empty(a_shape), np.empty(b_shape)).shape)
    )  # TODO: Fix this

    a_shaped = a
    b_shaped = b

    if a_shape != output_shape:
        a_shaped = broadcast_tensor(ctx, a, output_shape)
    if b_shape != output_shape:
        b_shaped = broadcast_tensor(ctx, b, output_shape)

    return a_shaped, b_shaped


def get_final_dtype(tensor: ggml.ggml_tensor_p, pattern: str = r"<(.*?)>"):
    tensor_name = tensor.contents.name.decode()
    tensor_dtype = get_tensor_dtype(tensor)

    match = re.search(pattern, tensor_name)

    if match:
        dtype_str = match.group(1)
        tensor_dtype = np.dtype(dtype_str)

    return tensor_dtype


# ------ Operators ------


@ggml_operator("Abs")
def ggml_operator_abs(
    backend: "GgmlBackendRep",
    node: NodeProto,
    tensors_dict: Dict[str, ggml.ggml_tensor_p],
    context: ggml.ggml_context_p,
    refs: List[Any],
):
    node_inputs = [tensors_dict[inp] for inp in node.input]

    if len(node_inputs) != 1:
        raise ValueError(
            f'Error for node "{node.name}": Operation "Abs" requires exactly one input. Actual number of inputs: {len(node_inputs)}'
        )

    output_name = node.output[0]
    a = node_inputs[0]

    abs_result = ggml.ggml_abs(
        context,
        a,
    )
    tensors_dict[output_name] = abs_result
    return abs_result


@ggml_operator("Add")
def ggml_operator_add(
    backend: "GgmlBackendRep",
    node: NodeProto,
    tensors_dict: Dict[str, ggml.ggml_tensor_p],
    context: ggml.ggml_context_p,
    refs: List[Any],
):
    node_inputs = [tensors_dict[inp] for inp in node.input]

    if len(node_inputs) != 2:
        raise ValueError(
            f'Error for node "{node.name}": Operation "Add" requires exactly two inputs. Actual number of inputs: {len(node_inputs)}'
        )

    output_name = node.output[0]

    a, b = node_inputs
    a, b = broadcast_shapes(context, a, b)

    add_result = ggml.ggml_add(
        context,
        a,
        b,
    )
    tensors_dict[output_name] = add_result
    return add_result


@ggml.ggml_custom2_op_t
def custom_cast(
    tensor_out: ggml.ggml_tensor_p,
    tensor_in_1: ggml.ggml_tensor_p,
    tensor_in_2: ggml.ggml_tensor_p,
    ith: int,
    nth: int,
    userdata: Optional[ctypes.c_void_p],
):
    dtype = ctypes.cast(userdata, ctypes.POINTER(ctypes.c_int)).contents.value
    tensor = ggml.utils.to_numpy(tensor_in_2)
    np_data_type = tensor_dtype_to_np_dtype(dtype)
    np_data_type_limit = np.dtype(str(np_data_type).replace("64", "32"))

    set_tensor_out(tensor_out, tensor.astype(np_data_type_limit))


@ggml_operator("Cast")
def ggml_operator_cast(
    backend: "GgmlBackendRep",
    node: NodeProto,
    tensors_dict: Dict[str, ggml.ggml_tensor_p],
    context: ggml.ggml_context_p,
    refs: List[Any],
):
    node_inputs = [tensors_dict[inp] for inp in node.input]

    if len(node_inputs) != 1:
        raise ValueError(
            f'Error for node "{node.name}": Operation "Cast" requires exactly one input and a dtype. Actual number of inputs: {len(node_inputs)}'
        )

    onnx_type = next(attr.i for attr in node.attribute if attr.name == "to")
    onnx_type_c = ctypes.c_int(onnx_type)

    a = node_inputs[0]
    np_data_type = tensor_dtype_to_np_dtype(onnx_type)
    np_data_type_limit = np.dtype(str(np_data_type).replace("64", "32"))
    x = np.empty(get_tensor_shape(a), dtype=np_data_type_limit)

    x_t = ggml.utils.from_numpy(x, context)

    new_tensor = tensors_dict[node.output[0]] = ggml.ggml_map_custom2_inplace(
        context,
        x_t,
        a,
        custom_cast,
        1,
        ctypes.pointer(onnx_type_c),
    )

    refs.append(onnx_type_c)

    return new_tensor


@ggml_operator("CastLike")
def ggml_operator_castlike(
    backend: "GgmlBackendRep",
    node: NodeProto,
    tensors_dict: Dict[str, ggml.ggml_tensor_p],
    context: ggml.ggml_context_p,
    refs: List[Any],
):
    node_inputs = [tensors_dict[inp] for inp in node.input]

    if len(node_inputs) != 2:
        raise ValueError(
            f'Error for node "{node.name}": Operation "CastLike" requires exactly two inputs. Actual number of inputs: {len(node_inputs)}'
        )
    a = node_inputs[0]
    b = node_inputs[1]

    np_data_dtype = get_tensor_dtype(b)
    np_data_type_limit = np.dtype(str(np_data_dtype).replace("64", "32"))

    onnx_type = np_dtype_to_tensor_dtype(np_data_dtype)
    onnx_type_c = ctypes.c_int(onnx_type)

    x = np.empty(get_tensor_shape(b), dtype=np_data_type_limit)
    x_t = ggml.utils.from_numpy(x, context)

    new_tensor = tensors_dict[node.output[0]] = ggml.ggml_map_custom2_inplace(
        context,
        x_t,
        a,
        custom_cast,
        1,
        ctypes.pointer(onnx_type_c),
    )

    refs.append(onnx_type_c)

    return new_tensor


@ggml_operator("Concat")
def ggml_operator_concat(
    backend: "GgmlBackendRep",
    node: NodeProto,
    tensors_dict: Dict[str, ggml.ggml_tensor_p],
    context: ggml.ggml_context_p,
    refs: List[Any],
):
    node_inputs = [tensors_dict[inp] for inp in node.input]

    if len(node_inputs) < 2:
        raise ValueError(
            f'Error for node "{node.name}": Operation "Concat" requires at least two inputs. Actual number of inputs: {len(node_inputs)}'
        )

    axis = next((attr.i for attr in node.attribute if attr.name == "axis"), 0)
    shapes = [get_tensor_shape(tensor) for tensor in node_inputs]

    if not all(
        shape[:axis] == shapes[0][:axis] and shape[axis + 1 :] == shapes[0][axis + 1 :]
        for shape in shapes
    ):
        raise ValueError(
            "All tensors must have the same shape along the specified axis."
        )

    total_dim = sum(shape[axis] for shape in shapes)
    output_shape = list(shapes[0])
    output_shape[axis] = total_dim

    x = np.empty(output_shape, dtype=get_tensor_dtype(node_inputs[0]))
    x_t = ggml.utils.from_numpy(x, context)

    @ggml.ggml_custom1_op_t
    def custom_concat(
        tensor_out: ggml.ggml_tensor_p,
        tensor_in_1: ggml.ggml_tensor_p,
        ith: int,
        nth: int,
        userdata: Optional[ctypes.c_void_p],
    ):
        tensors = [ggml.utils.to_numpy(node_input) for node_input in node_inputs]
        x = np.concatenate(tensors, axis=axis)

        set_tensor_out(tensor_out, x)

    new_tensor = tensors_dict[node.output[0]] = ggml.ggml_map_custom1_inplace(
        context,
        x_t,
        custom_concat,
        1,
        None,
    )

    refs.append(custom_concat)

    return new_tensor

@ggml.ggml_custom2_op_t
def custom_constant(
    tensor_out: ggml.ggml_tensor_p,
    tensor_in_1: ggml.ggml_tensor_p,
    tensor_in_2: ggml.ggml_tensor_p,
    ith: int,
    nth: int,
    userdata: Optional[ctypes.c_void_p],
):
    shape = get_tensor_shape(tensor_in_1)
    constant_data = ggml.utils.to_numpy(tensor_in_2)
    new_tenor = constant_data.reshape(shape)

    set_tensor_out(tensor_out, new_tenor)


@ggml_operator("Constant")
def ggml_operator_constant(
    backend: "GgmlBackendRep",
    node: NodeProto,
    tensors_dict: Dict[str, ggml.ggml_tensor_p],
    context: ggml.ggml_context_p,
    refs: List[Any],
):
    node_attributes = node.attribute
    name = node.output[0]

    value_attr = next(attr for attr in node_attributes if attr.name == "value")
    tensor = value_attr.t
    data_type = tensor.data_type

    np_data_type = tensor_dtype_to_np_dtype(data_type)
    np_data_type_limit = np.dtype(str(np_data_type).replace("64", "32"))

    if tensor.raw_data:
        data_value = np.frombuffer(tensor.raw_data, dtype=np_data_type)
    else:
        data_value = onnx.numpy_helper.to_array(tensor)

    data_tensor = ggml.utils.from_numpy(
        data_value.astype(np_data_type_limit),
        context,
    )

    tensor_shape = tensor.dims or ()

    x = np.empty(tensor_shape, dtype=np_data_type_limit)
    x_t = None

    if tensor_shape == ():
        ggml_type = map_to_ggml_type(np_data_type_limit)

        x_t = ggml.ggml_new_tensor(
            context,
            ggml_type.value,
            len(tensor_shape),
            (ctypes.c_int64 * len(tensor_shape))(*tensor_shape),
        )

    else:
        x_t = ggml.utils.from_numpy(x, context)

    new_tensor = tensors_dict[node.output[0]] = ggml.ggml_map_custom2_inplace(
        context,
        x_t,
        data_tensor,
        custom_constant,
        1,
        None,
    )

    ggml.ggml_set_name(new_tensor, (name + f"<{np_data_type}>").encode())
    return new_tensor


@ggml.ggml_custom2_op_t
def custom_constant_of_shape(
    tensor_out: ggml.ggml_tensor_p,
    tensor_in_1: ggml.ggml_tensor_p,
    tensor_in_2: ggml.ggml_tensor_p,
    ith: int,
    nth: int,
    userdata: Optional[ctypes.c_void_p],
):
    shape = get_tensor_shape(tensor_out)
    value = ggml.utils.to_numpy(tensor_in_2)
    new_tenor = np.full(tuple(shape), value)

    set_tensor_out(tensor_out, new_tenor)


@ggml_operator("ConstantOfShape")
def ggml_operator_constant_of_shape(
    backend: "GgmlBackendRep",
    node: NodeProto,
    tensors_dict: Dict[str, ggml.ggml_tensor_p],
    context: ggml.ggml_context_p,
    refs: List[Any],
):
    node_inputs = [tensors_dict[inp] for inp in node.input]

    if len(node_inputs) != 1:
        raise ValueError(
            f'Error for node "{node.name}": Operation "ConstantOfShape" requires exactly one input. Actual number of inputs: {len(node_inputs)}'
        )

    node_attributes = node.attribute

    value_attr = next(attr for attr in node_attributes if attr.name == "value")
    tensor = value_attr.t
    data_type = tensor.data_type
    np_data_type = tensor_dtype_to_np_dtype(data_type)

    np_data_type_limit = np.dtype(str(np_data_type).replace("64", "32"))

    data_value = np.frombuffer(tensor.raw_data, dtype=np_data_type)

    data_tensor = ggml.utils.from_numpy(
        data_value.astype(np_data_type_limit),
        context,
    )

    shape = ggml.utils.to_numpy(node_inputs[0])

    x = np.empty(shape, dtype=np_data_type_limit)
    x_t = ggml.utils.from_numpy(x, context)

    new_tensor = tensors_dict[node.output[0]] = ggml.ggml_map_custom2_inplace(
        context,
        x_t,
        data_tensor,
        custom_constant_of_shape,
        1,
        None,
    )

    return new_tensor


@ggml_operator("Div")
def ggml_operator_div(
    backend: "GgmlBackendRep",
    node: NodeProto,
    tensors_dict: Dict[str, ggml.ggml_tensor_p],
    context: ggml.ggml_context_p,
    refs: List[Any],
):
    node_inputs = [tensors_dict[inp] for inp in node.input]

    if len(node_inputs) != 2:
        raise ValueError(
            f'Error for node "{node.name}": Operation "Div" requires exactly two inputs. Actual number of inputs: {len(node_inputs)}'
        )

    output_name = node.output[0]
    a = node_inputs[0]
    b = node_inputs[1]

    a, b = broadcast_shapes(context, a, b)

    div_result = ggml.ggml_div(
        context,
        a,
        b,
    )
    tensors_dict[output_name] = div_result
    return div_result


@ggml.ggml_custom3_op_t
def custom_equal(
    tensor_out: ggml.ggml_tensor_p,
    tensor_in_1: ggml.ggml_tensor_p,
    tensor_in_2: ggml.ggml_tensor_p,
    tensor_in_3: ggml.ggml_tensor_p,
    ith: int,
    nth: int,
    userdata: Optional[ctypes.c_void_p],
):
    a = ggml.utils.to_numpy(tensor_in_2)
    b = ggml.utils.to_numpy(tensor_in_3)

    x = np.equal(a, b)

    set_tensor_out(tensor_out, x)


@ggml_operator("Equal")
def ggml_operator_equal(
    backend: "GgmlBackendRep",
    node: NodeProto,
    tensors_dict: Dict[str, ggml.ggml_tensor_p],
    context: ggml.ggml_context_p,
    refs: List[Any],
):
    node_inputs = [tensors_dict[inp] for inp in node.input]

    if len(node_inputs) != 2:
        raise ValueError(
            f'Error for node "{node.name}": Operation "Equal" requires exactly two inputs. Actual number of inputs: {len(node_inputs)}'
        )

    a_shape = get_tensor_shape(node_inputs[0])
    a_dtype = get_tensor_dtype(node_inputs[0])
    b_shape = get_tensor_shape(node_inputs[1])
    name = node.output[0]

    output_shape = np.broadcast(np.empty(a_shape), np.empty(b_shape)).shape

    x = np.empty(output_shape, dtype=a_dtype)
    x_t = ggml.utils.from_numpy(x, context)

    new_tensor = tensors_dict[name] = ggml.ggml_map_custom3_inplace(
        context,
        x_t,
        node_inputs[0],
        node_inputs[1],
        custom_equal,
        1,
        None,
    )

    ggml.ggml_set_name(new_tensor, (name + f"<bool>").encode())

    return new_tensor


@ggml_operator("Exp")
def ggml_operator_exp(
    backend: "GgmlBackendRep",
    node: NodeProto,
    tensors_dict: Dict[str, ggml.ggml_tensor_p],
    context: ggml.ggml_context_p,
    refs: List[Any],
):
    node_inputs = [tensors_dict[inp] for inp in node.input]

    if len(node_inputs) < 1:
        raise ValueError(
            f'Error for node "{node.name}": Operation "Exp" requires at least one input. Actual number of inputs: {len(node_inputs)}'
        )
    a = node_inputs[0]
    np_dtype = get_tensor_dtype(a)

    x = np.empty(get_tensor_shape(a), dtype=np_dtype)
    x_t = ggml.utils.from_numpy(x, context)

    @ggml.ggml_custom1_op_t
    def custom_exp(
        tensor_out: ggml.ggml_tensor_p,
        tensor_in_1: ggml.ggml_tensor_p,
        ith: int,
        nth: int,
        userdata: Optional[ctypes.c_void_p],
    ):
        tensor = ggml.utils.to_numpy(tensor_in_1)
        x = np.exp(tensor)
        set_tensor_out(tensor_out, np.array(x))

    new_tensor = tensors_dict[node.output[0]] = ggml.ggml_map_custom1_inplace(
        context,
        x_t,
        custom_exp,
        1,
        None,
    )

    refs.append(custom_exp)

    return new_tensor


@ggml.ggml_custom3_op_t
def custom_gather(
    tensor_out: ggml.ggml_tensor_p,
    tensor_in_1: ggml.ggml_tensor_p,
    tensor_in_2: ggml.ggml_tensor_p,
    tensor_in_3: ggml.ggml_tensor_p,
    ith: int,
    nth: int,
    userdata: Optional[ctypes.c_void_p],
):
    input_array = ggml.utils.to_numpy(tensor_in_2)
    index_array = ggml.utils.to_numpy(tensor_in_3)
    axis = ctypes.cast(userdata, ctypes.POINTER(ctypes.c_int)).contents.value

    new_array = np.take(input_array, index_array, axis=axis)

    set_tensor_out(tensor_out, new_array)


@ggml_operator("Gather")
def ggml_operator_gather(
    backend: "GgmlBackendRep",
    node: NodeProto,
    tensors_dict: Dict[str, ggml.ggml_tensor_p],
    context: ggml.ggml_context_p,
    refs: List[Any],
):
    node_inputs = [tensors_dict[inp] for inp in node.input]

    if len(node_inputs) != 2:
        raise ValueError(
            f'Error for node "{node.name}": Operation "Gather" requires exactly two inputs and one axis. Actual number of inputs: {len(node_inputs)}'
        )

    axis = next((attr.i for attr in node.attribute if attr.name == "axis"), 0)
    axis_c = ctypes.c_int(axis)

    input_shape = get_tensor_shape(node_inputs[0])
    input_dtype = get_tensor_dtype(node_inputs[0])
    index_shape = get_tensor_shape(node_inputs[1])

    Ni = input_shape[:axis]
    Nk = input_shape[axis + 1 :]
    Nj = index_shape

    output_shape = tuple(list(Ni) + list(Nj) + list(Nk))
    x = np.empty(output_shape, dtype=input_dtype)
    x_t = ggml.utils.from_numpy(x, context)

    new_tensor = tensors_dict[node.output[0]] = ggml.ggml_map_custom3_inplace(
        context,
        x_t,
        node_inputs[0],
        node_inputs[1],
        custom_gather,
        1,
        ctypes.pointer(axis_c),
    )

    refs.append(axis_c)

    return new_tensor


@ggml.ggml_custom3_op_t
def custom_greater(
    tensor_out: ggml.ggml_tensor_p,
    tensor_in_1: ggml.ggml_tensor_p,
    tensor_in_2: ggml.ggml_tensor_p,
    tensor_in_3: ggml.ggml_tensor_p,
    ith: int,
    nth: int,
    userdata: Optional[ctypes.c_void_p],
):
    a = ggml.utils.to_numpy(tensor_in_2)
    b = ggml.utils.to_numpy(tensor_in_3)

    x = np.greater(a, b)

    set_tensor_out(tensor_out, x)


@ggml_operator("Greater")
def ggml_operator_greater(
    backend: "GgmlBackendRep",
    node: NodeProto,
    tensors_dict: Dict[str, ggml.ggml_tensor_p],
    context: ggml.ggml_context_p,
    refs: List[Any],
):
    node_inputs = [tensors_dict[inp] for inp in node.input]

    if len(node_inputs) != 2:
        raise ValueError(
            f'Error for node "{node.name}": Operation "Greater" requires exactly two inputs. Actual number of inputs: {len(node_inputs)}'
        )

    a_shape = get_tensor_shape(node_inputs[0])
    a_dtype = get_tensor_dtype(node_inputs[0])
    b_shape = get_tensor_shape(node_inputs[1])
    name = node.output[0]

    output_shape = np.broadcast(np.empty(a_shape), np.empty(b_shape)).shape

    x = np.empty(output_shape, dtype=a_dtype)
    x_t = ggml.utils.from_numpy(x, context)

    new_tensor = tensors_dict[name] = ggml.ggml_map_custom3_inplace(
        context,
        x_t,
        node_inputs[0],
        node_inputs[1],
        custom_greater,
        1,
        None,
    )

    ggml.ggml_set_name(new_tensor, (name + f"<bool>").encode())

    return new_tensor


@ggml.ggml_custom3_op_t
def custom_greater_equal(
    tensor_out: ggml.ggml_tensor_p,
    tensor_in_1: ggml.ggml_tensor_p,
    tensor_in_2: ggml.ggml_tensor_p,
    tensor_in_3: ggml.ggml_tensor_p,
    ith: int,
    nth: int,
    userdata: Optional[ctypes.c_void_p],
):
    a = ggml.utils.to_numpy(tensor_in_2)
    b = ggml.utils.to_numpy(tensor_in_3)

    x = np.greater_equal(a, b)

    set_tensor_out(tensor_out, x)


@ggml_operator("GreaterOrEqual")
def ggml_operator_greater_or_equal(
    backend: "GgmlBackendRep",
    node: NodeProto,
    tensors_dict: Dict[str, ggml.ggml_tensor_p],
    context: ggml.ggml_context_p,
    refs: List[Any],
):
    node_inputs = [tensors_dict[inp] for inp in node.input]

    if len(node_inputs) != 2:
        raise ValueError(
            f'Error for node "{node.name}": Operation "GreaterOrEqual" requires exactly two inputs. Actual number of inputs: {len(node_inputs)}'
        )

    a_shape = get_tensor_shape(node_inputs[0])
    a_dtype = get_tensor_dtype(node_inputs[0])
    b_shape = get_tensor_shape(node_inputs[1])
    name = node.output[0]

    output_shape = np.broadcast(np.empty(a_shape), np.empty(b_shape)).shape

    x = np.empty(output_shape, dtype=a_dtype)
    x_t = ggml.utils.from_numpy(x, context)

    new_tensor = tensors_dict[name] = ggml.ggml_map_custom3_inplace(
        context,
        x_t,
        node_inputs[0],
        node_inputs[1],
        custom_greater_equal,
        1,
        None,
    )

    ggml.ggml_set_name(new_tensor, (name + f"<bool>").encode())

    return new_tensor


@ggml.ggml_custom3_op_t
def custom_less(
    tensor_out: ggml.ggml_tensor_p,
    tensor_in_1: ggml.ggml_tensor_p,
    tensor_in_2: ggml.ggml_tensor_p,
    tensor_in_3: ggml.ggml_tensor_p,
    ith: int,
    nth: int,
    userdata: Optional[ctypes.c_void_p],
):
    a = ggml.utils.to_numpy(tensor_in_2)
    b = ggml.utils.to_numpy(tensor_in_3)

    x = np.less(a, b)

    set_tensor_out(tensor_out, x)


@ggml_operator("Less")
def ggml_operator_less(
    backend: "GgmlBackendRep",
    node: NodeProto,
    tensors_dict: Dict[str, ggml.ggml_tensor_p],
    context: ggml.ggml_context_p,
    refs: List[Any],
):
    node_inputs = [tensors_dict[inp] for inp in node.input]

    if len(node_inputs) != 2:
        raise ValueError(
            f'Error for node "{node.name}": Operation "Less" requires exactly two inputs. Actual number of inputs: {len(node_inputs)}'
        )

    a_shape = get_tensor_shape(node_inputs[0])
    a_dtype = get_tensor_dtype(node_inputs[0])
    b_shape = get_tensor_shape(node_inputs[1])
    name = node.output[0]

    output_shape = np.broadcast(np.empty(a_shape), np.empty(b_shape)).shape

    x = np.empty(output_shape, dtype=a_dtype)
    x_t = ggml.utils.from_numpy(x, context)

    new_tensor = tensors_dict[name] = ggml.ggml_map_custom3_inplace(
        context,
        x_t,
        node_inputs[0],
        node_inputs[1],
        custom_less,
        1,
        None,
    )

    ggml.ggml_set_name(new_tensor, (name + f"<bool>").encode())

    return new_tensor


@ggml.ggml_custom3_op_t
def custom_less_equal(
    tensor_out: ggml.ggml_tensor_p,
    tensor_in_1: ggml.ggml_tensor_p,
    tensor_in_2: ggml.ggml_tensor_p,
    tensor_in_3: ggml.ggml_tensor_p,
    ith: int,
    nth: int,
    userdata: Optional[ctypes.c_void_p],
):
    a = ggml.utils.to_numpy(tensor_in_2)
    b = ggml.utils.to_numpy(tensor_in_3)

    x = np.less_equal(a, b)

    set_tensor_out(tensor_out, x)


@ggml_operator("LessOrEqual")
def ggml_operator_less_or_equal(
    backend: "GgmlBackendRep",
    node: NodeProto,
    tensors_dict: Dict[str, ggml.ggml_tensor_p],
    context: ggml.ggml_context_p,
    refs: List[Any],
):
    node_inputs = [tensors_dict[inp] for inp in node.input]

    if len(node_inputs) != 2:
        raise ValueError(
            f'Error for node "{node.name}": Operation "LessOrEqual" requires exactly two inputs. Actual number of inputs: {len(node_inputs)}'
        )

    a_shape = get_tensor_shape(node_inputs[0])
    a_dtype = get_tensor_dtype(node_inputs[0])
    b_shape = get_tensor_shape(node_inputs[1])
    name = node.output[0]

    output_shape = np.broadcast(np.empty(a_shape), np.empty(b_shape)).shape

    x = np.empty(output_shape, dtype=a_dtype)
    x_t = ggml.utils.from_numpy(x, context)

    new_tensor = tensors_dict[name] = ggml.ggml_map_custom3_inplace(
        context,
        x_t,
        node_inputs[0],
        node_inputs[1],
        custom_less_equal,
        1,
        None,
    )

    ggml.ggml_set_name(new_tensor, (name + f"<bool>").encode())

    return new_tensor


@ggml_operator("Log")
def ggml_operator_log(
    backend: "GgmlBackendRep",
    node: NodeProto,
    tensors_dict: Dict[str, ggml.ggml_tensor_p],
    context: ggml.ggml_context_p,
    refs: List[Any],
):
    node_inputs = [tensors_dict[inp] for inp in node.input]

    if len(node_inputs) != 1:
        raise ValueError(
            f'Error for node "{node.name}": Operation "Log" requires exactly one input. Actual number of inputs: {len(node_inputs)}'
        )

    output_name = node.output[0]
    a = node_inputs[0]

    log_result = ggml.ggml_log(
        context,
        a,
    )
    tensors_dict[output_name] = log_result
    return log_result


@ggml_operator("LogSoftmax")
def ggml_operator_log_soft_max(
    backend: "GgmlBackendRep",
    node: NodeProto,
    tensors_dict: Dict[str, ggml.ggml_tensor_p],
    context: ggml.ggml_context_p,
    refs: List[Any],
):
    node_inputs = [tensors_dict[inp] for inp in node.input]

    if len(node_inputs) != 1:
        raise ValueError(
            f'Error for node "{node.name}": Operation "LogSoftmax" requires exactly one input. Actual number of inputs: {len(node_inputs)}'
        )

    output_name = node.output[0]
    a = node_inputs[0]
    soft_max_result = ggml.ggml_soft_max(context, a)
    log_result = ggml.ggml_log(
        context,
        soft_max_result,
    )
    tensors_dict[output_name] = log_result
    return log_result


@ggml_operator("MatMul")
def ggml_operator_mat_mul(
    backend: "GgmlBackendRep",
    node: NodeProto,
    tensors_dict: Dict[str, ggml.ggml_tensor_p],
    context: ggml.ggml_context_p,
    refs: List[Any],
):
    node_inputs = [tensors_dict[inp] for inp in node.input]

    if len(node_inputs) != 2:
        raise ValueError(
            f'Error for node "{node.name}": Operation "MatMul" requires exactly two inputs. Actual number of inputs: {len(node_inputs)}'
        )

    output_name = node.output[0]
    a, b = node_inputs
    b_shape = get_tensor_shape(b)
    a_shape = get_tensor_shape(a)

    # TODO: is this check required? broadcast alone wont pass ONNX tests but is broadcasting itself even required or should it fail if a,b are not correct?
    try:
        np.matmul(np.empty(a_shape), np.empty(b_shape))
    except:
        a, b = broadcast_shapes(context, a, b)

    b_dtype = get_tensor_dtype(b)

    b_permute = ggml.ggml_transpose(
        context,
        b,
    )

    b_shape = ggml.utils.get_shape(b_permute)

    b_transposed = ggml.ggml_cpy(
        context,
        b_permute,
        ggml.ggml_new_tensor(
            context,
            map_to_ggml_type(b_dtype).value,
            len(b_shape),
            (ctypes.c_int64 * len(b_shape))(*b_shape),
        ),
    )

    mul_mat_result = ggml.ggml_mul_mat(
        context,
        b_transposed,
        a,
    )

    tensors_dict[output_name] = mul_mat_result
    return mul_mat_result


@ggml_operator("Max")
def ggml_operator_max(
    backend: "GgmlBackendRep",
    node: NodeProto,
    tensors_dict: Dict[str, ggml.ggml_tensor_p],
    context: ggml.ggml_context_p,
    refs: List[Any],
):
    node_inputs = [tensors_dict[inp] for inp in node.input]

    if len(node_inputs) < 1:
        raise ValueError(
            f'Error for node "{node.name}": Operation "Max" requires at least one input. Actual number of inputs: {len(node_inputs)}'
        )

    a_dtype = get_tensor_dtype(node_inputs[0])
    ggml_type = map_to_ggml_type(a_dtype)

    input_shapes = [get_tensor_shape(node_input) for node_input in node_inputs]
    output_shape = input_shapes[0]

    for shape in input_shapes[1:]:
        output_shape = np.maximum(output_shape, shape)

    output_shape = tuple(reversed(output_shape))

    x_t = ggml.ggml_new_tensor(
        context,
        ggml_type.value,
        len(output_shape),
        (ctypes.c_int64 * len(output_shape))(*output_shape),
    )

    @ggml.ggml_custom1_op_t
    def custom_max(
        tensor_out: ggml.ggml_tensor_p,
        tensor_in_1: ggml.ggml_tensor_p,
        ith: int,
        nth: int,
        userdata: Optional[ctypes.c_void_p],
    ):
        tensors = [ggml.utils.to_numpy(node_input) for node_input in node_inputs]
        x = np.max(tensors, axis=0)
        set_tensor_out(tensor_out, np.array(x))

    new_tensor = tensors_dict[node.output[0]] = ggml.ggml_map_custom1_inplace(
        context,
        x_t,
        custom_max,
        1,
        None,
    )

    refs.append(custom_max)

    return new_tensor


@ggml_operator("Min")
def ggml_operator_min(
    backend: "GgmlBackendRep",
    node: NodeProto,
    tensors_dict: Dict[str, ggml.ggml_tensor_p],
    context: ggml.ggml_context_p,
    refs: List[Any],
):
    node_inputs = [tensors_dict[inp] for inp in node.input]

    if len(node_inputs) < 1:
        raise ValueError(
            f'Error for node "{node.name}": Operation "Min" requires at least one input. Actual number of inputs: {len(node_inputs)}'
        )

    a_dtype = get_tensor_dtype(node_inputs[0])
    ggml_type = map_to_ggml_type(a_dtype)

    input_shapes = [get_tensor_shape(node_input) for node_input in node_inputs]
    output_shape = input_shapes[0]

    for shape in input_shapes[1:]:
        output_shape = np.minimum(output_shape, shape)

    output_shape = tuple(reversed(output_shape))

    x_t = ggml.ggml_new_tensor(
        context,
        ggml_type.value,
        len(output_shape),
        (ctypes.c_int64 * len(output_shape))(*output_shape),
    )

    @ggml.ggml_custom1_op_t
    def custom_min(
        tensor_out: ggml.ggml_tensor_p,
        tensor_in_1: ggml.ggml_tensor_p,
        ith: int,
        nth: int,
        userdata: Optional[ctypes.c_void_p],
    ):
        tensors = [ggml.utils.to_numpy(node_input) for node_input in node_inputs]
        x = np.min(tensors, axis=0)
        set_tensor_out(tensor_out, np.array(x))

    new_tensor = tensors_dict[node.output[0]] = ggml.ggml_map_custom1_inplace(
        context,
        x_t,
        custom_min,
        1,
        None,
    )

    refs.append(custom_min)

    return new_tensor


@ggml_operator("Mul")
def ggml_operator_mul(
    backend: "GgmlBackendRep",
    node: NodeProto,
    tensors_dict: Dict[str, ggml.ggml_tensor_p],
    context: ggml.ggml_context_p,
    refs: List[Any],
):
    node_inputs = [tensors_dict[inp] for inp in node.input]

    if len(node_inputs) != 2:
        raise ValueError(
            f'Error for node "{node.name}": Operation "Mul" requires exactly two inputs. Actual number of inputs: {len(node_inputs)}'
        )

    output_name = node.output[0]
    a = node_inputs[0]
    b = node_inputs[1]

    a, b = broadcast_shapes(context, a, b)

    mul_result = ggml.ggml_mul(
        context,
        a,
        b,
    )

    tensors_dict[output_name] = mul_result
    return mul_result


@ggml.ggml_custom2_op_t
def custom_pow(
    tensor_out: ggml.ggml_tensor_p,
    tensor_in_1: ggml.ggml_tensor_p,
    tensor_in_2: ggml.ggml_tensor_p,
    ith: int,
    nth: int,
    userdata: Optional[ctypes.c_void_p],
):
    x1 = ggml.utils.to_numpy(tensor_in_1)
    x2 = ggml.utils.to_numpy(tensor_in_2)

    new_tensor = np.power(x1, x2)

    set_tensor_out(tensor_out, new_tensor)


@ggml.ggml_custom3_op_t
def custom_or(
    tensor_out: ggml.ggml_tensor_p,
    tensor_in_1: ggml.ggml_tensor_p,
    tensor_in_2: ggml.ggml_tensor_p,
    tensor_in_3: ggml.ggml_tensor_p,
    ith: int,
    nth: int,
    userdata: Optional[ctypes.c_void_p],
):
    a = ggml.utils.to_numpy(tensor_in_2)
    b = ggml.utils.to_numpy(tensor_in_3)

    x = np.logical_or(a, b)

    set_tensor_out(tensor_out, x)


@ggml_operator("Or")
def ggml_operator_or(
    backend: "GgmlBackendRep",
    node: NodeProto,
    tensors_dict: Dict[str, ggml.ggml_tensor_p],
    context: ggml.ggml_context_p,
    refs: List[Any],
):
    node_inputs = [tensors_dict[inp] for inp in node.input]

    if len(node_inputs) != 2:
        raise ValueError(
            f'Error for node "{node.name}": Operation "Or" requires exactly two inputs. Actual number of inputs: {len(node_inputs)}'
        )

    a_shape = get_tensor_shape(node_inputs[0])
    a_dtype = get_tensor_dtype(node_inputs[0])
    b_shape = get_tensor_shape(node_inputs[1])
    name = node.output[0]

    output_shape = np.broadcast(np.empty(a_shape), np.empty(b_shape)).shape

    x = np.empty(output_shape, dtype=a_dtype)
    x_t = ggml.utils.from_numpy(x, context)

    new_tensor = tensors_dict[name] = ggml.ggml_map_custom3_inplace(
        context,
        x_t,
        node_inputs[0],
        node_inputs[1],
        custom_or,
        1,
        None,
    )

    ggml.ggml_set_name(new_tensor, (name + f"<bool>").encode())

    return new_tensor


@ggml_operator("Pow")
def ggml_operator_pow(
    backend: "GgmlBackendRep",
    node: NodeProto,
    tensors_dict: Dict[str, ggml.ggml_tensor_p],
    context: ggml.ggml_context_p,
    refs: List[Any],
):
    node_inputs = [tensors_dict[inp] for inp in node.input]

    if len(node_inputs) != 2:
        raise ValueError(
            f'Error for node "{node.name}": Operation "Pow" requires exactly two inputs. Actual number of inputs: {len(node_inputs)}'
        )

    x1 = node_inputs[0]
    x2 = node_inputs[1]

    new_tensor = tensors_dict[node.output[0]] = ggml.ggml_map_custom2_inplace(
        context,
        x1,
        x2,
        custom_pow,
        1,
        None,
    )

    return new_tensor


@ggml.ggml_custom2_op_t
def custom_range(
    tensor_out: ggml.ggml_tensor_p,
    tensor_in_1: ggml.ggml_tensor_p,
    tensor_in_2: ggml.ggml_tensor_p,
    ith: int,
    nth: int,
    userdata: Optional[ctypes.c_void_p],
):
    tensors = ggml.utils.to_numpy(tensor_in_2)
    start_array, limit_array, delta_array = tensors

    new_tensor = np.arange(start_array, limit_array, delta_array)

    set_tensor_out(tensor_out, new_tensor)


@ggml_operator("Range")
def ggml_operator_range(
    backend: "GgmlBackendRep",
    node: NodeProto,
    tensors_dict: Dict[str, ggml.ggml_tensor_p],
    context: ggml.ggml_context_p,
    refs: List[Any],
):
    node_inputs = [tensors_dict[inp] for inp in node.input]

    if len(node_inputs) != 3:
        raise ValueError(
            f'Error for node "{node.name}": Operation "Range" requires exactly three inputs. Actual number of inputs: {len(node_inputs)}'
        )

    tensors = [ggml.utils.to_numpy(node_input) for node_input in node_inputs]

    start, stop, step = tensors
    output_shape = (int(np.ceil((stop - start) / step)),)

    x = np.empty(output_shape, dtype=step.dtype)
    x_t = ggml.utils.from_numpy(x, context)

    input_tensors = ggml.utils.from_numpy(np.array(tensors), context)

    new_tensor = tensors_dict[node.output[0]] = ggml.ggml_map_custom2_inplace(
        context,
        x_t,
        input_tensors,
        custom_range,
        1,
        None,
    )

    return new_tensor


class ReduceMaxUserData(ctypes.Structure):
    _fields_ = [
        ("axes", ctypes.POINTER(ctypes.c_int)),
        ("axes_length", ctypes.c_int),
        ("keepdims", ctypes.c_int),
    ]

    def __init__(self, axes, keepdims):
        if isinstance(axes, list):
            self.axes_length = len(axes)
            self.axes = (ctypes.c_int * self.axes_length)(*axes)
        else:
            raise ValueError("axes should be a list of integers")

        self.keepdims = keepdims


@ggml.ggml_custom2_op_t
def custom_reduce_max(
    tensor_out: ggml.ggml_tensor_p,
    tensor_in_1: ggml.ggml_tensor_p,
    tensor_in_2: ggml.ggml_tensor_p,
    ith: int,
    nth: int,
    userdata: Optional[ctypes.c_void_p],
):
    userdata_data_ptr = ctypes.cast(userdata, ctypes.POINTER(ReduceMaxUserData))
    userdata_data = userdata_data_ptr.contents

    tensor = ggml.utils.to_numpy(tensor_in_2)
    axes = [userdata_data.axes[i] for i in range(userdata_data.axes_length)]
    keepdims = userdata_data.keepdims

    axes = tuple(axes) if len(axes) else None
    rmean_result = np.max(tensor, axis=axes, keepdims=keepdims)

    set_tensor_out(tensor_out, rmean_result)


@ggml_operator("ReduceMax")
def ggml_operator_reduce_max(
    backend: "GgmlBackendRep",
    node: NodeProto,
    tensors_dict: Dict[str, ggml.ggml_tensor_p],
    context: ggml.ggml_context_p,
    refs: List[Any],
):
    node_inputs = [tensors_dict[inp] for inp in node.input]

    if len(node_inputs) > 2 or len(node_inputs) < 1:
        raise ValueError(
            f'Error for node "{node.name}": Operation "ReduceMax" requires at least one input. Actual number of inputs: {len(node_inputs)}'
        )

    input_tensor = node_inputs[0]

    tensor_shape = get_tensor_shape(input_tensor)
    tensor_dtype = get_tensor_dtype(input_tensor)

    axes = next((attr.ints for attr in node.attribute if attr.name == "axes"), None)
    if not axes:
        if len(node_inputs) != 2:
            raise ValueError(
                f'Error for node "{node.name}": Operation "ReduceMean" requires an axis.'
            )

        axes_eval = backend.eval_tensor(node_inputs[1], context)
        axes = ggml.utils.to_numpy(axes_eval)

    keepdims = next((attr.i for attr in node.attribute if attr.name == "keepdims"), 0)

    rmean_userdata = ReduceMeanUserData(list(axes), keepdims)
    userdata_p = ctypes.cast(ctypes.pointer(rmean_userdata), ctypes.c_void_p)

    output_shape = tuple([1] * len(tensor_shape)) if keepdims else ()

    if len(axes):
        output_shape = list(tensor_shape)
        sorted_axes = sorted(axes, reverse=True)

        for axis in sorted_axes:
            if keepdims:
                output_shape[axis] = 1
            else:
                output_shape.pop(axis)

    output_shape = tuple(output_shape)
    x = np.empty(output_shape, dtype=tensor_dtype)
    x_t = ggml.utils.from_numpy(x, context)

    new_tensor = tensors_dict[node.output[0]] = ggml.ggml_map_custom2_inplace(
        context,
        x_t,
        input_tensor,
        custom_reduce_max,
        1,
        userdata_p,
    )

    refs.append(rmean_userdata)

    return new_tensor


class ReduceMeanUserData(ctypes.Structure):
    _fields_ = [
        ("axes", ctypes.POINTER(ctypes.c_int)),
        ("axes_length", ctypes.c_int),
        ("keepdims", ctypes.c_int),
    ]

    def __init__(self, axes, keepdims):
        if isinstance(axes, list):
            self.axes_length = len(axes)
            self.axes = (ctypes.c_int * self.axes_length)(*axes)
        else:
            raise ValueError("axes should be a list of integers")

        self.keepdims = keepdims


@ggml.ggml_custom2_op_t
def custom_reduce_mean(
    tensor_out: ggml.ggml_tensor_p,
    tensor_in_1: ggml.ggml_tensor_p,
    tensor_in_2: ggml.ggml_tensor_p,
    ith: int,
    nth: int,
    userdata: Optional[ctypes.c_void_p],
):
    userdata_data_ptr = ctypes.cast(userdata, ctypes.POINTER(ReduceMeanUserData))
    userdata_data = userdata_data_ptr.contents

    tensor = ggml.utils.to_numpy(tensor_in_2)
    axes = [userdata_data.axes[i] for i in range(userdata_data.axes_length)]
    keepdims = userdata_data.keepdims

    axes = tuple(axes) if len(axes) else None
    rmean_result = np.mean(tensor, axis=axes, keepdims=keepdims)

    set_tensor_out(tensor_out, rmean_result)


@ggml_operator("ReduceMean")
def ggml_operator_reduce_mean(
    backend: "GgmlBackendRep",
    node: NodeProto,
    tensors_dict: Dict[str, ggml.ggml_tensor_p],
    context: ggml.ggml_context_p,
    refs: List[Any],
):
    node_inputs = [tensors_dict[inp] for inp in node.input]

    if len(node_inputs) > 2 or len(node_inputs) < 1:
        raise ValueError(
            f'Error for node "{node.name}": Operation "ReduceMean" requires at least one input. Actual number of inputs: {len(node_inputs)}'
        )

    input_tensor = node_inputs[0]

    tensor_shape = get_tensor_shape(input_tensor)
    tensor_dtype = get_tensor_dtype(input_tensor)

    axes = next((attr.ints for attr in node.attribute if attr.name == "axes"), None)
    if not axes:
        if len(node_inputs) != 2:
            raise ValueError(
                f'Error for node "{node.name}": Operation "ReduceMean" requires an axis.'
            )

        axes_eval = backend.eval_tensor(node_inputs[1], context)
        axes = ggml.utils.to_numpy(axes_eval)

    keepdims = next((attr.i for attr in node.attribute if attr.name == "keepdims"), 0)

    rmean_userdata = ReduceMeanUserData(list(axes), keepdims)
    userdata_p = ctypes.cast(ctypes.pointer(rmean_userdata), ctypes.c_void_p)

    output_shape = tuple([1] * len(tensor_shape)) if keepdims else ()

    if len(axes):
        output_shape = list(tensor_shape)
        sorted_axes = sorted(axes, reverse=True)

        for axis in sorted_axes:
            if keepdims:
                output_shape[axis] = 1
            else:
                output_shape.pop(axis)

    output_shape = tuple(output_shape)
    x = np.empty(output_shape, dtype=tensor_dtype)
    x_t = ggml.utils.from_numpy(x, context)

    new_tensor = tensors_dict[node.output[0]] = ggml.ggml_map_custom2_inplace(
        context,
        x_t,
        input_tensor,
        custom_reduce_mean,
        1,
        userdata_p,
    )

    refs.append(rmean_userdata)

    return new_tensor


class ReduceSumUserData(ctypes.Structure):
    _fields_ = [
        ("axes", ctypes.POINTER(ctypes.c_int)),
        ("axes_length", ctypes.c_int),
        ("keepdims", ctypes.c_int),
    ]

    def __init__(self, axes, keepdims):
        if isinstance(axes, list):
            self.axes_length = len(axes)
            self.axes = (ctypes.c_int * self.axes_length)(*axes)
        else:
            raise ValueError("axes should be a list of integers")

        self.keepdims = keepdims


@ggml.ggml_custom2_op_t
def custom_reduce_sum(
    tensor_out: ggml.ggml_tensor_p,
    tensor_in_1: ggml.ggml_tensor_p,
    tensor_in_2: ggml.ggml_tensor_p,
    ith: int,
    nth: int,
    userdata: Optional[ctypes.c_void_p],
):
    userdata_data_ptr = ctypes.cast(userdata, ctypes.POINTER(ReduceSumUserData))
    userdata_data = userdata_data_ptr.contents

    tensor = ggml.utils.to_numpy(tensor_in_2)
    axes = [userdata_data.axes[i] for i in range(userdata_data.axes_length)]
    keepdims = userdata_data.keepdims

    axes = tuple(axes) if len(axes) else None
    result = np.sum(tensor, axis=axes, keepdims=keepdims)

    set_tensor_out(tensor_out, result)


@ggml_operator("ReduceSum")
def ggml_operator_reduce_sum(
    backend: "GgmlBackendRep",
    node: NodeProto,
    tensors_dict: Dict[str, ggml.ggml_tensor_p],
    context: ggml.ggml_context_p,
    refs: List[Any],
):
    node_inputs = [tensors_dict[inp] for inp in node.input]

    if len(node_inputs) > 2 or len(node_inputs) < 1:
        raise ValueError(
            f'Error for node "{node.name}": Operation "ReduceSum" requires at least one input. Actual number of inputs: {len(node_inputs)}'
        )

    input_tensor = node_inputs[0]

    tensor_shape = get_tensor_shape(input_tensor)
    tensor_dtype = get_tensor_dtype(input_tensor)

    axes = next((attr.ints for attr in node.attribute if attr.name == "axes"), None)
    if not axes:
        if len(node_inputs) != 2:
            raise ValueError(
                f'Error for node "{node.name}": Operation "ReduceMean" requires an axis.'
            )

        axes_eval = backend.eval_tensor(node_inputs[1], context)
        axes = ggml.utils.to_numpy(axes_eval)

    keepdims = next((attr.i for attr in node.attribute if attr.name == "keepdims"), 0)

    rmean_userdata = ReduceSumUserData(list(axes), keepdims)
    userdata_p = ctypes.cast(ctypes.pointer(rmean_userdata), ctypes.c_void_p)

    output_shape = tuple([1] * len(tensor_shape)) if keepdims else ()

    if len(axes):
        output_shape = list(tensor_shape)
        sorted_axes = sorted(axes, reverse=True)

        for axis in sorted_axes:
            if keepdims:
                output_shape[axis] = 1
            else:
                output_shape.pop(axis)

    output_shape = tuple(output_shape)
    x = np.empty(output_shape, dtype=tensor_dtype)
    x_t = ggml.utils.from_numpy(x, context)

    new_tensor = tensors_dict[node.output[0]] = ggml.ggml_map_custom2_inplace(
        context,
        x_t,
        input_tensor,
        custom_reduce_sum,
        1,
        userdata_p,
    )

    refs.append(rmean_userdata)

    return new_tensor


@ggml_operator("Relu")
def ggml_operator_relu(
    backend: "GgmlBackendRep",
    node: NodeProto,
    tensors_dict: Dict[str, ggml.ggml_tensor_p],
    context: ggml.ggml_context_p,
    refs: List[Any],
):
    node_inputs = [tensors_dict[inp] for inp in node.input]

    if len(node_inputs) != 1:
        raise ValueError(
            f'Error for node "{node.name}": Operation "Relu" requires exactly one input. Actual number of inputs: {len(node_inputs)}'
        )

    output_name = node.output[0]
    a = node_inputs[0]

    relu_result = ggml.ggml_relu(
        context,
        a,
    )
    tensors_dict[output_name] = relu_result
    return relu_result


@ggml_operator("Reshape")
def ggml_operator_reshape(
    backend: "GgmlBackendRep",
    node: NodeProto,
    tensors_dict: Dict[str, ggml.ggml_tensor_p],
    context: ggml.ggml_context_p,
    refs: List[Any],
):
    node_inputs = [tensors_dict[inp] for inp in node.input]
    if len(node_inputs) != 2:
        raise ValueError(
            f'Error for node "{node.name}": Operation "Reshape" requires exactly two inputs. Actual number of inputs: {len(node_inputs)}'
        )

    try:
        allowzero_attr = next(
            attr for attr in node.attribute if attr.name == "allowzero"
        )
        allowzero = allowzero_attr.i == 1
    except StopIteration:
        allowzero = False

    a = node_inputs[0]
    b = node_inputs[1]
    eval_b = backend.eval_tensor(b, context)

    new_shape = ggml.utils.to_numpy(eval_b).astype(dtype=np.int32)

    old_shape = get_tensor_shape(a)
    if not allowzero:
        keep_idxs = np.where(new_shape == 0)[0]
        new_shape[keep_idxs] = np.array(old_shape)[keep_idxs]

    temp_a = np.empty(old_shape, dtype=get_tensor_dtype(a))
    x = temp_a.reshape(new_shape)
    x_t = ggml.utils.from_numpy(x, context)

    @ggml.ggml_custom2_op_t
    def custom_reshape(
        tensor_out: ggml.ggml_tensor_p,
        tensor_in_1: ggml.ggml_tensor_p,
        tensor_in_2: ggml.ggml_tensor_p,
        ith: int,
        nth: int,
        userdata: Optional[ctypes.c_void_p],
    ):
        x = ggml.utils.to_numpy(tensor_in_2)
        x_reshape = np.reshape(x, new_shape)
        set_tensor_out(tensor_out, x_reshape)

    new_tensor = tensors_dict[node.output[0]] = ggml.ggml_map_custom2_inplace(
        context,
        x_t,
        a,
        custom_reshape,
        1,
        None,
    )

    refs.append(custom_reshape)

    return new_tensor


@ggml_operator("Shape")
def ggml_operator_shape(
    backend: "GgmlBackendRep",
    node: NodeProto,
    tensors_dict: Dict[str, ggml.ggml_tensor_p],
    context: ggml.ggml_context_p,
    refs: List[Any],
):
    node_inputs = [tensors_dict[inp] for inp in node.input]

    if len(node_inputs) != 1:
        raise ValueError(
            f'Error for node "{node.name}": Operation "Shape" requires exactly one input. Actual number of inputs: {len(node_inputs)}'
        )

    tensor_shape = np.array(get_tensor_shape(node_inputs[0]), dtype=np.int32)
    name = node.output[0]
    start = next((attr.i for attr in node.attribute if attr.name == "start"), None)
    end = next(
        (attr.i for attr in node.attribute if attr.name == "end"),
        None,
    )
    shape_slice = tensor_shape[start:end]
    new_tensor = tensors_dict[name] = ggml.utils.from_numpy(shape_slice, context)

    ggml.ggml_set_name(new_tensor, (name + f"<int64>").encode())

    return new_tensor


@ggml_operator("Softmax")
def ggml_operator_softmax(
    backend: "GgmlBackendRep",
    node: NodeProto,
    tensors_dict: Dict[str, ggml.ggml_tensor_p],
    context: ggml.ggml_context_p,
    refs: List[Any],
):
    node_inputs = [tensors_dict[inp] for inp in node.input]

    if len(node_inputs) != 1:
        raise ValueError(
            f'Error for node "{node.name}": Operation "Softmax" requires exactly one input. Actual number of inputs: {len(node_inputs)}'
        )

    output_name = node.output[0]
    a = node_inputs[0]

    soft_max_result = ggml.ggml_soft_max(
        context,
        a,
    )
    tensors_dict[output_name] = soft_max_result
    return soft_max_result


@ggml_operator("Sqrt")
def ggml_operator_sqrt(
    backend: "GgmlBackendRep",
    node: NodeProto,
    tensors_dict: Dict[str, ggml.ggml_tensor_p],
    context: ggml.ggml_context_p,
    refs: List[Any],
):
    node_inputs = [tensors_dict[inp] for inp in node.input]

    if len(node_inputs) != 1:
        raise ValueError(
            f'Error for node "{node.name}": Operation "Sqrt" requires exactly one input. Actual number of inputs: {len(node_inputs)}'
        )

    output_name = node.output[0]
    a = node_inputs[0]

    sqrt_result = ggml.ggml_sqrt(
        context,
        a,
    )
    tensors_dict[output_name] = sqrt_result
    return sqrt_result


@ggml_operator("Sub")
def ggml_operator_sub(
    backend: "GgmlBackendRep",
    node: NodeProto,
    tensors_dict: Dict[str, ggml.ggml_tensor_p],
    context: ggml.ggml_context_p,
    refs: List[Any],
):
    node_inputs = [tensors_dict[inp] for inp in node.input]

    if len(node_inputs) != 2:
        raise ValueError(
            f'Error for node "{node.name}": Operation "Sub" requires exactly two inputs. Actual number of inputs: {len(node_inputs)}'
        )

    output_name = node.output[0]
    a, b = node_inputs
    a, b = broadcast_shapes(context, a, b)

    sub_result = ggml.ggml_sub(
        context,
        a,
        b,
    )
    tensors_dict[output_name] = sub_result
    return sub_result


@ggml_operator("Transpose")
def ggml_operator_transpose(
    backend: "GgmlBackendRep",
    node: NodeProto,
    tensors_dict: Dict[str, ggml.ggml_tensor_p],
    context: ggml.ggml_context_p,
    refs: List[Any],
):
    node_inputs = [tensors_dict[inp] for inp in node.input]

    if len(node_inputs) != 1:
        raise ValueError(
            f'Error for node "{node.name}": Operation "Transpose" requires exactly one input. Actual number of inputs: {len(node_inputs)}'
        )

    output_name = node.output[0]
    x = node_inputs[0]
    input_shape = get_tensor_shape(x)

    perm_map = {1: [0, 1, 2, 3], 2: [1, 0, 2, 3], 3: [2, 1, 0, 3], 4: [3, 2, 1, 0]}

    perm_attr = next((attr for attr in node.attribute if attr.name == "perm"), None)

    # add special case and -> fix me comments

    if perm_attr is None:
        perms = perm_map.get(len(input_shape), [1, 0, 2, 3])
    else:
        perms = list(perm_attr.ints)
        perms += [0, 1, 2, 3][len(perms) :]

    ax0, ax1, ax2, ax3 = perms
    dims = ggml.utils.get_ndims(x)

    if dims > 3:
        raise ValueError(
            "n_dims cannot be more than 3. 4D permutations may not work"
        )  # FIXME: 2,3D permutations are fine 4d is not. Passes ONNX test

    if dims == 3 and f"02" in "".join([str(perm) for perm in perms]):
        x = ggml.ggml_transpose(context, x)

    transpose_result = ggml.ggml_permute(context, x, ax0, ax1, ax2, ax3)

    if dims == 3 and f"02" in "".join([str(perm) for perm in perms]):
        transpose_result = ggml.ggml_permute(context, transpose_result, 0, 2, 1, 3)

    tensors_dict[output_name] = transpose_result
    return transpose_result


@ggml.ggml_custom3_op_t
def custom_unsqueeze(
    tensor_out: ggml.ggml_tensor_p,
    tensor_in_1: ggml.ggml_tensor_p,
    tensor_in_2: ggml.ggml_tensor_p,
    tensor_in_3: ggml.ggml_tensor_p,
    ith: int,
    nth: int,
    userdata: Optional[ctypes.c_void_p],
):
    x = ggml.utils.to_numpy(tensor_in_2)
    axes = ggml.utils.to_numpy(tensor_in_3)

    axes_values = [ax if ax >= 0 else ax + x.ndim + 1 for ax in axes]
    axes_values.sort()
    axes_values = np.array(axes_values)
    for axis in axes_values:
        x = np.expand_dims(x, axis=axis)

    set_tensor_out(tensor_out, x)


@ggml_operator("Unsqueeze")
def ggml_operator_unsqueeze(
    backend: "GgmlBackendRep",
    node: NodeProto,
    tensors_dict: Dict[str, ggml.ggml_tensor_p],
    context: ggml.ggml_context_p,
    refs: List[Any],
):
    node_inputs = [tensors_dict[inp] for inp in node.input]

    if len(node_inputs) != 2:
        raise ValueError(
            f'Error for node "{node.name}": Operation "Unsqueeze" requires exactly two inputs, data and axes. Actual number of inputs: {len(node_inputs)}'
        )

    data = node_inputs[0]
    axes_input = node_inputs[1]

    x_shape = get_tensor_shape(data)
    x_dtype = get_tensor_dtype(data)
    x_ndims = ggml.utils.get_ndims(data)

    axes_eval = backend.eval_tensor(axes_input, context)
    axes = ggml.utils.to_numpy(axes_eval).astype(dtype=np.int32)

    axes_values = [ax if ax >= 0 else ax + x_ndims + 1 for ax in axes]
    axes_values.sort()

    dummy_data = np.empty(x_shape)
    for axis in axes_values:
        dummy_data = np.expand_dims(dummy_data, axis=axis)

    ggml_type = map_to_ggml_type(x_dtype)
    new_shape = tuple(reversed(dummy_data.shape))

    if len(new_shape) > 4:
        raise ValueError(
            f'Error for node "{node.name}": {len(new_shape)}D arrays are not allowed.'
        )

    x_t = ggml.ggml_new_tensor(
        context,
        ggml_type.value,
        len(new_shape),
        (ctypes.c_int64 * len(new_shape))(*new_shape),
    )

    new_tensor = tensors_dict[node.output[0]] = ggml.ggml_map_custom3_inplace(
        context,
        x_t,
        data,
        axes_input,
        custom_unsqueeze,
        1,
        None,
    )

    return new_tensor


@ggml.ggml_custom3_op_t
def custom_where(
    tensor_out: ggml.ggml_tensor_p,
    tensor_in_1: ggml.ggml_tensor_p,
    tensor_in_2: ggml.ggml_tensor_p,
    tensor_in_3: ggml.ggml_tensor_p,
    ith: int,
    nth: int,
    userdata: Optional[ctypes.c_void_p],
):
    x = ggml.utils.to_numpy(tensor_in_1)
    y = ggml.utils.to_numpy(tensor_in_2)
    condition_array = ggml.utils.to_numpy(tensor_in_3)
    new_tensor = np.where(condition_array, x, y)
    set_tensor_out(tensor_out, new_tensor)


@ggml_operator("Where")
def ggml_operator_where(
    backend: "GgmlBackendRep",
    node: NodeProto,
    tensors_dict: Dict[str, ggml.ggml_tensor_p],
    context: ggml.ggml_context_p,
    refs: List[Any],
):
    node_inputs = [tensors_dict[inp] for inp in node.input]

    if len(node_inputs) != 3:
        raise ValueError(
            f'Error for node "{node.name}": Operation "Where" requires exactly three inputs. Actual number of inputs: {len(node_inputs)}'
        )

    new_tensor = tensors_dict[node.output[0]] = ggml.ggml_map_custom3_inplace(
        context,
        node_inputs[1],
        node_inputs[2],
        node_inputs[0],
        custom_where,
        1,
        None,
    )

    return new_tensor


class GgmlBackendRep(BackendRep):
    def __init__(
        self,
        graph,
        weights,
        weights_buffer,
        inputs,
        outputs,
        ggml_context,
        ggml_init_params,
    ):
        super(GgmlBackendRep, self).__init__()
        self.graph = graph
        self.weights = weights
        self.weights_buffer = weights_buffer
        self.inputs = inputs
        self.outputs = outputs
        self.ggml_context = ggml_context
        self.ggml_init_params = ggml_init_params

    def __del__(self):
        if hasattr(self, "ggml_context"):
            ggml.ggml_free(self.ggml_context)

    def eval_tensor(self, tensor: ggml.ggml_tensor_p, context: ggml.ggml_context_p):
        gf = ggml.ggml_build_forward(tensor)
        ggml.ggml_graph_compute_with_ctx(context, ctypes.pointer(gf), 1)

        return tensor

    def run(self, inputs: Any, **kwargs: Any) -> Tuple[Any, ...]:
        """Run the model with the specified inputs."""

        if isinstance(inputs, list):
            inputs = {k.name: v for k, v in zip(self.inputs, inputs)}

        assert isinstance(inputs, dict)

        model_graph = self.graph
        exit_node = None
        ggml_tensors = self.weights

        # Define context
        params = ggml.ggml_init_params(mem_size=16 * 1024 * 1024, mem_buffer=None)
        context = ggml.ggml_init(params=params)

        refs: List[Any] = []

        # Create entry inputs
        for model_input in model_graph.input:
            input_name = model_input.name
            input_data = np.array(inputs[input_name])

            # Check if the input includes expected values
            if input_name not in inputs:
                raise KeyError(f'"{input_name}" must be included in the inputs.')

            # Check for rank of input
            expected_rank = len(list(model_input.type.tensor_type.shape.dim))
            actual_rank = input_data.ndim

            if expected_rank != actual_rank:
                raise ValueError(
                    f"INVALID_ARGUMENT : Invalid rank for input: {input_name} Got: {actual_rank} Expected: {expected_rank} Please fix either the inputs or the model."
                )

            # Check for input types + allow for type casting
            expected_dtype = onnx_dtype_map[model_input.type.tensor_type.elem_type]

            try:
                input_data.astype(expected_dtype)
            except:
                raise ValueError(
                    f'INVALID_ARGUMENT : Unexpected input data type for "{input_name}". Actual: {input_data.dtype}, expected: {expected_dtype}'
                )

            # Create the input tensors with the correct type/shape
            ggml_type = map_to_ggml_type(input_data.dtype)
            shape = tuple(reversed(input_data.shape))

            tensor = ggml.ggml_new_tensor(
                context,
                ggml_type.value,
                len(shape),
                (ctypes.c_int64 * len(shape))(*shape),
            )

            ggml_tensors[input_name] = tensor

        # Set user inputs
        for key, value in inputs.items():
            set_tensor_out(ggml_tensors[key], value)

        gf = ggml.ggml_cgraph()
        gf_p = ctypes.pointer(gf)
        output_names = [output.name for output in model_graph.output]

        # Build layers
        for node in model_graph.node:
            operator_func = ggml_operators.get(node.op_type)
            if operator_func is None:
                raise NotImplementedError(f'Operator "{node.op_type}" not implemented')

            operator_func(
                self,
                node,
                ggml_tensors,
                context,
                refs,
            )

            for output in node.output:
                if output in output_names:
                    ggml.ggml_build_forward_expand(gf_p, ggml_tensors[output])

        # Compute graph
        ggml.ggml_graph_compute_with_ctx(context, gf_p, 1)

        graph_outputs = []
        for output in self.outputs:
            exit_node = ggml_tensors[output.name]
            graph_output = ggml.utils.to_numpy(
                exit_node
            )  # TODO: Add checks to convert values back to bool or etc types
            graph_output = graph_output.astype(
                get_final_dtype(exit_node)
            )  # TODO: add a second dict to keep track of types and use that instead
            graph_outputs.append(graph_output)

        ggml.ggml_free(context)

        return graph_outputs


class GgmlRuntimeBackend(Backend):
    @classmethod
    def is_opset_supported(cls, model: ModelProto):
        return True, ""

    @classmethod
    def prepare(cls, model: ModelProto, device: str="CPU", **kwargs):
        """Load the model and creates the ggml runtime backend representation
        for the onnx graph.

        Parameters:
            model: ModelProto (returned by `onnx.load`),
            device: requested device for the computation

        Returns:
            GGML Backend Representation"""

        super(GgmlRuntimeBackend, cls).prepare(model, device, **kwargs)
        graph = model.graph
        weights = {}

        n_tensors = len(graph.initializer)
        init_params = ggml.ggml_init_params(
            mem_size=n_tensors * ggml.ggml_tensor_overhead(),
            no_alloc=True,
        )

        context = ggml.ggml_init(init_params)
        total_nbytes = 0

        pairs = []

        for initializer in graph.initializer:
            name = initializer.name
            np_array = onnx.numpy_helper.to_array(initializer)
            if can_quantize(np_array, name, graph):
                ggml_qtype = ggml.utils.GGML_TYPE.Q8_0
                shape = tuple(reversed(np_array.shape))
                tensor = ggml.ggml_new_tensor(
                    context,
                    ggml_qtype.value,
                    len(shape),
                    (ctypes.c_int64 * len(shape))(*shape),
                )

            else:
                tensor = ggml.utils.from_numpy(x=np_array, ctx=context)

            ggml.ggml_set_name(tensor=tensor, name=name.encode())
            total_nbytes += ggml.ggml_nbytes_pad(tensor)
            weights[name] = tensor
            pairs.append((tensor, initializer))

        buffer = (ctypes.c_uint8 * total_nbytes)()
        offset = 0

        for tensor, initializer in pairs:
            nbytes = ggml.ggml_nbytes_pad(tensor)
            tensor.contents.data = ctypes.cast(
                ctypes.addressof(buffer) + offset, ctypes.c_void_p
            )

            np_array = onnx.numpy_helper.to_array(initializer)
            if ggml.ggml_is_quantized(tensor.contents.type):
                np_c_float_data = (ctypes.c_float * np_array.size).from_address(
                    ctypes.addressof(np_array.ctypes.data)
                )

                ggml.utils.quantize_0(
                    np_c_float_data,
                    np_array.size,
                    np_array.shape[0],
                    ggml_qtype,
                    work=ctypes.cast(
                        ctypes.addressof(buffer) + offset, ctypes.c_void_p
                    ),
                )

            else:
                set_tensor_out(tensor, np_array)

            offset += nbytes

        return GgmlBackendRep(
            graph=graph,
            weights=weights,
            weights_buffer=buffer,
            inputs=graph.input,
            outputs=graph.output,
            ggml_context=context,
            ggml_init_params=init_params,
        )

    @classmethod
    def run_model(
        cls, model: ModelProto, inputs: Any, device: Optional[str]=None, **kwargs
    ) -> Tuple[Any, ...]:
        """Compute the prediction.
        """
        rep = cls.prepare(model, device, **kwargs)
        return rep.run(inputs, **kwargs)

    @classmethod
    def run_node(
        cls, node: NodeProto, inputs: Any, device: Optional[str]=None, outputs_info=None, **kwargs
    ) -> Tuple[Any, ...]:
        """
        This method is not implemented as it is much more efficient
        to run a whole model than every node independently.
        """
        raise NotImplementedError(
            "It is much more efficient to run a whole model than every node independently."
        )

class GgmlOnnxGraphOptimizerRule:
    """Base class for a graph optimization rule."""

    def __init__(self, name: str):
        self.name = name

    def apply(self, model: ModelProto) -> Optional[ModelProto]:
        """Apply the optimization rule to the given ONNX model."""
        raise NotImplementedError()

class GgmlOnnxGraphOptimizer:
    """Optimize an ONNX graph for the GGML runtime."""
    def __init__(self, model: ModelProto, rules: List[GgmlOnnxGraphOptimizerRule]):
        self.model = model
        self.rules = rules

    def optimize(self) -> ModelProto:
        """Apply the optimization rules to the ONNX model until there are no
        more optimizations left to perform.
        
        NOTE: This is a naive implementation that applies the rules in order until
        no more rules can be applied."""
        model = self.model
        while True:
            for rule in self.rules:
                new_model = rule.apply(model)
                if new_model is not None:
                    model = new_model
                    break
            else:
                break
        return model