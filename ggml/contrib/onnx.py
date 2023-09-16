"""GGML ONNX backend.

This module implements a GGML backend for ONNX models and operators.
"""
import ctypes
import math
import re
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import onnx
from onnx.backend.base import Backend, BackendRep
from onnx.helper import np_dtype_to_tensor_dtype, tensor_dtype_to_np_dtype
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
def ggml_operator_abs(ctx: "GgmlOnnxExecutionContext", node: NodeProto):
    node_inputs = [ctx.tensors_dict[inp] for inp in node.input]

    if len(node_inputs) != 1:
        raise ValueError(
            f'Error for node "{node.name}": Operation "Abs" requires exactly one input. Actual number of inputs: {len(node_inputs)}'
        )

    output_name = node.output[0]
    a = node_inputs[0]

    abs_result = ggml.ggml_abs(
        ctx.ggml_context,
        a,
    )
    ctx.tensors_dict[output_name] = abs_result
    return abs_result


@ggml_operator("Add")
def ggml_operator_add(ctx: "GgmlOnnxExecutionContext", node: NodeProto):
    node_inputs = [ctx.tensors_dict[inp] for inp in node.input]

    if len(node_inputs) != 2:
        raise ValueError(
            f'Error for node "{node.name}": Operation "Add" requires exactly two inputs. Actual number of inputs: {len(node_inputs)}'
        )

    output_name = node.output[0]

    a, b = node_inputs
    a, b = broadcast_shapes(ctx.ggml_context, a, b)

    add_result = ggml.ggml_add(
        ctx.ggml_context,
        a,
        b,
    )
    ctx.tensors_dict[output_name] = add_result
    return add_result


@ggml.ggml_custom3_op_t
def custom_and(
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

    x = np.logical_and(a, b)

    set_tensor_out(tensor_out, x)


@ggml_operator("And")
def ggml_operator_and(ctx: "GgmlOnnxExecutionContext", node: NodeProto):
    node_inputs = [ctx.tensors_dict[inp] for inp in node.input]

    if len(node_inputs) != 2:
        raise ValueError(
            f'Error for node "{node.name}": Operation "And" requires exactly two inputs. Actual number of inputs: {len(node_inputs)}'
        )

    a_shape = get_tensor_shape(node_inputs[0])
    a_dtype = get_tensor_dtype(node_inputs[0])
    b_shape = get_tensor_shape(node_inputs[1])
    name = node.output[0]

    output_shape = np.broadcast(np.empty(a_shape), np.empty(b_shape)).shape

    x = np.empty(output_shape, dtype=a_dtype)
    x_t = ggml.utils.from_numpy(x, ctx.ggml_context)

    new_tensor = ctx.tensors_dict[name] = ggml.ggml_map_custom3_inplace(
        ctx.ggml_context,
        x_t,
        node_inputs[0],
        node_inputs[1],
        custom_and,
        1,
        None,
    )

    ggml.ggml_set_name(new_tensor, (name + f"<bool>").encode())

    return new_tensor


class ArgOpsUserData(ctypes.Structure):
    _fields_ = [
        ("axis", ctypes.c_int),
        ("keepdims", ctypes.c_int),
        ("select_last_index", ctypes.c_int),
    ]


@ggml.ggml_custom2_op_t
def custom_arg_max(
    tensor_out: ggml.ggml_tensor_p,
    tensor_in_1: ggml.ggml_tensor_p,
    tensor_in_2: ggml.ggml_tensor_p,
    ith: int,
    nth: int,
    userdata: Optional[ctypes.c_void_p],
):
    x = ggml.utils.to_numpy(tensor_in_2)
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

    set_tensor_out(tensor_out, y)


@ggml_operator("ArgMax")
def ggml_operator_arg_max(ctx: "GgmlOnnxExecutionContext", node: NodeProto):
    node_inputs = [ctx.tensors_dict[inp] for inp in node.input]

    if len(node_inputs) != 1:
        raise ValueError(
            f'Error for node "{node.name}": Operation "ArgMax" requires exactly two inputs, data and axes. Actual number of inputs: {len(node_inputs)}'
        )

    data = node_inputs[0]
    name = node.output[0]

    axis = next((attr.i for attr in node.attribute if attr.name == "axis"), 0)
    keepdims = next((attr.i for attr in node.attribute if attr.name == "keepdims"), 1)
    select_last_index = next(
        (attr.i for attr in node.attribute if attr.name == "select_last_index"), 0
    )

    x_shape = get_tensor_shape(data)
    x_dtype = get_tensor_dtype(data)
    x_ndims = ggml.utils.get_ndims(data)

    dummpy_data = np.empty(x_shape, dtype=np.int32)

    if select_last_index:
        dummpy_data = np.flip(dummpy_data, axis)

    dummy_result = np.argmax(dummpy_data, axis=axis)

    if select_last_index:
        dummy_result = dummpy_data.shape[axis] - dummy_result - 1

    if keepdims:
        dummy_result = np.expand_dims(dummy_result, axis)

    dummy_result = dummy_result.astype(np.int32)

    x_t = ggml.utils.from_numpy(dummy_result, ctx.ggml_context)

    argmax_userdata = ArgOpsUserData(axis, keepdims, select_last_index)
    userdata_p = ctypes.cast(ctypes.pointer(argmax_userdata), ctypes.c_void_p)

    new_tensor = ctx.tensors_dict[name] = ggml.ggml_map_custom2_inplace(
        ctx.ggml_context,
        x_t,
        data,
        custom_arg_max,
        1,
        userdata_p,
    )

    ggml.ggml_set_name(new_tensor, (name + "<int64>").encode())
    ctx.refs.append(argmax_userdata)

    return new_tensor


@ggml.ggml_custom2_op_t
def custom_arg_min(
    tensor_out: ggml.ggml_tensor_p,
    tensor_in_1: ggml.ggml_tensor_p,
    tensor_in_2: ggml.ggml_tensor_p,
    ith: int,
    nth: int,
    userdata: Optional[ctypes.c_void_p],
):
    x = ggml.utils.to_numpy(tensor_in_2)
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

    set_tensor_out(tensor_out, y)


@ggml_operator("ArgMin")
def ggml_operator_arg_max(ctx: "GgmlOnnxExecutionContext", node: NodeProto):
    node_inputs = [ctx.tensors_dict[inp] for inp in node.input]

    if len(node_inputs) != 1:
        raise ValueError(
            f'Error for node "{node.name}": Operation "ArgMin" requires exactly two inputs, data and axes. Actual number of inputs: {len(node_inputs)}'
        )

    data = node_inputs[0]
    name = node.output[0]

    axis = next((attr.i for attr in node.attribute if attr.name == "axis"), 0)
    keepdims = next((attr.i for attr in node.attribute if attr.name == "keepdims"), 1)
    select_last_index = next(
        (attr.i for attr in node.attribute if attr.name == "select_last_index"), 0
    )

    x_shape = get_tensor_shape(data)

    dummpy_data = np.empty(x_shape, dtype=np.int32)

    if select_last_index:
        dummpy_data = np.flip(dummpy_data, axis)

    dummy_result = np.argmin(dummpy_data, axis=axis)

    if select_last_index:
        dummy_result = dummpy_data.shape[axis] - dummy_result - 1

    if keepdims:
        dummy_result = np.expand_dims(dummy_result, axis)

    dummy_result = dummy_result.astype(np.int32)

    x_t = ggml.utils.from_numpy(dummy_result, ctx.ggml_context)

    argmax_userdata = ArgOpsUserData(axis, keepdims, select_last_index)
    userdata_p = ctypes.cast(ctypes.pointer(argmax_userdata), ctypes.c_void_p)

    new_tensor = ctx.tensors_dict[name] = ggml.ggml_map_custom2_inplace(
        ctx.ggml_context,
        x_t,
        data,
        custom_arg_min,
        1,
        userdata_p,
    )

    ggml.ggml_set_name(new_tensor, (name + "<int64>").encode())
    ctx.refs.append(argmax_userdata)

    return new_tensor


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
def ggml_operator_cast(ctx: "GgmlOnnxExecutionContext", node: NodeProto):
    node_inputs = [ctx.tensors_dict[inp] for inp in node.input]

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

    x_t = ggml.utils.from_numpy(x, ctx.ggml_context)

    new_tensor = ctx.tensors_dict[node.output[0]] = ggml.ggml_map_custom2_inplace(
        ctx.ggml_context,
        x_t,
        a,
        custom_cast,
        1,
        ctypes.pointer(onnx_type_c),
    )

    ctx.refs.append(onnx_type_c)

    return new_tensor


@ggml_operator("CastLike")
def ggml_operator_castlike(ctx: "GgmlOnnxExecutionContext", node: NodeProto):
    node_inputs = [ctx.tensors_dict[inp] for inp in node.input]

    if len(node_inputs) != 2:
        raise ValueError(
            f'Error for node "{node.name}": Operation "CastLike" requires exactly two inputs. Actual number of inputs: {len(node_inputs)}'
        )
    a, b = node_inputs

    np_data_dtype = get_tensor_dtype(b)
    np_data_type_limit = np.dtype(str(np_data_dtype).replace("64", "32"))

    onnx_type = np_dtype_to_tensor_dtype(np_data_dtype)
    onnx_type_c = ctypes.c_int(onnx_type)

    x = np.empty(get_tensor_shape(a), dtype=np_data_type_limit)
    x_t = ggml.utils.from_numpy(x, ctx.ggml_context)

    new_tensor = ctx.tensors_dict[node.output[0]] = ggml.ggml_map_custom2_inplace(
        ctx.ggml_context,
        x_t,
        a,
        custom_cast,
        1,
        ctypes.pointer(onnx_type_c),
    )

    ctx.refs.append(onnx_type_c)

    return new_tensor


@ggml_operator("Ceil")
def ggml_operator_exp(ctx: "GgmlOnnxExecutionContext", node: NodeProto):
    node_inputs = [ctx.tensors_dict[inp] for inp in node.input]

    if len(node_inputs) != 1:
        raise ValueError(
            f'Error for node "{node.name}": Operation "Ceil" requires exactly one input. Actual number of inputs: {len(node_inputs)}'
        )
    a = node_inputs[0]
    np_dtype = get_tensor_dtype(a)

    x = np.empty(get_tensor_shape(a), dtype=np_dtype)
    x_t = ggml.utils.from_numpy(x, ctx.ggml_context)

    @ggml.ggml_custom1_op_t
    def custom_ceil(
        tensor_out: ggml.ggml_tensor_p,
        tensor_in_1: ggml.ggml_tensor_p,
        ith: int,
        nth: int,
        userdata: Optional[ctypes.c_void_p],
    ):
        tensor = ggml.utils.to_numpy(tensor_in_1)
        x = np.ceil(tensor)
        set_tensor_out(tensor_out, np.array(x))

    new_tensor = ctx.tensors_dict[node.output[0]] = ggml.ggml_map_custom1_inplace(
        ctx.ggml_context,
        x_t,
        custom_ceil,
        1,
        None,
    )

    ctx.refs.append(custom_ceil)

    return new_tensor


@ggml_operator("Concat")
def ggml_operator_concat(ctx: "GgmlOnnxExecutionContext", node: NodeProto):
    node_inputs = [ctx.tensors_dict[inp] for inp in node.input]

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
    x_t = ggml.utils.from_numpy(x, ctx.ggml_context)

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

    new_tensor = ctx.tensors_dict[node.output[0]] = ggml.ggml_map_custom1_inplace(
        ctx.ggml_context,
        x_t,
        custom_concat,
        1,
        None,
    )

    ctx.refs.append(custom_concat)

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
def ggml_operator_constant(ctx: "GgmlOnnxExecutionContext", node: NodeProto):
    node_attributes = node.attribute
    name = node.output[0]

    value_attr = next(attr for attr in node_attributes if "value" in attr.name)

    if value_attr.HasField("t"):
        tensor = value_attr.t
        data_type = tensor.data_type
        np_data_type = tensor_dtype_to_np_dtype(data_type)
        np_data_type_limit = np.dtype(str(np_data_type).replace("64", "32"))

        if tensor.raw_data:
            data_value = np.frombuffer(tensor.raw_data, dtype=np_data_type)
        else:
            data_value = onnx.numpy_helper.to_array(tensor)

    else:
        data_type = value_attr.type
        np_data_type = tensor_dtype_to_np_dtype(data_type)
        np_data_type_limit = np.dtype(str(np_data_type).replace("64", "32"))
        if np.issubdtype(np_data_type, np.floating):
            data_value = np.array(value_attr.f)
        elif np.issubdtype(np_data_type, np.integer):
            data_value = np.array(value_attr.i)
        else:
            raise ValueError(
                f'Error for node "{node.name}": Constant node not set correctly or incomplete implantation.'
            )

    data_tensor = ggml.utils.from_numpy(
        data_value.astype(np_data_type_limit),
        ctx.ggml_context,
    )

    tensor_shape = data_value.shape
    x = np.empty(tensor_shape, dtype=np_data_type_limit)

    x_t = ggml.utils.from_numpy(x, ctx.ggml_context)

    new_tensor = ctx.tensors_dict[node.output[0]] = ggml.ggml_map_custom2_inplace(
        ctx.ggml_context,
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
def ggml_operator_constant_of_shape(ctx: "GgmlOnnxExecutionContext", node: NodeProto):
    node_inputs = [ctx.tensors_dict[inp] for inp in node.input]

    if len(node_inputs) != 1:
        raise ValueError(
            f'Error for node "{node.name}": Operation "ConstantOfShape" requires exactly one input. Actual number of inputs: {len(node_inputs)}'
        )

    node_attributes = node.attribute
    value_attr = next(attr for attr in node_attributes if "value" in attr.name)

    if value_attr.HasField("t"):
        tensor = value_attr.t
        data_type = tensor.data_type
        np_data_type = tensor_dtype_to_np_dtype(data_type)
        np_data_type_limit = np.dtype(str(np_data_type).replace("64", "32"))

        if tensor.raw_data:
            data_value = np.frombuffer(tensor.raw_data, dtype=np_data_type)
        else:
            data_value = onnx.numpy_helper.to_array(tensor)

    else:
        data_type = value_attr.type
        np_data_type = tensor_dtype_to_np_dtype(data_type)
        np_data_type_limit = np.dtype(str(np_data_type).replace("64", "32"))
        if np.issubdtype(np_data_type, np.floating):
            data_value = np.array(value_attr.f)
        elif np.issubdtype(np_data_type, np.integer):
            data_value = np.array(value_attr.i)
        else:
            raise ValueError(
                f'Error for node "{node.name}": Constant node not set correctly or incomplete implantation.'
            )

    data_tensor = ggml.utils.from_numpy(
        data_value.astype(np_data_type_limit),
        ctx.ggml_context,
    )

    shape = ggml.utils.to_numpy(node_inputs[0])

    x = np.empty(shape, dtype=np_data_type_limit)
    x_t = ggml.utils.from_numpy(x, ctx.ggml_context)

    new_tensor = ctx.tensors_dict[node.output[0]] = ggml.ggml_map_custom2_inplace(
        ctx.ggml_context,
        x_t,
        data_tensor,
        custom_constant_of_shape,
        1,
        None,
    )

    return new_tensor


@ggml_operator("Conv")
def ggml_operator_conv(ctx: "GgmlOnnxExecutionContext", node: NodeProto):
    node_inputs = [ctx.tensors_dict[inp] for inp in node.input]

    if len(node_inputs) < 2:
        raise ValueError(
            f'Error for node "{node.name}": Operation "Conv" requires 2 - 3 inputs. Actual number of inputs: {len(node_inputs)}'
        )

    node_inputs_iter = iter(node_inputs)
    x = next(node_inputs_iter)
    x_shape = get_tensor_shape(x)
    w = next(node_inputs_iter)
    w_shape = get_tensor_shape(w)
    m = w_shape[0]
    bias = next(
        node_inputs_iter,
        ggml.utils.from_numpy(
            np.full(m, 0, dtype=get_tensor_dtype(x)), ctx.ggml_context
        ),
    )

    auto_pad = next(
        (attr.s.decode() for attr in node.attribute if attr.name == "auto_pad"),
        "NOTSET",
    )
    dilations = next(
        (attr.ints for attr in node.attribute if attr.name == "dilations"),
        [1 for _ in x_shape[2:]],
    )
    group = next((attr.i for attr in node.attribute if attr.name == "group"), 1)
    kernel_shape = next(
        (attr.ints for attr in node.attribute if attr.name == "kernel_shape"),
        w_shape[2:],
    )
    pads = next(
        (attr.ints for attr in node.attribute if attr.name == "pads"),
        [0 for _ in x_shape[2:]] * 2,
    )
    strides = next(
        (attr.ints for attr in node.attribute if attr.name == "strides"),
        [1 for _ in x_shape[2:]],
    )

    # Source: https://github.com/onnx/onnx/blob/main/onnx/reference/ops/op_conv.py

    if auto_pad in {"SAME_LOWER", "SAME_UPPER", "VALID"}:
        head = []
        tail = []
        for i in range(len(x_shape) - 2):
            d = x_shape[i]
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

    if len(strides) != 2:
        raise NotImplementedError("Cannot handle other than 2 strides")

    raise NotImplementedError(f'Operator "Conv" not implemented')
    # FIXME: ggml can only work with F16
    conv_result = ggml.ggml_conv_2d(
        ctx.ggml_context,
        x,
        bias,
        strides[0],
        strides[1],
        pads[0],
        pads[1],
        dilations[0],
        dilations[1],
    )

    ctx.tensors_dict[node.output[0]] = conv_result
    return conv_result


@ggml_operator("ConvTranspose")
def ggml_operator_convtranspose(ctx: "GgmlOnnxExecutionContext", node: NodeProto):
    node_inputs = [ctx.tensors_dict[inp] for inp in node.input]

    if len(node_inputs) < 2:
        raise ValueError(
            f'Error for node "{node.name}": Operation "ConvTranspose" requires 2 - 3 inputs. Actual number of inputs: {len(node_inputs)}'
        )

    node_inputs_iter = iter(node_inputs)
    x = next(node_inputs_iter)
    x_shape = get_tensor_shape(x)
    w = next(node_inputs_iter)
    w_shape = get_tensor_shape(w)
    m = w_shape[0]
    bias = next(
        node_inputs_iter,
        ggml.utils.from_numpy(
            np.full(m, 0, dtype=get_tensor_dtype(x)), ctx.ggml_context
        ),
    )

    auto_pad = next(
        (attr.s.decode() for attr in node.attribute if attr.name == "auto_pad"),
        "NOTSET",
    )
    dilations = next(
        (attr.ints for attr in node.attribute if attr.name == "dilations"),
        [1 for _ in x_shape[2:]],
    )
    group = next((attr.i for attr in node.attribute if attr.name == "group"), 1)
    kernel_shape = next(
        (attr.ints for attr in node.attribute if attr.name == "kernel_shape"),
        w_shape[2:],
    )
    output_padding = next(
        (attr.ints for attr in node.attribute if attr.name == "output_padding"),
        [0 for _ in x_shape[2:]] * 2,
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

    # https://github.com/onnx/onnx/blob/main/onnx/reference/ops/op_conv_transpose.py

    if pads is None and auto_pad not in {"SAME_UPPER", "SAME_LOWER"}:
        pads = [0 for i in range(2 * len(strides))]
    if pads is None:
        if output_shape is None:
            output_shape = [x_shape[i + 2] * strides[i] for i in range(len(strides))]
        total_padding = [
            strides[i] * (x_shape[i + 2] - 1)
            + output_padding[i]
            + ((kernel_shape[i] - 1) * dilations[i] + 1)
            - output_shape[i]
            for i in range(len(output_shape))
        ]
        pads_1 = []
        pads_2 = []
        for i in range(len(output_shape)):
            if auto_pad == "SAME_UPPER":
                pads_1.append(total_padding[i] // 2)
                pads_2.append(total_padding[i] - (total_padding[i] // 2))
            else:
                pads_1.append(total_padding[i] - (total_padding[i] // 2))
                pads_2.append(total_padding[i] // 2)
        pads = pads_1 + pads_2
        n_dims = len(pads) // 2
    else:
        n_dims = len(x_shape) - 2
        new_pads = np.array([(pads[i], pads[i + n_dims]) for i in range(n_dims)])
        if output_shape is None:
            output_shape = [
                strides[i] * (x_shape[i + 2] - 1)
                + output_padding[i]
                + ((kernel_shape[i] - 1) * dilations[i] + 1)
                - new_pads[i, :].sum()
                for i in range(n_dims)
            ]

    kernel_shape = w_shape[2:]
    kernel_size = np.prod(kernel_shape)
    num_output_channels = w_shape[1] * group
    kernel_dim = num_output_channels // group * kernel_size

    C = x_shape[1]  # num_inputs_channels
    m = kernel_dim  # kernel_dim
    n = np.prod(x_shape[2:])  # input_image_size
    k = C // group

    if group != 1:
        raise NotImplementedError(
            f'Error for node "{node.name}": Implementation for group={group} > 1 is not available yet.'
        )

    raise NotImplementedError(f'Operator "ConvTranspose" not implemented')


class DepthToSpaceUserData(ctypes.Structure):
    _fields_ = [
        ("blocksize", ctypes.c_int),
        ("mode", ctypes.c_char_p),
    ]


@ggml.ggml_custom2_op_t
def custom_depth_to_space(
    tensor_out: ggml.ggml_tensor_p,
    tensor_in_1: ggml.ggml_tensor_p,
    tensor_in_2: ggml.ggml_tensor_p,
    ith: int,
    nth: int,
    userdata: Optional[ctypes.c_void_p],
):
    x = ggml.utils.to_numpy(tensor_in_2)
    userdata_data_ptr = ctypes.cast(userdata, ctypes.POINTER(DepthToSpaceUserData))
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

    set_tensor_out(tensor_out, y)


@ggml_operator("DepthToSpace")
def ggml_operator_depth_to_space(ctx: "GgmlOnnxExecutionContext", node: NodeProto):
    node_inputs = [ctx.tensors_dict[inp] for inp in node.input]

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

    N, C, H, W = get_tensor_shape(x)

    new_C = C // (blocksize**2)
    new_H = H * blocksize
    new_W = W * blocksize

    output_shape = (N, new_C, new_H, new_W)

    x_t = ggml.utils.from_numpy(
        np.empty(output_shape, dtype=get_tensor_dtype(x)), ctx.ggml_context
    )
    depthtospace_userdata = DepthToSpaceUserData(blocksize, mode)
    userdata_p = ctypes.cast(ctypes.pointer(depthtospace_userdata), ctypes.c_void_p)

    new_tensor = ctx.tensors_dict[node.output[0]] = ggml.ggml_map_custom2_inplace(
        ctx.ggml_context,
        x_t,
        x,
        custom_depth_to_space,
        1,
        userdata_p,
    )

    ctx.refs.append(depthtospace_userdata)

    return new_tensor


@ggml_operator("Div")
def ggml_operator_div(ctx: "GgmlOnnxExecutionContext", node: NodeProto):
    node_inputs = [ctx.tensors_dict[inp] for inp in node.input]

    if len(node_inputs) != 2:
        raise ValueError(
            f'Error for node "{node.name}": Operation "Div" requires exactly two inputs. Actual number of inputs: {len(node_inputs)}'
        )

    output_name = node.output[0]
    a = node_inputs[0]
    b = node_inputs[1]

    a, b = broadcast_shapes(ctx.ggml_context, a, b)

    div_result = ggml.ggml_div(
        ctx.ggml_context,
        a,
        b,
    )
    ctx.tensors_dict[output_name] = div_result
    return div_result


class DropoutUserData(ctypes.Structure):
    _fields_ = [
        ("seed", ctypes.c_int),
        ("training_mode", ctypes.c_bool),
    ]


@ggml.ggml_custom2_op_t
def custom_dropout_mask(
    tensor_out: ggml.ggml_tensor_p,
    tensor_in_1: ggml.ggml_tensor_p,
    tensor_in_2: ggml.ggml_tensor_p,
    ith: int,
    nth: int,
    userdata: Optional[ctypes.c_void_p],
):
    x = ggml.utils.to_numpy(tensor_in_1)
    ratio = ggml.utils.to_numpy(tensor_in_2)

    userdata_data_ptr = ctypes.cast(userdata, ctypes.POINTER(DropoutUserData))
    userdata_data = userdata_data_ptr.contents

    seed = userdata_data.seed
    training_mode = userdata_data.training_mode

    if np.equal(0, np.array(ratio)) or training_mode is False:
        mask = np.ones(x.shape, dtype=np.int32)

    else:
        np.random.seed(seed)
        mask = np.random.uniform(0, 1.0, x.shape) >= ratio

    set_tensor_out(tensor_out, mask)


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
    x = ggml.utils.to_numpy(tensor_in_1)
    ratio = ggml.utils.to_numpy(tensor_in_2)
    mask = ggml.utils.to_numpy(tensor_in_3)

    userdata_data_ptr = ctypes.cast(userdata, ctypes.POINTER(DropoutUserData))
    userdata_data = userdata_data_ptr.contents

    training_mode = userdata_data.training_mode

    if np.equal(0, np.array(ratio)) or training_mode is False:
        y = x

    else:
        scale = 1 / (1 - ratio)
        y = mask * x * scale

    set_tensor_out(tensor_out, y)


@ggml_operator("Dropout")
def ggml_operator_dropout(ctx: "GgmlOnnxExecutionContext", node: NodeProto):
    node_inputs = [ctx.tensors_dict[inp] for inp in node.input]

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
        ratio = ggml.utils.from_numpy(
            np.array([ratio]).astype(np.float32), ctx.ggml_context
        )

    seed = next((attr.i for attr in node.attribute if attr.name == "seed"), 6)

    if type(training_mode) is ggml.ggml_tensor_p:
        training_mode_eval = ctx.backend.eval_tensor(training_mode, ctx.ggml_context)
        training_mode = ggml.utils.to_numpy(training_mode_eval)

    droput_userdata = DropoutUserData(seed, bool(training_mode))
    userdata_p = ctypes.cast(ctypes.pointer(droput_userdata), ctypes.c_void_p)

    mask = ggml.ggml_map_custom2_inplace(
        ctx.ggml_context,
        data,
        ratio,
        custom_dropout_mask,
        1,
        userdata_p,
    )

    output = ggml.ggml_map_custom3_inplace(
        ctx.ggml_context,
        data,
        ratio,
        mask,
        custom_dropout_output,
        1,
        userdata_p,
    )

    ctx.refs.append(droput_userdata)

    if len(node.output) == 2:
        ggml.ggml_set_name(mask, (node.output[1] + f"<bool>").encode())
        ctx.tensors_dict[node.output[0]] = output
        ctx.tensors_dict[node.output[1]] = mask

        return output, mask

    ctx.tensors_dict[node.output[0]] = output
    return output


@ggml_operator("Elu")
def ggml_operator_elu(ctx: "GgmlOnnxExecutionContext", node: NodeProto):
    node_inputs = [ctx.tensors_dict[inp] for inp in node.input]

    if len(node_inputs) != 1:
        raise ValueError(
            f'Error for node "{node.name}": Operation "Elu" requires exactly one input. Actual number of inputs: {len(node_inputs)}'
        )

    output_name = node.output[0]
    x = node_inputs[0]
    alpha = next((attr.f for attr in node.attribute if attr.name == "alpha"), 1.0)

    Y = ggml.ggml_elu(
        ctx.ggml_context,
        x,
    )

    if alpha != 1.0:
        Y_eval = ctx.backend.eval_tensor(Y, ctx.ggml_context)
        Y_np = ggml.utils.to_numpy(Y_eval)
        Y_alpha = np.where(Y_np < 0, alpha * Y_np, Y_np)

        Y = ggml.utils.from_numpy(Y_alpha, ctx.ggml_context)

    ctx.tensors_dict[output_name] = Y
    return Y


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
def ggml_operator_equal(ctx: "GgmlOnnxExecutionContext", node: NodeProto):
    node_inputs = [ctx.tensors_dict[inp] for inp in node.input]

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
    x_t = ggml.utils.from_numpy(x, ctx.ggml_context)

    new_tensor = ctx.tensors_dict[name] = ggml.ggml_map_custom3_inplace(
        ctx.ggml_context,
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
def ggml_operator_exp(ctx: "GgmlOnnxExecutionContext", node: NodeProto):
    node_inputs = [ctx.tensors_dict[inp] for inp in node.input]

    if len(node_inputs) < 1:
        raise ValueError(
            f'Error for node "{node.name}": Operation "Exp" requires at least one input. Actual number of inputs: {len(node_inputs)}'
        )
    a = node_inputs[0]
    np_dtype = get_tensor_dtype(a)

    x = np.empty(get_tensor_shape(a), dtype=np_dtype)
    x_t = ggml.utils.from_numpy(x, ctx.ggml_context)

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

    new_tensor = ctx.tensors_dict[node.output[0]] = ggml.ggml_map_custom1_inplace(
        ctx.ggml_context,
        x_t,
        custom_exp,
        1,
        None,
    )

    ctx.refs.append(custom_exp)

    return new_tensor


@ggml.ggml_custom2_op_t
def custom_flatten(
    tensor_out: ggml.ggml_tensor_p,
    tensor_in_1: ggml.ggml_tensor_p,
    tensor_in_2: ggml.ggml_tensor_p,
    ith: int,
    nth: int,
    userdata: Optional[ctypes.c_void_p],
):
    x = ggml.utils.to_numpy(tensor_in_2)
    axis = ctypes.cast(userdata, ctypes.POINTER(ctypes.c_int)).contents.value

    if axis < 0:
        axis += len(x.shape)
    new_shape = (np.prod(x.shape[:axis]).astype(np.int32), -1)

    y = x.reshape(new_shape)

    set_tensor_out(tensor_out, y)


@ggml_operator("Flatten")
def ggml_operator_flatten(ctx: "GgmlOnnxExecutionContext", node: NodeProto):
    node_inputs = [ctx.tensors_dict[inp] for inp in node.input]

    if len(node_inputs) != 1:
        raise ValueError(
            f'Error for node "{node.name}": Operation "Flatten" requires exactly one input. Actual number of inputs: {len(node_inputs)}'
        )

    x = node_inputs[0]
    x_shape = get_tensor_shape(x)
    x_dtype = get_tensor_dtype(x)

    axis = next((attr.i for attr in node.attribute if attr.name == "axis"), 1)

    if axis < 0:
        axis += len(x_shape)

    new_shape = (np.prod(x_shape[:axis]).astype(np.int32), -1)

    x_out = np.empty(x_shape, dtype=x_dtype)
    x_out = x_out.reshape(new_shape)
    x_t = ggml.utils.from_numpy(x_out, ctx.ggml_context)

    axis_c = ctypes.c_int(axis)

    new_tensor = ctx.tensors_dict[node.output[0]] = ggml.ggml_map_custom2_inplace(
        ctx.ggml_context,
        x_t,
        x,
        custom_flatten,
        1,
        ctypes.pointer(axis_c),
    )

    ctx.refs.append(axis_c)

    return new_tensor


@ggml.ggml_custom1_op_t
def custom_floor(
    tensor_out: ggml.ggml_tensor_p,
    tensor_in_1: ggml.ggml_tensor_p,
    ith: int,
    nth: int,
    userdata: Optional[ctypes.c_void_p],
):
    x = ggml.utils.to_numpy(tensor_in_1)
    y = np.floor(x)

    set_tensor_out(tensor_out, y)


@ggml_operator("Floor")
def ggml_operator_floor(ctx: "GgmlOnnxExecutionContext", node: NodeProto):
    node_inputs = [ctx.tensors_dict[inp] for inp in node.input]

    if len(node_inputs) != 1:
        raise ValueError(
            f'Error for node "{node.name}": Operation "Floor" requires exactly one input. Actual number of inputs: {len(node_inputs)}'
        )

    x = node_inputs[0]

    new_tensor = ctx.tensors_dict[node.output[0]] = ggml.ggml_map_custom1_inplace(
        ctx.ggml_context,
        x,
        custom_floor,
        1,
        None,
    )

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
def ggml_operator_gather(ctx: "GgmlOnnxExecutionContext", node: NodeProto):
    node_inputs = [ctx.tensors_dict[inp] for inp in node.input]

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
    x_t = ggml.utils.from_numpy(x, ctx.ggml_context)

    new_tensor = ctx.tensors_dict[node.output[0]] = ggml.ggml_map_custom3_inplace(
        ctx.ggml_context,
        x_t,
        node_inputs[0],
        node_inputs[1],
        custom_gather,
        1,
        ctypes.pointer(axis_c),
    )

    ctx.refs.append(axis_c)

    return new_tensor


@ggml_operator("Gemm")
def ggml_operator_gemm(ctx: "GgmlOnnxExecutionContext", node: NodeProto):
    node_inputs = [ctx.tensors_dict[inp] for inp in node.input]

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

    b_shape = get_tensor_shape(b)
    a_shape = get_tensor_shape(a)

    # TODO: broadcast? Current broadcasting method fails during tests

    a_dtype = get_tensor_dtype(a)
    b_dtype = get_tensor_dtype(b)

    a_transposed = a
    b_transposed = b

    if transA:
        a_permute = ggml.ggml_transpose(
            ctx.ggml_context,
            a,
        )
        a_shape = ggml.utils.get_shape(a_permute)
        a_transposed = ggml.ggml_cpy(
            ctx.ggml_context,
            a_permute,
            ggml.ggml_new_tensor(
                ctx.ggml_context,
                map_to_ggml_type(a_dtype).value,
                len(a_shape),
                (ctypes.c_int64 * len(a_shape))(*a_shape),
            ),
        )

    if not transB:
        b_permute = ggml.ggml_transpose(
            ctx.ggml_context,
            b,
        )
        b_shape = ggml.utils.get_shape(b_permute)
        b_transposed = ggml.ggml_cpy(
            ctx.ggml_context,
            b_permute,
            ggml.ggml_new_tensor(
                ctx.ggml_context,
                map_to_ggml_type(b_dtype).value,
                len(b_shape),
                (ctypes.c_int64 * len(b_shape))(*b_shape),
            ),
        )

    # Y = alpha * np.dot(A, B) + beta * C
    # ref: https://github.com/onnx/onnx/blob/main/onnx/backend/test/case/node/gemm.py

    mul_mat_result = ggml.ggml_mul_mat(
        ctx.ggml_context,
        b_transposed,
        a_transposed,
    )

    alpha_t = ggml.utils.from_numpy(
        np.full(
            get_tensor_shape(mul_mat_result),
            alpha,
            dtype=get_tensor_dtype(mul_mat_result),
        ),
        ctx.ggml_context,
    )

    mul_mat_result = ggml.ggml_mul_inplace(ctx.ggml_context, mul_mat_result, alpha_t)

    if c is None:
        c = ggml.utils.from_numpy(
            np.full(
                get_tensor_shape(mul_mat_result),
                0,
                dtype=get_tensor_dtype(mul_mat_result),
            ),
            ctx.ggml_context,
        )

    c, mul_mat_result = broadcast_shapes(ctx.ggml_context, c, mul_mat_result)

    beta_t = ggml.utils.from_numpy(
        np.full(
            get_tensor_shape(mul_mat_result),
            beta,
            dtype=get_tensor_dtype(mul_mat_result),
        ),
        ctx.ggml_context,
    )

    mul_mat_result = ggml.ggml_add_inplace(
        ctx.ggml_context,
        mul_mat_result,
        ggml.ggml_mul_inplace(ctx.ggml_context, c, beta_t),
    )

    ctx.tensors_dict[node.output[0]] = mul_mat_result
    return mul_mat_result


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
def ggml_operator_greater(ctx: "GgmlOnnxExecutionContext", node: NodeProto):
    node_inputs = [ctx.tensors_dict[inp] for inp in node.input]

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
    x_t = ggml.utils.from_numpy(x, ctx.ggml_context)

    new_tensor = ctx.tensors_dict[name] = ggml.ggml_map_custom3_inplace(
        ctx.ggml_context,
        x_t,
        node_inputs[0],
        node_inputs[1],
        custom_greater,
        1,
        None,
    )

    ggml.ggml_set_name(new_tensor, (name + f"<bool>").encode())

    return new_tensor


class HardSigmoidUserData(ctypes.Structure):
    _fields_ = [
        ("alpha", ctypes.c_float),
        ("beta", ctypes.c_float),
    ]


@ggml.ggml_custom1_op_t
def custom_hard_sigmoid(
    tensor_out: ggml.ggml_tensor_p,
    tensor_in_1: ggml.ggml_tensor_p,
    ith: int,
    nth: int,
    userdata: Optional[ctypes.c_void_p],
):
    userdata_data_ptr = ctypes.cast(userdata, ctypes.POINTER(HardSigmoidUserData))
    userdata_data = userdata_data_ptr.contents
    x = ggml.utils.to_numpy(tensor_in_1)
    alpha = userdata_data.alpha
    beta = userdata_data.beta

    y = np.clip((x * alpha) + beta, 0, 1)

    set_tensor_out(tensor_out, y)


@ggml_operator("HardSigmoid")
def ggml_operator_size(ctx: "GgmlOnnxExecutionContext", node: NodeProto):
    node_inputs = [ctx.tensors_dict[inp] for inp in node.input]

    if len(node_inputs) != 1:
        raise ValueError(
            f'Error for node "{node.name}": Operation "Sigmoid" requires exactly one input. Actual number of inputs: {len(node_inputs)}'
        )

    x = node_inputs[0]
    alpha = next((attr.f for attr in node.attribute if attr.name == "alpha"), 0.2)
    beta = next((attr.f for attr in node.attribute if attr.name == "beta"), 0.5)

    hsig_userdata = HardSigmoidUserData(alpha, beta)
    userdata_p = ctypes.cast(ctypes.pointer(hsig_userdata), ctypes.c_void_p)

    new_tensor = ctx.tensors_dict[node.output[0]] = ggml.ggml_map_custom1_inplace(
        ctx.ggml_context,
        x,
        custom_hard_sigmoid,
        1,
        userdata_p,
    )

    ctx.refs.append(hsig_userdata)

    return new_tensor


@ggml.ggml_custom1_op_t
def custom_hardmax(
    tensor_out: ggml.ggml_tensor_p,
    tensor_in_1: ggml.ggml_tensor_p,
    ith: int,
    nth: int,
    userdata: Optional[ctypes.c_void_p],
):
    axis = ctypes.cast(userdata, ctypes.POINTER(ctypes.c_int)).contents.value
    x = ggml.utils.to_numpy(tensor_in_1)

    max_indices = np.argmax(x, axis=axis, keepdims=True)
    y = np.zeros_like(x)
    np.put_along_axis(y, max_indices, 1, axis=axis)

    set_tensor_out(tensor_out, y)


@ggml_operator("Hardmax")
def ggml_operator_hardmax(ctx: "GgmlOnnxExecutionContext", node: NodeProto):
    node_inputs = [ctx.tensors_dict[inp] for inp in node.input]

    if len(node_inputs) != 1:
        raise ValueError(
            f'Error for node "{node.name}": Operation "Hardmax" requires exactly one input. Actual number of inputs: {len(node_inputs)}'
        )

    x = node_inputs[0]
    axis = next((attr.i for attr in node.attribute if attr.name == "axis"), -1)
    axis_c = ctypes.c_int(axis)

    new_tensor = ctx.tensors_dict[node.output[0]] = ggml.ggml_map_custom1_inplace(
        ctx.ggml_context,
        x,
        custom_hardmax,
        1,
        ctypes.pointer(axis_c),
    )

    ctx.refs.append(axis_c)

    return new_tensor


@ggml_operator("Identity")
def ggml_operator_floor(ctx: "GgmlOnnxExecutionContext", node: NodeProto):
    node_inputs = [ctx.tensors_dict[inp] for inp in node.input]

    if len(node_inputs) != 1:
        raise ValueError(
            f'Error for node "{node.name}": Operation "Identity" requires exactly one input. Actual number of inputs: {len(node_inputs)}'
        )

    x = node_inputs[0]
    output_name = node.output[0]
    y = ggml.ggml_dup(
        ctx.ggml_context, x
    )  # NOTE: This will freeze the tensor in time, may not be expected.
    ggml.ggml_set_name(y, output_name.encode())

    ctx.tensors_dict[output_name] = y

    return y


@ggml.ggml_custom3_op_t
def custom_instancenorm(
    tensor_out: ggml.ggml_tensor_p,
    tensor_in_1: ggml.ggml_tensor_p,
    tensor_in_2: ggml.ggml_tensor_p,
    tensor_in_3: ggml.ggml_tensor_p,
    ith: int,
    nth: int,
    userdata: Optional[ctypes.c_void_p],
):
    x = ggml.utils.to_numpy(tensor_in_1)
    scale = ggml.utils.to_numpy(tensor_in_2)
    B = ggml.utils.to_numpy(tensor_in_3)
    epsilon = ctypes.cast(userdata, ctypes.POINTER(ctypes.c_double)).contents.value

    mean = np.mean(x, axis=(2, 3), keepdims=True)
    variance = np.var(x, axis=(2, 3), keepdims=True)
    normalized = (x - mean) / np.sqrt(variance + epsilon)
    y = scale.reshape(1, -1, 1, 1) * normalized + B.reshape(1, -1, 1, 1)

    set_tensor_out(tensor_out, y)


@ggml_operator("InstanceNormalization")
def ggml_operator_instancenorm(ctx: "GgmlOnnxExecutionContext", node: NodeProto):
    node_inputs = [ctx.tensors_dict[inp] for inp in node.input]

    if len(node_inputs) != 3:
        raise ValueError(
            f'Error for node "{node.name}": Operation "InstanceNormalization" requires exactly three inputs. Actual number of inputs: {len(node_inputs)}'
        )
    input_tensor, scale, B = node_inputs
    epsilon = next((attr.f for attr in node.attribute if attr.name == "epsilon"), 1e-05)
    epsilon_c = ctypes.c_double(epsilon)
    new_tensor = ctx.tensors_dict[node.output[0]] = ggml.ggml_map_custom3_inplace(
        ctx.ggml_context,
        input_tensor,
        scale,
        B,
        custom_instancenorm,
        1,
        ctypes.pointer(epsilon_c),
    )

    ctx.refs.append(epsilon_c)
    return new_tensor


class LRNUserData(ctypes.Structure):
    _fields_ = [
        ("alpha", ctypes.c_double),
        ("beta", ctypes.c_double),
        ("bias", ctypes.c_double),
        ("size", ctypes.c_int),
    ]


@ggml.ggml_custom1_op_t
def custom_leaky_lrn(
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

    x = ggml.utils.to_numpy(tensor_in_1)

    square_sum = np.zeros(x.shape).astype(x.dtype)
    for n, c, h, w in np.ndindex(x.shape):
        square_sum[n, c, h, w] = sum(
            x[
                n,
                max(0, c - int(math.floor((size - 1) / 2))) : min(
                    5, c + int(math.ceil((size - 1) / 2)) + 1
                ),
                h,
                w,
            ]
            ** 2
        )
    y = x / ((bias + (alpha / size) * square_sum) ** beta)

    set_tensor_out(tensor_out, y)


@ggml_operator("LRN")
def ggml_operator_leaky_relu(ctx: "GgmlOnnxExecutionContext", node: NodeProto):
    node_inputs = [ctx.tensors_dict[inp] for inp in node.input]

    if len(node_inputs) != 1:
        raise ValueError(
            f'Error for node "{node.name}": Operation "LRN" requires exactly one input. Actual number of inputs: {len(node_inputs)}'
        )

    x = node_inputs[0]
    alpha = next((attr.f for attr in node.attribute if attr.name == "alpha"), 0.0001)
    beta = next((attr.f for attr in node.attribute if attr.name == "beta"), 0.75)
    bias = next((attr.f for attr in node.attribute if attr.name == "bias"), 1.0)
    size = next((attr.i for attr in node.attribute if attr.name == "size"), None)

    if size is None:
        raise ValueError(
            f'Error for node "{node.name}": Operation "LRN" requires "size" attibute.'
        )

    lrn_userdata = LRNUserData(alpha, beta, bias, size)
    userdata_p = ctypes.cast(ctypes.pointer(lrn_userdata), ctypes.c_void_p)

    new_tensor = ctx.tensors_dict[node.output[0]] = ggml.ggml_map_custom1_inplace(
        ctx.ggml_context,
        x,
        custom_leaky_lrn,
        1,
        userdata_p,
    )

    ctx.refs.append(lrn_userdata)

    return new_tensor


@ggml.ggml_custom1_op_t
def custom_leaky_relu(
    tensor_out: ggml.ggml_tensor_p,
    tensor_in_1: ggml.ggml_tensor_p,
    ith: int,
    nth: int,
    userdata: Optional[ctypes.c_void_p],
):
    alpha = ctypes.cast(userdata, ctypes.POINTER(ctypes.c_double)).contents.value
    x = ggml.utils.to_numpy(tensor_in_1)
    y = np.clip(x, 0, np.inf) + np.clip(x, -np.inf, 0) * alpha

    set_tensor_out(tensor_out, y)


@ggml_operator("LeakyRelu")
def ggml_operator_leaky_relu(ctx: "GgmlOnnxExecutionContext", node: NodeProto):
    node_inputs = [ctx.tensors_dict[inp] for inp in node.input]

    if len(node_inputs) != 1:
        raise ValueError(
            f'Error for node "{node.name}": Operation "LeakyRelu" requires exactly one input. Actual number of inputs: {len(node_inputs)}'
        )

    x = node_inputs[0]
    alpha = next((attr.f for attr in node.attribute if attr.name == "alpha"), 0.01)

    axis_c = ctypes.c_double(alpha)

    new_tensor = ctx.tensors_dict[node.output[0]] = ggml.ggml_map_custom1_inplace(
        ctx.ggml_context,
        x,
        custom_leaky_relu,
        1,
        ctypes.pointer(axis_c),
    )

    ctx.refs.append(axis_c)

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
def ggml_operator_greater_or_equal(ctx: "GgmlOnnxExecutionContext", node: NodeProto):
    node_inputs = [ctx.tensors_dict[inp] for inp in node.input]

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
    x_t = ggml.utils.from_numpy(x, ctx.ggml_context)

    new_tensor = ctx.tensors_dict[name] = ggml.ggml_map_custom3_inplace(
        ctx.ggml_context,
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
def ggml_operator_less(ctx: "GgmlOnnxExecutionContext", node: NodeProto):
    node_inputs = [ctx.tensors_dict[inp] for inp in node.input]

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
    x_t = ggml.utils.from_numpy(x, ctx.ggml_context)

    new_tensor = ctx.tensors_dict[name] = ggml.ggml_map_custom3_inplace(
        ctx.ggml_context,
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
def ggml_operator_less_or_equal(ctx: "GgmlOnnxExecutionContext", node: NodeProto):
    node_inputs = [ctx.tensors_dict[inp] for inp in node.input]

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
    x_t = ggml.utils.from_numpy(x, ctx.ggml_context)

    new_tensor = ctx.tensors_dict[name] = ggml.ggml_map_custom3_inplace(
        ctx.ggml_context,
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
def ggml_operator_log(ctx: "GgmlOnnxExecutionContext", node: NodeProto):
    node_inputs = [ctx.tensors_dict[inp] for inp in node.input]

    if len(node_inputs) != 1:
        raise ValueError(
            f'Error for node "{node.name}": Operation "Log" requires exactly one input. Actual number of inputs: {len(node_inputs)}'
        )

    output_name = node.output[0]
    a = node_inputs[0]

    log_result = ggml.ggml_log(
        ctx.ggml_context,
        a,
    )
    ctx.tensors_dict[output_name] = log_result
    return log_result


@ggml_operator("LogSoftmax")
def ggml_operator_log_soft_max(ctx: "GgmlOnnxExecutionContext", node: NodeProto):
    node_inputs = [ctx.tensors_dict[inp] for inp in node.input]

    if len(node_inputs) != 1:
        raise ValueError(
            f'Error for node "{node.name}": Operation "LogSoftmax" requires exactly one input. Actual number of inputs: {len(node_inputs)}'
        )

    output_name = node.output[0]
    a = node_inputs[0]
    soft_max_result = ggml.ggml_soft_max(ctx.ggml_context, a)
    log_result = ggml.ggml_log(
        ctx.ggml_context,
        soft_max_result,
    )
    ctx.tensors_dict[output_name] = log_result
    return log_result


@ggml_operator("MatMul")
def ggml_operator_mat_mul(ctx: "GgmlOnnxExecutionContext", node: NodeProto):
    node_inputs = [ctx.tensors_dict[inp] for inp in node.input]

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
        a, b = broadcast_shapes(ctx.ggml_context, a, b)

    b_dtype = get_tensor_dtype(b)

    b_permute = ggml.ggml_transpose(
        ctx.ggml_context,
        b,
    )

    b_shape = ggml.utils.get_shape(b_permute)

    b_transposed = ggml.ggml_cpy(
        ctx.ggml_context,
        b_permute,
        ggml.ggml_new_tensor(
            ctx.ggml_context,
            map_to_ggml_type(b_dtype).value,
            len(b_shape),
            (ctypes.c_int64 * len(b_shape))(*b_shape),
        ),
    )

    mul_mat_result = ggml.ggml_mul_mat(
        ctx.ggml_context,
        b_transposed,
        a,
    )

    ctx.tensors_dict[output_name] = mul_mat_result
    return mul_mat_result


@ggml_operator("Max")
def ggml_operator_max(ctx: "GgmlOnnxExecutionContext", node: NodeProto):
    node_inputs = [ctx.tensors_dict[inp] for inp in node.input]

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
        ctx.ggml_context,
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

    new_tensor = ctx.tensors_dict[node.output[0]] = ggml.ggml_map_custom1_inplace(
        ctx.ggml_context,
        x_t,
        custom_max,
        1,
        None,
    )

    ctx.refs.append(custom_max)

    return new_tensor


@ggml_operator("Mean")
def ggml_operator_mean(ctx: "GgmlOnnxExecutionContext", node: NodeProto):
    node_inputs = [ctx.tensors_dict[inp] for inp in node.input]

    if len(node_inputs) < 1:
        raise ValueError(
            f'Error for node "{node.name}": Operation "Mean" requires at least one input. Actual number of inputs: {len(node_inputs)}'
        )

    output_name = node.output[0]
    sums = node_inputs[0]

    for tensor in node_inputs[1:]:
        sums = ggml.ggml_add(ctx.ggml_context, sums, tensor)

    coef_np = np.full(get_tensor_shape(sums), len(node_inputs), dtype=np.float32)
    coef_t = ggml.utils.from_numpy(coef_np, ctx.ggml_context)

    mean = ggml.ggml_div(
        ctx.ggml_context,
        sums,
        coef_t,
    )

    ctx.tensors_dict[output_name] = mean
    return mean


@ggml_operator("Min")
def ggml_operator_min(ctx: "GgmlOnnxExecutionContext", node: NodeProto):
    node_inputs = [ctx.tensors_dict[inp] for inp in node.input]

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
        ctx.ggml_context,
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

    new_tensor = ctx.tensors_dict[node.output[0]] = ggml.ggml_map_custom1_inplace(
        ctx.ggml_context,
        x_t,
        custom_min,
        1,
        None,
    )

    ctx.refs.append(custom_min)

    return new_tensor


@ggml_operator("Mul")
def ggml_operator_mul(ctx: "GgmlOnnxExecutionContext", node: NodeProto):
    node_inputs = [ctx.tensors_dict[inp] for inp in node.input]

    if len(node_inputs) != 2:
        raise ValueError(
            f'Error for node "{node.name}": Operation "Mul" requires exactly two inputs. Actual number of inputs: {len(node_inputs)}'
        )

    output_name = node.output[0]
    a = node_inputs[0]
    b = node_inputs[1]

    a, b = broadcast_shapes(ctx.ggml_context, a, b)

    mul_result = ggml.ggml_mul(
        ctx.ggml_context,
        a,
        b,
    )

    ctx.tensors_dict[output_name] = mul_result
    return mul_result


@ggml_operator("Neg")
def ggml_operator_neg(ctx: "GgmlOnnxExecutionContext", node: NodeProto):
    node_inputs = [ctx.tensors_dict[inp] for inp in node.input]

    if len(node_inputs) != 1:
        raise ValueError(
            f'Error for node "{node.name}": Operation "Neg" requires exactly one input. Actual number of inputs: {len(node_inputs)}'
        )

    x = node_inputs[0]
    output_name = node.output[0]

    x_neg = ggml.ggml_neg(
        ctx.ggml_context,
        x,
    )
    ctx.tensors_dict[output_name] = x_neg
    return x_neg


@ggml.ggml_custom1_op_t
def custom_not(
    tensor_out: ggml.ggml_tensor_p,
    tensor_in_1: ggml.ggml_tensor_p,
    ith: int,
    nth: int,
    userdata: Optional[ctypes.c_void_p],
):
    a = ggml.utils.to_numpy(tensor_in_1)
    x = np.logical_not(a)

    set_tensor_out(tensor_out, x)


@ggml_operator("Not")
def ggml_operator_not(ctx: "GgmlOnnxExecutionContext", node: NodeProto):
    node_inputs = [ctx.tensors_dict[inp] for inp in node.input]

    if len(node_inputs) != 1:
        raise ValueError(
            f'Error for node "{node.name}": Operation "Not" requires exactly one input. Actual number of inputs: {len(node_inputs)}'
        )
    name = node.output[0]

    new_tensor = ctx.tensors_dict[name] = ggml.ggml_map_custom1_inplace(
        ctx.ggml_context,
        node_inputs[0],
        custom_not,
        1,
        None,
    )

    ggml.ggml_set_name(new_tensor, (name + f"<bool>").encode())

    return new_tensor


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
def ggml_operator_or(ctx: "GgmlOnnxExecutionContext", node: NodeProto):
    node_inputs = [ctx.tensors_dict[inp] for inp in node.input]

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
    x_t = ggml.utils.from_numpy(x, ctx.ggml_context)

    new_tensor = ctx.tensors_dict[name] = ggml.ggml_map_custom3_inplace(
        ctx.ggml_context,
        x_t,
        node_inputs[0],
        node_inputs[1],
        custom_or,
        1,
        None,
    )

    ggml.ggml_set_name(new_tensor, (name + f"<bool>").encode())

    return new_tensor


@ggml_operator("Pad")
def ggml_operator_pad(ctx: "GgmlOnnxExecutionContext", node: NodeProto):
    # x, pads, value, axes
    if len(ctx.tensors_dict) < 2:
        raise ValueError(
            f'Error for node "{node.name}": Operation "Pad" requires at least two inputs. Actual number of inputs: {len(node_inputs)}'
        )
    input_rank = ctx.tensors_dict["x"].contents.n_dims
    mode = next(
        (attr.s for attr in node.attribute if attr.name == "mode"), b"constant"
    ).decode("utf-8")

    if "axes" not in ctx.tensors_dict:
        axes = list(range(input_rank))
    else:
        axes_eval = ctx.backend.eval_tensor(ctx.tensors_dict["axes"], ctx.ggml_context)
        axes = ggml.utils.to_numpy(axes_eval)
        axes = [axis if axis >= 0 else axis + input_rank for axis in axes]
    num_axes = len(axes)
    pad_width = []
    for _ in range(input_rank):
        pad_width += [[0, 0]]  # init to zero

    raw_pads = ggml.utils.to_numpy(
        ctx.backend.eval_tensor(ctx.tensors_dict["pads"], ctx.ggml_context)
    )

    # re-order to np.pad accepted order ((x1_begin, x1_end), (x2_begin, x2_end), ...)
    for i in range(num_axes):
        axis = axes[i]
        if axis < 0:
            axis = input_rank + axis
        pad_width[axis] = [raw_pads[i], raw_pads[i + num_axes]]

    expand_by = [sum(pad) for pad in pad_width]
    prev_shape = get_tensor_shape(ctx.tensors_dict["x"])
    output_shape = [sum(x) for x in zip(prev_shape, expand_by)]
    a_dtype = get_tensor_dtype(ctx.tensors_dict["x"])
    x = np.empty(output_shape, dtype=a_dtype)
    x_t = ggml.utils.from_numpy(x, ctx.ggml_context)

    constant_value = None
    if "value" in ctx.tensors_dict:
        constant_values = ggml.utils.to_numpy(
            ctx.backend.eval_tensor(ctx.tensors_dict["value"], ctx.ggml_context)
        )

    @ggml.ggml_custom2_op_t
    def custom_pad(
        tensor_out: ggml.ggml_tensor_p,
        tensor_in_1: ggml.ggml_tensor_p,
        tensor_in_2: ggml.ggml_tensor_p,
        ith: int,
        nth: int,
        userdata: Optional[ctypes.c_void_p],
    ):
        a = ggml.utils.to_numpy(tensor_in_2)
        if mode == "constant":
            x = np.pad(
                a,
                pad_width=pad_width,
                mode=mode,
                constant_values=constant_values,
            )

        else:
            x = np.pad(
                a,
                pad_width=pad_width,
                mode=mode,
            )
        set_tensor_out(tensor_out, x)

    new_tensor = ctx.tensors_dict[node.output[0]] = ggml.ggml_map_custom2_inplace(
        ctx.ggml_context,
        x_t,
        ctx.tensors_dict["x"],
        custom_pad,
        1,
        None,
    )
    ctx.refs.append(custom_pad)
    return new_tensor


@ggml.ggml_custom2_op_t
def custom_leaky_prelu(
    tensor_out: ggml.ggml_tensor_p,
    tensor_in_1: ggml.ggml_tensor_p,
    tensor_in_2: ggml.ggml_tensor_p,
    ith: int,
    nth: int,
    userdata: Optional[ctypes.c_void_p],
):
    x = ggml.utils.to_numpy(tensor_in_1)
    slope = ggml.utils.to_numpy(tensor_in_2)

    y = np.clip(x, 0, np.inf) + np.clip(x, -np.inf, 0) * slope

    set_tensor_out(tensor_out, y)


@ggml_operator("PRelu")
def ggml_operator_leaky_relu(ctx: "GgmlOnnxExecutionContext", node: NodeProto):
    node_inputs = [ctx.tensors_dict[inp] for inp in node.input]

    if len(node_inputs) != 2:
        raise ValueError(
            f'Error for node "{node.name}": Operation "PRelu" requires exactly two inputs. Actual number of inputs: {len(node_inputs)}'
        )
    x, slope = node_inputs

    new_tensor = ctx.tensors_dict[node.output[0]] = ggml.ggml_map_custom2_inplace(
        ctx.ggml_context,
        x,
        slope,
        custom_leaky_prelu,
        1,
        None,
    )

    return new_tensor


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


@ggml_operator("Pow")
def ggml_operator_pow(ctx: "GgmlOnnxExecutionContext", node: NodeProto):
    node_inputs = [ctx.tensors_dict[inp] for inp in node.input]

    if len(node_inputs) != 2:
        raise ValueError(
            f'Error for node "{node.name}": Operation "Pow" requires exactly two inputs. Actual number of inputs: {len(node_inputs)}'
        )

    x1 = node_inputs[0]
    x2 = node_inputs[1]

    new_tensor = ctx.tensors_dict[node.output[0]] = ggml.ggml_map_custom2_inplace(
        ctx.ggml_context,
        x1,
        x2,
        custom_pow,
        1,
        None,
    )

    return new_tensor


@ggml.ggml_custom1_op_t
def custom_reciprocal(
    tensor_out: ggml.ggml_tensor_p,
    tensor_in_1: ggml.ggml_tensor_p,
    ith: int,
    nth: int,
    userdata: Optional[ctypes.c_void_p],
):
    x = ggml.utils.to_numpy(tensor_in_1)
    y = np.reciprocal(x)

    set_tensor_out(tensor_out, y)


@ggml_operator("Reciprocal")
def ggml_operator_reciprocal(ctx: "GgmlOnnxExecutionContext", node: NodeProto):
    node_inputs = [ctx.tensors_dict[inp] for inp in node.input]

    if len(node_inputs) != 1:
        raise ValueError(
            f'Error for node "{node.name}": Operation "Reciprocal" requires exactly one input. Actual number of inputs: {len(node_inputs)}'
        )

    x = node_inputs[0]
    new_tensor = ctx.tensors_dict[node.output[0]] = ggml.ggml_map_custom1_inplace(
        ctx.ggml_context,
        x,
        custom_reciprocal,
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
def ggml_operator_range(ctx: "GgmlOnnxExecutionContext", node: NodeProto):
    node_inputs = [ctx.tensors_dict[inp] for inp in node.input]

    if len(node_inputs) != 3:
        raise ValueError(
            f'Error for node "{node.name}": Operation "Range" requires exactly three inputs. Actual number of inputs: {len(node_inputs)}'
        )

    tensors = [ggml.utils.to_numpy(node_input) for node_input in node_inputs]

    start, stop, step = tensors
    output_shape = (int(np.ceil((stop - start) / step)),)

    x = np.empty(output_shape, dtype=step.dtype)
    x_t = ggml.utils.from_numpy(x, ctx.ggml_context)

    input_tensors = ggml.utils.from_numpy(np.array(tensors), ctx.ggml_context)

    new_tensor = ctx.tensors_dict[node.output[0]] = ggml.ggml_map_custom2_inplace(
        ctx.ggml_context,
        x_t,
        input_tensors,
        custom_range,
        1,
        None,
    )

    return new_tensor


class ReduceOpsUserData(ctypes.Structure):
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
def custom_reduce_l1(
    tensor_out: ggml.ggml_tensor_p,
    tensor_in_1: ggml.ggml_tensor_p,
    tensor_in_2: ggml.ggml_tensor_p,
    ith: int,
    nth: int,
    userdata: Optional[ctypes.c_void_p],
):
    userdata_data_ptr = ctypes.cast(userdata, ctypes.POINTER(ReduceOpsUserData))
    userdata_data = userdata_data_ptr.contents

    tensor = ggml.utils.to_numpy(tensor_in_2)
    axes = [userdata_data.axes[i] for i in range(userdata_data.axes_length)]
    keepdims = userdata_data.keepdims

    axes = tuple(axes) if len(axes) else None

    shape = tensor.shape
    data = np.reshape(np.arange(1, np.prod(shape) + 1, dtype=np.float32), shape)
    rl1_result = np.sum(a=np.abs(tensor), axis=axes, keepdims=keepdims)

    set_tensor_out(tensor_out, rl1_result)


@ggml_operator("ReduceL1")
def ggml_operator_reduce_l1(ctx: "GgmlOnnxExecutionContext", node: NodeProto):
    node_inputs = [ctx.tensors_dict[inp] for inp in node.input]

    if len(node_inputs) > 2 or len(node_inputs) < 1:
        raise ValueError(
            f'Error for node "{node.name}": Operation "ReduceL1" requires at least one input. Actual number of inputs: {len(node_inputs)}'
        )

    input_tensor = node_inputs[0]

    noop_with_empty_axes = next(
        (attr.i for attr in node.attribute if attr.name == "noop_with_empty_axes"), None
    )

    if noop_with_empty_axes == 1:
        ctx.tensors_dict[node.output[0]] = input_tensor
        return input_tensor

    tensor_shape = get_tensor_shape(input_tensor)
    tensor_dtype = get_tensor_dtype(input_tensor)

    axes = next((attr.ints for attr in node.attribute if attr.name == "axes"), None)
    if not axes:
        if len(node_inputs) > 1:
            axes_eval = ctx.backend.eval_tensor(node_inputs[1], ctx.ggml_context)
            axes = ggml.utils.to_numpy(axes_eval)
        else:
            axes = []

    keepdims = next((attr.i for attr in node.attribute if attr.name == "keepdims"), 1)

    rmean_userdata = ReduceOpsUserData(list(axes), keepdims)
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
    x_t = ggml.utils.from_numpy(x, ctx.ggml_context)

    new_tensor = ctx.tensors_dict[node.output[0]] = ggml.ggml_map_custom2_inplace(
        ctx.ggml_context,
        x_t,
        input_tensor,
        custom_reduce_l1,
        1,
        userdata_p,
    )

    ctx.refs.append(rmean_userdata)

    return new_tensor


@ggml.ggml_custom2_op_t
def custom_reduce_l2(
    tensor_out: ggml.ggml_tensor_p,
    tensor_in_1: ggml.ggml_tensor_p,
    tensor_in_2: ggml.ggml_tensor_p,
    ith: int,
    nth: int,
    userdata: Optional[ctypes.c_void_p],
):
    userdata_data_ptr = ctypes.cast(userdata, ctypes.POINTER(ReduceOpsUserData))
    userdata_data = userdata_data_ptr.contents

    tensor = ggml.utils.to_numpy(tensor_in_2)
    axes = [userdata_data.axes[i] for i in range(userdata_data.axes_length)]
    keepdims = userdata_data.keepdims

    axes = tuple(axes) if len(axes) else None

    rl2_result = np.sqrt(np.sum(a=np.square(tensor), axis=axes, keepdims=keepdims))

    set_tensor_out(tensor_out, rl2_result)


@ggml_operator("ReduceL2")
def ggml_operator_reduce_l2(ctx: "GgmlOnnxExecutionContext", node: NodeProto):
    node_inputs = [ctx.tensors_dict[inp] for inp in node.input]

    if len(node_inputs) > 2 or len(node_inputs) < 1:
        raise ValueError(
            f'Error for node "{node.name}": Operation "ReduceL2" requires at least one input. Actual number of inputs: {len(node_inputs)}'
        )

    input_tensor = node_inputs[0]

    noop_with_empty_axes = next(
        (attr.i for attr in node.attribute if attr.name == "noop_with_empty_axes"), None
    )

    if noop_with_empty_axes == 1:
        ctx.tensors_dict[node.output[0]] = input_tensor
        return input_tensor

    tensor_shape = get_tensor_shape(input_tensor)
    tensor_dtype = get_tensor_dtype(input_tensor)

    axes = next((attr.ints for attr in node.attribute if attr.name == "axes"), None)
    if not axes:
        if len(node_inputs) > 1:
            axes_eval = ctx.backend.eval_tensor(node_inputs[1], ctx.ggml_context)
            axes = ggml.utils.to_numpy(axes_eval)
        else:
            axes = []

    keepdims = next((attr.i for attr in node.attribute if attr.name == "keepdims"), 1)

    rmean_userdata = ReduceOpsUserData(list(axes), keepdims)
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
    x_t = ggml.utils.from_numpy(x, ctx.ggml_context)

    new_tensor = ctx.tensors_dict[node.output[0]] = ggml.ggml_map_custom2_inplace(
        ctx.ggml_context,
        x_t,
        input_tensor,
        custom_reduce_l2,
        1,
        userdata_p,
    )

    ctx.refs.append(rmean_userdata)

    return new_tensor


@ggml.ggml_custom2_op_t
def custom_reduce_log_sum(
    tensor_out: ggml.ggml_tensor_p,
    tensor_in_1: ggml.ggml_tensor_p,
    tensor_in_2: ggml.ggml_tensor_p,
    ith: int,
    nth: int,
    userdata: Optional[ctypes.c_void_p],
):
    userdata_data_ptr = ctypes.cast(userdata, ctypes.POINTER(ReduceOpsUserData))
    userdata_data = userdata_data_ptr.contents

    tensor = ggml.utils.to_numpy(tensor_in_2)
    axes = [userdata_data.axes[i] for i in range(userdata_data.axes_length)]
    keepdims = userdata_data.keepdims

    axes = tuple(axes) if len(axes) else None
    rlogsum_result = np.log(np.sum(tensor, axis=axes, keepdims=keepdims))

    set_tensor_out(tensor_out, rlogsum_result)


@ggml_operator("ReduceLogSum")
def ggml_operator_reduce_log_sum(ctx: "GgmlOnnxExecutionContext", node: NodeProto):
    node_inputs = [ctx.tensors_dict[inp] for inp in node.input]

    if len(node_inputs) > 2 or len(node_inputs) < 1:
        raise ValueError(
            f'Error for node "{node.name}": Operation "ReduceLogSum" requires at least one input. Actual number of inputs: {len(node_inputs)}'
        )

    input_tensor = node_inputs[0]

    noop_with_empty_axes = next(
        (attr.i for attr in node.attribute if attr.name == "noop_with_empty_axes"), None
    )

    if noop_with_empty_axes == 1:
        ctx.tensors_dict[node.output[0]] = input_tensor
        return input_tensor

    tensor_shape = get_tensor_shape(input_tensor)
    tensor_dtype = get_tensor_dtype(input_tensor)

    axes = next((attr.ints for attr in node.attribute if attr.name == "axes"), None)
    if not axes:
        if len(node_inputs) > 1:
            axes_eval = ctx.backend.eval_tensor(node_inputs[1], ctx.ggml_context)
            axes = ggml.utils.to_numpy(axes_eval)
        else:
            axes = []

    keepdims = next((attr.i for attr in node.attribute if attr.name == "keepdims"), 1)

    rmean_userdata = ReduceOpsUserData(list(axes), keepdims)
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
    x_t = ggml.utils.from_numpy(x, ctx.ggml_context)

    new_tensor = ctx.tensors_dict[node.output[0]] = ggml.ggml_map_custom2_inplace(
        ctx.ggml_context,
        x_t,
        input_tensor,
        custom_reduce_log_sum,
        1,
        userdata_p,
    )

    ctx.refs.append(rmean_userdata)

    return new_tensor


@ggml.ggml_custom2_op_t
def custom_reduce_log_sum_exp(
    tensor_out: ggml.ggml_tensor_p,
    tensor_in_1: ggml.ggml_tensor_p,
    tensor_in_2: ggml.ggml_tensor_p,
    ith: int,
    nth: int,
    userdata: Optional[ctypes.c_void_p],
):
    userdata_data_ptr = ctypes.cast(userdata, ctypes.POINTER(ReduceOpsUserData))
    userdata_data = userdata_data_ptr.contents

    tensor = ggml.utils.to_numpy(tensor_in_2)
    axes = [userdata_data.axes[i] for i in range(userdata_data.axes_length)]
    keepdims = userdata_data.keepdims

    axes = tuple(axes) if len(axes) else None
    rlogsum_result = np.log(np.sum(np.exp(tensor), axis=axes, keepdims=keepdims))

    set_tensor_out(tensor_out, rlogsum_result)


@ggml_operator("ReduceLogSumExp")
def ggml_operator_reduce_log_sum_exp(ctx: "GgmlOnnxExecutionContext", node: NodeProto):
    raise NotImplementedError(
        f'Error for node "{node.name}": Operation "ReduceLogSumExp" is not implemented.'
    )
    node_inputs = [ctx.tensors_dict[inp] for inp in node.input]

    if len(node_inputs) > 2 or len(node_inputs) < 1:
        raise ValueError(
            f'Error for node "{node.name}": Operation "ReduceLogSumExp" requires at least one input. Actual number of inputs: {len(node_inputs)}'
        )

    input_tensor = node_inputs[0]

    noop_with_empty_axes = next(
        (attr.i for attr in node.attribute if attr.name == "noop_with_empty_axes"), None
    )

    if noop_with_empty_axes == 1:
        ctx.tensors_dict[node.output[0]] = input_tensor
        return input_tensor

    tensor_shape = get_tensor_shape(input_tensor)
    tensor_dtype = get_tensor_dtype(input_tensor)

    axes = next((attr.ints for attr in node.attribute if attr.name == "axes"), None)
    if not axes:
        if len(node_inputs) > 1:
            axes_eval = ctx.backend.eval_tensor(node_inputs[1], ctx.ggml_context)
            axes = ggml.utils.to_numpy(axes_eval)
        else:
            axes = []

    keepdims = next((attr.i for attr in node.attribute if attr.name == "keepdims"), 1)

    rmean_userdata = ReduceOpsUserData(list(axes), keepdims)
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
    x_t = ggml.utils.from_numpy(x, ctx.ggml_context)

    new_tensor = ctx.tensors_dict[node.output[0]] = ggml.ggml_map_custom2_inplace(
        ctx.ggml_context,
        x_t,
        input_tensor,
        custom_reduce_log_sum_exp,
        1,
        userdata_p,
    )

    ctx.refs.append(rmean_userdata)

    return new_tensor


@ggml.ggml_custom2_op_t
def custom_reduce_max(
    tensor_out: ggml.ggml_tensor_p,
    tensor_in_1: ggml.ggml_tensor_p,
    tensor_in_2: ggml.ggml_tensor_p,
    ith: int,
    nth: int,
    userdata: Optional[ctypes.c_void_p],
):
    userdata_data_ptr = ctypes.cast(userdata, ctypes.POINTER(ReduceOpsUserData))
    userdata_data = userdata_data_ptr.contents

    tensor = ggml.utils.to_numpy(tensor_in_2)
    axes = [userdata_data.axes[i] for i in range(userdata_data.axes_length)]
    keepdims = userdata_data.keepdims

    axes = tuple(axes) if len(axes) else None
    rmean_result = np.max(tensor, axis=axes, keepdims=keepdims)

    set_tensor_out(tensor_out, rmean_result)


@ggml_operator("ReduceMax")
def ggml_operator_reduce_max(ctx: "GgmlOnnxExecutionContext", node: NodeProto):
    node_inputs = [ctx.tensors_dict[inp] for inp in node.input]

    if len(node_inputs) > 2 or len(node_inputs) < 1:
        raise ValueError(
            f'Error for node "{node.name}": Operation "ReduceMax" requires at least one input. Actual number of inputs: {len(node_inputs)}'
        )

    input_tensor = node_inputs[0]

    noop_with_empty_axes = next(
        (attr.i for attr in node.attribute if attr.name == "noop_with_empty_axes"), None
    )

    if noop_with_empty_axes == 1:
        ctx.tensors_dict[node.output[0]] = input_tensor
        return input_tensor

    tensor_shape = get_tensor_shape(input_tensor)
    tensor_dtype = get_tensor_dtype(input_tensor)

    axes = next((attr.ints for attr in node.attribute if attr.name == "axes"), None)
    if not axes:
        if len(node_inputs) > 1:
            axes_eval = ctx.backend.eval_tensor(node_inputs[1], ctx.ggml_context)
            axes = ggml.utils.to_numpy(axes_eval)
        else:
            axes = []

    keepdims = next((attr.i for attr in node.attribute if attr.name == "keepdims"), 1)

    rmean_userdata = ReduceOpsUserData(list(axes), keepdims)
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
    x_t = ggml.utils.from_numpy(x, ctx.ggml_context)

    new_tensor = ctx.tensors_dict[node.output[0]] = ggml.ggml_map_custom2_inplace(
        ctx.ggml_context,
        x_t,
        input_tensor,
        custom_reduce_max,
        1,
        userdata_p,
    )

    ctx.refs.append(rmean_userdata)

    return new_tensor


@ggml.ggml_custom2_op_t
def custom_reduce_mean(
    tensor_out: ggml.ggml_tensor_p,
    tensor_in_1: ggml.ggml_tensor_p,
    tensor_in_2: ggml.ggml_tensor_p,
    ith: int,
    nth: int,
    userdata: Optional[ctypes.c_void_p],
):
    userdata_data_ptr = ctypes.cast(userdata, ctypes.POINTER(ReduceOpsUserData))
    userdata_data = userdata_data_ptr.contents

    tensor = ggml.utils.to_numpy(tensor_in_2)
    axes = [userdata_data.axes[i] for i in range(userdata_data.axes_length)]
    keepdims = userdata_data.keepdims

    axes = tuple(axes) if len(axes) else None
    rmean_result = np.mean(tensor, axis=axes, keepdims=keepdims)

    set_tensor_out(tensor_out, rmean_result)


@ggml_operator("ReduceMean")
def ggml_operator_reduce_mean(ctx: "GgmlOnnxExecutionContext", node: NodeProto):
    node_inputs = [ctx.tensors_dict[inp] for inp in node.input]

    if len(node_inputs) > 2 or len(node_inputs) < 1:
        raise ValueError(
            f'Error for node "{node.name}": Operation "ReduceMean" requires at least one input. Actual number of inputs: {len(node_inputs)}'
        )

    input_tensor = node_inputs[0]

    noop_with_empty_axes = next(
        (attr.i for attr in node.attribute if attr.name == "noop_with_empty_axes"), None
    )

    if noop_with_empty_axes == 1:
        ctx.tensors_dict[node.output[0]] = input_tensor
        return input_tensor

    tensor_shape = get_tensor_shape(input_tensor)
    tensor_dtype = get_tensor_dtype(input_tensor)

    axes = next((attr.ints for attr in node.attribute if attr.name == "axes"), None)
    if not axes:
        if len(node_inputs) > 1:
            axes_eval = ctx.backend.eval_tensor(node_inputs[1], ctx.ggml_context)
            axes = ggml.utils.to_numpy(axes_eval)
        else:
            axes = []

    keepdims = next((attr.i for attr in node.attribute if attr.name == "keepdims"), 1)

    rmean_userdata = ReduceOpsUserData(list(axes), keepdims)
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
    x_t = ggml.utils.from_numpy(x, ctx.ggml_context)

    new_tensor = ctx.tensors_dict[node.output[0]] = ggml.ggml_map_custom2_inplace(
        ctx.ggml_context,
        x_t,
        input_tensor,
        custom_reduce_mean,
        1,
        userdata_p,
    )

    ctx.refs.append(rmean_userdata)

    return new_tensor


@ggml.ggml_custom2_op_t
def custom_reduce_min(
    tensor_out: ggml.ggml_tensor_p,
    tensor_in_1: ggml.ggml_tensor_p,
    tensor_in_2: ggml.ggml_tensor_p,
    ith: int,
    nth: int,
    userdata: Optional[ctypes.c_void_p],
):
    userdata_data_ptr = ctypes.cast(userdata, ctypes.POINTER(ReduceOpsUserData))
    userdata_data = userdata_data_ptr.contents

    tensor = ggml.utils.to_numpy(tensor_in_2)
    axes = [userdata_data.axes[i] for i in range(userdata_data.axes_length)]
    keepdims = userdata_data.keepdims

    axes = tuple(axes) if len(axes) else None
    rmean_result = np.minimum.reduce(tensor, axis=axes, keepdims=keepdims)

    set_tensor_out(tensor_out, rmean_result)


@ggml_operator("ReduceMin")
def ggml_operator_reduce_mean(ctx: "GgmlOnnxExecutionContext", node: NodeProto):
    node_inputs = [ctx.tensors_dict[inp] for inp in node.input]

    if len(node_inputs) > 2 or len(node_inputs) < 1:
        raise ValueError(
            f'Error for node "{node.name}": Operation "ReduceMin" requires at least one input. Actual number of inputs: {len(node_inputs)}'
        )

    input_tensor = node_inputs[0]

    noop_with_empty_axes = next(
        (attr.i for attr in node.attribute if attr.name == "noop_with_empty_axes"), None
    )

    if noop_with_empty_axes == 1:
        ctx.tensors_dict[node.output[0]] = input_tensor
        return input_tensor

    tensor_shape = get_tensor_shape(input_tensor)
    tensor_dtype = get_tensor_dtype(input_tensor)

    axes = next((attr.ints for attr in node.attribute if attr.name == "axes"), None)
    if not axes:
        if len(node_inputs) > 1:
            axes_eval = ctx.backend.eval_tensor(node_inputs[1], ctx.ggml_context)
            axes = ggml.utils.to_numpy(axes_eval)
        else:
            axes = []

    keepdims = next((attr.i for attr in node.attribute if attr.name == "keepdims"), 1)

    rmean_userdata = ReduceOpsUserData(list(axes), keepdims)
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
    x_t = ggml.utils.from_numpy(x, ctx.ggml_context)

    new_tensor = ctx.tensors_dict[node.output[0]] = ggml.ggml_map_custom2_inplace(
        ctx.ggml_context,
        x_t,
        input_tensor,
        custom_reduce_min,
        1,
        userdata_p,
    )

    ctx.refs.append(rmean_userdata)

    return new_tensor


@ggml.ggml_custom2_op_t
def custom_reduce_prod(
    tensor_out: ggml.ggml_tensor_p,
    tensor_in_1: ggml.ggml_tensor_p,
    tensor_in_2: ggml.ggml_tensor_p,
    ith: int,
    nth: int,
    userdata: Optional[ctypes.c_void_p],
):
    userdata_data_ptr = ctypes.cast(userdata, ctypes.POINTER(ReduceOpsUserData))
    userdata_data = userdata_data_ptr.contents

    tensor = ggml.utils.to_numpy(tensor_in_2)
    axes = [userdata_data.axes[i] for i in range(userdata_data.axes_length)]
    keepdims = userdata_data.keepdims

    axes = tuple(axes) if len(axes) else None
    rmean_result = np.prod(tensor, axis=axes, keepdims=keepdims)

    set_tensor_out(tensor_out, rmean_result)


@ggml_operator("ReduceProd")
def ggml_operator_reduce_prod(ctx: "GgmlOnnxExecutionContext", node: NodeProto):
    node_inputs = [ctx.tensors_dict[inp] for inp in node.input]

    if len(node_inputs) > 2 or len(node_inputs) < 1:
        raise ValueError(
            f'Error for node "{node.name}": Operation "ReduceProd" requires at least one input. Actual number of inputs: {len(node_inputs)}'
        )

    input_tensor = node_inputs[0]

    noop_with_empty_axes = next(
        (attr.i for attr in node.attribute if attr.name == "noop_with_empty_axes"), None
    )

    if noop_with_empty_axes == 1:
        ctx.tensors_dict[node.output[0]] = input_tensor
        return input_tensor

    tensor_shape = get_tensor_shape(input_tensor)
    tensor_dtype = get_tensor_dtype(input_tensor)

    axes = next((attr.ints for attr in node.attribute if attr.name == "axes"), None)
    if not axes:
        if len(node_inputs) > 1:
            axes_eval = ctx.backend.eval_tensor(node_inputs[1], ctx.ggml_context)
            axes = ggml.utils.to_numpy(axes_eval)
        else:
            axes = []

    keepdims = next((attr.i for attr in node.attribute if attr.name == "keepdims"), 1)

    rmean_userdata = ReduceOpsUserData(list(axes), keepdims)
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
    x_t = ggml.utils.from_numpy(x, ctx.ggml_context)

    new_tensor = ctx.tensors_dict[node.output[0]] = ggml.ggml_map_custom2_inplace(
        ctx.ggml_context,
        x_t,
        input_tensor,
        custom_reduce_prod,
        1,
        userdata_p,
    )

    ctx.refs.append(rmean_userdata)

    return new_tensor


@ggml.ggml_custom2_op_t
def custom_reduce_sum(
    tensor_out: ggml.ggml_tensor_p,
    tensor_in_1: ggml.ggml_tensor_p,
    tensor_in_2: ggml.ggml_tensor_p,
    ith: int,
    nth: int,
    userdata: Optional[ctypes.c_void_p],
):
    userdata_data_ptr = ctypes.cast(userdata, ctypes.POINTER(ReduceOpsUserData))
    userdata_data = userdata_data_ptr.contents

    tensor = ggml.utils.to_numpy(tensor_in_2)
    axes = [userdata_data.axes[i] for i in range(userdata_data.axes_length)]
    keepdims = userdata_data.keepdims

    axes = tuple(axes) if len(axes) else None
    result = np.sum(tensor, axis=axes, keepdims=keepdims)
    set_tensor_out(tensor_out, result)


@ggml_operator("ReduceSum")
def ggml_operator_reduce_sum(ctx: "GgmlOnnxExecutionContext", node: NodeProto):
    node_inputs = [ctx.tensors_dict[inp] for inp in node.input]

    if len(node_inputs) > 2 or len(node_inputs) < 1:
        raise ValueError(
            f'Error for node "{node.name}": Operation "ReduceSum" requires at least one input. Actual number of inputs: {len(node_inputs)}'
        )

    input_tensor = node_inputs[0]

    noop_with_empty_axes = next(
        (attr.i for attr in node.attribute if attr.name == "noop_with_empty_axes"), None
    )

    if noop_with_empty_axes == 1:
        ctx.tensors_dict[node.output[0]] = input_tensor
        return input_tensor

    tensor_shape = get_tensor_shape(input_tensor)
    tensor_dtype = get_tensor_dtype(input_tensor)

    axes = next((attr.ints for attr in node.attribute if attr.name == "axes"), None)
    if not axes:
        if len(node_inputs) > 1:
            axes_eval = ctx.backend.eval_tensor(node_inputs[1], ctx.ggml_context)
            axes = ggml.utils.to_numpy(axes_eval)
        else:
            axes = []

    keepdims = next((attr.i for attr in node.attribute if attr.name == "keepdims"), 1)

    rmean_userdata = ReduceOpsUserData(list(axes), keepdims)
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
    x_t = ggml.utils.from_numpy(x, ctx.ggml_context)

    new_tensor = ctx.tensors_dict[node.output[0]] = ggml.ggml_map_custom2_inplace(
        ctx.ggml_context,
        x_t,
        input_tensor,
        custom_reduce_sum,
        1,
        userdata_p,
    )

    ctx.refs.append(rmean_userdata)

    return new_tensor


@ggml.ggml_custom2_op_t
def custom_reduce_sum_square(
    tensor_out: ggml.ggml_tensor_p,
    tensor_in_1: ggml.ggml_tensor_p,
    tensor_in_2: ggml.ggml_tensor_p,
    ith: int,
    nth: int,
    userdata: Optional[ctypes.c_void_p],
):
    userdata_data_ptr = ctypes.cast(userdata, ctypes.POINTER(ReduceOpsUserData))
    userdata_data = userdata_data_ptr.contents

    tensor = ggml.utils.to_numpy(tensor_in_2)
    axes = [userdata_data.axes[i] for i in range(userdata_data.axes_length)]
    keepdims = userdata_data.keepdims

    axes = tuple(axes) if len(axes) else None
    result = np.sum(np.square(tensor), axis=axes, keepdims=keepdims)

    set_tensor_out(tensor_out, result)


@ggml_operator("ReduceSumSquare")
def ggml_operator_reduce_sum_square(ctx: "GgmlOnnxExecutionContext", node: NodeProto):
    node_inputs = [ctx.tensors_dict[inp] for inp in node.input]

    if len(node_inputs) > 2 or len(node_inputs) < 1:
        raise ValueError(
            f'Error for node "{node.name}": Operation "ReduceSumSquare" requires at least one input. Actual number of inputs: {len(node_inputs)}'
        )

    input_tensor = node_inputs[0]

    noop_with_empty_axes = next(
        (attr.i for attr in node.attribute if attr.name == "noop_with_empty_axes"), None
    )

    if noop_with_empty_axes == 1:
        ctx.tensors_dict[node.output[0]] = input_tensor
        return input_tensor

    tensor_shape = get_tensor_shape(input_tensor)
    tensor_dtype = get_tensor_dtype(input_tensor)

    axes = next((attr.ints for attr in node.attribute if attr.name == "axes"), None)
    if not axes:
        if len(node_inputs) > 1:
            axes_eval = ctx.backend.eval_tensor(node_inputs[1], ctx.ggml_context)
            axes = ggml.utils.to_numpy(axes_eval)
        else:
            axes = []

    keepdims = next((attr.i for attr in node.attribute if attr.name == "keepdims"), 1)

    rmean_userdata = ReduceOpsUserData(list(axes), keepdims)
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
    x_t = ggml.utils.from_numpy(x, ctx.ggml_context)

    new_tensor = ctx.tensors_dict[node.output[0]] = ggml.ggml_map_custom2_inplace(
        ctx.ggml_context,
        x_t,
        input_tensor,
        custom_reduce_sum_square,
        1,
        userdata_p,
    )

    ctx.refs.append(rmean_userdata)

    return new_tensor


@ggml_operator("Relu")
def ggml_operator_relu(ctx: "GgmlOnnxExecutionContext", node: NodeProto):
    node_inputs = [ctx.tensors_dict[inp] for inp in node.input]

    if len(node_inputs) != 1:
        raise ValueError(
            f'Error for node "{node.name}": Operation "Relu" requires exactly one input. Actual number of inputs: {len(node_inputs)}'
        )

    output_name = node.output[0]
    a = node_inputs[0]

    relu_result = ggml.ggml_relu(
        ctx.ggml_context,
        a,
    )
    ctx.tensors_dict[output_name] = relu_result
    return relu_result


@ggml_operator("Reshape")
def ggml_operator_reshape(ctx: "GgmlOnnxExecutionContext", node: NodeProto):
    node_inputs = [ctx.tensors_dict[inp] for inp in node.input]
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
    eval_b = ctx.backend.eval_tensor(b, ctx.ggml_context)

    new_shape = ggml.utils.to_numpy(eval_b).astype(dtype=np.int32)

    old_shape = get_tensor_shape(a)
    if not allowzero:
        keep_idxs = np.where(new_shape == 0)[0]
        new_shape[keep_idxs] = np.array(old_shape)[keep_idxs]

    temp_a = np.empty(old_shape, dtype=get_tensor_dtype(a))
    x = temp_a.reshape(new_shape)
    x_t = ggml.utils.from_numpy(x, ctx.ggml_context)

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

    new_tensor = ctx.tensors_dict[node.output[0]] = ggml.ggml_map_custom2_inplace(
        ctx.ggml_context,
        x_t,
        a,
        custom_reshape,
        1,
        None,
    )

    ctx.refs.append(custom_reshape)

    return new_tensor


class SeluUserData(ctypes.Structure):
    _fields_ = [
        ("alpha", ctypes.c_double),
        ("gamma", ctypes.c_double),
    ]


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
    x = ggml.utils.to_numpy(tensor_in_1)

    alpha = userdata_data.alpha
    gamma = userdata_data.gamma

    y = (
        np.clip(x, 0, np.inf) * gamma
        + (np.exp(np.clip(x, -np.inf, 0)) - 1) * alpha * gamma
    )

    set_tensor_out(tensor_out, y)


@ggml_operator("Selu")
def ggml_operator_selu(ctx: "GgmlOnnxExecutionContext", node: NodeProto):
    node_inputs = [ctx.tensors_dict[inp] for inp in node.input]

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

    new_tensor = ctx.tensors_dict[node.output[0]] = ggml.ggml_map_custom1_inplace(
        ctx.ggml_context,
        x,
        custom_selu,
        1,
        userdata_p,
    )

    ctx.refs.append(selu_userdata)

    return new_tensor


@ggml_operator("Shape")
def ggml_operator_shape(ctx: "GgmlOnnxExecutionContext", node: NodeProto):
    node_inputs = [ctx.tensors_dict[inp] for inp in node.input]

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
    new_tensor = ctx.tensors_dict[name] = ggml.utils.from_numpy(
        shape_slice, ctx.ggml_context
    )

    ggml.ggml_set_name(new_tensor, (name + f"<int64>").encode())

    return new_tensor


@ggml.ggml_custom1_op_t
def custom_sigmoid(
    tensor_out: ggml.ggml_tensor_p,
    tensor_in_1: ggml.ggml_tensor_p,
    ith: int,
    nth: int,
    userdata: Optional[ctypes.c_void_p],
):
    x = ggml.utils.to_numpy(tensor_in_1)

    y = 1.0 / (1.0 + np.exp(np.negative(x)))

    set_tensor_out(tensor_out, y)


@ggml_operator("Sigmoid")
def ggml_operator_sigmoid(ctx: "GgmlOnnxExecutionContext", node: NodeProto):
    node_inputs = [ctx.tensors_dict[inp] for inp in node.input]

    if len(node_inputs) != 1:
        raise ValueError(
            f'Error for node "{node.name}": Operation "Sigmoid" requires exactly one input. Actual number of inputs: {len(node_inputs)}'
        )

    x = node_inputs[0]
    new_tensor = ctx.tensors_dict[node.output[0]] = ggml.ggml_map_custom1_inplace(
        ctx.ggml_context,
        x,
        custom_sigmoid,
        1,
        None,
    )

    return new_tensor


@ggml.ggml_custom2_op_t
def custom_size(
    tensor_out: ggml.ggml_tensor_p,
    tensor_in_1: ggml.ggml_tensor_p,
    tensor_in_2: ggml.ggml_tensor_p,
    ith: int,
    nth: int,
    userdata: Optional[ctypes.c_void_p],
):
    tensor = ggml.utils.to_numpy(tensor_in_2)
    set_tensor_out(tensor_out, tensor)


@ggml_operator("Size")
def ggml_operator_size(ctx: "GgmlOnnxExecutionContext", node: NodeProto):
    node_inputs = [ctx.tensors_dict[inp] for inp in node.input]

    if len(node_inputs) != 1:
        raise ValueError(
            f'Error for node "{node.name}": Operation "Size" requires exactly one input. Actual number of inputs: {len(node_inputs)}'
        )

    tensor_shape = np.array(get_tensor_shape(node_inputs[0]), dtype=np.int32)
    name = node.output[0]
    tensor_size_np = np.prod(tensor_shape).astype(np.int32)
    tensor_size_np = np.array(
        [tensor_size_np]
    )  # Add a rank so ggml doesnt break the value, inside the custom reshape to scalar as expected TODO: Fix the ranking, ggml skalars or make sure broadcasting works fine
    tensor_size_t = ggml.utils.from_numpy(np.array([tensor_size_np]), ctx.ggml_context)

    ggml_type = map_to_ggml_type(tensor_size_np.dtype).value
    x = np.empty(tensor_shape, dtype=tensor_size_np.dtype)
    x_t = ggml.utils.from_numpy(x, ctx.ggml_context)

    new_tensor = ctx.tensors_dict[name] = ggml.ggml_map_custom2_inplace(
        ctx.ggml_context,
        x_t,
        tensor_size_t,
        custom_size,
        1,
        None,
    )

    ggml.ggml_set_name(new_tensor, (name + f"<int64>").encode())

    return new_tensor


@ggml_operator("Softmax")
def ggml_operator_softmax(ctx: "GgmlOnnxExecutionContext", node: NodeProto):
    node_inputs = [ctx.tensors_dict[inp] for inp in node.input]

    if len(node_inputs) != 1:
        raise ValueError(
            f'Error for node "{node.name}": Operation "Softmax" requires exactly one input. Actual number of inputs: {len(node_inputs)}'
        )

    output_name = node.output[0]
    a = node_inputs[0]

    soft_max_result = ggml.ggml_soft_max(
        ctx.ggml_context,
        a,
    )
    ctx.tensors_dict[output_name] = soft_max_result
    return soft_max_result


@ggml.ggml_custom1_op_t
def custom_softplus(
    tensor_out: ggml.ggml_tensor_p,
    tensor_in_1: ggml.ggml_tensor_p,
    ith: int,
    nth: int,
    userdata: Optional[ctypes.c_void_p],
):
    x = ggml.utils.to_numpy(tensor_in_1)
    y = np.log(np.exp(x) + 1)
    set_tensor_out(tensor_out, y)


@ggml_operator("Softplus")
def ggml_operator_softplus(ctx: "GgmlOnnxExecutionContext", node: NodeProto):
    node_inputs = [ctx.tensors_dict[inp] for inp in node.input]

    if len(node_inputs) != 1:
        raise ValueError(
            f'Error for node "{node.name}": Operation "Softplus" requires exactly one input. Actual number of inputs: {len(node_inputs)}'
        )

    x = node_inputs[0]

    new_tensor = ctx.tensors_dict[node.output[0]] = ggml.ggml_map_custom1_inplace(
        ctx.ggml_context,
        x,
        custom_softplus,
        1,
        None,
    )

    return new_tensor


@ggml_operator("Softsign")
def ggml_operator_softsign(ctx: "GgmlOnnxExecutionContext", node: NodeProto):
    node_inputs = [ctx.tensors_dict[inp] for inp in node.input]

    if len(node_inputs) != 1:
        raise ValueError(
            f'Error for node "{node.name}": Operation "Softsign" requires exactly one input. Actual number of inputs: {len(node_inputs)}'
        )

    x = node_inputs[0]
    x_shape = get_tensor_shape(x)
    x_dtype = get_tensor_dtype(x)

    # y = x / (1 + abs(x))
    one_np = np.full(x_shape, 1, dtype=x_dtype)
    one_t = ggml.utils.from_numpy(one_np, ctx.ggml_context)
    x_abs = ggml.ggml_abs(ctx.ggml_context, x)
    one_plus_abs = ggml.ggml_add(ctx.ggml_context, one_t, x_abs)
    y = ggml.ggml_div(ctx.ggml_context, x, one_plus_abs)
    ctx.tensors_dict[node.output[0]] = y

    return y


@ggml.ggml_custom2_op_t
def custom_space_to_depth(
    tensor_out: ggml.ggml_tensor_p,
    tensor_in_1: ggml.ggml_tensor_p,
    tensor_in_2: ggml.ggml_tensor_p,
    ith: int,
    nth: int,
    userdata: Optional[ctypes.c_void_p],
):
    x = ggml.utils.to_numpy(tensor_in_2)
    blocksize = ctypes.cast(userdata, ctypes.POINTER(ctypes.c_int)).contents.value

    N, C, H, W = x.shape
    new_H = H // blocksize
    new_W = W // blocksize

    reshaped = x.reshape(N, C, new_H, blocksize, new_W, blocksize)
    transposed = reshaped.transpose(
        0, 3, 5, 1, 2, 4
    )  # ONNX specification TODO: Test more examples
    y = transposed.reshape(N, C * (blocksize**2), new_H, new_W)

    set_tensor_out(tensor_out, y)


@ggml_operator("SpaceToDepth")
def ggml_operator_space_to_depth(ctx: "GgmlOnnxExecutionContext", node: NodeProto):
    node_inputs = [ctx.tensors_dict[inp] for inp in node.input]

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

    N, C, H, W = get_tensor_shape(x)
    new_H = H // blocksize
    new_W = W // blocksize
    output_shape = (N, C * blocksize * blocksize, new_H, new_W)

    x_t = ggml.utils.from_numpy(
        np.empty(output_shape, dtype=get_tensor_dtype(x)), ctx.ggml_context
    )

    blocksize_c = ctypes.c_int(blocksize)

    new_tensor = ctx.tensors_dict[node.output[0]] = ggml.ggml_map_custom2_inplace(
        ctx.ggml_context,
        x_t,
        x,
        custom_space_to_depth,
        1,
        ctypes.pointer(blocksize_c),
    )

    ctx.refs.append(blocksize_c)

    return new_tensor


class SplitUserData(ctypes.Structure):
    _fields_ = [
        ("axis", ctypes.c_int),
        ("split_index", ctypes.c_int),
    ]


@ggml.ggml_custom3_op_t
def custom_split(
    tensor_out: ggml.ggml_tensor_p,
    tensor_in_1: ggml.ggml_tensor_p,
    tensor_in_2: ggml.ggml_tensor_p,
    tensor_in_3: ggml.ggml_tensor_p,
    ith: int,
    nth: int,
    userdata: Optional[ctypes.c_void_p],
):
    userdata_data_ptr = ctypes.cast(userdata, ctypes.POINTER(SplitUserData))
    userdata_data = userdata_data_ptr.contents

    axis = userdata_data.axis
    split_index = userdata_data.split_index

    tensor = ggml.utils.to_numpy(tensor_in_2)

    split_shapes = ggml.utils.to_numpy(tensor_in_3)
    split_shape = list(ggml.utils.to_numpy(tensor_in_1).shape)

    split_size = split_shape[axis]
    split_start = sum(split_shapes[i][axis] for i in range(split_index))
    split_end = split_start + split_size

    split_output = np.take(tensor, range(split_start, split_end), axis=axis)

    set_tensor_out(tensor_out, split_output)


@ggml_operator("Split")
def ggml_operator_split(ctx: "GgmlOnnxExecutionContext", node: NodeProto):
    node_inputs = [ctx.tensors_dict[inp] for inp in node.input]

    if len(node_inputs) < 1 or len(node_inputs) > 2:
        raise ValueError(
            f'Error for node "{node.name}": Operation "Split" requires 1 - 2 inputs. Actual number of inputs: {len(node_inputs)}'
        )

    input_tensor = node_inputs.pop(0)
    split_tensor = node_inputs.pop(0) if len(node_inputs) else None

    axis = next((attr.i for attr in node.attribute if attr.name == "axis"), 0)
    num_outputs = next(
        (attr.i for attr in node.attribute if attr.name == "num_outputs"),
        len(node.output),
    )

    input_shape = list(get_tensor_shape(input_tensor))
    dtype = get_tensor_dtype(input_tensor)

    if split_tensor is None:
        split_size = input_shape[axis] // num_outputs
        remainder = input_shape[axis] % num_outputs
        split_shapes = [list(input_shape) for _ in range(num_outputs)]

        for i in range(num_outputs):
            split_shapes[i][axis] = split_size
            if i < remainder:
                split_shapes[i][axis] += 1

        split_shapes = [tuple(split_shape) for split_shape in split_shapes]

    else:
        split_eval = ctx.backend.eval_tensor(split_tensor, ctx.ggml_context)
        split_values = ggml.utils.to_numpy(split_eval)
        split_shapes = [list(input_shape) for _ in range(num_outputs)]

        for i, split_value in enumerate(split_values):
            split_shapes[i][axis] = split_value

        split_shapes = tuple(map(tuple, split_shapes))

    split_shapes_np = np.array(split_shapes, dtype=np.int32)
    split_shapes_t = ggml.utils.from_numpy(split_shapes_np, ctx.ggml_context)

    outputs = []

    for split_index, split_shape in enumerate(split_shapes):
        split_userdata = SplitUserData(axis, split_index)
        userdata_p = ctypes.cast(ctypes.pointer(split_userdata), ctypes.c_void_p)

        x_t = ggml.utils.from_numpy(
            np.empty(split_shape, dtype=dtype), ctx.ggml_context
        )
        new_tensor = ctx.tensors_dict[
            node.output[split_index]
        ] = ggml.ggml_map_custom3_inplace(
            ctx.ggml_context,
            x_t,
            input_tensor,
            split_shapes_t,
            custom_split,
            1,
            userdata_p,
        )

        ctx.refs.append(split_userdata)
        outputs.append(new_tensor)

    return outputs


@ggml_operator("Sqrt")
def ggml_operator_sqrt(ctx: "GgmlOnnxExecutionContext", node: NodeProto):
    node_inputs = [ctx.tensors_dict[inp] for inp in node.input]

    if len(node_inputs) != 1:
        raise ValueError(
            f'Error for node "{node.name}": Operation "Sqrt" requires exactly one input. Actual number of inputs: {len(node_inputs)}'
        )

    output_name = node.output[0]
    a = node_inputs[0]

    sqrt_result = ggml.ggml_sqrt(
        ctx.ggml_context,
        a,
    )
    ctx.tensors_dict[output_name] = sqrt_result
    return sqrt_result


@ggml.ggml_custom3_op_t
def custom_squeeze(
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

    y = np.squeeze(x, axis=axes[0])

    set_tensor_out(tensor_out, y)


@ggml_operator("Squeeze")
def ggml_operator_squeeze(ctx: "GgmlOnnxExecutionContext", node: NodeProto):
    node_inputs = [ctx.tensors_dict[inp] for inp in node.input]

    if len(node_inputs) != 2:
        raise ValueError(
            f'Error for node "{node.name}": Operation "Squeeze" requires exactly two inputs, data and axes. Actual number of inputs: {len(node_inputs)}'
        )

    data, axes_input = node_inputs

    x_shape = get_tensor_shape(data)
    x_dtype = get_tensor_dtype(data)

    axes_eval = ctx.backend.eval_tensor(axes_input, ctx.ggml_context)
    axes = ggml.utils.to_numpy(axes_eval).astype(dtype=np.int32)

    dummy_data = np.empty(x_shape, dtype=x_dtype)
    dummy_data = np.squeeze(dummy_data, axis=axes[0])

    if len(dummy_data.shape) > 4:
        raise ValueError(
            f'Error for node "{node.name}": {len(dummy_data.shape)}D arrays are not allowed.'
        )

    x_t = ggml.utils.from_numpy(dummy_data, ctx.ggml_context)

    new_tensor = ctx.tensors_dict[node.output[0]] = ggml.ggml_map_custom3_inplace(
        ctx.ggml_context,
        x_t,
        data,
        axes_input,
        custom_squeeze,
        1,
        None,
    )

    return new_tensor


@ggml_operator("Sub")
def ggml_operator_sub(ctx: "GgmlOnnxExecutionContext", node: NodeProto):
    node_inputs = [ctx.tensors_dict[inp] for inp in node.input]

    if len(node_inputs) != 2:
        raise ValueError(
            f'Error for node "{node.name}": Operation "Sub" requires exactly two inputs. Actual number of inputs: {len(node_inputs)}'
        )

    output_name = node.output[0]
    a, b = node_inputs
    a, b = broadcast_shapes(ctx.ggml_context, a, b)

    sub_result = ggml.ggml_sub(
        ctx.ggml_context,
        a,
        b,
    )
    ctx.tensors_dict[output_name] = sub_result
    return sub_result


@ggml_operator("Sum")
def ggml_operator_sum(ctx: "GgmlOnnxExecutionContext", node: NodeProto):
    node_inputs = [ctx.tensors_dict[inp] for inp in node.input]

    if len(node_inputs) < 1:
        raise ValueError(
            f'Error for node "{node.name}": Operation "Sum" requires at least one input. Actual number of inputs: {len(node_inputs)}'
        )

    output_name = node.output[0]
    shape = get_tensor_shape(node_inputs[0])
    dtype = get_tensor_dtype(node_inputs[0])

    empty_np = np.full(shape, 0, dtype=dtype)
    next_item = ggml.utils.from_numpy(empty_np, ctx.ggml_context)

    for tensor in node_inputs:
        tensor, next_item = broadcast_shapes(ctx.ggml_context, tensor, next_item)
        next_item = ggml.ggml_add(
            ctx.ggml_context,
            tensor,
            next_item,
        )

    ctx.tensors_dict[output_name] = next_item

    return next_item


@ggml_operator("Tanh")
def ggml_operator_tanh(ctx: "GgmlOnnxExecutionContext", node: NodeProto):
    node_inputs = [ctx.tensors_dict[inp] for inp in node.input]

    if len(node_inputs) != 1:
        raise ValueError(
            f'Error for node "{node.name}": Operation "Tanh" requires exactly one input. Actual number of inputs: {len(node_inputs)}'
        )

    x = node_inputs[0]
    tanh_result = ggml.ggml_tanh(
        ctx.ggml_context,
        x,
    )

    ctx.tensors_dict[node.output[0]] = tanh_result

    return tanh_result


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
    x = ggml.utils.to_numpy(tensor_in_2)
    repeats = ggml.utils.to_numpy(tensor_in_3)

    y = np.tile(x, repeats)

    set_tensor_out(tensor_out, y)


@ggml_operator("Tile")
def ggml_operator_tile(ctx: "GgmlOnnxExecutionContext", node: NodeProto):
    node_inputs = [ctx.tensors_dict[inp] for inp in node.input]

    if len(node_inputs) != 2:
        raise ValueError(
            f'Error for node "{node.name}": Operation "Tile" requires exactly two inputs. Actual number of inputs: {len(node_inputs)}'
        )

    x, repeats = node_inputs

    repeats_eval = ctx.backend.eval_tensor(repeats, ctx.ggml_context)
    repeats_vals = ggml.utils.to_numpy(repeats_eval).astype(dtype=np.int32)

    output_shape = list(get_tensor_shape(x))
    for i in range(len(output_shape)):
        output_shape[i] = output_shape[i] * repeats_vals[i]

    x_t = ggml.utils.from_numpy(
        np.empty(output_shape, dtype=get_tensor_dtype(x)),
        ctx.ggml_context,
    )

    new_tensor = ctx.tensors_dict[node.output[0]] = ggml.ggml_map_custom3_inplace(
        ctx.ggml_context,
        x_t,
        x,
        repeats,
        custom_tile,
        1,
        None,
    )

    return new_tensor


class TopKUserData(ctypes.Structure):
    _fields_ = [
        ("axis", ctypes.c_int),
        ("largest", ctypes.c_int),
        ("sorted", ctypes.c_int),
        ("k", ctypes.c_int),
    ]


@ggml.ggml_custom2_op_t
def custom_top_k_indices(
    tensor_out: ggml.ggml_tensor_p,
    tensor_in_1: ggml.ggml_tensor_p,
    tensor_in_2: ggml.ggml_tensor_p,
    ith: int,
    nth: int,
    userdata: Optional[ctypes.c_void_p],
):
    x = ggml.utils.to_numpy(tensor_in_2)

    userdata_data_ptr = ctypes.cast(userdata, ctypes.POINTER(TopKUserData))
    userdata_data = userdata_data_ptr.contents

    axis = userdata_data.axis
    largest = bool(userdata_data.largest)

    k = userdata_data.k

    if largest:
        sorted_indices = np.argsort(x, axis=axis)[:, ::-1]
    else:
        sorted_indices = np.argsort(x, axis=axis)

    topk_indices = sorted_indices[:, :k]

    set_tensor_out(tensor_out, topk_indices)


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
    x = ggml.utils.to_numpy(tensor_in_2)
    topk_indices = ggml.utils.to_numpy(tensor_in_3).astype(np.int32)

    userdata_data_ptr = ctypes.cast(userdata, ctypes.POINTER(TopKUserData))
    userdata_data = userdata_data_ptr.contents

    axis = userdata_data.axis
    sorted_flag = bool(userdata_data.sorted)

    topk_values = np.take_along_axis(x, topk_indices, axis=axis)
    if sorted_flag:
        topk_values_sorted = np.sort(topk_values, axis=axis)
    else:
        topk_values_sorted = topk_values

    set_tensor_out(tensor_out, topk_values_sorted)


@ggml_operator("TopK")
def ggml_operator_top_k(ctx: "GgmlOnnxExecutionContext", node: NodeProto):
    node_inputs = [ctx.tensors_dict[inp] for inp in node.input]

    if len(node_inputs) != 2:
        raise ValueError(
            f'Error for node "{node.name}": Operation "TopK" requires exactly two inputs. Actual number of inputs: {len(node_inputs)}'
        )

    x, k = node_inputs

    input_shape = get_tensor_shape(x)

    axis = next((attr.i for attr in node.attribute if attr.name == "axis"), -1)
    largest = next((attr.i for attr in node.attribute if attr.name == "largest"), 1)
    sorted_flag = next((attr.i for attr in node.attribute if attr.name == "sorted"), 0)

    k_eval = ctx.backend.eval_tensor(k, ctx.ggml_context)
    k_np = ggml.utils.to_numpy(k_eval)[0]

    topk_userdata = TopKUserData(axis, largest, sorted_flag, k_np)
    userdata_p = ctypes.cast(ctypes.pointer(topk_userdata), ctypes.c_void_p)

    output_shape = list(input_shape)
    output_shape[axis] = k_np
    output_shape = tuple(output_shape)

    indices_t = ggml.utils.from_numpy(
        np.empty(output_shape, dtype=np.int32),
        ctx.ggml_context,
    )

    values_t = ggml.utils.from_numpy(
        np.empty(output_shape, dtype=get_tensor_dtype(x)),
        ctx.ggml_context,
    )

    indices = ggml.ggml_map_custom2_inplace(
        ctx.ggml_context,
        indices_t,
        x,
        custom_top_k_indices,
        1,
        userdata_p,
    )

    values = ggml.ggml_map_custom3_inplace(
        ctx.ggml_context,
        values_t,
        x,
        indices,
        custom_top_k_values,
        1,
        userdata_p,
    )

    ctx.tensors_dict[node.output[0]] = values
    ctx.tensors_dict[node.output[1]] = indices

    ctx.refs.append(topk_userdata)

    ggml.ggml_set_name(indices, (node.output[1] + f"<int64>").encode())

    return values, indices


@ggml_operator("Transpose")
def ggml_operator_transpose(ctx: "GgmlOnnxExecutionContext", node: NodeProto):
    node_inputs = [ctx.tensors_dict[inp] for inp in node.input]

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
        x = ggml.ggml_transpose(ctx.ggml_context, x)

    transpose_result = ggml.ggml_permute(ctx.ggml_context, x, ax0, ax1, ax2, ax3)

    if dims == 3 and f"02" in "".join([str(perm) for perm in perms]):
        transpose_result = ggml.ggml_permute(
            ctx.ggml_context, transpose_result, 0, 2, 1, 3
        )

    ctx.tensors_dict[output_name] = transpose_result
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
def ggml_operator_unsqueeze(ctx: "GgmlOnnxExecutionContext", node: NodeProto):
    node_inputs = [ctx.tensors_dict[inp] for inp in node.input]

    if len(node_inputs) != 2:
        raise ValueError(
            f'Error for node "{node.name}": Operation "Unsqueeze" requires exactly two inputs, data and axes. Actual number of inputs: {len(node_inputs)}'
        )

    data = node_inputs[0]
    axes_input = node_inputs[1]

    x_shape = get_tensor_shape(data)
    x_dtype = get_tensor_dtype(data)
    x_ndims = ggml.utils.get_ndims(data)

    axes_eval = ctx.backend.eval_tensor(axes_input, ctx.ggml_context)
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
        ctx.ggml_context,
        ggml_type.value,
        len(new_shape),
        (ctypes.c_int64 * len(new_shape))(*new_shape),
    )

    new_tensor = ctx.tensors_dict[node.output[0]] = ggml.ggml_map_custom3_inplace(
        ctx.ggml_context,
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
def ggml_operator_where(ctx: "GgmlOnnxExecutionContext", node: NodeProto):
    node_inputs = [ctx.tensors_dict[inp] for inp in node.input]

    if len(node_inputs) != 3:
        raise ValueError(
            f'Error for node "{node.name}": Operation "Where" requires exactly three inputs. Actual number of inputs: {len(node_inputs)}'
        )

    new_tensor = ctx.tensors_dict[node.output[0]] = ggml.ggml_map_custom3_inplace(
        ctx.ggml_context,
        node_inputs[1],
        node_inputs[2],
        node_inputs[0],
        custom_where,
        1,
        None,
    )

    return new_tensor


@ggml.ggml_custom3_op_t
def custom_xor(
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

    x = np.logical_xor(a, b)

    set_tensor_out(tensor_out, x)


@ggml_operator("Xor")
def ggml_operator_xor(ctx: "GgmlOnnxExecutionContext", node: NodeProto):
    node_inputs = [ctx.tensors_dict[inp] for inp in node.input]

    if len(node_inputs) != 2:
        raise ValueError(
            f'Error for node "{node.name}": Operation "Xor" requires exactly two inputs. Actual number of inputs: {len(node_inputs)}'
        )

    a_shape = get_tensor_shape(node_inputs[0])
    a_dtype = get_tensor_dtype(node_inputs[0])
    b_shape = get_tensor_shape(node_inputs[1])
    name = node.output[0]

    output_shape = np.broadcast(np.empty(a_shape), np.empty(b_shape)).shape

    x = np.empty(output_shape, dtype=a_dtype)
    x_t = ggml.utils.from_numpy(x, ctx.ggml_context)

    new_tensor = ctx.tensors_dict[name] = ggml.ggml_map_custom3_inplace(
        ctx.ggml_context,
        x_t,
        node_inputs[0],
        node_inputs[1],
        custom_xor,
        1,
        None,
    )

    ggml.ggml_set_name(new_tensor, (name + f"<bool>").encode())

    return new_tensor


class GgmlOnnxExecutionContext:
    def __init__(
        self,
        backend: "GgmlBackendRep",
        tensors_dict: Dict[str, ggml.ggml_tensor_p],
        ggml_context: ggml.ggml_context_p,
        refs: List[Any],
    ):
        self.backend = backend
        self.tensors_dict = tensors_dict
        self.ggml_context = ggml_context
        self.refs = refs


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

    def eval_tensor(
        self, tensor: ggml.ggml_tensor_p, ggml_context: ggml.ggml_context_p
    ):
        gf = ggml.ggml_build_forward(tensor)
        ggml.ggml_graph_compute_with_ctx(ggml_context, ctypes.pointer(gf), 1)

        return tensor

    def run(self, inputs: Any, **kwargs: Any) -> Tuple[Any, ...]:
        """Run the model with the specified inputs."""

        if isinstance(inputs, list):
            inputs = {k.name: v for k, v in zip(self.inputs, inputs)}

        assert isinstance(inputs, dict)

        model_graph = self.graph
        exit_node = None
        ggml_tensors = self.weights

        # Define ggml_context
        params = ggml.ggml_init_params(mem_size=16 * 1024 * 1024, mem_buffer=None)
        ggml_context = ggml.ggml_init(params=params)

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

            # Handle scalars
            if len(shape) == 0:
                shape = (1,)

            tensor = ggml.ggml_new_tensor(
                ggml_context,
                ggml_type.value,
                len(shape),
                (ctypes.c_int64 * len(shape))(*shape),
            )

            ggml_tensors[input_name] = tensor

        # Set user inputs
        for key, value in inputs.items():
            set_tensor_out(ggml_tensors[key], np.array(value))

        gf = ggml.ggml_cgraph()
        gf_p = ctypes.pointer(gf)
        output_names = [output.name for output in model_graph.output]

        ctx = GgmlOnnxExecutionContext(self, ggml_tensors, ggml_context, refs)

        # Build layers
        for node in model_graph.node:
            operator_func = ggml_operators.get(node.op_type)
            if operator_func is None:
                raise NotImplementedError(f'Operator "{node.op_type}" not implemented')

            operator_func(
                ctx,
                node,
            )

            for output in node.output:
                if output in output_names:
                    ggml.ggml_build_forward_expand(gf_p, ggml_tensors[output])

        # Compute graph
        ggml.ggml_graph_compute_with_ctx(ggml_context, gf_p, 1)

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

        ggml.ggml_free(ggml_context)

        return graph_outputs


class GgmlRuntimeBackend(Backend):
    @classmethod
    def is_opset_supported(cls, model: ModelProto):
        return True, ""

    @classmethod
    def prepare(cls, model: ModelProto, device: str = "CPU", **kwargs):
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

        ggml_context = ggml.ggml_init(init_params)
        total_nbytes = 0

        pairs = []

        for initializer in graph.initializer:
            name = initializer.name
            np_array = onnx.numpy_helper.to_array(initializer)
            tensor = ggml.utils.from_numpy(x=np_array, ctx=ggml_context)
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
            set_tensor_out(tensor, np_array)

            offset += nbytes

        return GgmlBackendRep(
            graph=graph,
            weights=weights,
            weights_buffer=buffer,
            inputs=graph.input,
            outputs=graph.output,
            ggml_context=ggml_context,
            ggml_init_params=init_params,
        )

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
        outputs_info=None,
        **kwargs,
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
