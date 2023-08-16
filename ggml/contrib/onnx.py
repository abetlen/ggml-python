import ctypes
from typing import Any, Tuple, List

import numpy as np
import onnx
from onnx import defs
from onnx.backend.base import Backend, BackendRep
from onnx.helper import make_opsetid
from onnx.onnx_ml_pb2 import GraphProto, ModelProto, NodeProto
from onnx.helper import tensor_dtype_to_np_dtype

import ggml
import ggml.utils
from typing import Optional

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
    ggml_type = ggml.utils.NUMPY_DTYPE_TO_GGML_TYPE.get(
        dtype.type,
        ggml.utils.GGML_TYPE.I32,  # TODO: Add i64 but for now, use i32 if looking for i64 or f64
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


# ------ Operators ------


@ggml_operator("Abs")
def ggml_operator_abs(
    node: NodeProto, tensors_dict: dict, context: ggml.ggml_context_p, refs: List[Any]
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
    node: NodeProto, tensors_dict: dict, context: ggml.ggml_context_p, refs: List[Any]
):
    node_inputs = [tensors_dict[inp] for inp in node.input]

    if len(node_inputs) != 2:
        raise ValueError(
            f'Error for node "{node.name}": Operation "Add" requires exactly two inputs. Actual number of inputs: {len(node_inputs)}'
        )

    output_name = node.output[0]

    a = node_inputs[0]
    b = node_inputs[1]

    add_result = ggml.ggml_add(
        context,
        a,
        b,
    )
    tensors_dict[output_name] = add_result
    return add_result


@ggml_operator("Cast")
def ggml_operator_cast(
    node: NodeProto, tensors_dict: dict, context: ggml.ggml_context_p, refs: List[Any]
):
    raise NotImplementedError(f'Operator "Cast" not implemented')


@ggml_operator("Concat")
def ggml_operator_concat(
    node: NodeProto, tensors_dict: dict, context: ggml.ggml_context_p, refs: List[Any]
):
    node_inputs = [tensors_dict[inp] for inp in node.input]

    if len(node_inputs) < 2:
        raise ValueError(
            f'Error for node "{node.name}": Operation "Concat" requires at least two inputs and an axis. Actual number of inputs: {len(node_inputs)}'
        )

    shapes = [get_tensor_shape(tensor) for tensor in node_inputs]

    axis = next(attr for attr in node.attribute if attr.name == "axis").i
    output_shape = np.concatenate([np.empty(shape) for shape in shapes]).shape

    print()
    print()
    print("axis:", axis)
    print("shapes:", shapes)
    print("output_shape:", output_shape)
    print()
    print()

    raise NotImplementedError(f'Operator "Concat" not implemented')


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
    node: NodeProto, tensors_dict: dict, context: ggml.ggml_context_p, refs: List[Any]
):
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
    node: NodeProto, tensors_dict: dict, context: ggml.ggml_context_p, refs: List[Any]
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
    node: NodeProto, tensors_dict: dict, context: ggml.ggml_context_p, refs: List[Any]
):
    node_inputs = [tensors_dict[inp] for inp in node.input]

    if len(node_inputs) != 2:
        raise ValueError(
            f'Error for node "{node.name}": Operation "Div" requires exactly two inputs. Actual number of inputs: {len(node_inputs)}'
        )

    output_name = node.output[0]
    a = node_inputs[0]
    b = node_inputs[1]

    div_result = ggml.ggml_div(
        context,
        a,
        b,
    )
    tensors_dict[output_name] = div_result
    return div_result


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

    ggml.utils.to_numpy(tensor_out)[:] = new_array


@ggml_operator("Gather")
def ggml_operator_gather(
    node: NodeProto, tensors_dict: dict, context: ggml.ggml_context_p, refs: List[Any]
):
    node_inputs = [tensors_dict[inp] for inp in node.input]

    if len(node_inputs) != 2:
        raise ValueError(
            f'Error for node "{node.name}": Operation "Gather" requires exactly two inputs and one axis. Actual number of inputs: {len(node_inputs)}'
        )

    axis = node.attribute[0].i if len(node.attribute) > 0 else -1

    axis_c = ctypes.c_int(axis)

    input_ndim = ggml.utils.get_ndims(node_inputs[0])
    input_dtype = get_tensor_dtype(node_inputs[0])
    index_shape = get_tensor_shape(node_inputs[1])

    output_shape = (input_ndim - 1) * (1,) + index_shape

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
    node: NodeProto, tensors_dict: dict, context: ggml.ggml_context_p, refs: List[Any]
):
    node_inputs = [tensors_dict[inp] for inp in node.input]

    if len(node_inputs) != 2:
        raise ValueError(
            f'Error for node "{node.name}": Operation "Greater" requires exactly two inputs. Actual number of inputs: {len(node_inputs)}'
        )

    a_shape = get_tensor_shape(node_inputs[0])
    a_dtype = get_tensor_dtype(node_inputs[0])
    b_shape = get_tensor_shape(node_inputs[1])

    output_shape = np.broadcast(np.empty(a_shape), np.empty(b_shape)).shape

    x = np.empty(output_shape, dtype=a_dtype)
    x_t = ggml.utils.from_numpy(x, context)

    new_tensor = tensors_dict[node.output[0]] = ggml.ggml_map_custom3_inplace(
        context,
        x_t,
        node_inputs[0],
        node_inputs[1],
        custom_greater,
        1,
        None,
    )

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
    node: NodeProto, tensors_dict: dict, context: ggml.ggml_context_p, refs: List[Any]
):
    node_inputs = [tensors_dict[inp] for inp in node.input]

    if len(node_inputs) != 2:
        raise ValueError(
            f'Error for node "{node.name}": Operation "Less" requires exactly two inputs. Actual number of inputs: {len(node_inputs)}'
        )

    a_shape = get_tensor_shape(node_inputs[0])
    a_dtype = get_tensor_dtype(node_inputs[0])
    b_shape = get_tensor_shape(node_inputs[1])

    output_shape = np.broadcast(np.empty(a_shape), np.empty(b_shape)).shape

    x = np.empty(output_shape, dtype=a_dtype)
    x_t = ggml.utils.from_numpy(x, context)

    new_tensor = tensors_dict[node.output[0]] = ggml.ggml_map_custom3_inplace(
        context,
        x_t,
        node_inputs[0],
        node_inputs[1],
        custom_less,
        1,
        None,
    )

    return new_tensor


@ggml_operator("Log")
def ggml_operator_log(
    node: NodeProto, tensors_dict: dict, context: ggml.ggml_context_p, refs: List[Any]
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


@ggml_operator("MatMul")
def ggml_operator_mat_mul(
    node: NodeProto, tensors_dict: dict, context: ggml.ggml_context_p, refs: List[Any]
):
    node_inputs = [tensors_dict[inp] for inp in node.input]

    if len(node_inputs) != 2:
        raise ValueError(
            f'Error for node "{node.name}": Operation "MatMul" requires exactly two inputs. Actual number of inputs: {len(node_inputs)}'
        )

    output_name = node.output[0]
    a = node_inputs[0]
    b = node_inputs[1]
    b_shape = get_tensor_shape(node_inputs[1])
    b_dtype = get_tensor_dtype(node_inputs[1])

    b_transposed = ggml.ggml_cpy(
        context,
        ggml.ggml_transpose(context, b),
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


@ggml.ggml_custom2_op_t
def custom_max(
    tensor_out: ggml.ggml_tensor_p,
    tensor_in_1: ggml.ggml_tensor_p,
    tensor_in_2: ggml.ggml_tensor_p,
    ith: int,
    nth: int,
    userdata: Optional[ctypes.c_void_p],
):
    a = ggml.utils.to_numpy(tensor_in_2)
    x = np.max(a)
    set_tensor_out(tensor_out, np.array(x))


@ggml_operator("Max")
def ggml_operator_max(
    node: NodeProto, tensors_dict: dict, context: ggml.ggml_context_p, refs: List[Any]
):
    node_inputs = [tensors_dict[inp] for inp in node.input]

    if len(node_inputs) != 1:
        raise ValueError(
            f'Error for node "{node.name}": Operation "Max" requires exactly two inputs. Actual number of inputs: {len(node_inputs)}'
        )

    a_dtype = get_tensor_dtype(node_inputs[0])

    output_shape = ()
    ggml_type = map_to_ggml_type(a_dtype)

    x_t = ggml.ggml_new_tensor(
        context,
        ggml_type.value,
        len(output_shape),
        (ctypes.c_int64 * len(output_shape))(*output_shape),
    )

    new_tensor = tensors_dict[node.output[0]] = ggml.ggml_map_custom2_inplace(
        context,
        x_t,
        node_inputs[0],
        custom_max,
        1,
        None,
    )

    return new_tensor


@ggml.ggml_custom2_op_t
def custom_min(
    tensor_out: ggml.ggml_tensor_p,
    tensor_in_1: ggml.ggml_tensor_p,
    tensor_in_2: ggml.ggml_tensor_p,
    ith: int,
    nth: int,
    userdata: Optional[ctypes.c_void_p],
):
    a = ggml.utils.to_numpy(tensor_in_2)
    x = np.min(a)
    set_tensor_out(tensor_out, np.array(x))


@ggml_operator("Min")
def ggml_operator_min(
    node: NodeProto, tensors_dict: dict, context: ggml.ggml_context_p, refs: List[Any]
):
    node_inputs = [tensors_dict[inp] for inp in node.input]

    if len(node_inputs) != 1:
        raise ValueError(
            f'Error for node "{node.name}": Operation "Min" requires exactly two inputs. Actual number of inputs: {len(node_inputs)}'
        )

    a_dtype = get_tensor_dtype(node_inputs[0])

    output_shape = ()
    ggml_type = map_to_ggml_type(a_dtype)

    x_t = ggml.ggml_new_tensor(
        context,
        ggml_type.value,
        len(output_shape),
        (ctypes.c_int64 * len(output_shape))(*output_shape),
    )

    new_tensor = tensors_dict[node.output[0]] = ggml.ggml_map_custom2_inplace(
        context,
        x_t,
        node_inputs[0],
        custom_min,
        1,
        None,
    )

    return new_tensor


@ggml_operator("Mul")
def ggml_operator_mul(
    node: NodeProto, tensors_dict: dict, context: ggml.ggml_context_p, refs: List[Any]
):
    node_inputs = [tensors_dict[inp] for inp in node.input]

    if len(node_inputs) != 2:
        raise ValueError(
            f'Error for node "{node.name}": Operation "Mul" requires exactly two inputs. Actual number of inputs: {len(node_inputs)}'
        )

    output_name = node.output[0]
    a = node_inputs[0]
    b = node_inputs[1]

    mul_result = ggml.ggml_mul(
        context,
        a,
        b,
    )

    tensors_dict[output_name] = mul_result
    return mul_result


@ggml_operator("Pow")
def ggml_operator_pow(
    node: NodeProto, tensors_dict: dict, context: ggml.ggml_context_p, refs: List[Any]
):
    raise NotImplementedError(f'Operator "Pow" not implemented')


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
    node: NodeProto, tensors_dict: dict, context: ggml.ggml_context_p, refs: List[Any]
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


class RedueMeanUserData(ctypes.Structure):
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
    userdata_data_ptr = ctypes.cast(userdata, ctypes.POINTER(RedueMeanUserData))
    userdata_data = userdata_data_ptr.contents

    tensor = ggml.utils.to_numpy(tensor_in_2)
    axes = [userdata_data.axes[i] for i in range(userdata_data.axes_length)]
    keepdims = userdata_data.keepdims

    rmean_result = np.mean(tensor, tuple(axes), keepdims=keepdims)

    set_tensor_out(tensor_out, rmean_result)


@ggml_operator("ReduceMean")
def ggml_operator_reduce_mean(
    node: NodeProto, tensors_dict: dict, context: ggml.ggml_context_p, refs: List[Any]
):
    node_inputs = [tensors_dict[inp] for inp in node.input]

    if len(node_inputs) != 1:
        raise ValueError(
            f'Error for node "{node.name}": Operation "ReduceMean" requires exactly one input. Actual number of inputs: {len(node_inputs)}'
        )

    tensor_shape = get_tensor_shape(node_inputs[0])
    tensor_dtype = get_tensor_dtype(node_inputs[0])
    axes = next(attr for attr in node.attribute if attr.name == "axes").ints
    keepdims = next(attr for attr in node.attribute if attr.name == "keepdims").i

    rmean_userdata = RedueMeanUserData(list(axes), keepdims)
    userdata_p = ctypes.cast(ctypes.pointer(rmean_userdata), ctypes.c_void_p)

    output_shape = list(tensor_shape)
    for axis in axes:
        output_shape[axis] = 1
    for axis in axes:
        if not keepdims:
            output_shape.pop(0)

    output_shape = tuple(output_shape)

    x = np.empty(output_shape, dtype=tensor_dtype)

    x_t = ggml.utils.from_numpy(x, context)

    new_tensor = tensors_dict[node.output[0]] = ggml.ggml_map_custom2_inplace(
        context,
        x_t,
        node_inputs[0],
        custom_reduce_mean,
        1,
        userdata_p,
    )

    refs.append(rmean_userdata)

    return new_tensor


@ggml_operator("Relu")
def ggml_operator_relu(
    node: NodeProto, tensors_dict: dict, context: ggml.ggml_context_p, refs: List[Any]
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
    node: NodeProto, tensors_dict: dict, context: ggml.ggml_context_p, refs: List[Any]
):
    node_inputs = [tensors_dict[inp] for inp in node.input]

    if len(node_inputs) != 2:
        raise ValueError(
            f'Error for node "{node.name}": Operation "Reshape" requires exactly two inputs. Actual number of inputs: {len(node_inputs)}'
        )

    output_name = node.output[0]
    a = node_inputs[0]
    b = node_inputs[1]

    b_numpy_reverse: list = ggml.utils.to_numpy(b).tolist()
    b_numpy_reverse.reverse()

    dims = len(b_numpy_reverse)

    if dims > 4:
        raise NotImplementedError(
            f'Operator "Reshape" not implemented for over 4D arrays.'
        )

    b_numpy_reverse += [0, 0, 0][:dims]

    ne0, ne1, ne2, ne3 = b_numpy_reverse

    dim_map = {
        1: (ggml.ggml_reshape_1d, (context, a, ne0)),
        2: (ggml.ggml_reshape_2d, (context, a, ne0, ne1)),
        3: (ggml.ggml_reshape_3d, (context, a, ne0, ne1, ne2)),
        4: (ggml.ggml_reshape_4d, (context, a, ne0, ne1, ne2, ne3)),
    }

    func = dim_map[dims][0]
    args = dim_map[dims][1]
    reshape_result = func(*args)
    tensors_dict[output_name] = reshape_result
    return reshape_result


class ShapeUserData(ctypes.Structure):
    _fields_ = [("start", ctypes.c_int), ("end", ctypes.c_int)]


@ggml.ggml_custom2_op_t
def custom_shape(
    tensor_out: ggml.ggml_tensor_p,
    tensor_in_1: ggml.ggml_tensor_p,
    tensor_in_2: ggml.ggml_tensor_p,
    ith: int,
    nth: int,
    userdata: Optional[ctypes.c_void_p],
):
    userdata_data_ptr = ctypes.cast(userdata, ctypes.POINTER(ShapeUserData))
    userdata_data = userdata_data_ptr.contents

    tensor = ggml.utils.to_numpy(tensor_in_2)
    start = userdata_data.start
    end = userdata_data.end

    shaped_tensor = tensor[start:end]
    tensor_shape = np.array(shaped_tensor.shape, dtype=np.int32)

    ggml.utils.to_numpy(tensor_out)[:] = tensor_shape


@ggml_operator("Shape")
def ggml_operator_shape(
    node: NodeProto, tensors_dict: dict, context: ggml.ggml_context_p, refs: List[Any]
):
    node_inputs = [tensors_dict[inp] for inp in node.input]

    if len(node_inputs) == 0 or len(node_inputs) > 3:
        raise ValueError(
            f'Error for node "{node.name}": Operation "Shape" requires at least 1 and maximum of 3 inputs. Actual number of inputs: {len(node_inputs)}'
        )

    tensor_shape = get_tensor_shape(node_inputs[0])
    tensor_dtype = get_tensor_dtype(node_inputs[0])
    start = (
        ggml.utils.to_numpy(node_inputs[1])
        if len(node_inputs) > 1
        else [ctypes.c_int(0)]
    )
    end = (
        ggml.utils.to_numpy(node_inputs[2])
        if len(node_inputs) > 2
        else [ctypes.c_int(tensor_shape[-1])]
    )

    start = start[0] if len(start) else ctypes.c_int(0)
    end = end[0] if len(end) else ctypes.c_int(tensor_shape[-1])

    shape_userdata = ShapeUserData(start, end)
    userdata_p = ctypes.cast(ctypes.pointer(shape_userdata), ctypes.c_void_p)

    output_shape = len(list(tensor_shape))

    x = np.empty(output_shape, dtype=tensor_dtype)

    x_t = ggml.utils.from_numpy(x, context)

    new_tensor = tensors_dict[node.output[0]] = ggml.ggml_map_custom2_inplace(
        context,
        x_t,
        node_inputs[0],
        custom_shape,
        1,
        userdata_p,
    )

    refs.append(shape_userdata)

    return new_tensor


@ggml_operator("Softmax")
def ggml_operator_softmax(
    node: NodeProto, tensors_dict: dict, context: ggml.ggml_context_p, refs: List[Any]
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
    node: NodeProto, tensors_dict: dict, context: ggml.ggml_context_p, refs: List[Any]
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
    node: NodeProto, tensors_dict: dict, context: ggml.ggml_context_p, refs: List[Any]
):
    node_inputs = [tensors_dict[inp] for inp in node.input]

    if len(node_inputs) != 2:
        raise ValueError(
            f'Error for node "{node.name}": Operation "Sub" requires exactly two inputs. Actual number of inputs: {len(node_inputs)}'
        )

    output_name = node.output[0]
    a = node_inputs[0]
    b = node_inputs[1]

    sub_result = ggml.ggml_sub(
        context,
        a,
        b,
    )
    tensors_dict[output_name] = sub_result
    return sub_result


@ggml_operator("Transpose")
def ggml_operator_transpose(
    node: NodeProto, tensors_dict: dict, context: ggml.ggml_context_p, refs: List[Any]
):
    node_inputs = [tensors_dict[inp] for inp in node.input]

    if len(node_inputs) != 1:
        raise ValueError(
            f'Error for node "{node.name}": Operation "Transpose" requires exactly one input. Actual number of inputs: {len(node_inputs)}'
        )

    output_name = node.output[0]
    input_shape = get_tensor_shape(node_inputs[0])
    perm_map = {1: [0, 1, 2, 3], 2: [1, 0, 2, 3], 3: [2, 1, 0, 3], 4: [3, 2, 1, 0]}

    perm_attr = next((attr for attr in node.attribute if attr.name == "perm"), None)

    if perm_attr is None:
        perm = perm_map.get(len(input_shape), [1, 0, 2, 3])
    else:
        perm = list(perm_attr.ints)
        perm += [0, 1, 2, 3][len(perm) :]

    ax0, ax1, ax2, ax3 = perm
    transpose_result = ggml.ggml_permute(context, node_inputs[0], ax0, ax1, ax2, ax3)
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

    for axis in np.nditer(axes):
        x = np.expand_dims(x, axis=axis)

    ggml.utils.to_numpy(tensor_out)[:] = x


@ggml_operator("Unsqueeze")
def ggml_operator_unsqueeze(
    node: NodeProto, tensors_dict: dict, context: ggml.ggml_context_p, refs: List[Any]
):
    node_inputs = [tensors_dict[inp] for inp in node.input]

    if len(node_inputs) != 2:
        raise ValueError(
            f'Error for node "{node.name}": Operation "Unsqueeze" requires exactly two inputs, data and axes. Actual number of inputs: {len(node_inputs)}'
        )

    x_shape = get_tensor_shape(node_inputs[0])
    x_dtype = get_tensor_dtype(node_inputs[0])
    axes = ggml.utils.to_numpy(node_inputs[1])

    for axis in np.nditer(axes):
        x_shape = np.insert(x_shape, axis, 1)

    x_shape = x_shape.astype(np.int32)

    x = np.empty(x_shape, dtype=x_dtype)
    x_t = ggml.utils.from_numpy(x, context)

    new_tensor = tensors_dict[node.output[0]] = ggml.ggml_map_custom3_inplace(
        context,
        x_t,
        node_inputs[0],
        node_inputs[1],
        custom_unsqueeze,
        1,
        None,
    )

    return new_tensor


@ggml_operator("Where")
def ggml_operator_where(
    node: NodeProto, tensors_dict: dict, context: ggml.ggml_context_p, refs: List[Any]
):
    raise NotImplementedError(f'Operator "Where" not implemented')


class GgmlBackendRep(BackendRep):
    def __init__(self):
        super(GgmlBackendRep, self).__init__()

    def __del__(self):
        if hasattr(self, "ggml_context"):
            ggml.ggml_free(self.ggml_context)

    def run(self, inputs: Any, **kwargs: Any) -> Tuple[Any, ...]:
        """Abstract function."""

        # check: data is should be on CPU

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

        # Build layers
        for node in model_graph.node:
            node_output = ggml_operators[node.op_type](
                node,
                ggml_tensors,
                context,
                refs,
            )

            if node.output[-1] == self.graph.output[-1].name:
                exit_node = node_output

        # Build graph
        gf = ggml.ggml_build_forward(exit_node)

        # Set user inputs
        for key, value in inputs.items():
            ggml.utils.to_numpy(ggml_tensors[key])[:] = value

        # Compute graph
        ggml.ggml_graph_compute_with_ctx(context, ctypes.pointer(gf), 1)

        graph_output = ggml.utils.to_numpy(exit_node)

        return [graph_output]


class GgmlRuntimeBackend(Backend):
    @classmethod
    def is_opset_supported(cls, model):  # pylint: disable=unused-argument
        return True, ""

    @classmethod
    def prepare(cls, model: ModelProto, device="CPU", **kwargs):
        """
        Load the model and creates a :class:`onnxruntime.InferenceSession`
        ready to be used as a backend.

        :param model: ModelProto (returned by `onnx.load`),
            string for a filename or bytes for a serialized model
        :param device: requested device for the computation,
            None means the default one which depends on
            the compilation settings
        :param kwargs: see :class:`onnxruntime.SessionOptions`
        :return: :class:`onnxruntime.InferenceSession`
        """

        super(GgmlRuntimeBackend, cls).prepare(model, device, **kwargs)
        ggml_rep = cls.onnx_model_to_ggml_rep(model, **kwargs)

        return ggml_rep

    @classmethod
    def onnx_model_to_ggml_rep(cls, model: ModelProto, **kwargs):
        """Convert ONNX model to GgmlRep.

        :param model: ONNX ModelProto object.
        and the converted tensorflow model.
        :return: GgmlRep object.
        """

        # Models with IR_VERSION less than 3 does not have opset_import set.
        # We default to minimum opset, this behavior is consistent with
        # onnx checker.
        # c.f. https://github.com/onnx/onnx/blob/427ac0c1b792363d373e3d7e4eef97fa46458420/onnx/checker.cc#L478
        if model.ir_version < 3:
            opset_import = [make_opsetid(defs.ONNX_DOMAIN, 1)]
        else:
            opset_import = model.opset_import

        return cls._onnx_graph_to_ggml_rep(model.graph, opset_import, **kwargs)

    @classmethod
    def _onnx_graph_to_ggml_rep(cls, graph_def: GraphProto, opset, **kwargs):
        ggml_backend_rep = GgmlBackendRep()
        ggml_backend_rep.graph = graph_def

        weights = {}

        n_tensors = len(graph_def.initializer)
        init_params = ggml.ggml_init_params(
            mem_size=n_tensors * ggml.ggml_tensor_overhead(),
            no_alloc=True,
        )

        context = ggml.ggml_init(init_params)
        ggml_backend_rep.ggml_context = context
        ggml_backend_rep.ggml_init_params = init_params
        total_nbytes = 0

        pairs = []

        for initializer in graph_def.initializer:
            name = initializer.name
            np_array = onnx.numpy_helper.to_array(initializer)
            tensor = ggml.utils.from_numpy(x=np_array, ctx=context)

            ggml.ggml_set_name(tensor=tensor, name=name.encode())
            total_nbytes += ggml.ggml_nbytes(tensor)
            weights[name] = tensor
            pairs.append((tensor, initializer))

        buffer = (ctypes.c_uint8 * total_nbytes)()
        offset = 0

        for tensor, initializer in pairs:
            nbytes = ggml.ggml_nbytes(tensor)
            tensor.contents.data = ctypes.cast(
                ctypes.addressof(buffer) + offset, ctypes.c_void_p
            )
            ggml.utils.to_numpy(tensor)[:] = onnx.numpy_helper.to_array(initializer)
            offset += nbytes

        ggml_backend_rep.ggml_buffer = buffer
        ggml_backend_rep.weights = weights
        ggml_backend_rep.inputs = graph_def.input
        ggml_backend_rep.outputs = graph_def.output

        return ggml_backend_rep

    @classmethod
    def run_model(cls, model, inputs, device=None, **kwargs):
        """
        Compute the prediction.

        :param model: :class:`onnxruntime.InferenceSession` returned
            by function *prepare*
        :param inputs: inputs
        :param device: requested device for the computation,
            None means the default one which depends on
            the compilation settings
        :param kwargs: see :class:`onnxruntime.RunOptions`
        :return: predictions
        """
        rep = cls.prepare(model, device, **kwargs)
        return rep.run(inputs, **kwargs)

    @classmethod
    def run_node(cls, node, inputs, device=None, outputs_info=None, **kwargs):
        """
        This method is not implemented as it is much more efficient
        to run a whole model than every node independently.
        """
        raise NotImplementedError(
            "It is much more efficient to run a whole model than every node independently."
        )
