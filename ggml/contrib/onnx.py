import ctypes
import struct
from typing import Any, Tuple

import numpy as np
import onnx
from onnx import defs
from onnx.backend.base import Backend, BackendRep
from onnx.helper import make_opsetid
from onnx.onnx_ml_pb2 import GraphProto, ModelProto, NodeProto

import ggml
import ggml.utils
import torch


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


@ggml_operator("Add")
def ggml_operator_add(
    node: NodeProto, tensors_dict: dict, context: ggml.ggml_context_p
):
    node_inputs = [tensors_dict[inp] for inp in node.input]

    if len(node_inputs) != 2:
        raise ValueError(
            f'Error for node "{node.name}": Operation "Add" requires exactly two inputs. Actual number of inputs: {len(node_inputs)}'
        )

    output_name = node.output[0]

    add_result = ggml.ggml_add(
        context,
        *node_inputs,
    )
    tensors_dict[output_name] = add_result
    return add_result


@ggml_operator("Shape")
def ggml_operator_shape(
    node: NodeProto, tensors_dict: dict, context: ggml.ggml_context_p
):
    node_inputs = [tensors_dict[inp] for inp in node.input]

    if len(node_inputs) == 0 or len(node_inputs) > 3:
        raise ValueError(
            f'Error for node "{node.name}": Operation "Shape" requires at least 1 and maximum of 3 inputs. Actual number of inputs: {len(node_inputs)}'
        )

    output_name = node.output[0]

    tensor = ggml.utils.to_numpy(node_inputs[0])
    start = ggml.utils.to_numpy(node_inputs[1]) if len(node_inputs) > 1 else [None]
    end = ggml.utils.to_numpy(node_inputs[2]) if len(node_inputs) > 2 else [None]

    start = start[0] if len(start) > 0 else None
    end = end[0] if len(end) > 0 else None

    og_dtype = tensor.dtype

    shaped_tensor = tensor[start:end]

    # clamp the rank to two
    shaped_tensor = np.array([shaped_tensor], dtype=og_dtype)
    shaped_tensor = np.reshape(shaped_tensor, [1, -1])

    new_tensor = ggml.utils.from_numpy(shaped_tensor, context)
    tensors_dict[output_name] = new_tensor

    return new_tensor


@ggml_operator("Constant")
def ggml_operator_constant(
    node: NodeProto, tensors_dict: dict, context: ggml.ggml_context_p
):
    data_type_to_struct_format = {
        1: "f",  # FLOAT (4 bytes)
        2: "b",  # INT8 (1 byte)
        3: "h",  # INT16 (2 bytes)
        4: "i",  # INT32 (4 bytes)
        5: "q",  # INT64 (8 bytes)
        6: "B",  # UINT8 (1 byte)
        7: "Q",  # UINT64 (8 bytes)
        10: "e",  # FLOAT16 (half-precision floating-point) (2 bytes)
        11: "d",  # DOUBLE (8 bytes)
    }

    node_attributes = node.attribute
    raw_data = node_attributes[0].t.raw_data
    data_type = node_attributes[0].t.data_type
    output_name = node.output[0]

    constant_tensor_data = np.array(
        struct.unpack(
            f"={len(raw_data)//struct.calcsize(data_type_to_struct_format[data_type][0])}{data_type_to_struct_format[data_type][0]}",
            raw_data,
        ),
        dtype=data_type_to_struct_format[data_type][0],
    )

    # clamp the rank to two
    constant_tensor_data = np.reshape(constant_tensor_data, [1, -1])

    tensors_dict[output_name] = constant_tensor_data
    return constant_tensor_data


# ------ Operators ------


@ggml_operator("Mul")
def ggml_operator_mul(
    node: NodeProto, tensors_dict: dict, context: ggml.ggml_context_p
):
    node_inputs = [tensors_dict[inp] for inp in node.input]

    if len(node_inputs) != 2:
        raise ValueError(
            f'Error for node "{node.name}": Operation "Mul" requires exactly two inputs. Actual number of inputs: {len(node_inputs)}'
        )

    output_name = node.output[0]

    mul_result = ggml.ggml_mul(
        context,
        *node_inputs,
    )

    tensors_dict[output_name] = mul_result

    return mul_result


@ggml_operator("ConstantOfShape")
def ggml_operator_constant_of_shape(
    node: NodeProto, tensors_dict: dict, context: ggml.ggml_context_p
):
    raise NotImplementedError(f'Operator "ConstantOfShape" not implemented')


@ggml_operator("Softmax")
def ggml_operator_softmax(
    node: NodeProto, tensors_dict: dict, context: ggml.ggml_context_p
):
    raise NotImplementedError(f'Operator "Softmax" not implemented')


@ggml_operator("Gather")
def ggml_operator_gather(
    node: NodeProto, tensors_dict: dict, context: ggml.ggml_context_p
):
    ## For now only handles axis = 0 TODO: add axis=1 case
    node_inputs = [tensors_dict[inp] for inp in node.input]

    if len(node_inputs) != 2:
        raise ValueError(
            f'Error for node "{node.name}": Operation "Gather" requires exactly two inputs. Actual number of inputs: {len(node_inputs)}'
        )

    input_array = (
        ggml.utils.to_numpy(node_inputs[0])
        if type(node_inputs[0])
        != np.ndarray  # better to check if its ggml.ggml_tensor_p but it doesnt work like that TODO: make type(tensor) work with ggml.ggml_tensor_p
        else node_inputs[0]
    )
    index_array = (
        ggml.utils.to_numpy(node_inputs[1])
        if type(node_inputs[1]) != np.ndarray
        else node_inputs[1]
    )

    og_dtype = input_array.dtype
    new_array = np.take(input_array, index_array.astype(og_dtype), axis=-1)

    # clamp the rank to two
    new_array = np.array([new_array], dtype=og_dtype)
    new_array = np.reshape(new_array, [1, -1])

    new_tensor = ggml.utils.from_numpy(new_array, context)
    tensors_dict[node.output[0]] = new_tensor

    return new_tensor


@ggml_operator("Relu")
def ggml_operator_relu(
    node: NodeProto, tensors_dict: dict, context: ggml.ggml_context_p
):
    raise NotImplementedError(f'Operator "Relu" not implemented')


@ggml_operator("MatMul")
def ggml_operator_mat_mul(
    node: NodeProto, tensors_dict: dict, context: ggml.ggml_context_p
):
    raise NotImplementedError(f'Operator "MatMul" not implemented')


@ggml_operator("Abs")
def ggml_operator_abs(
    node: NodeProto, tensors_dict: dict, context: ggml.ggml_context_p
):
    raise NotImplementedError(f'Operator "Abs" not implemented')


@ggml_operator("Unsqueeze")
def ggml_operator_unsqueeze(
    node: NodeProto, tensors_dict: dict, context: ggml.ggml_context_p
):
    node_inputs = [tensors_dict[inp] for inp in node.input]

    if len(node_inputs) != 2:
        raise ValueError(
            f'Error for node "{node.name}": Operation "Unsqueeze" requires exactly two inputs, data and axes. Actual number of inputs: {len(node_inputs)}'
        )

    x = (
        ggml.utils.to_numpy(node_inputs[0])
        if type(node_inputs[0]) != np.ndarray
        else node_inputs[0]
    )
    axes = (
        ggml.utils.to_numpy(node_inputs[1])
        if type(node_inputs[1]) != np.ndarray
        else node_inputs[1]
    )

    og_dtype = x.dtype

    for axis in np.nditer(axes):
        x = np.expand_dims(x, axis=axis)

    # clamp the rank to 3
    x = np.array([x], dtype=og_dtype)
    # x = np.reshape(x, (1, 1, -1))

    new_tensor = ggml.utils.from_numpy(x, context)
    tensors_dict[node.output[0]] = new_tensor

    return new_tensor


@ggml_operator("Sqrt")
def ggml_operator_sqrt(
    node: NodeProto, tensors_dict: dict, context: ggml.ggml_context_p
):
    raise NotImplementedError(f'Operator "Sqrt" not implemented')


@ggml_operator("ReduceMean")
def ggml_operator_reduce_mean(
    node: NodeProto, tensors_dict: dict, context: ggml.ggml_context_p
):
    raise NotImplementedError(f'Operator "ReduceMean" not implemented')


@ggml_operator("Less")
def ggml_operator_less(
    node: NodeProto, tensors_dict: dict, context: ggml.ggml_context_p
):
    raise NotImplementedError(f'Operator "Less" not implemented')


@ggml_operator("Where")
def ggml_operator_where(
    node: NodeProto, tensors_dict: dict, context: ggml.ggml_context_p
):
    raise NotImplementedError(f'Operator "Where" not implemented')


@ggml_operator("Concat")
def ggml_operator_concat(
    node: NodeProto, tensors_dict: dict, context: ggml.ggml_context_p
):
    raise NotImplementedError(f'Operator "Concat" not implemented')


@ggml_operator("Div")
def ggml_operator_div(
    node: NodeProto, tensors_dict: dict, context: ggml.ggml_context_p
):
    raise NotImplementedError(f'Operator "Div" not implemented')


@ggml_operator("Range")
def ggml_operator_range(
    node: NodeProto, tensors_dict: dict, context: ggml.ggml_context_p
):
    raise NotImplementedError(f'Operator "Range" not implemented')


@ggml_operator("Sub")
def ggml_operator_sub(
    node: NodeProto, tensors_dict: dict, context: ggml.ggml_context_p
):
    raise NotImplementedError(f'Operator "Sub" not implemented')


@ggml_operator("Pow")
def ggml_operator_pow(
    node: NodeProto, tensors_dict: dict, context: ggml.ggml_context_p
):
    raise NotImplementedError(f'Operator "Pow" not implemented')


@ggml_operator("Cast")
def ggml_operator_cast(
    node: NodeProto, tensors_dict: dict, context: ggml.ggml_context_p
):
    raise NotImplementedError(f'Operator "Cast" not implemented')


@ggml_operator("Reshape")
def ggml_operator_reshape(
    node: NodeProto, tensors_dict: dict, context: ggml.ggml_context_p
):
    raise NotImplementedError(f'Operator "Reshape" not implemented')


@ggml_operator("Transpose")
def ggml_operator_transpose(
    node: NodeProto, tensors_dict: dict, context: ggml.ggml_context_p
):
    raise NotImplementedError(f'Operator "Transpose" not implemented')


@ggml_operator("Log")
def ggml_operator_log(
    node: NodeProto, tensors_dict: dict, context: ggml.ggml_context_p
):
    raise NotImplementedError(f'Operator "Log" not implemented')


@ggml_operator("Greater")
def ggml_operator_greater(
    node: NodeProto, tensors_dict: dict, context: ggml.ggml_context_p
):
    raise NotImplementedError(f'Operator "Greater" not implemented')


@ggml_operator("Min")
def ggml_operator_min(
    node: NodeProto, tensors_dict: dict, context: ggml.ggml_context_p
):
    raise NotImplementedError(f'Operator "Min" not implemented')


class GgmlBackendRep(BackendRep):
    def __init__(self):
        super(GgmlBackendRep, self).__init__()

    def __del__(self):
        if hasattr(self, "ggml_context"):
            ggml.ggml_free(self.ggml_context)

    def run(self, inputs: Any, **kwargs: Any) -> Tuple[Any, ...]:
        """Abstract function."""

        # check where data is should be on CPU

        model_graph = self.graph
        exit_node = None
        ggml_tensors = self.weights

        # Define context
        params = ggml.ggml_init_params(mem_size=16 * 1024 * 1024, mem_buffer=None)
        context = ggml.ggml_init(params=params)

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
            ggml_type = ggml.utils.NUMPY_DTYPE_TO_GGML_TYPE.get(
                input_data.dtype.type,
                ggml.utils.GGML_TYPE.I32,  # TODO: Add i64 but for now, use i32 if looking for i64 or f64
            )
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
            node_output = ggml_operators[node.op_type](node, ggml_tensors, context)

            if node.output[-1] == self.graph.output[-1].name:
                exit_node = node_output

        # Build graph
        gf = ggml.ggml_build_forward(exit_node)

        # Set user inputs
        for key, value in inputs.items():
            ggml.ggml_set_f32(ggml_tensors[key], value)

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
