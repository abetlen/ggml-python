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
ggml_inputs = {}

onnx_ggml_dtype = {
    1: ggml.GGML_TYPE_F32,  # torch.float32
    2: ggml.GGML_FTYPE_UNKNOWN,  # torch.uint8
    3: ggml.GGML_TYPE_I8,  # torch.int8
    4: ggml.GGML_FTYPE_UNKNOWN,  # torch.uint16
    5: ggml.GGML_TYPE_I16,  # torch.int16
    6: ggml.GGML_TYPE_I32,  # torch.int32
    7: ggml.GGML_FTYPE_UNKNOWN,  # torch.int64
}


def ggml_operator(operator):
    def inner(func):
        ggml_operators[operator] = func
        return func

    return inner


def ggml_input_tensor(tensor_type):
    def inner(func):
        ggml_inputs[tensor_type] = func
        return func

    return inner


@ggml_operator("Add")
def ggml_operator_add(node: NodeProto, tensors_dict, context):
    node_inputs = [tensors_dict[inp] for inp in node.input]

    add_result = ggml.ggml_add(
        context,
        *node_inputs,
    )
    tensors_dict[node.output[0]] = add_result
    return add_result


@ggml_operator("Shape")
def ggml_operator_shape(node: NodeProto, tensors_dict, context):
    # raise NotImplementedError(f'Operator "Shape" not implemented')
    pass


@ggml_operator("Constant")
def ggml_operator_constant(node: NodeProto, tensors_dict, context):
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

    constant_tensor_data = np.array(
        struct.unpack(
            f"={len(raw_data)//struct.calcsize(data_type_to_struct_format[data_type][0])}{data_type_to_struct_format[data_type][0]}",
            raw_data,
        ),
        dtype=data_type_to_struct_format[data_type][0],
    )

    tensors_dict[node.output[0]] = constant_tensor_data
    return constant_tensor_data


# ------ Operators ------


@ggml_operator("Mul")
def ggml_operator_mul(node: NodeProto, tensors_dict, context):
    node_inputs = [tensors_dict[inp] for inp in node.input]

    mul_result = ggml.ggml_mul(
        context,
        *node_inputs,
    )
    tensors_dict[node.output[0]] = mul_result
    return mul_result


@ggml_operator("ConstantOfShape")
def ggml_operator_constant_of_shape(node: NodeProto, tensors_dict, context):
    raise NotImplementedError(f'Operator "ConstantOfShape" not implemented')


@ggml_operator("Softmax")
def ggml_operator_softmax(node: NodeProto, tensors_dict, context):
    raise NotImplementedError(f'Operator "Softmax" not implemented')


@ggml_operator("Gather")
def ggml_operator_gather(node: NodeProto, tensors_dict, context):
    raise NotImplementedError(f'Operator "Gather" not implemented')


@ggml_operator("Relu")
def ggml_operator_relu(node: NodeProto, tensors_dict, context):
    raise NotImplementedError(f'Operator "Relu" not implemented')


@ggml_operator("MatMul")
def ggml_operator_mat_mul(node: NodeProto, tensors_dict, context):
    raise NotImplementedError(f'Operator "MatMul" not implemented')


@ggml_operator("Abs")
def ggml_operator_abs(node: NodeProto, tensors_dict, context):
    raise NotImplementedError(f'Operator "Abs" not implemented')


@ggml_operator("Unsqueeze")
def ggml_operator_unsqueeze(node: NodeProto, tensors_dict, context):
    raise NotImplementedError(f'Operator "Unsqueeze" not implemented')


@ggml_operator("Sqrt")
def ggml_operator_sqrt(node: NodeProto, tensors_dict, context):
    raise NotImplementedError(f'Operator "Sqrt" not implemented')


@ggml_operator("ReduceMean")
def ggml_operator_reduce_mean(node: NodeProto, tensors_dict, context):
    raise NotImplementedError(f'Operator "ReduceMean" not implemented')


@ggml_operator("Less")
def ggml_operator_less(node: NodeProto, tensors_dict, context):
    raise NotImplementedError(f'Operator "Less" not implemented')


@ggml_operator("Where")
def ggml_operator_where(node: NodeProto, tensors_dict, context):
    raise NotImplementedError(f'Operator "Where" not implemented')


@ggml_operator("Concat")
def ggml_operator_concat(node: NodeProto, tensors_dict, context):
    raise NotImplementedError(f'Operator "Concat" not implemented')


@ggml_operator("Div")
def ggml_operator_div(node: NodeProto, tensors_dict, context):
    raise NotImplementedError(f'Operator "Div" not implemented')


@ggml_operator("Range")
def ggml_operator_range(node: NodeProto, tensors_dict, context):
    raise NotImplementedError(f'Operator "Range" not implemented')


@ggml_operator("Sub")
def ggml_operator_sub(node: NodeProto, tensors_dict, context):
    raise NotImplementedError(f'Operator "Sub" not implemented')


@ggml_operator("Pow")
def ggml_operator_pow(node: NodeProto, tensors_dict, context):
    raise NotImplementedError(f'Operator "Pow" not implemented')


@ggml_operator("Cast")
def ggml_operator_cast(node: NodeProto, tensors_dict, context):
    raise NotImplementedError(f'Operator "Cast" not implemented')


@ggml_operator("Reshape")
def ggml_operator_reshape(node: NodeProto, tensors_dict, context):
    raise NotImplementedError(f'Operator "Reshape" not implemented')


@ggml_operator("Transpose")
def ggml_operator_transpose(node: NodeProto, tensors_dict, context):
    raise NotImplementedError(f'Operator "Transpose" not implemented')


@ggml_operator("Log")
def ggml_operator_log(node: NodeProto, tensors_dict, context):
    raise NotImplementedError(f'Operator "Log" not implemented')


@ggml_operator("Greater")
def ggml_operator_greater(node: NodeProto, tensors_dict, context):
    raise NotImplementedError(f'Operator "Greater" not implemented')


@ggml_operator("Min")
def ggml_operator_min(node: NodeProto, tensors_dict, context):
    raise NotImplementedError(f'Operator "Min" not implemented')


## ------- Inputs --------
@ggml_input_tensor("1")
def ggml_input_1d(node: NodeProto, tensors_dict, context):
    ggml_type = node.type.tensor_type.elem_type

    inp = ggml.ggml_new_tensor_1d(
        context,
        onnx_ggml_dtype[ggml_type],
        1,
    )
    tensors_dict[node.name] = inp


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

        # handle types same as operators
        tensor_types = {1: ggml.ggml_new_tensor_1d, 2: ggml.ggml_new_tensor_2d}

        # Define context
        params = ggml.ggml_init_params(mem_size=16 * 1024 * 1024, mem_buffer=None)
        context = ggml.ggml_init(params=params)

        # Create entry inputs
        for model_input in model_graph.input:
            shape_dim_value = [
                dim.dim_value
                for dim in model_input.type.tensor_type.shape.dim
                if dim.dim_value > 0
            ][-1]
            ggml_inputs[str(shape_dim_value)](model_input, ggml_tensors, context)

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
