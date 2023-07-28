import ctypes
from typing import Any, Tuple

import onnx
from onnx import defs
from onnx.backend.base import Backend, BackendRep
from onnx.helper import make_opsetid
from onnx.onnx_ml_pb2 import GraphProto, ModelProto

import ggml
import ggml.utils


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

        tensor_types = {1: ggml.ggml_new_tensor_1d, 2: ggml.ggml_new_tensor_2d}
        operation_types = {"Mul": ggml.ggml_mul, "Add": ggml.ggml_add}

        # Define context
        params = ggml.ggml_init_params(mem_size=16 * 1024 * 1024, mem_buffer=None)
        ctx = ggml.ggml_init(params=params)

        # Create entry inputs
        for model_input in model_graph.input:
            inp = ggml.ggml_new_tensor_1d(
                ctx,
                ggml.GGML_TYPE_F32,
                1,
            )
            ggml_tensors[model_input.name] = inp

        # Build layers
        for node in model_graph.node:
            node_inputs = [ggml_tensors[inp] for inp in node.input]
            layer = operation_types[node.op_type](
                ctx,
                *node_inputs,
            )
            ggml_tensors[node.output[0]] = layer
            if node.output[-1] == self.graph.output[-1].name:
                exit_node = layer

        # Build graph
        gf = ggml.ggml_build_forward(exit_node)

        # Set user inputs
        for key, value in inputs.items():
            ggml.ggml_set_f32(ggml_tensors[key], value)

        # Compute graph
        ggml.ggml_graph_compute_with_ctx(ctx, ctypes.pointer(gf), 1)

        output = ggml.utils.to_numpy(exit_node)

        return [output]


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
