from typing import Any, Tuple

from onnx import defs
from onnx.backend.base import Backend, BackendRep
from onnx.helper import make_opsetid
from onnx.onnx_ml_pb2 import GraphProto


class GgmlBackendRep(BackendRep):
    def __init__(self, graph=None, inputs=None, outputs=None, tensor_dict=None):
        super(GgmlRuntimeBackend, self).__init__()
        self._graph = graph
        self._inputs = inputs or {}
        self._outputs = outputs or {}
        self._tensor_dict = tensor_dict or {}

    @property
    def graph(self):
        return self._graph

    @graph.setter
    def graph(self, graph):
        self._graph = graph

    @property
    def inputs(self):
        return self._inputs

    @inputs.setter
    def inputs(self, inputs):
        self._inputs = inputs

    @property
    def outputs(self):
        return self._outputs

    @outputs.setter
    def outputs(self, outputs):
        self._outputs = outputs

    @property
    def tensor_dict(self):
        return self._tensor_dict

    @tensor_dict.setter
    def tensor_dict(self, tensor_dict):
        self._tensor_dict = tensor_dict

    def run(self, inputs: Any, **kwargs: Any) -> Tuple[Any, ...]:
        """Abstract function."""

        # check where data is should be on CPU
        return (None,)


class GgmlRuntimeBackend(Backend):
    @classmethod
    def is_opset_supported(cls, model):  # pylint: disable=unused-argument
        return True, ""

    @classmethod
    def prepare(cls, model, device="CPU", **kwargs):
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
    def onnx_model_to_ggml_rep(cls, model, **kwargs):
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
        inputs = {}
        outputs = {}
        weights = {}

        for node in graph_def.node:
            inputs[node.name] = list(node.input)
            outputs[node.name] = list(node.output)

        for initializer in graph_def.initializer:
            weights[initializer.name] = initializer.raw_data

        return GgmlBackendRep(
            graph_def, inputs=inputs, outputs=outputs, tensor_dict=weights
        )

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
