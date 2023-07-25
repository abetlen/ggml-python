import onnx
from onnx.backend.base import Backend


class GgmlRuntimeBackend(Backend):
    @classmethod
    def prepare(cls, model, device=None, **kwargs):
        pass

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
