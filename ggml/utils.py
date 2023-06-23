"""Utility functions for ggml-python.
"""
import enum
import ctypes
import contextlib

from typing import Any

from ggml import ggml

import numpy as np
import numpy.typing as npt


class GGML_TYPE(enum.Enum):
    F32 = ggml.GGML_TYPE_F32.value
    F16 = ggml.GGML_TYPE_F16.value
    Q4_0 = ggml.GGML_TYPE_Q4_0.value
    Q4_1 = ggml.GGML_TYPE_Q4_1.value
    Q5_0 = ggml.GGML_TYPE_Q5_0.value
    Q5_1 = ggml.GGML_TYPE_Q5_1.value
    Q8_0 = ggml.GGML_TYPE_Q8_0.value
    Q8_1 = ggml.GGML_TYPE_Q8_1.value
    I8 = ggml.GGML_TYPE_I8.value
    I16 = ggml.GGML_TYPE_I16.value
    I32 = ggml.GGML_TYPE_I32.value


NUMPY_DTYPE_TO_GGML_TYPE = {
    np.float16: GGML_TYPE.F16,
    np.float32: GGML_TYPE.F32,
    np.int8: GGML_TYPE.I8,
    np.int16: GGML_TYPE.I16,
    np.int32: GGML_TYPE.I32,
}

GGML_TYPE_TO_NUMPY_DTYPE = {v: k for k, v in NUMPY_DTYPE_TO_GGML_TYPE.items()}


def to_numpy(
    tensor,  # type: ctypes._Pointer[ggml.ggml_tensor] # type: ignore
) -> npt.NDArray[Any]:
    """Get the data of a ggml tensor as a numpy array.

    Parameters:
        tensor (ggml_tensor_p): ggml tensor

    Returns:
        Numpy array with a view of data from tensor
    """
    ggml_type = GGML_TYPE(tensor.contents.type)
    ctypes_type = np.ctypeslib.as_ctypes_type(GGML_TYPE_TO_NUMPY_DTYPE[ggml_type])
    array = ctypes.cast(tensor.contents.data, ctypes.POINTER(ctypes_type))
    shape = tuple(reversed(tensor.contents.ne[: tensor.contents.n_dims]))
    return np.ctypeslib.as_array(array, shape=shape).T


def from_numpy(
    x: npt.NDArray[Any], ctx: ggml.ggml_context_p
):  # type: (...) -> ctypes._Pointer[ggml.ggml_tensor] # type: ignore
    """Create a new ggml tensor with data copied from a numpy array.

    Parameters:
        x: numpy array
        ctx: ggml context

    Returns:
        (ggml_tensor_p): New ggml tensor with data copied from x
    """
    ggml_type = NUMPY_DTYPE_TO_GGML_TYPE[x.dtype.type]
    ctypes_type = np.ctypeslib.as_ctypes_type(x.dtype)
    shape = x.shape
    tensor = ggml.ggml_new_tensor(
        ctx,
        ctypes.c_int(ggml_type.value),
        ctypes.c_int(len(shape)),
        (ctypes.c_int64 * len(shape))(*shape),
    )
    array = ctypes.cast(ggml.ggml_get_data(tensor), ctypes.POINTER(ctypes_type))
    arr = np.ctypeslib.as_array(array, shape=x.shape)
    arr[:] = x
    return tensor


@contextlib.contextmanager
def ggml_context_manager(params: ggml.ggml_init_params):
    """Creates a context manager for a new ggml context that free's it after use.

    Example:
        ```python
        import ggml
        from ggml.utils import ggml_context_manager

        params = ggml.ggml_init_params(mem_size=16 * 1024 * 1024)
        with ggml_context_manager(params) as ctx:
            # do stuff with ctx
        ```

    Parameters:
        params: context parameters

    Returns:
        ggml_context_p context manager
    """
    ctx = ggml.ggml_init(params)
    try:
        yield ctx
    finally:
        ggml.ggml_free(ctx)
