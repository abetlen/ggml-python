"""Utility functions for ggml-python.
"""
import enum
import ctypes
import contextlib

from typing import Any, Callable, Tuple

from ggml import ggml

import numpy as np
import numpy.typing as npt


class GGML_TYPE(enum.Enum):
    F32 = ggml.GGML_TYPE_F32
    F16 = ggml.GGML_TYPE_F16
    Q4_0 = ggml.GGML_TYPE_Q4_0
    Q4_1 = ggml.GGML_TYPE_Q4_1
    Q5_0 = ggml.GGML_TYPE_Q5_0
    Q5_1 = ggml.GGML_TYPE_Q5_1
    Q8_0 = ggml.GGML_TYPE_Q8_0
    Q8_1 = ggml.GGML_TYPE_Q8_1
    I8 = ggml.GGML_TYPE_I8
    I16 = ggml.GGML_TYPE_I16
    I32 = ggml.GGML_TYPE_I32


NUMPY_DTYPE_TO_GGML_TYPE = {
    np.float16: GGML_TYPE.F16,
    np.float32: GGML_TYPE.F32,
    np.int8: GGML_TYPE.I8,
    np.int16: GGML_TYPE.I16,
    np.int32: GGML_TYPE.I32,
}

GGML_TYPE_TO_NUMPY_DTYPE = {v: k for k, v in NUMPY_DTYPE_TO_GGML_TYPE.items()}


def to_numpy(
    tensor: ggml.ggml_tensor_p,
) -> npt.NDArray[Any]:
    """Get the data of a ggml tensor as a numpy array.

    Parameters:
        tensor: ggml tensor

    Returns:
        Numpy array with a view of data from tensor
    """
    ggml_type = GGML_TYPE(tensor.contents.type)
    if ggml_type == GGML_TYPE.F16:
        ctypes_type = ctypes.c_uint16
    else:
        ctypes_type = np.ctypeslib.as_ctypes_type(GGML_TYPE_TO_NUMPY_DTYPE[ggml_type])

    array = ctypes.cast(ggml.ggml_get_data(tensor), ctypes.POINTER(ctypes_type))
    shape = tuple(reversed(tensor.contents.ne[: tensor.contents.n_dims]))
    output = np.ctypeslib.as_array(array, shape=shape)
    if ggml_type == GGML_TYPE.F16:
        output.dtype = np.float16
    return np.lib.stride_tricks.as_strided(
        output, strides=tuple(reversed(tensor.contents.nb[: tensor.contents.n_dims]))
    )


def from_numpy(x: npt.NDArray[Any], ctx: ggml.ggml_context_p) -> ggml.ggml_tensor_p:
    """Create a new ggml tensor with data copied from a numpy array.

    Parameters:
        x: numpy array
        ctx: ggml context

    Returns:
        New ggml tensor with data copied from x
    """
    ggml_type = NUMPY_DTYPE_TO_GGML_TYPE[x.dtype.type]
    shape = tuple(reversed(x.shape))
    tensor = ggml.ggml_new_tensor(
        ctx,
        ggml_type.value,
        len(shape),
        (ctypes.c_int64 * len(shape))(*shape),
    )
    tensor.contents.nb[: len(shape)] = (ctypes.c_int64 * len(shape))(
        *tuple(reversed(x.strides))
    )
    if tensor.contents.data is not None:
        to_numpy(tensor)[:] = x
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
        (contextlib.AbstractContextManager): ggml_context_p context manager
    """
    ctx = ggml.ggml_init(params)
    try:
        yield ctx
    finally:
        ggml.ggml_free(ctx)


def copy_to_cpu(
    ctx: ggml.ggml_context_p, tensor: ggml.ggml_tensor_p
) -> ggml.ggml_tensor_p:
    """Copy a ggml tensor from a GPU backend to CPU.

    Parameters:
        ctx: ggml context
        tensor: ggml tensor

    Returns:
        New ggml tensor with data copied from tensor on CPU backend"""
    tmp = ggml.ggml_dup_tensor(ctx, tensor)
    to_numpy(tmp)[:] = 0
    return ggml.ggml_add_inplace(ctx, tmp, tensor)
