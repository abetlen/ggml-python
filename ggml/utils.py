"""Utility functions for ggml-python.
"""
import enum
import ctypes
import contextlib

from typing import Any, List, Optional, Sequence, Tuple

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
    n_dims = ggml.ggml_n_dims(tensor)
    shape = tuple(reversed(tensor.contents.ne[:n_dims]))
    output = np.ctypeslib.as_array(array, shape=shape)
    if ggml_type == GGML_TYPE.F16:
        output.dtype = np.float16
    return np.lib.stride_tricks.as_strided(
        output, strides=tuple(reversed(tensor.contents.nb[:n_dims]))
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
    if ggml.ggml_get_data(tensor) is not None:
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


def quantize_0(
    data_f32: ggml.CFloatArray,
    nelements: int,
    ne0: int,
    ttype: GGML_TYPE,
    hist: Optional[ggml.CInt64Array] = None,
    work: Optional[ggml.CFloatArray] = None,
):
    """Quantize a float32 array.

    Parameters:
        data_f32: float32 array
        nelements: number of elements in data_f32
        ne0: number of elements in data_f32 that are zero
        ttype: ggml type to quantize to
        hist: histogram of quantized values
        work: work buffer

    Returns:
        (work, hist, cur_size): outpuut buffer, histogram, number of bytes in work buffer
    """
    work = work or (ctypes.c_float * nelements)()
    hist = hist or (ctypes.c_int64 * (1 << 4))()
    quantize_func = {
        GGML_TYPE.Q4_0: ggml.ggml_quantize_q4_0,
        GGML_TYPE.Q4_1: ggml.ggml_quantize_q4_1,
        GGML_TYPE.Q5_0: ggml.ggml_quantize_q5_0,
        GGML_TYPE.Q5_1: ggml.ggml_quantize_q5_1,
        GGML_TYPE.Q8_0: ggml.ggml_quantize_q8_0,
    }[ttype]
    cur_size = quantize_func(
        data_f32,
        ctypes.cast(work, ctypes.c_void_p),
        nelements,
        ne0,
        hist,
    )
    return ctypes.cast(work, ctypes.c_void_p), hist, cur_size


def quantize_row(
    data_f32: ggml.CFloatArray,
    nelements: int,
    ttype: GGML_TYPE,
    work: Optional[ctypes.c_void_p] = None,
):
    """Quantize a row of a ggml tensor.

    Parameters:
        data_f32: float32 array
        nelements: number of elements in data_f32
        ttype: ggml type to quantize to
        work: work buffer

    Returns:
        output buffer"""
    type_traits = ggml.ggml_internal_get_type_traits(ttype.value)
    from_float = type_traits.from_float
    work = work or ctypes.cast((ctypes.c_float * nelements)(), ctypes.c_void_p)
    from_float(data_f32, work, nelements)
    return work


def dequantize_row(
    data_q: ctypes.c_void_p,
    nelements: int,
    ttype: GGML_TYPE,
    work: Optional[ctypes.c_void_p] = None,
):
    """Dequantize a row of a ggml tensor.

    Parameters:
        data_q: quantized data
        nelements: number of elements in data_q
        ttype: ggml type to dequantize from
        work: work buffer

    Returns:
        output buffer"""
    type_traits = ggml.ggml_internal_get_type_traits(ttype.value)
    to_float = type_traits.to_float
    work = work or ctypes.cast((ctypes.c_float * nelements)(), ctypes.c_void_p)
    to_float(data_q, work, nelements)
    return work


def get_ndims(tensor: ggml.ggml_tensor_p) -> int:
    """Get the number of dimensions of a ggml tensor.

    Parameters:
        tensor: ggml tensor

    Returns:
        Number of dimensions of tensor
    """
    return ggml.ggml_n_dims(tensor)


def get_shape(tensor: ggml.ggml_tensor_p) -> Tuple[int, ...]:
    """Get the shape of a ggml tensor.

    Parameters:
        tensor: ggml tensor

    Returns:
        Shape of tensor
    """
    return tensor.contents.ne[: ggml.ggml_n_dims(tensor)]


def get_strides(tensor: ggml.ggml_tensor_p) -> Tuple[int, ...]:
    """Get the strides of a ggml tensor.

    Parameters:
        tensor: ggml tensor

    Returns:
        Strides of tensor
    """
    return tensor.contents.nb[: ggml.ggml_n_dims(tensor)]


def slice_tensor(
    ctx: ggml.ggml_context_p, tensor: ggml.ggml_tensor_p, indices: Sequence[slice]
):
    """Slice a ggml tensor along multiple dimensions.

    The slice is a view of the original tensor with the same number of dimensions.

    Parameters:
        ctx: ggml context
        tensor: ggml tensor
        indices: indices to slice along

    Returns:
        New ggml tensor slice view"""
    ndims = ggml.ggml_n_dims(tensor)

    # check that the number of dimensions match
    if len(indices) != ndims:
        raise ValueError(
            f"tensor has {ndims} dimensions but {len(indices)} indices were given"
        )

    # calculate slice
    start = tuple(idx.start or 0 for idx in indices)
    end = tuple(idx.stop or get_shape(tensor)[i] for i, idx in enumerate(indices))
    step = tuple(idx.step or 1 for idx in indices)

    # get the shape of the slice
    shape = tuple((end[i] - start[i] + step[i] - 1) // step[i] for i in range(ndims))

    # get the strides of the slice
    strides = tuple(get_strides(tensor)[i] * step[i] for i in range(ndims))

    # get the offset of the slice
    offset = sum(get_strides(tensor)[i] * start[i] for i in range(ndims))

    if ndims == 1:
        return ggml.ggml_view_1d(
            ctx,
            tensor,
            shape[0],
            offset,
        )
    elif ndims == 2:
        return ggml.ggml_view_2d(
            ctx,
            tensor,
            shape[0],
            shape[1],
            strides[1],
            offset,
        )
    elif ndims == 3:
        return ggml.ggml_view_3d(
            ctx,
            tensor,
            shape[0],
            shape[1],
            shape[2],
            strides[1],
            strides[2],
            offset,
        )
    elif ndims == 4:
        return ggml.ggml_view_4d(
            ctx,
            tensor,
            shape[0],
            shape[1],
            shape[2],
            shape[3],
            strides[1],
            strides[2],
            strides[3],
            offset,
        )
    else:
        raise NotImplementedError(
            f"ggml tensors with {ndims} dimensions are not supported"
        )


def alloc_graph_measure(
    graph: ggml.ggml_cgraph,
    alignment: int,
    alloc_tensors: Optional[List[ggml.ggml_tensor_p]] = None,
) -> int:
    """Returns the number of bytes required by a ggml_allocr allocator to allocate the tensors in the graph.

    NOTE: This implementation saves a copy of the current data pointers of all graph nodes and leafs and restores them
    after measuring the allocation size so that the graph can be re-used.

    Parameters:
        graph: ggml graph
        alignment: alignment of the allocation
        alloc_tensors: list of tensors to allocate individually using ggml_allocr_alloc

    Returns:
        Size of the required allocation buffer in bytes"""
    alloc_tensors = alloc_tensors or []
    leaf_data = [ggml.ggml_get_data(graph.leafs[i]) for i in range(graph.n_leafs)]
    node_data = [ggml.ggml_get_data(graph.nodes[i]) for i in range(graph.n_nodes)]
    alloc = ggml.ggml_allocr_new_measure(alignment)
    for tensor in alloc_tensors:
        ggml.ggml_allocr_alloc(alloc, tensor)
    alloc_size = (
        ggml.ggml_allocr_alloc_graph(alloc, ctypes.byref(graph)) + alignment  # type: ignore
    )
    ggml.ggml_allocr_free(alloc)
    for i in range(graph.n_leafs):
        graph.leafs[i].contents.data = leaf_data[i]
    for i in range(graph.n_nodes):
        graph.nodes[i].contents.data = node_data[i]
    return alloc_size
