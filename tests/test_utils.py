import ggml
import ggml.utils

import pytest

import numpy as np


def test_utils():
    params = ggml.ggml_init_params(mem_size=16 * 1024 * 1024)
    ctx = ggml.ggml_init(params)
    assert ctx is not None
    x = np.ones((3,), dtype=np.float32)
    assert x.shape == (3,)
    t = ggml.utils.from_numpy(x, ctx)
    assert t.contents.ne[:1] == [3]
    assert t.contents.type == ggml.GGML_TYPE_F32
    assert np.allclose(ggml.utils.to_numpy(t), x)
    ggml.ggml_free(ctx)


def test_numpy_arrays():
    params = ggml.ggml_init_params(mem_size=16 * 1024 * 1024)
    ctx = ggml.ggml_init(params)
    assert ctx is not None
    x = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32, order="F")
    assert x.shape == (2, 3)
    t = ggml.utils.from_numpy(x, ctx)
    assert t.contents.ne[:2] == [3, 2]
    y = ggml.utils.to_numpy(t)
    assert y.shape == (2, 3)
    ggml.ggml_free(ctx)


def test_numpy_arrays_transposed():
    params = ggml.ggml_init_params(mem_size=16 * 1024 * 1024)
    ctx = ggml.ggml_init(params)
    assert ctx is not None
    x = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.int32)
    t = ggml.utils.from_numpy(x, ctx)

    t_T = ggml.ggml_transpose(ctx, t)

    # ggml_transpose currently modifies the original tensor in place, input must be
    # set _after_ the transpose operation
    ggml.utils.to_numpy(t)[:] = x

    assert ggml.utils.get_shape(t_T) == (2, 3)
    assert ggml.utils.get_strides(t_T) == (12, 4)

    assert np.array_equal(ggml.utils.to_numpy(t_T, shape=x.T.shape), x.T)

    ggml.ggml_free(ctx)


def test_numpy_arrays_permute_transpose():
    params = ggml.ggml_init_params(mem_size=16 * 1024 * 1024)
    ctx = ggml.ggml_init(params)
    assert ctx is not None

    x = np.array(
        [[[1, 2], [3, 4], [5, 6]], [[7, 8], [9, 10], [11, 12]]], dtype=np.int32
    )
    t = ggml.utils.from_numpy(x, ctx)

    t_T = ggml.ggml_permute(ctx, t, 2, 1, 0, 3)

    ggml.utils.to_numpy(t)[:] = x

    x_T = ggml.utils.to_numpy(t_T)
    assert np.array_equal(x_T, x.T)

    ggml.ggml_free(ctx)


def test_slice_tensor():
    params = ggml.ggml_init_params(mem_size=16 * 1024 * 1024)
    ctx = ggml.ggml_init(params)
    assert ctx is not None
    x = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.int32)
    t = ggml.utils.from_numpy(x, ctx)
    t_slice = ggml.utils.slice_tensor(ctx, t, [
        slice(0, 1),
        slice(0, 2)
    ])
    x_slice = x[:2, :1]
    t_slice_array = ggml.utils.to_numpy(t_slice)
    assert np.array_equal(t_slice_array, x_slice)
    ggml.ggml_free(ctx)
