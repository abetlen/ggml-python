import ctypes

import ggml
import ggml.utils

import pytest

import numpy as np


def test_utils():
    params = ggml.ggml_init_params(mem_size=16 * 1024 * 1024)
    with ggml.utils.ggml_context_manager(params) as ctx:
        x = np.ones((3,), dtype=np.float32)
        assert x.shape == (3,)
        t = ggml.utils.from_numpy(x, ctx)
        assert t.contents.ne[:1] == [3]
        assert t.contents.type == ggml.GGML_TYPE_F32
        assert np.allclose(ggml.utils.to_numpy(t), x)


def test_numpy_arrays():
    params = ggml.ggml_init_params(mem_size=16 * 1024 * 1024)
    with ggml.utils.ggml_context_manager(params) as ctx:
        x = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32, order="F")
        assert x.shape == (2, 3)
        t = ggml.utils.from_numpy(x, ctx)
        assert t.contents.ne[:2] == [3, 2]
        y = ggml.utils.to_numpy(t)
        assert y.shape == (2, 3)


def test_numpy_arrays_transposed():
    params = ggml.ggml_init_params(mem_size=16 * 1024 * 1024)
    with ggml.utils.ggml_context_manager(params) as ctx:
        # 2D
        x = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.int32)
        t = ggml.utils.from_numpy(x, ctx)
        t_t = ggml.ggml_transpose(ctx, t)
        x_t = ggml.utils.to_numpy(t_t)
        assert np.array_equal(x_t, x.T)

        t = ggml.utils.from_numpy(x.T, ctx)
        x_t = ggml.utils.to_numpy(t)
        assert np.array_equal(x.T, x_t)

        # 3D
        x = np.array(
            [[[1, 2], [3, 4], [5, 6]], [[7, 8], [9, 10], [11, 12]]], dtype=np.int32
        )
        t = ggml.utils.from_numpy(x, ctx)
        t_t = ggml.ggml_permute(ctx, t, 2, 1, 0, 3)
        x_t = ggml.utils.to_numpy(t_t)
        assert np.array_equal(x_t, x.T)

        t = ggml.utils.from_numpy(x.T, ctx)
        x_t = ggml.utils.to_numpy(t)
        assert np.array_equal(x.T, x_t)


def test_slice_tensor():
    params = ggml.ggml_init_params(mem_size=16 * 1024 * 1024)
    with ggml.utils.ggml_context_manager(params) as ctx:
        x = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.int32)
        t = ggml.utils.from_numpy(x, ctx)
        t_slice = ggml.utils.slice_tensor(ctx, t, [
            slice(0, 2),
            slice(0, 1)
        ])
        x_slice = ggml.utils.to_numpy(t_slice)
        assert np.array_equal(x_slice, x[:1, :2].squeeze())


def test_alloc_graph_measure():
    max_overhead = ggml.ggml_tensor_overhead() * ggml.GGML_DEFAULT_GRAPH_SIZE  + ggml.ggml_graph_overhead()
    assert max_overhead < 16 * 1024 * 1024  # 16MB
    params = ggml.ggml_init_params(
        mem_size=max_overhead, mem_buffer=None, no_alloc=True
    )
    ctx = ggml.ggml_init(params=params)

    # define the graph
    x = ggml.ggml_new_tensor_1d(ctx, ggml.GGML_TYPE_F32, 1)
    a = ggml.ggml_new_tensor_1d(ctx, ggml.GGML_TYPE_F32, 1)
    b = ggml.ggml_new_tensor_1d(ctx, ggml.GGML_TYPE_F32, 1)

    x2 = ggml.ggml_mul(ctx, x, x)
    tmp = ggml.ggml_mul(ctx, a, x2)

    # outputs
    f = ggml.ggml_add(ctx, tmp, b)

    # build graph
    gf = ggml.ggml_new_graph(ctx)
    ggml.ggml_build_forward_expand(gf, f)

    # create measure allocator
    tensor_alignment = 32
    input_tensors = [x, a, b]
    alloc_size = ggml.utils.alloc_graph_measure(gf.contents, tensor_alignment, input_tensors)

    # allocate tensor memory
    buffer = (ctypes.c_uint8 * alloc_size)()
    alloc = ggml.ggml_allocr_new(
        ctypes.cast(buffer, ctypes.c_void_p), alloc_size, tensor_alignment
    )
    ggml.ggml_allocr_alloc(alloc, x)
    ggml.ggml_allocr_alloc(alloc, a)
    ggml.ggml_allocr_alloc(alloc, b)
    ggml.ggml_allocr_alloc_graph(alloc, gf)

    # set input values
    ggml.ggml_set_f32(x, 2.0)
    ggml.ggml_set_f32(a, 3.0)
    ggml.ggml_set_f32(b, 4.0)

    gp = ggml.ggml_graph_plan(gf, 1)
    assert gp.work_size == 0

    # compute
    ggml.ggml_graph_compute(gf, ctypes.pointer(gp))

    output = ggml.ggml_get_f32_1d(f, 0)
    assert output == 16.0

    ggml.ggml_free(ctx)
    ggml.ggml_allocr_free(alloc)