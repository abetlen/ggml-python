"""Simple example of graph offloading to a non-cpu backend."""

import ggml
import ctypes

import numpy as np

def test_ggml_backend():
    def get_backend():
        return ggml.ggml_backend_cpu_init()

    n_tensors = 1 + 2 # input (x) and weights (a, b)
    params = ggml.ggml_init_params(
        mem_size=ggml.ggml_tensor_overhead() * n_tensors, mem_buffer=None, no_alloc=True
    )
    ctx = ggml.ggml_init(params=params)
    assert ctx is not None

    backend = get_backend()

    assert backend is not None

    # create the tensors for input and weights
    x = ggml.ggml_new_tensor_1d(ctx, ggml.GGML_TYPE_F32, 1)

    a = ggml.ggml_new_tensor_1d(ctx, ggml.GGML_TYPE_F32, 1)
    b = ggml.ggml_new_tensor_1d(ctx, ggml.GGML_TYPE_F32, 1)

    # allocate the tensors in the backend
    buffer = ggml.ggml_backend_alloc_ctx_tensors(ctx, backend)

    # set the values of the weights
    ggml.ggml_backend_tensor_set(
        a,
        ctypes.cast(np.array([3.0], dtype=np.single).ctypes.data, ctypes.c_void_p),
        0,
        ggml.ggml_nbytes(a),
    )
    ggml.ggml_backend_tensor_set(
        b,
        ctypes.cast(np.array([4.0], dtype=np.single).ctypes.data, ctypes.c_void_p),
        0,
        ggml.ggml_nbytes(a),
    )

    max_nodes = 4096

    buf_size = ggml.ggml_tensor_overhead() * max_nodes + ggml.ggml_graph_overhead_custom(max_nodes, False)
    buf = (ctypes.c_uint8 * buf_size)()

    def build_graph(x: ggml.ggml_tensor_p, a: ggml.ggml_tensor_p, b: ggml.ggml_tensor_p):
        params = ggml.ggml_init_params(
            mem_size=buf_size, mem_buffer=ctypes.cast(buf, ctypes.c_void_p), no_alloc=True
        )
        ctx0 = ggml.ggml_init(params=params)

        assert ctx0 is not None

        gf = ggml.ggml_new_graph_custom(ctx0, max_nodes, False)

        x2 = ggml.ggml_mul(ctx0, x, x)
        ax2 = ggml.ggml_mul(ctx0, a, x2)
        f = ggml.ggml_add(ctx0, ax2, b)

        ggml.ggml_set_name(x2, b"x2")
        ggml.ggml_set_name(ax2, b"ax2")
        ggml.ggml_set_name(f, b"f")

        ggml.ggml_build_forward_expand(gf, f)

        ggml.ggml_free(ctx0)

        return gf

    allocr = ggml.ggml_allocr_new_measure_from_backend(backend)

    gf = build_graph(x, a, b)

    mem_size = ggml.ggml_allocr_alloc_graph(allocr, gf)

    ggml.ggml_allocr_free(allocr)

    buf_compute = ggml.ggml_backend_alloc_buffer(backend, mem_size)
    allocr = ggml.ggml_allocr_new_from_buffer(buf_compute)

    ggml.ggml_allocr_reset(allocr)

    gf = build_graph(x, a, b)

    ggml.ggml_allocr_alloc_graph(allocr, gf)

    ggml.ggml_backend_tensor_set(
        x,
        ctypes.cast(np.array([2.0], dtype=np.single).ctypes.data, ctypes.c_void_p),
        0,
        ggml.ggml_nbytes(x),
    )

    ggml.ggml_backend_graph_compute(backend, gf)

    f = ggml.ggml_graph_get_tensor(gf, b"f")

    output = np.zeros(1, dtype=np.single)
    ggml.ggml_backend_tensor_get(f, ctypes.cast(output.ctypes.data, ctypes.c_void_p), 0, ggml.ggml_nbytes(x))

    assert output[0] == 16.0

    ggml.ggml_backend_buffer_free(buffer)
    ggml.ggml_backend_buffer_free(buf_compute)
    ggml.ggml_backend_free(backend)
    ggml.ggml_free(ctx)
