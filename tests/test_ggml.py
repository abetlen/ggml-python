import ctypes

from typing import Optional

import ggml

import numpy as np


def test_ggml():
    assert ggml.GGML_FILE_VERSION == 1

    params = ggml.ggml_init_params(mem_size=16 * 1024 * 1024, mem_buffer=None)
    ctx = ggml.ggml_init(params=params)
    assert ctx is not None
    assert ggml.ggml_used_mem(ctx) == 0
    x = ggml.ggml_new_tensor_1d(ctx, ggml.GGML_TYPE_F32, 1)
    a = ggml.ggml_new_tensor_1d(ctx, ggml.GGML_TYPE_F32, 1)
    b = ggml.ggml_new_tensor_1d(ctx, ggml.GGML_TYPE_F32, 1)
    x2 = ggml.ggml_mul(ctx, x, x)
    f = ggml.ggml_add(ctx, ggml.ggml_mul(ctx, a, x2), b)
    gf = ggml.ggml_new_graph(ctx)
    ggml.ggml_build_forward_expand(gf, f)

    ggml.ggml_set_f32(x, 2.0)
    ggml.ggml_set_f32(a, 3.0)
    ggml.ggml_set_f32(b, 4.0)

    ggml.ggml_graph_compute_with_ctx(ctx, gf, 1)
    output = ggml.ggml_get_f32_1d(f, 0)
    assert output == 16.0
    ggml.ggml_free(ctx)


def test_ggml_custom_op():
    params = ggml.ggml_init_params(mem_size=16 * 1024 * 1024, mem_buffer=None)
    ctx = ggml.ggml_init(params=params)
    assert ctx is not None
    x_in = ggml.ggml_new_tensor_1d(ctx, ggml.GGML_TYPE_F32, 1)

    @ggml.ggml_custom1_op_t
    def double(
        tensor_out: ggml.ggml_tensor_p,
        tensor_in: ggml.ggml_tensor_p,
        ith: int,
        nth: int,
        userdata: Optional[ctypes.c_void_p],
    ):
        value = ggml.ggml_get_f32_1d(tensor_in, 0)
        ggml.ggml_set_f32(tensor_out, 2 * value)

    x_out = ggml.ggml_map_custom1(ctx, x_in, double, 1, None)
    gf = ggml.ggml_new_graph(ctx)
    ggml.ggml_build_forward_expand(gf, x_out)

    ggml.ggml_set_f32(x_in, 21.0)

    ggml.ggml_graph_compute_with_ctx(ctx, gf, 1)
    output = ggml.ggml_get_f32_1d(x_out, 0)
    assert output == 42.0
    ggml.ggml_free(ctx)


def test_ggml_min_alloc():
    max_overhead = (
        ggml.ggml_tensor_overhead() * ggml.GGML_DEFAULT_GRAPH_SIZE
        + ggml.ggml_graph_overhead()
    )
    assert max_overhead < 16 * 1024 * 1024  # 16MB
    params = ggml.ggml_init_params(
        mem_size=max_overhead, mem_buffer=None, no_alloc=True
    )
    ctx = ggml.ggml_init(params=params)
    assert ctx is not None

    def build_graph(ctx: ggml.ggml_context_p):
        x = ggml.ggml_new_tensor_1d(ctx, ggml.GGML_TYPE_F32, 1)
        ggml.ggml_set_name(x, b"x")
        a = ggml.ggml_new_tensor_1d(ctx, ggml.GGML_TYPE_F32, 1)
        ggml.ggml_set_name(a, b"a")
        b = ggml.ggml_new_tensor_1d(ctx, ggml.GGML_TYPE_F32, 1)
        ggml.ggml_set_name(b, b"b")
        x2 = ggml.ggml_mul(ctx, x, x)
        tmp = ggml.ggml_mul(ctx, a, x2)
        f = ggml.ggml_add(ctx, tmp, b)
        ggml.ggml_set_name(f, b"f")
        gf = ggml.ggml_new_graph(ctx)
        ggml.ggml_build_forward_expand(gf, f)
        return gf

    gf = build_graph(ctx)
    gp = ggml.ggml_graph_plan(gf, 1)

    n_nodes = gf.contents.n_nodes
    nodes_size = sum(ggml.ggml_nbytes_pad(gf.contents.nodes[i]) for i in range(n_nodes))
    n_leafs = gf.contents.n_leafs
    leafs_size = sum(ggml.ggml_nbytes_pad(gf.contents.leafs[i]) for i in range(n_leafs))

    mem_size = (
        nodes_size
        + leafs_size
        + ggml.ggml_tensor_overhead() * (n_nodes + n_leafs)
        + ggml.ggml_graph_overhead()
    )

    ggml.ggml_free(ctx)

    assert n_nodes == 3  # 3 nodes: mul, mul, add
    assert n_leafs == 3  # 3 leafs: x, a, b

    params = ggml.ggml_init_params(mem_size=mem_size, mem_buffer=None)
    ctx = ggml.ggml_init(params=params)
    assert ctx is not None
    gf = build_graph(ctx)

    a = ggml.ggml_get_tensor(ctx, b"a")
    b = ggml.ggml_get_tensor(ctx, b"b")
    x = ggml.ggml_get_tensor(ctx, b"x")
    f = ggml.ggml_get_tensor(ctx, b"f")

    assert a is not None and b is not None and x is not None and f is not None

    ggml.ggml_set_f32(x, 2.0)
    ggml.ggml_set_f32(a, 3.0)
    ggml.ggml_set_f32(b, 4.0)

    gp = ggml.ggml_graph_plan(gf, 1)
    ggml.ggml_graph_compute(gf, ctypes.pointer(gp))
    output = ggml.ggml_get_f32_1d(f, 0)
    assert output == 16.0
    ggml.ggml_free(ctx)


def test_quantize():
    ne0 = 32
    ne1 = 1
    nelements = ne0 * ne1
    data = [float(i) for i in range(nelements)]
    data_f32 = (ctypes.c_float * len(data))(*data)
    work = (ctypes.c_float * nelements)(0)
    hist = (ctypes.c_int64 * (1 << 4))(0)
    cur_size = ggml.ggml_quantize_q8_0(
        data_f32,
        ctypes.cast(work, ctypes.c_void_p),
        nelements,
        ne0,
        hist,
    )
    assert cur_size == 34

    type_traits = ggml.ggml_internal_get_type_traits(ggml.GGML_TYPE_Q8_0)
    work2 = (ctypes.c_float * nelements)(0)
    type_traits.to_float(
        ctypes.cast(work, ctypes.c_void_p),
        ctypes.cast(work2, ctypes.POINTER(ctypes.c_float)),
        nelements,
    )

    eps = 0.5
    for i in range(nelements):
        assert abs(work2[i] - data[i]) < eps


def test_ggml_cpu_backend():
    n_tensors = 1 + 2  # input (x) and weights (a, b)
    params = ggml.ggml_init_params(
        mem_size=ggml.ggml_tensor_overhead() * n_tensors, mem_buffer=None, no_alloc=True
    )
    ctx = ggml.ggml_init(params=params)
    assert ctx is not None

    backend = ggml.ggml_backend_cpu_init()

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

    buf_size = (
        ggml.ggml_tensor_overhead() * max_nodes
        + ggml.ggml_graph_overhead_custom(max_nodes, False)
    )
    buf = (ctypes.c_uint8 * buf_size)()

    def build_graph(
        x: ggml.ggml_tensor_p, a: ggml.ggml_tensor_p, b: ggml.ggml_tensor_p
    ):
        params = ggml.ggml_init_params(
            mem_size=buf_size,
            mem_buffer=ctypes.cast(buf, ctypes.c_void_p),
            no_alloc=True,
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

    allocr = ggml.ggml_gallocr_new(ggml.ggml_backend_get_default_buffer_type(backend))

    gf = build_graph(x, a, b)

    ggml.ggml_gallocr_reserve(allocr, gf)

    gf = build_graph(x, a, b)

    ggml.ggml_gallocr_alloc_graph(allocr, gf)

    ggml.ggml_backend_tensor_set(
        x,
        ctypes.cast(np.array([2.0], dtype=np.single).ctypes.data, ctypes.c_void_p),
        0,
        ggml.ggml_nbytes(x),
    )

    ggml.ggml_backend_graph_compute(backend, gf)

    f = ggml.ggml_graph_get_tensor(gf, b"f")

    output = np.zeros(1, dtype=np.single)
    ggml.ggml_backend_tensor_get(
        f, ctypes.cast(output.ctypes.data, ctypes.c_void_p), 0, ggml.ggml_nbytes(x)
    )

    assert output[0] == 16.0

    ggml.ggml_gallocr_free(allocr)
    ggml.ggml_backend_buffer_free(buffer)
    ggml.ggml_backend_free(backend)
    ggml.ggml_free(ctx)
