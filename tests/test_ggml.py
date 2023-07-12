import ctypes
import ggml


def test_ggml():
    assert ggml.GGML_FILE_VERSION == 1

    params = ggml.ggml_init_params(mem_size=16 * 1024 * 1024, mem_buffer=None)
    ctx = ggml.ggml_init(params=params)
    assert ggml.ggml_used_mem(ctx) == 0
    x = ggml.ggml_new_tensor_1d(ctx, ggml.GGML_TYPE_F32, 1)
    a = ggml.ggml_new_tensor_1d(ctx, ggml.GGML_TYPE_F32, 1)
    b = ggml.ggml_new_tensor_1d(ctx, ggml.GGML_TYPE_F32, 1)
    x2 = ggml.ggml_mul(ctx, x, x)
    f = ggml.ggml_add(ctx, ggml.ggml_mul(ctx, a, x2), b)
    gf = ggml.ggml_build_forward(f)

    ggml.ggml_set_f32(x, 2.0)
    ggml.ggml_set_f32(a, 3.0)
    ggml.ggml_set_f32(b, 4.0)

    ggml.ggml_graph_compute_with_ctx(ctx, ctypes.pointer(gf), 1)
    output = ggml.ggml_get_f32_1d(f, 0)
    assert output == 16.0
    ggml.ggml_free(ctx)


def test_ggml_custom_op():
    params = ggml.ggml_init_params(mem_size=16 * 1024 * 1024, mem_buffer=None)
    ctx = ggml.ggml_init(params=params)
    x_in = ggml.ggml_new_tensor_1d(ctx, ggml.GGML_TYPE_F32, 1)

    @ggml.ggml_custom1_op_f32_t
    def double(tensor_out: ggml.ggml_tensor_p, tensor_in: ggml.ggml_tensor_p):
        value = ggml.ggml_get_f32_1d(tensor_in, 0)
        ggml.ggml_set_f32(tensor_out, 2 * value)

    x_out = ggml.ggml_map_custom1_f32(ctx, x_in, double)
    gf = ggml.ggml_build_forward(x_out)

    ggml.ggml_set_f32(x_in, 21.0)

    ggml.ggml_graph_compute_with_ctx(ctx, ctypes.pointer(gf), 1)
    output = ggml.ggml_get_f32_1d(x_out, 0)
    assert output == 42.0
    ggml.ggml_free(ctx)


def test_ggml_min_alloc():
    overhead = ggml.ggml_tensor_overhead()
    max_overhead = overhead * 2 * ggml.GGML_MAX_NODES
    assert max_overhead < 16 * 1024 * 1024  # 16MB
    params = ggml.ggml_init_params(
        mem_size=max_overhead, mem_buffer=None, no_alloc=True
    )
    ctx = ggml.ggml_init(params=params)

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
        gf = ggml.ggml_build_forward(f)
        return gf

    gf = build_graph(ctx)
    gp = ggml.ggml_graph_plan(ctypes.pointer(gf), 1)

    n_nodes = gf.n_nodes
    nodes_size = sum(ggml.ggml_nbytes(gf.nodes[i]) for i in range(n_nodes))
    n_leafs = gf.n_leafs
    leafs_size = sum(ggml.ggml_nbytes(gf.leafs[i]) for i in range(n_leafs))

    ggml.ggml_free(ctx)

    assert n_nodes == 3  # 3 nodes: mul, mul, add
    assert n_leafs == 3  # 3 leafs: x, a, b

    mem_size = nodes_size + leafs_size + overhead * (n_nodes + n_leafs + 1) # TODO: why +1?
    params = ggml.ggml_init_params(mem_size=mem_size, mem_buffer=None)
    ctx = ggml.ggml_init(params=params)
    gf = build_graph(ctx)

    a = ggml.ggml_get_tensor(ctx, b"a")
    b = ggml.ggml_get_tensor(ctx, b"b")
    x = ggml.ggml_get_tensor(ctx, b"x")
    f = ggml.ggml_get_tensor(ctx, b"f")

    assert a is not None and b is not None and x is not None and f is not None

    ggml.ggml_set_f32(x, 2.0)
    ggml.ggml_set_f32(a, 3.0)
    ggml.ggml_set_f32(b, 4.0)

    gp = ggml.ggml_graph_plan(ctypes.pointer(gf), 1)
    ggml.ggml_graph_compute(ctypes.pointer(gf), ctypes.pointer(gp))
    output = ggml.ggml_get_f32_1d(f, 0)
    assert output == 16.0
    ggml.ggml_free(ctx)
