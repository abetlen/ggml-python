import ctypes
from typing import Optional
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
    nodes_size = sum(ggml.ggml_nbytes_pad(gf.nodes[i]) for i in range(n_nodes))
    n_leafs = gf.n_leafs
    leafs_size = sum(ggml.ggml_nbytes_pad(gf.leafs[i]) for i in range(n_leafs))

    ggml.ggml_free(ctx)

    assert n_nodes == 3  # 3 nodes: mul, mul, add
    assert n_leafs == 3  # 3 leafs: x, a, b

    mem_size = nodes_size + leafs_size + overhead * (n_nodes + n_leafs)
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


def test_ggml_alloc():
    def build_graph(ctx: ggml.ggml_context_p, alloc: ggml.ggml_allocr_p):
        # inputs
        x = ggml.ggml_new_tensor_1d(ctx, ggml.GGML_TYPE_F32, 1)
        ggml.ggml_set_name(x, b"x")
        ggml.ggml_allocr_alloc(alloc, x)
        a = ggml.ggml_new_tensor_1d(ctx, ggml.GGML_TYPE_F32, 1)
        ggml.ggml_set_name(a, b"a")
        ggml.ggml_allocr_alloc(alloc, a)
        b = ggml.ggml_new_tensor_1d(ctx, ggml.GGML_TYPE_F32, 1)
        ggml.ggml_set_name(b, b"b")
        ggml.ggml_allocr_alloc(alloc, b)

        x2 = ggml.ggml_mul(ctx, x, x)
        tmp = ggml.ggml_mul(ctx, a, x2)

        # outputs
        f = ggml.ggml_add(ctx, tmp, b)
        ggml.ggml_set_name(f, b"f")

        # build graph
        gf = ggml.ggml_build_forward(f)

        return gf

    overhead = ggml.ggml_tensor_overhead()
    max_overhead = overhead * 2 * ggml.GGML_MAX_NODES
    assert max_overhead < 16 * 1024 * 1024  # 16MB
    params = ggml.ggml_init_params(
        mem_size=max_overhead, mem_buffer=None, no_alloc=True
    )
    ctx = ggml.ggml_init(params=params)

    tensor_alignment = 32
    alloc = ggml.ggml_allocr_new_measure(tensor_alignment)
    assert alloc is not None
    assert ggml.ggml_allocr_is_measure(alloc)

    gf = build_graph(ctx, alloc)
    gp = ggml.ggml_graph_plan(ctypes.pointer(gf), 1)
    assert gp.work_size == 0

    alloc_size = (
        ggml.ggml_allocr_alloc_graph(alloc, ctypes.pointer(gf)) + tensor_alignment
    )

    ggml.ggml_free(ctx)
    ggml.ggml_allocr_free(alloc)

    params = ggml.ggml_init_params(
        mem_size=max_overhead, mem_buffer=None, no_alloc=True
    )
    ctx = ggml.ggml_init(params=params)
    buffer = (ctypes.c_uint8 * alloc_size)()
    alloc = ggml.ggml_allocr_new(
        ctypes.cast(buffer, ctypes.c_void_p), alloc_size, tensor_alignment
    )
    gf = build_graph(ctx, alloc)
    ggml.ggml_allocr_alloc_graph(alloc, ctypes.pointer(gf))

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
    ggml.ggml_allocr_free(alloc)

def test_ggml_alloc_one_pass():
    overhead = ggml.ggml_tensor_overhead()
    max_overhead = overhead * 2 * ggml.GGML_MAX_NODES
    assert max_overhead < 16 * 1024 * 1024  # 16MB
    params = ggml.ggml_init_params(
        mem_size=max_overhead, mem_buffer=None, no_alloc=True
    )
    ctx = ggml.ggml_init(params=params)

    # define the graph
    x = ggml.ggml_new_tensor_1d(ctx, ggml.GGML_TYPE_F32, 1)
    ggml.ggml_set_name(x, b"x")
    a = ggml.ggml_new_tensor_1d(ctx, ggml.GGML_TYPE_F32, 1)
    ggml.ggml_set_name(a, b"a")
    b = ggml.ggml_new_tensor_1d(ctx, ggml.GGML_TYPE_F32, 1)
    ggml.ggml_set_name(b, b"b")

    x2 = ggml.ggml_mul(ctx, x, x)
    tmp = ggml.ggml_mul(ctx, a, x2)

    # outputs
    f = ggml.ggml_add(ctx, tmp, b)
    ggml.ggml_set_name(f, b"f")

    # build graph
    gf = ggml.ggml_build_forward(f)

    # save old data pointers
    leaf_data = [ggml.ggml_get_data(gf.leafs[i]) for i in range(gf.n_leafs)]
    node_data = [ggml.ggml_get_data(gf.nodes[i]) for i in range(gf.n_nodes)]

    # create measure allocator
    tensor_alignment = 32
    alloc = ggml.ggml_allocr_new_measure(tensor_alignment)
    assert alloc is not None
    assert ggml.ggml_allocr_is_measure(alloc)

    # allocate input tensors
    ggml.ggml_allocr_alloc(alloc, x)
    ggml.ggml_allocr_alloc(alloc, a)
    ggml.ggml_allocr_alloc(alloc, b)
    # allocate graph
    alloc_size = (
        ggml.ggml_allocr_alloc_graph(alloc, ctypes.pointer(gf)) + tensor_alignment
    )
    assert alloc_size > 0

    # restore old data pointers
    for i in range(gf.n_leafs):
        gf.leafs[i].contents.data = leaf_data[i]
    
    for i in range(gf.n_nodes):
        gf.nodes[i].contents.data = node_data[i]

    # free measure allocator
    ggml.ggml_allocr_free(alloc)

    # allocate tensor memory
    buffer = (ctypes.c_uint8 * alloc_size)()
    alloc = ggml.ggml_allocr_new(
        ctypes.cast(buffer, ctypes.c_void_p), alloc_size, tensor_alignment
    )
    ggml.ggml_allocr_alloc(alloc, x)
    ggml.ggml_allocr_alloc(alloc, a)
    ggml.ggml_allocr_alloc(alloc, b)
    ggml.ggml_allocr_alloc_graph(alloc, ctypes.pointer(gf))

    # set input values
    ggml.ggml_set_f32(x, 2.0)
    ggml.ggml_set_f32(a, 3.0)
    ggml.ggml_set_f32(b, 4.0)

    gp = ggml.ggml_graph_plan(ctypes.pointer(gf), 1)
    assert gp.work_size == 0

    # compute
    ggml.ggml_graph_compute(ctypes.pointer(gf), ctypes.pointer(gp))

    output = ggml.ggml_get_f32_1d(f, 0)
    assert output == 16.0

    ggml.ggml_free(ctx)
    ggml.ggml_allocr_free(alloc)

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
