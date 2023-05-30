import ctypes
import ggml


def test_ggml():
    assert ggml.GGML_FILE_VERSION.value == 1

    params = ggml.ggml_init_params(
        mem_size=16 * 1024 * 1024, mem_buffer=None, no_alloc=False
    )
    ctx = ggml.ggml_init(params=params)
    assert ggml.ggml_used_mem(ctx) == 0
    x = ggml.ggml_new_tensor_1d(
        ctx,
        ggml.GGML_TYPE_F32,
        ctypes.c_int64(1),
    )
    ggml.ggml_set_param(ctx, x)
    a = ggml.ggml_new_tensor_1d(
        ctx,
        ggml.GGML_TYPE_F32,
        ctypes.c_int64(1),
    )
    b = ggml.ggml_new_tensor_1d(
        ctx,
        ggml.GGML_TYPE_F32,
        ctypes.c_int64(1),
    )
    x2 = ggml.ggml_mul(ctx, x, x)
    f = ggml.ggml_add(ctx, ggml.ggml_mul(ctx, a, x2), b)
    gf = ggml.ggml_build_forward(f)

    ggml.ggml_set_f32(x, ctypes.c_float(2.0))
    ggml.ggml_set_f32(a, ctypes.c_float(3.0))
    ggml.ggml_set_f32(b, ctypes.c_float(4.0))

    ggml.ggml_graph_compute(ctx, ctypes.pointer(gf))
    output = ggml.ggml_get_f32_1d(f, ctypes.c_int(0))
    assert output == 16.0
    ggml.ggml_free(ctx)
