import ggml.ggml_ext as ggml


def test_ggml():
    assert ggml.GGML_FILE_VERSION == 1

    params = ggml.ggml_init_params(
        mem_size=16 * 1024 * 1024,
        no_alloc=False
    )
    ctx = ggml.ggml_init(params=params)
    assert ggml.ggml_used_mem(ctx) == 0
    assert ggml.ggml_type_size(ggml.GGML_TYPE_F32) == 4
    assert params.mem_size > 0
    assert params.no_alloc == False
    assert params.mem_buffer == None
    x = ggml.ggml_new_tensor_1d(
        ctx=ctx,
        type=ggml.GGML_TYPE_F32,
        ne0=1,
    )
    ggml.ggml_set_param(ctx, x)
    a = ggml.ggml_new_tensor_1d(
        ctx,
        ggml.GGML_TYPE_F32,
        1,
    )
    b = ggml.ggml_new_tensor_1d(
        ctx,
        ggml.GGML_TYPE_F32,
        1,
    )
    x2 = ggml.ggml_mul(ctx, x, x)
    f = ggml.ggml_add(ctx, ggml.ggml_mul(ctx, a, x2), b)
    gf = ggml.ggml_build_forward(f)

    ggml.ggml_set_f32(x, 2.0)
    ggml.ggml_set_f32(a, 3.0)
    ggml.ggml_set_f32(b, 4.0)

    ggml.ggml_graph_compute(ctx, gf)
    output = ggml.ggml_get_f32_1d(f, 0)
    assert output == 16.0
    ggml.ggml_free(ctx)
