import ggml

def test_ggml():
    assert ggml.GGML_FILE_VERSION.value == 1

    params = ggml.ggml_init_params(0, None, False)
    ctx = ggml.ggml_init(params=params)
    assert ggml.ggml_used_mem(ctx) == 0
    ggml.ggml_free(ctx)