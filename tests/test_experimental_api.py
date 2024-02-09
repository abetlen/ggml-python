from ggml.experimental import ggml_context, ggml_cgraph, Tensor, GGML_TYPE, Backend

def test_experimental_api():
    backend = Backend.cpu()

    with ggml_context():
        a = Tensor.with_shape((1,), ggml_type=GGML_TYPE.F32)
        b = Tensor.with_shape((1,), ggml_type=GGML_TYPE.F32)
        x = Tensor.with_shape((1,), ggml_type=GGML_TYPE.F32)

        f = a * x * x + b

        assert f.shape == (1,)

        backend.alloc_ctx_tensors()

        a[0] = 3.0
        b[0] = 4.0
        x[0] = 2.0

        assert a[0] == 3.0
        assert b[0] == 4.0
        assert x[0] == 2.0

        graph = ggml_cgraph(f)
        graph.compute(backend)

        assert f[0] == 16.0

    with ggml_context():
        a = Tensor.with_shape((1,), ggml_type=GGML_TYPE.F32)
        b = Tensor.with_shape((1,), ggml_type=GGML_TYPE.F32)

        backend.alloc_ctx_tensors()

        a[0] = 3.0
        b[0] = 4.0

        with ggml_context():
            x = Tensor.with_shape((1,), ggml_type=GGML_TYPE.F32)

            f = a * x * x + b

            assert f.shape == (1,)

            backend.alloc_ctx_tensors()

            x[0] = 2.0

            assert a[0] == 3.0
            assert b[0] == 4.0
            assert x[0] == 2.0

            graph = ggml_cgraph(f)
            graph.compute(backend)

            assert f[0] == 16.0

    with ggml_context():
        a = Tensor.with_shape((1,), ggml_type=GGML_TYPE.F32)
        b = Tensor.with_shape((1,), ggml_type=GGML_TYPE.F32)

        backend.alloc_ctx_tensors()

        a[0] = 3.0
        b[0] = 4.0

        with ggml_context():
            x = Tensor.with_shape((1,), ggml_type=GGML_TYPE.F32)

            backend.alloc_ctx_tensors()

            x[0] = 2.0

            f = a * x * x + b

            assert f.shape == (1,)

            measure_allocr = backend.new_measure()

            graph = ggml_cgraph(f)

            mem_size = measure_allocr.alloc_graph(graph)

            buffer = backend.alloc_buffer(mem_size)

            allocr = buffer.new_allocr()
            allocr.alloc_graph(graph)

            graph.compute(backend)

            assert f[0] == 16.0
