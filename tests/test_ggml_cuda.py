import ggml
import ctypes
import pytest

ggml_cuda_available = ggml.GGML_USE_CUBLAS

run_if_ggml_cuda_available = pytest.mark.skipif(
    not ggml_cuda_available,
    reason="CUDA not available",
)


@run_if_ggml_cuda_available
def test_cuda():
    mem_size = 16 * 1024 * 1024
    buf = ggml.ggml_cuda_host_malloc(mem_size)
    assert buf is not None
    params = ggml.ggml_init_params(mem_size, mem_buffer=buf)
    ctx = ggml.ggml_init(params=params)

    x = ggml.ggml_new_tensor_1d(ctx, ggml.GGML_TYPE_F32, 1)
    ggml.ggml_set_f32(x, 2.0)
    x.contents.backend = ggml.GGML_BACKEND_GPU
    ggml.ggml_cuda_transform_tensor(x.contents.data, x)

    a = ggml.ggml_new_tensor_1d(ctx, ggml.GGML_TYPE_F32, 1)
    a.contents.backend = ggml.GGML_BACKEND_GPU
    ggml.ggml_set_f32(a, 3.0)
    ggml.ggml_cuda_transform_tensor(a.contents.data, a)

    b = ggml.ggml_new_tensor_1d(ctx, ggml.GGML_TYPE_F32, 1)
    b.contents.backend = ggml.GGML_BACKEND_GPU
    ggml.ggml_set_f32(b, 4.0)
    ggml.ggml_cuda_transform_tensor(b.contents.data, b)

    x2 = ggml.ggml_mul(ctx, x, x)
    ggml.ggml_cuda_assign_buffers_no_scratch(x2)

    tmp = ggml.ggml_mul(ctx, a, x2)
    ggml.ggml_cuda_assign_buffers_no_scratch(tmp)

    f = ggml.ggml_add(ctx, tmp, b)

    gf = ggml.ggml_build_forward(f)

    ggml.ggml_graph_compute(ctx, ctypes.pointer(gf))
    output = ggml.ggml_get_f32_1d(f, 0)
    assert output == 16.0
    ggml.ggml_free(ctx)
    ggml.ggml_cuda_free_data(a)
    ggml.ggml_cuda_free_data(b)
    ggml.ggml_cuda_free_data(x)
    ggml.ggml_cuda_host_free(buf)
