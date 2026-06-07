import ctypes
import re

from pathlib import Path
from typing import Optional

import ggml

import numpy as np


def test_ggml_h_public_api_bindings_match_header():
    header_path = Path(__file__).parents[1] / "vendor" / "ggml" / "include" / "ggml.h"
    header = header_path.read_text()
    header = re.sub(r"/\*.*?\*/", "", header, flags=re.S)
    header = re.sub(r"//.*", "", header)
    pattern = re.compile(
        r"GGML_API\s+(?:GGML_CALL\s+)?(?:[\w\s\*]+?)\s+(ggml_\w+)\s*\((.*?)\)\s*;",
        re.S,
    )
    ignored = {"ggml_abort", "ggml_format_name", "ggml_unused_vars_impl"}
    missing = []
    mismatched = []

    for name, parameters in pattern.findall(header):
        if name in ignored:
            continue
        if not hasattr(ggml, name):
            missing.append(name)
            continue
        function = getattr(ggml.lib, name)
        argtypes = getattr(function, "argtypes", None)
        if argtypes is None:
            continue
        expected_count = _ggml_h_parameter_count(parameters)
        if expected_count != len(argtypes):
            mismatched.append((name, expected_count, len(argtypes)))

    assert missing == []
    assert mismatched == []


def _ggml_h_parameter_count(parameters: str) -> int:
    parameters = parameters.strip()
    if not parameters or parameters == "void":
        return 0
    count = 1
    depth = 0
    for character in parameters:
        if character == "(":
            depth += 1
        elif character == ")":
            depth -= 1
        elif character == "," and depth == 0:
            count += 1
    return count


def test_ggml():
    assert ggml.GGML_FILE_VERSION == 2

    params = ggml.ggml_init_params(mem_size=16 * 1024 * 1024, mem_buffer=None)
    ctx = ggml.ggml_init(params)
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


def test_ggml_pythonic():
    import contextlib

    with contextlib.ExitStack() as stack:
        backend = ggml.ggml_backend_cpu_init()
        assert backend is not None
        stack.callback(ggml.ggml_backend_free, backend)

        params = ggml.ggml_init_params(
            mem_size=ggml.ggml_tensor_overhead() * 6 + ggml.ggml_graph_overhead(),
            no_alloc=True,
        )
        ctx = ggml.ggml_init(params)
        assert ctx is not None
        stack.callback(ggml.ggml_free, ctx)

        x = ggml.ggml_new_tensor_1d(ctx, ggml.GGML_TYPE_F32, 1)
        a = ggml.ggml_new_tensor_1d(ctx, ggml.GGML_TYPE_F32, 1)
        b = ggml.ggml_new_tensor_1d(ctx, ggml.GGML_TYPE_F32, 1)
        x2 = ggml.ggml_mul(ctx, x, x)
        f = ggml.ggml_add(ctx, ggml.ggml_mul(ctx, a, x2), b)
        gf = ggml.ggml_new_graph(ctx)

        ggml.ggml_build_forward_expand(gf, f)

        buffer = ggml.ggml_backend_alloc_ctx_tensors(ctx, backend)
        assert buffer is not None
        stack.callback(ggml.ggml_backend_buffer_free, buffer)

        ggml.ggml_set_f32(x, 2.0)
        ggml.ggml_set_f32(a, 3.0)
        ggml.ggml_set_f32(b, 4.0)

        ggml.ggml_backend_graph_compute(backend, gf)

        output = ggml.ggml_get_f32_1d(f, 0)

        assert output == 16.0


def test_ggml_backend_feature_flags_match_exported_symbols():
    assert ggml.GGML_USE_CUDA == hasattr(ggml.lib, "ggml_backend_cuda_init")
    assert ggml.GGML_USE_METAL == hasattr(ggml.lib, "ggml_backend_metal_init")
    assert ggml.GGML_USE_OPENCL == hasattr(ggml.lib, "ggml_backend_opencl_init")
    assert ggml.GGML_USE_CLBLAST == hasattr(ggml.lib, "ggml_cl_init")
    assert ggml.GGML_USE_VULKAN == hasattr(ggml.lib, "ggml_backend_vk_init")
    assert ggml.GGML_USE_VULKAN_CPU_ASSIST == hasattr(
        ggml.lib, "ggml_vk_init_cpu_assist"
    )
    assert ggml.GGML_USE_RPC == hasattr(ggml.lib, "ggml_backend_rpc_init")


def test_ggml_custom_op():
    params = ggml.ggml_init_params(mem_size=16 * 1024 * 1024, mem_buffer=None)
    ctx = ggml.ggml_init(params)
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


def test_quantize():
    ne0 = 32
    ne1 = 1
    nelements = ne0 * ne1
    data = [float(i) for i in range(nelements)]
    data_f32 = (ctypes.c_float * len(data))(*data)
    work = (ctypes.c_float * nelements)(0)
    # TODO: convert to ggml.ggml_quantize_chunk
    # cur_size = ggml.ggml_quantize_q8_0(
    cur_size = ggml.ggml_quantize_chunk(
        ggml.GGML_TYPE_Q8_0,
        data_f32,
        ctypes.cast(work, ctypes.c_void_p),
        0,
        nelements // ne0,
        ne0,
        None,
    )
    assert cur_size == 34

    type_traits = ggml.ggml_get_type_traits(ggml.GGML_TYPE_Q8_0).contents
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
    ctx = ggml.ggml_init(params)
    assert ctx is not None

    backend = ggml.ggml_backend_cpu_init()

    assert backend is not None

    # create the tensors for input and weights
    x = ggml.ggml_new_tensor_1d(ctx, ggml.GGML_TYPE_F32, 1)

    a = ggml.ggml_new_tensor_1d(ctx, ggml.GGML_TYPE_F32, 1)
    b = ggml.ggml_new_tensor_1d(ctx, ggml.GGML_TYPE_F32, 1)

    # allocate the tensors in the backend
    buffer = ggml.ggml_backend_alloc_ctx_tensors(ctx, backend)
    assert buffer is not None

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
        ctx0 = ggml.ggml_init(params)

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

    buffer_type = ggml.ggml_backend_get_default_buffer_type(backend)
    assert buffer_type is not None
    allocr = ggml.ggml_gallocr_new(buffer_type)
    assert allocr is not None

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


def test_grad():
    nthreads = 1
    params = ggml.ggml_init_params(
        mem_size=128 * 1024 * 1024, mem_buffer=None, no_alloc=False
    )
    ctx0 = ggml.ggml_init(params)
    assert ctx0 is not None

    x = ggml.ggml_new_tensor_1d(ctx0, ggml.GGML_TYPE_F32, 1)

    ggml.ggml_set_param(x)

    a = ggml.ggml_new_tensor_1d(ctx0, ggml.GGML_TYPE_F32, 1)
    b = ggml.ggml_mul(ctx0, x, x)
    f = ggml.ggml_mul(ctx0, a, b)

    gf = ggml.ggml_new_graph_custom(ctx0, ggml.GGML_DEFAULT_GRAPH_SIZE, True)
    ggml.ggml_build_forward_expand(gf, f)

    ggml.ggml_set_loss(f)
    gb = ggml.ggml_graph_dup(ctx0, gf, True)

    ggml.ggml_build_backward_expand(ctx0, gb, None)

    ggml.ggml_set_f32(x, 2.0)
    ggml.ggml_set_f32(a, 3.0)

    ggml.ggml_graph_reset(gb)

    ggml.ggml_graph_compute_with_ctx(ctx0, gb, nthreads)

    f_grad = ggml.ggml_graph_get_grad(gb, f)
    x_grad = ggml.ggml_graph_get_grad(gb, x)
    assert f_grad is not None
    assert x_grad is not None

    assert ggml.ggml_get_f32_1d(f, 0) == 12.0
    assert ggml.ggml_get_f32_1d(f_grad, 0) == 1.0
    assert ggml.ggml_get_f32_1d(x_grad, 0) == 12.0

    ggml.ggml_free(ctx0)
