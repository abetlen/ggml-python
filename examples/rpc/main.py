import ctypes
import argparse
import contextlib

import ggml

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=9091)
    args = parser.parse_args()

    with contextlib.ExitStack() as stack:
        backend = ggml.ggml_backend_rpc_init(f"{args.host}:{args.port}".encode("utf-8"))
        assert backend is not None
        stack.callback(ggml.ggml_backend_free, backend)

        params = ggml.ggml_init_params(
            mem_size=ggml.ggml_tensor_overhead() * 6 + ggml.ggml_graph_overhead() + 10000,
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

        x_data = (ctypes.c_float * 1)(2.0)
        ggml.ggml_backend_tensor_set(
            x,  # tensor
            x_data,  # data
            0,  # offset
            ctypes.sizeof(x_data),  # size
        )

        a_data = (ctypes.c_float * 1)(3.0)
        ggml.ggml_backend_tensor_set(
            a,  # tensor
            a_data,  # data
            0,  # offset
            ctypes.sizeof(a_data),  # size
        )

        b_data = (ctypes.c_float * 1)(4.0)
        ggml.ggml_backend_tensor_set(
            b,  # tensor
            b_data,  # data
            0,  # offset
            ctypes.sizeof(b_data),  # size
        )

        ggml.ggml_backend_graph_compute(backend, gf)

        output = ctypes.c_float()
        ggml.ggml_backend_tensor_get(
            f,  # tensor
            ctypes.byref(output),  # data
            0,  # offset
            ctypes.sizeof(output),  # size
        )

        print(f"Output: {output.value}")

        assert output.value == 16.0

if __name__ == "__main__":
    main()