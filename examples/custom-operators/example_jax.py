import ctypes

import ggml
import ggml.utils

import jax

from typing import Optional

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
    x = jax.device_put(ggml.utils.to_numpy(tensor_in))
    x *= 2
    ggml.utils.to_numpy(tensor_out)[:] = jax.device_get(x)

x_out = ggml.ggml_map_custom1(ctx, x_in, double, 1, None)
gf = ggml.ggml_build_forward(x_out)

ggml.ggml_set_f32(x_in, 21.0)

ggml.ggml_graph_compute_with_ctx(ctx, ctypes.pointer(gf), 1)
output = ggml.ggml_get_f32_1d(x_out, 0)
assert output == 42.0
print("GGML output: ", output)
ggml.ggml_free(ctx)