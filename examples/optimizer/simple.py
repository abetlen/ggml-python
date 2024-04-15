# %% [markdown]
"""
# Single-batch stochastic gradient descent example using ggml

This example demonstrates how to use ggml to implement a simple SGD optimizer.
"""
# %%
import ggml
import random

a_real = 3.0
b_real = 4.0

ctx0 = ggml.ggml_init(ggml.ggml_init_params(
    mem_size=128 * 1024 * 1024, mem_buffer=None, no_alloc=False
))

assert ctx0 is not None

# define parameters
a = ggml.ggml_new_tensor_1d(ctx0, ggml.GGML_TYPE_F32, 1)
ggml.ggml_set_param(ctx0, a)

b = ggml.ggml_new_tensor_1d(ctx0, ggml.GGML_TYPE_F32, 1)
ggml.ggml_set_param(ctx0, b)

# define input and output
x = ggml.ggml_new_tensor_1d(ctx0, ggml.GGML_TYPE_F32, 1)
ggml.ggml_set_input(x)

tmp = ggml.ggml_mul(ctx0, a, x)
f = ggml.ggml_add(ctx0, tmp, b)

# define loss
f_true = ggml.ggml_new_tensor_1d(ctx0, ggml.GGML_TYPE_F32, 1)
ggml.ggml_set_input(f_true)

tmp = ggml.ggml_sub(ctx0, f, f_true)
loss = ggml.ggml_mul(ctx0, tmp, tmp)

# build forward and backward graph
gf = ggml.ggml_new_graph_custom(ctx0, ggml.GGML_DEFAULT_GRAPH_SIZE, True)
ggml.ggml_build_forward_expand(gf, loss)
gb = ggml.ggml_graph_dup(ctx0, gf)
ggml.ggml_build_backward_expand(ctx0, gf, gb, False)

# initialize parameters
ggml.ggml_set_f32(a, 1.0)
ggml.ggml_set_f32(b, 1.0)

# SGD
lr = 1e-2
nsteps = 1000
decay = 1e-3

for i in range(nsteps):
    # sample data
    x_sample = random.uniform(-10, 10)
    f_sample = a_real * x_sample + b_real

    # set input
    ggml.ggml_set_f32(x, x_sample)
    ggml.ggml_set_f32(f_true, f_sample)

    # reset graph
    ggml.ggml_graph_reset(gf)
    ggml.ggml_set_f32(loss.contents.grad, 1.0)

    # compute forward and backward
    ggml.ggml_graph_compute_with_ctx(ctx0, gb, 1)

    # print loss
    loss_ = ggml.ggml_get_f32_1d(loss, 0)
    print(f"step {i}: loss = {loss_}")

    # decay learning rate
    lr *= (1.0 - decay)

    # update parameters
    ggml.ggml_set_f32(a, ggml.ggml_get_f32_1d(a, 0) - lr * ggml.ggml_get_f32_1d(a.contents.grad, 0))
    ggml.ggml_set_f32(b, ggml.ggml_get_f32_1d(b, 0) - lr * ggml.ggml_get_f32_1d(b.contents.grad, 0))

    # print parameters
    print(f"a = {ggml.ggml_get_f32_1d(a, 0):.2f}, b = {ggml.ggml_get_f32_1d(b, 0):.2f}")


ggml.ggml_free(ctx0)

# %%
