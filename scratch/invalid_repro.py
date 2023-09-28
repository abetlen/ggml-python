import ggml
import ggml.utils
import numpy as np

init_params = ggml.ggml_init_params(
    mem_size=1024,
    no_alloc=False,
)

context = ggml.ggml_init(init_params)

old_shape = (0, 3, 4)
new_shape = np.array([3, 4, 0], dtype=np.int32)

temp_a = np.empty(old_shape, dtype=np.dtype("float32"))
x = temp_a.reshape(new_shape)
x_t = ggml.utils.from_numpy(x, context)
