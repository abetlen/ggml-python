import ggml
import ggml.utils

import numpy as np

def test_utils():
    params = ggml.ggml_init_params(mem_size=16 * 1024 * 1024)
    with ggml.utils.ggml_context_manager(params) as ctx:
        x = np.ones((3,), dtype=np.float32)
        assert x.shape == (3,)
        t = ggml.utils.from_numpy(x, ctx)
        assert t.contents.ne[:1] == [3]
        assert t.contents.type == ggml.GGML_TYPE_F32
        assert np.allclose(ggml.utils.to_numpy(t), x)
