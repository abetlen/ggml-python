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


def test_numpy_arrays():
    params = ggml.ggml_init_params(mem_size=16 * 1024 * 1024)
    with ggml.utils.ggml_context_manager(params) as ctx:
        x = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32, order="F")
        assert x.shape == (2, 3)
        t = ggml.utils.from_numpy(x, ctx)
        assert t.contents.ne[:2] == [3, 2]
        y = ggml.utils.to_numpy(t)
        assert y.shape == (2, 3)


def test_numpy_arrays_transposed():
    params = ggml.ggml_init_params(mem_size=16 * 1024 * 1024)
    with ggml.utils.ggml_context_manager(params) as ctx:
        # 2D
        x = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.int32)
        t = ggml.utils.from_numpy(x, ctx)
        t_t = ggml.ggml_transpose(ctx, t)
        x_t = ggml.utils.to_numpy(t_t)
        assert np.array_equal(x_t, x.T)

        t = ggml.utils.from_numpy(x.T, ctx)
        x_t = ggml.utils.to_numpy(t)
        assert np.array_equal(x.T, x_t)

        # 3D
        x = np.array(
            [[[1, 2], [3, 4], [5, 6]], [[7, 8], [9, 10], [11, 12]]], dtype=np.int32
        )
        t = ggml.utils.from_numpy(x, ctx)
        t_t = ggml.ggml_permute(ctx, t, 2, 1, 0, 3)
        x_t = ggml.utils.to_numpy(t_t)
        assert np.array_equal(x_t, x.T)

        t = ggml.utils.from_numpy(x.T, ctx)
        x_t = ggml.utils.to_numpy(t)
        assert np.array_equal(x.T, x_t)
