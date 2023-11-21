from ggml.experimental import CGraph, Tensor, GGML_TYPE

import pytest

import numpy as np


@pytest.mark.skip(reason="not implemented")
def test_tensor():
    x = np.ones((3,), dtype=np.float32)
    assert x.shape == (3,)
    t = Tensor.from_numpy(x)
    assert t.shape == (3,)
    assert t.ggml_type == GGML_TYPE.F32
    assert np.allclose(t.numpy(), x)

@pytest.mark.skip(reason="not implemented")
def test_tensor_compute():
    x = Tensor.from_numpy(np.array([2.0], dtype=np.float32))
    a = Tensor.from_numpy(np.array([3.0], dtype=np.float32))
    b = Tensor.from_numpy(np.array([4.0], dtype=np.float32))
    x2 = x * x
    f = a * x2 + b
    gf = CGraph.build_forward(f)
    gf.compute()
    assert np.allclose(f.numpy(), np.array([16.0], dtype=np.float32))
