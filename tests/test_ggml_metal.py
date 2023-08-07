import ggml
import ggml.utils
import ctypes
import pytest
import numpy as np

ggml_metal_available = ggml.GGML_USE_METAL

run_if_ggml_metal_available = pytest.mark.skipif(
    not ggml_metal_available,
    reason="METAL not available",
)


@run_if_ggml_metal_available
def test_metal():
    assert False
