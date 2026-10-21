import importlib.util
import sys
import types
from pathlib import Path
from types import SimpleNamespace


def test_slice_tensor_preserves_explicit_zero_stop(monkeypatch):
    # Exercise the Python wrapper without loading a native ggml shared library.
    fake_ggml = types.ModuleType("ggml.ggml")
    type_names = (
        "F32",
        "F16",
        "Q4_0",
        "Q4_1",
        "Q5_0",
        "Q5_1",
        "Q8_0",
        "Q8_1",
        "Q2_K",
        "Q3_K",
        "Q4_K",
        "Q5_K",
        "Q6_K",
        "Q8_K",
        "I8",
        "I16",
        "I32",
        "I64",
        "F64",
        "BF16",
    )
    for value, name in enumerate(type_names):
        setattr(fake_ggml, f"GGML_TYPE_{name}", value)

    fake_package = types.ModuleType("ggml")
    fake_package.__path__ = []
    fake_package.ggml = fake_ggml
    monkeypatch.setitem(sys.modules, "ggml", fake_package)
    monkeypatch.setitem(sys.modules, "ggml.ggml", fake_ggml)

    utils_path = Path(__file__).resolve().parents[1] / "ggml" / "utils.py"
    spec = importlib.util.spec_from_file_location("ggml_utils_slice_test", utils_path)
    utils = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(utils)

    tensor = SimpleNamespace(
        contents=SimpleNamespace(ne=[4, 1, 1, 1], nb=[4, 16, 16, 16])
    )
    fake_ggml.ggml_n_dims = lambda _: 1
    views = []

    def view_1d(_, __, ne0, offset):
        views.append((ne0, offset))
        return object()

    fake_ggml.ggml_view_1d = view_1d
    utils.slice_tensor(None, tensor, [slice(0, 0)])

    assert views == [(0, 0)]
