# Python bindings for [`ggml`](https://github.com/ggerganov/ggml)

[![Documentation Status](https://readthedocs.org/projects/ggml-python/badge/?version=latest)](https://ggml-python.readthedocs.io/en/latest/?badge=latest)
[![Tests](https://github.com/abetlen/ggml-python/actions/workflows/test.yaml/badge.svg)](https://github.com/abetlen/ggml-python/actions/workflows/test.yaml)
[![PyPI](https://img.shields.io/pypi/v/ggml-python)](https://pypi.org/project/ggml-python/)
[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/ggml-python)](https://pypi.org/project/ggml-python/)
[![PyPI - License](https://img.shields.io/pypi/l/ggml-python)](https://pypi.org/project/ggml-python/)
[![PyPI - Downloads](https://img.shields.io/pypi/dm/ggml-python)](https://pypi.org/project/ggml-python/)


Python bindings for the [`ggml`](https://github.com/ggerganov/ggml) tensor library for machine learning.

> **⚠️ This project is in a very early state and currently only offers the basic low-level bindings to ggml**

# Documentation

- [Getting Started](https://ggml-python.readthedocs.io/en/latest/)
- [API Reference](https://ggml-python.readthedocs.io/en/latest/api-reference/)
- [Examples](https://github.com/abetlen/ggml-python/tree/main/examples)

# Installation


Requirements
- Python 3.7+
- C compiler (gcc, clang, msvc, etc)

You can install `ggml-python` using `pip`:

```bash
pip install ggml-python
```

This will compile ggml using cmake which requires a c compiler installed on your system.
To build ggml with specific features (ie. OpenBLAS, GPU Support, etc) you can pass specific flags through the `CMAKE_ARGS` environment variable. For example to install ggml-python with cuBLAS support you can run:

```bash
CMAKE_ARGS="-DGGML_CUBLAS=ON" pip install ggml-python
```

## Options

| Option | Description | Default |
| --- | --- | --- |
| `GGML_CUBLAS` | Enable cuBLAS support | `OFF` |
| `GGML_CLBLAST` | Enable CLBlast support | `OFF` |
| `GGML_OPENBLAS` | Enable OpenBLAS support | `OFF` |

# Usage

```python
import ggml
import ctypes

# Allocate a new context with 16 MB of memory
params = ggml.ggml_init_params(
    mem_size=16 * 1024 * 1024,
    mem_buffer=None
)
ctx = ggml.ggml_init(params=params)

# Instantiate tensors
x = ggml.ggml_new_tensor_1d(
    ctx,
    ggml.GGML_TYPE_F32,
    ctypes.c_int64(1),
)
a = ggml.ggml_new_tensor_1d(
    ctx,
    ggml.GGML_TYPE_F32,
    ctypes.c_int64(1),
)
b = ggml.ggml_new_tensor_1d(
    ctx,
    ggml.GGML_TYPE_F32,
    ctypes.c_int64(1),
)

# Use ggml operations to build a computational graph
x2 = ggml.ggml_mul(ctx, x, x)
f = ggml.ggml_add(ctx, ggml.ggml_mul(ctx, a, x2), b)

gf = ggml.ggml_build_forward(f)

# Set the input values
ggml.ggml_set_f32(x, ctypes.c_float(2.0))
ggml.ggml_set_f32(a, ctypes.c_float(3.0))
ggml.ggml_set_f32(b, ctypes.c_float(4.0))

# Compute the graph
ggml.ggml_graph_compute(ctx, ctypes.pointer(gf))

# Get the output value
output = ggml.ggml_get_f32_1d(f, ctypes.c_int(0))
assert output == 16.0

# Free the context
ggml.ggml_free(ctx)
```

# Troubleshooting

If you are having trouble installing `ggml-python` or activating specific features please try to install it with the `--verbose` and `--no-cache-dir` flags to get more information about any issues:

```bash
[options] pip install ggml-python --verbose --no-cache-dir --force-reinstall --upgrade
```

# License

This project is licensed under the terms of the MIT license.
