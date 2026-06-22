# Python bindings for [`ggml`](https://github.com/ggml-org/ggml)

[![Documentation Status](https://readthedocs.org/projects/ggml-python/badge/?version=latest)](https://ggml-python.readthedocs.io/en/latest/?badge=latest)
[![Tests](https://github.com/abetlen/ggml-python/actions/workflows/test.yaml/badge.svg)](https://github.com/abetlen/ggml-python/actions/workflows/test.yaml)
[![PyPI](https://img.shields.io/pypi/v/ggml-python)](https://pypi.org/project/ggml-python/)
[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/ggml-python)](https://pypi.org/project/ggml-python/)
[![PyPI - License](https://img.shields.io/pypi/l/ggml-python)](https://pypi.org/project/ggml-python/)
[![PyPI - Downloads](https://img.shields.io/pypi/dm/ggml-python)](https://pypi.org/project/ggml-python/)


Python bindings for the [`ggml`](https://github.com/ggml-org/ggml) tensor library for machine learning.

> ⚠️ Neither this project nor `ggml` currently guarantee backwards-compatibility, if you are using this library in other applications I strongly recommend pinning to specific releases in your `requirements.txt` file.

# Documentation

- [Getting Started](https://ggml-python.readthedocs.io/en/latest/)
- [API Reference](https://ggml-python.readthedocs.io/en/latest/api-reference/)
- [Examples](https://github.com/abetlen/ggml-python/tree/main/examples)

# Installation


Requirements
- Python 3.8+
- C compiler (gcc, clang, msvc, etc)

You can install `ggml-python` using `pip`:

```bash
pip install ggml-python
```

## Pre-built Wheels

It is also possible to install a pre-built wheel with basic CPU support:

```bash
pip install ggml-python \
  --extra-index-url https://abetlen.github.io/ggml-python/whl/cpu
```

Pre-built CUDA wheels are available for CUDA 11.8, 12.1, 12.2, 12.3, 12.4, 12.5, 13.0, and 13.2:

```bash
pip install ggml-python \
  --extra-index-url https://abetlen.github.io/ggml-python/whl/<cuda-version>
```

Where `<cuda-version>` is one of the following:

- `cu118`: CUDA 11.8
- `cu121`: CUDA 12.1
- `cu122`: CUDA 12.2
- `cu123`: CUDA 12.3
- `cu124`: CUDA 12.4
- `cu125`: CUDA 12.5
- `cu130`: CUDA 13.0
- `cu132`: CUDA 13.2

For example, to install the CUDA 12.1 wheel:

```bash
pip install ggml-python \
  --extra-index-url https://abetlen.github.io/ggml-python/whl/cu121
```

Pre-built Metal wheels are available for Apple Silicon macOS:

```bash
pip install ggml-python \
  --extra-index-url https://abetlen.github.io/ggml-python/whl/metal
```

Pre-built Vulkan wheels are available for Linux and Windows:

```bash
pip install ggml-python \
  --extra-index-url https://abetlen.github.io/ggml-python/whl/vulkan
```

Pre-built ROCm wheels are available for Linux x86_64 with ROCm 7.2:

```bash
pip install ggml-python \
  --extra-index-url https://abetlen.github.io/ggml-python/whl/rocm72
```

Pre-built HIP Radeon wheels are available for Windows x86_64:

```bash
pip install ggml-python \
  --extra-index-url https://abetlen.github.io/ggml-python/whl/hip-radeon
```

Pre-built Pyodide wheels are available for browser runtimes:

```python
import micropip

await micropip.install(["numpy", "typing_extensions"])
await micropip.install(
    "ggml-python",
    deps=False,
    index_urls=["https://abetlen.github.io/ggml-python/whl/cpu"],
)
```

When installing from source, pip compiles ggml with CMake and requires a C compiler installed on your system.
To build ggml with specific features (ie. OpenBLAS, GPU Support, etc) you can pass specific cmake options through the `cmake.args` pip install configuration setting. For example to install ggml-python with cuBLAS support you can run:

```bash
pip install --upgrade pip
pip install ggml-python --config-settings=cmake.args='-DGGML_CUDA=ON'
```

## Options

| Option | Description | Default |
| --- | --- | --- |
| `GGML_CUDA` | Enable cuBLAS support | `OFF` |
| `GGML_HIP` | Enable HIP / ROCm support | `OFF` |
| `GGML_OPENCL` | Enable OpenCL support | `OFF` |
| `GGML_BLAS` | Enable BLAS support | `OFF` |
| `GGML_BLAS_VENDOR` | Select BLAS vendor, for example `OpenBLAS` | unset |
| `GGML_METAL` | Enable Metal support | `OFF` |
| `GGML_VULKAN` | Enable Vulkan support | `OFF` |
| `GGML_RPC` | Enable RPC support | `OFF` |

# Usage

You can also try this example in the browser using the [ggml-python playground](https://ggml-python.readthedocs.io/en/latest/playground/).

```python
import ggml
import ctypes

# Allocate a new context with 16 MB of memory
params = ggml.ggml_init_params(mem_size=16 * 1024 * 1024, mem_buffer=None)
ctx = ggml.ggml_init(params)

# Instantiate tensors
x = ggml.ggml_new_tensor_1d(ctx, ggml.GGML_TYPE_F32, 1)
a = ggml.ggml_new_tensor_1d(ctx, ggml.GGML_TYPE_F32, 1)
b = ggml.ggml_new_tensor_1d(ctx, ggml.GGML_TYPE_F32, 1)

# Use ggml operations to build a computational graph
x2 = ggml.ggml_mul(ctx, x, x)
f = ggml.ggml_add(ctx, ggml.ggml_mul(ctx, a, x2), b)

gf = ggml.ggml_new_graph(ctx)
ggml.ggml_build_forward_expand(gf, f)

# Set the input values
ggml.ggml_set_f32(x, 2.0)
ggml.ggml_set_f32(a, 3.0)
ggml.ggml_set_f32(b, 4.0)

# Compute the graph
ggml.ggml_graph_compute_with_ctx(ctx, gf, 1)

# Get the output value
output = ggml.ggml_get_f32_1d(f, 0)
assert output == 16.0

# Free the context
ggml.ggml_free(ctx)
```

# Troubleshooting

If you are having trouble installing `ggml-python` or activating specific features please try to install it with the `--verbose` and `--no-cache-dir` flags to get more information about any issues:

```bash
pip install ggml-python --verbose --no-cache-dir --force-reinstall --upgrade
```

## Native Crashes

If you are debugging a `SIGSEGV`, `SIGABRT`, or `Aborted (core dumped)` failure, reproduce it with a debug build so native symbols are available.

```bash
git clone https://github.com/abetlen/ggml-python.git
cd ggml-python
make build.debug
gdb --args python3 your_script.py
```

Inside `gdb`, use `run` to start the script, `bt` to inspect the native backtrace, and `py-bt` to inspect the Python backtrace when Python debug symbols are available.

You can also use Python's built-in `breakpoint()` to stop before the native call that crashes.

# License

This project is licensed under the terms of the MIT license.
