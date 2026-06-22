---
title: Getting Started
---

## Introduction

ggml-python is a python library for working with [ggml](https://github.com/ggml-org/ggml).

ggml is a tensor library for machine learning developed by Georgi Gerganov, the library has been used to run models like Whisper and LLaMa on a wide range of devices.
ggml is written in C/C++ and is designed to be fast, portable and easily embeddable; making use of various hardware acceleration systems like BLAS, CUDA, HIP / ROCm, OpenCL, and Metal.
ggml supports quantized inference for reduced memory footprint and faster inference.

You can use ggml-python to:

- Convert and quantize model weights from Python-based ML frameworks (Pytorch, Tensorflow, etc) to ggml.
- Port existing ML models to ggml and run them from Python.

## Installation

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

When installing from source, pip compiles ggml with CMake and requires a C compiler installed on your system.

Below are the available options for building ggml-python with additional options for optimized inference.

=== "**BLAS**"

    ```bash
    CMAKE_ARGS="-DGGML_BLAS=ON -DGGML_BLAS_VENDOR=OpenBLAS" pip install ggml-python
    ```

=== "**CUDA**"

    ```bash
    CMAKE_ARGS="-DGGML_CUDA=ON" pip install ggml-python
    ```

=== "**HIP / ROCm**"

    ```bash
    CMAKE_ARGS="-DGGML_HIP=ON" pip install ggml-python
    ```

=== "**Metal**"

    ```bash
    CMAKE_ARGS="-DGGML_METAL=ON" pip install ggml-python
    ```

=== "**OpenCL**"

    ```bash
    CMAKE_ARGS="-DGGML_OPENCL=ON" pip install ggml-python
    ```

## Basic Example

Below is a simple example of using ggml-python low level api to compute the value of a function.
You can also try this example in the browser using the [ggml-python playground](playground.md).

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

## Next Steps

To learn more about ggml-python, check out the following resources:

- [API Reference](api-reference.md)
- Examples
    - [Code Completion Server](https://github.com/abetlen/ggml-python/tree/main/examples/replit) - A code completion server using ggml-python and the replit-code-v1-3b model that you can drop into your editor as a local Github Copilot replacement.
    - [CLIP Embeddings](https://github.com/abetlen/ggml-python/tree/main/examples/clip) - A simple example of using ggml-python to implement CLIP text / image embeddings.

## Development

```bash
git clone https://github.com/abetlen/ggml-python.git
cd ggml-python
# (Optional) Create a virtual environment
python -m venv venv
source venv/bin/activate
# Install dependencies
make build
```

## Contributing

If you would like to contribute to ggml-python, please open an issue or submit a pull request on [GitHub](https://github.com/abetlen/ggml-python).


## License

This project is licensed under the terms of the MIT license.
