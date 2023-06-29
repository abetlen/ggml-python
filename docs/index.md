---
title: Getting Started
---

## Introduction

ggml-python is a python library for working with [ggml](https://github.com/ggerganov/ggml).

ggml is a tensor library for machine learning developed by Georgi Gerganov, the library has been used to run models like Whisper and LLaMa on a wide range of devices.
ggml is written in C/C++ and is designed to be fast, portable and easily embeddable; making use of various hardware acceleration systems like BLAS, CUDA, OpenCL, and Metal.
ggml supports quantized inference for reduced memory footprint and faster inference.

You can use ggml-python to:

- Convert and quantize model weights from Python-based ML frameworks (Pytorch, Tensorflow, etc) to ggml.
- Port existing ML models to ggml and run them from Python.

## Installation

Requirements

- Python 3.7+
- C compiler (gcc, clang, msvc, etc)

You can install `ggml-python` using `pip`:

```bash
pip install ggml-python
```

This will compile ggml using cmake which requires a c compiler installed on your system.

Below are the available options for building ggml-python with additional options for optimized inference.

=== "**BLAS**"

    ```bash
    CMAKE_ARGS="-DGGML_OPENBLAS=ON" pip install ggml-python
    ```

=== "**CUDA**"

    ```bash
    CMAKE_ARGS="-DGGML_CUBLAS=ON" pip install ggml-python
    ```

=== "**Metal**"

    ```bash
    CMAKE_ARGS="-DGGML_METAL=ON" pip install ggml-python
    ```

=== "**OpenCL**"

    ```bash
    CMAKE_ARGS="-DGGML_CLBLAST=ON" pip install ggml-python
    ```

## Basic Example

Below is a simple example of using ggml-python low level api to compute the value of a function.

```python
import ggml
import ctypes

# Allocate a new context with 16 MB of memory
params = ggml.ggml_init_params(mem_size=16 * 1024 * 1024, mem_buffer=None)
ctx = ggml.ggml_init(params=params)

# Instantiate tensors
x = ggml.ggml_new_tensor_1d(ctx, ggml.GGML_TYPE_F32, 1)
a = ggml.ggml_new_tensor_1d(ctx, ggml.GGML_TYPE_F32, 1)
b = ggml.ggml_new_tensor_1d(ctx, ggml.GGML_TYPE_F32, 1)

# Use ggml operations to build a computational graph
x2 = ggml.ggml_mul(ctx, x, x)
f = ggml.ggml_add(ctx, ggml.ggml_mul(ctx, a, x2), b)

gf = ggml.ggml_build_forward(f)

# Set the input values
ggml.ggml_set_f32(x, 2.0)
ggml.ggml_set_f32(a, 3.0)
ggml.ggml_set_f32(b, 4.0)

# Compute the graph
ggml.ggml_graph_compute(ctx, ctypes.pointer(gf))

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