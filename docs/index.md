---
title: Getting Started
---

## Introduction

ggml-python is a python library for working with [ggml](https://github.com/ggerganov/ggml).

ggml is a tensor library for machine learning developed by Georgi Gerganov.
The library is written in C/C++ and is designed to be fast, portable and easily embeddable.
It offers efficient implementations of various tensor operations in SIMD, BLAS, CUDA, OpenCL, and Metal.
It supports quantized inference for reduced memory footprint and faster inference.

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


## API Reference

::: ggml.ggml
    options:
        show_root_heading: true


::: ggml.utils
    options:
        show_root_heading: true