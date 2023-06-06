# Python bindings for [`ggml`](https://github.com/ggerganov/ggml)

[![PyPI](https://img.shields.io/pypi/v/ggml-python)](https://pypi.org/project/ggml-python/)
[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/ggml-python)](https://pypi.org/project/ggml-python/)
[![PyPI - License](https://img.shields.io/pypi/l/ggml-python)](https://pypi.org/project/ggml-python/)
[![PyPI - Downloads](https://img.shields.io/pypi/dm/ggml-python)](https://pypi.org/project/ggml-python/)


Python bindings for the [`ggml`](https://github.com/ggerganov/ggml) tensor library for machine learning.

> **⚠️ This project is in a very early state and currently only offers the basic low-level bindings to ggml**

# Installation

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

## Troubleshooting

If you are having trouble installing `ggml-python` or activating specific features please try to install it with the `--verbose` and `--no-cache-dir` flags to get more information about any issues:

```bash
[options] pip install ggml-python --verbose --no-cache-dir --force-reinstall --upgrade
```