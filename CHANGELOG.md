# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

- feat: add missing ggml.h bindings by @abetlen in #118
- feat: add CUDA backend bindings by @abetlen in #119
- feat: add RPC backend bindings by @abetlen in #120
- feat: add Vulkan backend registry binding by @abetlen in #121
- feat: add BLAS backend bindings by @abetlen in #122
- feat: add CANN backend bindings by @abetlen in #123
- feat: add SYCL backend bindings by @abetlen in #124
- feat: add OpenVINO backend bindings by @abetlen in #125
- feat: add Hexagon backend bindings by @abetlen in #126
- feat: add WebGPU backend bindings by @abetlen in #127
- feat: add arithmetic op bindings by @aisk in #114
- fix: detect current optional backend symbols by @abetlen in #117
- ci: increase Windows HIP SDK installer timeout by @abetlen in #112
- ci: add tag triggers for release publishing workflows by @abetlen in #113

## [0.0.39]

- ci: add Vulkan wheel builds by @abetlen in #108
- ci: add ROCm and Windows HIP wheel builds by @abetlen in #109
- feat: add tensor layout helper bindings by @abetlen in #106
- feat: add ggml_is_contiguous_{0,1,2} by @aisk in #95

## [0.0.38]

- ci: update pre-built wheels to py3-none and add CUDA 11.8-13.2 support by @abetlen in #104
- feat: Update ggml to ggml-org/ggml@1e33fed3 and sync Python bindings by @abetlen in #100
