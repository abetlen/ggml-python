# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

- feat: add graph allocator reserve size binding by @abetlen in #159
- feat(gguf): add buffer and file pointer APIs by @aisk in #158
- fix: sync updated GGUF API signatures by @aisk in #155
- feat: update vendored ggml to 0.15.1 by @abetlen in #157
- feat(contrib): map additional ggml forward ops to onnx backend by @abetlen in #146
- fix(ci): run CUDA auditwheel from conda environment by @abetlen in #145
- fix(ci): install auditwheel in accelerator repair steps by @abetlen in #144
- fix(ci): repair accelerator wheel release dispatch by @abetlen in #143

## [0.0.41]

- fix(build): reduce duplicate Linux wheel library files by @abetlen in #141
- fix(build): default GGML_METAL to OFF to match docs by @abetlen in #140
- fix(ci): repair Linux CUDA wheel tags by @abetlen in #138
- fix(ci): repair Linux ROCm wheel tags by @abetlen in #139
- fix(ci): build Linux Vulkan wheels on a manylinux baseline by @abetlen in #137

## [0.0.40]

- fix(ci): rebuild wheel index after release asset workflows by @abetlen in #133
- ci: update GitHub Actions workflow dependencies by @abetlen in #134
- fix(ci): repair release wheel builds for cibuildwheel v3 by @abetlen in #135
- feat(contrib): add ONNX backend by @dmille, @mrezanvari, and @abetlen in #23
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
- feat: add ZenDNN backend bindings by @abetlen in #128
- feat: add zDNN backend bindings by @abetlen in #129
- feat: add CPU conversion bindings by @abetlen in #130
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
