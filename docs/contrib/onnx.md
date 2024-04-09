# GGML ONNX Runtime

## Getting Started

### Installation

```bash
pip install ggml-python[onnx]
```

### Usage

```python
import onnx
from ggml.contrib.onnx import GgmlRuntimeBackend

# Load an ONNX model
model = onnx.load("model.onnx")

# Create a runtime session
ggml_backend_rep = GgmlRuntimeBackend.prepare(model)

# Run inference
input = np.random.randn(1, 3, 224, 224).astype(np.float32)
output = ggml_backend_rep.run([input])
```

## Technical Overview

The GGML ONNX runtime is a backend for the [ONNX](https://onnx.ai/) model format. It is designed to be used as a drop-in replacement for the ONNX Runtime which leverages ggml for efficient model inference on a wide range of devices.

To use the runtime:

- Models are first converted from PyTorch, TensorFlow, and other frameworks to ONNX
- ONNX models are then optimized for ggml inference. This includes:
    - Weight Quantization
    - Dynamic Subgraph Detection
    - GPU Offloading
- The optimized ONNX models are then executed in the GGML ONNX runtime


## Operator Support

This table is generated from [`operator_sets.h`](https://github.com/onnx/onnx/blob/main/onnx/defs/operator_sets.h) and may not include all ONNX operators. These are core operators available in all versions starting from ai.onnx version 1.

| ONNX Operator | Status | Implementation Method |
|:--------------------------------------------------------------------------------------------------|:------------------:|:----------------|
| [Abs](https://github.com/onnx/onnx/blob/main/docs/Operators.md#Abs)                               | :white_check_mark: | `ggml_abs`       |
| [Add](https://github.com/onnx/onnx/blob/main/docs/Operators.md#Add)                               | :white_check_mark: | `ggml_add`       |
| [And](https://github.com/onnx/onnx/blob/main/docs/Operators.md#And)                               | :white_check_mark: |                  |
| [ArgMax](https://github.com/onnx/onnx/blob/main/docs/Operators.md#ArgMax)                         | :white_check_mark: |                  |
| [ArgMin](https://github.com/onnx/onnx/blob/main/docs/Operators.md#ArgMin)                         | :white_check_mark: |                  |
| [AveragePool](https://github.com/onnx/onnx/blob/main/docs/Operators.md#AveragePool)               |                    |                  |
| [BatchNormalization](https://github.com/onnx/onnx/blob/main/docs/Operators.md#BatchNormalization) |                    |                  |
| [Cast](https://github.com/onnx/onnx/blob/main/docs/Operators.md#Cast)                             | :white_check_mark: |                  |
| [Ceil](https://github.com/onnx/onnx/blob/main/docs/Operators.md#Ceil)                             | :white_check_mark: |                  |
| [Clip](https://github.com/onnx/onnx/blob/main/docs/Operators.md#Clip)                             |                    |                  |
| [Concat](https://github.com/onnx/onnx/blob/main/docs/Operators.md#Concat)                         | :white_check_mark: | `ggml_concat`    |
| [Constant](https://github.com/onnx/onnx/blob/main/docs/Operators.md#Constant)                     | :white_check_mark: |                  |
| [ConstantOfShape](https://github.com/onnx/onnx/blob/main/docs/Operators.md#ConstantOfShape)       | :white_check_mark: |                  |
| [Conv](https://github.com/onnx/onnx/blob/main/docs/Operators.md#Conv)                             |  ⚙️ (in progress)   |                  |
| [ConvTranspose](https://github.com/onnx/onnx/blob/main/docs/Operators.md#ConvTranspose)           |  ⚙️ (in progress)   |                   |
| [DepthToSpace](https://github.com/onnx/onnx/blob/main/docs/Operators.md#DepthToSpace)             | :white_check_mark: |                  |
| [Div](https://github.com/onnx/onnx/blob/main/docs/Operators.md#Div)                               | :white_check_mark: | `ggml_div`       |
| [Dropout](https://github.com/onnx/onnx/blob/main/docs/Operators.md#Dropout)                       |  ⚙️ (in progress)   |                   |
| [Elu](https://github.com/onnx/onnx/blob/main/docs/Operators.md#Elu)                               | :white_check_mark: | `ggml_elu`       |
| [Equal](https://github.com/onnx/onnx/blob/main/docs/Operators.md#Equal)                           | :white_check_mark: |                  |
| [Exp](https://github.com/onnx/onnx/blob/main/docs/Operators.md#Exp)                               | :white_check_mark: |                  |
| [Flatten](https://github.com/onnx/onnx/blob/main/docs/Operators.md#Flatten)                       | :white_check_mark: |                  |
| [Floor](https://github.com/onnx/onnx/blob/main/docs/Operators.md#Floor)                           | :white_check_mark: |                  |
| [GRU](https://github.com/onnx/onnx/blob/main/docs/Operators.md#GRU)                               |       :x:          |                  |
| [Gather](https://github.com/onnx/onnx/blob/main/docs/Operators.md#Gather)                         | :white_check_mark: |                  |
| [Gemm](https://github.com/onnx/onnx/blob/main/docs/Operators.md#Gemm)                             | :white_check_mark: |                  |
| [GlobalAveragePool](https://github.com/onnx/onnx/blob/main/docs/Operators.md#GlobalAveragePool)   |                    |                  |
| [GlobalLpPool](https://github.com/onnx/onnx/blob/main/docs/Operators.md#GlobalLpPool)             |                    |                  |
| [GlobalMaxPool](https://github.com/onnx/onnx/blob/main/docs/Operators.md#GlobalMaxPool)           |                    |                  |
| [Greater](https://github.com/onnx/onnx/blob/main/docs/Operators.md#Greater)                       | :white_check_mark: |                  |
| [HardSigmoid](https://github.com/onnx/onnx/blob/main/docs/Operators.md#HardSigmoid)               | :white_check_mark: |                  |
| [Hardmax](https://github.com/onnx/onnx/blob/main/docs/Operators.md#Hardmax)                       | :white_check_mark: |                  |
| [Identity](https://github.com/onnx/onnx/blob/main/docs/Operators.md#Identity)                     | :white_check_mark: |                  |
| [If](https://github.com/onnx/onnx/blob/main/docs/Operators.md#If)                                 |       :x:          |                  |
| [InstanceNormalization](https://github.com/onnx/onnx/blob/main/docs/Operators.md#InstanceNormalization)| :white_check_mark: |             |
| [LRN](https://github.com/onnx/onnx/blob/main/docs/Operators.md#LRN)                               | :white_check_mark: |                  |
| [LSTM](https://github.com/onnx/onnx/blob/main/docs/Operators.md#LSTM)                             |       :x:          |                  |
| [LeakyRelu](https://github.com/onnx/onnx/blob/main/docs/Operators.md#LeakyRelu)                   |         ⚙️          |                  |
| [Less](https://github.com/onnx/onnx/blob/main/docs/Operators.md#Less)                             | :white_check_mark: |                  |
| [Log](https://github.com/onnx/onnx/blob/main/docs/Operators.md#Log)                               | :white_check_mark: | `ggml_log`       |
| [LogSoftmax](https://github.com/onnx/onnx/blob/main/docs/Operators.md#LogSoftmax)                 | :white_check_mark: |                  |
| [Loop](https://github.com/onnx/onnx/blob/main/docs/Operators.md#Loop)                             |       :x:          |                  |
| [LpNormalization](https://github.com/onnx/onnx/blob/main/docs/Operators.md#LpNormalization)       |:x: (Test case not provided)|                  |
| [LpPool](https://github.com/onnx/onnx/blob/main/docs/Operators.md#LpPool)                         |                    |                  |
| [MatMul](https://github.com/onnx/onnx/blob/main/docs/Operators.md#MatMul)                         | :white_check_mark: | `ggml_mul_mat`   |
| [Max](https://github.com/onnx/onnx/blob/main/docs/Operators.md#Max)                               | :white_check_mark: | `ggml_max`       |
| [MaxPool](https://github.com/onnx/onnx/blob/main/docs/Operators.md#MaxPool)                       |                    |`ggml.ggml_pool_2d`|
| [MaxRoiPool](https://github.com/onnx/onnx/blob/main/docs/Operators.md#MaxRoiPool)                 |                    |                  |
| [Mean](https://github.com/onnx/onnx/blob/main/docs/Operators.md#Mean)                             | :white_check_mark: |~~`ggml_mean`~~<br />`ggml_add` + `ggml_div`|
| [Min](https://github.com/onnx/onnx/blob/main/docs/Operators.md#Min)                               | :white_check_mark: |                  |
| [Mul](https://github.com/onnx/onnx/blob/main/docs/Operators.md#Mul)                               | :white_check_mark: | `ggml_mul`       |
| [Neg](https://github.com/onnx/onnx/blob/main/docs/Operators.md#Neg)                               | :white_check_mark: | `ggml_neg`       |
| [Not](https://github.com/onnx/onnx/blob/main/docs/Operators.md#Not)                               | :white_check_mark: |                  |
| [Or](https://github.com/onnx/onnx/blob/main/docs/Operators.md#Or)                                 | :white_check_mark: |                  |
| [PRelu](https://github.com/onnx/onnx/blob/main/docs/Operators.md#PRelu)                           | :white_check_mark: |                  |
| [Pad](https://github.com/onnx/onnx/blob/main/docs/Operators.md#Pad)                               |                    |                  |
| [Pow](https://github.com/onnx/onnx/blob/main/docs/Operators.md#Pow)                               | :white_check_mark: |                  |
| [RNN](https://github.com/onnx/onnx/blob/main/docs/Operators.md#RNN)                               |       :x:          |                  |
| [RandomNormal](https://github.com/onnx/onnx/blob/main/docs/Operators.md#RandomNormal)             |:x: (Test case not provided)|                  |
| [RandomNormalLike](https://github.com/onnx/onnx/blob/main/docs/Operators.md#RandomNormalLike)     |:x: (Test case not provided)|                  |
| [RandomUniform](https://github.com/onnx/onnx/blob/main/docs/Operators.md#RandomUniform)           |:x: (Test case not provided)|                  |
| [RandomUniformLike](https://github.com/onnx/onnx/blob/main/docs/Operators.md#RandomUniformLike)   |:x: (Test case not provided)|                  |
| [Reciprocal](https://github.com/onnx/onnx/blob/main/docs/Operators.md#Reciprocal)                 | :white_check_mark: |                  |
| [ReduceL1](https://github.com/onnx/onnx/blob/main/docs/Operators.md#ReduceL1)                     | :white_check_mark: |                  |
| [ReduceL2](https://github.com/onnx/onnx/blob/main/docs/Operators.md#ReduceL2)                     | :white_check_mark: |                  |
| [ReduceLogSum](https://github.com/onnx/onnx/blob/main/docs/Operators.md#ReduceLogSum)             | :white_check_mark: |                  |
| [ReduceLogSumExp](https://github.com/onnx/onnx/blob/main/docs/Operators.md#ReduceLogSumExp)       |        ⚙️           |                  |
| [ReduceMax](https://github.com/onnx/onnx/blob/main/docs/Operators.md#ReduceMax)                   | :white_check_mark: |                  |
| [ReduceMean](https://github.com/onnx/onnx/blob/main/docs/Operators.md#ReduceMean)                 | :white_check_mark: |                  |
| [ReduceMin](https://github.com/onnx/onnx/blob/main/docs/Operators.md#ReduceMin)                   | :white_check_mark: |                  |
| [ReduceProd](https://github.com/onnx/onnx/blob/main/docs/Operators.md#ReduceProd)                 | :white_check_mark: |                  |
| [ReduceSum](https://github.com/onnx/onnx/blob/main/docs/Operators.md#ReduceSum)                   | :white_check_mark: |                  |
| [ReduceSumSquare](https://github.com/onnx/onnx/blob/main/docs/Operators.md#ReduceSumSquare)       | :white_check_mark: |                  |
| [Relu](https://github.com/onnx/onnx/blob/main/docs/Operators.md#Relu)                             | :white_check_mark: | `ggml_relu`      |
| [Reshape](https://github.com/onnx/onnx/blob/main/docs/Operators.md#Reshape)                       | :white_check_mark: | `ggml_reshape`   |
| [Selu](https://github.com/onnx/onnx/blob/main/docs/Operators.md#Selu)                             | :white_check_mark: |                  |
| [Shape](https://github.com/onnx/onnx/blob/main/docs/Operators.md#Shape)                           | :white_check_mark: |                  |
| [Sigmoid](https://github.com/onnx/onnx/blob/main/docs/Operators.md#Sigmoid)                       | :white_check_mark: |                  |
| [Size](https://github.com/onnx/onnx/blob/main/docs/Operators.md#Size)                             | :white_check_mark: |                  |
| [Slice](https://github.com/onnx/onnx/blob/main/docs/Operators.md#Slice)                           |                    |                  |
| [Softmax](https://github.com/onnx/onnx/blob/main/docs/Operators.md#Softmax)                       | :white_check_mark: | `ggml_soft_max`  |
| [Softplus](https://github.com/onnx/onnx/blob/main/docs/Operators.md#Softplus)                     | :white_check_mark: |                  |
| [Softsign](https://github.com/onnx/onnx/blob/main/docs/Operators.md#Softsign)                     | :white_check_mark: |                  |
| [SpaceToDepth](https://github.com/onnx/onnx/blob/main/docs/Operators.md#SpaceToDepth)             | :white_check_mark: |                  |
| [Split](https://github.com/onnx/onnx/blob/main/docs/Operators.md#Split)                           | :white_check_mark: |                  |
| [Sqrt](https://github.com/onnx/onnx/blob/main/docs/Operators.md#Sqrt)                             | :white_check_mark: | `ggml_sqrt`      |
| [Squeeze](https://github.com/onnx/onnx/blob/main/docs/Operators.md#Squeeze)                       | :white_check_mark: |                  |
| [Sub](https://github.com/onnx/onnx/blob/main/docs/Operators.md#Sub)                               | :white_check_mark: | `ggml_sub`       |
| [Sum](https://github.com/onnx/onnx/blob/main/docs/Operators.md#Sum)                               | :white_check_mark: | `ggml_sum`       |
| [Tanh](https://github.com/onnx/onnx/blob/main/docs/Operators.md#Tanh)                             | :white_check_mark: | `ggml_tanh`      |
| [Tile](https://github.com/onnx/onnx/blob/main/docs/Operators.md#Tile)                             | :white_check_mark: |                  |
| [TopK](https://github.com/onnx/onnx/blob/main/docs/Operators.md#TopK)                             | :white_check_mark: |                  |
| [Transpose](https://github.com/onnx/onnx/blob/main/docs/Operators.md#Transpose)                   | :white_check_mark: | `ggml_transpose` |
| [Unsqueeze](https://github.com/onnx/onnx/blob/main/docs/Operators.md#Unsqueeze)                   | :white_check_mark: |                  |
| ~~[Upsample](https://github.com/onnx/onnx/blob/main/docs/Operators.md#Upsample)~~                 |  :x: (Deprecated)  |                  |
| [Xor](https://github.com/onnx/onnx/blob/main/docs/Operators.md#Xor)                               | :white_check_mark: |                  |

## Acknowledgements

The GGML ONNX runtime is built on top of the [ONNX](https://onnx.ai/) and [GGML](ggml.ai)

The core of the runtime was written by Andrei Betlen (@abetlen), David Miller (@dmille), and 
Mohammadreza Anvari (@mrezanvari)

This work would also not be possible without the ggml community, in particular @slaren for their work on the ggml backends and memory allocation api.
