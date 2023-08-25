# Operator Implementation Progress


| ONNX Operators | Implemented | ggml Equivalent |
|:--------------------------------------------------------------------------------------------------|:------------------:|:----------------:|
| [Abs](https://github.com/onnx/onnx/blob/main/docs/Operators.md#Abs)                               | :white_check_mark: | `ggml_abs`       |
| [Add](https://github.com/onnx/onnx/blob/main/docs/Operators.md#Add)                               | :white_check_mark: | `ggml_add`       |
| [And](https://github.com/onnx/onnx/blob/main/docs/Operators.md#And)                               | :white_check_mark: |                  |
| [ArgMax](https://github.com/onnx/onnx/blob/main/docs/Operators.md#ArgMax)                         |                    |                  |
| [ArgMin](https://github.com/onnx/onnx/blob/main/docs/Operators.md#ArgMin)                         |                    |                  |
| [AveragePool](https://github.com/onnx/onnx/blob/main/docs/Operators.md#AveragePool)               |                    |                  |
| [BatchNormalizatio](https://github.com/onnx/onnx/blob/main/docs/Operators.md#BatchNormalizatio)   |                    |                  |
| [Cast](https://github.com/onnx/onnx/blob/main/docs/Operators.md#Cast)                             | :white_check_mark: |                  |
| [Ceil](https://github.com/onnx/onnx/blob/main/docs/Operators.md#Ceil)                             |                    |                  |
| [Clip](https://github.com/onnx/onnx/blob/main/docs/Operators.md#Clip)                             |                    |                  |
| [Concat](https://github.com/onnx/onnx/blob/main/docs/Operators.md#Concat)                         | :white_check_mark: | `ggml_concat`    |
| [Constant](https://github.com/onnx/onnx/blob/main/docs/Operators.md#Constant)                     | :white_check_mark: |                  |
| [Conv](https://github.com/onnx/onnx/blob/main/docs/Operators.md#Conv)                             |                    |                  |
| [ConvTranspose](https://github.com/onnx/onnx/blob/main/docs/Operators.md#ConvTranspose)           |                    |                  |
| [DepthToSpace](https://github.com/onnx/onnx/blob/main/docs/Operators.md#DepthToSpace)             |                    |                  |
| [Div](https://github.com/onnx/onnx/blob/main/docs/Operators.md#Div)                               | :white_check_mark: | `ggml_div`       |
| [Dropout](https://github.com/onnx/onnx/blob/main/docs/Operators.md#Dropout)                       |                    |                  |
| [Elu](https://github.com/onnx/onnx/blob/main/docs/Operators.md#Elu)                               | :white_check_mark: | `ggml_elu`       |
| [Equal](https://github.com/onnx/onnx/blob/main/docs/Operators.md#Equal)                           | :white_check_mark: |                  |
| [Exp](https://github.com/onnx/onnx/blob/main/docs/Operators.md#Exp)                               | :white_check_mark: |                  |
| [Flatten](https://github.com/onnx/onnx/blob/main/docs/Operators.md#Flatten)                       |                    |                  |
| [Floor](https://github.com/onnx/onnx/blob/main/docs/Operators.md#Floor)                           |                    |                  |
| [GRU](https://github.com/onnx/onnx/blob/main/docs/Operators.md#GRU)                               |                    |                  |
| [Gather](https://github.com/onnx/onnx/blob/main/docs/Operators.md#Gather)                         | :white_check_mark: |                  |
| [Gemm](https://github.com/onnx/onnx/blob/main/docs/Operators.md#Gemm)                             |                    |                  |
| [GlobalAveragePool](https://github.com/onnx/onnx/blob/main/docs/Operators.md#GlobalAveragePool)   |                    |                  |
| [GlobalLpPool](https://github.com/onnx/onnx/blob/main/docs/Operators.md#GlobalLpPool)             |                    |                  |
| [GlobalMaxPool](https://github.com/onnx/onnx/blob/main/docs/Operators.md#GlobalMaxPool)           |                    |                  |
| [Greater](https://github.com/onnx/onnx/blob/main/docs/Operators.md#Greater)                       |                    |                  |
| [HardSigmoid](https://github.com/onnx/onnx/blob/main/docs/Operators.md#HardSigmoid)               |                    |                  |
| [Hardmax](https://github.com/onnx/onnx/blob/main/docs/Operators.md#Hardmax)                       |                    |                  |
| [Identity](https://github.com/onnx/onnx/blob/main/docs/Operators.md#Identity)                     |                    |                  |
| [If](https://github.com/onnx/onnx/blob/main/docs/Operators.md#If)                                 |                    |                  |
| [InstanceNormaliza](https://github.com/onnx/onnx/blob/main/docs/Operators.md#InstanceNormaliza)   |                    |                  |
| [LRN](https://github.com/onnx/onnx/blob/main/docs/Operators.md#LRN)                               |                    |                  |
| [LSTM](https://github.com/onnx/onnx/blob/main/docs/Operators.md#LSTM)                             |                    |                  |
| [LeakyRelu](https://github.com/onnx/onnx/blob/main/docs/Operators.md#LeakyRelu)                   |                    |                  |
| [Less](https://github.com/onnx/onnx/blob/main/docs/Operators.md#Less)                             |                    |                  |
| [Log](https://github.com/onnx/onnx/blob/main/docs/Operators.md#Log)                               | :white_check_mark: | `ggml_log`       |
| [LogSoftmax](https://github.com/onnx/onnx/blob/main/docs/Operators.md#LogSoftmax)                 | :white_check_mark: |                  |
| [Loop](https://github.com/onnx/onnx/blob/main/docs/Operators.md#Loop)                             |                    |                  |
| [LpNormalization](https://github.com/onnx/onnx/blob/main/docs/Operators.md#LpNormalization)       |                    |                  |
| [LpPool](https://github.com/onnx/onnx/blob/main/docs/Operators.md#LpPool)                         |                    |                  |
| [MatMul](https://github.com/onnx/onnx/blob/main/docs/Operators.md#MatMul)                         | :white_check_mark: | `ggml_mul_mat`   |
| [Max](https://github.com/onnx/onnx/blob/main/docs/Operators.md#Max)                               | :white_check_mark: | `ggml_max`       |
| [MaxPool](https://github.com/onnx/onnx/blob/main/docs/Operators.md#MaxPool)                       |                    |                  |
| [MaxRoiPool](https://github.com/onnx/onnx/blob/main/docs/Operators.md#MaxRoiPool)                 |                    |                  |
| [Mean](https://github.com/onnx/onnx/blob/main/docs/Operators.md#Mean)                             | :white_check_mark: |~~`ggml_mean`~~<br />`ggml_add` + `ggml_div`|
| [Min](https://github.com/onnx/onnx/blob/main/docs/Operators.md#Min)                               | :white_check_mark: |                  |
| [Mul](https://github.com/onnx/onnx/blob/main/docs/Operators.md#Mul)                               | :white_check_mark: | `ggml_mul`       |
| [Neg](https://github.com/onnx/onnx/blob/main/docs/Operators.md#Neg)                               | :white_check_mark: | `ggml_neg`       |
| [Not](https://github.com/onnx/onnx/blob/main/docs/Operators.md#Not)                               | :white_check_mark: |                  |
| [Or](https://github.com/onnx/onnx/blob/main/docs/Operators.md#Or)                                 | :white_check_mark: |                  |
| [PRelu](https://github.com/onnx/onnx/blob/main/docs/Operators.md#PRelu)                           |                    |                  |
| [Pad](https://github.com/onnx/onnx/blob/main/docs/Operators.md#Pad)                               |                    |                  |
| [Pow](https://github.com/onnx/onnx/blob/main/docs/Operators.md#Pow)                               | :white_check_mark: |                  |
| [RNN](https://github.com/onnx/onnx/blob/main/docs/Operators.md#RNN)                               |                    |                  |
| [RandomNormal](https://github.com/onnx/onnx/blob/main/docs/Operators.md#RandomNormal)             |                    |                  |
| [RandomNormalLike](https://github.com/onnx/onnx/blob/main/docs/Operators.md#RandomNormalLike)     |                    |                  |
| [RandomUniform](https://github.com/onnx/onnx/blob/main/docs/Operators.md#RandomUniform)           |                    |                  |
| [RandomUniformLike](https://github.com/onnx/onnx/blob/main/docs/Operators.md#RandomUniformLike)   |                    |                  |
| [Reciprocal](https://github.com/onnx/onnx/blob/main/docs/Operators.md#Reciprocal)                 |                    |                  |
| [ReduceL1](https://github.com/onnx/onnx/blob/main/docs/Operators.md#ReduceL1)                     |                    |                  |
| [ReduceL2](https://github.com/onnx/onnx/blob/main/docs/Operators.md#ReduceL2)                     |                    |                  |
| [ReduceLogSum](https://github.com/onnx/onnx/blob/main/docs/Operators.md#ReduceLogSum)             |                    |                  |
| [ReduceLogSumExp](https://github.com/onnx/onnx/blob/main/docs/Operators.md#ReduceLogSumExp)       |                    |                  |
| [ReduceMax](https://github.com/onnx/onnx/blob/main/docs/Operators.md#ReduceMax)                   | :white_check_mark: |                  |
| [ReduceMean](https://github.com/onnx/onnx/blob/main/docs/Operators.md#ReduceMean)                 | :white_check_mark: |                  |
| [ReduceMin](https://github.com/onnx/onnx/blob/main/docs/Operators.md#ReduceMin)                   |                    |                  |
| [ReduceProd](https://github.com/onnx/onnx/blob/main/docs/Operators.md#ReduceProd)                 |                    |                  |
| [ReduceSum](https://github.com/onnx/onnx/blob/main/docs/Operators.md#ReduceSum)                   | :white_check_mark: | `ggml_sum`?      |
| [ReduceSumSquare](https://github.com/onnx/onnx/blob/main/docs/Operators.md#ReduceSumSquare)       |                    |                  |
| [Relu](https://github.com/onnx/onnx/blob/main/docs/Operators.md#Relu)                             | :white_check_mark: | `ggml_relu`      |
| [Reshape](https://github.com/onnx/onnx/blob/main/docs/Operators.md#Reshape)                       | :white_check_mark: | `ggml_reshape`   |
| [Selu](https://github.com/onnx/onnx/blob/main/docs/Operators.md#Selu)                             |                    |                  |
| [Shape](https://github.com/onnx/onnx/blob/main/docs/Operators.md#Shape)                           | :white_check_mark: |                  |
| [Sigmoid](https://github.com/onnx/onnx/blob/main/docs/Operators.md#Sigmoid)                       |                    |                  |
| [Size](https://github.com/onnx/onnx/blob/main/docs/Operators.md#Size)                             |                    |                  |
| [Slice](https://github.com/onnx/onnx/blob/main/docs/Operators.md#Slice)                           |                    |                  |
| [Softmax](https://github.com/onnx/onnx/blob/main/docs/Operators.md#Softmax)                       | :white_check_mark: | `ggml_soft_max`  |
| [Softplus](https://github.com/onnx/onnx/blob/main/docs/Operators.md#Softplus)                     |                    |                  |
| [Softsign](https://github.com/onnx/onnx/blob/main/docs/Operators.md#Softsign)                     |                    |                  |
| [SpaceToDepth](https://github.com/onnx/onnx/blob/main/docs/Operators.md#SpaceToDepth)             |                    |                  |
| [Split](https://github.com/onnx/onnx/blob/main/docs/Operators.md#Split)                           |                    |                  |
| [Sqrt](https://github.com/onnx/onnx/blob/main/docs/Operators.md#Sqrt)                             | :white_check_mark: | `ggml_sqrt`      |
| [Squeeze](https://github.com/onnx/onnx/blob/main/docs/Operators.md#Squeeze)                       |                    |                  |
| [Sub](https://github.com/onnx/onnx/blob/main/docs/Operators.md#Sub)                               | :white_check_mark: | `ggml_sub`       |
| [Sum](https://github.com/onnx/onnx/blob/main/docs/Operators.md#Sum)                               | :white_check_mark: | `ggml_sum`       |
| [Tanh](https://github.com/onnx/onnx/blob/main/docs/Operators.md#Tanh)                             | :white_check_mark: | `ggml_tanh`      |
| [Tile](https://github.com/onnx/onnx/blob/main/docs/Operators.md#Tile)                             |                    |                  |
| [TopK](https://github.com/onnx/onnx/blob/main/docs/Operators.md#TopK)                             |                    |                  |
| [Transpose](https://github.com/onnx/onnx/blob/main/docs/Operators.md#Transpose)                   | :white_check_mark: | `ggml_transpose` |
| [Unsqueeze](https://github.com/onnx/onnx/blob/main/docs/Operators.md#Unsqueeze)                   | :white_check_mark: |                  |
| [Upsample](https://github.com/onnx/onnx/blob/main/docs/Operators.md#Upsample)                     |                    |                  |
| [Xor](https://github.com/onnx/onnx/blob/main/docs/Operators.md#Xor)                               | :white_check_mark: |                  |
