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

## Why?

The original purpose of the ggml onnx runtime was a belief that onnx could offer a faster way to add new model support by directly converting pytorch models into onnx graphs and then automatically lowering those into ggml computational graphs.

I don't believe this to be the case however I still believe this is a useful project for another reason.

Models don't actually have one canonical compute graph.
Even the smallest models you can think of typically have one graph for training (usually split across multiple gpus and nodes) and one graph for inference.

But this is not correct either because different stages of training require different loss computation.
Inference can be dissagregated from a single machien across embeddings, prefill, and decoding stages.

For this reason I think it's useful to have an easy way to reference the same weight names but be able to push computational graphs to different machines over a network to support all of this.


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

This table reflects the current `ggml.contrib.onnx` backend handlers and selected ONNX backend test coverage. `Supported` means the selected backend tests for the operator pass. `Partial` means the operator has a handler, but known variants are skipped or not implemented. `Implemented` means a handler exists, but the current selected backend suite does not cover it.

| ONNX Operator | Status | Notes |
|:--------------|:------:|:------|
| [Abs](https://github.com/onnx/onnx/blob/main/docs/Operators.md#Abs) | Supported |  |
| [Acos](https://github.com/onnx/onnx/blob/main/docs/Operators.md#Acos) | Supported |  |
| [Acosh](https://github.com/onnx/onnx/blob/main/docs/Operators.md#Acosh) | Supported |  |
| [Add](https://github.com/onnx/onnx/blob/main/docs/Operators.md#Add) | Supported |  |
| [And](https://github.com/onnx/onnx/blob/main/docs/Operators.md#And) | Supported |  |
| [ArgMax](https://github.com/onnx/onnx/blob/main/docs/Operators.md#ArgMax) | Supported |  |
| [ArgMin](https://github.com/onnx/onnx/blob/main/docs/Operators.md#ArgMin) | Supported |  |
| [ArrayFeatureExtractor](https://github.com/onnx/onnx/blob/main/docs/Operators.md#ArrayFeatureExtractor) | Supported |  |
| [Asin](https://github.com/onnx/onnx/blob/main/docs/Operators.md#Asin) | Supported |  |
| [Asinh](https://github.com/onnx/onnx/blob/main/docs/Operators.md#Asinh) | Supported |  |
| [Atan](https://github.com/onnx/onnx/blob/main/docs/Operators.md#Atan) | Supported |  |
| [Atanh](https://github.com/onnx/onnx/blob/main/docs/Operators.md#Atanh) | Supported |  |
| [AveragePool](https://github.com/onnx/onnx/blob/main/docs/Operators.md#AveragePool) | Supported |  |
| [BatchNormalization](https://github.com/onnx/onnx/blob/main/docs/Operators.md#BatchNormalization) | Supported |  |
| [Binarizer](https://github.com/onnx/onnx/blob/main/docs/Operators.md#Binarizer) | Supported |  |
| [BitShift](https://github.com/onnx/onnx/blob/main/docs/Operators.md#BitShift) | Supported |  |
| [BitwiseAnd](https://github.com/onnx/onnx/blob/main/docs/Operators.md#BitwiseAnd) | Supported |  |
| [BitwiseNot](https://github.com/onnx/onnx/blob/main/docs/Operators.md#BitwiseNot) | Supported |  |
| [BitwiseOr](https://github.com/onnx/onnx/blob/main/docs/Operators.md#BitwiseOr) | Supported |  |
| [BitwiseXor](https://github.com/onnx/onnx/blob/main/docs/Operators.md#BitwiseXor) | Supported |  |
| [BlackmanWindow](https://github.com/onnx/onnx/blob/main/docs/Operators.md#BlackmanWindow) | Supported |  |
| [Cast](https://github.com/onnx/onnx/blob/main/docs/Operators.md#Cast) | Partial | Selected numeric and string-to-float forms pass; bfloat16, float8, and string output forms are not fully covered. |
| [CastLike](https://github.com/onnx/onnx/blob/main/docs/Operators.md#CastLike) | Partial | Selected numeric and string-to-float-like forms pass; bfloat16, float8, and string output forms are not fully covered. |
| [Ceil](https://github.com/onnx/onnx/blob/main/docs/Operators.md#Ceil) | Supported |  |
| [Celu](https://github.com/onnx/onnx/blob/main/docs/Operators.md#Celu) | Supported |  |
| [CenterCropPad](https://github.com/onnx/onnx/blob/main/docs/Operators.md#CenterCropPad) | Supported |  |
| [Clip](https://github.com/onnx/onnx/blob/main/docs/Operators.md#Clip) | Supported |  |
| [Col2Im](https://github.com/onnx/onnx/blob/main/docs/Operators.md#Col2Im) | Supported |  |
| [Compress](https://github.com/onnx/onnx/blob/main/docs/Operators.md#Compress) | Supported |  |
| [Concat](https://github.com/onnx/onnx/blob/main/docs/Operators.md#Concat) | Supported |  |
| [Constant](https://github.com/onnx/onnx/blob/main/docs/Operators.md#Constant) | Supported |  |
| [ConstantOfShape](https://github.com/onnx/onnx/blob/main/docs/Operators.md#ConstantOfShape) | Supported |  |
| [Conv](https://github.com/onnx/onnx/blob/main/docs/Operators.md#Conv) | Partial | Common 1D, 2D, 3D, grouped, and asymmetric-padding cases pass; auto_pad variants remain limited. |
| [ConvInteger](https://github.com/onnx/onnx/blob/main/docs/Operators.md#ConvInteger) | Partial | Padding cases pass; auto_pad is not supported. |
| [ConvTranspose](https://github.com/onnx/onnx/blob/main/docs/Operators.md#ConvTranspose) | Partial | 2D and 3D cases pass; auto_pad variants remain limited. |
| [Cos](https://github.com/onnx/onnx/blob/main/docs/Operators.md#Cos) | Supported |  |
| [Cosh](https://github.com/onnx/onnx/blob/main/docs/Operators.md#Cosh) | Supported |  |
| [CumSum](https://github.com/onnx/onnx/blob/main/docs/Operators.md#CumSum) | Supported |  |
| [DFT](https://github.com/onnx/onnx/blob/main/docs/Operators.md#DFT) | Supported |  |
| [DepthToSpace](https://github.com/onnx/onnx/blob/main/docs/Operators.md#DepthToSpace) | Supported |  |
| [DequantizeLinear](https://github.com/onnx/onnx/blob/main/docs/Operators.md#DequantizeLinear) | Supported |  |
| [Det](https://github.com/onnx/onnx/blob/main/docs/Operators.md#Det) | Supported |  |
| [Div](https://github.com/onnx/onnx/blob/main/docs/Operators.md#Div) | Supported |  |
| [Dropout](https://github.com/onnx/onnx/blob/main/docs/Operators.md#Dropout) | Supported |  |
| [DynamicQuantizeLinear](https://github.com/onnx/onnx/blob/main/docs/Operators.md#DynamicQuantizeLinear) | Supported |  |
| [Einsum](https://github.com/onnx/onnx/blob/main/docs/Operators.md#Einsum) | Supported |  |
| [Elu](https://github.com/onnx/onnx/blob/main/docs/Operators.md#Elu) | Supported |  |
| [Equal](https://github.com/onnx/onnx/blob/main/docs/Operators.md#Equal) | Partial | Numeric and boolean tensor cases pass; string tensor cases are excluded. |
| [Erf](https://github.com/onnx/onnx/blob/main/docs/Operators.md#Erf) | Supported |  |
| [Exp](https://github.com/onnx/onnx/blob/main/docs/Operators.md#Exp) | Supported |  |
| [Expand](https://github.com/onnx/onnx/blob/main/docs/Operators.md#Expand) | Supported |  |
| [EyeLike](https://github.com/onnx/onnx/blob/main/docs/Operators.md#EyeLike) | Supported |  |
| [Flatten](https://github.com/onnx/onnx/blob/main/docs/Operators.md#Flatten) | Supported |  |
| [Floor](https://github.com/onnx/onnx/blob/main/docs/Operators.md#Floor) | Supported |  |
| [Gather](https://github.com/onnx/onnx/blob/main/docs/Operators.md#Gather) | Supported |  |
| [GatherElements](https://github.com/onnx/onnx/blob/main/docs/Operators.md#GatherElements) | Supported |  |
| [GatherND](https://github.com/onnx/onnx/blob/main/docs/Operators.md#GatherND) | Supported |  |
| [Gemm](https://github.com/onnx/onnx/blob/main/docs/Operators.md#Gemm) | Supported |  |
| [GlobalAveragePool](https://github.com/onnx/onnx/blob/main/docs/Operators.md#GlobalAveragePool) | Supported |  |
| [GlobalMaxPool](https://github.com/onnx/onnx/blob/main/docs/Operators.md#GlobalMaxPool) | Supported |  |
| [Greater](https://github.com/onnx/onnx/blob/main/docs/Operators.md#Greater) | Supported |  |
| [GreaterOrEqual](https://github.com/onnx/onnx/blob/main/docs/Operators.md#GreaterOrEqual) | Supported |  |
| [GridSample](https://github.com/onnx/onnx/blob/main/docs/Operators.md#GridSample) | Supported |  |
| [GroupNormalization](https://github.com/onnx/onnx/blob/main/docs/Operators.md#GroupNormalization) | Supported |  |
| [HammingWindow](https://github.com/onnx/onnx/blob/main/docs/Operators.md#HammingWindow) | Supported |  |
| [HannWindow](https://github.com/onnx/onnx/blob/main/docs/Operators.md#HannWindow) | Supported |  |
| [HardSigmoid](https://github.com/onnx/onnx/blob/main/docs/Operators.md#HardSigmoid) | Supported |  |
| [HardSwish](https://github.com/onnx/onnx/blob/main/docs/Operators.md#HardSwish) | Supported |  |
| [Hardmax](https://github.com/onnx/onnx/blob/main/docs/Operators.md#Hardmax) | Supported |  |
| [Identity](https://github.com/onnx/onnx/blob/main/docs/Operators.md#Identity) | Partial | Tensor identity passes; optional and sequence identity cases are excluded. |
| [InstanceNormalization](https://github.com/onnx/onnx/blob/main/docs/Operators.md#InstanceNormalization) | Supported |  |
| [IsInf](https://github.com/onnx/onnx/blob/main/docs/Operators.md#IsInf) | Supported |  |
| [IsNaN](https://github.com/onnx/onnx/blob/main/docs/Operators.md#IsNaN) | Supported |  |
| [LRN](https://github.com/onnx/onnx/blob/main/docs/Operators.md#LRN) | Supported |  |
| [LayerNormalization](https://github.com/onnx/onnx/blob/main/docs/Operators.md#LayerNormalization) | Partial | Direct operator cases pass; expanded function graph cases remain excluded. |
| [LeakyRelu](https://github.com/onnx/onnx/blob/main/docs/Operators.md#LeakyRelu) | Supported |  |
| [Less](https://github.com/onnx/onnx/blob/main/docs/Operators.md#Less) | Supported |  |
| [LessOrEqual](https://github.com/onnx/onnx/blob/main/docs/Operators.md#LessOrEqual) | Supported |  |
| [Log](https://github.com/onnx/onnx/blob/main/docs/Operators.md#Log) | Supported |  |
| [LogSoftmax](https://github.com/onnx/onnx/blob/main/docs/Operators.md#LogSoftmax) | Supported |  |
| [LpPool](https://github.com/onnx/onnx/blob/main/docs/Operators.md#LpPool) | Supported |  |
| [MatMul](https://github.com/onnx/onnx/blob/main/docs/Operators.md#MatMul) | Supported |  |
| [MatMulInteger](https://github.com/onnx/onnx/blob/main/docs/Operators.md#MatMulInteger) | Supported |  |
| [Max](https://github.com/onnx/onnx/blob/main/docs/Operators.md#Max) | Supported |  |
| [MaxPool](https://github.com/onnx/onnx/blob/main/docs/Operators.md#MaxPool) | Supported |  |
| [MaxUnpool](https://github.com/onnx/onnx/blob/main/docs/Operators.md#MaxUnpool) | Partial | Covered for current backend tests; rank greater than 4 is not supported. |
| [Mean](https://github.com/onnx/onnx/blob/main/docs/Operators.md#Mean) | Supported |  |
| [MeanVarianceNormalization](https://github.com/onnx/onnx/blob/main/docs/Operators.md#MeanVarianceNormalization) | Supported |  |
| [MelWeightMatrix](https://github.com/onnx/onnx/blob/main/docs/Operators.md#MelWeightMatrix) | Supported |  |
| [Min](https://github.com/onnx/onnx/blob/main/docs/Operators.md#Min) | Supported |  |
| [Mish](https://github.com/onnx/onnx/blob/main/docs/Operators.md#Mish) | Supported |  |
| [Mod](https://github.com/onnx/onnx/blob/main/docs/Operators.md#Mod) | Supported |  |
| [Mul](https://github.com/onnx/onnx/blob/main/docs/Operators.md#Mul) | Supported |  |
| [Neg](https://github.com/onnx/onnx/blob/main/docs/Operators.md#Neg) | Supported |  |
| [NegativeLogLikelihoodLoss](https://github.com/onnx/onnx/blob/main/docs/Operators.md#NegativeLogLikelihoodLoss) | Supported |  |
| [NonMaxSuppression](https://github.com/onnx/onnx/blob/main/docs/Operators.md#NonMaxSuppression) | Supported |  |
| [NonZero](https://github.com/onnx/onnx/blob/main/docs/Operators.md#NonZero) | Supported |  |
| [Not](https://github.com/onnx/onnx/blob/main/docs/Operators.md#Not) | Supported |  |
| [OneHot](https://github.com/onnx/onnx/blob/main/docs/Operators.md#OneHot) | Supported |  |
| [Or](https://github.com/onnx/onnx/blob/main/docs/Operators.md#Or) | Supported |  |
| [PRelu](https://github.com/onnx/onnx/blob/main/docs/Operators.md#PRelu) | Supported |  |
| [Pad](https://github.com/onnx/onnx/blob/main/docs/Operators.md#Pad) | Supported |  |
| [Pow](https://github.com/onnx/onnx/blob/main/docs/Operators.md#Pow) | Supported |  |
| [QLinearConv](https://github.com/onnx/onnx/blob/main/docs/Operators.md#QLinearConv) | Partial | Selected quantized convolution cases pass; auto_pad and some per-channel forms remain limited. |
| [QLinearMatMul](https://github.com/onnx/onnx/blob/main/docs/Operators.md#QLinearMatMul) | Supported |  |
| [QuantizeLinear](https://github.com/onnx/onnx/blob/main/docs/Operators.md#QuantizeLinear) | Partial | Integer quantization cases pass; float8 output cases require ONNX element-type tracking. |
| [RandomNormalLike](https://github.com/onnx/onnx/blob/main/docs/Operators.md#RandomNormalLike) | Implemented | Registered handler; no selected backend coverage in the current suite. |
| [Range](https://github.com/onnx/onnx/blob/main/docs/Operators.md#Range) | Partial | Direct Range cases pass; expanded Loop-based forms are excluded. |
| [Reciprocal](https://github.com/onnx/onnx/blob/main/docs/Operators.md#Reciprocal) | Supported |  |
| [ReduceL1](https://github.com/onnx/onnx/blob/main/docs/Operators.md#ReduceL1) | Supported |  |
| [ReduceL2](https://github.com/onnx/onnx/blob/main/docs/Operators.md#ReduceL2) | Supported |  |
| [ReduceLogSum](https://github.com/onnx/onnx/blob/main/docs/Operators.md#ReduceLogSum) | Supported |  |
| [ReduceLogSumExp](https://github.com/onnx/onnx/blob/main/docs/Operators.md#ReduceLogSumExp) | Supported |  |
| [ReduceMax](https://github.com/onnx/onnx/blob/main/docs/Operators.md#ReduceMax) | Supported |  |
| [ReduceMean](https://github.com/onnx/onnx/blob/main/docs/Operators.md#ReduceMean) | Supported |  |
| [ReduceMin](https://github.com/onnx/onnx/blob/main/docs/Operators.md#ReduceMin) | Supported |  |
| [ReduceProd](https://github.com/onnx/onnx/blob/main/docs/Operators.md#ReduceProd) | Supported |  |
| [ReduceSum](https://github.com/onnx/onnx/blob/main/docs/Operators.md#ReduceSum) | Supported |  |
| [ReduceSumSquare](https://github.com/onnx/onnx/blob/main/docs/Operators.md#ReduceSumSquare) | Supported |  |
| [Relu](https://github.com/onnx/onnx/blob/main/docs/Operators.md#Relu) | Supported |  |
| [Reshape](https://github.com/onnx/onnx/blob/main/docs/Operators.md#Reshape) | Supported |  |
| [Resize](https://github.com/onnx/onnx/blob/main/docs/Operators.md#Resize) | Partial | Common nearest, linear, and cubic cases pass; not all resize modes and policies are implemented. |
| [ReverseSequence](https://github.com/onnx/onnx/blob/main/docs/Operators.md#ReverseSequence) | Supported |  |
| [RoiAlign](https://github.com/onnx/onnx/blob/main/docs/Operators.md#RoiAlign) | Supported |  |
| [Round](https://github.com/onnx/onnx/blob/main/docs/Operators.md#Round) | Supported |  |
| [STFT](https://github.com/onnx/onnx/blob/main/docs/Operators.md#STFT) | Supported |  |
| [Scatter](https://github.com/onnx/onnx/blob/main/docs/Operators.md#Scatter) | Supported |  |
| [ScatterElements](https://github.com/onnx/onnx/blob/main/docs/Operators.md#ScatterElements) | Supported |  |
| [ScatterND](https://github.com/onnx/onnx/blob/main/docs/Operators.md#ScatterND) | Supported |  |
| [Selu](https://github.com/onnx/onnx/blob/main/docs/Operators.md#Selu) | Supported |  |
| [Shape](https://github.com/onnx/onnx/blob/main/docs/Operators.md#Shape) | Supported |  |
| [Shrink](https://github.com/onnx/onnx/blob/main/docs/Operators.md#Shrink) | Supported |  |
| [Sigmoid](https://github.com/onnx/onnx/blob/main/docs/Operators.md#Sigmoid) | Supported |  |
| [Sign](https://github.com/onnx/onnx/blob/main/docs/Operators.md#Sign) | Supported |  |
| [Sin](https://github.com/onnx/onnx/blob/main/docs/Operators.md#Sin) | Supported |  |
| [Sinh](https://github.com/onnx/onnx/blob/main/docs/Operators.md#Sinh) | Supported |  |
| [Size](https://github.com/onnx/onnx/blob/main/docs/Operators.md#Size) | Supported |  |
| [Slice](https://github.com/onnx/onnx/blob/main/docs/Operators.md#Slice) | Supported |  |
| [Softmax](https://github.com/onnx/onnx/blob/main/docs/Operators.md#Softmax) | Supported |  |
| [SoftmaxCrossEntropyLoss](https://github.com/onnx/onnx/blob/main/docs/Operators.md#SoftmaxCrossEntropyLoss) | Supported |  |
| [Softplus](https://github.com/onnx/onnx/blob/main/docs/Operators.md#Softplus) | Supported |  |
| [Softsign](https://github.com/onnx/onnx/blob/main/docs/Operators.md#Softsign) | Supported |  |
| [SpaceToDepth](https://github.com/onnx/onnx/blob/main/docs/Operators.md#SpaceToDepth) | Supported |  |
| [Split](https://github.com/onnx/onnx/blob/main/docs/Operators.md#Split) | Supported |  |
| [Sqrt](https://github.com/onnx/onnx/blob/main/docs/Operators.md#Sqrt) | Supported |  |
| [Squeeze](https://github.com/onnx/onnx/blob/main/docs/Operators.md#Squeeze) | Supported |  |
| [Sub](https://github.com/onnx/onnx/blob/main/docs/Operators.md#Sub) | Supported |  |
| [Sum](https://github.com/onnx/onnx/blob/main/docs/Operators.md#Sum) | Supported |  |
| [Tan](https://github.com/onnx/onnx/blob/main/docs/Operators.md#Tan) | Supported |  |
| [Tanh](https://github.com/onnx/onnx/blob/main/docs/Operators.md#Tanh) | Supported |  |
| [ThresholdedRelu](https://github.com/onnx/onnx/blob/main/docs/Operators.md#ThresholdedRelu) | Supported |  |
| [Tile](https://github.com/onnx/onnx/blob/main/docs/Operators.md#Tile) | Supported |  |
| [TopK](https://github.com/onnx/onnx/blob/main/docs/Operators.md#TopK) | Supported |  |
| [Transpose](https://github.com/onnx/onnx/blob/main/docs/Operators.md#Transpose) | Supported |  |
| [Trilu](https://github.com/onnx/onnx/blob/main/docs/Operators.md#Trilu) | Supported |  |
| [Unique](https://github.com/onnx/onnx/blob/main/docs/Operators.md#Unique) | Supported |  |
| [Unsqueeze](https://github.com/onnx/onnx/blob/main/docs/Operators.md#Unsqueeze) | Supported |  |
| [Upsample](https://github.com/onnx/onnx/blob/main/docs/Operators.md#Upsample) | Deprecated | Deprecated ONNX operator; legacy nearest-mode handler is covered. |
| [Where](https://github.com/onnx/onnx/blob/main/docs/Operators.md#Where) | Supported |  |
| [Xor](https://github.com/onnx/onnx/blob/main/docs/Operators.md#Xor) | Supported |  |
| [Adagrad](https://github.com/onnx/onnx/blob/main/docs/Operators.md#Adagrad) | Unsupported | Training optimizer operator is not implemented. |
| [Adam](https://github.com/onnx/onnx/blob/main/docs/Operators.md#Adam) | Unsupported | Training optimizer operator is not implemented. |
| [Bernoulli](https://github.com/onnx/onnx/blob/main/docs/Operators.md#Bernoulli) | Unsupported | Random sampling operator is not implemented. |
| [DeformConv](https://github.com/onnx/onnx/blob/main/docs/Operators.md#DeformConv) | Unsupported | Deformable convolution is not implemented. |
| [GRU](https://github.com/onnx/onnx/blob/main/docs/Operators.md#GRU) | Unsupported | Recurrent operator is not implemented. |
| [GlobalLpPool](https://github.com/onnx/onnx/blob/main/docs/Operators.md#GlobalLpPool) | Unsupported | No registered handler. |
| [Gradient](https://github.com/onnx/onnx/blob/main/docs/Operators.md#Gradient) | Unsupported | Training gradient operator is not implemented. |
| [If](https://github.com/onnx/onnx/blob/main/docs/Operators.md#If) | Unsupported | Control-flow operator is not implemented. |
| [LSTM](https://github.com/onnx/onnx/blob/main/docs/Operators.md#LSTM) | Unsupported | Recurrent operator is not implemented. |
| [Loop](https://github.com/onnx/onnx/blob/main/docs/Operators.md#Loop) | Unsupported | Control-flow operator is not implemented. |
| [LpNormalization](https://github.com/onnx/onnx/blob/main/docs/Operators.md#LpNormalization) | Unsupported | No registered handler. |
| [MaxRoiPool](https://github.com/onnx/onnx/blob/main/docs/Operators.md#MaxRoiPool) | Unsupported | No registered handler. |
| [Momentum](https://github.com/onnx/onnx/blob/main/docs/Operators.md#Momentum) | Unsupported | Training optimizer operator is not implemented. |
| [OptionalGetElement](https://github.com/onnx/onnx/blob/main/docs/Operators.md#OptionalGetElement) | Unsupported | Optional value support is not implemented. |
| [OptionalHasElement](https://github.com/onnx/onnx/blob/main/docs/Operators.md#OptionalHasElement) | Unsupported | Optional value support is not implemented. |
| [RNN](https://github.com/onnx/onnx/blob/main/docs/Operators.md#RNN) | Unsupported | Recurrent operator is not implemented. |
| [RandomNormal](https://github.com/onnx/onnx/blob/main/docs/Operators.md#RandomNormal) | Unsupported | Random sampling operator is not implemented. |
| [RandomUniform](https://github.com/onnx/onnx/blob/main/docs/Operators.md#RandomUniform) | Unsupported | Random sampling operator is not implemented. |
| [RandomUniformLike](https://github.com/onnx/onnx/blob/main/docs/Operators.md#RandomUniformLike) | Unsupported | Random sampling operator is not implemented. |
| [Scan](https://github.com/onnx/onnx/blob/main/docs/Operators.md#Scan) | Unsupported | Control-flow operator is not implemented. |
| [SequenceAt](https://github.com/onnx/onnx/blob/main/docs/Operators.md#SequenceAt) | Unsupported | Sequence value support is not implemented. |
| [SequenceConstruct](https://github.com/onnx/onnx/blob/main/docs/Operators.md#SequenceConstruct) | Unsupported | Sequence value support is not implemented. |
| [SequenceEmpty](https://github.com/onnx/onnx/blob/main/docs/Operators.md#SequenceEmpty) | Unsupported | Sequence value support is not implemented. |
| [SequenceErase](https://github.com/onnx/onnx/blob/main/docs/Operators.md#SequenceErase) | Unsupported | Sequence value support is not implemented. |
| [SequenceInsert](https://github.com/onnx/onnx/blob/main/docs/Operators.md#SequenceInsert) | Unsupported | Sequence value support is not implemented. |
| [SequenceLength](https://github.com/onnx/onnx/blob/main/docs/Operators.md#SequenceLength) | Unsupported | Sequence value support is not implemented. |
| [SequenceMap](https://github.com/onnx/onnx/blob/main/docs/Operators.md#SequenceMap) | Unsupported | Sequence value support is not implemented. |
| [SplitToSequence](https://github.com/onnx/onnx/blob/main/docs/Operators.md#SplitToSequence) | Unsupported | Sequence output support is not implemented. |
| [StringNormalizer](https://github.com/onnx/onnx/blob/main/docs/Operators.md#StringNormalizer) | Unsupported | String tensor support is not implemented. |
| [TfIdfVectorizer](https://github.com/onnx/onnx/blob/main/docs/Operators.md#TfIdfVectorizer) | Unsupported | String/ML vectorizer support is not implemented. |

## Acknowledgements

The GGML ONNX runtime is built on top of the [ONNX](https://onnx.ai/) and [GGML](https://ggml.ai)

The core of the runtime was written by Andrei Betlen (@abetlen), David Miller (@dmille), and
Mohammadreza Anvari (@mrezanvari)

This work would also not be possible without the ggml community, in particular @slaren for their work on the ggml backends and memory allocation api.
