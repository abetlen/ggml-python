"""ggml-python implemention of the CLIP model
"""
import IPython
import os
import ctypes
import struct
import argparse
import numpy as np
from typing import List, Tuple
from CLIP.clip import simple_tokenizer
import ggml
from ggml.experimental import GGML_FTYPE, Context, InitParams, Tensor, GGML_TYPE, CGraph


class ResidualAttentionBlock:
    def __init__(self, ctx: Context, wtype: GGML_TYPE, embed_dim: int, heads: int):
        # Attention Block (attn)
        self.in_proj_weight = Tensor.new_tensor_2d(
            wtype, 3 * embed_dim, embed_dim, ctx=ctx
        )
        self.in_proj_bias = Tensor.new_tensor_1d(wtype, 3 * embed_dim, ctx=ctx)
        self.out_proj_weight = Tensor.new_tensor_2d(
            wtype, embed_dim, embed_dim, ctx=ctx
        )
        self.out_proj_bias = Tensor.new_tensor_1d(wtype, embed_dim, ctx=ctx)

        # Layer Norm 1 (ln_1)
        self.ln_1_weight = Tensor.new_tensor_1d(wtype, embed_dim, ctx=ctx)
        self.ln_1_bias = Tensor.new_tensor_1d(wtype, embed_dim, ctx=ctx)

        # MLP (mlp)
        self.mlp_weight = Tensor.new_tensor_2d(wtype, embed_dim * 4, embed_dim, ctx=ctx)
        self.mlp_bias = Tensor.new_tensor_1d(wtype, embed_dim * 4, ctx=ctx)

        # Layer Norm 2 (ln_2)
        self.ln_2_weight = Tensor.new_tensor_1d(wtype, embed_dim, ctx=ctx)
        self.ln_2_bias = Tensor.new_tensor_1d(wtype, embed_dim, ctx=ctx)


class VisionTransformer:
    def __init__(
        self,
        ctx: Context,
        wtype: GGML_TYPE,
        input_resolution: int,
        patch_size: int,
        width: int,
        heads: int,
        layers: int,
        output_dim: int,
    ):
        # Class Embedding (visual.class_embedding)
        self.visual_class_embedding = Tensor.new_tensor_1d(wtype, width, ctx=ctx)

        # Positional Embedding (visual.positional_embedding)
        self.visual_positional_embedding = Tensor.new_tensor_2d(
            wtype, (input_resolution // patch_size) ** 2 + 1, width, ctx=ctx
        )

        # Visual Projection (visual.proj)
        self.visual_proj = Tensor.new_tensor_2d(wtype, width, output_dim, ctx=ctx)

        # Convolutional Layer (visual.conv1)
        self.visual_conv1_weight = Tensor.new_tensor_4d(
            wtype, patch_size, patch_size, 3, width, ctx=ctx
        )

        # pre Layer Norm (visual.ln_pre)
        self.visual_ln_pre_weight = Tensor.new_tensor_1d(wtype, width, ctx=ctx)
        self.visual_ln_pre_bias = Tensor.new_tensor_1d(wtype, width, ctx=ctx)

        # Transformer Layer (visual.transformer)
        self.visual_transformer_blocks = [
            ResidualAttentionBlock(ctx=ctx, wtype=wtype, embed_dim=width, heads=heads)
            for _ in range(layers)
        ]

        # post Layer Norm (visual.ln_post)
        self.visual_ln_post_weight = Tensor.new_tensor_1d(wtype, width, ctx=ctx)
        self.visual_ln_post_bias = Tensor.new_tensor_1d(wtype, width, ctx=ctx)


class ClipModel:
    def __init__(
        self,
        ctx: Context,
        wtype: GGML_TYPE,
        vision_width: int,
        vision_layers: int,
        vision_patch_size: int,
        image_resolution: int,
        embed_dim: int,
        context_length: int,
        vocab_size: int,
        transformer_width: int,
        transformer_heads: int,
        transformer_layers: int,
    ):
        self.context_length = context_length

        self.positional_embedding = Tensor.new_tensor_2d(
            wtype, context_length, transformer_width, ctx=ctx
        )
        self.text_projection = Tensor.new_tensor_2d(
            wtype, transformer_width, embed_dim, ctx=ctx
        )
        self.logit_scale = Tensor.new_tensor_1d(wtype, 1, ctx=ctx)

        vision_heads = vision_width // 64
        self.visual = VisionTransformer(
            ctx=ctx,
            wtype=wtype,
            input_resolution=image_resolution,
            patch_size=vision_patch_size,
            width=vision_width,
            layers=vision_layers,
            heads=vision_heads,
            output_dim=embed_dim,
        )

        self.transformer_blocks = [
            ResidualAttentionBlock(
                ctx=ctx,
                wtype=wtype,
                embed_dim=transformer_width,
                heads=transformer_heads,
            )
            for _ in range(transformer_layers)
        ]
        self.token_embedding = Tensor.new_tensor_2d(
            wtype, vocab_size, transformer_width, ctx=ctx
        )
        self.ln_final_weight = Tensor.new_tensor_1d(wtype, transformer_width, ctx=ctx)
        self.ln_final_bias = Tensor.new_tensor_1d(wtype, transformer_width, ctx=ctx)

    @staticmethod
    def init_from_file(model_file: str, verbose=True):
        with open(model_file, "rb") as fin:
            # Magic Number
            (magic,) = struct.unpack("i", (fin.read(struct.calcsize("i"))))
            assert magic == ggml.GGML_FILE_MAGIC.value
            if verbose:
                print("magic number =", hex(magic))
            # Hyperparameters
            (
                vision_width,
                vision_layers,
                vision_patch_size,
                grid_size,
                image_resolution,
                embed_dim,
                context_length,
                transformer_width,
                transformer_heads,
                transformer_layers,
                ftype,
                vocab_size,
            ) = struct.unpack("iiiiiiiiiiii", fin.read(struct.calcsize("iiiiiiiiiiii")))
            qntvr = ftype // ggml.GGML_QNT_VERSION_FACTOR.value
            if verbose:
                print("vision_width    =", vision_width)
                print("vision_layers    =", vision_layers)
                print("vision_patch_size =", vision_patch_size)
                print("grid_size       =", grid_size)
                print("image_resolution =", image_resolution)
                print("embed_dim       =", embed_dim)
                print("context_length  =", context_length)
                print("transformer_width =", transformer_width)
                print("transformer_heads =", transformer_heads)
                print("transformer_layers =", transformer_layers)
                print("ftype           =", ftype)
                print("qntvr           =", qntvr)
                print("vocab_size      =", vocab_size)
            ftype %= ggml.GGML_QNT_VERSION_FACTOR.value
            ftype = GGML_FTYPE(int(ftype))

            # Vocabulary
            vocab: List[Tuple[int, str]] = []
            for i in range(vocab_size):
                (s_len,) = struct.unpack("i", fin.read(struct.calcsize("i")))
                s = fin.read(s_len).decode("utf-8")
                vocab.append((i, s))

            # Model Weights
            wtype = GGML_TYPE(ggml.ggml_ftype_to_ggml_type(ctypes.c_int(ftype.value)))

            # Save current position in file and get file size, then return
            position = fin.tell()
            n_bytes = fin.seek(0, os.SEEK_END)
            fin.seek(position)

            # Create Memory Buffer and Initialize Context
            ctx_size = n_bytes - position

            mem_buffer = np.empty(ctx_size, dtype=np.uint8)
            init_params = InitParams(
                mem_size=ctx_size, mem_buffer=mem_buffer.ctypes.data_as(ctypes.c_void_p)
            )
            ctx = Context(init_params=init_params)

            # Create Model
            model = ClipModel(
                ctx=ctx,
                wtype=wtype,
                vision_width=vision_width,
                vision_layers=vision_layers,
                vision_patch_size=vision_patch_size,
                image_resolution=image_resolution,
                embed_dim=embed_dim,
                context_length=context_length,
                vocab_size=vocab_size,
                transformer_width=transformer_width,
                transformer_heads=transformer_heads,
                transformer_layers=transformer_layers,
            )

            # Load Weights
            breakpoint()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model", type=str, default=None)

    args = parser.parse_args()

    model_file = args.model

    model = ClipModel.init_from_file(model_file)
