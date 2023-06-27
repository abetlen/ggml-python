"""ggml-python implemention of the CLIP model
"""
import io
import os
import ctypes
import struct
import argparse
import numpy as np
from typing import List, Tuple, Dict
import ggml
from ggml.experimental import GGML_FTYPE, Context, InitParams, Tensor, GGML_TYPE, CGraph


def compute_ctx_size(fin: io.BufferedReader) -> int:
    # Save current position in file and get file size, then return
    position = fin.tell()

    ctx_size = 0
    while True:
        nbytes = struct.calcsize("iii")
        data = fin.read(nbytes)
        if len(data) != nbytes:
            break
        (n_dims, s_len, ftype) = struct.unpack("iii", data)
        dims = struct.unpack("i" * n_dims, fin.read(struct.calcsize("i" * n_dims)))
        if ftype == 0:
            _format = "f"
        if ftype == 1:
            _format = "e"
        n_bytes = struct.calcsize(_format * int(np.prod(dims)))
        ctx_size += n_bytes
        ctx_size += 256  # Padding?
        name = fin.read(s_len).decode("utf-8")
        # print(f"Name: {name}, dims: {dims}, n_bytes: {n_bytes}")

        fin.seek(n_bytes, os.SEEK_CUR)

    # Seek back to saved position
    fin.seek(position)
    return ctx_size


class ResidualAttentionBlock:
    def __init__(
        self,
        ctx: Context,
        wtype: GGML_TYPE,
        embed_dim: int,
        heads: int,
        use_attn_mask: bool = False,
    ):
        self.tensors: Dict[str, Tensor] = {}
        self.n_head = heads
        self.embed_dim = embed_dim
        self.use_attn_mask = use_attn_mask
        # Layer Norm 1 (ln_1)
        self.ln_1_weight = Tensor.new_tensor_1d(wtype, embed_dim, ctx=ctx)
        self.ln_1_bias = Tensor.new_tensor_1d(wtype, embed_dim, ctx=ctx)
        self.tensors["ln_1.weight"] = self.ln_1_weight
        self.tensors["ln_1.bias"] = self.ln_1_bias

        # Attention Block (attn)
        self.in_proj_weight = Tensor.new_tensor_2d(
            wtype, embed_dim, 3 * embed_dim, ctx=ctx
        )
        self.in_proj_bias = Tensor.new_tensor_1d(wtype, 3 * embed_dim, ctx=ctx)
        self.out_proj_weight = Tensor.new_tensor_2d(
            wtype, embed_dim, embed_dim, ctx=ctx
        )
        self.out_proj_bias = Tensor.new_tensor_1d(wtype, embed_dim, ctx=ctx)
        self.tensors["attn.in_proj_weight"] = self.in_proj_weight
        self.tensors["attn.in_proj_bias"] = self.in_proj_bias
        self.tensors["attn.out_proj.weight"] = self.out_proj_weight
        self.tensors["attn.out_proj.bias"] = self.out_proj_bias

        # Layer Norm 2 (ln_2)
        self.ln_2_weight = Tensor.new_tensor_1d(wtype, embed_dim, ctx=ctx)
        self.ln_2_bias = Tensor.new_tensor_1d(wtype, embed_dim, ctx=ctx)
        self.tensors["ln_2.weight"] = self.ln_2_weight
        self.tensors["ln_2.bias"] = self.ln_2_bias

        # MLP (mlp)
        self.mlp_c_fc_weight = Tensor.new_tensor_2d(
            wtype, embed_dim, embed_dim * 4, ctx=ctx
        )
        self.mlp_c_fc_bias = Tensor.new_tensor_1d(wtype, embed_dim * 4, ctx=ctx)
        self.mlp_c_proj_weight = Tensor.new_tensor_2d(
            wtype, embed_dim * 4, embed_dim, ctx=ctx
        )
        self.mlp_c_proj_bias = Tensor.new_tensor_1d(wtype, embed_dim, ctx=ctx)
        self.tensors["mlp.c_fc.weight"] = self.mlp_c_fc_weight
        self.tensors["mlp.c_fc.bias"] = self.mlp_c_fc_bias
        self.tensors["mlp.c_proj.weight"] = self.mlp_c_proj_weight
        self.tensors["mlp.c_proj.bias"] = self.mlp_c_proj_bias

    @staticmethod
    def compute_forward_mem_size(
        N: int, width: int, n_heads: int, use_attn_mask: bool = False
    ) -> int:
        e_size = 4
        ggml_overhead = 256
        mem_size = 0
        mem_size += (
            e_size * width * N + ggml_overhead
        ) * 5  # ln_1: repeat, repeat, mul, add, norm
        mem_size += (
            e_size * width * 3 * N + ggml_overhead
        ) * 3  # in_proj: mul_mat, repeat, add
        mem_size += ggml_overhead * 3  # view_2d: Qcur, Kcur, Vcur
        mem_size += (
            (e_size * (width // n_heads) * n_heads * N + ggml_overhead)
            + 2 * ggml_overhead
        ) * 2  # K,Q: new_tensor, cpy, permute
        mem_size += e_size * N * N * n_heads + ggml_overhead  # KQ
        mem_size += e_size * 4 + 256  # KQ_scaled: new_f32
        mem_size += e_size * N * N * n_heads + ggml_overhead  # KQ_scaled
        if use_attn_mask:
            mem_size += (
                e_size * N * N * n_heads + ggml_overhead + e_size * 4 + ggml_overhead
            )  # diag_mask_inf
        mem_size += e_size * N * N * n_heads + ggml_overhead  # KQ_soft_max
        mem_size += (
            e_size * (width // n_heads) * n_heads * N + ggml_overhead
        )  # V_trans: new_tensor_3d
        mem_size += ggml_overhead * 2  # V_trans: cpy and permute
        mem_size += (
            e_size * (width // n_heads) * n_heads * N + ggml_overhead
        )  # V_trans: new_tensor_3d
        mem_size += ggml_overhead  # V_trans: cpy
        mem_size += (
            e_size * (width // n_heads) * n_heads * N + ggml_overhead
        )  # KQV: mul_mat
        mem_size += ggml_overhead  # KQV_merged: permute
        mem_size += e_size * width * N + ggml_overhead  # KQV_merged: new_tensor_2d
        mem_size += ggml_overhead  # KQV_merged: cpy
        mem_size += (
            e_size * width * N + ggml_overhead
        ) * 3  # out_proj: mul_mat, repeat, add
        mem_size += e_size * width * N + ggml_overhead  # Add residual
        mem_size += (
            e_size * width * N + ggml_overhead
        ) * 5  # ln_2: norm, add, repeat, repeat, mul
        mem_size += (
            e_size * width * 4 * N + ggml_overhead
        ) * 3  # MLP: mul_mat, repeat, add
        mem_size += (e_size * 4 + 256) * 2  # SiLU: sf_in, sf_out
        mem_size += (
            e_size * width * 4 * N + ggml_overhead
        ) * 3  # SiLU: scale, silu, scale
        mem_size += (
            e_size * width * N + ggml_overhead
        ) * 3  # mlp_c_proj: mul_mat, repeat, add
        mem_size += e_size * width * N + ggml_overhead  # Add Residual

        return mem_size

    def forward(self, inpL: Tensor, ctx: Context, gf: CGraph) -> Tensor:
        N = inpL.shape[1]

        # [768, N]
        cur = Tensor.norm(inpL, ctx=ctx)
        # cur = ln_1_weight * cur + ln_1_bias
        # [768, N]
        cur = Tensor.add(
            Tensor.mul(Tensor.repeat(self.ln_1_weight, cur, ctx=ctx), cur, ctx=ctx),
            Tensor.repeat(self.ln_1_bias, cur, ctx=ctx),
            ctx=ctx,
        )

        # cur = in_proj_weight * cur + in_proj_bias
        # [768, N] - cur (in)
        # [2304, 768] - in_proj_weight
        # [2304, 1] - in_proj_bias
        # [2304, N] - cur (out)
        cur = Tensor.mul_mat(self.in_proj_weight, cur, ctx=ctx)

        cur = Tensor.add(Tensor.repeat(self.in_proj_bias, cur, ctx=ctx), cur, ctx=ctx)

        # Self-Attention
        n_embd = cur.shape[0] // 3

        Qcur = Tensor.view_2d(
            cur,
            n_embd,
            N,
            cur.tensor.contents.nb[1],
            0 * ctypes.sizeof(ctypes.c_float) * n_embd,
            ctx=ctx,
        )

        Kcur = Tensor.view_2d(
            cur,
            n_embd,
            N,
            cur.tensor.contents.nb[1],
            1 * ctypes.sizeof(ctypes.c_float) * n_embd,
            ctx=ctx,
        )

        Vcur = Tensor.view_2d(
            cur,
            n_embd,
            N,
            cur.tensor.contents.nb[1],
            2 * ctypes.sizeof(ctypes.c_float) * n_embd,
            ctx=ctx,
        )

        Q = Tensor.permute(
            Tensor.cpy(
                Qcur,
                Tensor.new_tensor_3d(
                    GGML_TYPE.F32, n_embd // self.n_head, self.n_head, N, ctx=ctx
                ),
                ctx=ctx,
            ),
            0,
            2,
            1,
            3,
            ctx=ctx,
        )

        K = Tensor.permute(
            Tensor.cpy(
                Kcur,
                Tensor.new_tensor_3d(
                    GGML_TYPE.F32, n_embd // self.n_head, self.n_head, N, ctx=ctx
                ),
                ctx=ctx,
            ),
            0,
            2,
            1,
            3,
            ctx=ctx,
        )

        KQ = Tensor.mul_mat(K, Q, ctx=ctx)

        KQ_scaled = Tensor.scale(
            KQ,
            Tensor.new_f32(
                1.0 / np.sqrt(float(n_embd) / self.n_head),
                ctx=ctx,
            ),
            ctx=ctx,
        )
        if self.use_attn_mask:
            KQ_masked = Tensor.diag_mask_inf(KQ_scaled, 0, ctx=ctx)
            KQ_soft_max = Tensor.soft_max(KQ_masked, ctx=ctx)
        else:
            KQ_soft_max = Tensor.soft_max(KQ_scaled, ctx=ctx)

        V_trans = Tensor.cpy(
            Tensor.permute(
                Tensor.cpy(
                    Vcur,
                    Tensor.new_tensor_3d(
                        GGML_TYPE.F32, n_embd // self.n_head, self.n_head, N, ctx=ctx
                    ),
                    ctx=ctx,
                ),
                1,
                2,
                0,
                3,
                ctx=ctx,
            ),
            Tensor.new_tensor_3d(
                GGML_TYPE.F32, N, n_embd // self.n_head, self.n_head, ctx=ctx
            ),
            ctx=ctx,
        )

        KQV = Tensor.mul_mat(V_trans, KQ_soft_max, ctx=ctx)

        KQV_merged = Tensor.permute(
            KQV,
            0,
            2,
            1,
            3,
            ctx=ctx,
        )

        cur = Tensor.cpy(
            KQV_merged,
            Tensor.new_tensor_2d(
                GGML_TYPE.F32,
                n_embd,
                N,
                ctx=ctx,
            ),
            ctx=ctx,
        )

        cur = Tensor.mul_mat(
            self.out_proj_weight,
            cur,
            ctx=ctx,
        )

        cur = Tensor.add(Tensor.repeat(self.out_proj_bias, cur, ctx=ctx), cur, ctx=ctx)

        # Add Residual
        inpL = Tensor.add(inpL, cur, ctx=ctx)

        # LN2
        cur = Tensor.norm(inpL, ctx=ctx)
        cur = Tensor.add(
            Tensor.mul(Tensor.repeat(self.ln_2_weight, cur, ctx=ctx), cur, ctx=ctx),
            Tensor.repeat(self.ln_2_bias, cur, ctx=ctx),
            ctx=ctx,
        )

        # MLP
        # c_fc
        cur = Tensor.mul_mat(self.mlp_c_fc_weight, cur, ctx=ctx)
        cur = Tensor.add(Tensor.repeat(self.mlp_c_fc_bias, cur, ctx=ctx), cur, ctx=ctx)

        # QuickGELU -  x * sigmoid(1.702 * x)
        cur = Tensor.scale(cur, Tensor.new_f32(1.702, ctx=ctx), ctx=ctx)

        cur = Tensor.silu(cur, ctx=ctx)

        cur = Tensor.scale(cur, Tensor.new_f32(1 / 1.702, ctx=ctx), ctx=ctx)

        # c_proj
        cur = Tensor.mul_mat(self.mlp_c_proj_weight, cur, ctx=ctx)
        cur = Tensor.add(
            Tensor.repeat(self.mlp_c_proj_bias, cur, ctx=ctx), cur, ctx=ctx
        )

        # Add Residual
        cur = Tensor.add(inpL, cur, ctx=ctx)
        return cur


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
        self.layers = layers
        self.tensors: Dict[str, Tensor] = {}

        # Class Embedding (visual.class_embedding)
        self.visual_class_embedding = Tensor.new_tensor_1d(wtype, width, ctx=ctx)
        self.tensors["visual.class_embedding"] = self.visual_class_embedding

        # Positional Embedding (visual.positional_embedding)
        self.visual_positional_embedding = Tensor.new_tensor_2d(
            wtype, width, (input_resolution // patch_size) ** 2 + 1, ctx=ctx
        )
        self.tensors["visual.positional_embedding"] = self.visual_positional_embedding

        # Convolutional Layer (visual.conv1.weight)
        wtype_f16 = GGML_TYPE(ggml.ggml_ftype_to_ggml_type(ctypes.c_int(1)))
        self.visual_conv1_weight = Tensor.new_tensor_4d(
            wtype_f16, patch_size, patch_size, 3, width, ctx=ctx
        )
        self.tensors["visual.conv1.weight"] = self.visual_conv1_weight

        # pre Layer Norm Weight (visual.ln_pre.weight)
        self.visual_ln_pre_weight = Tensor.new_tensor_1d(wtype, width, ctx=ctx)
        self.tensors["visual.ln_pre.weight"] = self.visual_ln_pre_weight

        # pre Layer Norm Bias (visual.ln_pre.bias)
        self.visual_ln_pre_bias = Tensor.new_tensor_1d(wtype, width, ctx=ctx)
        self.tensors["visual.ln_pre.bias"] = self.visual_ln_pre_bias
        self.resblocks = []
        for i in range(layers):
            resblock = ResidualAttentionBlock(
                ctx=ctx, wtype=wtype, embed_dim=width, heads=heads, use_attn_mask=False
            )
            self.resblocks.append(resblock)
            self.tensors.update(
                {
                    f"visual.transformer.resblocks.{i}." + k: v
                    for k, v in resblock.tensors.items()
                }
            )

        # post Layer Norm (visual.ln_post)
        self.visual_ln_post_weight = Tensor.new_tensor_1d(wtype, width, ctx=ctx)
        self.visual_ln_post_bias = Tensor.new_tensor_1d(wtype, width, ctx=ctx)
        self.tensors["visual.ln_post.weight"] = self.visual_ln_post_weight
        self.tensors["visual.ln_post.bias"] = self.visual_ln_post_bias

        # Visual Projection (visual.proj)
        self.visual_proj = Tensor.new_tensor_2d(wtype, output_dim, width, ctx=ctx)
        self.tensors["visual.proj"] = self.visual_proj


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
        n_threads: int,
    ):
        self.n_threads = n_threads
        self.tensors: Dict[str, Tensor] = {}

        # Vision Transformer
        self.vision_layers = vision_layers
        self.vision_patch_size = vision_patch_size
        self.vision_width = vision_width
        self.vision_heads = vision_width // 64
        self.image_resolution = image_resolution
        self.grid_size = image_resolution // vision_patch_size

        # Text Transformer
        self.context_length = context_length
        self.transformer_width = transformer_width
        self.transformer_heads = transformer_heads
        self.transformer_layers = transformer_layers

        self.embed_dim = embed_dim

        # Positional Embedding (position_embedding)
        self.positional_embedding = Tensor.new_tensor_2d(
            wtype, transformer_width, context_length, ctx=ctx
        )
        self.tensors["positional_embedding"] = self.positional_embedding

        # Text Projection (text_projection)
        self.text_projection = Tensor.new_tensor_2d(
            wtype, transformer_width, embed_dim, ctx=ctx
        )
        self.tensors["text_projection"] = self.text_projection

        # Logit Scale (logit_scale)
        self.logit_scale = Tensor.new_tensor_1d(wtype, 1, ctx=ctx)
        self.tensors["logit_scale"] = self.logit_scale

        # Visual Transformer (visual.)
        self.visual = VisionTransformer(
            ctx=ctx,
            wtype=wtype,
            input_resolution=image_resolution,
            patch_size=vision_patch_size,
            width=vision_width,
            layers=vision_layers,
            heads=self.vision_heads,
            output_dim=embed_dim,
        )
        self.tensors.update(self.visual.tensors)

        # Transformer (transformer.)
        self.transformer_res_blocks = []
        for i in range(transformer_layers):
            res_block = ResidualAttentionBlock(
                ctx=ctx,
                wtype=wtype,
                embed_dim=transformer_width,
                heads=transformer_heads,
                use_attn_mask=True,
            )
            self.transformer_res_blocks.append(res_block)
            self.tensors.update(
                {
                    f"transformer.resblocks.{i}." + k: v
                    for k, v in res_block.tensors.items()
                }
            )

        # Token Embedding (token_embedding.weight)
        self.token_embedding = Tensor.new_tensor_2d(
            wtype, transformer_width, vocab_size, ctx=ctx
        )
        self.tensors["token_embedding.weight"] = self.token_embedding

        # Final Layer Norm (ln_final.weight)
        self.ln_final_weight = Tensor.new_tensor_1d(wtype, transformer_width, ctx=ctx)
        self.tensors["ln_final.weight"] = self.ln_final_weight

        # Final Layer Norm (ln_final.bias)
        self.ln_final_bias = Tensor.new_tensor_1d(wtype, transformer_width, ctx=ctx)
        self.tensors["ln_final.bias"] = self.ln_final_bias

    def encode_image(self, image):
        tensor = self._encode_image_internal(image)
        return tensor.numpy().copy().reshape(1, -1)

    def encode_text(self, text_embds):
        encodings = []
        # TODO: batchify
        for text_embd in text_embds:
            tensor = self._encode_text_internal(text_embd)
            encodings.append(tensor.numpy().copy().reshape(1, -1))
        return np.concatenate(encodings, axis=0)

    def __call__(self, image, text):
        image_features = self.encode_image(image)
        text_features = self.encode_text(text)

        # normalized features
        image_features = image_features / np.linalg.norm(
            image_features, axis=1, keepdims=True
        )
        text_features = text_features / np.linalg.norm(
            text_features, axis=1, keepdims=True
        )

        # cosine similarity as logits
        logit_scale = np.exp(self.logit_scale.numpy().copy())
        logits_per_image = logit_scale * image_features @ text_features.T
        logits_per_text = logits_per_image.T

        # shape = [global_batch_size, global_batch_size]
        return logits_per_image, logits_per_text

    def _text_encoder_compute_forward_memsize(self):
        mem_size = 0
        e_size = 4
        ggml_overhead = 256
        mem_size += e_size * self.context_length + ggml_overhead  # input embd

        mem_size += (
            e_size * self.context_length * self.embed_dim + ggml_overhead
        )  # token embedding

        mem_size += (
            e_size * self.context_length * self.embed_dim + ggml_overhead
        )  # add positional embedding
        res_block_mem_size = ResidualAttentionBlock.compute_forward_mem_size(
            self.context_length,
            self.transformer_width,
            self.transformer_heads,
            use_attn_mask=True,
        )
        mem_size += res_block_mem_size * self.transformer_layers
        mem_size += (
            e_size * self.transformer_width * self.context_length + ggml_overhead
        ) * 5  # ln_final

        mem_size += ggml_overhead  # view

        mem_size += e_size * self.embed_dim + ggml_overhead  # Text Proj: output
        mem_size += ggml_overhead  # Text Proj: Transpose
        mem_size += (
            e_size * self.embed_dim * self.embed_dim * ggml_overhead
        )  # Text Proj: cpy
        return mem_size

    def _encode_text_internal(self, embd_inp: np.ndarray):
        wtype = GGML_TYPE(ggml.ggml_ftype_to_ggml_type(ctypes.c_int(0)))
        N = self.context_length
        mem_size = self._text_encoder_compute_forward_memsize()
        mem_buffer = np.empty(mem_size, dtype=np.uint8)
        init_params = InitParams(
            mem_size=mem_size, mem_buffer=mem_buffer.ctypes.data_as(ctypes.c_void_p)
        )
        ctx0 = Context(init_params=init_params)

        gf = CGraph(cgraph=ggml.ggml_cgraph(n_threads=self.n_threads), ctx=ctx0)

        embd = Tensor.new_tensor_1d(GGML_TYPE.I32, N, ctx=ctx0)
        embd.numpy()[:] = np.array(embd_inp, dtype=np.int32)
        inpL = Tensor.get_rows(self.token_embedding, embd, ctx=ctx0)
        cur = Tensor.add(inpL, self.positional_embedding, ctx=ctx0)

        for il in range(self.transformer_layers):
            resblock = self.transformer_res_blocks[il]
            cur = resblock.forward(cur, ctx=ctx0, gf=gf)

        cur = Tensor.norm(cur, ctx=ctx0)
        cur = Tensor.add(
            Tensor.mul(
                Tensor.repeat(self.ln_final_weight, cur, ctx=ctx0),
                cur,
                ctx=ctx0,
            ),
            Tensor.repeat(self.ln_final_bias, cur, ctx=ctx0),
            ctx=ctx0,
        )

        # Use the embedding from the EOT token
        eot_idx = embd_inp.argmax()
        cur = Tensor.view_2d(
            cur,
            self.embed_dim,
            1,
            cur.tensor.contents.nb[1],
            eot_idx * cur.tensor.contents.nb[1],
            ctx=ctx0,
        )

        cur = Tensor.mul_mat(
            Tensor.cpy(
                Tensor.transpose(self.text_projection, ctx=ctx0),
                Tensor.new_tensor_2d(wtype, self.embed_dim, self.embed_dim, ctx=ctx0),
            ),
            cur,
            ctx=ctx0,
        )
        gf.build_forward_expand(cur)
        gf.compute()
        return cur

    def _image_encoder_compute_forward_memsize(self):
        e_size = 4
        N = self.grid_size * self.grid_size + 1
        ggml_overhead = 256

        mem_size = 0
        mem_size += 256
        mem_size += (
            e_size * self.image_resolution * self.image_resolution * 3 + ggml_overhead
        )  # image
        mem_size += (
            e_size * self.grid_size * self.grid_size * self.vision_width + ggml_overhead
        )  # conv
        mem_size += e_size * self.vision_width * N + ggml_overhead  # concat

        mem_size += (
            e_size * self.vision_width * N + ggml_overhead
        ) * 2  # Copy in visual features
        mem_size += e_size * 8 + 256
        mem_size += 2 * ggml_overhead  # cpy and transpose
        mem_size += e_size * 8 + 256  # ???
        mem_size += (
            e_size * self.vision_width * N + ggml_overhead
        )  # Copy in positional embeddings (new tensor 2d)
        mem_size += (
            e_size * self.vision_width * N + ggml_overhead
        )  # copy visual features: ret

        mem_size += e_size * self.vision_width * N + ggml_overhead  # add

        mem_size += e_size * self.vision_width * N + ggml_overhead  # ln_pre: norm
        mem_size += e_size * self.vision_width * N + ggml_overhead  # ln_pre: repeat
        mem_size += e_size * self.vision_width * N + ggml_overhead  # ln_pre: repeat
        mem_size += e_size * self.vision_width * N + ggml_overhead  # ln_pre: mul
        mem_size += e_size * self.vision_width * N + ggml_overhead  # ln_pre: add

        res_block_mem_size = ResidualAttentionBlock.compute_forward_mem_size(
            N, self.vision_width, self.vision_heads, use_attn_mask=False
        )

        mem_size += res_block_mem_size * self.vision_layers
        mem_size += ggml_overhead  # ln_post: transpose
        mem_size += (e_size * self.vision_width + ggml_overhead) * 3  # ln_post

        mem_size += e_size * self.vision_width * self.embed_dim + ggml_overhead
        mem_size += ggml_overhead  # cpy
        mem_size += 159808  # Compute Overhead ??
        return mem_size

    def _encode_image_internal(self, image):
        wtype = GGML_TYPE(ggml.ggml_ftype_to_ggml_type(ctypes.c_int(0)))

        mem_size = self._image_encoder_compute_forward_memsize()
        mem_buffer = np.empty(mem_size, dtype=np.uint8)
        init_params = InitParams(
            mem_size=mem_size, mem_buffer=mem_buffer.ctypes.data_as(ctypes.c_void_p)
        )
        ctx0 = Context(init_params=init_params)

        gf = CGraph(cgraph=ggml.ggml_cgraph(n_threads=self.n_threads), ctx=ctx0)

        img_tensor = Tensor.new_tensor_4d(
            wtype,
            image.shape[3],
            image.shape[2],
            image.shape[1],
            image.shape[0],
            ctx=ctx0,
        )
        img_tensor.numpy()[:] = image.permute(3, 2, 1, 0)
        cur = Tensor.conv_2d_sk_p0(
            self.visual.visual_conv1_weight, img_tensor, ctx=ctx0
        )

        cur = Tensor.reshape_2d(
            cur,
            cur.shape[0] * cur.shape[1],
            cur.shape[2],
            ctx=ctx0,
        )

        concat = Tensor.new_tensor_2d(wtype, cur.shape[0] + 1, cur.shape[1], ctx=ctx0)

        concat = Tensor.set_1d(
            concat,
            Tensor.view_1d(
                self.visual.visual_class_embedding,
                self.visual.visual_class_embedding.shape[0],
                0,
            ),
            0,
            ctx=ctx0,
        )

        # Copy in the visual features
        concat = Tensor.set_2d(
            concat,
            Tensor.cpy(
                Tensor.transpose(cur, ctx=ctx0),
                Tensor.new_tensor_2d(wtype, cur.shape[0], cur.shape[1]),
                ctx=ctx0,
            ),
            cur.tensor.contents.nb[1],
            self.visual.visual_class_embedding.nbytes(),
            ctx=ctx0,
        )

        # Copy in the positional embeddings
        cur = Tensor.cpy(
            concat,
            Tensor.new_tensor_2d(wtype, concat.shape[1], concat.shape[0], ctx=ctx0),
            ctx=ctx0,
        )

        pEmb = self.visual.visual_positional_embedding

        cur = Tensor.add(cur, pEmb, ctx=ctx0)

        # ln_pre
        cur = Tensor.norm(cur, ctx=ctx0)

        cur = Tensor.add(
            Tensor.mul(
                Tensor.repeat(self.visual.visual_ln_pre_weight, cur, ctx=ctx0),
                cur,
                ctx=ctx0,
            ),
            Tensor.repeat(self.visual.visual_ln_pre_bias, cur, ctx=ctx0),
            ctx=ctx0,
        )

        # Transformer
        for il in range(self.visual.layers):
            resblock = self.visual.resblocks[il]
            cur = resblock.forward(cur, ctx=ctx0, gf=gf)

        # ln_post
        cur = Tensor.norm(
            Tensor.view_2d(Tensor.transpose(cur, ctx=ctx0), cur.shape[0], 1, 1, 0),
            ctx=ctx0,
        )

        cur = Tensor.add(
            Tensor.mul(
                Tensor.repeat(self.visual.visual_ln_post_weight, cur, ctx=ctx0),
                cur,
                ctx=ctx0,
            ),
            Tensor.repeat(self.visual.visual_ln_post_bias, cur, ctx=ctx0),
            ctx=ctx0,
        )

        # Token Projection
        cur = Tensor.mul_mat(
            Tensor.cpy(
                Tensor.transpose(self.visual.visual_proj),
                Tensor.new_tensor_2d(
                    wtype,
                    self.visual.visual_proj.shape[1],
                    self.visual.visual_proj.shape[0],
                    ctx=ctx0,
                ),
                ctx=ctx0,
            ),
            Tensor.reshape_2d(cur, cur.shape[0], 1),
            ctx=ctx0,
        )

        gf.build_forward_expand(cur)
        gf.compute()

        return cur

    @staticmethod
    def init_from_file(model_file: str, verbose=True, n_threads=1):
        with open(model_file, "rb") as fin:
            # Magic Number
            (magic,) = struct.unpack("i", (fin.read(struct.calcsize("i"))))

            assert magic == ggml.GGML_FILE_MAGIC
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

            qntvr = ftype // ggml.GGML_QNT_VERSION_FACTOR
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
            ftype %= ggml.GGML_QNT_VERSION_FACTOR
            ftype = GGML_FTYPE(int(ftype))

            # Vocabulary
            vocab: List[Tuple[int, str]] = []
            for i in range(vocab_size):
                (s_len,) = struct.unpack("i", fin.read(struct.calcsize("i")))
                s = fin.read(s_len).decode("utf-8")
                vocab.append((i, s))

            # Model Weights
            wtype = GGML_TYPE(ggml.ggml_ftype_to_ggml_type(ctypes.c_int(ftype.value)))

            ctx_size = compute_ctx_size(fin)

            mem_buffer = np.empty(ctx_size, dtype=np.uint8)
            init_params = InitParams(
                mem_size=ctx_size,
                mem_buffer=mem_buffer.ctypes.data_as(ctypes.c_void_p),
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
                n_threads=n_threads,
            )

            # Load Weights
            while True:
                nbytes = struct.calcsize("iii")
                data = fin.read(nbytes)
                if len(data) != nbytes:
                    break
                (n_dims, s_len, ftype) = struct.unpack("iii", data)
                dims = struct.unpack(
                    "i" * n_dims, fin.read(struct.calcsize("i" * n_dims))
                )
                tensor_name = fin.read(s_len).decode("utf-8")
                tensor = model.tensors[tensor_name]
                n_elements = tensor.nelements()
                expected_n_elements = np.prod(dims)
                if n_elements != expected_n_elements:
                    raise ValueError(
                        f"tensor {tensor_name} has {n_elements} elements, but {expected_n_elements} were expected"
                    )

                buf = (ctypes.c_char * tensor.nbytes()).from_address(tensor.data)
                offset = fin.tell()
                fname = fin.name.encode("utf-8")
                fin.readinto(buf)

            return model


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model", type=str, default=None)
    parser.add_argument("--use-gpu", action="store_true")
    args = parser.parse_args()

    model_file = args.model
    model = ClipModel.init_from_file(model_file, n_threads=1, use_gpu=args.use_gpu)
    image = np.random.rand(3, 224, 224).astype(np.float32)
    output = model.eval([image, image])
    print(output)
