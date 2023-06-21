"""ggml-python implemention of the CLIP model
"""
import IPython
import io
import os
import ctypes
import struct
import argparse
import numpy as np
from typing import List, Tuple, Dict
from CLIP.clip import simple_tokenizer
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
    def __init__(self, ctx: Context, wtype: GGML_TYPE, embed_dim: int, heads: int):
        self.tensors: Dict[str, Tensor] = {}
        self.n_head = heads
        self.embed_dim = embed_dim

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
        # attn_weight = torch.softmax((Q @ K.transpose(-2, -1) / math.sqrt(Q.size(-1))) + attn_mask, dim=-1)

        KQ = Tensor.mul_mat(K, Q, ctx=ctx)

        KQ_scaled = Tensor.scale(
            KQ,
            Tensor.new_f32(
                1.0 / np.sqrt(float(n_embd) / self.n_head),
                ctx=ctx,
            ),
            ctx=ctx,
        )

        KQ_soft_max = Tensor.soft_max(KQ_scaled, ctx=ctx)
        # GOOD ^^^
        # BAD  vvv Fix double copy
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
        gf.build_forward_expand(V_trans)
        gf.compute()

        IPython.embed()
        exit(0)

        # V_trans = Tensor.cpy(
        #     Tensor.permute(
        #         Tensor.reshape_3d(Vcur, n_embd // self.n_head, self.n_head, N, ctx=ctx),
        #         1,
        #         2,
        #         0,
        #         3,
        #         ctx=ctx,
        #     ),
        #     Tensor.new_tensor_3d(
        #         GGML_TYPE.F32, N, n_embd // self.n_head, self.n_head, ctx=ctx
        #     ),
        # )
        # V_trans = Tensor.cpy(
        #     Tensor.permute(
        #         Vcur,
        #         1,
        #         2,
        #         0,
        #         3,
        #         ctx=ctx,
        #     ),
        #     Tensor.new_tensor_3d(
        #         GGML_TYPE.F32, N, n_embd // self.n_head, self.n_head, ctx=ctx
        #     ),
        # )
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
        # BAD
        gf.build_forward_expand(cur)
        gf.compute()

        IPython.embed()
        exit(0)

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
        # GELU
        cur = Tensor.gelu(cur, ctx=ctx)
        # c_proj
        cur = Tensor.mul_mat(self.mlp_c_proj_weight, cur, ctx=ctx)
        cur = Tensor.add(
            Tensor.repeat(self.mlp_c_proj_bias, cur, ctx=ctx), cur, ctx=ctx
        )
        # Add Residual
        inpL = Tensor.add(inpL, cur, ctx=ctx)

        return inpL


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
                ctx=ctx, wtype=wtype, embed_dim=width, heads=heads
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
        self.visual_proj = Tensor.new_tensor_2d(wtype, width, output_dim, ctx=ctx)
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
        self.context_length = context_length
        self.tensors: Dict[str, Tensor] = {}

        # Positional Embedding (position_embedding)
        self.positional_embedding = Tensor.new_tensor_2d(
            wtype, context_length, transformer_width, ctx=ctx
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
        self.tensors.update(self.visual.tensors)

        # Transformer (transformer.)
        for i in range(transformer_layers):
            res_block = ResidualAttentionBlock(
                ctx=ctx,
                wtype=wtype,
                embed_dim=transformer_width,
                heads=transformer_heads,
            )
            self.tensors.update(
                {
                    f"transformer.resblocks.{i}." + k: v
                    for k, v in res_block.tensors.items()
                }
            )

        # Token Embedding (token_embedding.weight)
        self.token_embedding = Tensor.new_tensor_2d(
            wtype, vocab_size, transformer_width, ctx=ctx
        )
        self.tensors["token_embedding.weight"] = self.token_embedding

        # Final Layer Norm (ln_final.weight)
        self.ln_final_weight = Tensor.new_tensor_1d(wtype, transformer_width, ctx=ctx)
        self.tensors["ln_final.weight"] = self.ln_final_weight

        # Final Layer Norm (ln_final.bias)
        self.ln_final_bias = Tensor.new_tensor_1d(wtype, transformer_width, ctx=ctx)
        self.tensors["ln_final.bias"] = self.ln_final_bias

    def encode_image(self, image):
        wtype = GGML_TYPE(ggml.ggml_ftype_to_ggml_type(ctypes.c_int(0)))
        mem_size = np.prod(image.shape) * 4 + 256 + 256 * 1024 * 1024 * 2
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
        cur = Tensor.norm(cur, ctx=ctx0)
        cur = Tensor.add(
            Tensor.mul(
                Tensor.repeat(self.visual.visual_ln_post_weight, cur, ctx=ctx0),
                cur,
                ctx=ctx0,
            ),
            Tensor.repeat(self.visual.visual_ln_post_bias, cur, ctx=ctx0),
            ctx=ctx0,
        )

        cur = Tensor.view_2d(cur, 1, cur.shape[0], 1, 0, ctx=ctx0)
        cur = Tensor.mul_mat(cur, self.visual.visual_proj, ctx=ctx0)
        # Token Projection
        gf.build_forward_expand(cur)
        gf.compute()

        IPython.embed()

        return cur

    def encode_text(self, text):
        # Encode text using the transformer
        pass

    def _eval_internal(self, image: np.ndarray, text: np.ndarray):
        ## Encode Image
        image_features = self.encode_image(image)
        ## Encode Text

        IPython.embed()

    def eval(self, image: np.ndarray, text: np.ndarray):
        output = self._eval_internal(image, text)
        pass

    @staticmethod
    def init_from_file(model_file: str, verbose=True, n_threads=1):
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

            ctx_size = compute_ctx_size(fin)

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
                fin.readinto(buf)
            return model


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model", type=str, default=None)

    args = parser.parse_args()

    model_file = args.model

    model = ClipModel.init_from_file(model_file, n_threads=1)
    image = np.random.rand(3, 224, 224).astype(np.float32)
    output = model.eval([image, image])
    print(output)
