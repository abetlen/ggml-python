"""ggml-python implemention of the Replit code model

Model is available at:
https://huggingface.co/replit/replit-code-v1-3b

This implementation is based on the example model code and ggml model file format from:
https://github.com/ggerganov/ggml/tree/master/examples/replit
"""
import math
import struct
import ctypes
from collections import deque
from dataclasses import dataclass
from typing import Deque, List, Tuple, Dict

import numpy as np

import ggml
from ggml.experimental import GGML_FTYPE, Context, InitParams, Tensor, GGML_TYPE, CGraph


@dataclass
class MPTParams:
    d_model: int
    max_seq_len: int
    n_heads: int
    n_layers: int
    vocab_size: int
    ftype: int


@dataclass
class ReplitLayer:
    # pre normalization
    ln_1_weight: Tensor
    # attention
    c_attn_wqkv_weight: Tensor
    c_attn_out_proj_weight: Tensor
    # post normalization
    ln_2_weight: Tensor
    # ff
    c_mlp_mlp_up_weight: Tensor
    c_mlp_mlp_down_weight: Tensor


@dataclass
class ReplitModel:
    hparams: MPTParams
    wte_weight: Tensor
    ln_f_weight: Tensor

    layers: List[ReplitLayer]

    # key + value memory
    memory_k: Tensor
    memory_v: Tensor

    ctx: Context
    tensors: Dict[str, Tensor]

    vocab: List[Tuple[int, str, float]]

    @staticmethod
    def encode_word(
        word: str, model: Dict[str, Tuple[int, float]]
    ) -> Tuple[List[int], float]:
        best_segmentation_starts = [-1] * (len(word) + 1)
        best_segmentation_starts[0] = 0

        best_segmentation_scores = [-math.inf] * (len(word) + 1)
        best_segmentation_scores[0] = 1.0

        for start_idx in range(len(word)):
            best_score_at_start = best_segmentation_scores[start_idx]
            for end_idx in range(start_idx + 1, len(word) + 1):
                token = word[start_idx:end_idx]
                if token in model and best_score_at_start != -math.inf:
                    token_score = model[token][1]
                    score = token_score + best_score_at_start
                    if (
                        best_segmentation_scores[end_idx] == -math.inf
                        or best_segmentation_scores[end_idx] > score
                    ):
                        best_segmentation_starts[end_idx] = start_idx
                        best_segmentation_scores[end_idx] = score

        if best_segmentation_scores[-1] == -math.inf:
            return [], 0.0

        score = best_segmentation_scores[-1]
        start = best_segmentation_starts[-1]
        end = len(word)
        tokens: Deque[int] = deque()
        while start != 0:
            token_id = model[word[start:end]][0]
            tokens.appendleft(token_id)
            next_start = best_segmentation_starts[start]
            end = start
            start = next_start
        token_id = model[word[start:end]][0]
        tokens.appendleft(token_id)
        return list(tokens), score

    def tokenize(self, text: str) -> List[int]:
        ws_symbol = b"\342\226\201"
        piece_map = {piece: (i, -score) for i, piece, score in self.vocab}
        normalized_text = text.replace(" ", ws_symbol.decode("utf-8"))
        tokenized, _ = ReplitModel.encode_word(normalized_text, piece_map)
        return tokenized

    def detokenize(self, tokens: List[int]) -> str:
        id_to_token = self.vocab
        ws_symbol = b"\342\226\201"
        text = ""
        for token in tokens:
            text += id_to_token[token][1]
        detokenized = text.replace(ws_symbol.decode("utf-8"), " ")
        return detokenized

    def eval(self, embd_inp: List[int], n_past: int, n_threads: int):
        N = len(embd_inp)

        n_embd = self.hparams.d_model
        n_layer = self.hparams.n_layers
        n_ctx = self.hparams.max_seq_len
        n_head = self.hparams.n_heads
        n_vocab = self.hparams.vocab_size

        buf_size = 256 * 1024 * 1024
        if not hasattr(self, "buf"):
            self.buf = (ctypes.c_char * buf_size)()

        init_params = InitParams(
            mem_size=buf_size,
            mem_buffer=ctypes.c_void_p(ctypes.addressof(self.buf)),
            no_alloc=False,
        )
        ctx0 = Context(init_params=init_params)
        gf = CGraph(
            cgraph=ggml.ggml_cgraph(
                n_threads=n_threads,
            ),
            ctx=ctx0,
        )

        embd = Tensor.new_tensor_1d(
            GGML_TYPE.I32,
            N,
            ctx=ctx0,
        )
        embd.numpy()[:] = np.array(embd_inp, dtype=np.int32)

        inpL = Tensor.get_rows(self.wte_weight, embd, ctx=ctx0)

        for il in range(n_layer):
            # // a = self.ln_1(x)
            cur = Tensor.norm(inpL, ctx=ctx0)
            cur = Tensor.mul(
                Tensor.repeat(self.layers[il].ln_1_weight, cur, ctx=ctx0),
                cur,
                ctx=ctx0,
            )

            # // self-attention
            # //  b, _, past_key_value = self.attn(a, past_key_value=past_key_value,
            # //  attn_bias=attn_bias, attention_mask=attention_mask,
            # //  is_causal=is_causal)

            # // compute QKV
            cur = Tensor.mul_mat(model.layers[il].c_attn_wqkv_weight, cur, ctx=ctx0)

            Qcur = Tensor.view_2d(
                cur,
                n_embd,
                N,
                cur.tensor.contents.nb[1],
                0 * ctypes.sizeof(ctypes.c_float) * n_embd,
                ctx=ctx0,
            )
            Kcur = Tensor.view_2d(
                cur,
                n_embd,
                N,
                cur.tensor.contents.nb[1],
                1 * ctypes.sizeof(ctypes.c_float) * n_embd,
                ctx=ctx0,
            )
            Vcur = Tensor.view_2d(
                cur,
                n_embd,
                N,
                cur.tensor.contents.nb[1],
                2 * ctypes.sizeof(ctypes.c_float) * n_embd,
                ctx=ctx0,
            )

            # // store key and value to memory
            k = Tensor.view_1d(
                self.memory_k,
                N * n_embd,
                (self.memory_k.element_size() * n_embd) * (il * n_ctx + n_past),
                ctx=ctx0,
            )
            v = Tensor.view_1d(
                self.memory_v,
                N * n_embd,
                (self.memory_v.element_size() * n_embd) * (il * n_ctx + n_past),
                ctx=ctx0,
            )

            gf.build_forward_expand(
                Tensor.cpy(
                    Kcur,
                    k,
                    ctx=ctx0,
                )
            )
            gf.build_forward_expand(
                Tensor.cpy(
                    Vcur,
                    v,
                    ctx=ctx0,
                )
            )

            # // Q = Qcur.contiguous().view(n_embd/n_head, n_head, N).permute(0,
            # // 2, 1, 3) [64, N, 12]
            Q = Tensor.permute(
                Tensor.cpy(
                    Qcur,
                    Tensor.new_tensor_3d(
                        GGML_TYPE.F32, n_embd // n_head, n_head, N, ctx=ctx0
                    ),
                    ctx=ctx0,
                ),
                0,
                2,
                1,
                3,
                ctx=ctx0,
            )

            # // K = Kmem.view(n_embd/n_head, n_head, n_past + N).permute(0, 2, 1,
            # // 3) [64, n_past + N, 12]
            K = Tensor.permute(
                Tensor.reshape_3d(
                    Tensor.view_1d(
                        self.memory_k,
                        (n_past + N) * n_embd,
                        il * n_ctx * self.memory_k.element_size() * n_embd,
                        ctx=ctx0,
                    ),
                    n_embd // n_head,
                    n_head,
                    n_past + N,
                    ctx=ctx0,
                ),
                0,
                2,
                1,
                3,
                ctx=ctx0,
            )

            # // K * Q
            KQ = Tensor.mul_mat(K, Q, ctx=ctx0)

            # // KQ_scaled = KQ / sqrt(n_embd/n_head)
            KQ_scaled = Tensor.scale(
                KQ,
                Tensor.new_f32(
                    1.0 / np.sqrt(float(n_embd) / n_head),
                    ctx=ctx0,
                ),
                ctx=ctx0,
            )

            KQ_scaled_alibi = Tensor.alibi(
                KQ_scaled,
                n_past,
                n_head,
                8.0,
                ctx=ctx0,
            )

            # // KQ_masked = mask_past(KQ_scaled)
            KQ_masked = Tensor.diag_mask_inf(
                KQ_scaled_alibi,
                n_past,
                ctx=ctx0,
            )

            # // KQ = soft_max(KQ_masked)
            KQ_soft_max = Tensor.soft_max(
                KQ_masked,
                ctx=ctx0,
            )

            # // V_trans = Vmem.view(n_embd/n_head, n_head, n_past + N).permute(1,
            # // 2, 0, 3).contiguous() [n_past + N, 64, 12]
            V_trans = Tensor.cpy(
                Tensor.permute(
                    Tensor.reshape_3d(
                        Tensor.view_1d(
                            self.memory_v,
                            (n_past + N) * n_embd,
                            il * n_ctx * self.memory_v.element_size() * n_embd,
                            ctx=ctx0,
                        ),
                        n_embd // n_head,
                        n_head,
                        n_past + N,
                        ctx=ctx0,
                    ),
                    1,
                    2,
                    0,
                    3,
                    ctx=ctx0,
                ),
                Tensor.new_tensor_3d(
                    self.memory_v.ggml_type,
                    n_past + N,
                    n_embd // n_head,
                    n_head,
                    ctx=ctx0,
                ),
            )

            # // KQV = transpose(V) * KQ_soft_max
            KQV = Tensor.mul_mat(V_trans, KQ_soft_max, ctx=ctx0)

            # // KQV_merged = KQV.permute(0, 2, 1, 3)
            KQV_merged = Tensor.permute(
                KQV,
                0,
                2,
                1,
                3,
                ctx=ctx0,
            )

            # // cur = KQV_merged.contiguous().view(n_embd, N)
            cur = Tensor.cpy(
                KQV_merged,
                Tensor.new_tensor_2d(
                    GGML_TYPE.F32,
                    n_embd,
                    N,
                    ctx=ctx0,
                ),
                ctx=ctx0,
            )

            # // projection
            cur = Tensor.mul_mat(
                self.layers[il].c_attn_out_proj_weight,
                cur,
                ctx=ctx0,
            )

            inpL = Tensor.add(
                inpL,
                cur,
                ctx=ctx0,
            )

            # // m = self.ln_2(x)
            cur = Tensor.norm(inpL, ctx=ctx0)
            cur = Tensor.mul(
                Tensor.repeat(self.layers[il].ln_2_weight, cur, ctx=ctx0),
                cur,
                ctx=ctx0,
            )

            # // n = self.mlp(m)
            cur = Tensor.mul_mat(
                self.layers[il].c_mlp_mlp_up_weight,
                cur,
                ctx=ctx0,
            )
            # // GELU activation
            cur = Tensor.gelu(
                cur,
                ctx=ctx0,
            )
            # // projection
            # // cur = proj_w*cur + proj_b
            cur = Tensor.mul_mat(
                self.layers[il].c_mlp_mlp_down_weight,
                cur,
                ctx=ctx0,
            )

            # // x = x + n
            inpL = Tensor.add(
                inpL,
                cur,
                ctx=ctx0,
            )

        # // norm
        inpL = Tensor.norm(inpL, ctx=ctx0)
        # // inpL = ln_f_g*inpL
        inpL = Tensor.mul(
            Tensor.repeat(self.ln_f_weight, inpL, ctx=ctx0),
            inpL,
            ctx=ctx0,
        )

        # // output embedding weight tied to input embedding
        inpL = Tensor.mul_mat(
            self.wte_weight,
            inpL,
            ctx=ctx0,
        )

        gf.build_forward_expand(inpL)
        gf.compute()

        embd_w = inpL.numpy().reshape(n_vocab, -1).copy()

        return embd_w

    @staticmethod
    def init_from_file(model_file: str, verbose: bool = True) -> "ReplitModel":
        verbose = True

        with open(model_file, "rb") as fin:
            # Magic Number
            (magic,) = struct.unpack("i", (fin.read(struct.calcsize("i"))))
            assert magic == ggml.GGML_FILE_MAGIC.value
            if verbose:
                print("magic number =", hex(magic))
            # Hyperparameters
            d_model, max_seq_len, n_heads, n_layers, vocab_size, ftype = struct.unpack(
                "iiiiii", (fin.read(struct.calcsize("iiiiii")))
            )
            qntvr = ftype // ggml.GGML_QNT_VERSION_FACTOR.value
            if verbose:
                print("d_model      =", d_model)
                print("max_seq_len  =", max_seq_len)
                print("n_heads      =", n_heads)
                print("n_layers     =", n_layers)
                print("vocab_size   =", vocab_size)
                print("ftype        =", ftype)
                print("qntvr        =", qntvr)
            ftype %= ggml.GGML_QNT_VERSION_FACTOR.value
            ftype = GGML_FTYPE(int(ftype))
            # Vocabulary
            vocab: List[Tuple[int, str, float]] = []
            for i in range(vocab_size):
                (s_len,) = struct.unpack("i", (fin.read(struct.calcsize("i"))))
                s = fin.read(s_len).decode("utf-8")
                (score,) = struct.unpack("f", (fin.read(struct.calcsize("f"))))
                vocab.append((i, s, score))
            # Model Weights
            wtype = GGML_TYPE(ggml.ggml_ftype_to_ggml_type(ctypes.c_int(ftype.value)))
            wtype_sizef = ggml.ggml_type_sizef(ctypes.c_int(wtype.value))
            f32_sizef = ggml.ggml_type_sizef(ctypes.c_int(GGML_TYPE.F32.value))
            f16_sizef = ggml.ggml_type_sizef(ctypes.c_int(GGML_TYPE.F16.value))

            ctx_size = 0

            n_embd = d_model
            n_layer = n_layers
            n_ctx = max_seq_len
            n_vocab = vocab_size

            # compute ctx size
            ctx_size += n_embd * n_vocab * wtype_sizef
            ctx_size += n_embd * f32_sizef

            ctx_size += n_layer * (n_embd * f32_sizef)
            ctx_size += n_layer * (3 * n_embd * n_embd * wtype_sizef)
            ctx_size += n_layer * (n_embd * n_embd * wtype_sizef)
            ctx_size += n_layer * (n_embd * f32_sizef)
            ctx_size += n_layer * (4 * n_embd * n_embd * wtype_sizef)
            ctx_size += n_layer * (n_embd * n_embd * 4 * wtype_sizef)

            ctx_size += n_ctx * n_layer * n_embd * f16_sizef
            ctx_size += n_ctx * n_layer * n_embd * f16_sizef

            ctx_size += (1 + 6 * n_layer) * 512
            ctx_size = int(ctx_size)

            if verbose:
                print("ctx size     =", ctx_size // (1024 * 1024), "MB")

            # create context
            mem_buffer = np.empty(ctx_size, dtype=np.uint8)
            init_params = InitParams(
                mem_size=ctx_size,
                mem_buffer=mem_buffer.ctypes.data_as(ctypes.c_void_p),
            )
            ctx = Context(init_params=init_params)

            hparams = MPTParams(
                d_model=d_model,
                max_seq_len=max_seq_len,
                n_heads=n_heads,
                n_layers=n_layers,
                vocab_size=vocab_size,
                ftype=ftype.value,
            )

            n_mem = n_layer * n_ctx
            n_elements = n_embd * n_mem

            model = ReplitModel(
                hparams=hparams,
                wte_weight=Tensor.new_tensor_2d(wtype, n_embd, n_vocab, ctx=ctx),
                ln_f_weight=Tensor.new_tensor_1d(GGML_TYPE.F32, n_embd, ctx=ctx),
                memory_k=Tensor.new_tensor_1d(GGML_TYPE.F16, n_elements, ctx=ctx),
                memory_v=Tensor.new_tensor_1d(GGML_TYPE.F16, n_elements, ctx=ctx),
                ctx=ctx,
                layers=[],
                tensors={},
                vocab=vocab,
            )

            model.tensors["transformer.wte.weight"] = model.wte_weight
            model.tensors["transformer.norm_f.weight"] = model.ln_f_weight

            for i in range(n_layer):
                layer = ReplitLayer(
                    ln_1_weight=Tensor.new_tensor_1d(GGML_TYPE.F32, n_embd, ctx=ctx),
                    c_attn_wqkv_weight=Tensor.new_tensor_2d(
                        wtype, n_embd, 3 * n_embd, ctx=ctx
                    ),
                    c_attn_out_proj_weight=Tensor.new_tensor_2d(
                        wtype, n_embd, n_embd, ctx=ctx
                    ),
                    ln_2_weight=Tensor.new_tensor_1d(GGML_TYPE.F32, n_embd, ctx=ctx),
                    c_mlp_mlp_up_weight=Tensor.new_tensor_2d(
                        wtype, n_embd, 4 * n_embd, ctx=ctx
                    ),
                    c_mlp_mlp_down_weight=Tensor.new_tensor_2d(
                        wtype, 4 * n_embd, n_embd, ctx=ctx
                    ),
                )
                model.layers.append(layer)

                model.tensors[
                    f"transformer.blocks.{i}.norm_1.weight"
                ] = layer.ln_1_weight
                model.tensors[
                    f"transformer.blocks.{i}.attn.Wqkv.weight"
                ] = layer.c_attn_wqkv_weight
                model.tensors[
                    f"transformer.blocks.{i}.attn.out_proj.weight"
                ] = layer.c_attn_out_proj_weight
                model.tensors[
                    f"transformer.blocks.{i}.norm_2.weight"
                ] = layer.ln_2_weight
                model.tensors[
                    f"transformer.blocks.{i}.ffn.up_proj.weight"
                ] = layer.c_mlp_mlp_up_weight
                model.tensors[
                    f"transformer.blocks.{i}.ffn.down_proj.weight"
                ] = layer.c_mlp_mlp_down_weight

            n_tensors = 0
            total_size = 0

            while True:
                nbytes = struct.calcsize("iii")
                data = fin.read(nbytes)
                if len(data) != nbytes:
                    break
                n_dims, length, ttype = struct.unpack("iii", data)
                nelements = 1
                ne = [1, 1]
                for i in range(n_dims):
                    (dim,) = struct.unpack("i", (fin.read(struct.calcsize("i"))))
                    ne[i] = dim
                    nelements *= ne[i]
                name = fin.read(length).decode("utf-8")
                # if verbose:
                #     print(name, ne, ttype)
                if name not in model.tensors:
                    raise ValueError(f"Tensor {name} not found in model")
                tensor = model.tensors[name]
                if tensor.nelements() != nelements:
                    raise ValueError(
                        f"Tensor {name} has {tensor.nelements()} elements, but {nelements} expected"
                    )
                if (
                    tensor.tensor.contents.ne[0] != ne[0]
                    or tensor.tensor.contents.ne[1] != ne[1]
                ):
                    raise ValueError(
                        f"Tensor {name} has {tensor.tensor.contents.ne[0]}x{tensor.tensor.contents.ne[1]} shape, but {ne[0]}x{ne[1]} expected"
                    )
                bpe = ggml.ggml_type_size(ctypes.c_int(GGML_TYPE(ttype).value))
                if (
                    (nelements * bpe) / ggml.ggml_blck_size(tensor.tensor.contents.type)
                ) != ggml.ggml_nbytes(tensor.tensor):
                    raise ValueError(
                        f"Tensor {name} has {ggml.ggml_nbytes(tensor.tensor)} bytes, but {(nelements * bpe) / ggml.ggml_blck_size(tensor.tensor.contents.type)} expected"
                    )
                buf = (ctypes.c_char * tensor.nbytes()).from_address(tensor.data)
                fin.readinto(buf)

                total_size += tensor.nbytes()
                if n_tensors % 8 == 0:
                    print(".", end="", flush=True)
                n_tensors += 1
            print("done")
            print(
                "model size =",
                total_size // (1024 * 1024),
                "MB",
                "num tensors =",
                n_tensors,
            )

        return model


if __name__ == "__main__":
    model_file = "../../models/replit-code-v1-3b/ggml-model-q4_0.bin"
    model = ReplitModel.init_from_file(model_file)

    prompt = "def fib(n):"
    prompt_tokens = model.tokenize(prompt)
    all_tokens: List[int] = prompt_tokens[:]  # type: ignore
    n_past = 0
    tokens: List[int] = prompt_tokens[:]  # type: ignore

    print("number of tokens in prompt =", len(prompt_tokens))
    for i, token_id in enumerate(prompt_tokens):
        print(f"token[{i}] =", token_id)

    print()
    print(prompt, end="", flush=True)
    for i in range(32):
        # eval
        scores = model.eval(tokens, n_past, 6)
        # sample
        logits = scores[:, -1]
        token_id = int(np.argmax(logits))
        # update
        all_tokens.append(token_id)
        print(model.detokenize([token_id]), end="", flush=True)
        n_past += len(tokens)
        tokens = [token_id]
    print()
