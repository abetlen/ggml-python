"""ggml-python implemention of the Replit code model

Model is available at:
https://huggingface.co/replit/replit-code-v1-3b

This implementation is based on the example model code and ggml model file format from:
https://github.com/ggerganov/ggml/tree/master/examples/replit
"""
from __future__ import annotations
import abc
import math
import time
import uuid
import struct
import ctypes
import argparse
import multiprocessing
from collections import deque

from typing_extensions import TypedDict, Literal
from typing import Deque, Iterator, List, Optional, Sequence, Tuple, Dict, Union

import numpy as np
import numpy.typing as npt

import ggml

from ggml.utils import to_numpy


## Types
class CompletionLogprobs(TypedDict):
    text_offset: List[int]
    token_logprobs: List[Optional[float]]
    tokens: List[str]
    top_logprobs: List[Optional[Dict[str, float]]]


class CompletionChoice(TypedDict):
    text: str
    index: int
    logprobs: Optional[CompletionLogprobs]
    finish_reason: Optional[str]


class CompletionUsage(TypedDict):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class CompletionChunk(TypedDict):
    id: str
    object: Literal["text_completion"]
    created: int
    model: str
    choices: List[CompletionChoice]


class Completion(TypedDict):
    id: str
    object: Literal["text_completion"]
    created: int
    model: str
    choices: List[CompletionChoice]
    usage: CompletionUsage


## Generic Sampling Functions


def logits_to_logprobs(logits: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]:
    return np.exp(logits) / (np.sum(np.exp(logits)))  # type: ignore


def sample(
    logits: npt.NDArray[np.float32],
    last_tokens: Optional[List[int]] = None,
    presence_penalty: float = 0.0,
    frequency_penalty: float = 0.0,
    temperature: float = 1.0,
    top_p: float = 0.0,
) -> int:
    if last_tokens is None:
        last_tokens = []
    if temperature == 0.0:
        return int(np.argmax(logits))
    logits = frequency_and_presence_penalties(
        logits, last_tokens, frequency_penalty, presence_penalty
    )
    return nucleus_sampling(logits, top_p=top_p, temperature=temperature)


# TODO: this is likely incorrect
def frequency_and_presence_penalties(
    logits: npt.NDArray[np.float32],
    last_tokens: Sequence[int],
    alpha_frequency: float,
    alpha_presence: float,
):
    if len(last_tokens) == 0:
        return logits

    if alpha_frequency == 0.0 and alpha_presence == 0.0:
        return logits

    # Calculate the frequency penalty contribution
    frequency_penalty = alpha_frequency * np.log(np.unique(last_tokens).size + 1)

    # Calculate the presence penalty contribution
    presence_penalty = alpha_presence * np.array(
        [float(token in last_tokens) for token in range(len(logits))]
    )

    # Apply penalties to the logits
    penalized_logits = logits - frequency_penalty - presence_penalty

    return penalized_logits


def nucleus_sampling(
    logits: npt.NDArray[np.float32], top_p: float, temperature: float = 1.0
):
    # Apply temperature to logits
    logits /= temperature

    # Subtract the maximum value for numerical stability
    logits -= logits.max()  # type: ignore

    # Calculate probabilities using softmax function with epsilon
    epsilon = 1e-8
    probabilities = np.exp(logits) / ((np.exp(logits)).sum() + epsilon)  # type: ignore

    # Filter out NaN values from probabilities
    probabilities = np.nan_to_num(probabilities)

    # Sort the probabilities in descending order and get the corresponding indices
    sorted_indices = np.argsort(probabilities)[::-1]

    # Select the indices within the nucleus
    nucleus_indices = sorted_indices[: int(len(sorted_indices) * top_p)]

    # Calculate the updated probabilities within the nucleus
    nucleus_probabilities = probabilities[nucleus_indices]

    # Normalize the probabilities within the nucleus
    nucleus_probabilities /= nucleus_probabilities.sum()  # type: ignore

    # Sample from the updated probabilities
    selected_token = np.random.choice(nucleus_indices, p=nucleus_probabilities)

    return selected_token


### Context Buffer


class ContextBuffer(abc.ABC):
    @abc.abstractmethod
    def resize(self, new_size: int) -> None:
        raise NotImplementedError

    @abc.abstractproperty
    def buffer(self) -> ctypes.c_void_p:
        raise NotImplementedError


class CpuContextBuffer(ContextBuffer):
    def __init__(self, buffer_size: int = 256 * 1024 * 1024):
        self.buffer_size = buffer_size
        self._buffer = (ctypes.c_uint8 * self.buffer_size)()

    def resize(self, new_size: int):
        assert new_size > self.buffer_size

        self.buffer_size = new_size
        ctypes.resize(self._buffer, self.buffer_size)

    @property
    def buffer(self) -> ctypes.c_void_p:
        return ctypes.c_void_p(ctypes.addressof(self._buffer))


class CudaContextBuffer(ContextBuffer):
    def __init__(self, buffer_size: int = 256 * 1024 * 1024):
        self.buffer_size = buffer_size
        self._buffer = CudaContextBuffer.try_alloc(self.buffer_size)

    @staticmethod
    def try_alloc(buffer_size: int):
        buffer = ggml.ggml_cuda_host_malloc(buffer_size)
        if buffer is None:  # type: ignore
            raise RuntimeError("Failed to allocate CUDA buffer")
        return buffer

    @property
    def buffer(self) -> ctypes.c_void_p:
        if self._buffer is None:
            return ctypes.c_void_p(0)
        return self._buffer

    def resize(self, new_size: int):
        assert new_size > self.buffer_size

        if self._buffer is not None:
            ggml.ggml_cuda_host_free(self._buffer)

        self.buffer_size = new_size
        self._buffer = CudaContextBuffer.try_alloc(self.buffer_size)

    def __del__(self):
        if self._buffer is not None:
            ggml.ggml_cuda_host_free(self._buffer)
            self._buffer = None


### Replit Model Definition


class ReplitLayer:
    def __init__(self, wtype: int, n_embd: int, ctx: ggml.ggml_context_p):
        self.ln_1_weight = ggml.ggml_new_tensor_1d(ctx, ggml.GGML_TYPE_F32, n_embd)
        self.c_attn_wqkv_weight = ggml.ggml_new_tensor_2d(
            ctx, wtype, n_embd, 3 * n_embd
        )
        self.c_attn_out_proj_weight = ggml.ggml_new_tensor_2d(
            ctx, wtype, n_embd, n_embd
        )
        self.ln_2_weight = ggml.ggml_new_tensor_1d(ctx, ggml.GGML_TYPE_F32, n_embd)
        self.c_mlp_mlp_up_weight = ggml.ggml_new_tensor_2d(
            ctx, wtype, n_embd, 4 * n_embd
        )
        self.c_mlp_mlp_down_weight = ggml.ggml_new_tensor_2d(
            ctx, wtype, 4 * n_embd, n_embd
        )


class ReplitModel:
    def __init__(
        self,
        d_model: int,
        max_seq_len: int,
        n_heads: int,
        n_layers: int,
        vocab_size: int,
        ftype: int,
        vocab: List[Tuple[int, str, float]],
        n_batch: int,
        n_threads: int,
        weights_buffer: ContextBuffer,
        ctx: ggml.ggml_context_p,
    ):
        self.d_model = d_model
        self.max_seq_len = max_seq_len
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.vocab_size = vocab_size
        self.ftype = ftype
        self.ctx = ctx
        self.layers: List[ReplitLayer] = []
        self.tensors: Dict[str, ggml.ggml_tensor_p] = {}
        self.vocab = vocab
        self.n_batch = n_batch
        self.n_threads = n_threads
        self.weights_buffer = weights_buffer

        n_layer = self.n_layers
        n_embd = self.d_model
        n_ctx = self.max_seq_len
        n_vocab = self.vocab_size
        wtype = ggml.ggml_ftype_to_ggml_type(ftype)

        n_mem = n_layer * n_ctx
        n_elements = n_embd * n_mem

        self.memory_k = ggml.ggml_new_tensor_1d(ctx, ggml.GGML_TYPE_F16, n_elements)
        self.memory_v = ggml.ggml_new_tensor_1d(ctx, ggml.GGML_TYPE_F16, n_elements)

        if ggml.GGML_USE_CUBLAS:
            self.memory_k.contents.backend = ggml.GGML_BACKEND_GPU
            ggml.ggml_cuda_transform_tensor(self.memory_k.contents.data, self.memory_k)
            self.memory_v.contents.backend = ggml.GGML_BACKEND_GPU
            ggml.ggml_cuda_transform_tensor(self.memory_v.contents.data, self.memory_v)

        self.wte_weight = ggml.ggml_new_tensor_2d(ctx, wtype, n_embd, n_vocab)
        self.ln_f_weight = ggml.ggml_new_tensor_1d(ctx, ggml.GGML_TYPE_F32, n_embd)
        self.tensors["transformer.wte.weight"] = self.wte_weight
        self.tensors["transformer.norm_f.weight"] = self.ln_f_weight

        self.mem_per_token = 0
        if ggml.GGML_USE_CUBLAS:
            self.eval_buffer = CudaContextBuffer()
        else:
            self.eval_buffer = CpuContextBuffer()

        for i in range(n_layer):
            layer = ReplitLayer(
                wtype=wtype,
                n_embd=n_embd,
                ctx=ctx,
            )
            self.layers.append(layer)

            self.tensors[f"transformer.blocks.{i}.norm_1.weight"] = layer.ln_1_weight
            self.tensors[
                f"transformer.blocks.{i}.attn.Wqkv.weight"
            ] = layer.c_attn_wqkv_weight
            self.tensors[
                f"transformer.blocks.{i}.attn.out_proj.weight"
            ] = layer.c_attn_out_proj_weight
            self.tensors[f"transformer.blocks.{i}.norm_2.weight"] = layer.ln_2_weight
            self.tensors[
                f"transformer.blocks.{i}.ffn.up_proj.weight"
            ] = layer.c_mlp_mlp_up_weight
            self.tensors[
                f"transformer.blocks.{i}.ffn.down_proj.weight"
            ] = layer.c_mlp_mlp_down_weight

        self._input_ids: npt.NDArray[np.intc] = np.array([], dtype=np.intc)
        self._scores: npt.NDArray[np.single] = np.ndarray((n_vocab, 0), dtype=np.single)

    def __del__(self):
        ggml.ggml_free(self.ctx)

        if ggml.GGML_USE_CUBLAS:
            for tensor in self.tensors.values():
                # TODO: check tensor backend before freeing
                ggml.ggml_cuda_free_data(tensor)
            ggml.ggml_cuda_free_data(self.memory_k)
            ggml.ggml_cuda_free_data(self.memory_v)

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
        text = "".join(id_to_token[token][1] for token in tokens)
        detokenized = text.replace(ws_symbol.decode("utf-8"), " ")
        return detokenized

    def reset(self):
        self._input_ids = np.array([], dtype=np.intc)
        self._scores = np.ndarray((self.vocab_size, 0), dtype=np.single)

    def _eval_internal(self, embd_inp: Sequence[int], n_past: int, n_threads: int):
        N = len(embd_inp)

        n_embd = self.d_model
        n_layer = self.n_layers
        n_ctx = self.max_seq_len
        n_head = self.n_heads
        n_vocab = self.vocab_size

        def offload_nop(tensor: ggml.ggml_tensor_p):
            pass

        offload_func_nr = offload_nop
        offload_func_kq = offload_nop
        offload_func_v = offload_nop

        required_buffer_size = int(self.mem_per_token * N * 2.0)

        if (
            self.mem_per_token > 0
            and self.eval_buffer.buffer_size < required_buffer_size
        ):
            self.eval_buffer.resize(required_buffer_size)

        init_params = ggml.ggml_init_params(
            mem_size=self.eval_buffer.buffer_size,
            mem_buffer=self.eval_buffer.buffer,
            no_alloc=False,
        )
        ctx0 = ggml.ggml_init(init_params)
        gf = ggml.ggml_cgraph(n_threads=n_threads)

        embd = ggml.ggml_new_tensor_1d(
            ctx0,
            ggml.GGML_TYPE_I32,
            N,
        )
        ggml.ggml_set_name(embd, b"embd")
        to_numpy(embd)[:] = np.array(embd_inp, dtype=np.int32)

        inpL = ggml.ggml_get_rows(ctx0, self.wte_weight, embd)

        if ggml.GGML_USE_CUBLAS:
            offload_func_nr = ggml.ggml_cuda_assign_buffers_no_scratch
            offload_func_kq = ggml.ggml_cuda_assign_buffers_no_scratch
            offload_func_v = ggml.ggml_cuda_assign_buffers_no_scratch

        for il in range(n_layer):
            offload_func = offload_nop

            if ggml.GGML_USE_CUBLAS:
                offload_func = ggml.ggml_cuda_assign_buffers_no_scratch

            # // lctx.use_buf(ctx0, 0)

            # // a = self.ln_1(x)
            cur = ggml.ggml_norm(ctx0, inpL)
            # offload_func(cur)
            ggml.ggml_set_name(cur, b"norm_0")
            cur = ggml.ggml_mul(
                ctx0,
                ggml.ggml_repeat(ctx0, self.layers[il].ln_1_weight, cur),
                cur,
            )
            offload_func(cur)
            ggml.ggml_set_name(cur, b"attention_norm_0")

            # // self-attention
            # //  b, _, past_key_value = self.attn(a, past_key_value=past_key_value,
            # //  attn_bias=attn_bias, attention_mask=attention_mask,
            # //  is_causal=is_causal)

            # // compute QKV
            cur = ggml.ggml_mul_mat(ctx0, self.layers[il].c_attn_wqkv_weight, cur)
            if ggml.GGML_USE_CUBLAS:
                cur = ggml.ggml_cont(ctx0, cur) # NOTE: needed for CUDA
                offload_func_kq(cur)
            ggml.ggml_set_name(cur, b"tmpkqv")

            Qcur = ggml.ggml_view_2d(
                ctx0,
                cur,
                n_embd,
                N,
                cur.contents.nb[1],
                0 * ctypes.sizeof(ctypes.c_float) * n_embd,
            )
            # offload_func_kq(Qcur)
            ggml.ggml_set_name(Qcur, b"Qcur")
            Kcur = ggml.ggml_view_2d(
                ctx0,
                cur,
                n_embd,
                N,
                cur.contents.nb[1],
                1 * ctypes.sizeof(ctypes.c_float) * n_embd,
            )
            # offload_func_kq(Kcur)
            ggml.ggml_set_name(Kcur, b"Kcur")
            Vcur = ggml.ggml_view_2d(
                ctx0,
                cur,
                n_embd,
                N,
                cur.contents.nb[1],
                2 * ctypes.sizeof(ctypes.c_float) * n_embd,
            )
            # offload_func_v(Vcur)
            ggml.ggml_set_name(Vcur, b"Vcur")

            # // store key and value to memory
            k = ggml.ggml_view_1d(
                ctx0,
                self.memory_k,
                N * n_embd,
                (ggml.ggml_element_size(self.memory_k) * n_embd)
                * (il * n_ctx + n_past),
            )
            # offload_func_kq(k)
            ggml.ggml_set_name(k, b"k")
            v = ggml.ggml_view_1d(
                ctx0,
                self.memory_v,
                N * n_embd,
                (ggml.ggml_element_size(self.memory_v) * n_embd)
                * (il * n_ctx + n_past),
            )
            # offload_func_v(v)
            ggml.ggml_set_name(v, b"v")

            ggml.ggml_build_forward_expand(
                ctypes.pointer(gf),
                ggml.ggml_cpy(
                    ctx0,
                    Kcur,
                    k,
                ),
            )
            ggml.ggml_build_forward_expand(
                ctypes.pointer(gf),
                ggml.ggml_cpy(
                    ctx0,
                    Vcur,
                    v,
                ),
            )

            # // Q = Qcur.contiguous().view(n_embd/n_head, n_head, N).permute(0,
            # // 2, 1, 3) [64, N, 12]
            Q = ggml.ggml_permute(
                ctx0,
                ggml.ggml_cpy(
                    ctx0,
                    Qcur,
                    ggml.ggml_new_tensor_3d(
                        ctx0,
                        ggml.GGML_TYPE_F32,
                        n_embd // n_head,
                        n_head,
                        N,
                    ),
                ),
                0,
                2,
                1,
                3,
            )
            # offload_func_kq(Q)
            ggml.ggml_set_name(Q, b"Q")

            # // K = Kmem.view(n_embd/n_head, n_head, n_past + N).permute(0, 2, 1,
            # // 3) [64, n_past + N, 12]
            K = ggml.ggml_permute(
                ctx0,
                ggml.ggml_reshape_3d(
                    ctx0,
                    ggml.ggml_view_1d(
                        ctx0,
                        self.memory_k,
                        (n_past + N) * n_embd,
                        il * n_ctx * ggml.ggml_element_size(self.memory_k) * n_embd,
                    ),
                    n_embd // n_head,
                    n_head,
                    n_past + N,
                ),
                0,
                2,
                1,
                3,
            )
            # offload_func_kq(K)
            ggml.ggml_set_name(K, b"K")

            # // K * Q
            KQ = ggml.ggml_mul_mat(ctx0, K, Q)
            offload_func_kq(KQ)
            ggml.ggml_set_name(KQ, b"KQ")

            # // KQ_scaled = KQ / sqrt(n_embd/n_head)
            KQ_scaled = ggml.ggml_scale(
                ctx0,
                KQ,
                ggml.ggml_new_f32(
                    ctx0,
                    1.0 / np.sqrt(float(n_embd) / n_head),
                ),
            )
            # offload_func_kq(KQ_scaled)
            ggml.ggml_set_name(KQ_scaled, b"KQ_scaled")

            KQ_scaled_alibi = ggml.ggml_alibi(
                ctx0,
                KQ_scaled,
                n_past,
                n_head,
                8.0,
            )
            # offload_func_kq(KQ_scaled_alibi)
            ggml.ggml_set_name(KQ_scaled_alibi, b"KQ_scaled_alibi")

            # // KQ_masked = mask_past(KQ_scaled)
            KQ_masked = ggml.ggml_diag_mask_inf(
                ctx0,
                KQ_scaled_alibi,
                n_past,
            )
            # offload_func_kq(KQ_masked)
            ggml.ggml_set_name(KQ_masked, b"KQ_masked")

            # // KQ = soft_max(KQ_masked)
            KQ_soft_max = ggml.ggml_soft_max(
                ctx0,
                KQ_masked,
            )
            # offload_func_kq(KQ_soft_max)
            ggml.ggml_set_name(KQ_soft_max, b"KQ_soft_max")

            # // V_trans = Vmem.view(n_embd/n_head, n_head, n_past + N).permute(1,
            # // 2, 0, 3).contiguous() [n_past + N, 64, 12]
            V_trans = ggml.ggml_cpy(
                ctx0,
                ggml.ggml_permute(
                    ctx0,
                    ggml.ggml_reshape_3d(
                        ctx0,
                        ggml.ggml_view_1d(
                            ctx0,
                            self.memory_v,
                            (n_past + N) * n_embd,
                            il * n_ctx * ggml.ggml_element_size(self.memory_v) * n_embd,
                        ),
                        n_embd // n_head,
                        n_head,
                        n_past + N,
                    ),
                    1,
                    2,
                    0,
                    3,
                ),
                ggml.ggml_new_tensor_3d(
                    ctx0,
                    self.memory_v.contents.type,
                    n_past + N,
                    n_embd // n_head,
                    n_head,
                ),
            )
            # offload_func_v(V_trans)
            ggml.ggml_set_name(V_trans, b"V_trans")

            # // KQV = transpose(V) * KQ_soft_max
            KQV = ggml.ggml_mul_mat(ctx0, V_trans, KQ_soft_max)
            # offload_func_v(KQV)
            ggml.ggml_set_name(KQV, b"KQV")

            # // KQV_merged = KQV.permute(0, 2, 1, 3)
            KQV_merged = ggml.ggml_permute(
                ctx0,
                KQV,
                0,
                2,
                1,
                3,
            )
            # offload_func_v(KQV_merged)
            ggml.ggml_set_name(KQV_merged, b"KQV_merged")

            # // cur = KQV_merged.contiguous().view(n_embd, N)
            cur = ggml.ggml_cpy(
                ctx0,
                KQV_merged,
                ggml.ggml_new_tensor_2d(
                    ctx0,
                    ggml.GGML_TYPE_F32,
                    n_embd,
                    N,
                ),
            )
            # offload_func_v(cur)
            ggml.ggml_set_name(cur, b"KQV_merged_contiguous")

            # // projection
            cur = ggml.ggml_mul_mat(
                ctx0,
                self.layers[il].c_attn_out_proj_weight,
                cur,
            )
            offload_func_v(cur)
            ggml.ggml_set_name(cur, b"result_wo")

            # // lctx.use_buf(ctx0, 1)

            inpL = ggml.ggml_add(
                ctx0,
                inpL,
                cur,
            )
            offload_func(cur)
            ggml.ggml_set_name(cur, b"inpFF")

            # // m = self.ln_2(x)
            cur = ggml.ggml_norm(ctx0, inpL)
            # offload_func(cur)
            ggml.ggml_set_name(cur, b"norm_1")
            cur = ggml.ggml_mul(
                ctx0,
                ggml.ggml_repeat(ctx0, self.layers[il].ln_2_weight, cur),
                cur,
            )
            # offload_func(cur)
            ggml.ggml_set_name(cur, b"norm")

            # // n = self.mlp(m)
            cur = ggml.ggml_mul_mat(
                ctx0,
                self.layers[il].c_mlp_mlp_up_weight,
                cur,
            )
            # offload_func(cur)
            ggml.ggml_set_name(cur, b"result_mlp_up")

            # // GELU activation
            cur = ggml.ggml_gelu(
                ctx0,
                cur,
            )
            # offload_func(cur)
            ggml.ggml_set_name(cur, b"gelu")
            # // projection
            # // cur = proj_w*cur + proj_b
            cur = ggml.ggml_mul_mat(
                ctx0,
                self.layers[il].c_mlp_mlp_down_weight,
                cur,
            )
            offload_func(cur)
            ggml.ggml_set_name(cur, b"result_mlp_down")

            # // x = x + n
            inpL = ggml.ggml_add(
                ctx0,
                inpL,
                cur,
            )
            offload_func(cur)
            ggml.ggml_set_name(cur, b"inpFF_+_result_mlp_down")

        # // lctx.use_buf(ctx0, 0)

        # // norm
        inpL = ggml.ggml_norm(ctx0, inpL)
        # offload_func_nr(inpL)
        ggml.ggml_set_name(inpL, b"norm_f")

        # // inpL = ln_f_g*inpL
        inpL = ggml.ggml_mul(
            ctx0,
            ggml.ggml_repeat(ctx0, self.ln_f_weight, inpL),
            inpL,
        )
        ggml.ggml_set_name(inpL, b"norm_f_mul")

        # // output embedding weight tied to input embedding
        inpL = ggml.ggml_mul_mat(
            ctx0,
            self.wte_weight,
            inpL,
        )
        ggml.ggml_set_name(inpL, b"result_output")

        # // lctx.use_buf(ctx0, -1)

        ggml.ggml_build_forward_expand(ctypes.pointer(gf), inpL)
        ggml.ggml_graph_compute(ctx0, ctypes.pointer(gf))

        embd_w = to_numpy(inpL).reshape(n_vocab, -1).copy()

        self.mem_per_token = int(ggml.ggml_used_mem(ctx0) / N)

        ggml.ggml_free(ctx0)

        return embd_w

    def eval(self, tokens: Sequence[int]):
        if self.mem_per_token == 0:
            self._eval_internal([1, 2, 3, 4], n_past=0, n_threads=self.n_threads)
        n_ctx = self.max_seq_len
        for i in range(0, len(tokens), self.n_batch):
            batch = tokens[i : min(len(tokens), i + self.n_batch)]
            n_past = min(n_ctx - len(batch), len(self._input_ids))
            scores = self._eval_internal(
                batch,
                n_past,
                self.n_threads,
            )
            # Save tokens
            self._input_ids: npt.NDArray[np.intc] = np.concatenate(
                (self._input_ids, np.array(batch, dtype=np.intc)), axis=0
            )
            # Save logits
            logits = scores
            self._scores: npt.NDArray[np.single] = np.concatenate(
                (self._scores, np.array(logits, dtype=np.single)), axis=1
            )
        return self._scores

    def generate(
        self,
        tokens: Sequence[int],
        top_p: float = 0.95,
        temperature: float = 0.80,
        frequency_penalty: float = 0.0,
        presence_penalty: float = 0.0,
    ) -> Iterator[int]:
        reset = True
        if len(self._input_ids) > 0:
            longest_prefix = 0
            for a, b in zip(self._input_ids, tokens[:-1]):
                if a == b:
                    longest_prefix += 1
                else:
                    break
            if longest_prefix > 0:
                reset = False
                tokens = tokens[longest_prefix:]
                self._input_ids = self._input_ids[:longest_prefix]
                self._scores = self._scores[:, :longest_prefix]

        if reset:
            self.reset()

        while True:
            scores = self.eval(tokens)
            logits = scores[:, -1]
            token = sample(
                logits,
                top_p=top_p,
                temperature=temperature,
                frequency_penalty=frequency_penalty,
                presence_penalty=presence_penalty,
            )
            yield token
            tokens = [token]

    def _create_completion(
        self,
        prompt: str,
        suffix: Optional[str] = None,
        max_tokens: int = 16,
        temperature: float = 0.8,
        top_p: float = 0.95,
        logprobs: Optional[int] = None,
        echo: bool = False,
        stop: Optional[Union[str, List[str]]] = [],
        frequency_penalty: float = 0.0,
        presence_penalty: float = 0.0,
        stream: bool = False,
        model: Optional[str] = None,
    ) -> Union[Iterator[Completion], Iterator[CompletionChunk]]:
        completion_id: str = f"cmpl-{str(uuid.uuid4())}"
        created: int = int(time.time())
        completion_tokens: List[int] = []
        # Add blank space to start of prompt to match OG llama tokenizer
        prompt_tokens: List[int] = self.tokenize(prompt)
        text: str = ""
        returned_tokens: int = 0
        stop = (
            stop if isinstance(stop, list) else [stop] if isinstance(stop, str) else []
        )
        model_name: str = model if model is not None else "replit-code-v1-3b"

        # Truncate prompt if it is too long
        max_tokens = min(max_tokens, max(0, self.max_seq_len - len(prompt_tokens)))
        if len(prompt_tokens) + max_tokens > self.max_seq_len:
            raise ValueError(
                f"Requested tokens exceed context window of {self.max_seq_len}"
            )

        stop_sequences = stop if stop != [] else []
        finish_reason = "length"
        for token in self.generate(
            prompt_tokens,
            top_p=top_p,
            temperature=temperature,
            frequency_penalty=frequency_penalty,
            presence_penalty=presence_penalty,
        ):
            if token == self.eos_token():
                text = self.detokenize(completion_tokens)
                finish_reason = "stop"
                break

            completion_tokens.append(token)

            all_text = self.detokenize(completion_tokens)
            any_stop = [s for s in stop_sequences if s in all_text]
            if len(any_stop) > 0:
                first_stop = any_stop[0]
                text = all_text[: all_text.index(first_stop)]
                finish_reason = "stop"
                break

            if stream:
                remaining_tokens = completion_tokens[returned_tokens:]
                remaining_text = self.detokenize(remaining_tokens)
                remaining_length = len(remaining_text)

                # We want to avoid yielding any characters from
                # the generated text if they are part of a stop
                # sequence.
                first_stop_position = 0
                for s in stop_sequences:
                    for i in range(min(len(s), remaining_length), 0, -1):
                        if remaining_text.endswith(s[:i]):
                            if i > first_stop_position:
                                first_stop_position = i
                            break

                token_end_position = 0
                for token in remaining_tokens:
                    token_end_position += len(self.detokenize([token]))
                    # Check if stop sequence is in the token
                    if token_end_position >= (
                        remaining_length - first_stop_position - 1
                    ):
                        break
                    logprobs_or_none: Optional[CompletionLogprobs] = None
                    if logprobs is not None:
                        token_str = self.detokenize([token])
                        text_offset = len(prompt) + len(
                            self.detokenize(completion_tokens[:returned_tokens])
                        )
                        token_offset = len(prompt_tokens) + returned_tokens
                        logits = self._scores[token_offset - 1, :].tolist()
                        current_logprobs = logits_to_logprobs(logits)
                        sorted_logprobs = list(
                            sorted(
                                zip(current_logprobs, range(len(current_logprobs))),
                                reverse=True,
                            )
                        )
                        top_logprob = {
                            self.detokenize([i]): logprob
                            for logprob, i in sorted_logprobs[:logprobs]
                        }
                        top_logprob[token_str] = current_logprobs[int(token)]
                        logprobs_or_none = {
                            "tokens": [self.detokenize([token])],
                            "text_offset": [text_offset],
                            "token_logprobs": [sorted_logprobs[int(token)][0]],
                            "top_logprobs": [top_logprob],
                        }
                    returned_tokens += 1
                    yield {
                        "id": completion_id,
                        "object": "text_completion",
                        "created": created,
                        "model": model_name,
                        "choices": [
                            {
                                "text": self.detokenize([token]),
                                "index": 0,
                                "logprobs": logprobs_or_none,
                                "finish_reason": None,
                            }
                        ],
                    }

            if len(completion_tokens) >= max_tokens:
                text = self.detokenize(completion_tokens)
                finish_reason = "length"
                break

        if stream:
            remaining_tokens = completion_tokens[returned_tokens:]
            all_text = self.detokenize(remaining_tokens)
            any_stop = [s for s in stop_sequences if s in all_text]
            if len(any_stop) > 0:
                end = min(all_text.index(stop) for stop in any_stop)
            else:
                end = len(all_text)

            token_end_position = 0
            for token in remaining_tokens:
                token_end_position += len(self.detokenize([token]))

                logprobs_or_none: Optional[CompletionLogprobs] = None
                if logprobs is not None:
                    token_str = self.detokenize([token])
                    text_offset = len(prompt) + len(
                        self.detokenize(completion_tokens[:returned_tokens])
                    )
                    token_offset = len(prompt_tokens) + returned_tokens - 1
                    logits = self._scores[token_offset, :].tolist()
                    current_logprobs = logits_to_logprobs(logits)
                    sorted_logprobs = list(
                        sorted(
                            zip(current_logprobs, range(len(current_logprobs))),
                            reverse=True,
                        )
                    )
                    top_logprob = {
                        self.detokenize([i]): logprob
                        for logprob, i in sorted_logprobs[:logprobs]
                    }
                    top_logprob[token_str] = current_logprobs[int(token)]
                    logprobs_or_none = {
                        "tokens": [self.detokenize([token])],
                        "text_offset": [text_offset],
                        "token_logprobs": [sorted_logprobs[int(token)][0]],
                        "top_logprobs": [top_logprob],
                    }

                if token_end_position >= end:
                    last_text = self.detokenize([token])
                    if token_end_position == end - 1:
                        break
                    returned_tokens += 1
                    yield {
                        "id": completion_id,
                        "object": "text_completion",
                        "created": created,
                        "model": model_name,
                        "choices": [
                            {
                                "text": last_text[
                                    : len(last_text) - (token_end_position - end)
                                ],
                                "index": 0,
                                "logprobs": logprobs_or_none,
                                "finish_reason": finish_reason,
                            }
                        ],
                    }
                    break
                returned_tokens += 1
                yield {
                    "id": completion_id,
                    "object": "text_completion",
                    "created": created,
                    "model": model_name,
                    "choices": [
                        {
                            "text": self.detokenize([token]),
                            "index": 0,
                            "logprobs": logprobs_or_none,
                            "finish_reason": finish_reason
                            if returned_tokens == len(completion_tokens)
                            else None,
                        }
                    ],
                }
            return

        text_str = text

        if echo:
            text_str = prompt + text_str

        if suffix is not None:
            text_str = text_str + suffix

        logprobs_or_none: Optional[CompletionLogprobs] = None
        if logprobs is not None:
            text_offset = 0 if echo else len(prompt)
            token_offset = 0 if echo else len(prompt_tokens[1:])
            text_offsets: List[int] = []
            token_logprobs: List[Optional[float]] = []
            tokens: List[str] = []
            top_logprobs: List[Optional[Dict[str, float]]] = []

            if echo:
                # Remove leading BOS token
                all_tokens = prompt_tokens[1:] + completion_tokens
            else:
                all_tokens = completion_tokens

            all_token_strs = [self.detokenize([token]) for token in all_tokens]
            all_logprobs = [logits_to_logprobs(row.tolist()) for row in self._scores][
                token_offset:
            ]
            for token, token_str, logprobs_token in zip(
                all_tokens, all_token_strs, all_logprobs
            ):
                text_offsets.append(text_offset)
                text_offset += len(token_str)
                tokens.append(token_str)
                sorted_logprobs = list(
                    sorted(
                        zip(logprobs_token, range(len(logprobs_token))), reverse=True
                    )
                )
                token_logprobs.append(sorted_logprobs[int(token)][0])
                top_logprob: Optional[Dict[str, float]] = {
                    self.detokenize([i]): logprob
                    for logprob, i in sorted_logprobs[:logprobs]
                }
                top_logprob.update({token_str: logprobs_token[int(token)]})
                top_logprobs.append(top_logprob)
            # Weird idosincracy of the OpenAI API where
            # token_logprobs and top_logprobs are null for
            # the first token.
            if echo and len(all_tokens) > 0:
                token_logprobs[0] = None
                top_logprobs[0] = None
            logprobs_or_none = {
                "tokens": tokens,
                "text_offset": text_offsets,
                "token_logprobs": token_logprobs,
                "top_logprobs": top_logprobs,
            }

        yield {
            "id": completion_id,
            "object": "text_completion",
            "created": created,
            "model": model_name,
            "choices": [
                {
                    "text": text_str,
                    "index": 0,
                    "logprobs": logprobs_or_none,
                    "finish_reason": finish_reason,
                }
            ],
            "usage": {
                "prompt_tokens": len(prompt_tokens),
                "completion_tokens": len(completion_tokens),
                "total_tokens": len(prompt_tokens) + len(completion_tokens),
            },
        }

    def create_completion(
        self,
        prompt: str,
        suffix: Optional[str] = None,
        max_tokens: int = 128,
        temperature: float = 0.8,
        top_p: float = 0.95,
        logprobs: Optional[int] = None,
        echo: bool = False,
        stop: Optional[Union[str, List[str]]] = [],
        frequency_penalty: float = 0.0,
        presence_penalty: float = 0.0,
        stream: bool = False,
        model: Optional[str] = None,
    ) -> Union[Completion, Iterator[CompletionChunk]]:
        """Generate text from a prompt.

        Args:
            prompt: The prompt to generate text from.
            suffix: A suffix to append to the generated text. If None, no suffix is appended.
            max_tokens: The maximum number of tokens to generate.
            temperature: The temperature to use for sampling.
            top_p: The top-p value to use for sampling.
            logprobs: The number of logprobs to return. If None, no logprobs are returned.
            echo: Whether to echo the prompt.
            stop: A list of strings to stop generation when encountered.
            repeat_penalty: The penalty to apply to repeated tokens.
            top_k: The top-k value to use for sampling.
            stream: Whether to stream the results.

        Raises:
            ValueError: If the requested tokens exceed the context window.
            RuntimeError: If the prompt fails to tokenize or the model fails to evaluate the prompt.

        Returns:
            Response object containing the generated text.
        """
        completion_or_chunks = self._create_completion(
            prompt=prompt,
            suffix=suffix,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            logprobs=logprobs,
            echo=echo,
            stop=stop,
            frequency_penalty=frequency_penalty,
            presence_penalty=presence_penalty,
            stream=stream,
            model=model,
        )
        if stream:
            chunks: Iterator[CompletionChunk] = completion_or_chunks
            return chunks
        completion: Completion = next(completion_or_chunks)  # type: ignore
        return completion

    @staticmethod
    def eos_token():
        return 1

    @staticmethod
    def init_from_file(
        model_file: str,
        n_gpu_layers: int = 0,
        n_batch: int = 1,
        n_threads: int = 1,
        verbose: bool = True,
    ) -> ReplitModel:
        with open(model_file, "rb") as fin:
            # Magic Number
            (magic,) = struct.unpack("i", (fin.read(struct.calcsize("i"))))
            assert magic == ggml.GGML_FILE_MAGIC
            if verbose:
                print("magic number =", hex(magic))
            # Hyperparameters
            d_model, max_seq_len, n_heads, n_layers, vocab_size, ftype = struct.unpack(
                "iiiiii", (fin.read(struct.calcsize("iiiiii")))
            )
            qntvr = ftype // ggml.GGML_QNT_VERSION_FACTOR
            if verbose:
                print("d_model      =", d_model)
                print("max_seq_len  =", max_seq_len)
                print("n_heads      =", n_heads)
                print("n_layers     =", n_layers)
                print("vocab_size   =", vocab_size)
                print("ftype        =", ftype)
                print("qntvr        =", qntvr)
            ftype %= ggml.GGML_QNT_VERSION_FACTOR
            # Vocabulary
            vocab: List[Tuple[int, str, float]] = []
            for i in range(vocab_size):
                (s_len,) = struct.unpack("i", (fin.read(struct.calcsize("i"))))
                s = fin.read(s_len).decode("utf-8")
                (score,) = struct.unpack("f", (fin.read(struct.calcsize("f"))))
                vocab.append((i, s, score))
            # Model Weights
            wtype = ggml.ggml_ftype_to_ggml_type(ftype)

            n_embd = d_model
            n_layer = n_layers
            n_ctx = max_seq_len
            n_vocab = vocab_size

            ctx_size = ReplitModel.compute_ctx_size(
                n_embd=n_embd,
                n_layer=n_layer,
                n_ctx=n_ctx,
                n_vocab=n_vocab,
                wtype=wtype,
            )

            if verbose:
                print("ctx size     =", ctx_size // (1024 * 1024), "MB")

            # create context
            if ggml.GGML_USE_CUBLAS:
                weights_buffer = CudaContextBuffer(ctx_size)
                # ggml.ggml_cuda_set_main_device(0)
            else:
                weights_buffer = CpuContextBuffer(ctx_size)
            init_params = ggml.ggml_init_params(
                mem_size=ctx_size,
                mem_buffer=weights_buffer.buffer,
                no_alloc=False,
            )
            ctx = ggml.ggml_init(init_params)

            model = ReplitModel(
                # hyperparameters
                d_model=d_model,
                max_seq_len=max_seq_len,
                n_heads=n_heads,
                n_layers=n_layers,
                vocab_size=vocab_size,
                ftype=ftype,
                # vocabulary
                vocab=vocab,
                ctx=ctx,
                n_batch=n_batch,
                n_threads=n_threads,
                weights_buffer=weights_buffer,
            )

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
                if name not in model.tensors:
                    raise ValueError(f"Tensor {name} not found in model")
                tensor = model.tensors[name]
                if ggml.ggml_nelements(tensor) != nelements:
                    raise ValueError(
                        f"Tensor {name} has {ggml.ggml_nelements(tensor)} elements, but {nelements} expected"
                    )
                if tensor.contents.ne[0] != ne[0] or tensor.contents.ne[1] != ne[1]:
                    raise ValueError(
                        f"Tensor {name} has {tensor.contents.ne[0]}x{tensor.contents.ne[1]} shape, but {ne[0]}x{ne[1]} expected"
                    )
                bpe = ggml.ggml_type_size(ttype)
                if (
                    (nelements * bpe) / ggml.ggml_blck_size(tensor.contents.type)
                ) != ggml.ggml_nbytes(tensor):
                    raise ValueError(
                        f"Tensor {name} has {ggml.ggml_nbytes(tensor)} bytes, but {(nelements * bpe) / ggml.ggml_blck_size(tensor.contents.type)} expected"
                    )
                fin.readinto(
                    (ctypes.c_uint8 * ggml.ggml_nbytes(tensor)).from_address(
                        ggml.ggml_get_data(tensor)
                    )
                )
                if ggml.GGML_USE_CUBLAS and name.startswith("transformer.block"):
                    should_offload_suffix = [
                        # "norm_1.weight",
                        "attn.Wqkv.weight",
                        "attn.out_proj.weight",
                        # "norm_2.weight",
                        "ffn.up_proj.weight",
                        "ffn.down_proj.weight",
                    ]
                    if any(name.endswith(s) for s in should_offload_suffix):
                        tensor.contents.backend = ggml.GGML_BACKEND_GPU
                        ggml.ggml_cuda_transform_tensor(tensor.contents.data, tensor)
                    if name == "transformer.wte.weight" or name == "transformer.norm_f.weight":
                        tensor.contents.backend = ggml.GGML_BACKEND_GPU
                        ggml.ggml_cuda_transform_tensor(tensor.contents.data, tensor)

                total_size += ggml.ggml_nbytes(tensor)
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

        model.eval([1, 2, 3, 4])
        print("mem_per_token =", model.mem_per_token)
        return model

    @staticmethod
    def compute_ctx_size(
        n_embd: int,
        n_layer: int,
        n_ctx: int,
        n_vocab: int,
        wtype: int,
    ) -> int:
        wtype_sizef = ggml.ggml_type_sizef(wtype)
        f32_sizef = ggml.ggml_type_sizef(ggml.GGML_TYPE_F32)
        f16_sizef = ggml.ggml_type_sizef(ggml.GGML_TYPE_F16)

        ctx_size = 0
        ctx_size += n_embd * n_vocab * wtype_sizef
        ctx_size += n_embd * f32_sizef

        ctx_size += n_layer * (n_embd * f32_sizef)
        ctx_size += n_layer * (3 * n_embd * n_embd * wtype_sizef)
        ctx_size += n_layer * (n_embd**2 * wtype_sizef)
        ctx_size += n_layer * (n_embd * f32_sizef)
        ctx_size += n_layer * (4 * n_embd * n_embd * wtype_sizef)
        ctx_size += n_layer * (n_embd**2 * 4 * wtype_sizef)

        ctx_size += n_ctx * n_layer * n_embd * f16_sizef
        ctx_size += n_ctx * n_layer * n_embd * f16_sizef

        ctx_size += (1 + 6 * n_layer) * 512
        ctx_size = int(ctx_size)
        return ctx_size


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model", type=str, default=None)
    parser.add_argument("-p", "--prompt", type=str, default="def fib(n):")
    parser.add_argument(
        "--n_threads", type=int, default=max(1, multiprocessing.cpu_count() // 2)
    )
    parser.add_argument("--n_batch", type=int, default=512)
    parser.add_argument("--n_gpu_layers", type=int, default=0)
    parser.add_argument("--max_tokens", type=int, default=32)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top_p", type=float, default=1.0)
    parser.add_argument("--presence_penalty", type=float, default=0.0)
    parser.add_argument("--frequency_penalty", type=float, default=0.0)
    args = parser.parse_args()

    model_file = args.model
    n_threads = args.n_threads
    n_batch = args.n_batch
    n_gpu_layers = args.n_gpu_layers
    max_tokens = args.max_tokens
    temperature = args.temperature
    top_p = args.top_p
    presence_penalty = args.presence_penalty
    frequency_penalty = args.frequency_penalty

    model = ReplitModel.init_from_file(
        model_file, n_gpu_layers=n_gpu_layers, n_threads=n_threads, n_batch=n_batch
    )

    prompt = args.prompt
    prompt_tokens = model.tokenize(prompt)
    all_tokens: List[int] = prompt_tokens[:]  # type: ignore
    n_past = 0
    tokens: List[int] = prompt_tokens[:]  # type: ignore

    print("number of tokens in prompt =", len(prompt_tokens))
    for i, token_id in enumerate(prompt_tokens):
        print(f"token[{i}] =", token_id)

    print()
    print(prompt, end="", flush=True)
    for _ in range(max_tokens):
        # eval
        scores = model.eval(tokens)
        logits = scores[:, -1]
        # sample
        token_id = sample(
            logits,
            last_tokens=all_tokens,
            temperature=temperature,
            top_p=top_p,
            presence_penalty=presence_penalty,
            frequency_penalty=frequency_penalty,
        )
        if token_id == model.eos_token():
            break
        # update
        all_tokens.append(token_id)
        print(model.detokenize([token_id]), end="", flush=True)
        n_past += len(tokens)
        tokens = [token_id]
    print()
