"""ggml-python implemention of the CLIP model
"""
import ctypes
import struct
import argparse
import numpy as np
from typing import List, Tuple
from CLIP.clip import simple_tokenizer
import ggml
from ggml.experimental import GGML_FTYPE, Context, InitParams, Tensor, GGML_TYPE, CGraph


class ClipModel:
    def __init__(
        self,
    ):
        pass

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

            # mem_buffer = np.empty(ctx_size, dtype=np.uint8)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model", type=str, default=None)

    args = parser.parse_args()

    model_file = args.model

    model = ClipModel.init_from_file(model_file)
