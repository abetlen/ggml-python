# Convert CLIP model from PyTorch to ggml format
#
# Usage: python convert-pt-to-ggml.py ViT-B-32 ./models
#
# This script loads the specified model and clip assets and saves them in ggml format.
# The output is a single binary file containing the following information:
#
#  - hparams
#  - tokenizer vocab
#  - model variables
#
# For each variable, write the following:
#
#  - Number of dimensions (int)
#  - Name length (int)
#  - Dimensions (int[n_dims])
#  - Name (char[name_length])
#  - Data (float[n_dims])
#

import os
import sys
import struct
import gzip
import numpy as np
import clip

if len(sys.argv) < 3:
    print("Usage: convert-pt-to-ggml.py clip_model dir-output\n")
    sys.exit(1)

clip_model = sys.argv[1]
dir_out = sys.argv[2]

# CLIP repo needs to exist at the root directory
MODELS = clip.clip._MODELS
model_filename = os.path.basename(MODELS[clip_model]).replace(".pt", "")

model = clip.load(clip_model, device="cpu")
state_dict = model[0].state_dict()

# output in the same directory as the model
fname_out = os.path.join(dir_out, model_filename + ".ggml")
os.makedirs(dir_out, exist_ok=True)

fout = open(fname_out, "wb")

# Get HParams
# Only ViT models supported for now
vit = True
if vit:
    vision_width = state_dict["visual.conv1.weight"].shape[0]
    vision_layers = len(
        [
            k
            for k in state_dict.keys()
            if k.startswith("visual.") and k.endswith(".attn.in_proj_weight")
        ]
    )
    vision_patch_size = state_dict["visual.conv1.weight"].shape[-1]
    grid_size = round((state_dict["visual.positional_embedding"].shape[0] - 1) ** 0.5)
    image_resolution = vision_patch_size * grid_size
embed_dim = state_dict["text_projection"].shape[1]
context_length = state_dict["positional_embedding"].shape[0]
vocab_size = state_dict["token_embedding.weight"].shape[0]
transformer_width = state_dict["ln_final.weight"].shape[0]
transformer_heads = transformer_width // 64
transformer_layers = len(
    set(k.split(".")[2] for k in state_dict if k.startswith("transformer.resblocks"))
)
print("HParams:")
print("  vision_width:", vision_width)
print("  vision_layers:", vision_layers)
print("  vision_patch_size:", vision_patch_size)
print("  grid_size:", grid_size)
print("  image_resolution:", image_resolution)
print("  embed_dim:", embed_dim)
print("  context_length:", context_length)
print("  vocab_size:", vocab_size)
print("  transformer_width:", transformer_width)
print("  transformer_heads:", transformer_heads)
print("  transformer_layers:", transformer_layers)


ftype = 0

# Write hparams
fout.write(struct.pack("i", 0x67676D6C))  # magic: ggml in hex
fout.write(struct.pack("i", vision_width))
fout.write(struct.pack("i", vision_layers))
fout.write(struct.pack("i", vision_patch_size))
fout.write(struct.pack("i", grid_size))
fout.write(struct.pack("i", image_resolution))
fout.write(struct.pack("i", embed_dim))
fout.write(struct.pack("i", context_length))
fout.write(struct.pack("i", transformer_width))
fout.write(struct.pack("i", transformer_heads))
fout.write(struct.pack("i", transformer_layers))
fout.write(struct.pack("i", ftype))  # ftype: 0 = float32, 1 = float16

bpe_path = os.path.join(os.path.dirname(clip.__file__), "bpe_simple_vocab_16e6.txt.gz")
merges = gzip.open(bpe_path).read().decode("utf-8").split("\n")
merges = merges[1 : 49152 - 256 - 2 + 1]
merges = [tuple(merge.split()) for merge in merges]

vocab = list(clip.simple_tokenizer.bytes_to_unicode().values())
tokens = vocab + [v + "</w>" for v in vocab]
for merge in merges:
    tokens.append("".join(merge))
tokens.extend(["<|startoftext|>", "<|endoftext|>"])
# byte_decoder = {v: k for k, v in clip.simple_tokenizer.bytes_to_unicode().items()}

fout.write(struct.pack("i", len(tokens)))

for key in tokens:
    text = key.encode("utf-8")
    fout.write(struct.pack("i", len(text)))
    fout.write(text)

for name in state_dict.keys():
    data = state_dict[name].squeeze().numpy()
    print("Processing variable: " + name + " with shape: ", data.shape)
    # ftype == 0 -> float32, ftype == 1 -> float16
    ftype = 0

    if name == "visual.conv1.weight":
        data = data.astype(np.float16)
        ftype = 1
    n_dims = len(data.shape)

    # header
    str = name.encode("utf-8")
    fout.write(struct.pack("iii", n_dims, len(str), ftype))
    for i in range(n_dims):
        fout.write(struct.pack("i", data.shape[n_dims - 1 - i]))
    fout.write(str)

    # data
    data.tofile(fout)

fout.close()

print("Done. Output file: " + fname_out)
print("")
