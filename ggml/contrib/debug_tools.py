import ggml

from argparse import Namespace

output_names = [
    "/text_model/Constant_6_output_0",
    "/text_model/Slice_output_0",
    "/text_model/Constant_3_output_0",
    "/text_model/Constant_4_output_0",
    "/text_model/Constant_5_output_0",
    "/text_model/Shape_2_output_0",
]


def debug_handle(ctx):
    pclass = ctx["self"]
    del ctx["self"]

    ns = Namespace(**ctx)

    for output in ns.node.output:
        if output in output_names:
            tensor = ns.ggml_tensors[output]
            ggml.ggml_build_forward_expand(ns.gf_p, tensor)
            array = ggml.utils.to_numpy(tensor)
            print(f"{ns.node.name} -> {output} {array.shape}")
            print(array)
