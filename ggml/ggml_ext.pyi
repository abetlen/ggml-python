from typing import *
from enum import Enum
import ggml.ggml_ext

GGML_BACKEND_CL: ggml_backend

GGML_BACKEND_CPU: ggml_backend

GGML_BACKEND_CUDA: ggml_backend

GGML_DEFAULT_N_THREADS: int

GGML_FILE_MAGIC: int

GGML_FILE_VERSION: int

GGML_FTYPE_ALL_F32: ggml_ftype

GGML_FTYPE_MOSTLY_F16: ggml_ftype

GGML_FTYPE_MOSTLY_Q4_0: ggml_ftype

GGML_FTYPE_MOSTLY_Q4_1: ggml_ftype

GGML_FTYPE_MOSTLY_Q4_1_SOME_F16: ggml_ftype

GGML_FTYPE_MOSTLY_Q5_0: ggml_ftype

GGML_FTYPE_MOSTLY_Q5_1: ggml_ftype

GGML_FTYPE_MOSTLY_Q8_0: ggml_ftype

GGML_FTYPE_UNKNOWN: ggml_ftype

GGML_LINESEARCH_BACKTRACKING_ARMIJO: ggml_linesearch

GGML_LINESEARCH_BACKTRACKING_STRONG_WOLFE: ggml_linesearch

GGML_LINESEARCH_BACKTRACKING_WOLFE: ggml_linesearch

GGML_LINESEARCH_FAIL: ggml_opt_result

GGML_LINESEARCH_INVALID_PARAMETERS: ggml_opt_result

GGML_LINESEARCH_MAXIMUM_ITERATIONS: ggml_opt_result

GGML_LINESEARCH_MAXIMUM_STEP: ggml_opt_result

GGML_LINESEARCH_MINIMUM_STEP: ggml_opt_result

GGML_MAX_CONTEXTS: int

GGML_MAX_DIMS: int

GGML_MAX_NAME: int

GGML_MAX_NODES: int

GGML_MAX_OPT: int

GGML_MAX_PARAMS: int

GGML_OBJECT_SIZE: int

GGML_OPT_ADAM: ggml_opt_type

GGML_OPT_DID_NOT_CONVERGE: ggml_opt_result

GGML_OPT_FAIL: ggml_opt_result

GGML_OPT_INVALID_WOLFE: ggml_opt_result

GGML_OPT_LBFGS: ggml_opt_type

GGML_OPT_NO_CONTEXT: ggml_opt_result

GGML_OPT_OK: ggml_opt_result

GGML_OP_ABS: ggml_op

GGML_OP_ACC: ggml_op

GGML_OP_ADD: ggml_op

GGML_OP_ADD1: ggml_op

GGML_OP_ALIBI: ggml_op

GGML_OP_CLAMP: ggml_op

GGML_OP_CONT: ggml_op

GGML_OP_CONV_1D_S1_PH: ggml_op

GGML_OP_CONV_1D_S2_PH: ggml_op

GGML_OP_CONV_2D_SK_P0: ggml_op

GGML_OP_COUNT: ggml_op

GGML_OP_CPY: ggml_op

GGML_OP_DIAG: ggml_op

GGML_OP_DIAG_MASK_INF: ggml_op

GGML_OP_DIAG_MASK_ZERO: ggml_op

GGML_OP_DIV: ggml_op

GGML_OP_DUP: ggml_op

GGML_OP_FLASH_ATTN: ggml_op

GGML_OP_FLASH_FF: ggml_op

GGML_OP_GELU: ggml_op

GGML_OP_GELU_QUICK: ggml_op

GGML_OP_GET_ROWS: ggml_op

GGML_OP_GET_ROWS_BACK: ggml_op

GGML_OP_LOG: ggml_op

GGML_OP_MAP_BINARY: ggml_op

GGML_OP_MAP_UNARY: ggml_op

GGML_OP_MEAN: ggml_op

GGML_OP_MUL: ggml_op

GGML_OP_MUL_MAT: ggml_op

GGML_OP_NEG: ggml_op

GGML_OP_NONE: ggml_op

GGML_OP_NORM: ggml_op

GGML_OP_PERMUTE: ggml_op

GGML_OP_RELU: ggml_op

GGML_OP_REPEAT: ggml_op

GGML_OP_RESHAPE: ggml_op

GGML_OP_RMS_NORM: ggml_op

GGML_OP_RMS_NORM_BACK: ggml_op

GGML_OP_ROPE: ggml_op

GGML_OP_ROPE_BACK: ggml_op

GGML_OP_SCALE: ggml_op

GGML_OP_SET: ggml_op

GGML_OP_SGN: ggml_op

GGML_OP_SILU: ggml_op

GGML_OP_SILU_BACK: ggml_op

GGML_OP_SOFT_MAX: ggml_op

GGML_OP_SQR: ggml_op

GGML_OP_SQRT: ggml_op

GGML_OP_STEP: ggml_op

GGML_OP_SUB: ggml_op

GGML_OP_SUM: ggml_op

GGML_OP_SUM_ROWS: ggml_op

GGML_OP_TRANSPOSE: ggml_op

GGML_OP_VIEW: ggml_op

GGML_OP_WIN_PART: ggml_op

GGML_OP_WIN_UNPART: ggml_op

GGML_QNT_VERSION: int

GGML_QNT_VERSION_FACTOR: int

GGML_TENSOR_SIZE: int

GGML_TYPE_COUNT: ggml_type

GGML_TYPE_F16: ggml_type

GGML_TYPE_F32: ggml_type

GGML_TYPE_I16: ggml_type

GGML_TYPE_I32: ggml_type

GGML_TYPE_I8: ggml_type

GGML_TYPE_Q4_0: ggml_type

GGML_TYPE_Q4_1: ggml_type

GGML_TYPE_Q5_0: ggml_type

GGML_TYPE_Q5_1: ggml_type

GGML_TYPE_Q8_0: ggml_type

GGML_TYPE_Q8_1: ggml_type

def ggml_abs(ctx: ggml.ggml_ext.ggml_context_p, a: ggml.ggml_ext.ggml_tensor) -> ggml.ggml_ext.ggml_tensor:
    ...

def ggml_abs_inplace(ctx: ggml.ggml_ext.ggml_context_p, a: ggml.ggml_ext.ggml_tensor) -> ggml.ggml_ext.ggml_tensor:
    ...

def ggml_acc(ctx: ggml.ggml_ext.ggml_context_p, a: ggml.ggml_ext.ggml_tensor, b: ggml.ggml_ext.ggml_tensor, nb1: int, nb2: int, nb3: int, offset: int) -> ggml.ggml_ext.ggml_tensor:
    ...

def ggml_acc_inplace(ctx: ggml.ggml_ext.ggml_context_p, a: ggml.ggml_ext.ggml_tensor, b: ggml.ggml_ext.ggml_tensor, nb1: int, nb2: int, nb3: int, offset: int) -> ggml.ggml_ext.ggml_tensor:
    ...

def ggml_add(ctx: ggml.ggml_ext.ggml_context_p, a: ggml.ggml_ext.ggml_tensor, b: ggml.ggml_ext.ggml_tensor) -> ggml.ggml_ext.ggml_tensor:
    ...

def ggml_add1(ctx: ggml.ggml_ext.ggml_context_p, a: ggml.ggml_ext.ggml_tensor, b: ggml.ggml_ext.ggml_tensor) -> ggml.ggml_ext.ggml_tensor:
    ...

def ggml_add1_inplace(ctx: ggml.ggml_ext.ggml_context_p, a: ggml.ggml_ext.ggml_tensor, b: ggml.ggml_ext.ggml_tensor) -> ggml.ggml_ext.ggml_tensor:
    ...

def ggml_add_inplace(ctx: ggml.ggml_ext.ggml_context_p, a: ggml.ggml_ext.ggml_tensor, b: ggml.ggml_ext.ggml_tensor) -> ggml.ggml_ext.ggml_tensor:
    ...

def ggml_alibi(ctx: ggml.ggml_ext.ggml_context_p, a: ggml.ggml_ext.ggml_tensor, n_past: int, n_head: int, bias_max: float) -> ggml.ggml_ext.ggml_tensor:
    ...

class ggml_backend(Enum):
    """
    <attribute '__doc__' of 'ggml_backend' objects>
    """

    @entries: dict
    
    GGML_BACKEND_CL: Any
    
    GGML_BACKEND_CPU: Any
    
    GGML_BACKEND_CUDA: Any
    
def ggml_blck_size(arg: ggml.ggml_ext.ggml_type, /) -> int:
    ...

def ggml_build_backward(ctx: ggml.ggml_ext.ggml_context_p, gf: ggml.ggml_ext.ggml_cgraph, keep: bool) -> ggml.ggml_ext.ggml_cgraph:
    ...

def ggml_build_forward(tensor: ggml.ggml_ext.ggml_tensor) -> ggml.ggml_ext.ggml_cgraph:
    ...

def ggml_build_forward_expand(cgraph: ggml.ggml_ext.ggml_cgraph, tensor: ggml.ggml_ext.ggml_tensor) -> None:
    ...

class ggml_cgraph:

    def __init__(*args, **kwargs):
        """
        Initialize self.  See help(type(self)) for accurate signature.
        """
        ...
    
def ggml_clamp(ctx: ggml.ggml_ext.ggml_context_p, a: ggml.ggml_ext.ggml_tensor, min: float, max: float) -> ggml.ggml_ext.ggml_tensor:
    ...

def ggml_cont(ctx: ggml.ggml_ext.ggml_context_p, a: ggml.ggml_ext.ggml_tensor) -> ggml.ggml_ext.ggml_tensor:
    ...

class ggml_context_p:

    def __init__(*args, **kwargs):
        """
        Initialize self.  See help(type(self)) for accurate signature.
        """
        ...
    
def ggml_conv_1d_s1_ph(ctx: ggml.ggml_ext.ggml_context_p, a: ggml.ggml_ext.ggml_tensor, b: ggml.ggml_ext.ggml_tensor) -> ggml.ggml_ext.ggml_tensor:
    ...

def ggml_conv_1d_s2_ph(ctx: ggml.ggml_ext.ggml_context_p, a: ggml.ggml_ext.ggml_tensor, b: ggml.ggml_ext.ggml_tensor) -> ggml.ggml_ext.ggml_tensor:
    ...

def ggml_conv_2d_sk_p0(ctx: ggml.ggml_ext.ggml_context_p, a: ggml.ggml_ext.ggml_tensor, b: ggml.ggml_ext.ggml_tensor) -> ggml.ggml_ext.ggml_tensor:
    ...

def ggml_cpu_has_arm_fma() -> int:
    ...

def ggml_cpu_has_avx() -> int:
    ...

def ggml_cpu_has_avx2() -> int:
    ...

def ggml_cpu_has_avx512() -> int:
    ...

def ggml_cpu_has_avx512_vbmi() -> int:
    ...

def ggml_cpu_has_avx512_vnni() -> int:
    ...

def ggml_cpu_has_blas() -> int:
    ...

def ggml_cpu_has_clblast() -> int:
    ...

def ggml_cpu_has_cublas() -> int:
    ...

def ggml_cpu_has_f16c() -> int:
    ...

def ggml_cpu_has_fma() -> int:
    ...

def ggml_cpu_has_fp16_va() -> int:
    ...

def ggml_cpu_has_gpublas() -> int:
    ...

def ggml_cpu_has_neon() -> int:
    ...

def ggml_cpu_has_sse3() -> int:
    ...

def ggml_cpu_has_vsx() -> int:
    ...

def ggml_cpu_has_wasm_simd() -> int:
    ...

def ggml_cpy(ctx: ggml.ggml_ext.ggml_context_p, a: ggml.ggml_ext.ggml_tensor, b: ggml.ggml_ext.ggml_tensor) -> ggml.ggml_ext.ggml_tensor:
    ...

def ggml_cycles() -> int:
    ...

def ggml_cycles_per_ms() -> int:
    ...

def ggml_diag(ctx: ggml.ggml_ext.ggml_context_p, a: ggml.ggml_ext.ggml_tensor) -> ggml.ggml_ext.ggml_tensor:
    ...

def ggml_diag_mask_inf(ctx: ggml.ggml_ext.ggml_context_p, a: ggml.ggml_ext.ggml_tensor, n_past: int) -> ggml.ggml_ext.ggml_tensor:
    ...

def ggml_diag_mask_inf_inplace(ctx: ggml.ggml_ext.ggml_context_p, a: ggml.ggml_ext.ggml_tensor, n_past: int) -> ggml.ggml_ext.ggml_tensor:
    ...

def ggml_diag_mask_zero(ctx: ggml.ggml_ext.ggml_context_p, a: ggml.ggml_ext.ggml_tensor, n_past: int) -> ggml.ggml_ext.ggml_tensor:
    ...

def ggml_diag_mask_zero_inplace(ctx: ggml.ggml_ext.ggml_context_p, a: ggml.ggml_ext.ggml_tensor, n_past: int) -> ggml.ggml_ext.ggml_tensor:
    ...

def ggml_div(ctx: ggml.ggml_ext.ggml_context_p, a: ggml.ggml_ext.ggml_tensor, b: ggml.ggml_ext.ggml_tensor) -> ggml.ggml_ext.ggml_tensor:
    ...

def ggml_div_inplace(ctx: ggml.ggml_ext.ggml_context_p, a: ggml.ggml_ext.ggml_tensor, b: ggml.ggml_ext.ggml_tensor) -> ggml.ggml_ext.ggml_tensor:
    ...

def ggml_dup(ctx: ggml.ggml_ext.ggml_context_p, a: ggml.ggml_ext.ggml_tensor) -> ggml.ggml_ext.ggml_tensor:
    ...

def ggml_dup_tensor(ctx: ggml.ggml_ext.ggml_context_p, src: ggml.ggml_ext.ggml_tensor) -> ggml.ggml_ext.ggml_tensor:
    ...

def ggml_element_size(arg: ggml.ggml_ext.ggml_tensor, /) -> int:
    ...

def ggml_flash_attn(ctx: ggml.ggml_ext.ggml_context_p, q: ggml.ggml_ext.ggml_tensor, k: ggml.ggml_ext.ggml_tensor, v: ggml.ggml_ext.ggml_tensor, masked: bool) -> ggml.ggml_ext.ggml_tensor:
    ...

def ggml_flash_ff(ctx: ggml.ggml_ext.ggml_context_p, a: ggml.ggml_ext.ggml_tensor, b0: ggml.ggml_ext.ggml_tensor, b1: ggml.ggml_ext.ggml_tensor, c0: ggml.ggml_ext.ggml_tensor, c1: ggml.ggml_ext.ggml_tensor) -> ggml.ggml_ext.ggml_tensor:
    ...

class ggml_fp16_t:

    def __init__(*args, **kwargs):
        """
        Initialize self.  See help(type(self)) for accurate signature.
        """
        ...
    
def ggml_fp16_to_fp32(x: int) -> float:
    ...

def ggml_fp16_to_fp32_row(x: int, y: float, n: int) -> None:
    ...

def ggml_fp32_to_fp16(x: float) -> int:
    ...

def ggml_fp32_to_fp16_row(x: float, y: int, n: int) -> None:
    ...

def ggml_free(ctx: ggml.ggml_ext.ggml_context_p) -> None:
    ...

class ggml_ftype(Enum):
    """
    <attribute '__doc__' of 'ggml_ftype' objects>
    """

    @entries: dict
    
    GGML_FTYPE_ALL_F32: Any
    
    GGML_FTYPE_MOSTLY_F16: Any
    
    GGML_FTYPE_MOSTLY_Q4_0: Any
    
    GGML_FTYPE_MOSTLY_Q4_1: Any
    
    GGML_FTYPE_MOSTLY_Q4_1_SOME_F16: Any
    
    GGML_FTYPE_MOSTLY_Q5_0: Any
    
    GGML_FTYPE_MOSTLY_Q5_1: Any
    
    GGML_FTYPE_MOSTLY_Q8_0: Any
    
    GGML_FTYPE_UNKNOWN: Any
    
def ggml_ftype_to_ggml_type(arg: ggml.ggml_ext.ggml_ftype, /) -> ggml.ggml_ext.ggml_type:
    ...

def ggml_gelu(ctx: ggml.ggml_ext.ggml_context_p, a: ggml.ggml_ext.ggml_tensor) -> ggml.ggml_ext.ggml_tensor:
    ...

def ggml_gelu_inplace(ctx: ggml.ggml_ext.ggml_context_p, a: ggml.ggml_ext.ggml_tensor) -> ggml.ggml_ext.ggml_tensor:
    ...

def ggml_gelu_quick(ctx: ggml.ggml_ext.ggml_context_p, a: ggml.ggml_ext.ggml_tensor) -> ggml.ggml_ext.ggml_tensor:
    ...

def ggml_gelu_quick_inplace(ctx: ggml.ggml_ext.ggml_context_p, a: ggml.ggml_ext.ggml_tensor) -> ggml.ggml_ext.ggml_tensor:
    ...

def ggml_get_data(tensor: ggml.ggml_ext.ggml_tensor) -> capsule:
    ...

def ggml_get_data_f32(tensor: ggml.ggml_ext.ggml_tensor) -> float:
    ...

def ggml_get_f32_1d(tensor: ggml.ggml_ext.ggml_tensor, i: int) -> float:
    ...

def ggml_get_i32_1d(tensor: ggml.ggml_ext.ggml_tensor, i: int) -> int:
    ...

def ggml_get_mem_buffer(arg: ggml.ggml_ext.ggml_context_p, /) -> capsule:
    ...

def ggml_get_mem_size(arg: ggml.ggml_ext.ggml_context_p, /) -> int:
    ...

def ggml_get_name(tensor: ggml.ggml_ext.ggml_tensor) -> str:
    ...

def ggml_get_rows(ctx: ggml.ggml_ext.ggml_context_p, a: ggml.ggml_ext.ggml_tensor, b: ggml.ggml_ext.ggml_tensor) -> ggml.ggml_ext.ggml_tensor:
    ...

def ggml_get_rows_back(ctx: ggml.ggml_ext.ggml_context_p, a: ggml.ggml_ext.ggml_tensor, b: ggml.ggml_ext.ggml_tensor, c: ggml.ggml_ext.ggml_tensor) -> ggml.ggml_ext.ggml_tensor:
    ...

def ggml_get_tensor(ctx: ggml.ggml_ext.ggml_context_p, name: str) -> ggml.ggml_ext.ggml_tensor:
    ...

def ggml_graph_compute(ctx: ggml.ggml_ext.ggml_context_p, cgraph: ggml.ggml_ext.ggml_cgraph) -> None:
    ...

def ggml_graph_dump_dot(gb: ggml.ggml_ext.ggml_cgraph, gf: ggml.ggml_ext.ggml_cgraph, filename: str) -> None:
    ...

def ggml_graph_export(cgraph: ggml.ggml_ext.ggml_cgraph, fname: str) -> None:
    ...

def ggml_graph_get_tensor(cgraph: ggml.ggml_ext.ggml_cgraph, name: str) -> ggml.ggml_ext.ggml_tensor:
    ...

def ggml_graph_import(fname: str, ctx_data: ggml.ggml_ext.ggml_context_p, ctx_eval: ggml.ggml_ext.ggml_context_p) -> ggml.ggml_ext.ggml_cgraph:
    ...

def ggml_graph_print(cgraph: ggml.ggml_ext.ggml_cgraph) -> None:
    ...

def ggml_graph_reset(cgraph: ggml.ggml_ext.ggml_cgraph) -> None:
    ...

def ggml_init(params: ggml.ggml_ext.ggml_init_params) -> ggml.ggml_ext.ggml_context_p:
    ...

class ggml_init_params:

    def __init__(self, mem_size: int = 0, mem_buffer: Optional[capsule] = None, no_alloc: bool = True) -> None:
        ...
    
    @property
    def mem_buffer(self) -> capsule:
        ...
    @mem_buffer.setter
    def mem_buffer(self, arg: ndarray[], /) -> None:
        ...
    
    @property
    def mem_size(self) -> int:
        ...
    @mem_size.setter
    def mem_size(self, arg: int, /) -> None:
        ...
    
    @property
    def no_alloc(self) -> bool:
        ...
    @no_alloc.setter
    def no_alloc(self, arg: bool, /) -> None:
        ...
    
def ggml_is_quantized(arg: ggml.ggml_ext.ggml_type, /) -> bool:
    ...

class ggml_linesearch(Enum):
    """
    <attribute '__doc__' of 'ggml_linesearch' objects>
    """

    @entries: dict
    
    GGML_LINESEARCH_BACKTRACKING_ARMIJO: Any
    
    GGML_LINESEARCH_BACKTRACKING_STRONG_WOLFE: Any
    
    GGML_LINESEARCH_BACKTRACKING_WOLFE: Any
    
    GGML_LINESEARCH_DEFAULT: Any
    
def ggml_log(ctx: ggml.ggml_ext.ggml_context_p, a: ggml.ggml_ext.ggml_tensor) -> ggml.ggml_ext.ggml_tensor:
    ...

def ggml_log_inplace(ctx: ggml.ggml_ext.ggml_context_p, a: ggml.ggml_ext.ggml_tensor) -> ggml.ggml_ext.ggml_tensor:
    ...

def ggml_map_binary_f32(*args, **kwargs):
    """
    ggml_map_binary_f32(ctx: ggml.ggml_ext.ggml_context_p, a: ggml.ggml_ext.ggml_tensor, b: ggml.ggml_ext.ggml_tensor, fun: void (int, float*, float const*, float const*)) -> ggml.ggml_ext.ggml_tensor
    """
    ...

def ggml_map_unary_f32(*args, **kwargs):
    """
    ggml_map_unary_f32(ctx: ggml.ggml_ext.ggml_context_p, a: ggml.ggml_ext.ggml_tensor, fun: void (int, float*, float const*)) -> ggml.ggml_ext.ggml_tensor
    """
    ...

def ggml_mean(ctx: ggml.ggml_ext.ggml_context_p, a: ggml.ggml_ext.ggml_tensor) -> ggml.ggml_ext.ggml_tensor:
    ...

def ggml_mul(ctx: ggml.ggml_ext.ggml_context_p, a: ggml.ggml_ext.ggml_tensor, b: ggml.ggml_ext.ggml_tensor) -> ggml.ggml_ext.ggml_tensor:
    ...

def ggml_mul_inplace(ctx: ggml.ggml_ext.ggml_context_p, a: ggml.ggml_ext.ggml_tensor, b: ggml.ggml_ext.ggml_tensor) -> ggml.ggml_ext.ggml_tensor:
    ...

def ggml_mul_mat(ctx: ggml.ggml_ext.ggml_context_p, a: ggml.ggml_ext.ggml_tensor, b: ggml.ggml_ext.ggml_tensor) -> ggml.ggml_ext.ggml_tensor:
    ...

def ggml_nbytes(arg: ggml.ggml_ext.ggml_tensor, /) -> int:
    ...

def ggml_neg(ctx: ggml.ggml_ext.ggml_context_p, a: ggml.ggml_ext.ggml_tensor) -> ggml.ggml_ext.ggml_tensor:
    ...

def ggml_neg_inplace(ctx: ggml.ggml_ext.ggml_context_p, a: ggml.ggml_ext.ggml_tensor) -> ggml.ggml_ext.ggml_tensor:
    ...

def ggml_nelements(arg: ggml.ggml_ext.ggml_tensor, /) -> int:
    ...

def ggml_new_f32(ctx: ggml.ggml_ext.ggml_context_p, value: float) -> ggml.ggml_ext.ggml_tensor:
    ...

def ggml_new_i32(ctx: ggml.ggml_ext.ggml_context_p, value: int) -> ggml.ggml_ext.ggml_tensor:
    ...

def ggml_new_tensor(ctx: ggml.ggml_ext.ggml_context_p, type: ggml.ggml_ext.ggml_type, n_dims: int, ne: int) -> ggml.ggml_ext.ggml_tensor:
    ...

def ggml_new_tensor_1d(ctx: ggml.ggml_ext.ggml_context_p, type: ggml.ggml_ext.ggml_type, ne0: int) -> ggml.ggml_ext.ggml_tensor:
    ...

def ggml_new_tensor_2d(ctx: ggml.ggml_ext.ggml_context_p, type: ggml.ggml_ext.ggml_type, ne0: int, ne1: int) -> ggml.ggml_ext.ggml_tensor:
    ...

def ggml_new_tensor_3d(ctx: ggml.ggml_ext.ggml_context_p, type: ggml.ggml_ext.ggml_type, ne0: int, ne1: int, ne2: int) -> ggml.ggml_ext.ggml_tensor:
    ...

def ggml_new_tensor_4d(ctx: ggml.ggml_ext.ggml_context_p, type: ggml.ggml_ext.ggml_type, ne0: int, ne1: int, ne2: int, ne3: int) -> ggml.ggml_ext.ggml_tensor:
    ...

def ggml_norm(ctx: ggml.ggml_ext.ggml_context_p, a: ggml.ggml_ext.ggml_tensor) -> ggml.ggml_ext.ggml_tensor:
    ...

def ggml_norm_inplace(ctx: ggml.ggml_ext.ggml_context_p, a: ggml.ggml_ext.ggml_tensor) -> ggml.ggml_ext.ggml_tensor:
    ...

class ggml_object:

    def __init__(*args, **kwargs):
        """
        Initialize self.  See help(type(self)) for accurate signature.
        """
        ...
    
class ggml_op(Enum):
    """
    <attribute '__doc__' of 'ggml_op' objects>
    """

    @entries: dict
    
    GGML_OP_ABS: Any
    
    GGML_OP_ACC: Any
    
    GGML_OP_ADD: Any
    
    GGML_OP_ADD1: Any
    
    GGML_OP_ALIBI: Any
    
    GGML_OP_CLAMP: Any
    
    GGML_OP_CONT: Any
    
    GGML_OP_CONV_1D_S1_PH: Any
    
    GGML_OP_CONV_1D_S2_PH: Any
    
    GGML_OP_CONV_2D_SK_P0: Any
    
    GGML_OP_COUNT: Any
    
    GGML_OP_CPY: Any
    
    GGML_OP_DIAG: Any
    
    GGML_OP_DIAG_MASK_INF: Any
    
    GGML_OP_DIAG_MASK_ZERO: Any
    
    GGML_OP_DIV: Any
    
    GGML_OP_DUP: Any
    
    GGML_OP_FLASH_ATTN: Any
    
    GGML_OP_FLASH_FF: Any
    
    GGML_OP_GELU: Any
    
    GGML_OP_GELU_QUICK: Any
    
    GGML_OP_GET_ROWS: Any
    
    GGML_OP_GET_ROWS_BACK: Any
    
    GGML_OP_LOG: Any
    
    GGML_OP_MAP_BINARY: Any
    
    GGML_OP_MAP_UNARY: Any
    
    GGML_OP_MEAN: Any
    
    GGML_OP_MUL: Any
    
    GGML_OP_MUL_MAT: Any
    
    GGML_OP_NEG: Any
    
    GGML_OP_NONE: Any
    
    GGML_OP_NORM: Any
    
    GGML_OP_PERMUTE: Any
    
    GGML_OP_RELU: Any
    
    GGML_OP_REPEAT: Any
    
    GGML_OP_RESHAPE: Any
    
    GGML_OP_RMS_NORM: Any
    
    GGML_OP_RMS_NORM_BACK: Any
    
    GGML_OP_ROPE: Any
    
    GGML_OP_ROPE_BACK: Any
    
    GGML_OP_SCALE: Any
    
    GGML_OP_SET: Any
    
    GGML_OP_SGN: Any
    
    GGML_OP_SILU: Any
    
    GGML_OP_SILU_BACK: Any
    
    GGML_OP_SOFT_MAX: Any
    
    GGML_OP_SQR: Any
    
    GGML_OP_SQRT: Any
    
    GGML_OP_STEP: Any
    
    GGML_OP_SUB: Any
    
    GGML_OP_SUM: Any
    
    GGML_OP_SUM_ROWS: Any
    
    GGML_OP_TRANSPOSE: Any
    
    GGML_OP_VIEW: Any
    
    GGML_OP_WIN_PART: Any
    
    GGML_OP_WIN_UNPART: Any
    
def ggml_op_name(arg: ggml.ggml_ext.ggml_op, /) -> str:
    ...

def ggml_opt(ctx: ggml.ggml_ext.ggml_context_p, params: ggml.ggml_ext.ggml_opt_params, f: ggml.ggml_ext.ggml_tensor) -> ggml.ggml_ext.ggml_opt_result:
    ...

def ggml_opt_default_params(type: ggml.ggml_ext.ggml_opt_type) -> ggml.ggml_ext.ggml_opt_params:
    ...

class ggml_opt_params:

    def __init__(*args, **kwargs):
        """
        Initialize self.  See help(type(self)) for accurate signature.
        """
        ...
    
class ggml_opt_result(Enum):
    """
    <attribute '__doc__' of 'ggml_opt_result' objects>
    """

    @entries: dict
    
    GGML_LINESEARCH_FAIL: Any
    
    GGML_LINESEARCH_INVALID_PARAMETERS: Any
    
    GGML_LINESEARCH_MAXIMUM_ITERATIONS: Any
    
    GGML_LINESEARCH_MAXIMUM_STEP: Any
    
    GGML_LINESEARCH_MINIMUM_STEP: Any
    
    GGML_OPT_DID_NOT_CONVERGE: Any
    
    GGML_OPT_FAIL: Any
    
    GGML_OPT_INVALID_WOLFE: Any
    
    GGML_OPT_NO_CONTEXT: Any
    
    GGML_OPT_OK: Any
    
class ggml_opt_type(Enum):
    """
    <attribute '__doc__' of 'ggml_opt_type' objects>
    """

    @entries: dict
    
    GGML_OPT_ADAM: Any
    
    GGML_OPT_LBFGS: Any
    
def ggml_permute(ctx: ggml.ggml_ext.ggml_context_p, a: ggml.ggml_ext.ggml_tensor, axis0: int, axis1: int, axis2: int, axis3: int) -> ggml.ggml_ext.ggml_tensor:
    ...

def ggml_print_object(arg: ggml.ggml_ext.ggml_object, /) -> None:
    ...

def ggml_print_objects(arg: ggml.ggml_ext.ggml_context_p, /) -> None:
    ...

def ggml_quantize_chunk(type: ggml.ggml_ext.ggml_type, src: float, dst: capsule, start: int, n: int, hist: int) -> int:
    ...

def ggml_quantize_q4_0(src: float, dst: capsule, n: int, k: int, hist: int) -> int:
    ...

def ggml_quantize_q4_1(src: float, dst: capsule, n: int, k: int, hist: int) -> int:
    ...

def ggml_quantize_q5_0(src: float, dst: capsule, n: int, k: int, hist: int) -> int:
    ...

def ggml_quantize_q5_1(src: float, dst: capsule, n: int, k: int, hist: int) -> int:
    ...

def ggml_quantize_q8_0(src: float, dst: capsule, n: int, k: int, hist: int) -> int:
    ...

def ggml_relu(ctx: ggml.ggml_ext.ggml_context_p, a: ggml.ggml_ext.ggml_tensor) -> ggml.ggml_ext.ggml_tensor:
    ...

def ggml_relu_inplace(ctx: ggml.ggml_ext.ggml_context_p, a: ggml.ggml_ext.ggml_tensor) -> ggml.ggml_ext.ggml_tensor:
    ...

def ggml_repeat(ctx: ggml.ggml_ext.ggml_context_p, a: ggml.ggml_ext.ggml_tensor, b: ggml.ggml_ext.ggml_tensor) -> ggml.ggml_ext.ggml_tensor:
    ...

def ggml_reshape(ctx: ggml.ggml_ext.ggml_context_p, a: ggml.ggml_ext.ggml_tensor, b: ggml.ggml_ext.ggml_tensor) -> ggml.ggml_ext.ggml_tensor:
    ...

def ggml_reshape_1d(ctx: ggml.ggml_ext.ggml_context_p, a: ggml.ggml_ext.ggml_tensor, ne0: int) -> ggml.ggml_ext.ggml_tensor:
    ...

def ggml_reshape_2d(ctx: ggml.ggml_ext.ggml_context_p, a: ggml.ggml_ext.ggml_tensor, ne0: int, ne1: int) -> ggml.ggml_ext.ggml_tensor:
    ...

def ggml_reshape_3d(ctx: ggml.ggml_ext.ggml_context_p, a: ggml.ggml_ext.ggml_tensor, ne0: int, ne1: int, ne2: int) -> ggml.ggml_ext.ggml_tensor:
    ...

def ggml_reshape_4d(ctx: ggml.ggml_ext.ggml_context_p, a: ggml.ggml_ext.ggml_tensor, ne0: int, ne1: int, ne2: int, ne3: int) -> ggml.ggml_ext.ggml_tensor:
    ...

def ggml_rms_norm(ctx: ggml.ggml_ext.ggml_context_p, a: ggml.ggml_ext.ggml_tensor) -> ggml.ggml_ext.ggml_tensor:
    ...

def ggml_rms_norm_back(ctx: ggml.ggml_ext.ggml_context_p, a: ggml.ggml_ext.ggml_tensor, b: ggml.ggml_ext.ggml_tensor) -> ggml.ggml_ext.ggml_tensor:
    ...

def ggml_rms_norm_inplace(ctx: ggml.ggml_ext.ggml_context_p, a: ggml.ggml_ext.ggml_tensor) -> ggml.ggml_ext.ggml_tensor:
    ...

def ggml_rope(ctx: ggml.ggml_ext.ggml_context_p, a: ggml.ggml_ext.ggml_tensor, n_past: int, n_dims: int, mode: int) -> ggml.ggml_ext.ggml_tensor:
    ...

def ggml_rope_back(ctx: ggml.ggml_ext.ggml_context_p, a: ggml.ggml_ext.ggml_tensor, n_past: int, n_dims: int, mode: int) -> ggml.ggml_ext.ggml_tensor:
    ...

def ggml_rope_inplace(ctx: ggml.ggml_ext.ggml_context_p, a: ggml.ggml_ext.ggml_tensor, n_past: int, n_dims: int, mode: int) -> ggml.ggml_ext.ggml_tensor:
    ...

def ggml_scale(ctx: ggml.ggml_ext.ggml_context_p, a: ggml.ggml_ext.ggml_tensor, b: ggml.ggml_ext.ggml_tensor) -> ggml.ggml_ext.ggml_tensor:
    ...

def ggml_scale_inplace(ctx: ggml.ggml_ext.ggml_context_p, a: ggml.ggml_ext.ggml_tensor, b: ggml.ggml_ext.ggml_tensor) -> ggml.ggml_ext.ggml_tensor:
    ...

class ggml_scratch:

    def __init__(*args, **kwargs):
        """
        Initialize self.  See help(type(self)) for accurate signature.
        """
        ...
    
def ggml_set(ctx: ggml.ggml_ext.ggml_context_p, a: ggml.ggml_ext.ggml_tensor, b: ggml.ggml_ext.ggml_tensor, nb1: int, nb2: int, nb3: int, offset: int) -> ggml.ggml_ext.ggml_tensor:
    ...

def ggml_set_1d(ctx: ggml.ggml_ext.ggml_context_p, a: ggml.ggml_ext.ggml_tensor, b: ggml.ggml_ext.ggml_tensor, offset: int) -> ggml.ggml_ext.ggml_tensor:
    ...

def ggml_set_1d_inplace(ctx: ggml.ggml_ext.ggml_context_p, a: ggml.ggml_ext.ggml_tensor, b: ggml.ggml_ext.ggml_tensor, offset: int) -> ggml.ggml_ext.ggml_tensor:
    ...

def ggml_set_2d(ctx: ggml.ggml_ext.ggml_context_p, a: ggml.ggml_ext.ggml_tensor, b: ggml.ggml_ext.ggml_tensor, nb1: int, offset: int) -> ggml.ggml_ext.ggml_tensor:
    ...

def ggml_set_2d_inplace(ctx: ggml.ggml_ext.ggml_context_p, a: ggml.ggml_ext.ggml_tensor, b: ggml.ggml_ext.ggml_tensor, nb1: int, offset: int) -> ggml.ggml_ext.ggml_tensor:
    ...

def ggml_set_f32(tensor: ggml.ggml_ext.ggml_tensor, value: float) -> ggml.ggml_ext.ggml_tensor:
    ...

def ggml_set_f32_1d(tensor: ggml.ggml_ext.ggml_tensor, i: int, value: float) -> None:
    ...

def ggml_set_i32(tensor: ggml.ggml_ext.ggml_tensor, value: int) -> ggml.ggml_ext.ggml_tensor:
    ...

def ggml_set_i32_1d(tensor: ggml.ggml_ext.ggml_tensor, i: int, value: int) -> None:
    ...

def ggml_set_inplace(ctx: ggml.ggml_ext.ggml_context_p, a: ggml.ggml_ext.ggml_tensor, b: ggml.ggml_ext.ggml_tensor, nb1: int, nb2: int, nb3: int, offset: int) -> ggml.ggml_ext.ggml_tensor:
    ...

def ggml_set_name(tensor: ggml.ggml_ext.ggml_tensor, name: str) -> ggml.ggml_ext.ggml_tensor:
    ...

def ggml_set_no_alloc(ctx: ggml.ggml_ext.ggml_context_p, no_alloc: bool) -> None:
    ...

def ggml_set_param(ctx: ggml.ggml_ext.ggml_context_p, tensor: ggml.ggml_ext.ggml_tensor) -> None:
    ...

def ggml_set_scratch(ctx: ggml.ggml_ext.ggml_context_p, scratch: ggml.ggml_ext.ggml_scratch) -> int:
    ...

def ggml_set_zero(tensor: ggml.ggml_ext.ggml_tensor) -> ggml.ggml_ext.ggml_tensor:
    ...

def ggml_sgn(ctx: ggml.ggml_ext.ggml_context_p, a: ggml.ggml_ext.ggml_tensor) -> ggml.ggml_ext.ggml_tensor:
    ...

def ggml_sgn_inplace(ctx: ggml.ggml_ext.ggml_context_p, a: ggml.ggml_ext.ggml_tensor) -> ggml.ggml_ext.ggml_tensor:
    ...

def ggml_silu(ctx: ggml.ggml_ext.ggml_context_p, a: ggml.ggml_ext.ggml_tensor) -> ggml.ggml_ext.ggml_tensor:
    ...

def ggml_silu_back(ctx: ggml.ggml_ext.ggml_context_p, a: ggml.ggml_ext.ggml_tensor, b: ggml.ggml_ext.ggml_tensor) -> ggml.ggml_ext.ggml_tensor:
    ...

def ggml_silu_inplace(ctx: ggml.ggml_ext.ggml_context_p, a: ggml.ggml_ext.ggml_tensor) -> ggml.ggml_ext.ggml_tensor:
    ...

def ggml_soft_max(ctx: ggml.ggml_ext.ggml_context_p, a: ggml.ggml_ext.ggml_tensor) -> ggml.ggml_ext.ggml_tensor:
    ...

def ggml_soft_max_inplace(ctx: ggml.ggml_ext.ggml_context_p, a: ggml.ggml_ext.ggml_tensor) -> ggml.ggml_ext.ggml_tensor:
    ...

def ggml_sqr(ctx: ggml.ggml_ext.ggml_context_p, a: ggml.ggml_ext.ggml_tensor) -> ggml.ggml_ext.ggml_tensor:
    ...

def ggml_sqr_inplace(ctx: ggml.ggml_ext.ggml_context_p, a: ggml.ggml_ext.ggml_tensor) -> ggml.ggml_ext.ggml_tensor:
    ...

def ggml_sqrt(ctx: ggml.ggml_ext.ggml_context_p, a: ggml.ggml_ext.ggml_tensor) -> ggml.ggml_ext.ggml_tensor:
    ...

def ggml_sqrt_inplace(ctx: ggml.ggml_ext.ggml_context_p, a: ggml.ggml_ext.ggml_tensor) -> ggml.ggml_ext.ggml_tensor:
    ...

def ggml_step(ctx: ggml.ggml_ext.ggml_context_p, a: ggml.ggml_ext.ggml_tensor) -> ggml.ggml_ext.ggml_tensor:
    ...

def ggml_step_inplace(ctx: ggml.ggml_ext.ggml_context_p, a: ggml.ggml_ext.ggml_tensor) -> ggml.ggml_ext.ggml_tensor:
    ...

def ggml_sub(ctx: ggml.ggml_ext.ggml_context_p, a: ggml.ggml_ext.ggml_tensor, b: ggml.ggml_ext.ggml_tensor) -> ggml.ggml_ext.ggml_tensor:
    ...

def ggml_sub_inplace(ctx: ggml.ggml_ext.ggml_context_p, a: ggml.ggml_ext.ggml_tensor, b: ggml.ggml_ext.ggml_tensor) -> ggml.ggml_ext.ggml_tensor:
    ...

def ggml_sum(ctx: ggml.ggml_ext.ggml_context_p, a: ggml.ggml_ext.ggml_tensor) -> ggml.ggml_ext.ggml_tensor:
    ...

def ggml_sum_rows(ctx: ggml.ggml_ext.ggml_context_p, a: ggml.ggml_ext.ggml_tensor) -> ggml.ggml_ext.ggml_tensor:
    ...

class ggml_tensor:

    def __init__(*args, **kwargs):
        """
        Initialize self.  See help(type(self)) for accurate signature.
        """
        ...
    
    @property
    def backend(self) -> ggml.ggml_ext.ggml_backend:
        ...
    @backend.setter
    def backend(self, arg: ggml.ggml_ext.ggml_backend, /) -> None:
        ...
    
    @property
    def data(self) -> capsule:
        ...
    @data.setter
    def data(self, arg: capsule, /) -> None:
        ...
    
    @property
    def grad(self) -> ggml.ggml_ext.ggml_tensor:
        ...
    @grad.setter
    def grad(self, arg: ggml.ggml_ext.ggml_tensor, /) -> None:
        ...
    
    @property
    def is_param(self) -> bool:
        ...
    @is_param.setter
    def is_param(self, arg: bool, /) -> None:
        ...
    
    @property
    def n_dims(self) -> int:
        ...
    @n_dims.setter
    def n_dims(self, arg: int, /) -> None:
        ...
    
    @property
    def n_tasks(self) -> int:
        ...
    @n_tasks.setter
    def n_tasks(self, arg: int, /) -> None:
        ...
    
    @property
    def name(self) -> std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> >:
        ...
    @name.setter
    def name(self, arg: std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> >, /) -> None:
        ...
    
    @property
    def nb(self) -> List[int]:
        ...
    @nb.setter
    def nb(self, arg: List[int], /) -> None:
        ...
    
    @property
    def ne(self) -> List[int]:
        ...
    @ne.setter
    def ne(self, arg: List[int], /) -> None:
        ...
    
    @property
    def op(self) -> ggml.ggml_ext.ggml_op:
        ...
    @op.setter
    def op(self, arg: ggml.ggml_ext.ggml_op, /) -> None:
        ...
    
    @property
    def opt(self) -> List[ggml.ggml_ext.ggml_tensor]:
        ...
    @opt.setter
    def opt(self, arg: List[ggml.ggml_ext.ggml_tensor], /) -> None:
        ...
    
    @property
    def padding(self) -> List[str]:
        ...
    @padding.setter
    def padding(self, arg: List[str], /) -> None:
        ...
    
    @property
    def perf_cycles(self) -> int:
        ...
    @perf_cycles.setter
    def perf_cycles(self, arg: int, /) -> None:
        ...
    
    @property
    def perf_runs(self) -> int:
        ...
    @perf_runs.setter
    def perf_runs(self, arg: int, /) -> None:
        ...
    
    @property
    def perf_time_us(self) -> int:
        ...
    @perf_time_us.setter
    def perf_time_us(self, arg: int, /) -> None:
        ...
    
    @property
    def src0(self) -> ggml.ggml_ext.ggml_tensor:
        ...
    @src0.setter
    def src0(self, arg: ggml.ggml_ext.ggml_tensor, /) -> None:
        ...
    
    @property
    def src1(self) -> ggml.ggml_ext.ggml_tensor:
        ...
    @src1.setter
    def src1(self, arg: ggml.ggml_ext.ggml_tensor, /) -> None:
        ...
    
    @property
    def type(self) -> ggml.ggml_ext.ggml_type:
        ...
    @type.setter
    def type(self, arg: ggml.ggml_ext.ggml_type, /) -> None:
        ...
    
def ggml_tensor_overhead() -> int:
    ...

def ggml_time_init() -> None:
    ...

def ggml_time_ms() -> int:
    ...

def ggml_time_us() -> int:
    ...

def ggml_transpose(ctx: ggml.ggml_ext.ggml_context_p, a: ggml.ggml_ext.ggml_tensor) -> ggml.ggml_ext.ggml_tensor:
    ...

class ggml_type(Enum):
    """
    <attribute '__doc__' of 'ggml_type' objects>
    """

    @entries: dict
    
    GGML_TYPE_COUNT: Any
    
    GGML_TYPE_F16: Any
    
    GGML_TYPE_F32: Any
    
    GGML_TYPE_I16: Any
    
    GGML_TYPE_I32: Any
    
    GGML_TYPE_I8: Any
    
    GGML_TYPE_Q4_0: Any
    
    GGML_TYPE_Q4_1: Any
    
    GGML_TYPE_Q5_0: Any
    
    GGML_TYPE_Q5_1: Any
    
    GGML_TYPE_Q8_0: Any
    
    GGML_TYPE_Q8_1: Any
    
def ggml_type_name(arg: ggml.ggml_ext.ggml_type, /) -> str:
    ...

def ggml_type_size(arg: ggml.ggml_ext.ggml_type, /) -> int:
    ...

def ggml_type_sizef(arg: ggml.ggml_ext.ggml_type, /) -> float:
    ...

def ggml_used_mem(ctx: ggml.ggml_ext.ggml_context_p) -> int:
    ...

def ggml_view_1d(ctx: ggml.ggml_ext.ggml_context_p, a: ggml.ggml_ext.ggml_tensor, ne0: int, offset: int) -> ggml.ggml_ext.ggml_tensor:
    ...

def ggml_view_2d(ctx: ggml.ggml_ext.ggml_context_p, a: ggml.ggml_ext.ggml_tensor, ne0: int, ne1: int, nb1: int, offset: int) -> ggml.ggml_ext.ggml_tensor:
    ...

def ggml_view_3d(ctx: ggml.ggml_ext.ggml_context_p, a: ggml.ggml_ext.ggml_tensor, ne0: int, ne1: int, ne2: int, nb1: int, nb2: int, offset: int) -> ggml.ggml_ext.ggml_tensor:
    ...

def ggml_view_4d(ctx: ggml.ggml_ext.ggml_context_p, a: ggml.ggml_ext.ggml_tensor, ne0: int, ne1: int, ne2: int, ne3: int, nb1: int, nb2: int, nb3: int, offset: int) -> ggml.ggml_ext.ggml_tensor:
    ...

def ggml_view_tensor(ctx: ggml.ggml_ext.ggml_context_p, src: ggml.ggml_ext.ggml_tensor) -> ggml.ggml_ext.ggml_tensor:
    ...

def ggml_win_part(ctx: ggml.ggml_ext.ggml_context_p, a: ggml.ggml_ext.ggml_tensor, w: int) -> ggml.ggml_ext.ggml_tensor:
    ...

def ggml_win_unpart(ctx: ggml.ggml_ext.ggml_context_p, a: ggml.ggml_ext.ggml_tensor, w0: int, h0: int, w: int) -> ggml.ggml_ext.ggml_tensor:
    ...

