import os
import sys
import ctypes
import pathlib

# Load the library
def load_shared_library(lib_base_name: str):
    # Determine the file extension based on the platform
    if sys.platform.startswith("linux"):
        lib_ext = ".so"
    elif sys.platform == "darwin":
        lib_ext = ".so"
    elif sys.platform == "win32":
        lib_ext = ".dll"
    else:
        raise RuntimeError("Unsupported platform")

    # Construct the paths to the possible shared library names
    _base_path = pathlib.Path(__file__).parent.resolve()
    # Searching for the library in the current directory under the name "libllama" (default name
    # for llamacpp) and "llama" (default name for this repo)
    _lib_paths = [
        _base_path / f"lib{lib_base_name}{lib_ext}",
        _base_path / f"{lib_base_name}{lib_ext}",
    ]

    cdll_args = dict() # type: ignore
    # Add the library directory to the DLL search path on Windows (if needed)
    if sys.platform == "win32" and sys.version_info >= (3, 8):
        os.add_dll_directory(str(_base_path))
        cdll_args["winmode"] = 0

    # Try to load the shared library, handling potential errors
    for _lib_path in _lib_paths:
        if _lib_path.exists():
            try:
                return ctypes.CDLL(str(_lib_path), **cdll_args)
            except Exception as e:
                raise RuntimeError(f"Failed to load shared library '{_lib_path}': {e}")

    raise FileNotFoundError(
        f"Shared library with base name '{lib_base_name}' not found"
    )


lib_base_name = "ggml"
lib = load_shared_library(lib_base_name)


# #define GGML_FILE_MAGIC   0x67676d6c // "ggml"
GGML_FILE_MAGIC = ctypes.c_int(int("0x67676d6c", 16))
# #define GGML_FILE_VERSION 1
GGML_FILE_VERSION = ctypes.c_int(1)
# #define GGML_QNT_VERSION        1    // bump this on quantization format changes
GGML_QNT_VERSION = ctypes.c_int(1)
# #define GGML_QNT_VERSION_FACTOR 1000 // do not change this
GGML_QNT_VERSION_FACTOR = ctypes.c_int(1000)
# #define GGML_MAX_DIMS          4
GGML_MAX_DIMS = ctypes.c_int(4)
# #define GGML_MAX_NODES         4096
GGML_MAX_NODES = ctypes.c_int(4096)
# #define GGML_MAX_PARAMS        256
GGML_MAX_PARAMS = ctypes.c_int(256)
# #define GGML_MAX_CONTEXTS      64
GGML_MAX_CONTEXTS = ctypes.c_int(64)
# #define GGML_MAX_OPT           4
GGML_MAX_OPT = ctypes.c_int(4) 
# #define GGML_DEFAULT_N_THREADS 4
GGML_DEFAULT_N_THREADS = ctypes.c_int(4)


# typedef uint16_t ggml_fp16_t;
ggml_fp16_t = ctypes.c_uint16


# GGML_API float       ggml_fp16_to_fp32(ggml_fp16_t x);
def ggml_fp16_to_fp32(x):
    return lib.ggml_fp16_to_fp32(x)

lib.ggml_fp16_to_fp32.argtypes = [ggml_fp16_t]
lib.ggml_fp16_to_fp32.restype = ctypes.c_float

# GGML_API ggml_fp16_t ggml_fp32_to_fp16(float x);
def ggml_fp32_to_fp16(x):
    return lib.ggml_fp32_to_fp16(x)

lib.ggml_fp32_to_fp16.argtypes = [ctypes.c_float]
lib.ggml_fp32_to_fp16.restype = ggml_fp16_t

# GGML_API void ggml_fp16_to_fp32_row(const ggml_fp16_t * x, float * y, size_t n);
def ggml_fp16_to_fp32_row(x, y, n):
    return lib.ggml_fp16_to_fp32_row(x, y, n)

lib.ggml_fp16_to_fp32_row.argtypes = [ctypes.POINTER(ggml_fp16_t), ctypes.POINTER(ctypes.c_float), ctypes.c_size_t]
lib.ggml_fp16_to_fp32_row.restype = None

# GGML_API void ggml_fp32_to_fp16_row(const float * x, ggml_fp16_t * y, size_t n);
def ggml_fp32_to_fp16_row(x, y, n):
    return lib.ggml_fp32_to_fp16_row(x, y, n)

lib.ggml_fp32_to_fp16_row.argtypes = [ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ggml_fp16_t), ctypes.c_size_t]
lib.ggml_fp32_to_fp16_row.restype = None


ggml_context_p = ctypes.c_void_p


# enum ggml_type {
#     GGML_TYPE_F32  = 0,
#     GGML_TYPE_F16  = 1,
#     GGML_TYPE_Q4_0 = 2,
#     GGML_TYPE_Q4_1 = 3,
#     // GGML_TYPE_Q4_2 = 4, support has been removed
#     // GGML_TYPE_Q4_3 (5) support has been removed
#     GGML_TYPE_Q5_0 = 6,
#     GGML_TYPE_Q5_1 = 7,
#     GGML_TYPE_Q8_0 = 8,
#     GGML_TYPE_Q8_1 = 9,
#     GGML_TYPE_I8,
#     GGML_TYPE_I16,
#     GGML_TYPE_I32,
#     GGML_TYPE_COUNT,
# };
GGML_TYPE_F32 = ctypes.c_int(0)
GGML_TYPE_F16 = ctypes.c_int(1)
GGML_TYPE_Q4_0 = ctypes.c_int(2)
GGML_TYPE_Q4_1 = ctypes.c_int(3)
GGML_TYPE_Q5_0 = ctypes.c_int(6)
GGML_TYPE_Q5_1 = ctypes.c_int(7)
GGML_TYPE_Q8_0 = ctypes.c_int(8)
GGML_TYPE_Q8_1 = ctypes.c_int(9)
GGML_TYPE_I8 = ctypes.c_int(10)
GGML_TYPE_I16 = ctypes.c_int(11)
GGML_TYPE_I32 = ctypes.c_int(12)
GGML_TYPE_COUNT = ctypes.c_int(13)


# enum ggml_backend {
#     GGML_BACKEND_CPU = 0,
#     GGML_BACKEND_CUDA = 1,
# };
GGML_BACKEND_CPU = ctypes.c_int(0)
GGML_BACKEND_CUDA = ctypes.c_int(1)


# // model file types
# enum ggml_ftype {
#     GGML_FTYPE_UNKNOWN     = -1,
#     GGML_FTYPE_ALL_F32     = 0,
#     GGML_FTYPE_MOSTLY_F16  = 1,  // except 1d tensors
#     GGML_FTYPE_MOSTLY_Q4_0 = 2,  // except 1d tensors
#     GGML_FTYPE_MOSTLY_Q4_1 = 3,  // except 1d tensors
#     GGML_FTYPE_MOSTLY_Q4_1_SOME_F16 = 4, // tok_embeddings.weight and output.weight are F16
#     GGML_FTYPE_MOSTLY_Q8_0 = 7,  // except 1d tensors
#     GGML_FTYPE_MOSTLY_Q5_0 = 8,  // except 1d tensors
#     GGML_FTYPE_MOSTLY_Q5_1 = 9,  // except 1d tensors
# };
GGML_FTYPE_UNKNOWN = ctypes.c_int(-1)
GGML_FTYPE_ALL_F32 = ctypes.c_int(0)
GGML_FTYPE_MOSTLY_F16 = ctypes.c_int(1)
GGML_FTYPE_MOSTLY_Q4_0 = ctypes.c_int(2)
GGML_FTYPE_MOSTLY_Q4_1 = ctypes.c_int(3)
GGML_FTYPE_MOSTLY_Q4_1_SOME_F16 = ctypes.c_int(4)
GGML_FTYPE_MOSTLY_Q8_0 = ctypes.c_int(7)
GGML_FTYPE_MOSTLY_Q5_0 = ctypes.c_int(8)
GGML_FTYPE_MOSTLY_Q5_1 = ctypes.c_int(9)


# // available tensor operations:
# enum ggml_op {
#     GGML_OP_NONE = 0,

#     GGML_OP_DUP,
#     GGML_OP_ADD,
#     GGML_OP_ADD1,
#     GGML_OP_ACC,
#     GGML_OP_SUB,
#     GGML_OP_MUL,
#     GGML_OP_DIV,
#     GGML_OP_SQR,
#     GGML_OP_SQRT,
#     GGML_OP_LOG,
#     GGML_OP_SUM,
#     GGML_OP_SUM_ROWS,
#     GGML_OP_MEAN,
#     GGML_OP_REPEAT,
#     GGML_OP_ABS,
#     GGML_OP_SGN,
#     GGML_OP_NEG,
#     GGML_OP_STEP,
#     GGML_OP_RELU,
#     GGML_OP_GELU,
#     GGML_OP_SILU,
#     GGML_OP_SILU_BACK,
#     GGML_OP_NORM, // normalize
#     GGML_OP_RMS_NORM,
#     GGML_OP_RMS_NORM_BACK,

#     GGML_OP_MUL_MAT,

#     GGML_OP_SCALE,
#     GGML_OP_SET,
#     GGML_OP_CPY,
#     GGML_OP_CONT,
#     GGML_OP_RESHAPE,
#     GGML_OP_VIEW,
#     GGML_OP_PERMUTE,
#     GGML_OP_TRANSPOSE,
#     GGML_OP_GET_ROWS,
#     GGML_OP_GET_ROWS_BACK,
#     GGML_OP_DIAG,
#     GGML_OP_DIAG_MASK_INF,
#     GGML_OP_DIAG_MASK_ZERO,
#     GGML_OP_SOFT_MAX,
#     GGML_OP_ROPE,
#     GGML_OP_ROPE_BACK,
#     GGML_OP_ALIBI,
#     GGML_OP_CONV_1D_1S,
#     GGML_OP_CONV_1D_2S,

#     GGML_OP_FLASH_ATTN,
#     GGML_OP_FLASH_FF,

#     GGML_OP_MAP_UNARY,
#     GGML_OP_MAP_BINARY,

#     GGML_OP_COUNT,
# };
GGML_OP_NONE = ctypes.c_int(0)

GGML_OP_DUP = ctypes.c_int(1)
GGML_OP_ADD = ctypes.c_int(2)
GGML_OP_ADD1 = ctypes.c_int(3)
GGML_OP_ACC = ctypes.c_int(4)
GGML_OP_SUB = ctypes.c_int(5)
GGML_OP_MUL = ctypes.c_int(6)
GGML_OP_DIV = ctypes.c_int(7)
GGML_OP_SQR = ctypes.c_int(8)
GGML_OP_SQRT = ctypes.c_int(9)
GGML_OP_LOG = ctypes.c_int(10)
GGML_OP_SUM = ctypes.c_int(11)
GGML_OP_SUM_ROWS = ctypes.c_int(12)
GGML_OP_MEAN = ctypes.c_int(13)
GGML_OP_REPEAT = ctypes.c_int(14)
GGML_OP_ABS = ctypes.c_int(15)
GGML_OP_SGN = ctypes.c_int(16)
GGML_OP_NEG = ctypes.c_int(17)
GGML_OP_STEP = ctypes.c_int(18)
GGML_OP_RELU = ctypes.c_int(19)
GGML_OP_GELU = ctypes.c_int(20)
GGML_OP_SILU = ctypes.c_int(21)
GGML_OP_SILU_BACK = ctypes.c_int(22)
GGML_OP_NORM = ctypes.c_int(23)
GGML_OP_RMS_NORM = ctypes.c_int(24)
GGML_OP_RMS_NORM_BACK = ctypes.c_int(25)

GGML_OP_MUL_MAT = ctypes.c_int(26)

GGML_OP_SCALE = ctypes.c_int(27)
GGML_OP_SET = ctypes.c_int(28)
GGML_OP_CPY = ctypes.c_int(29)
GGML_OP_CONT = ctypes.c_int(30)
GGML_OP_RESHAPE = ctypes.c_int(31)
GGML_OP_VIEW = ctypes.c_int(32)
GGML_OP_PERMUTE = ctypes.c_int(33)
GGML_OP_TRANSPOSE = ctypes.c_int(34)
GGML_OP_GET_ROWS = ctypes.c_int(35)
GGML_OP_GET_ROWS_BACK = ctypes.c_int(36)
GGML_OP_DIAG = ctypes.c_int(37)
GGML_OP_DIAG_MASK_INF = ctypes.c_int(38)
GGML_OP_DIAG_MASK_ZERO = ctypes.c_int(39)
GGML_OP_SOFT_MAX = ctypes.c_int(40)
GGML_OP_ROPE = ctypes.c_int(41)
GGML_OP_ROPE_BACK = ctypes.c_int(42)
GGML_OP_ALIBI = ctypes.c_int(43)
GGML_OP_CONV_1D_1S = ctypes.c_int(44)
GGML_OP_CONV_1D_2S = ctypes.c_int(45)

GGML_OP_FLASH_ATTN = ctypes.c_int(46)
GGML_OP_FLASH_FF = ctypes.c_int(47)

GGML_OP_MAP_UNARY = ctypes.c_int(48)
GGML_OP_MAP_BINARY = ctypes.c_int(49)

GGML_OP_COUNT = ctypes.c_int(50)


# struct ggml_object {
#     size_t offs;
#     size_t size;

#     struct ggml_object * next;

#     char padding[8];
# };
class ggml_object(ctypes.Structure):
    pass

ggml_object._fields_ = [
        ("offs", ctypes.c_size_t),
        ("size", ctypes.c_size_t),
        ("next", ctypes.POINTER(ggml_object)),
        ("padding", ctypes.c_char * 8)
    ]

GGML_OBJECT_SIZE = ctypes.sizeof(ggml_object)


# struct ggml_tensor {
#     enum ggml_type    type;
#     enum ggml_backend backend;

#     int     n_dims;
#     int64_t ne[GGML_MAX_DIMS]; // number of elements
#     size_t  nb[GGML_MAX_DIMS]; // stride in bytes:
#                                // nb[0] = sizeof(type)
#                                // nb[1] = nb[0]   * ne[0] + padding
#                                // nb[i] = nb[i-1] * ne[i-1]

#     // compute data
#     enum ggml_op op;

#     bool is_param;

#     struct ggml_tensor * grad;
#     struct ggml_tensor * src0;
#     struct ggml_tensor * src1;
#     struct ggml_tensor * opt[GGML_MAX_OPT];

#     // thread scheduling
#     int n_tasks;

#     // performance
#     int     perf_runs;
#     int64_t perf_cycles;
#     int64_t perf_time_us;

#     void * data;

#     char name[32];

#     char padding[16];
# };
class ggml_tensor(ctypes.Structure):
    pass

ggml_tensor._fields_ = [
        ("type", ctypes.c_int),
        ("backend", ctypes.c_int),
        ("n_dims", ctypes.c_int),
        ("ne", ctypes.c_int64 * int(GGML_MAX_DIMS.value)),
        ("nb", ctypes.c_size_t * int(GGML_MAX_DIMS.value)),
        ("op", ctypes.c_int),
        ("is_param", ctypes.c_bool),
        ("grad", ctypes.POINTER(ggml_tensor)),
        ("src0", ctypes.POINTER(ggml_tensor)),
        ("src1", ctypes.POINTER(ggml_tensor)),
        ("opt", ctypes.POINTER(ggml_tensor) * int(GGML_MAX_OPT.value)),
        ("n_tasks", ctypes.c_int),
        ("perf_runs", ctypes.c_int),
        ("perf_cycles", ctypes.c_int64),
        ("perf_time_us", ctypes.c_int64),
        ("data", ctypes.POINTER(ctypes.c_void_p)),
        ("name", ctypes.c_char * 32),
        ("padding", ctypes.c_char * 16)
    ]

# struct ggml_cgraph {
#     int n_nodes;
#     int n_leafs;
#     int n_threads;

#     size_t work_size;
#     struct ggml_tensor * work;

#     struct ggml_tensor * nodes[GGML_MAX_NODES];
#     struct ggml_tensor * grads[GGML_MAX_NODES];
#     struct ggml_tensor * leafs[GGML_MAX_NODES];

#     // performance
#     int     perf_runs;
#     int64_t perf_cycles;
#     int64_t perf_time_us;
# };
class ggml_cgraph(ctypes.Structure):
    _fields_ = [
        ("n_nodes", ctypes.c_int),
        ("n_leafs", ctypes.c_int),
        ("n_threads", ctypes.c_int),
        ("work_size", ctypes.c_size_t),
        ("work", ctypes.POINTER(ggml_tensor)),
        ("nodes", ctypes.POINTER(ggml_tensor) * int(GGML_MAX_NODES.value)),
        ("grads", ctypes.POINTER(ggml_tensor) * int(GGML_MAX_NODES.value)),
        ("leafs", ctypes.POINTER(ggml_tensor) * int(GGML_MAX_NODES.value)),
        ("perf_runs", ctypes.c_int),
        ("perf_cycles", ctypes.c_int64),
        ("perf_time_us", ctypes.c_int64)
    ]

# struct ggml_scratch {
#     size_t offs;
#     size_t size;
#     void * data;
# };
class ggml_scratch(ctypes.Structure):
    _fields_ = [
        ("offs", ctypes.c_size_t),
        ("size", ctypes.c_size_t),
        ("data", ctypes.POINTER(ctypes.c_void_p))
    ]


# struct ggml_init_params {
#     // memory pool
#     size_t mem_size;   // bytes
#     void * mem_buffer; // if NULL, memory will be allocated internally
#     bool   no_alloc;   // don't allocate memory for the tensor data
# };
class ggml_init_params(ctypes.Structure):
    _fields_ = [
        ("mem_size", ctypes.c_size_t),
        ("mem_buffer", ctypes.POINTER(ctypes.c_void_p)),
        ("no_alloc", ctypes.c_bool)
    ]


# GGML_API void    ggml_time_init(void); // call this once at the beginning of the program
def ggml_time_init():
    return lib.ggml_time_init()

lib.ggml_time_init.argtypes = []
lib.ggml_time_init.restype = None

# GGML_API int64_t ggml_time_ms(void);
def ggml_time_ms():
    return lib.ggml_time_ms()

lib.ggml_time_ms.argtypes = []
lib.ggml_time_ms.restype = ctypes.c_int64

# GGML_API int64_t ggml_time_us(void);
def ggml_time_us():
    return lib.ggml_time_us()

lib.ggml_time_us.argtypes = []
lib.ggml_time_us.restype = ctypes.c_int64

# GGML_API int64_t ggml_cycles(void);
def ggml_cycles():
    return lib.ggml_cycles()

lib.ggml_cycles.argtypes = []
lib.ggml_cycles.restype = ctypes.c_int64

# GGML_API int64_t ggml_cycles_per_ms(void);
def ggml_cycles_per_ms():
    return lib.ggml_cycles_per_ms()

lib.ggml_cycles_per_ms.argtypes = []
lib.ggml_cycles_per_ms.restype = ctypes.c_int64

# GGML_API void    ggml_print_object (const struct ggml_object * obj);
def ggml_print_object(obj):
    return lib.ggml_print_object(obj)

lib.ggml_print_object.argtypes = [ctypes.POINTER(ggml_object)]
lib.ggml_print_object.restype = None

# GGML_API void    ggml_print_objects(const struct ggml_context * ctx);
def ggml_print_objects(ctx):
    return lib.ggml_print_objects(ctx)

lib.ggml_print_objects.argtypes = [ggml_context_p]
lib.ggml_print_objects.restype = None

# GGML_API int64_t ggml_nelements(const struct ggml_tensor * tensor);
def ggml_nelements(tensor):
    return lib.ggml_nelements(tensor)

lib.ggml_nelements.argtypes = [ctypes.POINTER(ggml_tensor)]
lib.ggml_nelements.restype = ctypes.c_int64

# GGML_API size_t  ggml_nbytes   (const struct ggml_tensor * tensor);
def ggml_nbytes(tensor):
    return lib.ggml_nbytes(tensor)

lib.ggml_nbytes.argtypes = [ctypes.POINTER(ggml_tensor)]
lib.ggml_nbytes.restype = ctypes.c_size_t

# GGML_API int     ggml_blck_size (enum ggml_type type);
def ggml_blck_size(type):
    return lib.ggml_blck_size(type)

lib.ggml_blck_size.argtypes = [ctypes.c_int]
lib.ggml_blck_size.restype = ctypes.c_int

# GGML_API size_t  ggml_type_size (enum ggml_type type); // size in bytes for all elements in a block
def ggml_type_size(type):
    return lib.ggml_type_size(type)

lib.ggml_type_size.argtypes = [ctypes.c_int]
lib.ggml_type_size.restype = ctypes.c_size_t

# GGML_API float   ggml_type_sizef(enum ggml_type type); // ggml_type_size()/ggml_blck_size() as float
def ggml_type_sizef(type):
    return lib.ggml_type_sizef(type)

lib.ggml_type_sizef.argtypes = [ctypes.c_int]
lib.ggml_type_sizef.restype = ctypes.c_float

# GGML_API const char * ggml_type_name(enum ggml_type type);
def ggml_type_name(type):
    return lib.ggml_type_name(type)

lib.ggml_type_name.argtypes = [ctypes.c_int]
lib.ggml_type_name.restype = ctypes.c_char_p

# GGML_API size_t  ggml_element_size(const struct ggml_tensor * tensor);
def ggml_element_size(tensor):
    return lib.ggml_element_size(tensor)

lib.ggml_element_size.argtypes = [ctypes.POINTER(ggml_tensor)]
lib.ggml_element_size.restype = ctypes.c_size_t

# GGML_API bool    ggml_is_quantized(enum ggml_type type);
def ggml_is_quantized(type):
    return lib.ggml_is_quantized(type)

lib.ggml_is_quantized.argtypes = [ctypes.c_int]
lib.ggml_is_quantized.restype = ctypes.c_bool


# // TODO: temporary until model loading of ggml examples is refactored
# GGML_API enum ggml_type ggml_ftype_to_ggml_type(enum ggml_ftype ftype);
def ggml_ftype_to_ggml_type(ftype):
    return lib.ggml_ftype_to_ggml_type(ftype)

lib.ggml_ftype_to_ggml_type.argtypes = [ctypes.c_int]
lib.ggml_ftype_to_ggml_type.restype = ctypes.c_int

# // main

# GGML_API struct ggml_context * ggml_init(struct ggml_init_params params);
def ggml_init(params):
    return lib.ggml_init(params)

lib.ggml_init.argtypes = [ggml_init_params]
lib.ggml_init.restype = ggml_context_p

# GGML_API void    ggml_free(struct ggml_context * ctx);
def ggml_free(ctx):
    return lib.ggml_free(ctx)

lib.ggml_free.argtypes = [ggml_context_p]
lib.ggml_free.restype = None

# GGML_API size_t  ggml_used_mem(const struct ggml_context * ctx);
def ggml_used_mem(ctx):
    return lib.ggml_used_mem(ctx)

lib.ggml_used_mem.argtypes = [ggml_context_p]
lib.ggml_used_mem.restype = ctypes.c_size_t

# GGML_API size_t  ggml_set_scratch(struct ggml_context * ctx, struct ggml_scratch scratch);
def ggml_set_scratch(ctx, scratch):
    return lib.ggml_set_scratch(ctx, scratch)

lib.ggml_set_scratch.argtypes = [ggml_context_p, ggml_scratch]
lib.ggml_set_scratch.restype = ctypes.c_size_t

# GGML_API struct ggml_tensor * ggml_new_tensor(
#         struct ggml_context * ctx,
#         enum   ggml_type type,
#         int    n_dims,
#         const int64_t *ne);
def ggml_new_tensor(ctx, type, n_dims, ne):
    return lib.ggml_new_tensor(ctx, type, n_dims, ne)

lib.ggml_new_tensor.argtypes = [ggml_context_p, ctypes.c_int, ctypes.c_int, ctypes.POINTER(ctypes.c_int64)]
lib.ggml_new_tensor.restype = ctypes.POINTER(ggml_tensor)

# GGML_API struct ggml_tensor * ggml_new_tensor_1d(
#         struct ggml_context * ctx,
#         enum   ggml_type type,
#         int64_t ne0);
def ggml_new_tensor_1d(ctx, type, ne0):
    return lib.ggml_new_tensor_1d(ctx, type, ne0)

lib.ggml_new_tensor_1d.argtypes = [ggml_context_p, ctypes.c_int, ctypes.c_int64]
lib.ggml_new_tensor_1d.restype = ctypes.POINTER(ggml_tensor)

# GGML_API struct ggml_tensor * ggml_new_tensor_2d(
#         struct ggml_context * ctx,
#         enum   ggml_type type,
#         int64_t ne0,
#         int64_t ne1);
def ggml_new_tensor_2d(ctx, type, ne0, ne1):
    return lib.ggml_new_tensor_2d(ctx, type, ne0, ne1)

lib.ggml_new_tensor_2d.argtypes = [ggml_context_p, ctypes.c_int, ctypes.c_int64, ctypes.c_int64]
lib.ggml_new_tensor_2d.restype = ctypes.POINTER(ggml_tensor)

# GGML_API struct ggml_tensor * ggml_new_tensor_3d(
#         struct ggml_context * ctx,
#         enum   ggml_type type,
#         int64_t ne0,
#         int64_t ne1,
#         int64_t ne2);
def ggml_new_tensor_3d(ctx, type, ne0, ne1, ne2):
    return lib.ggml_new_tensor_3d(ctx, type, ne0, ne1, ne2)

lib.ggml_new_tensor_3d.argtypes = [ggml_context_p, ctypes.c_int, ctypes.c_int64, ctypes.c_int64, ctypes.c_int64]
lib.ggml_new_tensor_3d.restype = ctypes.POINTER(ggml_tensor)

# GGML_API struct ggml_tensor * ggml_new_tensor_4d(
#         struct ggml_context * ctx,
#         enum   ggml_type type,
#         int64_t ne0,
#         int64_t ne1,
#         int64_t ne2,
#         int64_t ne3);
def ggml_new_tensor_4d(ctx, type, ne0, ne1, ne2, ne3):
    return lib.ggml_new_tensor_4d(ctx, type, ne0, ne1, ne2, ne3)

lib.ggml_new_tensor_4d.argtypes = [ggml_context_p, ctypes.c_int, ctypes.c_int64, ctypes.c_int64, ctypes.c_int64, ctypes.c_int64]
lib.ggml_new_tensor_4d.restype = ctypes.POINTER(ggml_tensor)

# GGML_API struct ggml_tensor * ggml_new_i32(struct ggml_context * ctx, int32_t value);
def ggml_new_i32(ctx, value):
    return lib.ggml_new_i32(ctx, value)

lib.ggml_new_i32.argtypes = [ggml_context_p, ctypes.c_int32]
lib.ggml_new_i32.restype = ctypes.POINTER(ggml_tensor)

# GGML_API struct ggml_tensor * ggml_new_f32(struct ggml_context * ctx, float value);
def ggml_new_f32(ctx, value):
    return lib.ggml_new_f32(ctx, value)

lib.ggml_new_f32.argtypes = [ggml_context_p, ctypes.c_float]
lib.ggml_new_f32.restype = ctypes.POINTER(ggml_tensor)

# GGML_API struct ggml_tensor * ggml_dup_tensor (struct ggml_context * ctx, const struct ggml_tensor * src);
def ggml_dup_tensor(ctx, src):
    return lib.ggml_dup_tensor(ctx, src)

lib.ggml_dup_tensor.argtypes = [ggml_context_p, ctypes.POINTER(ggml_tensor)]
lib.ggml_dup_tensor.restype = ctypes.POINTER(ggml_tensor)

# GGML_API struct ggml_tensor * ggml_view_tensor(struct ggml_context * ctx, const struct ggml_tensor * src);
def ggml_view_tensor(ctx, src):
    return lib.ggml_view_tensor(ctx, src)

lib.ggml_view_tensor.argtypes = [ggml_context_p, ctypes.POINTER(ggml_tensor)]
lib.ggml_view_tensor.restype = ctypes.POINTER(ggml_tensor)

# GGML_API struct ggml_tensor * ggml_set_zero(struct ggml_tensor * tensor);
def ggml_set_zero(tensor):
    return lib.ggml_set_zero(tensor)

lib.ggml_set_zero.argtypes = [ctypes.POINTER(ggml_tensor)]
lib.ggml_set_zero.restype = ctypes.POINTER(ggml_tensor)

# GGML_API struct ggml_tensor * ggml_set_i32 (struct ggml_tensor * tensor, int32_t value);
def ggml_set_i32(tensor, value):
    return lib.ggml_set_i32(tensor, value)

lib.ggml_set_i32.argtypes = [ctypes.POINTER(ggml_tensor), ctypes.c_int32]
lib.ggml_set_i32.restype = ctypes.POINTER(ggml_tensor)

# GGML_API struct ggml_tensor * ggml_set_f32 (struct ggml_tensor * tensor, float value);
def ggml_set_f32(tensor, value):
    return lib.ggml_set_f32(tensor, value)

lib.ggml_set_f32.argtypes = [ctypes.POINTER(ggml_tensor), ctypes.c_float]
lib.ggml_set_f32.restype = ctypes.POINTER(ggml_tensor)

# GGML_API int32_t ggml_get_i32_1d(const struct ggml_tensor * tensor, int i);
def ggml_get_i32_1d(tensor, i):
    return lib.ggml_get_i32_1d(tensor, i)

lib.ggml_get_i32_1d.argtypes = [ctypes.POINTER(ggml_tensor), ctypes.c_int]
lib.ggml_get_i32_1d.restype = ctypes.c_int32

# GGML_API void    ggml_set_i32_1d(const struct ggml_tensor * tensor, int i, int32_t value);
def ggml_set_i32_1d(tensor, i, value):
    return lib.ggml_set_i32_1d(tensor, i, value)

lib.ggml_set_i32_1d.argtypes = [ctypes.POINTER(ggml_tensor), ctypes.c_int, ctypes.c_int32]
lib.ggml_set_i32_1d.restype = None

# GGML_API float   ggml_get_f32_1d(const struct ggml_tensor * tensor, int i);
def ggml_get_f32_1d(tensor, i):
    return lib.ggml_get_f32_1d(tensor, i)

lib.ggml_get_f32_1d.argtypes = [ctypes.POINTER(ggml_tensor), ctypes.c_int]
lib.ggml_get_f32_1d.restype = ctypes.c_float

# GGML_API void    ggml_set_f32_1d(const struct ggml_tensor * tensor, int i, float value);
def ggml_set_f32_1d(tensor, i, value):
    return lib.ggml_set_f32_1d(tensor, i, value)

lib.ggml_set_f32_1d.argtypes = [ctypes.POINTER(ggml_tensor), ctypes.c_int, ctypes.c_float]
lib.ggml_set_f32_1d.restype = None

# GGML_API void *  ggml_get_data    (const struct ggml_tensor * tensor);
def ggml_get_data(tensor):
    return lib.ggml_get_data(tensor)

lib.ggml_get_data.argtypes = [ctypes.POINTER(ggml_tensor)]
lib.ggml_get_data.restype = ctypes.POINTER(ctypes.c_void_p)

# GGML_API float * ggml_get_data_f32(const struct ggml_tensor * tensor);
def ggml_get_data_f32(tensor):
    return lib.ggml_get_data_f32(tensor)

lib.ggml_get_data_f32.argtypes = [ctypes.POINTER(ggml_tensor)]
lib.ggml_get_data_f32.restype = ctypes.POINTER(ctypes.c_float)

# GGML_API const char * ggml_get_name(const struct ggml_tensor * tensor);
def ggml_get_name(tensor):
    return lib.ggml_get_name(tensor)

lib.ggml_get_name.argtypes = [ctypes.POINTER(ggml_tensor)]
lib.ggml_get_name.restype = ctypes.c_char_p

# GGML_API void         ggml_set_name(struct ggml_tensor * tensor, const char * name);
def ggml_set_name(tensor, name):
    return lib.ggml_set_name(tensor, name)

lib.ggml_set_name.argtypes = [ctypes.POINTER(ggml_tensor), ctypes.c_char_p]
lib.ggml_set_name.restype = None



# GGML_API struct ggml_tensor * ggml_dup(
#         struct ggml_context * ctx,
#         struct ggml_tensor  * a);
def ggml_dup(ctx, a):
    return lib.ggml_dup(ctx, a)

lib.ggml_dup.argtypes = [ggml_context_p, ctypes.POINTER(ggml_tensor)]
lib.ggml_dup.restype = ctypes.POINTER(ggml_tensor)

# GGML_API struct ggml_tensor * ggml_add(
#         struct ggml_context * ctx,
#         struct ggml_tensor  * a,
#         struct ggml_tensor  * b);
def ggml_add(ctx, a, b):
    return lib.ggml_add(ctx, a, b)

lib.ggml_add.argtypes = [ggml_context_p, ctypes.POINTER(ggml_tensor), ctypes.POINTER(ggml_tensor)]
lib.ggml_add.restype = ctypes.POINTER(ggml_tensor)

# GGML_API struct ggml_tensor * ggml_add_inplace(
#         struct ggml_context * ctx,
#         struct ggml_tensor  * a,
#         struct ggml_tensor  * b);
def ggml_add_inplace(ctx, a, b):
    return lib.ggml_add_inplace(ctx, a, b)

lib.ggml_add_inplace.argtypes = [ggml_context_p, ctypes.POINTER(ggml_tensor), ctypes.POINTER(ggml_tensor)]
lib.ggml_add_inplace.restype = ctypes.POINTER(ggml_tensor)

# GGML_API struct ggml_tensor * ggml_add1(
#         struct ggml_context * ctx,
#         struct ggml_tensor  * a,
#         struct ggml_tensor  * b);
def ggml_add1(ctx, a, b):
    return lib.ggml_add1(ctx, a, b)

lib.ggml_add1.argtypes = [ggml_context_p, ctypes.POINTER(ggml_tensor), ctypes.POINTER(ggml_tensor)]
lib.ggml_add1.restype = ctypes.POINTER(ggml_tensor)

# GGML_API struct ggml_tensor * ggml_acc(
#         struct ggml_context * ctx,
#         struct ggml_tensor  * a,
#         struct ggml_tensor  * b,
#         size_t                nb1,
#         size_t                nb2,
#         size_t                nb3,
#         size_t                offset);
def ggml_acc(ctx, a, b, nb1, nb2, nb3, offset):
    return lib.ggml_acc(ctx, a, b, nb1, nb2, nb3, offset)

lib.ggml_acc.argtypes = [ggml_context_p, ctypes.POINTER(ggml_tensor), ctypes.POINTER(ggml_tensor), ctypes.c_size_t, ctypes.c_size_t, ctypes.c_size_t, ctypes.c_size_t]
lib.ggml_acc.restype = ctypes.POINTER(ggml_tensor)

# GGML_API struct ggml_tensor * ggml_acc_inplace(
#         struct ggml_context * ctx,
#         struct ggml_tensor  * a,
#         struct ggml_tensor  * b,
#         size_t                nb1,
#         size_t                nb2,
#         size_t                nb3,
#         size_t                offset);
def ggml_acc_inplace(ctx, a, b, nb1, nb2, nb3, offset):
    return lib.ggml_acc_inplace(ctx, a, b, nb1, nb2, nb3, offset)

lib.ggml_acc_inplace.argtypes = [ggml_context_p, ctypes.POINTER(ggml_tensor), ctypes.POINTER(ggml_tensor), ctypes.c_size_t, ctypes.c_size_t, ctypes.c_size_t, ctypes.c_size_t]
lib.ggml_acc_inplace.restype = ctypes.POINTER(ggml_tensor)

# GGML_API struct ggml_tensor * ggml_sub(
#         struct ggml_context * ctx,
#         struct ggml_tensor  * a,
#         struct ggml_tensor  * b);
def ggml_sub(ctx, a, b):
    return lib.ggml_sub(ctx, a, b)

lib.ggml_sub.argtypes = [ggml_context_p, ctypes.POINTER(ggml_tensor), ctypes.POINTER(ggml_tensor)]
lib.ggml_sub.restype = ctypes.POINTER(ggml_tensor)

# GGML_API struct ggml_tensor * ggml_mul(
#         struct ggml_context * ctx,
#         struct ggml_tensor  * a,
#         struct ggml_tensor  * b);
def ggml_mul(ctx, a, b):
    return lib.ggml_mul(ctx, a, b)

lib.ggml_mul.argtypes = [ggml_context_p, ctypes.POINTER(ggml_tensor), ctypes.POINTER(ggml_tensor)]
lib.ggml_mul.restype = ctypes.POINTER(ggml_tensor)

# GGML_API struct ggml_tensor * ggml_div(
#         struct ggml_context * ctx,
#         struct ggml_tensor  * a,
#         struct ggml_tensor  * b);
def ggml_div(ctx, a, b):
    return lib.ggml_div(ctx, a, b)

lib.ggml_div.argtypes = [ggml_context_p, ctypes.POINTER(ggml_tensor), ctypes.POINTER(ggml_tensor)]
lib.ggml_div.restype = ctypes.POINTER(ggml_tensor)

# GGML_API struct ggml_tensor * ggml_sqr(
#         struct ggml_context * ctx,
#         struct ggml_tensor  * a);
def ggml_sqr(ctx, a):
    return lib.ggml_sqr(ctx, a)

lib.ggml_sqr.argtypes = [ggml_context_p, ctypes.POINTER(ggml_tensor)]
lib.ggml_sqr.restype = ctypes.POINTER(ggml_tensor)

# GGML_API struct ggml_tensor * ggml_sqrt(
#         struct ggml_context * ctx,
#         struct ggml_tensor  * a);
def ggml_sqrt(ctx, a):
    return lib.ggml_sqrt(ctx, a)

lib.ggml_sqrt.argtypes = [ggml_context_p, ctypes.POINTER(ggml_tensor)]
lib.ggml_sqrt.restype = ctypes.POINTER(ggml_tensor)

# GGML_API struct ggml_tensor * ggml_log(
#         struct ggml_context * ctx,
#         struct ggml_tensor  * a);
def ggml_log(ctx, a):
    return lib.ggml_log(ctx, a)

lib.ggml_log.argtypes = [ggml_context_p, ctypes.POINTER(ggml_tensor)]
lib.ggml_log.restype = ctypes.POINTER(ggml_tensor)

# GGML_API struct ggml_tensor * ggml_log_inplace(
#         struct ggml_context * ctx,
#         struct ggml_tensor  * a);
def ggml_log_inplace(ctx, a):
    return lib.ggml_log_inplace(ctx, a)

lib.ggml_log_inplace.argtypes = [ggml_context_p, ctypes.POINTER(ggml_tensor)]
lib.ggml_log_inplace.restype = ctypes.POINTER(ggml_tensor)

# // return scalar
# GGML_API struct ggml_tensor * ggml_sum(
#         struct ggml_context * ctx,
#         struct ggml_tensor  * a);
def ggml_sum(ctx, a):
    return lib.ggml_sum(ctx, a)

lib.ggml_sum.argtypes = [ggml_context_p, ctypes.POINTER(ggml_tensor)]
lib.ggml_sum.restype = ctypes.POINTER(ggml_tensor)

# // sums along rows, with input shape [a,b,c,d] return shape [1,b,c,d]
# GGML_API struct ggml_tensor * ggml_sum_rows(
#         struct ggml_context * ctx,
#         struct ggml_tensor  * a);
def ggml_sum_rows(ctx, a):
    return lib.ggml_sum_rows(ctx, a)

lib.ggml_sum_rows.argtypes = [ggml_context_p, ctypes.POINTER(ggml_tensor)]
lib.ggml_sum_rows.restype = ctypes.POINTER(ggml_tensor)

# // mean along rows
# GGML_API struct ggml_tensor * ggml_mean(
#         struct ggml_context * ctx,
#         struct ggml_tensor  * a);
def ggml_mean(ctx, a):
    return lib.ggml_mean(ctx, a)

lib.ggml_mean.argtypes = [ggml_context_p, ctypes.POINTER(ggml_tensor)]
lib.ggml_mean.restype = ctypes.POINTER(ggml_tensor)

# // if a is the same shape as b, and a is not parameter, return a
# // otherwise, return a new tensor: repeat(a) to fit in b
# GGML_API struct ggml_tensor * ggml_repeat(
#         struct ggml_context * ctx,
#         struct ggml_tensor  * a,
#         struct ggml_tensor  * b);
def ggml_repeat(ctx, a, b):
    return lib.ggml_repeat(ctx, a, b)

lib.ggml_repeat.argtypes = [ggml_context_p, ctypes.POINTER(ggml_tensor), ctypes.POINTER(ggml_tensor)]
lib.ggml_repeat.restype = ctypes.POINTER(ggml_tensor)

# GGML_API struct ggml_tensor * ggml_abs(
#         struct ggml_context * ctx,
#         struct ggml_tensor  * a);
def ggml_abs(ctx, a):
    return lib.ggml_abs(ctx, a)

lib.ggml_abs.argtypes = [ggml_context_p, ctypes.POINTER(ggml_tensor)]
lib.ggml_abs.restype = ctypes.POINTER(ggml_tensor)

# GGML_API struct ggml_tensor * ggml_sgn(
#         struct ggml_context * ctx,
#         struct ggml_tensor  * a);
def ggml_sgn(ctx, a):
    return lib.ggml_sgn(ctx, a)

lib.ggml_sgn.argtypes = [ggml_context_p, ctypes.POINTER(ggml_tensor)]
lib.ggml_sgn.restype = ctypes.POINTER(ggml_tensor)

# GGML_API struct ggml_tensor * ggml_neg(
#         struct ggml_context * ctx,
#         struct ggml_tensor  * a);
def ggml_neg(ctx, a):
    return lib.ggml_neg(ctx, a)

lib.ggml_neg.argtypes = [ggml_context_p, ctypes.POINTER(ggml_tensor)]
lib.ggml_neg.restype = ctypes.POINTER(ggml_tensor)

# GGML_API struct ggml_tensor * ggml_step(
#         struct ggml_context * ctx,
#         struct ggml_tensor  * a);
def ggml_step(ctx, a):
    return lib.ggml_step(ctx, a)

lib.ggml_step.argtypes = [ggml_context_p, ctypes.POINTER(ggml_tensor)]
lib.ggml_step.restype = ctypes.POINTER(ggml_tensor)

# GGML_API struct ggml_tensor * ggml_relu(
#         struct ggml_context * ctx,
#         struct ggml_tensor  * a);
def ggml_relu(ctx, a):
    return lib.ggml_relu(ctx, a)

lib.ggml_relu.argtypes = [ggml_context_p, ctypes.POINTER(ggml_tensor)]
lib.ggml_relu.restype = ctypes.POINTER(ggml_tensor)

# // TODO: double-check this computation is correct
# GGML_API struct ggml_tensor * ggml_gelu(
#         struct ggml_context * ctx,
#         struct ggml_tensor  * a);
def ggml_gelu(ctx, a):
    return lib.ggml_gelu(ctx, a)

lib.ggml_gelu.argtypes = [ggml_context_p, ctypes.POINTER(ggml_tensor)]
lib.ggml_gelu.restype = ctypes.POINTER(ggml_tensor)

# GGML_API struct ggml_tensor * ggml_silu(
#         struct ggml_context * ctx,
#         struct ggml_tensor  * a);
def ggml_silu(ctx, a):
    return lib.ggml_silu(ctx, a)

lib.ggml_silu.argtypes = [ggml_context_p, ctypes.POINTER(ggml_tensor)]
lib.ggml_silu.restype = ctypes.POINTER(ggml_tensor)

# // a - x
# // b - dy
# GGML_API struct ggml_tensor * ggml_silu_back(
#         struct ggml_context * ctx,
#         struct ggml_tensor  * a,
#         struct ggml_tensor  * b);
def ggml_silu_back(ctx, a, b):
    return lib.ggml_silu_back(ctx, a, b)

lib.ggml_silu_back.argtypes = [ggml_context_p, ctypes.POINTER(ggml_tensor), ctypes.POINTER(ggml_tensor)]
lib.ggml_silu_back.restype = ctypes.POINTER(ggml_tensor)

# // normalize along rows
# // TODO: eps is hardcoded to 1e-5 for now
# GGML_API struct ggml_tensor * ggml_norm(
#         struct ggml_context * ctx,
#         struct ggml_tensor  * a);
def ggml_norm(ctx, a):
    return lib.ggml_norm(ctx, a)

lib.ggml_norm.argtypes = [ggml_context_p, ctypes.POINTER(ggml_tensor)]
lib.ggml_norm.restype = ctypes.POINTER(ggml_tensor)

# GGML_API struct ggml_tensor * ggml_rms_norm(
#         struct ggml_context * ctx,
#         struct ggml_tensor  * a);
def ggml_rms_norm(ctx, a):
    return lib.ggml_rms_norm(ctx, a)

lib.ggml_rms_norm.argtypes = [ggml_context_p, ctypes.POINTER(ggml_tensor)]
lib.ggml_rms_norm.restype = ctypes.POINTER(ggml_tensor)

# // a - x
# // b - dy
# GGML_API struct ggml_tensor * ggml_rms_norm_back(
#         struct ggml_context * ctx,
#         struct ggml_tensor  * a,
#         struct ggml_tensor  * b);
def ggml_rms_norm_back(ctx, a, b):
    return lib.ggml_rms_norm_back(ctx, a, b)

lib.ggml_rms_norm_back.argtypes = [ggml_context_p, ctypes.POINTER(ggml_tensor), ctypes.POINTER(ggml_tensor)]
lib.ggml_rms_norm_back.restype = ctypes.POINTER(ggml_tensor)

# // A: m rows, n columns
# // B: p rows, n columns (i.e. we transpose it internally)
# // result is m columns, p rows
# GGML_API struct ggml_tensor * ggml_mul_mat(
#         struct ggml_context * ctx,
#         struct ggml_tensor  * a,
#         struct ggml_tensor  * b);
def ggml_mul_mat(ctx, a, b):
    return lib.ggml_mul_mat(ctx, a, b)

lib.ggml_mul_mat.argtypes = [ggml_context_p, ctypes.POINTER(ggml_tensor), ctypes.POINTER(ggml_tensor)]
lib.ggml_mul_mat.restype = ctypes.POINTER(ggml_tensor)

# //
# // operations on tensors without backpropagation
# //

# GGML_API struct ggml_tensor * ggml_scale(
#         struct ggml_context * ctx,
#         struct ggml_tensor  * a,
#         struct ggml_tensor  * b);
def ggml_scale(ctx, a, b):
    return lib.ggml_scale(ctx, a, b)

lib.ggml_scale.argtypes = [ggml_context_p, ctypes.POINTER(ggml_tensor), ctypes.POINTER(ggml_tensor)]
lib.ggml_scale.restype = ctypes.POINTER(ggml_tensor)

# // in-place, returns view(a)
# GGML_API struct ggml_tensor * ggml_scale_inplace(
#         struct ggml_context * ctx,
#         struct ggml_tensor  * a,
#         struct ggml_tensor  * b);
def ggml_scale_inplace(ctx, a, b):
    return lib.ggml_scale_inplace(ctx, a, b)

lib.ggml_scale_inplace.argtypes = [ggml_context_p, ctypes.POINTER(ggml_tensor), ctypes.POINTER(ggml_tensor)]
lib.ggml_scale_inplace.restype = ctypes.POINTER(ggml_tensor)

# // b -> view(a,offset,nb1,nb2,3), return modified a
# GGML_API struct ggml_tensor * ggml_set(
#         struct ggml_context * ctx,
#         struct ggml_tensor  * a,
#         struct ggml_tensor  * b,
#         size_t                nb1,
#         size_t                nb2,
#         size_t                nb3,
#         size_t                offset);
def ggml_set(ctx, a, b, nb1, nb2, nb3, offset):
    return lib.ggml_set(ctx, a, b, nb1, nb2, nb3, offset)

lib.ggml_set.argtypes = [ggml_context_p, ctypes.POINTER(ggml_tensor), ctypes.POINTER(ggml_tensor), ctypes.c_size_t, ctypes.c_size_t, ctypes.c_size_t, ctypes.c_size_t]
lib.ggml_set.restype = ctypes.POINTER(ggml_tensor)

# // b -> view(a,offset,nb1,nb2,3), return view(a)
# GGML_API struct ggml_tensor * ggml_set_inplace(
#         struct ggml_context * ctx,
#         struct ggml_tensor  * a,
#         struct ggml_tensor  * b,
#         size_t                nb1,
#         size_t                nb2,
#         size_t                nb3,
#         size_t                offset);
def ggml_set_inplace(ctx, a, b, nb1, nb2, nb3, offset):
    return lib.ggml_set_inplace(ctx, a, b, nb1, nb2, nb3, offset)

lib.ggml_set_inplace.argtypes = [ggml_context_p, ctypes.POINTER(ggml_tensor), ctypes.POINTER(ggml_tensor), ctypes.c_size_t, ctypes.c_size_t, ctypes.c_size_t, ctypes.c_size_t]
lib.ggml_set_inplace.restype = ctypes.POINTER(ggml_tensor)

# GGML_API struct ggml_tensor * ggml_set_1d(
#         struct ggml_context * ctx,
#         struct ggml_tensor  * a,
#         struct ggml_tensor  * b,
#         size_t                offset);
def ggml_set_1d(ctx, a, b, offset):
    return lib.ggml_set_1d(ctx, a, b, offset)

lib.ggml_set_1d.argtypes = [ggml_context_p, ctypes.POINTER(ggml_tensor), ctypes.POINTER(ggml_tensor), ctypes.c_size_t]
lib.ggml_set_1d.restype = ctypes.POINTER(ggml_tensor)

# GGML_API struct ggml_tensor * ggml_set_1d_inplace(
#         struct ggml_context * ctx,
#         struct ggml_tensor  * a,
#         struct ggml_tensor  * b,
#         size_t                offset);
def ggml_set_1d_inplace(ctx, a, b, offset):
    return lib.ggml_set_1d_inplace(ctx, a, b, offset)

lib.ggml_set_1d_inplace.argtypes = [ggml_context_p, ctypes.POINTER(ggml_tensor), ctypes.POINTER(ggml_tensor), ctypes.c_size_t]
lib.ggml_set_1d_inplace.restype = ctypes.POINTER(ggml_tensor)

# // b -> view(a,offset,nb1,nb2,3), return modified a
# GGML_API struct ggml_tensor * ggml_set_2d(
#         struct ggml_context * ctx,
#         struct ggml_tensor  * a,
#         struct ggml_tensor  * b,
#         size_t                nb1,
#         size_t                offset);
def ggml_set_2d(ctx, a, b, nb1, offset):
    return lib.ggml_set_2d(ctx, a, b, nb1, offset)

lib.ggml_set_2d.argtypes = [ggml_context_p, ctypes.POINTER(ggml_tensor), ctypes.POINTER(ggml_tensor), ctypes.c_size_t, ctypes.c_size_t]
lib.ggml_set_2d.restype = ctypes.POINTER(ggml_tensor)

# // b -> view(a,offset,nb1,nb2,3), return view(a)
# GGML_API struct ggml_tensor * ggml_set_2d_inplace(
#         struct ggml_context * ctx,
#         struct ggml_tensor  * a,
#         struct ggml_tensor  * b,
#         size_t                nb1,
#         size_t                offset);
def ggml_set_2d_inplace(ctx, a, b, nb1, offset):
    return lib.ggml_set_2d_inplace(ctx, a, b, nb1, offset)

lib.ggml_set_2d_inplace.argtypes = [ggml_context_p, ctypes.POINTER(ggml_tensor), ctypes.POINTER(ggml_tensor), ctypes.c_size_t, ctypes.c_size_t]
lib.ggml_set_2d_inplace.restype = ctypes.POINTER(ggml_tensor)

# // a -> b, return view(b)
# GGML_API struct ggml_tensor * ggml_cpy(
#         struct ggml_context * ctx,
#         struct ggml_tensor  * a,
#         struct ggml_tensor  * b);
def ggml_cpy(ctx, a, b):
    return lib.ggml_cpy(ctx, a, b)

lib.ggml_cpy.argtypes = [ggml_context_p, ctypes.POINTER(ggml_tensor), ctypes.POINTER(ggml_tensor)]
lib.ggml_cpy.restype = ctypes.POINTER(ggml_tensor)

# // make contiguous
# GGML_API struct ggml_tensor * ggml_cont(
#         struct ggml_context * ctx,
#         struct ggml_tensor  * a);
def ggml_cont(ctx, a):
    return lib.ggml_cont(ctx, a)

lib.ggml_cont.argtypes = [ggml_context_p, ctypes.POINTER(ggml_tensor)]
lib.ggml_cont.restype = ctypes.POINTER(ggml_tensor)

# // return view(a), b specifies the new shape
# // TODO: when we start computing gradient, make a copy instead of view
# GGML_API struct ggml_tensor * ggml_reshape(
#         struct ggml_context * ctx,
#         struct ggml_tensor  * a,
#         struct ggml_tensor  * b);
def ggml_reshape(ctx, a, b):
    return lib.ggml_reshape(ctx, a, b)

lib.ggml_reshape.argtypes = [ggml_context_p, ctypes.POINTER(ggml_tensor), ctypes.POINTER(ggml_tensor)]
lib.ggml_reshape.restype = ctypes.POINTER(ggml_tensor)

# // return view(a)
# // TODO: when we start computing gradient, make a copy instead of view
# GGML_API struct ggml_tensor * ggml_reshape_1d(
#         struct ggml_context * ctx,
#         struct ggml_tensor  * a,
#         int64_t               ne0);
def ggml_reshape_1d(ctx, a, ne0):
    return lib.ggml_reshape_1d(ctx, a, ne0)

lib.ggml_reshape_1d.argtypes = [ggml_context_p, ctypes.POINTER(ggml_tensor), ctypes.c_int64]
lib.ggml_reshape_1d.restype = ctypes.POINTER(ggml_tensor)

# GGML_API struct ggml_tensor * ggml_reshape_2d(
#         struct ggml_context * ctx,
#         struct ggml_tensor  * a,
#         int64_t               ne0,
#         int64_t               ne1);
def ggml_reshape_2d(ctx, a, ne0, ne1):
    return lib.ggml_reshape_2d(ctx, a, ne0, ne1)

lib.ggml_reshape_2d.argtypes = [ggml_context_p, ctypes.POINTER(ggml_tensor), ctypes.c_int64, ctypes.c_int64]
lib.ggml_reshape_2d.restype = ctypes.POINTER(ggml_tensor)

# // return view(a)
# // TODO: when we start computing gradient, make a copy instead of view
# GGML_API struct ggml_tensor * ggml_reshape_3d(
#         struct ggml_context * ctx,
#         struct ggml_tensor  * a,
#         int64_t               ne0,
#         int64_t               ne1,
#         int64_t               ne2);
def ggml_reshape_3d(ctx, a, ne0, ne1, ne2):
    return lib.ggml_reshape_3d(ctx, a, ne0, ne1, ne2)

lib.ggml_reshape_3d.argtypes = [ggml_context_p, ctypes.POINTER(ggml_tensor), ctypes.c_int64, ctypes.c_int64, ctypes.c_int64]
lib.ggml_reshape_3d.restype = ctypes.POINTER(ggml_tensor)

# GGML_API struct ggml_tensor * ggml_reshape_4d(
#         struct ggml_context * ctx,
#         struct ggml_tensor  * a,
#         int64_t               ne0,
#         int64_t               ne1,
#         int64_t               ne2,
#         int64_t               ne3);
def ggml_reshape_4d(ctx, a, ne0, ne1, ne2, ne3):
    return lib.ggml_reshape_4d(ctx, a, ne0, ne1, ne2, ne3)

lib.ggml_reshape_4d.argtypes = [ggml_context_p, ctypes.POINTER(ggml_tensor), ctypes.c_int64, ctypes.c_int64, ctypes.c_int64, ctypes.c_int64]
lib.ggml_reshape_4d.restype = ctypes.POINTER(ggml_tensor)

# // offset in bytes
# GGML_API struct ggml_tensor * ggml_view_1d(
#         struct ggml_context * ctx,
#         struct ggml_tensor  * a,
#         int64_t               ne0,
#         size_t                offset);
def ggml_view_1d(ctx, a, ne0, offset):
    return lib.ggml_view_1d(ctx, a, ne0, offset)

lib.ggml_view_1d.argtypes = [ggml_context_p, ctypes.POINTER(ggml_tensor), ctypes.c_int64, ctypes.c_size_t]
lib.ggml_view_1d.restype = ctypes.POINTER(ggml_tensor)

# GGML_API struct ggml_tensor * ggml_view_2d(
#         struct ggml_context * ctx,
#         struct ggml_tensor  * a,
#         int64_t               ne0,
#         int64_t               ne1,
#         size_t                nb1, // row stride in bytes
#         size_t                offset);
def ggml_view_2d(ctx, a, ne0, ne1, nb1, offset):
    return lib.ggml_view_2d(ctx, a, ne0, ne1, nb1, offset)

lib.ggml_view_2d.argtypes = [ggml_context_p, ctypes.POINTER(ggml_tensor), ctypes.c_int64, ctypes.c_int64, ctypes.c_size_t, ctypes.c_size_t]
lib.ggml_view_2d.restype = ctypes.POINTER(ggml_tensor)

# GGML_API struct ggml_tensor * ggml_view_3d(
#         struct ggml_context * ctx,
#         struct ggml_tensor  * a,
#         int64_t               ne0,
#         int64_t               ne1,
#         int64_t               ne2,
#         size_t                nb1, // row   stride in bytes
#         size_t                nb2, // slice stride in bytes
#         size_t                offset);
def ggml_view_3d(ctx, a, ne0, ne1, ne2, nb1, nb2, offset):
    return lib.ggml_view_3d(ctx, a, ne0, ne1, ne2, nb1, nb2, offset)

lib.ggml_view_3d.argtypes = [ggml_context_p, ctypes.POINTER(ggml_tensor), ctypes.c_int64, ctypes.c_int64, ctypes.c_int64, ctypes.c_size_t, ctypes.c_size_t, ctypes.c_size_t]
lib.ggml_view_3d.restype = ctypes.POINTER(ggml_tensor)

# GGML_API struct ggml_tensor * ggml_view_4d(
#         struct ggml_context * ctx,
#         struct ggml_tensor  * a,
#         int64_t               ne0,
#         int64_t               ne1,
#         int64_t               ne2,
#         int64_t               ne3,
#         size_t                nb1, // row   stride in bytes
#         size_t                nb2, // slice stride in bytes
#         size_t                nb3,
#         size_t                offset);
def ggml_view_4d(ctx, a, ne0, ne1, ne2, ne3, nb1, nb2, nb3, offset):
    return lib.ggml_view_4d(ctx, a, ne0, ne1, ne2, ne3, nb1, nb2, nb3, offset)

lib.ggml_view_4d.argtypes = [ggml_context_p, ctypes.POINTER(ggml_tensor), ctypes.c_int64, ctypes.c_int64, ctypes.c_int64, ctypes.c_int64, ctypes.c_size_t, ctypes.c_size_t, ctypes.c_size_t, ctypes.c_size_t]
lib.ggml_view_4d.restype = ctypes.POINTER(ggml_tensor)

# GGML_API struct ggml_tensor * ggml_permute(
#         struct ggml_context * ctx,
#         struct ggml_tensor  * a,
#         int                   axis0,
#         int                   axis1,
#         int                   axis2,
#         int                   axis3);
def ggml_permute(ctx, a, axis0, axis1, axis2, axis3):
    return lib.ggml_permute(ctx, a, axis0, axis1, axis2, axis3)

lib.ggml_permute.argtypes = [ggml_context_p, ctypes.POINTER(ggml_tensor), ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int]
lib.ggml_permute.restype = ctypes.POINTER(ggml_tensor)

# // alias for ggml_permute(ctx, a, 1, 0, 2, 3)
# GGML_API struct ggml_tensor * ggml_transpose(
#         struct ggml_context * ctx,
#         struct ggml_tensor  * a);
def ggml_transpose(ctx, a):
    return lib.ggml_transpose(ctx, a)

lib.ggml_transpose.argtypes = [ggml_context_p, ctypes.POINTER(ggml_tensor)]
lib.ggml_transpose.restype = ctypes.POINTER(ggml_tensor)

# GGML_API struct ggml_tensor * ggml_get_rows(
#         struct ggml_context * ctx,
#         struct ggml_tensor  * a,
#         struct ggml_tensor  * b);
def ggml_get_rows(ctx, a, b):
    return lib.ggml_get_rows(ctx, a, b)

lib.ggml_get_rows.argtypes = [ggml_context_p, ctypes.POINTER(ggml_tensor), ctypes.POINTER(ggml_tensor)]
lib.ggml_get_rows.restype = ctypes.POINTER(ggml_tensor)

# GGML_API struct ggml_tensor * ggml_get_rows_back(
#         struct ggml_context * ctx,
#         struct ggml_tensor  * a,
#         struct ggml_tensor  * b,
#         struct ggml_tensor  * c);
def ggml_get_rows_back(ctx, a, b, c):
    return lib.ggml_get_rows_back(ctx, a, b, c)

lib.ggml_get_rows_back.argtypes = [ggml_context_p, ctypes.POINTER(ggml_tensor), ctypes.POINTER(ggml_tensor), ctypes.POINTER(ggml_tensor)]
lib.ggml_get_rows_back.restype = ctypes.POINTER(ggml_tensor)

# GGML_API struct ggml_tensor * ggml_diag(
#     struct ggml_context     * ctx,
#     struct ggml_tensor      * a);
def ggml_diag(ctx, a):
    return lib.ggml_diag(ctx, a)

lib.ggml_diag.argtypes = [ggml_context_p, ctypes.POINTER(ggml_tensor)]
lib.ggml_diag.restype = ctypes.POINTER(ggml_tensor)

# // set elements above the diagonal to -INF
# GGML_API struct ggml_tensor * ggml_diag_mask_inf(
#         struct ggml_context * ctx,
#         struct ggml_tensor  * a,
#         int                   n_past);
def ggml_diag_mask_inf(ctx, a, n_past):
    return lib.ggml_diag_mask_inf(ctx, a, n_past)

lib.ggml_diag_mask_inf.argtypes = [ggml_context_p, ctypes.POINTER(ggml_tensor), ctypes.c_int]
lib.ggml_diag_mask_inf.restype = ctypes.POINTER(ggml_tensor)

# // in-place, returns view(a)
# GGML_API struct ggml_tensor * ggml_diag_mask_inf_inplace(
#         struct ggml_context * ctx,
#         struct ggml_tensor  * a,
#         int                   n_past);
def ggml_diag_mask_inf_inplace(ctx, a, n_past):
    return lib.ggml_diag_mask_inf_inplace(ctx, a, n_past)

lib.ggml_diag_mask_inf_inplace.argtypes = [ggml_context_p, ctypes.POINTER(ggml_tensor), ctypes.c_int]
lib.ggml_diag_mask_inf_inplace.restype = ctypes.POINTER(ggml_tensor)

# // set elements above the diagonal to 0
# GGML_API struct ggml_tensor * ggml_diag_mask_zero(
#         struct ggml_context * ctx,
#         struct ggml_tensor  * a,
#         int                   n_past);
def ggml_diag_mask_zero(ctx, a, n_past):
    return lib.ggml_diag_mask_zero(ctx, a, n_past)

lib.ggml_diag_mask_zero.argtypes = [ggml_context_p, ctypes.POINTER(ggml_tensor), ctypes.c_int]
lib.ggml_diag_mask_zero.restype = ctypes.POINTER(ggml_tensor)

# // in-place, returns view(a)
# GGML_API struct ggml_tensor * ggml_diag_mask_zero_inplace(
#         struct ggml_context * ctx,
#         struct ggml_tensor  * a,
#         int                   n_past);
def ggml_diag_mask_zero_inplace(ctx, a, n_past):
    return lib.ggml_diag_mask_zero_inplace(ctx, a, n_past)

lib.ggml_diag_mask_zero_inplace.argtypes = [ggml_context_p, ctypes.POINTER(ggml_tensor), ctypes.c_int]
lib.ggml_diag_mask_zero_inplace.restype = ctypes.POINTER(ggml_tensor)

# GGML_API struct ggml_tensor * ggml_soft_max(
#         struct ggml_context * ctx,
#         struct ggml_tensor  * a);
def ggml_soft_max(ctx, a):
    return lib.ggml_soft_max(ctx, a)

lib.ggml_soft_max.argtypes = [ggml_context_p, ctypes.POINTER(ggml_tensor)]
lib.ggml_soft_max.restype = ctypes.POINTER(ggml_tensor)

# // in-place, returns view(a)
# GGML_API struct ggml_tensor * ggml_soft_max_inplace(
#         struct ggml_context * ctx,
#         struct ggml_tensor  * a);
def ggml_soft_max_inplace(ctx, a):
    return lib.ggml_soft_max_inplace(ctx, a)

lib.ggml_soft_max_inplace.argtypes = [ggml_context_p, ctypes.POINTER(ggml_tensor)]
lib.ggml_soft_max_inplace.restype = ctypes.POINTER(ggml_tensor)

# // rotary position embedding
# // if mode & 1 == 1, skip n_past elements
# // if mode & 2 == 1, GPT-NeoX style
# // TODO: avoid creating a new tensor every time
# GGML_API struct ggml_tensor * ggml_rope(
#         struct ggml_context * ctx,
#         struct ggml_tensor  * a,
#         int                   n_past,
#         int                   n_dims,
#         int                   mode);
def ggml_rope(ctx, a, n_past, n_dims, mode):
    return lib.ggml_rope(ctx, a, n_past, n_dims, mode)

lib.ggml_rope.argtypes = [ggml_context_p, ctypes.POINTER(ggml_tensor), ctypes.c_int, ctypes.c_int, ctypes.c_int]
lib.ggml_rope.restype = ctypes.POINTER(ggml_tensor)

# // in-place, returns view(a)
# GGML_API struct ggml_tensor * ggml_rope_inplace(
#         struct ggml_context * ctx,
#         struct ggml_tensor  * a,
#         int                   n_past,
#         int                   n_dims,
#         int                   mode);
def ggml_rope_inplace(ctx, a, n_past, n_dims, mode):
    return lib.ggml_rope_inplace(ctx, a, n_past, n_dims, mode)

lib.ggml_rope_inplace.argtypes = [ggml_context_p, ctypes.POINTER(ggml_tensor), ctypes.c_int, ctypes.c_int, ctypes.c_int]
lib.ggml_rope_inplace.restype = ctypes.POINTER(ggml_tensor)

# // rotary position embedding backward, i.e compute dx from dy
# // a - dy
# GGML_API struct ggml_tensor * ggml_rope_back(
#         struct ggml_context * ctx,
#         struct ggml_tensor  * a,
#         int                   n_past,
#         int                   n_dims,
#         int                   mode);
def ggml_rope_back(ctx, a, n_past, n_dims, mode):
    return lib.ggml_rope_back(ctx, a, n_past, n_dims, mode)

lib.ggml_rope_back.argtypes = [ggml_context_p, ctypes.POINTER(ggml_tensor), ctypes.c_int, ctypes.c_int, ctypes.c_int]
lib.ggml_rope_back.restype = ctypes.POINTER(ggml_tensor)

# // alibi position embedding
# // in-place, returns view(a)
# struct ggml_tensor * ggml_alibi(
#         struct ggml_context * ctx,
#         struct ggml_tensor  * a,
#         int                   n_past,
#         int                   n_head);
def ggml_alibi(ctx, a, n_past, n_head):
    return lib.ggml_alibi(ctx, a, n_past, n_head)

lib.ggml_alibi.argtypes = [ggml_context_p, ctypes.POINTER(ggml_tensor), ctypes.c_int, ctypes.c_int]
lib.ggml_alibi.restype = ctypes.POINTER(ggml_tensor)

# // padding = 1
# // TODO: we don't support extra parameters for now
# //       that's why we are hard-coding the stride, padding, and dilation
# //       not great ..
# GGML_API struct ggml_tensor * ggml_conv_1d_1s(
#         struct ggml_context * ctx,
#         struct ggml_tensor  * a,
#         struct ggml_tensor  * b);
def ggml_conv_1d_1s(ctx, a, b):
    return lib.ggml_conv_1d_1s(ctx, a, b)

lib.ggml_conv_1d_1s.argtypes = [ggml_context_p, ctypes.POINTER(ggml_tensor), ctypes.POINTER(ggml_tensor)]
lib.ggml_conv_1d_1s.restype = ctypes.POINTER(ggml_tensor)

# GGML_API struct ggml_tensor * ggml_conv_1d_2s(
#         struct ggml_context * ctx,
#         struct ggml_tensor  * a,
#         struct ggml_tensor  * b);
def ggml_conv_1d_2s(ctx, a, b):
    return lib.ggml_conv_1d_2s(ctx, a, b)

lib.ggml_conv_1d_2s.argtypes = [ggml_context_p, ctypes.POINTER(ggml_tensor), ctypes.POINTER(ggml_tensor)]
lib.ggml_conv_1d_2s.restype = ctypes.POINTER(ggml_tensor)

# GGML_API struct ggml_tensor * ggml_flash_attn(
#         struct ggml_context * ctx,
#         struct ggml_tensor  * q,
#         struct ggml_tensor  * k,
#         struct ggml_tensor  * v,
#         bool                  masked);
def ggml_flash_attn(ctx, q, k, v, masked):
    return lib.ggml_flash_attn(ctx, q, k, v, masked)

lib.ggml_flash_attn.argtypes = [ggml_context_p, ctypes.POINTER(ggml_tensor), ctypes.POINTER(ggml_tensor), ctypes.POINTER(ggml_tensor), ctypes.c_bool]
lib.ggml_flash_attn.restype = ctypes.POINTER(ggml_tensor)

# GGML_API struct ggml_tensor * ggml_flash_ff(
#         struct ggml_context * ctx,
#         struct ggml_tensor  * a,
#         struct ggml_tensor  * b0,
#         struct ggml_tensor  * b1,
#         struct ggml_tensor  * c0,
#         struct ggml_tensor  * c1);
def ggml_flash_ff(ctx, a, b0, b1, c0, c1):
    return lib.ggml_flash_ff(ctx, a, b0, b1, c0, c1)

lib.ggml_flash_ff.argtypes = [ggml_context_p, ctypes.POINTER(ggml_tensor), ctypes.POINTER(ggml_tensor), ctypes.POINTER(ggml_tensor), ctypes.POINTER(ggml_tensor), ctypes.POINTER(ggml_tensor)]
lib.ggml_flash_ff.restype = ctypes.POINTER(ggml_tensor)

# // Mapping operations
# typedef void (*ggml_unary_op_f32_t)(const int, float *, const float *);
ggml_unary_op_f32_t = ctypes.CFUNCTYPE(None, ctypes.c_int, ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float))

# typedef void (*ggml_binary_op_f32_t)(const int, float *, const float *, const float *);
ggml_binary_op_f32_t = ctypes.CFUNCTYPE(None, ctypes.c_int, ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float))

# GGML_API struct ggml_tensor * ggml_map_unary_f32(
#         struct ggml_context        * ctx,
#         struct ggml_tensor         * a,
#                ggml_unary_op_f32_t   fun);
def ggml_map_unary_f32(ctx, a, fun):
    return lib.ggml_map_unary_f32(ctx, a, fun)

lib.ggml_map_unary_f32.argtypes = [ggml_context_p, ctypes.POINTER(ggml_tensor), ggml_unary_op_f32_t]
lib.ggml_map_unary_f32.restype = ctypes.POINTER(ggml_tensor)

# GGML_API struct ggml_tensor * ggml_map_binary_f32(
#         struct ggml_context         * ctx,
#         struct ggml_tensor          * a,
#         struct ggml_tensor          * b,
#                ggml_binary_op_f32_t   fun);
def ggml_map_binary_f32(ctx, a, b, fun):
    return lib.ggml_map_binary_f32(ctx, a, b, fun)

lib.ggml_map_binary_f32.argtypes = [ggml_context_p, ctypes.POINTER(ggml_tensor), ctypes.POINTER(ggml_tensor), ggml_binary_op_f32_t]
lib.ggml_map_binary_f32.restype = ctypes.POINTER(ggml_tensor)


# GGML_API void ggml_set_param(
#         struct ggml_context * ctx,
#         struct ggml_tensor * tensor);
def ggml_set_param(ctx, tensor):
    return lib.ggml_set_param(ctx, tensor)

lib.ggml_set_param.argtypes = [ggml_context_p, ctypes.POINTER(ggml_tensor)]
lib.ggml_set_param.restype = None

# GGML_API void ggml_build_forward_expand(struct ggml_cgraph * cgraph, struct ggml_tensor * tensor);
def ggml_build_forward_expand(cgraph, tensor):
    return lib.ggml_build_forward_expand(cgraph, tensor)

lib.ggml_build_forward_expand.argtypes = [ctypes.POINTER(ggml_cgraph), ctypes.POINTER(ggml_tensor)]
lib.ggml_build_forward_expand.restype = None

# GGML_API struct ggml_cgraph ggml_build_forward (struct ggml_tensor * tensor);
def ggml_build_forward(tensor):
    return lib.ggml_build_forward(tensor)

lib.ggml_build_forward.argtypes = [ctypes.POINTER(ggml_tensor)]
lib.ggml_build_forward.restype = ggml_cgraph

# GGML_API struct ggml_cgraph ggml_build_backward(struct ggml_context * ctx, struct ggml_cgraph * gf, bool keep);
def ggml_build_backward(ctx, gf, keep):
    return lib.ggml_build_backward(ctx, gf, keep)

lib.ggml_build_backward.argtypes = [ggml_context_p, ctypes.POINTER(ggml_cgraph), ctypes.c_bool]
lib.ggml_build_backward.restype = ggml_cgraph

# GGML_API void ggml_graph_compute(struct ggml_context * ctx, struct ggml_cgraph * cgraph);
def ggml_graph_compute(ctx, cgraph):
    return lib.ggml_graph_compute(ctx, cgraph)

lib.ggml_graph_compute.argtypes = [ggml_context_p, ctypes.POINTER(ggml_cgraph)]
lib.ggml_graph_compute.restype = None

# GGML_API void ggml_graph_reset  (struct ggml_cgraph * cgraph);
def ggml_graph_reset(cgraph):
    return lib.ggml_graph_reset(cgraph)

lib.ggml_graph_reset.argtypes = [ctypes.POINTER(ggml_cgraph)]
lib.ggml_graph_reset.restype = None

# // print info and performance information for the graph
# GGML_API void ggml_graph_print(const struct ggml_cgraph * cgraph);
def ggml_graph_print(cgraph):
    return lib.ggml_graph_print(cgraph)

lib.ggml_graph_print.argtypes = [ctypes.POINTER(ggml_cgraph)]
lib.ggml_graph_print.restype = None

# // dump the graph into a file using the dot format
# GGML_API void ggml_graph_dump_dot(const struct ggml_cgraph * gb, const struct ggml_cgraph * gf, const char * filename);
def ggml_graph_dump_dot(gb, gf, filename):
    return lib.ggml_graph_dump_dot(gb, gf, filename)

lib.ggml_graph_dump_dot.argtypes = [ctypes.POINTER(ggml_cgraph), ctypes.POINTER(ggml_cgraph), ctypes.c_char_p]
lib.ggml_graph_dump_dot.restype = None


# //
# // optimization
# //

# // optimization methods
# enum ggml_opt_type {
#     GGML_OPT_ADAM,
#     GGML_OPT_LBFGS,
# };
GGML_OPT_ADAM = ctypes.c_int(0)
GGML_OPT_LBFGS = ctypes.c_int(1)

# // linesearch methods
# enum ggml_linesearch {
#     GGML_LINESEARCH_DEFAULT = 1,

#     GGML_LINESEARCH_BACKTRACKING_ARMIJO       = 0,
#     GGML_LINESEARCH_BACKTRACKING_WOLFE        = 1,
#     GGML_LINESEARCH_BACKTRACKING_STRONG_WOLFE = 2,
# };
GGML_LINESEARCH_DEFAULT = ctypes.c_int(1)
GGML_LINESEARCH_BACKTRACKING_ARMIJO = ctypes.c_int(0)
GGML_LINESEARCH_BACKTRACKING_WOLFE = ctypes.c_int(1)
GGML_LINESEARCH_BACKTRACKING_STRONG_WOLFE = ctypes.c_int(2)

# // optimization return values
# enum ggml_opt_result {
#     GGML_OPT_OK = 0,
#     GGML_OPT_DID_NOT_CONVERGE,
#     GGML_OPT_NO_CONTEXT,
#     GGML_OPT_INVALID_WOLFE,
#     GGML_OPT_FAIL,

#     GGML_LINESEARCH_FAIL = -128,
#     GGML_LINESEARCH_MINIMUM_STEP,
#     GGML_LINESEARCH_MAXIMUM_STEP,
#     GGML_LINESEARCH_MAXIMUM_ITERATIONS,
#     GGML_LINESEARCH_INVALID_PARAMETERS,
# };
GGML_OPT_OK = ctypes.c_int(0)
GGML_OPT_DID_NOT_CONVERGE = ctypes.c_int(1)
GGML_OPT_NO_CONTEXT = ctypes.c_int(2)
GGML_OPT_INVALID_WOLFE = ctypes.c_int(3)
GGML_OPT_FAIL = ctypes.c_int(4)
GGML_LINESEARCH_FAIL = ctypes.c_int(-128)
GGML_LINESEARCH_MINIMUM_STEP = ctypes.c_int(-127)
GGML_LINESEARCH_MAXIMUM_STEP = ctypes.c_int(-126)
GGML_LINESEARCH_MAXIMUM_ITERATIONS = ctypes.c_int(-125)
GGML_LINESEARCH_INVALID_PARAMETERS = ctypes.c_int(-124)


# // optimization parameters
# //
# //   see ggml.c (ggml_opt_default_params) for default values
# //
# struct ggml_opt_params {
#     enum ggml_opt_type type;

#     int n_threads;

#     // delta-based convergence test
#     //
#     //   if past == 0 - disabled
#     //   if past > 0:
#     //     stop if |f(x) - f(x_past)| < delta * max(1, |f(x)|)
#     //
#     int past;
#     float delta;

#     // maximum number of iterations without improvement
#     //
#     //   if 0 - disabled
#     //   if > 0:
#     //     assume convergence if no cost improvement in this number of iterations
#     //
#     int max_no_improvement;

#     bool print_forward_graph;
#     bool print_backward_graph;

#     // ADAM parameters
#     struct {
#         int n_iter;

#         float alpha; // learning rate
#         float beta1;
#         float beta2;
#         float eps;   // epsilon for numerical stability
#         float eps_f; // epsilon for convergence test
#         float eps_g; // epsilon for convergence test
#     } adam;

#     // LBFGS parameters
#     struct {
#         int m; // number of corrections to approximate the inv. Hessian
#         int n_iter;
#         int max_linesearch;

#         float eps;      // convergence tolerance
#         float ftol;     // line search tolerance
#         float wolfe;
#         float min_step;
#         float max_step;

#         enum ggml_linesearch linesearch;
#     } lbfgs;
# };

class ggml_opt_params_adam(ctypes.Structure):
    _fields_ = [
        ("n_iter", ctypes.c_int),
        ("alpha", ctypes.c_float),
        ("beta1", ctypes.c_float),
        ("beta2", ctypes.c_float),
        ("eps", ctypes.c_float),
        ("eps_f", ctypes.c_float),
        ("eps_g", ctypes.c_float),
    ]

class ggml_opt_params_lbfgs(ctypes.Structure):
    _fields_ = [
        ("m", ctypes.c_int),
        ("n_iter", ctypes.c_int),
        ("max_linesearch", ctypes.c_int),
        ("eps", ctypes.c_float),
        ("ftol", ctypes.c_float),
        ("wolfe", ctypes.c_float),
        ("min_step", ctypes.c_float),
        ("max_step", ctypes.c_float),
        ("linesearch", ctypes.c_int),
    ]

class ggml_opt_params(ctypes.Structure):
    _fields_ = [
        ("type", ctypes.c_int),
        ("n_threads", ctypes.c_int),
        ("past", ctypes.c_int),
        ("delta", ctypes.c_float),
        ("max_no_improvement", ctypes.c_int),
        ("print_forward_graph", ctypes.c_bool),
        ("print_backward_graph", ctypes.c_bool),
        ("adam", ggml_opt_params_adam),
        ("lbfgs", ggml_opt_params_lbfgs),
    ]

# GGML_API struct ggml_opt_params ggml_opt_default_params(enum ggml_opt_type type);
def ggml_opt_default_params(type):
    return lib.ggml_opt_default_params(type)

lib.ggml_opt_default_params.argtypes = [ctypes.c_int]
lib.ggml_opt_default_params.restype = ggml_opt_params

# // optimize the function defined by the tensor f
# GGML_API enum ggml_opt_result ggml_opt(
#         struct ggml_context * ctx,
#         struct ggml_opt_params params,
#         struct ggml_tensor * f);
def ggml_opt(ctx, params, f):
    return lib.ggml_opt(ctx, params, f)

lib.ggml_opt.argtypes = [ggml_context_p, ggml_opt_params, ctypes.POINTER(ggml_tensor)]
lib.ggml_opt.restype = ctypes.c_int



# //
# // quantization
# //

# GGML_API size_t ggml_quantize_q4_0(const float * src, void * dst, int n, int k, int64_t * hist);
def ggml_quantize_q4_0(src, dst, n, k, hist):
    return lib.ggml_quantize_q4_0(src, dst, n, k, hist)

lib.ggml_quantize_q4_0.argtypes = [ctypes.POINTER(ctypes.c_float), ctypes.c_void_p, ctypes.c_int, ctypes.c_int, ctypes.POINTER(ctypes.c_int64)]
lib.ggml_quantize_q4_0.restype = ctypes.c_int

# GGML_API size_t ggml_quantize_q4_1(const float * src, void * dst, int n, int k, int64_t * hist);
def ggml_quantize_q4_1(src, dst, n, k, hist):
    return lib.ggml_quantize_q4_1(src, dst, n, k, hist)

lib.ggml_quantize_q4_1.argtypes = [ctypes.POINTER(ctypes.c_float), ctypes.c_void_p, ctypes.c_int, ctypes.c_int, ctypes.POINTER(ctypes.c_int64)]
lib.ggml_quantize_q4_1.restype = ctypes.c_int

# GGML_API size_t ggml_quantize_q5_0(const float * src, void * dst, int n, int k, int64_t * hist);
def ggml_quantize_q5_0(src, dst, n, k, hist):
    return lib.ggml_quantize_q5_0(src, dst, n, k, hist)

lib.ggml_quantize_q5_0.argtypes = [ctypes.POINTER(ctypes.c_float), ctypes.c_void_p, ctypes.c_int, ctypes.c_int, ctypes.POINTER(ctypes.c_int64)]
lib.ggml_quantize_q5_0.restype = ctypes.c_int

# GGML_API size_t ggml_quantize_q5_1(const float * src, void * dst, int n, int k, int64_t * hist);
def ggml_quantize_q5_1(src, dst, n, k, hist):
    return lib.ggml_quantize_q5_1(src, dst, n, k, hist)

lib.ggml_quantize_q5_1.argtypes = [ctypes.POINTER(ctypes.c_float), ctypes.c_void_p, ctypes.c_int, ctypes.c_int, ctypes.POINTER(ctypes.c_int64)]
lib.ggml_quantize_q5_1.restype = ctypes.c_int

# GGML_API size_t ggml_quantize_q8_0(const float * src, void * dst, int n, int k, int64_t * hist);
def ggml_quantize_q8_0(src, dst, n, k, hist):
    return lib.ggml_quantize_q8_0(src, dst, n, k, hist)

lib.ggml_quantize_q8_0.argtypes = [ctypes.POINTER(ctypes.c_float), ctypes.c_void_p, ctypes.c_int, ctypes.c_int, ctypes.POINTER(ctypes.c_int64)]
lib.ggml_quantize_q8_0.restype = ctypes.c_int

# GGML_API size_t ggml_quantize_chunk(enum ggml_type type, const float * src, void * dst, int start, int n, int64_t * hist);
def ggml_quantize_chunk(type, src, dst, start, n, hist):
    return lib.ggml_quantize_chunk(type, src, dst, start, n, hist)

lib.ggml_quantize_chunk.argtypes = [ctypes.c_int, ctypes.POINTER(ctypes.c_float), ctypes.c_void_p, ctypes.c_int, ctypes.c_int, ctypes.POINTER(ctypes.c_int64)]
lib.ggml_quantize_chunk.restype = ctypes.c_int

# //
# // system info
# //

# GGML_API int ggml_cpu_has_avx        (void);
def ggml_cpu_has_avx():
    return lib.ggml_cpu_has_avx()

lib.ggml_cpu_has_avx.argtypes = []
lib.ggml_cpu_has_avx.restype = ctypes.c_int

# GGML_API int ggml_cpu_has_avx2       (void);
def ggml_cpu_has_avx2():
    return lib.ggml_cpu_has_avx2()

lib.ggml_cpu_has_avx2.argtypes = []
lib.ggml_cpu_has_avx2.restype = ctypes.c_int

# GGML_API int ggml_cpu_has_avx512     (void);
def ggml_cpu_has_avx512():
    return lib.ggml_cpu_has_avx512()

lib.ggml_cpu_has_avx512.argtypes = []
lib.ggml_cpu_has_avx512.restype = ctypes.c_int

# GGML_API int ggml_cpu_has_avx512_vbmi(void);
def ggml_cpu_has_avx512_vbmi():
    return lib.ggml_cpu_has_avx512_vbmi()

lib.ggml_cpu_has_avx512_vbmi.argtypes = []
lib.ggml_cpu_has_avx512_vbmi.restype = ctypes.c_int

# GGML_API int ggml_cpu_has_avx512_vnni(void);
def ggml_cpu_has_avx512_vnni():
    return lib.ggml_cpu_has_avx512_vnni()

lib.ggml_cpu_has_avx512_vnni.argtypes = []
lib.ggml_cpu_has_avx512_vnni.restype = ctypes.c_int

# GGML_API int ggml_cpu_has_fma        (void);
def ggml_cpu_has_fma():
    return lib.ggml_cpu_has_fma()

lib.ggml_cpu_has_fma.argtypes = []
lib.ggml_cpu_has_fma.restype = ctypes.c_int

# GGML_API int ggml_cpu_has_neon       (void);
def ggml_cpu_has_neon():
    return lib.ggml_cpu_has_neon()

lib.ggml_cpu_has_neon.argtypes = []
lib.ggml_cpu_has_neon.restype = ctypes.c_int

# GGML_API int ggml_cpu_has_arm_fma    (void);
def ggml_cpu_has_arm_fma():
    return lib.ggml_cpu_has_arm_fma()

lib.ggml_cpu_has_arm_fma.argtypes = []
lib.ggml_cpu_has_arm_fma.restype = ctypes.c_int

# GGML_API int ggml_cpu_has_f16c       (void);
def ggml_cpu_has_f16c():
    return lib.ggml_cpu_has_f16c()

lib.ggml_cpu_has_f16c.argtypes = []
lib.ggml_cpu_has_f16c.restype = ctypes.c_int

# GGML_API int ggml_cpu_has_fp16_va    (void);
def ggml_cpu_has_fp16_va():
    return lib.ggml_cpu_has_fp16_va()

lib.ggml_cpu_has_fp16_va.argtypes = []
lib.ggml_cpu_has_fp16_va.restype = ctypes.c_int

# GGML_API int ggml_cpu_has_wasm_simd  (void);
def ggml_cpu_has_wasm_simd():
    return lib.ggml_cpu_has_wasm_simd()

lib.ggml_cpu_has_wasm_simd.argtypes = []
lib.ggml_cpu_has_wasm_simd.restype = ctypes.c_int

# GGML_API int ggml_cpu_has_blas       (void);
def ggml_cpu_has_blas():
    return lib.ggml_cpu_has_blas()

lib.ggml_cpu_has_blas.argtypes = []
lib.ggml_cpu_has_blas.restype = ctypes.c_int

# GGML_API int ggml_cpu_has_cublas     (void);
def ggml_cpu_has_cublas():
    return lib.ggml_cpu_has_cublas()

lib.ggml_cpu_has_cublas.argtypes = []
lib.ggml_cpu_has_cublas.restype = ctypes.c_int

# GGML_API int ggml_cpu_has_clblast    (void);
def ggml_cpu_has_clblast():
    return lib.ggml_cpu_has_clblast()

lib.ggml_cpu_has_clblast.argtypes = []
lib.ggml_cpu_has_clblast.restype = ctypes.c_int

# GGML_API int ggml_cpu_has_gpublas    (void);
def ggml_cpu_has_gpublas():
    return lib.ggml_cpu_has_gpublas()

lib.ggml_cpu_has_gpublas.argtypes = []
lib.ggml_cpu_has_gpublas.restype = ctypes.c_int

# GGML_API int ggml_cpu_has_sse3       (void);
def ggml_cpu_has_sse3():
    return lib.ggml_cpu_has_sse3()

lib.ggml_cpu_has_sse3.argtypes = []
lib.ggml_cpu_has_sse3.restype = ctypes.c_int

# GGML_API int ggml_cpu_has_vsx        (void);
def ggml_cpu_has_vsx():
    return lib.ggml_cpu_has_vsx()

lib.ggml_cpu_has_vsx.argtypes = []
lib.ggml_cpu_has_vsx.restype = ctypes.c_int