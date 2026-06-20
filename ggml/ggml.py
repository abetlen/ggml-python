"""This module is the core of the ggml-python library, it exposes a low-level [ctypes](https://docs.python.org/3/library/ctypes.html)-based interface for ggml.

Structures and functions in the `ggml.ggml` module map directly to the original ggml C library and
they operate at a fairly low level.
No additional runtime checks checks are performed nor is memory management handled automatically.
You've been warned :).

With that in mind here are some useful things to keep in mind

- While runtime checks are avoided for performance reasons, this module attempts to provide a type-safe interface by using Python's type annotations. Please report any issues you find.
- Functions accept both ctypes types (c_int, c_bool, c_float, etc.) and Python types (int, bool, float, etc.) as parameters.
- Functions return Python types for simple values (int, bool, float, etc.) and ctypes types for complex values ([ggml_context_p][ggml.ggml_context_p], [ggml_tensor_p][ggml.ggml_tensor_p], etc.).
- Memory management is the responsibility of the user. The user must call `ggml.ggml_free` on the context after calling `ggml.ggml_init`.
- Opaque pointers that are returned by ggml functions (e.g. `ggml.ggml_init`) are returned as int's or None in Python. For some additional static type safety these pointers are wrapped in [NewType](https://docs.python.org/3/library/typing.html#typing.NewType) definitions (e.g. [ggml.ggml_context_p][ggml.ggml_context_p]).

Example

```python
import ggml
import ctypes

# Allocate a new context with 16 MB of memory
params = ggml.ggml_init_params(mem_size=16 * 1024 * 1024, mem_buffer=None)
ctx = ggml.ggml_init(params)

# Instantiate tensors
x = ggml.ggml_new_tensor_1d(ctx, ggml.GGML_TYPE_F32, 1)
a = ggml.ggml_new_tensor_1d(ctx, ggml.GGML_TYPE_F32, 1)
b = ggml.ggml_new_tensor_1d(ctx, ggml.GGML_TYPE_F32, 1)

# Use ggml operations to build a computational graph
x2 = ggml.ggml_mul(ctx, x, x)
f = ggml.ggml_add(ctx, ggml.ggml_mul(ctx, a, x2), b)

gf = ggml.ggml_new_graph(ctx)
ggml.ggml_build_forward_expand(gf, f)

# Set the input values
ggml.ggml_set_f32(x, 2.0)
ggml.ggml_set_f32(a, 3.0)
ggml.ggml_set_f32(b, 4.0)

# Compute the graph
ggml.ggml_graph_compute_with_ctx(ctx, gf, 1)

# Get the output value
output = ggml.ggml_get_f32_1d(f, 0)
assert output == 16.0

# Free the context
ggml.ggml_free(ctx)
```

"""
from __future__ import annotations

import os
import sys
import ctypes
import pathlib
import functools
try:
    # Python < 3.9
    import importlib_resources
except ImportError:
    import importlib.resources as importlib_resources
from typing import (
    cast,
    Any,
    Callable,
    List,
    Optional,
    Sequence,
    Union,
    NewType,
    TYPE_CHECKING,
    TypeVar,
    Generic,
)
from typing_extensions import TypeAlias


# Load the library
def load_shared_library(module_name: str, lib_base_name: str):
    # Construct the paths to the possible shared library names
    base_path = pathlib.Path(__file__).parent.resolve() / "lib"
    # Searching for the library in the current directory under the name "libggml" (default name
    # for ggml) and "ggml" (default name for this repo)
    lib_names: List[str] = [
        f"lib{lib_base_name}.so",
        f"lib{lib_base_name}.dylib",
        f"{lib_base_name}.dll",
    ]

    path: Optional[pathlib.Path] = None

    for lib_name in lib_names:
        try:
            with importlib_resources.as_file(importlib_resources.files(module_name).joinpath("lib", lib_name)) as p: # type: ignore
                p = cast(pathlib.Path, p)
                if os.path.exists(p):
                    path = p
                    break
        except FileNotFoundError:
            pass

    if path is None:
        raise FileNotFoundError(
            f"Shared library with base name '{lib_base_name}' not found"
        )

    cdll_args = dict()  # type: ignore
    # Add the library directory to the DLL search path on Windows (if needed)
    if sys.platform == "win32" and sys.version_info >= (3, 8):
        os.environ["PATH"] = str(base_path) + os.pathsep + os.environ["PATH"]
        os.add_dll_directory(str(base_path))
        cdll_args["winmode"] = 0

    preloaded_libraries: List[ctypes.CDLL] = []

    def preload_local_library(dep_base_name: str):
        dep_names = [
            f"lib{dep_base_name}.so",
            f"lib{dep_base_name}.dylib",
            f"{dep_base_name}.dll",
        ]
        cdll_dep_args = dict(cdll_args)
        if hasattr(ctypes, "RTLD_GLOBAL") and sys.platform != "win32":
            cdll_dep_args["mode"] = ctypes.RTLD_GLOBAL
        for dep_name in dep_names:
            dep_path = base_path / dep_name
            if dep_path.exists():
                preloaded_libraries.append(ctypes.CDLL(str(dep_path), **cdll_dep_args))  # type: ignore
                break

    if lib_base_name == "ggml":
        for dep_base_name in (
            "ggml-base",
            "ggml-cpu",
            "ggml-blas",
            "ggml-cann",
            "ggml-cuda",
            "ggml-hexagon",
            "ggml-hip",
            "ggml-metal",
            "ggml-musa",
            "ggml-opencl",
            "ggml-openvino",
            "ggml-rpc",
            "ggml-sycl",
            "ggml-vulkan",
            "ggml-virtgpu",
            "ggml-webgpu",
            "ggml-zdnn",
            "ggml-zendnn",
        ):
            preload_local_library(dep_base_name)

    class SharedLibraryProxy:
        def __init__(self, libraries: List[ctypes.CDLL]):
            self._libraries = libraries

        def __getattr__(self, name: str):
            for library in self._libraries:
                try:
                    return getattr(library, name)
                except AttributeError:
                    pass
            raise AttributeError(name)

    # Try to load the shared library, handling potential errors
    try:
        primary = ctypes.CDLL(str(path), **cdll_args)  # type: ignore
        return SharedLibraryProxy([primary] + preloaded_libraries)
    except Exception as e:
        raise RuntimeError(f"Failed to load shared library '{path}': {e}")


module_name = "ggml"
lib_base_name = "ggml"
lib = load_shared_library(module_name, lib_base_name)


#####################################################
# GGML Utility Types
#####################################################


if TYPE_CHECKING:
    CtypesCData = TypeVar("CtypesCData", bound=ctypes._CData)  # type: ignore

    CtypesArray: TypeAlias = ctypes.Array[CtypesCData]  # type: ignore

    CtypesPointer: TypeAlias = ctypes._Pointer[CtypesCData]  # type: ignore

    CtypesVoidPointer: TypeAlias = ctypes.c_void_p

    class CtypesRef(Generic[CtypesCData]):
        pass

    CtypesPointerOrRef: TypeAlias = Union[
        CtypesPointer[CtypesCData], CtypesRef[CtypesCData]
    ]

    CtypesFuncPointer: TypeAlias = ctypes._FuncPointer  # type: ignore


def byref(obj: CtypesCData, offset: Optional[int] = None) -> CtypesRef[CtypesCData]:
    """Type-annotated version of ctypes.byref"""
    return ctypes.byref(obj, offset) if offset is not None else ctypes.byref(obj)  # type: ignore

F = TypeVar("F", bound=Callable[..., Any])

def ctypes_function_for_shared_library(lib: ctypes.CDLL):
    def ctypes_function(
        name: str, argtypes: List[Any], restype: Any, enabled: bool = True
    ):
        def decorator(f: F) -> F:
            if enabled:
                func = getattr(lib, name)
                func.argtypes = argtypes
                func.restype = restype
                functools.wraps(f)(func)
                return func
            else:
                def f_(*args: Any, **kwargs: Any):
                    raise RuntimeError(
                        f"Function '{name}' is not available in the shared library (enabled=False)"
                    )
                return cast(F, f_)

        return decorator

    return ctypes_function


ggml_function = ctypes_function_for_shared_library(lib)


#####################################################
# GGML API
# source: include/ggml/ggml.h
#####################################################


# define GGML_FILE_MAGIC   0x67676d6c // "ggml"
# define GGML_FILE_VERSION 2
GGML_FILE_MAGIC = 0x67676D6C
GGML_FILE_VERSION = 2

# define GGML_QNT_VERSION        2    // bump this on quantization format changes
# define GGML_QNT_VERSION_FACTOR 1000 // do not change this
GGML_QNT_VERSION = 2
GGML_QNT_VERSION_FACTOR = 1000

# define GGML_MAX_DIMS           4
# define GGML_MAX_PARAMS         2048
# define GGML_MAX_SRC            10
# define GGML_MAX_N_THREADS      512
# define GGML_MAX_NAME           64
# define GGML_MAX_OP_PARAMS      64
# define GGML_DEFAULT_N_THREADS  4
# define GGML_DEFAULT_GRAPH_SIZE 2048
GGML_MAX_DIMS = 4
GGML_MAX_PARAMS = 2048
GGML_MAX_SRC = 10
GGML_MAX_N_THREADS = 512
GGML_MAX_NAME = 64
GGML_MAX_OP_PARAMS = 64
GGML_DEFAULT_N_THREADS = 4
GGML_DEFAULT_GRAPH_SIZE = 2048

# #if UINTPTR_MAX == 0xFFFFFFFF
#     #define GGML_MEM_ALIGN 4
# #else
#     #define GGML_MEM_ALIGN 16
# #endif
GGML_MEM_ALIGN = 4 if ctypes.sizeof(ctypes.c_void_p) == 4 else 16
GGML_MEMALIGN = GGML_MEM_ALIGN

# #define GGML_EXIT_SUCCESS 0
GGML_EXIT_SUCCESS = 0
# #define GGML_EXIT_ABORTED 1
GGML_EXIT_ABORTED = 1

GGML_VERSION_MAJOR = 0
GGML_VERSION_MINOR = 15
GGML_VERSION_PATCH = 2
GGML_VERSION = "0.15.2"

GGML_ROPE_TYPE_NORMAL = 0
GGML_ROPE_TYPE_NEOX = 2
GGML_ROPE_TYPE_MROPE = 8
GGML_ROPE_TYPE_VISION = 24
GGML_ROPE_TYPE_IMROPE = 40
GGML_MROPE_SECTIONS = 4

# define GGUF_MAGIC "GGUF"
GGUF_MAGIC = "GGUF"

# define GGUF_VERSION 3
GGUF_VERSION = 3

# #define GGUF_DEFAULT_ALIGNMENT 32
GGUF_DEFAULT_ALIGNMENT = 32

# enum ggml_status {
#     GGML_STATUS_ALLOC_FAILED = -2,
#     GGML_STATUS_FAILED = -1,
#     GGML_STATUS_SUCCESS = 0,
#     GGML_STATUS_ABORTED = 1,
# };
GGML_STATUS_ALLOC_FAILED = -2
GGML_STATUS_FAILED = -1
GGML_STATUS_SUCCESS = 0
GGML_STATUS_ABORTED = 1


# // get ggml_status name string
# GGML_API GGML_CALL const char * ggml_status_to_string(enum ggml_status status);
@ggml_function("ggml_status_to_string", [ctypes.c_int], ctypes.c_char_p)
def ggml_status_to_string(status: int, /) -> bytes:
    ...


ggml_abort_callback_t = ctypes.CFUNCTYPE(None, ctypes.c_char_p)


@ggml_function(
    "ggml_set_abort_callback", [ggml_abort_callback_t], ggml_abort_callback_t
)
def ggml_set_abort_callback(callback: ggml_abort_callback_t, /) -> ggml_abort_callback_t:
    ...


# GGML_API void ggml_abort(const char * file, int line, const char * fmt, ...);
@ggml_function(
    "ggml_abort",
    [ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p],
    None,
)
def ggml_abort(
    file: bytes,
    line: Union[ctypes.c_int, int],
    fmt: bytes,
    *args: Any,
) -> None:
    ...


# // ieee 754-2008 half-precision float16
# // todo: make this not an integral type
# typedef uint16_t ggml_fp16_t;
ggml_fp16_t = ctypes.c_uint16


# GGML_API float       ggml_fp16_to_fp32(ggml_fp16_t);
@ggml_function("ggml_fp16_to_fp32", [ggml_fp16_t], ctypes.c_float)
def ggml_fp16_to_fp32(fp16: ggml_fp16_t, /) -> float:
    ...


# GGML_API ggml_fp16_t ggml_fp32_to_fp16(float);
@ggml_function("ggml_fp32_to_fp16", [ctypes.c_float], ggml_fp16_t)
def ggml_fp32_to_fp16(fp32: float, /) -> ggml_fp16_t:
    ...


# GGML_API void        ggml_fp16_to_fp32_row(const ggml_fp16_t *, float *, int64_t);
@ggml_function("ggml_fp16_to_fp32_row", [ctypes.POINTER(ggml_fp16_t), ctypes.POINTER(ctypes.c_float), ctypes.c_int64], None)
def ggml_fp16_to_fp32_row(fp16: CtypesPointer[ggml_fp16_t], fp32: CtypesPointer[ctypes.c_float], n: int, /) -> None:
    ...


# GGML_API void        ggml_fp32_to_fp16_row(const float *, ggml_fp16_t *, int64_t);
@ggml_function("ggml_fp32_to_fp16_row", [ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ggml_fp16_t), ctypes.c_int64], None)
def ggml_fp32_to_fp16_row(fp32: CtypesPointer[ctypes.c_float], fp16: CtypesPointer[ggml_fp16_t], n: int, /) -> None:
    ...


# // google brain half-precision bfloat16
# typedef struct { uint16_t bits; } ggml_bf16_t;
class ggml_bf16_t(ctypes.Structure):
    _fields_ = [("bits", ctypes.c_uint16)]


# GGML_API ggml_bf16_t ggml_fp32_to_bf16(float);
@ggml_function("ggml_fp32_to_bf16", [ctypes.c_float], ggml_bf16_t)
def ggml_fp32_to_bf16(fp32: float, /) -> ggml_bf16_t:
    ...


# GGML_API float       ggml_bf16_to_fp32(ggml_bf16_t);  // consider just doing << 16
@ggml_function("ggml_bf16_to_fp32", [ggml_bf16_t], ctypes.c_float)
def ggml_bf16_to_fp32(bf16: ggml_bf16_t, /) -> float:
    ...


# GGML_API void        ggml_bf16_to_fp32_row(const ggml_bf16_t *, float *, int64_t);
@ggml_function("ggml_bf16_to_fp32_row", [ctypes.POINTER(ggml_bf16_t), ctypes.POINTER(ctypes.c_float), ctypes.c_int64], None)
def ggml_bf16_to_fp32_row(bf16: CtypesPointer[ggml_bf16_t], fp32: CtypesPointer[ctypes.c_float], n: int, /) -> None:
    ...


# GGML_API void        ggml_fp32_to_bf16_row_ref(const float *, ggml_bf16_t *, int64_t);
@ggml_function("ggml_fp32_to_bf16_row_ref", [ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ggml_bf16_t), ctypes.c_int64], None)
def ggml_fp32_to_bf16_row_ref(fp32: CtypesPointer[ctypes.c_float], bf16: CtypesPointer[ggml_bf16_t], n: int, /) -> None:
    ...


# GGML_API void        ggml_fp32_to_bf16_row(const float *, ggml_bf16_t *, int64_t);
@ggml_function("ggml_fp32_to_bf16_row", [ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ggml_bf16_t), ctypes.c_int64], None)
def ggml_fp32_to_bf16_row(fp32: CtypesPointer[ctypes.c_float], bf16: CtypesPointer[ggml_bf16_t], n: int, /) -> None:
    ...


# struct ggml_context;
ggml_context_p = NewType("ggml_context_p", int)
"""Opaque pointer to a ggml_context.

ggml_context structs are not accessed directly instead they must be created using
`ggml_init` and freed using `ggml_free`."""

ggml_context_p_ctypes = ctypes.c_void_p  # type: ignore

# enum ggml_type {
#     GGML_TYPE_F32     = 0,
#     GGML_TYPE_F16     = 1,
#     GGML_TYPE_Q4_0    = 2,
#     GGML_TYPE_Q4_1    = 3,
#     // GGML_TYPE_Q4_2 = 4, support has been removed
#     // GGML_TYPE_Q4_3 = 5, support has been removed
#     GGML_TYPE_Q5_0    = 6,
#     GGML_TYPE_Q5_1    = 7,
#     GGML_TYPE_Q8_0    = 8,
#     GGML_TYPE_Q8_1    = 9,
#     GGML_TYPE_Q2_K    = 10,
#     GGML_TYPE_Q3_K    = 11,
#     GGML_TYPE_Q4_K    = 12,
#     GGML_TYPE_Q5_K    = 13,
#     GGML_TYPE_Q6_K    = 14,
#     GGML_TYPE_Q8_K    = 15,
#     GGML_TYPE_IQ2_XXS = 16,
#     GGML_TYPE_IQ2_XS  = 17,
#     GGML_TYPE_IQ3_XXS = 18,
#     GGML_TYPE_IQ1_S   = 19,
#     GGML_TYPE_IQ4_NL  = 20,
#     GGML_TYPE_IQ3_S   = 21,
#     GGML_TYPE_IQ2_S   = 22,
#     GGML_TYPE_IQ4_XS  = 23,
#     GGML_TYPE_I8      = 24,
#     GGML_TYPE_I16     = 25,
#     GGML_TYPE_I32     = 26,
#     GGML_TYPE_I64     = 27,
#     GGML_TYPE_F64     = 28,
#     GGML_TYPE_IQ1_M   = 29,
#     GGML_TYPE_BF16    = 30,
#     GGML_TYPE_TQ1_0   = 34,
#     GGML_TYPE_TQ2_0   = 35,
#     GGML_TYPE_MXFP4   = 39,
#     GGML_TYPE_NVFP4   = 40,
#     GGML_TYPE_Q1_0    = 41,
#     GGML_TYPE_COUNT   = 42,
# };
GGML_TYPE_F32 = 0
GGML_TYPE_F16 = 1
GGML_TYPE_Q4_0 = 2
GGML_TYPE_Q4_1 = 3
GGML_TYPE_Q5_0 = 6
GGML_TYPE_Q5_1 = 7
GGML_TYPE_Q8_0 = 8
GGML_TYPE_Q8_1 = 9
GGML_TYPE_Q2_K = 10
GGML_TYPE_Q3_K = 11
GGML_TYPE_Q4_K = 12
GGML_TYPE_Q5_K = 13
GGML_TYPE_Q6_K = 14
GGML_TYPE_Q8_K = 15
GGML_TYPE_IQ2_XXS = 16
GGML_TYPE_IQ2_XS = 17
GGML_TYPE_IQ3_XXS = 18
GGML_TYPE_IQ1_S = 19
GGML_TYPE_IQ4_NL = 20
GGML_TYPE_IQ3_S = 21
GGML_TYPE_IQ2_S = 22
GGML_TYPE_IQ4_XS = 23
GGML_TYPE_I8 = 24
GGML_TYPE_I16 = 25
GGML_TYPE_I32 = 26
GGML_TYPE_I64 = 27
GGML_TYPE_F64 = 28
GGML_TYPE_IQ1_M = 29
GGML_TYPE_BF16 = 30
GGML_TYPE_TQ1_0 = 34
GGML_TYPE_TQ2_0 = 35
GGML_TYPE_MXFP4 = 39
GGML_TYPE_NVFP4 = 40
GGML_TYPE_Q1_0 = 41
GGML_TYPE_COUNT = 42


# // precision
# enum ggml_prec {
#     GGML_PREC_DEFAULT =  0,
#     GGML_PREC_F32     = 10,
# };
GGML_PREC_DEFAULT = 0
GGML_PREC_F32 = 10

# enum ggml_op_hint {
#     GGML_HINT_NONE             = 0,
#     GGML_HINT_SRC0_IS_HADAMARD = 1,
# };
GGML_HINT_NONE = 0
GGML_HINT_SRC0_IS_HADAMARD = 1

# enum ggml_backend_type {
#     GGML_BACKEND_TYPE_CPU = 0,
#     GGML_BACKEND_TYPE_GPU = 10,
#     GGML_BACKEND_TYPE_GPU_SPLIT = 20,
# };
GGML_BACKEND_TYPE_CPU = 0
GGML_BACKEND_TYPE_GPU = 10
GGML_BACKEND_TYPE_GPU_SPLIT = 20


# // model file types
# enum ggml_ftype {
#     GGML_FTYPE_UNKNOWN        = -1,
#     GGML_FTYPE_ALL_F32        = 0,
#     GGML_FTYPE_MOSTLY_F16     = 1,  // except 1d tensors
#     GGML_FTYPE_MOSTLY_Q4_0    = 2,  // except 1d tensors
#     GGML_FTYPE_MOSTLY_Q4_1    = 3,  // except 1d tensors
#     GGML_FTYPE_MOSTLY_Q4_1_SOME_F16 = 4, // tok_embeddings.weight and output.weight are F16
#     GGML_FTYPE_MOSTLY_Q8_0    = 7,  // except 1d tensors
#     GGML_FTYPE_MOSTLY_Q5_0    = 8,  // except 1d tensors
#     GGML_FTYPE_MOSTLY_Q5_1    = 9,  // except 1d tensors
#     GGML_FTYPE_MOSTLY_Q2_K    = 10, // except 1d tensors
#     GGML_FTYPE_MOSTLY_Q3_K    = 11, // except 1d tensors
#     GGML_FTYPE_MOSTLY_Q4_K    = 12, // except 1d tensors
#     GGML_FTYPE_MOSTLY_Q5_K    = 13, // except 1d tensors
#     GGML_FTYPE_MOSTLY_Q6_K    = 14, // except 1d tensors
#     GGML_FTYPE_MOSTLY_IQ2_XXS = 15, // except 1d tensors
#     GGML_FTYPE_MOSTLY_IQ2_XS  = 16, // except 1d tensors
#     GGML_FTYPE_MOSTLY_IQ3_XXS = 17, // except 1d tensors
#     GGML_FTYPE_MOSTLY_IQ1_S   = 18, // except 1d tensors
#     GGML_FTYPE_MOSTLY_IQ4_NL  = 19, // except 1d tensors
#     GGML_FTYPE_MOSTLY_IQ3_S   = 20, // except 1d tensors
#     GGML_FTYPE_MOSTLY_IQ2_S   = 21, // except 1d tensors
#     GGML_FTYPE_MOSTLY_IQ4_XS  = 22, // except 1d tensors
#     GGML_FTYPE_MOSTLY_IQ1_M   = 23, // except 1d tensors
#     GGML_FTYPE_MOSTLY_BF16    = 24, // except 1d tensors
#     GGML_FTYPE_MOSTLY_MXFP4   = 25, // except 1d tensors
#     GGML_FTYPE_MOSTLY_NVFP4   = 26, // except 1d tensors
#     GGML_FTYPE_MOSTLY_Q1_0    = 27, // except 1d tensors
# };
GGML_FTYPE_UNKNOWN = -1
GGML_FTYPE_ALL_F32 = 0
GGML_FTYPE_MOSTLY_F16 = 1
GGML_FTYPE_MOSTLY_Q4_0 = 2
GGML_FTYPE_MOSTLY_Q4_1 = 3
GGML_FTYPE_MOSTLY_Q4_1_SOME_F16 = 4
GGML_FTYPE_MOSTLY_Q8_0 = 7
GGML_FTYPE_MOSTLY_Q5_0 = 8
GGML_FTYPE_MOSTLY_Q5_1 = 9
GGML_FTYPE_MOSTLY_Q2_K = 10
GGML_FTYPE_MOSTLY_Q3_K = 11
GGML_FTYPE_MOSTLY_Q4_K = 12
GGML_FTYPE_MOSTLY_Q5_K = 13
GGML_FTYPE_MOSTLY_Q6_K = 14
GGML_FTYPE_MOSTLY_IQ2_XXS = 15
GGML_FTYPE_MOSTLY_IQ2_XS = 16
GGML_FTYPE_MOSTLY_IQ3_XXS = 17
GGML_FTYPE_MOSTLY_IQ1_S = 18
GGML_FTYPE_MOSTLY_IQ4_NL = 19
GGML_FTYPE_MOSTLY_IQ3_S = 20
GGML_FTYPE_MOSTLY_IQ2_S = 21
GGML_FTYPE_MOSTLY_IQ4_XS = 22
GGML_FTYPE_MOSTLY_IQ1_M = 23
GGML_FTYPE_MOSTLY_BF16 = 24
GGML_FTYPE_MOSTLY_MXFP4 = 25
GGML_FTYPE_MOSTLY_NVFP4 = 26
GGML_FTYPE_MOSTLY_Q1_0 = 27


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
#     GGML_OP_ARGMAX,
#     GGML_OP_REPEAT,
#     GGML_OP_REPEAT_BACK,
#     GGML_OP_CONCAT,
#     GGML_OP_SILU_BACK,
#     GGML_OP_NORM, // normalize
#     GGML_OP_RMS_NORM,
#     GGML_OP_RMS_NORM_BACK,
#     GGML_OP_GROUP_NORM,

#     GGML_OP_MUL_MAT,
#     GGML_OP_MUL_MAT_ID,
#     GGML_OP_OUT_PROD,

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
#     GGML_OP_SOFT_MAX_BACK,
#     GGML_OP_ROPE,
#     GGML_OP_ROPE_BACK,
#     GGML_OP_CLAMP,
#     GGML_OP_CONV_TRANSPOSE_1D,
#     GGML_OP_IM2COL,
#     GGML_OP_CONV_TRANSPOSE_2D,
#     GGML_OP_POOL_1D,
#     GGML_OP_POOL_2D,
#     GGML_OP_UPSCALE, // nearest interpolate
#     GGML_OP_PAD,
#     GGML_OP_ARANGE,
#     GGML_OP_TIMESTEP_EMBEDDING,
#     GGML_OP_ARGSORT,
#     GGML_OP_LEAKY_RELU,

#     GGML_OP_FLASH_ATTN_EXT,
#     GGML_OP_FLASH_ATTN_BACK,
#     GGML_OP_SSM_CONV,
#     GGML_OP_SSM_SCAN,
#     GGML_OP_WIN_PART,
#     GGML_OP_WIN_UNPART,
#     GGML_OP_GET_REL_POS,
#     GGML_OP_ADD_REL_POS,

#     GGML_OP_UNARY,

#     GGML_OP_MAP_UNARY,
#     GGML_OP_MAP_BINARY,

#     GGML_OP_MAP_CUSTOM1_F32,
#     GGML_OP_MAP_CUSTOM2_F32,
#     GGML_OP_MAP_CUSTOM3_F32,

#     GGML_OP_MAP_CUSTOM1,
#     GGML_OP_MAP_CUSTOM2,
#     GGML_OP_MAP_CUSTOM3,

#     GGML_OP_CROSS_ENTROPY_LOSS,
#     GGML_OP_CROSS_ENTROPY_LOSS_BACK,

#     GGML_OP_COUNT,
# };
GGML_OP_NONE = 0
GGML_OP_DUP = 1
GGML_OP_ADD = 2
GGML_OP_ADD_ID = 3
GGML_OP_ADD1 = 4
GGML_OP_ACC = 5
GGML_OP_SUB = 6
GGML_OP_MUL = 7
GGML_OP_DIV = 8
GGML_OP_SQR = 9
GGML_OP_SQRT = 10
GGML_OP_LOG = 11
GGML_OP_SIN = 12
GGML_OP_COS = 13
GGML_OP_SUM = 14
GGML_OP_SUM_ROWS = 15
GGML_OP_CUMSUM = 16
GGML_OP_MEAN = 17
GGML_OP_ARGMAX = 18
GGML_OP_COUNT_EQUAL = 19
GGML_OP_REPEAT = 20
GGML_OP_REPEAT_BACK = 21
GGML_OP_CONCAT = 22
GGML_OP_SILU_BACK = 23
GGML_OP_NORM = 24
GGML_OP_RMS_NORM = 25
GGML_OP_RMS_NORM_BACK = 26
GGML_OP_GROUP_NORM = 27
GGML_OP_L2_NORM = 28
GGML_OP_MUL_MAT = 29
GGML_OP_MUL_MAT_ID = 30
GGML_OP_OUT_PROD = 31
GGML_OP_SCALE = 32
GGML_OP_SET = 33
GGML_OP_CPY = 34
GGML_OP_CONT = 35
GGML_OP_RESHAPE = 36
GGML_OP_VIEW = 37
GGML_OP_PERMUTE = 38
GGML_OP_TRANSPOSE = 39
GGML_OP_GET_ROWS = 40
GGML_OP_GET_ROWS_BACK = 41
GGML_OP_SET_ROWS = 42
GGML_OP_DIAG = 43
GGML_OP_DIAG_MASK_INF = 44
GGML_OP_DIAG_MASK_ZERO = 45
GGML_OP_SOFT_MAX = 46
GGML_OP_SOFT_MAX_BACK = 47
GGML_OP_ROPE = 48
GGML_OP_ROPE_BACK = 49
GGML_OP_CLAMP = 50
GGML_OP_CONV_TRANSPOSE_1D = 51
GGML_OP_IM2COL = 52
GGML_OP_IM2COL_BACK = 53
GGML_OP_IM2COL_3D = 54
GGML_OP_COL2IM_1D = 55
GGML_OP_CONV_2D = 56
GGML_OP_CONV_3D = 57
GGML_OP_CONV_2D_DW = 58
GGML_OP_CONV_TRANSPOSE_2D = 59
GGML_OP_POOL_1D = 60
GGML_OP_POOL_2D = 61
GGML_OP_POOL_2D_BACK = 62
GGML_OP_UPSCALE = 63
GGML_OP_PAD = 64
GGML_OP_PAD_REFLECT_1D = 65
GGML_OP_ROLL = 66
GGML_OP_ARANGE = 67
GGML_OP_TIMESTEP_EMBEDDING = 68
GGML_OP_ARGSORT = 69
GGML_OP_TOP_K = 70
GGML_OP_LEAKY_RELU = 71
GGML_OP_TRI = 72
GGML_OP_FILL = 73
GGML_OP_FLASH_ATTN_EXT = 74
GGML_OP_FLASH_ATTN_BACK = 75
GGML_OP_SSM_CONV = 76
GGML_OP_SSM_SCAN = 77
GGML_OP_WIN_PART = 78
GGML_OP_WIN_UNPART = 79
GGML_OP_GET_REL_POS = 80
GGML_OP_ADD_REL_POS = 81
GGML_OP_RWKV_WKV6 = 82
GGML_OP_GATED_LINEAR_ATTN = 83
GGML_OP_RWKV_WKV7 = 84
GGML_OP_SOLVE_TRI = 85
GGML_OP_GATED_DELTA_NET = 86
GGML_OP_UNARY = 87
GGML_OP_MAP_CUSTOM1 = 88
GGML_OP_MAP_CUSTOM2 = 89
GGML_OP_MAP_CUSTOM3 = 90
GGML_OP_CUSTOM = 91
GGML_OP_CROSS_ENTROPY_LOSS = 92
GGML_OP_CROSS_ENTROPY_LOSS_BACK = 93
GGML_OP_OPT_STEP_ADAMW = 94
GGML_OP_OPT_STEP_SGD = 95
GGML_OP_GLU = 96
GGML_OP_COUNT = 97


# enum ggml_unary_op {
#     GGML_UNARY_OP_ABS,
#     GGML_UNARY_OP_SGN,
#     GGML_UNARY_OP_NEG,
#     GGML_UNARY_OP_STEP,
#     GGML_UNARY_OP_TANH,
#     GGML_UNARY_OP_ELU,
#     GGML_UNARY_OP_RELU,
#     GGML_UNARY_OP_SIGMOID,
#     GGML_UNARY_OP_GELU,
#     GGML_UNARY_OP_GELU_QUICK,
#     GGML_UNARY_OP_SILU,
#     GGML_UNARY_OP_HARDSWISH,
#     GGML_UNARY_OP_HARDSIGMOID,

#     GGML_UNARY_OP_COUNT,
# };
GGML_UNARY_OP_ABS = 0
GGML_UNARY_OP_SGN = 1
GGML_UNARY_OP_NEG = 2
GGML_UNARY_OP_STEP = 3
GGML_UNARY_OP_TANH = 4
GGML_UNARY_OP_ELU = 5
GGML_UNARY_OP_RELU = 6
GGML_UNARY_OP_SIGMOID = 7
GGML_UNARY_OP_GELU = 8
GGML_UNARY_OP_GELU_QUICK = 9
GGML_UNARY_OP_SILU = 10
GGML_UNARY_OP_HARDSWISH = 11
GGML_UNARY_OP_HARDSIGMOID = 12
GGML_UNARY_OP_EXP = 13
GGML_UNARY_OP_EXPM1 = 14
GGML_UNARY_OP_SOFTPLUS = 15
GGML_UNARY_OP_GELU_ERF = 16
GGML_UNARY_OP_XIELU = 17
GGML_UNARY_OP_FLOOR = 18
GGML_UNARY_OP_CEIL = 19
GGML_UNARY_OP_ROUND = 20
GGML_UNARY_OP_TRUNC = 21
GGML_UNARY_OP_COUNT = 22

# enum ggml_glu_op {
#     GGML_GLU_OP_REGLU,
#     GGML_GLU_OP_GEGLU,
#     GGML_GLU_OP_SWIGLU,
#     GGML_GLU_OP_SWIGLU_OAI,
#     GGML_GLU_OP_GEGLU_ERF,
#     GGML_GLU_OP_GEGLU_QUICK,
#     GGML_GLU_OP_COUNT,
# };
GGML_GLU_OP_REGLU = 0
GGML_GLU_OP_GEGLU = 1
GGML_GLU_OP_SWIGLU = 2
GGML_GLU_OP_SWIGLU_OAI = 3
GGML_GLU_OP_GEGLU_ERF = 4
GGML_GLU_OP_GEGLU_QUICK = 5
GGML_GLU_OP_COUNT = 6

# enum ggml_object_type {
#     GGML_OBJECT_TYPE_TENSOR,
#     GGML_OBJECT_TYPE_GRAPH,
#     GGML_OBJECT_TYPE_WORK_BUFFER
# };
GGML_OBJECT_TYPE_TENSOR = 0
GGML_OBJECT_TYPE_GRAPH = 1
GGML_OBJECT_TYPE_WORK_BUFFER = 2

# enum ggml_log_level {
#     GGML_LOG_LEVEL_NONE  = 0,
#     GGML_LOG_LEVEL_DEBUG = 1,
#     GGML_LOG_LEVEL_INFO  = 2,
#     GGML_LOG_LEVEL_WARN  = 3,
#     GGML_LOG_LEVEL_ERROR = 4,
#     GGML_LOG_LEVEL_CONT  = 5,
# };
GGML_LOG_LEVEL_NONE = 0
GGML_LOG_LEVEL_DEBUG = 1
GGML_LOG_LEVEL_INFO = 2
GGML_LOG_LEVEL_WARN = 3
GGML_LOG_LEVEL_ERROR = 4
GGML_LOG_LEVEL_CONT = 5


# enum ggml_tensor_flag {
#     GGML_TENSOR_FLAG_INPUT  = 1,
#     GGML_TENSOR_FLAG_OUTPUT = 2,
#     GGML_TENSOR_FLAG_PARAM  = 4,
#     GGML_TENSOR_FLAG_LOSS   = 8,
#     GGML_TENSOR_FLAG_COMPUTE = 16,
# };
GGML_TENSOR_FLAG_INPUT = 1
GGML_TENSOR_FLAG_OUTPUT = 2
GGML_TENSOR_FLAG_PARAM = 4
GGML_TENSOR_FLAG_LOSS = 8
GGML_TENSOR_FLAG_COMPUTE = 16

# enum ggml_tri_type {
#     GGML_TRI_TYPE_UPPER_DIAG = 0,
#     GGML_TRI_TYPE_UPPER      = 1,
#     GGML_TRI_TYPE_LOWER_DIAG = 2,
#     GGML_TRI_TYPE_LOWER      = 3
# };
GGML_TRI_TYPE_UPPER_DIAG = 0
GGML_TRI_TYPE_UPPER = 1
GGML_TRI_TYPE_LOWER_DIAG = 2
GGML_TRI_TYPE_LOWER = 3

# enum ggml_scale_mode {
#     GGML_SCALE_MODE_NEAREST  = 0,
#     GGML_SCALE_MODE_BILINEAR = 1,
#     GGML_SCALE_MODE_BICUBIC  = 2,
#
#     GGML_SCALE_MODE_COUNT
# };
GGML_SCALE_MODE_NEAREST = 0
GGML_SCALE_MODE_BILINEAR = 1
GGML_SCALE_MODE_BICUBIC = 2
GGML_SCALE_MODE_COUNT = 3

# enum ggml_scale_flag {
#     GGML_SCALE_FLAG_ALIGN_CORNERS = (1 << 8),
#     GGML_SCALE_FLAG_ANTIALIAS     = (1 << 9),
# };
GGML_SCALE_FLAG_ALIGN_CORNERS = 1 << 8
GGML_SCALE_FLAG_ANTIALIAS = 1 << 9

GGML_SCHED_PRIO_LOW = -1
GGML_SCHED_PRIO_NORMAL = 0
GGML_SCHED_PRIO_MEDIUM = 1
GGML_SCHED_PRIO_HIGH = 2
GGML_SCHED_PRIO_REALTIME = 3


# // ggml object
# struct ggml_object {
#     size_t offs;
#     size_t size;

#     struct ggml_object * next;

#     enum ggml_object_type type;


#     char padding[4];
# };
class ggml_object(ctypes.Structure):
    """ggml object
    
    Attributes:
        offs (int): offset
        size (int): size
        next (ctypes.pointer[ggml_object]): pointer to next object
        type (int): ggml object type
        padding (bytes): padding"""

    if TYPE_CHECKING:
        offs: int
        size: int
        next: CtypesPointer[ggml_object]
        type: int
        padding: bytes


ggml_object._fields_ = [
    ("offs", ctypes.c_size_t),
    ("size", ctypes.c_size_t),
    ("next", ctypes.POINTER(ggml_object)),
    ("type", ctypes.c_int),
    ("padding", ctypes.c_char * 4),
]

ggml_object_p: TypeAlias = "ctypes._Pointer[ggml_object]"  # type: ignore

GGML_OBJECT_SIZE = ctypes.sizeof(ggml_object)


# // n-dimensional tensor
# struct ggml_tensor {
#     enum ggml_type         type;
#     GGML_DEPRECATED(enum ggml_backend_type backend, "use the buffer type to find the storage location of the tensor");

#     struct ggml_backend_buffer * buffer;

#     int64_t ne[GGML_MAX_DIMS]; // number of elements
#     size_t  nb[GGML_MAX_DIMS]; // stride in bytes:
#                                // nb[0] = ggml_type_size(type)
#                                // nb[1] = nb[0]   * (ne[0] / ggml_blck_size(type)) + padding
#                                // nb[i] = nb[i-1] * ne[i-1]

#     // compute data
#     enum ggml_op op;

#     // op params - allocated as int32_t for alignment
#     int32_t op_params[GGML_MAX_OP_PARAMS / sizeof(int32_t)];

#     int32_t flags;

#     struct ggml_tensor * src[GGML_MAX_SRC];

#     // source tensor and offset for views
#     struct ggml_tensor * view_src;
#     size_t               view_offs;

#     void * data;

#     char name[GGML_MAX_NAME];

#     void * extra; // extra things e.g. for ggml-cuda.cu


#     char padding[8];
# };
class ggml_tensor(ctypes.Structure):
    """n-dimensional tensor

    Attributes:
        type (int): ggml_type
        buffer (ctypes.pointer[ggml_backend_buffer]): pointer to backend buffer
        ne (ctypes.Array[ctypes.c_int64]): number of elements in each dimension
        nb (ctypes.Array[ctypes.c_size_t]): stride in bytes for each dimension
        op (int): ggml operation
        op_params (ctypes.Array[ctypes.c_int32]): `GGML_MAX_OP_PARAMS`-length array of operation parameters
        flags (int): tensor flags
        src (ctypes.Array[ggml_tensor_p]): `GGML_MAX_SRC`-length array of source tensors
        view_src (ggml_tensor_p): pointer to tensor if this tensor is a view, None if the tensor is not a view
        view_offs (ctypes.c_size_t): offset into the data pointer of the view tensor
        data (ctypes.c_void_p): reference to raw tensor data
        name (bytes): name of tensor
        extra (ctypes.c_void_p): extra data (e.g. for CUDA)
    """

    if TYPE_CHECKING:
        type: int
        buffer: Optional[ctypes.c_void_p]
        ne: CtypesArray[ctypes.c_int64]
        nb: CtypesArray[ctypes.c_size_t]
        op: int
        op_params: CtypesArray[ctypes.c_int32]
        flags: int
        src: CtypesArray[ggml_tensor_p]
        view_src: CtypesPointer[ggml_tensor]
        view_offs: int
        data: Optional[ctypes.c_void_p]
        name: bytes
        extra: Optional[ctypes.c_void_p]


ggml_tensor._fields_ = [
    ("type", ctypes.c_int),
    ("buffer", ctypes.c_void_p),
    ("ne", ctypes.c_int64 * GGML_MAX_DIMS),
    ("nb", ctypes.c_size_t * GGML_MAX_DIMS),
    ("op", ctypes.c_int),
    (
        "op_params",
        ctypes.c_int32 * (GGML_MAX_OP_PARAMS // ctypes.sizeof(ctypes.c_int32)),
    ),
    ("flags", ctypes.c_int),
    ("src", ctypes.POINTER(ggml_tensor) * GGML_MAX_SRC),
    ("view_src", ctypes.POINTER(ggml_tensor)),
    ("view_offs", ctypes.c_size_t),
    ("data", ctypes.c_void_p),
    ("name", ctypes.c_char * GGML_MAX_NAME),
    ("extra", ctypes.c_void_p),
    ("padding", ctypes.c_char * 8),
]

GGML_TENSOR_SIZE = ctypes.sizeof(ggml_tensor)

ggml_tensor_p: TypeAlias = "ctypes._Pointer[ggml_tensor]"  # type: ignore
"""ctypes pointer to a [ggml_tensor][ggml.ggml_tensor]

Can be dereferenced to a [ggml_tensor][ggml.ggml_tensor] object using
the `.contents` attribute."""


# // Abort callback
# // If not NULL, called before ggml computation
# // If it returns true, the computation is aborted
# typedef bool (*ggml_abort_callback)(void * data);
ggml_abort_callback = ctypes.CFUNCTYPE(ctypes.c_bool, ctypes.c_void_p)


# // the compute plan that needs to be prepared for ggml_graph_compute()
# // since https://github.com/ggerganov/ggml/issues/287
# struct ggml_cplan {
#     size_t    work_size; // size of work buffer, calculated by `ggml_graph_plan()`
#     uint8_t * work_data; // work buffer, to be allocated by caller before calling to `ggml_graph_compute()`

#     int n_threads;
#     struct ggml_threadpool * threadpool;


#     // abort ggml_graph_compute when true
#     ggml_abort_callback abort_callback;
#     void *              abort_callback_data;
#
#     // use only reference implementations
#     bool use_ref;
# };
class ggml_cplan(ctypes.Structure):
    """Compute plan for a ggml computation graph

    Attributes:
        work_size (int): size of work buffer
        work_data (ctypes.pointer[ctypes.c_uint8]): work buffer
        n_threads (int): number of threads
        threadpool (ctypes.c_void_p): optional ggml_threadpool pointer
        abort_callback (ggml_abort_callback): abort callback
        abort_callback_data (ctypes.c_void_p): abort callback data
        use_ref (bool): use only reference implementations
    """

    if TYPE_CHECKING:
        work_size: int
        work_data: CtypesPointer[ctypes.c_uint8]
        n_threads: int
        threadpool: Optional[ctypes.c_void_p]
        abort_callback: Callable[[ctypes.c_void_p], bool]
        abort_callback_data: Optional[ctypes.c_void_p]
        use_ref: bool

    _fields_ = [
        ("work_size", ctypes.c_size_t),
        ("work_data", ctypes.POINTER(ctypes.c_uint8)),
        ("n_threads", ctypes.c_int),
        ("threadpool", ctypes.c_void_p),
        (
            "abort_callback",
            ggml_abort_callback,
        ),
        ("abort_callback_data", ctypes.c_void_p),
        ("use_ref", ctypes.c_bool),
    ]


GGML_CPLAN_SIZE = ctypes.sizeof(ggml_cplan)

ggml_cplan_p: TypeAlias = "ctypes._Pointer[ggml_cplan]"  # type: ignore
"""ctypes pointer to a [ggml_cplan][ggml.ggml_cplan]

Can be dereferenced to a [ggml_cplan][ggml.ggml_cplan] object using
the `.contents` attribute."""


ggml_threadpool_t = NewType("ggml_threadpool_t", int)
ggml_threadpool_t_ctypes: TypeAlias = ctypes.c_void_p


# struct ggml_threadpool_params {
#     bool cpumask[GGML_MAX_N_THREADS];
#     int n_threads;
#     enum ggml_sched_priority prio;
#     uint32_t poll;
#     bool strict_cpu;
#     bool paused;
# };
class ggml_threadpool_params(ctypes.Structure):
    if TYPE_CHECKING:
        cpumask: CtypesArray[ctypes.c_bool]
        n_threads: int
        prio: int
        poll: int
        strict_cpu: bool
        paused: bool

    _fields_ = [
        ("cpumask", ctypes.c_bool * GGML_MAX_N_THREADS),
        ("n_threads", ctypes.c_int),
        ("prio", ctypes.c_int),
        ("poll", ctypes.c_uint32),
        ("strict_cpu", ctypes.c_bool),
        ("paused", ctypes.c_bool),
    ]


# GGML_API struct ggml_threadpool_params ggml_threadpool_params_default(int n_threads);
@ggml_function("ggml_threadpool_params_default", [ctypes.c_int], ggml_threadpool_params)
def ggml_threadpool_params_default(n_threads: Union[ctypes.c_int, int]) -> ggml_threadpool_params:
    ...


# GGML_API void ggml_threadpool_params_init(struct ggml_threadpool_params * p, int n_threads);
@ggml_function("ggml_threadpool_params_init", [ctypes.POINTER(ggml_threadpool_params), ctypes.c_int], None)
def ggml_threadpool_params_init(p: CtypesPointer[ggml_threadpool_params], n_threads: Union[ctypes.c_int, int]):
    ...


# GGML_API bool ggml_threadpool_params_match(const struct ggml_threadpool_params * p0, const struct ggml_threadpool_params * p1);
@ggml_function("ggml_threadpool_params_match", [ctypes.POINTER(ggml_threadpool_params), ctypes.POINTER(ggml_threadpool_params)], ctypes.c_bool)
def ggml_threadpool_params_match(
    p0: CtypesPointer[ggml_threadpool_params],
    p1: CtypesPointer[ggml_threadpool_params],
) -> bool:
    ...


# GGML_BACKEND_API struct ggml_threadpool * ggml_threadpool_new(struct ggml_threadpool_params * params);
@ggml_function("ggml_threadpool_new", [ctypes.POINTER(ggml_threadpool_params)], ggml_threadpool_t_ctypes)
def ggml_threadpool_new(params: CtypesPointer[ggml_threadpool_params]) -> Optional[ggml_threadpool_t]:
    ...


# GGML_BACKEND_API void ggml_threadpool_free(struct ggml_threadpool * threadpool);
@ggml_function("ggml_threadpool_free", [ggml_threadpool_t_ctypes], None)
def ggml_threadpool_free(threadpool: Union[ggml_threadpool_t, int]):
    ...


# GGML_BACKEND_API int ggml_threadpool_get_n_threads(struct ggml_threadpool * threadpool);
@ggml_function(
    "ggml_threadpool_get_n_threads",
    [ggml_threadpool_t_ctypes],
    ctypes.c_int,
    enabled=hasattr(lib, "ggml_threadpool_get_n_threads"),
)
def ggml_threadpool_get_n_threads(threadpool: Union[ggml_threadpool_t, int]) -> int:
    ...


# GGML_BACKEND_API void ggml_threadpool_pause(struct ggml_threadpool * threadpool);
@ggml_function("ggml_threadpool_pause", [ggml_threadpool_t_ctypes], None)
def ggml_threadpool_pause(threadpool: Union[ggml_threadpool_t, int]):
    ...


# GGML_BACKEND_API void ggml_threadpool_resume(struct ggml_threadpool * threadpool);
@ggml_function("ggml_threadpool_resume", [ggml_threadpool_t_ctypes], None)
def ggml_threadpool_resume(threadpool: Union[ggml_threadpool_t, int]):
    ...


# enum ggml_cgraph_eval_order {
#     GGML_CGRAPH_EVAL_ORDER_LEFT_TO_RIGHT = 0,
#     GGML_CGRAPH_EVAL_ORDER_RIGHT_TO_LEFT,
#     GGML_CGRAPH_EVAL_ORDER_COUNT
# };
GGML_CGRAPH_EVAL_ORDER_LEFT_TO_RIGHT = 0
GGML_CGRAPH_EVAL_ORDER_RIGHT_TO_LEFT = 1
GGML_CGRAPH_EVAL_ORDER_COUNT = 2


# struct ggml_hash_set {
#     size_t size;
#     ggml_bitset_t * used;
#     struct ggml_tensor ** keys;
# };
class ggml_hash_set(ctypes.Structure):
    """ggml hash set
    
    Attributes:
        size (int): size
        used (ctypes.pointer[ctypes.c_uint32]): bitset of used entries
        keys (ctypes.Array[ctypes.POINTER(ggml_tensor)]): array of tensor keys"""

    if TYPE_CHECKING:
        size: int
        used: CtypesPointer[ctypes.c_uint32]
        keys: CtypesArray[CtypesPointer[ggml_tensor]]

    _fields_ = [
        ("size", ctypes.c_size_t),
        ("used", ctypes.POINTER(ctypes.c_uint32)),
        ("keys", ctypes.POINTER(ctypes.POINTER(ggml_tensor))),
    ]


# // computation graph
# struct ggml_cgraph {
#     int size;
#     int n_nodes;
#     int n_leafs;

#     struct ggml_tensor ** nodes;
#     struct ggml_tensor ** grads;
#     struct ggml_tensor ** grad_accs;
#     struct ggml_tensor ** leafs;
#     int32_t             * use_counts;

#     struct ggml_hash_set visited_hash_set;

#     enum ggml_cgraph_eval_order order;


#     uint64_t uid;
# };
class ggml_cgraph(ctypes.Structure):
    """ggml computation graph

    Attributes:
        size (int): size
        n_nodes (int): number of nodes
        n_leafs (int): number of leafs
        nodes (ctypes.Array[ggml_tensor_p]): `n_nodes`-length array of compute tensors
        grads (ctypes.Array[ggml_tensor_p]): `n_nodes`-length array of gradient tensors
        grad_accs (ctypes.Array[ggml_tensor_p]): `n_nodes`-length array of gradient accumulators
        leafs (ctypes.Array[ggml_tensor_p]): `n_leafs`-length array of parameter tensors
        use_counts (ctypes.Array[ctypes.c_int32]): tensor use counts indexed by hash slot
        visited_hash_set (ggml_hash_set): hash set of visited tensors
        order (int): evaluation order
        uid (int): optional graph identifier"""
    
    if TYPE_CHECKING:
        size: int
        n_nodes: int
        n_leafs: int
        nodes: CtypesArray[CtypesPointer[ggml_tensor]]
        grads: CtypesArray[CtypesPointer[ggml_tensor]]
        grad_accs: CtypesArray[CtypesPointer[ggml_tensor]]
        leafs: CtypesArray[CtypesPointer[ggml_tensor]]
        use_counts: CtypesPointer[ctypes.c_int32]
        visited_hash_set: ggml_hash_set
        order: int
        uid: int

    _fields_ = [
        ("size", ctypes.c_int),
        ("n_nodes", ctypes.c_int),
        ("n_leafs", ctypes.c_int),
        ("nodes", ctypes.POINTER(ctypes.POINTER(ggml_tensor))),
        ("grads", ctypes.POINTER(ctypes.POINTER(ggml_tensor))),
        ("grad_accs", ctypes.POINTER(ctypes.POINTER(ggml_tensor))),
        ("leafs", ctypes.POINTER(ctypes.POINTER(ggml_tensor))),
        ("use_counts", ctypes.POINTER(ctypes.c_int32)),
        ("visited_hash_set", ggml_hash_set),
        ("order", ctypes.c_int),
        ("uid", ctypes.c_uint64),
    ]


ggml_cgraph_p: TypeAlias = "ctypes._Pointer[ggml_cgraph]"  # type: ignore
"""ctypes pointer to a [ggml_cgraph][ggml.ggml_cgraph]

Can be dereferenced to a [ggml_cgraph][ggml.ggml_cgraph] object using
the `.contents` attribute."""


# struct ggml_scratch {
#     size_t offs;
#     size_t size;
#     void * data;
# };
class ggml_scratch(ctypes.Structure):
    """Scratch memory for ggml
    
    Attributes:
        offs (int): offset
        size (int): size
        data (ctypes.c_void_p): data pointer"""
    
    if TYPE_CHECKING:
        offs: int
        size: int
        data: Optional[ctypes.c_void_p]

    _fields_ = [
        ("offs", ctypes.c_size_t),
        ("size", ctypes.c_size_t),
        ("data", ctypes.c_void_p),
    ]


# struct ggml_init_params {
#     // memory pool
#     size_t mem_size;   // bytes
#     void * mem_buffer; // if NULL, memory will be allocated internally
#     bool   no_alloc;   // don't allocate memory for the tensor data
# };
class ggml_init_params(ctypes.Structure):
    """Initialization parameters for a ggml context

    **NOTE**: Reference counting does not cross into ggml, if you allocate a memory buffer
    in python using ctypes Arrays or a numpy array, you must keep a reference to it until
    you free the ggml context otherwise you will encounter a segmentation fault.

    Attributes:
        mem_size (int): size of memory pool in bytes
        mem_buffer (ctypes.c_void_p): pointer to memory pool, if None, memory will be allocated internally
        no_alloc (bool): don't allocate memory for tensor data
    """

    if TYPE_CHECKING:
        mem_size: int
        mem_buffer: Optional[ctypes.c_void_p]
        no_alloc: bool

    _fields_ = [
        ("mem_size", ctypes.c_size_t),
        ("mem_buffer", ctypes.c_void_p),
        ("no_alloc", ctypes.c_bool),
    ]


# // compute types

# // NOTE: the INIT or FINALIZE pass is not scheduled unless explicitly enabled.
# // This behavior was changed since https://github.com/ggerganov/llama.cpp/pull/1995.
# enum ggml_task_type {
#     GGML_TASK_TYPE_INIT = 0,
#     GGML_TASK_TYPE_COMPUTE,
#     GGML_TASK_TYPE_FINALIZE,
# };
GGML_TASK_TYPE_INIT = 0
GGML_TASK_TYPE_COMPUTE = 1
GGML_TASK_TYPE_FINALIZE = 2

# struct ggml_compute_params {
#     enum ggml_task_type type;

#     // ith = thread index, nth = number of threads
#     int ith, nth;


#     // work buffer for all threads
#     size_t wsize;
#     void * wdata;
# };
class ggml_compute_params(ctypes.Structure):
    """Compute parameters for ggml
    
    Attributes:
        type (int): task type
        ith (int): thread index
        nth (int): number of threads
        wsize (int): work buffer size
        wdata (ctypes.c_void_p): work buffer data"""

    if TYPE_CHECKING:
        type: int
        ith: int
        nth: int
        wsize: int
        wdata: Optional[ctypes.c_void_p]

    _fields_ = [
        ("type", ctypes.c_int),
        ("ith", ctypes.c_int),
        ("nth", ctypes.c_int),
        ("wsize", ctypes.c_size_t),
        ("wdata", ctypes.c_void_p),
    ]


ggml_compute_params_p: TypeAlias = "ctypes._Pointer[ggml_compute_params]"  # type: ignore


# // numa strategies
# enum ggml_numa_strategy {
#     GGML_NUMA_STRATEGY_DISABLED   = 0,
#     GGML_NUMA_STRATEGY_DISTRIBUTE = 1,
#     GGML_NUMA_STRATEGY_ISOLATE    = 2,
#     GGML_NUMA_STRATEGY_NUMACTL    = 3,
#     GGML_NUMA_STRATEGY_MIRROR     = 4,
#     GGML_NUMA_STRATEGY_COUNT
# };
GGML_NUMA_STRATEGY_DISABLED = 0
GGML_NUMA_STRATEGY_DISTRIBUTE = 1
GGML_NUMA_STRATEGY_ISOLATE = 2
GGML_NUMA_STRATEGY_NUMACTL = 3
GGML_NUMA_STRATEGY_MIRROR = 4
GGML_NUMA_STRATEGY_COUNT = 5


# //
# // GUID
# //

# // GUID types
# typedef uint8_t ggml_guid[16];
# typedef ggml_guid * ggml_guid_t;
if TYPE_CHECKING:
    ggml_guid = CtypesArray[ctypes.c_uint8]
    ggml_guid_t = CtypesPointer[ggml_guid]


ggml_guid_ctypes = ctypes.c_uint8 * 16
ggml_guid_t_ctypes = ctypes.POINTER(ggml_guid_ctypes)


# GGML_API bool ggml_guid_matches(ggml_guid_t guid_a, ggml_guid_t guid_b);
# TODO: Verify type signature
@ggml_function("ggml_guid_matches", [ggml_guid_t_ctypes, ggml_guid_t_ctypes], ctypes.c_bool)
def ggml_guid_matches(guid_a: ggml_guid_t, guid_b: ggml_guid_t, /) -> bool:
    ...


# // misc


# GGML_API const char * ggml_version(void);
@ggml_function("ggml_version", [], ctypes.c_char_p)
def ggml_version() -> bytes:
    ...


# GGML_API const char * ggml_commit(void);
@ggml_function("ggml_commit", [], ctypes.c_char_p)
def ggml_commit() -> bytes:
    ...


# GGML_API void    ggml_time_init(void); // call this once at the beginning of the program
@ggml_function("ggml_time_init", [], None)
def ggml_time_init():
    ...


# GGML_API int64_t ggml_time_ms(void);
@ggml_function("ggml_time_ms", [], ctypes.c_int64)
def ggml_time_ms() -> int:
    ...


# GGML_API int64_t ggml_time_us(void);
@ggml_function("ggml_time_us", [], ctypes.c_int64)
def ggml_time_us() -> int:
    ...


# GGML_API int64_t ggml_cycles(void);
@ggml_function("ggml_cycles", [], ctypes.c_int64)
def ggml_cycles() -> int:
    ...


# GGML_API int64_t ggml_cycles_per_ms(void);
@ggml_function("ggml_cycles_per_ms", [], ctypes.c_int64)
def ggml_cycles_per_ms() -> int:
    ...


# GGML_API void    ggml_print_backtrace(void);
@ggml_function("ggml_print_backtrace", [], None, enabled=hasattr(lib, "ggml_print_backtrace"))
def ggml_print_backtrace():
    ...


# // accepts a UTF-8 path, even on Windows
# GGML_API FILE *  ggml_fopen(const char * fname, const char * mode);
@ggml_function("ggml_fopen", [ctypes.c_char_p, ctypes.c_char_p], ctypes.c_void_p)
def ggml_fopen(fname: bytes, mode: bytes, /) -> ctypes.c_void_p:
    ...


# GGML_API void    ggml_numa_init(enum ggml_numa_strategy numa); // call once for better performance on NUMA systems
@ggml_function("ggml_numa_init", [ctypes.c_int], None)
def ggml_numa_init(numa: Union[ctypes.c_int, int], /):
    ...


# GGML_API bool    ggml_is_numa(void); // true if init detected that system has >1 NUMA node
@ggml_function("ggml_is_numa", [], ctypes.c_bool)
def ggml_is_numa() -> bool:
    ...


# GGML_API void    ggml_print_object (const struct ggml_object * obj);
@ggml_function("ggml_print_object", [ctypes.POINTER(ggml_object)], None)
def ggml_print_object(obj: ggml_object_p, /):
    ...


# GGML_API void    ggml_print_objects(const struct ggml_context * ctx);
@ggml_function("ggml_print_objects", [ggml_context_p_ctypes], None)
def ggml_print_objects(ctx: ggml_context_p, /):
    ...


# GGML_API GGML_CALL int64_t ggml_nelements   (const struct ggml_tensor * tensor);
@ggml_function("ggml_nelements", [ctypes.POINTER(ggml_tensor)], ctypes.c_int64)
def ggml_nelements(tensor: ggml_tensor_p, /) -> int:
    """Get the number of elements in a tensor

    Parameters:
        tensor: tensor

    Returns:
        number of elements"""
    ...


# GGML_API GGML_CALL int64_t ggml_nrows       (const struct ggml_tensor * tensor);
@ggml_function("ggml_nrows", [ctypes.POINTER(ggml_tensor)], ctypes.c_int64)
def ggml_nrows(tensor: ggml_tensor_p, /) -> int:
    """Get the number of rows in a tensor

    Parameters:
        tensor: tensor

    Returns:
        number of rows"""
    ...


# GGML_API GGML_CALL size_t  ggml_nbytes      (const struct ggml_tensor * tensor);
@ggml_function("ggml_nbytes", [ctypes.POINTER(ggml_tensor)], ctypes.c_size_t)
def ggml_nbytes(tensor: ggml_tensor_p, /) -> int:
    """Get the number of bytes required to store tensor data

    Parameters:
        tensor: tensor

    Returns:
        number of bytes"""
    ...


# GGML_API           size_t  ggml_nbytes_pad  (const struct ggml_tensor * tensor); // same as ggml_nbytes() but padded to GGML_MEM_ALIGN
@ggml_function("ggml_nbytes_pad", [ctypes.POINTER(ggml_tensor)], ctypes.c_size_t)
def ggml_nbytes_pad(tensor: ggml_tensor_p, /) -> int:
    """Get the number of bytes required to store tensor data, padded to GGML_MEM_ALIGN

    Parameters:
        tensor: tensor

    Returns:
        number of bytes"""
    ...


# GGML_API GGML_CALL int64_t ggml_blck_size(enum ggml_type type);
@ggml_function("ggml_blck_size", [ctypes.c_int], ctypes.c_int64)
def ggml_blck_size(type: Union[ctypes.c_int, int], /) -> int:
    ...


# GGML_API GGML_CALL size_t ggml_type_size(enum ggml_type type);             // size in bytes for all elements in a block
@ggml_function("ggml_type_size", [ctypes.c_int], ctypes.c_size_t)
def ggml_type_size(type: Union[ctypes.c_int, int], /) -> int:
    ...


# GGML_API GGML_CALL size_t ggml_row_size (enum ggml_type type, int64_t ne); // size in bytes for all elements in a row
@ggml_function("ggml_row_size", [ctypes.c_int, ctypes.c_int64], ctypes.c_size_t)
def ggml_row_size(type: Union[ctypes.c_int, int], ne: int, /) -> int:
    ...


# GGML_DEPRECATED(
# GGML_API double ggml_type_sizef(enum ggml_type type), // ggml_type_size()/ggml_blck_size() as float
# "use ggml_row_size() instead");
@ggml_function("ggml_type_sizef", [ctypes.c_int], ctypes.c_double)
def ggml_type_sizef(type: Union[ctypes.c_int, int], /) -> float:
    ...


# GGML_API GGML_CALL const char * ggml_type_name(enum ggml_type type);
@ggml_function("ggml_type_name", [ctypes.c_int], ctypes.c_char_p)
def ggml_type_name(type: Union[ctypes.c_int, int], /) -> bytes:
    ...


# GGML_API GGML_CALL const char * ggml_op_name  (enum ggml_op   op);
@ggml_function("ggml_op_name", [ctypes.c_int], ctypes.c_char_p)
def ggml_op_name(op: Union[ctypes.c_int, int], /) -> bytes:
    ...


# GGML_API           const char * ggml_op_symbol(enum ggml_op   op);
@ggml_function("ggml_op_symbol", [ctypes.c_int], ctypes.c_char_p)
def ggml_op_symbol(op: Union[ctypes.c_int, int], /) -> bytes:
    ...


# GGML_API           const char * ggml_unary_op_name(enum ggml_unary_op op);
@ggml_function("ggml_unary_op_name", [ctypes.c_int], ctypes.c_char_p)
def ggml_unary_op_name(op: Union[ctypes.c_int, int], /) -> bytes:
    ...


# GGML_API           const char * ggml_glu_op_name(enum ggml_glu_op op);
@ggml_function("ggml_glu_op_name", [ctypes.c_int], ctypes.c_char_p)
def ggml_glu_op_name(op: Union[ctypes.c_int, int], /) -> bytes:
    ...


# GGML_API GGML_CALL const char * ggml_op_desc(const struct ggml_tensor * t); // unary or op name
@ggml_function("ggml_op_desc", [ctypes.POINTER(ggml_tensor)], ctypes.c_char_p)
def ggml_op_desc(t: ggml_tensor_p, /) -> bytes:
    ...


# GGML_API GGML_CALL size_t  ggml_element_size(const struct ggml_tensor * tensor);
@ggml_function("ggml_element_size", [ctypes.POINTER(ggml_tensor)], ctypes.c_size_t)
def ggml_element_size(tensor: ggml_tensor_p, /) -> int:
    ...


# GGML_API GGML_CALL bool    ggml_is_quantized(enum ggml_type type);
@ggml_function("ggml_is_quantized", [ctypes.c_int], ctypes.c_bool)
def ggml_is_quantized(type: Union[ctypes.c_int, int], /) -> bool:
    ...


# // TODO: temporary until model loading of ggml examples is refactored
# GGML_API enum ggml_type ggml_ftype_to_ggml_type(enum ggml_ftype ftype);
@ggml_function("ggml_ftype_to_ggml_type", [ctypes.c_int], ctypes.c_int)
def ggml_ftype_to_ggml_type(ftype: Union[ctypes.c_int, int], /) -> int:
    ...


# GGML_API GGML_CALL bool ggml_is_transposed(const struct ggml_tensor * tensor);
@ggml_function("ggml_is_transposed", [ctypes.POINTER(ggml_tensor)], ctypes.c_bool)
def ggml_is_transposed(tensor: ggml_tensor_p, /) -> bool:
    """Check if a tensor is transposed

    Parameters:
        tensor: tensor

    Returns:
        True if tensor is transposed else False"""
    ...


# GGML_API GGML_CALL bool ggml_is_contiguous(const struct ggml_tensor * tensor);
@ggml_function("ggml_is_contiguous", [ctypes.POINTER(ggml_tensor)], ctypes.c_bool)
def ggml_is_contiguous(tensor: ggml_tensor_p, /) -> bool:
    """Check if a tensor is contiguous

    Parameters:
        tensor: tensor

    Returns:
        True if tensor is contiguous else False"""
    ...


# GGML_API GGML_CALL bool ggml_is_contiguous_0(const struct ggml_tensor * tensor);
@ggml_function("ggml_is_contiguous_0", [ctypes.POINTER(ggml_tensor)], ctypes.c_bool)
def ggml_is_contiguous_0(tensor: ggml_tensor_p, /) -> bool:
    """Check if a tensor is contiguous (same as ggml_is_contiguous)

    Parameters:
        tensor: tensor

    Returns:
        True if tensor is contiguous else False"""
    ...


# GGML_API GGML_CALL bool ggml_is_contiguous_1(const struct ggml_tensor * tensor);
@ggml_function("ggml_is_contiguous_1", [ctypes.POINTER(ggml_tensor)], ctypes.c_bool)
def ggml_is_contiguous_1(tensor: ggml_tensor_p, /) -> bool:
    """Check if a tensor is contiguous for dimensions >= 1

    Parameters:
        tensor: tensor

    Returns:
        True if tensor is contiguous for dims >= 1 else False"""
    ...


# GGML_API GGML_CALL bool ggml_is_contiguous_2(const struct ggml_tensor * tensor);
@ggml_function("ggml_is_contiguous_2", [ctypes.POINTER(ggml_tensor)], ctypes.c_bool)
def ggml_is_contiguous_2(tensor: ggml_tensor_p, /) -> bool:
    """Check if a tensor is contiguous for dimensions >= 2

    Parameters:
        tensor: tensor

    Returns:
        True if tensor is contiguous for dims >= 2 else False"""
    ...


# GGML_API bool ggml_is_contiguously_allocated(const struct ggml_tensor * tensor);
@ggml_function("ggml_is_contiguously_allocated", [ctypes.POINTER(ggml_tensor)], ctypes.c_bool)
def ggml_is_contiguously_allocated(tensor: ggml_tensor_p, /) -> bool:
    """Check if a tensor is allocated as one contiguous block"""
    ...


# GGML_API bool ggml_is_contiguous_channels(const struct ggml_tensor * tensor);
@ggml_function("ggml_is_contiguous_channels", [ctypes.POINTER(ggml_tensor)], ctypes.c_bool)
def ggml_is_contiguous_channels(tensor: ggml_tensor_p, /) -> bool:
    """Check if a tensor is stored as contiguous channels"""
    ...


# GGML_API bool ggml_is_contiguous_rows(const struct ggml_tensor * tensor);
@ggml_function("ggml_is_contiguous_rows", [ctypes.POINTER(ggml_tensor)], ctypes.c_bool)
def ggml_is_contiguous_rows(tensor: ggml_tensor_p, /) -> bool:
    """Check if a tensor has contiguous rows"""
    ...


# GGML_API GGML_CALL bool ggml_is_permuted  (const struct ggml_tensor * tensor);
@ggml_function("ggml_is_permuted", [ctypes.POINTER(ggml_tensor)], ctypes.c_bool)
def ggml_is_permuted(tensor: ggml_tensor_p, /) -> bool:
    """Check if a tensor is permuted

    Parameters:
        tensor: tensor

    Returns:
        True if tensor is permuted else False"""
    ...


# GGML_API GGML_CALL bool ggml_is_empty     (const struct ggml_tensor * tensor);
@ggml_function("ggml_is_empty", [ctypes.POINTER(ggml_tensor)], ctypes.c_bool)
def ggml_is_empty(tensor: ggml_tensor_p, /) -> bool:
    ...


# GGML_API bool ggml_is_view      (const struct ggml_tensor * tensor);
@ggml_function("ggml_is_view", [ctypes.POINTER(ggml_tensor)], ctypes.c_bool)
def ggml_is_view(tensor: ggml_tensor_p, /) -> bool:
    """Check if a tensor is a view"""
    ...


# GGML_API           bool ggml_is_scalar    (const struct ggml_tensor * tensor);
@ggml_function("ggml_is_scalar", [ctypes.POINTER(ggml_tensor)], ctypes.c_bool)
def ggml_is_scalar(tensor: ggml_tensor_p, /) -> bool:
    """Check if a tensor is a scalar"""
    ...


# GGML_API           bool ggml_is_vector    (const struct ggml_tensor * tensor);
@ggml_function("ggml_is_vector", [ctypes.POINTER(ggml_tensor)], ctypes.c_bool)
def ggml_is_vector(tensor: ggml_tensor_p, /) -> bool:
    """Check if a tensor is a vector"""
    ...


# GGML_API           bool ggml_is_matrix    (const struct ggml_tensor * tensor);
@ggml_function("ggml_is_matrix", [ctypes.POINTER(ggml_tensor)], ctypes.c_bool)
def ggml_is_matrix(tensor: ggml_tensor_p, /) -> bool:
    """Check if a tensor is a matrix"""
    ...


# GGML_API           bool ggml_is_3d        (const struct ggml_tensor * tensor);
@ggml_function("ggml_is_3d", [ctypes.POINTER(ggml_tensor)], ctypes.c_bool)
def ggml_is_3d(tensor: ggml_tensor_p, /) -> bool:
    """Check if a tensor is 3d"""
    ...


# GGML_API           int  ggml_n_dims       (const struct ggml_tensor * tensor); // returns 1 for scalars
@ggml_function("ggml_n_dims", [ctypes.POINTER(ggml_tensor)], ctypes.c_int)
def ggml_n_dims(tensor: ggml_tensor_p, /) -> int:
    """Get the number of dimensions in a tensor"""
    ...


# GGML_API bool ggml_are_same_shape (const struct ggml_tensor * t0, const struct ggml_tensor * t1);
@ggml_function(
    "ggml_are_same_shape",
    [ctypes.POINTER(ggml_tensor), ctypes.POINTER(ggml_tensor)],
    ctypes.c_bool,
)
def ggml_are_same_shape(t0: ggml_tensor_p, t1: ggml_tensor_p, /) -> bool:
    """Check if two tensors have the same shape

    Parameters:
        t0: tensor 0
        t1: tensor 1

    Returns:
        True if tensors have the same shape else False"""
    ...


# GGML_API bool ggml_are_same_stride(const struct ggml_tensor * t0, const struct ggml_tensor * t1);
@ggml_function(
    "ggml_are_same_stride",
    [ctypes.POINTER(ggml_tensor), ctypes.POINTER(ggml_tensor)],
    ctypes.c_bool,
)
def ggml_are_same_stride(t0: ggml_tensor_p, t1: ggml_tensor_p, /) -> bool:
    """Check if two tensors have the same stride

    Parameters:
        t0: tensor 0
        t1: tensor 1

    Returns:
        True if tensors have the same stride else False"""
    ...


# GGML_API bool ggml_can_repeat(const struct ggml_tensor * t0, const struct ggml_tensor * t1);
@ggml_function(
    "ggml_can_repeat",
    [ctypes.POINTER(ggml_tensor), ctypes.POINTER(ggml_tensor)],
    ctypes.c_bool,
)
def ggml_can_repeat(t0: ggml_tensor_p, t1: ggml_tensor_p, /) -> bool:
    ...


# // use this to compute the memory overhead of a tensor
# GGML_API size_t ggml_tensor_overhead(void);
@ggml_function("ggml_tensor_overhead", [], ctypes.c_size_t)
def ggml_tensor_overhead() -> int:
    """Overhead required for a tensor struct in bytes

    Returns:
        size of tensor struct in bytes"""
    ...


# GGML_API bool ggml_validate_row_data(enum ggml_type type, const void * data, size_t nbytes);
@ggml_function("ggml_validate_row_data", [ctypes.c_int, ctypes.c_void_p, ctypes.c_size_t], ctypes.c_bool)
def ggml_validate_row_data(type: Union[ctypes.c_int, int], data: ctypes.c_void_p, nbytes: int, /) -> bool:
    ...


# // main


# GGML_API struct ggml_context * ggml_init(struct ggml_init_params params);
@ggml_function("ggml_init", [ggml_init_params], ggml_context_p_ctypes)
def ggml_init(params: ggml_init_params, /) -> Optional[ggml_context_p]:
    """Instantiate a new ggml context with params.

    You must call `ggml_free()` to free the context.

    Parameters:
        params: ggml init params

    Returns:
        Pointer to ggml_context or None if failed to initialize context."""
    ...


# GGML_API void ggml_reset(struct ggml_context * ctx);
@ggml_function("ggml_reset", [ggml_context_p_ctypes], None)
def ggml_reset(ctx: ggml_context_p, /):
    """Reset the ggml context.

    Parameters:
        ctx: ggml context"""
    ...


# GGML_API void                  ggml_free(struct ggml_context * ctx);
@ggml_function("ggml_free", [ggml_context_p_ctypes], None)
def ggml_free(ctx: ggml_context_p, /):
    """Free the ggml context.

    Parameters:
        ctx: ggml context"""
    ...


# GGML_API size_t  ggml_used_mem(const struct ggml_context * ctx);
@ggml_function("ggml_used_mem", [ggml_context_p_ctypes], ctypes.c_size_t)
def ggml_used_mem(ctx: ggml_context_p, /) -> int:
    """Return the amount of memory used by the ggml context in bytes.

    Parameters:
        ctx: ggml context

    Returns:
        amount of memory used in bytes"""
    ...


# GGML_API size_t  ggml_set_scratch(struct ggml_context * ctx, struct ggml_scratch scratch);
@ggml_function(
    "ggml_set_scratch",
    [ggml_context_p_ctypes, ggml_scratch],
    ctypes.c_size_t,
    enabled=hasattr(lib, "ggml_set_scratch"),
)
def ggml_set_scratch(ctx: ggml_context_p, scratch: ggml_scratch, /) -> int:
    """Set the scratch buffer for the ggml context."""
    ...


# GGML_API bool    ggml_get_no_alloc(struct ggml_context * ctx);
@ggml_function("ggml_get_no_alloc", [ggml_context_p_ctypes], ctypes.c_bool)
def ggml_get_no_alloc(ctx: ggml_context_p, /) -> bool:
    """Return the no_alloc flag for the ggml context."""
    ...


# GGML_API void    ggml_set_no_alloc(struct ggml_context * ctx, bool no_alloc);
@ggml_function("ggml_set_no_alloc", [ggml_context_p_ctypes, ctypes.c_bool], None)
def ggml_set_no_alloc(ctx: ggml_context_p, no_alloc: Union[ctypes.c_bool, bool], /):
    """Set the no_alloc flag for the ggml context."""
    ...


# GGML_API void *  ggml_get_mem_buffer     (struct ggml_context * ctx);
@ggml_function("ggml_get_mem_buffer", [ggml_context_p_ctypes], ctypes.c_void_p)
def ggml_get_mem_buffer(ctx: ggml_context_p, /) -> Optional[int]:
    """Return the memory buffer for the ggml context."""
    ...


# GGML_API size_t  ggml_get_mem_size       (struct ggml_context * ctx);
@ggml_function("ggml_get_mem_size", [ggml_context_p_ctypes], ctypes.c_size_t)
def ggml_get_mem_size(ctx: ggml_context_p, /) -> int:
    """Return the size of the memory buffer for the ggml context in bytes."""
    ...


# GGML_API size_t  ggml_get_max_tensor_size(const struct ggml_context * ctx);
@ggml_function("ggml_get_max_tensor_size", [ggml_context_p_ctypes], ctypes.c_size_t)
def ggml_get_max_tensor_size(ctx: ggml_context_p, /) -> int:
    """Return the maximum size of a tensor in bytes."""
    ...


# GGML_API struct ggml_tensor * ggml_new_tensor(
#         struct ggml_context * ctx,
#         enum   ggml_type type,
#         int    n_dims,
#         const int64_t *ne);
@ggml_function(
    "ggml_new_tensor",
    [ggml_context_p_ctypes, ctypes.c_int, ctypes.c_int, ctypes.POINTER(ctypes.c_int64)],
    ctypes.POINTER(ggml_tensor),
)
def ggml_new_tensor(
    ctx: ggml_context_p,
    type: Union[ctypes.c_int, int],
    n_dims: Union[ctypes.c_int, int],
    ne: CtypesArray[ctypes.c_int64],
    /,
) -> ggml_tensor_p:
    """Create a new tensor with the given type, number of dimensions, and number of elements in each dimension.

    Parameters:
        ctx: ggml context
        type: ggml type
        n_dims: number of dimensions
        ne (ctypes.Array[ctypes.c_int64]): number of elements in each dimension (array of length n_dims)

    Returns:
        Pointer to ggml_tensor"""
    ...


# GGML_API struct ggml_tensor * ggml_new_tensor_1d(
#         struct ggml_context * ctx,
#         enum   ggml_type type,
#         int64_t ne0);
@ggml_function(
    "ggml_new_tensor_1d",
    [ggml_context_p_ctypes, ctypes.c_int, ctypes.c_int64],
    ctypes.POINTER(ggml_tensor),
)
def ggml_new_tensor_1d(
    ctx: ggml_context_p,
    type: Union[ctypes.c_int, int],
    ne0: Union[ctypes.c_int64, int],
    /,
) -> ggml_tensor_p:
    """Create a new 1-dimensional tensor with the given type and number of elements.

    Parameters:
        ctx: ggml context
        type: ggml type
        ne0: number of elements in dimension 0

    Returns:
        Pointer to ggml_tensor"""
    ...


# GGML_API struct ggml_tensor * ggml_new_tensor_2d(
#         struct ggml_context * ctx,
#         enum   ggml_type type,
#         int64_t ne0,
#         int64_t ne1);
@ggml_function(
    "ggml_new_tensor_2d",
    [ggml_context_p_ctypes, ctypes.c_int, ctypes.c_int64, ctypes.c_int64],
    ctypes.POINTER(ggml_tensor),
)
def ggml_new_tensor_2d(
    ctx: ggml_context_p,
    type: Union[ctypes.c_int, int],
    ne0: Union[ctypes.c_int64, int],
    ne1: Union[ctypes.c_int64, int],
    /,
) -> ggml_tensor_p:
    """Create a new 2-dimensional tensor with the given type and number of elements in each dimension.

    Parameters:
        ctx: ggml context
        type: ggml type
        ne0: number of elements in dimension 0
        ne1: number of elements in dimension 1

    Returns:
        Pointer to ggml_tensor"""
    ...


# GGML_API struct ggml_tensor * ggml_new_tensor_3d(
#         struct ggml_context * ctx,
#         enum   ggml_type type,
#         int64_t ne0,
#         int64_t ne1,
#         int64_t ne2);
@ggml_function(
    "ggml_new_tensor_3d",
    [
        ggml_context_p_ctypes,
        ctypes.c_int,
        ctypes.c_int64,
        ctypes.c_int64,
        ctypes.c_int64,
    ],
    ctypes.POINTER(ggml_tensor),
)
def ggml_new_tensor_3d(
    ctx: ggml_context_p,
    type: Union[ctypes.c_int, int],
    ne0: Union[ctypes.c_int64, int],
    ne1: Union[ctypes.c_int64, int],
    ne2: Union[ctypes.c_int64, int],
    /,
) -> ggml_tensor_p:
    """Create a new 3-dimensional tensor with the given type and number of elements in each dimension.

    Parameters:
        ctx: ggml context
        type: ggml type
        ne0: number of elements in dimension 0
        ne1: number of elements in dimension 1
        ne2: number of elements in dimension 2

    Returns:
        Pointer to ggml_tensor"""
    ...


# GGML_API struct ggml_tensor * ggml_new_tensor_4d(
#         struct ggml_context * ctx,
#         enum   ggml_type type,
#         int64_t ne0,
#         int64_t ne1,
#         int64_t ne2,
#         int64_t ne3);
@ggml_function(
    "ggml_new_tensor_4d",
    [
        ggml_context_p_ctypes,
        ctypes.c_int,
        ctypes.c_int64,
        ctypes.c_int64,
        ctypes.c_int64,
        ctypes.c_int64,
    ],
    ctypes.POINTER(ggml_tensor),
)
def ggml_new_tensor_4d(
    ctx: ggml_context_p,
    type: Union[ctypes.c_int, int],
    ne0: Union[ctypes.c_int64, int],
    ne1: Union[ctypes.c_int64, int],
    ne2: Union[ctypes.c_int64, int],
    ne3: Union[ctypes.c_int64, int],
    /,
) -> ggml_tensor_p:
    """Create a new 4-dimensional tensor with the given type and number of elements in each dimension.

    Parameters:
        ctx: ggml context
        type: ggml type
        ne0: number of elements in dimension 0
        ne1: number of elements in dimension 1
        ne2: number of elements in dimension 2

    Returns:
        Pointer to ggml_tensor"""
    ...


# GGML_API void * ggml_new_buffer(struct ggml_context * ctx, size_t nbytes);
@ggml_function("ggml_new_buffer", [ggml_context_p_ctypes, ctypes.c_size_t], ctypes.c_void_p)
def ggml_new_buffer(
    ctx: ggml_context_p,
    nbytes: Union[ctypes.c_size_t, int],
    /,
) -> Optional[int]:
    ...


# GGML_API struct ggml_tensor * ggml_new_i32(struct ggml_context * ctx, int32_t value);
@ggml_function(
    "ggml_new_i32", [ggml_context_p_ctypes, ctypes.c_int32], ctypes.POINTER(ggml_tensor)
)
def ggml_new_i32(
    ctx: ggml_context_p, value: Union[ctypes.c_int32, int], /
) -> ggml_tensor_p:
    """Create a 1 element tensor with the given integer value.

    Parameters:
        ctx: ggml context
        value: integer value

    Returns:
        Pointer to ggml_tensor"""
    ...


# GGML_API struct ggml_tensor * ggml_new_f32(struct ggml_context * ctx, float value);
@ggml_function(
    "ggml_new_f32", [ggml_context_p_ctypes, ctypes.c_float], ctypes.POINTER(ggml_tensor)
)
def ggml_new_f32(
    ctx: ggml_context_p, value: Union[ctypes.c_float, float], /
) -> ggml_tensor_p:
    """Create a 1 element tensor with the given float value.

    Parameters:
        ctx: ggml context
        value: float value

    Returns:
        Pointer to ggml_tensor"""
    ...


# GGML_API struct ggml_tensor * ggml_dup_tensor (struct ggml_context * ctx, const struct ggml_tensor * src);
@ggml_function(
    "ggml_dup_tensor",
    [ggml_context_p_ctypes, ctypes.POINTER(ggml_tensor)],
    ctypes.POINTER(ggml_tensor),
)
def ggml_dup_tensor(ctx: ggml_context_p, src: ggml_tensor_p, /) -> ggml_tensor_p:
    """Create a new tensor with the same type and dimensions as the source tensor.

    Parameters:
        ctx: ggml context
        src: source tensor

    Returns:
        Pointer to ggml_tensor"""
    ...


# GGML_API struct ggml_tensor * ggml_view_tensor(struct ggml_context * ctx, struct ggml_tensor * src);
@ggml_function(
    "ggml_view_tensor",
    [ggml_context_p_ctypes, ctypes.POINTER(ggml_tensor)],
    ctypes.POINTER(ggml_tensor),
)
def ggml_view_tensor(ctx: ggml_context_p, src: ggml_tensor_p, /) -> ggml_tensor_p:
    """Create a new tensor with the same type, dimensions and data as the source tensor.

    Parameters:
        ctx: ggml context
        src: source tensor

    Returns:
        Pointer to ggml_tensor"""
    ...


# // Context tensor enumeration and lookup
# GGML_API struct ggml_tensor * ggml_get_first_tensor(const struct ggml_context * ctx);
@ggml_function(
    "ggml_get_first_tensor", [ggml_context_p_ctypes], ctypes.POINTER(ggml_tensor)
)
def ggml_get_first_tensor(ctx: ggml_context_p, /) -> ggml_tensor_p:
    """Get the first tensor from the ggml context.

    Parameters:
        ctx: ggml context

    Returns:
        Pointer to ggml_tensor"""
    ...


# GGML_API struct ggml_tensor * ggml_get_next_tensor (const struct ggml_context * ctx, struct ggml_tensor * tensor);
@ggml_function(
    "ggml_get_next_tensor",
    [ggml_context_p_ctypes, ctypes.POINTER(ggml_tensor)],
    ctypes.POINTER(ggml_tensor),
)
def ggml_get_next_tensor(
    ctx: ggml_context_p, tensor: ggml_tensor_p, /
) -> ggml_tensor_p:
    """Get the next tensor from the ggml context.

    Parameters:
        ctx: ggml context
        tensor: tensor

    Returns:
        Pointer to ggml_tensor"""
    ...


# GGML_API struct ggml_tensor * ggml_get_tensor(struct ggml_context * ctx, const char * name);
@ggml_function(
    "ggml_get_tensor",
    [ggml_context_p_ctypes, ctypes.c_char_p],
    ctypes.POINTER(ggml_tensor),
)
def ggml_get_tensor(ctx: ggml_context_p, name: bytes, /) -> ggml_tensor_p:
    """Get a tensor from the ggml context by name.

    Parameters:
        ctx: ggml context
        name: name of tensor

    Returns:
        Pointer to ggml_tensor"""
    ...


# GGML_API struct ggml_tensor * ggml_set_zero(struct ggml_tensor * tensor);
@ggml_function(
    "ggml_set_zero", [ctypes.POINTER(ggml_tensor)], ctypes.POINTER(ggml_tensor)
)
def ggml_set_zero(tensor: ggml_tensor_p, /) -> ggml_tensor_p:
    """Zero all elements in a tensor.

    Parameters:
        tensor: tensor

    Returns:
        Pointer to ggml_tensor"""
    ...


# GGML_API struct ggml_tensor * ggml_set_i32 (struct ggml_tensor * tensor, int32_t value);
@ggml_function(
    "ggml_set_i32",
    [ctypes.POINTER(ggml_tensor), ctypes.c_int32],
    ctypes.POINTER(ggml_tensor),
)
def ggml_set_i32(
    tensor: ggml_tensor_p, value: Union[ctypes.c_int32, int], /
) -> ggml_tensor_p:
    """Set all elements in a tensor to the given integer value.

    Parameters:
        tensor: tensor
        value: integer value

    Returns:
        Pointer to ggml_tensor"""
    ...


# GGML_API struct ggml_tensor * ggml_set_f32 (struct ggml_tensor * tensor, float value);
@ggml_function(
    "ggml_set_f32",
    [ctypes.POINTER(ggml_tensor), ctypes.c_float],
    ctypes.POINTER(ggml_tensor),
)
def ggml_set_f32(
    tensor: ggml_tensor_p, value: Union[ctypes.c_float, float], /
) -> ggml_tensor_p:
    """Set all elements in a tensor to the given float value.

    Parameters:
        tensor: tensor
        value: float value

    Returns:
        Pointer to ggml_tensor"""
    ...


# // Converts a flat index into coordinates
# GGML_API void    ggml_unravel_index(const struct ggml_tensor * tensor, int64_t i, int64_t * i0, int64_t * i1, int64_t * i2, int64_t * i3);
@ggml_function(
    "ggml_unravel_index",
    [
        ctypes.POINTER(ggml_tensor),
        ctypes.c_int64,
        ctypes.POINTER(ctypes.c_int64),
        ctypes.POINTER(ctypes.c_int64),
        ctypes.POINTER(ctypes.c_int64),
        ctypes.POINTER(ctypes.c_int64),
    ],
    None,
)
def ggml_unravel_index(
    tensor: ggml_tensor_p,
    i: Union[ctypes.c_int64, int],
    i0: CtypesPointer[ctypes.c_int64],
    i1: CtypesPointer[ctypes.c_int64],
    i2: CtypesPointer[ctypes.c_int64],
    i3: CtypesPointer[ctypes.c_int64],
    /,
):
    """Convert a flat index into coordinates.

    Parameters:
        tensor: tensor
        i: flat index
        i0: pointer to index 0
        i1: pointer to index 1
        i2: pointer to index 2
        i3: pointer to index 3"""
    ...


# GGML_API int32_t ggml_get_i32_1d(const struct ggml_tensor * tensor, int i);
@ggml_function(
    "ggml_get_i32_1d", [ctypes.POINTER(ggml_tensor), ctypes.c_int], ctypes.c_int32
)
def ggml_get_i32_1d(tensor: ggml_tensor_p, i: Union[ctypes.c_int, int], /) -> int:
    """Get the integer value of the i-th element in a 1-dimensional tensor.

    Parameters:
        tensor: tensor
        i: index of element

    Returns:
        integer value of element at index i"""
    ...


# GGML_API void    ggml_set_i32_1d(const struct ggml_tensor * tensor, int i, int32_t value);
@ggml_function(
    "ggml_set_i32_1d",
    [
        ctypes.POINTER(ggml_tensor),
        ctypes.c_int,
        ctypes.c_int32,
    ],
    None,
)
def ggml_set_i32_1d(
    tensor: ggml_tensor_p,
    i: Union[ctypes.c_int, int],
    value: Union[ctypes.c_int32, int],
    /,
):
    """Set the integer value of the i-th element in a 1-dimensional tensor.

    Parameters:
        tensor: tensor
        i: index of element
        value: integer value to set element to"""
    ...


# GGML_API int32_t ggml_get_i32_nd(const struct ggml_tensor * tensor, int i0, int i1, int i2, int i3);
@ggml_function(
    "ggml_get_i32_nd",
    [
        ctypes.POINTER(ggml_tensor),
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
    ],
    ctypes.c_int32,
)
def ggml_get_i32_nd(
    tensor: ggml_tensor_p,
    i0: Union[ctypes.c_int, int],
    i1: Union[ctypes.c_int, int],
    i2: Union[ctypes.c_int, int],
    i3: Union[ctypes.c_int, int],
    /,
) -> int:
    """Get the integer value of the element at the given coordinates in a 4-dimensional tensor.

    Parameters:
        tensor: tensor
        i0: index of element in dimension 0
        i1: index of element in dimension 1
        i2: index of element in dimension 2
        i3: index of element in dimension 3

    Returns:
        integer value of element at coordinates"""
    ...


# GGML_API void    ggml_set_i32_nd(const struct ggml_tensor * tensor, int i0, int i1, int i2, int i3, int32_t value);
@ggml_function(
    "ggml_set_i32_nd",
    [
        ctypes.POINTER(ggml_tensor),
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int32,
    ],
    None,
)
def ggml_set_i32_nd(
    tensor: ggml_tensor_p,
    i0: Union[ctypes.c_int, int],
    i1: Union[ctypes.c_int, int],
    i2: Union[ctypes.c_int, int],
    i3: Union[ctypes.c_int, int],
    value: Union[ctypes.c_int32, int],
    /,
):
    """Set the integer value of the element at the given coordinates in a 4-dimensional tensor.

    Parameters:
        tensor: tensor
        i0: index of element in dimension 0
        i1: index of element in dimension 1
        i2: index of element in dimension 2
        i3: index of element in dimension 3
        value: integer value to set element to"""
    ...


# GGML_API float   ggml_get_f32_1d(const struct ggml_tensor * tensor, int i);
@ggml_function(
    "ggml_get_f32_1d", [ctypes.POINTER(ggml_tensor), ctypes.c_int], ctypes.c_float
)
def ggml_get_f32_1d(tensor: ggml_tensor_p, i: Union[ctypes.c_int, int], /) -> float:
    """Get the float value of the i-th element in a 1-dimensional tensor.

    Parameters:
        tensor: tensor

    Returns:
        float value of element at index i"""
    ...


# GGML_API void    ggml_set_f32_1d(const struct ggml_tensor * tensor, int i, float value);
@ggml_function(
    "ggml_set_f32_1d",
    [
        ctypes.POINTER(ggml_tensor),
        ctypes.c_int,
        ctypes.c_float,
    ],
    None,
)
def ggml_set_f32_1d(
    tensor: ggml_tensor_p,
    i: Union[ctypes.c_int, int],
    value: Union[ctypes.c_float, float],
    /,
):
    """Set the float value of the i-th element in a 1-dimensional tensor.

    Parameters:
        tensor: tensor
        i: index of element
        value: float value to set element to"""
    ...


# GGML_API float   ggml_get_f32_nd(const struct ggml_tensor * tensor, int i0, int i1, int i2, int i3);
@ggml_function(
    "ggml_get_f32_nd",
    [
        ctypes.POINTER(ggml_tensor),
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
    ],
    ctypes.c_float,
)
def ggml_get_f32_nd(
    tensor: ggml_tensor_p,
    i0: Union[ctypes.c_int, int],
    i1: Union[ctypes.c_int, int],
    i2: Union[ctypes.c_int, int],
    i3: Union[ctypes.c_int, int],
    /,
) -> float:
    """Get the float value of the element at the given coordinates in a 4-dimensional tensor.

    Parameters:
        tensor: tensor
        i0: index of element in dimension 0
        i1: index of element in dimension 1
        i2: index of element in dimension 2
        i3: index of element in dimension 3

    Returns:
        float value of element at coordinates"""
    ...


# GGML_API void    ggml_set_f32_nd(const struct ggml_tensor * tensor, int i0, int i1, int i2, int i3, float value);
@ggml_function(
    "ggml_set_f32_nd",
    [
        ctypes.POINTER(ggml_tensor),
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_float,
    ],
    None,
)
def ggml_set_f32_nd(
    tensor: ggml_tensor_p,
    i0: Union[ctypes.c_int, int],
    i1: Union[ctypes.c_int, int],
    i2: Union[ctypes.c_int, int],
    i3: Union[ctypes.c_int, int],
    value: Union[ctypes.c_float, float],
    /,
):
    """Set the float value of the element at the given coordinates in a 4-dimensional tensor.

    Parameters:
        tensor: tensor
        i0: index of element in dimension 0
        i1: index of element in dimension 1
        i2: index of element in dimension 2
        i3: index of element in dimension 3
        value: float value to set element to"""
    ...


# GGML_API void *  ggml_get_data    (const struct ggml_tensor * tensor);
@ggml_function("ggml_get_data", [ctypes.POINTER(ggml_tensor)], ctypes.c_void_p)
def ggml_get_data(tensor: ggml_tensor_p, /) -> Optional[int]:
    """Get the data pointer of a tensor.

    Parameters:
        tensor: tensor

    Returns:
        Pointer to data, or None if tensor has no data"""
    ...


# GGML_API float * ggml_get_data_f32(const struct ggml_tensor * tensor);
@ggml_function(
    "ggml_get_data_f32", [ctypes.POINTER(ggml_tensor)], ctypes.POINTER(ctypes.c_float)
)
def ggml_get_data_f32(
    tensor: ggml_tensor_p, /
) -> Optional[CtypesArray[ctypes.c_float]]:
    """Get the data pointer of a tensor as a float array.

    Parameters:
        tensor: tensor

    Returns:
        (Optional[ctypes.Array[ctypes.c_float]]): array of float to data, or None if tensor has no data
    """
    ...


# GGML_API GGML_CALL enum ggml_unary_op ggml_get_unary_op(const struct ggml_tensor * tensor);
@ggml_function("ggml_get_unary_op", [ctypes.POINTER(ggml_tensor)], ctypes.c_int)
def ggml_get_unary_op(tensor: ggml_tensor_p, /) -> int:
    """Get the unary operation of a tensor.

    Parameters:
        tensor: tensor

    Returns:
        unary operation"""
    ...


# GGML_API enum ggml_glu_op ggml_get_glu_op(const struct ggml_tensor * tensor);
@ggml_function("ggml_get_glu_op", [ctypes.POINTER(ggml_tensor)], ctypes.c_int)
def ggml_get_glu_op(tensor: ggml_tensor_p, /) -> int:
    """Get the GLU operation of a tensor."""
    ...


# GGML_API const char *         ggml_get_name(const struct ggml_tensor * tensor);
@ggml_function("ggml_get_name", [ctypes.POINTER(ggml_tensor)], ctypes.c_char_p)
def ggml_get_name(tensor: ggml_tensor_p, /) -> bytes:
    """Get the name of a tensor.

    Parameters:
        tensor: tensor

    Returns:
        name of tensor"""
    ...


# GGML_API struct ggml_tensor * ggml_set_name(struct ggml_tensor * tensor, const char * name);
@ggml_function(
    "ggml_set_name",
    [ctypes.POINTER(ggml_tensor), ctypes.c_char_p],
    ctypes.POINTER(ggml_tensor),
)
def ggml_set_name(tensor: ggml_tensor_p, name: bytes, /) -> ggml_tensor_p:
    """Set the name of a tensor.

    Parameters:
        tensor: tensor
        name: name to set tensor to

    Returns:
        Pointer to ggml_tensor"""
    ...


# GGML_API struct ggml_tensor * ggml_format_name(struct ggml_tensor * tensor, const char * fmt, ...);
@ggml_function(
    "ggml_format_name",
    [ctypes.POINTER(ggml_tensor), ctypes.c_char_p],
    ctypes.POINTER(ggml_tensor),
)
def ggml_format_name(
    tensor: ggml_tensor_p,
    fmt: bytes,
    /,
    *args: Sequence[Union[bool, int, float, str]],
) -> ggml_tensor_p:
    """Format the name of a tensor using the given format c string and arguments.

    Parameters:
        tensor: tensor
        fmt: format c string
        args: arguments to format string

    Returns:
        Pointer to ggml_tensor"""
    ...


# //
# // operations on tensors with backpropagation
# //


# GGML_API struct ggml_tensor * ggml_dup(
#         struct ggml_context * ctx,
#         struct ggml_tensor  * a);
@ggml_function(
    "ggml_dup",
    [ggml_context_p_ctypes, ctypes.POINTER(ggml_tensor)],
    ctypes.POINTER(ggml_tensor),
)
def ggml_dup(ctx: ggml_context_p, a: ggml_tensor_p, /) -> ggml_tensor_p:
    ...


# // in-place, returns view(a)
# GGML_API struct ggml_tensor * ggml_dup_inplace(
#         struct ggml_context * ctx,
#         struct ggml_tensor  * a);
@ggml_function(
    "ggml_dup_inplace",
    [ggml_context_p_ctypes, ctypes.POINTER(ggml_tensor)],
    ctypes.POINTER(ggml_tensor),
)
def ggml_dup_inplace(ctx: ggml_context_p, a: ggml_tensor_p, /) -> ggml_tensor_p:
    ...


# GGML_API struct ggml_tensor * ggml_add(
#         struct ggml_context * ctx,
#         struct ggml_tensor  * a,
#         struct ggml_tensor  * b);
@ggml_function(
    "ggml_add",
    [
        ggml_context_p_ctypes,
        ctypes.POINTER(ggml_tensor),
        ctypes.POINTER(ggml_tensor),
    ],
    ctypes.POINTER(ggml_tensor),
)
def ggml_add(
    ctx: ggml_context_p, a: ggml_tensor_p, b: ggml_tensor_p, /
) -> ggml_tensor_p:
    """Add two tensors together and return the result.

    Parameters:
        ctx: ggml context
        a: first tensor
        b: second tensor

    Returns:
        Pointer to ggml_tensor"""
    ...


# GGML_API struct ggml_tensor * ggml_add_inplace(
#         struct ggml_context * ctx,
#         struct ggml_tensor  * a,
#         struct ggml_tensor  * b);
@ggml_function(
    "ggml_add_inplace",
    [
        ggml_context_p_ctypes,
        ctypes.POINTER(ggml_tensor),
        ctypes.POINTER(ggml_tensor),
    ],
    ctypes.POINTER(ggml_tensor),
)
def ggml_add_inplace(
    ctx: ggml_context_p, a: ggml_tensor_p, b: ggml_tensor_p, /
) -> ggml_tensor_p:
    """Add two tensors together and store the result in the first tensor.

    Parameters:
        ctx: ggml context
        a: first tensor
        b: second tensor

    Returns:
        Pointer to ggml_tensor"""
    ...


# GGML_API struct ggml_tensor * ggml_add_cast(
#         struct ggml_context * ctx,
#         struct ggml_tensor  * a,
#         struct ggml_tensor  * b,
#         enum   ggml_type      type);
@ggml_function(
    "ggml_add_cast",
    [
        ggml_context_p_ctypes,
        ctypes.POINTER(ggml_tensor),
        ctypes.POINTER(ggml_tensor),
        ctypes.c_int,
    ],
    ctypes.POINTER(ggml_tensor),
)
def ggml_add_cast(
    ctx: ggml_context_p,
    a: ggml_tensor_p,
    b: ggml_tensor_p,
    type: Union[ctypes.c_int, int],
    /,
) -> ggml_tensor_p:
    """Add two tensors together and cast the result to the given type.

    Parameters:
        ctx: ggml context
        a: first tensor
        b: second tensor
        type: type to cast result to

    Returns:
        Pointer to ggml_tensor"""
    ...


# GGML_API struct ggml_tensor * ggml_add_id(
#         struct ggml_context * ctx,
#         struct ggml_tensor  * a,
#         struct ggml_tensor  * b,
#         struct ggml_tensor  * ids);
@ggml_function(
    "ggml_add_id",
    [
        ggml_context_p_ctypes,
        ctypes.POINTER(ggml_tensor),
        ctypes.POINTER(ggml_tensor),
        ctypes.POINTER(ggml_tensor),
    ],
    ctypes.POINTER(ggml_tensor),
)
def ggml_add_id(
    ctx: ggml_context_p,
    a: ggml_tensor_p,
    b: ggml_tensor_p,
    ids: ggml_tensor_p,
    /,
) -> ggml_tensor_p:
    ...


# GGML_API struct ggml_tensor * ggml_add1(
#         struct ggml_context * ctx,
#         struct ggml_tensor  * a,
#         struct ggml_tensor  * b);
@ggml_function(
    "ggml_add1",
    [
        ggml_context_p_ctypes,
        ctypes.POINTER(ggml_tensor),
        ctypes.POINTER(ggml_tensor),
    ],
    ctypes.POINTER(ggml_tensor),
)
def ggml_add1(
    ctx: ggml_context_p, a: ggml_tensor_p, b: ggml_tensor_p, /
) -> ggml_tensor_p:
    ...


# GGML_API struct ggml_tensor * ggml_add1_inplace(
#         struct ggml_context * ctx,
#         struct ggml_tensor  * a,
#         struct ggml_tensor  * b);
@ggml_function(
    "ggml_add1_inplace",
    [
        ggml_context_p_ctypes,
        ctypes.POINTER(ggml_tensor),
        ctypes.POINTER(ggml_tensor),
    ],
    ctypes.POINTER(ggml_tensor),
)
def ggml_add1_inplace(
    ctx: ggml_context_p, a: ggml_tensor_p, b: ggml_tensor_p, /
) -> ggml_tensor_p:
    ...


# // dst = a
# // view(dst, nb1, nb2, nb3, offset) += b
# // return dst
# GGML_API struct ggml_tensor * ggml_acc(
#         struct ggml_context * ctx,
#         struct ggml_tensor  * a,
#         struct ggml_tensor  * b,
#         size_t                nb1,
#         size_t                nb2,
#         size_t                nb3,
#         size_t                offset);
@ggml_function(
    "ggml_acc",
    [
        ggml_context_p_ctypes,
        ctypes.POINTER(ggml_tensor),
        ctypes.POINTER(ggml_tensor),
        ctypes.c_size_t,
        ctypes.c_size_t,
        ctypes.c_size_t,
        ctypes.c_size_t,
    ],
    ctypes.POINTER(ggml_tensor),
)
def ggml_acc(
    ctx: ggml_context_p,
    a: ggml_tensor_p,
    b: ggml_tensor_p,
    nb1: Union[ctypes.c_size_t, int],
    nb2: Union[ctypes.c_size_t, int],
    nb3: Union[ctypes.c_size_t, int],
    offset: Union[ctypes.c_size_t, int],
    /,
) -> ggml_tensor_p:
    ...


# GGML_API struct ggml_tensor * ggml_acc_inplace(
#         struct ggml_context * ctx,
#         struct ggml_tensor  * a,
#         struct ggml_tensor  * b,
#         size_t                nb1,
#         size_t                nb2,
#         size_t                nb3,
#         size_t                offset);
@ggml_function(
    "ggml_acc_inplace",
    [
        ggml_context_p_ctypes,
        ctypes.POINTER(ggml_tensor),
        ctypes.POINTER(ggml_tensor),
        ctypes.c_size_t,
        ctypes.c_size_t,
        ctypes.c_size_t,
        ctypes.c_size_t,
    ],
    ctypes.POINTER(ggml_tensor),
)
def ggml_acc_inplace(
    ctx: ggml_context_p,
    a: ggml_tensor_p,
    b: ggml_tensor_p,
    nb1: Union[ctypes.c_size_t, int],
    nb2: Union[ctypes.c_size_t, int],
    nb3: Union[ctypes.c_size_t, int],
    offset: Union[ctypes.c_size_t, int],
    /,
) -> ggml_tensor_p:
    ...


# GGML_API struct ggml_tensor * ggml_sub(
#         struct ggml_context * ctx,
#         struct ggml_tensor  * a,
#         struct ggml_tensor  * b);
@ggml_function(
    "ggml_sub",
    [
        ggml_context_p_ctypes,
        ctypes.POINTER(ggml_tensor),
        ctypes.POINTER(ggml_tensor),
    ],
    ctypes.POINTER(ggml_tensor),
)
def ggml_sub(
    ctx: ggml_context_p, a: ggml_tensor_p, b: ggml_tensor_p, /
) -> ggml_tensor_p:
    """Subtract two tensors and return the result.

    Parameters:
        ctx: ggml context
        a: first tensor
        b: second tensor

    Returns:
        Pointer to ggml_tensor"""
    ...


# GGML_API struct ggml_tensor * ggml_sub_inplace(
#         struct ggml_context * ctx,
#         struct ggml_tensor  * a,
#         struct ggml_tensor  * b);
@ggml_function(
    "ggml_sub_inplace",
    [
        ggml_context_p_ctypes,
        ctypes.POINTER(ggml_tensor),
        ctypes.POINTER(ggml_tensor),
    ],
    ctypes.POINTER(ggml_tensor),
)
def ggml_sub_inplace(
    ctx: ggml_context_p, a: ggml_tensor_p, b: ggml_tensor_p, /
) -> ggml_tensor_p:
    """Subtract two tensors and store the result in the first tensor.

    Parameters:
        ctx: ggml context
        a: first tensor
        b: second tensor

    Returns:
        Pointer to ggml_tensor"""
    ...


# GGML_API struct ggml_tensor * ggml_mul(
#         struct ggml_context * ctx,
#         struct ggml_tensor  * a,
#         struct ggml_tensor  * b);
@ggml_function(
    "ggml_mul",
    [
        ggml_context_p_ctypes,
        ctypes.POINTER(ggml_tensor),
        ctypes.POINTER(ggml_tensor),
    ],
    ctypes.POINTER(ggml_tensor),
)
def ggml_mul(
    ctx: ggml_context_p, a: ggml_tensor_p, b: ggml_tensor_p, /
) -> ggml_tensor_p:
    """Element-wise multiply two tensors and return the result.

    Parameters:
        ctx: ggml context
        a: first tensor
        b: second tensor

    Returns:
        Pointer to ggml_tensor"""
    ...


# GGML_API struct ggml_tensor * ggml_mul_inplace(
#         struct ggml_context * ctx,
#         struct ggml_tensor  * a,
#         struct ggml_tensor  * b);
@ggml_function(
    "ggml_mul_inplace",
    [
        ggml_context_p_ctypes,
        ctypes.POINTER(ggml_tensor),
        ctypes.POINTER(ggml_tensor),
    ],
    ctypes.POINTER(ggml_tensor),
)
def ggml_mul_inplace(
    ctx: ggml_context_p, a: ggml_tensor_p, b: ggml_tensor_p, /
) -> ggml_tensor_p:
    """Element-wise multiply two tensors and store the result in the first tensor.

    Parameters:
        ctx: ggml context
        a: first tensor
        b: second tensor

    Returns:
        Pointer to ggml_tensor"""
    ...


# GGML_API struct ggml_tensor * ggml_div(
#         struct ggml_context * ctx,
#         struct ggml_tensor  * a,
#         struct ggml_tensor  * b);
@ggml_function(
    "ggml_div",
    [
        ggml_context_p_ctypes,
        ctypes.POINTER(ggml_tensor),
        ctypes.POINTER(ggml_tensor),
    ],
    ctypes.POINTER(ggml_tensor),
)
def ggml_div(
    ctx: ggml_context_p, a: ggml_tensor_p, b: ggml_tensor_p, /
) -> ggml_tensor_p:
    """Element-wise divide two tensors and return the result.

    Parameters:
        ctx: ggml context
        a: first tensor
        b: second tensor

    Returns:
        Pointer to ggml_tensor"""
    ...


# GGML_API struct ggml_tensor * ggml_div_inplace(
#         struct ggml_context * ctx,
#         struct ggml_tensor  * a,
#         struct ggml_tensor  * b);
@ggml_function(
    "ggml_div_inplace",
    [
        ggml_context_p_ctypes,
        ctypes.POINTER(ggml_tensor),
        ctypes.POINTER(ggml_tensor),
    ],
    ctypes.POINTER(ggml_tensor),
)
def ggml_div_inplace(
    ctx: ggml_context_p, a: ggml_tensor_p, b: ggml_tensor_p, /
) -> ggml_tensor_p:
    """Element-wise divide two tensors and store the result in the first tensor.

    Parameters:
        ctx: ggml context
        a: first tensor
        b: second tensor

    Returns:
        Pointer to ggml_tensor"""
    ...


# GGML_API struct ggml_tensor * ggml_sqr(
#         struct ggml_context * ctx,
#         struct ggml_tensor  * a);
@ggml_function(
    "ggml_sqr",
    [ggml_context_p_ctypes, ctypes.POINTER(ggml_tensor)],
    ctypes.POINTER(ggml_tensor),
)
def ggml_sqr(ctx: ggml_context_p, a: ggml_tensor_p, /) -> ggml_tensor_p:
    """Square all elements in a tensor and return the result.

    Parameters:
        ctx: ggml context
        a: tensor

    Returns:
        Pointer to ggml_tensor"""
    ...


# GGML_API struct ggml_tensor * ggml_sqr_inplace(
#         struct ggml_context * ctx,
#         struct ggml_tensor  * a);
@ggml_function(
    "ggml_sqr_inplace",
    [ggml_context_p_ctypes, ctypes.POINTER(ggml_tensor)],
    ctypes.POINTER(ggml_tensor),
)
def ggml_sqr_inplace(ctx: ggml_context_p, a: ggml_tensor_p, /) -> ggml_tensor_p:
    """Square all elements in a tensor and store the result in the first tensor.

    Parameters:
        ctx: ggml context
        a: tensor

    Returns:
        Pointer to ggml_tensor"""
    ...


# GGML_API struct ggml_tensor * ggml_sqrt(
#         struct ggml_context * ctx,
#         struct ggml_tensor  * a);
@ggml_function(
    "ggml_sqrt",
    [ggml_context_p_ctypes, ctypes.POINTER(ggml_tensor)],
    ctypes.POINTER(ggml_tensor),
)
def ggml_sqrt(ctx: ggml_context_p, a: ggml_tensor_p, /) -> ggml_tensor_p:
    """Square root all elements in a tensor and return the result.

    Parameters:
        ctx: ggml context
        a: tensor

    Returns:
        Pointer to ggml_tensor"""
    ...


# GGML_API struct ggml_tensor * ggml_sqrt_inplace(
#         struct ggml_context * ctx,
#         struct ggml_tensor  * a);
@ggml_function(
    "ggml_sqrt_inplace",
    [ggml_context_p_ctypes, ctypes.POINTER(ggml_tensor)],
    ctypes.POINTER(ggml_tensor),
)
def ggml_sqrt_inplace(ctx: ggml_context_p, a: ggml_tensor_p, /) -> ggml_tensor_p:
    """Square root all elements in a tensor and store the result in the first tensor.

    Parameters:
        ctx: ggml context

    Returns:
        Pointer to ggml_tensor"""
    ...


# GGML_API struct ggml_tensor * ggml_log(
#         struct ggml_context * ctx,
#         struct ggml_tensor  * a);
@ggml_function(
    "ggml_log",
    [ggml_context_p_ctypes, ctypes.POINTER(ggml_tensor)],
    ctypes.POINTER(ggml_tensor),
)
def ggml_log(ctx: ggml_context_p, a: ggml_tensor_p, /) -> ggml_tensor_p:
    """Take the natural logarithm of all elements in a tensor and return the result.

    Parameters:
        ctx: ggml context
        a: tensor

    Returns:
        Pointer to ggml_tensor"""
    ...


# GGML_API struct ggml_tensor * ggml_log_inplace(
#         struct ggml_context * ctx,
#         struct ggml_tensor  * a);
@ggml_function(
    "ggml_log_inplace",
    [ggml_context_p_ctypes, ctypes.POINTER(ggml_tensor)],
    ctypes.POINTER(ggml_tensor),
)
def ggml_log_inplace(ctx: ggml_context_p, a: ggml_tensor_p, /) -> ggml_tensor_p:
    """Take the natural logarithm of all elements in a tensor and store the result in the first tensor.

    Parameters:
        ctx: ggml context
        a: tensor

    Returns:
        Pointer to ggml_tensor"""
    ...


# GGML_API struct ggml_tensor * ggml_expm1(
#         struct ggml_context * ctx,
#         struct ggml_tensor  * a);
@ggml_function(
    "ggml_expm1",
    [ggml_context_p_ctypes, ctypes.POINTER(ggml_tensor)],
    ctypes.POINTER(ggml_tensor),
    enabled=hasattr(lib, "ggml_expm1"),
)
def ggml_expm1(ctx: ggml_context_p, a: ggml_tensor_p, /) -> ggml_tensor_p:
    """Compute exp(a) - 1 for all elements in a tensor and return the result.

    Parameters:
        ctx: ggml context
        a: tensor

    Returns:
        Pointer to ggml_tensor"""
    ...


# GGML_API struct ggml_tensor * ggml_expm1_inplace(
#         struct ggml_context * ctx,
#         struct ggml_tensor  * a);
@ggml_function(
    "ggml_expm1_inplace",
    [ggml_context_p_ctypes, ctypes.POINTER(ggml_tensor)],
    ctypes.POINTER(ggml_tensor),
    enabled=hasattr(lib, "ggml_expm1_inplace"),
)
def ggml_expm1_inplace(ctx: ggml_context_p, a: ggml_tensor_p, /) -> ggml_tensor_p:
    """Compute exp(a) - 1 for all elements in a tensor and store the result in the first tensor.

    Parameters:
        ctx: ggml context
        a: tensor

    Returns:
        Pointer to ggml_tensor"""
    ...


# GGML_API struct ggml_tensor * ggml_softplus(
#         struct ggml_context * ctx,
#         struct ggml_tensor  * a);
@ggml_function(
    "ggml_softplus",
    [ggml_context_p_ctypes, ctypes.POINTER(ggml_tensor)],
    ctypes.POINTER(ggml_tensor),
    enabled=hasattr(lib, "ggml_softplus"),
)
def ggml_softplus(ctx: ggml_context_p, a: ggml_tensor_p, /) -> ggml_tensor_p:
    """Apply the softplus activation function to all elements in a tensor and return the result.

    Parameters:
        ctx: ggml context
        a: tensor

    Returns:
        Pointer to ggml_tensor"""
    ...


# GGML_API struct ggml_tensor * ggml_softplus_inplace(
#         struct ggml_context * ctx,
#         struct ggml_tensor  * a);
@ggml_function(
    "ggml_softplus_inplace",
    [ggml_context_p_ctypes, ctypes.POINTER(ggml_tensor)],
    ctypes.POINTER(ggml_tensor),
    enabled=hasattr(lib, "ggml_softplus_inplace"),
)
def ggml_softplus_inplace(ctx: ggml_context_p, a: ggml_tensor_p, /) -> ggml_tensor_p:
    """Apply the softplus activation function to all elements in a tensor and store the result in the first tensor.

    Parameters:
        ctx: ggml context
        a: tensor

    Returns:
        Pointer to ggml_tensor"""
    ...


# GGML_API struct ggml_tensor * ggml_sin(
#         struct ggml_context * ctx,
#         struct ggml_tensor  * a);
@ggml_function(
    "ggml_sin",
    [ggml_context_p_ctypes, ctypes.POINTER(ggml_tensor)],
    ctypes.POINTER(ggml_tensor),
)
def ggml_sin(ctx: ggml_context_p, a: ggml_tensor_p, /) -> ggml_tensor_p:
    """Compute the sine of all elements in a tensor and return the result.

    Parameters:
        ctx: ggml context
        a: tensor

    Returns:
        Pointer to ggml_tensor"""
    ...


# GGML_API struct ggml_tensor * ggml_sin_inplace(
#         struct ggml_context * ctx,
#         struct ggml_tensor  * a);
@ggml_function(
    "ggml_sin_inplace",
    [ggml_context_p_ctypes, ctypes.POINTER(ggml_tensor)],
    ctypes.POINTER(ggml_tensor),
)
def ggml_sin_inplace(ctx: ggml_context_p, a: ggml_tensor_p, /) -> ggml_tensor_p:
    """Compute the sine of all elements in a tensor and store the result in the first tensor.

    Parameters:
        ctx: ggml context
        a: tensor

    Returns:
        Pointer to ggml_tensor"""
    ...


# GGML_API struct ggml_tensor * ggml_cos(
#         struct ggml_context * ctx,
#         struct ggml_tensor  * a);
@ggml_function(
    "ggml_cos",
    [ggml_context_p_ctypes, ctypes.POINTER(ggml_tensor)],
    ctypes.POINTER(ggml_tensor),
)
def ggml_cos(ctx: ggml_context_p, a: ggml_tensor_p, /) -> ggml_tensor_p:
    """Compute the cosine of all elements in a tensor and return the result.

    Parameters:
        ctx: ggml context
        a: tensor

    Returns:
        Pointer to ggml_tensor"""
    ...


# GGML_API struct ggml_tensor * ggml_cos_inplace(
#         struct ggml_context * ctx,
#         struct ggml_tensor  * a);
@ggml_function(
    "ggml_cos_inplace",
    [ggml_context_p_ctypes, ctypes.POINTER(ggml_tensor)],
    ctypes.POINTER(ggml_tensor),
)
def ggml_cos_inplace(ctx: ggml_context_p, a: ggml_tensor_p, /) -> ggml_tensor_p:
    """Compute the cosine of all elements in a tensor and store the result in the first tensor.

    Parameters:
        ctx: ggml context
        a: tensor

    Returns:
        Pointer to ggml_tensor"""
    ...


# // return scalar
# GGML_API struct ggml_tensor * ggml_sum(
#         struct ggml_context * ctx,
#         struct ggml_tensor  * a);
@ggml_function(
    "ggml_sum",
    [ggml_context_p_ctypes, ctypes.POINTER(ggml_tensor)],
    ctypes.POINTER(ggml_tensor),
)
def ggml_sum(ctx: ggml_context_p, a: ggml_tensor_p, /) -> ggml_tensor_p:
    """Sum all elements in a tensor and return the result.

    Parameters:
        ctx: ggml context
        a: tensor

    Returns:
        Pointer to ggml_tensor"""
    ...


# // sums along rows, with input shape [a,b,c,d] return shape [1,b,c,d]
# GGML_API struct ggml_tensor * ggml_sum_rows(
#         struct ggml_context * ctx,
#         struct ggml_tensor  * a);
@ggml_function(
    "ggml_sum_rows",
    [ggml_context_p_ctypes, ctypes.POINTER(ggml_tensor)],
    ctypes.POINTER(ggml_tensor),
)
def ggml_sum_rows(ctx: ggml_context_p, a: ggml_tensor_p, /) -> ggml_tensor_p:
    """Sum all elements in a tensor along the first axis and return the result.

    sums along rows, with input shape [a,b,c,d] return shape [1,b,c,d]

    Parameters:
        ctx: ggml context
        a: tensor

    Returns:
        Pointer to ggml_tensor"""
    ...


# GGML_API struct ggml_tensor * ggml_cumsum(
#         struct ggml_context * ctx,
#         struct ggml_tensor  * a);
@ggml_function(
    "ggml_cumsum",
    [ggml_context_p_ctypes, ctypes.POINTER(ggml_tensor)],
    ctypes.POINTER(ggml_tensor),
)
def ggml_cumsum(ctx: ggml_context_p, a: ggml_tensor_p, /) -> ggml_tensor_p:
    ...


# // mean along rows
# GGML_API struct ggml_tensor * ggml_mean(
#         struct ggml_context * ctx,
#         struct ggml_tensor  * a);
@ggml_function(
    "ggml_mean",
    [ggml_context_p_ctypes, ctypes.POINTER(ggml_tensor)],
    ctypes.POINTER(ggml_tensor),
)
def ggml_mean(ctx: ggml_context_p, a: ggml_tensor_p, /) -> ggml_tensor_p:
    """Take the mean of all elements in a tensor and return the result.

    Parameters:
        ctx: ggml context
        a: tensor

    Returns:
        Pointer to ggml_tensor"""
    ...


# // argmax along rows
# GGML_API struct ggml_tensor * ggml_argmax(
#         struct ggml_context * ctx,
#         struct ggml_tensor  * a);
@ggml_function(
    "ggml_argmax",
    [ggml_context_p_ctypes, ctypes.POINTER(ggml_tensor)],
    ctypes.POINTER(ggml_tensor),
)
def ggml_argmax(ctx: ggml_context_p, a: ggml_tensor_p, /) -> ggml_tensor_p:
    """Take the argmax of all elements in a tensor and return the result.

    argmax along rows

    Parameters:
        ctx: ggml context
        a: tensor

    Returns:
        Pointer to ggml_tensor"""
    ...


# GGML_API struct ggml_tensor * ggml_count_equal(
#         struct ggml_context * ctx,
#         struct ggml_tensor  * a,
#         struct ggml_tensor  * b);
@ggml_function(
    "ggml_count_equal",
    [
        ggml_context_p_ctypes,
        ctypes.POINTER(ggml_tensor),
        ctypes.POINTER(ggml_tensor),
    ],
    ctypes.POINTER(ggml_tensor),
)
def ggml_count_equal(
    ctx: ggml_context_p,
    a: ggml_tensor_p,
    b: ggml_tensor_p,
    /,
) -> ggml_tensor_p:
    ...


# // if a is the same shape as b, and a is not parameter, return a
# // otherwise, return a new tensor: repeat(a) to fit in b
# GGML_API struct ggml_tensor * ggml_repeat(
#         struct ggml_context * ctx,
#         struct ggml_tensor  * a,
#         struct ggml_tensor  * b);
@ggml_function(
    "ggml_repeat",
    [
        ggml_context_p_ctypes,
        ctypes.POINTER(ggml_tensor),
        ctypes.POINTER(ggml_tensor),
    ],
    ctypes.POINTER(ggml_tensor),
)
def ggml_repeat(
    ctx: ggml_context_p, a: ggml_tensor_p, b: ggml_tensor_p, /
) -> ggml_tensor_p:
    """Repeat a tensor to fit the shape of another tensor.

    If a is the same shape as b, and a is not parameter, return a

    Parameters:
        ctx: ggml context
        a: tensor to repeat
        b: tensor to fit

    Returns:
        Pointer to ggml_tensor"""
    ...


# GGML_API struct ggml_tensor * ggml_repeat_4d(
#         struct ggml_context * ctx,
#         struct ggml_tensor  * a,
#                    int64_t    ne0,
#                    int64_t    ne1,
#                    int64_t    ne2,
#                    int64_t    ne3);
@ggml_function(
    "ggml_repeat_4d",
    [
        ggml_context_p_ctypes,
        ctypes.POINTER(ggml_tensor),
        ctypes.c_int64,
        ctypes.c_int64,
        ctypes.c_int64,
        ctypes.c_int64,
    ],
    ctypes.POINTER(ggml_tensor),
)
def ggml_repeat_4d(
    ctx: ggml_context_p,
    a: ggml_tensor_p,
    ne0: Union[ctypes.c_int64, int],
    ne1: Union[ctypes.c_int64, int],
    ne2: Union[ctypes.c_int64, int],
    ne3: Union[ctypes.c_int64, int],
    /,
) -> ggml_tensor_p:
    ...


# // sums repetitions in a into shape of b
# GGML_API struct ggml_tensor * ggml_repeat_back(
#         struct ggml_context * ctx,
#         struct ggml_tensor  * a,
#         struct ggml_tensor  * b);
@ggml_function(
    "ggml_repeat_back",
    [
        ggml_context_p_ctypes,
        ctypes.POINTER(ggml_tensor),
        ctypes.POINTER(ggml_tensor),
    ],
    ctypes.POINTER(ggml_tensor),
)
def ggml_repeat_back(
    ctx: ggml_context_p, a: ggml_tensor_p, b: ggml_tensor_p, /
) -> ggml_tensor_p:
    ...


# // concat a and b on dim 2
# // used in stable-diffusion
# GGML_API struct ggml_tensor * ggml_concat(
#         struct ggml_context * ctx,
#         struct ggml_tensor  * a,
#         struct ggml_tensor  * b,
#         int                   dim);
@ggml_function(
    "ggml_concat",
    [
        ggml_context_p_ctypes,
        ctypes.POINTER(ggml_tensor),
        ctypes.POINTER(ggml_tensor),
        ctypes.c_int,
    ],
    ctypes.POINTER(ggml_tensor),
)
def ggml_concat(
    ctx: ggml_context_p, a: ggml_tensor_p, b: ggml_tensor_p, dim: int, /
) -> ggml_tensor_p:
    """Concatenate two tensors along the second axis and return the result.

    Parameters:
        ctx: ggml context
        a: first tensor
        b: second tensor
        dim: dimension to concatenate along

    Returns:
        Pointer to ggml_tensor"""
    ...


# GGML_API struct ggml_tensor * ggml_abs(
#         struct ggml_context * ctx,
#         struct ggml_tensor  * a);
@ggml_function(
    "ggml_abs",
    [ggml_context_p_ctypes, ctypes.POINTER(ggml_tensor)],
    ctypes.POINTER(ggml_tensor),
)
def ggml_abs(ctx: ggml_context_p, a: ggml_tensor_p, /) -> ggml_tensor_p:
    """Take the absolute value of all elements in a tensor and return the result.

    Parameters:
        ctx: ggml context
        a: tensor

    Returns:
        Pointer to ggml_tensor"""
    ...


# GGML_API struct ggml_tensor * ggml_abs_inplace(
#         struct ggml_context * ctx,
#         struct ggml_tensor  * a);
@ggml_function(
    "ggml_abs_inplace",
    [ggml_context_p_ctypes, ctypes.POINTER(ggml_tensor)],
    ctypes.POINTER(ggml_tensor),
)
def ggml_abs_inplace(ctx: ggml_context_p, a: ggml_tensor_p, /) -> ggml_tensor_p:
    """Take the absolute value of all elements in a tensor and store the result in the first tensor.

    Parameters:
        ctx: ggml context
        a: tensor

    Returns:
        Pointer to ggml_tensor"""
    ...


# GGML_API struct ggml_tensor * ggml_sgn(
#         struct ggml_context * ctx,
#         struct ggml_tensor  * a);
@ggml_function(
    "ggml_sgn",
    [ggml_context_p_ctypes, ctypes.POINTER(ggml_tensor)],
    ctypes.POINTER(ggml_tensor),
)
def ggml_sgn(ctx: ggml_context_p, a: ggml_tensor_p, /) -> ggml_tensor_p:
    """Get the sign of all elements in a tensor and return the result.

    Parameters:
        ctx: ggml context
        a: tensor

    Returns:
        Pointer to ggml_tensor"""
    ...


# GGML_API struct ggml_tensor * ggml_sgn_inplace(
#         struct ggml_context * ctx,
#         struct ggml_tensor  * a);
@ggml_function(
    "ggml_sgn_inplace",
    [ggml_context_p_ctypes, ctypes.POINTER(ggml_tensor)],
    ctypes.POINTER(ggml_tensor),
)
def ggml_sgn_inplace(ctx: ggml_context_p, a: ggml_tensor_p, /) -> ggml_tensor_p:
    """Get the sign of all elements in a tensor and store the result in the first tensor.

    Parameters:
        ctx: ggml context
        a: tensor

    Returns:
        Pointer to ggml_tensor"""
    ...


# GGML_API struct ggml_tensor * ggml_neg(
#         struct ggml_context * ctx,
#         struct ggml_tensor  * a);
@ggml_function(
    "ggml_neg",
    [ggml_context_p_ctypes, ctypes.POINTER(ggml_tensor)],
    ctypes.POINTER(ggml_tensor),
)
def ggml_neg(ctx: ggml_context_p, a: ggml_tensor_p, /) -> ggml_tensor_p:
    """Negate all elements in a tensor and return the result.

    Parameters:
        ctx: ggml context
        a: tensor

    Returns:
        Pointer to ggml_tensor"""
    ...


# GGML_API struct ggml_tensor * ggml_neg_inplace(
#         struct ggml_context * ctx,
#         struct ggml_tensor  * a);
@ggml_function(
    "ggml_neg_inplace",
    [ggml_context_p_ctypes, ctypes.POINTER(ggml_tensor)],
    ctypes.POINTER(ggml_tensor),
)
def ggml_neg_inplace(ctx: ggml_context_p, a: ggml_tensor_p, /) -> ggml_tensor_p:
    """Negate all elements in a tensor and store the result in the first tensor.

    Parameters:
        ctx: ggml context
        a: tensor

    Returns:
        Pointer to ggml_tensor"""
    ...


# GGML_API struct ggml_tensor * ggml_step(
#         struct ggml_context * ctx,
#         struct ggml_tensor  * a);
@ggml_function(
    "ggml_step",
    [ggml_context_p_ctypes, ctypes.POINTER(ggml_tensor)],
    ctypes.POINTER(ggml_tensor),
)
def ggml_step(ctx: ggml_context_p, a: ggml_tensor_p, /) -> ggml_tensor_p:
    ...


# GGML_API struct ggml_tensor * ggml_step_inplace(
#         struct ggml_context * ctx,
#         struct ggml_tensor  * a);
@ggml_function(
    "ggml_step_inplace",
    [ggml_context_p_ctypes, ctypes.POINTER(ggml_tensor)],
    ctypes.POINTER(ggml_tensor),
)
def ggml_step_inplace(ctx: ggml_context_p, a: ggml_tensor_p, /) -> ggml_tensor_p:
    """Apply the step function to all elements in a tensor and store the result in the first tensor.

    Parameters:
        ctx: ggml context
        a: tensor

    Returns:
        Pointer to ggml_tensor"""
    ...


# GGML_API struct ggml_tensor * ggml_tanh(
#         struct ggml_context * ctx,
#         struct ggml_tensor  * a);
@ggml_function(
    "ggml_tanh",
    [ggml_context_p_ctypes, ctypes.POINTER(ggml_tensor)],
    ctypes.POINTER(ggml_tensor),
)
def ggml_tanh(ctx: ggml_context_p, a: ggml_tensor_p, /) -> ggml_tensor_p:
    """Apply the tanh activation function to all elements in a tensor and return the result.

    Parameters:
        ctx: ggml context
        a: tensor

    Returns:
        Pointer to ggml_tensor"""
    ...


# GGML_API struct ggml_tensor * ggml_tanh_inplace(
#         struct ggml_context * ctx,
#         struct ggml_tensor  * a);
@ggml_function(
    "ggml_tanh_inplace",
    [ggml_context_p_ctypes, ctypes.POINTER(ggml_tensor)],
    ctypes.POINTER(ggml_tensor),
)
def ggml_tanh_inplace(ctx: ggml_context_p, a: ggml_tensor_p, /) -> ggml_tensor_p:
    """Apply the tanh activation function to all elements in a tensor and store the result in the first tensor.

    Parameters:
        ctx: ggml context
        a: tensor

    Returns:
        Pointer to ggml_tensor"""
    ...


# GGML_API struct ggml_tensor * ggml_elu(
#         struct ggml_context * ctx,
#         struct ggml_tensor  * a);
@ggml_function(
    "ggml_elu",
    [ggml_context_p_ctypes, ctypes.POINTER(ggml_tensor)],
    ctypes.POINTER(ggml_tensor),
)
def ggml_elu(ctx: ggml_context_p, a: ggml_tensor_p, /) -> ggml_tensor_p:
    """Apply the ELU activation function to all elements in a tensor and return the result.

    Parameters:
        ctx: ggml context
        a: tensor

    Returns:
        Pointer to ggml_tensor"""
    ...


# GGML_API struct ggml_tensor * ggml_elu_inplace(
#         struct ggml_context * ctx,
#         struct ggml_tensor  * a);
@ggml_function(
    "ggml_elu_inplace",
    [ggml_context_p_ctypes, ctypes.POINTER(ggml_tensor)],
    ctypes.POINTER(ggml_tensor),
)
def ggml_elu_inplace(ctx: ggml_context_p, a: ggml_tensor_p, /) -> ggml_tensor_p:
    """Apply the ELU activation function to all elements in a tensor and store the result in the first tensor.

    Parameters:
        ctx: ggml context
        a: tensor

    Returns:
        Pointer to ggml_tensor"""
    ...


# GGML_API struct ggml_tensor * ggml_relu(
#         struct ggml_context * ctx,
#         struct ggml_tensor  * a);
@ggml_function(
    "ggml_relu",
    [ggml_context_p_ctypes, ctypes.POINTER(ggml_tensor)],
    ctypes.POINTER(ggml_tensor),
)
def ggml_relu(ctx: ggml_context_p, a: ggml_tensor_p, /) -> ggml_tensor_p:
    """Apply the ReLU activation function to all elements in a tensor and return the result.

    Parameters:
        ctx: ggml context
        a: tensor

    Returns:
        Pointer to ggml_tensor"""
    ...


# GGML_API struct ggml_tensor * ggml_leaky_relu(
#         struct ggml_context * ctx,
#         struct ggml_tensor  * a, float negative_slope, bool inplace);
@ggml_function(
    "ggml_leaky_relu",
    [
        ggml_context_p_ctypes,
        ctypes.POINTER(ggml_tensor),
        ctypes.c_float,
        ctypes.c_bool,
    ],
    ctypes.POINTER(ggml_tensor),
)
def ggml_leaky_relu(
    ctx: ggml_context_p, a: ggml_tensor_p, negative_slope: float, inplace: bool, /
) -> ggml_tensor_p:
    """Apply the Leaky ReLU activation function to all elements in a tensor and return the result.

    Parameters:
        ctx: ggml context
        a: tensor
        negative_slope: negative slope
        inplace: whether to store the result in the first tensor

    Returns:
        Pointer to ggml_tensor"""
    ...


# GGML_API struct ggml_tensor * ggml_xielu(
#         struct ggml_context * ctx,
#         struct ggml_tensor  * a,
#         float                 alpha_n,
#         float                 alpha_p,
#         float                 beta,
#         float                 eps);
@ggml_function(
    "ggml_xielu",
    [
        ggml_context_p_ctypes,
        ctypes.POINTER(ggml_tensor),
        ctypes.c_float,
        ctypes.c_float,
        ctypes.c_float,
        ctypes.c_float,
    ],
    ctypes.POINTER(ggml_tensor),
)
def ggml_xielu(
    ctx: ggml_context_p,
    a: ggml_tensor_p,
    alpha_n: Union[ctypes.c_float, float],
    alpha_p: Union[ctypes.c_float, float],
    beta: Union[ctypes.c_float, float],
    eps: Union[ctypes.c_float, float],
    /,
) -> ggml_tensor_p:
    ...


# GGML_API struct ggml_tensor * ggml_relu_inplace(
#         struct ggml_context * ctx,
#         struct ggml_tensor  * a);
@ggml_function(
    "ggml_relu_inplace",
    [ggml_context_p_ctypes, ctypes.POINTER(ggml_tensor)],
    ctypes.POINTER(ggml_tensor),
)
def ggml_relu_inplace(ctx: ggml_context_p, a: ggml_tensor_p, /) -> ggml_tensor_p:
    """Apply the ReLU activation function to all elements in a tensor and store the result in the first tensor.

    Parameters:
        ctx: ggml context
        a: tensor

    Returns:
        Pointer to ggml_tensor"""
    ...


# GGML_API struct ggml_tensor * ggml_sigmoid(
#         struct ggml_context * ctx,
#         struct ggml_tensor  * a);
@ggml_function(
    "ggml_sigmoid",
    [ggml_context_p_ctypes, ctypes.POINTER(ggml_tensor)],
    ctypes.POINTER(ggml_tensor),
)
def ggml_sigmoid(ctx: ggml_context_p, a: ggml_tensor_p, /) -> ggml_tensor_p:
    """Apply the Sigmoid activation function to all elements in a tensor and return the result.

    Parameters:
        ctx: ggml context
        a: tensor
        
    Returns:
        Pointer to ggml_tensor"""
    ...


# GGML_API struct ggml_tensor * ggml_sigmoid_inplace(
#         struct ggml_context * ctx,
#         struct ggml_tensor  * a);
@ggml_function(
    "ggml_sigmoid_inplace",
    [ggml_context_p_ctypes, ctypes.POINTER(ggml_tensor)],
    ctypes.POINTER(ggml_tensor),
)
def ggml_sigmoid_inplace(ctx: ggml_context_p, a: ggml_tensor_p, /) -> ggml_tensor_p:
    """Apply the Sigmoid activation function to all elements in a tensor and store the result in the first tensor.

    Parameters:
        ctx: ggml context
        a: tensor
    
    Returns:
        Pointer to ggml_tensor"""
    ...


# GGML_API struct ggml_tensor * ggml_gelu(
#         struct ggml_context * ctx,
#         struct ggml_tensor  * a);
@ggml_function(
    "ggml_gelu",
    [ggml_context_p_ctypes, ctypes.POINTER(ggml_tensor)],
    ctypes.POINTER(ggml_tensor),
)
def ggml_gelu(ctx: ggml_context_p, a: ggml_tensor_p, /) -> ggml_tensor_p:
    """Apply the Gaussian Error Linear Unit activation function to all elements in a tensor and return the result.

    Parameters:
        ctx: ggml context
        a: tensor

    Returns:
        Pointer to ggml_tensor"""
    ...


# GGML_API struct ggml_tensor * ggml_gelu_inplace(
#         struct ggml_context * ctx,
#         struct ggml_tensor  * a);
@ggml_function(
    "ggml_gelu_inplace",
    [ggml_context_p_ctypes, ctypes.POINTER(ggml_tensor)],
    ctypes.POINTER(ggml_tensor),
)
def ggml_gelu_inplace(ctx: ggml_context_p, a: ggml_tensor_p, /) -> ggml_tensor_p:
    """Apply the Gaussian Error Linear Unit activation function to all elements in a tensor and store the result in the first tensor.

    Parameters:
        ctx: ggml context
        a: tensor

    Returns:
        Pointer to ggml_tensor"""
    ...


# GGML_API struct ggml_tensor * ggml_gelu_erf(
#         struct ggml_context * ctx,
#         struct ggml_tensor  * a);
@ggml_function(
    "ggml_gelu_erf",
    [ggml_context_p_ctypes, ctypes.POINTER(ggml_tensor)],
    ctypes.POINTER(ggml_tensor),
)
def ggml_gelu_erf(ctx: ggml_context_p, a: ggml_tensor_p, /) -> ggml_tensor_p:
    """Apply the exact GELU activation function to all elements in a tensor and return the result.

    Parameters:
        ctx: ggml context
        a: tensor

    Returns:
        Pointer to ggml_tensor"""
    ...


# GGML_API struct ggml_tensor * ggml_gelu_erf_inplace(
#         struct ggml_context * ctx,
#         struct ggml_tensor  * a);
@ggml_function(
    "ggml_gelu_erf_inplace",
    [ggml_context_p_ctypes, ctypes.POINTER(ggml_tensor)],
    ctypes.POINTER(ggml_tensor),
)
def ggml_gelu_erf_inplace(ctx: ggml_context_p, a: ggml_tensor_p, /) -> ggml_tensor_p:
    """Apply the exact GELU activation function to all elements in a tensor and store the result in the first tensor.

    Parameters:
        ctx: ggml context
        a: tensor

    Returns:
        Pointer to ggml_tensor"""
    ...


# GGML_API struct ggml_tensor * ggml_gelu_quick(
#         struct ggml_context * ctx,
#         struct ggml_tensor  * a);
@ggml_function(
    "ggml_gelu_quick",
    [ggml_context_p_ctypes, ctypes.POINTER(ggml_tensor)],
    ctypes.POINTER(ggml_tensor),
)
def ggml_gelu_quick(ctx: ggml_context_p, a: ggml_tensor_p, /) -> ggml_tensor_p:
    """Apply the Gaussian Error Linear Unit activation function to all elements in a tensor and return the result.

    Parameters:
        ctx: ggml context
        a: tensor

    Returns:
        Pointer to ggml_tensor"""
    ...


# GGML_API struct ggml_tensor * ggml_gelu_quick_inplace(
#         struct ggml_context * ctx,
#         struct ggml_tensor  * a);
@ggml_function(
    "ggml_gelu_quick_inplace",
    [ggml_context_p_ctypes, ctypes.POINTER(ggml_tensor)],
    ctypes.POINTER(ggml_tensor),
)
def ggml_gelu_quick_inplace(ctx: ggml_context_p, a: ggml_tensor_p, /) -> ggml_tensor_p:
    """Apply the Gaussian Error Linear Unit activation function to all elements in a tensor and store the result in the first tensor.

    Parameters:
        ctx: ggml context
        a: tensor

    Returns:
        Pointer to ggml_tensor"""
    ...


# GGML_API struct ggml_tensor * ggml_silu(
#         struct ggml_context * ctx,
#         struct ggml_tensor  * a);
@ggml_function(
    "ggml_silu",
    [ggml_context_p_ctypes, ctypes.POINTER(ggml_tensor)],
    ctypes.POINTER(ggml_tensor),
)
def ggml_silu(ctx: ggml_context_p, a: ggml_tensor_p, /) -> ggml_tensor_p:
    """Apply the Sigmoid Linear Unit activation function to all elements in a tensor and return the result.

    Parameters:
        ctx: ggml context
        a: tensor

    Returns:
        Pointer to ggml_tensor"""
    ...


# GGML_API struct ggml_tensor * ggml_silu_inplace(
#         struct ggml_context * ctx,
#         struct ggml_tensor  * a);
@ggml_function(
    "ggml_silu_inplace",
    [ggml_context_p_ctypes, ctypes.POINTER(ggml_tensor)],
    ctypes.POINTER(ggml_tensor),
)
def ggml_silu_inplace(ctx: ggml_context_p, a: ggml_tensor_p, /) -> ggml_tensor_p:
    """Apply the Sigmoid Linear Unit activation function to all elements in a tensor and store the result in the first tensor.

    Parameters:
        ctx: ggml context
        a: tensor

    Returns:
        Pointer to ggml_tensor"""
    ...


# // a - x
# // b - dy
# GGML_API struct ggml_tensor * ggml_silu_back(
#         struct ggml_context * ctx,
#         struct ggml_tensor  * a,
#         struct ggml_tensor  * b);
@ggml_function(
    "ggml_silu_back",
    [
        ggml_context_p_ctypes,
        ctypes.POINTER(ggml_tensor),
        ctypes.POINTER(ggml_tensor),
    ],
    ctypes.POINTER(ggml_tensor),
)
def ggml_silu_back(
    ctx: ggml_context_p, a: ggml_tensor_p, b: ggml_tensor_p, /
) -> ggml_tensor_p:
    ...


# // hardswish(x) = x * relu6(x + 3) / 6
# GGML_API struct ggml_tensor * ggml_hardswish(
#         struct ggml_context * ctx,
#         struct ggml_tensor  * a);
@ggml_function(
    "ggml_hardswish",
    [ggml_context_p_ctypes, ctypes.POINTER(ggml_tensor)],
    ctypes.POINTER(ggml_tensor),
)
def ggml_hardswish(ctx: ggml_context_p, a: ggml_tensor_p, /) -> ggml_tensor_p:
    """Apply the Hardswish activation function to all elements in a tensor and return the result.

    Parameters:
        ctx: ggml context
        a: tensor

    Returns:
        Pointer to ggml_tensor"""
    ...


# // hardsigmoid(x) = relu6(x + 3) / 6
# GGML_API struct ggml_tensor * ggml_hardsigmoid(
#         struct ggml_context * ctx,
#         struct ggml_tensor  * a);
@ggml_function(
    "ggml_hardsigmoid",
    [ggml_context_p_ctypes, ctypes.POINTER(ggml_tensor)],
    ctypes.POINTER(ggml_tensor),
)
def ggml_hardsigmoid(ctx: ggml_context_p, a: ggml_tensor_p, /) -> ggml_tensor_p:
    """Apply the Hardsigmoid activation function to all elements in a tensor and return the result.

    Parameters:
        ctx: ggml context
        a: tensor

    Returns:
        Pointer to ggml_tensor"""

    ...


# GGML_API struct ggml_tensor * ggml_exp(
#         struct ggml_context * ctx,
#         struct ggml_tensor  * a);
@ggml_function(
    "ggml_exp",
    [ggml_context_p_ctypes, ctypes.POINTER(ggml_tensor)],
    ctypes.POINTER(ggml_tensor),
)
def ggml_exp(ctx: ggml_context_p, a: ggml_tensor_p, /) -> ggml_tensor_p:
    """Compute the exponential of all elements in a tensor and return the result.

    Parameters:
        ctx: ggml context
        a: tensor

    Returns:
        Pointer to ggml_tensor"""
    ...


# GGML_API struct ggml_tensor * ggml_exp_inplace(
#         struct ggml_context * ctx,
#         struct ggml_tensor  * a);
@ggml_function(
    "ggml_exp_inplace",
    [ggml_context_p_ctypes, ctypes.POINTER(ggml_tensor)],
    ctypes.POINTER(ggml_tensor),
)
def ggml_exp_inplace(ctx: ggml_context_p, a: ggml_tensor_p, /) -> ggml_tensor_p:
    """Compute the exponential of all elements in a tensor and store the result in the first tensor.

    Parameters:
        ctx: ggml context
        a: tensor

    Returns:
        Pointer to ggml_tensor"""
    ...


# GGML_API struct ggml_tensor * ggml_floor(
#         struct ggml_context * ctx,
#         struct ggml_tensor  * a);
@ggml_function(
    "ggml_floor",
    [ggml_context_p_ctypes, ctypes.POINTER(ggml_tensor)],
    ctypes.POINTER(ggml_tensor),
    enabled=hasattr(lib, "ggml_floor"),
)
def ggml_floor(ctx: ggml_context_p, a: ggml_tensor_p, /) -> ggml_tensor_p:
    """Compute the floor of all elements in a tensor and return the result.

    Parameters:
        ctx: ggml context
        a: tensor

    Returns:
        Pointer to ggml_tensor"""
    ...


# GGML_API struct ggml_tensor * ggml_floor_inplace(
#         struct ggml_context * ctx,
#         struct ggml_tensor  * a);
@ggml_function(
    "ggml_floor_inplace",
    [ggml_context_p_ctypes, ctypes.POINTER(ggml_tensor)],
    ctypes.POINTER(ggml_tensor),
    enabled=hasattr(lib, "ggml_floor_inplace"),
)
def ggml_floor_inplace(ctx: ggml_context_p, a: ggml_tensor_p, /) -> ggml_tensor_p:
    """Compute the floor of all elements in a tensor and store the result in the first tensor.

    Parameters:
        ctx: ggml context
        a: tensor

    Returns:
        Pointer to ggml_tensor"""
    ...


# GGML_API struct ggml_tensor * ggml_ceil(
#         struct ggml_context * ctx,
#         struct ggml_tensor  * a);
@ggml_function(
    "ggml_ceil",
    [ggml_context_p_ctypes, ctypes.POINTER(ggml_tensor)],
    ctypes.POINTER(ggml_tensor),
    enabled=hasattr(lib, "ggml_ceil"),
)
def ggml_ceil(ctx: ggml_context_p, a: ggml_tensor_p, /) -> ggml_tensor_p:
    """Compute the ceiling of all elements in a tensor and return the result.

    Parameters:
        ctx: ggml context
        a: tensor

    Returns:
        Pointer to ggml_tensor"""
    ...


# GGML_API struct ggml_tensor * ggml_ceil_inplace(
#         struct ggml_context * ctx,
#         struct ggml_tensor  * a);
@ggml_function(
    "ggml_ceil_inplace",
    [ggml_context_p_ctypes, ctypes.POINTER(ggml_tensor)],
    ctypes.POINTER(ggml_tensor),
    enabled=hasattr(lib, "ggml_ceil_inplace"),
)
def ggml_ceil_inplace(ctx: ggml_context_p, a: ggml_tensor_p, /) -> ggml_tensor_p:
    """Compute the ceiling of all elements in a tensor and store the result in the first tensor.

    Parameters:
        ctx: ggml context
        a: tensor

    Returns:
        Pointer to ggml_tensor"""
    ...


# GGML_API struct ggml_tensor * ggml_round(
#         struct ggml_context * ctx,
#         struct ggml_tensor  * a);
@ggml_function(
    "ggml_round",
    [ggml_context_p_ctypes, ctypes.POINTER(ggml_tensor)],
    ctypes.POINTER(ggml_tensor),
    enabled=hasattr(lib, "ggml_round"),
)
def ggml_round(ctx: ggml_context_p, a: ggml_tensor_p, /) -> ggml_tensor_p:
    """Round all elements in a tensor and return the result.

    Parameters:
        ctx: ggml context
        a: tensor

    Returns:
        Pointer to ggml_tensor"""
    ...


# GGML_API struct ggml_tensor * ggml_round_inplace(
#         struct ggml_context * ctx,
#         struct ggml_tensor  * a);
@ggml_function(
    "ggml_round_inplace",
    [ggml_context_p_ctypes, ctypes.POINTER(ggml_tensor)],
    ctypes.POINTER(ggml_tensor),
    enabled=hasattr(lib, "ggml_round_inplace"),
)
def ggml_round_inplace(ctx: ggml_context_p, a: ggml_tensor_p, /) -> ggml_tensor_p:
    """Round all elements in a tensor and store the result in the first tensor.

    Parameters:
        ctx: ggml context
        a: tensor

    Returns:
        Pointer to ggml_tensor"""
    ...


# GGML_API struct ggml_tensor * ggml_trunc(
#         struct ggml_context * ctx,
#         struct ggml_tensor  * a);
@ggml_function(
    "ggml_trunc",
    [ggml_context_p_ctypes, ctypes.POINTER(ggml_tensor)],
    ctypes.POINTER(ggml_tensor),
    enabled=hasattr(lib, "ggml_trunc"),
)
def ggml_trunc(ctx: ggml_context_p, a: ggml_tensor_p, /) -> ggml_tensor_p:
    """Truncate all elements in a tensor toward zero and return the result.

    Parameters:
        ctx: ggml context
        a: tensor

    Returns:
        Pointer to ggml_tensor"""
    ...


# GGML_API struct ggml_tensor * ggml_trunc_inplace(
#         struct ggml_context * ctx,
#         struct ggml_tensor  * a);
@ggml_function(
    "ggml_trunc_inplace",
    [ggml_context_p_ctypes, ctypes.POINTER(ggml_tensor)],
    ctypes.POINTER(ggml_tensor),
    enabled=hasattr(lib, "ggml_trunc_inplace"),
)
def ggml_trunc_inplace(ctx: ggml_context_p, a: ggml_tensor_p, /) -> ggml_tensor_p:
    """Truncate all elements in a tensor toward zero and store the result in the first tensor.

    Parameters:
        ctx: ggml context
        a: tensor

    Returns:
        Pointer to ggml_tensor"""
    ...


# GGML_API struct ggml_tensor * ggml_glu(
#         struct ggml_context * ctx,
#         struct ggml_tensor  * a,
#         enum ggml_glu_op      op,
#         bool                  swapped);
@ggml_function(
    "ggml_glu",
    [
        ggml_context_p_ctypes,
        ctypes.POINTER(ggml_tensor),
        ctypes.c_int,
        ctypes.c_bool,
    ],
    ctypes.POINTER(ggml_tensor),
)
def ggml_glu(
    ctx: ggml_context_p,
    a: ggml_tensor_p,
    op: Union[ctypes.c_int, int],
    swapped: Union[ctypes.c_bool, bool],
    /,
) -> ggml_tensor_p:
    ...


# GGML_API struct ggml_tensor * ggml_reglu(
#         struct ggml_context * ctx,
#         struct ggml_tensor  * a);
@ggml_function(
    "ggml_reglu",
    [ggml_context_p_ctypes, ctypes.POINTER(ggml_tensor)],
    ctypes.POINTER(ggml_tensor),
)
def ggml_reglu(ctx: ggml_context_p, a: ggml_tensor_p, /) -> ggml_tensor_p:
    ...


# GGML_API struct ggml_tensor * ggml_reglu_swapped(
#         struct ggml_context * ctx,
#         struct ggml_tensor  * a);
@ggml_function(
    "ggml_reglu_swapped",
    [ggml_context_p_ctypes, ctypes.POINTER(ggml_tensor)],
    ctypes.POINTER(ggml_tensor),
)
def ggml_reglu_swapped(ctx: ggml_context_p, a: ggml_tensor_p, /) -> ggml_tensor_p:
    ...


# GGML_API struct ggml_tensor * ggml_geglu(
#         struct ggml_context * ctx,
#         struct ggml_tensor  * a);
@ggml_function(
    "ggml_geglu",
    [ggml_context_p_ctypes, ctypes.POINTER(ggml_tensor)],
    ctypes.POINTER(ggml_tensor),
)
def ggml_geglu(ctx: ggml_context_p, a: ggml_tensor_p, /) -> ggml_tensor_p:
    ...


# GGML_API struct ggml_tensor * ggml_geglu_swapped(
#         struct ggml_context * ctx,
#         struct ggml_tensor  * a);
@ggml_function(
    "ggml_geglu_swapped",
    [ggml_context_p_ctypes, ctypes.POINTER(ggml_tensor)],
    ctypes.POINTER(ggml_tensor),
)
def ggml_geglu_swapped(ctx: ggml_context_p, a: ggml_tensor_p, /) -> ggml_tensor_p:
    ...


# GGML_API struct ggml_tensor * ggml_swiglu(
#         struct ggml_context * ctx,
#         struct ggml_tensor  * a);
@ggml_function(
    "ggml_swiglu",
    [ggml_context_p_ctypes, ctypes.POINTER(ggml_tensor)],
    ctypes.POINTER(ggml_tensor),
)
def ggml_swiglu(ctx: ggml_context_p, a: ggml_tensor_p, /) -> ggml_tensor_p:
    ...


# GGML_API struct ggml_tensor * ggml_swiglu_swapped(
#         struct ggml_context * ctx,
#         struct ggml_tensor  * a);
@ggml_function(
    "ggml_swiglu_swapped",
    [ggml_context_p_ctypes, ctypes.POINTER(ggml_tensor)],
    ctypes.POINTER(ggml_tensor),
)
def ggml_swiglu_swapped(ctx: ggml_context_p, a: ggml_tensor_p, /) -> ggml_tensor_p:
    ...


# GGML_API struct ggml_tensor * ggml_geglu_erf(
#         struct ggml_context * ctx,
#         struct ggml_tensor  * a);
@ggml_function(
    "ggml_geglu_erf",
    [ggml_context_p_ctypes, ctypes.POINTER(ggml_tensor)],
    ctypes.POINTER(ggml_tensor),
)
def ggml_geglu_erf(ctx: ggml_context_p, a: ggml_tensor_p, /) -> ggml_tensor_p:
    ...


# GGML_API struct ggml_tensor * ggml_geglu_erf_swapped(
#         struct ggml_context * ctx,
#         struct ggml_tensor  * a);
@ggml_function(
    "ggml_geglu_erf_swapped",
    [ggml_context_p_ctypes, ctypes.POINTER(ggml_tensor)],
    ctypes.POINTER(ggml_tensor),
)
def ggml_geglu_erf_swapped(ctx: ggml_context_p, a: ggml_tensor_p, /) -> ggml_tensor_p:
    ...


# GGML_API struct ggml_tensor * ggml_geglu_quick(
#         struct ggml_context * ctx,
#         struct ggml_tensor  * a);
@ggml_function(
    "ggml_geglu_quick",
    [ggml_context_p_ctypes, ctypes.POINTER(ggml_tensor)],
    ctypes.POINTER(ggml_tensor),
)
def ggml_geglu_quick(ctx: ggml_context_p, a: ggml_tensor_p, /) -> ggml_tensor_p:
    ...


# GGML_API struct ggml_tensor * ggml_geglu_quick_swapped(
#         struct ggml_context * ctx,
#         struct ggml_tensor  * a);
@ggml_function(
    "ggml_geglu_quick_swapped",
    [ggml_context_p_ctypes, ctypes.POINTER(ggml_tensor)],
    ctypes.POINTER(ggml_tensor),
)
def ggml_geglu_quick_swapped(ctx: ggml_context_p, a: ggml_tensor_p, /) -> ggml_tensor_p:
    ...


# GGML_API struct ggml_tensor * ggml_glu_split(
#         struct ggml_context * ctx,
#          struct ggml_tensor * a,
#          struct ggml_tensor * b,
#          enum ggml_glu_op     op);
@ggml_function(
    "ggml_glu_split",
    [
        ggml_context_p_ctypes,
        ctypes.POINTER(ggml_tensor),
        ctypes.POINTER(ggml_tensor),
        ctypes.c_int,
    ],
    ctypes.POINTER(ggml_tensor),
)
def ggml_glu_split(
    ctx: ggml_context_p,
    a: ggml_tensor_p,
    b: ggml_tensor_p,
    op: Union[ctypes.c_int, int],
    /,
) -> ggml_tensor_p:
    ...


# GGML_API struct ggml_tensor * ggml_reglu_split(
#         struct ggml_context * ctx,
#         struct ggml_tensor  * a,
#         struct ggml_tensor  * b);
@ggml_function(
    "ggml_reglu_split",
    [
        ggml_context_p_ctypes,
        ctypes.POINTER(ggml_tensor),
        ctypes.POINTER(ggml_tensor),
    ],
    ctypes.POINTER(ggml_tensor),
)
def ggml_reglu_split(
    ctx: ggml_context_p,
    a: ggml_tensor_p,
    b: ggml_tensor_p,
    /,
) -> ggml_tensor_p:
    ...


# GGML_API struct ggml_tensor * ggml_geglu_split(
#         struct ggml_context * ctx,
#         struct ggml_tensor  * a,
#         struct ggml_tensor  * b);
@ggml_function(
    "ggml_geglu_split",
    [
        ggml_context_p_ctypes,
        ctypes.POINTER(ggml_tensor),
        ctypes.POINTER(ggml_tensor),
    ],
    ctypes.POINTER(ggml_tensor),
)
def ggml_geglu_split(
    ctx: ggml_context_p,
    a: ggml_tensor_p,
    b: ggml_tensor_p,
    /,
) -> ggml_tensor_p:
    ...


# GGML_API struct ggml_tensor * ggml_swiglu_split(
#         struct ggml_context * ctx,
#         struct ggml_tensor  * a,
#         struct ggml_tensor  * b);
@ggml_function(
    "ggml_swiglu_split",
    [
        ggml_context_p_ctypes,
        ctypes.POINTER(ggml_tensor),
        ctypes.POINTER(ggml_tensor),
    ],
    ctypes.POINTER(ggml_tensor),
)
def ggml_swiglu_split(
    ctx: ggml_context_p,
    a: ggml_tensor_p,
    b: ggml_tensor_p,
    /,
) -> ggml_tensor_p:
    ...


# GGML_API struct ggml_tensor * ggml_geglu_erf_split(
#         struct ggml_context * ctx,
#         struct ggml_tensor  * a,
#         struct ggml_tensor  * b);
@ggml_function(
    "ggml_geglu_erf_split",
    [
        ggml_context_p_ctypes,
        ctypes.POINTER(ggml_tensor),
        ctypes.POINTER(ggml_tensor),
    ],
    ctypes.POINTER(ggml_tensor),
)
def ggml_geglu_erf_split(
    ctx: ggml_context_p,
    a: ggml_tensor_p,
    b: ggml_tensor_p,
    /,
) -> ggml_tensor_p:
    ...


# GGML_API struct ggml_tensor * ggml_geglu_quick_split(
#         struct ggml_context * ctx,
#         struct ggml_tensor  * a,
#         struct ggml_tensor  * b);
@ggml_function(
    "ggml_geglu_quick_split",
    [
        ggml_context_p_ctypes,
        ctypes.POINTER(ggml_tensor),
        ctypes.POINTER(ggml_tensor),
    ],
    ctypes.POINTER(ggml_tensor),
)
def ggml_geglu_quick_split(
    ctx: ggml_context_p,
    a: ggml_tensor_p,
    b: ggml_tensor_p,
    /,
) -> ggml_tensor_p:
    ...


# GGML_API struct ggml_tensor * ggml_swiglu_oai(
#         struct ggml_context * ctx,
#         struct ggml_tensor  * a,
#         struct ggml_tensor  * b,
#         float                 alpha,
#         float                 limit);
@ggml_function(
    "ggml_swiglu_oai",
    [
        ggml_context_p_ctypes,
        ctypes.POINTER(ggml_tensor),
        ctypes.POINTER(ggml_tensor),
        ctypes.c_float,
        ctypes.c_float,
    ],
    ctypes.POINTER(ggml_tensor),
)
def ggml_swiglu_oai(
    ctx: ggml_context_p,
    a: ggml_tensor_p,
    b: ggml_tensor_p,
    alpha: Union[ctypes.c_float, float],
    limit: Union[ctypes.c_float, float],
    /,
) -> ggml_tensor_p:
    ...


# // normalize along rows
# GGML_API struct ggml_tensor * ggml_norm(
#         struct ggml_context * ctx,
#         struct ggml_tensor  * a
#         float                eps);
@ggml_function(
    "ggml_norm",
    [
        ggml_context_p_ctypes,
        ctypes.POINTER(ggml_tensor),
        ctypes.c_float,
    ],
    ctypes.POINTER(ggml_tensor),
)
def ggml_norm(
    ctx: ggml_context_p, a: ggml_tensor_p, eps: Union[ctypes.c_float, float], /
) -> ggml_tensor_p:
    """Normalize all elements in a tensor along the first axis and return the result.

    normalize along rows.

    Parameters:
        ctx: ggml context
        a: tensor
        eps: minimum value to avoid division by zero

    Returns:
        Pointer to ggml_tensor"""
    ...


# GGML_API struct ggml_tensor * ggml_norm_inplace(
#         struct ggml_context * ctx,
#         struct ggml_tensor  * a
#         float                eps);
@ggml_function(
    "ggml_norm_inplace",
    [
        ggml_context_p_ctypes,
        ctypes.POINTER(ggml_tensor),
        ctypes.c_float,
    ],
    ctypes.POINTER(ggml_tensor),
)
def ggml_norm_inplace(
    ctx: ggml_context_p, a: ggml_tensor_p, eps: Union[ctypes.c_float, float], /
) -> ggml_tensor_p:
    """Normalize all elements in a tensor along the first axis and store the result in the first tensor.

    normalize along rows.

    Parameters:
        ctx: ggml context
        a: tensor
        eps: minimum value to avoid division by zero

    Returns:
        Pointer to ggml_tensor"""
    ...


# GGML_API struct ggml_tensor * ggml_rms_norm(
#         struct ggml_context * ctx,
#         struct ggml_tensor  * a,
#         float                 eps);
@ggml_function(
    "ggml_rms_norm",
    [
        ggml_context_p_ctypes,
        ctypes.POINTER(ggml_tensor),
        ctypes.c_float,
    ],
    ctypes.POINTER(ggml_tensor),
)
def ggml_rms_norm(
    ctx: ggml_context_p, a: ggml_tensor_p, eps: Union[ctypes.c_float, float], /
) -> ggml_tensor_p:
    """Compute the RMS norm of a tensor and return the result.

    Parameters:
        ctx: ggml context
        a: tensor
        eps: float

    Returns:
        Pointer to ggml_tensor"""
    ...


# GGML_API struct ggml_tensor * ggml_rms_norm_inplace(
#         struct ggml_context * ctx,
#         struct ggml_tensor  * a,
#         float                 eps);
@ggml_function(
    "ggml_rms_norm_inplace",
    [
        ggml_context_p_ctypes,
        ctypes.POINTER(ggml_tensor),
        ctypes.c_float,
    ],
    ctypes.POINTER(ggml_tensor),
)
def ggml_rms_norm_inplace(
    ctx: ggml_context_p, a: ggml_tensor_p, eps: Union[ctypes.c_float, float], /
) -> ggml_tensor_p:
    ...


# // group normalize along ne0*ne1*n_groups
# // used in stable-diffusion
# GGML_API struct ggml_tensor * ggml_group_norm(
#         struct ggml_context * ctx,
#         struct ggml_tensor  * a,
#         int                   n_groups,
#         float                 eps);
@ggml_function(
    "ggml_group_norm",
    [
        ggml_context_p_ctypes,
        ctypes.POINTER(ggml_tensor),
        ctypes.c_int,
        ctypes.c_float,
    ],
    ctypes.POINTER(ggml_tensor),
)
def ggml_group_norm(
    ctx: ggml_context_p,
    a: ggml_tensor_p,
    n_groups: Union[ctypes.c_int, int],
    eps: Union[ctypes.c_float, float],
    /,
) -> ggml_tensor_p:
    """Group normalize a tensor and return the result.

    Parameters:
        ctx: ggml context
        a: tensor
        n_groups: int
        eps: minimum value to avoid division by zero

    Returns:
        Pointer to ggml_tensor"""
    ...


# GGML_API struct ggml_tensor * ggml_group_norm_inplace(
#         struct ggml_context * ctx,
#         struct ggml_tensor  * a,
#         int                   n_groups,
#         float                 eps);
@ggml_function(
    "ggml_group_norm_inplace",
    [
        ggml_context_p_ctypes,
        ctypes.POINTER(ggml_tensor),
        ctypes.c_int,
        ctypes.c_float,
    ],
    ctypes.POINTER(ggml_tensor),
)
def ggml_group_norm_inplace(
    ctx: ggml_context_p,
    a: ggml_tensor_p,
    n_groups: Union[ctypes.c_int, int],
    eps: Union[ctypes.c_float, float],
    /,
) -> ggml_tensor_p:
    """Group normalize a tensor and store the result in the first tensor.

    Parameters:
        ctx: ggml context
        a: tensor
        n_groups: int
        eps: minimum value to avoid division by zero

    Returns:
        Pointer to ggml_tensor"""
    ...


# GGML_API struct ggml_tensor * ggml_l2_norm(
#         struct ggml_context * ctx,
#         struct ggml_tensor  * a,
#         float                 eps);
@ggml_function(
    "ggml_l2_norm",
    [
        ggml_context_p_ctypes,
        ctypes.POINTER(ggml_tensor),
        ctypes.c_float,
    ],
    ctypes.POINTER(ggml_tensor),
)
def ggml_l2_norm(
    ctx: ggml_context_p, a: ggml_tensor_p, eps: Union[ctypes.c_float, float], /
) -> ggml_tensor_p:
    """L2 normalize a tensor along rows and return the result.

    Parameters:
        ctx: ggml context
        a: tensor
        eps: minimum value to avoid division by zero

    Returns:
        Pointer to ggml_tensor"""
    ...


# GGML_API struct ggml_tensor * ggml_l2_norm_inplace(
#         struct ggml_context * ctx,
#         struct ggml_tensor  * a,
#         float                 eps);
@ggml_function(
    "ggml_l2_norm_inplace",
    [
        ggml_context_p_ctypes,
        ctypes.POINTER(ggml_tensor),
        ctypes.c_float,
    ],
    ctypes.POINTER(ggml_tensor),
)
def ggml_l2_norm_inplace(
    ctx: ggml_context_p, a: ggml_tensor_p, eps: Union[ctypes.c_float, float], /
) -> ggml_tensor_p:
    """L2 normalize a tensor along rows and store the result in the first tensor.

    Parameters:
        ctx: ggml context
        a: tensor
        eps: minimum value to avoid division by zero

    Returns:
        Pointer to ggml_tensor"""
    ...


# // a - x
# // b - dy
# GGML_API struct ggml_tensor * ggml_rms_norm_back(
#         struct ggml_context * ctx,
#         struct ggml_tensor  * a,
#         struct ggml_tensor  * b
#         float                 eps);
@ggml_function(
    "ggml_rms_norm_back",
    [
        ggml_context_p_ctypes,
        ctypes.POINTER(ggml_tensor),
        ctypes.POINTER(ggml_tensor),
        ctypes.c_float,
    ],
    ctypes.POINTER(ggml_tensor),
)
def ggml_rms_norm_back(
    ctx: ggml_context_p,
    a: ggml_tensor_p,
    b: ggml_tensor_p,
    eps: Union[ctypes.c_float, float],
    /,
) -> ggml_tensor_p:
    ...


# // A: k columns, n rows => [ne03, ne02, n, k]
# // B: k columns, m rows  (i.e. we transpose it internally) => [ne03 * x, ne02 * y, m, k]
# // result is n columns, m rows => [ne03 * x, ne02 * y, m, n]
# GGML_API struct ggml_tensor * ggml_mul_mat(
#         struct ggml_context * ctx,
#         struct ggml_tensor  * a,
#         struct ggml_tensor  * b);
@ggml_function(
    "ggml_mul_mat",
    [
        ggml_context_p_ctypes,
        ctypes.POINTER(ggml_tensor),
        ctypes.POINTER(ggml_tensor),
    ],
    ctypes.POINTER(ggml_tensor),
)
def ggml_mul_mat(
    ctx: ggml_context_p, a: ggml_tensor_p, b: ggml_tensor_p, /
) -> ggml_tensor_p:
    """Multiply two matrices and return the result.

    A: k columns, n rows => [ne03, ne02, n, k]
    B: k columns, m rows  (i.e. we transpose it internally) => [ne03 * x, ne02 * y, m, k]
    result is n columns, m rows => [ne03 * x, ne02 * y, m, n]

    Parameters:
        ctx: ggml context
        a: tensor
        b: tensor

    Returns:
        Pointer to ggml_tensor"""
    ...


# // change the precision of a matrix multiplication
# // set to GGML_PREC_F32 for higher precision (useful for phi-2)
# GGML_API void ggml_mul_mat_set_prec(
#         struct ggml_tensor * a,
#         enum ggml_prec       prec);
@ggml_function(
    "ggml_mul_mat_set_prec", [ctypes.POINTER(ggml_tensor), ctypes.c_int], None
)
def ggml_mul_mat_set_prec(a: ggml_tensor_p, prec: Union[ctypes.c_int, int], /) -> None:
    """Change the precision of a matrix multiplication.

    set to GGML_PREC_F32 for higher precision (useful for phi-2)

    Parameters:
        a: tensor
        prec: precision"""
    ...


# GGML_API void ggml_mul_mat_set_hint(
#         struct ggml_tensor * a,
#         enum ggml_op_hint    hint);
@ggml_function(
    "ggml_mul_mat_set_hint",
    [ctypes.POINTER(ggml_tensor), ctypes.c_int],
    None,
)
def ggml_mul_mat_set_hint(a: ggml_tensor_p, hint: Union[ctypes.c_int, int], /) -> None:
    ...


# // indirect matrix multiplication
# GGML_API struct ggml_tensor * ggml_mul_mat_id(
#         struct ggml_context * ctx,
#         struct ggml_tensor  * as,
#         struct ggml_tensor  * b,
#         struct ggml_tensor  * ids);
@ggml_function(
    "ggml_mul_mat_id",
    [
        ggml_context_p_ctypes,
        ctypes.POINTER(ggml_tensor),
        ctypes.POINTER(ggml_tensor),
        ctypes.POINTER(ggml_tensor),
    ],
    ctypes.POINTER(ggml_tensor),
)
def ggml_mul_mat_id(
    ctx: ggml_context_p,
    as_: ggml_tensor_p,
    b: ggml_tensor_p,
    ids: ggml_tensor_p,
    /,
) -> ggml_tensor_p:
    """Multiply two matrices indirectly and return the result.

    Parameters:
        ctx: ggml context
        as_: tensor
        b: tensor
        ids: tensor

    Returns:
        Pointer to ggml_tensor"""
    ...


# // A: m columns, n rows,
# // B: p columns, n rows,
# // result is m columns, p rows
# GGML_API struct ggml_tensor * ggml_out_prod(
#         struct ggml_context * ctx,
#         struct ggml_tensor  * a,
#         struct ggml_tensor  * b);
@ggml_function(
    "ggml_out_prod",
    [
        ggml_context_p_ctypes,
        ctypes.POINTER(ggml_tensor),
        ctypes.POINTER(ggml_tensor),
    ],
    ctypes.POINTER(ggml_tensor),
)
def ggml_out_prod(
    ctx: ggml_context_p, a: ggml_tensor_p, b: ggml_tensor_p, /
) -> ggml_tensor_p:
    """Compute the outer product of two matrices and return the result.

    A: m columns, n rows,
    B: p columns, n rows,
    result is m columns, p rows

    Parameters:
        ctx: ggml context
        a: tensor
        b: tensor

    Returns:
        Pointer to ggml_tensor"""
    ...


# //
# // operations on tensors without backpropagation
# //


# GGML_API struct ggml_tensor * ggml_scale(
#         struct ggml_context * ctx,
#         struct ggml_tensor  * a,
#         float                 s);
@ggml_function(
    "ggml_scale",
    [
        ggml_context_p_ctypes,
        ctypes.POINTER(ggml_tensor),
        ctypes.c_float,
    ],
    ctypes.POINTER(ggml_tensor),
)
def ggml_scale(
    ctx: ggml_context_p, a: ggml_tensor_p, s: Union[ctypes.c_float, float], /
) -> ggml_tensor_p:
    """Scale a tensor by another tensor and return the result.

    Parameters:
        ctx: ggml context
        a: tensor
        s: float

    Returns:
        Pointer to ggml_tensor"""
    ...


# // in-place, returns view(a)
# GGML_API struct ggml_tensor * ggml_scale_inplace(
#         struct ggml_context * ctx,
#         struct ggml_tensor  * a,
#         float                 s);
@ggml_function(
    "ggml_scale_inplace",
    [
        ggml_context_p_ctypes,
        ctypes.POINTER(ggml_tensor),
        ctypes.c_float,
    ],
    ctypes.POINTER(ggml_tensor),
)
def ggml_scale_inplace(
    ctx: ggml_context_p, a: ggml_tensor_p, s: Union[ctypes.c_float, float], /
) -> ggml_tensor_p:
    """Scale a tensor by another tensor and store the result in the first tensor.

    Parameters:
        ctx: ggml context
        a: tensor
        s: float

    Returns:
        Pointer to ggml_tensor"""
    ...


# // x = s * a + b
# GGML_API struct ggml_tensor * ggml_scale_bias(
#         struct ggml_context * ctx,
#         struct ggml_tensor  * a,
#         float                 s,
#         float                 b);
@ggml_function(
    "ggml_scale_bias",
    [
        ggml_context_p_ctypes,
        ctypes.POINTER(ggml_tensor),
        ctypes.c_float,
        ctypes.c_float,
    ],
    ctypes.POINTER(ggml_tensor),
)
def ggml_scale_bias(
    ctx: ggml_context_p,
    a: ggml_tensor_p,
    s: Union[ctypes.c_float, float],
    b: Union[ctypes.c_float, float],
    /,
) -> ggml_tensor_p:
    ...


# GGML_API struct ggml_tensor * ggml_scale_bias_inplace(
#         struct ggml_context * ctx,
#         struct ggml_tensor  * a,
#         float                 s,
#         float                 b);
@ggml_function(
    "ggml_scale_bias_inplace",
    [
        ggml_context_p_ctypes,
        ctypes.POINTER(ggml_tensor),
        ctypes.c_float,
        ctypes.c_float,
    ],
    ctypes.POINTER(ggml_tensor),
)
def ggml_scale_bias_inplace(
    ctx: ggml_context_p,
    a: ggml_tensor_p,
    s: Union[ctypes.c_float, float],
    b: Union[ctypes.c_float, float],
    /,
) -> ggml_tensor_p:
    ...


# // b -> view(a,offset,nb1,nb2,3), return modified a
# GGML_API struct ggml_tensor * ggml_set(
#         struct ggml_context * ctx,
#         struct ggml_tensor  * a,
#         struct ggml_tensor  * b,
#         size_t                nb1,
#         size_t                nb2,
#         size_t                nb3,
#         size_t                offset);
@ggml_function(
    "ggml_set",
    [
        ggml_context_p_ctypes,
        ctypes.POINTER(ggml_tensor),
        ctypes.POINTER(ggml_tensor),
        ctypes.c_size_t,
        ctypes.c_size_t,
        ctypes.c_size_t,
        ctypes.c_size_t,
    ],
    ctypes.POINTER(ggml_tensor),
)
def ggml_set(
    ctx: ggml_context_p,
    a: ggml_tensor_p,
    b: ggml_tensor_p,
    nb1: Union[ctypes.c_size_t, int],
    nb2: Union[ctypes.c_size_t, int],
    nb3: Union[ctypes.c_size_t, int],
    offset: Union[ctypes.c_size_t, int],
    /,
) -> ggml_tensor_p:
    ...


# // b -> view(a,offset,nb1,nb2,3), return view(a)
# GGML_API struct ggml_tensor * ggml_set_inplace(
#         struct ggml_context * ctx,
#         struct ggml_tensor  * a,
#         struct ggml_tensor  * b,
#         size_t                nb1,
#         size_t                nb2,
#         size_t                nb3,
#         size_t                offset);
@ggml_function(
    "ggml_set_inplace",
    [
        ggml_context_p_ctypes,
        ctypes.POINTER(ggml_tensor),
        ctypes.POINTER(ggml_tensor),
        ctypes.c_size_t,
        ctypes.c_size_t,
        ctypes.c_size_t,
        ctypes.c_size_t,
    ],
    ctypes.POINTER(ggml_tensor),
)
def ggml_set_inplace(
    ctx: ggml_context_p,
    a: ggml_tensor_p,
    b: ggml_tensor_p,
    nb1: Union[ctypes.c_size_t, int],
    nb2: Union[ctypes.c_size_t, int],
    nb3: Union[ctypes.c_size_t, int],
    offset: Union[ctypes.c_size_t, int],
    /,
) -> ggml_tensor_p:
    ...


# GGML_API struct ggml_tensor * ggml_set_1d(
#         struct ggml_context * ctx,
#         struct ggml_tensor  * a,
#         struct ggml_tensor  * b,
#         size_t                offset);
@ggml_function(
    "ggml_set_1d",
    [
        ggml_context_p_ctypes,
        ctypes.POINTER(ggml_tensor),
        ctypes.POINTER(ggml_tensor),
        ctypes.c_size_t,
    ],
    ctypes.POINTER(ggml_tensor),
)
def ggml_set_1d(
    ctx: ggml_context_p,
    a: ggml_tensor_p,
    b: ggml_tensor_p,
    offset: Union[ctypes.c_size_t, int],
    /,
) -> ggml_tensor_p:
    ...


# GGML_API struct ggml_tensor * ggml_set_1d_inplace(
#         struct ggml_context * ctx,
#         struct ggml_tensor  * a,
#         struct ggml_tensor  * b,
#         size_t                offset);
@ggml_function(
    "ggml_set_1d_inplace",
    [
        ggml_context_p_ctypes,
        ctypes.POINTER(ggml_tensor),
        ctypes.POINTER(ggml_tensor),
        ctypes.c_size_t,
    ],
    ctypes.POINTER(ggml_tensor),
)
def ggml_set_1d_inplace(
    ctx: ggml_context_p,
    a: ggml_tensor_p,
    b: ggml_tensor_p,
    offset: Union[ctypes.c_size_t, int],
    /,
) -> ggml_tensor_p:
    ...


# // b -> view(a,offset,nb1,nb2,3), return modified a
# GGML_API struct ggml_tensor * ggml_set_2d(
#         struct ggml_context * ctx,
#         struct ggml_tensor  * a,
#         struct ggml_tensor  * b,
#         size_t                nb1,
#         size_t                offset);
@ggml_function(
    "ggml_set_2d",
    [
        ggml_context_p_ctypes,
        ctypes.POINTER(ggml_tensor),
        ctypes.POINTER(ggml_tensor),
        ctypes.c_size_t,
        ctypes.c_size_t,
    ],
    ctypes.POINTER(ggml_tensor),
)
def ggml_set_2d(
    ctx: ggml_context_p,
    a: ggml_tensor_p,
    b: ggml_tensor_p,
    nb1: Union[ctypes.c_size_t, int],
    offset: Union[ctypes.c_size_t, int],
    /,
) -> ggml_tensor_p:
    ...


# // b -> view(a,offset,nb1,nb2,3), return view(a)
# GGML_API struct ggml_tensor * ggml_set_2d_inplace(
#         struct ggml_context * ctx,
#         struct ggml_tensor  * a,
#         struct ggml_tensor  * b,
#         size_t                nb1,
#         size_t                offset);
@ggml_function(
    "ggml_set_2d_inplace",
    [
        ggml_context_p_ctypes,
        ctypes.POINTER(ggml_tensor),
        ctypes.POINTER(ggml_tensor),
        ctypes.c_size_t,
        ctypes.c_size_t,
    ],
    ctypes.POINTER(ggml_tensor),
)
def ggml_set_2d_inplace(
    ctx: ggml_context_p,
    a: ggml_tensor_p,
    b: ggml_tensor_p,
    nb1: Union[ctypes.c_size_t, int],
    offset: Union[ctypes.c_size_t, int],
    /,
) -> ggml_tensor_p:
    ...


# // a -> b, return view(b)
# GGML_API struct ggml_tensor * ggml_cpy(
#         struct ggml_context * ctx,
#         struct ggml_tensor  * a,
#         struct ggml_tensor  * b);
@ggml_function(
    "ggml_cpy",
    [
        ggml_context_p_ctypes,
        ctypes.POINTER(ggml_tensor),
        ctypes.POINTER(ggml_tensor),
    ],
    ctypes.POINTER(ggml_tensor),
)
def ggml_cpy(
    ctx: ggml_context_p, a: ggml_tensor_p, b: ggml_tensor_p, /
) -> ggml_tensor_p:
    ...


# GGML_API struct ggml_tensor * ggml_cast(
#         struct ggml_context * ctx,
#         struct ggml_tensor  * a,
#         enum   ggml_type      type);
@ggml_function(
    "ggml_cast",
    [
        ggml_context_p_ctypes,
        ctypes.POINTER(ggml_tensor),
        ctypes.c_int,
    ],
    ctypes.POINTER(ggml_tensor),
)
def ggml_cast(
    ctx: ggml_context_p, a: ggml_tensor_p, type_: Union[ctypes.c_int, int], /
) -> ggml_tensor_p:
    ...


# // make contiguous
# GGML_API struct ggml_tensor * ggml_cont(
#         struct ggml_context * ctx,
#         struct ggml_tensor  * a);
@ggml_function(
    "ggml_cont",
    [ggml_context_p_ctypes, ctypes.POINTER(ggml_tensor)],
    ctypes.POINTER(ggml_tensor),
)
def ggml_cont(ctx: ggml_context_p, a: ggml_tensor_p, /) -> ggml_tensor_p:
    """Make a tensor contiguous and return the result.

    Parameters:
        ctx: ggml context
        a: tensor

    Returns:
        Pointer to ggml_tensor"""
    ...


# // make contiguous, with new shape
# GGML_API struct ggml_tensor * ggml_cont_1d(
#         struct ggml_context * ctx,
#         struct ggml_tensor  * a,
#         int64_t               ne0);
@ggml_function(
    "ggml_cont_1d",
    [
        ggml_context_p_ctypes,
        ctypes.POINTER(ggml_tensor),
        ctypes.c_int64,
    ],
    ctypes.POINTER(ggml_tensor),
)
def ggml_cont_1d(
    ctx: ggml_context_p, a: ggml_tensor_p, ne0: Union[ctypes.c_int64, int], /
) -> ggml_tensor_p:
    ...


# GGML_API struct ggml_tensor * ggml_cont_2d(
#         struct ggml_context * ctx,
#         struct ggml_tensor  * a,
#         int64_t               ne0,
#         int64_t               ne1);
@ggml_function(
    "ggml_cont_2d",
    [
        ggml_context_p_ctypes,
        ctypes.POINTER(ggml_tensor),
        ctypes.c_int64,
        ctypes.c_int64,
    ],
    ctypes.POINTER(ggml_tensor),
)
def ggml_cont_2d(
    ctx: ggml_context_p,
    a: ggml_tensor_p,
    ne0: Union[ctypes.c_int64, int],
    ne1: Union[ctypes.c_int64, int],
    /,
) -> ggml_tensor_p:
    ...


# GGML_API struct ggml_tensor * ggml_cont_3d(
#         struct ggml_context * ctx,
#         struct ggml_tensor  * a,
#         int64_t               ne0,
#         int64_t               ne1,
#         int64_t               ne2);
@ggml_function(
    "ggml_cont_3d",
    [
        ggml_context_p_ctypes,
        ctypes.POINTER(ggml_tensor),
        ctypes.c_int64,
        ctypes.c_int64,
        ctypes.c_int64,
    ],
    ctypes.POINTER(ggml_tensor),
)
def ggml_cont_3d(
    ctx: ggml_context_p,
    a: ggml_tensor_p,
    ne0: Union[ctypes.c_int64, int],
    ne1: Union[ctypes.c_int64, int],
    ne2: Union[ctypes.c_int64, int],
    /,
) -> ggml_tensor_p:
    ...


# GGML_API struct ggml_tensor * ggml_cont_4d(
#         struct ggml_context * ctx,
#         struct ggml_tensor  * a,
#         int64_t               ne0,
#         int64_t               ne1,
#         int64_t               ne2,
#         int64_t               ne3);
@ggml_function(
    "ggml_cont_4d",
    [
        ggml_context_p_ctypes,
        ctypes.POINTER(ggml_tensor),
        ctypes.c_int64,
        ctypes.c_int64,
        ctypes.c_int64,
        ctypes.c_int64,
    ],
    ctypes.POINTER(ggml_tensor),
)
def ggml_cont_4d(
    ctx: ggml_context_p,
    a: ggml_tensor_p,
    ne0: Union[ctypes.c_int64, int],
    ne1: Union[ctypes.c_int64, int],
    ne2: Union[ctypes.c_int64, int],
    ne3: Union[ctypes.c_int64, int],
    /,
) -> ggml_tensor_p:
    ...


# // return view(a), b specifies the new shape
# // TODO: when we start computing gradient, make a copy instead of view
# GGML_API struct ggml_tensor * ggml_reshape(
#         struct ggml_context * ctx,
#         struct ggml_tensor  * a,
#         struct ggml_tensor  * b);
@ggml_function(
    "ggml_reshape",
    [
        ggml_context_p_ctypes,
        ctypes.POINTER(ggml_tensor),
        ctypes.POINTER(ggml_tensor),
    ],
    ctypes.POINTER(ggml_tensor),
)
def ggml_reshape(
    ctx: ggml_context_p, a: ggml_tensor_p, b: ggml_tensor_p, /
) -> ggml_tensor_p:
    ...


# // return view(a)
# // TODO: when we start computing gradient, make a copy instead of view
# GGML_API struct ggml_tensor * ggml_reshape_1d(
#         struct ggml_context * ctx,
#         struct ggml_tensor  * a,
#         int64_t               ne0);
@ggml_function(
    "ggml_reshape_1d",
    [
        ggml_context_p_ctypes,
        ctypes.POINTER(ggml_tensor),
        ctypes.c_int64,
    ],
    ctypes.POINTER(ggml_tensor),
)
def ggml_reshape_1d(
    ctx: ggml_context_p, a: ggml_tensor_p, ne0: Union[ctypes.c_int64, int], /
) -> ggml_tensor_p:
    ...


# GGML_API struct ggml_tensor * ggml_reshape_2d(
#         struct ggml_context * ctx,
#         struct ggml_tensor  * a,
#         int64_t               ne0,
#         int64_t               ne1);
@ggml_function(
    "ggml_reshape_2d",
    [
        ggml_context_p_ctypes,
        ctypes.POINTER(ggml_tensor),
        ctypes.c_int64,
        ctypes.c_int64,
    ],
    ctypes.POINTER(ggml_tensor),
)
def ggml_reshape_2d(
    ctx: ggml_context_p,
    a: ggml_tensor_p,
    ne0: Union[ctypes.c_int64, int],
    ne1: Union[ctypes.c_int64, int],
    /,
) -> ggml_tensor_p:
    ...


# // return view(a)
# // TODO: when we start computing gradient, make a copy instead of view
# GGML_API struct ggml_tensor * ggml_reshape_3d(
#         struct ggml_context * ctx,
#         struct ggml_tensor  * a,
#         int64_t               ne0,
#         int64_t               ne1,
#         int64_t               ne2);
@ggml_function(
    "ggml_reshape_3d",
    [
        ggml_context_p_ctypes,
        ctypes.POINTER(ggml_tensor),
        ctypes.c_int64,
        ctypes.c_int64,
        ctypes.c_int64,
    ],
    ctypes.POINTER(ggml_tensor),
)
def ggml_reshape_3d(
    ctx: ggml_context_p,
    a: ggml_tensor_p,
    ne0: Union[ctypes.c_int64, int],
    ne1: Union[ctypes.c_int64, int],
    ne2: Union[ctypes.c_int64, int],
    /,
) -> ggml_tensor_p:
    ...


# GGML_API struct ggml_tensor * ggml_reshape_4d(
#         struct ggml_context * ctx,
#         struct ggml_tensor  * a,
#         int64_t               ne0,
#         int64_t               ne1,
#         int64_t               ne2,
#         int64_t               ne3);
@ggml_function(
    "ggml_reshape_4d",
    [
        ggml_context_p_ctypes,
        ctypes.POINTER(ggml_tensor),
        ctypes.c_int64,
        ctypes.c_int64,
        ctypes.c_int64,
        ctypes.c_int64,
    ],
    ctypes.POINTER(ggml_tensor),
)
def ggml_reshape_4d(
    ctx: ggml_context_p,
    a: ggml_tensor_p,
    ne0: Union[ctypes.c_int64, int],
    ne1: Union[ctypes.c_int64, int],
    ne2: Union[ctypes.c_int64, int],
    ne3: Union[ctypes.c_int64, int],
    /,
) -> ggml_tensor_p:
    ...


# // offset in bytes
# GGML_API struct ggml_tensor * ggml_view_1d(
#         struct ggml_context * ctx,
#         struct ggml_tensor  * a,
#         int64_t               ne0,
#         size_t                offset);
@ggml_function(
    "ggml_view_1d",
    [
        ggml_context_p_ctypes,
        ctypes.POINTER(ggml_tensor),
        ctypes.c_int64,
        ctypes.c_size_t,
    ],
    ctypes.POINTER(ggml_tensor),
)
def ggml_view_1d(
    ctx: ggml_context_p,
    a: ggml_tensor_p,
    ne0: Union[ctypes.c_int64, int],
    offset: Union[ctypes.c_size_t, int],
    /,
) -> ggml_tensor_p:
    ...


# GGML_API struct ggml_tensor * ggml_view_2d(
#         struct ggml_context * ctx,
#         struct ggml_tensor  * a,
#         int64_t               ne0,
#         int64_t               ne1,
#         size_t                nb1, // row stride in bytes
#         size_t                offset);
@ggml_function(
    "ggml_view_2d",
    [
        ggml_context_p_ctypes,
        ctypes.POINTER(ggml_tensor),
        ctypes.c_int64,
        ctypes.c_int64,
        ctypes.c_size_t,
        ctypes.c_size_t,
    ],
    ctypes.POINTER(ggml_tensor),
)
def ggml_view_2d(
    ctx: ggml_context_p,
    a: ggml_tensor_p,
    ne0: Union[ctypes.c_int64, int],
    ne1: Union[ctypes.c_int64, int],
    nb1: Union[ctypes.c_size_t, int],
    offset: Union[ctypes.c_size_t, int],
    /,
) -> ggml_tensor_p:
    ...


# GGML_API struct ggml_tensor * ggml_view_3d(
#         struct ggml_context * ctx,
#         struct ggml_tensor  * a,
#         int64_t               ne0,
#         int64_t               ne1,
#         int64_t               ne2,
#         size_t                nb1, // row   stride in bytes
#         size_t                nb2, // slice stride in bytes
#         size_t                offset);
@ggml_function(
    "ggml_view_3d",
    [
        ggml_context_p_ctypes,
        ctypes.POINTER(ggml_tensor),
        ctypes.c_int64,
        ctypes.c_int64,
        ctypes.c_int64,
        ctypes.c_size_t,
        ctypes.c_size_t,
        ctypes.c_size_t,
    ],
    ctypes.POINTER(ggml_tensor),
)
def ggml_view_3d(
    ctx: ggml_context_p,
    a: ggml_tensor_p,
    ne0: Union[ctypes.c_int64, int],
    ne1: Union[ctypes.c_int64, int],
    ne2: Union[ctypes.c_int64, int],
    nb1: Union[ctypes.c_size_t, int],
    nb2: Union[ctypes.c_size_t, int],
    offset: Union[ctypes.c_size_t, int],
    /,
) -> ggml_tensor_p:
    ...


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
@ggml_function(
    "ggml_view_4d",
    [
        ggml_context_p_ctypes,
        ctypes.POINTER(ggml_tensor),
        ctypes.c_int64,
        ctypes.c_int64,
        ctypes.c_int64,
        ctypes.c_int64,
        ctypes.c_size_t,
        ctypes.c_size_t,
        ctypes.c_size_t,
        ctypes.c_size_t,
    ],
    ctypes.POINTER(ggml_tensor),
)
def ggml_view_4d(
    ctx: ggml_context_p,
    a: ggml_tensor_p,
    ne0: Union[ctypes.c_int64, int],
    ne1: Union[ctypes.c_int64, int],
    ne2: Union[ctypes.c_int64, int],
    ne3: Union[ctypes.c_int64, int],
    nb1: Union[ctypes.c_size_t, int],
    nb2: Union[ctypes.c_size_t, int],
    nb3: Union[ctypes.c_size_t, int],
    offset: Union[ctypes.c_size_t, int],
    /,
) -> ggml_tensor_p:
    ...


# GGML_API struct ggml_tensor * ggml_permute(
#         struct ggml_context * ctx,
#         struct ggml_tensor  * a,
#         int                   axis0,
#         int                   axis1,
#         int                   axis2,
#         int                   axis3);
@ggml_function(
    "ggml_permute",
    [
        ggml_context_p_ctypes,
        ctypes.POINTER(ggml_tensor),
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
    ],
    ctypes.POINTER(ggml_tensor),
)
def ggml_permute(
    ctx: ggml_context_p,
    a: ggml_tensor_p,
    axis0: Union[ctypes.c_int, int],
    axis1: Union[ctypes.c_int, int],
    axis2: Union[ctypes.c_int, int],
    axis3: Union[ctypes.c_int, int],
    /,
) -> ggml_tensor_p:
    ...


# // alias for ggml_permute(ctx, a, 1, 0, 2, 3)
# GGML_API struct ggml_tensor * ggml_transpose(
#         struct ggml_context * ctx,
#         struct ggml_tensor  * a);
@ggml_function(
    "ggml_transpose",
    [ggml_context_p_ctypes, ctypes.POINTER(ggml_tensor)],
    ctypes.POINTER(ggml_tensor),
)
def ggml_transpose(ctx: ggml_context_p, a: ggml_tensor_p, /) -> ggml_tensor_p:
    """Transpose *the first two dimensions* of a tensor and return the result.

    alias for `ggml_permute(ctx, a, 1, 0, 2, 3)`

    Parameters:
        ctx: ggml context
        a: tensor

    Returns:
        Pointer to ggml_tensor"""
    ...


# // supports 3D: a->ne[2] == b->ne[1]
# GGML_API struct ggml_tensor * ggml_get_rows(
#         struct ggml_context * ctx,
#         struct ggml_tensor  * a,
#         struct ggml_tensor  * b);
@ggml_function(
    "ggml_get_rows",
    [
        ggml_context_p_ctypes,
        ctypes.POINTER(ggml_tensor),
        ctypes.POINTER(ggml_tensor),
    ],
    ctypes.POINTER(ggml_tensor),
)
def ggml_get_rows(
    ctx: ggml_context_p, a: ggml_tensor_p, b: ggml_tensor_p, /
) -> ggml_tensor_p:
    ...


# GGML_API struct ggml_tensor * ggml_get_rows_back(
#         struct ggml_context * ctx,
#         struct ggml_tensor  * a,
#         struct ggml_tensor  * b,
#         struct ggml_tensor  * c);
@ggml_function(
    "ggml_get_rows_back",
    [
        ggml_context_p_ctypes,
        ctypes.POINTER(ggml_tensor),
        ctypes.POINTER(ggml_tensor),
        ctypes.POINTER(ggml_tensor),
    ],
    ctypes.POINTER(ggml_tensor),
)
def ggml_get_rows_back(
    ctx: ggml_context_p, a: ggml_tensor_p, b: ggml_tensor_p, c: ggml_tensor_p, /
) -> ggml_tensor_p:
    ...


# GGML_API struct ggml_tensor * ggml_set_rows(
#         struct ggml_context * ctx,
#         struct ggml_tensor  * a,
#         struct ggml_tensor  * b,
#         struct ggml_tensor  * c);
@ggml_function(
    "ggml_set_rows",
    [
        ggml_context_p_ctypes,
        ctypes.POINTER(ggml_tensor),
        ctypes.POINTER(ggml_tensor),
        ctypes.POINTER(ggml_tensor),
    ],
    ctypes.POINTER(ggml_tensor),
)
def ggml_set_rows(
    ctx: ggml_context_p,
    a: ggml_tensor_p,
    b: ggml_tensor_p,
    c: ggml_tensor_p,
    /,
) -> ggml_tensor_p:
    ...


# GGML_API struct ggml_tensor * ggml_diag(
#     struct ggml_context     * ctx,
#     struct ggml_tensor      * a);
@ggml_function(
    "ggml_diag",
    [ggml_context_p_ctypes, ctypes.POINTER(ggml_tensor)],
    ctypes.POINTER(ggml_tensor),
)
def ggml_diag(ctx: ggml_context_p, a: ggml_tensor_p, /) -> ggml_tensor_p:
    ...


# // set elements above the diagonal to -INF
# GGML_API struct ggml_tensor * ggml_diag_mask_inf(
#         struct ggml_context * ctx,
#         struct ggml_tensor  * a,
#         int                   n_past);
@ggml_function(
    "ggml_diag_mask_inf",
    [
        ggml_context_p_ctypes,
        ctypes.POINTER(ggml_tensor),
        ctypes.c_int,
    ],
    ctypes.POINTER(ggml_tensor),
)
def ggml_diag_mask_inf(
    ctx: ggml_context_p, a: ggml_tensor_p, n_past: Union[ctypes.c_int, int], /
) -> ggml_tensor_p:
    ...


# // in-place, returns view(a)
# GGML_API struct ggml_tensor * ggml_diag_mask_inf_inplace(
#         struct ggml_context * ctx,
#         struct ggml_tensor  * a,
#         int                   n_past);
@ggml_function(
    "ggml_diag_mask_inf_inplace",
    [
        ggml_context_p_ctypes,
        ctypes.POINTER(ggml_tensor),
        ctypes.c_int,
    ],
    ctypes.POINTER(ggml_tensor),
)
def ggml_diag_mask_inf_inplace(
    ctx: ggml_context_p, a: ggml_tensor_p, n_past: Union[ctypes.c_int, int], /
) -> ggml_tensor_p:
    ...


# // set elements above the diagonal to 0
# GGML_API struct ggml_tensor * ggml_diag_mask_zero(
#         struct ggml_context * ctx,
#         struct ggml_tensor  * a,
#         int                   n_past);
@ggml_function(
    "ggml_diag_mask_zero",
    [
        ggml_context_p_ctypes,
        ctypes.POINTER(ggml_tensor),
        ctypes.c_int,
    ],
    ctypes.POINTER(ggml_tensor),
)
def ggml_diag_mask_zero(
    ctx: ggml_context_p, a: ggml_tensor_p, n_past: Union[ctypes.c_int, int], /
) -> ggml_tensor_p:
    ...


# // in-place, returns view(a)
# GGML_API struct ggml_tensor * ggml_diag_mask_zero_inplace(
#         struct ggml_context * ctx,
#         struct ggml_tensor  * a,
#         int                   n_past);
@ggml_function(
    "ggml_diag_mask_zero_inplace",
    [
        ggml_context_p_ctypes,
        ctypes.POINTER(ggml_tensor),
        ctypes.c_int,
    ],
    ctypes.POINTER(ggml_tensor),
)
def ggml_diag_mask_zero_inplace(
    ctx: ggml_context_p, a: ggml_tensor_p, n_past: Union[ctypes.c_int, int], /
) -> ggml_tensor_p:
    ...


# GGML_API struct ggml_tensor * ggml_soft_max(
#         struct ggml_context * ctx,
#         struct ggml_tensor  * a);
@ggml_function(
    "ggml_soft_max",
    [ggml_context_p_ctypes, ctypes.POINTER(ggml_tensor)],
    ctypes.POINTER(ggml_tensor),
)
def ggml_soft_max(ctx: ggml_context_p, a: ggml_tensor_p, /) -> ggml_tensor_p:
    ...


# // in-place, returns view(a)
# GGML_API struct ggml_tensor * ggml_soft_max_inplace(
#         struct ggml_context * ctx,
#         struct ggml_tensor  * a);
@ggml_function(
    "ggml_soft_max_inplace",
    [ggml_context_p_ctypes, ctypes.POINTER(ggml_tensor)],
    ctypes.POINTER(ggml_tensor),
)
def ggml_soft_max_inplace(ctx: ggml_context_p, a: ggml_tensor_p, /) -> ggml_tensor_p:
    ...


# // fused soft_max(a*scale + mask*(ALiBi slope))
# // mask is optional
# // max_bias = 0.0f for no ALiBi
# GGML_API struct ggml_tensor * ggml_soft_max_ext(
#         struct ggml_context * ctx,
#         struct ggml_tensor  * a,
#         struct ggml_tensor  * mask,
#         float                 scale,
#         float                 max_bias);
@ggml_function(
    "ggml_soft_max_ext",
    [
        ggml_context_p_ctypes,
        ctypes.POINTER(ggml_tensor),
        ctypes.POINTER(ggml_tensor),
        ctypes.c_float,
        ctypes.c_float,
    ],
    ctypes.POINTER(ggml_tensor),
)
def ggml_soft_max_ext(
    ctx: ggml_context_p,
    a: ggml_tensor_p,
    mask: ggml_tensor_p,
    scale: Union[ctypes.c_float, float],
    max_bias: Union[ctypes.c_float, float],
    /,
) -> ggml_tensor_p:
    ...


# GGML_API struct ggml_tensor * ggml_soft_max_ext_inplace(
#         struct ggml_context * ctx,
#         struct ggml_tensor  * a,
#         struct ggml_tensor  * mask,
#         float                 scale,
#         float                 max_bias);
@ggml_function(
    "ggml_soft_max_ext_inplace",
    [
        ggml_context_p_ctypes,
        ctypes.POINTER(ggml_tensor),
        ctypes.POINTER(ggml_tensor),
        ctypes.c_float,
        ctypes.c_float,
    ],
    ctypes.POINTER(ggml_tensor),
)
def ggml_soft_max_ext_inplace(
    ctx: ggml_context_p,
    a: ggml_tensor_p,
    mask: ggml_tensor_p,
    scale: Union[ctypes.c_float, float],
    max_bias: Union[ctypes.c_float, float],
    /,
) -> ggml_tensor_p:
    ...


# GGML_API void ggml_soft_max_add_sinks(
#         struct ggml_tensor * a,
#         struct ggml_tensor * sinks);
@ggml_function(
    "ggml_soft_max_add_sinks",
    [ctypes.POINTER(ggml_tensor), ctypes.POINTER(ggml_tensor)],
    None,
)
def ggml_soft_max_add_sinks(
    a: ggml_tensor_p,
    sinks: ggml_tensor_p,
    /,
) -> None:
    ...


# GGML_API struct ggml_tensor * ggml_soft_max_ext_back(
#         struct ggml_context * ctx,
#         struct ggml_tensor  * a,
#         struct ggml_tensor  * b,
#         float                 scale,
#         float                 max_bias);
@ggml_function(
    "ggml_soft_max_ext_back",
    [
        ggml_context_p_ctypes,
        ctypes.POINTER(ggml_tensor),
        ctypes.POINTER(ggml_tensor),
        ctypes.c_float,
        ctypes.c_float,
    ],
    ctypes.POINTER(ggml_tensor),
)
def ggml_soft_max_ext_back(
    ctx: ggml_context_p,
    a: ggml_tensor_p,
    b: ggml_tensor_p,
    scale: Union[ctypes.c_float, float],
    max_bias: Union[ctypes.c_float, float],
    /,
) -> ggml_tensor_p:
    ...


# GGML_API struct ggml_tensor * ggml_soft_max_ext_back_inplace(
#         struct ggml_context * ctx,
#         struct ggml_tensor  * a,
#         struct ggml_tensor  * b,
#         float                 scale,
#         float                 max_bias);
@ggml_function(
    "ggml_soft_max_ext_back_inplace",
    [
        ggml_context_p_ctypes,
        ctypes.POINTER(ggml_tensor),
        ctypes.POINTER(ggml_tensor),
        ctypes.c_float,
        ctypes.c_float,
    ],
    ctypes.POINTER(ggml_tensor),
)
def ggml_soft_max_ext_back_inplace(
    ctx: ggml_context_p,
    a: ggml_tensor_p,
    b: ggml_tensor_p,
    scale: Union[ctypes.c_float, float],
    max_bias: Union[ctypes.c_float, float],
    /,
) -> ggml_tensor_p:
    ...


# GGML_API struct ggml_tensor * ggml_soft_max_back(
#         struct ggml_context * ctx,
#         struct ggml_tensor  * a,
#         struct ggml_tensor  * b);
@ggml_function(
    "ggml_soft_max_back",
    [
        ggml_context_p_ctypes,
        ctypes.POINTER(ggml_tensor),
        ctypes.POINTER(ggml_tensor),
    ],
    ctypes.POINTER(ggml_tensor),
    enabled=hasattr(lib, "ggml_soft_max_back"),
)
def ggml_soft_max_back(
    ctx: ggml_context_p, a: ggml_tensor_p, b: ggml_tensor_p, /
) -> ggml_tensor_p:
    ...


# // in-place, returns view(a)
# GGML_API struct ggml_tensor * ggml_soft_max_back_inplace(
#         struct ggml_context * ctx,
#         struct ggml_tensor  * a,
#         struct ggml_tensor  * b);
@ggml_function(
    "ggml_soft_max_back_inplace",
    [
        ggml_context_p_ctypes,
        ctypes.POINTER(ggml_tensor),
        ctypes.POINTER(ggml_tensor),
    ],
    ctypes.POINTER(ggml_tensor),
    enabled=hasattr(lib, "ggml_soft_max_back_inplace"),
)
def ggml_soft_max_back_inplace(
    ctx: ggml_context_p, a: ggml_tensor_p, b: ggml_tensor_p, /
) -> ggml_tensor_p:
    ...


# // rotary position embedding
# // if mode & 1 == 1, skip n_past elements (NOT SUPPORTED)
# // if mode & 2 == 1, GPT-NeoX style
# // if mode & 4 == 1, ChatGLM style
# //
# // b is an int32 vector with size a->ne[2], it contains the positions
# // c is freq factors (e.g. phi3-128k), (optional)
# GGML_API struct ggml_tensor * ggml_rope(
#         struct ggml_context * ctx,
#         struct ggml_tensor  * a,
#         struct ggml_tensor  * b,
#         int                   n_dims,
#         int                   mode);
@ggml_function(
    "ggml_rope",
    [
        ggml_context_p_ctypes,
        ctypes.POINTER(ggml_tensor),
        ctypes.POINTER(ggml_tensor),
        ctypes.c_int,
        ctypes.c_int,
    ],
    ctypes.POINTER(ggml_tensor),
)
def ggml_rope(
    ctx: ggml_context_p,
    a: ggml_tensor_p,
    b: ggml_tensor_p,
    n_dims: Union[ctypes.c_int, int],
    mode: Union[ctypes.c_int, int],
    /,
) -> ggml_tensor_p:
    """Rotary position embedding

    Parameters:
        ctx: ggml context
        a: tensor
        b: int32 vector with size a->ne[2], it contains the positions
        n_dims: number of dimensions
        mode: if mode & 1 == 1, skip n_past elements (DEPRECATED)
                if mode & 2 == 1, GPT-NeoX style
                if mode & 4 == 1, ChatGLM style

    Returns:
        Pointer to ggml_tensor"""
    ...


# // in-place, returns view(a)
# GGML_API struct ggml_tensor * ggml_rope_inplace(
#         struct ggml_context * ctx,
#         struct ggml_tensor  * a,
#         struct ggml_tensor  * b,
#         int                   n_dims,
#         int                   mode);
@ggml_function(
    "ggml_rope_inplace",
    [
        ggml_context_p_ctypes,
        ctypes.POINTER(ggml_tensor),
        ctypes.POINTER(ggml_tensor),
        ctypes.c_int,
        ctypes.c_int,
    ],
    ctypes.POINTER(ggml_tensor),
)
def ggml_rope_inplace(
    ctx: ggml_context_p,
    a: ggml_tensor_p,
    b: ggml_tensor_p,
    n_dims: Union[ctypes.c_int, int],
    mode: Union[ctypes.c_int, int],
    /,
) -> ggml_tensor_p:
    """Rotary position embedding inplace

    Parameters:
        ctx: ggml context
        a: tensor
        b: int32 vector with size a->ne[2], it contains the positions
        n_dims: number of dimensions
        mode: if mode & 1 == 1, skip n_past elements (DEPRECATED)
                if mode & 2 == 1, GPT-NeoX style
                if mode & 4 == 1, ChatGLM style

    Returns:
        Pointer to ggml_tensor"""
    ...


# // custom RoPE
# GGML_API struct ggml_tensor * ggml_rope_ext(
#         struct ggml_context * ctx,
#         struct ggml_tensor  * a,
#         struct ggml_tensor  * b,
#         struct ggml_tensor  * c,
#         int                   n_dims,
#         int                   mode,
#         int                   n_ctx_orig,
#         float                 freq_base,
#         float                 freq_scale,
#         float                 ext_factor,
#         float                 attn_factor,
#         float                 beta_fast,
#         float                 beta_slow);
@ggml_function(
    "ggml_rope_ext",
    [
        ggml_context_p_ctypes,
        ctypes.POINTER(ggml_tensor),
        ctypes.POINTER(ggml_tensor),
        ctypes.POINTER(ggml_tensor),
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_float,
        ctypes.c_float,
        ctypes.c_float,
        ctypes.c_float,
        ctypes.c_float,
        ctypes.c_float,
    ],
    ctypes.POINTER(ggml_tensor),
)
def ggml_rope_ext(
    ctx: ggml_context_p,
    a: ggml_tensor_p,
    b: ggml_tensor_p,
    c: ggml_tensor_p,
    n_dims: Union[ctypes.c_int, int],
    mode: Union[ctypes.c_int, int],
    n_ctx_orig: Union[ctypes.c_int, int],
    freq_base: Union[ctypes.c_float, float],
    freq_scale: Union[ctypes.c_float, float],
    ext_factor: Union[ctypes.c_float, float],
    attn_factor: Union[ctypes.c_float, float],
    beta_fast: Union[ctypes.c_float, float],
    beta_slow: Union[ctypes.c_float, float],
    /,
) -> ggml_tensor_p:
    ...


# // in-place, returns view(a)
# GGML_API struct ggml_tensor * ggml_rope_ext_inplace(
#         struct ggml_context * ctx,
#         struct ggml_tensor  * a,
#         struct ggml_tensor  * b,
#         struct ggml_tensor  * c,
#         int                   n_dims,
#         int                   mode,
#         int                   n_ctx_orig,
#         float                 freq_base,
#         float                 freq_scale,
#         float                 ext_factor,
#         float                 attn_factor,
#         float                 beta_fast,
#         float                 beta_slow);
@ggml_function(
    "ggml_rope_ext_inplace",
    [
        ggml_context_p_ctypes,
        ctypes.POINTER(ggml_tensor),
        ctypes.POINTER(ggml_tensor),
        ctypes.POINTER(ggml_tensor),
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_float,
        ctypes.c_float,
        ctypes.c_float,
        ctypes.c_float,
        ctypes.c_float,
        ctypes.c_float,
    ],
    ctypes.POINTER(ggml_tensor),
)
def ggml_rope_ext_inplace(
    ctx: ggml_context_p,
    a: ggml_tensor_p,
    b: ggml_tensor_p,
    c: ggml_tensor_p,
    n_dims: Union[ctypes.c_int, int],
    mode: Union[ctypes.c_int, int],
    n_ctx_orig: Union[ctypes.c_int, int],
    freq_base: Union[ctypes.c_float, float],
    freq_scale: Union[ctypes.c_float, float],
    ext_factor: Union[ctypes.c_float, float],
    attn_factor: Union[ctypes.c_float, float],
    beta_fast: Union[ctypes.c_float, float],
    beta_slow: Union[ctypes.c_float, float],
    /,
) -> ggml_tensor_p:
    ...


# GGML_API struct ggml_tensor * ggml_rope_multi(
#         struct ggml_context * ctx,
#         struct ggml_tensor  * a,
#         struct ggml_tensor  * b,
#         struct ggml_tensor  * c,
#         int                   n_dims,
#         int                   sections[GGML_MROPE_SECTIONS],
#         int                   mode,
#         int                   n_ctx_orig,
#         float                 freq_base,
#         float                 freq_scale,
#         float                 ext_factor,
#         float                 attn_factor,
#         float                 beta_fast,
#         float                 beta_slow);
@ggml_function(
    "ggml_rope_multi",
    [
        ggml_context_p_ctypes,
        ctypes.POINTER(ggml_tensor),
        ctypes.POINTER(ggml_tensor),
        ctypes.POINTER(ggml_tensor),
        ctypes.c_int,
        ctypes.POINTER(ctypes.c_int),
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_float,
        ctypes.c_float,
        ctypes.c_float,
        ctypes.c_float,
        ctypes.c_float,
        ctypes.c_float,
    ],
    ctypes.POINTER(ggml_tensor),
)
def ggml_rope_multi(
    ctx: ggml_context_p,
    a: ggml_tensor_p,
    b: ggml_tensor_p,
    c: ggml_tensor_p,
    n_dims: Union[ctypes.c_int, int],
    sections: CtypesPointer[ctypes.c_int],
    mode: Union[ctypes.c_int, int],
    n_ctx_orig: Union[ctypes.c_int, int],
    freq_base: Union[ctypes.c_float, float],
    freq_scale: Union[ctypes.c_float, float],
    ext_factor: Union[ctypes.c_float, float],
    attn_factor: Union[ctypes.c_float, float],
    beta_fast: Union[ctypes.c_float, float],
    beta_slow: Union[ctypes.c_float, float],
    /,
) -> ggml_tensor_p:
    ...


# GGML_API struct ggml_tensor * ggml_rope_multi_inplace(
#         struct ggml_context * ctx,
#         struct ggml_tensor  * a,
#         struct ggml_tensor  * b,
#         struct ggml_tensor  * c,
#         int                   n_dims,
#         int                   sections[GGML_MROPE_SECTIONS],
#         int                   mode,
#         int                   n_ctx_orig,
#         float                 freq_base,
#         float                 freq_scale,
#         float                 ext_factor,
#         float                 attn_factor,
#         float                 beta_fast,
#         float                 beta_slow);
@ggml_function(
    "ggml_rope_multi_inplace",
    [
        ggml_context_p_ctypes,
        ctypes.POINTER(ggml_tensor),
        ctypes.POINTER(ggml_tensor),
        ctypes.POINTER(ggml_tensor),
        ctypes.c_int,
        ctypes.POINTER(ctypes.c_int),
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_float,
        ctypes.c_float,
        ctypes.c_float,
        ctypes.c_float,
        ctypes.c_float,
        ctypes.c_float,
    ],
    ctypes.POINTER(ggml_tensor),
)
def ggml_rope_multi_inplace(
    ctx: ggml_context_p,
    a: ggml_tensor_p,
    b: ggml_tensor_p,
    c: ggml_tensor_p,
    n_dims: Union[ctypes.c_int, int],
    sections: CtypesPointer[ctypes.c_int],
    mode: Union[ctypes.c_int, int],
    n_ctx_orig: Union[ctypes.c_int, int],
    freq_base: Union[ctypes.c_float, float],
    freq_scale: Union[ctypes.c_float, float],
    ext_factor: Union[ctypes.c_float, float],
    attn_factor: Union[ctypes.c_float, float],
    beta_fast: Union[ctypes.c_float, float],
    beta_slow: Union[ctypes.c_float, float],
    /,
) -> ggml_tensor_p:
    ...


# GGML_DEPRECATED(GGML_API struct ggml_tensor * ggml_rope_custom(
#         struct ggml_context * ctx,
#         struct ggml_tensor  * a,
#         struct ggml_tensor  * b,
#         int                   n_dims,
#         int                   mode,
#         int                   n_ctx_orig,
#         float                 freq_base,
#         float                 freq_scale,
#         float                 ext_factor,
#         float                 attn_factor,
#         float                 beta_fast,
#         float                 beta_slow),
#     "use ggml_rope_ext instead");
@ggml_function(
    "ggml_rope_custom",
    [
        ggml_context_p_ctypes,
        ctypes.POINTER(ggml_tensor),
        ctypes.POINTER(ggml_tensor),
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_float,
        ctypes.c_float,
        ctypes.c_float,
        ctypes.c_float,
        ctypes.c_float,
        ctypes.c_float,
    ],
    ctypes.POINTER(ggml_tensor),
)
def ggml_rope_custom(
    ctx: ggml_context_p,
    a: ggml_tensor_p,
    b: ggml_tensor_p,
    n_dims: Union[ctypes.c_int, int],
    mode: Union[ctypes.c_int, int],
    n_ctx_orig: Union[ctypes.c_int, int],
    freq_base: Union[ctypes.c_float, float],
    freq_scale: Union[ctypes.c_float, float],
    ext_factor: Union[ctypes.c_float, float],
    attn_factor: Union[ctypes.c_float, float],
    beta_fast: Union[ctypes.c_float, float],
    beta_slow: Union[ctypes.c_float, float],
    /,
) -> ggml_tensor_p:
    """Custom rotary position embedding"""
    ...


# GGML_DEPRECATED(GGML_API struct ggml_tensor * ggml_rope_custom_inplace(
#         struct ggml_context * ctx,
#         struct ggml_tensor  * a,
#         struct ggml_tensor  * b,
#         int                   n_dims,
#         int                   mode,
#         int                   n_ctx_orig,
#         float                 freq_base,
#         float                 freq_scale,
#         float                 ext_factor,
#         float                 attn_factor,
#         float                 beta_fast,
#         float                 beta_slow),
#     "use ggml_rope_ext_inplace instead");
@ggml_function(
    "ggml_rope_custom_inplace",
    [
        ggml_context_p_ctypes,
        ctypes.POINTER(ggml_tensor),
        ctypes.POINTER(ggml_tensor),
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_float,
        ctypes.c_float,
        ctypes.c_float,
        ctypes.c_float,
        ctypes.c_float,
        ctypes.c_float,
    ],
    ctypes.POINTER(ggml_tensor),
)
def ggml_rope_custom_inplace(
    ctx: ggml_context_p,
    a: ggml_tensor_p,
    b: ggml_tensor_p,
    n_dims: Union[ctypes.c_int, int],
    mode: Union[ctypes.c_int, int],
    n_ctx_orig: Union[ctypes.c_int, int],
    freq_base: Union[ctypes.c_float, float],
    freq_scale: Union[ctypes.c_float, float],
    ext_factor: Union[ctypes.c_float, float],
    attn_factor: Union[ctypes.c_float, float],
    beta_fast: Union[ctypes.c_float, float],
    beta_slow: Union[ctypes.c_float, float],
    /,
) -> ggml_tensor_p:
    """Custom rotary position embedding inplace"""
    ...


# // compute correction dims for YaRN RoPE scaling
# GGML_CALL void ggml_rope_yarn_corr_dims(
#     int n_dims, int n_orig_ctx, float freq_base, float beta_fast, float beta_slow, float dims[2]);
@ggml_function(
    "ggml_rope_yarn_corr_dims",
    [
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_float,
        ctypes.c_float,
        ctypes.c_float,
        ctypes.POINTER(ctypes.c_float),
    ],
    None,
)
def ggml_rope_yarn_corr_dims(
    n_dims: Union[ctypes.c_int, int],
    n_orig_ctx: Union[ctypes.c_int, int],
    freq_base: Union[ctypes.c_float, float],
    beta_fast: Union[ctypes.c_float, float],
    beta_slow: Union[ctypes.c_float, float],
    dims: CtypesArray[ctypes.c_float],
    /,
) -> None:
    """Compute correction dims for YaRN RoPE scaling"""
    ...


# // rotary position embedding backward, i.e compute dx from dy
# // a - dy
# GGML_API struct ggml_tensor * ggml_rope_back(
#         struct ggml_context * ctx,
#         struct ggml_tensor  * a,
#         struct ggml_tensor  * b,
#         struct ggml_tensor  * c,
#         int                   n_dims,
#         int                   mode,
#         int                   n_ctx,
#         int                   n_orig_ctx,
#         float                 freq_base,
#         float                 freq_scale,
#         float                 ext_factor,
#         float                 attn_factor,
#         float                 beta_fast,
#         float                 beta_slow,
#         float                 xpos_base,
#         bool                  xpos_down);
@ggml_function(
    "ggml_rope_back",
    [
        ggml_context_p_ctypes,
        ctypes.POINTER(ggml_tensor),
        ctypes.POINTER(ggml_tensor),
        ctypes.POINTER(ggml_tensor),
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_float,
        ctypes.c_float,
        ctypes.c_float,
        ctypes.c_float,
        ctypes.c_float,
        ctypes.c_float,
        ctypes.c_float,
        ctypes.c_bool,
    ],
    ctypes.POINTER(ggml_tensor),
    enabled=hasattr(lib, "ggml_rope_back"),
)
def ggml_rope_back(
    ctx: ggml_context_p,
    a: ggml_tensor_p,
    b: ggml_tensor_p,
    c: ggml_tensor_p,
    n_dims: Union[ctypes.c_int, int],
    mode: Union[ctypes.c_int, int],
    n_ctx: Union[ctypes.c_int, int],
    n_orig_ctx: Union[ctypes.c_int, int],
    freq_base: Union[ctypes.c_float, float],
    freq_scale: Union[ctypes.c_float, float],
    ext_factor: Union[ctypes.c_float, float],
    attn_factor: Union[ctypes.c_float, float],
    beta_fast: Union[ctypes.c_float, float],
    beta_slow: Union[ctypes.c_float, float],
    xpos_base: Union[ctypes.c_float, float],
    xpos_down: Union[ctypes.c_bool, bool],
    /,
) -> ggml_tensor_p:
    """Rotary position embedding backward pass"""
    ...


# GGML_API struct ggml_tensor * ggml_rope_ext_back(
#         struct ggml_context * ctx,
#         struct ggml_tensor  * a,
#         struct ggml_tensor  * b,
#         struct ggml_tensor  * c,
#         int                   n_dims,
#         int                   mode,
#         int                   n_ctx_orig,
#         float                 freq_base,
#         float                 freq_scale,
#         float                 ext_factor,
#         float                 attn_factor,
#         float                 beta_fast,
#         float                 beta_slow);
@ggml_function(
    "ggml_rope_ext_back",
    [
        ggml_context_p_ctypes,
        ctypes.POINTER(ggml_tensor),
        ctypes.POINTER(ggml_tensor),
        ctypes.POINTER(ggml_tensor),
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_float,
        ctypes.c_float,
        ctypes.c_float,
        ctypes.c_float,
        ctypes.c_float,
        ctypes.c_float,
    ],
    ctypes.POINTER(ggml_tensor),
)
def ggml_rope_ext_back(
    ctx: ggml_context_p,
    a: ggml_tensor_p,
    b: ggml_tensor_p,
    c: ggml_tensor_p,
    n_dims: Union[ctypes.c_int, int],
    mode: Union[ctypes.c_int, int],
    n_ctx_orig: Union[ctypes.c_int, int],
    freq_base: Union[ctypes.c_float, float],
    freq_scale: Union[ctypes.c_float, float],
    ext_factor: Union[ctypes.c_float, float],
    attn_factor: Union[ctypes.c_float, float],
    beta_fast: Union[ctypes.c_float, float],
    beta_slow: Union[ctypes.c_float, float],
    /,
) -> ggml_tensor_p:
    ...


# GGML_API struct ggml_tensor * ggml_rope_multi_back(
#         struct ggml_context * ctx,
#         struct ggml_tensor  * a,
#         struct ggml_tensor  * b,
#         struct ggml_tensor  * c,
#         int                   n_dims,
#         int                   sections[4],
#         int                   mode,
#         int                   n_ctx_orig,
#         float                 freq_base,
#         float                 freq_scale,
#         float                 ext_factor,
#         float                 attn_factor,
#         float                 beta_fast,
#         float                 beta_slow);
@ggml_function(
    "ggml_rope_multi_back",
    [
        ggml_context_p_ctypes,
        ctypes.POINTER(ggml_tensor),
        ctypes.POINTER(ggml_tensor),
        ctypes.POINTER(ggml_tensor),
        ctypes.c_int,
        ctypes.POINTER(ctypes.c_int),
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_float,
        ctypes.c_float,
        ctypes.c_float,
        ctypes.c_float,
        ctypes.c_float,
        ctypes.c_float,
    ],
    ctypes.POINTER(ggml_tensor),
)
def ggml_rope_multi_back(
    ctx: ggml_context_p,
    a: ggml_tensor_p,
    b: ggml_tensor_p,
    c: ggml_tensor_p,
    n_dims: Union[ctypes.c_int, int],
    sections: CtypesPointer[ctypes.c_int],
    mode: Union[ctypes.c_int, int],
    n_ctx_orig: Union[ctypes.c_int, int],
    freq_base: Union[ctypes.c_float, float],
    freq_scale: Union[ctypes.c_float, float],
    ext_factor: Union[ctypes.c_float, float],
    attn_factor: Union[ctypes.c_float, float],
    beta_fast: Union[ctypes.c_float, float],
    beta_slow: Union[ctypes.c_float, float],
    /,
) -> ggml_tensor_p:
    ...


# // clamp
# // in-place, returns view(a)
# GGML_API struct ggml_tensor * ggml_clamp(
#         struct ggml_context * ctx,
#         struct ggml_tensor  * a,
#         float                 min,
#         float                 max);
@ggml_function(
    "ggml_clamp",
    [
        ggml_context_p_ctypes,
        ctypes.POINTER(ggml_tensor),
        ctypes.c_float,
        ctypes.c_float,
    ],
    ctypes.POINTER(ggml_tensor),
)
def ggml_clamp(
    ctx: ggml_context_p,
    a: ggml_tensor_p,
    min: Union[ctypes.c_float, float],
    max: Union[ctypes.c_float, float],
    /,
) -> ggml_tensor_p:
    """Clamp tensor values between min and max

    Parameters:
        ctx: ggml context
        a: tensor
        min: minimum value
        max: maximum value

    Returns:
        Pointer to ggml_tensor"""
    ...


# GGML_API struct ggml_tensor * ggml_im2col(
#         struct ggml_context * ctx,
#         struct ggml_tensor  * a,
#         struct ggml_tensor  * b,
#         int                  s0,
#         int                  s1,
#         int                  p0,
#         int                  p1,
#         int                  d0,
#         int                  d1,
#         bool                 is_2D,
#         enum ggml_type       dst_type);
@ggml_function(
    "ggml_im2col",
    [
        ggml_context_p_ctypes,
        ctypes.POINTER(ggml_tensor),
        ctypes.POINTER(ggml_tensor),
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_bool,
        ctypes.c_int,
    ],
    ctypes.POINTER(ggml_tensor),
)
def ggml_im2col(
    ctx: ggml_context_p,
    a: ggml_tensor_p,
    b: ggml_tensor_p,
    s0: Union[ctypes.c_int, int],
    s1: Union[ctypes.c_int, int],
    p0: Union[ctypes.c_int, int],
    p1: Union[ctypes.c_int, int],
    d0: Union[ctypes.c_int, int],
    d1: Union[ctypes.c_int, int],
    is_2D: Union[ctypes.c_bool, bool],
    dst_type: Union[ctypes.c_int, int],
    /,
) -> ggml_tensor_p:
    ...


# GGML_API struct ggml_tensor * ggml_im2col_back(
#         struct ggml_context * ctx,
#         struct ggml_tensor  * a,
#         struct ggml_tensor  * b,
#         int64_t             * ne,
#         int                   s0,
#         int                   s1,
#         int                   p0,
#         int                   p1,
#         int                   d0,
#         int                   d1,
#         bool                  is_2D);
@ggml_function(
    "ggml_im2col_back",
    [
        ggml_context_p_ctypes,
        ctypes.POINTER(ggml_tensor),
        ctypes.POINTER(ggml_tensor),
        ctypes.POINTER(ctypes.c_int64),
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_bool,
    ],
    ctypes.POINTER(ggml_tensor),
)
def ggml_im2col_back(
    ctx: ggml_context_p,
    a: ggml_tensor_p,
    b: ggml_tensor_p,
    ne: CtypesPointer[ctypes.c_int64],
    s0: Union[ctypes.c_int, int],
    s1: Union[ctypes.c_int, int],
    p0: Union[ctypes.c_int, int],
    p1: Union[ctypes.c_int, int],
    d0: Union[ctypes.c_int, int],
    d1: Union[ctypes.c_int, int],
    is_2D: Union[ctypes.c_bool, bool],
    /,
) -> ggml_tensor_p:
    ...


# // col2im_1d: scatter-add GEMM columns back to 1D signal
# // a: [K*OC, T_in]  (columns from matmul, K = a->ne[0]/OC)
# // result: [T_out, OC]  where T_out = (T_in - 1)*s0 + K - 2*p0
# GGML_API struct ggml_tensor * ggml_col2im_1d(
#         struct ggml_context * ctx,
#         struct ggml_tensor  * a,
#         int                   s0,
#         int                   oc,
#         int                   p0);
@ggml_function(
    "ggml_col2im_1d",
    [
        ggml_context_p_ctypes,
        ctypes.POINTER(ggml_tensor),
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
    ],
    ctypes.POINTER(ggml_tensor),
)
def ggml_col2im_1d(
    ctx: ggml_context_p,
    a: ggml_tensor_p,
    s0: Union[ctypes.c_int, int],
    oc: Union[ctypes.c_int, int],
    p0: Union[ctypes.c_int, int],
    /,
) -> ggml_tensor_p:
    ...


# GGML_API struct ggml_tensor * ggml_conv_depthwise_2d(
#         struct ggml_context * ctx,
#         struct ggml_tensor  * a,
#         struct ggml_tensor  * b,
#         int                  s0,
#         int                  s1,
#         int                  p0,
#         int                  p1,
#         int                  d0,
#         int                  d1);
@ggml_function(
    "ggml_conv_depthwise_2d",
    [
        ggml_context_p_ctypes,
        ctypes.POINTER(ggml_tensor),
        ctypes.POINTER(ggml_tensor),
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
    ],
    ctypes.POINTER(ggml_tensor),
    enabled=hasattr(lib, "ggml_conv_depthwise_2d"),
)
def ggml_conv_depthwise_2d(
    ctx: ggml_context_p,
    a: ggml_tensor_p,
    b: ggml_tensor_p,
    s0: Union[ctypes.c_int, int],
    s1: Union[ctypes.c_int, int],
    p0: Union[ctypes.c_int, int],
    p1: Union[ctypes.c_int, int],
    d0: Union[ctypes.c_int, int],
    d1: Union[ctypes.c_int, int],
    /,
) -> ggml_tensor_p:
    ...


# GGML_API struct ggml_tensor * ggml_conv_1d(
#         struct ggml_context * ctx,
#         struct ggml_tensor  * a,
#         struct ggml_tensor  * b,
#         int                   s0,  // stride
#         int                   p0,  // padding
#         int                   d0); // dilation
@ggml_function(
    "ggml_conv_1d",
    [
        ggml_context_p_ctypes,
        ctypes.POINTER(ggml_tensor),
        ctypes.POINTER(ggml_tensor),
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
    ],
    ctypes.POINTER(ggml_tensor),
)
def ggml_conv_1d(
    ctx: ggml_context_p,
    a: ggml_tensor_p,
    b: ggml_tensor_p,
    s0: Union[ctypes.c_int, int],
    p0: Union[ctypes.c_int, int],
    d0: Union[ctypes.c_int, int],
    /,
) -> ggml_tensor_p:
    """Convolution 1D

    Parameters:
        a: input tensor
        b: filter tensor
        s0: stride
        p0: padding
        d0: dilation

    Returns:
        output tensor"""
    ...


# GGML_API struct ggml_tensor * ggml_conv_1d_dw(
#         struct ggml_context * ctx,
#         struct ggml_tensor  * a,
#         struct ggml_tensor  * b,
#         int                   s0,
#         int                   p0,
#         int                   d0);
@ggml_function(
    "ggml_conv_1d_dw",
    [
        ggml_context_p_ctypes,
        ctypes.POINTER(ggml_tensor),
        ctypes.POINTER(ggml_tensor),
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
    ],
    ctypes.POINTER(ggml_tensor),
)
def ggml_conv_1d_dw(
    ctx: ggml_context_p,
    a: ggml_tensor_p,
    b: ggml_tensor_p,
    s0: Union[ctypes.c_int, int],
    p0: Union[ctypes.c_int, int],
    d0: Union[ctypes.c_int, int],
    /,
) -> ggml_tensor_p:
    ...


# // conv_1d with padding = half
# // alias for ggml_conv_1d(a, b, s, a->ne[0]/2, d)
# GGML_API struct ggml_tensor* ggml_conv_1d_ph(
#         struct ggml_context * ctx,
#         struct ggml_tensor  * a,
#         struct ggml_tensor  * b,
#         int                   s,
#         int                   d);
@ggml_function(
    "ggml_conv_1d_ph",
    [
        ggml_context_p_ctypes,
        ctypes.POINTER(ggml_tensor),
        ctypes.POINTER(ggml_tensor),
        ctypes.c_int,
        ctypes.c_int,
    ],
    ctypes.POINTER(ggml_tensor),
)
def ggml_conv_1d_ph(
    ctx: ggml_context_p,
    a: ggml_tensor_p,
    b: ggml_tensor_p,
    s: Union[ctypes.c_int, int],
    d: Union[ctypes.c_int, int],
    /,
) -> ggml_tensor_p:
    """Convolution 1D with padding = half

    Parameters:
        a: input tensor
        b: filter tensor
        s: stride
        d: dilation

    Returns:
        output tensor"""
    ...


# GGML_API struct ggml_tensor * ggml_conv_1d_dw_ph(
#         struct ggml_context * ctx,
#         struct ggml_tensor  * a,
#         struct ggml_tensor  * b,
#         int                   s0,
#         int                   d0);
@ggml_function(
    "ggml_conv_1d_dw_ph",
    [
        ggml_context_p_ctypes,
        ctypes.POINTER(ggml_tensor),
        ctypes.POINTER(ggml_tensor),
        ctypes.c_int,
        ctypes.c_int,
    ],
    ctypes.POINTER(ggml_tensor),
)
def ggml_conv_1d_dw_ph(
    ctx: ggml_context_p,
    a: ggml_tensor_p,
    b: ggml_tensor_p,
    s0: Union[ctypes.c_int, int],
    d0: Union[ctypes.c_int, int],
    /,
) -> ggml_tensor_p:
    ...


# GGML_API struct ggml_tensor * ggml_conv_transpose_1d(
#         struct ggml_context * ctx,
#         struct ggml_tensor  * a,
#         struct ggml_tensor  * b,
#         int                   s0,
#         int                   p0,
#         int                   d0);
@ggml_function(
    "ggml_conv_transpose_1d",
    [
        ggml_context_p_ctypes,
        ctypes.POINTER(ggml_tensor),
        ctypes.POINTER(ggml_tensor),
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
    ],
    ctypes.POINTER(ggml_tensor),
)
def ggml_conv_transpose_1d(
    ctx: ggml_context_p,
    a: ggml_tensor_p,
    b: ggml_tensor_p,
    s0: Union[ctypes.c_int, int],
    p0: Union[ctypes.c_int, int],
    d0: Union[ctypes.c_int, int],
    /,
) -> ggml_tensor_p:
    """Convolution transpose 1D

    Parameters:
        a: input tensor
        b: filter tensor
        s0: stride
        p0: padding
        d0: dilation

    Returns:
        output tensor"""
    ...


# GGML_API struct ggml_tensor * ggml_conv_2d(
#         struct ggml_context * ctx,
#         struct ggml_tensor  * a,
#         struct ggml_tensor  * b,
#         int                   s0,
#         int                   s1,
#         int                   p0,
#         int                   p1,
#         int                   d0,
#         int                   d1);
@ggml_function(
    "ggml_conv_2d",
    [
        ggml_context_p_ctypes,
        ctypes.POINTER(ggml_tensor),
        ctypes.POINTER(ggml_tensor),
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
    ],
    ctypes.POINTER(ggml_tensor),
)
def ggml_conv_2d(
    ctx: ggml_context_p,
    a: ggml_tensor_p,
    b: ggml_tensor_p,
    s0: Union[ctypes.c_int, int],
    s1: Union[ctypes.c_int, int],
    p0: Union[ctypes.c_int, int],
    p1: Union[ctypes.c_int, int],
    d0: Union[ctypes.c_int, int],
    d1: Union[ctypes.c_int, int],
    /,
) -> ggml_tensor_p:
    """Convolution 2D

    Parameters:
        a: input tensor
        b: filter tensor
        s0: stride
        s1: stride
        p0: padding
        p1: padding
        d0: dilation
        d1: dilation

    Returns:
        output tensor"""
    ...


# GGML_API struct ggml_tensor * ggml_im2col_3d(
#         struct ggml_context * ctx,
#         struct ggml_tensor  * a,
#         struct ggml_tensor  * b,
#         int64_t               IC,
#         int                   s0,
#         int                   s1,
#         int                   s2,
#         int                   p0,
#         int                   p1,
#         int                   p2,
#         int                   d0,
#         int                   d1,
#         int                   d2,
#         enum ggml_type        dst_type);
@ggml_function(
    "ggml_im2col_3d",
    [
        ggml_context_p_ctypes,
        ctypes.POINTER(ggml_tensor),
        ctypes.POINTER(ggml_tensor),
        ctypes.c_int64,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
    ],
    ctypes.POINTER(ggml_tensor),
)
def ggml_im2col_3d(
    ctx: ggml_context_p,
    a: ggml_tensor_p,
    b: ggml_tensor_p,
    IC: Union[ctypes.c_int64, int],
    s0: Union[ctypes.c_int, int],
    s1: Union[ctypes.c_int, int],
    s2: Union[ctypes.c_int, int],
    p0: Union[ctypes.c_int, int],
    p1: Union[ctypes.c_int, int],
    p2: Union[ctypes.c_int, int],
    d0: Union[ctypes.c_int, int],
    d1: Union[ctypes.c_int, int],
    d2: Union[ctypes.c_int, int],
    dst_type: Union[ctypes.c_int, int],
    /,
) -> ggml_tensor_p:
    ...


# GGML_API struct ggml_tensor * ggml_conv_3d(
#         struct ggml_context * ctx,
#         struct ggml_tensor  * a,
#         struct ggml_tensor  * b,
#         int64_t               IC,
#         int                   s0,
#         int                   s1,
#         int                   s2,
#         int                   p0,
#         int                   p1,
#         int                   p2,
#         int                   d0,
#         int                   d1,
#         int                   d2);
@ggml_function(
    "ggml_conv_3d",
    [
        ggml_context_p_ctypes,
        ctypes.POINTER(ggml_tensor),
        ctypes.POINTER(ggml_tensor),
        ctypes.c_int64,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
    ],
    ctypes.POINTER(ggml_tensor),
)
def ggml_conv_3d(
    ctx: ggml_context_p,
    a: ggml_tensor_p,
    b: ggml_tensor_p,
    IC: Union[ctypes.c_int64, int],
    s0: Union[ctypes.c_int, int],
    s1: Union[ctypes.c_int, int],
    s2: Union[ctypes.c_int, int],
    p0: Union[ctypes.c_int, int],
    p1: Union[ctypes.c_int, int],
    p2: Union[ctypes.c_int, int],
    d0: Union[ctypes.c_int, int],
    d1: Union[ctypes.c_int, int],
    d2: Union[ctypes.c_int, int],
    /,
) -> ggml_tensor_p:
    ...


# // kernel size is a->ne[0] x a->ne[1]
# // stride is equal to kernel size
# // padding is zero
# // example:
# // a:     16   16    3  768
# // b:   1024 1024    3    1
# // res:   64   64  768    1
# // used in sam
# GGML_API struct ggml_tensor * ggml_conv_2d_sk_p0(
#         struct ggml_context * ctx,
#         struct ggml_tensor  * a,
#         struct ggml_tensor  * b);
@ggml_function(
    "ggml_conv_2d_sk_p0",
    [
        ggml_context_p_ctypes,
        ctypes.POINTER(ggml_tensor),
        ctypes.POINTER(ggml_tensor),
    ],
    ctypes.POINTER(ggml_tensor),
)
def ggml_conv_2d_sk_p0(
    ctx: ggml_context_p, a: ggml_tensor_p, b: ggml_tensor_p, /
) -> ggml_tensor_p:
    """Convolution 2D

    Parameters:
        a: input tensor
        b: filter tensor

    Returns:
        output tensor"""
    ...


# // kernel size is a->ne[0] x a->ne[1]
# // stride is 1
# // padding is half
# // example:
# // a:      3    3    256  256
# // b:     64   64    256    1
# // res:   64   64    256    1
# // used in sam
# GGML_API struct ggml_tensor * ggml_conv_2d_s1_ph(
#         struct ggml_context * ctx,
#         struct ggml_tensor  * a,
#         struct ggml_tensor  * b);
@ggml_function(
    "ggml_conv_2d_s1_ph",
    [
        ggml_context_p_ctypes,
        ctypes.POINTER(ggml_tensor),
        ctypes.POINTER(ggml_tensor),
    ],
    ctypes.POINTER(ggml_tensor),
)
def ggml_conv_2d_s1_ph(
    ctx: ggml_context_p, a: ggml_tensor_p, b: ggml_tensor_p, /
) -> ggml_tensor_p:
    """Convolution 2D with stride = 1 and padding = half

    Parameters:
        a: input tensor
        b: filter tensor

    Returns:
        output tensor"""
    ...


# GGML_API struct ggml_tensor * ggml_conv_2d_dw(
#         struct ggml_context * ctx,
#         struct ggml_tensor  * a,
#         struct ggml_tensor  * b,
#         int                  s0,
#         int                  s1,
#         int                  p0,
#         int                  p1,
#         int                  d0,
#         int                  d1);
@ggml_function(
    "ggml_conv_2d_dw",
    [
        ggml_context_p_ctypes,
        ctypes.POINTER(ggml_tensor),
        ctypes.POINTER(ggml_tensor),
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
    ],
    ctypes.POINTER(ggml_tensor),
)
def ggml_conv_2d_dw(
    ctx: ggml_context_p,
    a: ggml_tensor_p,
    b: ggml_tensor_p,
    s0: Union[ctypes.c_int, int],
    s1: Union[ctypes.c_int, int],
    p0: Union[ctypes.c_int, int],
    p1: Union[ctypes.c_int, int],
    d0: Union[ctypes.c_int, int],
    d1: Union[ctypes.c_int, int],
    /,
) -> ggml_tensor_p:
    ...


# GGML_API struct ggml_tensor * ggml_conv_2d_dw_direct(
#         struct ggml_context * ctx,
#         struct ggml_tensor  * a,
#         struct ggml_tensor  * b,
#         int                   stride0,
#         int                   stride1,
#         int                   pad0,
#         int                   pad1,
#         int                   dilation0,
#         int                   dilation1);
@ggml_function(
    "ggml_conv_2d_dw_direct",
    [
        ggml_context_p_ctypes,
        ctypes.POINTER(ggml_tensor),
        ctypes.POINTER(ggml_tensor),
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
    ],
    ctypes.POINTER(ggml_tensor),
)
def ggml_conv_2d_dw_direct(
    ctx: ggml_context_p,
    a: ggml_tensor_p,
    b: ggml_tensor_p,
    stride0: Union[ctypes.c_int, int],
    stride1: Union[ctypes.c_int, int],
    pad0: Union[ctypes.c_int, int],
    pad1: Union[ctypes.c_int, int],
    dilation0: Union[ctypes.c_int, int],
    dilation1: Union[ctypes.c_int, int],
    /,
) -> ggml_tensor_p:
    ...


# GGML_API struct ggml_tensor * ggml_conv_transpose_2d_p0(
#         struct ggml_context * ctx,
#         struct ggml_tensor  * a,
#         struct ggml_tensor  * b,
#         int                   stride);
@ggml_function(
    "ggml_conv_transpose_2d_p0",
    [
        ggml_context_p_ctypes,
        ctypes.POINTER(ggml_tensor),
        ctypes.POINTER(ggml_tensor),
        ctypes.c_int,
    ],
    ctypes.POINTER(ggml_tensor),
)
def ggml_conv_transpose_2d_p0(
    ctx: ggml_context_p,
    a: ggml_tensor_p,
    b: ggml_tensor_p,
    stride: Union[ctypes.c_int, int],
    /,
) -> ggml_tensor_p:
    """Convolution Transpose 2D with padding = zero

    Parameters:
        a: input tensor
        b: filter tensor
        stride: stride

    Returns:
        output tensor"""
    ...


# GGML_API struct ggml_tensor * ggml_conv_2d_direct(
#         struct ggml_context * ctx,
#         struct ggml_tensor  * a,
#         struct ggml_tensor  * b,
#         int                   s0,
#         int                   s1,
#         int                   p0,
#         int                   p1,
#         int                   d0,
#         int                   d1);
@ggml_function(
    "ggml_conv_2d_direct",
    [
        ggml_context_p_ctypes,
        ctypes.POINTER(ggml_tensor),
        ctypes.POINTER(ggml_tensor),
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
    ],
    ctypes.POINTER(ggml_tensor),
)
def ggml_conv_2d_direct(
    ctx: ggml_context_p,
    a: ggml_tensor_p,
    b: ggml_tensor_p,
    s0: Union[ctypes.c_int, int],
    s1: Union[ctypes.c_int, int],
    p0: Union[ctypes.c_int, int],
    p1: Union[ctypes.c_int, int],
    d0: Union[ctypes.c_int, int],
    d1: Union[ctypes.c_int, int],
    /,
) -> ggml_tensor_p:
    ...


# GGML_API struct ggml_tensor * ggml_conv_3d_direct(
#         struct ggml_context * ctx,
#         struct ggml_tensor  * a,
#         struct ggml_tensor  * b,
#         int                   s0,
#         int                   s1,
#         int                   s2,
#         int                   p0,
#         int                   p1,
#         int                   p2,
#         int                   d0,
#         int                   d1,
#         int                   d2,
#         int                   n_channels,
#         int                   n_batch,
#         int                   n_channels_out);
@ggml_function(
    "ggml_conv_3d_direct",
    [
        ggml_context_p_ctypes,
        ctypes.POINTER(ggml_tensor),
        ctypes.POINTER(ggml_tensor),
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
    ],
    ctypes.POINTER(ggml_tensor),
)
def ggml_conv_3d_direct(
    ctx: ggml_context_p,
    a: ggml_tensor_p,
    b: ggml_tensor_p,
    s0: Union[ctypes.c_int, int],
    s1: Union[ctypes.c_int, int],
    s2: Union[ctypes.c_int, int],
    p0: Union[ctypes.c_int, int],
    p1: Union[ctypes.c_int, int],
    p2: Union[ctypes.c_int, int],
    d0: Union[ctypes.c_int, int],
    d1: Union[ctypes.c_int, int],
    d2: Union[ctypes.c_int, int],
    n_channels: Union[ctypes.c_int, int],
    n_batch: Union[ctypes.c_int, int],
    n_channels_out: Union[ctypes.c_int, int],
    /,
) -> ggml_tensor_p:
    ...


# enum ggml_op_pool {
#     GGML_OP_POOL_MAX,
#     GGML_OP_POOL_AVG,
#     GGML_OP_POOL_COUNT,
# };
GGML_OP_POOL_MAX = 0
GGML_OP_POOL_AVG = 1
GGML_OP_POOL_COUNT = 2


# GGML_API struct ggml_tensor * ggml_pool_1d(
#         struct ggml_context * ctx,
#         struct ggml_tensor  * a,
#         enum ggml_op_pool     op,
#         int                   k0, // kernel size
#         int                   s0, // stride
#         int                   p0); // padding
@ggml_function(
    "ggml_pool_1d",
    [
        ggml_context_p_ctypes,
        ctypes.POINTER(ggml_tensor),
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
    ],
    ctypes.POINTER(ggml_tensor),
)
def ggml_pool_1d(
    ctx: ggml_context_p,
    a: ggml_tensor_p,
    op: Union[ctypes.c_int, int],
    k0: Union[ctypes.c_int, int],
    s0: Union[ctypes.c_int, int],
    p0: Union[ctypes.c_int, int],
    /,
) -> ggml_tensor_p:
    """1D Pooling

    Parameters:
        a: input tensor
        op: pooling operation
        k0: kernel size
        s0: stride
        p0: padding

    Returns:
        output tensor"""
    ...


# // the result will have 2*p0 padding for the first dimension
# // and 2*p1 padding for the second dimension
# GGML_API struct ggml_tensor * ggml_pool_2d(
#         struct ggml_context * ctx,
#         struct ggml_tensor  * a,
#         enum ggml_op_pool     op,
#         int                   k0,
#         int                   k1,
#         int                   s0,
#         int                   s1,
#         float                 p0,
#         float                 p1);
@ggml_function(
    "ggml_pool_2d",
    [
        ggml_context_p_ctypes,
        ctypes.POINTER(ggml_tensor),
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_float,
        ctypes.c_float,
    ],
    ctypes.POINTER(ggml_tensor),
)
def ggml_pool_2d(
    ctx: ggml_context_p,
    a: ggml_tensor_p,
    op: Union[ctypes.c_int, int],
    k0: Union[ctypes.c_int, int],
    k1: Union[ctypes.c_int, int],
    s0: Union[ctypes.c_int, int],
    s1: Union[ctypes.c_int, int],
    p0: Union[ctypes.c_float, float],
    p1: Union[ctypes.c_float, float],
    /,
) -> ggml_tensor_p:
    """2D Pooling

    Parameters:
        a: input tensor
        op: pooling operation
        k0: kernel size
        k1: kernel size
        s0: stride
        s1: stride
        p0: padding
        p1: padding

    Returns:
        output tensor"""
    ...


# GGML_API struct ggml_tensor * ggml_pool_2d_back(
#         struct ggml_context * ctx,
#         struct ggml_tensor  * a,
#         struct ggml_tensor  * af,
#         enum ggml_op_pool     op,
#         int                   k0,
#         int                   k1,
#         int                   s0,
#         int                   s1,
#         float                 p0,
#         float                 p1);
@ggml_function(
    "ggml_pool_2d_back",
    [
        ggml_context_p_ctypes,
        ctypes.POINTER(ggml_tensor),
        ctypes.POINTER(ggml_tensor),
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_float,
        ctypes.c_float,
    ],
    ctypes.POINTER(ggml_tensor),
)
def ggml_pool_2d_back(
    ctx: ggml_context_p,
    a: ggml_tensor_p,
    af: ggml_tensor_p,
    op: Union[ctypes.c_int, int],
    k0: Union[ctypes.c_int, int],
    k1: Union[ctypes.c_int, int],
    s0: Union[ctypes.c_int, int],
    s1: Union[ctypes.c_int, int],
    p0: Union[ctypes.c_float, float],
    p1: Union[ctypes.c_float, float],
    /,
) -> ggml_tensor_p:
    ...


# // nearest interpolate
# // multiplies ne0 and ne1 by scale factor
# // used in stable-diffusion
# GGML_API struct ggml_tensor * ggml_upscale(
#         struct ggml_context * ctx,
#         struct ggml_tensor  * a,
#         int                   scale_factor,
#         enum ggml_scale_mode  mode);
@ggml_function(
    "ggml_upscale",
    [
        ggml_context_p_ctypes,
        ctypes.POINTER(ggml_tensor),
        ctypes.c_int,
        ctypes.c_int,
    ],
    ctypes.POINTER(ggml_tensor),
)
def ggml_upscale(
    ctx: ggml_context_p,
    a: ggml_tensor_p,
    scale_factor: Union[ctypes.c_int, int],
    mode: Union[ctypes.c_int, int],
    /,
) -> ggml_tensor_p:
    """Upscale

    Multiply ne0 and ne1 by scale factor

    Parameters:
        a: input tensor
        scale_factor: scale factor
        mode: scale mode

    Returns:
        output tensor"""
    ...


# // nearest interpolate
# // nearest interpolate to specified dimensions
# // used in tortoise.cpp
# GGML_API struct ggml_tensor * ggml_upscale_ext(
#         struct ggml_context * ctx,
#         struct ggml_tensor  * a,
#         int                   ne0,
#         int                   ne1,
#         int                   ne2,
#         int                   ne3,
#         enum ggml_scale_mode  mode);
@ggml_function(
    "ggml_upscale_ext",
    [
        ggml_context_p_ctypes,
        ctypes.POINTER(ggml_tensor),
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
    ],
    ctypes.POINTER(ggml_tensor),
)
def ggml_upscale_ext(
    ctx: ggml_context_p,
    a: ggml_tensor_p,
    ne0: Union[ctypes.c_int, int],
    ne1: Union[ctypes.c_int, int],
    ne2: Union[ctypes.c_int, int],
    ne3: Union[ctypes.c_int, int],
    mode: Union[ctypes.c_int, int],
    /,
) -> ggml_tensor_p:
    """Upscale to specified dimensions

    Parameters:
        a: input tensor
        ne0: dimension 0
        ne1: dimension 1
        ne2: dimension 2
        ne3: dimension 3
        mode: scale mode

    Returns:
        output tensor"""
    ...


# GGML_API struct ggml_tensor * ggml_interpolate(
#         struct ggml_context * ctx,
#         struct ggml_tensor  * a,
#         int64_t               ne0,
#         int64_t               ne1,
#         int64_t               ne2,
#         int64_t               ne3,
#         uint32_t              mode);
@ggml_function(
    "ggml_interpolate",
    [
        ggml_context_p_ctypes,
        ctypes.POINTER(ggml_tensor),
        ctypes.c_int64,
        ctypes.c_int64,
        ctypes.c_int64,
        ctypes.c_int64,
        ctypes.c_uint32,
    ],
    ctypes.POINTER(ggml_tensor),
)
def ggml_interpolate(
    ctx: ggml_context_p,
    a: ggml_tensor_p,
    ne0: Union[ctypes.c_int64, int],
    ne1: Union[ctypes.c_int64, int],
    ne2: Union[ctypes.c_int64, int],
    ne3: Union[ctypes.c_int64, int],
    mode: Union[ctypes.c_uint32, int],
    /,
) -> ggml_tensor_p:
    ...


# // pad each dimension with zeros: [x, ..., x] -> [x, ..., x, 0, ..., 0]
# GGML_API struct ggml_tensor * ggml_pad(
#         struct ggml_context * ctx,
#         struct ggml_tensor  * a,
#         int                  p0,
#         int                  p1,
#         int                  p2,
#         int                  p3);
@ggml_function(
    "ggml_pad",
    [
        ggml_context_p_ctypes,
        ctypes.POINTER(ggml_tensor),
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
    ],
    ctypes.POINTER(ggml_tensor),
)
def ggml_pad(
    ctx: ggml_context_p,
    a: ggml_tensor_p,
    p0: Union[ctypes.c_int, int],
    p1: Union[ctypes.c_int, int],
    p2: Union[ctypes.c_int, int],
    p3: Union[ctypes.c_int, int],
    /,
) -> ggml_tensor_p:
    """Pad tensor with zeros

    Parameters:
        a: input tensor
        p0: padding
        p1: padding
        p2: padding
        p3: padding

    Returns:
        output tensor"""
    ...


# GGML_API struct ggml_tensor * ggml_pad_circular(
#         struct ggml_context * ctx,
#         struct ggml_tensor  * a,
#         int                   p0,
#         int                   p1,
#         int                   p2,
#         int                   p3);
@ggml_function(
    "ggml_pad_circular",
    [
        ggml_context_p_ctypes,
        ctypes.POINTER(ggml_tensor),
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
    ],
    ctypes.POINTER(ggml_tensor),
)
def ggml_pad_circular(
    ctx: ggml_context_p,
    a: ggml_tensor_p,
    p0: Union[ctypes.c_int, int],
    p1: Union[ctypes.c_int, int],
    p2: Union[ctypes.c_int, int],
    p3: Union[ctypes.c_int, int],
    /,
) -> ggml_tensor_p:
    ...


# GGML_API struct ggml_tensor * ggml_pad_ext(
#         struct ggml_context * ctx,
#         struct ggml_tensor  * a,
#         int                  lp0,
#         int                  rp0,
#         int                  lp1,
#         int                  rp1,
#         int                  lp2,
#         int                  rp2,
#         int                  lp3,
#         int                  rp3);
@ggml_function(
    "ggml_pad_ext",
    [
        ggml_context_p_ctypes,
        ctypes.POINTER(ggml_tensor),
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
    ],
    ctypes.POINTER(ggml_tensor),
)
def ggml_pad_ext(
    ctx: ggml_context_p,
    a: ggml_tensor_p,
    lp0: Union[ctypes.c_int, int],
    rp0: Union[ctypes.c_int, int],
    lp1: Union[ctypes.c_int, int],
    rp1: Union[ctypes.c_int, int],
    lp2: Union[ctypes.c_int, int],
    rp2: Union[ctypes.c_int, int],
    lp3: Union[ctypes.c_int, int],
    rp3: Union[ctypes.c_int, int],
    /,
) -> ggml_tensor_p:
    ...


# GGML_API struct ggml_tensor * ggml_pad_ext_circular(
#         struct ggml_context * ctx,
#         struct ggml_tensor  * a,
#         int                   lp0,
#         int                   rp0,
#         int                   lp1,
#         int                   rp1,
#         int                   lp2,
#         int                   rp2,
#         int                   lp3,
#         int                   rp3);
@ggml_function(
    "ggml_pad_ext_circular",
    [
        ggml_context_p_ctypes,
        ctypes.POINTER(ggml_tensor),
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
    ],
    ctypes.POINTER(ggml_tensor),
)
def ggml_pad_ext_circular(
    ctx: ggml_context_p,
    a: ggml_tensor_p,
    lp0: Union[ctypes.c_int, int],
    rp0: Union[ctypes.c_int, int],
    lp1: Union[ctypes.c_int, int],
    rp1: Union[ctypes.c_int, int],
    lp2: Union[ctypes.c_int, int],
    rp2: Union[ctypes.c_int, int],
    lp3: Union[ctypes.c_int, int],
    rp3: Union[ctypes.c_int, int],
    /,
) -> ggml_tensor_p:
    ...


# GGML_API struct ggml_tensor * ggml_pad_reflect_1d(
#         struct ggml_context * ctx,
#         struct ggml_tensor  * a,
#         int                   p0,
#         int                   p1);
@ggml_function(
    "ggml_pad_reflect_1d",
    [
        ggml_context_p_ctypes,
        ctypes.POINTER(ggml_tensor),
        ctypes.c_int,
        ctypes.c_int,
    ],
    ctypes.POINTER(ggml_tensor),
)
def ggml_pad_reflect_1d(
    ctx: ggml_context_p,
    a: ggml_tensor_p,
    p0: Union[ctypes.c_int, int],
    p1: Union[ctypes.c_int, int],
    /,
) -> ggml_tensor_p:
    ...


# GGML_API struct ggml_tensor * ggml_roll(
#         struct ggml_context * ctx,
#         struct ggml_tensor  * a,
#         int                   shift0,
#         int                   shift1,
#         int                   shift2,
#         int                   shift3);
@ggml_function(
    "ggml_roll",
    [
        ggml_context_p_ctypes,
        ctypes.POINTER(ggml_tensor),
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
    ],
    ctypes.POINTER(ggml_tensor),
)
def ggml_roll(
    ctx: ggml_context_p,
    a: ggml_tensor_p,
    shift0: Union[ctypes.c_int, int],
    shift1: Union[ctypes.c_int, int],
    shift2: Union[ctypes.c_int, int],
    shift3: Union[ctypes.c_int, int],
    /,
) -> ggml_tensor_p:
    ...


# GGML_API struct ggml_tensor * ggml_tri(
#         struct ggml_context * ctx,
#         struct ggml_tensor  * a,
#         enum ggml_tri_type    type);
@ggml_function(
    "ggml_tri",
    [
        ggml_context_p_ctypes,
        ctypes.POINTER(ggml_tensor),
        ctypes.c_int,
    ],
    ctypes.POINTER(ggml_tensor),
)
def ggml_tri(
    ctx: ggml_context_p,
    a: ggml_tensor_p,
    type_: Union[ctypes.c_int, int],
    /,
) -> ggml_tensor_p:
    ...


# // Fill tensor a with constant c
# GGML_API struct ggml_tensor * ggml_fill(
#         struct ggml_context * ctx,
#         struct ggml_tensor  * a,
#         float                 c);
@ggml_function(
    "ggml_fill",
    [
        ggml_context_p_ctypes,
        ctypes.POINTER(ggml_tensor),
        ctypes.c_float,
    ],
    ctypes.POINTER(ggml_tensor),
    enabled=hasattr(lib, "ggml_fill"),
)
def ggml_fill(
    ctx: ggml_context_p, a: ggml_tensor_p, c: Union[ctypes.c_float, float], /
) -> ggml_tensor_p:
    """Fill a tensor with a constant and return the result.

    Parameters:
        ctx: ggml context
        a: tensor
        c: fill value

    Returns:
        Pointer to ggml_tensor"""
    ...


# GGML_API struct ggml_tensor * ggml_fill_inplace(
#         struct ggml_context * ctx,
#         struct ggml_tensor  * a,
#         float                 c);
@ggml_function(
    "ggml_fill_inplace",
    [
        ggml_context_p_ctypes,
        ctypes.POINTER(ggml_tensor),
        ctypes.c_float,
    ],
    ctypes.POINTER(ggml_tensor),
    enabled=hasattr(lib, "ggml_fill_inplace"),
)
def ggml_fill_inplace(
    ctx: ggml_context_p, a: ggml_tensor_p, c: Union[ctypes.c_float, float], /
) -> ggml_tensor_p:
    """Fill a tensor with a constant and store the result in the first tensor.

    Parameters:
        ctx: ggml context
        a: tensor
        c: fill value

    Returns:
        Pointer to ggml_tensor"""
    ...


# // Ref: https://github.com/CompVis/stable-diffusion/blob/main/ldm/modules/diffusionmodules/util.py#L151
# // timesteps: [N,]
# // return: [N, dim]
# GGML_API struct ggml_tensor * ggml_timestep_embedding(
#         struct ggml_context * ctx,
#         struct ggml_tensor  * timesteps,
#         int                   dim,
#         int                   max_period);
@ggml_function(
    "ggml_timestep_embedding",
    [
        ggml_context_p_ctypes,
        ctypes.POINTER(ggml_tensor),
        ctypes.c_int,
        ctypes.c_int,
    ],
    ctypes.POINTER(ggml_tensor),
)
def ggml_timestep_embedding(
    ctx: ggml_context_p,
    timesteps: ggml_tensor_p,
    dim: Union[ctypes.c_int, int],
    max_period: Union[ctypes.c_int, int],
    /,
) -> ggml_tensor_p:
    """Timestep embedding

    Parameters:
        timesteps: input tensor
        dim: embedding dimension
        max_period: maximum period"""
    ...


# // sort rows
# enum ggml_sort_order {
#     GGML_SORT_ORDER_ASC,
#     GGML_SORT_ORDER_DESC,
# };
GGML_SORT_ORDER_ASC = 0
GGML_SORT_ORDER_DESC = 1


# GGML_API struct ggml_tensor * ggml_argsort(
#         struct ggml_context * ctx,
#         struct ggml_tensor  * a,
#         enum ggml_sort_order  order);
@ggml_function(
    "ggml_argsort",
    [
        ggml_context_p_ctypes,
        ctypes.POINTER(ggml_tensor),
        ctypes.c_int,
    ],
    ctypes.POINTER(ggml_tensor),
)
def ggml_argsort(
    ctx: ggml_context_p, a: ggml_tensor_p, order: Union[ctypes.c_int, int], /
) -> ggml_tensor_p:
    """Argsort

    Parameters:
        a: input tensor
        order: sort order

    Returns:
        output tensor"""
    ...


# GGML_API struct ggml_tensor * ggml_argsort_top_k(
#         struct ggml_context * ctx,
#         struct ggml_tensor  * a,
#         int                   k);
@ggml_function(
    "ggml_argsort_top_k",
    [ggml_context_p_ctypes, ctypes.POINTER(ggml_tensor), ctypes.c_int],
    ctypes.POINTER(ggml_tensor),
)
def ggml_argsort_top_k(
    ctx: ggml_context_p,
    a: ggml_tensor_p,
    k: Union[ctypes.c_int, int],
    /,
) -> ggml_tensor_p:
    ...


# GGML_API struct ggml_tensor * ggml_arange(
#         struct ggml_context * ctx,
#         float                 start,
#         float                 stop,
#         float                 step);
@ggml_function(
    "ggml_arange",
    [
        ggml_context_p_ctypes,
        ctypes.c_float,
        ctypes.c_float,
        ctypes.c_float,
    ],
    ctypes.POINTER(ggml_tensor),
)
def ggml_arange(
    ctx: ggml_context_p,
    start: Union[ctypes.c_float, float],
    stop: Union[ctypes.c_float, float],
    step: Union[ctypes.c_float, float],
    /,
) -> ggml_tensor_p:
    """Arange

    Parameters:
        start: start
        stop: stop
        step: step"""
    ...

# // top k elements per row
# GGML_API struct ggml_tensor * ggml_top_k(
#         struct ggml_context * ctx,
#         struct ggml_tensor  * a,
#         int                   k);
@ggml_function(
    "ggml_top_k",
    [ggml_context_p_ctypes, ctypes.POINTER(ggml_tensor), ctypes.c_int],
    ctypes.POINTER(ggml_tensor),
)
def ggml_top_k(
    ctx: ggml_context_p, a: ggml_tensor_p, k: Union[ctypes.c_int, int], /
) -> ggml_tensor_p:
    """Top k elements per row

    Parameters:
        a: input tensor
        k: number of elements

    Returns:
        output tensor"""
    ...


#define GGML_KQ_MASK_PAD 32
GGML_KQ_MASK_PAD = 32

# // q:    [n_embd, n_batch,     n_head,    1]
# // k:    [n_embd, n_kv,        n_head_kv, 1]
# // v:    [n_embd, n_kv,        n_head_kv, 1] !! not transposed !!
# // mask: [n_kv,   n_batch_pad, 1,         1] !! n_batch_pad = GGML_PAD(n_batch, GGML_KQ_MASK_PAD) !!
# // res:  [n_embd, n_head,      n_batch,   1] !! permuted !!
# GGML_API struct ggml_tensor * ggml_flash_attn_ext(
#         struct ggml_context * ctx,
#         struct ggml_tensor  * q,
#         struct ggml_tensor  * k,
#         struct ggml_tensor  * v,
#         struct ggml_tensor  * mask,
#         float                 scale,
#         float                 max_bias,
#         float                 logit_softcap);
@ggml_function(
    "ggml_flash_attn_ext",
    [
        ggml_context_p_ctypes,
        ctypes.POINTER(ggml_tensor),
        ctypes.POINTER(ggml_tensor),
        ctypes.POINTER(ggml_tensor),
        ctypes.POINTER(ggml_tensor),
        ctypes.c_float,
        ctypes.c_float,
        ctypes.c_float,
    ],
    ctypes.POINTER(ggml_tensor),
)
def ggml_flash_attn_ext(
    ctx: ggml_context_p,
    q: ggml_tensor_p,
    k: ggml_tensor_p,
    v: ggml_tensor_p,
    mask: ggml_tensor_p,
    scale: Union[ctypes.c_float, float],
    max_bias: Union[ctypes.c_float, float],
    logit_softcap: Union[ctypes.c_float, float],
    /,
) -> ggml_tensor_p:
    ...


# GGML_API void ggml_flash_attn_ext_set_prec(
#         struct ggml_tensor * a,
#         enum ggml_prec       prec);
@ggml_function(
    "ggml_flash_attn_ext_set_prec",
    [ctypes.POINTER(ggml_tensor), ctypes.c_int],
    None,
)
def ggml_flash_attn_ext_set_prec(
    a: ggml_tensor_p, prec: Union[ctypes.c_int, int], /
) -> None:
    ...


# GGML_API enum ggml_prec ggml_flash_attn_ext_get_prec(
#         const struct ggml_tensor * a);
@ggml_function(
    "ggml_flash_attn_ext_get_prec",
    [ctypes.POINTER(ggml_tensor)],
    ctypes.c_int,
)
def ggml_flash_attn_ext_get_prec(a: ggml_tensor_p, /) -> int:
    ...


# GGML_API void ggml_flash_attn_ext_add_sinks(
#         struct ggml_tensor * a,
#         struct ggml_tensor * sinks);
@ggml_function(
    "ggml_flash_attn_ext_add_sinks",
    [ctypes.POINTER(ggml_tensor), ctypes.POINTER(ggml_tensor)],
    None,
)
def ggml_flash_attn_ext_add_sinks(
    a: ggml_tensor_p,
    sinks: ggml_tensor_p,
    /,
) -> None:
    ...


# // TODO: needs to be adapted to ggml_flash_attn_ext
# GGML_API struct ggml_tensor * ggml_flash_attn_back(
#         struct ggml_context * ctx,
#         struct ggml_tensor  * q,
#         struct ggml_tensor  * k,
#         struct ggml_tensor  * v,
#         struct ggml_tensor  * d,
#         bool                  masked);
@ggml_function(
    "ggml_flash_attn_back",
    [
        ggml_context_p_ctypes,
        ctypes.POINTER(ggml_tensor),
        ctypes.POINTER(ggml_tensor),
        ctypes.POINTER(ggml_tensor),
        ctypes.POINTER(ggml_tensor),
        ctypes.c_bool,
    ],
    ctypes.POINTER(ggml_tensor),
)
def ggml_flash_attn_back(
    ctx: ggml_context_p,
    q: ggml_tensor_p,
    k: ggml_tensor_p,
    v: ggml_tensor_p,
    d: ggml_tensor_p,
    masked: Union[ctypes.c_bool, bool],
    /,
) -> ggml_tensor_p:
    ...


# GGML_API struct ggml_tensor * ggml_ssm_conv(
#         struct ggml_context * ctx,
#         struct ggml_tensor  * sx,
#         struct ggml_tensor  * c);
@ggml_function(
    "ggml_ssm_conv",
    [
        ggml_context_p_ctypes,
        ctypes.POINTER(ggml_tensor),
        ctypes.POINTER(ggml_tensor),
    ],
    ctypes.POINTER(ggml_tensor),
)
def ggml_ssm_conv(
    ctx: ggml_context_p,
    sx: ggml_tensor_p,
    c: ggml_tensor_p,
    /,
) -> ggml_tensor_p:
    ...


# GGML_API struct ggml_tensor * ggml_ssm_scan(
#         struct ggml_context * ctx,
#         struct ggml_tensor  * s,
#         struct ggml_tensor  * x,
#         struct ggml_tensor  * dt,
#         struct ggml_tensor  * A,
#         struct ggml_tensor  * B,
#         struct ggml_tensor  * C,
#         struct ggml_tensor  * ids);
@ggml_function(
    "ggml_ssm_scan",
    [
        ggml_context_p_ctypes,
        ctypes.POINTER(ggml_tensor),
        ctypes.POINTER(ggml_tensor),
        ctypes.POINTER(ggml_tensor),
        ctypes.POINTER(ggml_tensor),
        ctypes.POINTER(ggml_tensor),
        ctypes.POINTER(ggml_tensor),
        ctypes.POINTER(ggml_tensor),
    ],
    ctypes.POINTER(ggml_tensor),
)
def ggml_ssm_scan(
    ctx: ggml_context_p,
    s: ggml_tensor_p,
    x: ggml_tensor_p,
    dt: ggml_tensor_p,
    A: ggml_tensor_p,
    B: ggml_tensor_p,
    C: ggml_tensor_p,
    ids: ggml_tensor_p,
    /,
) -> ggml_tensor_p:
    ...


# // partition into non-overlapping windows with padding if needed
# // example:
# // a:   768   64   64    1
# // w:    14
# // res: 768   14   14    25
# // used in sam
# GGML_API struct ggml_tensor * ggml_win_part(
#         struct ggml_context * ctx,
#         struct ggml_tensor  * a,
#         int                   w);
@ggml_function(
    "ggml_win_part",
    [
        ggml_context_p_ctypes,
        ctypes.POINTER(ggml_tensor),
        ctypes.c_int,
    ],
    ctypes.POINTER(ggml_tensor),
)
def ggml_win_part(
    ctx: ggml_context_p, a: ggml_tensor_p, w: Union[ctypes.c_int, int], /
) -> ggml_tensor_p:
    ...


# // reverse of ggml_win_part
# // used in sam
# GGML_API struct ggml_tensor * ggml_win_unpart(
#         struct ggml_context * ctx,
#         struct ggml_tensor  * a,
#         int                   w0,
#         int                   h0,
#         int                   w);
@ggml_function(
    "ggml_win_unpart",
    [
        ggml_context_p_ctypes,
        ctypes.POINTER(ggml_tensor),
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
    ],
    ctypes.POINTER(ggml_tensor),
)
def ggml_win_unpart(
    ctx: ggml_context_p,
    a: ggml_tensor_p,
    w0: Union[ctypes.c_int, int],
    h0: Union[ctypes.c_int, int],
    w: Union[ctypes.c_int, int],
    /,
) -> ggml_tensor_p:
    ...


# GGML_API struct ggml_tensor * ggml_unary(
#         struct ggml_context * ctx,
#             struct ggml_tensor * a,
#             enum ggml_unary_op op);
@ggml_function(
    "ggml_unary",
    [
        ggml_context_p_ctypes,
        ctypes.POINTER(ggml_tensor),
        ctypes.c_int,
    ],
    ctypes.POINTER(ggml_tensor),
)
def ggml_unary(
    ctx: ggml_context_p, a: ggml_tensor_p, op: Union[ctypes.c_int, int], /
) -> ggml_tensor_p:
    ...


# GGML_API struct ggml_tensor * ggml_unary_inplace(
#     struct ggml_context * ctx,
#     struct ggml_tensor  * a,
#     enum ggml_unary_op op);
@ggml_function(
    "ggml_unary_inplace",
    [
        ggml_context_p_ctypes,
        ctypes.POINTER(ggml_tensor),
        ctypes.c_int,
    ],
    ctypes.POINTER(ggml_tensor),
)
def ggml_unary_inplace(
    ctx: ggml_context_p, a: ggml_tensor_p, op: Union[ctypes.c_int, int], /
) -> ggml_tensor_p:
    ...


# // used in sam
# GGML_API struct ggml_tensor * ggml_get_rel_pos(
#         struct ggml_context * ctx,
#         struct ggml_tensor  * a,
#         int                   qh,
#         int                   kh);
@ggml_function(
    "ggml_get_rel_pos",
    [
        ggml_context_p_ctypes,
        ctypes.POINTER(ggml_tensor),
        ctypes.c_int,
        ctypes.c_int,
    ],
    ctypes.POINTER(ggml_tensor),
)
def ggml_get_rel_pos(
    ctx: ggml_context_p,
    a: ggml_tensor_p,
    qh: Union[ctypes.c_int, int],
    kh: Union[ctypes.c_int, int],
    /,
) -> ggml_tensor_p:
    ...


# // used in sam
# GGML_API struct ggml_tensor * ggml_add_rel_pos(
#         struct ggml_context * ctx,
#         struct ggml_tensor  * a,
#         struct ggml_tensor  * pw,
#         struct ggml_tensor  * ph);
@ggml_function(
    "ggml_add_rel_pos",
    [
        ggml_context_p_ctypes,
        ctypes.POINTER(ggml_tensor),
        ctypes.POINTER(ggml_tensor),
        ctypes.POINTER(ggml_tensor),
    ],
    ctypes.POINTER(ggml_tensor),
)
def ggml_add_rel_pos(
    ctx: ggml_context_p, a: ggml_tensor_p, pw: ggml_tensor_p, ph: ggml_tensor_p, /
) -> ggml_tensor_p:
    ...


# GGML_API struct ggml_tensor * ggml_add_rel_pos_inplace(
#         struct ggml_context * ctx,
#         struct ggml_tensor  * a,
#         struct ggml_tensor  * pw,
#         struct ggml_tensor  * ph);
@ggml_function(
    "ggml_add_rel_pos_inplace",
    [
        ggml_context_p_ctypes,
        ctypes.POINTER(ggml_tensor),
        ctypes.POINTER(ggml_tensor),
        ctypes.POINTER(ggml_tensor),
    ],
    ctypes.POINTER(ggml_tensor),
)
def ggml_add_rel_pos_inplace(
    ctx: ggml_context_p, a: ggml_tensor_p, pw: ggml_tensor_p, ph: ggml_tensor_p, /
) -> ggml_tensor_p:
    ...


# GGML_API struct ggml_tensor * ggml_rwkv_wkv6(
#         struct ggml_context * ctx,
#         struct ggml_tensor  * k,
#         struct ggml_tensor  * v,
#         struct ggml_tensor  * r,
#         struct ggml_tensor  * tf,
#         struct ggml_tensor  * td,
#         struct ggml_tensor  * state);
@ggml_function(
    "ggml_rwkv_wkv6",
    [
        ggml_context_p_ctypes,
        ctypes.POINTER(ggml_tensor),
        ctypes.POINTER(ggml_tensor),
        ctypes.POINTER(ggml_tensor),
        ctypes.POINTER(ggml_tensor),
        ctypes.POINTER(ggml_tensor),
        ctypes.POINTER(ggml_tensor),
    ],
    ctypes.POINTER(ggml_tensor),
)
def ggml_rwkv_wkv6(
    ctx: ggml_context_p,
    k: ggml_tensor_p,
    v: ggml_tensor_p,
    r: ggml_tensor_p,
    tf: ggml_tensor_p,
    td: ggml_tensor_p,
    state: ggml_tensor_p,
    /,
) -> ggml_tensor_p:
    ...


# GGML_API struct ggml_tensor * ggml_gated_linear_attn(
#         struct ggml_context * ctx,
#         struct ggml_tensor  * k,
#         struct ggml_tensor  * v,
#         struct ggml_tensor  * q,
#         struct ggml_tensor  * g,
#         struct ggml_tensor  * state,
#         float scale);
@ggml_function(
    "ggml_gated_linear_attn",
    [
        ggml_context_p_ctypes,
        ctypes.POINTER(ggml_tensor),
        ctypes.POINTER(ggml_tensor),
        ctypes.POINTER(ggml_tensor),
        ctypes.POINTER(ggml_tensor),
        ctypes.POINTER(ggml_tensor),
        ctypes.c_float,
    ],
    ctypes.POINTER(ggml_tensor),
)
def ggml_gated_linear_attn(
    ctx: ggml_context_p,
    k: ggml_tensor_p,
    v: ggml_tensor_p,
    q: ggml_tensor_p,
    g: ggml_tensor_p,
    state: ggml_tensor_p,
    scale: Union[ctypes.c_float, float],
    /,
) -> ggml_tensor_p:
    ...


# GGML_API struct ggml_tensor * ggml_rwkv_wkv7(
#         struct ggml_context * ctx,
#         struct ggml_tensor  * r,
#         struct ggml_tensor  * w,
#         struct ggml_tensor  * k,
#         struct ggml_tensor  * v,
#         struct ggml_tensor  * a,
#         struct ggml_tensor  * b,
#         struct ggml_tensor  * state);
@ggml_function(
    "ggml_rwkv_wkv7",
    [
        ggml_context_p_ctypes,
        ctypes.POINTER(ggml_tensor),
        ctypes.POINTER(ggml_tensor),
        ctypes.POINTER(ggml_tensor),
        ctypes.POINTER(ggml_tensor),
        ctypes.POINTER(ggml_tensor),
        ctypes.POINTER(ggml_tensor),
        ctypes.POINTER(ggml_tensor),
    ],
    ctypes.POINTER(ggml_tensor),
)
def ggml_rwkv_wkv7(
    ctx: ggml_context_p,
    r: ggml_tensor_p,
    w: ggml_tensor_p,
    k: ggml_tensor_p,
    v: ggml_tensor_p,
    a: ggml_tensor_p,
    b: ggml_tensor_p,
    state: ggml_tensor_p,
    /,
) -> ggml_tensor_p:
    ...


# GGML_API struct ggml_tensor * ggml_solve_tri(
#         struct ggml_context * ctx,
#         struct ggml_tensor  * a,
#         struct ggml_tensor  * b,
#         bool                  left,
#         bool                  lower,
#         bool                  uni);
@ggml_function(
    "ggml_solve_tri",
    [
        ggml_context_p_ctypes,
        ctypes.POINTER(ggml_tensor),
        ctypes.POINTER(ggml_tensor),
        ctypes.c_bool,
        ctypes.c_bool,
        ctypes.c_bool,
    ],
    ctypes.POINTER(ggml_tensor),
)
def ggml_solve_tri(
    ctx: ggml_context_p,
    a: ggml_tensor_p,
    b: ggml_tensor_p,
    left: Union[ctypes.c_bool, bool],
    lower: Union[ctypes.c_bool, bool],
    uni: Union[ctypes.c_bool, bool],
    /,
) -> ggml_tensor_p:
    ...


# GGML_API struct ggml_tensor * ggml_gated_delta_net(
#         struct ggml_context * ctx,
#         struct ggml_tensor  * q,
#         struct ggml_tensor  * k,
#         struct ggml_tensor  * v,
#         struct ggml_tensor  * g,
#         struct ggml_tensor  * beta,
#         struct ggml_tensor  * state,
#         int64_t               K);
@ggml_function(
    "ggml_gated_delta_net",
    [
        ggml_context_p_ctypes,
        ctypes.POINTER(ggml_tensor),
        ctypes.POINTER(ggml_tensor),
        ctypes.POINTER(ggml_tensor),
        ctypes.POINTER(ggml_tensor),
        ctypes.POINTER(ggml_tensor),
        ctypes.POINTER(ggml_tensor),
        ctypes.c_int64,
    ],
    ctypes.POINTER(ggml_tensor),
)
def ggml_gated_delta_net(
    ctx: ggml_context_p,
    q: ggml_tensor_p,
    k: ggml_tensor_p,
    v: ggml_tensor_p,
    g: ggml_tensor_p,
    beta: ggml_tensor_p,
    state: ggml_tensor_p,
    K: Union[ctypes.c_int64, int],
    /,
) -> ggml_tensor_p:
    ...


# // custom operators (DEPRECATED)

# typedef void (*ggml_unary_op_f32_t)(const int, float *, const float *);
ggml_unary_op_f32_t = ctypes.CFUNCTYPE(
    None, ctypes.c_int, ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float)
)

# typedef void (*ggml_binary_op_f32_t)(const int, float *, const float *, const float *);
ggml_binary_op_f32_t = ctypes.CFUNCTYPE(
    None,
    ctypes.c_int,
    ctypes.POINTER(ctypes.c_float),
    ctypes.POINTER(ctypes.c_float),
    ctypes.POINTER(ctypes.c_float),
)

# typedef void (*ggml_custom1_op_f32_t)(struct ggml_tensor *, const struct ggml_tensor *);
ggml_custom1_op_f32_t = ctypes.CFUNCTYPE(
    None, ctypes.POINTER(ggml_tensor), ctypes.POINTER(ggml_tensor)
)
"""Unary operator function type"""

# typedef void (*ggml_custom2_op_f32_t)(struct ggml_tensor *, const struct ggml_tensor *, const struct ggml_tensor *);
ggml_custom2_op_f32_t = ctypes.CFUNCTYPE(
    None,
    ctypes.POINTER(ggml_tensor),
    ctypes.POINTER(ggml_tensor),
    ctypes.POINTER(ggml_tensor),
)
"""Binary operator function type"""

# typedef void (*ggml_custom3_op_f32_t)(struct ggml_tensor *, const struct ggml_tensor *, const struct ggml_tensor *, const struct ggml_tensor *);
ggml_custom3_op_f32_t = ctypes.CFUNCTYPE(
    None,
    ctypes.POINTER(ggml_tensor),
    ctypes.POINTER(ggml_tensor),
    ctypes.POINTER(ggml_tensor),
    ctypes.POINTER(ggml_tensor),
)
"""Ternary operator function type"""


# GGML_API struct ggml_tensor * ggml_map_unary_f32(
#         struct ggml_context        * ctx,
#         struct ggml_tensor         * a,
#                ggml_unary_op_f32_t   fun);
@ggml_function(
    "ggml_map_unary_f32",
    [
        ggml_context_p_ctypes,
        ctypes.POINTER(ggml_tensor),
        ggml_unary_op_f32_t,
    ],
    ctypes.POINTER(ggml_tensor),
    enabled=hasattr(lib, "ggml_map_unary_f32"),
)
def ggml_map_unary_f32(
    ctx: ggml_context_p, a: ggml_tensor_p, fun: CtypesFuncPointer, /  # type: ignore
) -> ggml_tensor_p:
    ...


# GGML_API struct ggml_tensor * ggml_map_unary_inplace_f32(
#         struct ggml_context        * ctx,
#         struct ggml_tensor         * a,
#                 ggml_unary_op_f32_t   fun);
@ggml_function(
    "ggml_map_unary_inplace_f32",
    [
        ggml_context_p_ctypes,
        ctypes.POINTER(ggml_tensor),
        ggml_unary_op_f32_t,
    ],
    ctypes.POINTER(ggml_tensor),
    enabled=hasattr(lib, "ggml_map_unary_inplace_f32"),
)
def ggml_map_unary_inplace_f32(
    ctx: ggml_context_p, a: ggml_tensor_p, fun: CtypesFuncPointer, /  # type: ignore
) -> ggml_tensor_p:
    ...


# GGML_API struct ggml_tensor * ggml_map_binary_f32(
#         struct ggml_context         * ctx,
#         struct ggml_tensor          * a,
#         struct ggml_tensor          * b,
#                ggml_binary_op_f32_t   fun);
@ggml_function(
    "ggml_map_binary_f32",
    [
        ggml_context_p_ctypes,
        ctypes.POINTER(ggml_tensor),
        ctypes.POINTER(ggml_tensor),
        ggml_binary_op_f32_t,
    ],
    ctypes.POINTER(ggml_tensor),
    enabled=hasattr(lib, "ggml_map_binary_f32"),
)
def ggml_map_binary_f32(
    ctx: ggml_context_p,
    a: ggml_tensor_p,
    b: ggml_tensor_p,
    fun: CtypesFuncPointer,  # type: ignore
    /,
) -> ggml_tensor_p:
    ...


# GGML_API struct ggml_tensor * ggml_map_binary_inplace_f32(
#         struct ggml_context         * ctx,
#         struct ggml_tensor          * a,
#         struct ggml_tensor          * b,
#                 ggml_binary_op_f32_t   fun);
@ggml_function(
    "ggml_map_binary_inplace_f32",
    [
        ggml_context_p_ctypes,
        ctypes.POINTER(ggml_tensor),
        ctypes.POINTER(ggml_tensor),
        ggml_binary_op_f32_t,
    ],
    ctypes.POINTER(ggml_tensor),
    enabled=hasattr(lib, "ggml_map_binary_inplace_f32"),
)
def ggml_map_binary_inplace_f32(
    ctx: ggml_context_p,
    a: ggml_tensor_p,
    b: ggml_tensor_p,
    fun: CtypesFuncPointer,  # type: ignore
    /,
) -> ggml_tensor_p:
    ...


# GGML_API struct ggml_tensor * ggml_map_custom1_f32(
#         struct ggml_context          * ctx,
#         struct ggml_tensor           * a,
#                 ggml_custom1_op_f32_t   fun);
@ggml_function(
    "ggml_map_custom1_f32",
    [ggml_context_p_ctypes, ctypes.POINTER(ggml_tensor), ggml_custom1_op_f32_t],
    ctypes.POINTER(ggml_tensor),
    enabled=hasattr(lib, "ggml_map_custom1_f32"),
)
def ggml_map_custom1_f32(
    ctx: ggml_context_p, a: ggml_tensor_p, fun: CtypesFuncPointer, /  # type: ignore
) -> ggml_tensor_p:
    """Custom unary operator on a tensor.

    Example:
        ```python
        import ggml

        @ggml.ggml_custom1_op_f32_t
        def custom_op(b: ggml.tensor_p, a: ggml.tensor_p):
            # do something with a and copy to b
            return

        ...

        b = ggml.ggml_map_custom1_f32(ctx, a, custom_op)
        ```

    Parameters:
        a: input tensor
        fun (ggml.ggml_custom1_op_f32_t): function to apply to each element

    Returns:
        output tensor"""
    ...


# GGML_API struct ggml_tensor * ggml_map_custom1_inplace_f32(
#         struct ggml_context          * ctx,
#         struct ggml_tensor           * a,
#                 ggml_custom1_op_f32_t   fun);
def ggml_map_custom1_inplace_f32(
    ctx: ggml_context_p, a: ggml_tensor_p, fun: "ctypes._CFuncPtr", /  # type: ignore
) -> ggml_tensor_p:
    """Custom unary operator on a tensor inplace.

    Parameters:
        a: input tensor
        fun (ggml.ggml_custom1_op_f32_t): function to apply to each element

    Returns:
        output tensor"""
    ...


# GGML_API struct ggml_tensor * ggml_map_custom2_f32(
#         struct ggml_context          * ctx,
#         struct ggml_tensor           * a,
#         struct ggml_tensor           * b,
#                 ggml_custom2_op_f32_t   fun);
@ggml_function(
    "ggml_map_custom2_f32",
    [
        ggml_context_p_ctypes,
        ctypes.POINTER(ggml_tensor),
        ctypes.POINTER(ggml_tensor),
        ggml_custom2_op_f32_t,
    ],
    ctypes.POINTER(ggml_tensor),
    enabled=hasattr(lib, "ggml_map_custom2_f32"),
)
def ggml_map_custom2_f32(
    ctx: ggml_context_p,
    a: ggml_tensor_p,
    b: ggml_tensor_p,
    fun: CtypesFuncPointer,  # type: ignore
    /,
) -> ggml_tensor_p:
    """Custom binary operator on two tensors.

    Parameters:
        a: input tensor
        b: input tensor
        fun (ggml.ggml_custom2_op_f32_t): function to apply to each element

    Returns:
        output tensor"""
    ...


# GGML_API struct ggml_tensor * ggml_map_custom2_inplace_f32(
#         struct ggml_context          * ctx,
#         struct ggml_tensor           * a,
#         struct ggml_tensor           * b,
#                 ggml_custom2_op_f32_t   fun);
@ggml_function(
    "ggml_map_custom2_inplace_f32",
    [
        ggml_context_p_ctypes,
        ctypes.POINTER(ggml_tensor),
        ctypes.POINTER(ggml_tensor),
        ggml_custom2_op_f32_t,
    ],
    ctypes.POINTER(ggml_tensor),
    enabled=hasattr(lib, "ggml_map_custom2_inplace_f32"),
)
def ggml_map_custom2_inplace_f32(
    ctx: ggml_context_p,
    a: ggml_tensor_p,
    b: ggml_tensor_p,
    fun: CtypesFuncPointer,  # type: ignore
) -> ggml_tensor_p:
    """Custom binary operator on two tensors inplace.

    Parameters:
        a: input tensor
        b: input tensor
        fun (ggml.ggml_custom2_op_f32_t): function to apply to each element

    Returns:
        output tensor"""
    ...


# GGML_API struct ggml_tensor * ggml_map_custom3_f32(
#         struct ggml_context          * ctx,
#         struct ggml_tensor           * a,
#         struct ggml_tensor           * b,
#         struct ggml_tensor           * c,
#                 ggml_custom3_op_f32_t   fun);
@ggml_function(
    "ggml_map_custom3_f32",
    [
        ggml_context_p_ctypes,
        ctypes.POINTER(ggml_tensor),
        ctypes.POINTER(ggml_tensor),
        ctypes.POINTER(ggml_tensor),
        ggml_custom3_op_f32_t,
    ],
    ctypes.POINTER(ggml_tensor),
    enabled=hasattr(lib, "ggml_map_custom3_f32"),
)
def ggml_map_custom3_f32(
    ctx: ggml_context_p,
    a: ggml_tensor_p,
    b: ggml_tensor_p,
    c: ggml_tensor_p,
    fun: CtypesFuncPointer,  # type: ignore
) -> ggml_tensor_p:
    """Custom ternary operator on three tensors.

    Parameters:
        a: input tensor
        b: input tensor
        c: input tensor
        fun (ggml.ggml_custom3_op_f32_t): function to apply to each element

    Returns:
        output tensor"""
    ...


# GGML_API struct ggml_tensor * ggml_map_custom3_inplace_f32(
#         struct ggml_context          * ctx,
#         struct ggml_tensor           * a,
#         struct ggml_tensor           * b,
#         struct ggml_tensor           * c,
#                 ggml_custom3_op_f32_t   fun);
@ggml_function(
    "ggml_map_custom3_inplace_f32",
    [
        ggml_context_p_ctypes,
        ctypes.POINTER(ggml_tensor),
        ctypes.POINTER(ggml_tensor),
        ctypes.POINTER(ggml_tensor),
        ggml_custom3_op_f32_t,
    ],
    ctypes.POINTER(ggml_tensor),
    enabled=hasattr(lib, "ggml_map_custom3_inplace_f32"),
)
def ggml_map_custom3_inplace_f32(
    ctx: ggml_context_p,
    a: ggml_tensor_p,
    b: ggml_tensor_p,
    c: ggml_tensor_p,
    fun: CtypesFuncPointer,  # type: ignore
) -> ggml_tensor_p:
    """Custom ternary operator on three tensors inplace.

    Parameters:
        a: input tensor
        b: input tensor
        c: input tensor
        fun (ggml.ggml_custom3_op_f32_t): function to apply to each element

    Returns:
        output tensor"""
    ...


# // custom operators v2

# typedef void (*ggml_custom1_op_t)(struct ggml_tensor * dst , const struct ggml_tensor * a, int ith, int nth, void * userdata);
ggml_custom1_op_t = ctypes.CFUNCTYPE(
    None,
    ctypes.POINTER(ggml_tensor),
    ctypes.POINTER(ggml_tensor),
    ctypes.c_int,
    ctypes.c_int,
    ctypes.c_void_p,
)
"""Custom unary operator on a tensor."""

# typedef void (*ggml_custom2_op_t)(struct ggml_tensor * dst , const struct ggml_tensor * a, const struct ggml_tensor * b, int ith, int nth, void * userdata);
ggml_custom2_op_t = ctypes.CFUNCTYPE(
    None,
    ctypes.POINTER(ggml_tensor),
    ctypes.POINTER(ggml_tensor),
    ctypes.POINTER(ggml_tensor),
    ctypes.c_int,
    ctypes.c_int,
    ctypes.c_void_p,
)
"""Custom binary operator on two tensors."""

# typedef void (*ggml_custom3_op_t)(struct ggml_tensor * dst , const struct ggml_tensor * a, const struct ggml_tensor * b, const struct ggml_tensor * c, int ith, int nth, void * userdata);
ggml_custom3_op_t = ctypes.CFUNCTYPE(
    None,
    ctypes.POINTER(ggml_tensor),
    ctypes.POINTER(ggml_tensor),
    ctypes.POINTER(ggml_tensor),
    ctypes.POINTER(ggml_tensor),
    ctypes.c_int,
    ctypes.c_int,
    ctypes.c_void_p,
)
"""Custom ternary operator on three tensors."""

# typedef void (*ggml_custom_op_t)(struct ggml_tensor * dst , int ith, int nth, void * userdata);
ggml_custom_op_t = ctypes.CFUNCTYPE(
    None,
    ctypes.POINTER(ggml_tensor),
    ctypes.c_int,
    ctypes.c_int,
    ctypes.c_void_p,
)
"""Custom operator on a tensor with variadic tensor inputs."""

# #define GGML_N_TASKS_MAX -1
GGML_N_TASKS_MAX = -1


# GGML_API struct ggml_tensor * ggml_map_custom1(
#         struct ggml_context   * ctx,
#         struct ggml_tensor    * a,
#         ggml_custom1_op_t       fun,
#         int                     n_tasks,
#         void                  * userdata);
@ggml_function(
    "ggml_map_custom1",
    [
        ggml_context_p_ctypes,
        ctypes.POINTER(ggml_tensor),
        ggml_custom1_op_t,
        ctypes.c_int,
        ctypes.c_void_p,
    ],
    ctypes.POINTER(ggml_tensor),
)
def ggml_map_custom1(
    ctx: ggml_context_p,
    a: ggml_tensor_p,
    fun: CtypesFuncPointer,  # type: ignore
    n_tasks: Union[ctypes.c_int, int],
    userdata: Union[ctypes.c_void_p, int, None],
) -> ggml_tensor_p:
    ...


# GGML_API struct ggml_tensor * ggml_map_custom1_inplace(
#         struct ggml_context   * ctx,
#         struct ggml_tensor    * a,
#         ggml_custom1_op_t       fun,
#         int                     n_tasks,
#         void                  * userdata);
@ggml_function(
    "ggml_map_custom1_inplace",
    [
        ggml_context_p_ctypes,
        ctypes.POINTER(ggml_tensor),
        ggml_custom1_op_t,
        ctypes.c_int,
        ctypes.c_void_p,
    ],
    ctypes.POINTER(ggml_tensor),
)
def ggml_map_custom1_inplace(
    ctx: ggml_context_p,
    a: ggml_tensor_p,
    fun: CtypesFuncPointer,  # type: ignore
    n_tasks: Union[ctypes.c_int, int],
    userdata: Union[ctypes.c_void_p, int, None],
) -> ggml_tensor_p:
    ...


# GGML_API struct ggml_tensor * ggml_map_custom2(
#         struct ggml_context   * ctx,
#         struct ggml_tensor    * a,
#         struct ggml_tensor    * b,
#         ggml_custom2_op_t       fun,
#         int                     n_tasks,
#         void                  * userdata);
@ggml_function(
    "ggml_map_custom2",
    [
        ggml_context_p_ctypes,
        ctypes.POINTER(ggml_tensor),
        ctypes.POINTER(ggml_tensor),
        ggml_custom2_op_t,
        ctypes.c_int,
        ctypes.c_void_p,
    ],
    ctypes.POINTER(ggml_tensor),
)
def ggml_map_custom2(
    ctx: ggml_context_p,
    a: ggml_tensor_p,
    b: ggml_tensor_p,
    fun: CtypesFuncPointer,  # type: ignore
    n_tasks: Union[ctypes.c_int, int],
    userdata: Union[ctypes.c_void_p, int, None],
) -> ggml_tensor_p:
    ...


# GGML_API struct ggml_tensor * ggml_map_custom2_inplace(
#         struct ggml_context   * ctx,
#         struct ggml_tensor    * a,
#         struct ggml_tensor    * b,
#         ggml_custom2_op_t       fun,
#         int                     n_tasks,
#         void                  * userdata);
@ggml_function(
    "ggml_map_custom2_inplace",
    [
        ggml_context_p_ctypes,
        ctypes.POINTER(ggml_tensor),
        ctypes.POINTER(ggml_tensor),
        ggml_custom2_op_t,
        ctypes.c_int,
        ctypes.c_void_p,
    ],
    ctypes.POINTER(ggml_tensor),
)
def ggml_map_custom2_inplace(
    ctx: ggml_context_p,
    a: ggml_tensor_p,
    b: ggml_tensor_p,
    fun: CtypesFuncPointer,  # type: ignore
    n_tasks: Union[ctypes.c_int, int],
    userdata: Union[ctypes.c_void_p, int, None],
) -> ggml_tensor_p:
    ...


# GGML_API struct ggml_tensor * ggml_map_custom3(
#         struct ggml_context   * ctx,
#         struct ggml_tensor    * a,
#         struct ggml_tensor    * b,
#         struct ggml_tensor    * c,
#         ggml_custom3_op_t       fun,
#         int                     n_tasks,
#         void                  * userdata);
@ggml_function(
    "ggml_map_custom3",
    [
        ggml_context_p_ctypes,
        ctypes.POINTER(ggml_tensor),
        ctypes.POINTER(ggml_tensor),
        ctypes.POINTER(ggml_tensor),
        ggml_custom3_op_t,
        ctypes.c_int,
        ctypes.c_void_p,
    ],
    ctypes.POINTER(ggml_tensor),
)
def ggml_map_custom3(
    ctx: ggml_context_p,
    a: ggml_tensor_p,
    b: ggml_tensor_p,
    c: ggml_tensor_p,
    fun: CtypesFuncPointer,  # type: ignore
    n_tasks: Union[ctypes.c_int, int],
    userdata: Union[ctypes.c_void_p, int, None],
) -> ggml_tensor_p:
    ...


# GGML_API struct ggml_tensor * ggml_map_custom3_inplace(
#         struct ggml_context   * ctx,
#         struct ggml_tensor    * a,
#         struct ggml_tensor    * b,
#         struct ggml_tensor    * c,
#         ggml_custom3_op_t       fun,
#         int                     n_tasks,
#         void                  * userdata);
@ggml_function(
    "ggml_map_custom3_inplace",
    [
        ggml_context_p_ctypes,
        ctypes.POINTER(ggml_tensor),
        ctypes.POINTER(ggml_tensor),
        ctypes.POINTER(ggml_tensor),
        ggml_custom3_op_t,
        ctypes.c_int,
        ctypes.c_void_p,
    ],
    ctypes.POINTER(ggml_tensor),
)
def ggml_map_custom3_inplace(
    ctx: ggml_context_p,
    a: ggml_tensor_p,
    b: ggml_tensor_p,
    c: ggml_tensor_p,
    fun: CtypesFuncPointer,  # type: ignore
    n_tasks: Union[ctypes.c_int, int],
    userdata: Union[ctypes.c_void_p, int, None],
) -> ggml_tensor_p:
    ...


# GGML_API struct ggml_tensor * ggml_custom_4d(
#         struct ggml_context * ctx,
#         enum ggml_type        type,
#         int64_t               ne0,
#         int64_t               ne1,
#         int64_t               ne2,
#         int64_t               ne3,
#         struct ggml_tensor ** args,
#         int                   n_args,
#         ggml_custom_op_t      fun,
#         int                   n_tasks,
#         void                * userdata);
@ggml_function(
    "ggml_custom_4d",
    [
        ggml_context_p_ctypes,
        ctypes.c_int,
        ctypes.c_int64,
        ctypes.c_int64,
        ctypes.c_int64,
        ctypes.c_int64,
        ctypes.POINTER(ctypes.POINTER(ggml_tensor)),
        ctypes.c_int,
        ggml_custom_op_t,
        ctypes.c_int,
        ctypes.c_void_p,
    ],
    ctypes.POINTER(ggml_tensor),
)
def ggml_custom_4d(
    ctx: ggml_context_p,
    type_: Union[ctypes.c_int, int],
    ne0: Union[ctypes.c_int64, int],
    ne1: Union[ctypes.c_int64, int],
    ne2: Union[ctypes.c_int64, int],
    ne3: Union[ctypes.c_int64, int],
    args: CtypesPointer[ggml_tensor_p],
    n_args: Union[ctypes.c_int, int],
    fun: CtypesFuncPointer,  # type: ignore
    n_tasks: Union[ctypes.c_int, int],
    userdata: Union[ctypes.c_void_p, int, None],
    /,
) -> ggml_tensor_p:
    ...


# GGML_API struct ggml_tensor * ggml_custom_inplace(
#         struct ggml_context * ctx,
#         struct ggml_tensor  * a,
#         struct ggml_tensor ** args,
#         int                   n_args,
#         ggml_custom_op_t      fun,
#         int                   n_tasks,
#         void                * userdata);
@ggml_function(
    "ggml_custom_inplace",
    [
        ggml_context_p_ctypes,
        ctypes.POINTER(ggml_tensor),
        ctypes.POINTER(ctypes.POINTER(ggml_tensor)),
        ctypes.c_int,
        ggml_custom_op_t,
        ctypes.c_int,
        ctypes.c_void_p,
    ],
    ctypes.POINTER(ggml_tensor),
)
def ggml_custom_inplace(
    ctx: ggml_context_p,
    a: ggml_tensor_p,
    args: CtypesPointer[ggml_tensor_p],
    n_args: Union[ctypes.c_int, int],
    fun: CtypesFuncPointer,  # type: ignore
    n_tasks: Union[ctypes.c_int, int],
    userdata: Union[ctypes.c_void_p, int, None],
    /,
) -> ggml_tensor_p:
    ...


# // loss function


# GGML_API struct ggml_tensor * ggml_cross_entropy_loss(
#         struct ggml_context         * ctx,
#         struct ggml_tensor          * a,
#         struct ggml_tensor          * b);
@ggml_function(
    "ggml_cross_entropy_loss",
    [
        ggml_context_p_ctypes,
        ctypes.POINTER(ggml_tensor),
        ctypes.POINTER(ggml_tensor),
    ],
    ctypes.POINTER(ggml_tensor),
)
def ggml_cross_entropy_loss(
    ctx: ggml_context_p,
    a: ggml_tensor_p,
    b: ggml_tensor_p,
) -> ggml_tensor_p:
    ...


# GGML_API struct ggml_tensor * ggml_cross_entropy_loss_back(
#         struct ggml_context         * ctx,
#         struct ggml_tensor          * a,
#         struct ggml_tensor          * b,
#         struct ggml_tensor          * c);
@ggml_function(
    "ggml_cross_entropy_loss_back",
    [
        ggml_context_p_ctypes,
        ctypes.POINTER(ggml_tensor),
        ctypes.POINTER(ggml_tensor),
        ctypes.POINTER(ggml_tensor),
    ],
    ctypes.POINTER(ggml_tensor),
)
def ggml_cross_entropy_loss_back(
    ctx: ggml_context_p,
    a: ggml_tensor_p,
    b: ggml_tensor_p,
    c: ggml_tensor_p,
) -> ggml_tensor_p:
    ...


# GGML_API struct ggml_tensor * ggml_opt_step_adamw(
#         struct ggml_context * ctx,
#         struct ggml_tensor  * a,
#         struct ggml_tensor  * grad,
#         struct ggml_tensor  * m,
#         struct ggml_tensor  * v,
#         struct ggml_tensor  * adamw_params);
@ggml_function(
    "ggml_opt_step_adamw",
    [
        ggml_context_p_ctypes,
        ctypes.POINTER(ggml_tensor),
        ctypes.POINTER(ggml_tensor),
        ctypes.POINTER(ggml_tensor),
        ctypes.POINTER(ggml_tensor),
        ctypes.POINTER(ggml_tensor),
    ],
    ctypes.POINTER(ggml_tensor),
)
def ggml_opt_step_adamw(
    ctx: ggml_context_p,
    a: ggml_tensor_p,
    grad: ggml_tensor_p,
    m: ggml_tensor_p,
    v: ggml_tensor_p,
    adamw_params: ggml_tensor_p,
    /,
) -> ggml_tensor_p:
    ...


# GGML_API struct ggml_tensor * ggml_opt_step_sgd(
#         struct ggml_context * ctx,
#         struct ggml_tensor  * a,
#         struct ggml_tensor  * grad,
#         struct ggml_tensor  * sgd_params);
@ggml_function(
    "ggml_opt_step_sgd",
    [
        ggml_context_p_ctypes,
        ctypes.POINTER(ggml_tensor),
        ctypes.POINTER(ggml_tensor),
        ctypes.POINTER(ggml_tensor),
    ],
    ctypes.POINTER(ggml_tensor),
)
def ggml_opt_step_sgd(
    ctx: ggml_context_p,
    a: ggml_tensor_p,
    grad: ggml_tensor_p,
    sgd_params: ggml_tensor_p,
    /,
) -> ggml_tensor_p:
    ...


# //
# // automatic differentiation
# //


# GGML_API void ggml_set_param(struct ggml_tensor * tensor);
_ggml_set_param = lib.ggml_set_param
_ggml_set_param.argtypes = [ctypes.POINTER(ggml_tensor)]
_ggml_set_param.restype = None


def ggml_set_param(*args: Any):
    if len(args) == 1:
        tensor = args[0]
    elif len(args) == 2:
        tensor = args[1]
    else:
        raise TypeError("ggml_set_param expects tensor or ctx, tensor")
    return _ggml_set_param(tensor)


# GGML_API void ggml_build_forward_expand (struct ggml_cgraph * cgraph, struct ggml_tensor * tensor);
@ggml_function(
    "ggml_build_forward_expand",
    [
        ctypes.POINTER(ggml_cgraph),
        ctypes.POINTER(ggml_tensor),
    ],
    None,
)
def ggml_build_forward_expand(
    cgraph: ggml_cgraph_p,
    tensor: ggml_tensor_p,
):
    """Add a tensor to the forward computation graph. This is used to
    compute and save the value of the tensor.

    Parameters:
        cgraph: The graph.
        tensor: The tensor."""
    ...


# GGML_API struct ggml_tensor * ggml_build_forward_select(
#         struct ggml_cgraph * cgraph,
#         struct ggml_tensor ** tensors,
#         int n_tensors,
#         int idx);
@ggml_function(
    "ggml_build_forward_select",
    [
        ctypes.POINTER(ggml_cgraph),
        ctypes.POINTER(ctypes.POINTER(ggml_tensor)),
        ctypes.c_int,
        ctypes.c_int,
    ],
    ctypes.POINTER(ggml_tensor),
)
def ggml_build_forward_select(
    cgraph: ggml_cgraph_p,
    tensors: CtypesPointer[ggml_tensor_p],
    n_tensors: Union[ctypes.c_int, int],
    idx: Union[ctypes.c_int, int],
    /,
) -> ggml_tensor_p:
    ...


# GGML_API void ggml_build_backward_expand(
#     struct ggml_context * ctx,
#     struct ggml_cgraph  * cgraph,
#     struct ggml_tensor ** grad_accs);
_ggml_build_backward_expand = lib.ggml_build_backward_expand
_ggml_build_backward_expand.argtypes = [
    ggml_context_p_ctypes,
    ctypes.POINTER(ggml_cgraph),
    ctypes.POINTER(ctypes.POINTER(ggml_tensor)),
]
_ggml_build_backward_expand.restype = None


def ggml_build_backward_expand(*args: Any):
    """Add backward pass nodes to a graph.

    Parameters:
        args: Either `(ctx, cgraph, grad_accs)` or the legacy
            `(ctx, gf, gb, keep)` call shape."""
    if len(args) == 3:
        ctx, cgraph, grad_accs = args
    elif len(args) == 4:
        ctx, _gf, cgraph, _keep = args
        grad_accs = None
    else:
        raise TypeError("ggml_build_backward_expand expects ctx, cgraph, grad_accs or ctx, gf, gb, keep")
    return _ggml_build_backward_expand(ctx, cgraph, grad_accs)


# // graph allocation in a context
# GGML_API struct ggml_cgraph * ggml_new_graph         (struct ggml_context * ctx); // size = GGML_DEFAULT_GRAPH_SIZE, grads = false
@ggml_function("ggml_new_graph", [ggml_context_p_ctypes], ctypes.POINTER(ggml_cgraph))
def ggml_new_graph(ctx: ggml_context_p) -> ggml_cgraph_p:
    """Create a new graph.

    Parameters:
        ctx: The context.

    Returns:
        The graph."""
    ...


# GGML_API struct ggml_cgraph * ggml_new_graph_custom  (struct ggml_context * ctx, size_t size, bool grads);
@ggml_function(
    "ggml_new_graph_custom",
    [ggml_context_p_ctypes, ctypes.c_size_t, ctypes.c_bool],
    ctypes.POINTER(ggml_cgraph),
)
def ggml_new_graph_custom(
    ctx: ggml_context_p,
    size: Union[ctypes.c_size_t, int],
    grads: Union[ctypes.c_bool, bool],
) -> ggml_cgraph_p:
    """Create a new graph with custom size and grads.

    Parameters:
        ctx: The context.
        size: The size of the graph.
        grads: Whether to keep the gradients.

    Returns:
        The graph."""
    ...


# GGML_API struct ggml_cgraph * ggml_graph_dup         (struct ggml_context * ctx, struct ggml_cgraph * cgraph, bool force_grads);
_ggml_graph_dup = lib.ggml_graph_dup
_ggml_graph_dup.argtypes = [
    ggml_context_p_ctypes,
    ctypes.POINTER(ggml_cgraph),
    ctypes.c_bool,
]
_ggml_graph_dup.restype = ctypes.POINTER(ggml_cgraph)


def ggml_graph_dup(
    ctx: ggml_context_p,
    cgraph: ggml_cgraph_p,
    force_grads: Union[ctypes.c_bool, bool] = False,
) -> ggml_cgraph_p:
    """Duplicate a graph.

    Parameters:
        ctx: The context.
        cgraph: The graph.
        force_grads: Whether to force allocation of graph gradients.

    Returns:
        The graph."""
    return _ggml_graph_dup(ctx, cgraph, force_grads)


# GGML_API struct ggml_cgraph   ggml_graph_view        (struct ggml_cgraph * cgraph, int i0, int i1);
@ggml_function(
    "ggml_graph_view",
    [ctypes.POINTER(ggml_cgraph), ctypes.c_int, ctypes.c_int],
    ggml_cgraph,
    enabled=hasattr(lib, "ggml_graph_view"),
)
def ggml_graph_view(
    cgraph: ggml_cgraph_p,
    i0: Union[ctypes.c_int, int],
    i1: Union[ctypes.c_int, int],
) -> ggml_cgraph:
    """View a graph.

    Parameters:
        cgraph: The graph.
        i0: The start index.
        i1: The end index.

    Returns:
        The graph."""
    ...


# GGML_API void                 ggml_graph_cpy         (struct ggml_cgraph * src, struct ggml_cgraph * dst);
@ggml_function(
    "ggml_graph_cpy", [ctypes.POINTER(ggml_cgraph), ctypes.POINTER(ggml_cgraph)], None
)
def ggml_graph_cpy(
    src: ggml_cgraph_p,
    dst: ggml_cgraph_p,
):
    """Copy a graph.

    Parameters:
        src: The source graph.
        dst: The destination graph."""
    ...


# GGML_API void                 ggml_graph_reset       (struct ggml_cgraph * cgraph);  // zero grads
@ggml_function("ggml_graph_reset", [ctypes.POINTER(ggml_cgraph)], None)
def ggml_graph_reset(
    cgraph: ggml_cgraph_p,
):
    """Reset a graph.

    Parameters:
        cgraph: The graph."""
    ...


# GGML_API void                 ggml_graph_clear       (struct ggml_cgraph * cgraph);
@ggml_function("ggml_graph_clear", [ctypes.POINTER(ggml_cgraph)], None)
def ggml_graph_clear(
    cgraph: ggml_cgraph_p,
):
    """Clear a graph.

    Parameters:
        cgraph: The graph."""
    ...


# GGML_API size_t ggml_graph_overhead(void);
@ggml_function("ggml_graph_overhead", [], ctypes.c_size_t)
def ggml_graph_overhead() -> int:
    """Get the overhead of the graph."""
    ...


# GGML_API size_t ggml_graph_overhead_custom(size_t size, bool grads);
@ggml_function(
    "ggml_graph_overhead_custom", [ctypes.c_size_t, ctypes.c_bool], ctypes.c_size_t
)
def ggml_graph_overhead_custom(
    size: Union[ctypes.c_size_t, int],
    grads: Union[ctypes.c_bool, bool],
) -> int:
    ...


# GGML_API int ggml_graph_size(struct ggml_cgraph * cgraph);
@ggml_function("ggml_graph_size", [ctypes.POINTER(ggml_cgraph)], ctypes.c_int)
def ggml_graph_size(cgraph: ggml_cgraph_p, /) -> int:
    ...


# GGML_API struct ggml_tensor * ggml_graph_node(struct ggml_cgraph * cgraph, int i);
@ggml_function(
    "ggml_graph_node",
    [ctypes.POINTER(ggml_cgraph), ctypes.c_int],
    ctypes.POINTER(ggml_tensor),
)
def ggml_graph_node(
    cgraph: ggml_cgraph_p,
    i: Union[ctypes.c_int, int],
    /,
) -> ggml_tensor_p:
    ...


# GGML_API struct ggml_tensor ** ggml_graph_nodes(struct ggml_cgraph * cgraph);
@ggml_function(
    "ggml_graph_nodes",
    [ctypes.POINTER(ggml_cgraph)],
    ctypes.POINTER(ctypes.POINTER(ggml_tensor)),
)
def ggml_graph_nodes(cgraph: ggml_cgraph_p, /) -> CtypesPointer[ggml_tensor_p]:
    ...


# GGML_API int ggml_graph_n_nodes(struct ggml_cgraph * cgraph);
@ggml_function("ggml_graph_n_nodes", [ctypes.POINTER(ggml_cgraph)], ctypes.c_int)
def ggml_graph_n_nodes(cgraph: ggml_cgraph_p, /) -> int:
    ...


# GGML_API void ggml_graph_add_node(struct ggml_cgraph * cgraph, struct ggml_tensor * tensor);
@ggml_function(
    "ggml_graph_add_node",
    [ctypes.POINTER(ggml_cgraph), ctypes.POINTER(ggml_tensor)],
    None,
)
def ggml_graph_add_node(
    cgraph: ggml_cgraph_p,
    tensor: ggml_tensor_p,
    /,
) -> None:
    ...


# // ggml_graph_plan() has to be called before ggml_graph_compute()
# // when plan.work_size > 0, caller must allocate memory for plan.work_data
# GGML_API struct ggml_cplan ggml_graph_plan(const struct ggml_cgraph * cgraph, int n_threads, struct ggml_threadpool * threadpool);
_ggml_graph_plan = lib.ggml_graph_plan
_ggml_graph_plan.argtypes = [
    ctypes.POINTER(ggml_cgraph),
    ctypes.c_int,
    ctypes.c_void_p,
]
_ggml_graph_plan.restype = ggml_cplan


def ggml_graph_plan(
    cgraph: ggml_cgraph_p,
    n_threads: Union[ctypes.c_int, int] = GGML_DEFAULT_N_THREADS,
    threadpool: Union[ctypes.c_void_p, int, None] = None,
) -> ggml_cplan:
    """Plan the computation graph.

    Parameters:
        cgraph: The graph.
        n_threads: The number of threads to use.
        threadpool: Optional ggml_threadpool pointer.

    Returns:
        The plan."""
    return _ggml_graph_plan(cgraph, n_threads, threadpool)


# GGML_API enum ggml_status  ggml_graph_compute         (      struct ggml_cgraph * cgraph, struct ggml_cplan * cplan);
@ggml_function(
    "ggml_graph_compute",
    [
        ctypes.POINTER(ggml_cgraph),
        ctypes.POINTER(ggml_cplan),
    ],
    ctypes.c_int,
)
def ggml_graph_compute(
    cgraph: ggml_cgraph_p,
    cplan: ggml_cplan_p,
) -> int:
    ...


# // same as ggml_graph_compute() but the work data is allocated as a part of the context
# // note: the drawback of this API is that you must have ensured that the context has enough memory for the work data
# GGML_API enum ggml_status  ggml_graph_compute_with_ctx(struct ggml_context * ctx, struct ggml_cgraph * cgraph, int n_threads);
@ggml_function(
    "ggml_graph_compute_with_ctx",
    [
        ggml_context_p_ctypes,
        ctypes.POINTER(ggml_cgraph),
        ctypes.c_int,
    ],
    ctypes.c_int,
)
def ggml_graph_compute_with_ctx(
    ctx: ggml_context_p,
    cgraph: ggml_cgraph_p,
    n_threads: Union[ctypes.c_int, int],
) -> int:
    """Compute the graph with a context.

    Parameters:
        ctx: The context.
        cgraph: The graph.
        n_threads: The number of threads to use."""
    ...


# GGML_API struct ggml_tensor * ggml_graph_get_tensor(struct ggml_cgraph * cgraph, const char * name);
@ggml_function(
    "ggml_graph_get_tensor",
    [
        ctypes.POINTER(ggml_cgraph),
        ctypes.c_char_p,
    ],
    ctypes.POINTER(ggml_tensor),
)
def ggml_graph_get_tensor(
    cgraph: ggml_cgraph_p,
    name: bytes,
) -> ggml_tensor_p:
    """Get a tensor from the graph by name.

    Parameters:
        cgraph: The graph.
        name: The name of the tensor.

    Returns:
        The tensor."""
    ...


# GGML_API struct ggml_tensor * ggml_graph_get_grad(const struct ggml_cgraph * cgraph, const struct ggml_tensor * node);
@ggml_function(
    "ggml_graph_get_grad",
    [
        ctypes.POINTER(ggml_cgraph),
        ctypes.POINTER(ggml_tensor),
    ],
    ctypes.POINTER(ggml_tensor),
)
def ggml_graph_get_grad(
    cgraph: ggml_cgraph_p,
    node: ggml_tensor_p,
) -> ggml_tensor_p:
    ...


# GGML_API struct ggml_tensor * ggml_graph_get_grad_acc(const struct ggml_cgraph * cgraph, const struct ggml_tensor * node);
@ggml_function(
    "ggml_graph_get_grad_acc",
    [
        ctypes.POINTER(ggml_cgraph),
        ctypes.POINTER(ggml_tensor),
    ],
    ctypes.POINTER(ggml_tensor),
)
def ggml_graph_get_grad_acc(
    cgraph: ggml_cgraph_p,
    node: ggml_tensor_p,
) -> ggml_tensor_p:
    ...


# GGML_API void                 ggml_graph_export(const struct ggml_cgraph * cgraph, const char * fname);
@ggml_function(
    "ggml_graph_export",
    [
        ctypes.POINTER(ggml_cgraph),
        ctypes.c_char_p,
    ],
    None,
    enabled=hasattr(lib, "ggml_graph_export"),
)
def ggml_graph_export(
    cgraph: ggml_cgraph_p,
    fname: bytes,
):
    ...


# GGML_API struct ggml_cgraph * ggml_graph_import(const char * fname, struct ggml_context ** ctx_data, struct ggml_context ** ctx_eval);
@ggml_function(
    "ggml_graph_import",
    [
        ctypes.c_char_p,
        ctypes.POINTER(ggml_context_p_ctypes),
        ctypes.POINTER(ggml_context_p_ctypes),
    ],
    ctypes.POINTER(ggml_cgraph),
    enabled=hasattr(lib, "ggml_graph_import"),
)
def ggml_graph_import(
    fname: bytes,
    ctx_data: "ctypes._Pointer[ggml_context_p]",  # type: ignore
    ctx_eval: "ctypes._Pointer[ggml_context_p]",  # type: ignore
) -> ggml_cgraph_p:
    ...


# // print info and performance information for the graph
# GGML_API void ggml_graph_print(const struct ggml_cgraph * cgraph);
@ggml_function("ggml_graph_print", [ctypes.POINTER(ggml_cgraph)], None)
def ggml_graph_print(
    cgraph: ggml_cgraph_p,
):
    ...


# // dump the graph into a file using the dot format
# GGML_API void ggml_graph_dump_dot(const struct ggml_cgraph * gb, const struct ggml_cgraph * gf, const char * filename);
@ggml_function(
    "ggml_graph_dump_dot",
    [
        ctypes.POINTER(ggml_cgraph),
        ctypes.POINTER(ggml_cgraph),
        ctypes.c_char_p,
    ],
    None,
)
def ggml_graph_dump_dot(
    gb: ggml_cgraph_p,
    gf: ggml_cgraph_p,
    filename: bytes,
):
    ...


# // build gradient checkpointing backward graph gb for gf using provided checkpoints
# // gb_tmp will contain original backward graph with rewritten backward process nodes,
# // but without the second forward pass nodes.
# GGML_API void ggml_build_backward_gradient_checkpointing(
#         struct ggml_context   * ctx,
#         struct ggml_cgraph    * gf,
#         struct ggml_cgraph    * gb,
#         struct ggml_cgraph    * gb_tmp,
#         struct ggml_tensor  * * checkpoints,
#         int                     n_checkpoints);
@ggml_function(
    "ggml_build_backward_gradient_checkpointing",
    [
        ggml_context_p_ctypes,
        ctypes.POINTER(ggml_cgraph),
        ctypes.POINTER(ggml_cgraph),
        ctypes.POINTER(ggml_cgraph),
        ctypes.POINTER(ctypes.POINTER(ggml_tensor)),
        ctypes.c_int,
    ],
    None,
    enabled=hasattr(lib, "ggml_build_backward_gradient_checkpointing"),
)
def ggml_build_backward_gradient_checkpointing(
    ctx: ggml_context_p,
    gf: ggml_cgraph_p,
    gb: ggml_cgraph_p,
    gb_tmp: ggml_cgraph_p,
    checkpoints: "ctypes._Pointer[ggml_tensor_p]",  # type: ignore
    n_checkpoints: Union[ctypes.c_int, int],
):
    ...


# //
# // optimization
# //

# typedef void (*ggml_log_callback)(enum ggml_log_level level, const char * text, void * user_data);
ggml_log_callback = ctypes.CFUNCTYPE(
    None, ctypes.c_int, ctypes.c_char_p, ctypes.c_void_p
)


# GGML_API void ggml_log_get(ggml_log_callback * log_callback, void ** user_data);
@ggml_function(
    "ggml_log_get",
    [
        ctypes.POINTER(ggml_log_callback),
        ctypes.POINTER(ctypes.c_void_p),
    ],
    None,
)
def ggml_log_get(
    log_callback: CtypesPointer[ggml_log_callback],
    user_data: CtypesPointer[ctypes.c_void_p],
    /,
) -> None:
    ...


# GGML_API void ggml_log_set(ggml_log_callback log_callback, void * user_data);
@ggml_function(
    "ggml_log_set",
    [
        ggml_log_callback,
        ctypes.c_void_p,
    ],
    None,
)
def ggml_log_set(
    log_callback: CtypesFuncPointer,  # type: ignore
    user_data: Union[ctypes.c_void_p, int, None],
    /,
) -> None:
    ...


# struct ggml_opt_dataset;
ggml_opt_dataset_t = NewType("ggml_opt_dataset_t", int)
ggml_opt_dataset_t_ctypes: TypeAlias = ctypes.c_void_p

# struct ggml_opt_context;
ggml_opt_context_t = NewType("ggml_opt_context_t", int)
ggml_opt_context_t_ctypes: TypeAlias = ctypes.c_void_p

# struct ggml_opt_result;
ggml_opt_result_t = NewType("ggml_opt_result_t", int)
ggml_opt_result_t_ctypes: TypeAlias = ctypes.c_void_p

# enum ggml_opt_loss_type {
#     GGML_OPT_LOSS_TYPE_MEAN,
#     GGML_OPT_LOSS_TYPE_SUM,
#     GGML_OPT_LOSS_TYPE_CROSS_ENTROPY,
#     GGML_OPT_LOSS_TYPE_MEAN_SQUARED_ERROR,
# };
GGML_OPT_LOSS_TYPE_MEAN = 0
GGML_OPT_LOSS_TYPE_SUM = 1
GGML_OPT_LOSS_TYPE_CROSS_ENTROPY = 2
GGML_OPT_LOSS_TYPE_MEAN_SQUARED_ERROR = 3

# enum ggml_opt_build_type {
#     GGML_OPT_BUILD_TYPE_FORWARD = 10,
#     GGML_OPT_BUILD_TYPE_GRAD    = 20,
#     GGML_OPT_BUILD_TYPE_OPT     = 30,
# };
GGML_OPT_BUILD_TYPE_FORWARD = 10
GGML_OPT_BUILD_TYPE_GRAD = 20
GGML_OPT_BUILD_TYPE_OPT = 30

# enum ggml_opt_optimizer_type {
#     GGML_OPT_OPTIMIZER_TYPE_ADAMW,
#     GGML_OPT_OPTIMIZER_TYPE_SGD,
#     GGML_OPT_OPTIMIZER_TYPE_COUNT
# };
GGML_OPT_OPTIMIZER_TYPE_ADAMW = 0
GGML_OPT_OPTIMIZER_TYPE_SGD = 1
GGML_OPT_OPTIMIZER_TYPE_COUNT = 2


# struct ggml_opt_optimizer_params {
#     struct {
#         float alpha; // learning rate
#         float beta1; // first AdamW momentum
#         float beta2; // second AdamW momentum
#         float eps;   // epsilon for numerical stability
#         float wd;    // weight decay - 0.0f to disable
#     } adamw;
class ggml_opt_optimizer_params_adamw(ctypes.Structure):
    _fields_ = [
        ("alpha", ctypes.c_float),
        ("beta1", ctypes.c_float),
        ("beta2", ctypes.c_float),
        ("eps", ctypes.c_float),
        ("wd", ctypes.c_float),
    ]


#     struct {
#         float alpha; // learning rate
#         float wd;    // weight decay
#     } sgd;
# };
class ggml_opt_optimizer_params_sgd(ctypes.Structure):
    _fields_ = [
        ("alpha", ctypes.c_float),
        ("wd", ctypes.c_float),
    ]


# parameters that control which optimizer is used and how said optimizer tries to find the minimal loss
class ggml_opt_optimizer_params(ctypes.Structure):
    _fields_ = [
        ("adamw", ggml_opt_optimizer_params_adamw),
        ("sgd", ggml_opt_optimizer_params_sgd),
    ]


# typedef struct ggml_opt_optimizer_params (*ggml_opt_get_optimizer_params)(void * userdata);
# ctypes cannot create Python callbacks that return structs by value, so this
# callback is exposed as a raw native function pointer.
ggml_opt_get_optimizer_params = ctypes.c_void_p


# struct ggml_opt_params {
#     ggml_backend_sched_t backend_sched; // defines which backends are used to construct the compute graphs
#
#     // by default the forward graph needs to be reconstructed for each eval
#     // if ctx_compute, inputs, and outputs are set the graphs are instead allocated statically
#     struct ggml_context * ctx_compute;
#     struct ggml_tensor  * inputs;
#     struct ggml_tensor  * outputs;
#
#     enum ggml_opt_loss_type  loss_type;
#     enum ggml_opt_build_type build_type;
#
#     int32_t opt_period; // after how many gradient accumulation steps an optimizer step should be done
#
#     ggml_opt_get_optimizer_params get_opt_pars;    // callback for calculating optimizer parameters
#     void *                        get_opt_pars_ud; // userdata for calculating optimizer parameters
#
#     // only GGML_OPT_OPTIMIZER_TYPE_ADAMW needs m, v momenta per parameter tensor
#     enum ggml_opt_optimizer_type optimizer;
# };
class ggml_opt_params(ctypes.Structure):
    _fields_ = [
        ("backend_sched", ctypes.c_void_p),
        ("ctx_compute", ggml_context_p_ctypes),
        ("inputs", ctypes.POINTER(ggml_tensor)),
        ("outputs", ctypes.POINTER(ggml_tensor)),
        ("loss_type", ctypes.c_int),
        ("build_type", ctypes.c_int),
        ("opt_period", ctypes.c_int32),
        ("get_opt_pars", ggml_opt_get_optimizer_params),
        ("get_opt_pars_ud", ctypes.c_void_p),
        ("optimizer", ctypes.c_int),
    ]


# typedef void (*ggml_opt_epoch_callback)(bool train, ggml_opt_context_t opt_ctx, ggml_opt_dataset_t dataset, ggml_opt_result_t result, int64_t ibatch, int64_t ibatch_max, int64_t t_start_us);
ggml_opt_epoch_callback = ctypes.CFUNCTYPE(
    None,
    ctypes.c_bool,
    ggml_opt_context_t_ctypes,
    ggml_opt_dataset_t_ctypes,
    ggml_opt_result_t_ctypes,
    ctypes.c_int64,
    ctypes.c_int64,
    ctypes.c_int64,
)


# GGML_API ggml_opt_dataset_t ggml_opt_dataset_init(enum ggml_type type_data, enum ggml_type type_label, int64_t ne_datapoint, int64_t ne_label, int64_t ndata, int64_t ndata_shard);
@ggml_function(
    "ggml_opt_dataset_init",
    [
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int64,
        ctypes.c_int64,
        ctypes.c_int64,
        ctypes.c_int64,
    ],
    ggml_opt_dataset_t_ctypes,
)
def ggml_opt_dataset_init(
    type_data: Union[ctypes.c_int, int],
    type_label: Union[ctypes.c_int, int],
    ne_datapoint: Union[ctypes.c_int64, int],
    ne_label: Union[ctypes.c_int64, int],
    ndata: Union[ctypes.c_int64, int],
    ndata_shard: Union[ctypes.c_int64, int],
    /,
) -> Optional[ggml_opt_dataset_t]:
    ...


# GGML_API void ggml_opt_dataset_free(ggml_opt_dataset_t dataset);
@ggml_function("ggml_opt_dataset_free", [ggml_opt_dataset_t_ctypes], None)
def ggml_opt_dataset_free(dataset: Union[ggml_opt_dataset_t, int], /):
    ...


# GGML_API int64_t ggml_opt_dataset_ndata(ggml_opt_dataset_t dataset);
@ggml_function("ggml_opt_dataset_ndata", [ggml_opt_dataset_t_ctypes], ctypes.c_int64)
def ggml_opt_dataset_ndata(dataset: Union[ggml_opt_dataset_t, int], /) -> int:
    ...


# GGML_API struct ggml_tensor * ggml_opt_dataset_data(ggml_opt_dataset_t dataset);
@ggml_function(
    "ggml_opt_dataset_data",
    [ggml_opt_dataset_t_ctypes],
    ctypes.POINTER(ggml_tensor),
)
def ggml_opt_dataset_data(
    dataset: Union[ggml_opt_dataset_t, int],
) -> Optional[ggml_tensor_p]:
    ...


# GGML_API struct ggml_tensor * ggml_opt_dataset_labels(ggml_opt_dataset_t dataset);
@ggml_function(
    "ggml_opt_dataset_labels",
    [ggml_opt_dataset_t_ctypes],
    ctypes.POINTER(ggml_tensor),
)
def ggml_opt_dataset_labels(
    dataset: Union[ggml_opt_dataset_t, int],
) -> Optional[ggml_tensor_p]:
    ...


# GGML_API void ggml_opt_dataset_shuffle(ggml_opt_context_t opt_ctx, ggml_opt_dataset_t dataset, int64_t idata);
@ggml_function(
    "ggml_opt_dataset_shuffle",
    [ggml_opt_context_t_ctypes, ggml_opt_dataset_t_ctypes, ctypes.c_int64],
    None,
)
def ggml_opt_dataset_shuffle(
    opt_ctx: Union[ggml_opt_context_t, int],
    dataset: Union[ggml_opt_dataset_t, int],
    idata: Union[ctypes.c_int64, int],
    /,
):
    ...


# GGML_API void ggml_opt_dataset_get_batch(ggml_opt_dataset_t dataset, struct ggml_tensor * data_batch, struct ggml_tensor * labels_batch, int64_t ibatch);
@ggml_function(
    "ggml_opt_dataset_get_batch",
    [
        ggml_opt_dataset_t_ctypes,
        ctypes.POINTER(ggml_tensor),
        ctypes.POINTER(ggml_tensor),
        ctypes.c_int64,
    ],
    None,
)
def ggml_opt_dataset_get_batch(
    dataset: Union[ggml_opt_dataset_t, int],
    data_batch: ggml_tensor_p,
    labels_batch: ggml_tensor_p,
    ibatch: Union[ctypes.c_int64, int],
    /,
):
    ...


# GGML_API void ggml_opt_dataset_get_batch_host(ggml_opt_dataset_t dataset, void * data_batch, size_t nb_data_batch, void * labels_batch, int64_t ibatch);
@ggml_function(
    "ggml_opt_dataset_get_batch_host",
    [
        ggml_opt_dataset_t_ctypes,
        ctypes.c_void_p,
        ctypes.c_size_t,
        ctypes.c_void_p,
        ctypes.c_int64,
    ],
    None,
)
def ggml_opt_dataset_get_batch_host(
    dataset: Union[ggml_opt_dataset_t, int],
    data_batch: Union[ctypes.c_void_p, int, None],
    nb_data_batch: Union[ctypes.c_size_t, int],
    labels_batch: Union[ctypes.c_void_p, int, None],
    ibatch: Union[ctypes.c_int64, int],
    /,
):
    ...


# GGML_API struct ggml_opt_optimizer_params ggml_opt_get_default_optimizer_params(void * userdata);
@ggml_function(
    "ggml_opt_get_default_optimizer_params",
    [ctypes.c_void_p],
    ggml_opt_optimizer_params,
)
def ggml_opt_get_default_optimizer_params(
    userdata: Union[ctypes.c_void_p, int, None],
) -> ggml_opt_optimizer_params:
    ...


# GGML_API struct ggml_opt_optimizer_params ggml_opt_get_constant_optimizer_params(void * userdata);
@ggml_function(
    "ggml_opt_get_constant_optimizer_params",
    [ctypes.c_void_p],
    ggml_opt_optimizer_params,
)
def ggml_opt_get_constant_optimizer_params(
    userdata: Union[ctypes.c_void_p, int, None],
) -> ggml_opt_optimizer_params:
    ...


# GGML_API struct ggml_opt_params ggml_opt_default_params(ggml_backend_sched_t backend_sched, enum ggml_opt_loss_type loss_type);
@ggml_function(
    "ggml_opt_default_params",
    [ctypes.c_void_p, ctypes.c_int],
    ggml_opt_params,
)
def ggml_opt_default_params(
    backend_sched: Union[ctypes.c_void_p, int, None],
    loss_type: Union[ctypes.c_int, int],
    /,
) -> ggml_opt_params:
    ...


# GGML_API ggml_opt_context_t ggml_opt_init(struct ggml_opt_params params);
@ggml_function("ggml_opt_init", [ggml_opt_params], ggml_opt_context_t_ctypes)
def ggml_opt_init(params: ggml_opt_params, /) -> Optional[ggml_opt_context_t]:
    ...


# GGML_API void ggml_opt_free(ggml_opt_context_t opt_ctx);
@ggml_function("ggml_opt_free", [ggml_opt_context_t_ctypes], None)
def ggml_opt_free(opt_ctx: Union[ggml_opt_context_t, int], /):
    ...


# GGML_API void ggml_opt_reset(ggml_opt_context_t opt_ctx, bool optimizer);
@ggml_function("ggml_opt_reset", [ggml_opt_context_t_ctypes, ctypes.c_bool], None)
def ggml_opt_reset(
    opt_ctx: Union[ggml_opt_context_t, int],
    optimizer: Union[ctypes.c_bool, bool],
    /,
):
    ...


# GGML_API bool ggml_opt_static_graphs(ggml_opt_context_t opt_ctx);
@ggml_function("ggml_opt_static_graphs", [ggml_opt_context_t_ctypes], ctypes.c_bool)
def ggml_opt_static_graphs(opt_ctx: Union[ggml_opt_context_t, int], /) -> bool:
    ...


# GGML_API struct ggml_tensor * ggml_opt_inputs(ggml_opt_context_t opt_ctx);
@ggml_function(
    "ggml_opt_inputs", [ggml_opt_context_t_ctypes], ctypes.POINTER(ggml_tensor)
)
def ggml_opt_inputs(opt_ctx: Union[ggml_opt_context_t, int], /) -> Optional[ggml_tensor_p]:
    ...


# GGML_API struct ggml_tensor * ggml_opt_outputs(ggml_opt_context_t opt_ctx);
@ggml_function(
    "ggml_opt_outputs", [ggml_opt_context_t_ctypes], ctypes.POINTER(ggml_tensor)
)
def ggml_opt_outputs(opt_ctx: Union[ggml_opt_context_t, int], /) -> Optional[ggml_tensor_p]:
    ...


# GGML_API struct ggml_tensor * ggml_opt_labels(ggml_opt_context_t opt_ctx);
@ggml_function(
    "ggml_opt_labels", [ggml_opt_context_t_ctypes], ctypes.POINTER(ggml_tensor)
)
def ggml_opt_labels(opt_ctx: Union[ggml_opt_context_t, int], /) -> Optional[ggml_tensor_p]:
    ...


# GGML_API struct ggml_tensor * ggml_opt_loss(ggml_opt_context_t opt_ctx);
@ggml_function(
    "ggml_opt_loss", [ggml_opt_context_t_ctypes], ctypes.POINTER(ggml_tensor)
)
def ggml_opt_loss(opt_ctx: Union[ggml_opt_context_t, int], /) -> Optional[ggml_tensor_p]:
    ...


# GGML_API struct ggml_tensor * ggml_opt_pred(ggml_opt_context_t opt_ctx);
@ggml_function(
    "ggml_opt_pred", [ggml_opt_context_t_ctypes], ctypes.POINTER(ggml_tensor)
)
def ggml_opt_pred(opt_ctx: Union[ggml_opt_context_t, int], /) -> Optional[ggml_tensor_p]:
    ...


# GGML_API struct ggml_tensor * ggml_opt_ncorrect(ggml_opt_context_t opt_ctx);
@ggml_function(
    "ggml_opt_ncorrect", [ggml_opt_context_t_ctypes], ctypes.POINTER(ggml_tensor)
)
def ggml_opt_ncorrect(
    opt_ctx: Union[ggml_opt_context_t, int],
) -> Optional[ggml_tensor_p]:
    ...


# GGML_API struct ggml_tensor * ggml_opt_grad_acc(ggml_opt_context_t opt_ctx, struct ggml_tensor * node);
@ggml_function(
    "ggml_opt_grad_acc",
    [ggml_opt_context_t_ctypes, ctypes.POINTER(ggml_tensor)],
    ctypes.POINTER(ggml_tensor),
)
def ggml_opt_grad_acc(
    opt_ctx: Union[ggml_opt_context_t, int],
    node: ggml_tensor_p,
    /,
) -> Optional[ggml_tensor_p]:
    ...


# GGML_API enum ggml_opt_optimizer_type ggml_opt_context_optimizer_type(ggml_opt_context_t);
@ggml_function(
    "ggml_opt_context_optimizer_type", [ggml_opt_context_t_ctypes], ctypes.c_int
)
def ggml_opt_context_optimizer_type(opt_ctx: Union[ggml_opt_context_t, int], /) -> int:
    ...


# GGML_API const char * ggml_opt_optimizer_name(enum ggml_opt_optimizer_type);
@ggml_function("ggml_opt_optimizer_name", [ctypes.c_int], ctypes.c_char_p)
def ggml_opt_optimizer_name(optimizer: Union[ctypes.c_int, int], /) -> bytes:
    ...


# GGML_API ggml_opt_result_t ggml_opt_result_init(void);
@ggml_function("ggml_opt_result_init", [], ggml_opt_result_t_ctypes)
def ggml_opt_result_init() -> Optional[ggml_opt_result_t]:
    ...


# GGML_API void ggml_opt_result_free(ggml_opt_result_t result);
@ggml_function("ggml_opt_result_free", [ggml_opt_result_t_ctypes], None)
def ggml_opt_result_free(result: Union[ggml_opt_result_t, int], /):
    ...


# GGML_API void ggml_opt_result_reset(ggml_opt_result_t result);
@ggml_function("ggml_opt_result_reset", [ggml_opt_result_t_ctypes], None)
def ggml_opt_result_reset(result: Union[ggml_opt_result_t, int], /):
    ...


# GGML_API void ggml_opt_result_ndata(ggml_opt_result_t result, int64_t * ndata);
@ggml_function(
    "ggml_opt_result_ndata",
    [ggml_opt_result_t_ctypes, ctypes.POINTER(ctypes.c_int64)],
    None,
)
def ggml_opt_result_ndata(
    result: Union[ggml_opt_result_t, int],
    ndata: CtypesPointer[ctypes.c_int64],
    /,
):
    ...


# GGML_API void ggml_opt_result_loss(ggml_opt_result_t result, double * loss, double * unc);
@ggml_function(
    "ggml_opt_result_loss",
    [
        ggml_opt_result_t_ctypes,
        ctypes.POINTER(ctypes.c_double),
        ctypes.POINTER(ctypes.c_double),
    ],
    None,
)
def ggml_opt_result_loss(
    result: Union[ggml_opt_result_t, int],
    loss: CtypesPointer[ctypes.c_double],
    unc: Optional[CtypesPointer[ctypes.c_double]],
    /,
):
    ...


# GGML_API void ggml_opt_result_pred(ggml_opt_result_t result, int32_t * pred);
@ggml_function(
    "ggml_opt_result_pred",
    [ggml_opt_result_t_ctypes, ctypes.POINTER(ctypes.c_int32)],
    None,
)
def ggml_opt_result_pred(
    result: Union[ggml_opt_result_t, int],
    pred: CtypesPointer[ctypes.c_int32],
    /,
):
    ...


# GGML_API void ggml_opt_result_accuracy(ggml_opt_result_t result, double * accuracy, double * unc);
@ggml_function(
    "ggml_opt_result_accuracy",
    [
        ggml_opt_result_t_ctypes,
        ctypes.POINTER(ctypes.c_double),
        ctypes.POINTER(ctypes.c_double),
    ],
    None,
)
def ggml_opt_result_accuracy(
    result: Union[ggml_opt_result_t, int],
    accuracy: CtypesPointer[ctypes.c_double],
    unc: Optional[CtypesPointer[ctypes.c_double]],
    /,
):
    ...


# GGML_API void ggml_opt_prepare_alloc(ggml_opt_context_t opt_ctx, struct ggml_context * ctx_compute, struct ggml_cgraph * gf, struct ggml_tensor * inputs, struct ggml_tensor * outputs);
@ggml_function(
    "ggml_opt_prepare_alloc",
    [
        ggml_opt_context_t_ctypes,
        ggml_context_p_ctypes,
        ctypes.POINTER(ggml_cgraph),
        ctypes.POINTER(ggml_tensor),
        ctypes.POINTER(ggml_tensor),
    ],
    None,
)
def ggml_opt_prepare_alloc(
    opt_ctx: Union[ggml_opt_context_t, int],
    ctx_compute: ggml_context_p,
    gf: ggml_cgraph_p,
    inputs: ggml_tensor_p,
    outputs: ggml_tensor_p,
    /,
):
    ...


# GGML_API void ggml_opt_alloc(ggml_opt_context_t opt_ctx, bool backward);
@ggml_function("ggml_opt_alloc", [ggml_opt_context_t_ctypes, ctypes.c_bool], None)
def ggml_opt_alloc(
    opt_ctx: Union[ggml_opt_context_t, int],
    backward: Union[ctypes.c_bool, bool],
    /,
):
    ...


# GGML_API void ggml_opt_eval(ggml_opt_context_t opt_ctx, ggml_opt_result_t result);
@ggml_function(
    "ggml_opt_eval",
    [ggml_opt_context_t_ctypes, ggml_opt_result_t_ctypes],
    None,
)
def ggml_opt_eval(
    opt_ctx: Union[ggml_opt_context_t, int],
    result: Union[ggml_opt_result_t, int, None],
    /,
):
    ...


# GGML_API void ggml_opt_epoch(ggml_opt_context_t opt_ctx, ggml_opt_dataset_t dataset, ggml_opt_result_t result_train, ggml_opt_result_t result_eval, int64_t idata_split, ggml_opt_epoch_callback callback_train, ggml_opt_epoch_callback callback_eval);
@ggml_function(
    "ggml_opt_epoch",
    [
        ggml_opt_context_t_ctypes,
        ggml_opt_dataset_t_ctypes,
        ggml_opt_result_t_ctypes,
        ggml_opt_result_t_ctypes,
        ctypes.c_int64,
        ggml_opt_epoch_callback,
        ggml_opt_epoch_callback,
    ],
    None,
)
def ggml_opt_epoch(
    opt_ctx: Union[ggml_opt_context_t, int],
    dataset: Union[ggml_opt_dataset_t, int],
    result_train: Union[ggml_opt_result_t, int, None],
    result_eval: Union[ggml_opt_result_t, int, None],
    idata_split: Union[ctypes.c_int64, int],
    callback_train: Optional[ggml_opt_epoch_callback],
    callback_eval: Optional[ggml_opt_epoch_callback],
    /,
):
    ...


# GGML_API void ggml_opt_epoch_callback_progress_bar(bool train, ggml_opt_context_t opt_ctx, ggml_opt_dataset_t dataset, ggml_opt_result_t result, int64_t ibatch, int64_t ibatch_max, int64_t t_start_us);
@ggml_function(
    "ggml_opt_epoch_callback_progress_bar",
    [
        ctypes.c_bool,
        ggml_opt_context_t_ctypes,
        ggml_opt_dataset_t_ctypes,
        ggml_opt_result_t_ctypes,
        ctypes.c_int64,
        ctypes.c_int64,
        ctypes.c_int64,
    ],
    None,
)
def ggml_opt_epoch_callback_progress_bar(
    train: Union[ctypes.c_bool, bool],
    opt_ctx: Union[ggml_opt_context_t, int],
    dataset: Union[ggml_opt_dataset_t, int],
    result: Union[ggml_opt_result_t, int],
    ibatch: Union[ctypes.c_int64, int],
    ibatch_max: Union[ctypes.c_int64, int],
    t_start_us: Union[ctypes.c_int64, int],
    /,
):
    ...


# GGML_API void ggml_opt_fit(ggml_backend_sched_t backend_sched, struct ggml_context * ctx_compute, struct ggml_tensor * inputs, struct ggml_tensor * outputs, ggml_opt_dataset_t dataset, enum ggml_opt_loss_type loss_type, enum ggml_opt_optimizer_type optimizer, ggml_opt_get_optimizer_params get_opt_pars, int64_t nepoch, int64_t nbatch_logical, float val_split, bool silent);
@ggml_function(
    "ggml_opt_fit",
    [
        ctypes.c_void_p,
        ggml_context_p_ctypes,
        ctypes.POINTER(ggml_tensor),
        ctypes.POINTER(ggml_tensor),
        ggml_opt_dataset_t_ctypes,
        ctypes.c_int,
        ctypes.c_int,
        ggml_opt_get_optimizer_params,
        ctypes.c_int64,
        ctypes.c_int64,
        ctypes.c_float,
        ctypes.c_bool,
    ],
    None,
)
def ggml_opt_fit(
    backend_sched: Union[ctypes.c_void_p, int, None],
    ctx_compute: ggml_context_p,
    inputs: ggml_tensor_p,
    outputs: ggml_tensor_p,
    dataset: Union[ggml_opt_dataset_t, int],
    loss_type: Union[ctypes.c_int, int],
    optimizer: Union[ctypes.c_int, int],
    get_opt_pars: Union[ctypes.c_void_p, int, None],
    nepoch: Union[ctypes.c_int64, int],
    nbatch_logical: Union[ctypes.c_int64, int],
    val_split: Union[ctypes.c_float, float],
    silent: Union[ctypes.c_bool, bool],
    /,
):
    ...


# //
# // tensor flags
# //
# GGML_API void ggml_set_input(struct ggml_tensor * tensor);
@ggml_function("ggml_set_input", [ctypes.POINTER(ggml_tensor)], None)
def ggml_set_input(tensor: ggml_tensor_p):
    ...


# GGML_API void ggml_set_output(struct ggml_tensor * tensor);
@ggml_function("ggml_set_output", [ctypes.POINTER(ggml_tensor)], None)
def ggml_set_output(tensor: ggml_tensor_p):
    ...


# GGML_API void ggml_set_loss(struct ggml_tensor * tensor);
@ggml_function("ggml_set_loss", [ctypes.POINTER(ggml_tensor)], None)
def ggml_set_loss(tensor: ggml_tensor_p):
    ...


# //
# // quantization
# //


# // - ggml_quantize_init can be called multiple times with the same type
# //   it will only initialize the quantization tables for the first call or after ggml_quantize_free
# //   automatically called by ggml_quantize_chunk for convenience
# //
# // - ggml_quantize_free will free any memory allocated by ggml_quantize_init
# //   call this at the end of the program to avoid memory leaks
# //
# // note: these are thread-safe
# //
# GGML_API void ggml_quantize_init(enum ggml_type type);
@ggml_function("ggml_quantize_init", [ctypes.c_int], None)
def ggml_quantize_init(type: Union[ctypes.c_int, int]):
    ...


# GGML_API void ggml_quantize_free(void);
@ggml_function("ggml_quantize_free", [], None)
def ggml_quantize_free():
    ...


# // some quantization type cannot be used without an importance matrix
# GGML_API bool ggml_quantize_requires_imatrix(enum ggml_type type);
@ggml_function(
    "ggml_quantize_requires_imatrix",
    [
        ctypes.c_int,
    ],
    ctypes.c_bool,
)
def ggml_quantize_requires_imatrix(
    type: Union[ctypes.c_int, int],
) -> bool:
    ...

# // calls ggml_quantize_init internally (i.e. can allocate memory)
# GGML_API size_t ggml_quantize_chunk(
#         enum ggml_type   type,
#            const float * src,
#                   void * dst,
#                int64_t   start,
#                int64_t   nrows,
#                int64_t   n_per_row,
#            const float * imatrix);
@ggml_function(
    "ggml_quantize_chunk",
    [
        ctypes.c_int,
        ctypes.POINTER(ctypes.c_float),
        ctypes.c_void_p,
        ctypes.c_int64,
        ctypes.c_int64,
        ctypes.c_int64,
        ctypes.POINTER(ctypes.c_float),
    ],
    ctypes.c_size_t,
)
def ggml_quantize_chunk(
    type: Union[ctypes.c_int, int],
    src: CtypesArray[ctypes.c_float],
    dst: Union[ctypes.c_void_p, int, None],
    start: Union[ctypes.c_int64, int],
    nrows: Union[ctypes.c_int64, int],
    n_per_row: Union[ctypes.c_int64, int],
    imatrix: Optional[CtypesArray[ctypes.c_float]],
) -> int:
    ...


# //
# // gguf
# //

# enum gguf_type {
#     GGUF_TYPE_UINT8   = 0,
#     GGUF_TYPE_INT8    = 1,
#     GGUF_TYPE_UINT16  = 2,
#     GGUF_TYPE_INT16   = 3,
#     GGUF_TYPE_UINT32  = 4,
#     GGUF_TYPE_INT32   = 5,
#     GGUF_TYPE_FLOAT32 = 6,
#     GGUF_TYPE_BOOL    = 7,
#     GGUF_TYPE_STRING  = 8,
#     GGUF_TYPE_ARRAY   = 9,
#     GGUF_TYPE_UINT64  = 10,
#     GGUF_TYPE_INT64   = 11,
#     GGUF_TYPE_FLOAT64 = 12,
#     GGUF_TYPE_COUNT,       // marks the end of the enum
# };
GGUF_TYPE_UINT8 = 0
GGUF_TYPE_INT8 = 1
GGUF_TYPE_UINT16 = 2
GGUF_TYPE_INT16 = 3
GGUF_TYPE_UINT32 = 4
GGUF_TYPE_INT32 = 5
GGUF_TYPE_FLOAT32 = 6
GGUF_TYPE_BOOL = 7
GGUF_TYPE_STRING = 8
GGUF_TYPE_ARRAY = 9
GGUF_TYPE_UINT64 = 10
GGUF_TYPE_INT64 = 11
GGUF_TYPE_FLOAT64 = 12
GGUF_TYPE_COUNT = 13

# struct gguf_context;
gguf_context_p = NewType("gguf_context_p", int)
gguf_context_p_ctypes = ctypes.c_void_p

# struct gguf_init_params {
#     bool no_alloc;


#     // if not NULL, create a ggml_context and allocate the tensor data in it
#     struct ggml_context ** ctx;
# };
class gguf_init_params(ctypes.Structure):
    """Initialization parameters for gguf.

    Attributes:
        no_alloc: No allocation.
        ctx: The context."""

    if TYPE_CHECKING:
        no_alloc: bool
        ctx: CtypesPointer[ggml_context_p]

    _fields_ = [
        ("no_alloc", ctypes.c_bool),
        ("ctx", ctypes.POINTER(ggml_context_p_ctypes)),
    ]


gguf_reader_callback_t = ctypes.CFUNCTYPE(
    ctypes.c_size_t,
    ctypes.c_void_p,
    ctypes.c_void_p,
    ctypes.c_uint64,
    ctypes.c_size_t,
)


# GGML_API struct gguf_context * gguf_init_empty(void);
@ggml_function("gguf_init_empty", [], gguf_context_p_ctypes)
def gguf_init_empty() -> Optional[gguf_context_p]:
    ...


# GGML_API struct gguf_context * gguf_init_from_file_ptr(FILE * file, struct gguf_init_params params);
@ggml_function(
    "gguf_init_from_file_ptr",
    [
        ctypes.c_void_p,
        gguf_init_params,
    ],
    gguf_context_p_ctypes,
)
def gguf_init_from_file_ptr(
    file: Union[ctypes.c_void_p, int, None],
    params: gguf_init_params,
    /,
) -> Optional[gguf_context_p]:
    ...


# GGML_API struct gguf_context * gguf_init_from_file(const char * fname, struct gguf_init_params params);
@ggml_function(
    "gguf_init_from_file",
    [
        ctypes.c_char_p,
        gguf_init_params,
    ],
    gguf_context_p_ctypes,
)
def gguf_init_from_file(
    fname: bytes,
    params: gguf_init_params,
) -> Optional[gguf_context_p]:
    ...


# GGML_API struct gguf_context * gguf_init_from_buffer(const void * data, size_t size, struct gguf_init_params params);
@ggml_function(
    "gguf_init_from_buffer",
    [
        ctypes.c_void_p,
        ctypes.c_size_t,
        gguf_init_params,
    ],
    gguf_context_p_ctypes,
)
def gguf_init_from_buffer(
    data: Union[ctypes.c_void_p, int, None],
    size: Union[ctypes.c_size_t, int],
    params: gguf_init_params,
    /,
) -> Optional[gguf_context_p]:
    ...


# typedef size_t (*gguf_reader_callback_t)(void * userdata, void * output, uint64_t offset, size_t len);
# GGML_API struct gguf_context * gguf_init_from_callback(
#     gguf_reader_callback_t callback,
#     void * userdata,
#     size_t max_chunk_read,
#     uint64_t max_expected_size,
#     struct gguf_init_params params);
@ggml_function(
    "gguf_init_from_callback",
    [
        gguf_reader_callback_t,
        ctypes.c_void_p,
        ctypes.c_size_t,
        ctypes.c_uint64,
        gguf_init_params,
    ],
    gguf_context_p_ctypes,
)
def gguf_init_from_callback(
    callback: gguf_reader_callback_t,
    userdata: Union[ctypes.c_void_p, int, None],
    max_chunk_read: Union[ctypes.c_size_t, int],
    max_expected_size: Union[ctypes.c_uint64, int],
    params: gguf_init_params,
    /,
) -> Optional[gguf_context_p]:
    ...


# GGML_API void gguf_free(struct gguf_context * ctx);
@ggml_function(
    "gguf_free",
    [
        gguf_context_p_ctypes,
    ],
    None,
)
def gguf_free(
    ctx: gguf_context_p,
):
    ...


# GGML_API const char * gguf_type_name(enum gguf_type type);
@ggml_function(
    "gguf_type_name",
    [
        ctypes.c_int,
    ],
    ctypes.c_char_p,
)
def gguf_type_name(
    type: Union[ctypes.c_int, int],
) -> bytes:
    ...


# GGML_API uint32_t gguf_get_version    (const struct gguf_context * ctx);
@ggml_function(
    "gguf_get_version",
    [
        gguf_context_p_ctypes,
    ],
    ctypes.c_uint32,
)
def gguf_get_version(
    ctx: gguf_context_p,
) -> int:
    ...


# GGML_API size_t gguf_get_alignment  (const struct gguf_context * ctx);
@ggml_function(
    "gguf_get_alignment",
    [
        gguf_context_p_ctypes,
    ],
    ctypes.c_size_t,
)
def gguf_get_alignment(
    ctx: gguf_context_p,
) -> int:
    ...


# GGML_API size_t gguf_get_data_offset(const struct gguf_context * ctx);
@ggml_function(
    "gguf_get_data_offset",
    [
        gguf_context_p_ctypes,
    ],
    ctypes.c_size_t,
)
def gguf_get_data_offset(
    ctx: gguf_context_p,
) -> int:
    ...


# GGML_API void * gguf_get_data       (const struct gguf_context * ctx);
@ggml_function(
    "gguf_get_data",
    [
        gguf_context_p_ctypes,
    ],
    ctypes.c_void_p,
    enabled=hasattr(lib, "gguf_get_data"),
)
def gguf_get_data(
    ctx: gguf_context_p,
) -> Optional[int]:
    ...


# GGML_API int64_t      gguf_get_n_kv(const struct gguf_context * ctx);
@ggml_function(
    "gguf_get_n_kv",
    [
        gguf_context_p_ctypes,
    ],
    ctypes.c_int64,
)
def gguf_get_n_kv(
    ctx: gguf_context_p,
) -> int:
    ...


# GGML_API int64_t      gguf_find_key(const struct gguf_context * ctx, const char * key);
@ggml_function(
    "gguf_find_key",
    [
        gguf_context_p_ctypes,
        ctypes.c_char_p,
    ],
    ctypes.c_int64,
)
def gguf_find_key(
    ctx: gguf_context_p,
    key: bytes,
) -> int:
    ...


# GGML_API const char * gguf_get_key (const struct gguf_context * ctx, int64_t key_id);
@ggml_function(
    "gguf_get_key",
    [
        gguf_context_p_ctypes,
        ctypes.c_int64,
    ],
    ctypes.c_char_p,
)
def gguf_get_key(
    ctx: gguf_context_p,
    key_id: Union[ctypes.c_int64, int],
) -> bytes:
    ...


# GGML_API enum gguf_type gguf_get_kv_type (const struct gguf_context * ctx, int64_t key_id);
@ggml_function(
    "gguf_get_kv_type",
    [
        gguf_context_p_ctypes,
        ctypes.c_int64,
    ],
    ctypes.c_int,
)
def gguf_get_kv_type(
    ctx: gguf_context_p,
    key_id: Union[ctypes.c_int64, int],
) -> int:
    ...


# GGML_API enum gguf_type gguf_get_arr_type(const struct gguf_context * ctx, int64_t key_id);
@ggml_function(
    "gguf_get_arr_type",
    [
        gguf_context_p_ctypes,
        ctypes.c_int64,
    ],
    ctypes.c_int,
)
def gguf_get_arr_type(
    ctx: gguf_context_p,
    key_id: Union[ctypes.c_int64, int],
) -> int:
    ...


# // will abort if the wrong type is used for the key
# GGML_API uint8_t      gguf_get_val_u8  (const struct gguf_context * ctx, int64_t key_id);
@ggml_function(
    "gguf_get_val_u8",
    [
        gguf_context_p_ctypes,
        ctypes.c_int64,
    ],
    ctypes.c_uint8,
)
def gguf_get_val_u8(
    ctx: gguf_context_p,
    key_id: Union[ctypes.c_int64, int],
) -> int:
    ...


# GGML_API int8_t       gguf_get_val_i8  (const struct gguf_context * ctx, int64_t key_id);
@ggml_function(
    "gguf_get_val_i8",
    [
        gguf_context_p_ctypes,
        ctypes.c_int64,
    ],
    ctypes.c_int8,
)
def gguf_get_val_i8(
    ctx: gguf_context_p,
    key_id: Union[ctypes.c_int64, int],
) -> int:
    ...


# GGML_API uint16_t     gguf_get_val_u16 (const struct gguf_context * ctx, int64_t key_id);
@ggml_function(
    "gguf_get_val_u16",
    [
        gguf_context_p_ctypes,
        ctypes.c_int64,
    ],
    ctypes.c_uint16,
)
def gguf_get_val_u16(
    ctx: gguf_context_p,
    key_id: Union[ctypes.c_int64, int],
) -> int:
    ...


# GGML_API int16_t      gguf_get_val_i16 (const struct gguf_context * ctx, int64_t key_id);
@ggml_function(
    "gguf_get_val_i16",
    [
        gguf_context_p_ctypes,
        ctypes.c_int64,
    ],
    ctypes.c_int16,
)
def gguf_get_val_i16(
    ctx: gguf_context_p,
    key_id: Union[ctypes.c_int64, int],
) -> int:
    ...


# GGML_API uint32_t     gguf_get_val_u32 (const struct gguf_context * ctx, int64_t key_id);
@ggml_function(
    "gguf_get_val_u32",
    [
        gguf_context_p_ctypes,
        ctypes.c_int64,
    ],
    ctypes.c_uint32,
)
def gguf_get_val_u32(
    ctx: gguf_context_p,
    key_id: Union[ctypes.c_int64, int],
) -> int:
    ...


# GGML_API int32_t      gguf_get_val_i32 (const struct gguf_context * ctx, int64_t key_id);
@ggml_function(
    "gguf_get_val_i32",
    [
        gguf_context_p_ctypes,
        ctypes.c_int64,
    ],
    ctypes.c_int32,
)
def gguf_get_val_i32(
    ctx: gguf_context_p,
    key_id: Union[ctypes.c_int64, int],
) -> int:
    ...


# GGML_API float        gguf_get_val_f32 (const struct gguf_context * ctx, int64_t key_id);
@ggml_function(
    "gguf_get_val_f32",
    [
        gguf_context_p_ctypes,
        ctypes.c_int64,
    ],
    ctypes.c_float,
)
def gguf_get_val_f32(
    ctx: gguf_context_p,
    key_id: Union[ctypes.c_int64, int],
) -> float:
    ...


# GGML_API uint64_t     gguf_get_val_u64 (const struct gguf_context * ctx, int64_t key_id);
@ggml_function(
    "gguf_get_val_u64",
    [
        gguf_context_p_ctypes,
        ctypes.c_int64,
    ],
    ctypes.c_uint64,
)
def gguf_get_val_u64(
    ctx: gguf_context_p,
    key_id: Union[ctypes.c_int64, int],
) -> int:
    ...


# GGML_API int64_t      gguf_get_val_i64 (const struct gguf_context * ctx, int64_t key_id);
@ggml_function(
    "gguf_get_val_i64",
    [
        gguf_context_p_ctypes,
        ctypes.c_int64,
    ],
    ctypes.c_int64,
)
def gguf_get_val_i64(
    ctx: gguf_context_p,
    key_id: Union[ctypes.c_int64, int],
) -> int:
    ...


# GGML_API double       gguf_get_val_f64 (const struct gguf_context * ctx, int64_t key_id);
@ggml_function(
    "gguf_get_val_f64",
    [
        gguf_context_p_ctypes,
        ctypes.c_int64,
    ],
    ctypes.c_double,
)
def gguf_get_val_f64(
    ctx: gguf_context_p,
    key_id: Union[ctypes.c_int64, int],
) -> float:
    ...


# GGML_API bool         gguf_get_val_bool(const struct gguf_context * ctx, int64_t key_id);
@ggml_function(
    "gguf_get_val_bool",
    [
        gguf_context_p_ctypes,
        ctypes.c_int64,
    ],
    ctypes.c_bool,
)
def gguf_get_val_bool(
    ctx: gguf_context_p,
    key_id: Union[ctypes.c_int64, int],
) -> bool:
    ...


# GGML_API const char * gguf_get_val_str (const struct gguf_context * ctx, int64_t key_id);
@ggml_function(
    "gguf_get_val_str",
    [
        gguf_context_p_ctypes,
        ctypes.c_int64,
    ],
    ctypes.c_char_p,
)
def gguf_get_val_str(
    ctx: gguf_context_p,
    key_id: Union[ctypes.c_int64, int],
) -> bytes:
    ...


# GGML_API const void * gguf_get_val_data(const struct gguf_context * ctx, int64_t key_id);
@ggml_function(
    "gguf_get_val_data",
    [
        gguf_context_p_ctypes,
        ctypes.c_int64,
    ],
    ctypes.c_void_p,
)
def gguf_get_val_data(
    ctx: gguf_context_p,
    key_id: Union[ctypes.c_int64, int],
) -> Optional[int]:
    ...


# GGML_API size_t       gguf_get_arr_n   (const struct gguf_context * ctx, int64_t key_id);
@ggml_function(
    "gguf_get_arr_n",
    [
        gguf_context_p_ctypes,
        ctypes.c_int64,
    ],
    ctypes.c_size_t,
)
def gguf_get_arr_n(
    ctx: gguf_context_p,
    key_id: Union[ctypes.c_int64, int],
) -> int:
    ...


# GGML_API const void * gguf_get_arr_data(const struct gguf_context * ctx, int64_t key_id);
@ggml_function(
    "gguf_get_arr_data",
    [
        gguf_context_p_ctypes,
        ctypes.c_int64,
    ],
    ctypes.c_void_p,
)
def gguf_get_arr_data(
    ctx: gguf_context_p,
    key_id: Union[ctypes.c_int64, int],
) -> Optional[int]:
    ...


# GGML_API const char * gguf_get_arr_str (const struct gguf_context * ctx, int64_t key_id, size_t i);
@ggml_function(
    "gguf_get_arr_str",
    [
        gguf_context_p_ctypes,
        ctypes.c_int64,
        ctypes.c_size_t,
    ],
    ctypes.c_char_p,
)
def gguf_get_arr_str(
    ctx: gguf_context_p,
    key_id: Union[ctypes.c_int64, int],
    i: Union[ctypes.c_size_t, int],
) -> bytes:
    ...


# GGML_API int64_t        gguf_get_n_tensors    (const struct gguf_context * ctx);
@ggml_function(
    "gguf_get_n_tensors",
    [
        gguf_context_p_ctypes,
    ],
    ctypes.c_int64,
)
def gguf_get_n_tensors(
    ctx: gguf_context_p,
) -> int:
    ...


# GGML_API int64_t        gguf_find_tensor      (const struct gguf_context * ctx, const char * name);
@ggml_function(
    "gguf_find_tensor",
    [
        gguf_context_p_ctypes,
        ctypes.c_char_p,
    ],
    ctypes.c_int64,
)
def gguf_find_tensor(
    ctx: gguf_context_p,
    name: bytes,
) -> int:
    ...


# GGML_API size_t         gguf_get_tensor_offset(const struct gguf_context * ctx, int64_t tensor_id);
@ggml_function(
    "gguf_get_tensor_offset",
    [
        gguf_context_p_ctypes,
        ctypes.c_int64,
    ],
    ctypes.c_size_t,
)
def gguf_get_tensor_offset(
    ctx: gguf_context_p,
    tensor_id: Union[ctypes.c_int64, int],
) -> int:
    ...


# GGML_API const char *   gguf_get_tensor_name  (const struct gguf_context * ctx, int64_t tensor_id);
@ggml_function(
    "gguf_get_tensor_name",
    [
        gguf_context_p_ctypes,
        ctypes.c_int64,
    ],
    ctypes.c_char_p,
)
def gguf_get_tensor_name(
    ctx: gguf_context_p,
    tensor_id: Union[ctypes.c_int64, int],
) -> bytes:
    ...


# GGML_API enum ggml_type gguf_get_tensor_type  (const struct gguf_context * ctx, int64_t tensor_id);
@ggml_function(
    "gguf_get_tensor_type",
    [
        gguf_context_p_ctypes,
        ctypes.c_int64,
    ],
    ctypes.c_int,
)
def gguf_get_tensor_type(
    ctx: gguf_context_p,
    tensor_id: Union[ctypes.c_int64, int],
) -> int:
    ...


# GGML_API size_t         gguf_get_tensor_size  (const struct gguf_context * ctx, int64_t tensor_id);
@ggml_function(
    "gguf_get_tensor_size",
    [
        gguf_context_p_ctypes,
        ctypes.c_int64,
    ],
    ctypes.c_size_t,
)
def gguf_get_tensor_size(
    ctx: gguf_context_p,
    tensor_id: Union[ctypes.c_int64, int],
) -> int:
    ...


# // removes key if it exists, returns id that the key had prior to removal (-1 if it didn't exist)
# GGML_API int64_t gguf_remove_key(struct gguf_context * ctx, const char * key);
@ggml_function(
    "gguf_remove_key",
    [
        gguf_context_p_ctypes,
        ctypes.c_char_p,
    ],
    ctypes.c_int64,
)
def gguf_remove_key(
    ctx: gguf_context_p,
    key: bytes,
) -> int:
    ...


# // overrides existing values or adds a new one
# GGML_API void gguf_set_val_u8  (struct gguf_context * ctx, const char * key, uint8_t  val);
@ggml_function(
    "gguf_set_val_u8",
    [
        gguf_context_p_ctypes,
        ctypes.c_char_p,
        ctypes.c_uint8,
    ],
    None,
)
def gguf_set_val_u8(
    ctx: gguf_context_p,
    key: bytes,
    val: Union[ctypes.c_uint8, int],
):
    ...


# GGML_API void gguf_set_val_i8  (struct gguf_context * ctx, const char * key, int8_t   val);
@ggml_function(
    "gguf_set_val_i8",
    [
        gguf_context_p_ctypes,
        ctypes.c_char_p,
        ctypes.c_int8,
    ],
    None,
)
def gguf_set_val_i8(
    ctx: gguf_context_p,
    key: bytes,
    val: Union[ctypes.c_int8, int],
):
    ...


# GGML_API void gguf_set_val_u16 (struct gguf_context * ctx, const char * key, uint16_t val);
@ggml_function(
    "gguf_set_val_u16",
    [
        gguf_context_p_ctypes,
        ctypes.c_char_p,
        ctypes.c_uint16,
    ],
    None,
)
def gguf_set_val_u16(
    ctx: gguf_context_p,
    key: bytes,
    val: Union[ctypes.c_uint16, int],
):
    ...


# GGML_API void gguf_set_val_i16 (struct gguf_context * ctx, const char * key, int16_t  val);
@ggml_function(
    "gguf_set_val_i16",
    [
        gguf_context_p_ctypes,
        ctypes.c_char_p,
        ctypes.c_int16,
    ],
    None,
)
def gguf_set_val_i16(
    ctx: gguf_context_p,
    key: bytes,
    val: Union[ctypes.c_int16, int],
):
    ...


# GGML_API void gguf_set_val_u32 (struct gguf_context * ctx, const char * key, uint32_t val);
@ggml_function(
    "gguf_set_val_u32",
    [
        gguf_context_p_ctypes,
        ctypes.c_char_p,
        ctypes.c_uint32,
    ],
    None,
)
def gguf_set_val_u32(
    ctx: gguf_context_p,
    key: bytes,
    val: Union[ctypes.c_uint32, int],
):
    ...


# GGML_API void gguf_set_val_i32 (struct gguf_context * ctx, const char * key, int32_t  val);
@ggml_function(
    "gguf_set_val_i32",
    [
        gguf_context_p_ctypes,
        ctypes.c_char_p,
        ctypes.c_int32,
    ],
    None,
)
def gguf_set_val_i32(
    ctx: gguf_context_p,
    key: bytes,
    val: Union[ctypes.c_int32, int],
):
    ...


# GGML_API void gguf_set_val_f32 (struct gguf_context * ctx, const char * key, float    val);
@ggml_function(
    "gguf_set_val_f32",
    [
        gguf_context_p_ctypes,
        ctypes.c_char_p,
        ctypes.c_float,
    ],
    None,
)
def gguf_set_val_f32(
    ctx: gguf_context_p,
    key: bytes,
    val: Union[ctypes.c_float, float],
):
    ...


# GGML_API void gguf_set_val_u64 (struct gguf_context * ctx, const char * key, uint64_t val);
@ggml_function(
    "gguf_set_val_u64",
    [
        gguf_context_p_ctypes,
        ctypes.c_char_p,
        ctypes.c_uint64,
    ],
    None,
)
def gguf_set_val_u64(
    ctx: gguf_context_p,
    key: bytes,
    val: Union[ctypes.c_uint64, int],
):
    ...


# GGML_API void gguf_set_val_i64 (struct gguf_context * ctx, const char * key, int64_t  val);
@ggml_function(
    "gguf_set_val_i64",
    [
        gguf_context_p_ctypes,
        ctypes.c_char_p,
        ctypes.c_int64,
    ],
    None,
)
def gguf_set_val_i64(
    ctx: gguf_context_p,
    key: bytes,
    val: Union[ctypes.c_int64, int],
):
    ...


# GGML_API void gguf_set_val_f64 (struct gguf_context * ctx, const char * key, double   val);
@ggml_function(
    "gguf_set_val_f64",
    [
        gguf_context_p_ctypes,
        ctypes.c_char_p,
        ctypes.c_double,
    ],
    None,
)
def gguf_set_val_f64(
    ctx: gguf_context_p,
    key: bytes,
    val: Union[ctypes.c_double, float],
):
    ...


# GGML_API void gguf_set_val_bool(struct gguf_context * ctx, const char * key, bool     val);
@ggml_function(
    "gguf_set_val_bool",
    [
        gguf_context_p_ctypes,
        ctypes.c_char_p,
        ctypes.c_bool,
    ],
    None,
)
def gguf_set_val_bool(
    ctx: gguf_context_p,
    key: bytes,
    val: Union[ctypes.c_bool, bool],
):
    ...


# GGML_API void gguf_set_val_str (struct gguf_context * ctx, const char * key, const char * val);
@ggml_function(
    "gguf_set_val_str",
    [
        gguf_context_p_ctypes,
        ctypes.c_char_p,
        ctypes.c_char_p,
    ],
    None,
)
def gguf_set_val_str(
    ctx: gguf_context_p,
    key: bytes,
    val: bytes,
):
    ...


# GGML_API void gguf_set_arr_data(struct gguf_context * ctx, const char * key, enum gguf_type type, const void * data, size_t n);
@ggml_function(
    "gguf_set_arr_data",
    [
        gguf_context_p_ctypes,
        ctypes.c_char_p,
        ctypes.c_int,
        ctypes.c_void_p,
        ctypes.c_size_t,
    ],
    None,
)
def gguf_set_arr_data(
    ctx: gguf_context_p,
    key: bytes,
    type: Union[ctypes.c_int, int],
    data: Union[ctypes.c_void_p, int, None],
    n: Union[ctypes.c_size_t, int],
):
    ...


# GGML_API void gguf_set_arr_str (struct gguf_context * ctx, const char * key, const char ** data, size_t n);
@ggml_function(
    "gguf_set_arr_str",
    [
        gguf_context_p_ctypes,
        ctypes.c_char_p,
        ctypes.POINTER(ctypes.c_char_p),
        ctypes.c_size_t,
    ],
    None,
)
def gguf_set_arr_str(
    ctx: gguf_context_p,
    key: bytes,
    data: CtypesPointer[ctypes.c_char_p],
    n: Union[ctypes.c_size_t, int],
):
    ...


# // set or add KV pairs from another context
# GGML_API void gguf_set_kv(struct gguf_context * ctx, const struct gguf_context * src);
@ggml_function(
    "gguf_set_kv",
    [
        gguf_context_p_ctypes,
        gguf_context_p_ctypes,
    ],
    None,
)
def gguf_set_kv(
    ctx: gguf_context_p,
    src: gguf_context_p,
):
    ...


# // manage tensor info
# GGML_API void gguf_add_tensor(struct gguf_context * ctx, const struct ggml_tensor * tensor);
@ggml_function(
    "gguf_add_tensor",
    [
        gguf_context_p_ctypes,
        ctypes.POINTER(ggml_tensor),
    ],
    None,
)
def gguf_add_tensor(
    ctx: gguf_context_p,
    tensor: ggml_tensor_p,
):
    ...


# GGML_API void gguf_set_tensor_type(struct gguf_context * ctx, const char * name, enum ggml_type type);
@ggml_function(
    "gguf_set_tensor_type",
    [
        gguf_context_p_ctypes,
        ctypes.c_char_p,
        ctypes.c_int,
    ],
    None,
)
def gguf_set_tensor_type(
    ctx: gguf_context_p, name: bytes, type: Union[ctypes.c_int, int], /
):
    ...


# // assumes that at least gguf_get_tensor_size bytes can be read from data
# GGML_API void gguf_set_tensor_data(struct gguf_context * ctx, const char * name, const void * data);
@ggml_function(
    "gguf_set_tensor_data",
    [
        gguf_context_p_ctypes,
        ctypes.c_char_p,
        ctypes.c_void_p,
    ],
    None,
)
def gguf_set_tensor_data(
    ctx: gguf_context_p,
    name: bytes,
    data: Union[ctypes.c_void_p, int, None],
    /,
):
    ...


# // writing gguf files can be done in 2 ways:
# //
# // - write the entire gguf_context to a binary file in a single pass:
# //
# //   gguf_write_to_file(ctx, fname);
# //
# // - first prepare a file with a placeholder for the meta data, write the tensor data, then write the meta data:
# //
# //   FILE * f = fopen(fname, "wb");
# //   fseek(f, gguf_get_meta_size(ctx), SEEK_SET);
# //   fwrite(f, ...);
# //   void * data = gguf_meta_get_meta_data(ctx);
# //   fseek(f, 0, SEEK_SET);
# //   fwrite(f, data, gguf_get_meta_size(ctx));
# //   free(data);
# //   fclose(f);
# //


# // write the entire context to a binary file
# GGML_API bool gguf_write_to_file_ptr(const struct gguf_context * ctx, FILE * file, bool only_meta);
@ggml_function(
    "gguf_write_to_file_ptr",
    [
        gguf_context_p_ctypes,
        ctypes.c_void_p,
        ctypes.c_bool,
    ],
    ctypes.c_bool,
)
def gguf_write_to_file_ptr(
    ctx: gguf_context_p,
    file: Union[ctypes.c_void_p, int, None],
    only_meta: Union[ctypes.c_bool, bool],
    /,
) -> bool:
    ...


# GGML_API bool gguf_write_to_file(const struct gguf_context * ctx, const char * fname, bool only_meta);
@ggml_function(
    "gguf_write_to_file",
    [
        gguf_context_p_ctypes,
        ctypes.c_char_p,
        ctypes.c_bool,
    ],
    ctypes.c_bool,
)
def gguf_write_to_file(
    ctx: gguf_context_p, fname: bytes, only_meta: Union[ctypes.c_bool, bool], /
) -> bool:
    ...


# // get the size in bytes of the meta data (header, kv pairs, tensor info) including padding
# GGML_API size_t gguf_get_meta_size(const struct gguf_context * ctx);
@ggml_function(
    "gguf_get_meta_size",
    [
        gguf_context_p_ctypes,
    ],
    ctypes.c_size_t,
)
def gguf_get_meta_size(ctx: gguf_context_p, /) -> int:
    ...


# GGML_API void   gguf_get_meta_data(const struct gguf_context * ctx, void * data);
@ggml_function(
    "gguf_get_meta_data",
    [
        gguf_context_p_ctypes,
        ctypes.c_void_p,
    ],
    None,
)
def gguf_get_meta_data(ctx: gguf_context_p, data: Union[ctypes.c_void_p, int, None], /):
    ...


# //
# // system info
# //


# GGML_API int ggml_cpu_has_avx        (void);
@ggml_function("ggml_cpu_has_avx", [], ctypes.c_int)
def ggml_cpu_has_avx() -> int:
    ...


# GGML_API int ggml_cpu_has_avx_vnni   (void);
@ggml_function("ggml_cpu_has_avx_vnni", [], ctypes.c_int)
def ggml_cpu_has_avx_vnni() -> int:
    ...


# GGML_API int ggml_cpu_has_avx2       (void);
@ggml_function("ggml_cpu_has_avx2", [], ctypes.c_int)
def ggml_cpu_has_avx2() -> int:
    ...


# GGML_API int ggml_cpu_has_bmi2       (void);
@ggml_function("ggml_cpu_has_bmi2", [], ctypes.c_int)
def ggml_cpu_has_bmi2() -> int:
    ...


# GGML_API int ggml_cpu_has_avx512     (void);
@ggml_function("ggml_cpu_has_avx512", [], ctypes.c_int)
def ggml_cpu_has_avx512() -> int:
    ...


# GGML_API int ggml_cpu_has_avx512_vbmi(void);
@ggml_function("ggml_cpu_has_avx512_vbmi", [], ctypes.c_int)
def ggml_cpu_has_avx512_vbmi() -> int:
    ...


# GGML_API int ggml_cpu_has_avx512_vnni(void);
@ggml_function("ggml_cpu_has_avx512_vnni", [], ctypes.c_int)
def ggml_cpu_has_avx512_vnni() -> int:
    ...


# GGML_API int ggml_cpu_has_avx512_bf16(void);
@ggml_function("ggml_cpu_has_avx512_bf16", [], ctypes.c_int)
def ggml_cpu_has_avx512_bf16() -> int:
    ...


# GGML_API int ggml_cpu_has_amx_int8   (void);
@ggml_function("ggml_cpu_has_amx_int8", [], ctypes.c_int)
def ggml_cpu_has_amx_int8() -> int:
    ...


# GGML_API int ggml_cpu_has_fma        (void);
@ggml_function("ggml_cpu_has_fma", [], ctypes.c_int)
def ggml_cpu_has_fma() -> int:
    ...


# GGML_API int ggml_cpu_has_neon       (void);
@ggml_function("ggml_cpu_has_neon", [], ctypes.c_int)
def ggml_cpu_has_neon() -> int:
    ...


# GGML_API int ggml_cpu_has_sve        (void);
@ggml_function("ggml_cpu_has_sve", [], ctypes.c_int)
def ggml_cpu_has_sve() -> int:
    ...


# GGML_API int ggml_cpu_get_sve_cnt    (void);
@ggml_function("ggml_cpu_get_sve_cnt", [], ctypes.c_int)
def ggml_cpu_get_sve_cnt() -> int:
    ...


# GGML_API int ggml_cpu_has_sme        (void);
@ggml_function("ggml_cpu_has_sme", [], ctypes.c_int)
def ggml_cpu_has_sme() -> int:
    ...


# GGML_API int ggml_cpu_has_arm_fma    (void);
@ggml_function("ggml_cpu_has_arm_fma", [], ctypes.c_int)
def ggml_cpu_has_arm_fma() -> int:
    ...


# GGML_API int ggml_cpu_has_metal      (void);
@ggml_function("ggml_cpu_has_metal", [], ctypes.c_int, enabled=hasattr(lib, "ggml_cpu_has_metal"))
def ggml_cpu_has_metal() -> int:
    ...


# GGML_API int ggml_cpu_has_f16c       (void);
@ggml_function("ggml_cpu_has_f16c", [], ctypes.c_int)
def ggml_cpu_has_f16c() -> int:
    ...


# GGML_API int ggml_cpu_has_fp16_va    (void);
@ggml_function("ggml_cpu_has_fp16_va", [], ctypes.c_int)
def ggml_cpu_has_fp16_va() -> int:
    ...


# GGML_API int ggml_cpu_has_dotprod    (void);
@ggml_function("ggml_cpu_has_dotprod", [], ctypes.c_int)
def ggml_cpu_has_dotprod() -> int:
    ...


# GGML_API int ggml_cpu_has_wasm_simd  (void);
@ggml_function("ggml_cpu_has_wasm_simd", [], ctypes.c_int)
def ggml_cpu_has_wasm_simd() -> int:
    ...


# GGML_API int ggml_cpu_has_blas       (void);
@ggml_function("ggml_cpu_has_blas", [], ctypes.c_int, enabled=hasattr(lib, "ggml_cpu_has_blas"))
def ggml_cpu_has_blas() -> int:
    ...


# GGML_API int ggml_cpu_has_cuda       (void);
@ggml_function("ggml_cpu_has_cuda", [], ctypes.c_int, enabled=hasattr(lib, "ggml_cpu_has_cuda"))
def ggml_cpu_has_cuda() -> int:
    ...


# GGML_API int ggml_cpu_has_vulkan     (void);
@ggml_function("ggml_cpu_has_vulkan", [], ctypes.c_int, enabled=hasattr(lib, "ggml_cpu_has_vulkan"))
def ggml_cpu_has_vulkan() -> int:
    ...


# GGML_API int ggml_cpu_has_kompute    (void);
@ggml_function("ggml_cpu_has_kompute", [], ctypes.c_int, enabled=hasattr(lib, "ggml_cpu_has_kompute"))
def ggml_cpu_has_kompute() -> int:
    ...


# GGML_API int ggml_cpu_has_gpublas    (void);
@ggml_function("ggml_cpu_has_gpublas", [], ctypes.c_int, enabled=hasattr(lib, "ggml_cpu_has_gpublas"))
def ggml_cpu_has_gpublas() -> int:
    ...


# GGML_API int ggml_cpu_has_sse3       (void);
@ggml_function("ggml_cpu_has_sse3", [], ctypes.c_int)
def ggml_cpu_has_sse3() -> int:
    ...


# GGML_API int ggml_cpu_has_ssse3      (void);
@ggml_function("ggml_cpu_has_ssse3", [], ctypes.c_int)
def ggml_cpu_has_ssse3() -> int:
    ...


# GGML_API int ggml_cpu_has_sycl       (void);
@ggml_function("ggml_cpu_has_sycl", [], ctypes.c_int, enabled=hasattr(lib, "ggml_cpu_has_sycl"))
def ggml_cpu_has_sycl() -> int:
    ...


# GGML_API int ggml_cpu_has_vsx        (void);
@ggml_function("ggml_cpu_has_vsx", [], ctypes.c_int)
def ggml_cpu_has_vsx() -> int:
    ...


# GGML_API int ggml_cpu_has_vxe        (void);
@ggml_function("ggml_cpu_has_vxe", [], ctypes.c_int)
def ggml_cpu_has_vxe() -> int:
    ...


# GGML_API int ggml_cpu_has_matmul_int8(void);
@ggml_function("ggml_cpu_has_matmul_int8", [], ctypes.c_int)
def ggml_cpu_has_matmul_int8() -> int:
    ...


# GGML_API int ggml_cpu_has_riscv_v    (void);
@ggml_function("ggml_cpu_has_riscv_v", [], ctypes.c_int)
def ggml_cpu_has_riscv_v() -> int:
    ...


# GGML_API int ggml_cpu_get_rvv_vlen   (void);
@ggml_function("ggml_cpu_get_rvv_vlen", [], ctypes.c_int)
def ggml_cpu_get_rvv_vlen() -> int:
    ...


# GGML_API int ggml_cpu_has_llamafile  (void);
@ggml_function("ggml_cpu_has_llamafile", [], ctypes.c_int)
def ggml_cpu_has_llamafile() -> int:
    ...


# GGML_BACKEND_API void ggml_cpu_init(void);
@ggml_function("ggml_cpu_init", [], None)
def ggml_cpu_init():
    ...


# GGML_BACKEND_API void ggml_cpu_fp32_to_fp32(const float * x, float * y, int64_t k);
@ggml_function(
    "ggml_cpu_fp32_to_fp32",
    [
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.c_int64,
    ],
    None,
)
def ggml_cpu_fp32_to_fp32(
    x: CtypesPointer[ctypes.c_float],
    y: CtypesPointer[ctypes.c_float],
    k: Union[ctypes.c_int64, int],
    /,
) -> None:
    ...


# GGML_BACKEND_API void ggml_cpu_fp32_to_i32(const float * x, int32_t * y, int64_t k);
@ggml_function(
    "ggml_cpu_fp32_to_i32",
    [
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_int32),
        ctypes.c_int64,
    ],
    None,
)
def ggml_cpu_fp32_to_i32(
    x: CtypesPointer[ctypes.c_float],
    y: CtypesPointer[ctypes.c_int32],
    k: Union[ctypes.c_int64, int],
    /,
) -> None:
    ...


# GGML_BACKEND_API void ggml_cpu_fp32_to_fp16(const float * x, ggml_fp16_t * y, int64_t k);
@ggml_function(
    "ggml_cpu_fp32_to_fp16",
    [
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ggml_fp16_t),
        ctypes.c_int64,
    ],
    None,
)
def ggml_cpu_fp32_to_fp16(
    x: CtypesPointer[ctypes.c_float],
    y: CtypesPointer[ggml_fp16_t],
    k: Union[ctypes.c_int64, int],
    /,
) -> None:
    ...


# GGML_BACKEND_API void ggml_cpu_fp16_to_fp32(const ggml_fp16_t * x, float * y, int64_t k);
@ggml_function(
    "ggml_cpu_fp16_to_fp32",
    [
        ctypes.POINTER(ggml_fp16_t),
        ctypes.POINTER(ctypes.c_float),
        ctypes.c_int64,
    ],
    None,
)
def ggml_cpu_fp16_to_fp32(
    x: CtypesPointer[ggml_fp16_t],
    y: CtypesPointer[ctypes.c_float],
    k: Union[ctypes.c_int64, int],
    /,
) -> None:
    ...


# GGML_BACKEND_API void ggml_cpu_fp32_to_bf16(const float * x, ggml_bf16_t * y, int64_t k);
@ggml_function(
    "ggml_cpu_fp32_to_bf16",
    [
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ggml_bf16_t),
        ctypes.c_int64,
    ],
    None,
)
def ggml_cpu_fp32_to_bf16(
    x: CtypesPointer[ctypes.c_float],
    y: CtypesPointer[ggml_bf16_t],
    k: Union[ctypes.c_int64, int],
    /,
) -> None:
    ...


# GGML_BACKEND_API void ggml_cpu_bf16_to_fp32(const ggml_bf16_t * x, float * y, int64_t k);
@ggml_function(
    "ggml_cpu_bf16_to_fp32",
    [
        ctypes.POINTER(ggml_bf16_t),
        ctypes.POINTER(ctypes.c_float),
        ctypes.c_int64,
    ],
    None,
)
def ggml_cpu_bf16_to_fp32(
    x: CtypesPointer[ggml_bf16_t],
    y: CtypesPointer[ctypes.c_float],
    k: Union[ctypes.c_int64, int],
    /,
) -> None:
    ...


# //
# // Internal types and functions exposed for tests and benchmarks
# //

# typedef void (*ggml_to_float_t)(const void * x, float * y, int64_t k);
ggml_to_float_t = ctypes.CFUNCTYPE(
    None, ctypes.c_void_p, ctypes.POINTER(ctypes.c_float), ctypes.c_int64
)

# typedef void (*ggml_from_float_t)(const float * x, void * y, int64_t k);
ggml_from_float_t = ctypes.CFUNCTYPE(
    None, ctypes.POINTER(ctypes.c_float), ctypes.c_void_p, ctypes.c_int64
)

# typedef void (*ggml_vec_dot_t)   (int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT x, size_t bx,
#                                   const void * GGML_RESTRICT y, size_t by, int nrc);
ggml_vec_dot_t = ctypes.CFUNCTYPE(
    None,
    ctypes.c_int,
    ctypes.POINTER(ctypes.c_float),
    ctypes.c_size_t,
    ctypes.c_void_p,
    ctypes.c_size_t,
    ctypes.c_void_p,
    ctypes.c_size_t,
    ctypes.c_int,
)


# struct ggml_type_traits {
#     const char             * type_name;
#     int64_t                  blck_size;
#     int64_t                  blck_size_interleave;
#     size_t                   type_size;
#     bool                     is_quantized;
#     ggml_to_float_t          to_float;
#     ggml_from_float_t        from_float_ref;
# };
class ggml_type_traits(ctypes.Structure):
    """Internal types and functions exposed for tests and benchmarks.
    
    Attributes:
        type_name(bytes): Name of the type
        blck_size(int): Block size
        blck_size_interleave(int): Interleaved block size
        type_size(int): Size of the type
        is_quantized(bool): Is quantized
        to_float(ggml_to_float_t): Convert to float
        from_float_ref(ggml_from_float_t): Reference conversion from float"""

    if TYPE_CHECKING:
        type_name: bytes
        blck_size: int
        blck_size_interleave: int
        type_size: int
        is_quantized: bool
        to_float: Callable[[ctypes.c_void_p, CtypesPointer[ctypes.c_float], int], None]
        from_float_ref: Callable[[CtypesPointer[ctypes.c_float], ctypes.c_void_p, int], None]

    _fields_ = [
        ("type_name", ctypes.c_char_p),
        ("blck_size", ctypes.c_int64),
        ("blck_size_interleave", ctypes.c_int64),
        ("type_size", ctypes.c_size_t),
        ("is_quantized", ctypes.c_bool),
        ("to_float", ggml_to_float_t),
        ("from_float_ref", ggml_from_float_t),
    ]


ggml_type_traits_t = ggml_type_traits
ggml_type_traits_p: TypeAlias = "ctypes._Pointer[ggml_type_traits]"  # type: ignore


# GGML_API const struct ggml_type_traits * ggml_get_type_traits(enum ggml_type type);
@ggml_function("ggml_get_type_traits", [ctypes.c_int], ctypes.POINTER(ggml_type_traits))
def ggml_get_type_traits(
    type: Union[ctypes.c_int, int], /
) -> CtypesPointer[ggml_type_traits]:
    ...


def ggml_internal_get_type_traits(
    type: Union[ctypes.c_int, int], /
) -> ggml_type_traits:
    """Compatibility alias for the removed ggml_internal_get_type_traits API."""
    return ggml_get_type_traits(type).contents


# struct ggml_type_traits_cpu {
#     ggml_from_float_t from_float;
#     ggml_vec_dot_t    vec_dot;
#     enum ggml_type    vec_dot_type;
#     int64_t           nrows;
# };
class ggml_type_traits_cpu(ctypes.Structure):
    """CPU-specific conversion and dot-product functions."""

    if TYPE_CHECKING:
        from_float: Callable[[CtypesPointer[ctypes.c_float], ctypes.c_void_p, int], None]
        vec_dot: Callable[
            [
                int,
                CtypesPointer[ctypes.c_float],
                int,
                ctypes.c_void_p,
                int,
                ctypes.c_void_p,
                int,
                int,
            ],
            None,
        ]
        vec_dot_type: int
        nrows: int

    _fields_ = [
        ("from_float", ggml_from_float_t),
        ("vec_dot", ggml_vec_dot_t),
        ("vec_dot_type", ctypes.c_int),
        ("nrows", ctypes.c_int64),
    ]


ggml_type_traits_cpu_p: TypeAlias = "ctypes._Pointer[ggml_type_traits_cpu]"  # type: ignore


# GGML_BACKEND_API const struct ggml_type_traits_cpu * ggml_get_type_traits_cpu(enum ggml_type type);
@ggml_function("ggml_get_type_traits_cpu", [ctypes.c_int], ctypes.POINTER(ggml_type_traits_cpu))
def ggml_get_type_traits_cpu(
    type: Union[ctypes.c_int, int], /
) -> CtypesPointer[ggml_type_traits_cpu]:
    ...


#####################################################
# GGML ALLOC API
# source: include/ggml/ggml-alloc.h
#####################################################


# typedef struct ggml_backend_buffer_type * ggml_backend_buffer_type_t;
# typedef struct ggml_backend_buffer * ggml_backend_buffer_t;
# typedef struct ggml_backend * ggml_backend_t;
ggml_backend_buffer_type_t = NewType("ggml_backend_buffer_type_t", int)
ggml_backend_buffer_type_t_ctypes: TypeAlias = ctypes.c_void_p

ggml_backend_buffer_t = NewType("ggml_backend_buffer_t", int)
ggml_backend_buffer_t_ctypes: TypeAlias = ctypes.c_void_p

ggml_backend_t = NewType("ggml_backend_t", int)
ggml_backend_t_ctypes: TypeAlias = ctypes.c_void_p


# // Tensor allocator
# // Tensor allocator
# struct ggml_tallocr {
#     ggml_backend_buffer_t buffer;
#     void * base;
#     size_t alignment;
#     size_t offset;
# };
class ggml_tallocr(ctypes.Structure):
    """Tensor allocator

    Attributes:
        buffer: ggml_backend_buffer_t
        base: ctypes.c_void_p
        alignment: ctypes.c_size_t
        offset: ctypes.c_size_t"""

    if TYPE_CHECKING:
        buffer: ggml_backend_buffer_t
        base: ctypes.c_void_p
        alignment: int
        offset: int

    _fields_ = [
        ("buffer", ggml_backend_buffer_t_ctypes),
        ("base", ctypes.c_void_p),
        ("alignment", ctypes.c_size_t),
        ("offset", ctypes.c_size_t),
    ]

# typedef struct ggml_tallocr * ggml_tallocr_t;
if TYPE_CHECKING:
    ggml_tallocr_t = CtypesPointer[ggml_tallocr]
ggml_tallocr_t_ctypes = ctypes.POINTER(ggml_tallocr)


# GGML_API struct ggml_tallocr ggml_tallocr_new(ggml_backend_buffer_t buffer);
@ggml_function("ggml_tallocr_new", [ggml_backend_buffer_t_ctypes], ggml_tallocr)
def ggml_tallocr_new(buffer: Union[ggml_backend_buffer_t, int], /) -> ggml_tallocr:
    ...


# GGML_API enum ggml_status    ggml_tallocr_alloc(struct ggml_tallocr * talloc, struct ggml_tensor * tensor);
@ggml_function(
    "ggml_tallocr_alloc",
    [ggml_tallocr_t_ctypes, ctypes.POINTER(ggml_tensor)],
    ctypes.c_int,
)
def ggml_tallocr_alloc(
    talloc: ggml_tallocr_t, tensor: ggml_tensor_p, /
) -> int:
    ...


# // Graph allocator
# /*
#   Example usage:
#     ggml_gallocr_t galloc = ggml_gallocr_new(ggml_bacckend_cpu_buffer_type());

#     // optional: create a worst-case graph and reserve the buffers to avoid reallocations
#     ggml_gallocr_reserve(galloc, build_graph(max_batch));

#     // allocate the graph
#     struct ggml_cgraph * graph = build_graph(batch);
#     ggml_gallocr_alloc_graph(galloc, graph);

#     printf("compute buffer size: %zu bytes\n", ggml_gallocr_get_buffer_size(galloc, 0));

#     // evaluate the graph
#     ggml_backend_graph_compute(backend, graph);
# */

# // special tensor flags for use with the graph allocator:
# //   ggml_set_input(): all input tensors are allocated at the beginning of the graph in non-overlapping addresses
# //   ggml_set_output(): output tensors are never freed and never overwritten


# typedef struct ggml_gallocr * ggml_gallocr_t;
ggml_gallocr = NewType("ggml_gallocr", int)
ggml_gallocr_ctypes: TypeAlias = ctypes.c_void_p


# GGML_API ggml_gallocr_t ggml_gallocr_new(ggml_backend_buffer_type_t buft);
@ggml_function(
    "ggml_gallocr_new", [ggml_backend_buffer_type_t_ctypes], ggml_gallocr_ctypes
)
def ggml_gallocr_new(
    buft: Union[ggml_backend_buffer_type_t, int], /
) -> Optional[ggml_gallocr]:
    ...


# GGML_API ggml_gallocr_t ggml_gallocr_new_n(ggml_backend_buffer_type_t * bufts, int n_bufs);
@ggml_function(
    "ggml_gallocr_new_n",
    [ggml_backend_buffer_type_t_ctypes, ctypes.c_int],
    ggml_gallocr_ctypes,
)
def ggml_gallocr_new_n(
    bufts: Union[ggml_backend_buffer_type_t, int], n_bufs: int, /
) -> Optional[ggml_gallocr]:
    ...


# GGML_API void           ggml_gallocr_free(ggml_gallocr_t galloc);
@ggml_function("ggml_gallocr_free", [ggml_gallocr_ctypes], None)
def ggml_gallocr_free(galloc: Union[ggml_gallocr, int], /) -> None:
    ...


# // pre-allocate buffers from a measure graph - does not allocate or modify the graph
# // call with a worst-case graph to avoid buffer reallocations
# // not strictly required for single buffer usage: ggml_gallocr_alloc_graph will reallocate the buffers automatically if needed
# // returns false if the buffer allocation failed
# GGML_API bool ggml_gallocr_reserve(ggml_gallocr_t galloc, struct ggml_cgraph * graph);
@ggml_function(
    "ggml_gallocr_reserve",
    [ggml_gallocr_ctypes, ctypes.POINTER(ggml_cgraph)],
    ctypes.c_bool,
)
def ggml_gallocr_reserve(
    galloc: Union[ggml_gallocr, int], graph: ggml_cgraph_p, /
) -> bool:
    """pre-allocate buffers from a measure graph - does not allocate or modify the graph
    call with a worst-case graph to avoid buffer reallocations
    not strictly required for single buffer usage: ggml_gallocr_alloc_graph will reallocate the buffers automatically if needed
    returns false if the buffer allocation failed"""
    ...


# GGML_API void ggml_gallocr_reserve_n_size(
#     ggml_gallocr_t galloc,
#     struct ggml_cgraph * graph,
#     const int * node_buffer_ids,
#     const int * leaf_buffer_ids,
#     size_t * sizes);
@ggml_function(
    "ggml_gallocr_reserve_n_size",
    [
        ggml_gallocr_ctypes,
        ctypes.POINTER(ggml_cgraph),
        ctypes.POINTER(ctypes.c_int),
        ctypes.POINTER(ctypes.c_int),
        ctypes.POINTER(ctypes.c_size_t),
    ],
    None,
)
def ggml_gallocr_reserve_n_size(
    galloc: Union[ggml_gallocr, int],
    graph: ggml_cgraph_p,
    node_buffer_ids: CtypesPointer[ctypes.c_int],
    leaf_buffer_ids: CtypesPointer[ctypes.c_int],
    sizes: CtypesPointer[ctypes.c_size_t],
    /,
) -> None:
    """write the buffer sizes that would be allocated by ggml_gallocr_reserve_n"""
    ...


# GGML_API bool ggml_gallocr_reserve_n(
#     ggml_gallocr_t galloc,
#     struct ggml_cgraph * graph,
#     const int * node_buffer_ids,
#     const int * leaf_buffer_ids);
@ggml_function(
    "ggml_gallocr_reserve_n",
    [
        ggml_gallocr_ctypes,
        ctypes.POINTER(ggml_cgraph),
        ctypes.POINTER(ctypes.c_int),
        ctypes.POINTER(ctypes.c_int),
    ],
    ctypes.c_bool,
)
def ggml_gallocr_reserve_n(
    galloc: Union[ggml_gallocr, int],
    graph: ggml_cgraph_p,
    node_buffer_ids: CtypesPointer[ctypes.c_int],
    leaf_buffer_ids: CtypesPointer[ctypes.c_int],
    /,
) -> bool:
    ...


# // automatic reallocation if the topology changes when using a single buffer
# // returns false if using multiple buffers and a re-allocation is needed (call ggml_gallocr_reserve_n first to set the node buffers)
# GGML_API bool ggml_gallocr_alloc_graph(ggml_gallocr_t galloc, struct ggml_cgraph * graph);
@ggml_function(
    "ggml_gallocr_alloc_graph",
    [ggml_gallocr_ctypes, ctypes.POINTER(ggml_cgraph)],
    ctypes.c_bool,
)
def ggml_gallocr_alloc_graph(
    galloc: Union[ggml_gallocr, int], graph: ggml_cgraph_p, /
) -> bool:
    """automatic reallocation if the topology changes when using a single buffer
    returns false if using multiple buffers and a re-allocation is needed (call ggml_gallocr_reserve_n first to set the node buffers)
    """
    ...


# GGML_API size_t ggml_gallocr_get_buffer_size(ggml_gallocr_t galloc, int buffer_id);
@ggml_function(
    "ggml_gallocr_get_buffer_size", [ggml_gallocr_ctypes, ctypes.c_int], ctypes.c_size_t
)
def ggml_gallocr_get_buffer_size(
    galloc: Union[ggml_gallocr, int], buffer_id: Union[ctypes.c_int, int], /
) -> int:
    ...


# // Utils
# // Create a buffer and allocate all the tensors in a ggml_context
# GGML_API size_t ggml_backend_alloc_ctx_tensors_from_buft_size(struct ggml_context * ctx, ggml_backend_buffer_type_t buft);
@ggml_function(
    "ggml_backend_alloc_ctx_tensors_from_buft_size",
    [
        ggml_context_p_ctypes,
        ggml_backend_buffer_type_t_ctypes,
    ],
    ctypes.c_size_t,
)
def ggml_backend_alloc_ctx_tensors_from_buft_size(
    ctx: ggml_context_p, buft: Union[ggml_backend_buffer_type_t, int], /
) -> int:
    """Get the size of the buffer that would be allocated for all tensors in a context."""
    ...


# GGML_API struct ggml_backend_buffer * ggml_backend_alloc_ctx_tensors_from_buft(struct ggml_context * ctx, ggml_backend_buffer_type_t buft);
@ggml_function(
    "ggml_backend_alloc_ctx_tensors_from_buft",
    [
        ggml_context_p_ctypes,
        ggml_backend_buffer_type_t_ctypes,
    ],
    ggml_backend_buffer_t_ctypes,
)
def ggml_backend_alloc_ctx_tensors_from_buft(
    ctx: ggml_context_p, buft: Union[ggml_backend_buffer_type_t, int], /
) -> Optional[ggml_backend_buffer_t]:
    """Create a buffer and allocate all the tensors in a ggml_context"""
    ...


# GGML_API struct ggml_backend_buffer * ggml_backend_alloc_ctx_tensors(struct ggml_context * ctx, ggml_backend_t backend);
@ggml_function(
    "ggml_backend_alloc_ctx_tensors",
    [ggml_context_p_ctypes, ggml_backend_t_ctypes],
    ggml_backend_buffer_t_ctypes,
)
def ggml_backend_alloc_ctx_tensors(
    ctx: ggml_context_p, backend: Union[ggml_backend_t, int], /
) -> Optional[ggml_backend_buffer_t]:
    ...


#####################################################
# GGML Backend API
# source: include/ggml/ggml-backend.h
#####################################################

# typedef struct ggml_backend_buffer_type * ggml_backend_buffer_type_t;
# typedef struct ggml_backend_buffer * ggml_backend_buffer_t;
# typedef struct ggml_backend_event * ggml_backend_event_t;
# typedef struct ggml_backend * ggml_backend_t;
# typedef void * ggml_backend_graph_plan_t;
# typedef struct ggml_backend_reg * ggml_backend_reg_t;
# typedef struct ggml_backend_device * ggml_backend_dev_t;
ggml_backend_graph_plan_t = NewType("ggml_backend_graph_plan_t", int)
ggml_backend_graph_plan_t_ctypes: TypeAlias = ctypes.c_void_p

ggml_backend_event_t = NewType("ggml_backend_event_t", int)
ggml_backend_event_t_ctypes: TypeAlias = ctypes.c_void_p

ggml_backend_reg_t = NewType("ggml_backend_reg_t", int)
ggml_backend_reg_t_ctypes: TypeAlias = ctypes.c_void_p

ggml_backend_dev_t = NewType("ggml_backend_dev_t", int)
ggml_backend_dev_t_ctypes: TypeAlias = ctypes.c_void_p

# //
# // Backend buffer
# //


# // buffer type
# GGML_API           const char *          ggml_backend_buft_name            (ggml_backend_buffer_type_t buft);
@ggml_function(
    "ggml_backend_buft_name", [ggml_backend_buffer_type_t_ctypes], ctypes.c_char_p
)
def ggml_backend_buft_name(buft: Union[ggml_backend_buffer_type_t, int], /) -> bytes:
    ...


# GGML_API GGML_CALL ggml_backend_buffer_t ggml_backend_buft_alloc_buffer    (ggml_backend_buffer_type_t buft, size_t size);
@ggml_function(
    "ggml_backend_buft_alloc_buffer",
    [
        ggml_backend_buffer_type_t_ctypes,
        ctypes.c_size_t,
    ],
    ggml_backend_buffer_t_ctypes,
)
def ggml_backend_buft_alloc_buffer(
    buft: Union[ggml_backend_buffer_type_t, int], size: Union[ctypes.c_size_t, int], /
) -> Optional[ggml_backend_buffer_t]:
    ...


# GGML_API           size_t                ggml_backend_buft_get_alignment   (ggml_backend_buffer_type_t buft);
@ggml_function(
    "ggml_backend_buft_get_alignment",
    [ggml_backend_buffer_type_t_ctypes],
    ctypes.c_size_t,
)
def ggml_backend_buft_get_alignment(
    buft: Union[ggml_backend_buffer_type_t, int], /
) -> int:
    ...


# GGML_API           size_t                ggml_backend_buft_get_max_size    (ggml_backend_buffer_type_t buft);
@ggml_function(
    "ggml_backend_buft_get_max_size",
    [ggml_backend_buffer_type_t_ctypes],
    ctypes.c_size_t,
)
def ggml_backend_buft_get_max_size(
    buft: Union[ggml_backend_buffer_type_t, int], /
) -> int:
    ...


# GGML_API GGML_CALL size_t                ggml_backend_buft_get_alloc_size  (ggml_backend_buffer_type_t buft, struct ggml_tensor * tensor);
@ggml_function(
    "ggml_backend_buft_get_alloc_size",
    [
        ggml_backend_buffer_type_t_ctypes,
        ctypes.POINTER(ggml_tensor),
    ],
    ctypes.c_size_t,
)
def ggml_backend_buft_get_alloc_size(
    buft: Union[ggml_backend_buffer_type_t, int], tensor: ggml_tensor_p, /
) -> int:
    ...


# GGML_API           bool                  ggml_backend_buft_is_host         (ggml_backend_buffer_type_t buft);
@ggml_function(
    "ggml_backend_buft_is_host", [ggml_backend_buffer_type_t_ctypes], ctypes.c_bool
)
def ggml_backend_buft_is_host(buft: Union[ggml_backend_buffer_type_t, int], /) -> bool:
    ...


# GGML_API ggml_backend_dev_t    ggml_backend_buft_get_device    (ggml_backend_buffer_type_t buft);
@ggml_function(
    "ggml_backend_buft_get_device",
    [ggml_backend_buffer_type_t_ctypes],
    ggml_backend_dev_t_ctypes,
)
def ggml_backend_buft_get_device(
    buft: Union[ggml_backend_buffer_type_t, int], /
) -> Optional[ggml_backend_dev_t]:
    ...


# // buffer
# enum ggml_backend_buffer_usage {
#     GGML_BACKEND_BUFFER_USAGE_ANY = 0,
#     GGML_BACKEND_BUFFER_USAGE_WEIGHTS = 1,
#     GGML_BACKEND_BUFFER_USAGE_COMPUTE = 2,
# };
GGML_BACKEND_BUFFER_USAGE_ANY = 0
GGML_BACKEND_BUFFER_USAGE_WEIGHTS = 1
GGML_BACKEND_BUFFER_USAGE_COMPUTE = 2


# GGML_API           const char *               ggml_backend_buffer_name          (ggml_backend_buffer_t buffer);
@ggml_function(
    "ggml_backend_buffer_name", [ggml_backend_buffer_t_ctypes], ctypes.c_char_p
)
def ggml_backend_buffer_name(buffer: Union[ggml_backend_buffer_t, int], /) -> bytes:
    ...


# GGML_API           void                       ggml_backend_buffer_free          (ggml_backend_buffer_t buffer);
@ggml_function("ggml_backend_buffer_free", [ggml_backend_buffer_t_ctypes], None)
def ggml_backend_buffer_free(buffer: Union[ggml_backend_buffer_t, int], /):
    ...


# GGML_API           void *                     ggml_backend_buffer_get_base      (ggml_backend_buffer_t buffer);
@ggml_function(
    "ggml_backend_buffer_get_base", [ggml_backend_buffer_t_ctypes], ctypes.c_void_p
)
def ggml_backend_buffer_get_base(
    buffer: Union[ggml_backend_buffer_t, int], /
) -> Optional[int]:
    ...


# GGML_API           size_t                     ggml_backend_buffer_get_size      (ggml_backend_buffer_t buffer);
@ggml_function(
    "ggml_backend_buffer_get_size", [ggml_backend_buffer_t_ctypes], ctypes.c_size_t
)
def ggml_backend_buffer_get_size(buffer: Union[ggml_backend_buffer_t, int], /) -> int:
    ...


# GGML_API enum ggml_status                    ggml_backend_buffer_init_tensor   (ggml_backend_buffer_t buffer, struct ggml_tensor * tensor);
@ggml_function(
    "ggml_backend_buffer_init_tensor",
    [
        ggml_backend_buffer_t_ctypes,
        ctypes.POINTER(ggml_tensor),
    ],
    ctypes.c_int,
)
def ggml_backend_buffer_init_tensor(
    buffer: Union[ggml_backend_buffer_t, int], tensor: ggml_tensor_p, /
) -> int:
    ...


# GGML_API           size_t                     ggml_backend_buffer_get_alignment (ggml_backend_buffer_t buffer);
@ggml_function(
    "ggml_backend_buffer_get_alignment", [ggml_backend_buffer_t_ctypes], ctypes.c_size_t
)
def ggml_backend_buffer_get_alignment(
    buffer: Union[ggml_backend_buffer_t, int], /
) -> int:
    ...


# GGML_API           size_t                     ggml_backend_buffer_get_max_size  (ggml_backend_buffer_t buffer);
@ggml_function(
    "ggml_backend_buffer_get_max_size", [ggml_backend_buffer_t_ctypes], ctypes.c_size_t
)
def ggml_backend_buffer_get_max_size(
    buffer: Union[ggml_backend_buffer_t, int], /
) -> int:
    ...


# GGML_API           size_t                     ggml_backend_buffer_get_alloc_size(ggml_backend_buffer_t buffer, struct ggml_tensor * tensor);
@ggml_function(
    "ggml_backend_buffer_get_alloc_size",
    [
        ggml_backend_buffer_t_ctypes,
        ctypes.POINTER(ggml_tensor),
    ],
    ctypes.c_size_t,
)
def ggml_backend_buffer_get_alloc_size(
    buffer: Union[ggml_backend_buffer_t, int], tensor: ggml_tensor_p, /
) -> int:
    ...


# GGML_API           void                       ggml_backend_buffer_clear         (ggml_backend_buffer_t buffer, uint8_t value);
@ggml_function(
    "ggml_backend_buffer_clear", [ggml_backend_buffer_t_ctypes, ctypes.c_uint8], None
)
def ggml_backend_buffer_clear(
    buffer: Union[ggml_backend_buffer_t, int], value: ctypes.c_uint8, /
):
    ...


# GGML_API           bool                       ggml_backend_buffer_is_host       (ggml_backend_buffer_t buffer);
@ggml_function(
    "ggml_backend_buffer_is_host", [ggml_backend_buffer_t_ctypes], ctypes.c_bool
)
def ggml_backend_buffer_is_host(buffer: Union[ggml_backend_buffer_t, int], /) -> bool:
    ...


# GGML_API           void                       ggml_backend_buffer_set_usage     (ggml_backend_buffer_t buffer, enum ggml_backend_buffer_usage usage);
@ggml_function(
    "ggml_backend_buffer_set_usage", [ggml_backend_buffer_t_ctypes, ctypes.c_int], None
)
def ggml_backend_buffer_set_usage(
    buffer: Union[ggml_backend_buffer_t, int], usage: Union[ctypes.c_int, int], /
):
    ...


# GGML_API enum ggml_backend_buffer_usage ggml_backend_buffer_get_usage(ggml_backend_buffer_t buffer);
@ggml_function(
    "ggml_backend_buffer_get_usage", [ggml_backend_buffer_t_ctypes], ctypes.c_int
)
def ggml_backend_buffer_get_usage(buffer: Union[ggml_backend_buffer_t, int], /) -> int:
    ...


# GGML_API           ggml_backend_buffer_type_t ggml_backend_buffer_get_type      (ggml_backend_buffer_t buffer);
@ggml_function(
    "ggml_backend_buffer_get_type",
    [ggml_backend_buffer_t_ctypes],
    ggml_backend_buffer_type_t_ctypes,
)
def ggml_backend_buffer_get_type(
    buffer: Union[ggml_backend_buffer_t, int], /
) -> Optional[ggml_backend_buffer_type_t]:
    ...


# GGML_API           void                       ggml_backend_buffer_reset         (ggml_backend_buffer_t buffer);
@ggml_function("ggml_backend_buffer_reset", [ggml_backend_buffer_t_ctypes], None)
def ggml_backend_buffer_reset(buffer: Union[ggml_backend_buffer_t, int], /):
    ...


# //
# // Backend
# //


# GGML_API ggml_guid_t  ggml_backend_guid(ggml_backend_t backend);
@ggml_function(
    "ggml_backend_guid",
    [ggml_backend_t_ctypes],
    ggml_guid_t_ctypes,
)
def ggml_backend_guid(
    backend: Union[ggml_backend_t, int], /
) -> Optional[ggml_guid_t]:
    ...


# GGML_API const char * ggml_backend_name(ggml_backend_t backend);
@ggml_function("ggml_backend_name", [ggml_backend_t_ctypes], ctypes.c_char_p)
def ggml_backend_name(backend: Union[ggml_backend_t, int], /) -> bytes:
    ...


# GGML_API void         ggml_backend_free(ggml_backend_t backend);
@ggml_function("ggml_backend_free", [ggml_backend_t_ctypes], None)
def ggml_backend_free(backend: Union[ggml_backend_t, int], /):
    ...


# GGML_API ggml_backend_buffer_type_t ggml_backend_get_default_buffer_type(ggml_backend_t backend);
@ggml_function(
    "ggml_backend_get_default_buffer_type",
    [ggml_backend_t_ctypes],
    ggml_backend_buffer_type_t_ctypes,
)
def ggml_backend_get_default_buffer_type(
    backend: Union[ggml_backend_t, int], /
) -> Optional[ggml_backend_buffer_type_t]:
    ...


# GGML_API ggml_backend_buffer_t      ggml_backend_alloc_buffer(ggml_backend_t backend, size_t size);
@ggml_function(
    "ggml_backend_alloc_buffer",
    [ggml_backend_t_ctypes, ctypes.c_size_t],
    ggml_backend_buffer_t,
)
def ggml_backend_alloc_buffer(
    backend: Union[ggml_backend_t, int], size: Union[ctypes.c_size_t, int], /
) -> Optional[ggml_backend_buffer_t]:
    ...


# GGML_API size_t                     ggml_backend_get_alignment(ggml_backend_t backend);
@ggml_function("ggml_backend_get_alignment", [ggml_backend_t_ctypes], ctypes.c_size_t)
def ggml_backend_get_alignment(
    backend: Union[ggml_backend_t, int],
) -> int:
    ...


# GGML_API size_t                     ggml_backend_get_max_size(ggml_backend_t backend);
@ggml_function("ggml_backend_get_max_size", [ggml_backend_t_ctypes], ctypes.c_size_t)
def ggml_backend_get_max_size(
    backend: Union[ggml_backend_t, int],
) -> int:
    ...


# GGML_API void ggml_backend_tensor_set_async(ggml_backend_t backend,       struct ggml_tensor * tensor, const void * data, size_t offset, size_t size);
@ggml_function(
    "ggml_backend_tensor_set_async",
    [
        ggml_backend_t_ctypes,
        ctypes.POINTER(ggml_tensor),
        ctypes.c_void_p,
        ctypes.c_size_t,
        ctypes.c_size_t,
    ],
    None,
)
def ggml_backend_tensor_set_async(
    backend: Union[ggml_backend_t, int],
    tensor: ggml_tensor_p,
    data: Union[ctypes.c_void_p, int, None],
    offset: Union[ctypes.c_size_t, int],
    size: Union[ctypes.c_size_t, int],
    /,
):
    ...


# GGML_API void ggml_backend_tensor_get_async(ggml_backend_t backend, const struct ggml_tensor * tensor,       void * data, size_t offset, size_t size);
@ggml_function(
    "ggml_backend_tensor_get_async",
    [
        ggml_backend_t_ctypes,
        ctypes.POINTER(ggml_tensor),
        ctypes.c_void_p,
        ctypes.c_size_t,
        ctypes.c_size_t,
    ],
    None,
)
def ggml_backend_tensor_get_async(
    backend: Union[ggml_backend_t, int],
    tensor: ggml_tensor_p,
    data: Union[ctypes.c_void_p, int, None],
    offset: Union[ctypes.c_size_t, int],
    size: Union[ctypes.c_size_t, int],
    /,
):
    ...


# GGML_API void ggml_backend_tensor_set_2d_async(ggml_backend_t backend, struct ggml_tensor * tensor, const void * data, size_t offset, size_t size, size_t n_copies, size_t stride_tensor, size_t stride_data);
@ggml_function(
    "ggml_backend_tensor_set_2d_async",
    [
        ggml_backend_t_ctypes,
        ctypes.POINTER(ggml_tensor),
        ctypes.c_void_p,
        ctypes.c_size_t,
        ctypes.c_size_t,
        ctypes.c_size_t,
        ctypes.c_size_t,
        ctypes.c_size_t,
    ],
    None,
)
def ggml_backend_tensor_set_2d_async(
    backend: Union[ggml_backend_t, int],
    tensor: ggml_tensor_p,
    data: Union[ctypes.c_void_p, int, None],
    offset: Union[ctypes.c_size_t, int],
    size: Union[ctypes.c_size_t, int],
    n_copies: Union[ctypes.c_size_t, int],
    stride_tensor: Union[ctypes.c_size_t, int],
    stride_data: Union[ctypes.c_size_t, int],
    /,
):
    ...


# GGML_API void ggml_backend_tensor_get_2d_async(ggml_backend_t backend, const struct ggml_tensor * tensor, void * data, size_t offset, size_t size, size_t n_copies, size_t stride_tensor, size_t stride_data);
@ggml_function(
    "ggml_backend_tensor_get_2d_async",
    [
        ggml_backend_t_ctypes,
        ctypes.POINTER(ggml_tensor),
        ctypes.c_void_p,
        ctypes.c_size_t,
        ctypes.c_size_t,
        ctypes.c_size_t,
        ctypes.c_size_t,
        ctypes.c_size_t,
    ],
    None,
)
def ggml_backend_tensor_get_2d_async(
    backend: Union[ggml_backend_t, int],
    tensor: ggml_tensor_p,
    data: Union[ctypes.c_void_p, int, None],
    offset: Union[ctypes.c_size_t, int],
    size: Union[ctypes.c_size_t, int],
    n_copies: Union[ctypes.c_size_t, int],
    stride_tensor: Union[ctypes.c_size_t, int],
    stride_data: Union[ctypes.c_size_t, int],
    /,
):
    ...


# GGML_API GGML_CALL void ggml_backend_tensor_set(      struct ggml_tensor * tensor, const void * data, size_t offset, size_t size);
@ggml_function(
    "ggml_backend_tensor_set",
    [
        ctypes.POINTER(ggml_tensor),
        ctypes.c_void_p,
        ctypes.c_size_t,
        ctypes.c_size_t,
    ],
    None,
)
def ggml_backend_tensor_set(
    tensor: ggml_tensor_p,
    data: Union[ctypes.c_void_p, int, None],
    offset: Union[ctypes.c_size_t, int],
    size: Union[ctypes.c_size_t, int],
    /,
):
    ...


# GGML_API GGML_CALL void ggml_backend_tensor_get(const struct ggml_tensor * tensor,       void * data, size_t offset, size_t size);
@ggml_function(
    "ggml_backend_tensor_get",
    [
        ctypes.POINTER(ggml_tensor),
        ctypes.c_void_p,
        ctypes.c_size_t,
        ctypes.c_size_t,
    ],
    None,
)
def ggml_backend_tensor_get(
    tensor: ggml_tensor_p,
    data: Union[ctypes.c_void_p, int, None],
    offset: Union[ctypes.c_size_t, int],
    size: Union[ctypes.c_size_t, int],
    /,
):
    ...


# GGML_API void ggml_backend_tensor_set_2d(struct ggml_tensor * tensor, const void * data, size_t offset, size_t size, size_t n_copies, size_t stride_tensor, size_t stride_data);
@ggml_function(
    "ggml_backend_tensor_set_2d",
    [
        ctypes.POINTER(ggml_tensor),
        ctypes.c_void_p,
        ctypes.c_size_t,
        ctypes.c_size_t,
        ctypes.c_size_t,
        ctypes.c_size_t,
        ctypes.c_size_t,
    ],
    None,
)
def ggml_backend_tensor_set_2d(
    tensor: ggml_tensor_p,
    data: Union[ctypes.c_void_p, int, None],
    offset: Union[ctypes.c_size_t, int],
    size: Union[ctypes.c_size_t, int],
    n_copies: Union[ctypes.c_size_t, int],
    stride_tensor: Union[ctypes.c_size_t, int],
    stride_data: Union[ctypes.c_size_t, int],
    /,
):
    ...


# GGML_API void ggml_backend_tensor_get_2d(const struct ggml_tensor * tensor, void * data, size_t offset, size_t size, size_t n_copies, size_t stride_tensor, size_t stride_data);
@ggml_function(
    "ggml_backend_tensor_get_2d",
    [
        ctypes.POINTER(ggml_tensor),
        ctypes.c_void_p,
        ctypes.c_size_t,
        ctypes.c_size_t,
        ctypes.c_size_t,
        ctypes.c_size_t,
        ctypes.c_size_t,
    ],
    None,
)
def ggml_backend_tensor_get_2d(
    tensor: ggml_tensor_p,
    data: Union[ctypes.c_void_p, int, None],
    offset: Union[ctypes.c_size_t, int],
    size: Union[ctypes.c_size_t, int],
    n_copies: Union[ctypes.c_size_t, int],
    stride_tensor: Union[ctypes.c_size_t, int],
    stride_data: Union[ctypes.c_size_t, int],
    /,
):
    ...


# GGML_API void ggml_backend_tensor_memset(struct ggml_tensor * tensor, uint8_t value, size_t offset, size_t size);
@ggml_function(
    "ggml_backend_tensor_memset",
    [
        ctypes.POINTER(ggml_tensor),
        ctypes.c_uint8,
        ctypes.c_size_t,
        ctypes.c_size_t,
    ],
    None,
)
def ggml_backend_tensor_memset(
    tensor: ggml_tensor_p,
    value: Union[ctypes.c_uint8, int],
    offset: Union[ctypes.c_size_t, int],
    size: Union[ctypes.c_size_t, int],
    /,
):
    ...


# GGML_API void ggml_backend_synchronize(ggml_backend_t backend);
@ggml_function("ggml_backend_synchronize", [ggml_backend_t_ctypes], None)
def ggml_backend_synchronize(backend: Union[ggml_backend_t, int], /):
    ...


# GGML_API ggml_backend_graph_plan_t ggml_backend_graph_plan_create(ggml_backend_t backend, struct ggml_cgraph * cgraph);
@ggml_function(
    "ggml_backend_graph_plan_create",
    [
        ggml_backend_t_ctypes,
        ctypes.POINTER(ggml_cgraph),
    ],
    ggml_backend_graph_plan_t_ctypes,
)
def ggml_backend_graph_plan_create(
    backend: Union[ggml_backend_t, int],
    cgraph: ggml_cgraph_p,
) -> ggml_backend_graph_plan_t:
    ...


# GGML_API void                      ggml_backend_graph_plan_free  (ggml_backend_t backend, ggml_backend_graph_plan_t plan);
@ggml_function(
    "ggml_backend_graph_plan_free",
    [
        ggml_backend_t_ctypes,
        ggml_backend_graph_plan_t_ctypes,
    ],
    None,
)
def ggml_backend_graph_plan_free(
    backend: Union[ggml_backend_t, int], plan: ggml_backend_graph_plan_t, /
):
    ...


# GGML_API enum ggml_status ggml_backend_graph_plan_compute (ggml_backend_t backend, ggml_backend_graph_plan_t plan);
@ggml_function(
    "ggml_backend_graph_plan_compute",
    [ggml_backend_t_ctypes, ggml_backend_graph_plan_t_ctypes],
    ctypes.c_int,
)
def ggml_backend_graph_plan_compute(
    backend: Union[ggml_backend_t, int], plan: ggml_backend_graph_plan_t, /
) -> int:
    ...

# GGML_API enum ggml_status ggml_backend_graph_compute      (ggml_backend_t backend, struct ggml_cgraph * cgraph);
@ggml_function(
    "ggml_backend_graph_compute",
    [ggml_backend_t_ctypes, ctypes.POINTER(ggml_cgraph)],
    ctypes.c_int,
)
def ggml_backend_graph_compute(
    backend: Union[ggml_backend_t, int], cgraph: ggml_cgraph_p, /
) -> int:
    ...

# GGML_API enum ggml_status ggml_backend_graph_compute_async(ggml_backend_t backend, struct ggml_cgraph * cgraph);
@ggml_function(
    "ggml_backend_graph_compute_async",
    [ggml_backend_t_ctypes, ctypes.POINTER(ggml_cgraph)],
    ctypes.c_int,
)
def ggml_backend_graph_compute_async(
    backend: Union[ggml_backend_t, int], cgraph: ggml_cgraph_p, /
) -> int:
    ...


# GGML_API bool ggml_backend_supports_op(ggml_backend_t backend, const struct ggml_tensor * op);
@ggml_function(
    "ggml_backend_supports_op",
    [ggml_backend_t_ctypes, ctypes.POINTER(ggml_tensor)],
    ctypes.c_bool,
)
def ggml_backend_supports_op(
    backend: Union[ggml_backend_t, int],
    op: ggml_tensor_p,
) -> bool:
    ...


# GGML_API bool ggml_backend_supports_buft(ggml_backend_t backend, ggml_backend_buffer_type_t buft);
@ggml_function(
    "ggml_backend_supports_buft",
    [ggml_backend_t_ctypes, ggml_backend_buffer_type_t_ctypes],
    ctypes.c_bool,
)
def ggml_backend_supports_buft(
    backend: Union[ggml_backend_t, int],
    buft: Union[ggml_backend_buffer_type_t, int],
) -> bool:
    ...


# GGML_API bool ggml_backend_offload_op(ggml_backend_t backend, const struct ggml_tensor * op);
@ggml_function(
    "ggml_backend_offload_op",
    [ggml_backend_t_ctypes, ctypes.POINTER(ggml_tensor)],
    ctypes.c_bool,
)
def ggml_backend_offload_op(
    backend: Union[ggml_backend_t, int],
    op: ggml_tensor_p,
) -> bool:
    ...


# // tensor copy between different backends
# GGML_API void ggml_backend_tensor_copy(struct ggml_tensor * src, struct ggml_tensor * dst);
@ggml_function(
    "ggml_backend_tensor_copy",
    [
        ctypes.POINTER(ggml_tensor),
        ctypes.POINTER(ggml_tensor),
    ],
    None,
)
def ggml_backend_tensor_copy(src: ggml_tensor_p, dst: ggml_tensor_p, /):
    ...


# // asynchronous copy
# // the copy is performed after all the currently queued operations in backend_src
# // backend_dst will wait for the copy to complete before performing other operations
# // automatic fallback to sync copy if async is not supported
# GGML_API void ggml_backend_tensor_copy_async(ggml_backend_t backend_src, ggml_backend_t backend_dst, struct ggml_tensor * src, struct ggml_tensor * dst);
@ggml_function(
    "ggml_backend_tensor_copy_async",
    [
        ggml_backend_t_ctypes,
        ggml_backend_t_ctypes,
        ctypes.POINTER(ggml_tensor),
        ctypes.POINTER(ggml_tensor),
    ],
    None,
)
def ggml_backend_tensor_copy_async(
    backend_src: Union[ggml_backend_t, int],
    backend_dst: Union[ggml_backend_t, int],
    src: ggml_tensor_p,
    dst: ggml_tensor_p,
    /,
):
    ...


# GGML_API ggml_backend_dev_t ggml_backend_get_device(ggml_backend_t backend);
@ggml_function("ggml_backend_get_device", [ggml_backend_t_ctypes], ggml_backend_dev_t_ctypes)
def ggml_backend_get_device(
    backend: Union[ggml_backend_t, int],
) -> Optional[ggml_backend_dev_t]:
    ...


# // events
# GGML_API ggml_backend_event_t   ggml_backend_event_new        (ggml_backend_dev_t device);
@ggml_function("ggml_backend_event_new", [ggml_backend_dev_t_ctypes], ggml_backend_event_t_ctypes)
def ggml_backend_event_new(
    device: Union[ggml_backend_dev_t, int],
) -> Optional[ggml_backend_event_t]:
    ...


# GGML_API void                   ggml_backend_event_free       (ggml_backend_event_t event);
@ggml_function("ggml_backend_event_free", [ggml_backend_event_t_ctypes], None)
def ggml_backend_event_free(event: Union[ggml_backend_event_t, int], /):
    ...


# GGML_API void                   ggml_backend_event_record     (ggml_backend_event_t event, ggml_backend_t backend);
@ggml_function("ggml_backend_event_record", [ggml_backend_event_t_ctypes, ggml_backend_t_ctypes], None)
def ggml_backend_event_record(event: Union[ggml_backend_event_t, int], backend: Union[ggml_backend_t, int], /):
    ...


# GGML_API void                   ggml_backend_event_synchronize(ggml_backend_event_t event);
@ggml_function("ggml_backend_event_synchronize", [ggml_backend_event_t_ctypes], None)
def ggml_backend_event_synchronize(event: Union[ggml_backend_event_t, int], /):
    ...


# GGML_API void                   ggml_backend_event_wait       (ggml_backend_t backend, ggml_backend_event_t event); // wait async on event
@ggml_function(
    "ggml_backend_event_wait",
    [ggml_backend_t_ctypes, ggml_backend_event_t_ctypes],
    None,
)
def ggml_backend_event_wait(
    backend: Union[ggml_backend_t, int], event: ggml_backend_event_t, /
):
    ...


# //
# // CPU backend
# //


# GGML_API ggml_backend_t ggml_backend_cpu_init(void);
@ggml_function("ggml_backend_cpu_init", [], ggml_backend_t_ctypes)
def ggml_backend_cpu_init() -> Optional[ggml_backend_t]:
    ...


# GGML_API GGML_CALL bool ggml_backend_is_cpu                (ggml_backend_t backend);
@ggml_function("ggml_backend_is_cpu", [ggml_backend_t_ctypes], ctypes.c_bool)
def ggml_backend_is_cpu(
    backend: Union[ggml_backend_t, int],
) -> bool:
    ...


# GGML_API           void ggml_backend_cpu_set_n_threads     (ggml_backend_t backend_cpu, int n_threads);
@ggml_function(
    "ggml_backend_cpu_set_n_threads", [ggml_backend_t_ctypes, ctypes.c_int], None
)
def ggml_backend_cpu_set_n_threads(
    backend_cpu: Union[ggml_backend_t, int], n_threads: Union[ctypes.c_int, int], /
):
    ...


# GGML_BACKEND_API void ggml_backend_cpu_set_threadpool(ggml_backend_t backend_cpu, ggml_threadpool_t threadpool);
@ggml_function(
    "ggml_backend_cpu_set_threadpool", [ggml_backend_t_ctypes, ctypes.c_void_p], None
)
def ggml_backend_cpu_set_threadpool(
    backend_cpu: Union[ggml_backend_t, int], threadpool: Union[ctypes.c_void_p, int, None], /
):
    ...


# GGML_API           void ggml_backend_cpu_set_abort_callback(ggml_backend_t backend_cpu, ggml_abort_callback abort_callback, void * abort_callback_data);
@ggml_function(
    "ggml_backend_cpu_set_abort_callback",
    [
        ggml_backend_t_ctypes,
        ggml_abort_callback,
        ctypes.c_void_p,
    ],
    None,
)
def ggml_backend_cpu_set_abort_callback(
    backend_cpu: Union[ggml_backend_t, int],
    abort_callback,  # type: ignore
    abort_callback_data: Union[ctypes.c_void_p, int, None],
    /,
):
    ...


# GGML_BACKEND_API void ggml_backend_cpu_set_use_ref(ggml_backend_t backend_cpu, bool use_ref);
@ggml_function(
    "ggml_backend_cpu_set_use_ref", [ggml_backend_t_ctypes, ctypes.c_bool], None
)
def ggml_backend_cpu_set_use_ref(
    backend_cpu: Union[ggml_backend_t, int], use_ref: Union[ctypes.c_bool, bool], /
):
    ...


# // Create a backend buffer from an existing pointer
# GGML_API GGML_CALL ggml_backend_buffer_t ggml_backend_cpu_buffer_from_ptr(void * ptr, size_t size);
@ggml_function(
    "ggml_backend_cpu_buffer_from_ptr",
    [ctypes.c_void_p, ctypes.c_size_t],
    ggml_backend_buffer_t,
)
def ggml_backend_cpu_buffer_from_ptr(
    ptr: Union[ctypes.c_void_p, int, None],
    size: Union[ctypes.c_size_t, int],
) -> Optional[ggml_backend_buffer_t]:
    ...


# GGML_API GGML_CALL ggml_backend_buffer_type_t ggml_backend_cpu_buffer_type(void);
@ggml_function("ggml_backend_cpu_buffer_type", [], ggml_backend_buffer_type_t)
def ggml_backend_cpu_buffer_type() -> Optional[ggml_backend_buffer_type_t]:
    ...


# #ifdef GGML_USE_CPU_HBM
#     GGML_API ggml_backend_buffer_type_t ggml_backend_cpu_hbm_buffer_type(void);
# #endif
def ggml_backend_cpu_hbm_buffer_type() -> Optional[ggml_backend_buffer_type_t]:
    ...


if hasattr(lib, "ggml_backend_cpu_hbm_buffer_type"):
    ggml_backend_cpu_hbm_buffer_type = lib.ggml_backend_cpu_hbm_buffer_type
    ggml_backend_cpu_hbm_buffer_type.argtypes = []
    ggml_backend_cpu_hbm_buffer_type.restype = ggml_backend_buffer_type_t


# GGML_BACKEND_API ggml_backend_reg_t ggml_backend_cpu_reg(void);
@ggml_function("ggml_backend_cpu_reg", [], ggml_backend_reg_t_ctypes)
def ggml_backend_cpu_reg() -> Optional[ggml_backend_reg_t]:
    ...

# //
# // Backend registry
# //

# // The backend registry is a registry of all the available backends, and allows initializing backends in a generic way


# GGML_API size_t                     ggml_backend_reg_get_count(void);
@ggml_function("ggml_backend_reg_get_count", [], ctypes.c_size_t, enabled=hasattr(lib, "ggml_backend_reg_get_count"))
def ggml_backend_reg_get_count() -> int:
    ...


# GGML_API size_t                     ggml_backend_reg_find_by_name(const char * name);
@ggml_function("ggml_backend_reg_find_by_name", [ctypes.c_char_p], ctypes.c_size_t, enabled=hasattr(lib, "ggml_backend_reg_find_by_name"))
def ggml_backend_reg_find_by_name(
    name: bytes,
) -> int:
    ...


# GGML_API ggml_backend_t             ggml_backend_reg_init_backend_from_str(const char * backend_str); // str is name[:params]
@ggml_function(
    "ggml_backend_reg_init_backend_from_str",
    [ctypes.c_char_p],
    ggml_backend_t,
    enabled=hasattr(lib, "ggml_backend_reg_init_backend_from_str"),
)
def ggml_backend_reg_init_backend_from_str(
    backend_str: bytes,
) -> Optional[ggml_backend_t]:
    ...


# GGML_API const char *               ggml_backend_reg_get_name(size_t i);
@ggml_function("ggml_backend_reg_get_name", [ctypes.c_size_t], ctypes.c_char_p, enabled=hasattr(lib, "ggml_backend_reg_get_name"))
def ggml_backend_reg_get_name(
    i: Union[ctypes.c_size_t, int],
) -> bytes:
    ...


# GGML_API ggml_backend_t             ggml_backend_reg_init_backend(size_t i, const char * params); // params is backend-specific
@ggml_function(
    "ggml_backend_reg_init_backend",
    [ctypes.c_size_t, ctypes.c_char_p],
    ggml_backend_t,
    enabled=hasattr(lib, "ggml_backend_reg_init_backend"),
)
def ggml_backend_reg_init_backend(
    i: Union[ctypes.c_size_t, int],
    params: bytes,
) -> Optional[ggml_backend_t]:
    ...


# GGML_API ggml_backend_buffer_type_t ggml_backend_reg_get_default_buffer_type(size_t i);
@ggml_function(
    "ggml_backend_reg_get_default_buffer_type",
    [ctypes.c_size_t],
    ggml_backend_buffer_type_t,
    enabled=hasattr(lib, "ggml_backend_reg_get_default_buffer_type"),
)
def ggml_backend_reg_get_default_buffer_type(
    i: Union[ctypes.c_size_t, int],
) -> Optional[ggml_backend_buffer_type_t]:
    ...


# GGML_API ggml_backend_buffer_t      ggml_backend_reg_alloc_buffer(size_t i, size_t size);
@ggml_function(
    "ggml_backend_reg_alloc_buffer",
    [ctypes.c_size_t, ctypes.c_size_t],
    ggml_backend_buffer_t,
    enabled=hasattr(lib, "ggml_backend_reg_alloc_buffer"),
)
def ggml_backend_reg_alloc_buffer(
    i: Union[ctypes.c_size_t, int],
    size: Union[ctypes.c_size_t, int],
) -> Optional[ggml_backend_buffer_t]:
    ...


# GGML_API size_t ggml_backend_reg_count(void);
@ggml_function("ggml_backend_reg_count", [], ctypes.c_size_t)
def ggml_backend_reg_count() -> int:
    ...


# GGML_API ggml_backend_reg_t ggml_backend_reg_get(size_t index);
@ggml_function("ggml_backend_reg_get", [ctypes.c_size_t], ggml_backend_reg_t_ctypes)
def ggml_backend_reg_get(
    index: Union[ctypes.c_size_t, int],
) -> Optional[ggml_backend_reg_t]:
    ...


# GGML_API ggml_backend_reg_t ggml_backend_reg_by_name(const char * name);
@ggml_function("ggml_backend_reg_by_name", [ctypes.c_char_p], ggml_backend_reg_t_ctypes)
def ggml_backend_reg_by_name(name: bytes) -> Optional[ggml_backend_reg_t]:
    ...


# GGML_API const char * ggml_backend_reg_name(ggml_backend_reg_t reg);
@ggml_function("ggml_backend_reg_name", [ggml_backend_reg_t_ctypes], ctypes.c_char_p)
def ggml_backend_reg_name(reg: Union[ggml_backend_reg_t, int]) -> bytes:
    ...


# GGML_API size_t ggml_backend_reg_dev_count(ggml_backend_reg_t reg);
@ggml_function("ggml_backend_reg_dev_count", [ggml_backend_reg_t_ctypes], ctypes.c_size_t)
def ggml_backend_reg_dev_count(reg: Union[ggml_backend_reg_t, int]) -> int:
    ...


# GGML_API ggml_backend_dev_t ggml_backend_reg_dev_get(ggml_backend_reg_t reg, size_t index);
@ggml_function(
    "ggml_backend_reg_dev_get",
    [ggml_backend_reg_t_ctypes, ctypes.c_size_t],
    ggml_backend_dev_t_ctypes,
)
def ggml_backend_reg_dev_get(
    reg: Union[ggml_backend_reg_t, int],
    index: Union[ctypes.c_size_t, int],
) -> Optional[ggml_backend_dev_t]:
    ...


# GGML_API void * ggml_backend_reg_get_proc_address(ggml_backend_reg_t reg, const char * name);
@ggml_function(
    "ggml_backend_reg_get_proc_address",
    [ggml_backend_reg_t_ctypes, ctypes.c_char_p],
    ctypes.c_void_p,
)
def ggml_backend_reg_get_proc_address(
    reg: Union[ggml_backend_reg_t, int],
    name: bytes,
) -> Optional[int]:
    ...


# GGML_API void ggml_backend_device_register(ggml_backend_dev_t device);
@ggml_function("ggml_backend_device_register", [ggml_backend_dev_t_ctypes], None)
def ggml_backend_device_register(device: Union[ggml_backend_dev_t, int], /):
    ...


# Device enumeration
GGML_BACKEND_DEVICE_TYPE_CPU = 0
GGML_BACKEND_DEVICE_TYPE_GPU = 1
GGML_BACKEND_DEVICE_TYPE_IGPU = 2
GGML_BACKEND_DEVICE_TYPE_ACCEL = 3
GGML_BACKEND_DEVICE_TYPE_META = 4


# // functionality supported by the device
# struct ggml_backend_dev_caps {
#     // asynchronous operations
#     bool async;
#     // pinned host buffer
#     bool host_buffer;
#     // creating buffers from host ptr
#     bool buffer_from_host_ptr;
#     // event synchronization
#     bool events;
# };
class ggml_backend_dev_caps(ctypes.Structure):
    _fields_ = [
        ("async", ctypes.c_bool),
        ("host_buffer", ctypes.c_bool),
        ("buffer_from_host_ptr", ctypes.c_bool),
        ("events", ctypes.c_bool),
    ]


# // all the device properties
# struct ggml_backend_dev_props {
#     // device name
#     const char * name;
#     // device description
#     const char * description;
#     // device free memory in bytes
#     size_t memory_free;
#     // device total memory in bytes
#     size_t memory_total;
#     // device type
#     enum ggml_backend_dev_type type;
#     // device id
#     //   for PCI devices, this should be the lower-case PCI bus id formatted as "domain:bus:device.function" (e.g. "0000:c1:00.0")
#     //   if the id is unknown, this should be NULL
#     const char * device_id;
#     // device capabilities
#     struct ggml_backend_dev_caps caps;
# };
class ggml_backend_dev_props(ctypes.Structure):
    _fields_ = [
        ("name", ctypes.c_char_p),
        ("description", ctypes.c_char_p),
        ("memory_free", ctypes.c_size_t),
        ("memory_total", ctypes.c_size_t),
        ("type", ctypes.c_int),
        ("device_id", ctypes.c_char_p),
        ("caps", ggml_backend_dev_caps),
    ]


# GGML_API size_t ggml_backend_dev_count(void);
@ggml_function("ggml_backend_dev_count", [], ctypes.c_size_t)
def ggml_backend_dev_count() -> int:
    ...


# GGML_API ggml_backend_dev_t ggml_backend_dev_get(size_t index);
@ggml_function("ggml_backend_dev_get", [ctypes.c_size_t], ggml_backend_dev_t_ctypes)
def ggml_backend_dev_get(index: Union[ctypes.c_size_t, int]) -> Optional[ggml_backend_dev_t]:
    ...


# GGML_API ggml_backend_dev_t ggml_backend_dev_by_name(const char * name);
@ggml_function("ggml_backend_dev_by_name", [ctypes.c_char_p], ggml_backend_dev_t_ctypes)
def ggml_backend_dev_by_name(name: bytes) -> Optional[ggml_backend_dev_t]:
    ...


# GGML_API ggml_backend_dev_t ggml_backend_dev_by_type(enum ggml_backend_dev_type type);
@ggml_function("ggml_backend_dev_by_type", [ctypes.c_int], ggml_backend_dev_t_ctypes)
def ggml_backend_dev_by_type(type: Union[ctypes.c_int, int]) -> Optional[ggml_backend_dev_t]:
    ...


# GGML_API const char * ggml_backend_dev_name(ggml_backend_dev_t device);
@ggml_function("ggml_backend_dev_name", [ggml_backend_dev_t_ctypes], ctypes.c_char_p)
def ggml_backend_dev_name(device: Union[ggml_backend_dev_t, int]) -> bytes:
    ...


# GGML_API const char * ggml_backend_dev_description(ggml_backend_dev_t device);
@ggml_function("ggml_backend_dev_description", [ggml_backend_dev_t_ctypes], ctypes.c_char_p)
def ggml_backend_dev_description(device: Union[ggml_backend_dev_t, int]) -> bytes:
    ...


# GGML_API void ggml_backend_dev_memory(ggml_backend_dev_t device, size_t * free, size_t * total);
@ggml_function(
    "ggml_backend_dev_memory",
    [
        ggml_backend_dev_t_ctypes,
        ctypes.POINTER(ctypes.c_size_t),
        ctypes.POINTER(ctypes.c_size_t),
    ],
    None,
)
def ggml_backend_dev_memory(
    device: Union[ggml_backend_dev_t, int],
    free: CtypesPointer[ctypes.c_size_t],
    total: CtypesPointer[ctypes.c_size_t],
    /,
):
    ...


# GGML_API enum ggml_backend_dev_type ggml_backend_dev_type(ggml_backend_dev_t device);
@ggml_function("ggml_backend_dev_type", [ggml_backend_dev_t_ctypes], ctypes.c_int)
def ggml_backend_dev_type(device: Union[ggml_backend_dev_t, int]) -> int:
    ...


# GGML_API void ggml_backend_dev_get_props(ggml_backend_dev_t device, struct ggml_backend_dev_props * props);
@ggml_function(
    "ggml_backend_dev_get_props",
    [ggml_backend_dev_t_ctypes, ctypes.POINTER(ggml_backend_dev_props)],
    None,
)
def ggml_backend_dev_get_props(
    device: Union[ggml_backend_dev_t, int],
    props: CtypesPointer[ggml_backend_dev_props],
    /,
):
    ...


# GGML_API ggml_backend_reg_t ggml_backend_dev_backend_reg(ggml_backend_dev_t device);
@ggml_function("ggml_backend_dev_backend_reg", [ggml_backend_dev_t_ctypes], ggml_backend_reg_t_ctypes)
def ggml_backend_dev_backend_reg(device: Union[ggml_backend_dev_t, int]) -> Optional[ggml_backend_reg_t]:
    ...


# GGML_API ggml_backend_t ggml_backend_dev_init(ggml_backend_dev_t device, const char * params);
@ggml_function("ggml_backend_dev_init", [ggml_backend_dev_t_ctypes, ctypes.c_char_p], ggml_backend_t_ctypes)
def ggml_backend_dev_init(device: Union[ggml_backend_dev_t, int], params: Optional[bytes]) -> Optional[ggml_backend_t]:
    ...


# GGML_API ggml_backend_buffer_type_t ggml_backend_dev_buffer_type(ggml_backend_dev_t device);
@ggml_function("ggml_backend_dev_buffer_type", [ggml_backend_dev_t_ctypes], ggml_backend_buffer_type_t_ctypes)
def ggml_backend_dev_buffer_type(device: Union[ggml_backend_dev_t, int]) -> Optional[ggml_backend_buffer_type_t]:
    ...


# GGML_API ggml_backend_buffer_type_t ggml_backend_dev_host_buffer_type(ggml_backend_dev_t device);
@ggml_function(
    "ggml_backend_dev_host_buffer_type",
    [ggml_backend_dev_t_ctypes],
    ggml_backend_buffer_type_t_ctypes,
)
def ggml_backend_dev_host_buffer_type(
    device: Union[ggml_backend_dev_t, int],
) -> Optional[ggml_backend_buffer_type_t]:
    ...


# GGML_API ggml_backend_buffer_t ggml_backend_dev_buffer_from_host_ptr(ggml_backend_dev_t device, void * ptr, size_t size, size_t max_tensor_size);
@ggml_function(
    "ggml_backend_dev_buffer_from_host_ptr",
    [
        ggml_backend_dev_t_ctypes,
        ctypes.c_void_p,
        ctypes.c_size_t,
        ctypes.c_size_t,
    ],
    ggml_backend_buffer_t_ctypes,
)
def ggml_backend_dev_buffer_from_host_ptr(
    device: Union[ggml_backend_dev_t, int],
    ptr: Union[ctypes.c_void_p, int, None],
    size: Union[ctypes.c_size_t, int],
    max_tensor_size: Union[ctypes.c_size_t, int],
    /,
) -> Optional[ggml_backend_buffer_t]:
    ...


# GGML_API bool ggml_backend_dev_supports_op(ggml_backend_dev_t device, const struct ggml_tensor * op);
@ggml_function("ggml_backend_dev_supports_op", [ggml_backend_dev_t_ctypes, ctypes.POINTER(ggml_tensor)], ctypes.c_bool)
def ggml_backend_dev_supports_op(device: Union[ggml_backend_dev_t, int], op: ggml_tensor_p) -> bool:
    ...


# GGML_API bool ggml_backend_dev_supports_buft(ggml_backend_dev_t device, ggml_backend_buffer_type_t buft);
@ggml_function("ggml_backend_dev_supports_buft", [ggml_backend_dev_t_ctypes, ggml_backend_buffer_type_t_ctypes], ctypes.c_bool)
def ggml_backend_dev_supports_buft(device: Union[ggml_backend_dev_t, int], buft: Union[ggml_backend_buffer_type_t, int]) -> bool:
    ...


# GGML_API bool ggml_backend_dev_offload_op(ggml_backend_dev_t device, const struct ggml_tensor * op);
@ggml_function(
    "ggml_backend_dev_offload_op",
    [ggml_backend_dev_t_ctypes, ctypes.POINTER(ggml_tensor)],
    ctypes.c_bool,
)
def ggml_backend_dev_offload_op(
    device: Union[ggml_backend_dev_t, int],
    op: ggml_tensor_p,
) -> bool:
    ...


# GGML_API ggml_backend_t ggml_backend_init_by_name(const char * name, const char * params);
@ggml_function("ggml_backend_init_by_name", [ctypes.c_char_p, ctypes.c_char_p], ggml_backend_t_ctypes)
def ggml_backend_init_by_name(name: bytes, params: Optional[bytes]) -> Optional[ggml_backend_t]:
    ...


# GGML_API ggml_backend_t ggml_backend_init_by_type(enum ggml_backend_dev_type type, const char * params);
@ggml_function("ggml_backend_init_by_type", [ctypes.c_int, ctypes.c_char_p], ggml_backend_t_ctypes)
def ggml_backend_init_by_type(type: Union[ctypes.c_int, int], params: Optional[bytes]) -> Optional[ggml_backend_t]:
    ...


# GGML_API ggml_backend_t ggml_backend_init_best(void);
@ggml_function("ggml_backend_init_best", [], ggml_backend_t_ctypes)
def ggml_backend_init_best() -> Optional[ggml_backend_t]:
    ...


# GGML_API ggml_backend_reg_t ggml_backend_load(const char * path);
@ggml_function("ggml_backend_load", [ctypes.c_char_p], ggml_backend_reg_t_ctypes)
def ggml_backend_load(path: bytes) -> Optional[ggml_backend_reg_t]:
    ...


# GGML_API void ggml_backend_unload(ggml_backend_reg_t reg);
@ggml_function("ggml_backend_unload", [ggml_backend_reg_t_ctypes], None)
def ggml_backend_unload(reg: Union[ggml_backend_reg_t, int], /):
    ...


# GGML_API void ggml_backend_load_all(void);
@ggml_function("ggml_backend_load_all", [], None)
def ggml_backend_load_all():
    ...


# GGML_API void ggml_backend_load_all_from_path(const char * dir_path);
@ggml_function("ggml_backend_load_all_from_path", [ctypes.c_char_p], None)
def ggml_backend_load_all_from_path(dir_path: bytes):
    ...


# //
# // Backend scheduler
# //

# // The backend scheduler allows for multiple backends to be used together
# // Handles compute buffer allocation, assignment of tensors to backends, and copying of tensors between backends
# // The backends are selected based on:
# // - the backend that supports the operation
# // - the location of the pre-allocated tensors (e.g. the weights)
# /*
#   Example usage:

#     // operations that use tensors allocated in a buffer with USAGE_WEIGHTS will be asigned
#     // preferrably to run on the same backend as the buffer
#     ggml_backend_buffer_set_usage(buf_weights, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);

#     sched = ggml_backend_sched_new({backend_gpu, backend_gpu2, backend_cpu}, NULL, num_backends, GGML_DEFAULT_GRAPH_SIZE, false, true);

#     // initialize buffers from a max size graph (optional)
#     reserve_graph = build_graph(sched, max_batch_size);

#     // manually assign nodes to a backend (optional, should not be needed in most cases)
#     struct ggml_tensor * node = ggml_mul_mat(ctx, ...);
#     ggml_backend_sched_set_tensor_backend(sched, node, backend_gpu);

#     ggml_backend_sched_reserve(sched, reserve_graph);

#     // compute
#     graph = build_graph(sched);
#     ggml_backend_sched_graph_compute(sched, graph);

#     // if there are graph inputs:
#     ggml_backend_sched_reset(sched);
#     ggml_backend_sched_alloc_graph(sched, graph);
#     ggml_backend_tensor_set(input_tensor, ...);
#     ggml_backend_sched_graph_compute(sched, graph);
# }
# */

# struct ggml_backend_sched;
# typedef struct ggml_backend_sched * ggml_backend_sched_t;
ggml_backend_sched_t = NewType("ggml_backend_sched_t", int)
ggml_backend_sched_t_ctypes: TypeAlias = ctypes.c_void_p


# // when ask == true, the scheduler wants to know if the user wants to observe this node
# // this allows the scheduler to batch nodes together in order to evaluate them in a single call
# //
# // when ask == false, the scheduler is passing the node tensor to the user for observation
# // if the user returns false, the scheduler will cancel the graph compute
# //
# typedef bool (*ggml_backend_sched_eval_callback)(struct ggml_tensor * t, bool ask, void * user_data);
ggml_backend_sched_eval_callback = ctypes.CFUNCTYPE(
    ctypes.c_bool, ctypes.POINTER(ggml_tensor), ctypes.c_bool, ctypes.c_void_p
)


# // Initialize a backend scheduler
# GGML_API ggml_backend_sched_t ggml_backend_sched_new(ggml_backend_t * backends, ggml_backend_buffer_type_t * bufts, int n_backends, size_t graph_size, bool parallel, bool op_offload);
@ggml_function(
    "ggml_backend_sched_new",
    [
        ctypes.POINTER(ggml_backend_t_ctypes),
        ctypes.POINTER(ggml_backend_buffer_type_t_ctypes),
        ctypes.c_int,
        ctypes.c_size_t,
        ctypes.c_bool,
        ctypes.c_bool,
    ],
    ggml_backend_sched_t_ctypes,
)
def ggml_backend_sched_new(
    backends: "ctypes._Pointer[ggml_backend_t]",  # type: ignore
    bufts: "ctypes._Pointer[ggml_backend_buffer_type_t]",  # type: ignore
    n_backends: Union[ctypes.c_int, int],
    graph_size: Union[ctypes.c_size_t, int],
    parallel: Union[ctypes.c_bool, bool],
    op_offload: Union[ctypes.c_bool, bool],
) -> ggml_backend_sched_t:
    ...


# GGML_API void                 ggml_backend_sched_free(ggml_backend_sched_t sched);
@ggml_function("ggml_backend_sched_free", [ggml_backend_sched_t_ctypes], None)
def ggml_backend_sched_free(sched: ggml_backend_sched_t, /):
    ...


# // Initialize backend buffers from a measure graph
# GGML_API void                 ggml_backend_sched_reserve_size(ggml_backend_sched_t sched, struct ggml_cgraph * measure_graph, size_t * sizes);
@ggml_function(
    "ggml_backend_sched_reserve_size",
    [
        ggml_backend_sched_t_ctypes,
        ctypes.POINTER(ggml_cgraph),
        ctypes.POINTER(ctypes.c_size_t),
    ],
    None,
)
def ggml_backend_sched_reserve_size(
    sched: ggml_backend_sched_t,
    measure_graph: ggml_cgraph_p,
    sizes: CtypesPointer[ctypes.c_size_t],
    /,
):
    """Initialize backend buffers from a measure graph and write per-backend sizes."""
    ...


# GGML_API bool                 ggml_backend_sched_reserve(ggml_backend_sched_t sched, struct ggml_cgraph * measure_graph);
@ggml_function(
    "ggml_backend_sched_reserve",
    [
        ggml_backend_sched_t_ctypes,
        ctypes.POINTER(ggml_cgraph),
    ],
    ctypes.c_bool,
)
def ggml_backend_sched_reserve(
    sched: ggml_backend_sched_t,
    measure_graph: ggml_cgraph_p,
) -> bool:
    """Initialize backend buffers from a measure graph."""
    ...


# GGML_API int                  ggml_backend_sched_get_n_backends(ggml_backend_sched_t sched);
@ggml_function(
    "ggml_backend_sched_get_n_backends", [ggml_backend_sched_t_ctypes], ctypes.c_int
)
def ggml_backend_sched_get_n_backends(
    sched: ggml_backend_sched_t,
) -> int:
    ...


# GGML_API ggml_backend_t       ggml_backend_sched_get_backend(ggml_backend_sched_t sched, int i);
@ggml_function(
    "ggml_backend_sched_get_backend",
    [ggml_backend_sched_t_ctypes, ctypes.c_int],
    ggml_backend_t_ctypes,
)
def ggml_backend_sched_get_backend(
    sched: ggml_backend_sched_t,
    i: Union[ctypes.c_int, int],
) -> Optional[ggml_backend_t]:
    ...


# // Get the number of splits of the last graph
# GGML_API int                  ggml_backend_sched_get_n_splits(ggml_backend_sched_t sched);
@ggml_function(
    "ggml_backend_sched_get_n_splits", [ggml_backend_sched_t_ctypes], ctypes.c_int
)
def ggml_backend_sched_get_n_splits(
    sched: ggml_backend_sched_t,
) -> int:
    """Get the number of splits of the last graph."""
    ...


# GGML_API int                  ggml_backend_sched_get_n_copies(ggml_backend_sched_t sched);
@ggml_function(
    "ggml_backend_sched_get_n_copies", [ggml_backend_sched_t_ctypes], ctypes.c_int
)
def ggml_backend_sched_get_n_copies(
    sched: ggml_backend_sched_t,
) -> int:
    ...


# GGML_API ggml_backend_buffer_type_t ggml_backend_sched_get_buffer_type(ggml_backend_sched_t sched, ggml_backend_t backend);
@ggml_function(
    "ggml_backend_sched_get_buffer_type",
    [
        ggml_backend_sched_t_ctypes,
        ggml_backend_t_ctypes,
    ],
    ggml_backend_buffer_type_t_ctypes,
)
def ggml_backend_sched_get_buffer_type(
    sched: ggml_backend_sched_t,
    backend: Union[ggml_backend_t, int],
) -> Optional[ggml_backend_buffer_type_t]:
    ...


# GGML_API size_t               ggml_backend_sched_get_buffer_size(ggml_backend_sched_t sched, ggml_backend_t backend);
@ggml_function(
    "ggml_backend_sched_get_buffer_size",
    [
        ggml_backend_sched_t_ctypes,
        ggml_backend_t_ctypes,
    ],
    ctypes.c_size_t,
)
def ggml_backend_sched_get_buffer_size(
    sched: ggml_backend_sched_t,
    backend: Union[ggml_backend_t, int],
) -> int:
    ...


# GGML_API void                 ggml_backend_sched_set_tensor_backend(ggml_backend_sched_t sched, struct ggml_tensor * node, ggml_backend_t backend);
@ggml_function(
    "ggml_backend_sched_set_tensor_backend",
    [
        ggml_backend_sched_t_ctypes,
        ctypes.POINTER(ggml_tensor),
        ggml_backend_t_ctypes,
    ],
    None,
)
def ggml_backend_sched_set_tensor_backend(
    sched: ggml_backend_sched_t,
    node: ggml_tensor_p,
    backend: Union[ggml_backend_t, int],
    /,
):
    ...


# GGML_API ggml_backend_t       ggml_backend_sched_get_tensor_backend(ggml_backend_sched_t sched, struct ggml_tensor * node);
@ggml_function(
    "ggml_backend_sched_get_tensor_backend",
    [
        ggml_backend_sched_t_ctypes,
        ctypes.POINTER(ggml_tensor),
    ],
    ggml_backend_t,
)
def ggml_backend_sched_get_tensor_backend(
    sched: ggml_backend_sched_t,
    node: ggml_tensor_p,
) -> Optional[ggml_backend_t]:
    ...


# GGML_API void                 ggml_backend_sched_split_graph(ggml_backend_sched_t sched, struct ggml_cgraph * graph);
@ggml_function(
    "ggml_backend_sched_split_graph",
    [
        ggml_backend_sched_t_ctypes,
        ctypes.POINTER(ggml_cgraph),
    ],
    None,
)
def ggml_backend_sched_split_graph(
    sched: ggml_backend_sched_t,
    graph: ggml_cgraph_p,
    /,
):
    ...


# // Allocate and compute graph on the backend scheduler
# GGML_API bool                 ggml_backend_sched_alloc_graph(ggml_backend_sched_t sched, struct ggml_cgraph * graph);
@ggml_function(
    "ggml_backend_sched_alloc_graph",
    [
        ggml_backend_sched_t_ctypes,
        ctypes.POINTER(ggml_cgraph),
    ],
    ctypes.c_bool,
)
def ggml_backend_sched_alloc_graph(
    sched: ggml_backend_sched_t,
    graph: ggml_cgraph_p,
) -> bool:
    ...


# GGML_API enum ggml_status     ggml_backend_sched_graph_compute(ggml_backend_sched_t sched, struct ggml_cgraph * graph);
@ggml_function(
    "ggml_backend_sched_graph_compute",
    [
        ggml_backend_sched_t_ctypes,
        ctypes.POINTER(ggml_cgraph),
    ],
    ctypes.c_int,
)
def ggml_backend_sched_graph_compute(
    sched: ggml_backend_sched_t,
    graph: ggml_cgraph_p,
) -> int:
    """Allocate and compute graph on the backend scheduler."""
    ...


# GGML_API enum ggml_status     ggml_backend_sched_graph_compute_async(ggml_backend_sched_t sched, struct ggml_cgraph * graph);
@ggml_function(
    "ggml_backend_sched_graph_compute_async",
    [
        ggml_backend_sched_t_ctypes,
        ctypes.POINTER(ggml_cgraph),
    ],
    ctypes.c_int,
)
def ggml_backend_sched_graph_compute_async(
    sched: ggml_backend_sched_t,
    graph: ggml_cgraph_p,
) -> int:
    ...


# GGML_API void                 ggml_backend_sched_synchronize(ggml_backend_sched_t sched);
@ggml_function("ggml_backend_sched_synchronize", [ggml_backend_sched_t_ctypes], None)
def ggml_backend_sched_synchronize(sched: ggml_backend_sched_t, /):
    ...


# // Reset all assignments and allocators - must be called before changing the node backends
# GGML_API void                 ggml_backend_sched_reset(ggml_backend_sched_t sched);
@ggml_function("ggml_backend_sched_reset", [ggml_backend_sched_t_ctypes], None)
def ggml_backend_sched_reset(sched: ggml_backend_sched_t, /):
    """Reset all assignments and allocators - must be called before changing the node backends."""
    ...


# // Set a callback to be called for each resulting node during graph compute
# GGML_API void                 ggml_backend_sched_set_eval_callback(ggml_backend_sched_t sched, ggml_backend_sched_eval_callback callback, void * user_data);
@ggml_function(
    "ggml_backend_sched_set_eval_callback",
    [
        ggml_backend_sched_t_ctypes,
        ggml_backend_sched_eval_callback,  # TODO: this may need to also accept NULL
        ctypes.c_void_p,
    ],
    None,
)
def ggml_backend_sched_set_eval_callback(
    sched: ggml_backend_sched_t,
    callback,  # type: ignore
    user_data: Union[ctypes.c_void_p, int, None],
    /,
):
    ...


# //
# // Meta backend
# //
GGML_BACKEND_META_MAX_DEVICES = 16

GGML_BACKEND_SPLIT_AXIS_0 = 0
GGML_BACKEND_SPLIT_AXIS_1 = 1
GGML_BACKEND_SPLIT_AXIS_2 = 2
GGML_BACKEND_SPLIT_AXIS_3 = 3
GGML_BACKEND_SPLIT_AXIS_MIRRORED = 10
GGML_BACKEND_SPLIT_AXIS_PARTIAL = 11
GGML_BACKEND_SPLIT_AXIS_NONE = 98
GGML_BACKEND_SPLIT_AXIS_UNKNOWN = 99


# struct ggml_backend_meta_split_state {
#     enum ggml_backend_meta_split_axis axis;
#
#     // for tensors with axis >= 0 && axis < GGML_MAX_DIMS:
#     //   - each device has a slice of the tensor along the split axis
#     //   - most tensors have n_segments == 1 and a contiguous slice of the tensor data
#     //   - some tensors have an inhomogenenous data layout along the split axis,
#     //     those tensors are divided into segments which are each individually split across devices
#     //   - ne has one entry per segment and device and that segment repeats nr times,
#     //     in total when accounting for repetitions the segments add up to ggml_tensor::ne for that axis,
#     //     the outer/inner loops are over segments/devices like [seg0_dev0_r0, seg0_dev1_r0, seg0_dev0_r1, seg0_dev1_r1, seg1_dev0_r0, seg1_dev1_r0],
#     //   - for example, a transformer may have a fused QKV matrix rather than 3 matrices, those would be 3 separate segments
#     //     that each need to be split individually across devices so that each device gets a slice of Q, K, and V,
#     //     the Q matrix can be larger than the K and V matrices so this can either be expressed as 3 segments or as 2 segments
#     //     where the segment for K/V repeats twice
#     int64_t  ne[16*GGML_BACKEND_META_MAX_DEVICES];
#     uint32_t nr[16];
#     uint32_t n_segments;
# };
class ggml_backend_meta_split_state(ctypes.Structure):
    _fields_ = [
        ("axis", ctypes.c_int),
        ("ne", ctypes.c_int64 * (16 * GGML_BACKEND_META_MAX_DEVICES)),
        ("nr", ctypes.c_uint32 * 16),
        ("n_segments", ctypes.c_uint32),
    ]


# typedef struct ggml_backend_meta_split_state(*ggml_backend_meta_get_split_state_t)(const struct ggml_tensor * tensor, void * userdata);
#
# ctypes cannot create Python callbacks that return structs by value, so this
# callback is exposed as a raw native function pointer.
ggml_backend_meta_get_split_state_t = ctypes.c_void_p



# GGML_API const char * ggml_backend_meta_split_axis_name(enum ggml_backend_meta_split_axis split_axis);
@ggml_function("ggml_backend_meta_split_axis_name", [ctypes.c_int], ctypes.c_char_p)
def ggml_backend_meta_split_axis_name(split_axis: Union[ctypes.c_int, int]) -> bytes:
    ...


# GGML_API ggml_backend_dev_t ggml_backend_meta_device(ggml_backend_dev_t * devs, size_t n_devs, ggml_backend_meta_get_split_state_t get_split_state, void * get_split_state_ud);
@ggml_function(
    "ggml_backend_meta_device",
    [
        ctypes.POINTER(ggml_backend_dev_t_ctypes),
        ctypes.c_size_t,
        ctypes.c_void_p,
        ctypes.c_void_p,
    ],
    ggml_backend_dev_t_ctypes,
)
def ggml_backend_meta_device(
    devs: CtypesPointer[ggml_backend_dev_t_ctypes],
    n_devs: Union[ctypes.c_size_t, int],
    get_split_state: Union[ctypes.c_void_p, int, None],
    get_split_state_ud: Union[ctypes.c_void_p, int, None],
    /,
) -> Optional[ggml_backend_dev_t]:
    ...


# //
# // Utils
# //


# struct ggml_backend_graph_copy {
#     ggml_backend_buffer_t buffer;
#     struct ggml_context * ctx_allocated;
#     struct ggml_context * ctx_unallocated;
#     struct ggml_cgraph * graph;
# };
class ggml_backend_graph_copy(ctypes.Structure):
    """Structure for ggml_backend_graph_copy.
    
    Attributes:
        buffer: ggml_backend_buffer_t
        ctx_allocated: ggml_context_p
        ctx_unallocated: ggml_context_p
        graph: ctypes.POINTER(ggml_cgraph)"""

    if TYPE_CHECKING:
        buffer: ggml_backend_buffer_t
        ctx_allocated: ggml_context_p
        ctx_unallocated: ggml_context_p
        graph: CtypesPointer[ggml_cgraph]

    _fields_ = [
        ("buffer", ggml_backend_buffer_t_ctypes),
        ("ctx_allocated", ggml_context_p_ctypes),
        ("ctx_unallocated", ggml_context_p_ctypes),
        ("graph", ctypes.POINTER(ggml_cgraph)),
    ]


ggml_backend_graph_copy_t = ggml_backend_graph_copy


# // Copy a graph to a different backend
# GGML_API struct ggml_backend_graph_copy ggml_backend_graph_copy(ggml_backend_t backend, struct ggml_cgraph * graph);
@ggml_function(
    "ggml_backend_graph_copy",
    [ggml_backend_t_ctypes, ctypes.POINTER(ggml_cgraph)],
    ggml_backend_graph_copy_t,
)
def ggml_backend_graph_copy_(
    backend: Union[ggml_backend_t, int],
    graph: ggml_cgraph_p,
) -> ggml_backend_graph_copy_t:
    ...


# GGML_API void                           ggml_backend_graph_copy_free(struct ggml_backend_graph_copy copy);
@ggml_function("ggml_backend_graph_copy_free", [ggml_backend_graph_copy_t], None)
def ggml_backend_graph_copy_free(copy: ggml_backend_graph_copy_t, /):
    ...


# typedef bool (*GGML_CALL ggml_backend_eval_callback)(int node_index, struct ggml_tensor * t1, struct ggml_tensor * t2, void * user_data);
ggml_backend_eval_callback = ctypes.CFUNCTYPE(
    ctypes.c_bool,
    ctypes.c_int,
    ctypes.POINTER(ggml_tensor),
    ctypes.POINTER(ggml_tensor),
    ctypes.c_void_p,
)


# // Compare the output of two backends
# GGML_API bool ggml_backend_compare_graph_backend(ggml_backend_t backend1, ggml_backend_t backend2, struct ggml_cgraph * graph, ggml_backend_eval_callback callback, void * user_data, struct ggml_tensor const * const * test_nodes, size_t num_test_nodes);
@ggml_function(
    "ggml_backend_compare_graph_backend",
    [
        ggml_backend_t_ctypes,
        ggml_backend_t_ctypes,
        ctypes.POINTER(ggml_cgraph),
        ggml_backend_eval_callback,
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.POINTER(ggml_tensor)),
        ctypes.c_size_t,
    ],
    ctypes.c_bool,
)
def ggml_backend_compare_graph_backend(
    backend1: Union[ggml_backend_t, int],
    backend2: Union[ggml_backend_t, int],
    graph: ggml_cgraph_p,
    callback,  # type: ignore
    user_data: Union[ctypes.c_void_p, int, None],
    test_nodes: Optional[CtypesPointer[ggml_tensor_p]],
    num_test_nodes: Union[ctypes.c_size_t, int],
) -> bool:
    ...


# // Tensor initialization
# GGML_API enum ggml_status ggml_backend_tensor_alloc(ggml_backend_buffer_t buffer, struct ggml_tensor * tensor, void * addr);
@ggml_function(
    "ggml_backend_tensor_alloc",
    [
        ggml_backend_buffer_t_ctypes,
        ctypes.POINTER(ggml_tensor),
        ctypes.c_void_p,
    ],
    ctypes.c_int,
)
def ggml_backend_tensor_alloc(
    buffer: Union[ggml_backend_buffer_t, int],
    tensor: ggml_tensor_p,
    addr: Union[ctypes.c_void_p, int, None],
    /,
) -> int:
    ...


# GGML_API enum ggml_status ggml_backend_view_init(struct ggml_tensor * tensor);
@ggml_function(
    "ggml_backend_view_init",
    [
        ctypes.POINTER(ggml_tensor),
    ],
    ctypes.c_int,
)
def ggml_backend_view_init(tensor: ggml_tensor_p, /) -> int:
    ...


#####################################################
# GGML Backend Implementation API
# source: src/ggml-backend-impl.h
#####################################################

# NOTE: This API may be removed in the future from ggml-python

# //
# // Backend buffer
# //

# // buffer type
# typedef void * ggml_backend_buffer_type_context_t;
ggml_backend_buffer_type_context_t = NewType("ggml_backend_buffer_type_context_t", int)
ggml_backend_buffer_type_context_t_ctypes: TypeAlias = ctypes.c_void_p

# struct ggml_backend_buffer_type_i {
#     const char *          (*GGML_CALL get_name)        (ggml_backend_buffer_type_t buft);
#     ggml_backend_buffer_t (*GGML_CALL alloc_buffer)    (ggml_backend_buffer_type_t buft, size_t size);
#     size_t                (*GGML_CALL get_alignment)   (ggml_backend_buffer_type_t buft); // tensor alignment
#     size_t                (*GGML_CALL get_max_size)    (ggml_backend_buffer_type_t buft); // allocation max size
#     size_t                (*GGML_CALL get_alloc_size)  (ggml_backend_buffer_type_t buft, const struct ggml_tensor * tensor); // data size needed to allocate the tensor, including padding
#     bool                  (*GGML_CALL supports_backend)(ggml_backend_buffer_type_t buft, ggml_backend_t backend); // check if the buffer type is usable by the backend
#     // check if tensor data is in host memory
#     // should be equivalent to supports_backend(buft, ggml_backend_cpu_init())
#     bool                  (*GGML_CALL is_host)         (ggml_backend_buffer_type_t buft);
# };
ggml_backend_buffer_type_i_get_name = ctypes.CFUNCTYPE(
    ctypes.c_char_p, ggml_backend_buffer_type_t_ctypes
)
ggml_backend_buffer_i_alloc_buffer = ctypes.CFUNCTYPE(
    ggml_backend_buffer_t_ctypes, ggml_backend_buffer_type_t_ctypes, ctypes.c_size_t
)
ggml_backend_buffer_i_get_alignment = ctypes.CFUNCTYPE(
    ctypes.c_size_t, ggml_backend_buffer_type_t_ctypes
)
ggml_backend_buffer_i_get_max_size = ctypes.CFUNCTYPE(
    ctypes.c_size_t, ggml_backend_buffer_type_t_ctypes
)
ggml_backend_buffer_i_get_alloc_size = ctypes.CFUNCTYPE(
    ctypes.c_size_t, ggml_backend_buffer_type_t_ctypes, ctypes.POINTER(ggml_tensor)
)
ggml_backend_buffer_i_supports_backend = ctypes.CFUNCTYPE(
    ctypes.c_bool, ggml_backend_buffer_type_t_ctypes, ggml_backend_t_ctypes
)
ggml_backend_buffer_i_is_host = ctypes.CFUNCTYPE(
    ctypes.c_bool, ggml_backend_buffer_type_t_ctypes
)


class ggml_backend_buffer_type_i(ctypes.Structure):
    _fields_ = [
        ("get_name", ggml_backend_buffer_type_i_get_name),
        ("alloc_buffer", ggml_backend_buffer_i_alloc_buffer),
        ("get_alignment", ggml_backend_buffer_i_get_alignment),
        ("get_max_size", ggml_backend_buffer_i_get_max_size),
        ("get_alloc_size", ggml_backend_buffer_i_get_alloc_size),
        ("supports_backend", ggml_backend_buffer_i_supports_backend),
        ("is_host", ggml_backend_buffer_i_is_host),
    ]


# struct ggml_backend_buffer_type {
#     struct ggml_backend_buffer_type_i  iface;
#     ggml_backend_buffer_type_context_t context;
# };
class ggml_backend_buffer_type(ctypes.Structure):
    _fields_ = [
        ("iface", ggml_backend_buffer_type_i),
        ("context", ggml_backend_buffer_type_context_t_ctypes),
    ]


# typedef void * ggml_backend_buffer_context_t;
ggml_backend_buffer_context_t = NewType("ggml_backend_buffer_context_t", int)
ggml_backend_buffer_context_t_ctypes: TypeAlias = ctypes.c_void_p


# struct ggml_backend_buffer_i {
#     const char * (*GGML_CALL get_name)   (ggml_backend_buffer_t buffer);
#     void         (*GGML_CALL free_buffer)(ggml_backend_buffer_t buffer);
#     void *       (*GGML_CALL get_base)   (ggml_backend_buffer_t buffer);
#     void         (*GGML_CALL init_tensor)(ggml_backend_buffer_t buffer, struct ggml_tensor * tensor);
#     void         (*GGML_CALL set_tensor) (ggml_backend_buffer_t buffer,       struct ggml_tensor * tensor, const void * data, size_t offset, size_t size);
#     void         (*GGML_CALL get_tensor) (ggml_backend_buffer_t buffer, const struct ggml_tensor * tensor,       void * data, size_t offset, size_t size);
#     bool         (*GGML_CALL cpy_tensor) (ggml_backend_buffer_t buffer, const struct ggml_tensor * src, struct ggml_tensor * dst); // dst is in the buffer, src may be in any buffer
#     void         (*GGML_CALL clear)      (ggml_backend_buffer_t buffer, uint8_t value);
#     void         (*GGML_CALL reset)      (ggml_backend_buffer_t buffer); // reset any internal state due to tensor initialization, such as tensor extras
# };
ggml_backend_buffer_i_get_name = ctypes.CFUNCTYPE(
    ctypes.c_char_p, ggml_backend_buffer_t_ctypes
)
ggml_backend_buffer_i_free_buffer = ctypes.CFUNCTYPE(None, ggml_backend_buffer_t_ctypes)
ggml_backend_buffer_i_get_base = ctypes.CFUNCTYPE(
    ctypes.c_void_p, ggml_backend_buffer_t_ctypes
)
ggml_backend_buffer_i_init_tensor = ctypes.CFUNCTYPE(
    None, ggml_backend_buffer_t_ctypes, ctypes.POINTER(ggml_tensor)
)
ggml_backend_buffer_i_set_tensor = ctypes.CFUNCTYPE(
    None,
    ggml_backend_buffer_t_ctypes,
    ctypes.POINTER(ggml_tensor),
    ctypes.c_void_p,
    ctypes.c_size_t,
    ctypes.c_size_t,
)
ggml_backend_buffer_i_get_tensor = ctypes.CFUNCTYPE(
    None,
    ggml_backend_buffer_t_ctypes,
    ctypes.POINTER(ggml_tensor),
    ctypes.c_void_p,
    ctypes.c_size_t,
    ctypes.c_size_t,
)
ggml_backend_buffer_i_cpy_tensor = ctypes.CFUNCTYPE(
    ctypes.c_bool,
    ggml_backend_buffer_t_ctypes,
    ctypes.POINTER(ggml_tensor),
    ctypes.POINTER(ggml_tensor),
)
ggml_backend_buffer_i_clear = ctypes.CFUNCTYPE(
    None, ggml_backend_buffer_t_ctypes, ctypes.c_uint8
)
ggml_backend_buffer_i_reset = ctypes.CFUNCTYPE(None, ggml_backend_buffer_t_ctypes)


class ggml_backend_buffer_i(ctypes.Structure):
    _fields_ = [
        ("get_name", ggml_backend_buffer_i_get_name),
        ("free_buffer", ggml_backend_buffer_i_free_buffer),
        ("get_base", ggml_backend_buffer_i_get_base),
        ("init_tensor", ggml_backend_buffer_i_init_tensor),
        ("set_tensor", ggml_backend_buffer_i_set_tensor),
        ("get_tensor", ggml_backend_buffer_i_get_tensor),
        ("cpy_tensor", ggml_backend_buffer_i_cpy_tensor),
        ("clear", ggml_backend_buffer_i_clear),
        ("reset", ggml_backend_buffer_i_reset),
    ]


# struct ggml_backend_buffer {
#     struct ggml_backend_buffer_i  iface;
#     ggml_backend_buffer_type_t    buft;
#     ggml_backend_buffer_context_t context;
#     size_t size;
#     enum ggml_backend_buffer_usage usage;
# };
class ggml_backend_buffer(ctypes.Structure):
    _fields_ = [
        ("iface", ggml_backend_buffer_i),
        ("buft", ggml_backend_buffer_type_t_ctypes),
        ("context", ggml_backend_buffer_context_t_ctypes),
        ("size", ctypes.c_size_t),
    ]


# GGML_CALL ggml_backend_buffer_t ggml_backend_buffer_init(
#                ggml_backend_buffer_type_t      buft,
#         struct ggml_backend_buffer_i           iface,
#                ggml_backend_buffer_context_t   context,
#                size_t                          size);
@ggml_function(
    "ggml_backend_buffer_init",
    [
        ggml_backend_buffer_type_t_ctypes,
        ggml_backend_buffer_i,
        ggml_backend_buffer_context_t_ctypes,
        ctypes.c_size_t,
    ],
    ggml_backend_buffer_t_ctypes,
)
def ggml_backend_buffer_init(
    buft: Union[ggml_backend_buffer_type_t, int],
    iface: ggml_backend_buffer_i,
    context: ggml_backend_buffer_context_t,
    size: Union[ctypes.c_size_t, int],
) -> Optional[ggml_backend_buffer_t]:
    ...


# // do not use directly, use ggml_backend_tensor_copy instead
# bool ggml_backend_buffer_copy_tensor(const struct ggml_tensor * src, struct ggml_tensor * dst);
@ggml_function(
    "ggml_backend_buffer_copy_tensor",
    [
        ctypes.POINTER(ggml_tensor),
        ctypes.POINTER(ggml_tensor),
    ],
    ctypes.c_bool,
)
def ggml_backend_buffer_copy_tensor(
    src: ggml_tensor_p,
    dst: ggml_tensor_p,
) -> bool:
    ...


# // buffer that contains a collection of buffers
# GGML_CALL ggml_backend_buffer_t ggml_backend_multi_buffer_alloc_buffer(ggml_backend_buffer_t * buffers, size_t n_buffers);
# GGML_CALL bool                  ggml_backend_buffer_is_multi_buffer(ggml_backend_buffer_t buffer);
# GGML_CALL void                  ggml_backend_multi_buffer_set_usage(ggml_backend_buffer_t buffer, enum ggml_backend_buffer_usage usage);
@ggml_function(
    "ggml_backend_multi_buffer_alloc_buffer",
    [
        ctypes.POINTER(ggml_backend_buffer_t_ctypes),
        ctypes.c_size_t,
    ],
    ggml_backend_buffer_t_ctypes,
)
def ggml_backend_multi_buffer_alloc_buffer(
    buffers: "ctypes._Pointer(ggml_backend_buffer_t)",  # type: ignore
    n_buffers: Union[ctypes.c_size_t, int],
) -> Optional[ggml_backend_buffer_t]:
    ...


# //
# // Backend
# //

# typedef void * ggml_backend_context_t;
ggml_backend_context_t = NewType("ggml_backend_context_t", int)
ggml_backend_context_t_ctypes: TypeAlias = ctypes.c_void_p


# struct ggml_backend_i {
#     const char * (*GGML_CALL get_name)(ggml_backend_t backend);

#     void (*GGML_CALL free)(ggml_backend_t backend);

#     // buffer allocation
#     ggml_backend_buffer_type_t (*GGML_CALL get_default_buffer_type)(ggml_backend_t backend);

#     // (optional) asynchronous tensor data access
#     void (*GGML_CALL set_tensor_async)(ggml_backend_t backend,       struct ggml_tensor * tensor, const void * data, size_t offset, size_t size);
#     void (*GGML_CALL get_tensor_async)(ggml_backend_t backend, const struct ggml_tensor * tensor,       void * data, size_t offset, size_t size);
#     bool (*GGML_CALL cpy_tensor_async)(ggml_backend_t backend_src, ggml_backend_t backend_dst, const struct ggml_tensor * src, struct ggml_tensor * dst);

#     // (optional) complete all pending operations
#     void (*GGML_CALL synchronize)(ggml_backend_t backend);

#     // compute graph with a plan (not used currently)
#     ggml_backend_graph_plan_t (*GGML_CALL graph_plan_create) (ggml_backend_t backend, const struct ggml_cgraph * cgraph);
#     void                      (*GGML_CALL graph_plan_free)   (ggml_backend_t backend, ggml_backend_graph_plan_t plan);
#     void                      (*GGML_CALL graph_plan_compute)(ggml_backend_t backend, ggml_backend_graph_plan_t plan);

#     // compute graph without a plan (async)
#     bool (*GGML_CALL graph_compute)(ggml_backend_t backend, struct ggml_cgraph * cgraph);

#     // check if the backend supports an operation
#     bool (*GGML_CALL supports_op)(ggml_backend_t backend, const struct ggml_tensor * op);

#     // check if the backend wants to run an operation, even if the weights are allocated in a CPU buffer
#     // these should be expensive operations with large batch sizes that may benefit from running on this backend
#     // even if the weight has to be copied from the CPU temporarily
#     bool (*GGML_CALL offload_op)(ggml_backend_t backend, const struct ggml_tensor * op);

#     // (optional) event synchronization
#     ggml_backend_event_t (*GGML_CALL event_new)         (ggml_backend_t backend);
#     void                 (*GGML_CALL event_free)        (ggml_backend_event_t event);
#     void                 (*GGML_CALL event_record)      (ggml_backend_event_t event);
#     void                 (*GGML_CALL event_wait)        (ggml_backend_t backend, ggml_backend_event_t event);
#     void                 (*GGML_CALL event_synchronize) (ggml_backend_event_t event);
# };
ggml_backend_i_get_name = ctypes.CFUNCTYPE(ctypes.c_char_p, ggml_backend_t_ctypes)
ggml_backend_i_free = ctypes.CFUNCTYPE(None, ggml_backend_t_ctypes)
ggml_backend_i_get_default_buffer_type = ctypes.CFUNCTYPE(
    ggml_backend_buffer_type_t_ctypes, ggml_backend_t_ctypes
)
ggml_backend_i_set_tensor_async = ctypes.CFUNCTYPE(
    None,
    ggml_backend_t_ctypes,
    ctypes.POINTER(ggml_tensor),
    ctypes.c_void_p,
    ctypes.c_size_t,
    ctypes.c_size_t,
)
ggml_backend_i_get_tensor_async = ctypes.CFUNCTYPE(
    None,
    ggml_backend_t_ctypes,
    ctypes.POINTER(ggml_tensor),
    ctypes.c_void_p,
    ctypes.c_size_t,
    ctypes.c_size_t,
)
ggml_backend_i_cpy_tensor_async = ctypes.CFUNCTYPE(
    ctypes.c_bool,
    ggml_backend_t_ctypes,
    ggml_backend_t_ctypes,
    ctypes.POINTER(ggml_tensor),
    ctypes.POINTER(ggml_tensor),
)
ggml_backend_i_synchronize = ctypes.CFUNCTYPE(None, ggml_backend_t_ctypes)
ggml_backend_i_graph_plan_create = ctypes.CFUNCTYPE(
    ggml_backend_graph_plan_t_ctypes, ggml_backend_t_ctypes, ctypes.POINTER(ggml_cgraph)
)
ggml_backend_i_graph_plan_free = ctypes.CFUNCTYPE(
    None, ggml_backend_t_ctypes, ggml_backend_graph_plan_t_ctypes
)
ggml_backend_i_graph_plan_compute = ctypes.CFUNCTYPE(
    None, ggml_backend_t_ctypes, ggml_backend_graph_plan_t_ctypes
)
ggml_backend_i_graph_compute = ctypes.CFUNCTYPE(
    ctypes.c_bool, ggml_backend_t_ctypes, ctypes.POINTER(ggml_cgraph)
)
ggml_backend_i_supports_op = ctypes.CFUNCTYPE(
    ctypes.c_bool, ggml_backend_t_ctypes, ctypes.POINTER(ggml_tensor)
)
ggml_backend_i_offload_op = ctypes.CFUNCTYPE(
    ctypes.c_bool, ggml_backend_t_ctypes, ctypes.POINTER(ggml_tensor)
)
ggml_backend_i_event_new = ctypes.CFUNCTYPE(ggml_backend_event_t_ctypes, ggml_backend_t_ctypes)
ggml_backend_i_event_free = ctypes.CFUNCTYPE(None, ggml_backend_event_t_ctypes)
ggml_backend_i_event_record = ctypes.CFUNCTYPE(None, ggml_backend_event_t_ctypes)
ggml_backend_i_event_wait = ctypes.CFUNCTYPE(
    None, ggml_backend_t_ctypes, ggml_backend_event_t_ctypes
)
ggml_backend_i_event_synchronize = ctypes.CFUNCTYPE(None, ggml_backend_event_t_ctypes)


class ggml_backend_i(ctypes.Structure):
    _fields_ = [
        ("get_name", ggml_backend_i_get_name),
        ("free", ggml_backend_i_free),
        ("get_default_buffer_type", ggml_backend_i_get_default_buffer_type),
        ("set_tensor_async", ggml_backend_i_set_tensor_async),
        ("get_tensor_async", ggml_backend_i_get_tensor_async),
        ("cpy_tensor_async", ggml_backend_i_cpy_tensor_async),
        ("synchronize", ggml_backend_i_synchronize),
        ("graph_plan_create", ggml_backend_i_graph_plan_create),
        ("graph_plan_free", ggml_backend_i_graph_plan_free),
        ("graph_plan_compute", ggml_backend_i_graph_plan_compute),
        ("graph_compute", ggml_backend_i_graph_compute),
        ("supports_op", ggml_backend_i_supports_op),
        ("offload_op", ggml_backend_i_offload_op),
        ("event_new", ggml_backend_i_event_new),
        ("event_free", ggml_backend_i_event_free),
        ("event_record", ggml_backend_i_event_record),
        ("event_wait", ggml_backend_i_event_wait),
        ("event_synchronize", ggml_backend_i_event_synchronize),
    ]


# struct ggml_backend {
#     ggml_guid_t guid;

#     struct ggml_backend_i iface;

#     ggml_backend_context_t context;
# };
class ggml_backend(ctypes.Structure):
    _fields_ = [
        ("guid", ggml_guid_t_ctypes),
        ("iface", ggml_backend_i),
        ("context", ggml_backend_context_t_ctypes),
    ]


# struct ggml_backend_event {
#     ggml_backend_t backend;
#     void * context;
# };
class ggml_backend_event(ctypes.Structure):
    _fields_ = [
        ("backend", ggml_backend_t_ctypes),
        ("context", ctypes.c_void_p),
    ]


# //
# // Backend registry
# //

# GGML_API void ggml_backend_register(ggml_backend_reg_t reg);
@ggml_function("ggml_backend_register", [ggml_backend_reg_t_ctypes], None)
def ggml_backend_register(reg: Union[ggml_backend_reg_t, int], /):
    ...


#####################################################
# GGML BLAS API
# source: src/ggml-blas.h
#####################################################


GGML_USE_BLAS = hasattr(lib, "ggml_backend_blas_init")


# GGML_API GGML_CALL ggml_backend_t ggml_backend_blas_init(void);
@ggml_function(
    "ggml_backend_blas_init",
    [],
    ggml_backend_t_ctypes,
    enabled=GGML_USE_BLAS,
)
def ggml_backend_blas_init() -> Optional[ggml_backend_t]:
    ...


# GGML_API GGML_CALL bool ggml_backend_is_blas(ggml_backend_t backend);
@ggml_function(
    "ggml_backend_is_blas",
    [ggml_backend_t_ctypes],
    ctypes.c_bool,
    enabled=GGML_USE_BLAS,
)
def ggml_backend_is_blas(backend: Union[ggml_backend_t, int], /) -> bool:
    ...


# GGML_API GGML_CALL void ggml_backend_blas_set_n_threads(ggml_backend_t backend_blas, int n_threads);
@ggml_function(
    "ggml_backend_blas_set_n_threads",
    [ggml_backend_t_ctypes, ctypes.c_int],
    None,
    enabled=GGML_USE_BLAS,
)
def ggml_backend_blas_set_n_threads(
    backend_blas: Union[ggml_backend_t, int],
    n_threads: Union[ctypes.c_int, int],
    /,
) -> None:
    ...


# GGML_API GGML_CALL ggml_backend_reg_t ggml_backend_blas_reg(void);
@ggml_function(
    "ggml_backend_blas_reg",
    [],
    ggml_backend_reg_t_ctypes,
    enabled=GGML_USE_BLAS,
)
def ggml_backend_blas_reg() -> Optional[ggml_backend_reg_t]:
    ...


#####################################################
# GGML CANN API
# source: src/ggml-cann.h
#####################################################


GGML_USE_CANN = hasattr(lib, "ggml_backend_cann_init")


GGML_CANN_MAX_DEVICES = 16


# GGML_API GGML_CALL ggml_backend_reg_t ggml_backend_cann_reg(void);
@ggml_function(
    "ggml_backend_cann_reg",
    [],
    ggml_backend_reg_t_ctypes,
    enabled=GGML_USE_CANN,
)
def ggml_backend_cann_reg() -> Optional[ggml_backend_reg_t]:
    ...


# GGML_API GGML_CALL ggml_backend_t ggml_backend_cann_init(int32_t device);
@ggml_function(
    "ggml_backend_cann_init",
    [ctypes.c_int32],
    ggml_backend_t_ctypes,
    enabled=GGML_USE_CANN,
)
def ggml_backend_cann_init(
    device: Union[ctypes.c_int32, int],
    /,
) -> Optional[ggml_backend_t]:
    ...


# GGML_API GGML_CALL bool ggml_backend_is_cann(ggml_backend_t backend);
@ggml_function(
    "ggml_backend_is_cann",
    [ggml_backend_t_ctypes],
    ctypes.c_bool,
    enabled=GGML_USE_CANN,
)
def ggml_backend_is_cann(backend: Union[ggml_backend_t, int], /) -> bool:
    ...


# GGML_API GGML_CALL ggml_backend_buffer_type_t ggml_backend_cann_buffer_type(int32_t device);
@ggml_function(
    "ggml_backend_cann_buffer_type",
    [ctypes.c_int32],
    ggml_backend_buffer_type_t_ctypes,
    enabled=GGML_USE_CANN,
)
def ggml_backend_cann_buffer_type(
    device: Union[ctypes.c_int32, int],
    /,
) -> Optional[ggml_backend_buffer_type_t]:
    ...


# GGML_API GGML_CALL int32_t ggml_backend_cann_get_device_count(void);
@ggml_function(
    "ggml_backend_cann_get_device_count",
    [],
    ctypes.c_int32,
    enabled=GGML_USE_CANN,
)
def ggml_backend_cann_get_device_count() -> int:
    ...


# GGML_API GGML_CALL ggml_backend_buffer_type_t ggml_backend_cann_host_buffer_type(void);
@ggml_function(
    "ggml_backend_cann_host_buffer_type",
    [],
    ggml_backend_buffer_type_t_ctypes,
    enabled=GGML_USE_CANN,
)
def ggml_backend_cann_host_buffer_type() -> Optional[ggml_backend_buffer_type_t]:
    ...


# GGML_API GGML_CALL void ggml_backend_cann_get_device_description(int32_t device, char * description, size_t description_size);
@ggml_function(
    "ggml_backend_cann_get_device_description",
    [
        ctypes.c_int32,
        ctypes.c_char_p,
        ctypes.c_size_t,
    ],
    None,
    enabled=GGML_USE_CANN,
)
def ggml_backend_cann_get_device_description(
    device: Union[ctypes.c_int32, int],
    description: ctypes.c_char_p,
    description_size: Union[ctypes.c_size_t, int],
    /,
) -> None:
    ...


# GGML_API GGML_CALL void ggml_backend_cann_get_device_memory(int32_t device, size_t * free, size_t * total);
@ggml_function(
    "ggml_backend_cann_get_device_memory",
    [
        ctypes.c_int32,
        ctypes.POINTER(ctypes.c_size_t),
        ctypes.POINTER(ctypes.c_size_t),
    ],
    None,
    enabled=GGML_USE_CANN,
)
def ggml_backend_cann_get_device_memory(
    device: Union[ctypes.c_int32, int],
    free: CtypesPointer[ctypes.c_size_t],
    total: CtypesPointer[ctypes.c_size_t],
    /,
) -> None:
    ...


#####################################################
# GGML SYCL API
# source: src/ggml-sycl.h
#####################################################


GGML_USE_SYCL_BACKEND = hasattr(lib, "ggml_backend_sycl_init")
GGML_SYCL_NAME = "SYCL"
GGML_SYCL_MAX_DEVICES = 48


# GGML_API GGML_CALL ggml_backend_t ggml_backend_sycl_init(int device);
@ggml_function(
    "ggml_backend_sycl_init",
    [ctypes.c_int],
    ggml_backend_t_ctypes,
    enabled=GGML_USE_SYCL_BACKEND,
)
def ggml_backend_sycl_init(device: Union[ctypes.c_int, int], /) -> Optional[ggml_backend_t]:
    ...


# GGML_API GGML_CALL bool ggml_backend_is_sycl(ggml_backend_t backend);
@ggml_function(
    "ggml_backend_is_sycl",
    [ggml_backend_t_ctypes],
    ctypes.c_bool,
    enabled=GGML_USE_SYCL_BACKEND,
)
def ggml_backend_is_sycl(backend: Union[ggml_backend_t, int], /) -> bool:
    ...


# GGML_API GGML_CALL ggml_backend_buffer_type_t ggml_backend_sycl_buffer_type(int device);
@ggml_function(
    "ggml_backend_sycl_buffer_type",
    [ctypes.c_int],
    ggml_backend_buffer_type_t_ctypes,
    enabled=GGML_USE_SYCL_BACKEND,
)
def ggml_backend_sycl_buffer_type(
    device: Union[ctypes.c_int, int],
    /,
) -> Optional[ggml_backend_buffer_type_t]:
    ...


# GGML_API GGML_CALL ggml_backend_buffer_type_t ggml_backend_sycl_split_buffer_type(const float * tensor_split);
@ggml_function(
    "ggml_backend_sycl_split_buffer_type",
    [ctypes.POINTER(ctypes.c_float)],
    ggml_backend_buffer_type_t_ctypes,
    enabled=GGML_USE_SYCL_BACKEND,
)
def ggml_backend_sycl_split_buffer_type(
    tensor_split: CtypesArray[ctypes.c_float],
    /,
) -> Optional[ggml_backend_buffer_type_t]:
    ...


# GGML_API GGML_CALL ggml_backend_buffer_type_t ggml_backend_sycl_host_buffer_type(void);
@ggml_function(
    "ggml_backend_sycl_host_buffer_type",
    [],
    ggml_backend_buffer_type_t_ctypes,
    enabled=GGML_USE_SYCL_BACKEND,
)
def ggml_backend_sycl_host_buffer_type() -> Optional[ggml_backend_buffer_type_t]:
    ...


# GGML_API GGML_CALL void ggml_backend_sycl_print_sycl_devices(void);
@ggml_function(
    "ggml_backend_sycl_print_sycl_devices",
    [],
    None,
    enabled=GGML_USE_SYCL_BACKEND,
)
def ggml_backend_sycl_print_sycl_devices() -> None:
    ...


# GGML_API GGML_CALL void ggml_backend_sycl_get_gpu_list(int * id_list, int max_len);
@ggml_function(
    "ggml_backend_sycl_get_gpu_list",
    [ctypes.POINTER(ctypes.c_int), ctypes.c_int],
    None,
    enabled=GGML_USE_SYCL_BACKEND,
)
def ggml_backend_sycl_get_gpu_list(
    id_list: CtypesPointer[ctypes.c_int],
    max_len: Union[ctypes.c_int, int],
    /,
) -> None:
    ...


# GGML_API GGML_CALL void ggml_backend_sycl_get_device_description(int device, char * description, size_t description_size);
@ggml_function(
    "ggml_backend_sycl_get_device_description",
    [
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_size_t,
    ],
    None,
    enabled=GGML_USE_SYCL_BACKEND,
)
def ggml_backend_sycl_get_device_description(
    device: Union[ctypes.c_int, int],
    description: ctypes.c_char_p,
    description_size: Union[ctypes.c_size_t, int],
    /,
) -> None:
    ...


# GGML_API GGML_CALL int ggml_backend_sycl_get_device_count(void);
@ggml_function(
    "ggml_backend_sycl_get_device_count",
    [],
    ctypes.c_int,
    enabled=GGML_USE_SYCL_BACKEND,
)
def ggml_backend_sycl_get_device_count() -> int:
    ...


# GGML_API GGML_CALL void ggml_backend_sycl_get_device_memory(int device, size_t * free, size_t * total);
@ggml_function(
    "ggml_backend_sycl_get_device_memory",
    [
        ctypes.c_int,
        ctypes.POINTER(ctypes.c_size_t),
        ctypes.POINTER(ctypes.c_size_t),
    ],
    None,
    enabled=GGML_USE_SYCL_BACKEND,
)
def ggml_backend_sycl_get_device_memory(
    device: Union[ctypes.c_int, int],
    free: CtypesPointer[ctypes.c_size_t],
    total: CtypesPointer[ctypes.c_size_t],
    /,
) -> None:
    ...


# GGML_API GGML_CALL ggml_backend_reg_t ggml_backend_sycl_reg(void);
@ggml_function(
    "ggml_backend_sycl_reg",
    [],
    ggml_backend_reg_t_ctypes,
    enabled=GGML_USE_SYCL_BACKEND,
)
def ggml_backend_sycl_reg() -> Optional[ggml_backend_reg_t]:
    ...


#####################################################
# GGML OpenVINO API
# source: src/ggml-openvino.h
#####################################################


GGML_USE_OPENVINO = hasattr(lib, "ggml_backend_openvino_init")
GGML_OPENVINO_NAME = "OPENVINO"


# GGML_API GGML_CALL ggml_backend_t ggml_backend_openvino_init(int device);
@ggml_function(
    "ggml_backend_openvino_init",
    [ctypes.c_int],
    ggml_backend_t_ctypes,
    enabled=GGML_USE_OPENVINO,
)
def ggml_backend_openvino_init(device: Union[ctypes.c_int, int], /) -> Optional[ggml_backend_t]:
    ...


# GGML_API GGML_CALL bool ggml_backend_is_openvino(ggml_backend_t backend);
@ggml_function(
    "ggml_backend_is_openvino",
    [ggml_backend_t_ctypes],
    ctypes.c_bool,
    enabled=GGML_USE_OPENVINO,
)
def ggml_backend_is_openvino(backend: Union[ggml_backend_t, int], /) -> bool:
    ...


# GGML_API GGML_CALL bool ggml_backend_buffer_is_openvino(ggml_backend_buffer_t buffer);
@ggml_function(
    "ggml_backend_buffer_is_openvino",
    [ggml_backend_buffer_t_ctypes],
    ctypes.c_bool,
    enabled=GGML_USE_OPENVINO,
)
def ggml_backend_buffer_is_openvino(
    buffer: Union[ggml_backend_buffer_t, int],
    /,
) -> bool:
    ...


# GGML_API GGML_CALL bool ggml_backend_buft_is_openvino(ggml_backend_buffer_type_t buft);
@ggml_function(
    "ggml_backend_buft_is_openvino",
    [ggml_backend_buffer_type_t_ctypes],
    ctypes.c_bool,
    enabled=GGML_USE_OPENVINO,
)
def ggml_backend_buft_is_openvino(
    buft: Union[ggml_backend_buffer_type_t, int],
    /,
) -> bool:
    ...


# GGML_API GGML_CALL bool ggml_backend_buft_is_openvino_host(ggml_backend_buffer_type_t buft);
@ggml_function(
    "ggml_backend_buft_is_openvino_host",
    [ggml_backend_buffer_type_t_ctypes],
    ctypes.c_bool,
    enabled=GGML_USE_OPENVINO,
)
def ggml_backend_buft_is_openvino_host(
    buft: Union[ggml_backend_buffer_type_t, int],
    /,
) -> bool:
    ...


# GGML_API GGML_CALL size_t ggml_backend_openvino_buffer_get_ctx_id(ggml_backend_buffer_t buffer);
@ggml_function(
    "ggml_backend_openvino_buffer_get_ctx_id",
    [ggml_backend_buffer_t_ctypes],
    ctypes.c_size_t,
    enabled=GGML_USE_OPENVINO,
)
def ggml_backend_openvino_buffer_get_ctx_id(
    buffer: Union[ggml_backend_buffer_t, int],
    /,
) -> int:
    ...


# GGML_API GGML_CALL ggml_backend_buffer_type_t ggml_backend_openvino_buffer_type(int device);
@ggml_function(
    "ggml_backend_openvino_buffer_type",
    [ctypes.c_int],
    ggml_backend_buffer_type_t_ctypes,
    enabled=GGML_USE_OPENVINO,
)
def ggml_backend_openvino_buffer_type(
    device: Union[ctypes.c_int, int],
    /,
) -> Optional[ggml_backend_buffer_type_t]:
    ...


# GGML_API GGML_CALL ggml_backend_buffer_type_t ggml_backend_openvino_host_buffer_type(int device);
@ggml_function(
    "ggml_backend_openvino_host_buffer_type",
    [ctypes.c_int],
    ggml_backend_buffer_type_t_ctypes,
    enabled=GGML_USE_OPENVINO,
)
def ggml_backend_openvino_host_buffer_type(
    device: Union[ctypes.c_int, int],
    /,
) -> Optional[ggml_backend_buffer_type_t]:
    ...


# GGML_API GGML_CALL int ggml_backend_openvino_get_device_count(void);
@ggml_function(
    "ggml_backend_openvino_get_device_count",
    [],
    ctypes.c_int,
    enabled=GGML_USE_OPENVINO,
)
def ggml_backend_openvino_get_device_count() -> int:
    ...


# GGML_API GGML_CALL ggml_backend_reg_t ggml_backend_openvino_reg(void);
@ggml_function(
    "ggml_backend_openvino_reg",
    [],
    ggml_backend_reg_t_ctypes,
    enabled=GGML_USE_OPENVINO,
)
def ggml_backend_openvino_reg() -> Optional[ggml_backend_reg_t]:
    ...


#####################################################
# GGML Hexagon API
# source: src/ggml-hexagon.h
#####################################################


GGML_USE_HEXAGON = hasattr(lib, "ggml_backend_hexagon_init")


# GGML_API GGML_CALL ggml_backend_t ggml_backend_hexagon_init(void);
@ggml_function(
    "ggml_backend_hexagon_init",
    [],
    ggml_backend_t_ctypes,
    enabled=GGML_USE_HEXAGON,
)
def ggml_backend_hexagon_init() -> Optional[ggml_backend_t]:
    ...


# GGML_API GGML_CALL bool ggml_backend_is_hexagon(ggml_backend_t backend);
@ggml_function(
    "ggml_backend_is_hexagon",
    [ggml_backend_t_ctypes],
    ctypes.c_bool,
    enabled=GGML_USE_HEXAGON,
)
def ggml_backend_is_hexagon(backend: Union[ggml_backend_t, int], /) -> bool:
    ...


# GGML_API GGML_CALL ggml_backend_reg_t ggml_backend_hexagon_reg(void);
@ggml_function(
    "ggml_backend_hexagon_reg",
    [],
    ggml_backend_reg_t_ctypes,
    enabled=GGML_USE_HEXAGON,
)
def ggml_backend_hexagon_reg() -> Optional[ggml_backend_reg_t]:
    ...


#####################################################
# GGML WebGPU API
# source: src/ggml-webgpu.h
#####################################################


GGML_USE_WEBGPU = hasattr(lib, "ggml_backend_webgpu_init")


# GGML_API GGML_CALL ggml_backend_t ggml_backend_webgpu_init(void);
@ggml_function(
    "ggml_backend_webgpu_init",
    [],
    ggml_backend_t_ctypes,
    enabled=GGML_USE_WEBGPU,
)
def ggml_backend_webgpu_init() -> Optional[ggml_backend_t]:
    ...


# GGML_API GGML_CALL ggml_backend_reg_t ggml_backend_webgpu_reg(void);
@ggml_function(
    "ggml_backend_webgpu_reg",
    [],
    ggml_backend_reg_t_ctypes,
    enabled=GGML_USE_WEBGPU,
)
def ggml_backend_webgpu_reg() -> Optional[ggml_backend_reg_t]:
    ...


#####################################################
# GGML VirtGPU API
# source: include/ggml-virtgpu.h
#####################################################


GGML_USE_VIRTGPU = hasattr(lib, "ggml_backend_virtgpu_reg")


# GGML_BACKEND_API ggml_backend_reg_t ggml_backend_virtgpu_reg();
@ggml_function(
    "ggml_backend_virtgpu_reg",
    [],
    ggml_backend_reg_t_ctypes,
    enabled=GGML_USE_VIRTGPU,
)
def ggml_backend_virtgpu_reg() -> Optional[ggml_backend_reg_t]:
    ...


#####################################################
# GGML ZenDNN API
# source: src/ggml-zendnn.h
#####################################################


GGML_USE_ZENDNN = hasattr(lib, "ggml_backend_zendnn_init")


# GGML_API GGML_CALL ggml_backend_t ggml_backend_zendnn_init(void);
@ggml_function(
    "ggml_backend_zendnn_init",
    [],
    ggml_backend_t_ctypes,
    enabled=GGML_USE_ZENDNN,
)
def ggml_backend_zendnn_init() -> Optional[ggml_backend_t]:
    ...


# GGML_API GGML_CALL bool ggml_backend_is_zendnn(ggml_backend_t backend);
@ggml_function(
    "ggml_backend_is_zendnn",
    [ggml_backend_t_ctypes],
    ctypes.c_bool,
    enabled=GGML_USE_ZENDNN,
)
def ggml_backend_is_zendnn(backend: Union[ggml_backend_t, int], /) -> bool:
    ...


# GGML_API GGML_CALL void ggml_backend_zendnn_set_n_threads(ggml_backend_t backend_zendnn, int n_threads);
@ggml_function(
    "ggml_backend_zendnn_set_n_threads",
    [ggml_backend_t_ctypes, ctypes.c_int],
    None,
    enabled=GGML_USE_ZENDNN,
)
def ggml_backend_zendnn_set_n_threads(
    backend_zendnn: Union[ggml_backend_t, int],
    n_threads: Union[ctypes.c_int, int],
    /,
) -> None:
    ...


# GGML_API GGML_CALL ggml_backend_reg_t ggml_backend_zendnn_reg(void);
@ggml_function(
    "ggml_backend_zendnn_reg",
    [],
    ggml_backend_reg_t_ctypes,
    enabled=GGML_USE_ZENDNN,
)
def ggml_backend_zendnn_reg() -> Optional[ggml_backend_reg_t]:
    ...


#####################################################
# GGML zDNN API
# source: src/ggml-zdnn.h
#####################################################


GGML_USE_ZDNN = hasattr(lib, "ggml_backend_zdnn_buffer_type")


# GGML_API GGML_CALL ggml_backend_buffer_type_t ggml_backend_zdnn_buffer_type(void);
@ggml_function(
    "ggml_backend_zdnn_buffer_type",
    [],
    ggml_backend_buffer_type_t_ctypes,
    enabled=GGML_USE_ZDNN,
)
def ggml_backend_zdnn_buffer_type() -> Optional[ggml_backend_buffer_type_t]:
    ...


# GGML_API GGML_CALL ggml_backend_reg_t ggml_backend_zdnn_reg(void);
@ggml_function(
    "ggml_backend_zdnn_reg",
    [],
    ggml_backend_reg_t_ctypes,
    enabled=GGML_USE_ZDNN,
)
def ggml_backend_zdnn_reg() -> Optional[ggml_backend_reg_t]:
    ...


#####################################################
# GGML CUDA API
# source: src/ggml-cuda.h
#####################################################


GGML_USE_CUDA = hasattr(lib, "ggml_backend_cuda_init")
GGML_USE_CUDA_LOG_CALLBACK = hasattr(lib, "ggml_backend_cuda_log_set_callback")


GGML_CUDA_MAX_DEVICES = 16


# // backend API
# GGML_API GGML_CALL ggml_backend_t ggml_backend_cuda_init(int device);
@ggml_function(
    "ggml_backend_cuda_init", [ctypes.c_int], ggml_backend_t_ctypes, enabled=GGML_USE_CUDA
)
def ggml_backend_cuda_init(device: int) -> Optional[ggml_backend_t]:
    ...


# GGML_API GGML_CALL bool ggml_backend_is_cuda(ggml_backend_t backend);
@ggml_function(
    "ggml_backend_is_cuda",
    [ggml_backend_t_ctypes],
    ctypes.c_bool,
    enabled=GGML_USE_CUDA,
)
def ggml_backend_is_cuda(
    backend: Union[ggml_backend_t, int],
) -> bool:
    ...


# // device buffer
# GGML_API GGML_CALL ggml_backend_buffer_type_t ggml_backend_cuda_buffer_type(int device);
@ggml_function(
    "ggml_backend_cuda_buffer_type",
    [ctypes.c_int],
    ggml_backend_buffer_type_t_ctypes,
    enabled=GGML_USE_CUDA,
)
def ggml_backend_cuda_buffer_type(
    device: Union[ctypes.c_int, int],
) -> Optional[ggml_backend_buffer_type_t]:
    ...


# // split tensor buffer that splits matrices by rows across multiple devices
# GGML_API GGML_CALL bool ggml_backend_cuda_allreduce_tensor(ggml_backend_t * backends, struct ggml_tensor ** tensors, size_t n_backends);
@ggml_function(
    "ggml_backend_cuda_allreduce_tensor",
    [
        ctypes.POINTER(ggml_backend_t_ctypes),
        ctypes.POINTER(ctypes.POINTER(ggml_tensor)),
        ctypes.c_size_t,
    ],
    ctypes.c_bool,
    enabled=GGML_USE_CUDA,
)
def ggml_backend_cuda_allreduce_tensor(
    backends: CtypesPointer[ggml_backend_t_ctypes],
    tensors: CtypesPointer[ggml_tensor_p],
    n_backends: Union[ctypes.c_size_t, int],
    /,
) -> bool:
    ...


# GGML_API GGML_CALL ggml_backend_buffer_type_t ggml_backend_cuda_split_buffer_type(int main_device, const float * tensor_split);
@ggml_function(
    "ggml_backend_cuda_split_buffer_type",
    [ctypes.c_int, ctypes.POINTER(ctypes.c_float)],
    ggml_backend_buffer_type_t_ctypes,
    enabled=GGML_USE_CUDA,
)
def ggml_backend_cuda_split_buffer_type(
    main_device: Union[ctypes.c_int, int],
    tensor_split: CtypesArray[ctypes.c_float],
) -> Optional[ggml_backend_buffer_type_t]:
    ...


# // pinned host buffer for use with the CPU backend for faster copies between CPU and GPU
# GGML_API GGML_CALL ggml_backend_buffer_type_t ggml_backend_cuda_host_buffer_type(void);
@ggml_function(
    "ggml_backend_cuda_host_buffer_type",
    [],
    ggml_backend_buffer_type_t_ctypes,
    enabled=GGML_USE_CUDA,
)
def ggml_backend_cuda_host_buffer_type() -> (
    Union[ggml_backend_buffer_type_t, int, None]
):
    ...


# GGML_API GGML_CALL int  ggml_backend_cuda_get_device_count(void);
@ggml_function(
    "ggml_backend_cuda_get_device_count", [], ctypes.c_int, enabled=GGML_USE_CUDA
)
def ggml_backend_cuda_get_device_count() -> int:
    ...


# GGML_API GGML_CALL void ggml_backend_cuda_get_device_description(int device, char * description, size_t description_size);
@ggml_function(
    "ggml_backend_cuda_get_device_description",
    [
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_size_t,
    ],
    None,
    enabled=GGML_USE_CUDA,
)
def ggml_backend_cuda_get_device_description(
    device: Union[ctypes.c_int, int],
    description: ctypes.c_char_p,
    description_size: Union[ctypes.c_size_t, int],
    /,
):
    ...


# GGML_API GGML_CALL void ggml_backend_cuda_get_device_memory(int device, size_t * free, size_t * total);
@ggml_function(
    "ggml_backend_cuda_get_device_memory",
    [
        ctypes.c_int,
        ctypes.POINTER(ctypes.c_size_t),
        ctypes.POINTER(ctypes.c_size_t),
    ],
    None,
    enabled=GGML_USE_CUDA,
)
def ggml_backend_cuda_get_device_memory(
    device: Union[ctypes.c_int, int],
    free: "ctypes._Pointer[ctypes.c_size_t]",  # type: ignore
    total: "ctypes._Pointer[ctypes.c_size_t]",  # type: ignore
    /,
):
    ...


# GGML_API GGML_CALL bool ggml_backend_cuda_register_host_buffer(void * buffer, size_t size);
@ggml_function(
    "ggml_backend_cuda_register_host_buffer",
    [
        ctypes.c_void_p,
        ctypes.c_size_t,
    ],
    ctypes.c_bool,
    enabled=GGML_USE_CUDA,
)
def ggml_backend_cuda_register_host_buffer(
    buffer: Union[ctypes.c_void_p, int, None],
    size: Union[ctypes.c_size_t, int],
) -> bool:
    ...


# GGML_API GGML_CALL void ggml_backend_cuda_unregister_host_buffer(void * buffer);
@ggml_function(
    "ggml_backend_cuda_unregister_host_buffer",
    [ctypes.c_void_p],
    None,
    enabled=GGML_USE_CUDA,
)
def ggml_backend_cuda_unregister_host_buffer(
    buffer: Union[ctypes.c_void_p, int, None], /
):
    ...


# GGML_API GGML_CALL ggml_backend_reg_t ggml_backend_cuda_reg(void);
@ggml_function(
    "ggml_backend_cuda_reg",
    [],
    ggml_backend_reg_t_ctypes,
    enabled=GGML_USE_CUDA,
)
def ggml_backend_cuda_reg() -> Optional[ggml_backend_reg_t]:
    ...


# GGML_API void ggml_backend_cuda_log_set_callback(ggml_log_callback log_callback, void * user_data);
@ggml_function(
    "ggml_backend_cuda_log_set_callback",
    [
        ggml_log_callback,
        ctypes.c_void_p,
    ],
    None,
    enabled=GGML_USE_CUDA_LOG_CALLBACK,
)
def ggml_backend_cuda_log_set_callback(
    log_callback, user_data: Union[ctypes.c_void_p, int, None], /  # type: ignore
):
    ...


#####################################################
# GGML METAL API
# source: src/ggml-metal.h
#####################################################


GGML_USE_METAL = hasattr(lib, "ggml_backend_metal_init")


# // max memory buffers that can be mapped to the device
# #define GGML_METAL_MAX_BUFFERS 64
GGML_METAL_MAX_BUFFERS = 64

# //
# // backend API
# // user-code should use only these functions
# //


# GGML_API ggml_backend_t ggml_backend_metal_init(void);
@ggml_function(
    "ggml_backend_metal_init", [], ggml_backend_t_ctypes, enabled=GGML_USE_METAL
)
def ggml_backend_metal_init() -> Optional[ggml_backend_t]:
    ...


# GGML_API bool ggml_backend_is_metal(ggml_backend_t backend);
@ggml_function(
    "ggml_backend_is_metal",
    [ggml_backend_t_ctypes],
    ctypes.c_bool,
    enabled=GGML_USE_METAL,
)
def ggml_backend_is_metal(
    backend: Union[ggml_backend_t, int],
) -> bool:
    ...


# GGML_API void ggml_backend_metal_set_abort_callback(ggml_backend_t backend, ggml_abort_callback abort_callback, void * user_data);
@ggml_function(
    "ggml_backend_metal_set_abort_callback",
    [
        ggml_backend_t_ctypes,
        ggml_abort_callback,
        ctypes.c_void_p,
    ],
    None,
    enabled=GGML_USE_METAL,
)
def ggml_backend_metal_set_abort_callback(
    backend: Union[ggml_backend_t, int],
    abort_callback,  # type: ignore
    user_data: Union[ctypes.c_void_p, int, None],
    /,
):
    ...


# // helper to check if the device supports a specific family
# // ideally, the user code should be doing these checks
# // ref: https://developer.apple.com/metal/Metal-Feature-Set-Tables.pdf
# GGML_API bool ggml_backend_metal_supports_family(ggml_backend_t backend, int family);
@ggml_function(
    "ggml_backend_metal_supports_family",
    [
        ggml_backend_t_ctypes,
        ctypes.c_int,
    ],
    ctypes.c_bool,
    enabled=GGML_USE_METAL,
)
def ggml_backend_metal_supports_family(
    backend: Union[ggml_backend_t, int],
    family: Union[ctypes.c_int, int],
) -> bool:
    ...


# // capture all command buffers committed the next time `ggml_backend_graph_compute` is called
# GGML_API void ggml_backend_metal_capture_next_compute(ggml_backend_t backend);
@ggml_function(
    "ggml_backend_metal_capture_next_compute",
    [ggml_backend_t_ctypes],
    None,
    enabled=GGML_USE_METAL,
)
def ggml_backend_metal_capture_next_compute(backend: Union[ggml_backend_t, int], /):
    ...


# GGML_API ggml_backend_reg_t ggml_backend_metal_reg(void);
@ggml_function(
    "ggml_backend_metal_reg",
    [],
    ggml_backend_reg_t_ctypes,
    enabled=GGML_USE_METAL,
)
def ggml_backend_metal_reg() -> Optional[ggml_backend_reg_t]:
    ...


#####################################################
# GGML OPENCL API
# source: ggml-opencl.h
#####################################################


GGML_USE_OPENCL = hasattr(lib, "ggml_backend_opencl_init")
GGML_USE_CLBLAST = hasattr(lib, "ggml_cl_init")


# GGML_API void ggml_cl_init(void);
@ggml_function("ggml_cl_init", [], None, enabled=GGML_USE_CLBLAST)
def ggml_cl_init():
    ...


# GGML_API void   ggml_cl_mul(const struct ggml_tensor * src0, const struct ggml_tensor * src1, struct ggml_tensor * dst);
@ggml_function(
    "ggml_cl_mul",
    [
        ctypes.POINTER(ggml_tensor),
        ctypes.POINTER(ggml_tensor),
        ctypes.POINTER(ggml_tensor),
    ],
    None,
    enabled=GGML_USE_CLBLAST,
)
def ggml_cl_mul(src0: ggml_tensor_p, src1: ggml_tensor_p, dst: ggml_tensor_p, /):
    ...


# GGML_API void   ggml_cl_add(const struct ggml_tensor * src0, const struct ggml_tensor * src1, struct ggml_tensor * dst);
@ggml_function(
    "ggml_cl_add",
    [
        ctypes.POINTER(ggml_tensor),
        ctypes.POINTER(ggml_tensor),
        ctypes.POINTER(ggml_tensor),
    ],
    None,
    enabled=GGML_USE_CLBLAST,
)
def ggml_cl_add(src0: ggml_tensor_p, src1: ggml_tensor_p, dst: ggml_tensor_p, /):
    ...


# GGML_API bool   ggml_cl_can_mul_mat(const struct ggml_tensor * src0, const struct ggml_tensor * src1, const struct ggml_tensor * dst);
@ggml_function(
    "ggml_cl_can_mul_mat",
    [
        ctypes.POINTER(ggml_tensor),
        ctypes.POINTER(ggml_tensor),
        ctypes.POINTER(ggml_tensor),
    ],
    ctypes.c_bool,
    enabled=GGML_USE_CLBLAST,
)
def ggml_cl_can_mul_mat(
    src0: ggml_tensor_p,
    src1: ggml_tensor_p,
    dst: ggml_tensor_p,
) -> bool:
    ...


# GGML_API size_t ggml_cl_mul_mat_get_wsize(const struct ggml_tensor * src0, const struct ggml_tensor * src1, struct ggml_tensor * dst);
@ggml_function(
    "ggml_cl_mul_mat_get_wsize",
    [
        ctypes.POINTER(ggml_tensor),
        ctypes.POINTER(ggml_tensor),
        ctypes.POINTER(ggml_tensor),
    ],
    ctypes.c_size_t,
    enabled=GGML_USE_CLBLAST,
)
def ggml_cl_mul_mat_get_wsize(
    src0: ggml_tensor_p,
    src1: ggml_tensor_p,
    dst: ggml_tensor_p,
) -> int:
    ...


# GGML_API void   ggml_cl_mul_mat(const struct ggml_tensor * src0, const struct ggml_tensor * src1, struct ggml_tensor * dst, void * wdata, size_t wsize);
@ggml_function(
    "ggml_cl_mul_mat",
    [
        ctypes.POINTER(ggml_tensor),
        ctypes.POINTER(ggml_tensor),
        ctypes.POINTER(ggml_tensor),
        ctypes.c_void_p,
        ctypes.c_size_t,
    ],
    None,
    enabled=GGML_USE_CLBLAST,
)
def ggml_cl_mul_mat(
    src0: ggml_tensor_p,
    src1: ggml_tensor_p,
    dst: ggml_tensor_p,
    wdata: Union[ctypes.c_void_p, int, None],
    wsize: Union[ctypes.c_size_t, int],
    /,
):
    ...


# GGML_API void ggml_cl_free_data(const struct ggml_tensor* tensor);
@ggml_function(
    "ggml_cl_free_data",
    [
        ctypes.POINTER(ggml_tensor),
    ],
    None,
    enabled=GGML_USE_CLBLAST,
)
def ggml_cl_free_data(tensor: ggml_tensor_p, /):
    ...


# GGML_API void ggml_cl_transform_tensor(void * data, struct ggml_tensor * tensor);
@ggml_function(
    "ggml_cl_transform_tensor",
    [
        ctypes.c_void_p,
        ctypes.POINTER(ggml_tensor),
    ],
    None,
    enabled=GGML_USE_CLBLAST,
)
def ggml_cl_transform_tensor(
    data: Union[ctypes.c_void_p, int, None], tensor: ggml_tensor_p, /
):
    ...


# // backend API

# GGML_API ggml_backend_t ggml_backend_opencl_init(void);
@ggml_function(
    "ggml_backend_opencl_init",
    [],
    ggml_backend_t_ctypes,
    enabled=GGML_USE_OPENCL,
)
def ggml_backend_opencl_init() -> Optional[ggml_backend_t]:
    ...

# GGML_API bool ggml_backend_is_opencl(ggml_backend_t backend);
@ggml_function(
    "ggml_backend_is_opencl",
    [ggml_backend_t_ctypes],
    ctypes.c_bool,
    enabled=GGML_USE_OPENCL,
)
def ggml_backend_is_opencl(backend: Union[ggml_backend_t, int], /) -> bool:
    ...


# GGML_API ggml_backend_buffer_type_t ggml_backend_opencl_buffer_type(void);
@ggml_function(
    "ggml_backend_opencl_buffer_type",
    [],
    ggml_backend_buffer_type_t_ctypes,
    enabled=GGML_USE_OPENCL,
)
def ggml_backend_opencl_buffer_type() -> Optional[ggml_backend_buffer_type_t]:
    ...


# GGML_API ggml_backend_buffer_type_t ggml_backend_opencl_host_buffer_type(void);
@ggml_function(
    "ggml_backend_opencl_host_buffer_type",
    [],
    ggml_backend_buffer_type_t_ctypes,
    enabled=GGML_USE_OPENCL,
)
def ggml_backend_opencl_host_buffer_type() -> Optional[ggml_backend_buffer_type_t]:
    ...


# GGML_API ggml_backend_reg_t ggml_backend_opencl_reg(void);
@ggml_function(
    "ggml_backend_opencl_reg",
    [],
    ggml_backend_reg_t_ctypes,
    enabled=GGML_USE_OPENCL,
)
def ggml_backend_opencl_reg() -> Optional[ggml_backend_reg_t]:
    ...


# TODO: Add ggml-quants.h

#####################################################
# GGML Vulkan API
# source: src/ggml-vulkan.h
#####################################################

GGML_USE_VULKAN = hasattr(lib, "ggml_backend_vk_init")
GGML_USE_VULKAN_CPU_ASSIST = hasattr(lib, "ggml_vk_init_cpu_assist")

# #define GGML_VK_NAME "Vulkan"
# #define GGML_VK_MAX_DEVICES 16
GGML_VK_NAME = "Vulkan"
GGML_VK_MAX_DEVICES = 16


# GGML_API void ggml_vk_instance_init(void);
@ggml_function("ggml_vk_instance_init", [], None, enabled=GGML_USE_VULKAN_CPU_ASSIST)
def ggml_vk_instance_init():
    ...


# GGML_API void ggml_vk_init_cpu_assist(void);
@ggml_function("ggml_vk_init_cpu_assist", [], None, enabled=GGML_USE_VULKAN_CPU_ASSIST)
def ggml_vk_init_cpu_assist():
    ...


# GGML_API void ggml_vk_preallocate_buffers_graph_cpu_assist(struct ggml_tensor * node);
@ggml_function(
    "ggml_vk_preallocate_buffers_graph_cpu_assist",
    [ctypes.POINTER(ggml_tensor)],
    None,
    enabled=GGML_USE_VULKAN_CPU_ASSIST,
)
def ggml_vk_preallocate_buffers_graph_cpu_assist(node: ggml_tensor_p, /):
    ...


# GGML_API void ggml_vk_preallocate_buffers_cpu_assist(void);
@ggml_function(
    "ggml_vk_preallocate_buffers_cpu_assist",
    [],
    None,
    enabled=GGML_USE_VULKAN_CPU_ASSIST,
)
def ggml_vk_preallocate_buffers_cpu_assist():
    ...


# GGML_API void ggml_vk_build_graph_cpu_assist(struct ggml_tensor * node, bool last_node);
@ggml_function(
    "ggml_vk_build_graph_cpu_assist",
    [
        ctypes.POINTER(ggml_tensor),
        ctypes.c_bool,
    ],
    None,
    enabled=GGML_USE_VULKAN_CPU_ASSIST,
)
def ggml_vk_build_graph_cpu_assist(node: ggml_tensor_p, last_node: bool, /):
    ...


# GGML_API bool ggml_vk_compute_forward_cpu_assist(struct ggml_compute_params * params, struct ggml_tensor * tensor);
@ggml_function(
    "ggml_vk_compute_forward_cpu_assist",
    [
        ctypes.POINTER(ggml_compute_params),
        ctypes.POINTER(ggml_tensor),
    ],
    ctypes.c_bool,
    enabled=GGML_USE_VULKAN_CPU_ASSIST,
)
def ggml_vk_compute_forward_cpu_assist(
    params: ggml_compute_params_p, tensor: ggml_tensor_p, /
) -> bool:
    ...


# #ifdef GGML_VULKAN_CHECK_RESULTS
# void ggml_vk_check_results_1_cpu_assist(struct ggml_compute_params * params, struct ggml_tensor * tensor);
# #endif


# GGML_API void ggml_vk_graph_cleanup_cpu_assist(void);
@ggml_function(
    "ggml_vk_graph_cleanup_cpu_assist", [], None, enabled=GGML_USE_VULKAN_CPU_ASSIST
)
def ggml_vk_graph_cleanup_cpu_assist():
    ...


# GGML_API void ggml_vk_free_cpu_assist(void);
@ggml_function("ggml_vk_free_cpu_assist", [], None, enabled=GGML_USE_VULKAN_CPU_ASSIST)
def ggml_vk_free_cpu_assist():
    ...


# // backend API
# GGML_API GGML_CALL ggml_backend_t ggml_backend_vk_init(size_t dev_num);
@ggml_function(
    "ggml_backend_vk_init",
    [ctypes.c_size_t],
    ggml_backend_t_ctypes,
    enabled=GGML_USE_VULKAN,
)
def ggml_backend_vk_init(
    dev_num: Union[ctypes.c_size_t, int], /
) -> Optional[ggml_backend_t]:
    ...


# GGML_API GGML_CALL bool ggml_backend_is_vk(ggml_backend_t backend);
@ggml_function(
    "ggml_backend_is_vk",
    [ggml_backend_t_ctypes],
    ctypes.c_bool,
    enabled=GGML_USE_VULKAN,
)
def ggml_backend_is_vk(backend: Union[ggml_backend_t, int], /) -> bool:
    ...


# GGML_API GGML_CALL int  ggml_backend_vk_get_device_count(void);
@ggml_function(
    "ggml_backend_vk_get_device_count", [], ctypes.c_int, enabled=GGML_USE_VULKAN
)
def ggml_backend_vk_get_device_count() -> int:
    ...


# GGML_API GGML_CALL void ggml_backend_vk_get_device_description(int device, char * description, size_t description_size);
@ggml_function(
    "ggml_backend_vk_get_device_description",
    [
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_size_t,
    ],
    None,
    enabled=GGML_USE_VULKAN,
)
def ggml_backend_vk_get_device_description(
    device: Union[ctypes.c_int, int],
    description: bytes,
    description_size: Union[ctypes.c_size_t, int],
    /,
):
    ...


# GGML_API GGML_CALL void ggml_backend_vk_get_device_memory(int device, size_t * free, size_t * total);
@ggml_function(
    "ggml_backend_vk_get_device_memory",
    [
        ctypes.c_int,
        ctypes.POINTER(ctypes.c_size_t),
        ctypes.POINTER(ctypes.c_size_t),
    ],
    None,
    enabled=GGML_USE_VULKAN,
)
def ggml_backend_vk_get_device_memory(
    device: Union[ctypes.c_int, int],
    free: "ctypes._Pointer[ctypes.c_size_t]",  # type: ignore
    total: "ctypes._Pointer[ctypes.c_size_t]",  # type: ignore
    /,
):
    ...


# GGML_API GGML_CALL ggml_backend_buffer_type_t ggml_backend_vk_buffer_type(size_t dev_num);
@ggml_function(
    "ggml_backend_vk_buffer_type",
    [ctypes.c_size_t],
    ggml_backend_buffer_type_t,
    enabled=GGML_USE_VULKAN,
)
def ggml_backend_vk_buffer_type(
    dev_num: Union[ctypes.c_size_t, int], /
) -> Optional[ggml_backend_buffer_type_t]:
    ...


# // pinned host buffer for use with the CPU backend for faster copies between CPU and GPU
# GGML_API GGML_CALL ggml_backend_buffer_type_t ggml_backend_vk_host_buffer_type(void);
@ggml_function(
    "ggml_backend_vk_host_buffer_type",
    [],
    ggml_backend_buffer_type_t,
    enabled=GGML_USE_VULKAN,
)
def ggml_backend_vk_host_buffer_type() -> Optional[ggml_backend_buffer_type_t]:
    ...


# GGML_API GGML_CALL ggml_backend_reg_t ggml_backend_vk_reg(void);
@ggml_function(
    "ggml_backend_vk_reg",
    [],
    ggml_backend_reg_t_ctypes,
    enabled=GGML_USE_VULKAN,
)
def ggml_backend_vk_reg() -> Optional[ggml_backend_reg_t]:
    ...


#####################################################
# GGML Vulkan API
# source: src/ggml-rpc.h
#####################################################


GGML_USE_RPC = hasattr(lib, "ggml_backend_rpc_init")
GGML_USE_RPC_LEGACY_SERVER = hasattr(lib, "start_rpc_server")


#define GGML_RPC_MAX_SERVERS       16
GGML_RPC_MAX_SERVERS = 16


# // backend API
# GGML_API GGML_CALL ggml_backend_t ggml_backend_rpc_init(const char * endpoint, uint32_t device);
@ggml_function(
    "ggml_backend_rpc_init",
    [ctypes.c_char_p, ctypes.c_uint32],
    ggml_backend_t_ctypes,
    enabled=GGML_USE_RPC,
)
def ggml_backend_rpc_init(
    endpoint: bytes,
    device: Union[ctypes.c_uint32, int],
    /,
) -> Optional[ggml_backend_t]:
    ...


# GGML_API GGML_CALL bool ggml_backend_is_rpc(ggml_backend_t backend);
@ggml_function(
    "ggml_backend_is_rpc",
    [ggml_backend_t_ctypes],
    ctypes.c_bool,
    enabled=GGML_USE_RPC,
)
def ggml_backend_is_rpc(backend: Union[ggml_backend_t, int], /) -> bool:
    ...


# GGML_API GGML_CALL ggml_backend_buffer_type_t ggml_backend_rpc_buffer_type(const char * endpoint, uint32_t device);
@ggml_function(
    "ggml_backend_rpc_buffer_type",
    [ctypes.c_char_p, ctypes.c_uint32],
    ggml_backend_buffer_type_t_ctypes,
    enabled=GGML_USE_RPC,
)
def ggml_backend_rpc_buffer_type(
    endpoint: bytes,
    device: Union[ctypes.c_uint32, int],
    /,
) -> Optional[ggml_backend_buffer_type_t]:
    ...


# GGML_API GGML_CALL void ggml_backend_rpc_get_device_memory(const char * endpoint, uint32_t device, size_t * free, size_t * total);
@ggml_function(
    "ggml_backend_rpc_get_device_memory",
    [
        ctypes.c_char_p,
        ctypes.c_uint32,
        ctypes.POINTER(ctypes.c_size_t),
        ctypes.POINTER(ctypes.c_size_t),
    ],
    None,
    enabled=GGML_USE_RPC,
)
def ggml_backend_rpc_get_device_memory(
    endpoint: bytes,
    device: Union[ctypes.c_uint32, int],
    free: CtypesPointer[ctypes.c_size_t],
    total: CtypesPointer[ctypes.c_size_t],
    /,
):
    ...


# GGML_API GGML_CALL void ggml_backend_rpc_start_server(
#         const char * endpoint,
#         const char * cache_dir,
#         size_t n_threads,
#         size_t n_devices,
#         ggml_backend_dev_t * devices);
@ggml_function(
    "ggml_backend_rpc_start_server",
    [
        ctypes.c_char_p,
        ctypes.c_char_p,
        ctypes.c_size_t,
        ctypes.c_size_t,
        ctypes.POINTER(ggml_backend_dev_t_ctypes),
    ],
    None,
    enabled=GGML_USE_RPC,
)
def ggml_backend_rpc_start_server(
    endpoint: bytes,
    cache_dir: Optional[bytes],
    n_threads: Union[ctypes.c_size_t, int],
    n_devices: Union[ctypes.c_size_t, int],
    devices: CtypesPointer[ggml_backend_dev_t_ctypes],
    /,
):
    ...


# GGML_API GGML_CALL ggml_backend_reg_t ggml_backend_rpc_reg(void);
@ggml_function(
    "ggml_backend_rpc_reg",
    [],
    ggml_backend_reg_t_ctypes,
    enabled=GGML_USE_RPC,
)
def ggml_backend_rpc_reg() -> Optional[ggml_backend_reg_t]:
    ...


# GGML_API GGML_CALL ggml_backend_reg_t ggml_backend_rpc_add_server(const char * endpoint);
@ggml_function(
    "ggml_backend_rpc_add_server",
    [ctypes.c_char_p],
    ggml_backend_reg_t_ctypes,
    enabled=GGML_USE_RPC,
)
def ggml_backend_rpc_add_server(endpoint: bytes, /) -> Optional[ggml_backend_reg_t]:
    ...


# GGML_API GGML_CALL void start_rpc_server(ggml_backend_t backend, const char * endpoint, size_t free_mem, size_t total_mem);
@ggml_function(
    "start_rpc_server",
    [
        ggml_backend_t_ctypes,
        ctypes.c_char_p,
        ctypes.c_size_t,
        ctypes.c_size_t,
    ],
    None,
    enabled=GGML_USE_RPC_LEGACY_SERVER,
)
def start_rpc_server(
    backend: Union[ggml_backend_t, int],
    endpoint: bytes,
    free_mem: Union[ctypes.c_size_t, int],
    total_mem: Union[ctypes.c_size_t, int],
    /,
):
    ...


# TODO: Add ggml-sycl.h

# TODO: Add ggml-kompute.h
