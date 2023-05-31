import enum
import ctypes

from typing import Any, Callable, Optional, Sequence, Tuple

import ggml

import numpy as np
import numpy.typing as npt

_default_context: Optional["Context"] = None


def default_context() -> "Context":
    global _default_context
    if _default_context is None:
        _default_context = Context(InitParams())
    return _default_context


class InitParams:
    def __init__(self):
        self.mem_size = 16 * 1024 * 1024
        self.mem_buffer = None
        self.no_alloc = False
        self.params = ggml.ggml_init_params(
            mem_size=self.mem_size,
            mem_buffer=self.mem_buffer,
            no_alloc=self.no_alloc,
        )


class Context:
    def __init__(self, init_params: InitParams):
        self.init_params = init_params
        self.context: ggml.ggml_context_p = ggml.ggml_init(init_params.params)

    def __del__(self):
        ggml.ggml_free(self.context)


class DType(enum.Enum):
    F32 = ggml.GGML_TYPE_F32.value
    F16 = ggml.GGML_TYPE_F16.value
    Q4_0 = ggml.GGML_TYPE_Q4_0.value
    Q4_1 = ggml.GGML_TYPE_Q4_1.value
    Q5_0 = ggml.GGML_TYPE_Q5_0.value
    Q8_0 = ggml.GGML_TYPE_Q8_0.value
    Q8_1 = ggml.GGML_TYPE_Q8_1.value
    I8 = ggml.GGML_TYPE_I8.value
    I16 = ggml.GGML_TYPE_I16.value
    I32 = ggml.GGML_TYPE_I32.value


NUMPY_DTYPE_TO_GGML_DTYPE = {
    np.float32: DType.F32,
    np.float16: DType.F16,
    np.uint8: DType.Q8_0,
    np.uint8: DType.Q8_1,
    np.int8: DType.I8,
    np.int16: DType.I16,
    np.int32: DType.I32,
}

GGML_DTYPE_TO_NUMPY_DTYPE = {v: k for k, v in NUMPY_DTYPE_TO_GGML_DTYPE.items()}


class Tensor:
    def __init__(
        self,
        tensor,  # type: ctypes._Pointer[ggml.ggml_tensor] # type: ignore
        ctx: Optional[Context] = None,
    ):
        self.tensor = tensor
        self.ctx = ctx or default_context()

    def n_elements(self):
        return ggml.ggml_nelements(self.tensor)

    @property
    def name(self):
        return ggml.ggml_get_name(self.tensor)

    @name.setter
    def name(self, name: bytes):
        ggml.ggml_set_name(self.tensor, name)

    @property
    def dtype(self):
        return DType(self.tensor.contents.type)

    @property
    def shape(self) -> Tuple[int, ...]:
        return tuple(self.tensor.contents.ne[: self.tensor.contents.n_dims])

    def numpy(self):
        ctypes_type = np.ctypeslib.as_ctypes_type(GGML_DTYPE_TO_NUMPY_DTYPE[self.dtype])
        array = ctypes.cast(
            ggml.ggml_get_data(self.tensor), ctypes.POINTER(ctypes_type)
        )
        return np.ctypeslib.as_array(array, shape=self.shape)

    # Magic methods
    def __len__(self):
        return self.n_elements()

    def __add__(self, other: "Tensor"):
        op = ggml.ggml_add(self.ctx.context, self.tensor, other.tensor)
        return Tensor(tensor=op, ctx=self.ctx)

    def __sub__(self, other: "Tensor"):
        op = ggml.ggml_sub(self.ctx.context, self.tensor, other.tensor)
        return Tensor(tensor=op, ctx=self.ctx)

    def __mul__(self, other: "Tensor"):
        op = ggml.ggml_mul(self.ctx.context, self.tensor, other.tensor)
        return Tensor(tensor=op, ctx=self.ctx)

    def __truediv__(self, other: "Tensor"):
        op = ggml.ggml_div(self.ctx.context, self.tensor, other.tensor)
        return Tensor(tensor=op, ctx=self.ctx)

    def __neg__(self):
        op = ggml.ggml_neg(self.ctx.context, self.tensor)
        return Tensor(tensor=op, ctx=self.ctx)

    def __abs__(self):
        op = ggml.ggml_abs(self.ctx.context, self.tensor)
        return Tensor(tensor=op, ctx=self.ctx)

    @classmethod
    def with_shape(
        cls, shape: Sequence[int], dtype: DType, ctx: Optional[Context] = None
    ):
        ctx = ctx or default_context()
        tensor = ggml.ggml_new_tensor(
            ctx.context,
            ctypes.c_int(dtype.value),
            ctypes.c_int(len(shape)),
            (ctypes.c_int64 * len(shape))(*shape),
        )
        return cls(tensor=tensor, ctx=ctx)

    @classmethod
    def from_numpy(cls, x: npt.NDArray[Any], ctx: Optional[Context] = None):
        dtype = NUMPY_DTYPE_TO_GGML_DTYPE[x.dtype.type]
        ctypes_type = np.ctypeslib.as_ctypes_type(x.dtype)
        tensor = cls.with_shape(shape=x.shape, dtype=dtype, ctx=ctx)
        array = ctypes.cast(
            ggml.ggml_get_data(tensor.tensor), ctypes.POINTER(ctypes_type)
        )
        arr = np.ctypeslib.as_array(array, shape=x.shape)
        arr[:] = x
        return tensor

    @staticmethod
    def dup_tensor(a: "Tensor", ctx: Optional[Context] = None):
        ctx = ctx or default_context()
        op = ggml.ggml_dup_tensor(ctx.context, a.tensor)
        return Tensor(tensor=op, ctx=ctx)

    @staticmethod
    def view(a: "Tensor", ctx: Optional[Context] = None):
        ctx = ctx or default_context()
        op = ggml.ggml_view_tensor(ctx.context, a.tensor)
        return Tensor(tensor=op, ctx=ctx)

    @staticmethod
    def set_zero(a: "Tensor"):
        op = ggml.ggml_set_zero(a.tensor)
        return Tensor(tensor=op, ctx=a.ctx)

    @staticmethod
    def set_i32(a: "Tensor", value: int):
        op = ggml.ggml_set_i32(a.tensor, ctypes.c_int32(value))
        return Tensor(tensor=op, ctx=a.ctx)

    @staticmethod
    def get_i32_1d(a: "Tensor", i: int):
        return ggml.ggml_get_i32_1d(a.tensor, ctypes.c_int(i))

    @staticmethod
    def set_f32(a: "Tensor", i: int, value: float):
        op = ggml.ggml_set_f32_1d(a.tensor, ctypes.c_int(i), ctypes.c_float(value))
        return Tensor(tensor=op, ctx=a.ctx)

    @staticmethod
    def get_f32_1d(a: "Tensor", i: int):
        return ggml.ggml_get_f32_1d(a.tensor, ctypes.c_int(i))

    @staticmethod
    def set_f32_1d(a: "Tensor", i: int, value: float):
        op = ggml.ggml_set_f32_1d(a.tensor, ctypes.c_int(i), ctypes.c_float(value))
        return Tensor(tensor=op, ctx=a.ctx)

    @staticmethod
    def get_data(a: "Tensor"):
        return ggml.ggml_get_data(a.tensor)

    @staticmethod
    def get_data_f32(a: "Tensor"):
        return ggml.ggml_get_data_f32(a.tensor)

    @staticmethod
    def get_name(a: "Tensor"):
        return ggml.ggml_get_name(a.tensor)

    @staticmethod
    def set_name(a: "Tensor", name: bytes):
        ggml.ggml_set_name(a.tensor, name)

    @staticmethod
    def dup(a: "Tensor", ctx: Optional[Context] = None):
        ctx = ctx or default_context()
        op = ggml.ggml_dup(ctx.context, a.tensor)
        return Tensor(tensor=op, ctx=ctx)

    @staticmethod
    def add(a: "Tensor", b: "Tensor", ctx: Optional[Context] = None):
        ctx = ctx or default_context()
        op = ggml.ggml_add(ctx.context, a.tensor, b.tensor)
        return Tensor(tensor=op, ctx=ctx)

    @staticmethod
    def add_inplace(a: "Tensor", b: "Tensor", ctx: Optional[Context] = None):
        ctx = ctx or default_context()
        op = ggml.ggml_add_inplace(ctx.context, a.tensor, b.tensor)
        return Tensor(tensor=op, ctx=ctx)

    @staticmethod
    def add1(a: "Tensor", b: "Tensor", ctx: Optional[Context] = None):
        ctx = ctx or default_context()
        op = ggml.ggml_add1(ctx.context, a.tensor, b.tensor)
        return Tensor(tensor=op, ctx=ctx)

    @staticmethod
    def acc(
        a: "Tensor",
        b: "Tensor",
        nb1: int,
        nb2: int,
        nb3: int,
        offset: int,
        ctx: Optional[Context] = None,
    ):
        ctx = ctx or default_context()
        op = ggml.ggml_acc(
            ctx.context,
            a.tensor,
            b.tensor,
            ctypes.c_size_t(nb1),
            ctypes.c_size_t(nb2),
            ctypes.c_size_t(nb3),
            ctypes.c_size_t(offset),
        )
        return Tensor(tensor=op, ctx=ctx)

    @staticmethod
    def acc_inplace(
        a: "Tensor",
        b: "Tensor",
        nb1: int,
        nb2: int,
        nb3: int,
        offset: int,
        ctx: Optional[Context] = None,
    ):
        ctx = ctx or default_context()
        op = ggml.ggml_acc_inplace(
            ctx.context,
            a.tensor,
            b.tensor,
            ctypes.c_size_t(nb1),
            ctypes.c_size_t(nb2),
            ctypes.c_size_t(nb3),
            ctypes.c_size_t(offset),
        )
        return Tensor(tensor=op, ctx=ctx)

    @staticmethod
    def sub(a: "Tensor", b: "Tensor", ctx: Optional[Context] = None):
        ctx = ctx or default_context()
        op = ggml.ggml_sub(ctx.context, a.tensor, b.tensor)
        return Tensor(tensor=op, ctx=ctx)

    @staticmethod
    def mul(a: "Tensor", b: "Tensor", ctx: Optional[Context] = None):
        ctx = ctx or default_context()
        op = ggml.ggml_mul(ctx.context, a.tensor, b.tensor)
        return Tensor(tensor=op, ctx=ctx)

    @staticmethod
    def div(a: "Tensor", b: "Tensor", ctx: Optional[Context] = None):
        ctx = ctx or default_context()
        op = ggml.ggml_div(ctx.context, a.tensor, b.tensor)
        return Tensor(tensor=op, ctx=ctx)

    @staticmethod
    def sqr(a: "Tensor", ctx: Optional[Context] = None):
        ctx = ctx or default_context()
        op = ggml.ggml_sqr(ctx.context, a.tensor)
        return Tensor(tensor=op, ctx=ctx)

    @staticmethod
    def sqrt(a: "Tensor", ctx: Optional[Context] = None):
        ctx = ctx or default_context()
        op = ggml.ggml_sqrt(ctx.context, a.tensor)
        return Tensor(tensor=op, ctx=ctx)

    @staticmethod
    def log(a: "Tensor", ctx: Optional[Context] = None):
        ctx = ctx or default_context()
        op = ggml.ggml_log(ctx.context, a.tensor)
        return Tensor(tensor=op, ctx=ctx)

    @staticmethod
    def log_inplace(a: "Tensor", ctx: Optional[Context] = None):
        ctx = ctx or default_context()
        op = ggml.ggml_log_inplace(ctx.context, a.tensor)
        return Tensor(tensor=op, ctx=ctx)

    @staticmethod
    def sum(a: "Tensor", ctx: Optional[Context] = None):
        ctx = ctx or default_context()
        op = ggml.ggml_sum(ctx.context, a.tensor)
        return Tensor(tensor=op, ctx=ctx)

    @staticmethod
    def sum_rows(a: "Tensor", ctx: Optional[Context] = None):
        ctx = ctx or default_context()
        op = ggml.ggml_sum_rows(ctx.context, a.tensor)
        return Tensor(tensor=op, ctx=ctx)

    @staticmethod
    def mean(a: "Tensor", ctx: Optional[Context] = None):
        ctx = ctx or default_context()
        op = ggml.ggml_mean(ctx.context, a.tensor)
        return Tensor(tensor=op, ctx=ctx)

    @staticmethod
    def repeat(a: "Tensor", b: "Tensor", ctx: Optional[Context] = None):
        ctx = ctx or default_context()
        op = ggml.ggml_repeat(ctx.context, a.tensor, b.tensor)
        return Tensor(tensor=op, ctx=ctx)

    @staticmethod
    def abs(a: "Tensor", ctx: Optional[Context] = None):
        ctx = ctx or default_context()
        op = ggml.ggml_abs(ctx.context, a.tensor)
        return Tensor(tensor=op, ctx=ctx)

    @staticmethod
    def sgn(a: "Tensor", ctx: Optional[Context] = None):
        ctx = ctx or default_context()
        op = ggml.ggml_sgn(ctx.context, a.tensor)
        return Tensor(tensor=op, ctx=ctx)

    @staticmethod
    def neg(a: "Tensor", ctx: Optional[Context] = None):
        ctx = ctx or default_context()
        op = ggml.ggml_neg(ctx.context, a.tensor)
        return Tensor(tensor=op, ctx=ctx)

    @staticmethod
    def step(a: "Tensor", ctx: Optional[Context] = None):
        ctx = ctx or default_context()
        op = ggml.ggml_step(ctx.context, a.tensor)
        return Tensor(tensor=op, ctx=ctx)

    @staticmethod
    def relu(a: "Tensor", ctx: Optional[Context] = None):
        ctx = ctx or default_context()
        op = ggml.ggml_relu(ctx.context, a.tensor)
        return Tensor(tensor=op, ctx=ctx)

    @staticmethod
    def gelu(a: "Tensor", ctx: Optional[Context] = None):
        ctx = ctx or default_context()
        op = ggml.ggml_gelu(ctx.context, a.tensor)
        return Tensor(tensor=op, ctx=ctx)

    @staticmethod
    def silu(a: "Tensor", ctx: Optional[Context] = None):
        ctx = ctx or default_context()
        op = ggml.ggml_silu(ctx.context, a.tensor)
        return Tensor(tensor=op, ctx=ctx)

    @staticmethod
    def silu_back(a: "Tensor", b: "Tensor", ctx: Optional[Context] = None):
        ctx = ctx or default_context()
        op = ggml.ggml_silu_back(ctx.context, a.tensor, b.tensor)
        return Tensor(tensor=op, ctx=ctx)

    @staticmethod
    def norm(a: "Tensor", ctx: Optional[Context] = None):
        ctx = ctx or default_context()
        op = ggml.ggml_norm(ctx.context, a.tensor)
        return Tensor(tensor=op, ctx=ctx)

    @staticmethod
    def rms_norm(a: "Tensor", ctx: Optional[Context] = None):
        ctx = ctx or default_context()
        op = ggml.ggml_rms_norm(ctx.context, a.tensor)
        return Tensor(tensor=op, ctx=ctx)

    @staticmethod
    def rms_norm_back(a: "Tensor", b: "Tensor", ctx: Optional[Context] = None):
        ctx = ctx or default_context()
        op = ggml.ggml_rms_norm_back(ctx.context, a.tensor, b.tensor)
        return Tensor(tensor=op, ctx=ctx)

    @staticmethod
    def mul_mat(a: "Tensor", b: "Tensor", ctx: Optional[Context] = None):
        ctx = ctx or default_context()
        op = ggml.ggml_mul_mat(ctx.context, a.tensor, b.tensor)
        return Tensor(tensor=op, ctx=ctx)

    @staticmethod
    def scale(a: "Tensor", b: "Tensor", ctx: Optional[Context] = None):
        ctx = ctx or default_context()
        op = ggml.ggml_scale(ctx.context, a.tensor, b.tensor)
        return Tensor(tensor=op, ctx=ctx)

    @staticmethod
    def scale_inplace(a: "Tensor", b: "Tensor", ctx: Optional[Context] = None):
        ctx = ctx or default_context()
        op = ggml.ggml_scale_inplace(ctx.context, a.tensor, b.tensor)
        return Tensor(tensor=op, ctx=ctx)

    @staticmethod
    def set(
        a: "Tensor",
        b: "Tensor",
        nb1: int,
        nb2: int,
        nb3: int,
        offset: int,
        ctx: Optional[Context] = None,
    ):
        ctx = ctx or default_context()
        op = ggml.ggml_set(
            ctx.context,
            a.tensor,
            b.tensor,
            ctypes.c_size_t(nb1),
            ctypes.c_size_t(nb2),
            ctypes.c_size_t(nb3),
            ctypes.c_size_t(offset),
        )
        return Tensor(tensor=op, ctx=ctx)

    @staticmethod
    def set_inplace(
        a: "Tensor",
        b: "Tensor",
        nb1: int,
        nb2: int,
        nb3: int,
        offset: int,
        ctx: Optional[Context] = None,
    ):
        ctx = ctx or default_context()
        op = ggml.ggml_set_inplace(
            ctx.context,
            a.tensor,
            b.tensor,
            ctypes.c_size_t(nb1),
            ctypes.c_size_t(nb2),
            ctypes.c_size_t(nb3),
            ctypes.c_size_t(offset),
        )
        return Tensor(tensor=op, ctx=ctx)

    @staticmethod
    def set_1d(a: "Tensor", b: "Tensor", offset: int, ctx: Optional[Context] = None):
        ctx = ctx or default_context()
        op = ggml.ggml_set_1d(ctx.context, a.tensor, b.tensor, ctypes.c_size_t(offset))
        return Tensor(tensor=op, ctx=ctx)

    @staticmethod
    def set_1d_inplace(
        a: "Tensor", b: "Tensor", offset: int, ctx: Optional[Context] = None
    ):
        ctx = ctx or default_context()
        op = ggml.ggml_set_1d_inplace(
            ctx.context, a.tensor, b.tensor, ctypes.c_size_t(offset)
        )
        return Tensor(tensor=op, ctx=ctx)

    @staticmethod
    def set_2d(
        a: "Tensor",
        b: "Tensor",
        nb1: int,
        offset: int,
        ctx: Optional[Context] = None,
    ):
        ctx = ctx or default_context()
        op = ggml.ggml_set_2d(
            ctx.context,
            a.tensor,
            b.tensor,
            ctypes.c_size_t(nb1),
            ctypes.c_size_t(offset),
        )
        return Tensor(tensor=op, ctx=ctx)

    @staticmethod
    def set_2d_inplace(
        a: "Tensor",
        b: "Tensor",
        nb1: int,
        offset: int,
        ctx: Optional[Context] = None,
    ):
        ctx = ctx or default_context()
        op = ggml.ggml_set_2d_inplace(
            ctx.context,
            a.tensor,
            b.tensor,
            ctypes.c_size_t(nb1),
            ctypes.c_size_t(offset),
        )
        return Tensor(tensor=op, ctx=ctx)

    def cpy(self, a: "Tensor", b: "Tensor", ctx: Optional[Context] = None):
        ctx = ctx or default_context()
        op = ggml.ggml_cpy(ctx.context, a.tensor, b.tensor)
        return Tensor(tensor=op, ctx=ctx)

    def cont(self, a: "Tensor", ctx: Optional[Context] = None):
        ctx = ctx or default_context()
        op = ggml.ggml_cont(ctx.context, a.tensor)
        return Tensor(tensor=op, ctx=ctx)

    def reshape(self, a: "Tensor", b: "Tensor", ctx: Optional[Context] = None):
        ctx = ctx or default_context()
        op = ggml.ggml_reshape(ctx.context, a.tensor, b.tensor)
        return Tensor(tensor=op, ctx=ctx)

    def reshape_1d(self, a: "Tensor", ne0: int, ctx: Optional[Context] = None):
        ctx = ctx or default_context()
        op = ggml.ggml_reshape_1d(ctx.context, a.tensor, ctypes.c_int64(ne0))
        return Tensor(tensor=op, ctx=ctx)

    def reshape_2d(
        self, a: "Tensor", ne0: int, ne1: int, ctx: Optional[Context] = None
    ):
        ctx = ctx or default_context()
        op = ggml.ggml_reshape_2d(
            ctx.context, a.tensor, ctypes.c_int64(ne0), ctypes.c_int64(ne1)
        )
        return Tensor(tensor=op, ctx=ctx)

    def reshape_3d(
        self, a: "Tensor", ne0: int, ne1: int, ne2: int, ctx: Optional[Context] = None
    ):
        ctx = ctx or default_context()
        op = ggml.ggml_reshape_3d(
            ctx.context,
            a.tensor,
            ctypes.c_int64(ne0),
            ctypes.c_int64(ne1),
            ctypes.c_int64(ne2),
        )
        return Tensor(tensor=op, ctx=ctx)

    def reshape_4d(
        self,
        a: "Tensor",
        ne0: int,
        ne1: int,
        ne2: int,
        ne3: int,
        ctx: Optional[Context] = None,
    ):
        ctx = ctx or default_context()
        op = ggml.ggml_reshape_4d(
            ctx.context,
            a.tensor,
            ctypes.c_int64(ne0),
            ctypes.c_int64(ne1),
            ctypes.c_int64(ne2),
            ctypes.c_int64(ne3),
        )
        return Tensor(tensor=op, ctx=ctx)

    @staticmethod
    def view_1d(
        a: "Tensor",
        ne0: int,
        offset: int,
        ctx: Optional[Context] = None,
    ):
        ctx = ctx or default_context()
        op = ggml.ggml_view_1d(
            ctx.context, a.tensor, ctypes.c_int64(ne0), ctypes.c_size_t(offset)
        )
        return Tensor(tensor=op, ctx=ctx)

    @staticmethod
    def view_2d(
        a: "Tensor",
        ne0: int,
        ne1: int,
        nb1: int,
        offset: int,
        ctx: Optional[Context] = None,
    ):
        ctx = ctx or default_context()
        op = ggml.ggml_view_2d(
            ctx.context,
            a.tensor,
            ctypes.c_int64(ne0),
            ctypes.c_int64(ne1),
            ctypes.c_size_t(nb1),
            ctypes.c_size_t(offset),
        )
        return Tensor(tensor=op, ctx=ctx)

    @staticmethod
    def view_3d(
        a: "Tensor",
        ne0: int,
        ne1: int,
        ne2: int,
        nb1: int,
        nb2: int,
        offset: int,
        ctx: Optional[Context] = None,
    ):
        ctx = ctx or default_context()
        op = ggml.ggml_view_3d(
            ctx.context,
            a.tensor,
            ctypes.c_int64(ne0),
            ctypes.c_int64(ne1),
            ctypes.c_int64(ne2),
            ctypes.c_size_t(nb1),
            ctypes.c_size_t(nb2),
            ctypes.c_size_t(offset),
        )
        return Tensor(tensor=op, ctx=ctx)

    @staticmethod
    def view_4d(
        a: "Tensor",
        ne0: int,
        ne1: int,
        ne2: int,
        ne3: int,
        nb1: int,
        nb2: int,
        nb3: int,
        offset: int,
        ctx: Optional[Context] = None,
    ):
        ctx = ctx or default_context()
        op = ggml.ggml_view_4d(
            ctx.context,
            a.tensor,
            ctypes.c_int64(ne0),
            ctypes.c_int64(ne1),
            ctypes.c_int64(ne2),
            ctypes.c_int64(ne3),
            ctypes.c_size_t(nb1),
            ctypes.c_size_t(nb2),
            ctypes.c_size_t(nb3),
            ctypes.c_size_t(offset),
        )
        return Tensor(tensor=op, ctx=ctx)

    @staticmethod
    def permute(
        a: "Tensor",
        axis0: int,
        axis1: int,
        axis2: int,
        axis3: int,
        ctx: Optional[Context] = None,
    ):
        ctx = ctx or default_context()
        op = ggml.ggml_permute(
            ctx.context,
            a.tensor,
            ctypes.c_int(axis0),
            ctypes.c_int(axis1),
            ctypes.c_int(axis2),
            ctypes.c_int(axis3),
        )
        return Tensor(tensor=op, ctx=ctx)

    @staticmethod
    def transpose(a: "Tensor", ctx: Optional[Context] = None):
        ctx = ctx or default_context()
        op = ggml.ggml_transpose(ctx.context, a.tensor)
        return Tensor(tensor=op, ctx=ctx)

    @staticmethod
    def get_rows(a: "Tensor", b: "Tensor", ctx: Optional[Context] = None):
        ctx = ctx or default_context()
        op = ggml.ggml_get_rows(ctx.context, a.tensor, b.tensor)
        return Tensor(tensor=op, ctx=ctx)

    @staticmethod
    def get_rows_back(
        a: "Tensor", b: "Tensor", c: "Tensor", ctx: Optional[Context] = None
    ):
        ctx = ctx or default_context()
        op = ggml.ggml_get_rows_back(ctx.context, a.tensor, b.tensor, c.tensor)
        return Tensor(tensor=op, ctx=ctx)

    @staticmethod
    def diag(a: "Tensor", ctx: Optional[Context] = None):
        ctx = ctx or default_context()
        op = ggml.ggml_diag(ctx.context, a.tensor)
        return Tensor(tensor=op, ctx=ctx)

    @staticmethod
    def diag_mask_inf(a: "Tensor", n_past: int, ctx: Optional[Context] = None):
        ctx = ctx or default_context()
        op = ggml.ggml_diag_mask_inf(ctx.context, a.tensor, ctypes.c_int(n_past))
        return Tensor(tensor=op, ctx=ctx)

    @staticmethod
    def diag_mask_inf_inplace(a: "Tensor", n_past: int, ctx: Optional[Context] = None):
        ctx = ctx or default_context()
        op = ggml.ggml_diag_mask_inf_inplace(
            ctx.context, a.tensor, ctypes.c_int(n_past)
        )
        return Tensor(tensor=op, ctx=ctx)

    @staticmethod
    def diag_mask_zero(a: "Tensor", n_past: int, ctx: Optional[Context] = None):
        ctx = ctx or default_context()
        op = ggml.ggml_diag_mask_zero(ctx.context, a.tensor, ctypes.c_int(n_past))
        return Tensor(tensor=op, ctx=ctx)

    @staticmethod
    def diag_mask_zero_inplace(a: "Tensor", n_past: int, ctx: Optional[Context] = None):
        ctx = ctx or default_context()
        op = ggml.ggml_diag_mask_zero_inplace(
            ctx.context, a.tensor, ctypes.c_int(n_past)
        )
        return Tensor(tensor=op, ctx=ctx)

    @staticmethod
    def soft_max(a: "Tensor", ctx: Optional[Context] = None):
        ctx = ctx or default_context()
        op = ggml.ggml_soft_max(ctx.context, a.tensor)
        return Tensor(tensor=op, ctx=ctx)

    @staticmethod
    def soft_max_inplace(a: "Tensor", ctx: Optional[Context] = None):
        ctx = ctx or default_context()
        op = ggml.ggml_soft_max_inplace(ctx.context, a.tensor)
        return Tensor(tensor=op, ctx=ctx)

    @staticmethod
    def rope(
        a: "Tensor",
        n_past: int,
        n_dims: int,
        mode: int,
        ctx: Optional[Context] = None,
    ):
        ctx = ctx or default_context()
        op = ggml.ggml_rope(
            ctx.context,
            a.tensor,
            ctypes.c_int(n_past),
            ctypes.c_int(n_dims),
            ctypes.c_int(mode),
        )
        return Tensor(tensor=op, ctx=ctx)

    @staticmethod
    def rope_inplace(
        a: "Tensor",
        n_past: int,
        n_dims: int,
        mode: int,
        ctx: Optional[Context] = None,
    ):
        ctx = ctx or default_context()
        op = ggml.ggml_rope_inplace(
            ctx.context,
            a.tensor,
            ctypes.c_int(n_past),
            ctypes.c_int(n_dims),
            ctypes.c_int(mode),
        )
        return Tensor(tensor=op, ctx=ctx)

    @staticmethod
    def rope_back(
        a: "Tensor",
        n_past: int,
        n_dims: int,
        mode: int,
        ctx: Optional[Context] = None,
    ):
        ctx = ctx or default_context()
        op = ggml.ggml_rope_back(
            ctx.context,
            a.tensor,
            ctypes.c_int(n_past),
            ctypes.c_int(n_dims),
            ctypes.c_int(mode),
        )
        return Tensor(tensor=op, ctx=ctx)

    @staticmethod
    def alibi(
        a: "Tensor",
        n_past: int,
        n_head: int,
        bias_max: float,
        ctx: Optional[Context] = None,
    ):
        ctx = ctx or default_context()
        op = ggml.ggml_alibi(
            ctx.context,
            a.tensor,
            ctypes.c_int(n_past),
            ctypes.c_int(n_head),
            ctypes.c_float(bias_max),
        )
        return Tensor(tensor=op, ctx=ctx)

    @staticmethod
    def clamp(a: "Tensor", min: float, max: float, ctx: Optional[Context] = None):
        ctx = ctx or default_context()
        op = ggml.ggml_clamp(
            ctx.context, a.tensor, ctypes.c_float(min), ctypes.c_float(max)
        )
        return Tensor(tensor=op, ctx=ctx)

    @staticmethod
    def conv_1d_s1_ph(
        a: "Tensor",
        b: "Tensor",
        ctx: Optional[Context] = None,
    ):
        ctx = ctx or default_context()
        op = ggml.ggml_conv_1d_s1_ph(ctx.context, a.tensor, b.tensor)
        return Tensor(tensor=op, ctx=ctx)

    @staticmethod
    def conv_1d_s2_ph(
        a: "Tensor",
        b: "Tensor",
        ctx: Optional[Context] = None,
    ):
        ctx = ctx or default_context()
        op = ggml.ggml_conv_1d_s2_ph(ctx.context, a.tensor, b.tensor)
        return Tensor(tensor=op, ctx=ctx)

    @staticmethod
    def flash_attn(
        q: "Tensor",
        k: "Tensor",
        v: "Tensor",
        masked: bool,
        ctx: Optional[Context] = None,
    ):
        ctx = ctx or default_context()
        op = ggml.ggml_flash_attn(
            ctx.context,
            q.tensor,
            k.tensor,
            v.tensor,
            ctypes.c_bool(masked),
        )
        return Tensor(tensor=op, ctx=ctx)

    @staticmethod
    def flash_ff(
        a: "Tensor",
        b0: "Tensor",
        b1: "Tensor",
        c0: "Tensor",
        c1: "Tensor",
        ctx: Optional[Context] = None,
    ):
        ctx = ctx or default_context()
        op = ggml.ggml_flash_ff(
            ctx.context, a.tensor, b0.tensor, b1.tensor, c0.tensor, c1.tensor
        )
        return Tensor(tensor=op, ctx=ctx)

    # TODO: fix type signature
    @staticmethod
    def map_unary_f32(
        a: "Tensor",
        fun: Callable[[float], float],
        ctx: Optional[Context] = None,
    ):
        ctx = ctx or default_context()
        op = ggml.ggml_map_unary_f32(  # type: ignore
            ctx.context, a.tensor, ctypes.CFUNCTYPE(ctypes.c_float, ctypes.c_float)(fun)
        )
        return Tensor(tensor=op, ctx=ctx)

    # TODO: fix type signature
    @staticmethod
    def map_binary_f32(
        a: "Tensor",
        b: "Tensor",
        fun: Callable[[float, float], float],
        ctx: Optional[Context] = None,
    ):
        ctx = ctx or default_context()
        op = ggml.ggml_map_binary_f32(  # type: ignore
            ctx.context,
            a.tensor,
            b.tensor,
            ctypes.CFUNCTYPE(ctypes.c_float, ctypes.c_float, ctypes.c_float)(fun),
        )
        return Tensor(tensor=op, ctx=ctx)

    @staticmethod
    def set_param(
        a: "Tensor",
        ctx: Optional[Context] = None,
    ):
        ctx = ctx or default_context()
        op = ggml.ggml_set_param(ctx.context, a.tensor)
        return Tensor(tensor=op, ctx=ctx)


class CGraph:
    def __init__(self, cgraph: ggml.ggml_cgraph, ctx: Optional[Context] = None):
        self.cgraph = cgraph
        self.ctx = ctx or default_context()

    def compute(self):
        ggml.ggml_graph_compute(self.ctx.context, ctypes.pointer(self.cgraph))

    def reset(self):
        ggml.ggml_graph_reset(ctypes.pointer(self.cgraph))

    def get_tensor(self, name: bytes):
        return Tensor(
            tensor=ggml.ggml_graph_get_tensor(ctypes.pointer(self.cgraph), name),
            ctx=self.ctx,
        )

    def graph_export(self, fname: bytes):
        ggml.ggml_graph_export(ctypes.pointer(self.cgraph), fname)

    @staticmethod
    def print(a: "CGraph"):
        ggml.ggml_graph_print(ctypes.pointer(a.cgraph))

    @staticmethod
    def dump_dot(
        gb: "CGraph",
        gf: "CGraph",
        filename: bytes,
    ):
        ggml.ggml_graph_dump_dot(
            ctypes.pointer(gb.cgraph), ctypes.pointer(gf.cgraph), filename
        )

    @classmethod
    def build_forward(cls, tensor: Tensor):
        return CGraph(cgraph=ggml.ggml_build_forward(tensor.tensor), ctx=tensor.ctx)

    @classmethod
    def build_backward(
        cls, forward: "CGraph", keep: bool, ctx: Optional[Context] = None
    ):
        ctx = ctx or default_context()
        return CGraph(
            cgraph=ggml.ggml_build_backward(
                ctx.context, ctypes.pointer(forward.cgraph), ctypes.c_bool(keep)
            )
        )

    @classmethod
    def graph_import(cls, fname: bytes, ctx: Optional[Context] = None) -> "CGraph":
        ctx = ctx or default_context()
        return CGraph(
            cgraph=ggml.ggml_graph_import(
                fname, ctypes.pointer(ctx.context), ctypes.pointer(ctx.context)
            ),
            ctx=ctx,
        )


class OptType(enum.Enum):
    ADAM = ggml.GGML_OPT_ADAM.value
    LBFGS = ggml.GGML_OPT_LBFGS.value


class OptResult(enum.Enum):
    OK = ggml.GGML_OPT_OK.value
    DID_NOT_CONVERGE = ggml.GGML_OPT_DID_NOT_CONVERGE.value
    NO_CONTEXT = ggml.GGML_OPT_NO_CONTEXT.value
    INVALID_WOLFE = ggml.GGML_OPT_INVALID_WOLFE.value
    FAIL = ggml.GGML_OPT_FAIL.value
    LINESEARCH_FAIL = ggml.GGML_LINESEARCH_FAIL.value
    LINESEARCH_MINIMUM_STEP = ggml.GGML_LINESEARCH_MINIMUM_STEP.value
    LINESEARCH_MAXIMUM_STEP = ggml.GGML_LINESEARCH_MAXIMUM_STEP.value
    LINESEARCH_MAXIMUM_ITERATIONS = ggml.GGML_LINESEARCH_MAXIMUM_ITERATIONS.value
    LINESEARCH_INVALID_PARAMETERS = ggml.GGML_LINESEARCH_INVALID_PARAMETERS.value


class Optimizer:
    def __init__(self, opt_params: ggml.ggml_opt_params, ctx: Optional[Context] = None):
        self.opt_params = opt_params
        self.ctx = ctx or default_context()

    def opt(self, f: "Tensor"):
        result = ggml.ggml_opt(self.ctx.context, self.opt_params, f.tensor)
        return OptResult(result)

    @classmethod
    def default(cls, type: OptType = OptType.ADAM, ctx: Optional[Context] = None):
        ctx = ctx or default_context()
        return Optimizer(
            opt_params=ggml.ggml_opt_default_params(ctypes.c_int(type.value)), ctx=ctx
        )
