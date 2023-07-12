from __future__ import annotations
import enum
import ctypes

from typing import Any, Callable, Optional, Sequence, Tuple

import ggml
from ggml.utils import from_numpy, to_numpy

import numpy as np
import numpy.typing as npt

_default_context: Optional[Context] = None


def default_context() -> Context:
    global _default_context
    if _default_context is None:
        _default_context = Context(InitParams())
    return _default_context


class InitParams:
    def __init__(
        self,
        mem_size: int = 16 * 1024 * 1024,
        mem_buffer: Optional[ctypes.c_void_p] = None,
        no_alloc: bool = False,
    ):
        self.mem_size = mem_size
        self.mem_buffer = mem_buffer # NOTE: DO NOT REMOVE THIS
        self.no_alloc = no_alloc
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


class GGML_TYPE(enum.IntEnum):
    F32 = ggml.GGML_TYPE_F32
    F16 = ggml.GGML_TYPE_F16
    Q4_0 = ggml.GGML_TYPE_Q4_0
    Q4_1 = ggml.GGML_TYPE_Q4_1
    Q5_0 = ggml.GGML_TYPE_Q5_0
    Q5_1 = ggml.GGML_TYPE_Q5_1
    Q8_0 = ggml.GGML_TYPE_Q8_0
    Q8_1 = ggml.GGML_TYPE_Q8_1
    I8 = ggml.GGML_TYPE_I8
    I16 = ggml.GGML_TYPE_I16
    I32 = ggml.GGML_TYPE_I32


NUMPY_DTYPE_TO_GGML_TYPE = {
    np.float16: GGML_TYPE.F16,
    np.float32: GGML_TYPE.F32,
    np.int8: GGML_TYPE.I8,
    np.int16: GGML_TYPE.I16,
    np.int32: GGML_TYPE.I32,
}

GGML_TYPE_TO_NUMPY_DTYPE = {v: k for k, v in NUMPY_DTYPE_TO_GGML_TYPE.items()}


class GGML_FTYPE(enum.IntEnum):
    UNKNOWN = ggml.GGML_FTYPE_UNKNOWN
    ALL_F32 = ggml.GGML_FTYPE_ALL_F32
    MOSTLY_F16 = ggml.GGML_FTYPE_MOSTLY_F16
    MOSTLY_Q4_0 = ggml.GGML_FTYPE_MOSTLY_Q4_0
    MOSTLY_Q4_1_SOME_F16 = ggml.GGML_FTYPE_MOSTLY_Q4_1_SOME_F16
    MOSTLY_Q8_0 = ggml.GGML_FTYPE_MOSTLY_Q8_0
    MOSTLY_Q5_0 = ggml.GGML_FTYPE_MOSTLY_Q5_0
    MOSTLY_Q5_1 = ggml.GGML_FTYPE_MOSTLY_Q5_1


class Tensor:
    def __init__(
        self,
        tensor,  # type: ctypes._Pointer[ggml.ggml_tensor] # type: ignore
        ctx: Optional[Context] = None,
    ):
        self.tensor = tensor
        self.ctx = ctx or default_context()

    def nelements(self):
        return ggml.ggml_nelements(self.tensor)

    def nbytes(self):
        return ggml.ggml_nbytes(self.tensor)

    def element_size(self):
        return ggml.ggml_element_size(self.tensor)

    @property
    def name(self):
        return ggml.ggml_get_name(self.tensor)

    @name.setter
    def name(self, name: bytes):
        ggml.ggml_set_name(self.tensor, name)

    @property
    def ggml_type(self):
        return GGML_TYPE(self.tensor.contents.type)

    @property
    def shape(self) -> Tuple[int, ...]:
        return tuple(self.tensor.contents.ne[: self.tensor.contents.n_dims])

    @property
    def data(self):
        return ggml.ggml_get_data(self.tensor)

    def set_data(self, data: bytes):
        return ctypes.memmove(self.data, data, self.nbytes())

    def numpy(self):
        return to_numpy(self.tensor)

    # Magic methods
    def __len__(self):
        return self.nelements()

    def __add__(self, other: Tensor):
        op = ggml.ggml_add(self.ctx.context, self.tensor, other.tensor)
        return Tensor(tensor=op, ctx=self.ctx)

    def __sub__(self, other: Tensor):
        op = ggml.ggml_sub(self.ctx.context, self.tensor, other.tensor)
        return Tensor(tensor=op, ctx=self.ctx)

    def __mul__(self, other: Tensor):
        op = ggml.ggml_mul(self.ctx.context, self.tensor, other.tensor)
        return Tensor(tensor=op, ctx=self.ctx)

    def __truediv__(self, other: Tensor):
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
        cls, shape: Sequence[int], ggml_type: GGML_TYPE, ctx: Optional[Context] = None
    ):
        ctx = ctx or default_context()
        tensor = ggml.ggml_new_tensor(
            ctx.context,
            ctypes.c_int(ggml_type.value),
            ctypes.c_int(len(shape)),
            (ctypes.c_int64 * len(shape))(*shape),
        )
        return cls(tensor=tensor, ctx=ctx)

    @classmethod
    def from_numpy(cls, x: npt.NDArray[Any], ctx: Optional[Context] = None):
        ctx = ctx or default_context()
        tensor = from_numpy(x, ctx.context)
        return cls(tensor=tensor, ctx=ctx)

    @staticmethod
    def new_tensor(
        ggml_type: GGML_TYPE = GGML_TYPE.F32,
        shape: Sequence[int] = (),
        ctx: Optional[Context] = None,
    ):
        ctx = ctx or default_context()
        tensor = ggml.ggml_new_tensor(
            ctx.context,
            ctypes.c_int(ggml_type.value),
            ctypes.c_int(len(shape)),
            (ctypes.c_int64 * len(shape))(*shape),
        )
        return Tensor(tensor=tensor, ctx=ctx)

    @staticmethod
    def new_tensor_1d(
        ggml_type: GGML_TYPE = GGML_TYPE.F32,
        ne0: int = 0,
        ctx: Optional[Context] = None,
    ):
        ctx = ctx or default_context()
        tensor = ggml.ggml_new_tensor_1d(
            ctx.context,
            ctypes.c_int(ggml_type.value),
            ctypes.c_int64(ne0),
        )
        return Tensor(tensor=tensor, ctx=ctx)

    @staticmethod
    def new_tensor_2d(
        ggml_type: GGML_TYPE = GGML_TYPE.F32,
        ne0: int = 0,
        ne1: int = 0,
        ctx: Optional[Context] = None,
    ):
        ctx = ctx or default_context()
        tensor = ggml.ggml_new_tensor_2d(
            ctx.context,
            ctypes.c_int(ggml_type.value),
            ctypes.c_int64(ne0),
            ctypes.c_int64(ne1),
        )
        return Tensor(tensor=tensor, ctx=ctx)

    @staticmethod
    def new_tensor_3d(
        ggml_type: GGML_TYPE = GGML_TYPE.F32,
        ne0: int = 0,
        ne1: int = 0,
        ne2: int = 0,
        ctx: Optional[Context] = None,
    ):
        ctx = ctx or default_context()
        tensor = ggml.ggml_new_tensor_3d(
            ctx.context,
            ctypes.c_int(ggml_type.value),
            ctypes.c_int64(ne0),
            ctypes.c_int64(ne1),
            ctypes.c_int64(ne2),
        )
        return Tensor(tensor=tensor, ctx=ctx)

    @staticmethod
    def new_tensor_4d(
        ggml_type: GGML_TYPE = GGML_TYPE.F32,
        ne0: int = 0,
        ne1: int = 0,
        ne2: int = 0,
        ne3: int = 0,
        ctx: Optional[Context] = None,
    ):
        ctx = ctx or default_context()
        tensor = ggml.ggml_new_tensor_4d(
            ctx.context,
            ctypes.c_int(ggml_type.value),
            ctypes.c_int64(ne0),
            ctypes.c_int64(ne1),
            ctypes.c_int64(ne2),
            ctypes.c_int64(ne3),
        )
        return Tensor(tensor=tensor, ctx=ctx)

    @staticmethod
    def new_i32(
        value: int,
        ctx: Optional[Context] = None,
    ):
        ctx = ctx or default_context()
        tensor = ggml.ggml_new_i32(ctx.context, ctypes.c_int32(value))
        return Tensor(tensor=tensor, ctx=ctx)

    @staticmethod
    def new_f32(
        value: float,
        ctx: Optional[Context] = None,
    ):
        ctx = ctx or default_context()
        tensor = ggml.ggml_new_f32(ctx.context, ctypes.c_float(value))
        return Tensor(tensor=tensor, ctx=ctx)

    @staticmethod
    def dup_tensor(a: Tensor, ctx: Optional[Context] = None):
        ctx = ctx or default_context()
        op = ggml.ggml_dup_tensor(ctx.context, a.tensor)
        return Tensor(tensor=op, ctx=ctx)

    @staticmethod
    def view(a: Tensor, ctx: Optional[Context] = None):
        ctx = ctx or default_context()
        op = ggml.ggml_view_tensor(ctx.context, a.tensor)
        return Tensor(tensor=op, ctx=ctx)

    @staticmethod
    def set_zero(a: Tensor):
        op = ggml.ggml_set_zero(a.tensor)
        return Tensor(tensor=op, ctx=a.ctx)

    @staticmethod
    def set_i32(a: Tensor, value: int):
        op = ggml.ggml_set_i32(a.tensor, ctypes.c_int32(value))
        return Tensor(tensor=op, ctx=a.ctx)

    @staticmethod
    def get_i32_1d(a: Tensor, i: int):
        return ggml.ggml_get_i32_1d(a.tensor, ctypes.c_int(i))

    @staticmethod
    def set_f32(a: Tensor, i: int, value: float):
        op = ggml.ggml_set_f32_1d(a.tensor, ctypes.c_int(i), ctypes.c_float(value))
        return Tensor(tensor=op, ctx=a.ctx)

    @staticmethod
    def get_f32_1d(a: Tensor, i: int):
        return ggml.ggml_get_f32_1d(a.tensor, ctypes.c_int(i))

    @staticmethod
    def set_f32_1d(a: Tensor, i: int, value: float):
        op = ggml.ggml_set_f32_1d(a.tensor, ctypes.c_int(i), ctypes.c_float(value))
        return Tensor(tensor=op, ctx=a.ctx)

    @staticmethod
    def get_data(a: Tensor):
        return ggml.ggml_get_data(a.tensor)

    @staticmethod
    def get_data_f32(a: Tensor):
        return ggml.ggml_get_data_f32(a.tensor)

    @staticmethod
    def get_name(a: Tensor):
        return ggml.ggml_get_name(a.tensor)

    @staticmethod
    def set_name(a: Tensor, name: bytes):
        ggml.ggml_set_name(a.tensor, name)

    @staticmethod
    def dup(a: Tensor, ctx: Optional[Context] = None):
        ctx = ctx or default_context()
        op = ggml.ggml_dup(ctx.context, a.tensor)
        return Tensor(tensor=op, ctx=ctx)

    @staticmethod
    def add(a: Tensor, b: Tensor, ctx: Optional[Context] = None):
        ctx = ctx or default_context()
        op = ggml.ggml_add(ctx.context, a.tensor, b.tensor)
        return Tensor(tensor=op, ctx=ctx)

    @staticmethod
    def add_inplace(a: Tensor, b: Tensor, ctx: Optional[Context] = None):
        ctx = ctx or default_context()
        op = ggml.ggml_add_inplace(ctx.context, a.tensor, b.tensor)
        return Tensor(tensor=op, ctx=ctx)

    @staticmethod
    def add1(a: Tensor, b: Tensor, ctx: Optional[Context] = None):
        ctx = ctx or default_context()
        op = ggml.ggml_add1(ctx.context, a.tensor, b.tensor)
        return Tensor(tensor=op, ctx=ctx)

    @staticmethod
    def acc(
        a: Tensor,
        b: Tensor,
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
        a: Tensor,
        b: Tensor,
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
    def sub(a: Tensor, b: Tensor, ctx: Optional[Context] = None):
        ctx = ctx or default_context()
        op = ggml.ggml_sub(ctx.context, a.tensor, b.tensor)
        return Tensor(tensor=op, ctx=ctx)

    @staticmethod
    def mul(a: Tensor, b: Tensor, ctx: Optional[Context] = None):
        ctx = ctx or default_context()
        op = ggml.ggml_mul(ctx.context, a.tensor, b.tensor)
        return Tensor(tensor=op, ctx=ctx)

    @staticmethod
    def div(a: Tensor, b: Tensor, ctx: Optional[Context] = None):
        ctx = ctx or default_context()
        op = ggml.ggml_div(ctx.context, a.tensor, b.tensor)
        return Tensor(tensor=op, ctx=ctx)

    @staticmethod
    def sqr(a: Tensor, ctx: Optional[Context] = None):
        ctx = ctx or default_context()
        op = ggml.ggml_sqr(ctx.context, a.tensor)
        return Tensor(tensor=op, ctx=ctx)

    @staticmethod
    def sqrt(a: Tensor, ctx: Optional[Context] = None):
        ctx = ctx or default_context()
        op = ggml.ggml_sqrt(ctx.context, a.tensor)
        return Tensor(tensor=op, ctx=ctx)

    @staticmethod
    def log(a: Tensor, ctx: Optional[Context] = None):
        ctx = ctx or default_context()
        op = ggml.ggml_log(ctx.context, a.tensor)
        return Tensor(tensor=op, ctx=ctx)

    @staticmethod
    def log_inplace(a: Tensor, ctx: Optional[Context] = None):
        ctx = ctx or default_context()
        op = ggml.ggml_log_inplace(ctx.context, a.tensor)
        return Tensor(tensor=op, ctx=ctx)

    @staticmethod
    def sum(a: Tensor, ctx: Optional[Context] = None):
        ctx = ctx or default_context()
        op = ggml.ggml_sum(ctx.context, a.tensor)
        return Tensor(tensor=op, ctx=ctx)

    @staticmethod
    def sum_rows(a: Tensor, ctx: Optional[Context] = None):
        ctx = ctx or default_context()
        op = ggml.ggml_sum_rows(ctx.context, a.tensor)
        return Tensor(tensor=op, ctx=ctx)

    @staticmethod
    def mean(a: Tensor, ctx: Optional[Context] = None):
        ctx = ctx or default_context()
        op = ggml.ggml_mean(ctx.context, a.tensor)
        return Tensor(tensor=op, ctx=ctx)

    @staticmethod
    def repeat(a: Tensor, b: Tensor, ctx: Optional[Context] = None):
        ctx = ctx or default_context()
        op = ggml.ggml_repeat(ctx.context, a.tensor, b.tensor)
        return Tensor(tensor=op, ctx=ctx)

    @staticmethod
    def abs(a: Tensor, ctx: Optional[Context] = None):
        ctx = ctx or default_context()
        op = ggml.ggml_abs(ctx.context, a.tensor)
        return Tensor(tensor=op, ctx=ctx)

    @staticmethod
    def sgn(a: Tensor, ctx: Optional[Context] = None):
        ctx = ctx or default_context()
        op = ggml.ggml_sgn(ctx.context, a.tensor)
        return Tensor(tensor=op, ctx=ctx)

    @staticmethod
    def neg(a: Tensor, ctx: Optional[Context] = None):
        ctx = ctx or default_context()
        op = ggml.ggml_neg(ctx.context, a.tensor)
        return Tensor(tensor=op, ctx=ctx)

    @staticmethod
    def step(a: Tensor, ctx: Optional[Context] = None):
        ctx = ctx or default_context()
        op = ggml.ggml_step(ctx.context, a.tensor)
        return Tensor(tensor=op, ctx=ctx)

    @staticmethod
    def relu(a: Tensor, ctx: Optional[Context] = None):
        ctx = ctx or default_context()
        op = ggml.ggml_relu(ctx.context, a.tensor)
        return Tensor(tensor=op, ctx=ctx)

    @staticmethod
    def gelu(a: Tensor, ctx: Optional[Context] = None):
        ctx = ctx or default_context()
        op = ggml.ggml_gelu(ctx.context, a.tensor)
        return Tensor(tensor=op, ctx=ctx)

    @staticmethod
    def silu(a: Tensor, ctx: Optional[Context] = None):
        ctx = ctx or default_context()
        op = ggml.ggml_silu(ctx.context, a.tensor)
        return Tensor(tensor=op, ctx=ctx)

    @staticmethod
    def silu_back(a: Tensor, b: Tensor, ctx: Optional[Context] = None):
        ctx = ctx or default_context()
        op = ggml.ggml_silu_back(ctx.context, a.tensor, b.tensor)
        return Tensor(tensor=op, ctx=ctx)

    @staticmethod
    def norm(a: Tensor, ctx: Optional[Context] = None):
        ctx = ctx or default_context()
        op = ggml.ggml_norm(ctx.context, a.tensor)
        return Tensor(tensor=op, ctx=ctx)

    @staticmethod
    def rms_norm(a: Tensor, ctx: Optional[Context] = None):
        ctx = ctx or default_context()
        op = ggml.ggml_rms_norm(ctx.context, a.tensor)
        return Tensor(tensor=op, ctx=ctx)

    @staticmethod
    def rms_norm_back(a: Tensor, b: Tensor, ctx: Optional[Context] = None):
        ctx = ctx or default_context()
        op = ggml.ggml_rms_norm_back(ctx.context, a.tensor, b.tensor)
        return Tensor(tensor=op, ctx=ctx)

    @staticmethod
    def mul_mat(a: Tensor, b: Tensor, ctx: Optional[Context] = None):
        ctx = ctx or default_context()
        op = ggml.ggml_mul_mat(ctx.context, a.tensor, b.tensor)
        return Tensor(tensor=op, ctx=ctx)

    @staticmethod
    def scale(a: Tensor, b: Tensor, ctx: Optional[Context] = None):
        ctx = ctx or default_context()
        op = ggml.ggml_scale(ctx.context, a.tensor, b.tensor)
        return Tensor(tensor=op, ctx=ctx)

    @staticmethod
    def scale_inplace(a: Tensor, b: Tensor, ctx: Optional[Context] = None):
        ctx = ctx or default_context()
        op = ggml.ggml_scale_inplace(ctx.context, a.tensor, b.tensor)
        return Tensor(tensor=op, ctx=ctx)

    @staticmethod
    def set(
        a: Tensor,
        b: Tensor,
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
        a: Tensor,
        b: Tensor,
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
    def set_1d(a: Tensor, b: Tensor, offset: int, ctx: Optional[Context] = None):
        ctx = ctx or default_context()
        op = ggml.ggml_set_1d(ctx.context, a.tensor, b.tensor, ctypes.c_size_t(offset))
        return Tensor(tensor=op, ctx=ctx)

    @staticmethod
    def set_1d_inplace(
        a: Tensor, b: Tensor, offset: int, ctx: Optional[Context] = None
    ):
        ctx = ctx or default_context()
        op = ggml.ggml_set_1d_inplace(
            ctx.context, a.tensor, b.tensor, ctypes.c_size_t(offset)
        )
        return Tensor(tensor=op, ctx=ctx)

    @staticmethod
    def set_2d(
        a: Tensor,
        b: Tensor,
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
        a: Tensor,
        b: Tensor,
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

    @staticmethod
    def cpy(a: Tensor, b: Tensor, ctx: Optional[Context] = None):
        ctx = ctx or default_context()
        op = ggml.ggml_cpy(ctx.context, a.tensor, b.tensor)
        return Tensor(tensor=op, ctx=ctx)

    @staticmethod
    def cont(a: Tensor, ctx: Optional[Context] = None):
        ctx = ctx or default_context()
        op = ggml.ggml_cont(ctx.context, a.tensor)
        return Tensor(tensor=op, ctx=ctx)

    @staticmethod
    def reshape(a: Tensor, b: Tensor, ctx: Optional[Context] = None):
        ctx = ctx or default_context()
        op = ggml.ggml_reshape(ctx.context, a.tensor, b.tensor)
        return Tensor(tensor=op, ctx=ctx)

    @staticmethod
    def reshape_1d(a: Tensor, ne0: int, ctx: Optional[Context] = None):
        ctx = ctx or default_context()
        op = ggml.ggml_reshape_1d(ctx.context, a.tensor, ctypes.c_int64(ne0))
        return Tensor(tensor=op, ctx=ctx)

    @staticmethod
    def reshape_2d(a: Tensor, ne0: int, ne1: int, ctx: Optional[Context] = None):
        ctx = ctx or default_context()
        op = ggml.ggml_reshape_2d(
            ctx.context, a.tensor, ctypes.c_int64(ne0), ctypes.c_int64(ne1)
        )
        return Tensor(tensor=op, ctx=ctx)

    @staticmethod
    def reshape_3d(
        a: Tensor, ne0: int, ne1: int, ne2: int, ctx: Optional[Context] = None
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

    @staticmethod
    def reshape_4d(
        a: Tensor,
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
        a: Tensor,
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
        a: Tensor,
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
        a: Tensor,
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
        a: Tensor,
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
        a: Tensor,
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
    def transpose(a: Tensor, ctx: Optional[Context] = None):
        ctx = ctx or default_context()
        op = ggml.ggml_transpose(ctx.context, a.tensor)
        return Tensor(tensor=op, ctx=ctx)

    @staticmethod
    def get_rows(a: Tensor, b: Tensor, ctx: Optional[Context] = None):
        ctx = ctx or default_context()
        op = ggml.ggml_get_rows(ctx.context, a.tensor, b.tensor)
        return Tensor(tensor=op, ctx=ctx)

    @staticmethod
    def get_rows_back(a: Tensor, b: Tensor, c: Tensor, ctx: Optional[Context] = None):
        ctx = ctx or default_context()
        op = ggml.ggml_get_rows_back(ctx.context, a.tensor, b.tensor, c.tensor)
        return Tensor(tensor=op, ctx=ctx)

    @staticmethod
    def diag(a: Tensor, ctx: Optional[Context] = None):
        ctx = ctx or default_context()
        op = ggml.ggml_diag(ctx.context, a.tensor)
        return Tensor(tensor=op, ctx=ctx)

    @staticmethod
    def diag_mask_inf(a: Tensor, n_past: int, ctx: Optional[Context] = None):
        ctx = ctx or default_context()
        op = ggml.ggml_diag_mask_inf(ctx.context, a.tensor, ctypes.c_int(n_past))
        return Tensor(tensor=op, ctx=ctx)

    @staticmethod
    def diag_mask_inf_inplace(a: Tensor, n_past: int, ctx: Optional[Context] = None):
        ctx = ctx or default_context()
        op = ggml.ggml_diag_mask_inf_inplace(
            ctx.context, a.tensor, ctypes.c_int(n_past)
        )
        return Tensor(tensor=op, ctx=ctx)

    @staticmethod
    def diag_mask_zero(a: Tensor, n_past: int, ctx: Optional[Context] = None):
        ctx = ctx or default_context()
        op = ggml.ggml_diag_mask_zero(ctx.context, a.tensor, ctypes.c_int(n_past))
        return Tensor(tensor=op, ctx=ctx)

    @staticmethod
    def diag_mask_zero_inplace(a: Tensor, n_past: int, ctx: Optional[Context] = None):
        ctx = ctx or default_context()
        op = ggml.ggml_diag_mask_zero_inplace(
            ctx.context, a.tensor, ctypes.c_int(n_past)
        )
        return Tensor(tensor=op, ctx=ctx)

    @staticmethod
    def soft_max(a: Tensor, ctx: Optional[Context] = None):
        ctx = ctx or default_context()
        op = ggml.ggml_soft_max(ctx.context, a.tensor)
        return Tensor(tensor=op, ctx=ctx)

    @staticmethod
    def soft_max_inplace(a: Tensor, ctx: Optional[Context] = None):
        ctx = ctx or default_context()
        op = ggml.ggml_soft_max_inplace(ctx.context, a.tensor)
        return Tensor(tensor=op, ctx=ctx)

    @staticmethod
    def rope(
        a: Tensor,
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
        a: Tensor,
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
        a: Tensor,
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
        a: Tensor,
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
    def clamp(a: Tensor, min: float, max: float, ctx: Optional[Context] = None):
        ctx = ctx or default_context()
        op = ggml.ggml_clamp(
            ctx.context, a.tensor, ctypes.c_float(min), ctypes.c_float(max)
        )
        return Tensor(tensor=op, ctx=ctx)

    @staticmethod
    def conv_1d_s1_ph(
        a: Tensor,
        b: Tensor,
        ctx: Optional[Context] = None,
    ):
        ctx = ctx or default_context()
        op = ggml.ggml_conv_1d_s1_ph(ctx.context, a.tensor, b.tensor)
        return Tensor(tensor=op, ctx=ctx)

    @staticmethod
    def conv_2d_sk_p0(a: Tensor, b: Tensor, ctx: Optional[Context] = None):
        ctx = ctx or default_context()
        op = ggml.ggml_conv_2d_sk_p0(ctx.context, a.tensor, b.tensor)
        return Tensor(tensor=op, ctx=ctx)

    @staticmethod
    def conv_1d_s2_ph(
        a: Tensor,
        b: Tensor,
        ctx: Optional[Context] = None,
    ):
        ctx = ctx or default_context()
        op = ggml.ggml_conv_1d_s2_ph(ctx.context, a.tensor, b.tensor)
        return Tensor(tensor=op, ctx=ctx)

    @staticmethod
    def flash_attn(
        q: Tensor,
        k: Tensor,
        v: Tensor,
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
        a: Tensor,
        b0: Tensor,
        b1: Tensor,
        c0: Tensor,
        c1: Tensor,
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
        a: Tensor,
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
        a: Tensor,
        b: Tensor,
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
        a: Tensor,
        ctx: Optional[Context] = None,
    ):
        ctx = ctx or default_context()
        op = ggml.ggml_set_param(ctx.context, a.tensor)
        return Tensor(tensor=op, ctx=ctx)

    @staticmethod
    def ggml_ftype_to_ggml_type(ftype: int):
        return GGML_TYPE(ggml.ggml_ftype_to_ggml_type(ctypes.c_int(ftype)))


class CGraph:
    def __init__(self, cgraph: ggml.ggml_cgraph, ctx: Optional[Context] = None):
        self.cgraph = cgraph
        self.ctx = ctx or default_context()

    def compute(self, n_threads: int = 1):
        ggml.ggml_graph_compute_with_ctx(self.ctx.context, ctypes.pointer(self.cgraph), n_threads=n_threads)

    def reset(self):
        ggml.ggml_graph_reset(ctypes.pointer(self.cgraph))

    def get_tensor(self, name: bytes):
        return Tensor(
            tensor=ggml.ggml_graph_get_tensor(ctypes.pointer(self.cgraph), name),
            ctx=self.ctx,
        )

    def graph_export(self, fname: bytes):
        ggml.ggml_graph_export(ctypes.pointer(self.cgraph), fname)

    def build_forward_expand(self, tensor: Tensor):
        ggml.ggml_build_forward_expand(ctypes.pointer(self.cgraph), tensor.tensor)

    @staticmethod
    def print(a: CGraph):
        ggml.ggml_graph_print(ctypes.pointer(a.cgraph))

    @staticmethod
    def dump_dot(
        gb: CGraph,
        gf: Optional[CGraph],
        filename: bytes,
    ):
        gf_p = ctypes.pointer(gf.cgraph) if gf else None
        ggml.ggml_graph_dump_dot(
            ctypes.pointer(gb.cgraph), gf_p, filename  # type: ignore
        )

    @classmethod
    def build_forward(cls, tensor: Tensor):
        return CGraph(cgraph=ggml.ggml_build_forward(tensor.tensor), ctx=tensor.ctx)

    @classmethod
    def build_backward(cls, forward: CGraph, keep: bool, ctx: Optional[Context] = None):
        ctx = ctx or default_context()
        return CGraph(
            cgraph=ggml.ggml_build_backward(
                ctx.context, ctypes.pointer(forward.cgraph), ctypes.c_bool(keep)
            )
        )

    @classmethod
    def graph_import(cls, fname: bytes, ctx: Optional[Context] = None) -> CGraph:
        ctx = ctx or default_context()
        return CGraph(
            cgraph=ggml.ggml_graph_import(
                fname, ctypes.pointer(ctx.context), ctypes.pointer(ctx.context)
            ),
            ctx=ctx,
        )
