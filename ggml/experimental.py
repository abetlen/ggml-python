from __future__ import annotations
import enum
import ctypes

from typing import Any, Callable, List, Optional, Sequence, Tuple

import ggml
from ggml.utils import from_numpy, to_numpy

import numpy as np
import numpy.typing as npt


class InitParams:
    def __init__(
        self,
        mem_size: int = 16 * 1024 * 1024,
        mem_buffer: Optional[ctypes.c_void_p] = None,
        no_alloc: bool = False,
    ):
        self.mem_size = mem_size
        self.mem_buffer = mem_buffer  # NOTE: DO NOT REMOVE THIS
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

    @classmethod
    def with_tensor_overhead(cls):
        return cls(InitParams(mem_size=ggml.ggml_tensor_overhead(), no_alloc=True))


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
        tensor: ggml.ggml_tensor_p,
        ctx: Optional[Context] = None,
        data: Optional[Any] = None,
        src: Optional[List[Tensor]] = None,
    ):
        self.tensor = tensor
        self.ctx = ctx
        self._data = data
        self._src = src or []

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

    @classmethod
    def with_buffer(cls, tensor: ggml.ggml_tensor_p, ctx: Optional[Context] = None, src: Optional[List[Tensor]] = None):
        src = src or []
        if tensor.contents.data is not None:
            return cls(tensor=tensor, ctx=ctx, src=src)
        nbytes = ggml.ggml_nbytes(tensor)
        data = (ctypes.c_uint8 * nbytes)()
        tensor.contents.data = ctypes.cast(data, ctypes.c_void_p)
        return cls(tensor=tensor, ctx=ctx, data=data, src=src)

    def __add__(self, other: Tensor):
        ctx = Context.with_tensor_overhead()
        op = ggml.ggml_add(ctx.context, self.tensor, other.tensor)
        return Tensor.with_buffer(op, ctx, src=[self, other])

    def __sub__(self, other: Tensor):
        ctx = Context.with_tensor_overhead()
        op = ggml.ggml_sub(ctx.context, self.tensor, other.tensor)
        return Tensor.with_buffer(op, ctx, src=[self, other])

    def __mul__(self, other: Tensor):
        ctx = Context.with_tensor_overhead()
        op = ggml.ggml_mul(ctx.context, self.tensor, other.tensor)
        return Tensor.with_buffer(op, ctx, src=[self, other])

    def __truediv__(self, other: Tensor):
        ctx = Context.with_tensor_overhead()
        op = ggml.ggml_div(ctx.context, self.tensor, other.tensor)
        return Tensor.with_buffer(op, ctx, src=[self, other])

    def __neg__(self):
        ctx = Context.with_tensor_overhead()
        op = ggml.ggml_neg(ctx.context, self.tensor)
        return Tensor.with_buffer(op, ctx, src=[self])

    def __abs__(self):
        ctx = Context.with_tensor_overhead()
        op = ggml.ggml_abs(ctx.context, self.tensor)
        return Tensor.with_buffer(op, ctx, src=[self])

    @classmethod
    def with_shape(
        cls, shape: Sequence[int], ggml_type: GGML_TYPE, ctx: Optional[Context] = None
    ):
        ctx = ctx or Context.with_tensor_overhead()
        tensor = ggml.ggml_new_tensor(
            ctx.context,
            ggml_type.value,
            len(shape),
            (ctypes.c_int64 * len(shape))(*shape),
        )
        return cls.with_buffer(tensor, ctx)

    @classmethod
    def from_numpy(cls, x: npt.NDArray[Any], ctx: Optional[Context] = None):
        _ctx = ctx or Context.with_tensor_overhead()
        tensor = from_numpy(x, _ctx.context)
        obj = cls.with_buffer(tensor=tensor, ctx=_ctx)
        if ctx is None:
            obj.numpy()[:] = x
        return obj

    @staticmethod
    def new_tensor(
        ggml_type: GGML_TYPE = GGML_TYPE.F32,
        shape: Sequence[int] = (),
        ctx: Optional[Context] = None,
    ):
        ctx = ctx or Context.with_tensor_overhead()
        tensor = ggml.ggml_new_tensor(
            ctx.context,
            ggml_type.value,
            len(shape),
            (ctypes.c_int64 * len(shape))(*shape),
        )
        return Tensor.with_buffer(tensor=tensor, ctx=ctx)

    @staticmethod
    def new_tensor_1d(
        ggml_type: GGML_TYPE = GGML_TYPE.F32,
        ne0: int = 0,
        ctx: Optional[Context] = None,
    ):
        ctx = ctx or Context.with_tensor_overhead()
        tensor = ggml.ggml_new_tensor_1d(
            ctx.context,
            ggml_type.value,
            ne0,
        )
        return Tensor.with_buffer(tensor=tensor, ctx=ctx)

    @staticmethod
    def new_tensor_2d(
        ggml_type: GGML_TYPE = GGML_TYPE.F32,
        ne0: int = 0,
        ne1: int = 0,
        ctx: Optional[Context] = None,
    ):
        ctx = ctx or Context.with_tensor_overhead()
        tensor = ggml.ggml_new_tensor_2d(
            ctx.context,
            ggml_type.value,
            ne0,
            ne1,
        )
        return Tensor.with_buffer(tensor=tensor, ctx=ctx)

    @staticmethod
    def new_tensor_3d(
        ggml_type: GGML_TYPE = GGML_TYPE.F32,
        ne0: int = 0,
        ne1: int = 0,
        ne2: int = 0,
        ctx: Optional[Context] = None,
    ):
        ctx = ctx or Context.with_tensor_overhead()
        tensor = ggml.ggml_new_tensor_3d(
            ctx.context,
            ggml_type.value,
            ne0,
            ne1,
            ne2,
        )
        return Tensor.with_buffer(tensor=tensor, ctx=ctx)

    @staticmethod
    def new_tensor_4d(
        ggml_type: GGML_TYPE = GGML_TYPE.F32,
        ne0: int = 0,
        ne1: int = 0,
        ne2: int = 0,
        ne3: int = 0,
        ctx: Optional[Context] = None,
    ):
        ctx = ctx or Context.with_tensor_overhead()
        tensor = ggml.ggml_new_tensor_4d(
            ctx.context,
            ggml_type.value,
            ne0,
            ne1,
            ne2,
            ne3,
        )
        return Tensor.with_buffer(tensor=tensor, ctx=ctx)

    @staticmethod
    def new_i32(
        value: int,
        ctx: Optional[Context] = None,
    ):
        ctx = ctx or Context.with_tensor_overhead()
        tensor = ggml.ggml_new_i32(ctx.context, value)
        return Tensor.with_buffer(tensor=tensor, ctx=ctx)

    @staticmethod
    def new_f32(
        value: float,
        ctx: Optional[Context] = None,
    ):
        ctx = ctx or Context.with_tensor_overhead()
        tensor = ggml.ggml_new_f32(ctx.context, value)
        return Tensor.with_buffer(tensor=tensor, ctx=ctx)

    @staticmethod
    def dup_tensor(a: Tensor, ctx: Optional[Context] = None):
        ctx = ctx or Context.with_tensor_overhead()
        op = ggml.ggml_dup_tensor(ctx.context, a.tensor)
        return Tensor.with_buffer(tensor=op, ctx=ctx, src=[a])

    @staticmethod
    def view(a: Tensor, ctx: Optional[Context] = None):
        ctx = ctx or Context.with_tensor_overhead()
        op = ggml.ggml_view_tensor(ctx.context, a.tensor)
        return Tensor.with_buffer(tensor=op, ctx=ctx, src=[a])

    @staticmethod
    def set_zero(a: Tensor):
        op = ggml.ggml_set_zero(a.tensor)
        return Tensor.with_buffer(tensor=op, ctx=a.ctx)

    @staticmethod
    def set_i32(a: Tensor, value: int):
        op = ggml.ggml_set_i32(a.tensor, value)
        return Tensor.with_buffer(tensor=op, ctx=a.ctx)

    @staticmethod
    def get_i32_1d(a: Tensor, i: int):
        return ggml.ggml_get_i32_1d(a.tensor, i)

    @staticmethod
    def set_f32(a: Tensor, i: int, value: float):
        op = ggml.ggml_set_f32_1d(a.tensor, i, value)
        return Tensor(tensor=op, ctx=a.ctx)

    @staticmethod
    def get_f32_1d(a: Tensor, i: int):
        return ggml.ggml_get_f32_1d(a.tensor, i)

    @staticmethod
    def set_f32_1d(a: Tensor, i: int, value: float):
        op = ggml.ggml_set_f32_1d(a.tensor, i, value)
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
        ctx = ctx or Context.with_tensor_overhead()
        op = ggml.ggml_dup(ctx.context, a.tensor)
        return Tensor.with_buffer(tensor=op, ctx=ctx, src=[a])

    @staticmethod
    def add(a: Tensor, b: Tensor, ctx: Optional[Context] = None):
        ctx = ctx or Context.with_tensor_overhead()
        op = ggml.ggml_add(ctx.context, a.tensor, b.tensor)
        return Tensor.with_buffer(tensor=op, ctx=ctx, src=[a, b])

    @staticmethod
    def add_inplace(a: Tensor, b: Tensor, ctx: Optional[Context] = None):
        ctx = ctx or Context.with_tensor_overhead()
        op = ggml.ggml_add_inplace(ctx.context, a.tensor, b.tensor)
        return Tensor.with_buffer(tensor=op, ctx=ctx, src=[a, b])

    @staticmethod
    def add1(a: Tensor, b: Tensor, ctx: Optional[Context] = None):
        ctx = ctx or Context.with_tensor_overhead()
        op = ggml.ggml_add1(ctx.context, a.tensor, b.tensor)
        return Tensor.with_buffer(tensor=op, ctx=ctx, src=[a, b])

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
        ctx = ctx or Context.with_tensor_overhead()
        op = ggml.ggml_acc(
            ctx.context,
            a.tensor,
            b.tensor,
            nb1,
            nb2,
            nb3,
            offset,
        )
        return Tensor.with_buffer(tensor=op, ctx=ctx, src=[a, b])

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
        ctx = ctx or Context.with_tensor_overhead()
        op = ggml.ggml_acc_inplace(
            ctx.context,
            a.tensor,
            b.tensor,
            nb1,
            nb2,
            nb3,
            offset,
        )
        return Tensor.with_buffer(tensor=op, ctx=ctx, src=[a, b])

    @staticmethod
    def sub(a: Tensor, b: Tensor, ctx: Optional[Context] = None):
        ctx = ctx or Context.with_tensor_overhead()
        op = ggml.ggml_sub(ctx.context, a.tensor, b.tensor)
        return Tensor.with_buffer(tensor=op, ctx=ctx, src=[a, b])

    @staticmethod
    def mul(a: Tensor, b: Tensor, ctx: Optional[Context] = None):
        ctx = ctx or Context.with_tensor_overhead()
        op = ggml.ggml_mul(ctx.context, a.tensor, b.tensor)
        return Tensor.with_buffer(tensor=op, ctx=ctx, src=[a, b])

    @staticmethod
    def div(a: Tensor, b: Tensor, ctx: Optional[Context] = None):
        ctx = ctx or Context.with_tensor_overhead()
        op = ggml.ggml_div(ctx.context, a.tensor, b.tensor)
        return Tensor.with_buffer(tensor=op, ctx=ctx, src=[a, b])

    @staticmethod
    def sqr(a: Tensor, ctx: Optional[Context] = None):
        ctx = ctx or Context.with_tensor_overhead()
        op = ggml.ggml_sqr(ctx.context, a.tensor)
        return Tensor.with_buffer(tensor=op, ctx=ctx, src=[a])

    @staticmethod
    def sqrt(a: Tensor, ctx: Optional[Context] = None):
        ctx = ctx or Context.with_tensor_overhead()
        op = ggml.ggml_sqrt(ctx.context, a.tensor)
        return Tensor.with_buffer(tensor=op, ctx=ctx, src=[a])

    @staticmethod
    def log(a: Tensor, ctx: Optional[Context] = None):
        ctx = ctx or Context.with_tensor_overhead()
        op = ggml.ggml_log(ctx.context, a.tensor)
        return Tensor.with_buffer(tensor=op, ctx=ctx, src=[a])

    @staticmethod
    def log_inplace(a: Tensor, ctx: Optional[Context] = None):
        ctx = ctx or Context.with_tensor_overhead()
        op = ggml.ggml_log_inplace(ctx.context, a.tensor)
        return Tensor.with_buffer(tensor=op, ctx=ctx, src=[a])

    @staticmethod
    def sum(a: Tensor, ctx: Optional[Context] = None):
        ctx = ctx or Context.with_tensor_overhead()
        op = ggml.ggml_sum(ctx.context, a.tensor)
        return Tensor.with_buffer(tensor=op, ctx=ctx, src=[a])

    @staticmethod
    def sum_rows(a: Tensor, ctx: Optional[Context] = None):
        ctx = ctx or Context.with_tensor_overhead()
        op = ggml.ggml_sum_rows(ctx.context, a.tensor)
        return Tensor.with_buffer(tensor=op, ctx=ctx, src=[a])

    @staticmethod
    def mean(a: Tensor, ctx: Optional[Context] = None):
        ctx = ctx or Context.with_tensor_overhead()
        op = ggml.ggml_mean(ctx.context, a.tensor)
        return Tensor.with_buffer(tensor=op, ctx=ctx, src=[a])

    @staticmethod
    def repeat(a: Tensor, b: Tensor, ctx: Optional[Context] = None):
        ctx = ctx or Context.with_tensor_overhead()
        op = ggml.ggml_repeat(ctx.context, a.tensor, b.tensor)
        return Tensor.with_buffer(tensor=op, ctx=ctx, src=[a, b])

    @staticmethod
    def abs(a: Tensor, ctx: Optional[Context] = None):
        ctx = ctx or Context.with_tensor_overhead()
        op = ggml.ggml_abs(ctx.context, a.tensor)
        return Tensor.with_buffer(tensor=op, ctx=ctx, src=[a])

    @staticmethod
    def sgn(a: Tensor, ctx: Optional[Context] = None):
        ctx = ctx or Context.with_tensor_overhead()
        op = ggml.ggml_sgn(ctx.context, a.tensor)
        return Tensor.with_buffer(tensor=op, ctx=ctx, src=[a])

    @staticmethod
    def neg(a: Tensor, ctx: Optional[Context] = None):
        ctx = ctx or Context.with_tensor_overhead()
        op = ggml.ggml_neg(ctx.context, a.tensor)
        return Tensor.with_buffer(tensor=op, ctx=ctx, src=[a])

    @staticmethod
    def step(a: Tensor, ctx: Optional[Context] = None):
        ctx = ctx or Context.with_tensor_overhead()
        op = ggml.ggml_step(ctx.context, a.tensor)
        return Tensor.with_buffer(tensor=op, ctx=ctx, src=[a])

    @staticmethod
    def relu(a: Tensor, ctx: Optional[Context] = None):
        ctx = ctx or Context.with_tensor_overhead()
        op = ggml.ggml_relu(ctx.context, a.tensor)
        return Tensor.with_buffer(tensor=op, ctx=ctx, src=[a])

    @staticmethod
    def gelu(a: Tensor, ctx: Optional[Context] = None):
        ctx = ctx or Context.with_tensor_overhead()
        op = ggml.ggml_gelu(ctx.context, a.tensor)
        return Tensor.with_buffer(tensor=op, ctx=ctx, src=[a])

    @staticmethod
    def silu(a: Tensor, ctx: Optional[Context] = None):
        ctx = ctx or Context.with_tensor_overhead()
        op = ggml.ggml_silu(ctx.context, a.tensor)
        return Tensor.with_buffer(tensor=op, ctx=ctx, src=[a])

    @staticmethod
    def silu_back(a: Tensor, b: Tensor, ctx: Optional[Context] = None):
        ctx = ctx or Context.with_tensor_overhead()
        op = ggml.ggml_silu_back(ctx.context, a.tensor, b.tensor)
        return Tensor.with_buffer(tensor=op, ctx=ctx, src=[a, b])

    @staticmethod
    def norm(a: Tensor, eps: float, ctx: Optional[Context] = None):
        ctx = ctx or Context.with_tensor_overhead()
        op = ggml.ggml_norm(ctx.context, a.tensor, eps)
        return Tensor.with_buffer(tensor=op, ctx=ctx, src=[a])

    @staticmethod
    def rms_norm(a: Tensor,  eps: float, ctx: Optional[Context] = None):
        ctx = ctx or Context.with_tensor_overhead()
        op = ggml.ggml_rms_norm(ctx.context, a.tensor, eps)
        return Tensor.with_buffer(tensor=op, ctx=ctx, src=[a])

    @staticmethod
    def rms_norm_back(a: Tensor, b: Tensor, eps: float, ctx: Optional[Context] = None):
        ctx = ctx or Context.with_tensor_overhead()
        op = ggml.ggml_rms_norm_back(ctx.context, a.tensor, b.tensor, eps)
        return Tensor.with_buffer(tensor=op, ctx=ctx, src=[a, b])

    @staticmethod
    def mul_mat(a: Tensor, b: Tensor, ctx: Optional[Context] = None):
        ctx = ctx or Context.with_tensor_overhead()
        op = ggml.ggml_mul_mat(ctx.context, a.tensor, b.tensor)
        return Tensor.with_buffer(tensor=op, ctx=ctx, src=[a, b])

    @staticmethod
    def scale(a: Tensor, b: Tensor, ctx: Optional[Context] = None):
        ctx = ctx or Context.with_tensor_overhead()
        op = ggml.ggml_scale(ctx.context, a.tensor, b.tensor)
        return Tensor.with_buffer(tensor=op, ctx=ctx, src=[a, b])

    @staticmethod
    def scale_inplace(a: Tensor, b: Tensor, ctx: Optional[Context] = None):
        ctx = ctx or Context.with_tensor_overhead()
        op = ggml.ggml_scale_inplace(ctx.context, a.tensor, b.tensor)
        return Tensor.with_buffer(tensor=op, ctx=ctx, src=[a, b])

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
        ctx = ctx or Context.with_tensor_overhead()
        op = ggml.ggml_set(
            ctx.context,
            a.tensor,
            b.tensor,
            nb1,
            nb2,
            nb3,
            offset,
        )
        return Tensor.with_buffer(tensor=op, ctx=ctx, src=[a, b])

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
        ctx = ctx or Context.with_tensor_overhead()
        op = ggml.ggml_set_inplace(
            ctx.context,
            a.tensor,
            b.tensor,
            nb1,
            nb2,
            nb3,
            offset,
        )
        return Tensor.with_buffer(tensor=op, ctx=ctx, src=[a, b])

    @staticmethod
    def set_1d(a: Tensor, b: Tensor, offset: int, ctx: Optional[Context] = None):
        ctx = ctx or Context.with_tensor_overhead()
        op = ggml.ggml_set_1d(ctx.context, a.tensor, b.tensor, offset)
        return Tensor.with_buffer(tensor=op, ctx=ctx, src=[a, b])

    @staticmethod
    def set_1d_inplace(
        a: Tensor, b: Tensor, offset: int, ctx: Optional[Context] = None
    ):
        ctx = ctx or Context.with_tensor_overhead()
        op = ggml.ggml_set_1d_inplace(ctx.context, a.tensor, b.tensor, offset)
        return Tensor.with_buffer(tensor=op, ctx=ctx, src=[a, b])

    @staticmethod
    def set_2d(
        a: Tensor,
        b: Tensor,
        nb1: int,
        offset: int,
        ctx: Optional[Context] = None,
    ):
        ctx = ctx or Context.with_tensor_overhead()
        op = ggml.ggml_set_2d(
            ctx.context,
            a.tensor,
            b.tensor,
            nb1,
            offset,
        )
        return Tensor.with_buffer(tensor=op, ctx=ctx, src=[a, b])

    @staticmethod
    def set_2d_inplace(
        a: Tensor,
        b: Tensor,
        nb1: int,
        offset: int,
        ctx: Optional[Context] = None,
    ):
        ctx = ctx or Context.with_tensor_overhead()
        op = ggml.ggml_set_2d_inplace(
            ctx.context,
            a.tensor,
            b.tensor,
            nb1,
            offset,
        )
        return Tensor.with_buffer(tensor=op, ctx=ctx, src=[a, b])

    @staticmethod
    def cpy(a: Tensor, b: Tensor, ctx: Optional[Context] = None):
        ctx = ctx or Context.with_tensor_overhead()
        op = ggml.ggml_cpy(ctx.context, a.tensor, b.tensor)
        return Tensor.with_buffer(tensor=op, ctx=ctx, src=[a, b])

    @staticmethod
    def cont(a: Tensor, ctx: Optional[Context] = None):
        ctx = ctx or Context.with_tensor_overhead()
        op = ggml.ggml_cont(ctx.context, a.tensor)
        return Tensor.with_buffer(tensor=op, ctx=ctx, src=[a])

    @staticmethod
    def reshape(a: Tensor, b: Tensor, ctx: Optional[Context] = None):
        ctx = ctx or Context.with_tensor_overhead()
        op = ggml.ggml_reshape(ctx.context, a.tensor, b.tensor)
        return Tensor.with_buffer(tensor=op, ctx=ctx, src=[a, b])

    @staticmethod
    def reshape_1d(a: Tensor, ne0: int, ctx: Optional[Context] = None):
        ctx = ctx or Context.with_tensor_overhead()
        op = ggml.ggml_reshape_1d(ctx.context, a.tensor, ne0)
        return Tensor.with_buffer(tensor=op, ctx=ctx, src=[a])

    @staticmethod
    def reshape_2d(a: Tensor, ne0: int, ne1: int, ctx: Optional[Context] = None):
        ctx = ctx or Context.with_tensor_overhead()
        op = ggml.ggml_reshape_2d(ctx.context, a.tensor, ne0, ne1)
        return Tensor.with_buffer(tensor=op, ctx=ctx, src=[a])

    @staticmethod
    def reshape_3d(
        a: Tensor, ne0: int, ne1: int, ne2: int, ctx: Optional[Context] = None
    ):
        ctx = ctx or Context.with_tensor_overhead()
        op = ggml.ggml_reshape_3d(
            ctx.context,
            a.tensor,
            ne0,
            ne1,
            ne2,
        )
        return Tensor.with_buffer(tensor=op, ctx=ctx, src=[a])

    @staticmethod
    def reshape_4d(
        a: Tensor,
        ne0: int,
        ne1: int,
        ne2: int,
        ne3: int,
        ctx: Optional[Context] = None,
    ):
        ctx = ctx or Context.with_tensor_overhead()
        op = ggml.ggml_reshape_4d(
            ctx.context,
            a.tensor,
            ne0,
            ne1,
            ne2,
            ne3,
        )
        return Tensor.with_buffer(tensor=op, ctx=ctx, src=[a])

    @staticmethod
    def view_1d(
        a: Tensor,
        ne0: int,
        offset: int,
        ctx: Optional[Context] = None,
    ):
        ctx = ctx or Context.with_tensor_overhead()
        op = ggml.ggml_view_1d(ctx.context, a.tensor, ne0, offset)
        return Tensor.with_buffer(tensor=op, ctx=ctx, src=[a])

    @staticmethod
    def view_2d(
        a: Tensor,
        ne0: int,
        ne1: int,
        nb1: int,
        offset: int,
        ctx: Optional[Context] = None,
    ):
        ctx = ctx or Context.with_tensor_overhead()
        op = ggml.ggml_view_2d(
            ctx.context,
            a.tensor,
            ne0,
            ne1,
            nb1,
            offset,
        )
        return Tensor.with_buffer(tensor=op, ctx=ctx, src=[a])

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
        ctx = ctx or Context.with_tensor_overhead()
        op = ggml.ggml_view_3d(
            ctx.context,
            a.tensor,
            ne0,
            ne1,
            ne2,
            nb1,
            nb2,
            offset,
        )
        return Tensor.with_buffer(tensor=op, ctx=ctx, src=[a])

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
        ctx = ctx or Context.with_tensor_overhead()
        op = ggml.ggml_view_4d(
            ctx.context,
            a.tensor,
            ne0,
            ne1,
            ne2,
            ne3,
            nb1,
            nb2,
            nb3,
            offset,
        )
        return Tensor.with_buffer(tensor=op, ctx=ctx, src=[a])

    @staticmethod
    def permute(
        a: Tensor,
        axis0: int,
        axis1: int,
        axis2: int,
        axis3: int,
        ctx: Optional[Context] = None,
    ):
        ctx = ctx or Context.with_tensor_overhead()
        op = ggml.ggml_permute(
            ctx.context,
            a.tensor,
            axis0,
            axis1,
            axis2,
            axis3,
        )
        return Tensor.with_buffer(tensor=op, ctx=ctx, src=[a])

    @staticmethod
    def transpose(a: Tensor, ctx: Optional[Context] = None):
        ctx = ctx or Context.with_tensor_overhead()
        op = ggml.ggml_transpose(ctx.context, a.tensor)
        return Tensor.with_buffer(tensor=op, ctx=ctx, src=[a])

    @staticmethod
    def get_rows(a: Tensor, b: Tensor, ctx: Optional[Context] = None):
        ctx = ctx or Context.with_tensor_overhead()
        op = ggml.ggml_get_rows(ctx.context, a.tensor, b.tensor)
        return Tensor.with_buffer(tensor=op, ctx=ctx, src=[a, b])

    @staticmethod
    def get_rows_back(a: Tensor, b: Tensor, c: Tensor, ctx: Optional[Context] = None):
        ctx = ctx or Context.with_tensor_overhead()
        op = ggml.ggml_get_rows_back(ctx.context, a.tensor, b.tensor, c.tensor)
        return Tensor.with_buffer(tensor=op, ctx=ctx, src=[a, b, c])

    @staticmethod
    def diag(a: Tensor, ctx: Optional[Context] = None):
        ctx = ctx or Context.with_tensor_overhead()
        op = ggml.ggml_diag(ctx.context, a.tensor)
        return Tensor.with_buffer(tensor=op, ctx=ctx, src=[a])

    @staticmethod
    def diag_mask_inf(a: Tensor, n_past: int, ctx: Optional[Context] = None):
        ctx = ctx or Context.with_tensor_overhead()
        op = ggml.ggml_diag_mask_inf(ctx.context, a.tensor, n_past)
        return Tensor.with_buffer(tensor=op, ctx=ctx, src=[a])

    @staticmethod
    def diag_mask_inf_inplace(a: Tensor, n_past: int, ctx: Optional[Context] = None):
        ctx = ctx or Context.with_tensor_overhead()
        op = ggml.ggml_diag_mask_inf_inplace(ctx.context, a.tensor, n_past)
        return Tensor.with_buffer(tensor=op, ctx=ctx, src=[a])

    @staticmethod
    def diag_mask_zero(a: Tensor, n_past: int, ctx: Optional[Context] = None):
        ctx = ctx or Context.with_tensor_overhead()
        op = ggml.ggml_diag_mask_zero(ctx.context, a.tensor, n_past)
        return Tensor.with_buffer(tensor=op, ctx=ctx, src=[a])

    @staticmethod
    def diag_mask_zero_inplace(a: Tensor, n_past: int, ctx: Optional[Context] = None):
        ctx = ctx or Context.with_tensor_overhead()
        op = ggml.ggml_diag_mask_zero_inplace(ctx.context, a.tensor, n_past)
        return Tensor.with_buffer(tensor=op, ctx=ctx, src=[a])

    @staticmethod
    def soft_max(a: Tensor, ctx: Optional[Context] = None):
        ctx = ctx or Context.with_tensor_overhead()
        op = ggml.ggml_soft_max(ctx.context, a.tensor)
        return Tensor.with_buffer(tensor=op, ctx=ctx, src=[a])

    @staticmethod
    def soft_max_inplace(a: Tensor, ctx: Optional[Context] = None):
        ctx = ctx or Context.with_tensor_overhead()
        op = ggml.ggml_soft_max_inplace(ctx.context, a.tensor)
        return Tensor.with_buffer(tensor=op, ctx=ctx, src=[a])

    @staticmethod
    def rope(
        a: Tensor,
        b: Tensor,
        n_dims: int,
        mode: int,
        n_ctx: int,
        ctx: Optional[Context] = None,
    ):
        ctx = ctx or Context.with_tensor_overhead()
        op = ggml.ggml_rope(
            ctx.context,
            a.tensor,
            b.tensor,
            n_dims,
            mode,
            n_ctx,
        )
        return Tensor.with_buffer(tensor=op, ctx=ctx, src=[a])

    @staticmethod
    def rope_inplace(
        a: Tensor,
        b: Tensor,
        n_dims: int,
        mode: int,
        n_ctx: int,
        ctx: Optional[Context] = None,
    ):
        ctx = ctx or Context.with_tensor_overhead()
        op = ggml.ggml_rope_inplace(ctx.context, a.tensor, b.tensor, n_dims, mode, n_ctx)
        return Tensor.with_buffer(tensor=op, ctx=ctx, src=[a])

    @staticmethod
    def rope_back(
        a: Tensor,
        b: Tensor,
        n_dims: int,
        mode: int,
        n_ctx: int,
        freq_base: float,
        freq_scale: float,
        xpos_base: float,
        xpos_down: bool,
        ctx: Optional[Context] = None,
    ):
        ctx = ctx or Context.with_tensor_overhead()
        op = ggml.ggml_rope_back(
            ctx.context,
            a.tensor,
            b.tensor,
            n_dims,
            mode,
            n_ctx,
            freq_base,
            freq_scale,
            xpos_base,
            xpos_down,
        )
        return Tensor.with_buffer(tensor=op, ctx=ctx, src=[a])

    @staticmethod
    def alibi(
        a: Tensor,
        n_past: int,
        n_head: int,
        bias_max: float,
        ctx: Optional[Context] = None,
    ):
        ctx = ctx or Context.with_tensor_overhead()
        op = ggml.ggml_alibi(
            ctx.context,
            a.tensor,
            n_past,
            n_head,
            bias_max,
        )
        return Tensor.with_buffer(tensor=op, ctx=ctx, src=[a])

    @staticmethod
    def clamp(a: Tensor, min: float, max: float, ctx: Optional[Context] = None):
        ctx = ctx or Context.with_tensor_overhead()
        op = ggml.ggml_clamp(ctx.context, a.tensor, min, max)
        return Tensor.with_buffer(tensor=op, ctx=ctx, src=[a])

    @staticmethod
    def conv_1d(
        a: Tensor,
        b: Tensor,
        s0: int,
        p0: int,
        d0: int,
        ctx: Optional[Context] = None,
    ):
        ctx = ctx or Context.with_tensor_overhead()
        op = ggml.ggml_conv_1d(ctx.context, a.tensor, b.tensor, s0, p0, d0)
        return Tensor.with_buffer(tensor=op, ctx=ctx, src=[a, b])

    @staticmethod
    def conv_2d(
        a: Tensor,
        b: Tensor,
        s0: int,
        s1: int,
        p0: int,
        p1: int,
        d0: int,
        d1: int,
        ctx: Optional[Context] = None,
    ):
        ctx = ctx or Context.with_tensor_overhead()
        op = ggml.ggml_conv_2d(ctx.context, a.tensor, b.tensor, s0, s1, p0, p1, d0, d1)
        return Tensor.with_buffer(tensor=op, ctx=ctx, src=[a, b])

    @staticmethod
    def flash_attn(
        q: Tensor,
        k: Tensor,
        v: Tensor,
        masked: bool,
        ctx: Optional[Context] = None,
    ):
        ctx = ctx or Context.with_tensor_overhead()
        op = ggml.ggml_flash_attn(
            ctx.context,
            q.tensor,
            k.tensor,
            v.tensor,
            ctypes.c_bool(masked),
        )
        return Tensor.with_buffer(tensor=op, ctx=ctx, src=[q, k, v])

    @staticmethod
    def flash_ff(
        a: Tensor,
        b0: Tensor,
        b1: Tensor,
        c0: Tensor,
        c1: Tensor,
        ctx: Optional[Context] = None,
    ):
        ctx = ctx or Context.with_tensor_overhead()
        op = ggml.ggml_flash_ff(
            ctx.context, a.tensor, b0.tensor, b1.tensor, c0.tensor, c1.tensor
        )
        return Tensor.with_buffer(tensor=op, ctx=ctx, src=[a, b0, b1, c0, c1])

    # TODO: fix type signature
    @staticmethod
    def map_unary_f32(
        a: Tensor,
        fun: Callable[[float], float],
        ctx: Optional[Context] = None,
    ):
        ctx = ctx or Context.with_tensor_overhead()
        op = ggml.ggml_map_unary_f32(  # type: ignore
            ctx.context, a.tensor, ctypes.CFUNCTYPE(ctypes.c_float, ctypes.c_float)(fun)
        )
        return Tensor.with_buffer(tensor=op, ctx=ctx, src=[a])

    # TODO: fix type signature
    @staticmethod
    def map_binary_f32(
        a: Tensor,
        b: Tensor,
        fun: Callable[[float, float], float],
        ctx: Optional[Context] = None,
    ):
        ctx = ctx or Context.with_tensor_overhead()
        op = ggml.ggml_map_binary_f32(  # type: ignore
            ctx.context,
            a.tensor,
            b.tensor,
            ctypes.CFUNCTYPE(ctypes.c_float, ctypes.c_float, ctypes.c_float)(fun),
        )
        return Tensor.with_buffer(tensor=op, ctx=ctx, src=[a, b])

    @staticmethod
    def set_param(
        a: Tensor,
        ctx: Optional[Context] = None,
    ):
        ctx = ctx or Context.with_tensor_overhead()
        op = ggml.ggml_set_param(ctx.context, a.tensor)
        return Tensor.with_buffer(tensor=op, ctx=ctx)

    @staticmethod
    def ggml_ftype_to_ggml_type(ftype: int):
        return GGML_TYPE(ggml.ggml_ftype_to_ggml_type(ctypes.c_int(ftype)))


class CGraph:
    def __init__(self, cgraph: ggml.ggml_cgraph):
        self.cgraph = cgraph
        self._tensors: List[Tensor] = []

    def compute(self, n_threads: int = 1):
        gp = ggml.ggml_graph_plan(ctypes.pointer(self.cgraph), n_threads=n_threads)
        if gp.work_size > 0:
            work_data = (ctypes.c_uint8 * gp.work_size)()
            gp.work_data = ctypes.cast(work_data, ctypes.c_void_p)
            ggml.ggml_graph_compute(ctypes.pointer(self.cgraph), ctypes.pointer(gp))
        else:
            ggml.ggml_graph_compute(ctypes.pointer(self.cgraph), ctypes.pointer(gp))

    def reset(self):
        ggml.ggml_graph_reset(ctypes.pointer(self.cgraph))
        self._tensors = []

    def get_tensor(self, name: bytes):
        return Tensor(
            tensor=ggml.ggml_graph_get_tensor(ctypes.pointer(self.cgraph), name),
        )

    def graph_export(self, fname: bytes):
        ggml.ggml_graph_export(ctypes.pointer(self.cgraph), fname)

    def build_forward_expand(self, tensor: Tensor):
        ggml.ggml_build_forward_expand(ctypes.pointer(self.cgraph), tensor.tensor)
        self._tensors.append(tensor)

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
        obj = CGraph(cgraph=ggml.ggml_build_forward(tensor.tensor))
        obj._tensors.append(tensor)
        return obj
