import os
from enum import Enum, auto
from typing import Self
from functools import wraps

import numpy as np


DEBUG = int(os.getenv('DEBUG') or '0', 0)

TensorData = np.ndarray | tuple | list | int | float


class SOpCode(Enum):
    nop = auto()
    set = auto()
    get = auto()
    eye = auto()

class UnOpCode(Enum):
    pos = auto()
    neg = auto()
    exp = auto()
    tanh = auto()
    sigmoid = auto()
    relu = auto()
    sum = auto()
    transpose = auto()

class BinOpCode(Enum):
    add = auto()
    sub = auto()
    mul = auto()
    div = auto()
    pow = auto()
    matmul = auto()
    eq = auto()
    lt = auto()

OpCode: type = SOpCode | UnOpCode | BinOpCode


class OpError(Exception):
    pass


class Op:
    def __init__(self, opcode: OpCode, operands: list['Tensor'] | tuple['Tensor']=()):
        self.opcode = opcode
        self.operands = operands
        self.requires_grad = any(n.requires_grad for n in operands)

    def forward(self, *args, **kwargs):
        raise NotImplementedError('forward')

    def backward(self, *args, **kwargs):
        raise NotImplementedError('forward')


class NopOp(Op):
    # FIXME: implement, probably
    pass


class SetOp(Op):
    def __init__(self, data: TensorData, **kwargs):
        super().__init__(opcode=SOpCode.set, **kwargs)
        self.data = data

    def forward(self):
        pass

    def backward(self):
        pass


class Backend:
    pass


class NumPyBackend(Backend):
    pass


BACKENDS = {
    'numpy': NumPyBackend,
}


class TensorError(Exception):
    pass


class Tensor:
    def __init__(self, data: TensorData, *, requires_grad: bool=False, dtype=np.float32):
        if not isinstance(data, np.ndarray):
            data = np.array(data, dtype=dtype)
        
        self.data = data
        self.requires_grad = requires_grad
        self._grad = None
        self._grad_fn = None
        self._op = SetOp(data)

    def __repr__(self):
        if DEBUG == 0:
            return object.__repr__(self)
        else:
            items = []
            items.append(type(self).__name__)
            items.append('(')

            subitems = []
            subitems.append(f'data={self.data}')

            if self.requires_grad:
                subitems.append(f'requires_grad={self.requires_grad}')

            if self._grad:
                subitems.append(f'_grad={self._grad}')

            if self._grad_fn:
                subitems.append(f'_grad_fn={self._grad_fn}')

            items.append(' '.join(subitems))
            items.append(')')
            return ''.join(items)

    def __pos__(self) -> Self:
        return Tensor(self.data)

    def __neg__(self) -> Self:
        return Tensor(0.0 - self.data)

    def __add__(self, other: TensorData) -> Self:
        if not isinstance(other, Tensor):
            other = Tensor(other if isinstance(other, (int, float)) else other.data)

        return Tensor(self.data + other.data)

    def __radd__(self, other: TensorData) -> Self:
        return self + other

    def __sub__(self, other: TensorData) -> Self:
        if not isinstance(other, Tensor):
            other = Tensor(other if isinstance(other, (int, float)) else other.data)

        return Tensor(self.data - other.data)

    def __rsub__(self, other: TensorData) -> Self:
        return self - other

    def __mul__(self, other: TensorData) -> Self:
        if not isinstance(other, Tensor):
            other = Tensor(other if isinstance(other, (int, float)) else other.data)

        return Tensor(self.data * other.data)

    def __rmul__(self, other: TensorData) -> Self:
        return self * other

    def __truediv__(self, other: TensorData) -> Self:
        if not isinstance(other, Tensor):
            other = Tensor(other if isinstance(other, (int, float)) else other.data)

        return Tensor(self.data / other.data)

    div = __truediv__

    def __rtruediv__(self, other: TensorData) -> Self:
        return self / other

    def __floordiv__(self, other: TensorData) -> Self:
        if not isinstance(other, Tensor):
            other = Tensor(other if isinstance(other, (int, float)) else other.data)

        return Tensor(self.data // other.data)

    def __rfloordiv__(self, other: TensorData) -> Self:
        return self // other

    def __pow__(self, other: TensorData) -> Self:
        if not isinstance(other, Tensor):
            other = Tensor(other if isinstance(other, (int, float)) else other.data)

        return Tensor(self.data ** other.data)

    pow = __pow__

    def __rpow__(self, other: TensorData) -> Self:
        return self ** other

    def __matmul__(self, other: TensorData) -> Self:
        if not isinstance(other, Tensor):
            other = Tensor(other.data)

        return Tensor(self.data.dot(other.data))

    matmul = __matmul__

    def __eq__(self, other: Self) -> Self:
        return Tensor(np.equal(self.data, other.data)) # dtype=np.bool_

    def __lt__(self, other: Self) -> Self:
        return Tensor(np.less(self.data, other.data)) # dtype=np.bool_

    def exp(self) -> Self:
        return Tensor(np.exp(self.data))

    def tanh(self) -> Self:
        return Tensor(np.tanh(self.data))

    def sigmoid(self) -> Self:
        return Tensor(1.0 / (1.0 + np.exp(-self.data)))

    def relu(self) -> Self:
        return Tensor(np.maximum(0, self.data))

    @classmethod
    def eye(cls, dim: int, requires_grad: bool=False, dtype=np.float32) -> Self:
        return Tensor(np.eye(dim, dtype=dtype), requires_grad=requires_grad)

    def sum(self, axis=None, keepdims=False):
        # https://numpy.org/doc/stable/reference/generated/numpy.sum.html#numpy.sum
        return Tensor(self.data.sum(axis=axis, keepdims=keepdims))

    def transpose(self, axis0=1, axis1=0) -> Self:
        return Tensor(self.data.transpose(axis0, axis1))

    @property
    def T(self):
        return self.transpose()

    def numpy(self) -> np.ndarray:
        return self.data

    @property
    def dtype(self) -> np.dtype:
        return self.data.dtype

    @property
    def shape(self) -> tuple:
        return self.data.shape

    @property
    def grad(self):
        return self._grad

    @property
    def grad_fn(self):
        if not self.requires_grad:
            raise TensorError('This tensor is not backpropagated, requires_grad is set to False')

        return self._grad_fn

    def backward(self):
        pass
    
def demo0():
    a = Tensor([1, 2, 3])
    b = Tensor([2, 3, 4])
    print(a)
    print(b)
    c = a + b
    print(c)

def demo1():
    u = Tensor([[1, 2, 3], [4, 5, 6]])
    v = u.T
    w = v.T
    print(u)
    print(v)
    print(w)

if __name__ == '__main__':
    DEBUG = 1

    x = Tensor.eye(3, requires_grad=True)
    y = Tensor([[2.0,0,-2.0]], requires_grad=True)
    z = y.matmul(x)
    w = z.sum()
    w.backward()

    print(x)
    print(y)
    print(z)
    print(w)

    +w

    # print(x.grad.numpy())  # dw/dx
    # print(y.grad.numpy())  # dw/dy
    # print(z.grad.numpy())  # dw/dz