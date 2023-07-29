__all__ = ['TensorError', 'Tensor']

import os
from typing import Self

import numpy as np

from .defs import TensorData
# from ..op import *

DEBUG = int(os.getenv('DEBUG') or '0', 0)

class TensorError(Exception):
    pass

class Tensor:
    def __init__(self, data: TensorData, *, dtype=np.float32, requires_grad: bool=False):
        # NOTE: subclass should set its internal representation of data
        self.data = None
        # NOTE: subclass should set shape based on data
        self._shape = None
        self._dtype = dtype
        self.requires_grad = requires_grad
        self._backend = None
        self._grad = None
        self._grad_fn = None
        # self._op = self._backend.SetOp(data)

    def __repr__(self):
        if DEBUG == 0:
            return object.__repr__(self)
        
        items = []
        items.append(type(self).__name__)
        items.append('(')

        subitems = []

        if DEBUG > 1:
            subitems.append(f'data={self.data}')

        subitems.append(f'shape={self.shape}')

        if self.requires_grad:
            subitems.append(f'requires_grad={self.requires_grad}')

        if self._backend:
            subitems.append(f'_backend={self._backend}')

        if self._grad:
            subitems.append(f'_grad={self._grad}')

        if self._grad_fn:
            subitems.append(f'_grad_fn={self._grad_fn}')

        items.append(' '.join(subitems))
        items.append(')')
        return ''.join(items)

    def __pos__(self) -> Self:
        raise NotImplementedError('__pos__')

    def __neg__(self) -> Self:
        raise NotImplementedError('__neg__')

    def __add__(self, other: TensorData) -> Self:
        raise NotImplementedError('__add__')

    def __radd__(self, other: TensorData) -> Self:
        raise NotImplementedError('__radd__')

    def __sub__(self, other: TensorData) -> Self:
        raise NotImplementedError('__sub__')

    def __rsub__(self, other: TensorData) -> Self:
        raise NotImplementedError('__rsub__')

    def __mul__(self, other: TensorData) -> Self:
        raise NotImplementedError('__mul__')

    def __rmul__(self, other: TensorData) -> Self:
        raise NotImplementedError('__rmul__')

    def __truediv__(self, other: TensorData) -> Self:
        raise NotImplementedError('__truediv__')

    def div(self, other: TensorData) -> Self:
        raise NotImplementedError('div')

    def __rtruediv__(self, other: TensorData) -> Self:
        raise NotImplementedError('__rtruediv__')

    def __floordiv__(self, other: TensorData) -> Self:
        raise NotImplementedError('__floordiv__')

    def __rfloordiv__(self, other: TensorData) -> Self:
        raise NotImplementedError('__rfloordiv__')

    def __pow__(self, other: TensorData) -> Self:
        raise NotImplementedError('__pow__')

    def pow(self, other: TensorData) -> Self:
        raise NotImplementedError('pow')

    def __rpow__(self, other: TensorData) -> Self:
        raise NotImplementedError('__rpow__')

    def __matmul__(self, other: TensorData) -> Self:
        raise NotImplementedError('__matmul__')

    def matmul(self, other: TensorData) -> Self:
        raise NotImplementedError('matmul')

    def __eq__(self, other: Self) -> Self:
        raise NotImplementedError('__eq__')

    def __lt__(self, other: Self) -> Self:
        raise NotImplementedError('__lt__')

    def exp(self) -> Self:
        raise NotImplementedError('exp')

    def tanh(self) -> Self:
        raise NotImplementedError('tanh')

    def sigmoid(self) -> Self:
        raise NotImplementedError('sigmoid')

    def relu(self) -> Self:
        raise NotImplementedError('relu')

    @classmethod
    def eye(cls, dim: int, requires_grad: bool=False, dtype=np.float32) -> Self:
        raise NotImplementedError('eye')

    def sum(self, axis=None, keepdims=False):
        raise NotImplementedError('sum')

    def transpose(self, axis0=1, axis1=0) -> Self:
        raise NotImplementedError('transpose')

    @property
    def T(self):
        return self.transpose()

    def numpy(self) -> np.ndarray:
        raise NotImplementedError('numpy')

    @property
    def shape(self) -> tuple:
        return self._shape

    @property
    def dtype(self) -> np.dtype:
        return self._dtype

    @property
    def grad(self):
        return self._grad

    @property
    def grad_fn(self):
        if not self.requires_grad:
            raise TensorError('This tensor is not backpropagated, requires_grad is set to False')

        return self._grad_fn

    def backward(self):
        raise NotImplementedError('backward')

"""    
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

    # Tensor = Tensor.use_backend(NumPyBackend)

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
"""