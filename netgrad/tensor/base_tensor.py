__all__ = ['TensorError', 'BaseTensor']

import os
from typing import Self

import numpy as np

from ..defs import TensorData
from ..op import *

DEBUG = int(os.getenv('DEBUG') or '0', 0)

class TensorError(Exception):
    pass

class BaseTensor:
    def __init__(self, data: TensorData, *, dtype=np.float32, requires_grad: bool=False, op: Op=None):
        if not isinstance(data, np.ndarray):
            data = np.array(data, dtype=dtype)

        self.data = data
        self.requires_grad = requires_grad
        self._grad = None
        self._grad_fn = None
        self._op = self._backend.SetOp(operands=(self,)) if op is None else op

    def __repr__(self):
        if DEBUG == 0:
            return object.__repr__(self)
        
        items = []
        items.append(type(self).__name__)
        items.append('(')

        subitems = []

        if DEBUG > 2:
            subitems.append(f'data={self.data}')

        subitems.append(f'shape={self.shape}')
        subitems.append(f'dtype={self.dtype}')

        if self.requires_grad:
            subitems.append(f'requires_grad={self.requires_grad}')

        if DEBUG > 3:
            if self._backend:
                subitems.append(f'_backend={self._backend}')

            if self._grad:
                subitems.append(f'_grad={self._grad}')

            if self._grad_fn:
                subitems.append(f'_grad_fn={self._grad_fn}')

        if DEBUG > 1:
            if self._op:
                subitems.append(f'_op={self._op}')

        items.append(' '.join(subitems))
        items.append(')')
        return ''.join(items)
    
    #
    # specialized operations
    #
    @classmethod
    def eye(cls, dim: int, requires_grad: bool=False, dtype=np.float32) -> Self:
        raise NotImplementedError('eye')

    #
    # unary operatons
    #
    def __pos__(self) -> Self:
        raise NotImplementedError('__pos__')

    def __neg__(self) -> Self:
        raise NotImplementedError('__neg__')

    def exp(self) -> Self:
        raise NotImplementedError('exp')

    def tanh(self) -> Self:
        raise NotImplementedError('tanh')

    def sigmoid(self) -> Self:
        raise NotImplementedError('sigmoid')

    def relu(self) -> Self:
        raise NotImplementedError('relu')

    def sum(self, axis=None, keepdims=False):
        raise NotImplementedError('sum')

    def transpose(self, axis0=1, axis1=0) -> Self:
        raise NotImplementedError('transpose')

    @property
    def T(self):
        return self.transpose()

    #
    # binary operations
    #
    def __add__(self, other: TensorData) -> Self:
        raise NotImplementedError('__add__')

    def __radd__(self, other: TensorData) -> Self:
        return self + other

    def __sub__(self, other: TensorData) -> Self:
        raise NotImplementedError('__sub__')

    def __rsub__(self, other: TensorData) -> Self:
        return self - other

    def __mul__(self, other: TensorData) -> Self:
        raise NotImplementedError('__mul__')

    def __rmul__(self, other: TensorData) -> Self:
        raise NotImplementedError('__rmul__')

    def __truediv__(self, other: TensorData) -> Self:
        raise NotImplementedError('__truediv__')

    def div(self, other: TensorData) -> Self:
        return self.__truediv__(other)

    def __rtruediv__(self, other: TensorData) -> Self:
        return self / other

    def __floordiv__(self, other: TensorData) -> Self:
        raise NotImplementedError('__floordiv__')

    def __rfloordiv__(self, other: TensorData) -> Self:
        return self // other

    def __pow__(self, other: TensorData) -> Self:
        raise NotImplementedError('__pow__')

    def pow(self, other: TensorData) -> Self:
        return self.__pow__(other)

    def __rpow__(self, other: TensorData) -> Self:
        return self ** other

    def __matmul__(self, other: TensorData) -> Self:
        raise NotImplementedError('__matmul__')

    def matmul(self, other: TensorData) -> Self:
        return self.__matmul__(other)

    def __eq__(self, other: Self) -> Self:
        raise NotImplementedError('__eq__')

    def __lt__(self, other: Self) -> Self:
        raise NotImplementedError('__lt__')

    #
    # properties
    #
    def numpy(self) -> np.ndarray:
        return self.data

    @property
    def shape(self) -> tuple:
        return self.data.shape

    @property
    def dtype(self) -> np.dtype:
        return self.data.dtype

    @property
    def grad(self):
        return self._grad

    @property
    def grad_fn(self):
        if not self.requires_grad:
            raise TensorError('This tensor is not backpropagated, requires_grad is set to False')

        return self._grad_fn

    #
    # propagation
    #
    def backward(self):
        raise NotImplementedError('backward')
