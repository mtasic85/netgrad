__all__ = ['BaseTensor', 'TensorError']

from typing import Self

import numpy as np

from .defs import TensorData
from .op import Op

class TensorError(Exception):
    pass

class Tensor:
    def __init__(self, data: TensorData, *, dtype=np.float32, requires_grad: bool=False, op: Op=None):
        if not isinstance(data, np.ndarray):
            data = np.array(data, dtype=dtype)

        self.data = data
        self.requires_grad = requires_grad
        self.op = self.AssignOp(operands=(self,)) if op is None else op
        # self.grad = None

    def __repr__(self):
        if DEBUG == 0:
            return object.__repr__(self)
        
        items = []
        items.append(type(self).__name__)
        items.append('(')
        subitems = []

        if COMPACT:
            subitems.append(f'{self.shape}')

            if self.dtype != np.float32:
                subitems.append(f'{self.dtype}')

            if not self.requires_grad:
                subitems.append(f'requires_grad={self.requires_grad}')

            if DEBUG == 1:
                subitems.append(f'{self.op.opcode.name}')
            else:
                subitems.append(f'{self.op}')
        else:
            if DEBUG > 2:
                subitems.append(f'data={self.data}')

            subitems.append(f'shape={self.shape}')

            if self.dtype != np.float32:
                subitems.append(f'dtype={self.dtype}')
            
            if not self.requires_grad:
                subitems.append(f'requires_grad={self.requires_grad}')

            if DEBUG == 1:
                subitems.append(f'op={self.op.opcode.name}')
            elif DEBUG > 1:
                if self.op:
                    subitems.append(f'op={self.op}')

            if DEBUG > 2:
                if self._backend:
                    subitems.append(f'_backend={self._backend}')

        items.append(' '.join(subitems))
        items.append(')')
        return ''.join(items)

    def __hash__(self) -> int:
        return id(self)

    # def zero_grad(self):
    #     raise NotImplementedError
    
    def _build_top_ord(self) -> list[Self]:
        values = []
        visited = set()

        def __build_top_ord(v):
            if v not in visited:
                visited.add(v)

                for operand in v.op.operands:
                    __build_top_ord(operand)

                values.append(v)

        __build_top_ord(self)
        return values

    #
    # specialized operations
    #
    @classmethod
    def eye(cls, dim: int, requires_grad: bool=False, dtype=np.float32) -> Self:
        raise NotImplementedError('eye')

    #
    # unary operatons
    #
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
        return self * other

    def __truediv__(self, other: TensorData) -> Self:
        raise NotImplementedError('__truediv__')

    def div(self, other: TensorData) -> Self:
        return self / other

    def __rtruediv__(self, other: TensorData) -> Self:
        return self / other

    def __pow__(self, other: TensorData) -> Self:
        raise NotImplementedError('__pow__')

    def pow(self, other: TensorData) -> Self:
        return self ** other

    def __rpow__(self, other: TensorData) -> Self:
        return self ** other

    def __matmul__(self, other: TensorData) -> Self:
        raise NotImplementedError('__matmul__')

    def matmul(self, other: TensorData) -> Self:
        return self @ other

    def __eq__(self, other: Self) -> Self:
        raise NotImplementedError('__eq__')

    def __lt__(self, other: Self) -> Self:
        raise NotImplementedError('__lt__')

    #
    # move to device
    #
    def to(self, device: str) -> Self:
        raise NotImplementedError('to')

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
