__all__ = ['NPTensor']

from typing import Self

import numpy as np

from ..defs import TensorData
from ..op import Op
from ..backend.numpy_backend import NPBackend
from .base_tensor import TensorError, BaseTensor

class NPTensor(BaseTensor):
    _backend = NPBackend

    def __init__(self, data: TensorData, *, dtype=np.float32, requires_grad: bool=False, op: Op=None):
        super().__init__(data=data, dtype=dtype, requires_grad=requires_grad, op=op)
    
    #
    # specialized operations
    #
    @classmethod
    def eye(cls, dim: int, requires_grad: bool=False, dtype=np.float32) -> Self:
        return NPTensor(np.eye(dim, dtype=dtype), requires_grad=requires_grad)

    #
    # unary operatons
    #
    def __neg__(self) -> Self:
        rd = self.requires_grad
        op = self._backend.NegOp((self,))
        return NPTensor(0.0 - self.data)

    def exp(self) -> Self:
        rd = self.requires_grad
        op = self._backend.SumOp((self,))
        return NPTensor(np.exp(self.data))

    def tanh(self) -> Self:
        rd = self.requires_grad
        op = self._backend.SumOp((self,))
        return NPTensor(np.tanh(self.data))

    def sigmoid(self) -> Self:
        rd = self.requires_grad
        op = self._backend.SumOp((self,))
        return NPTensor(1.0 / (1.0 + np.exp(-self.data)))

    def relu(self) -> Self:
        rd = self.requires_grad
        op = self._backend.SumOp((self,))
        return NPTensor(np.maximum(0, self.data))

    def sum(self, axis=None, keepdims=False):
        # https://numpy.org/doc/stable/reference/generated/numpy.sum.html#numpy.sum
        rd = self.requires_grad
        op = self._backend.SumOp((self,))
        res = NPTensor(self.data.sum(axis=axis, keepdims=keepdims), requires_grad=rd, op=op)
        return res

    def transpose(self, axis0=1, axis1=0) -> Self:
        return NPTensor(self.data.transpose(axis0, axis1))

    #
    # binary operations
    #
    def __add__(self, other: TensorData) -> Self:
        if not isinstance(other, NPTensor):
            other = NPTensor(other if isinstance(other, (int, float)) else other.data)

        return NPTensor(self.data + other.data)

    def __sub__(self, other: TensorData) -> Self:
        if not isinstance(other, NPTensor):
            other = NPTensor(other if isinstance(other, (int, float)) else other.data)

        return NPTensor(self.data - other.data)

    def __mul__(self, other: TensorData) -> Self:
        if not isinstance(other, NPTensor):
            other = NPTensor(other if isinstance(other, (int, float)) else other.data)

        return NPTensor(self.data * other.data)

    def __truediv__(self, other: TensorData) -> Self:
        if not isinstance(other, NPTensor):
            other = NPTensor(other if isinstance(other, (int, float)) else other.data)

        return NPTensor(self.data / other.data)

    def __floordiv__(self, other: TensorData) -> Self:
        if not isinstance(other, NPTensor):
            other = NPTensor(other if isinstance(other, (int, float)) else other.data)

        return NPTensor(self.data // other.data)

    def __pow__(self, other: TensorData) -> Self:
        if not isinstance(other, NPTensor):
            other = NPTensor(other if isinstance(other, (int, float)) else other.data)

        return NPTensor(self.data ** other.data)

    def __matmul__(self, other: TensorData) -> Self:
        if not isinstance(other, NPTensor):
            other = NPTensor(other.data)

        rd = self.requires_grad or other.requires_grad
        op = self._backend.MatMulOp((self, other))
        res = NPTensor(self.data.dot(other.data), requires_grad=rd, op=op)
        return res

    #
    # comparison operations
    #
    def __eq__(self, other: Self) -> Self:
        return NPTensor(np.equal(self.data, other.data)) # dtype=np.bool_

    def __lt__(self, other: Self) -> Self:
        return NPTensor(np.less(self.data, other.data)) # dtype=np.bool_

    #
    # propagation
    #
    def backward(self):
        pass
