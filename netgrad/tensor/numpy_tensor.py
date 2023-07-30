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
        res = NPTensor(0.0 - self.data, requires_grad=rd, op=op)
        return res

    def exp(self) -> Self:
        rd = self.requires_grad
        op = self._backend.SumOp((self,))
        res = NPTensor(np.exp(self.data), requires_grad=rd, op=op)
        return res

    def tanh(self) -> Self:
        rd = self.requires_grad
        op = self._backend.TanhOp((self,))
        res = NPTensor(np.tanh(self.data), requires_grad=rd, op=op)
        return res

    def sigmoid(self) -> Self:
        rd = self.requires_grad
        op = self._backend.SigmoidOp((self,))
        res = NPTensor(1.0 / (1.0 + np.exp(-self.data)), requires_grad=rd, op=op)
        return res

    def relu(self) -> Self:
        rd = self.requires_grad
        op = self._backend.ReluOp((self,))
        res = NPTensor(np.maximum(0, self.data), requires_grad=rd, op=op)
        return res

    def sum(self, axis=None, keepdims=False):
        # https://numpy.org/doc/stable/reference/generated/numpy.sum.html#numpy.sum
        rd = self.requires_grad
        op = self._backend.SumOp((self,))
        res = NPTensor(self.data.sum(axis=axis, keepdims=keepdims), requires_grad=rd, op=op)
        return res

    def transpose(self, axis0=1, axis1=0) -> Self:
        rd = self.requires_grad
        op = self._backend.TransposeOp((self,))
        res = NPTensor(self.data.transpose(axis0, axis1), requires_grad=rd, op=op)
        return res

    #
    # binary operations
    #
    def __add__(self, other: TensorData) -> Self:
        if not isinstance(other, NPTensor):
            other = NPTensor(other if isinstance(other, (int, float)) else other.data)

        rd = self.requires_grad or other.requires_grad
        op = self._backend.AddOp((self, other))
        res = NPTensor(self.data + other.data)
        return res

    def __sub__(self, other: TensorData) -> Self:
        if not isinstance(other, NPTensor):
            other = NPTensor(other if isinstance(other, (int, float)) else other.data)

        rd = self.requires_grad or other.requires_grad
        op = self._backend.SubOp((self, other))
        res = NPTensor(self.data - other.data)
        return res

    def __mul__(self, other: TensorData) -> Self:
        if not isinstance(other, NPTensor):
            other = NPTensor(other if isinstance(other, (int, float)) else other.data)

        rd = self.requires_grad or other.requires_grad
        op = self._backend.MulOp((self, other))
        res = NPTensor(self.data * other.data)
        return res

    def __truediv__(self, other: TensorData) -> Self:
        if not isinstance(other, NPTensor):
            other = NPTensor(other if isinstance(other, (int, float)) else other.data)

        rd = self.requires_grad or other.requires_grad
        op = self._backend.DivOp((self, other))
        res = NPTensor(self.data / other.data)
        return res

    def __pow__(self, other: TensorData) -> Self:
        if not isinstance(other, NPTensor):
            other = NPTensor(other if isinstance(other, (int, float)) else other.data)

        rd = self.requires_grad or other.requires_grad
        op = self._backend.PowOp((self, other))
        res = NPTensor(self.data ** other.data)
        return res

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
        rd = self.requires_grad or other.requires_grad
        op = self._backend.MatMulOp((self, other))
        res = NPTensor(np.equal(self.data, other.data)) # dtype=np.bool_
        return res

    def __lt__(self, other: Self) -> Self:
        rd = self.requires_grad or other.requires_grad
        op = self._backend.MatMulOp((self, other))
        res = NPTensor(np.less(self.data, other.data)) # dtype=np.bool_
        return res

    #
    # propagation
    #
    def backward(self):
        pass
