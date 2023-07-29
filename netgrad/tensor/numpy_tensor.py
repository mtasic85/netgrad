__all__ = ['NPTensor']

from typing import Self

import numpy as np

from ..defs import TensorData
from ..backend.numpy_backend import NPBackend
from .base_tensor import TensorError, BaseTensor

class NPTensor(BaseTensor):
    def __init__(self, data: TensorData, *, dtype=np.float32, requires_grad: bool=False):
        super().__init__(data=data, dtype=dtype, requires_grad=requires_grad)
        self._backend = NPBackend

    def __pos__(self) -> Self:
        return NPTensor(self.data)

    def __neg__(self) -> Self:
        return NPTensor(0.0 - self.data)

    def __add__(self, other: TensorData) -> Self:
        if not isinstance(other, NPTensor):
            other = NPTensor(other if isinstance(other, (int, float)) else other.data)

        return NPTensor(self.data + other.data)

    def __radd__(self, other: TensorData) -> Self:
        return self + other

    def __sub__(self, other: TensorData) -> Self:
        if not isinstance(other, NPTensor):
            other = NPTensor(other if isinstance(other, (int, float)) else other.data)

        return NPTensor(self.data - other.data)

    def __rsub__(self, other: TensorData) -> Self:
        return self - other

    def __mul__(self, other: TensorData) -> Self:
        if not isinstance(other, NPTensor):
            other = NPTensor(other if isinstance(other, (int, float)) else other.data)

        return NPTensor(self.data * other.data)

    def __rmul__(self, other: TensorData) -> Self:
        return self * other

    def __truediv__(self, other: TensorData) -> Self:
        if not isinstance(other, NPTensor):
            other = NPTensor(other if isinstance(other, (int, float)) else other.data)

        return NPTensor(self.data / other.data)

    div = __truediv__

    def __rtruediv__(self, other: TensorData) -> Self:
        return self / other

    def __floordiv__(self, other: TensorData) -> Self:
        if not isinstance(other, NPTensor):
            other = NPTensor(other if isinstance(other, (int, float)) else other.data)

        return NPTensor(self.data // other.data)

    def __rfloordiv__(self, other: TensorData) -> Self:
        return self // other

    def __pow__(self, other: TensorData) -> Self:
        if not isinstance(other, NPTensor):
            other = NPTensor(other if isinstance(other, (int, float)) else other.data)

        return NPTensor(self.data ** other.data)

    pow = __pow__

    def __rpow__(self, other: TensorData) -> Self:
        return self ** other

    def __matmul__(self, other: TensorData) -> Self:
        if not isinstance(other, NPTensor):
            other = NPTensor(other.data)

        return NPTensor(self.data.dot(other.data))

    matmul = __matmul__

    def __eq__(self, other: Self) -> Self:
        return NPTensor(np.equal(self.data, other.data)) # dtype=np.bool_

    def __lt__(self, other: Self) -> Self:
        return NPTensor(np.less(self.data, other.data)) # dtype=np.bool_

    def exp(self) -> Self:
        return NPTensor(np.exp(self.data))

    def tanh(self) -> Self:
        return NPTensor(np.tanh(self.data))

    def sigmoid(self) -> Self:
        return NPTensor(1.0 / (1.0 + np.exp(-self.data)))

    def relu(self) -> Self:
        return NPTensor(np.maximum(0, self.data))

    @classmethod
    def eye(cls, dim: int, requires_grad: bool=False, dtype=np.float32) -> Self:
        return NPTensor(np.eye(dim, dtype=dtype), requires_grad=requires_grad)

    def sum(self, axis=None, keepdims=False):
        # https://numpy.org/doc/stable/reference/generated/numpy.sum.html#numpy.sum
        return NPTensor(self.data.sum(axis=axis, keepdims=keepdims))

    def transpose(self, axis0=1, axis1=0) -> Self:
        return NPTensor(self.data.transpose(axis0, axis1))

    def backward(self):
        pass
