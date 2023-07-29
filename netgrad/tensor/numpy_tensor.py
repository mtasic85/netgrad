__all__ = ['NumPyTensor']

from typing import Self

import numpy as np

from .defs import TensorData
from .tensor import TensorError, Tensor
from ..backend.numpy_backend import NumPyBackend

class NumPyTensor(Tensor):
    def __init__(self, data: TensorData, *, dtype=np.float32, requires_grad: bool=False):
        super().__init__(data=data, dtype=dtype, requires_grad=requires_grad)

        if not isinstance(data, np.ndarray):
            data = np.array(data, dtype=dtype)
            # FIXME: warning, data.dtype and dtype could be different?
            dtype = data.dtype
        
        self.data = data
        self._shape = data.shape
        self._dtype = dtype
        self.requires_grad = requires_grad
        self._backend = NumPyBackend
        self._grad = None
        self._grad_fn = None
        # self._op = self._backend.SetOp(data)

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

    def numpy(self) -> np.ndarray:
        return self.data

    def backward(self):
        pass
