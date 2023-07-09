import os
from typing import Self
from functools import wraps

import numpy as np


DEBUG = int(os.getenv('DEBUG') or '0', 0)

TensorDataArg = np.ndarray | tuple | list | int | float


class Tensor:
    def __init__(self, data: TensorDataArg, dtype=np.float32):
        if not isinstance(data, np.ndarray):
            data = np.array(data, dtype=dtype)
        
        self.data = data

    def __repr__(self):
        if DEBUG == 0:
            return object.__repr__(self)
        else:
            return f'{type(self).__name__}(data={self.data})'

    def __add__(self, other: TensorDataArg) -> Self:
        if not isinstance(other, Tensor):
            other = Tensor(other if isinstance(other, (int, float)) else other.data)

        return Tensor(self.data + other.data)

    def __radd__(self, other: TensorDataArg) -> Self:
        return self + other

    def __sub__(self, other: TensorDataArg) -> Self:
        if not isinstance(other, Tensor):
            other = Tensor(other if isinstance(other, (int, float)) else other.data)

        return Tensor(self.data - other.data)

    def __rsub__(self, other: TensorDataArg) -> Self:
        return self - other

    def __mul__(self, other: TensorDataArg) -> Self:
        if not isinstance(other, Tensor):
            other = Tensor(other if isinstance(other, (int, float)) else other.data)

        return Tensor(self.data * other.data)

    def __rmul__(self, other: TensorDataArg) -> Self:
        return self * other

    def __truediv__(self, other: TensorDataArg) -> Self:
        if not isinstance(other, Tensor):
            other = Tensor(other if isinstance(other, (int, float)) else other.data)

        return Tensor(self.data / other.data)

    def __rtruediv__(self, other: TensorDataArg) -> Self:
        return self / other

    def __floordiv__(self, other: TensorDataArg) -> Self:
        if not isinstance(other, Tensor):
            other = Tensor(other if isinstance(other, (int, float)) else other.data)

        return Tensor(self.data // other.data)

    def __rfloordiv__(self, other: TensorDataArg) -> Self:
        return self // other

    def __pow__(self, other: TensorDataArg) -> Self:
        if not isinstance(other, Tensor):
            other = Tensor(other if isinstance(other, (int, float)) else other.data)

        return Tensor(self.data ** other.data)

    def __rpow__(self, other: TensorDataArg) -> Self:
        return self ** other

    def __matmul__(self, other: TensorDataArg) -> Self:
        if not isinstance(other, Tensor):
            other = Tensor(other.data)

        return Tensor(self.data.dot(other.data))

    def __eq__(self, other: Self) -> Self:
        return Tensor(np.equal(self.data, other.data)) # dtype=np.bool_

    def __lt__(self, other: Self) -> Self:
        return Tensor(np.less(self.data, other.data)) # dtype=np.bool_

    def numpy(self) -> np.ndarray:
        return self.data

    @property
    def dtype(self) -> np.dtype:
        return self.data.dtype


if __name__ == '__main__':
    DEBUG = 1
    a = Tensor([1, 2, 3])
    b = Tensor([2, 3, 4])
    print(a)
    print(b)
    c = a + b
    print(c)