__all__ = ['NPTensor']

from typing import Self

import numpy as np

from .defs import TensorData, COMPACT, DEBUG
from .op import SOpCode, UOpCode, BOpCode, OpCode, Operands, OpError, Op
from .tensor import Tensor, TensorError

class NPTensor(Tensor):
    def __init__(self, data: TensorData, *, dtype=np.float32, requires_grad: bool=False, op: Op | None=None):
        super().__init__(data, dtype=dtype, requires_grad=requires_grad)
        self.op = NPAssignOp((self,)) if op is None else op

    #
    # specialized operations
    #
    @classmethod
    def eye(cls, dim: int, requires_grad: bool=False, dtype=np.float32) -> Self:
        return NPTensor(np.eye(dim, dtype=dtype), requires_grad=requires_grad)

    #
    # unary operatons
    #
    def sum(self, axis=None, keepdims=False):
        # https://numpy.org/doc/stable/reference/generated/numpy.sum.html#numpy.sum
        rd = self.requires_grad
        data = self.data.sum(axis=axis, keepdims=keepdims)
        res = NPTensor(data, requires_grad=rd)
        res.op = NPSumOp((self,))
        return res

    #
    # binary operations
    #
    def __matmul__(self, other: TensorData) -> Self:
        if not isinstance(other, NPTensor):
            other = NPTensor(other.data)

        rd = self.requires_grad or other.requires_grad
        data = np.matmul(self.data, other.data)
        res = NPTensor(data, requires_grad=rd)
        res.op = NPMatMulOp((self, other))
        return res

    #
    # propagation
    #
    def backward(self) -> Self:
        grad = NPTensor(1, requires_grad=False)
        grad.op = NPAssignOp((grad,))

        for t in reversed(self._build_top_ord()):
            if not t.requires_grad:
                continue

            print('!', t)
            t.op.backward(grad)


class NPOp(Op):
    def __init__(self, opcode: OpCode, operands: Operands):
        super().__init__('numpy', opcode, operands)

class NPAssignOp(NPOp):
    def __init__(self, operands: Operands):
        super().__init__(SOpCode.assign, operands)

    # def forward(self, x: Tensor) -> Tensor:
    #     return self

    def backward(self, grad: Tensor) -> Tensor:
        return self

class NPSumOp(NPOp):
    def __init__(self, operands: Operands):
        super().__init__(UOpCode.sum, operands)
    
    # def forward(self, x: Tensor) -> Tensor:
    #     data = np.sum(x.data)
    #     rd = x.requires_grad
    #     res = NPTensor(data, requires_grad=rd)
    #     return res

    def backward(self, grad: Tensor) -> Self:
        for tensor in self.operands:
            if tensor.requires_grad:
                if tensor.op.grad is None:
                    tensor.op.grad = NPTensor(grad.data * np.ones_like(tensor.data))
                else:
                    tensor.op.grad.data += grad.data * np.ones_like(tensor.data)
                
                tensor.op.backward(grad)

        return self

class NPMatMulOp(NPOp):
    def __init__(self, operands: Operands):
        super().__init__(BOpCode.matmul, operands)

    # def forward(self, x: Tensor) -> Tensor:
    #     pass

    def backward(self, grad: Tensor) -> Self:
        for tensor in self.operands:
            if tensor.requires_grad:
                if tensor.op.grad is None:
                    tensor.op.grad = NPTensor(grad.data @ self.other(tensor).data.T)
                else:
                    tensor.grad.data += grad.data @ self.other(tensor).data.T
                
                tensor.op.backward(grad)

        return self

    def other(self, tensor):
        if tensor is self.operands[0]:
            return self.operands[1]
        
        return self.operands[0]
