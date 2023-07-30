__all__ = ['NPBackend']

from .backend import Backend
from ..defs import TensorData
from ..op import *

class NopOp(Op):
    # FIXME: implement, probably
    pass

class SetOp(Op):
    def __init__(self, operands: list['Tensor'] | tuple['Tensor']=()):
        super().__init__(SOpCode.set, operands)

    def forward(self):
        pass

    def backward(self):
        pass

class MatMulOp(Op):
    def __init__(self, operands: list['Tensor'] | tuple['Tensor']=()):
        super().__init__(BOpCode.matmul, operands)

    def forward(self):
        pass

    def backward(self):
        pass

class NPBackend(Backend):
    # SOp
    NopOp = NopOp
    SetOp = SetOp

    # BinOp
    MatMulOp = MatMulOp
