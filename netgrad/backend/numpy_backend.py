__all__ = ['NPBackend']

from .backend import Backend
from ..defs import TensorData
from ..op import *

#
# specialized operations
#
class NopOp(Op):
    # FIXME: implement, probably
    pass

class AssignOp(Op):
    def __init__(self, operands: list['Tensor'] | tuple['Tensor']=()):
        super().__init__(SOpCode.assign, operands)

    def forward(self):
        pass

    def backward(self):
        pass

#
# unary operatons
#

#
# binary operations
#
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
    AssignOp = AssignOp

    # BinOp
    MatMulOp = MatMulOp
