__all__ = ['NumPyBackend']

from .backend import Backend
from ..tensor.defs import TensorData
from ..op import *

class NopOp(Op):
    # FIXME: implement, probably
    pass

class SetOp(Op):
    def __init__(self, data: TensorData, **kwargs):
        super().__init__(opcode=SOpCode.set, **kwargs)
        self.data = data

    def forward(self):
        pass

    def backward(self):
        pass

class NumPyBackend(Backend):
    NopOp = NopOp
    SetOp = SetOp
