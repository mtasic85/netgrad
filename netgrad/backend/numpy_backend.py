__all__ = ['NPBackend']

from .backend import Backend
from ..defs import TensorData
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

class NPBackend(Backend):
    NopOp = NopOp
    SetOp = SetOp
