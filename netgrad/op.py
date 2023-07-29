__all__ = ['SOpCode', 'UnOpCode', 'BinOpCode', 'OpCode', 'OpError', 'Op']

from enum import Enum, auto

class SOpCode(Enum):
    nop = auto()
    set = auto()
    get = auto()
    eye = auto()

class UnOpCode(Enum):
    pos = auto()
    neg = auto()
    exp = auto()
    tanh = auto()
    sigmoid = auto()
    relu = auto()
    sum = auto()
    transpose = auto()

class BinOpCode(Enum):
    add = auto()
    sub = auto()
    mul = auto()
    div = auto()
    pow = auto()
    matmul = auto()
    eq = auto()
    lt = auto()

OpCode: type = SOpCode | UnOpCode | BinOpCode

class OpError(Exception):
    pass

class Op:
    def __init__(self, opcode: OpCode, operands: list['Tensor'] | tuple['Tensor']=()):
        self.opcode = opcode
        self.operands = operands
        self.requires_grad = any(n.requires_grad for n in operands)

    def forward(self, *args, **kwargs):
        raise NotImplementedError('forward')

    def backward(self, *args, **kwargs):
        raise NotImplementedError('forward')
