__all__ = ['SOpCode', 'UOpCode', 'BOpCode', 'OpCode', 'OpError', 'Op']

from enum import Enum, auto

from .defs import COMPACT, DEBUG

class SOpCode(Enum):
    assign = auto()

class UOpCode(Enum):
    neg = auto()
    exp = auto()
    tanh = auto()
    sigmoid = auto()
    relu = auto()
    sum = auto()
    transpose = auto()

class BOpCode(Enum):
    add = auto()
    sub = auto()
    mul = auto()
    div = auto()
    pow = auto()
    matmul = auto()

class COPCode(Enum):
    eq = auto()
    lt = auto()

OpCode: type = SOpCode | UOpCode | BOpCode | COPCode

class OpError(Exception):
    pass

Operands: type = list['Tensor'] | tuple['Tensor']

class Op:
    opcode: OpCode
    operands: Operands

    def __init__(self, opcode: OpCode, operands: Operands=()):
        self.opcode = opcode
        self.operands = operands
        self.requires_grad = any(n.requires_grad for n in operands)