__all__ = ['SOpCode', 'UOpCode', 'BOpCode', 'OpCode', 'OpError', 'Op']

from enum import Enum, auto

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

class Op:
    def __init__(self, opcode: OpCode, operands: list['Tensor'] | tuple['Tensor']=()):
        self.opcode = opcode
        self.operands = operands
        self.requires_grad = any(n.requires_grad for n in operands)

    def __repr__(self) -> str:
        return f'<Op opcode={self.opcode} operands={self.operands}>'

    def forward(self, *args, **kwargs):
        raise NotImplementedError('forward')

    def backward(self, *args, **kwargs):
        raise NotImplementedError('forward')
