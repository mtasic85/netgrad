__all__ = ['SOpCode', 'UOpCode', 'BOpCode', 'OpCode', 'Operands', 'OpError', 'Op']

from typing import Self, Optional
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
Operands: type = list['Tensor'] | tuple['Tensor']

class OpError(Exception):
    pass

class Op:
    device: str
    opcode: OpCode
    operands: Operands
    grad: Optional['Tensor']

    def __init__(self, device: str, opcode: OpCode, operands: Operands):
        self.device = device
        self.opcode = opcode
        self.operands = operands
        self.requires_grad = any(n.requires_grad for n in operands)
        self.grad = None

    def __repr__(self) -> str:
        if DEBUG == 0:
            return object.__repr__(self)

        items = []
        items.append(type(self).__name__)
        items.append('(')
        subitems = []

        if COMPACT:
            subitems.append(self.device)

            if not self.requires_grad:
                subitems.append(f'requires_grad={self.requires_grad}')

            if DEBUG > 2:
                if self.grad is not None:
                    subitems.append(f'grad={self.grad}')
        else:
            subitems.append(f'device={self.device}')

            if DEBUG > 1:
                subitems.append(f'opcode={self.opcode}')

            if DEBUG > 1:
                if self.operands:
                    subitems.append(f'operands={self.operands}')

            if not self.requires_grad:
                subitems.append(f'requires_grad={self.requires_grad}')

            if DEBUG > 2:
                if self.grad is not None:
                    subitems.append(f'grad={self.grad}')

        items.append(' '.join(subitems))
        items.append(')')
        return ''.join(items)

    def forward(self, x: 'Tensor') -> 'Tensor':
        raise NotImplementedError('forward')

    def backward(self, grad: 'Tensor') -> Self:
        raise NotImplementedError('backward')