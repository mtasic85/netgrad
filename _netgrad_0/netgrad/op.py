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

class Op:
    def __init__(self, opcode: OpCode, operands: list['Tensor'] | tuple['Tensor']=()):
        self.opcode = opcode
        self.operands = operands
        self.requires_grad = any(n.requires_grad for n in operands)
        # self.grad = None
        # self.grad_fn = None

    def __repr__(self) -> str:
        if DEBUG == 0:
            return object.__repr__(self)

        # return f'<Op opcode={self.opcode} operands={self.operands}>'

        items = []
        items.append(type(self).__name__)
        items.append('(')
        subitems = []

        if COMPACT:
            if not self.requires_grad:
                subitems.append(f'requires_grad={self.requires_grad}')

            # if self.grad is not None:
            #     subitems.append(f'grad={self.grad}')
            #
            # if self.grad_fn is not None:
            #     subitems.append(f'grad_fn={self.grad_fn}')
        else:
            if DEBUG > 1:
                subitems.append(f'opcode={self.opcode}')

            if DEBUG > 1:
                if self.operands:
                    subitems.append(f'operands={self.operands}')

            if not self.requires_grad:
                subitems.append(f'requires_grad={self.requires_grad}')

            # if DEBUG > 2:
            #     if self.grad is not None:
            #         subitems.append(f'grad={self.grad}')
            #
            #     if self.grad_fn is not None:
            #         subitems.append(f'grad_fn={self.grad_fn}')

        items.append(' '.join(subitems))
        items.append(')')
        return ''.join(items)

    def forward(self, *args, **kwargs):
        raise NotImplementedError('forward')

    def backward(self, *args, **kwargs):
        raise NotImplementedError('forward')
