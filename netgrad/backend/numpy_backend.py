__all__ = ['NPBackend']

from .backend import Backend
from ..defs import TensorData
from ..op import *

#
# specialized operations
#
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
class NegOp(Op):
    def __init__(self, operands: list['Tensor'] | tuple['Tensor']=()):
        super().__init__(UOpCode.neg, operands)

    def forward(self):
        pass

    def backward(self):
        pass

class ExpOp(Op):
    def __init__(self, operands: list['Tensor'] | tuple['Tensor']=()):
        super().__init__(UOpCode.exp, operands)

    def forward(self):
        pass

    def backward(self):
        pass

class TanhOp(Op):
    def __init__(self, operands: list['Tensor'] | tuple['Tensor']=()):
        super().__init__(UOpCode.tanh, operands)

    def forward(self):
        pass

    def backward(self):
        pass

class SigmoidOp(Op):
    def __init__(self, operands: list['Tensor'] | tuple['Tensor']=()):
        super().__init__(UOpCode.sigmoid, operands)

    def forward(self):
        pass

    def backward(self):
        pass

class ReluOp(Op):
    def __init__(self, operands: list['Tensor'] | tuple['Tensor']=()):
        super().__init__(UOpCode.relu, operands)

    def forward(self):
        pass

    def backward(self):
        pass

class SumOp(Op):
    def __init__(self, operands: list['Tensor'] | tuple['Tensor']=()):
        super().__init__(UOpCode.sum, operands)

    def forward(self):
        pass

    def backward(self):
        pass

class TransposeOp(Op):
    def __init__(self, operands: list['Tensor'] | tuple['Tensor']=()):
        super().__init__(UOpCode.transpose, operands)

    def forward(self):
        pass

    def backward(self):
        pass

#
# binary operations
#
class AddOp(Op):
    def __init__(self, operands: list['Tensor'] | tuple['Tensor']=()):
        super().__init__(BOpCode.add, operands)

    def forward(self):
        pass

    def backward(self):
        pass

class SubOp(Op):
    def __init__(self, operands: list['Tensor'] | tuple['Tensor']=()):
        super().__init__(BOpCode.sub, operands)

    def forward(self):
        pass

    def backward(self):
        pass

class MulOp(Op):
    def __init__(self, operands: list['Tensor'] | tuple['Tensor']=()):
        super().__init__(BOpCode.mul, operands)

    def forward(self):
        pass

    def backward(self):
        pass

class DivOp(Op):
    def __init__(self, operands: list['Tensor'] | tuple['Tensor']=()):
        super().__init__(BOpCode.div, operands)

    def forward(self):
        pass

    def backward(self):
        pass

class PowOp(Op):
    def __init__(self, operands: list['Tensor'] | tuple['Tensor']=()):
        super().__init__(BOpCode.pow, operands)

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

#
# comparison operations
#
class EqOp(Op):
    def __init__(self, operands: list['Tensor'] | tuple['Tensor']=()):
        super().__init__(COpCode.eq, operands)

    def forward(self):
        pass

    def backward(self):
        pass

class LtOp(Op):
    def __init__(self, operands: list['Tensor'] | tuple['Tensor']=()):
        super().__init__(COpCode.lt, operands)

    def forward(self):
        pass

    def backward(self):
        pass

#
# NPBackend
#
class NPBackend(Backend):
    # SOp
    AssignOp = AssignOp

    # UOp
    NegOp = NegOp
    ExpOp = ExpOp
    TanhOp = TanhOp
    SigmoidOp = SigmoidOp
    ReluOp = ReluOp
    SumOp = SumOp
    TransposeOp = TransposeOp

    # BOp
    AddOp = AddOp
    SubOp = SumOp
    MulOp = MulOp
    DivOp = DivOp
    PowOp = PowOp
    MatMulOp = MatMulOp

    # COp
    EqOp = EqOp
    LtOp = LtOp
