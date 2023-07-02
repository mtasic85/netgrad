import math
from enum import Enum

DEBUG = 0

class SOp(Enum):
    nop = 0
    set = 1
    get = 2

class UnOp(Enum):
    pos = 10
    neg = 11
    tanh = 12
    exp = 13

class BinOp(Enum):
    add = 20
    sub = 21
    mul = 22
    div = 23
    pow = 24

Op: type = SOp | UnOp | BinOp

class Value:
    def __init__(self, v: int | float, label: str|None=None, op: Op=SOp.set, operands: tuple=()):
        self.v = v
        self.label = label
        self.op = op
        self.operands = operands
        self.grad = 0.0
        self._backward = lambda: None

    def __repr__(self) -> str:
        def _repr(obj: Value, ident: int=0) -> str:
            items = [f'v={obj.v}', f'grad={obj.grad}']

            if obj.label is not None:
                items.append(f'label={obj.label!r}')

            if obj.op != SOp.set:
                items.append(f'op={obj.op}')

            if obj.operands:
                items.append('operands=(\n')
    
                for operand in obj.operands:
                    item = _repr(operand, ident + 2)
                    items.append(f'{item},\n')

                ident_str = ' ' * ident
                items.append(f'{ident_str})')

            ident_str = ' ' * ident
            
            res = ''.join((
                ident_str,
                'Value(',
                ' '.join(items),
                ')',
            ))

            return res

        if DEBUG == 0:
            return f'Value(v={self.v} label={self.label})'
        elif DEBUG == 1:
            items = [f'v={self.v}', f'grad={self.grad}']

            if self.label is not None:
                items.append(f'label={self.label!r}')

            if self.op != SOp.set:
                items.append(f'op={self.op}')

            if self.operands is not None:
                items.append(f'operands={self.operands}')

            return f'Value({" ".join(items)})'
        elif DEBUG == 2:
            return _repr(self, 0)
        else:
            raise ValueError(f'Unsupported DEBUG level {DEBUG}')

    def __neg__(self) -> 'Value':
        return self * -1

    def __add__(self, other: 'Value') -> 'Value':
        if not isinstance(other, Value):
            other = Value(other)

        res = Value(self.v + other.v, op=BinOp.add, operands=(self, other))

        def _backward():
            self.grad += 1.0 * res.grad
            other.grad += 1.0 * res.grad
        
        res._backward = _backward
        return res

    def __radd__(self, other: 'Value') -> 'Value':
        return self + other

    def __sub__(self, other: 'Value') -> 'Value':
        return self + (-other)

    def __rsub__(self, other: 'Value') -> 'Value':
        return self - other

    def __mul__(self, other: 'Value') -> 'Value':
        if not isinstance(other, Value):
            other = Value(other)

        res = Value(self.v * other.v, op=BinOp.mul, operands=(self, other))

        def _backward():
            self.grad += other.v * res.grad
            other.grad += self.v * res.grad
        
        res._backward = _backward
        return res

    def __rmul__(self, other: 'Value') -> 'Value':
        return self * other

    def __truediv__(self, other: 'Value') -> 'Value':
        return self * (other ** -1.0)

    def __rtruediv__(self, other: 'Value') -> 'Value':
        return self / other

    def __pow__(self, other: int|float) -> 'Value':
        assert isinstance(other, (int, float))
        res = Value(self.v ** other, label=f'**{other}', op=BinOp.pow, operands=(self,))

        def _backward():
            self.grad += other * (self.v ** (other - 1)) * res.grad
        
        res._backward = _backward
        return res

    def tanh(self) -> 'Value':
        x = self.v
        t = (math.exp(2 * x) - 1) / (math.exp(2 * x) + 1)
        res = Value(t, op=UnOp.tanh, operands=(self,))

        def _backward():
            self.grad += (1.0 - t ** 2.0) * res.grad
        
        res._backward = _backward
        return res

    def exp(self) -> 'Value':
        x = self.v
        res = Value(math.exp(x), op=UnOp.exp, operands=(self,))

        def _backward():
            self.grad += res.v * res.grad
        
        res._backward = _backward
        return res

    def _build_top_ord(self) -> list['Value']:
        values = []
        visited = set()

        def __build_top_ord(v):
            if v not in visited:
                visited.add(v)

                for operand in v.operands:
                    __build_top_ord(operand)

                values.append(v)

        __build_top_ord(self)
        return values

    def backward(self):
        top_ord = self._build_top_ord()
        self.grad = 1.0

        for value in reversed(top_ord):
            value._backward()

def demo0():
    a = Value(2.0, 'a')
    b = Value(-3.0, 'b')
    c = Value(10.0, 'c')
    e = a * b; e.label = 'e'
    d = e + c; d.label = 'd'
    f = Value(-2.0, 'f')
    L = d * f; L.label = 'L'
    print(L)

def demo1():
    x1 = Value(2.0, 'x1')
    x2 = Value(0.0, 'x2')
    w1 = Value(-3.0, 'w1')
    w2 = Value(1.0, 'w2')
    b = Value(6.881373, 'b')
    x1w1 = x1 * w1; x1w1.label = 'x1*w1'
    x2w2 = x2 * w2; x2w2.label = 'x2*w2'
    x1w1x2w2 = x1w1 + x2w2; x1w1x2w2.label = 'x1w1x2w2'
    n = x1w1x2w2 + b; n.label = 'n'
    o = n.tanh(); o.label = 'o'
    o.backward()
    print(o)

def demo2():
    a = Value(3.0, 'a')
    b = a + a; b.label = 'b'
    b.backward()
    print(b)

def demo3():
    a = Value(-2.0, 'a')
    b = Value(3.0, 'b')
    d = a * b; d.label = 'd'
    e = a + b; e.label = 'e'
    f = d * e; f.label = 'f'
    f.backward()
    print(f)

def demo4():
    x1 = Value(2.0, 'x1')
    x2 = Value(0.0, 'x2')
    w1 = Value(-3.0, 'w1')
    w2 = Value(1.0, 'w2')
    b = Value(6.881373, 'b')
    x1w1 = x1 * w1; x1w1.label = 'x1*w1'
    x2w2 = x2 * w2; x2w2.label = 'x2*w2'
    x1w1x2w2 = x1w1 + x2w2; x1w1x2w2.label = 'x1w1x2w2'
    n = x1w1x2w2 + b; n.label = 'n'
    # o = n.tanh(); o.label = 'o'
    e = (2 * n).exp(); e.label = 'e'
    o = (e - 1) / (e + 1); o.label = 'o'
    o.backward()
    print(o)

if __name__ == '__main__':
    demo4()
