import os
import sys
sys.path.append(os.path.abspath('.'))

from netgrad import NPTensor as Tensor

def demo0():
    a = Tensor([1, 2, 3])
    b = Tensor([2, 3, 4])
    print(a)
    print(b)
    c = a + b
    print(c)

def demo1():
    u = Tensor([[1, 2, 3], [4, 5, 6]])
    v = u.T
    w = v.T
    print(u)
    print(v)
    print(w)

def demo2():
    x = Tensor.eye(3, requires_grad=True)
    y = Tensor([[2.0, 0, -2.0]], requires_grad=True)
    z = y.matmul(x)
    w = z.sum()

    print(x)
    print(y)
    print(z)
    print(w)
    print('---')

    w.backward()

    print('---')
    print(x.grad.numpy())  # dw/dx
    print(y.grad.numpy())  # dw/dy
    print(z.grad.numpy())  # dw/dz
    print(w.grad.numpy())  # dw/dz

if __name__ == '__main__':
    demo2()