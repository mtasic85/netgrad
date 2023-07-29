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

if __name__ == '__main__':
    DEBUG = 1

    # Tensor = Tensor.use_backend(NumPyBackend)

    x = Tensor.eye(3, requires_grad=True)
    y = Tensor([[2.0,0,-2.0]], requires_grad=True)
    z = y.matmul(x)
    w = z.sum()
    w.backward()

    print(x)
    print(y)
    print(z)
    print(w)

    +w

    # print(x.grad.numpy())  # dw/dx
    # print(y.grad.numpy())  # dw/dy
    # print(z.grad.numpy())  # dw/dz