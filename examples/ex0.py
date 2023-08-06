import os
import sys
sys.path.append(os.path.abspath('..'))

from netgrad import NPTensor as Tensor

x = Tensor.eye(3, requires_grad=True)
print(x)

y = Tensor([[2.0,0,-2.0]], requires_grad=True)
print(y)

z = y.matmul(x)
print(z)

w = z.sum()
print(w)

w.backward()

print('---')
print(x)
print(y)
print(z)
print(w)

print('---')
print(z.grad.numpy())  # dw/dz
print(y.grad.numpy())  # dw/dy
print(x.grad.numpy())  # dw/dx
