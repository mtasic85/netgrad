import os
import sys
sys.path.append(os.path.abspath('..'))

from netgrad import NPTensor as Tensor

x = Tensor.eye(3, requires_grad=True)
y = Tensor([[2.0,0,-2.0]], requires_grad=True)
z = y.matmul(x)
w = z.sum()
w.backward()

print(x)
print(y)
print(z)
print(w)

# print(x.grad.numpy())  # dw/dx
# print(y.grad.numpy())  # dw/dy
# print(z.grad.numpy())  # dw/dz
