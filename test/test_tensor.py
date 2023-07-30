import numpy as np

from netgrad import NPTensor as Tensor

#
# specialized operations
#
def test_eye():
    a = Tensor.eye(3)
    np.testing.assert_allclose(a.numpy(), np.eye(3))

#
# unary operatons
#
def test_neg():
    a = Tensor([1, 2, 3])
    b = -a
    
    np.testing.assert_allclose(b.numpy(), -a.numpy())

def test_exp():
    a = Tensor([1, 2, 3])
    b = a.exp()
    
    np.testing.assert_allclose(b.numpy(), np.exp(a.numpy()))

def test_tanh():
    a = Tensor([1, 2, 3])
    b = a.tanh()
    
    np.testing.assert_allclose(b.numpy(), np.tanh(a.numpy()))

def test_sigmoid():
    a = Tensor([1, 2, 3])
    b = a.sigmoid()
    
    np.testing.assert_allclose(b.numpy(), 1.0 / (1.0 + np.exp(-a.numpy())))

def test_relu():
    a = Tensor([1, 2, 3])
    b = a.relu()
    
    np.testing.assert_allclose(b.numpy(), np.maximum(0, a.numpy()))

def test_sum():
    a = Tensor([[1, 2, 3], [2, 3, 4]])
    b = a.sum()
    
    np.testing.assert_allclose(b.numpy(), a.numpy().sum())

def test_sum_1d():
    a = Tensor([1, 2, 3])
    b = a.sum()
    
    np.testing.assert_allclose(b.numpy(), a.numpy().sum())

def test_transpose():
    a = Tensor([[1, 2, 3], [2, 3, 4]])
    b = a.transpose(1, 0)
    
    np.testing.assert_allclose(b.numpy(), a.numpy().transpose())

def test_T():
    a = Tensor([[1, 2, 3], [2, 3, 4]])
    b = a.T
    
    np.testing.assert_allclose(b.numpy(), a.numpy().T)

#
# binary operations
#
def test_add():
    a = Tensor([1, 2, 3])
    b = Tensor([2, 3, 4])
    c = a + b
    
    np.testing.assert_allclose(c.numpy(), a.numpy() + b.numpy())

def test_add_scalar():
    a = Tensor([1, 2, 3])
    b = 1
    c = a + b
    
    np.testing.assert_allclose(c.numpy(), a.numpy() + b)

def test_sub():
    a = Tensor([1, 2, 3])
    b = Tensor([2, 3, 4])
    c = a - b
    
    np.testing.assert_allclose(c.numpy(), a.numpy() - b.numpy())

def test_sub_scalar():
    a = Tensor([1, 2, 3])
    b = 1
    c = a - b
    
    np.testing.assert_allclose(c.numpy(), a.numpy() - b)

def test_mul():
    a = Tensor([1, 2, 3])
    b = Tensor([2, 3, 4])
    c = a * b
    
    np.testing.assert_allclose(c.numpy(), a.numpy() * b.numpy())

def test_mul_scalar():
    a = Tensor([1, 2, 3])
    b = 2
    c = a * b
    
    np.testing.assert_allclose(c.numpy(), a.numpy() * b)

def test_truediv():
    a = Tensor([1, 2, 3])
    b = Tensor([2, 3, 4])
    c = a / b
    
    np.testing.assert_allclose(c.numpy(), a.numpy() / b.numpy())

def test_truediv_scalar():
    a = Tensor([1, 2, 3])
    b = 2
    c = a / b
    
    np.testing.assert_allclose(c.numpy(), a.numpy() / b)

def test_pow():
    a = Tensor([1, 2, 3])
    b = Tensor([2, 3, 4])
    c = a ** b
    
    np.testing.assert_allclose(c.numpy(), a.numpy() ** b.numpy())

def test_pow_scalar():
    a = Tensor([1, 2, 3])
    b = -1
    c = a ** b
    
    np.testing.assert_allclose(c.numpy(), a.numpy() ** b)

def test_matmul_1x3_0():
    a = Tensor([1, 2, 3])
    b = Tensor([2, 3, 4])
    c = a @ b
    
    np.testing.assert_allclose(c.numpy(), a.numpy() @ b.numpy())

def test_matmul_2x3_0():
    a = Tensor([[1, 2, 3], [2, 3, 4]])
    b = Tensor([[2, 3], [4, 5], [6, 7]])
    c = a @ b
    
    np.testing.assert_allclose(c.numpy(), a.numpy() @ b.numpy())

def test_eq():
    a = Tensor([1, 2, 3])
    b = Tensor([1, 2, 3])
    c = a == b
    
    np.testing.assert_allclose(c.numpy(), a.numpy() == b.numpy())

def test_lt():
    a = Tensor([1, 2, 3])
    b = Tensor([3, 2, 1])
    c = a < b
    
    np.testing.assert_allclose(c.numpy(), a.numpy() < b.numpy())
