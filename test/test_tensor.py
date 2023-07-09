import numpy as np

import netgrad


def test_pos():
    a = netgrad.Tensor([1, 2, 3])
    b = -a
    
    np.testing.assert_allclose(b.numpy(), -a.numpy())

def test_neg():
    a = netgrad.Tensor([1, 2, 3])
    b = -a
    
    np.testing.assert_allclose(b.numpy(), -a.numpy())

def test_add():
    a = netgrad.Tensor([1, 2, 3])
    b = netgrad.Tensor([2, 3, 4])
    c = a + b
    
    np.testing.assert_allclose(c.numpy(), a.numpy() + b.numpy())

def test_add_scalar():
    a = netgrad.Tensor([1, 2, 3])
    b = 1
    c = a + b
    
    np.testing.assert_allclose(c.numpy(), a.numpy() + b)

def test_sub():
    a = netgrad.Tensor([1, 2, 3])
    b = netgrad.Tensor([2, 3, 4])
    c = a - b
    
    np.testing.assert_allclose(c.numpy(), a.numpy() - b.numpy())

def test_sub_scalar():
    a = netgrad.Tensor([1, 2, 3])
    b = 1
    c = a - b
    
    np.testing.assert_allclose(c.numpy(), a.numpy() - b)

def test_mul():
    a = netgrad.Tensor([1, 2, 3])
    b = netgrad.Tensor([2, 3, 4])
    c = a * b
    
    np.testing.assert_allclose(c.numpy(), a.numpy() * b.numpy())

def test_mul_scalar():
    a = netgrad.Tensor([1, 2, 3])
    b = 2
    c = a * b
    
    np.testing.assert_allclose(c.numpy(), a.numpy() * b)

def test_truediv():
    a = netgrad.Tensor([1, 2, 3])
    b = netgrad.Tensor([2, 3, 4])
    c = a / b
    
    np.testing.assert_allclose(c.numpy(), a.numpy() / b.numpy())

def test_truediv_scalar():
    a = netgrad.Tensor([1, 2, 3])
    b = 2
    c = a / b
    
    np.testing.assert_allclose(c.numpy(), a.numpy() / b)

def test_floordiv():
    a = netgrad.Tensor([1, 2, 3])
    b = netgrad.Tensor([2, 3, 4])
    c = a // b
    
    np.testing.assert_allclose(c.numpy(), a.numpy() // b.numpy())

def test_floordiv():
    a = netgrad.Tensor([1, 2, 3])
    b = 2
    c = a // b
    
    np.testing.assert_allclose(c.numpy(), a.numpy() // b)

def test_pow():
    a = netgrad.Tensor([1, 2, 3])
    b = netgrad.Tensor([2, 3, 4])
    c = a ** b
    
    np.testing.assert_allclose(c.numpy(), a.numpy() ** b.numpy())

def test_pow_scalar():
    a = netgrad.Tensor([1, 2, 3])
    b = -1
    c = a ** b
    
    np.testing.assert_allclose(c.numpy(), a.numpy() ** b)

def test_matmul_1x3_0():
    a = netgrad.Tensor([1, 2, 3])
    b = netgrad.Tensor([2, 3, 4])
    c = a @ b
    
    np.testing.assert_allclose(c.numpy(), a.numpy() @ b.numpy())

def test_matmul_2x3_0():
    a = netgrad.Tensor([[1, 2, 3], [2, 3, 4]])
    b = netgrad.Tensor([[2, 3], [4, 5], [6, 7]])
    c = a @ b
    
    np.testing.assert_allclose(c.numpy(), a.numpy() @ b.numpy())

def test_eq():
    a = netgrad.Tensor([1, 2, 3])
    b = netgrad.Tensor([1, 2, 3])
    c = a == b
    
    np.testing.assert_allclose(c.numpy(), a.numpy() == b.numpy())

def test_lt():
    a = netgrad.Tensor([1, 2, 3])
    b = netgrad.Tensor([3, 2, 1])
    c = a < b
    
    np.testing.assert_allclose(c.numpy(), a.numpy() < b.numpy())

def test_exp():
    a = netgrad.Tensor([1, 2, 3])
    b = a.exp()
    
    np.testing.assert_allclose(b.numpy(), np.exp(a.numpy()))

def test_tanh():
    a = netgrad.Tensor([1, 2, 3])
    b = a.tanh()
    
    np.testing.assert_allclose(b.numpy(), np.tanh(a.numpy()))

def test_sigmoid():
    a = netgrad.Tensor([1, 2, 3])
    b = a.sigmoid()
    
    np.testing.assert_allclose(b.numpy(), 1.0 / (1.0 + np.exp(-a.numpy())))

def test_relu():
    a = netgrad.Tensor([1, 2, 3])
    b = a.relu()
    
    np.testing.assert_allclose(b.numpy(), np.maximum(0, a.numpy()))

def test_eye():
    a = netgrad.Tensor.eye(3)
    np.testing.assert_allclose(a.numpy(), np.eye(3))

def test_sum():
    a = netgrad.Tensor([[1, 2, 3], [2, 3, 4]])
    b = a.sum()
    
    np.testing.assert_allclose(b.numpy(), a.numpy().sum())

def test_transpose():
    a = netgrad.Tensor([[1, 2, 3], [2, 3, 4]])
    b = a.transpose(1, 0)
    
    np.testing.assert_allclose(b.numpy(), a.numpy().transpose())

def test_T():
    a = netgrad.Tensor([[1, 2, 3], [2, 3, 4]])
    b = a.T
    
    np.testing.assert_allclose(b.numpy(), a.numpy().T)