import numpy as np

from netgrad import tensor


def test_add():
    a = tensor.Tensor([1, 2, 3])
    b = tensor.Tensor([2, 3, 4])
    c = a + b
    
    np.testing.assert_allclose(c.numpy(), a.numpy() + b.numpy())

def test_add_scalar():
    a = tensor.Tensor([1, 2, 3])
    b = 1
    c = a + b
    
    np.testing.assert_allclose(c.numpy(), a.numpy() + b)

def test_sub():
    a = tensor.Tensor([1, 2, 3])
    b = tensor.Tensor([2, 3, 4])
    c = a - b
    
    np.testing.assert_allclose(c.numpy(), a.numpy() - b.numpy())

def test_sub_scalar():
    a = tensor.Tensor([1, 2, 3])
    b = 1
    c = a - b
    
    np.testing.assert_allclose(c.numpy(), a.numpy() - b)

def test_mul():
    a = tensor.Tensor([1, 2, 3])
    b = tensor.Tensor([2, 3, 4])
    c = a * b
    
    np.testing.assert_allclose(c.numpy(), a.numpy() * b.numpy())

def test_mul_scalar():
    a = tensor.Tensor([1, 2, 3])
    b = 2
    c = a * b
    
    np.testing.assert_allclose(c.numpy(), a.numpy() * b)

def test_truediv():
    a = tensor.Tensor([1, 2, 3])
    b = tensor.Tensor([2, 3, 4])
    c = a / b
    
    np.testing.assert_allclose(c.numpy(), a.numpy() / b.numpy())

def test_truediv_scalar():
    a = tensor.Tensor([1, 2, 3])
    b = 2
    c = a / b
    
    np.testing.assert_allclose(c.numpy(), a.numpy() / b)

def test_floordiv():
    a = tensor.Tensor([1, 2, 3])
    b = tensor.Tensor([2, 3, 4])
    c = a // b
    
    np.testing.assert_allclose(c.numpy(), a.numpy() // b.numpy())

def test_floordiv():
    a = tensor.Tensor([1, 2, 3])
    b = 2
    c = a // b
    
    np.testing.assert_allclose(c.numpy(), a.numpy() // b)

def test_pow():
    a = tensor.Tensor([1, 2, 3])
    b = tensor.Tensor([2, 3, 4])
    c = a ** b
    
    np.testing.assert_allclose(c.numpy(), a.numpy() ** b.numpy())

def test_pow_scalar():
    a = tensor.Tensor([1, 2, 3])
    b = -1
    c = a ** b
    
    np.testing.assert_allclose(c.numpy(), a.numpy() ** b)

def test_matmul_1x3_0():
    a = tensor.Tensor([1, 2, 3])
    b = tensor.Tensor([2, 3, 4])
    c = a @ b
    
    np.testing.assert_allclose(c.numpy(), a.numpy() @ b.numpy())

def test_matmul_2x3_0():
    a = tensor.Tensor([[1, 2, 3], [2, 3, 4]])
    b = tensor.Tensor([[2, 3], [4, 5], [6, 7]])
    c = a @ b
    
    np.testing.assert_allclose(c.numpy(), a.numpy() @ b.numpy())

def test_eq():
    a = tensor.Tensor([1, 2, 3])
    b = tensor.Tensor([1, 2, 3])
    c = a == b
    
    np.testing.assert_allclose(c.numpy(), a.numpy() == b.numpy())

def test_lt():
    a = tensor.Tensor([1, 2, 3])
    b = tensor.Tensor([3, 2, 1])
    c = a < b
    
    np.testing.assert_allclose(c.numpy(), a.numpy() < b.numpy())

def test_exp():
    a = tensor.Tensor([1, 2, 3])
    b = a.exp()
    
    np.testing.assert_allclose(b.numpy(), np.exp(a.numpy()))

def test_tanh():
    a = tensor.Tensor([1, 2, 3])
    b = a.tanh()
    
    np.testing.assert_allclose(b.numpy(), np.tanh(a.numpy()))

def test_sigmoid():
    a = tensor.Tensor([1, 2, 3])
    b = a.sigmoid()
    
    np.testing.assert_allclose(b.numpy(), 1.0 / (1.0 + np.exp(-a.numpy())))

def test_relu():
    a = tensor.Tensor([1, 2, 3])
    b = a.relu()
    
    np.testing.assert_allclose(b.numpy(), np.maximum(0, a.numpy()))

def test_eye():
    a = tensor.Tensor.eye(3)
    np.testing.assert_allclose(a.numpy(), np.eye(3))

def test_sum():
    a = tensor.Tensor([[1, 2, 3], [2, 3, 4]])
    b = a.sum()
    
    np.testing.assert_allclose(b.numpy(), a.numpy().sum())