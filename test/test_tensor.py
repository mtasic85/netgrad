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

def test_matmul():
    a = tensor.Tensor([1, 2, 3])
    b = tensor.Tensor([2, 3, 4])
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