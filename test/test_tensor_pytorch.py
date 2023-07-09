import numpy as np

import torch
from netgrad import tensor


def test_add():
    a_n = tensor.Tensor([1, 2, 3])
    b_n = tensor.Tensor([2, 3, 4])
    c_n = a_n + b_n

    a_t = torch.Tensor([1, 2, 3])
    b_t = torch.Tensor([2, 3, 4])
    c_t = a_t + b_t
    
    np.testing.assert_allclose(c_n.numpy(), c_t.numpy())

def test_add_scalar():
    a_n = tensor.Tensor([1, 2, 3])
    b_n = 1
    c_n = a_n + b_n

    a_t = torch.Tensor([1, 2, 3])
    b_t = 1
    c_t = a_t + b_t
    
    np.testing.assert_allclose(c_n.numpy(), c_t.numpy())

def test_sub():
    a_n = tensor.Tensor([1, 2, 3])
    b_n = tensor.Tensor([2, 3, 4])
    c_n = a_n - b_n

    a_t = torch.Tensor([1, 2, 3])
    b_t = torch.Tensor([2, 3, 4])
    c_t = a_t - b_t
    
    np.testing.assert_allclose(c_n.numpy(), c_t.numpy())

def test_sub_scalar():
    a_n = tensor.Tensor([1, 2, 3])
    b_n = 1
    c_n = a_n - b_n

    a_t = torch.Tensor([1, 2, 3])
    b_t = 1
    c_t = a_t - b_t
    
    np.testing.assert_allclose(c_n.numpy(), c_t.numpy())

def test_mul():
    a_n = tensor.Tensor([1, 2, 3])
    b_n = tensor.Tensor([2, 3, 4])
    c_n = a_n * b_n

    a_t = torch.Tensor([1, 2, 3])
    b_t = torch.Tensor([2, 3, 4])
    c_t = a_t * b_t
    
    np.testing.assert_allclose(c_n.numpy(), c_t.numpy())

def test_mul_scalar():
    a_n = tensor.Tensor([1, 2, 3])
    b_n = 2
    c_n = a_n * b_n

    a_t = torch.Tensor([1, 2, 3])
    b_t = 2
    c_t = a_t * b_t
    
    np.testing.assert_allclose(c_n.numpy(), c_t.numpy())

def test_truediv():
    a_n = tensor.Tensor([1, 2, 3])
    b_n = tensor.Tensor([2, 3, 4])
    c_n = a_n / b_n

    a_t = torch.Tensor([1, 2, 3])
    b_t = torch.Tensor([2, 3, 4])
    c_t = a_t / b_t
    
    np.testing.assert_allclose(c_n.numpy(), c_t.numpy())

def test_truediv_scalar():
    a_n = tensor.Tensor([1, 2, 3])
    b_n = 2
    c_n = a_n / b_n

    a_t = torch.Tensor([1, 2, 3])
    b_t = 2
    c_t = a_t / b_t
    
    np.testing.assert_allclose(c_n.numpy(), c_t.numpy())

def test_floordiv():
    a_n = tensor.Tensor([1, 2, 3])
    b_n = tensor.Tensor([2, 3, 4])
    c_n = a_n // b_n

    a_t = torch.Tensor([1, 2, 3])
    b_t = torch.Tensor([2, 3, 4])
    c_t = a_t // b_t
    
    np.testing.assert_allclose(c_n.numpy(), c_t.numpy())

def test_floordiv_scalar():
    a_n = tensor.Tensor([1, 2, 3])
    b_n = 2
    c_n = a_n // b_n

    a_t = torch.Tensor([1, 2, 3])
    b_t = 2
    c_t = a_t // b_t
    
    np.testing.assert_allclose(c_n.numpy(), c_t.numpy())

def test_pow():
    a_n = tensor.Tensor([1, 2, 3])
    b_n = tensor.Tensor([2, 3, 4])
    c_n = a_n ** b_n

    a_t = torch.Tensor([1, 2, 3])
    b_t = torch.Tensor([2, 3, 4])
    c_t = a_t ** b_t
    
    np.testing.assert_allclose(c_n.numpy(), c_t.numpy())

def test_pow_scalar():
    a_n = tensor.Tensor([1, 2, 3])
    b_n = -1
    c_n = a_n ** b_n

    a_t = torch.Tensor([1, 2, 3])
    b_t = -1
    c_t = a_t ** b_t
    
    np.testing.assert_allclose(c_n.numpy(), c_t.numpy())

def test_matmul_1x3_0():
    a_n = tensor.Tensor([1, 2, 3])
    b_n = tensor.Tensor([2, 3, 4])
    c_n = a_n @ b_n

    a_t = torch.Tensor([1, 2, 3])
    b_t = torch.Tensor([2, 3, 4])
    c_t = a_t @ b_t
    
    np.testing.assert_allclose(c_n.numpy(), c_t.numpy())

def test_matmul_2x3_0():
    a_n = tensor.Tensor([[1, 2, 3], [2, 3, 4]])
    b_n = tensor.Tensor([[2, 3], [4, 5], [6, 7]])
    c_n = a_n @ b_n

    a_t = torch.Tensor([[1, 2, 3], [2, 3, 4]])
    b_t = torch.Tensor([[2, 3], [4, 5], [6, 7]])
    c_t = a_t @ b_t
    
    np.testing.assert_allclose(c_n.numpy(), c_t.numpy())

def test_eq():
    a_n = tensor.Tensor([1, 2, 3])
    b_n = tensor.Tensor([1, 2, 3])
    c_n = a_n == b_n

    a_t = torch.Tensor([1, 2, 3])
    b_t = torch.Tensor([1, 2, 3])
    c_t = a_t == b_t
    
    np.testing.assert_allclose(c_n.numpy(), c_t.numpy())

def test_lt():
    a_n = tensor.Tensor([1, 2, 3])
    b_n = tensor.Tensor([3, 2, 1])
    c_n = a_n < b_n

    a_t = torch.Tensor([1, 2, 3])
    b_t = torch.Tensor([3, 2, 1])
    c_t = a_t < b_t
    
    np.testing.assert_allclose(c_n.numpy(), c_t.numpy())

def test_exp():
    a_n = tensor.Tensor([1, 2, 3])
    b_n = a_n.exp()

    a_t = torch.Tensor([1, 2, 3])
    b_t = a_t.exp()
    
    np.testing.assert_allclose(b_n.numpy(), b_t.numpy())

def test_tanh():
    a_n = tensor.Tensor([1, 2, 3])
    b_n = a_n.tanh()

    a_t = torch.Tensor([1, 2, 3])
    b_t = a_t.tanh()
    
    np.testing.assert_allclose(b_n.numpy(), b_t.numpy())

def test_sigmoid():
    a_n = tensor.Tensor([1, 2, 3])
    b_n = a_n.sigmoid()

    a_t = torch.Tensor([1, 2, 3])
    b_t = a_t.sigmoid()
    
    np.testing.assert_allclose(b_n.numpy(), b_t.numpy())

def test_relu():
    a_n = tensor.Tensor([1, 2, 3])
    b_n = a_n.relu()

    a_t = torch.Tensor([1, 2, 3])
    b_t = a_t.relu()
    
    np.testing.assert_allclose(b_n.numpy(), b_t.numpy())