import numpy as np

from netgrad import tensor


def test_add():
    a = tensor.Tensor([1, 2, 3])
    b = tensor.Tensor([2, 3, 4])
    c = a + b
    
    np.testing.assert_allclose(c.numpy(), np.array([3, 5, 7], dtype=c.dtype))

def test_sub():
    a = tensor.Tensor([1, 2, 3])
    b = tensor.Tensor([2, 3, 4])
    c = a - b
    
    np.testing.assert_allclose(c.numpy(), np.array([-1, -1, -1], dtype=c.dtype))

def test_mul():
    a = tensor.Tensor([1, 2, 3])
    b = tensor.Tensor([2, 3, 4])
    c = a * b
    
    np.testing.assert_allclose(c.numpy(), np.array([2, 6, 12], dtype=c.dtype))