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

def test_sub():
    a_n = tensor.Tensor([1, 2, 3])
    b_n = tensor.Tensor([2, 3, 4])
    c_n = a_n - b_n

    a_t = torch.Tensor([1, 2, 3])
    b_t = torch.Tensor([2, 3, 4])
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