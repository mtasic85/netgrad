__all__ = ['get_tensor_type']

from .numpy_tensor import NumPyTensor
from .opencl_tensor import OpenCLTensor


def get_tensor_type(backend: str='numpy', device: str='cpu') -> type:
    if backend == 'numpy' and device == 'cpu':
        return NumPyTensor
    elif backend == 'opencl' and device in ('cpu', 'gpu'):
        return NumPyTensor

    raise TypeError(f'unsupported tensor backend {backend!r} and device {device!r}')