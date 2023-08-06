__all__ = ['TensorData']

import os

import numpy as np

COMPACT = int(os.getenv('COMPACT') or '1')
DEBUG = int(os.getenv('DEBUG') or '0')

TensorData = np.ndarray | tuple[int|float] | list[int|float] | int | float
