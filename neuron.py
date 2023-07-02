import random
from value import Value

class Neuron:
    def __init__(self, n_in: int):
        self.w = [Value(random.uniform(-1, 1)) for _ in range(n_in)]
        self.b = Value(random.uniform(-1, 1))

    def __call__(self, x):
        # w * x + b
        act = sum((wi * xi for wi, xi in zip(self.w, x)), self.b)
        out = act.tanh()
        return out


if __name__ == '__main__':
    x = [2.0, 3.0]
    n = Neuron(2)
    print(n(x))

