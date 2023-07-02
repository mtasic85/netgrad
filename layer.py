from neuron import Neuron

class Layer:
    def __init__(self, n_in, n_out):
        self.neurons = [Neuron(n_in) for _ in range(n_out)]

    def __call__(self, x):
        outs = [n(x) for n in self.neurons]
        return outs

if __name__ == '__main__':
    x = [2.0, 3.0]
    n = Layer(2, 3)
    print(n(x))