from layer import Layer

class MLP:
    def __init__(self, n_in, n_outs):
        sz = [n_in] + n_outs
        self.layers = [Layer(sz[i], sz[i + 1]) for i in range(len(n_outs))]

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)

        return x

if __name__ == '__main__':
    x = [2.0, 3.0, -1.0]
    n = MLP(3, [4, 4, 1])
    # print(n(x))

    xs = [
        [2.0, 3.0, -1.0],
        [3.0, -1.0, 0.5],
        [0.5, 1.0, 1.0],
        [1.0, 1.0, -1.0],
    ]

    ys = [1.0, -1.0, -1.0, 1.0]
    ypred = [n(x) for x in xs]
    print(ypred)

    loss = sum([(yout - ygt) ** 2.0 for ygt, yout in zip(ys, ypred)])
    print(loss)