from layer import Layer

class MLP:
    def __init__(self, n_in, n_outs):
        sz = [n_in] + n_outs
        self.layers = [Layer(sz[i], sz[i + 1]) for i in range(len(n_outs))]

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)

        return x

    def parameters(self):
        return [p for layer in self.layers for p in layer.parameters()]

if __name__ == '__main__':
    x = [2.0, 3.0, -1.0]
    n = MLP(3, [4, 4, 1])
    n(x)

    print(n.layers[0].neurons[0].w[0].v)
    print(n.layers[0].neurons[0].w[0].grad)

    # print(len(n.parameters()), n.parameters())
    print(len(n.parameters()))

    xs = [
        [2.0, 3.0, -1.0],
        [3.0, -1.0, 0.5],
        [0.5, 1.0, 1.0],
        [1.0, 1.0, -1.0],
    ]

    ys = [1.0, -1.0, -1.0, 1.0]
    
    for i in range(1000):
        ypred = [n(x) for x in xs]
        print(ypred)
        loss = sum((yout - ygt) ** 2.0 for ygt, yout in zip(ys, ypred))
        print(loss)

        for p in n.parameters():
            p.grad = 0.0

        loss.backward()

        for p in n.parameters():
            p.v += -0.001 * p.grad

    ypred = [n(x) for x in xs]
    print(ypred)
    loss = sum((yout - ygt) ** 2.0 for ygt, yout in zip(ys, ypred))
    print(loss)
