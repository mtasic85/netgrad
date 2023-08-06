import numpy as np

class Tensor:
    def __init__(self, data, requires_grad=False):
        self.data = np.array(data)
        self.requires_grad = requires_grad
        
        # Gradient initialization
        if requires_grad:
            self.grad = np.zeros_like(self.data)
        else:
            self.grad = None
        
    def zero_grad(self):
        self.grad.fill(0)
        
    def backward(self, grad):
        assert self.requires_grad, "Called backward on non-requires-grad tensor"
        self.grad += grad
        
    def __add__(self, other):
        if isinstance(other, Tensor):
            requires_grad = self.requires_grad or other.requires_grad
        else:
            requires_grad = self.requires_grad
        
        out = Tensor(self.data + other.data, requires_grad)
        
        def backward(grad):
            self.backward(grad)
            other.backward(grad)
        out.backward = backward
        
        return out
    
    def sum(self):
        requires_grad = self.requires_grad
        s = Tensor(np.sum(self.data), requires_grad)
        def backward(grad):
            self.backward(grad*np.ones_like(self.data))
        s.backward = backward
        return s
    
x = Tensor([1, 2, 3], requires_grad=True)
y = Tensor([4, 5, 6], requires_grad=True)
z = x + y
loss = z.sum()
loss.backward(1)

print(x.grad) # [1, 1, 1] 
print(y.grad) # [1, 1, 1]