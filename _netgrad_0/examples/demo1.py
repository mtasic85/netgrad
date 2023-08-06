import numpy as np

class Tensor:
    
    def __init__(self, data, requires_grad=False):
        self.data = np.array(data)
        self.requires_grad = requires_grad
        
        self.grad = None
        if self.requires_grad:
            self.zero_grad()
            
    def zero_grad(self):
        self.grad = np.zeros_like(self.data)
        
    def __add__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        requires_grad = self.requires_grad or other.requires_grad
        result = Tensor(self.data + other.data, requires_grad)
        
        def backward():
            if self.requires_grad:
                result.grad += self.grad
            if other.requires_grad:
                result.grad += other.grad
        result.backward = backward
        return result
    
    def __mul__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        requires_grad = self.requires_grad or other.requires_grad
        result = Tensor(self.data * other.data, requires_grad)
        
        def backward():
            if self.requires_grad:
                result.grad += other.data * self.grad 
            if other.requires_grad:
                result.grad += self.data * other.grad
        result.backward = backward
        return result
        
    def sum(self):
        requires_grad = self.requires_grad
        result = Tensor(self.data.sum(), requires_grad)
        
        def backward():
            if self.requires_grad:
                result.grad += np.ones_like(self.data) * self.grad
        result.backward = backward
        return result
    
    def matmul(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        requires_grad = self.requires_grad or other.requires_grad
        result = Tensor(self.data @ other.data, requires_grad)
        
        def backward():
            if self.requires_grad:
                result.grad += other.data.T @ self.grad
            if other.requires_grad:
                result.grad += self.data.T @ other.grad
        result.backward = backward
        return result
    
x = Tensor([1,2,3], requires_grad=True)
y = Tensor([4,5,6], requires_grad=True)
z = x.matmul(y).sum()
z.backward()
print(x.grad) # [4 5 6] 
print(y.grad) # [1 2 3]