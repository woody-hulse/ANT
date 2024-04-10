import numpy as np
from .core import Diff

'''
Linear activation function
'''
class Linear(Diff):
    def __init__(self):
        super().__init__()
    
    def __call__(self, x):
        return x
    
    def dydx(self, output):
        return np.ones_like(output)


'''
Sigmoid activation function
'''
class Sigmoid(Diff):
    def __init__(self):
        super().__init__()
    
    def __call__(self, x):
        return 1 / (1 + np.exp(-x))

    def dydx(self, output):
        return self(output) * (1 - self(output))

'''
Tanh activation function
'''
class Tanh(Diff):
    def __init__(self):
        super().__init__()
        self.epsilon = 1e-7

    def __call__(self, x):
        return (np.exp(x) - np.exp(-x)) / (np.exp(x) + np.exp(-x) + self.epsilon)

    def dydx(self, output):
        return 1 - np.square(self(output))


'''
Sigmoid activation function between (-1, 1)
'''
class Sigmoid2(Diff):
    def __init__(self):
        super().__init__()
    
    def __call__(self, x):
        return 4 / (1 + np.exp(-x)) - 1

    def dydx(self, output):
        return 4 * self(output) * (1 - self(output))

    
'''
ReLU activation function
'''
class ReLU(Diff):
    def __init__(self):
        super().__init__()

    def __call__(self, x):
        return x * (x > 0)
    
    def dydx(self, output):
        return np.array(output > 0, dtype=np.float32)


'''
Leaky ReLU activation function
'''
class LeakyReLU(Diff):
    def __init__(self, alpha=0.5):
        super().__init__()
        self.alpha = alpha

    def __call__(self, x):
        return x * (x > 0) + x * self.alpha * (x < 0)
    
    def dydx(self, output):
        return np.array(output > 0, dtype=np.float32) + np.array(output < 0, dtype=np.float32) * self.alpha


'''
NReLU activation function
'''
class NReLU(Diff):
    def __init__(self):
        super().__init__()

    def __call__(self, x):
        return -x * (x > 0)
    
    def dydx(self, output):
        return -1 * np.array(output > 0, dtype=np.float32)
    

'''
Softmax activation function
'''
class Softmax(Diff):
    def __init__(self):
        super().__init__()
    
    def __call__(self, x):
        e_x = np.exp(x - np.max(x))
        return e_x / e_x.sum(axis=0)
    
    def dydx(self, output):
        s = output.reshape(-1, 1)
        return np.diagflat(s) - np.dot(s, s.T)
