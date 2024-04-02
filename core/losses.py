import numpy as np
from .core import Diff

'''
Mean squared error loss
'''
class MSE(Diff):
    def __init__(self):
        super().__init__()
    
    def __call__(self, y, y_hat):
        return np.mean(np.square(y - y_hat))
    
    def dydx(self, y, y_hat):
        return 2 / max(1, np.sum(y.shape)) * (y_hat - y)
    
'''
Experimental RL loss
'''
class RL(Diff):
    def __init__(self):
        super().__init__()
        self.epsilon = 1e-5
    
    def __call__(self, y, y_hat):
        return np.exp(y_hat - y)
    
    def dydx(self, y, y_hat):
        return np.exp(y_hat - y)


class BCE(Diff):
    def __init__(self):
        super().__init__()

    def __call__(self, y, y_hat):
        return -(y * np.log(y_hat) + (1 - y) * np.log(1 - y_hat))
    
    def dydx(self, y, y_hat):
        # print(y, y_hat)
        return -y * (1 / y_hat) - (1 - y) * (1 / (1 - y_hat))
