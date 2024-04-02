import numpy as np
from .core import Diff

'''
ADAM optimizer
'''
class ADAM():
    def __init__(self, alpha=0.01, beta1=0.9, beta2=0.9, reg=0.01, epsilon=1e-7):
        self.alpha = alpha
        self.beta1 = beta1
        self.beta2 = beta2
        self.reg = reg
        self.epsilon = epsilon

    def __call__(self, neuron, grads, t):
        dldw, dldb = grads

        # Update first moment estimate
        neuron.m_weights = self.beta1 * neuron.m_weights + (1 - self.beta1) * dldw
        neuron.m_bias = self.beta1 * neuron.m_bias + (1 - self.beta1) * dldb
        
        # Update second moment estimate
        neuron.v_weights = self.beta2 * neuron.v_weights + (1 - self.beta2) * (dldw ** 2)
        neuron.v_bias = self.beta2 * neuron.v_bias + (1 - self.beta2) * (dldb ** 2)
        
        # Compute bias-corrected first moment estimate
        m_hat_weights = neuron.m_weights / (1 - self.beta1 ** t)
        m_hat_bias = neuron.m_bias / (1 - self.beta1 ** t)
        
        # Compute bias-corrected second moment estimate
        v_hat_weights = neuron.v_weights / (1 - self.beta2 ** t)
        v_hat_bias = neuron.v_bias / (1 - self.beta2 ** t)

        weights_reg = neuron.weights * self.reg
        bias_reg = neuron.bias * self.reg

        neuron.weights -= self.alpha * m_hat_weights / (np.sqrt(v_hat_weights) + self.epsilon)
        neuron.bias -= self.alpha * m_hat_bias / (np.sqrt(v_hat_bias) + self.epsilon)

        neuron.weights -= weights_reg
        neuron.bias -= bias_reg


'''
RMSProp optimizer
'''
class RMSProp():
    def __init__(self, alpha=0.01, beta=0.9, reg=0.01, epsilon=1e-7):
        self.alpha = alpha
        self.beta = beta
        self.reg = reg
        self.epsilon = epsilon

    def __call__(self, neuron, grads):
        dldw, dldb = grads

        # Update moving average of the squared gradients
        neuron.m_weights = self.beta * neuron.m_weights + (1 - self.beta) * (dldw**2)
        neuron.m_bias = self.beta * neuron.m_bias + (1 - self.beta) * (dldb**2)

        weights_reg = neuron.weights * self.reg
        bias_reg = neuron.bias * self.reg

        # Apply RMSProp update
        neuron.weights -= self.alpha * dldw / (np.sqrt(neuron.m_weights) + self.epsilon)
        neuron.bias -= self.alpha * dldb / (np.sqrt(neuron.m_bias) + self.epsilon)

        neuron.weights -= weights_reg
        neuron.bias -= bias_reg


'''
SGD optimizer
'''
class SGD():
    def __init__(self, alpha=0.01):
        self.alpha = alpha

    def __call__(self, neuron, grads):
        dldw, dldb = grads
        
        neuron.weights -= self.alpha * dldw
        neuron.bias -= self.alpha * dldb