from core.activations import *
from core.optimizers import *
from core.losses import *

'''
Neuron class

         *
        /
    \  /
    - o --*
    / |

Contains:
    [Graph]
    id              : ID of neuron, corresponding to index in network list
    next            : List of pointers to next neurons in network
    prev            : List of pointers to previous neurons in network
    compiled_prev   : Array of (network_output_dim, x) of pointers to valid precomputed differentiation paths

    [Model]
    weights         : f(x) = x[w] + b
    bias            : f(x) = xw + [b]
    activation      : Activation function a(f(x))
    inputs          : Array of (network_output_dim, prev_dim) computed inputs
    outputs         : Array of (network_output_dim, next_dim) computed outputs
'''
class Neuron(Diff):
    def __init__(self, id):
        self.id = id
        self.next = []
        self.prev = []

        self.weights = None
        self.bias = None
        self.activation = Tanh()
        self.epsilon = 1e-5
        # self.lrJ_1 = 1

        self.initialize_weights()
    
    def initialize_weights(self):
        # np.random.seed(42) # deterministic weight initialization
        self.input_size = len(self.prev)
        self.output_size = len(self.next)
        self.weights = np.random.normal(size=(self.input_size, self.output_size)) * 0.1
        self.bias = np.random.normal(size=self.output_size) * 0.01

        self.accumulated_weight_gradients = [] # np.zeros_like(self.weights)
        self.accumulated_bias_gradients = [] # np.zeros_like(self.bias)

        self.inputs = np.zeros(self.input_size)
        self.next_inputs = np.zeros(self.input_size)
        self.outputs = np.zeros(self.output_size)

        self.input_J = np.zeros(self.output_size)
        self.next_input_J = np.zeros(self.output_size)

        self.init_optimizer_params()
    
    def init_optimizer_params(self):
        self.m_weights = np.zeros_like(self.weights)
        self.m_bias = np.zeros_like(self.bias)
        self.v_weights = np.zeros_like(self.weights)
        self.v_bias = np.zeros_like(self.bias)
    
    def forward(self):
        return self.activation(self.inputs @ self.weights + self.bias)

    def backward(self):
        # compute temporal term
        # print(self.weights.shape, self.inputs.shape)
        # dada = np.sum(self.weights.T * np.expand_dims(self.lrJ_1, axis=-1) / (self.epsilon + np.expand_dims(self.inputs, axis=0)), axis=1)
        # print(dada, self.input_J)
        # self.input_J = self.input_J * dada
        # print(dada)

        dady = self.activation.dydx(self.outputs)
        # print(dady, self.activation)
        # print(self.id, self.activation)
        dydx = self.weights.T
        dydw = self.inputs
        dydb = np.ones_like(self.bias)

        # self.lrJ_1 = 1 / (self.epsilon + 1e-6 * self.input_J * dady)

        dldw = np.outer(dydw, self.input_J * dady)
        dldb = self.input_J * dady * dydb
        dldx = (self.input_J * dady) @ dydx

        return dldw, dldb, dldx
    
    def insert_next_connection(self, neuron):
        self.next.append(neuron)
        self.output_size += 1

        self.input_J = np.concatenate([self.input_J, np.array([0])])
        self.next_input_J = np.concatenate([self.next_input_J, np.array([0])])
        self.outputs = np.concatenate([self.outputs, np.array([0])])

        self.weights = np.hstack([self.weights, np.array([np.random.normal(size=self.input_size) * 0.1]).T])
        self.bias = np.concatenate([self.bias, np.array([np.random.normal() * 0.01])])

        self.init_optimizer_params()

    def insert_prev_connection(self, neuron):
        self.prev.append(neuron)
        self.input_size += 1

        self.inputs = np.concatenate([self.inputs, np.array([0])])
        self.next_inputs = np.concatenate([self.next_inputs, np.array([0])])

        self.weights = np.vstack([self.weights, np.random.normal(size=self.output_size) * 0.1])

        self.init_optimizer_params()

    def remove_next_connection(self, neuron):
        index = self.next.index(neuron)
        self.next.remove(neuron)
        self.output_size -= 1

        self.input_J = np.concatenate([self.input_J[:index], self.input_J[index + 1:]])
        self.next_input_J = np.concatenate([self.next_input_J[:index], self.next_input_J[index + 1:]])
        self.outputs = np.concatenate([self.outputs[:index], self.outputs[index + 1:]])

        self.weights = np.hstack([self.weights[:, :index], self.weights[:, index + 1:]])
        self.bias = np.concatenate([self.bias[:index], self.bias[index + 1:]])

        self.init_optimizer_params()

    def remove_prev_connection(self, neuron):
        index = self.prev.index(neuron)
        self.prev.remove(neuron)
        self.input_size -= 1

        self.inputs = np.concatenate([self.inputs[:index], self.inputs[index + 1:]])
        self.next_inputs = np.concatenate([self.next_inputs[:index], self.next_inputs[index + 1:]])

        self.weights = np.vstack([self.weights[:index], self.weights[index + 1:]])

        self.init_optimizer_params()

    def mutate_weights(self, weight_mutation_rate=0.0001):
        self.weights += np.random.normal(size=self.weights.shape) * weight_mutation_rate
        self.bias += np.random.normal(size=self.bias.shape) * weight_mutation_rate

'''
Neuron updating function (for parallelization)
    - Investigate moving this into the neuron class
'''
def update_neuron(neuron, optimizer, update_metrics=False):
    dady = neuron.activation.dydx(neuron.outputs)
    dydx = neuron.weights
    dydw = neuron.inputs
    dydb = np.ones_like(neuron.bias)

    dldw = np.outer(dydw, neuron.input_J * dady)
    dldb = neuron.input_J * dady * dydb

    # Assume self.optimizer is adapted to be called here directly
    optimizer(neuron, [dldw, dldb])

    output_J = dydx @ (neuron.input_J * dady)

    metrics = None
    if update_metrics:
        metrics = {
            'energy': np.sum(np.abs(neuron.inputs)),
            'grad_energy': np.sum(np.abs(neuron.input_J)),
            'weight_energy': np.sum(np.abs(neuron.weights)),
            'prop_input': np.mean(neuron.inputs > 1e-7),
            'prop_grad': np.mean(neuron.input_J > 1e-7),
        }

    return output_J, metrics
