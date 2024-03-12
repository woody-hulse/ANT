import networkx as nx
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from collections import deque
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor

from utils import *


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


'''
Superclass of differentiable objects
'''
class Diff():
    def __init__(self):
        pass

    def dydx(self):
        raise Exception('Input gradient not defined')
    
    def dydw(self):
        raise Exception('Weight gradient not defined')
    
    def dydb(self):
        raise Exception('Bias gradient not defined')

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
        # print(x, 1 / (1 + np.exp(-x)))
        return 1 / (1 + np.exp(-x))

    def dydx(self, output):
        return self(output) * (1 - self(output))


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
    def __init__(self, alpha=0.05):
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
        self.valid_prev = []

        self.weights = None
        self.bias = None
        self.activation = LeakyReLU()
    
    def initialize_weights(self):
        # np.random.seed(42) # optional deterministic weight initialization
        self.input_size = max(len(self.prev), 1)
        self.output_size = len(self.next)
        self.weights = np.random.normal(size=(self.input_size, self.output_size)) * 0.1
        self.bias = np.random.normal(size=self.input_size) * 0.01

        self.inputs = np.zeros(self.input_size)
        self.next_inputs = np.zeros(self.input_size)
        self.outputs = np.zeros(self.output_size)

        self.input_J = np.zeros(self.output_size)
        self.output_J = np.zeros(self.input_size)
        self.next_input_J = np.zeros(self.output_size)

        self.m_weights = np.zeros_like(self.weights)
        self.m_bias = np.zeros_like(self.bias)
        self.v_weights = np.zeros_like(self.weights)
        self.v_bias = np.zeros_like(self.bias)


'''
Neuron updating function (for parallelization)
    - Investigate moving this into the neuron class
'''
def update_neuron(neuron, optimizer, decay=0.0, update_metrics=False):
    dady = neuron.activation.dydx(neuron.outputs)
    dydx = neuron.weights
    dydw = neuron.inputs
    dydb = np.ones_like(neuron.bias)

    dldw = np.outer(dydw, neuron.input_J * dady)
    dldb = neuron.input_J * dady * dydb

    # Assume self.optimizer is adapted to be called here directly
    optimizer(neuron, [dldw, dldb])

    output_J = dydx @ (neuron.input_J * dady) * (1 - decay) # signal dropoff

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


'''
Class defining the neuron network

Contains:
    [Network]
    neruons         : List of pointers to neurons in network
    input_neurons   : List of pointers to neurons designated as inputs
    output_neurons  : List of pointers to neurons designated as outputs
    adjacency_matrix: Adjacency matrix defining graph structure

    [Model]
    max_depth       : Maximum depth of backpropagation paths, defined in compilation
'''
class ContinuousNetwork():
    def __init__(self, num_neurons, edge_probability, num_input_neurons=10, num_output_neurons=10, name='network'):
        self.name = name
        self.num_neurons = num_neurons
        self.num_input_neurons = num_input_neurons
        self.num_output_neurons = num_output_neurons
        self.initialize_network_fast(num_neurons, edge_probability)
        self.num_edges = np.sum(self.adjacency_matrix, dtype=np.int32)
        self.initialize_neuron_graph()
        self.optimizer = ADAM()

        self.metrics = {
            'loss': [],
            'energy':  [],
            'prop_input': [],

            'grad_energy': [],
            'prop_grad': [],

            'weight_energy': [],

            'pred': [],
            'true': [],
            'output_grad': []
        }

    '''
    Initializes the network adjacency matrix and neuron positions [fast version]
        
    - Randomly assign node positions
    - Locally assign connections in new adjacency matrix
    - Apply graph rules to new adjacency matrix
    '''
    def initialize_network_fast(self, num_neurons, edge_probability):
        debug_print(['Initializing network'])
        def sort_by_x(neuron_positions):
            # Sort the spatially arranged neurons by x position
            sorted_neurons = sorted(neuron_positions.items(), key=lambda item: item[1][0])
            sorted_list = [(index, position) for index, position in sorted_neurons]
            return sorted_list
        def sort_by_y(neuron_positions):
            # Sort the spatially arranged neurons by x position
            sorted_neurons = sorted(neuron_positions.items(), key=lambda item: item[1][1])
            sorted_list = [(index, position) for index, position in sorted_neurons]
            return sorted_list

        def generate_postiional_adjacency_matrix(neuron_positions, prob):
            num_neurons = len(neuron_positions)
            adjacency_matrix = np.zeros((num_neurons, num_neurons))

            rand = np.random.normal(size=(num_neurons, num_neurons))
            for i in tqdm(range(num_neurons)):
                for j in range(num_neurons):
                    m_distance = np.abs(neuron_positions[i][0] - neuron_positions[j][0]) + np.abs(neuron_positions[i][1] - neuron_positions[j][1])
                    
                    connection_probability = prob - m_distance

                    if rand[i][j] < connection_probability:
                        adjacency_matrix[i][j] = 1
            
            return adjacency_matrix

        neuron_positions = {i: [0, 0] for i in range(self.num_neurons)}
        for i in range(self.num_neurons):
            x, y = 1, 1
            while x ** 2 + y ** 2 > 1:
                x, y = np.random.random(2) * 2 - 1
            neuron_positions[i] = [x, y]

        self.neuron_positions = neuron_positions
        sorted_adjacency_matrix_positions = sort_by_x(neuron_positions)
        self.input_neuron_indices = [i[0] for i in sorted_adjacency_matrix_positions[:self.num_input_neurons]]
        # sorted_adjacency_matrix_positions = sort_by_y(neuron_positions)
        i = self.num_input_neurons + self.num_neurons // 10
        self.output_neuron_indices = [i[0] for i in sorted_adjacency_matrix_positions[i:i+self.num_output_neurons]]

        adjacency_matrix = generate_postiional_adjacency_matrix(neuron_positions, edge_probability)

        # - Neurons can't connect to themselves
        # - Input neurons can't recieve input
        # - Output neurons can't output to other neurons (yet)
        for i in range(num_neurons): adjacency_matrix[i, i] = False
        for i in self.input_neuron_indices: adjacency_matrix[:, i] = False
        for i in self.output_neuron_indices: adjacency_matrix[i, :] = False # Something to revisit

        self.adjacency_matrix = adjacency_matrix


    '''
    Initializes the network adjacency matrix and neuron positions
        
    - Create a dummy adjacency matrix
    - Apply spring algorithm to matrix to position neurons
    - Locally assign connections in new adjacency matrix
    - Apply graph rules to new adjacency matrix
    '''
    def initialize_network(self, num_neurons, edge_probability):
        debug_print(['Initializing network'])
        def sort_by_x(neuron_positions):
            # Sort the spatially arranged neurons by x position
            sorted_neurons = sorted(neuron_positions.items(), key=lambda item: item[1][0])
            sorted_list = [(index, position) for index, position in sorted_neurons]
            return sorted_list
        def sort_by_y(neuron_positions):
            # Sort the spatially arranged neurons by x position
            sorted_neurons = sorted(neuron_positions.items(), key=lambda item: item[1][1])
            sorted_list = [(index, position) for index, position in sorted_neurons]
            return sorted_list

        def generate_postiional_adjacency_matrix(neuron_positions, prob):
            num_neurons = len(neuron_positions)
            adjacency_matrix = np.zeros((num_neurons, num_neurons))

            for i in range(num_neurons):
                for j in range(num_neurons):
                    distance = np.sqrt((neuron_positions[i][0] - neuron_positions[j][0]) ** 2 + \
                                       (neuron_positions[i][1] - neuron_positions[j][1]) ** 2)
                    
                    connection_probability = 1 / (1 + 2 ** (10 * (distance - prob))) / 2
                    # print(round(connection_probability, 3), round(distance, 3))

                    if np.random.uniform() < connection_probability:
                        adjacency_matrix[i][j] = 1
            
            return adjacency_matrix

        model_adjacency_matrix = np.random.random((num_neurons, num_neurons)) < 0.5
        adjacency_matrix_positions = nx.spring_layout(nx.Graph(model_adjacency_matrix))
        self.neuron_positions = adjacency_matrix_positions
        sorted_adjacency_matrix_positions = sort_by_x(adjacency_matrix_positions)
        self.input_neuron_indices = [i[0] for i in sorted_adjacency_matrix_positions[:self.num_input_neurons]]
        sorted_adjacency_matrix_positions = sort_by_y(adjacency_matrix_positions)
        self.output_neuron_indices = [i[0] for i in sorted_adjacency_matrix_positions[-self.num_output_neurons:]]

        adjacency_matrix = generate_postiional_adjacency_matrix(adjacency_matrix_positions, edge_probability)

        # - Neurons can't connect to themselves
        # - Input neurons can't recieve input
        # - Output neurons can't output to other neurons (yet)
        for i in range(num_neurons): adjacency_matrix[i, i] = False
        for i in self.input_neuron_indices: adjacency_matrix[:, i] = False
        for i in self.output_neuron_indices: adjacency_matrix[i, :] = False # Something to revisit

        self.adjacency_matrix = adjacency_matrix

    '''
    Creates a neuron graph from adjacency matrix
    '''
    def initialize_neuron_graph(self):
        debug_print(['Initializing neuron graph'])
        neurons = []
        for i in range(self.num_neurons):
            neuron = Neuron(i)
            neuron.valid_prev = [[] for _ in range(len(self.output_neuron_indices))]
            neurons.append(neuron)

        for i in range(self.adjacency_matrix.shape[0]):
            for j in range(self.adjacency_matrix.shape[1]):
                if self.adjacency_matrix[i, j]:
                    neurons[i].next.append(neurons[j])
                    neurons[j].prev.append(neurons[i])

        self.neurons = neurons
        self.input_neurons = [self.neurons[i] for i in self.input_neuron_indices]
        self.output_neurons = [self.neurons[i] for i in self.output_neuron_indices]

        for neuron in self.input_neurons: neuron.prev.append(None)
        for neuron in self.output_neurons: neuron.next.append(None)
        # for neuron in self.neurons:
        #     # Add inhibitory neurons (control energy)
        #     if np.random.random() < 0.4:
        #         neuron.activation = NReLU()
        for neuron in self.output_neurons: neuron.activation = Linear()

        for neuron in neurons:
            neuron.initialize_weights()

    '''
    Information display function for network
        - Look at putting in some KPIs in the future
    '''
    def print_info(self):
        debug_print([
            '\nNumber of neurons     :', self.num_neurons,
            '\nNumber of connections :', self.num_edges,
            '\nInput indices         :', self.input_neuron_indices,
            '\nOutput indices        :', self.output_neuron_indices])
    
    '''
    Set optimization parameters
    '''
    def set_optimizer(self, optimizer):
        self.optimizer = optimizer

    '''
    Compute forward pass step
    '''
    def forward_pass(self, inputs, decay=0):
        network_output = np.zeros(self.num_output_neurons)
        for i, neuron in enumerate(self.input_neurons):
            # print(inputs)
            neuron.inputs = np.array([inputs[i]])

        for neuron in self.neurons:
            if not neuron in self.input_neurons:
                neuron.inputs = neuron.next_inputs

        for neuron in self.neurons:
            neuron.outputs = neuron.activation((neuron.inputs + neuron.bias) @ neuron.weights) # * (1 - decay)
            if neuron in self.output_neurons:
                output_index = self.output_neurons.index(neuron)
                network_output[output_index] = neuron.outputs
            else:
                for next, output in zip(neuron.next, neuron.outputs):
                    input_index = next.prev.index(neuron)
                    next.next_inputs[input_index] = output

        return network_output

    '''
    Parallelized backward pass [SLOW! UNUSED]
    '''
    def parallel_backward_pass(self, loss, y, y_hat, decay=0, update_metrics=True):
        # Metrics
        energy = 0
        grad_energy = 0
        weight_energy = 0
        prop_input = 0
        prop_grad = 0

        dlda = loss.dydx(y, y_hat)
        for i, neuron in enumerate(self.output_neurons):
            neuron.input_J = np.array([dlda[i]])

        for neuron in self.neurons:
            if not neuron in self.output_neurons:
                neuron.input_J = neuron.next_input_J
        
        with ProcessPoolExecutor() as executor:
            results = list(executor.map(
                update_neuron, 
                self.neurons, 
                [self.optimizer] * len(self.neurons), 
                [decay] * len(self.neurons), 
                [update_metrics] * len(self.neurons)))

        for neuron, (output_J, metrics) in zip(self.neurons, results):
            if update_metrics:
                energy += metrics['energy']
                grad_energy += metrics['grad_energy']
                weight_energy += metrics['weight_energy']
                prop_input += metrics['prop_input']
                prop_grad += metrics['prop_grad']
            
            if neuron in self.input_neurons:
                continue
            for prev, prev_J in zip(neuron.prev, output_J):
                prev.input_J = prev_J
        
        if update_metrics:
            self.metrics['loss'].append(loss(y, y_hat))
            self.metrics['pred'].append(np.sum(y_hat))
            self.metrics['true'].append(np.sum(y))
            
            self.metrics['energy'].append(energy)
            self.metrics['grad_energy'].append(grad_energy)
            self.metrics['weight_energy'].append(weight_energy)
            self.metrics['prop_input'].append(prop_input)
            self.metrics['prop_grad'].append(prop_grad)

    '''
    Compute backward pass step
    '''
    def backward_pass(self, loss, y, y_hat, decay=0, update_metrics=True):
        # Metrics
        energy = 0
        grad_energy = 0
        weight_energy = 0
        prop_input = 0
        prop_grad = 0

        dlda = loss.dydx(y, y_hat)
        for i, neuron in enumerate(self.output_neurons):
            neuron.input_J = np.array([dlda[i]])

        for neuron in self.neurons:
            if not neuron in self.output_neurons:
                neuron.input_J = neuron.next_input_J

        for neuron in self.neurons:
            dady = neuron.activation.dydx(neuron.outputs)
            dydx = neuron.weights
            dydw = neuron.inputs
            dydb = neuron.weights

            dldw = np.outer(dydw, neuron.input_J * dady)
            dldb = dydb @ (neuron.input_J * dady)

            self.optimizer(neuron, [dldw, dldb])

            output_J = dydx @ (neuron.input_J * dady) # * (1 - decay) # signal dropoff

            # Metrics
            if update_metrics: #and False: # disable for speed reasons
                energy += np.sum(np.abs(neuron.inputs)) / self.num_neurons
                grad_energy += np.sum(np.abs(neuron.input_J)) / self.num_neurons
                weight_energy += np.sum(np.abs(neuron.weights)) / self.num_neurons
                prop_input += np.mean(neuron.inputs > 1e-7) / self.num_neurons
                prop_grad += np.mean(neuron.input_J > 1e-7) / self.num_neurons

            if neuron in self.input_neurons:
                continue
            for prev, prev_J in zip(neuron.prev, output_J):
                input_index = prev.next.index(neuron)
                prev.next_input_J[input_index] = prev_J

        if update_metrics:
            self.metrics['loss'].append(loss(y, y_hat))
            self.metrics['pred'].append(np.sum(y_hat))
            self.metrics['true'].append(np.sum(y))
            
            self.metrics['energy'].append(energy)
            self.metrics['grad_energy'].append(grad_energy)
            self.metrics['weight_energy'].append(weight_energy)
            self.metrics['prop_input'].append(prop_input)
            self.metrics['prop_grad'].append(prop_grad)

    '''
    Clears all network jacobians
    '''
    def clear_jacobians(self, alpha=0):
        for neuron in self.neurons:
            neuron.input_J *= alpha
            neuron.next_input_J *= alpha


    '''
    Applies some noise to network weights
    '''
    def noise(self, alpha=0.0001):
        for neuron in self.neurons:
            neuron.weights += np.random.normal(size=neuron.weights.shape) * alpha

        
    '''
    Unused - computes computable network metrics
    '''
    def update_metrics(self):
        def compute_internal_energy(self):
            return np.mean([np.sum(np.abs(neuron.inputs)) for neuron in self.neurons])
        
        def compute_gradient_energy(self):
            return np.mean([np.sum(np.abs(neuron.input_J)) for neuron in self.neurons])
        
        def compute_weight_energy(self):
            return np.mean([np.sum(np.abs(neuron.weights)) for neuron in self.neurons])
        
        def compute_prop_input(self):
            return 1 - np.mean(np.array([np.sum(np.abs(neuron.inputs)) for neuron in self.neurons]) < 1e-6)
        
        def compute_prop_grad(self):
            return 1 - np.mean(np.array([np.sum(np.abs(neuron.input_J)) for neuron in self.neurons]) < 1e-6)
        
        self.metrics['energy'].append(compute_internal_energy(self))
        self.metrics['grad_energy'].append(compute_gradient_energy(self))
        self.metrics['weight_energy'].append(compute_weight_energy(self))
        self.metrics['prop_input'].append(compute_prop_input(self))
        self.metrics['prop_grad'].append(compute_prop_grad(self))

    def get_metrics_string(self, metrics=['pred', 'loss', 'energy', 'grad_energy', 'prop_input', 'prop_grad']):
        metrics_string = ''
        for metric in metrics:
            metrics_string += metric + '=' + f'{np.sum(self.metrics[metric][-1]) : 10.3f}' + '; '
        return metrics_string[:-2]

    def plot_metrics(self, title='', metrics=['pred', 'true', 'loss', 'energy', 'grad_energy']):
        for metric in metrics:
            plt.plot([max(-100, min(1000, x)) for x in self.metrics[metric]], label=metric)

        plt.title(title)
        plt.legend()
        plt.show()



def main():
    network = ContinuousNetwork(
    num_neurons         = 64,
    edge_probability    = 0.8,
    num_input_neurons   = 4,
    num_output_neurons  = 1
    )
    # visualize_network(network)
    # visualize_network(network, paths=[[1, 4, 5, 3, 4, 7, 8]], show_weight=True)
    plot_graph(network)


if __name__ == '__main__':
    main()