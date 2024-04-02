import networkx as nx
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from collections import deque
from tqdm import tqdm

from utils import *


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
Mean squared error loss
'''
class MeanSquaredError(Diff):
    def __init__(self):
        super().__init__()
    
    def __call__(self, y, y_hat):
        return np.mean(np.square(y - y_hat))
    
    def dydx(self, y, y_hat):
        return 2 / max(1, np.sum(y.shape)) * (y - y_hat)


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

        self.weight = np.random.normal()
        self.bias = 0
        self.activation = Linear()

        self.inputs = []
        self.outputs = []

        self.compiled = False
    
    def initialize_weights(self):
        np.random.seed(42) # optional deterministic weight initialization
        self.input_size = max(len(self.prev), 1)
        self.output_size = len(self.next)
        self.weights = np.random.normal(size=(self.input_size, self.output_size))
        self.bias = np.zeros(self.output_size)


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
class Network():
    def __init__(self, num_neurons, edge_probability, num_input_neurons=10, num_output_neurons=10, name='network'):
        self.name = name
        self.num_neurons = num_neurons
        self.num_input_neurons = num_input_neurons
        self.num_output_neurons = num_output_neurons
        self.initialize_network(num_neurons, edge_probability)
        self.num_edges = np.sum(self.adjacency_matrix, dtype=np.int32)
        self.initialize_neuron_graph()

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

        model_adjacency_matrix = np.random.random((num_neurons, num_neurons)) < edge_probability
        adjacency_matrix_positions = nx.spring_layout(nx.Graph(model_adjacency_matrix))
        self.neuron_positions = adjacency_matrix_positions
        sorted_adjacency_matrix_positions = sort_by_x(adjacency_matrix_positions)
        
        self.input_neuron_indices = [i[0] for i in sorted_adjacency_matrix_positions[:self.num_input_neurons]]
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
    Find backpropagation paths from network via DFS
    '''
    def compile(self, max_depth=6):
        self.max_depth = max_depth
        debug_print(['Compiling network'])
        def find_neuron_inputs(neuron, depth, output_index):
            neuron.compiled = True
            neuron.inputs = [[0 for _ in range(neuron.weights.shape[0])] for _ in range(len(self.output_neuron_indices))]
            neuron.outputs = [[0 for _ in range(neuron.weights.shape[1])] for _ in range(len(self.output_neuron_indices))]

            if neuron.id in self.input_neuron_indices: return True
            if depth > max_depth: return False

            valid_prev = []
            for prev in neuron.prev:
                if find_neuron_inputs(prev, depth + 1, output_index):
                    valid_prev.append(prev)
            
            neuron.valid_prev[output_index] = valid_prev

            if len(valid_prev) > 0: return True
            else: return False

        for i, output_neuron in enumerate(self.output_neurons):
            find_neuron_inputs(output_neuron, depth=0, output_index=i)
    
    '''
    Compute forward pass depth-first
    '''
    def forward_pass(self, inputs):
        def compute_neuron(neuron, next, depth, output_index):
            if depth > self.max_depth: return 0
            if not neuron: return 0

            neuron_inputs = np.array([0])
            if neuron.id in self.input_neuron_indices:
                i = self.input_neuron_indices.index(neuron.id)
                neuron_inputs = np.array([inputs[i]])
            elif len(neuron.valid_prev[output_index]) == 0: return 0

            if neuron.id not in self.input_neuron_indices:
                neuron_inputs = np.array([
                    compute_neuron(prev, next=neuron, depth=depth + 1, output_index=output_index) 
                    for prev in neuron.prev])
            
                # if len(neuron_inputs.shape) == 2: neuron_inputs = neuron_inputs[:, 0]
            
            neuron.inputs[output_index] = neuron_inputs
            
            neuron_outputs = neuron.activation(neuron_inputs @ neuron.weights + neuron.bias)

            next_index = neuron.next.index(next)
            neuron_outputs = neuron_outputs[next_index]
            neuron.outputs[output_index][next_index] = neuron_outputs
            return neuron_outputs

        outputs = []
        for i, output_neuron in enumerate(self.output_neurons):
            outputs.append(compute_neuron(output_neuron, None, depth=0, output_index=i))
        
        return np.array(outputs)
    
    '''
    Compute backward pass breadth-first
    '''
    def backward_pass(self, loss, y, y_hat, alpha=0.01):
        queue = deque([])  # Each item is (neuron, depth, J, output_index, next)

        dlda = loss.dydx(y, y_hat)
        for i, neuron in enumerate(self.output_neurons):
            queue.append((neuron, 0, dlda[i], i, None))

        while queue:
            current, depth, J, output_index, next = queue.popleft()
            
            if depth > self.max_depth: continue
            if J is None or np.sum(J) == 0: continue
            if len(current.valid_prev[output_index]) == 0:
                if not current.id in self.input_neuron_indices:
                    continue
            
            inputs = current.inputs[output_index]
            outputs = current.outputs[output_index]

            if not next: next_index = 0
            else: next_index = current.next.index(next)

            outputs_next = outputs[next_index]
            dady = current.activation.dydx(outputs_next)
            dydx = current.weights[:, next_index]
            dydw = inputs
            dydb = 1

            dldw = J * dady * dydw
            dldb = J * dady * dydb

            current.weights[:, next_index] -= alpha * np.squeeze(dldw)
            current.bias[next_index] -= alpha * dldb
            J = J * dady * dydx
                        
            for i, prev in enumerate(current.valid_prev[output_index]):
                queue.append((prev, depth + 1, J[i], output_index, current))


def main():
    network = Network(
    num_neurons         = 64,
    edge_probability    = 0.8,
    num_input_neurons   = 4,
    num_output_neurons  = 1
    )
    # visualize_network(network)
    visualize_network(network, paths=[[1, 4, 5, 3, 4, 7, 8]], show_weight=True)


if __name__ == '__main__':
    main()