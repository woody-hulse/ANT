import networkx as nx
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from collections import deque
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor

from utils import *

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
    def __init__(self, num_neurons, edge_probability, num_input_neurons=8, num_output_neurons=8, num_critic_neurons=1, name='network'):
        self.name = name
        self.num_neurons = num_neurons
        self.num_input_neurons = num_input_neurons
        self.num_output_neurons = num_output_neurons
        self.num_critic_neurons = num_critic_neurons
        self.value = np.zeros(num_critic_neurons)
        self.initialize_network_fast(num_neurons, edge_probability)
        self.num_edges = np.sum(self.adjacency_matrix, dtype=np.int32)
        self.initialize_neuron_graph()
        self.optimizer = ADAM()
        self.epsilon = 1e-7

        self.metrics = {
            'loss': [],
            'energy':  [],
            'prop_input': [],

            'grad_energy': [],
            'prop_grad': [],

            'weight_energy': [],

            'pred': [],
            'true': [],
            'output_grad': [],

            'mean_var': [],
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

            rand = np.random.uniform(0, 1, size=(num_neurons, num_neurons))
            for i in tqdm(range(num_neurons)):
                for j in range(num_neurons):
                    distance = np.abs(neuron_positions[i][0] - neuron_positions[j][0]) + np.abs(neuron_positions[i][1] - neuron_positions[j][1])
                    if rand[i][j] * prob > distance:
                        adjacency_matrix[i][j] = 1
            
            return adjacency_matrix

        neuron_positions = {i: [0, 0] for i in range(self.num_neurons)}
        for i in range(self.num_neurons):
            x, y = 1, 1
            while x ** 2 + y ** 2 > 1:
                x, y = np.random.random(2) * 2 - 1
            neuron_positions[i] = [x, y]

        self.neuron_positions = neuron_positions
        allocated_neuron_indices = set()
        x_sorted_adjacency_matrix_positions = sort_by_x(neuron_positions)
        y_sorted_adjacency_matrix_positions = sort_by_y(neuron_positions)

        self.input_neuron_indices = []
        self.critic_neuron_indices = []
        self.output_neuron_indices = []
        for i, (x, y) in x_sorted_adjacency_matrix_positions:
            if not i in allocated_neuron_indices:
                self.input_neuron_indices.append(i)
                allocated_neuron_indices.add(i)
            if len(self.input_neuron_indices) >= self.num_input_neurons: break
        for i, (x, y) in y_sorted_adjacency_matrix_positions:
            if not i in allocated_neuron_indices:
                self.critic_neuron_indices.append(i)
                allocated_neuron_indices.add(i)
            if len(self.critic_neuron_indices) >= self.num_critic_neurons: break
        for i, (x, y) in x_sorted_adjacency_matrix_positions[::-1]:
            if not i in allocated_neuron_indices:
                self.output_neuron_indices.append(i) 
                allocated_neuron_indices.add(i)
            if len(self.output_neuron_indices) >= self.num_output_neurons: break

        adjacency_matrix = generate_postiional_adjacency_matrix(neuron_positions, edge_probability)

        # - Neurons can't connect to themselves
        # - Input neurons can't recieve input
        # - Output neurons can't output to other neurons (yet)
        for i in range(num_neurons): adjacency_matrix[i, i] = False
        for i in self.input_neuron_indices: adjacency_matrix[:, i] = False
        for i in self.output_neuron_indices: adjacency_matrix[i, :] = False # Something to revisit
        for i in self.critic_neuron_indices: adjacency_matrix[i, :] = False

        # Issue graph warnings
        for i in self.input_neuron_indices:
            if (1 - adjacency_matrix[i, :]).all(): debug_print(['WARNING: disconencted input neuron'])
        for i in self.critic_neuron_indices:
            if (1 - adjacency_matrix[:, i]).all(): debug_print(['WARNING: disconencted critic neuron'])
        for i in self.output_neuron_indices:
            if (1 - adjacency_matrix[:, i]).all(): debug_print(['WARNING: disconencted output neuron'])

        self.adjacency_matrix = adjacency_matrix

    '''
    Creates a neuron graph from adjacency matrix
    '''
    def initialize_neuron_graph(self):
        debug_print(['Initializing neuron graph'])
        neurons = []
        for i in range(self.num_neurons):
            neuron = Neuron(i)
            neurons.append(neuron)

        for i in range(self.adjacency_matrix.shape[0]):
            for j in range(self.adjacency_matrix.shape[1]):
                if self.adjacency_matrix[i, j]:
                    neurons[i].next.append(neurons[j])
                    neurons[j].prev.append(neurons[i])

        self.neurons = neurons
        self.input_neurons = [self.neurons[i] for i in self.input_neuron_indices]
        self.output_neurons = [self.neurons[i] for i in self.output_neuron_indices]
        self.critic_neurons = [self.neurons[i] for i in self.critic_neuron_indices]
        self.mu_neurons = [self.neurons[i] for i in self.output_neuron_indices[:self.num_output_neurons // 2]]
        self.var_neurons = [self.neurons[i] for i in self.output_neuron_indices[self.num_output_neurons // 2:]]

        for neuron in self.input_neurons: neuron.prev.append(None)
        for neuron in self.output_neurons: neuron.next.append(None)
        for neuron in self.critic_neurons: neuron.next.append(None)
        # for neuron in self.neurons:
        #     # Add inhibitory neurons (control energy)
        #     if np.random.random() < 0.4:
        #         neuron.activation = NReLU()
        for neuron in self.output_neurons: neuron.activation = Linear()
        for neuron in self.critic_neurons: neuron.activation = Linear()

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
            neuron.outputs = neuron.activation((neuron.inputs) @ neuron.weights) # * (1 - decay)
            if neuron in self.output_neurons:
                output_index = self.output_neurons.index(neuron)
                network_output[output_index] = neuron.outputs
            elif neuron in self.critic_neurons:
                critic_index = self.critic_neurons.index(neuron)
                self.value[critic_index] = neuron.outputs
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
            dldb = 0 # dydb @ (neuron.input_J * dady)

            self.optimizer(neuron, [dldw, dldb])

            output_J = dydx @ (neuron.input_J * dady) * (1 - decay) # signal dropoff

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
    Compute backward pass step given dL/dy, dL/dc
    '''
    def backward_pass_(self, dlda, dldc, decay=0, clip=None, update_metrics=True):
        # Metrics
        energy = 0
        grad_energy = 0
        weight_energy = 0
        prop_input = 0
        prop_grad = 0
        grad_mean_square_mag = 0

        for i, neuron in enumerate(self.output_neurons):
            neuron.input_J = np.array([dlda[i]])

        for i, neuron in enumerate(self.critic_neurons):
            neuron.input_J = np.array([dldc[i]])

        for neuron in self.neurons:
            if not neuron in self.output_neurons + self.critic_neurons:
                neuron.input_J = neuron.next_input_J

        for neuron in self.neurons:
            dady = neuron.activation.dydx(neuron.outputs)
            dydx = neuron.weights
            dydw = neuron.inputs
            dydb = neuron.weights

            dldw = np.outer(dydw, neuron.input_J * dady)
            dldb = 0 # dydb @ (neuron.input_J * dady)

            self.optimizer(neuron, [dldw, dldb])

            output_J = dydx @ (neuron.input_J * dady) # * (1 - decay) # signal dropoff

            # Metrics
            if update_metrics: #and False: # disable for speed reasons
                energy += np.sum(np.abs(neuron.inputs)) / self.num_neurons
                grad_energy += np.sum(np.abs(neuron.input_J)) / self.num_neurons
                weight_energy += np.sum(np.abs(neuron.weights)) / self.num_neurons
                prop_input += np.mean(neuron.inputs > 1e-7) / self.num_neurons
                prop_grad += np.mean(neuron.input_J > 1e-7) / self.num_neurons
            grad_mean_square_mag += np.sum(np.square(neuron.input_J)) / self.num_neurons

            if neuron in self.input_neurons:
                continue
            for prev, prev_J in zip(neuron.prev, output_J):
                input_index = prev.next.index(neuron)
                prev.next_input_J[input_index] = prev_J

        if clip:
            # print(np.sqrt(grad_mean_square_mag))
            if np.sqrt(grad_mean_square_mag) > clip:
                # print('CLIP')
                clip_factor = clip / np.sqrt(grad_mean_square_mag)
                for neuron in self.neurons:
                    neuron.next_input_J *= clip_factor


        if update_metrics:            
            self.metrics['energy'].append(energy)
            self.metrics['grad_energy'].append(grad_energy)
            self.metrics['weight_energy'].append(weight_energy)
            self.metrics['prop_input'].append(prop_input)
            self.metrics['prop_grad'].append(prop_grad)

    '''
    Gets predicted value of next action
    '''
    def get_next_value(self):
        value = np.zeros_like(self.value)
        for neuron in self.critic_neurons:
            neuron.outputs = neuron.activation((neuron.inputs + neuron.bias) @ neuron.weights)
            critic_index = self.critic_neurons.index(neuron)
            value[critic_index] = neuron.outputs
        
        return value

    '''
    Samples a continuous action
    '''
    def action(self, state, env):
        output = self.forward_pass(state)
        split = len(output) // 2
        mu, var = output[:split], output[split:]
        sigma = np.sqrt(var)
        actions = [np.random.normal(m, s) for m, s in zip(mu, sigma)]


        return actions
    
    '''
    Calculates log probability of continuous action space
    '''
    def calculate_logprobs(self, mu, sigma, action):
        a = -np.square(mu - action) / (2 * np.square(sigma) + self.epsilon)
        b = -np.log(np.sqrt(2 * np.pi * np.square(sigma)))
        return a + b

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