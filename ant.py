import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor

from old.datasets import *
from utils import *

from core.activations import *
from core.optimizers import *
from core.losses import *
from core.neuron import Neuron

from network import Network

'''
Class defining the artificial neural topology

Contains:
    [Network]
    neruons         : List of pointers to neurons in network
    input_neurons   : List of pointers to neurons designated as inputs
    output_neurons  : List of pointers to neurons designated as outputs
    adjacency_matrix: Adjacency matrix defining graph structure

    [Model]
    max_depth       : Maximum depth of backpropagation paths, defined in compilation
'''
class ANT(Network):
    def __init__(self, num_neurons, edge_probability, num_input_neurons=8, num_output_neurons=8, name='ant'):
        super().__init__(num_neurons, edge_probability, num_input_neurons, num_output_neurons, name)
        self.initialize_network(edge_probability)
        self.num_edges = np.sum(self.adjacency_matrix, dtype=np.int32)
        self.initialize_neuron_graph()
        self.optimizer = RMSProp(alpha=1e-5, beta=0.99)
        self.use_metrics = True

    '''
    Initializes the network adjacency matrix and neuron positions [fast version]
        
    - Randomly assign node positions
    - Locally assign connections in new adjacency matrix
    - Apply graph rules to new adjacency matrix
    '''
    def initialize_network(self, edge_probability):
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

        def generate_positional_adjacency_matrix(neuron_positions, prob):
            self.prob = prob
            num_neurons = len(neuron_positions)
            adjacency_matrix = np.zeros((num_neurons, num_neurons))

            rand = np.random.uniform(0, 1, size=(num_neurons, num_neurons))
            for i in tqdm(range(num_neurons)):
                for j in range(num_neurons):
                    distance = np.abs(neuron_positions[i][0] - neuron_positions[j][0]) + np.abs(neuron_positions[i][1] - neuron_positions[j][1])
                    if rand[i][j] * self.prob > distance:
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
        for i, (x, y) in x_sorted_adjacency_matrix_positions[::-1]:
            if not i in allocated_neuron_indices:
                self.output_neuron_indices.append(i) 
                allocated_neuron_indices.add(i)
            if len(self.output_neuron_indices) >= self.num_output_neurons: break

        adjacency_matrix = generate_positional_adjacency_matrix(neuron_positions, edge_probability)

        self.mandate_adjacency_rules(adjacency_matrix)

        # Issue graph warnings
        for i in self.input_neuron_indices:
            if (1 - adjacency_matrix[i, :]).all(): debug_print(['WARNING: disconencted input neuron'])
        for i in self.output_neuron_indices:
            if (1 - adjacency_matrix[:, i]).all(): debug_print(['WARNING: disconencted output neuron'])

        self.adjacency_matrix = adjacency_matrix

    def mandate_adjacency_rules(self, adjacency_matrix):
        # - Neurons can't connect to themselves
        # - Input neurons can't recieve input
        # - Output neurons can't output to other neurons (yet)
        for i in range(adjacency_matrix.shape[0]): adjacency_matrix[i, i] = False
        for i in self.input_neuron_indices: adjacency_matrix[:, i] = False
        for i in self.output_neuron_indices: adjacency_matrix[i, :] = False # Something to revisit

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
        self.mu_neurons = [self.neurons[i] for i in self.output_neuron_indices[:self.num_output_neurons // 2]]
        self.var_neurons = [self.neurons[i] for i in self.output_neuron_indices[self.num_output_neurons // 2:]]

        for neuron in self.input_neurons: neuron.prev.append(None)
        for neuron in self.output_neurons: neuron.next.append(None)
        for neuron in self.output_neurons: neuron.activation = Linear()

        for neuron in neurons:
            neuron.initialize_weights()

    '''
    Append neurons 
    '''

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
    Mutates parameters of network for genetic algorithm
    '''
    def mutate(self, neuron_mutation_rate=0.5, edge_mutation_rate=0.02, weight_mutation_rate=0.0001):
        num_new_neurons = np.random.poisson(neuron_mutation_rate)

        for n in range(self.num_neurons, self.num_neurons + num_new_neurons):
            neuron = Neuron(n)
            self.neurons.append(neuron)

        # Modify existing connections
        dense_edges = self.num_neurons * (self.num_neurons - 1) / 2
        probs = [self.num_edges / dense_edges, (dense_edges - self.num_edges) / dense_edges]
        rand = np.random.uniform(low=0, high=1, size=(self.num_neurons + num_new_neurons, self.num_neurons + num_new_neurons))
        for i in range(self.num_neurons):
            for j in range(self.num_neurons):
                if i not in self.output_neuron_indices and j not in self.input_neuron_indices and not i == j: # Maintaining graph rules
                    # CHECK: should consider looking at proximity here (only flipping nearest neighbors)
                    # distance = np.abs(self.neuron_positions[i][0] - self.neuron_positions[j][0]) + np.abs(self.neuron_positions[i][1] - self.neuron_positions[j][1])
                    # if rand[i][j] * edge_mutation_rate > distance:
                    connection = int(self.adjacency_matrix[i][j])
                    if rand[i][j] < edge_mutation_rate * probs[connection]:
                        if connection:
                            self.adjacency_matrix[i][j] = 0
                            self.neurons[i].remove_next_connection(self.neurons[j])
                            self.neurons[j].remove_prev_connection(self.neurons[i])
                        else:
                            self.adjacency_matrix[i][j] = 1
                            self.neurons[i].insert_next_connection(self.neurons[j])
                            self.neurons[j].insert_prev_connection(self.neurons[i])

        # Create new connections
        for n in range(self.num_neurons, self.num_neurons + num_new_neurons):
            # Define position
            x, y = 1, 1
            while x ** 2 + y ** 2 > 1:
                x, y = np.random.random(2) * 2 - 1
            self.neuron_positions[n] = [x, y]

            # Define adjacency
            row = np.zeros(self.adjacency_matrix.shape[1])
            col = np.zeros((self.adjacency_matrix.shape[0] + 1, 1))
            for i in range(n):
                distance = np.abs(self.neuron_positions[i][0] - x) + np.abs(self.neuron_positions[i][1] - y)
                if i not in self.input_neuron_indices:
                    if rand[n][i] * self.prob > distance:
                        row[i] = 1
                        self.neurons[n].insert_next_connection(self.neurons[i])
                        self.neurons[i].insert_prev_connection(self.neurons[n])
                if i not in self.output_neuron_indices:
                    if rand[i][n] * self.prob > distance:
                        col[i] = 1
                        self.neurons[i].insert_next_connection(self.neurons[n])
                        self.neurons[n].insert_prev_connection(self.neurons[i])
            self.adjacency_matrix = np.vstack([self.adjacency_matrix, row])
            self.adjacency_matrix = np.hstack([self.adjacency_matrix, col])

        self.num_neurons += num_new_neurons

        # Tickle weights
        for neuron in self.neurons:
            neuron.mutate_weights(weight_mutation_rate)

        # self.initialize_neuron_graph()
        self.num_edges = np.sum(self.adjacency_matrix, dtype=np.int32)

    
    '''
    Compute forward pass step
    '''
    def forward(self, inputs):
        network_output = np.zeros(self.num_output_neurons)
        for i, neuron in enumerate(self.input_neurons):
            neuron.inputs = np.array([inputs[i]])

        for neuron in self.neurons:
            if not neuron in self.input_neurons:
                neuron.inputs = neuron.next_inputs

        for neuron in self.neurons:
            neuron.outputs = neuron.forward()
            if neuron in self.output_neurons:
                output_index = self.output_neurons.index(neuron)
                network_output[output_index] = neuron.outputs
            else:
                for next, output in zip(neuron.next, neuron.outputs):
                    input_index = next.prev.index(neuron)
                    next.next_inputs[input_index] = output

        return network_output

    '''
    Compute backward pass step given dL/dy
    '''
    def backward(self, dlda, clip=None, update_metrics=True, t=0, accumulate=True):
        # Metrics
        energy = 0
        grad_energy = 0
        weight_energy = 0
        prop_input = 0
        prop_grad = 0
        grad_mean_square_mag = 0

        for i, neuron in enumerate(self.output_neurons):
            neuron.input_J = np.array([dlda[i]])

        for neuron in self.neurons:
            if not neuron in self.output_neurons:
                neuron.input_J = neuron.next_input_J

        for neuron in self.neurons:
            dldw, dldb, output_J = neuron.backward()

            if accumulate:
                neuron.accumulated_weight_gradients.append(dldw)
                neuron.accumulated_bias_gradients.append(dldb)
            else:
                if type(self.optimizer) == ADAM: self.optimizer(neuron, [dldw, dldb], t + 1)
                else: self.optimizer(neuron, [dldw, dldb])

            # Metrics
            if update_metrics: # disable for speed reasons
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
                debug_print(['GRADIENTS CLIPPED'])
                clip_factor = clip / np.sqrt(grad_mean_square_mag)
                for neuron in self.neurons:
                    neuron.next_input_J *= clip_factor


        if update_metrics:            
            self.metrics['energy'].append(energy)
            self.metrics['grad_energy'].append(grad_energy)
            self.metrics['weight_energy'].append(weight_energy)
            self.metrics['prop_input'].append(prop_input)
            self.metrics['prop_grad'].append(prop_grad)

    def discrete_act(self, state, pool=None):
        if pool: logits = self.parallel_forward(state, pool) 
        else: logits = self.forward(state)
        probs = Softmax()(logits)
        return np.argmax(probs), probs

    '''
    Parallelized forward step
    '''
    def neuron_forward(self, neuron):
        neuron.outputs = neuron.forward()
        if neuron in self.output_neurons:
            output_index = self.output_neurons.index(neuron)
            self.network_output[output_index] = neuron.outputs
        else:
            for next, output in zip(neuron.next, neuron.outputs):
                input_index = next.prev.index(neuron)
                next.next_inputs[input_index] = output

    def parallel_forward(self, inputs, pool):
        self.network_output = np.zeros(self.num_output_neurons)
        for i, neuron in enumerate(self.input_neurons):
            neuron.inputs = np.array([inputs[i]])

        for neuron in self.neurons:
            if not neuron in self.input_neurons:
                neuron.inputs = neuron.next_inputs
         
        pool.map(self.neuron_forward, self.neurons)

        return self.network_output


    '''
    Parallelized backward step
    '''
    def neuron_backward(self, neuron): 
        dldw, dldb, output_J = neuron.backward()
        self.optimizer(neuron, [dldw, dldb])

        if neuron in self.input_neurons: return
        for prev, prev_J in zip(neuron.prev, output_J):
            input_index = prev.next.index(neuron)
            prev.next_input_J[input_index] = prev_J

    def parallel_backward(self, dlda, pool, clip=None):
        for i, neuron in enumerate(self.output_neurons):
            neuron.input_J = np.array([dlda[i]])

        for neuron in self.neurons:
            if not neuron in self.output_neurons:
                neuron.input_J = neuron.next_input_J

        pool.map(self.neuron_backward, self.neurons)


def main():
    network = ANT(
    num_neurons         = 256,
    edge_probability    = 1.8,
    num_input_neurons   = 16,
    num_output_neurons  = 8
    )
    network.optimizer = RMSProp(alpha=1e-4)
    network.print_info()
    # plot_graph(network)

    # X, Y = ripple_sinusoidal_pulse(time=1000, n=10)
    # network.fit(X, Y, plot=False)

    mutation_args = {
        'neuron_mutation_rate': 1,
        'edge_mutation_rate': 0.01,
        'weight_mutation_rate': 0.001
    }

    visualize_evolution(network, mutation_args=mutation_args, gif=True, time=100)

    # network.fit(X, Y, plot=False)


if __name__ == '__main__':
    main()