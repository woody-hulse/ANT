import networkx as nx
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from collections import deque
from tqdm import tqdm

from utils import *


class Layer():
    def __init__(self):
        self.output = np.array([])

class MSE(Layer):
    def __init__(self):
        super().__init__()
    
    def __call__(self, y, y_hat):
        return np.mean(np.square(y - y_hat))
    
    def dydx(self, y, y_hat):
        return 2 / max(1, np.sum(y.shape)) * (y - y_hat)


class ReLU(Layer):
    def __init__(self):
        super().__init__()

    def __call__(self, x):
        y = x * (x > 0)
        self.output = y
        return y
    
    def dydx(self, output):
        return np.array([output > 0])

class Linear(Layer):
    def __init__(self):
        super().__init__()

    def __call__(self, x):
        self.output = x
        return x
    
    def dydx(self, output):
        return np.array([output])


class Neuron:
    def __init__(self, id):
        self.id = id
        self.next = []
        self.prev = []
        self.valid_prev = []

        self.weight = np.random.normal()
        self.bias = 0

        self.inputs = []
        self.outputs = []
        self.inputs_ = deque([])
        self.outputs_ = deque([])

    def initialize_weights(self):
        self.input_size = len(self.prev)
        self.output_size = len(self.next)
        self.weights = np.random.normal(size=(self.input_size, self.output_size))
        self.bias = np.zeros(self.output_size)
        self.activation = Linear()

class NeuralNetwork():
    def __init__(self, num_neurons, edge_probability, num_input_neurons=10, num_output_neurons=10):
        self.num_neurons = num_neurons
        self.num_input_neurons = num_input_neurons
        self.num_output_neurons = num_output_neurons
        self.prune = False
        self.initialize_network(num_neurons, edge_probability)
        self.num_edges = np.sum(self.adjacency_matrix, dtype=np.int32)
        self.initialize_neuron_graph()



    def naive_initialize_network(self, num_neurons, edge_probability):
        adjacency_matrix = np.random.random((num_neurons, num_neurons)) < edge_probability

        # Neurons cannot connect to themselves, input neurons can't recieve input, and output 
        # neurons can't output to other neurons (yet)
        for i in range(num_neurons): adjacency_matrix[i, i] = False
        # for i in self.input_neurons: adjacency_matrix[:, i] = False
        for i in self.output_neurons: adjacency_matrix[i, :] = False # Something to revisit
        self.neuron_positions = nx.spring_layout(nx.Graph(adjacency_matrix))

        self.input_neuron_indices = list(range(self.num_neurons))[:self.num_input_neurons]
        self.output_neuron_indices = list(range(self.num_neurons))[-self.num_output_neurons:]

        return adjacency_matrix

    def initialize_network(self, num_neurons, edge_probability):

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

        # allow pruning
        self.prune = True

        self.adjacency_matrix = adjacency_matrix

    def initialize_neuron_graph(self):
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

    def print_info(self):
        print('Number of neurons     :', self.num_neurons)
        print('Number of connections :', self.num_edges)
        print('Input indices         :', self.input_neuron_indices)
        print('Output indices        :', self.output_neuron_indices)

    def visualize_network(self, paths=[], show_weight=False):
        G = nx.Graph(self.adjacency_matrix)
        G_ = G.copy()

        pos = self.neuron_positions

        # Set neuron colors
        input_neuron_color = (0.1, 0.35, 0.95)
        output_neuron_color = (0.9, 0.7, 0.2)
        hidden_neuron_color = (0.2, 0.2, 0.2)
        neuron_colors = [hidden_neuron_color for _ in range(len(G.nodes))]
        neuron_colors = [input_neuron_color if i in self.input_neuron_indices else neuron_colors[i] for i in range(len(G.nodes))]
        neuron_colors = [output_neuron_color if i in self.output_neuron_indices else neuron_colors[i] for i in range(len(G.nodes))]

        # Set neuron sizes
        neuron_sizes = [15 for _ in range(len(G.nodes))]
        neuron_sizes = [30 if i in self.input_neuron_indices + self.output_neuron_indices else neuron_sizes[i] for i in range(len(G.nodes))]

        edge_attributes = {}
        for path in paths:
            path_edge_attributes = {edge: {'color': 'skyblue', 'width': 1} for edge in zip(path[:-1], path[1:])}
            edge_attributes = {**edge_attributes, **path_edge_attributes}

        min_weight_color = np.array([0, 0, 0])
        max_weight_color = np.array([0.6, 0.75, 0.9])
        if show_weight:
            max_weight = 1
            for neuron in self.neurons: 
                for weight in neuron.weights: max_weight = max(max_weight, np.abs(np.sum(weight)))
            print(max_weight)

            for i, neuron in enumerate(self.neurons):
                for weight, next in zip(neuron.weights, neuron.next):
                    if not next: continue
                    next_i = self.neurons.index(next)
                    weight_mag = np.abs(np.sum(weight)) / max_weight
                    print(weight_mag)
                    color_value = plt.cm.Blues(weight_mag + 0.1)
                    edge_attributes[(i, next_i)] = {'color': min_weight_color + 0.6 * max_weight_color * weight_mag}

        nx.set_edge_attributes(G_, edge_attributes)

        nx.draw(G, pos, with_labels=False, font_weight='bold', node_size=neuron_sizes, node_color=neuron_colors, font_color='black', font_size=8, edge_color=(0.4, 0.4, 0.4), width=0.2)
        nx.draw_networkx_edges(G_, pos, edgelist=edge_attributes.keys(), edge_color='skyblue', width=1)

        # Create legend
        legend_elements = [
            matplotlib.lines.Line2D([0], [0], marker='o', color='w', markerfacecolor=input_neuron_color, markersize=8, label='Input Neurons'),
            matplotlib.lines.Line2D([0], [0], marker='o', color='w', markerfacecolor=output_neuron_color, markersize=8, label='Output Neurons'),
            matplotlib.lines.Line2D([0], [0], marker='o', color='w', markerfacecolor=hidden_neuron_color, markersize=6, label='Hidden Neurons')]

        plt.legend(handles=legend_elements, loc='upper right')

        plt.show()

    def find_path(self, start, end, max_depth, print_info=True, prune=True):
        if start not in range(len(self.adjacency_matrix)) or end not in range(len(self.adjacency_matrix)):
            raise ValueError("Invalid start or end neuron")
        
        start_time = time.time()
        queue = deque([(start, [start])])

        paths = []
        num_prune = 0

        current_depth = 0
        while queue:
            current_neuron, path = queue.popleft()
            if len(path) > current_depth:
                current_depth += 1
                print('Depth', current_depth, '/', max_depth)

            if current_neuron in self.input_neuron_indices:
                paths.append(path)

            if len(path) < max_depth:
                if self.prune and prune and len(path) > 2:
                    if (1 + self.neuron_positions[current_neuron][0]) + (1 - len(path) / max_depth) < self.neuron_positions[end][0]:
                        num_prune += 1
                        continue

                for neighbor, is_connected in enumerate(self.adjacency_matrix[:, current_neuron]):
                    if is_connected and neighbor not in path:
                        queue.append((neighbor, path + [neighbor]))

        end_time = time.time()

        if print_info:
            print('Path finding:')
            print('   Time                 :', round(end_time - start_time, 4), 's')
            print('   Number of paths      :', len(paths))
            print('   Number of prunes     :', num_prune)
            print('   Number of parameters :', sum([len(path) for path in paths]))


        return paths
    

    def compile(self, max_depth=7):
        self.max_depth = max_depth
        print('Compiling network')
        def find_neuron_inputs(neuron, depth, output_index):
            neuron.outputs = [np.array([]) for _ in range(len(self.output_neuron_indices))]
            if neuron.id in self.input_neuron_indices: return True
            if depth > max_depth: return False

            valid_prev = []
            for prev in neuron.prev:
                if find_neuron_inputs(prev, depth + 1, output_index): pass
                valid_prev.append(prev)
            
            neuron.valid_prev[output_index] = valid_prev
            neuron.inputs = [np.array([]) for _ in range(len(self.output_neuron_indices))]

            if len(valid_prev) > 0: return True
            else: return False

        for i, output_neuron in enumerate(self.output_neurons):
            find_neuron_inputs(output_neuron, depth=0, output_index=i)

    
    def compiled_forward_pass(self, inputs):
        def compute_neuron(neuron, next, depth, output_index):
            if depth > self.max_depth: return 0
            if not neuron: return 0

            neuron_inputs = None
            if neuron.id in self.input_neuron_indices:
                i = self.input_neuron_indices.index(neuron.id)
                neuron_inputs = np.array([inputs[i]])
                # print(1, inputs[i])

            if neuron.id not in self.input_neuron_indices:
                if len(neuron.valid_prev[output_index]) == 0 or len(neuron.prev) == 0: return 0
                
                neuron_inputs = np.array([
                    compute_neuron(prev, next=neuron, depth=depth + 1, output_index=output_index) 
                    for prev in neuron.valid_prev[output_index]])
            
                if len(neuron_inputs.shape) == 2: neuron_inputs = neuron_inputs[:, 0]
                neuron.inputs[output_index] = neuron_inputs
                # print(neuron_inputs)

            neuron_outputs = neuron_inputs @ neuron.weights + neuron.bias

            neuron_outputs = neuron_outputs[neuron.next.index(next)]
            activation_outputs = neuron.activation(neuron_outputs)
            neuron.outputs[output_index] = activation_outputs
            # print(activation_outputs)
            return activation_outputs

        outputs = []
        for i, output_neuron in enumerate(self.output_neurons):
            outputs.append(compute_neuron(output_neuron, None, depth=0, output_index=i))
        outputs = np.array(outputs)
        
        return outputs
    

    def compiled_backward_pass(self, loss, y, y_hat, alpha=0.01):
        def differentiate_neuron(neuron, J, depth, output_index):
            if not neuron or depth > self.max_depth: return

            try:
                inputs = neuron.inputs[output_index]
                outputs = neuron.outputs[output_index]
            except IndexError:
                return

            dady = neuron.activation.dydx(outputs)
            dydx = neuron.weights
            dydw = inputs
            dydb = 1

            dldw = np.outer(J * dady, dydw)
            dldb = J * dady * dydb

            # print(J, dady, dydb)

            neuron.weights -= alpha * dldw.T
            neuron.bias -= alpha * dldb[0]

            J_next = np.mean(J * dady * dydx, keepdims=True)
            for prev in neuron.valid_prev[output_index]:
                differentiate_neuron(prev, J_next, depth=depth + 1, output_index=output_index)

        dlda = loss.dydx(y, y_hat)
        for i, output_neuron in enumerate(self.output_neurons):
            differentiate_neuron(output_neuron, dlda, depth=0, output_index=i)


    def forward_pass(self, inputs, max_depth=7):
        def compute_neuron(neuron, next, depth):
            if not neuron: return np.array([0])
            if neuron.id in self.input_neuron_indices:
                i = self.input_neuron_indices.index(neuron.id)
                neuron_inputs = np.array([inputs[i]])
            if depth > max_depth: return np.array([0])
            if len(neuron.prev) == 0: return np.array([0])
            
            if not neuron.id in self.input_neuron_indices:
                neuron_inputs = np.array([compute_neuron(prev, next=neuron, depth=depth + 1) for prev in neuron.prev])
            if len(neuron_inputs.shape) == 2: neuron_inputs = neuron_inputs[:, 0]
            neuron.inputs_.append(neuron_inputs)

            neuron_outputs = neuron_inputs @ neuron.weights + neuron.bias

            if next: neuron_outputs = neuron_outputs[neuron.next.index(next)]
            activation_outputs = neuron.activation(neuron_outputs)
            neuron.outputs_.append(activation_outputs)
            return activation_outputs
    
        outputs = np.array([compute_neuron(output_neuron, None, depth=0) for output_neuron in self.output_neurons])
        
        return outputs
    
    def backward_pass(self, loss, y, y_hat, alpha=0.01, max_depth=7):
        def differentiate_neuron(neuron, J, depth=0):
            if not neuron or depth > max_depth: return

            try:
                inputs = neuron.inputs_.popleft()
                outputs = neuron.outputs_.popleft()
            except IndexError:
                return

            dady = neuron.activation.dydx(outputs)
            dydx = neuron.weights
            dydw = inputs
            dydb = 1

            dldw = np.outer(J * dady, dydw)
            dldb = J * dady * dydb
            # print(J.shape, dady.shape, dydw.shape, dldw.shape, neuron.weights.shape)
            # print(dldb.shape, neuron.bias.shape)

            neuron.weights -= alpha * dldw.T
            neuron.bias -= alpha * dldb[0]

            # print(J.shape, dady.shape, dydx.shape)
            J_next = np.mean(J * dady * dydx, keepdims=True)
            for prev in neuron.prev:
                differentiate_neuron(prev, J_next, depth + 1)

        dlda = loss.dydx(y, y_hat)
        for output_neuron in self.output_neurons:
            differentiate_neuron(output_neuron, dlda)


network = NeuralNetwork(
    num_neurons         = 32,
    edge_probability    = 0.7,
    num_input_neurons   = 4,
    num_output_neurons  = 1
)
# network.visualize_network()
# network.visualize_network(show_weight=True)

network.compile(max_depth=5)
print('\n\n')
network.print_info()

print('Training network')
def train_compiled(X, Y, epochs=10):
    loss_layer = MSE()
    for epoch in range(epochs):
        total_loss = 0
        train_size = len(X)
        for i in tqdm(range(train_size), desc=f'           Epoch {epoch + 1} | '):
            x, y = X[i], Y[i]
            x, y = np.array(x), np.array(y)
            # print('Forward pass')
            y_hat = network.compiled_forward_pass(x)
            total_loss += loss_layer(y, y_hat)
            # print('Backward Pass')
            network.compiled_backward_pass(loss_layer, y, y_hat, alpha=5e-6)
        print(f'Epoch {epoch + 1} | loss : {total_loss / train_size}')

def train(X, Y, epochs=10, max_depth=5):
    loss_layer = MSE()
    for epoch in range(epochs):
        total_loss = 0
        train_size = len(X)
        for i in tqdm(range(train_size), desc=f'           Epoch {epoch + 1} | '):
            x, y = X[i], Y[i]
            x, y = np.array(x), np.array(y)
            # print(x)
            y_hat = network.forward_pass(x, max_depth=max_depth)
            # print(y, y_hat)
            total_loss += loss_layer(y, y_hat)
            network.backward_pass(loss_layer, y, y_hat, alpha=5e-6, max_depth=max_depth)
        print(f'Epoch {epoch + 1} | loss : {total_loss / train_size}')


X = [[0.5488135,  0.71518937, 0.60276338, 0.54488318],
 [0.4236548,  0.64589411, 0.43758721, 0.891773  ],
 [0.96366276, 0.38344152, 0.79172504, 0.52889492],
 [0.56804456, 0.92559664, 0.07103606, 0.0871293 ],
 [0.0202184,  0.83261985, 0.77815675, 0.87001215]]
Y = [7.66535545, 8.01574137, 8.4595805, 6.04627032, 6.64351887]

train_compiled(X, Y, epochs=10)
train(X, Y, epochs=10)


'''
paths = network.find_path(
    start               = network.output_neuron_indices[0],
    end                 = network.input_neuron_indices[0],
    max_depth           = 7,
    print_info          = True,
    prune               = True
)

for path in paths:
    print(path)
print('Input neurons:', network.input_neuron_indices)
print('Output neurons: ', network.output_neuron_indices)

paths = network.find_path(
    start               = network.output_neuron_indices[0],
    end                 = network.input_neuron_indices[0],
    max_depth           = 7,
    print_info          = True,
    prune               = False
)


network.visualize_network(paths=paths)

'''