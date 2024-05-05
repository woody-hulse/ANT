from network import *

from core.activations import *
from core.optimizers import *
from core.losses import *
from core.neuron import Neuron

'''
Conventional Artificial Neural Network (ANN) custom implementation
'''
class ANN(Network):
    def __init__(self, layers=[1], name='ann'):
        num_input_neurons = layers[0]
        num_output_neurons = layers[-1]
        self.num_neurons = sum(layers)
        self.num_layers = len(layers)
        self.layer_sizes = layers
        super().__init__(self.num_neurons, None, num_input_neurons, num_output_neurons, name)
        self.initialize_network(layers)
        self.num_edges = np.sum(self.adjacency_matrix, dtype=np.int32)
        self.initialize_neuron_graph()
        self.optimizer = RMSProp(alpha=1e-3, beta=0.99)
        self.use_metrics = False
    
    def initialize_network(self, layers):
        self.adjacency_matrix = np.zeros((self.num_neurons, self.num_neurons), dtype=np.int32)
        self.neurons = [Neuron(i) for i in range(self.num_neurons)]
        self.input_neurons = self.neurons[:self.num_input_neurons]
        self.output_neurons = self.neurons[-self.num_output_neurons:]
        self.input_neuron_indices = [i for i in range(self.num_input_neurons)]
        self.output_neuron_indices = [i for i in range(self.num_neurons - self.num_output_neurons, self.num_neurons)]
        self.neuron_layers = [[self.neurons[j] for j in range(sum(layers[:i]), sum(layers[:i + 1]))] for i in range(len(layers))]
        for i in range(len(layers) - 1):
            for j in range(layers[i]):
                for k in range(layers[i + 1]):
                    self.adjacency_matrix[j + sum(layers[:i])][k + sum(layers[:i + 1])] = 1

    def initialize_neuron_graph(self):
        for i in range(self.num_neurons):
            self.neurons[i].next = [self.neurons[j] for j in np.where(self.adjacency_matrix[i] == 1)[0]]
            self.neurons[i].prev = [self.neurons[j] for j in np.where(self.adjacency_matrix[:, i] == 1)[0]]
        self.neuron_positions = {}
        for i, neuron_layer in enumerate(self.neuron_layers):
            for j, neuron in enumerate(neuron_layer):
                self.neuron_positions[neuron.id] = [i, j]

        for neuron in self.input_neurons: neuron.prev.append(None)
        for neuron in self.output_neurons: neuron.next.append(None)
        for neuron in self.output_neurons: neuron.activation = Linear()

        for neuron in self.neurons:
            neuron.initialize_weights()
            if neuron not in self.output_neurons: neuron.activation = Tanh()
    
    '''
    Mutates parameters of network for genetic algorithm
    '''
    def mutate(self, neuron_mutation_rate=0.5, edge_mutation_rate=None, weight_mutation_rate=0.0001):
        num_new_neurons_layers = [np.random.poisson(neuron_mutation_rate / self.num_layers) for _ in range(self.num_layers - 2)]

        for i, num_new_neurons in enumerate(num_new_neurons_layers):
            layer_index = i + 1
            for n in range(self.num_neurons, self.num_neurons + num_new_neurons):
                neuron = Neuron(n)
                self.neurons.append(neuron)
                self.neuron_layers[layer_index].append(neuron)

            # Create new connections
            for n in range(self.num_neurons, self.num_neurons + num_new_neurons):
                self.neuron_positions[n] = [layer_index, self.layer_sizes[layer_index] + n - self.num_neurons]

                # Define adjacency
                row = np.zeros(self.adjacency_matrix.shape[1])
                col = np.zeros((self.adjacency_matrix.shape[0] + 1, 1))

                for i in range(n):
                    if self.neurons[i] in self.neuron_layers[layer_index + 1]:
                        row[i] = 1
                        self.neurons[n].insert_next_connection(self.neurons[i])
                        self.neurons[i].insert_prev_connection(self.neurons[n])
                    if self.neurons[i] in self.neuron_layers[layer_index - 1]:
                        col[i] = 1
                        self.neurons[i].insert_next_connection(self.neurons[n])
                        self.neurons[n].insert_prev_connection(self.neurons[i])
                
                self.adjacency_matrix = np.vstack([self.adjacency_matrix, row])
                self.adjacency_matrix = np.hstack([self.adjacency_matrix, col])

            self.num_neurons += num_new_neurons
            self.layer_sizes[layer_index] += num_new_neurons

            # Tickle weights
            for neuron in self.neurons:
                neuron.mutate_weights(weight_mutation_rate)

            # self.initialize_neuron_graph()
            self.num_edges = np.sum(self.adjacency_matrix, dtype=np.int32)
    


    def forward(self, inputs):
        network_output = np.zeros(self.num_output_neurons)
        for i in range(self.num_input_neurons):
            self.neurons[i].inputs = np.array([inputs[i]])

        for layer in self.neuron_layers:
            for neuron in layer:
                neuron.outputs = neuron.forward()
                if neuron in self.output_neurons:
                    index = self.output_neurons.index(neuron)
                    network_output[index] = neuron.outputs[0]
                else:
                    for next, output in zip(neuron.next, neuron.outputs):
                        input_index = next.prev.index(neuron)
                        next.inputs[input_index] = output
        
        return network_output

    def backward(self, dlda, clip=None, t=0, accumulate=True):
        for i, neuron in enumerate(self.output_neurons):
            neuron.input_J = np.array([dlda[i]])
        
        for layer in reversed(self.neuron_layers):
            for neuron in layer:
                dldw, dldb, output_J = neuron.backward()
                if accumulate:
                    neuron.accumulated_weight_gradients.append(dldw)
                    neuron.accumulated_bias_gradients.append(dldb)
                else:
                    if type(self.optimizer) == ADAM: self.optimizer(neuron, [dldw, dldb], t + 1)
                    else: self.optimizer(neuron, [dldw, dldb])
                if neuron in self.input_neurons:
                    continue
                for prev, prev_J in zip(neuron.prev, output_J):
                    input_index = prev.next.index(neuron)
                    prev.input_J[input_index] = prev_J

    def discrete_act(self, state, pool=None):
        logits = self.forward(state)
        probs = Softmax()(logits)
        return np.random.choice(len(probs), p=probs), probs
    

def main():
    network = ANN(layers=[4, 10, 10, 2])
    # plot_graph(network)

    mutation_args = {
        'neuron_mutation_rate': 1,
        'edge_mutation_rate': 0.01,
        'weight_mutation_rate': 0.001
    }

    visualize_evolution(network, mutation_args=mutation_args, gif=True, time=100, spring=False)


if __name__ == '__main__':
    main()