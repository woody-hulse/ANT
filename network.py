import pickle

from utils import *
from core.losses import *
from core.optimizers import *

'''
Superclass for ANN, ANT
'''

class Network():
    def __init__(self, num_neurons, edge_probability, num_input_neurons=8, num_output_neurons=8, name='network'):
        self.name = name
        self.num_neurons = num_neurons
        self.num_input_neurons = num_input_neurons
        self.num_output_neurons = num_output_neurons
        self.optimizer = RMSProp()
        self.epsilon = 1e-7
        self.neurons = []

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

            'rewards': []
        }

    '''
    Mutates parameters of network
    '''
    def mutate(self):
        raise NotImplementedError('network.mutate() undefined')

    '''
    Initializes the network adjacency matrix and neuron positions [fast version]
        
    - Randomly assign node positions
    - Locally assign connections in new adjacency matrix
    - Apply graph rules to new adjacency matrix
    '''
    def initialize_network(self):
        raise NotImplementedError('network.initialize_network() undefined')
    
    '''
    Creates a neuron graph from adjacency matrix
    '''
    def initialize_neuron_graph(self):
        raise NotImplementedError('network.initialize_neuron_graph() undefined')

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
    Compute forward pass step
    '''
    def forward(self, inputs):
        raise NotImplementedError('network.forward() undefined')

    '''
    Compute backward pass step
    '''
    def backward(self, grad):
        raise NotImplementedError('network.backward() undefined')


    '''
    Apply accumulated gradients for each neuron
    '''
    def apply_accumulated_gradients(self, t=0):
        for neuron in self.neurons:
            if type(self.optimizer) == ADAM: self.optimizer(neuron, [neuron.accumulated_weight_gradients, neuron.accumulated_bias_gradients], t + 1)
            else: self.optimizer(neuron, [neuron.accumulated_weight_gradients, neuron.accumulated_bias_gradients])
            neuron.accumulated_weight_gradients = np.zeros_like(neuron.weights)
            neuron.accumulated_bias_gradients = np.zeros_like(neuron.bias)

    def compute_discounted_gradients(self, saved_gradients, rewards, gamma=0.99):
        discounted_gradients = [np.zeros_like(grad) for grad in saved_gradients[0]]
        T = len(rewards) - 1
        for t in range(len(saved_gradients)):
            for i in range(len(saved_gradients[t])):
                discounted_gradients[i] += saved_gradients[t][i] * rewards[t] * np.pow(gamma, T - t)
        return discounted_gradients
    
    '''
    Apply discounted accumulated gradients for each neuron
    '''
    def apply_discounted_accumulated_gradients(self, rewards, gamma=0.99, t=0):
        discounted_rewards = np.zeros(len(rewards))

        total = 0
        for t in reversed(range(len(rewards))):
            total = total * gamma + rewards[t]
            discounted_rewards[t] = total

        for neuron in self.neurons:
            discounted_weight_gradients = np.sum(np.array(neuron.accumulated_weight_gradients) * discounted_rewards[:, np.newaxis, np.newaxis], axis=0)
            discounted_bias_gradients = np.sum(np.array(neuron.accumulated_bias_gradients) * discounted_rewards[:, np.newaxis], axis=0)
            #  print(discounted_bias_gradients.shape, np.array(neuron.accumulated_bias_gradients).shape, discounted_rewards.shape, neuron.bias.shape)

            if type(self.optimizer) == ADAM: self.optimizer(neuron, [discounted_weight_gradients, discounted_bias_gradients], t + 1)
            else: self.optimizer(neuron, [discounted_weight_gradients, discounted_bias_gradients])
            # print(discounted_rewards, rewards)
            # print(neuron.id, self.output_neuron_indices, self.input_neuron_indices, discounted_weight_gradients, discounted_bias_gradients)
            neuron.accumulated_weight_gradients = [] # np.zeros_like(neuron.weights)
            neuron.accumulated_bias_gradients = []# np.zeros_like(neuron.bias)

    '''
    Save the network to file
    '''
    def save(self, path):
        data = {'network': self}
        with open(path, 'wb') as file:
            pickle.dump(data, file)
    
    '''
    Load a network from file
    '''
    def load(self, path):
        with open(path, 'rb') as file:
            data = pickle.load(file)
            self.__dict__.update(data['network'].__dict__)


    def reset_weights(self):
        for neuron in self.neurons:
            neuron.initialize_weights()


    '''
    Fit a simple input/output mapping
    '''
    def fit(self, X, Y, plot=True):
        loss_function = MSE()
        Y_pred = np.zeros_like(Y)
        losses = np.zeros_like(Y)
        for t, (x, y) in tqdm(enumerate(zip(X, Y))):
            y_pred = self.forward(x)
            loss = loss_function(y, y_pred)

            dlda = loss_function.dydx(y, y_pred)
            self.backward_(dlda, np.array([0]), None, False)

            Y_pred[t] = np.sum(y_pred)
            losses[t] = loss
        
        if plot:
            plt.plot(Y_pred, label='y pred')
            plt.plot(Y, label='y')
            plt.plot(losses, label='loss')
            plt.legend()
            plt.show()
    
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
        try:
            metrics_string = ''
            for metric in metrics:
                metrics_string += metric + '=' + f'{np.sum(self.metrics[metric][-1]) : 10.3f}' + '; '
            return metrics_string[:-2]
        except: return ''

    def plot_metrics(self, title='', metrics=['pred', 'true', 'loss', 'energy', 'grad_energy']):
        for metric in metrics:
            plt.plot([max(-100, min(1000, x)) for x in self.metrics[metric]], label=metric)

        plt.title(title)
        plt.legend()
        plt.show()