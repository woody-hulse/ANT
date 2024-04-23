'''
Classes of gym environments and their properties
'''

import gym
import numpy as np

from core.activations import *
from core.optimizers import *
from core.losses import *

class Environment():
    def __init__(self, env, continuous, mu_var=False, num_actions=1, minimizer=None):

        self.env = env
        self.observation_space = env.observation_space.shape[0]
        if continuous:
            if mu_var: self.action_space = env.action_space.shape[0] * 2
            else: self.action_space = env.action_space.shape[0]
        else:
            self.action_space = num_actions
        self.continuous = continuous
        self.mu_var = mu_var
        self.minimizer = minimizer

        # defaults
        self.neuron_activation = Sigmoid()
        if continuous: self.output_activation = Sigmoid()
        else: self.output_activation = Linear()
        self.mu_activation = Tanh()
        self.var_activation = Sigmoid()
        self.critic_activation = Linear()

        self.optimizer = ADAM(alpha=1e-6, beta1=0.9, beta2=0.9)
        # self.optimizer = RMSProp(alpha=1e-7, beta=0.99)

        self.parametrization = None
        self.gamma = 0.99

    def discretize_output(self, action):
        cutoffs = np.array([(i + 1) / self.num_actions for i in range(self.num_actions)])
        for i, cutoff in enumerate(cutoffs):
            if action <= cutoff: return i
        return -1
    
    def get_class_output(self, action):
        return np.argmax(action)
    
    def get_probabilisitc_class_output(self, action):
        return np.random.choice(action, size=1)

    def configure_newtork(self, network):
        if self.mu_var:
            for neuron in network.mu_neurons: neuron.activation = self.mu_activation
            for neuron in network.var_neurons: neuron.activation = self.var_activation
        else:
            for neuron in network.output_neurons: neuron.activation = self.output_activation

        for neuron in network.critic_neurons: neuron.activation = self.critic_activation

        network.optimizer = self.optimizer

    def act(self, network, state):
        output = network.forward_pass(state)
        action_desc = {}
        action_desc['output'] = output

        if self.mu_var:
            split = output.shape[0] // 2
            mu, var = output[:split], output[split:] / 6
            sigma = np.sqrt(var)
            action = [np.random.normal(m, s) for m, s in zip(mu, sigma)]
            if not self.continuous: action = int(np.array(output) > 0)
            action_desc = {'mu': mu, 'var': var, 'sigma': sigma}
        elif not self.continuous: 
            # print(output)
            softmax = Softmax()(output)
            action = self.get_class_output(softmax)
        action_desc['action'] = action

        if not self.parametrization is None:
            action = self.parametrization(*action)

        next_state, reward, done, _, _ = self.env.step(action)
        
        return next_state, reward, done, action_desc

cartpole = gym.make('CartPole-v1', render_mode='rgb_array')
pendulum = gym.make('Pendulum-v1', render_mode='rgb_array')
mountaincar = gym.make('MountainCar-v0', render_mode='rgb_array')
lunarlander = gym.make('LunarLander-v2', render_mode='rgb_array')
mountaincar_continuous = gym.make('MountainCarContinuous-v0', render_mode='rgb_array')
bipedalwalker = gym.make('BipedalWalker-v3', render_mode='rgb_array')
ant = gym.make('Ant-v4', render_mode='rgb_array')

CARTPOLE = Environment(cartpole, continuous=False, mu_var=False, num_actions=2)
PENDULUM = Environment(pendulum, continuous=True, mu_var=True)
LUNARLANDER = Environment(lunarlander, continuous=False, mu_var=False, num_actions=4)
MOUNTAINCAR = Environment(mountaincar, continuous=False, mu_var=False, num_actions=3)
MOUNTAINCAR_CONTINUOUS = Environment(mountaincar_continuous, continuous=True, mu_var=True)
BIPEDALWALKER = Environment(bipedalwalker, continuous=True, mu_var=True)
ANT = Environment(ant, continuous=True, mu_var=True)

'''
Make environment-specific changes here
'''


def tilt_minimizer(o0, o1, a, o3): return np.abs(a)
CARTPOLE.minimizer = tilt_minimizer

def lander_minimizer(x, y, vx, vy, a, va, l, r):
    return np.abs(vx) + np.abs(vy) + np.abs(a) + np.abs(va)
LUNARLANDER.minimizer = lander_minimizer


def bipedalwalker_parametrization(t1, t2):
    h1, k1 = t1, t1
    h2, k2 = t2, t2
    return np.array([h1, k1, h2, k2])

PARAMETRIC_BIPEDALWALKER = Environment(bipedalwalker, continuous=True, mu_var=True)
PARAMETRIC_BIPEDALWALKER.mu_activation = Tanh()
PARAMETRIC_BIPEDALWALKER.parametrization = bipedalwalker_parametrization
PARAMETRIC_BIPEDALWALKER.action_space = 4
