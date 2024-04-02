'''
Classes of gym environments and their properties
'''

import gym
import numpy as np

from core.activations import *
from core.optimizers import *
from core.losses import *

class Environment():
    def __init__(self, env, continuous, mu_var=False):

        self.env = env
        self.observation_space = env.observation_space.shape[0]
        if mu_var: self.action_space = env.action_space.shape[0] * 2
        else: self.action_space = env.action_space.shape[0]
        self.continuous = continuous
        self.mu_var = mu_var

        # defaults
        self.neuron_activation = LeakyReLU()
        self.output_activation = Sigmoid()
        self.mu_activation = Tanh()
        self.var_activation = Sigmoid()
        self.critic_activation = Linear()

        self.optimizer = RMSProp(alpha=4e-4, beta=0.99)

        self.parametrization = None

    def configure_newtork(self, network):
        if self.mu_var:
            for neuron in network.mu_neurons:
                neuron.activation = self.mu_activation
            for neuron in network.var_neurons:
                neuron.activation = self.var_activation
        else:
            for neuron in network.output_neurons:
                neuron.activation = self.output_activation

        for neuron in network.critic_neurons:
            neuron.activation = self.critic_activation

        network.set_optimizer(self.optimizer)

    def act(self, network, state):
        action = network.forward_pass(state)
        action_desc = {}

        if self.mu_var:
            split = action.shape[0] // 2
            mu, var = action[:split], action[split:]
            sigma = np.sqrt(var)
            action = [np.random.normal(m, s) for m, s in zip(mu, sigma)]
            action_desc = {'mu': mu, 'var': var, 'sigma': sigma}
        
        action_desc['action'] = action

        if not self.parametrization is None:
            action = self.parametrization(*action)

        next_state, reward, done, _, _ = self.env.step(action)
        
        return next_state, reward, done, action_desc

pendulum = gym.make('Pendulum-v1', render_mode='rgb_array')
mountaincar_continuous = gym.make('MountainCarContinuous-v0', render_mode='rgb_array')
bipedalwalker = gym.make('BipedalWalker-v3', render_mode='rgb_array')
ant = gym.make('Ant-v4', render_mode='rgb_array')

PENDULUM = Environment(pendulum, continuous=True, mu_var=True)
MOUNTAINCAR_CONTINUOUS = Environment(mountaincar_continuous, continuous=True, mu_var=True)
BIPEDALWALKER = Environment(bipedalwalker, continuous=True, mu_var=True)
ANT = Environment(ant, continuous=True, mu_var=True)

'''
Make environment-specific changes here
'''


def bipedalwalker_parametrization(t1, t2):
    h1, k1 = t1, t1
    h2, k2 = t2, t2
    return np.array([h1, k1, h2, k2])

PARAMETRIC_BIPEDALWALKER = Environment(bipedalwalker, continuous=True, mu_var=True)
PARAMETRIC_BIPEDALWALKER.mu_activation = Tanh()
PARAMETRIC_BIPEDALWALKER.parametrization = bipedalwalker_parametrization
PARAMETRIC_BIPEDALWALKER.action_space = 4
