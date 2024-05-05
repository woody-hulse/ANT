'''
Classes of gym environments and their properties
'''

import sys
import os

import gym
import numpy as np

from core.activations import *
from core.optimizers import *
from core.losses import *
from custom_env.env import *

sys.path.append(os.path.abspath('../CARL/carl/envs/gymnasium/'))

from classic_control.carl_cartpole import CARLCartPole
from classic_control.carl_acrobot import CARLAcrobot

class Environment():
    def __init__(self, env, continuous, mu_var=False, num_observations=None, num_actions=1, minimizer=None, old=False, name=''):
        self.name = name
        self.env = env
        if num_observations: 
            self.observation_space = num_observations
            self.obs = True
        else: 
            self.observation_space = env.observation_space.shape[0]
            self.obs = False
        if continuous:
            if mu_var: self.action_space = env.action_space.shape[0] * 2
            else: self.action_space = env.action_space.shape[0]
        else:
            self.action_space = num_actions
        self.continuous = continuous
        self.mu_var = mu_var
        self.minimizer = minimizer

        # defaults
        self.neuron_activation = Linear()
        if continuous: self.output_activation = Sigmoid()
        else: self.output_activation = Linear()
        self.mu_activation = Tanh()
        self.var_activation = Sigmoid()

        # self.optimizer = ADAM(alpha=1e-6, beta1=0.9, beta2=0.9)
        self.optimizer = RMSProp(alpha=1e-5, beta=0.99, reg=0)

        self.parametrization = None
        self.gamma = 0.99
        self.epsilon = 1
        self.epsilon_decay = 0.99

        self.old = old

    def discrete_action(self, logits):
        probs = Softmax()(logits)
        argmax = np.argmax(probs)
        p = self.epsilon * probs + np.eye(len(logits))[argmax] * (1 - self.epsilon)
        return np.random.choice(len(logits), p=p)

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
        for neuron in network.neurons: neuron.activation = self.neuron_activation
        if self.mu_var:
            for neuron in network.mu_neurons: neuron.activation = self.mu_activation
            for neuron in network.var_neurons: neuron.activation = self.var_activation
        else:
            for neuron in network.output_neurons: neuron.activation = self.output_activation

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
    
    def step(self, action):
        if self.old: state, reward, done, _ = self.env.step(action)
        else: state, reward, done, _, _ = self.env.step(action)
        if self.obs: state = state['obs']
        return state, reward, done
    
    def reset(self):
        if self.old: state = self.env.reset()
        else: state = self.env.reset()[0]
        if self.obs: state = state['obs']
        return state

cartpole = gym.make('CartPole-v1', render_mode='rgb_array')
carl_cartpole = CARLCartPole()
pendulum = gym.make('Pendulum-v1', render_mode='rgb_array')
mountaincar = gym.make('MountainCar-v0', render_mode='rgb_array')
lunarlander = gym.make('LunarLander-v2', render_mode='rgb_array')
acrobot = gym.make('Acrobot-v1', render_mode='rgb_array')
carl_acrobot = CARLAcrobot()

mountaincar_continuous = gym.make('MountainCarContinuous-v0', render_mode='rgb_array')
bipedalwalker = gym.make('BipedalWalker-v3', render_mode='rgb_array')
ant = gym.make('Ant-v4', render_mode='rgb_array')
watermelon = Watermelon()

CARTPOLE = Environment(cartpole, continuous=False, mu_var=False, num_actions=2, name='cartpole')
CARL_CARTPOLE = Environment(cartpole, continuous=False, mu_var=False, num_actions=2, name='carl_cartpole')
PENDULUM = Environment(pendulum, continuous=True, mu_var=True, name='pendulum')
LUNARLANDER = Environment(lunarlander, continuous=False, mu_var=False, num_actions=4, name='lunarlander')
MOUNTAINCAR = Environment(mountaincar, continuous=False, mu_var=False, num_actions=3, name='mountaincar')
ACROBOT = Environment(acrobot, continuous=False, mu_var=False, num_actions=3, name='acrobot')
CARL_ACROBOT = Environment(carl_acrobot, continuous=False, mu_var=False, num_actions=3, num_observations=6, name='carl_acrobot')
WATERMELON = Environment(watermelon, continuous=False, mu_var=False, num_actions=6, old=True, name='watermelon')

MOUNTAINCAR_CONTINUOUS = Environment(mountaincar_continuous, continuous=True, mu_var=True, name='mountaincar_continuous')
BIPEDALWALKER = Environment(bipedalwalker, continuous=True, mu_var=True, name='bipedalwalker')
ANTENV = Environment(ant, continuous=True, mu_var=True, name='ant')

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
