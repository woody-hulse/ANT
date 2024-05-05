import numpy as np
import gym
import tensorflow as tf
from tqdm import tqdm
import time

from environments import *

import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style(style='whitegrid')


class PolicyNetwork(tf.keras.Model):
    def __init__(self, num_actions, hidden_size=128, num_hidden=1):
        super().__init__()

        self.network = tf.keras.Sequential([tf.keras.layers.Dense(hidden_size, activation='tanh') for _ in range(num_hidden)] \
                                                + [tf.keras.layers.Dense(num_actions, activation='softmax')])

    def call(self, x):
        return self.network(x)


def get_discounted_rewards(rewards, gamma):
    discounted_rewards = np.zeros_like(rewards)
    G = 0
    for t in reversed(range(len(rewards))):
        G = G * gamma + rewards[t]
        discounted_rewards[t] = G
    return discounted_rewards

    
def train(env, size, episodes=1000, learning_rate=0.001, plot=True):
    learning_rate = 0.001
    gamma = 0.99  # Discount factor
    time_ = 200

    num_actions = env.action_space
    num_inputs = env.observation_space

    model = PolicyNetwork(num_actions, hidden_size=size // 2, num_hidden=2)
    optimizer = tf.keras.optimizers.RMSprop(learning_rate=learning_rate)

    total_rewards = []
    for episode in range(episodes):
        state = env.reset()
        done = False
        action_probs_log = []
        rewards = []
        states = []
        actions = []

        pbar = tqdm(range(time_))
        for t in pbar:
            state = tf.convert_to_tensor([state], dtype=tf.float32)
            action_probs = model(state)
            action = np.random.choice(num_actions, p=np.squeeze(action_probs))
            log_prob = tf.math.log(action_probs[0, action])
            
            next_state, reward, done = env.step(action)
            loss = -tf.reduce_sum(log_prob)

            action_probs_log.append(log_prob)
            rewards.append(reward)
            states.append(state)
            actions.append(action)

            state = next_state

            pbar_string = f'Episode {episode : 05}: reward={sum(rewards) : 9.3f}'
            pbar.set_description(pbar_string)

            if done: break

        
        
        returns = get_discounted_rewards(rewards, gamma)
        # returns = (returns - np.mean(returns)) / (np.std(returns) + 1e-9)

        actions_tensor = tf.convert_to_tensor(actions, dtype=tf.int32)
        returns_tensor = tf.convert_to_tensor(returns, dtype=tf.float32)

        with tf.GradientTape() as tape:
            action_probs = model(tf.concat(states, axis=0))
            action_probs_log = tf.math.log(tf.gather(action_probs, actions_tensor, batch_dims=1, axis=1))
            loss = -tf.reduce_sum(action_probs_log * returns_tensor)

        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        # print(tf.reduce_sum([tf.reduce_sum(grad) for grad in gradients]).numpy())

        total_rewards.append(sum(rewards))

    if plot:
        plt.title('Total reward (baseline newtork)')
        plt.plot(total_rewards)
        plt.show()

    return total_rewards


def main():
    train(
        env=CARTPOLE,
        size=40,
        learning_rate=1e-5
    )


if __name__ == '__main__':
    main()
