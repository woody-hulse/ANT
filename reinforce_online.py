import numpy as np
import gym
import tensorflow as tf
from tqdm import tqdm

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

def train(env, size, episodes = 1000, learning_rate=0.001, plot=True):
    learning_rate = 0.001
    gamma = 0.99  # Discount factor
    time = 200

    num_actions = env.action_space
    num_inputs = env.observation_space

    model = PolicyNetwork(num_actions, hidden_size=size // 2, num_hidden=2)
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

    def compute_discounted_gradients(saved_gradients, rewards):
        discounted_gradients = [tf.zeros_like(grad) for grad in saved_gradients[0]]
        T = len(rewards) - 1
        for t in range(len(saved_gradients)):
            for i in range(len(saved_gradients[t])):
                discounted_gradients[i] += saved_gradients[t][i] * rewards[t] * tf.pow(gamma, T - t)
        return discounted_gradients

    total_rewards = []
    for episode in range(episodes):
        state = env.reset()
        done = False
        action_probs_log = []
        rewards = []
        states = []
        actions = []

        saved_gradients = []
        pbar = tqdm(range(time))
        for t in pbar:
            with tf.GradientTape() as tape:
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

            gradients = tape.gradient(loss, model.trainable_variables)
            saved_gradients.append(gradients)
            
            discounted_gradients = compute_discounted_gradients(saved_gradients, rewards)
            optimizer.apply_gradients(zip(discounted_gradients, model.trainable_variables))

            state = next_state

            pbar_string = f'Episode {episode : 05}: reward={sum(rewards) : 9.3f}'
            pbar.set_description(pbar_string)

            if done: break
        
        total_rewards.append(sum(rewards))

    if plot:
        plt.title('Total reward (baseline newtork)')
        plt.plot(total_rewards)
        plt.show()

    return total_rewards


def main():
    train(
        env=ACROBOT
    )


if __name__ == '__main__':
    main()