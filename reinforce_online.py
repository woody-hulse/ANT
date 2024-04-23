import numpy as np
import gym
import tensorflow as tf
from tqdm import tqdm

import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style(style='whitegrid')

class PolicyNetwork(tf.keras.Model):
    def __init__(self, num_actions, hidden_size=128, num_hidden=1):
        super().__init__()

        self.network = tf.keras.Sequential([tf.keras.layers.Dense(hidden_size, activation='relu') for _ in range(num_hidden)] \
                                                + [tf.keras.layers.Dense(num_actions, activation='softmax')])

    def call(self, x):
        return self.network(x)

learning_rate = 0.0002
gamma = 0.99  # Discount factor
alpha = tf.constant(1, dtype=tf.float32)
episodes = 1000
time = 512

env = gym.make('LunarLander-v2')
num_actions = env.action_space.n
num_inputs = env.observation_space.shape[0]

model = PolicyNetwork(num_actions, hidden_size=4, num_hidden=1)
optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

def get_discounted_rewards(rewards, gamma):
    discounted_rewards = np.zeros_like(rewards)
    G = 0
    for t in reversed(range(len(rewards))):
        G = G * gamma + rewards[t]
        discounted_rewards[t] = G
    return discounted_rewards

def compute_discounted_gradients(saved_gradients, rewards):
    discounted_gradients = [tf.zeros_like(grad) for grad in saved_gradients[0]]
    T = len(rewards) - 1
    for t in range(len(saved_gradients)):
        for i in range(len(saved_gradients[t])):
            discounted_gradients[i] += saved_gradients[t][i] * rewards[t] * tf.pow(gamma, T - t)
    return discounted_gradients

total_rewards = []
for episode in range(episodes):
    state = env.reset()[0]
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
            # print(action_probs)
            action = np.random.choice(num_actions, p=np.squeeze(action_probs))
            log_prob = tf.math.log(action_probs[0, action])
        
            next_state, reward, done, _, _ = env.step(action)
            # reward_tensor = tf.convert_to_tensor(reward, dtype=tf.float32)
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

    '''
    returns = get_discounted_rewards(rewards, gamma)
    # returns = (returns - np.mean(returns)) / (np.std(returns) + 1e-9)

    actions_tensor = tf.convert_to_tensor(actions, dtype=tf.int32)
    returns_tensor = tf.convert_to_tensor(returns, dtype=tf.float32)

    # Compute loss and perform a gradient ascent step
    with tf.GradientTape() as tape:
        action_probs = model(tf.concat(states, axis=0))
        action_probs_log = tf.math.log(tf.gather(action_probs, actions_tensor, batch_dims=1, axis=1))
        loss = -tf.reduce_sum(action_probs_log * returns_tensor)

    gradients = tape.gradient(loss, model.trainable_variables)
    discounted_gradients = compute_discounted_gradients(saved_gradients, rewards)

    print('grads:', gradients)
    print('discounted grads:', discounted_gradients)

    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    '''

    total_rewards.append(sum(rewards))

plt.title('Total reward (baseline newtork)')
plt.plot(total_rewards)
plt.show()