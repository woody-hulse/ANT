import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from continuous_rl import *


class Actor(tf.keras.Model):
    def __init__(self, env, hidden_size=32, num_hidden=3):
        super().__init__()

        self.dense_layers = tf.keras.Sequential([tf.keras.layers.Dense(hidden_size, activation='sigmoid') for _ in range(num_hidden)] \
                                                + [tf.keras.layers.Dense(env.action_space, activation='linear')])
        self.softmax_layer = tf.keras.activations.Softmax()
        
        self.env = env
    
    def __call__(self, x):
        for layer in self.dense_layers:
            x = layer(x)
        return x

    def train(self, episodes, time=512):
        for episode in range(episodes):
            state = self.env.env.reset()[0]

            total_reward = 0
            saved_log_probs = []
            rewards = []

            pbar = tqdm(range(time))
            for t in pbar:
                logits = self(state)
                probs = self.softmax_layer(logits)
                action = tf.random.categorical(logits, num_samples=1)

                soft = tf.math.reduce_sum(tf.math.exp(logits))
                logprobs = logits[action] - tf.math.log(soft)

                saved_log_probs.append(logprobs)
                state, reward, done, _, _ = self.env.env.step(action)
                rewards.append(reward)
                total_reward += reward
        
            # REINFORCE Update
            policy_loss = []
            returns = []
            G = 0
            for r in reversed(rewards):
                G = r + self.env.gamma * G
                returns.insert(0, G)
            '''
            r = mg.tensor(returns)
            returns = (r - np.mean(r)) / (np.std(r) + 1e-9)  # Normalize returns

            for log_prob, Gt in zip(saved_log_probs, returns):
                policy_loss.append(-log_prob * Gt)  # Negative for gradient ascent
        
            returns.backward()
            grad = np.array([np.mean(r.grad, axis=0) for _ in range(env.action_space)])

            network.backward_(-grad, np.array([0]))

            # print(grad.shape, grad)
            '''
            print(f'Episode {episode} Total Reward: {total_reward}')
        
        


class ActorCritic(tf.keras.Model):
    def __init__(self, env, hidden_size=32, num_hidden=3):
        super().__init__()

        self.env = env
        self.observation_size = env.observation_space.shape[0]
        self.action_size = env.action_space.shape[0]

        input = tf.keras.layers.Input((self.observation_size,))
        hidden_1 = tf.keras.layers.Dense(hidden_size, activation='sigmoid')(input)
        hidden_2 = tf.keras.layers.Dense(hidden_size, activation='sigmoid')(hidden_1)
        mu = tf.keras.layers.Dense(self.action_size, activation='tanh')(hidden_2)
        sigma = tf.keras.layers.Dense(self.action_size, activation='sigmoid')(hidden_2)
        self.actor = tf.keras.Model(inputs=input, outputs=[mu, sigma])
        self.actor.optimizer = tf.keras.optimizers.Adam(learning_rate=0.003)

        self.critic = tf.keras.Sequential()
        for _ in range(num_hidden):
            self.critic.add(tf.keras.layers.Dense(hidden_size, activation='sigmoid'))
        self.critic.add(tf.keras.layers.Dense(1, activation='linear'))
        self.critic.optimizer = tf.keras.optimizers.Adam(learning_rate=0.003)

    def act(self, state):
        mu, sigma = self.actor(state)
        sigma *= 0.3
        norm_dist = tf.compat.v1.distributions.Normal(mu, sigma)
        action = tf.squeeze(norm_dist.sample(1), axis=0)
        action = tf.clip_by_value(action, 
                                  self.env.action_space.low[0], 
                                  self.env.action_space.high[0])

        return action, norm_dist, sigma

    
    def get_state_scale(self):
        state_space_samples = np.array(
            [self.env.observation_space.sample() for _ in range(10000)])
        self.scaler = StandardScaler()
        self.scaler.fit(state_space_samples)

    def scale_state(self, state, many=False):
        if many: scaled = self.scaler.transform(state)
        else: scaled = self.scaler.transform([state])
        return tf.Variable(scaled)
    

    def train(self, episodes=1000, time=500, plot=True):
        self.get_state_scale()

        gamma = 0.99
        epsilon = 4e-4

        total_rewards = []
        for episode in range(episodes):
            state = self.env.reset()[0]
            total_reward = 0
            total_action = 0
            
            pbar = tqdm(range(time))
            for t in pbar:
                with tf.GradientTape(persistent=True) as tape:
                    action, norm_dist, sigma = self.act(self.scale_state(state))
                    value = self.critic(self.scale_state(state))
                    next_state, reward, done, _, _ = self.env.step(np.squeeze(action, axis=0))
                    total_reward += reward

                    value_next = self.critic(self.scale_state(state))
                    target = reward + gamma * np.squeeze(value_next)
                    td_error = target - np.squeeze(value)

                    loss_actor = -tf.math.log(norm_dist.prob(action) + 1e-8) * td_error + 1 / tf.math.sqrt(2 * np.pi * sigma) * epsilon
                    loss_critic = tf.keras.losses.MeanSquaredError()(value, np.array([target]))

                grads_actor = tape.gradient(loss_actor, self.actor.trainable_variables)
                grads_critic = tape.gradient(loss_critic, self.critic.trainable_variables)

                grads_actor, _ = tf.clip_by_global_norm(grads_actor, 5.0)
                grads_critic, _ = tf.clip_by_global_norm(grads_critic, 5.0)

                self.actor.optimizer.apply_gradients(zip(grads_actor, self.actor.trainable_variables))
                self.critic.optimizer.apply_gradients(zip(grads_critic, self.critic.trainable_variables))

                state = next_state
                mean = np.mean(action.numpy())
                total_action += np.mean(np.abs(action.numpy()))
                pbar_string = f'Episode {episode : 05}: total_reward={total_reward : 9.3f}; action={mean : 9.3f};' + \
                   f' avg_action_mag={total_action / (t + 1) : 9.3f}; sigma={np.mean(sigma.numpy()) : 9.3f} '
                pbar.set_description(pbar_string)

                if done:
                    break
            
            total_rewards.append(total_reward)

            # if episode % 10 == 0:
            #     self.watch_episode(name=str(episode))
    
        if plot:
            plt.plot(total_rewards)
            plt.show()

    
    def watch_episode(self, name='baseline_episode'):
        debug_print(['Visualizing episode'])

        state = self.env.reset()[0]
        image_frames = []
        
        for step in tqdm(range(500)):
            action, _, _ = self.act(np.array([state]))
            
            state, _, done, _, _ = self.env.step(action)
            
            image_array = self.env.render()
            image_frame = Image.fromarray(image_array)
            image_frames.append(image_frame)
            
            if done:
                break

        image_frames[0].save(name + '.gif', 
                            save_all = True, 
                            duration = 20,
                            loop = 0,
                            append_images = image_frames[1:])


import random
import numpy as np
import tensorflow as tf
from collections import deque
import matplotlib.pyplot as plt
from tqdm import tqdm

class Trainer:
    def __init__(self, env):
        maxlen = 100000
        self.env = env
        # self.replay_buffer = deque(maxlen=maxlen)
        self.batch_size = 128
        self.replay_buffer = {
            'states' : np.empty((self.batch_size, self.env.observation_space.shape[0],)),
            'next_states' : np.empty((self.batch_size, self.env.observation_space.shape[0],)),
            'actions' : np.empty((self.batch_size, self.env.action_space.shape[0],)),
            'rewards' : np.empty((self.batch_size,)),
            'done' : np.empty((self.batch_size,))
            }
        
        self.gamma = 0.99
        self.epsilon = 4e-4
        
    def replay(self, model):
        state = self.replay_buffer['states']
        next_state = self.replay_buffer['next_states']
        action = self.replay_buffer['actions']
        reward = self.replay_buffer['rewards']
        done = self.replay_buffer['done']
        with tf.GradientTape(persistent=True) as tape:
            scaled_state = model.scale_state(state, many=True)
            tape.watch(scaled_state)
            recalc_action, norm_dist, _ = model.act(scaled_state)

            value = model.critic(model.scale_state(state, many=True))
            value_next = model.critic(model.scale_state(next_state, many=True))
            target = reward + self.gamma * np.squeeze(value_next) # * (1 - int(done))
            td_error = target - np.squeeze(value)

            loss_actor = -tf.math.log(norm_dist.prob(action) + 1e-8) * td_error
            loss_critic = tf.keras.losses.MeanSquaredError()(value, np.array([target]))

        grads_actor = tape.gradient(loss_actor, model.actor.trainable_variables)
        grads_critic = tape.gradient(loss_critic, model.critic.trainable_variables)

        # grads_actor, _ = tf.clip_by_global_norm(grads_actor, 5.0)
        # grads_critic, _ = tf.clip_by_global_norm(grads_critic, 5.0)

        model.actor.optimizer.apply_gradients(zip(grads_actor, model.actor.trainable_variables))
        model.critic.optimizer.apply_gradients(zip(grads_critic, model.critic.trainable_variables)) 

    def train(self, model, episodes=1000, time=500, plot=True):
        model.get_state_scale()

        total_rewards = []
        for episode in range(episodes):
            state = self.env.reset()[0]
            total_reward = 0
            
            pbar = tqdm(range(time))
            for t in pbar:
                action, norm_dist, sigma = model.act(model.scale_state(state))
                next_state, reward, done, _, _ = self.env.step(np.squeeze(action, axis=0))
                total_reward += reward

                self.replay_buffer['states'][t % self.batch_size] = state
                self.replay_buffer['next_states'][t % self.batch_size] = next_state
                self.replay_buffer['rewards'][t % self.batch_size] = reward
                self.replay_buffer['done'][t % self.batch_size] = done
                self.replay_buffer['actions'][t % self.batch_size] = action

                if (t + 1) % self.batch_size == 0:
                    self.replay(model)

                state = next_state
                pbar.set_description(f'Episode {episode : 05}: total_reward={total_reward : 9.3f}; mu={np.mean(action) : 6.3f}; sigma={np.mean(sigma) : 6.3f}')

                if done:
                    break
            
            total_rewards.append(total_reward)
    
        if plot:
            plt.plot(total_rewards)
            plt.xlabel('Episode')
            plt.ylabel('Total Reward')
            plt.show()

# Note: Ensure that `self.env`, `self.actor`, and `self.critic` are properly initialized before calling `train`.



def train():
    env = gym.make('Pendulum-v1', render_mode='rgb_array')
    
    model = ActorCritic(env, hidden_size=64, num_hidden=3)
    # model.train(episodes=200, time=512)
    # model.watch_episode()

    trainer = Trainer(env)
    trainer.train(model, episodes=5000, time=512, plot=True)


if __name__ == '__main__':
    train()