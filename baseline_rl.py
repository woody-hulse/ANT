import tensorflow as tf
from continuous_rl import *


class SimpleModel(tf.keras.Model):
    def __init__(self, hidden_size, num_hidden_layers, action_size):
        super().__init__()
        self.hidden_layers = [tf.keras.layers.Dense(hidden_size, activation='sigmoid') for _ in range(num_hidden_layers)]
        self.action_layer = tf.keras.layers.Dense(action_size, activation='softmax')
        self.action_size = action_size

        self.epsilon = 1e-6

    def __call__(self, x):
        for hidden_layer in self.hidden_layers:
            x = hidden_layer(x)
        x = self.action_layer(x)

        return x
    

    def train(self, env, episodes=1000, time=500, plot=True):
        episode_rewards = []

        for episode in range(episodes):
            state = env.reset()[0]
            total_reward = 0

            pbar = tqdm(range(time))
            for t in pbar:
                with tf.GradientTape() as tape:
                    output = self(np.array([state]))
                    action = tf.random.categorical(output, 1).numpy()[0][0]
                    next_state, reward, done, _, _ = env.step(action)

                    score = np.square(np.sum(state) + state[2] * 10)
                    # print(output, score)
                    loss = -tf.math.log(output[0][action]) * score

                gradients = tape.gradient(loss, self.trainable_weights)
                self.optimizer.apply_gradients(zip(gradients, self.trainable_weights))
                state = next_state
                total_reward += reward
                pbar_string = f'Episode {episode : 05}: total_reward={total_reward : 9.3f}; action={action : 9.3f} '
                pbar.set_description(pbar_string)

                if done:
                    break
            
            episode_rewards.append(total_reward)
            
        if plot:
            # fig, axes = plt.subplots(1, 1, gridspec_kw={'height_ratios': [1]}, sharex=True)
            plt.title('ANN Rewards over episodes')
            plt.plot(episode_rewards, color='blue', label='Total Reward')
            plt.legend()
            plt.xlabel('Episode')
            plt.show()  


class ActorCriticModel(tf.keras.Model):
    def __init__(self, hidden_size, num_hidden_layers, action_size):
        super().__init__()
        self.hidden_layers = [tf.keras.layers.Dense(hidden_size, activation='selu') for _ in range(num_hidden_layers)]
        self.mu_layer = tf.keras.layers.Dense(action_size, activation='tanh')
        self.var_layer = tf.keras.layers.Dense(action_size, activation='sigmoid')
        # self.critic = tf.keras.layers.Dense(1, activation='linear')
        self.action_size = action_size

        self.critic = tf.keras.Sequential()
        for _ in range(num_hidden_layers):
            self.critic.add(tf.keras.layers.Dense(hidden_size, activation='relu'))
        self.critic.add(tf.keras.layers.Dense(1, activation='linear'))
        self.critic.optimizer = tf.keras.optimizers.RMSprop()

        self.epsilon = 1e-6

    def __call__(self, x):
        for hidden_layer in self.hidden_layers:
            x = hidden_layer(x)
        mu = self.mu_layer(x)
        var = self.var_layer(x)
        critic = None # self.critic(x)

        return mu, var, critic
    
    def action(self, x):
        mu, var, critic = self(x)
        sigma = tf.sqrt(var)
        action = tf.random.normal(mu, sigma)

        return action, critic
    
    def calculate_logprobs(self, mu, sigma, action):
        a = -tf.square(mu - action) / (2 * tf.square(sigma) + self.epsilon)
        b = -tf.math.log(tf.sqrt(2 * np.pi * tf.square(sigma)))
        return a + b

    def pdf(self, mu, sigma, action):
        return (1 / (tf.sqrt(2 * np.pi) * sigma)) * tf.exp(-0.5 * ((action - mu) / sigma)**2)
    
    def normal_dist_prob(self, mu, action):
        self.reduction = 0.01
        cov_inv = 1/float(self.reduction)
        y = tf.reduce_sum(tf.square((action - mu))*tf.ones(self.action_size)*cov_inv,1)
        Z = (2 * np.pi)**(0.5*4) * (self.reduction**self.action_size)**(0.5)
        pdf = tf.exp(-0.5 * y)/Z
        return pdf   
    
    def train(self, env, episodes=1000, time=500, plot=True):
        gamma = 0.99
        beta = 1e-4

        episode_rewards = []
        var_convergences = []
        
        for episode in range(episodes):
            state = env.reset()[0]
            total_reward = 0

            var_convergence = []
            pred = []
            true = []

            pbar = tqdm(range(time))
            for t in pbar:
                with tf.GradientTape(persistent=True) as tape:
                    mu, var, _ = self(np.array([state]))
                    critic = self.critic(np.array([state]))
                    sigma = tf.sqrt(var)
                    action = []
                    for m, s in zip(mu, sigma):
                        action.append(tf.random.normal([1], m, s))
                    action = tf.concat(action, axis=0)
                    next_state, reward, done, _, _ = env.step(action)
                    pred_reward = self.critic(np.array([next_state]))

                    actor_loss = -tf.math.log(self.normal_dist_prob(mu, action))
                    critic_loss = -critic

                    # pred_reward = self.critic(np.array([next_state]))
                    # pred_reward = critic

                    # td = reward + gamma * pred_reward
                    # critic_loss = tf.keras.losses.MeanSquaredError()(td, reward)
                    # critic_loss = tf.keras.losses.MeanSquaredError()(reward, pred_reward)
                    # pred.append(pred_reward[0])
                    # true.append(reward)

                    # advantage = reward - pred_reward
                    # log_probs = reward * self.calculate_logprobs(mu, sigma, action)
                    # loss = -tf.math.reduce_mean(log_probs)
                    # actor_loss = -td * tf.math.log(self.pdf(mu, sigma, action))
                    # print(td, critic)
                    # loss = -actor_loss + critic_loss

                gradients = tape.gradient(actor_loss, self.trainable_weights)
                self.optimizer.apply_gradients(zip(gradients, self.trainable_weights))

                critic_gradients = tape.gradient(critic_loss, self.critic.trainable_weights)
                self.critic.optimizer.apply_gradients(zip(critic_gradients, self.critic.trainable_weights))

                state = next_state
                total_reward += reward
                mean = np.mean(mu.numpy())
                convergence = np.mean(np.abs(var.numpy()))
                pbar_string = f'Episode {episode : 05}: total_reward={total_reward : 9.3f}; mean={mean : 9.3f}; var={convergence : 9.3f}; loss={np.mean(loss.numpy()) : 9.3f} '
                pbar.set_description(pbar_string)

                var_convergence.append(tf.reduce_mean(var).numpy())

                if done:
                    break
            
            var_convergences.append(np.mean(var_convergence))
            episode_rewards.append(total_reward)

            # plt.plot(pred)
            # plt.plot(true)
            # plt.show()
            
        if plot:
            fig, axes = plt.subplots(2, 1, gridspec_kw={'height_ratios': [1, 0.1]}, sharex=True)
            axes[0].set_title('ANN Rewards over episodes')
            axes[0].plot(episode_rewards, color='blue', label='Total Reward')
            axes[1].plot(var_convergences, label='Action Convergence')
            plt.legend()
            plt.xlabel('Episode')
            plt.show()  


def actor_critic():
    env = gym.make('MountainCarContinuous-v0', render_mode='rgb_array')

    try: action_size = env.action_space.shape[0]
    except: action_size = 1

    network = ActorCriticModel(200, 2, action_size=action_size)
    network.optimizer = tf.keras.optimizers.RMSprop()

    network.train(env, episodes=1000, time=500, plot=True)

def simple():
    env = gym.make('CartPole-v1', render_mode='rgb_array')

    network = SimpleModel(50, 5, action_size=2)
    network.optimizer = tf.keras.optimizers.RMSprop()

    network.train(env, episodes=1000, time=500, plot=True)

def main():
    actor_critic()



if __name__ == '__main__':
    main()