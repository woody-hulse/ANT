import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from continuous_rl import *

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
        self.actor.optimizer = tf.keras.optimizers.Adam(learning_rate=0.01)

        self.critic = tf.keras.Sequential()
        for _ in range(num_hidden):
            self.critic.add(tf.keras.layers.Dense(hidden_size, activation='sigmoid'))
        self.critic.add(tf.keras.layers.Dense(1, activation='linear'))
        self.critic.optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)

    def act(self, state):
        mu, sigma = self.actor(state)
        norm_dist = tf.compat.v1.distributions.Normal(mu, sigma)
        action = tf.squeeze(norm_dist.sample(1), axis=0)
        action = tf.clip_by_value(action, 
                                  self.env.action_space.low[0], 
                                  self.env.action_space.high[0])

        return action, norm_dist, sigma / 100

    
    def get_state_scale(self):
        state_space_samples = np.array(
            [self.env.observation_space.sample() for _ in range(10000)])
        self.scaler = StandardScaler()
        self.scaler.fit(state_space_samples)

    def scale_state(self, state):
        scaled = self.scaler.transform([state])
        return scaled
    

    def train(self, episodes=1000, time=500, plot=True):
        self.get_state_scale()

        gamma = 0.99
        epsilon = 1e-4

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

                    loss_actor = -tf.math.log(norm_dist.prob(action) + 1e-5) * td_error + 1 / tf.math.sqrt(2 * np.pi * sigma) * epsilon
                    loss_critic = tf.keras.losses.MeanSquaredError()(value, np.array([target]))

                grads_actor = tape.gradient(loss_actor, self.actor.trainable_variables)
                grads_critic = tape.gradient(loss_critic, self.critic.trainable_variables)

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


def train():
    env = gym.make('Pendulum-v1', render_mode='rgb_array')
    
    model = ActorCritic(env, hidden_size=16, num_hidden=2)
    model.train(episodes=200, time=512)
    model.watch_episode()


if __name__ == '__main__':
    train()