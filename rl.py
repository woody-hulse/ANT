import gym
# import tensorflow as tf
import copy
import time
import cv2
from PIL import Image
import IPython as ip
import seaborn as sns
sns.set_style(style='whitegrid',rc={'font.family': 'serif','font.serif':'Times'})

from utils import *
from continuous_network import *



def visualize_episode(network, env, name):
    debug_print(['Visualizing episode'])

    state = env.reset()[0]
    image_frames = []
    max_steps_per_episode = env.spec.max_episode_steps
    
    for step in tqdm(range(500)):
        action = int(np.ceil(network.forward_pass(state)[0] - 0.5))
        # action = network.forward_pass(state)
        
        state, reward, done, _, _ = env.step(action)
        
        image_array = env.render()
        image_frame = Image.fromarray(image_array)
        image_frames.append(image_frame)
        
        if done:
            break

    image_frames[0].save(name + '.gif', 
                        save_all = True, 
                        duration = 20,
                        loop = 0,
                        append_images = image_frames[1:])

    # debug_print(['Creating gif'])
    # ip.display.Image(open('rl_results/' + name + '.gif', 'rb').read())


def watch_rl_episode(network, env):
    state = env.reset()[0]
    done = False
    step = 0

    while not done and step < 1000:
        image = env.render()

        cv2.imshow(f't={step}', image)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            cv2.destroyAllWindows()

        action = network.forward_pass(state)
        state, reward, done, _, _ = env.step(action)
        # print(reward)
        step += 1
        if done:
            print(f'Finished after {step} steps.')
            break
    
    cv2.destroyAllWindows()
    env.close()



def tensorflow_train(layers, env, episodes=1000, time=500):
    import tensorflow as tf

    model = tf.keras.Sequential()
    for layer in layers[:-1]:
        model.add(tf.keras.layers.Dense(layer, activation='relu'))
    model.add(tf.keras.layers.Dense(layer, activation='linear'))
    model.compile(loss='mse', optimizer='rmsprop')
    optimizer = tf.keras.optimizers.RMSprop(learning_rate=1e-5)

    episode_rewards = []

    for episode in range(episodes):
        state = env.reset()[0]
        total_reward = 0

        pbar = tqdm(range(time))
        for t in pbar:
            with tf.GradientTape() as tape:
                action = int(np.ceil(model(tf.constant([state]))[0][0] - 0.5))
                next_state, reward, done, _, _ = env.step(action)

                y      = np.array([np.sum(state) + state[2] * 10])
                y_hat  = np.array([0])

                total_reward += reward

                loss = tf.keras.losses.mean_squared_error(y, y_hat)
            
            grads = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(grads, model.trainable_variables))

            if done:
                break
        
        episode_rewards.append(total_reward)

    fig, axes = plt.subplots(1, 1, sharex=True)
    axes[0].set_title('ANN rewards over episodes')
    axes[0].plot(episode_rewards, color='blue', label='Total Reward')
    plt.legend()
    plt.xlabel('Episode')
    plt.show()



def train(network, env, episodes=1000, time=500, render=False, plot=True, gif=False):
    gamma = 0.99  # Discount factor for future rewards

    loss_fn = MSE()

    episode_rewards = []

    NETWORK_METRICS = ['energy', 'grad_energy', 'weight_energy', 'prop_input', 'prop_grad']
    NETWORK_METRICS_LABELS = ['Energy', 'Gradient Energy', 'Weight Magnitude', 'Network Utilization', 'Gradient Utilization']
    num_metrics = len(NETWORK_METRICS)
    metrics = {}
    for metric in NETWORK_METRICS:
        metrics[metric] = ([], [])
    
    best_t = 0
    for episode in range(episodes):
        state = env.reset()[0]

        image_frames = []
        G = 0
        total_reward = 0
        p_reward = 0

        pbar = tqdm(range(time))
        for t in pbar:
            if render:
                image_array = env.render()
                image_frame = Image.fromarray(image_array)
                image_frames.append(image_frame)

            '''
            For pole cart: 
                action = int(np.ceil(network.forward_pass(state)[0] - 0.5))
                y      = np.array([state[2] + state[]])
                y_hat  = np.array([0])
            otherwise:
                action = network.forward_pass(state)
                y      = reward
                y_hat  = G
            '''
            output = network.forward_pass(state) / 2
            action = round(output[0] + 0.5)
            next_state, reward, done, _, _ = env.step(action)

            G = reward + gamma * G

            y      = np.array([np.sum(state) + state[2] * 10])
            y_hat  = np.array([0])
            
            network.backward_pass(loss_fn, y, y_hat)
            p_reward = reward

            max_chars = 160
            total_reward += reward
            pbar_string = f'Episode {episode : 05}: reward={total_reward : 9.3f}; action={np.sum(action) : 7.3f}; ' + \
                network.get_metrics_string(metrics=['loss', 'energy', 'prop_input', 'prop_grad'])
            
            if len(pbar_string) > max_chars: pbar_string = pbar_string[:max_chars - 3] + '...'

            pbar.set_description(pbar_string)

            state = next_state

            if gif and episode % 10 == 0:
                plot_graph(network, title=f't{t}', save=True, save_directory='graph_images/')

            if t > best_t:
                best_t = t
                if (done or t == time - 1) and render:
                    image_frames[0].save(f'rl_results/episode{episode}.gif', 
                            save_all = True, 
                            duration = 20,
                            loop = 0,
                            append_images = image_frames[1:])

            if done: break
        
        if gif and episode % 10 == 0:
            convert_files_to_gif(directory='graph_images/', name=f'graph_results/network_weights_episode{episode}.gif')
                        
        for metric in metrics.keys():
            metrics[metric][0].append(network.metrics[metric][-1])
            metrics[metric][1].append(np.mean(network.metrics[metric][-t:]))
        episode_rewards.append(total_reward)

    if plot:
        fig, axes = plt.subplots(num_metrics + 1, 1, gridspec_kw={'height_ratios': [1] + [0.1 for _ in range(num_metrics)]}, sharex=True)
        axes[0].set_title('Rewards over episodes')
        axes[0].plot(episode_rewards, color='blue', label='Total Reward')
        for i, ((metric, values), label) in enumerate(zip(metrics.items(), NETWORK_METRICS_LABELS)):
            axes[i + 1].plot(metrics[metric][0], label=label)
            axes[i + 1].plot(metrics[metric][1], label='Mean ' + label)
            axes[i + 1].legend()
        plt.legend()
        plt.xlabel('Episode')
        plt.show()


def main():
    # run internal mechansim until convergence

    # '*' indicates that model succeeds at this task
    # * 'CartPole-v1'
    #   'Pendulum-v1'
    #   'MountainCarContinuous-v0'
    #   'BipedalWalker-v3'
    #   'Ant-v2'
    env = gym.make('CartPole-v1', render_mode='rgb_array')

    try: num_output_neurons = env.action_space.shape[0]
    except: num_output_neurons = 1

    network = ContinuousNetwork(
    num_neurons         = 128,
    edge_probability    = 0.9,
    num_input_neurons   = env.observation_space.shape[0],
    num_output_neurons  = num_output_neurons
    )
    # plot_graph(network)
    # visualize_network(network, show_weight=True)

    for neuron in network.output_neurons:
        neuron.activation = Tanh()

    network.print_info()
    rmsprop = RMSProp(alpha=1e-5, beta=0.99, reg=2e-5)
    network.set_optimizer(rmsprop)

    train(network, env, episodes=200, render=False, plot=True, gif=False)
    # watch_rl_episode(network, env)
    visualize_episode(network, env, name='prelim_rl_results')


if __name__ == '__main__':
    main()