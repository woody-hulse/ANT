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



def visualize_episode(env, network, name):
    debug_print(['Visualizing episode'])

    state = env.reset()[0]
    image_frames = []
    max_steps_per_episode = env.spec.max_episode_steps
    
    for step in tqdm(range(1000)):
        action = network.forward_pass(state)
        
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

    debug_print(['Creating gif'])
    ip.display.Image(open('rl_results' + name + '.gif', 'rb').read())


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

def train(network, env, episodes=1000, time=500, render=False, plot=True, gif=False):
    gamma = 0.99  # Discount factor for future rewards

    loss_fn = MSE()

    episode_rewards = []
    episode_energies = []
    episode_mean_energies = []
    episode_utilizations = []
    episode_mean_utilizations = []
    episode_gradient_utilizations = []
    episode_mean_gradient_utilizations = []
    best_t = 0
    for episode in range(episodes):
        state = env.reset()[0]

        image_frames = []
        G = 0
        total_reward = 0
        p_reward = 0

        for i in range(15):
            network.forward_pass(np.zeros_like(state))

        pbar = tqdm(range(time))
        for t in pbar:
            if render:
                image_array = env.render()
                image_frame = Image.fromarray(image_array)
                image_frames.append(image_frame)

            '''
            For pole cart: 
                action = int(np.ceil(network.forward_pass(state)[0] - 0.5))
                y      = np.array([state[2]])
                y_hat  = np.array([0])
            otherwise:
                action = network.forward_pass(state)
                y      = reward
                y_hat  = G
            '''
            action = int(np.ceil(network.forward_pass(state)[0] - 0.5))
            next_state, reward, done, _, _ = env.step(action)

            G = reward + gamma * G

            y      = np.array([state[2]])
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

            if gif:
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
        
        if gif:
            convert_files_to_gif(directory='graph_images/', name=f'graph_results/network_weights_episode{episode}.gif')
                        

        episode_utilizations.append(network.metrics['prop_input'][-1])
        episode_mean_utilizations.append(np.mean(network.metrics['prop_input']))
        episode_gradient_utilizations.append(network.metrics['prop_grad'][-1])
        episode_mean_gradient_utilizations.append(np.mean(network.metrics['prop_grad']))
        episode_energies.append(network.metrics['energy'][-1])
        episode_mean_energies.append(np.mean(network.metrics['energy']))
        episode_rewards.append(total_reward)

    if plot:
        fig, axes = plt.subplots(4, 1, gridspec_kw={'height_ratios': [1, 0.15, 0.15, 0.15]}, sharex=True)
        axes[0].set_title('Rewards over episodes')
        axes[0].plot(episode_rewards, color='blue', label='Total Reward')
        axes[1].plot(episode_energies, color='orange', label='Energy')
        axes[1].plot(episode_mean_energies, color='green', label='Mean Energy')
        axes[2].plot(episode_utilizations, color='purple', label='Network Utilization')
        axes[2].plot(episode_mean_utilizations, color='red', label='Mean Network Utilization')
        axes[3].plot(episode_gradient_utilizations, color='teal', label='Network Gradient Utilization')
        axes[3].plot(episode_mean_gradient_utilizations, color='violet', label='Mean Network Gradient Utilization')
        axes[0].legend()
        axes[1].legend()
        axes[2].legend()
        axes[3].legend()
        plt.legend()
        plt.xlabel('Episode')
        plt.show()


def main():
    # run internal mechansim until convergence

    # '*' indicates that model succeeds at this task
    #   'CartPole-v1'
    #   'Pendulum-v1'
    #   'MountainCarContinuous-v0'
    #   'BipedalWalker-v3'
    #   'Ant-v2'
    env = gym.make('CartPole-v1', render_mode='rgb_array')

    try: num_output_neurons = env.action_space.shape[0]
    except: num_output_neurons = 1

    network = ContinuousNetwork(
    num_neurons         = 128,
    edge_probability    = 1.5,
    num_input_neurons   = env.observation_space.shape[0],
    num_output_neurons  = num_output_neurons
    )
    # visualize_network(network)
    # visualize_network(network, show_weight=True)

    for neuron in network.output_neurons:
        neuron.activation = Sigmoid()

    network.print_info()
    rmsprop = RMSProp(alpha=5e-6, beta=0.99, reg=0)
    network.set_optimizer(rmsprop)

    train(network, env, episodes=1000, render=False, plot=True, gif=False)
    # watch_rl_episode(network, env)
    # visualize_episode(env, network, name='prelim_rl_results')


if __name__ == '__main__':
    main()