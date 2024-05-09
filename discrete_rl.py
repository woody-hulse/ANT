# import mygrad as mg

import multiprocessing

from utils import *
from ant import ANT
from ann import ANN
from environments import *
import seaborn as sns
sns.set_style(style='whitegrid')

def compute_discounted_gradients(gradients, rewards, gamma=0.99):
    discounted_gradients = np.zeros_like(gradients[0])
    T = len(rewards) - 1
    for t in range(len(gradients)):
        discounted_gradients += gradients[t] * rewards[t] * np.power(gamma, T - t)
    return discounted_gradients

def train(network, env, episodes=1000, time=500, render=False, plot=True, gif=False, pool=None):
    episode_rewards = []

    NETWORK_METRICS = ['energy', 'grad_energy', 'weight_energy', 'prop_input', 'prop_grad']
    NETWORK_METRICS_LABELS = ['Energy', 'Gradient Energy', 'Weight Magnitude', 'Network Utilization', 'Gradient Utilization']
    num_metrics = len(NETWORK_METRICS)
    metrics = {}
    for metric in NETWORK_METRICS:
        metrics[metric] = ([], [])

    for episode in range(episodes):
        state = env.reset()
        total_reward = 0
        rewards = []
        gradients = []
        
        image_frames = []
        pbar = tqdm(range(time))
        e = 0

        for t in pbar:
            if render:
                image_array = env.env.render()
                image_frame = Image.fromarray(image_array)
                image_frames.append(image_frame)

            action, probs = network.discrete_act(state, pool)
            logprobs = np.log(probs[action])
            loss = -np.sum(logprobs)

            state, reward, done = env.step(action)
            grad = probs - np.eye(env.action_space)[action]
            network.metrics['loss'].append(loss)
            gradients.append(grad)
            rewards.append(reward)
            total_reward += reward

            # grad = compute_discounted_gradients(gradients, rewards, gamma=0.99)#  / (t + 1)
            if pool: network.parallel_backward(grad, pool)
            else: network.backward(grad, clip=100, t=t, accumulate=True)

            e += action

            max_chars = 160
            pbar_string = f'Episode {episode : 05}: reward={total_reward : 9.3f}; action={action : 2}; conf={probs[action] : 6.9f}; e={e / (t + 1) : 6.4f}; ' + \
            network.get_metrics_string(metrics=['energy'])
            if len(pbar_string) > max_chars: pbar_string = pbar_string[:max_chars - 3] + '...'
            pbar.set_description(pbar_string)

            if (done or t == time - 1) and render and (episode % 1 == 0):
                image_frames[0].save(f'rl_results/episode{episode}.gif', 
                            save_all = True, 
                            duration = 20,
                            loop = 0,
                            append_images = image_frames[1:])
            if gif and episode % 10 == 0:
                plot_graph(network, title=f't{t}', save=True, save_directory='graph_images/')

            if done: break

        env.epsilon *= env.epsilon_decay

        # network.apply_accumulated_gradients()
        network.apply_discounted_accumulated_gradients(rewards, gamma=0.99)

        if gif and episode % 10 == 0:
            convert_files_to_gif(directory='graph_images/', name=f'graph_results/network_weights_episode{episode}.gif')
        
        if network.use_metrics:
            for metric in metrics.keys():
                metrics[metric][0].append(network.metrics[metric][-1])
                metrics[metric][1].append(np.mean(network.metrics[metric][-t:]))
        episode_rewards.append(total_reward)

    if plot:
        fig, axes = plt.subplots(num_metrics + 1, 1, figsize=(10, 10), gridspec_kw={'height_ratios': [1] + [0.1 for _ in range(num_metrics)]}, sharex=True)
        axes[0].set_title('Training summary')
        for a in [1, 5, 25, 125, 625]:
            if a > len(episode_rewards): continue
            if a == 1:
                axes[0].plot(episode_rewards, color='blue', label='Total reward')
            else:
                ma = moving_average(episode_rewards, a)
                axes[0].plot(ma, label=f'Total reward ({a} episode moving average)')
        axes[0].legend()
        if network.use_metrics:
            for i, ((metric, values), label) in enumerate(zip(metrics.items(), NETWORK_METRICS_LABELS)):
                axes[i + 1].plot(metrics[metric][0], label=label)
                axes[i + 1].plot(metrics[metric][1], label='Mean ' + label)
                axes[i + 1].legend()
        plt.legend()
        plt.xlabel('Episode')
        plt.show()

    return episode_rewards


def convergence_train(env, size, learning_rate, episodes):
    network = ANT(
    num_neurons         = size,
    edge_probability    = 3,
    num_input_neurons   = env.observation_space,
    num_output_neurons  = env.action_space
    )
    env.optimizer = RMSProp(alpha=learning_rate, beta=0.99)
    env.configure_newtork(network)
    rewards = train(network, env, episodes=episodes, time=200, render=False, plot=False, gif=False)

    return np.array(rewards)
    

def main():
    # env = CARTPOLE
    # env = MOUNTAINCAR
    # env = LUNARLANDER
    # env = ACROBOT
    # env = CARL_CARTPOLE
    # env = WATERMELON
    # env = LUNARLANDER_EARTH
    env = CARL_ACROBOT

    base_ant = ANT(
    num_neurons         = 20,
    edge_probability    = 3,
    num_input_neurons   = env.observation_space,
    num_output_neurons  = env.action_space
    )
    env.configure_newtork(base_ant)
    base_ann = ANN([env.observation_space, 14, 14, env.action_space])

    network = base_ant
    # env.configure_newtork(network)
    # network.print_info()
    # plot_graph(network)
    # network.load('saved_networks/genetic_1_Acrobot.pkl')
    # network.reset_weights(seed=420)
    # network.optimizer = RMSProp(alpha=1e-6, beta=0.99)

    # network.load('saved_networks/ant_earth.pkl')
    # network.optimizer = RMSProp(alpha=1e-6, beta=0.99)

    # with multiprocessing.Pool(processes=12) as pool:
    train(network, env, episodes=200, time=200, render=True, plot=True, gif=False)


if __name__ == '__main__':
    main()


