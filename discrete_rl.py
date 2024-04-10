from utils import *
from network import *
from environments import *

def train(network, env, episodes=1000, time=500, render=False, plot=True, gif=False):
    loss_function = MSE()

    episode_rewards = []

    NETWORK_METRICS = ['energy', 'grad_energy', 'weight_energy', 'prop_input', 'prop_grad']
    NETWORK_METRICS_LABELS = ['Energy', 'Gradient Energy', 'Weight Magnitude', 'Network Utilization', 'Gradient Utilization']
    num_metrics = len(NETWORK_METRICS)
    metrics = {}
    for metric in NETWORK_METRICS:
        metrics[metric] = ([], [])
    
    best_t = 0
    for episode in range(episodes):
        state = env.env.reset()[0]
        image_frames = []
        total_reward = 0

        actor_losses, rewards, grads = [], [], []

        pbar = tqdm(range(time))
        for t in pbar:
            if render:
                image_array = env.env.render()
                image_frame = Image.fromarray(image_array)
                image_frames.append(image_frame)
            
            next_state, reward, done, action_desc = env.act(network, state)
            output = action_desc['output']
            action = action_desc['action']
            minimizer = env.minimizer(*next_state)
            dlda = 
            network.backward_pass(loss_function, np.array([minimizer for _ in range(env.action_space)]), np.array([0 for _ in range(env.action_space)]))

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
    env = CARTPOLE

    network = Network(
    num_neurons         = 64,
    edge_probability    = 1.5,
    num_input_neurons   = env.observation_space,
    num_output_neurons  = env.action_space
    )
    env.configure_newtork(network)
    network.print_info()

    train(network, env, episodes=1000, render=False, plot=True, gif=False)


if __name__ == '__main__':
    main()


