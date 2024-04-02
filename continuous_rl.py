import gym
import copy
import time
import cv2
from PIL import Image
import IPython as ip
import seaborn as sns
sns.set_style(style='whitegrid',rc={'font.family': 'serif','font.serif':'Times'})

from utils import *
from continuous_network import *
from environments import *


def visualize_episode(network, env, name):
    debug_print(['Visualizing episode'])

    state = env.reset()[0]
    image_frames = []
    max_steps_per_episode = env.spec.max_episode_steps
    
    for step in tqdm(range(500)):
        output = network.forward_pass(state)
        split = len(output) // 2
        mu, var = output[:split], output[split:]
        sigma = np.sqrt(var)
        action = [np.random.normal(m, s) for m, s in zip(mu, sigma)]
        
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
    

def train(network, env, episodes=1000, time=500, render=False, plot=True, gif=False):
    gamma = 0.99  # Discount factor for future rewards
    beta = 1e-4 # Beta for entropy loss

    loss_fn = MSE()

    episode_rewards = []

    NETWORK_METRICS = ['energy', 'grad_energy', 'weight_energy', 'prop_input', 'mean_var']
    NETWORK_METRICS_LABELS = ['Energy', 'Gradient Energy', 'Weight Magnitude', 'Network Utilization', 'Action Convergence']
    num_metrics = len(NETWORK_METRICS)
    metrics = {}
    for metric in NETWORK_METRICS:
        metrics[metric] = ([], [])
    
    best_t = 0
    for episode in range(episodes):
        state = env.env.reset()[0]

        image_frames = []
        G = 0
        total_reward = 0
        p_reward = 0

        actor_losses = []
        rewards = []
        grads = []

        pbar = tqdm(range(time))
        for t in pbar:
            if render:
                image_array = env.env.render()
                image_frame = Image.fromarray(image_array)
                image_frames.append(image_frame)

            # Perform action
            next_state, reward, done, action_desc = env.act(network, state)
            action = action_desc['action']
            mu = action_desc['mu']
            var = action_desc['var']
            sigma = action_desc['sigma']

            reward = np.array([reward])
            pred_reward = network.value
            # pred_next_reward = network.get_next_value()

            # Compute loss
            advantage = reward - pred_reward
            actor_loss = -np.mean(advantage * network.calculate_logprobs(mu, sigma, action))
            critic_loss = np.square(advantage)
            loss = np.mean(actor_loss) + np.mean(critic_loss)

            # print(actor_loss, critic_loss)
            
            mean_var = np.mean(var)
            network.metrics['loss'].append(loss)
            network.metrics['mean_var'].append(mean_var)

            # Compute derivatives
            dlogpdmu = - (mu - action) / (np.square(sigma) + network.epsilon)
            dldmu = - advantage * dlogpdmu
            dlds = -advantage * (-np.square(mu - action) / (np.power(sigma, 3) + network.epsilon) - 1 / (np.pi * sigma + network.epsilon))
            dldc = 2 * advantage
            dldy = np.concatenate([dldmu, np.zeros_like(dlds)])

            if not done:
                actor_losses.append(np.sum(actor_loss))
                rewards.append(reward[0])
                grads.append(dldc)

            # Backward propagate
            network.backward_pass_(dldy, dldc, decay=0, clip=100, update_metrics=True)
            # network.backward_pass(loss_fn, reward, pred_reward)

            '''

            # Compute loss
            pred_reward = network.value
            log_probs = (reward - pred_reward) * network.calculate_logprobs(mu, sigma, action)

            value_loss = loss_fn(reward, pred_reward)
            policy_loss = -np.mean(log_probs)
            entropy_loss = beta * np.mean(-(np.log(2 * np.pi * var) + 1) / 2)

            loss = value_loss + policy_loss + entropy_loss
            network.metrics['loss'].append(loss)

            # Compute dL/dy and perform backward pass
            dlvdy = loss_fn.dydx(np.array([reward]), pred_reward)
            dlpdy = (mu - action) / (np.square(sigma) + network.epsilon / 2)
            dledy = 0

            dlvds = 0
            dlpds = np.square(mu - action) / (np.power(sigma, 3) + network.epsilon) - (1 / sigma)
            dleds = dlpds

            dldy = np.concatenate([dlvdy + dlpdy + dledy, dlvds + dlpds + dleds])

            network.backward_pass_(dldy, decay=0, update_metrics=True)

            '''

            max_chars = 160
            total_reward += np.sum(reward)
            pbar_string = f'Episode {episode : 05}: reward={total_reward : 9.3f}; action={np.sum(action) : 7.3f}; convergence={mean_var : 7.3f}; ' + \
                network.get_metrics_string(metrics=['loss', 'energy', 'grad_energy'])
            
            if len(pbar_string) > max_chars: pbar_string = pbar_string[:max_chars - 3] + '...'

            pbar.set_description(pbar_string)

            state = next_state

            if gif:
                plot_graph(network, title=f't{t}', save=True, save_directory='graph_images/')
            
            if (done or t == time - 1) and render:
                image_frames[0].save(f'rl_results/episode{episode}.gif', 
                            save_all = True, 
                            duration = 20,
                            loop = 0,
                            append_images = image_frames[1:])

            '''
            if t > best_t:
                best_t = t
                if (done or t == time - 1) and render:
                    image_frames[0].save(f'rl_results/episode{episode}.gif', 
                            save_all = True, 
                            duration = 20,
                            loop = 0,
                            append_images = image_frames[1:])
            '''
            
            if done: break

            # if t == time - 1: print(dlds, var)

        # plt.plot(actor_losses)
        # plt.plot(rewards)
        # plt.plot(grads)
        # plt.show()

        if gif:
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
    #   PENDULUM
    #   MOUNTAINCAR_CONTINUOUS
    #   BIPEDALWAKER
    #   ANT
    env = PARAMETRIC_BIPEDALWALKER

    network = ContinuousNetwork(
    num_neurons         = 64,
    edge_probability    = 1.8,
    num_input_neurons   = env.observation_space,
    num_output_neurons  = env.action_space
    )
    # plot_graph(network)
    # visualize_network(network, show_weight=True)

    network.print_info()

    env.configure_newtork(network)

    train(network, env, episodes=300, render=True, plot=True, gif=False)
    # watch_rl_episode(network, env)
    visualize_episode(network, env, name='prelim_rl_results')


if __name__ == '__main__':
    main()