import multiprocessing
import warnings ; warnings.warn = lambda *args,**kwargs: None # for gym np.bool8

from utils import *
from network import *
from environments import *
import seaborn as sns
sns.set_style(style='whitegrid')

TRAIN = True

def compute_discounted_gradients(gradients, rewards, gamma=0.99):
    discounted_gradients = np.zeros_like(gradients[0])
    T = len(rewards) - 1
    for t in range(len(gradients)):
        discounted_gradients += gradients[t] * rewards[t] * np.power(gamma, T - t)
    return discounted_gradients

def simulate_episode(network_env, train=TRAIN, time=500):
    network, env = network_env
    state = env.env.reset()[0]
    total_reward = 0
    rewards = []
    gradients = []
    for t in range(time):
        logits = network.forward(state)
        action = np.argmax(logits)
        state, reward, done, _, _ = env.env.step(action)
        rewards.append(reward)
        total_reward += reward

        if train:
            probs = Softmax()(logits)
            logprobs = np.log(probs[action])
            grad = np.eye(env.env.action_space.n)[action] - probs
            gradients.append(grad)
            discounted_gradients = compute_discounted_gradients(gradients, rewards, gamma=0.99)
            network.backward_(discounted_gradients, np.array([0]), clip=10, update_metrics=True, t=t)

        if done: break
    return total_reward, network


def test_network(network, env, episodes=1, train=TRAIN, time=500):
    total_rewards = np.zeros(episodes)
    for e in range(episodes):
        state = env.env.reset()[0]
        total_reward = 0
        rewards = []
        gradients = []
        for t in range(time):
            logits = network.forward(state)
            action = np.argmax(logits)
            state, reward, done, _, _ = env.env.step(action)
            rewards.append(reward)
            total_reward += reward

            if train:
                probs = Softmax()(logits)
                logprobs = np.log(probs[action])
                grad = np.eye(env.env.action_space.n)[action] - probs
                gradients.append(grad)
                discounted_gradients = compute_discounted_gradients(gradients, rewards, gamma=0.99)
                # network.backward_(discounted_gradients, np.array([0]), clip=10, update_metrics=False, t=t)
            
            if done: break

        total_rewards[e] = total_reward
    return np.mean(total_rewards)


def evolve(network, env, episodes=1000, mutations=1000, graph=False, render=False, plot=True):

    train_rewards = []
    test_rewards = []
    NETWORK_METRICS = ['energy', 'grad_energy', 'weight_energy', 'prop_input', 'prop_grad']
    NETWORK_METRICS_LABELS = ['Energy', 'Gradient Energy', 'Weight Magnitude', 'Network Utilization', 'Gradient Utilization']
    num_metrics = len(NETWORK_METRICS)
    metrics = {}
    for metric in NETWORK_METRICS:
        metrics[metric] = ([], [])
    for e in range(episodes):
        state = env.env.reset()[0]
        networks = [copy.deepcopy(network) for _ in range(mutations)]
        envs = [copy.deepcopy(env.env) for _ in range(mutations)]

        # Mutate networks
        for network in networks:
            network.mutate(neuron_mutation_rate=0.01, edge_mutation_rate=0.01, weight_mutation_rate=1e-5)

        network_env_tuples = list(zip(networks, envs))

        with multiprocessing.Pool(processes=multiprocessing.cpu_count()) as pool:
            total_rewards_networks = pool.map(simulate_episode, network_env_tuples)
        
        total_rewards = np.zeros(len(total_rewards_networks))
        for i, (reward, network) in enumerate(total_rewards_networks):
            networks[i] = network
            total_rewards[i] = reward

        # total_rewards = []
        # for tup in network_env_state_tuples:
        #     total_rewards_networks = simulate_episode(tup)

        network = networks[np.argmax(total_rewards)]
        for metric in metrics.keys():
            metrics[metric][0].append(network.metrics[metric][-1])
            metrics[metric][1].append(np.mean(network.metrics[metric][-500:]))
        test_reward = test_network(network, env, episodes=10)
        debug_print([f'episode {e : 4} | total reward: {np.max(total_rewards) : 9.3f}; ' + \
                     f'average test reward: {test_reward : 9.3f}; neuron count: {network.num_neurons : 6}; ' + \
                     f'connection count: {network.num_edges : 6}; weight magnitude: {sum([np.sum(np.abs(neuron.weights)) for neuron in network.neurons]) : 9.3f}'])

        train_rewards.append(np.max(total_rewards))
        test_rewards.append(test_reward)

        if graph:
            plot_graph(network, spring=True)
        
        if render:
            visualize_episode(network, copy.deepcopy(env.env), f'rl_results/genetic_episode_{e}') 

    if plot:
        fig, axes = plt.subplots(num_metrics + 1, 1, figsize=(10, 10), gridspec_kw={'height_ratios': [1] + [0.1 for _ in range(num_metrics)]}, sharex=True)
        axes[0].set_title('Genetic training summary')
        for a in [1, 5, 25, 125, 625]:
            if a == 1:
                axes[0].plot(test_rewards, color='blue', label='Total reward')
            else:
                ma = moving_average(test_rewards, a)
                axes[0].plot(ma, label=f'Total reward ({a} episode moving average)')
        axes[0].legend()
        for i, ((metric, values), label) in enumerate(zip(metrics.items(), NETWORK_METRICS_LABELS)):
            axes[i + 1].plot(metrics[metric][0], label=label)
            axes[i + 1].plot(metrics[metric][1], label='Mean ' + label)
            axes[i + 1].legend()
        plt.legend()
        plt.xlabel('Episode')
        plt.show()



def main():
    env = CARTPOLE
    # env = MOUNTAINCAR
    # env = LUNARLANDER

    network = Network(
    num_neurons         = 16,
    edge_probability    = 3,
    num_input_neurons   = env.observation_space,
    num_output_neurons  = env.action_space
    )
    env.configure_newtork(network)
    network.print_info()
    # plot_graph(network)

    evolve(network, env, episodes=100, mutations=128, graph=True, render=False)


if __name__ == '__main__':
    main()