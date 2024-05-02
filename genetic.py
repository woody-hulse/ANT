import multiprocessing
import warnings ; warnings.warn = lambda *args,**kwargs: None # for gym np.bool8

from utils import *
from network import *
from environments import *
import seaborn as sns
sns.set_style(style='whitegrid')

EPISODES = 10
TRAIN = True
TIME = 200

def compute_discounted_gradients(gradients, rewards, gamma=0.99):
    discounted_gradients = np.zeros_like(gradients[0])
    T = len(rewards) - 1
    for t in range(len(gradients)):
        discounted_gradients += gradients[t] * rewards[t] * np.power(gamma, T - t)
    return discounted_gradients

def simulate_episode(network, env, seed=None, episodes=EPISODES, train=TRAIN, time=TIME):
    total_reward = 0
    for episode in range(episodes):
        if seed: env.env.set_seed(seed)
        state = env.reset()

        rewards = []
        gradients = []
        for t in range(time):
            logits = network.forward(state)
            action = np.argmax(logits)
            # action = np.random.choice(n=len(logits), p=logits)
            # action = env.discrete_action(logits)
            env.epsilon *= env.epsilon_decay
            state, reward, done = env.step(action)
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

        network.metrics['rewards'].append(total_reward)
    return total_reward / episodes, network


def test_network(network, env, seed=None, episodes=1, train=False, time=TIME):
    total_rewards = np.zeros(episodes)
    for episode in range(episodes):
        if seed: env.env.set_seed(seed)
        state = env.reset()
        total_reward = 0
        rewards = []
        gradients = []
        pbar = tqdm(range(time))
        for t in pbar:
            logits = network.forward(state)
            action = np.argmax(logits)
            # action = env.discrete_action(logits)
            state, reward, done = env.step(action)
            rewards.append(reward)
            total_reward += reward

            if train:
                probs = Softmax()(logits)
                logprobs = np.log(probs[action])
                grad = np.eye(env.env.action_space.n)[action] - probs
                gradients.append(grad)
                discounted_gradients = compute_discounted_gradients(gradients, rewards, gamma=0.99)
                network.backward_(discounted_gradients, np.array([0]), clip=10, update_metrics=False, t=t)

            pbar_string = f'Episode {episode : 05}: reward={sum(rewards) : 9.3f}'
            pbar.set_description(pbar_string) 
            
            if done: break

        total_rewards[episode] = total_reward
    return np.mean(total_rewards)


def mutate(network, args):
    network.mutate(**args)
    return network


def evolve(network, env, mutation_args, episodes=1000, mutations=1000, k=5, graph=False, render=False, plot=True, multiprocess=True):
    train_rewards = []
    test_rewards = []
    NETWORK_METRICS = ['energy', 'grad_energy', 'weight_energy', 'prop_input', 'prop_grad']
    NETWORK_METRICS_LABELS = ['Energy', 'Gradient Energy', 'Weight Magnitude', 'Network Utilization', 'Gradient Utilization']
    num_metrics = len(NETWORK_METRICS)
    metrics = {}

    best_networks = [copy.deepcopy(network) for _ in range(k)]
    with multiprocessing.Pool(processes=12) as pool:# multiprocessing.cpu_count()) as pool:
        for metric in NETWORK_METRICS:
            metrics[metric] = ([], [])
        for e in range(episodes):
            # t = time.time()

            # seeds = np.random.randint(1, 100000, size=2)
            if multiprocess:
                networks = []
                for network in best_networks:
                    networks.extend(pool.map(copy.deepcopy, [network for _ in range(mutations // k)]))
                envs = pool.map(copy.deepcopy, [env for _ in range(mutations // k * k)])
                networks = pool.starmap(mutate, zip(networks, [mutation_args for _ in range(mutations // k * k)]))
                # seeds = [seeds[0] for _ in range(mutations // k * k)]
                total_rewards_networks = pool.starmap(simulate_episode, zip(networks, envs))#, seeds))
            else:
                networks = [copy.deepcopy(network) for _ in range(mutations)]
                envs = [copy.deepcopy(env) for _ in range(mutations)]
                network_env_tuples = zip(networks, envs)
                for tup in network_env_tuples:
                    reward_network = simulate_episode(tup)
                    total_rewards_networks.append(reward_network)

            total_rewards = np.zeros(len(total_rewards_networks))
            for i, (reward, network) in enumerate(total_rewards_networks):
                networks[i] = network
                total_rewards[i] = reward
            
            best_networks = np.array(networks)[np.argsort(total_rewards)[-k:]]
            network = best_networks[-1]
            for metric in metrics.keys():
                metrics[metric][0].append(network.metrics[metric][-1])
                metrics[metric][1].append(np.mean(network.metrics[metric][-500:]))
            test_reward = test_network(network, env, episodes=100, train=False)#, seed=seeds[0])
            debug_print([f'episode {e : 4} | total reward: {np.max(total_rewards) : 9.3f}; ' + \
                        f'average test reward: {test_reward : 9.3f}; neuron count: {network.num_neurons : 6}; ' + \
                        f'connection count: {network.num_edges : 6}; weight magnitude: {sum([np.sum(np.abs(neuron.weights)) for neuron in network.neurons]) : 9.3f}'])

            train_rewards.append(np.max(total_rewards))
            test_rewards.append(test_reward)

            if graph:
                plot_graph(network, spring=True)
            
            if render:
                visualize_episode(network, copy.deepcopy(env), f'rl_results/genetic_episode_{e}', time=TIME) 

            # print('train time:', time.time() - t)

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

    
    return network

def train(env, size, learning_rate, episodes):
    network = Network(
    num_neurons         = size,
    edge_probability    = 1.5,
    num_input_neurons   = env.observation_space,
    num_output_neurons  = env.action_space
    )
    env.optimizer = RMSProp(alpha=learning_rate, beta=0.99)
    env.configure_newtork(network)

    mutation_args = {
        'neuron_mutation_rate': 1,
        'edge_mutation_rate': 0.05,
        'weight_mutation_rate': 1e-4
    }

    network = evolve(network, env, mutation_args, episodes=episodes//10, mutations=128, graph=False, plot=False, render=False, k=10)

    return network.metrics['rewards']



def main():
    # env = CARTPOLE        # *
    env = MOUNTAINCAR
    # env = LUNARLANDER     # *
    # env = WATERMELON
    # env = ACROBOT         # *
    # env = CARL_CARTPOLE   # *
    # env = CARL_ACROBOT

    network = Network(
    num_neurons         = 32,
    edge_probability    = 1.5,
    num_input_neurons   = env.observation_space,
    num_output_neurons  = env.action_space
    )
    env.configure_newtork(network)
    network.print_info()
    # plot_graph(network)

    '''
    mutation_args = {
        'neuron_mutation_rate': 0.01,
        'edge_mutation_rate': 0.01,
        'weight_mutation_rate': 1e-5
    }
    '''

    mutation_args = {
        'neuron_mutation_rate': 1,
        'edge_mutation_rate': 0.05,
        'weight_mutation_rate': 1e-4
    }

    evolve(network, env, mutation_args, episodes=30, mutations=500, graph=False, render=False)


if __name__ == '__main__':
    main()