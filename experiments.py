from tqdm import tqdm
import time
import copy
import multiprocessing

from utils import *
import discrete_rl
import genetic
from reinforce_offline import *
from environments import *
from ant import ANT
from ann import ANN

def test_train(env, size, learning_rate, episodes):
    return np.random.normal(loc=learning_rate, scale=size/5, size=(episodes,))

def convergence(env, ann_train, ant_train2, episodes=1000, trials=5):
    learning_rates = [1e-3, 1e-4, 1e-5, 1e-6]
    sizes = [16, 32, 64, 128]

    ann_lr_size_rewards = np.zeros((len(learning_rates), len(sizes), episodes))
    ant_lr_size_rewards = np.zeros((len(learning_rates), len(sizes), episodes))
    for i, learning_rate in tqdm(enumerate(learning_rates)):
        for j, size in enumerate(sizes):
            for trial in range(trials):
                print('[', i * len(sizes) * trials + j * trials + trial, '/', len(learning_rates) * len(sizes) * trials, ']')
                total_rewards = ann_train(env=env, size=size, learning_rate=learning_rate, episodes=episodes)
                ann_lr_size_rewards[i][j] += np.array(total_rewards) / trials

                total_rewards = ant_train2(env=env, size=size, learning_rate=learning_rate, episodes=episodes)
                ant_lr_size_rewards[i][j] += np.array(total_rewards) / trials
    
    plt.figure(figsize=(8, 6), dpi=300)
    for i in range(len(learning_rates)):
        for j in range(len(sizes)):
            plt.plot(ann_lr_size_rewards[i][j], color='blue', alpha=0.3)
            plt.plot(ant_lr_size_rewards[i][j], color='orange', alpha=0.3)

    ann_lr_size_rewards = np.reshape(ann_lr_size_rewards, (len(learning_rates) * len(sizes), episodes))
    ant_lr_size_rewards = np.reshape(ant_lr_size_rewards, (len(learning_rates) * len(sizes), episodes))
    ann_rewards_ind = np.argmax(np.mean(ann_lr_size_rewards, axis=1))
    ant_rewards_ind = np.argmax(np.mean(ant_lr_size_rewards, axis=1))

    plt.plot(ann_lr_size_rewards[ann_rewards_ind], color='blue', label='ANN quickest convergence', linewidth=2)
    plt.plot(ant_lr_size_rewards[ant_rewards_ind], color='orange', label='ANT quickest convergence', linewidth=2)

    plt.title('ANN vs ANT convergence across hyperparameter space', fontsize=14)
    plt.xlabel('Episodes', fontsize=12)
    plt.ylabel('Total reward', fontsize=12)

    plt.legend()
    plt.grid(True)
    # plt.tight_layout()
    plt.show()


def evolution():
    env = LUNARLANDER
    base_ant = ANT(
        num_neurons         = 32,
        edge_probability    = 2,
        num_input_neurons   = env.observation_space,
        num_output_neurons  = env.action_space
    )
    env.configure_newtork(base_ant)

    base_ann = ANN([env.observation_space, 14, 14, env.action_space])

    mutation_args = {
        'neuron_mutation_rate': 0.5,
        'edge_mutation_rate': 0.05,
        'weight_mutation_rate': 1e-4
    }

    base_ant.print_info()
    evolved_ant, ant_evolution_rewards = genetic.evolve(base_ant, env, mutation_args, episodes=10, mutations=100, graph=False, render=False, save=False, plot=False)
    evolved_ant_rewards = evolved_ant.metrics['rewards']

    base_ann.print_info()
    evolved_ann, ann_evolution_rewards = genetic.evolve(base_ann, env, mutation_args, episodes=10, mutations=100, graph=False, render=False, save=False, plot=False)
    evolved_ann_rewards = evolved_ann.metrics['rewards']
    
    # ann_highs, ann_lows = get_rewards_range(ann_evolution_rewards, len(evolved_ann_rewards))
    # ant_highs, ant_lows = get_rewards_range(ant_evolution_rewards, len(evolved_ant_rewards))
    
    plt.figure(figsize=(8, 6), dpi=150)
    # plt.fill_between(range(len(evolved_ann_rewards)), ann_highs, ann_lows, color='#ff8819', alpha=0.2)
    for rewards in ann_evolution_rewards:
        # plt.scatter(np.arange(len(rewards)), rewards, color='#ff8819', s=0.5, alpha=0.3)
        plt.plot(rewards, color='#ffc28a', linewidth=0.5, alpha=0.2)
    plt.plot(evolved_ann_rewards, color='#ff8819', label='ANN', linewidth=0.8)

    # plt.fill_between(range(len(evolved_ant_rewards)), ant_highs, ant_lows, color='#0e68cf', alpha=0.2)
    for rewards in ant_evolution_rewards:
        # plt.scatter(np.arange(len(rewards)), rewards, color='#0e68cf', s=0.5, alpha=0.3)
        plt.plot(rewards, color='#83a9d4', linewidth=0.5, alpha=0.2)
    plt.plot(evolved_ant_rewards, color='#0e68cf', label='ANT', linewidth=0.8)

    plt.title(f'ANN vs. ANT Evolution in {env.name}', fontsize=12)
    plt.xlabel('Episodes', fontsize=10)
    plt.ylabel('Total evaluation reward', fontsize=10)

    plt.legend()
    plt.xlim((0, len(evolved_ann_rewards)))
    plt.ylim((-250, 250))

    ax = plt.gca()
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['left'].set_visible(True)
    ax.spines['bottom'].set_visible(True)
    # ax.spines['bottom'].set_position(('data', 0))
    ax.spines['left'].set_position(('data', 0))
    ax.yaxis.set_ticks_position('left')
    ax.xaxis.set_ticks_position('bottom')

    # plt.grid(True)
    plt.grid(False)
    # plt.tight_layout()
    plt.show()


def average_evolution(iterations):
    episodes = 20
    mutations = 100

    evolution_size = episodes * genetic.EPISODES
    average_ann_rewards = np.zeros(evolution_size)
    average_ant_rewards = np.zeros(evolution_size)
    for i in range(iterations):
        env = LUNARLANDER
        base_ant = ANT(
            num_neurons         = 32,
            edge_probability    = 2,
            num_input_neurons   = env.observation_space,
            num_output_neurons  = env.action_space
        )
        env.configure_newtork(base_ant)

        base_ann = ANN([env.observation_space, 14, 14, env.action_space])

        mutation_args = {
            'neuron_mutation_rate': 0.5,
            'edge_mutation_rate': 0.05,
            'weight_mutation_rate': 1e-4
        }

        base_ant.print_info()
        evolved_ant, ant_evolution_rewards = genetic.evolve(base_ant, env, mutation_args, episodes=episodes, mutations=mutations, graph=False, render=False, save=False, plot=False)
        evolved_ant_rewards = np.array(evolved_ant.metrics['rewards'])[:evolution_size]

        base_ann.print_info()
        evolved_ann, ann_evolution_rewards = genetic.evolve(base_ann, env, mutation_args, episodes=episodes, mutations=mutations, graph=False, render=False, save=False, plot=False)
        evolved_ann_rewards = np.array(evolved_ann.metrics['rewards'])[:evolution_size]

        average_ann_rewards += evolved_ann_rewards / iterations
        average_ant_rewards += evolved_ant_rewards / iterations
    
    plt.figure(figsize=(8, 6), dpi=150)
    plt.plot(average_ann_rewards, color='#ff8819', label='ANN', linewidth=1)
    plt.plot(average_ant_rewards, color='#0e68cf', label='ANT', linewidth=1)
    plt.title(f'Average ANN vs. ANT Evolution in {env.name}', fontsize=12)

    plt.legend()
    plt.xlim((0, len(evolved_ann_rewards)))
    plt.ylim((-250, 250))

    ax = plt.gca()
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['left'].set_visible(True)
    ax.spines['bottom'].set_visible(True)
    # ax.spines['bottom'].set_position(('data', 0))
    ax.spines['left'].set_position(('data', 0))
    ax.yaxis.set_ticks_position('left')
    ax.xaxis.set_ticks_position('bottom')

    # plt.grid(True)
    plt.grid(False)
    # plt.tight_layout()
    plt.show()


def tensorflow_baseline_evolve(env, neuron_count, mutations=100):
    k = 5
    network = tf.keras.Sequential([tf.keras.layers.Dense(8, activation='tanh') for _ in range(neuron_count // 8)] + 
        [tf.keras.layers.Dense(env.action_space, activation='softmax')])

    best_networks = [tf.keras.models.clone_model(network) for _ in range(k)]
    with multiprocessing.Pool(processes=15) as pool:
        networks = []
        for network in best_networks:
            networks.extend(pool.map(tf.keras.models.clone_model, [network for _ in range(mutations // k)]))
        envs = pool.map(copy.deepcopy, [env for _ in range(mutations // k * k)])
        pool.starmap(train_with_model, zip(networks, envs))


def parallelization_performance():
    env = LUNARLANDER

    singleproc_times = []
    multiproc_times = []
    tensorflow_times = []
    neuron_counts = [8, 16, 32, 64, 128]
    for neuron_count in neuron_counts:
        debug_print(['Starting neuron count:', neuron_count])
        base_ant = ANT(
            num_neurons         = neuron_count,
            edge_probability    = 8 / np.sqrt(neuron_count),
            num_input_neurons   = env.observation_space,
            num_output_neurons  = env.action_space
        )

        env.configure_newtork(base_ant)

        mutation_args = {
            'neuron_mutation_rate': 0.5,
            'edge_mutation_rate': 0.05,
            'weight_mutation_rate': 1e-4
        }

        tensorflow_time = time.time()
        tensorflow_baseline_evolve(env, neuron_count, mutations=100)
        tensorflow_times.append(time.time() - tensorflow_time)

        singleproc_time = time.time()
        genetic.evolve(copy.deepcopy(base_ant), env, mutation_args, episodes=1, mutations=100, graph=False, render=False, multiprocess=False, save=False, plot=False)
        singleproc_times.append(time.time() - singleproc_time)

        multiproc_time = time.time()
        genetic.evolve(base_ant, env, mutation_args, episodes=1, mutations=100, graph=False, render=False, multiprocess=True, save=False, plot=False)
        multiproc_times.append(time.time() - multiproc_time)    


    plt.figure(figsize=(8, 4), dpi=150)
    plt.plot(neuron_counts, singleproc_times, color='#ff8819', label='Non-parallelized', linewidth=1)
    plt.plot(neuron_counts, multiproc_times, color='#0e68cf', label='Parallelized (12 CPU cores)', linewidth=1)
    plt.plot(neuron_counts, tensorflow_times, color='#555555', label='Tensorflow baseline', linewidth=0.7, linestyle='dashed')
    plt.title(f'ANT Performance on Single Evolution', fontsize=12)
    plt.legend()

    ax = plt.gca()
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['left'].set_visible(True)
    ax.spines['bottom'].set_visible(True)
    # ax.spines['bottom'].set_position(('data', 0))
    # ax.spines['left'].set_position(('data', 0))
    # ax.yaxis.set_ticks_position('left')
    # ax.xaxis.set_ticks_position('bottom')
    plt.xlabel('Neuron count')
    plt.ylabel('Time (s)')

    plt.ylim(0)
    plt.grid(axis='x')
    # plt.grid(which='major', axis='y', linestyle='-')
    plt.show()


def gd():
    env = ACROBOT

    base_ant = ANT(
        num_neurons         = 70,
        edge_probability    = 2,
        num_input_neurons   = env.observation_space,
        num_output_neurons  = env.action_space
    )
    env.configure_newtork(base_ant)
    base_ant.optimizer = RMSProp(alpha=1e-6, beta=0.99)

    base_ann = ANN([env.observation_space, 30, 30, env.action_space])
    base_ann.optimizer = RMSProp(alpha=3e-3, beta=0.99)
    
    ann_rewards = discrete_rl.train(base_ann, env, episodes=1000, time=200, render=False, plot=False, gif=False)
    ant_rewards = discrete_rl.train(base_ant, env, episodes=1000, time=200, render=False, plot=False, gif=False)

    ant_rewards_ma = moving_average(ant_rewards, 50)
    ann_rewards_ma = moving_average(ann_rewards, 50)

    plt.figure(figsize=(8, 6), dpi=150)
    plt.plot(ant_rewards, color='#0e68cf', label='ANT', alpha=0.3)
    plt.plot(ant_rewards_ma, color='#0e68cf', label='ANT (50 episode moving average)')

    plt.plot(ann_rewards, color='#ff8819', label='ANN', alpha=0.3)
    plt.plot(ann_rewards_ma, color='#ff8819', label='ANN (50 episode moving average)')

    ax = plt.gca()
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['left'].set_visible(True)
    ax.spines['bottom'].set_visible(True)
    plt.title(f'ANN vs. ANT Gradient Descent in {env.name}', fontsize=12)
    plt.grid(False)
    plt.legend()
    plt.xlabel('Episode')
    plt.ylabel('Total reward')
    plt.tight_layout()
    # plt.ylim(bottom=-500)
    plt.show()


def evolution2():
    env = ACROBOT
    base_ant = ANT(
        num_neurons         = 70,
        edge_probability    = 2,
        num_input_neurons   = env.observation_space,
        num_output_neurons  = env.action_space
    )
    env.configure_newtork(base_ant)

    base_ann = ANN([env.observation_space, 30, 30, env.action_space])

    mutation_args = {
        'neuron_mutation_rate': 0.5,
        'edge_mutation_rate': 0.05,
        'weight_mutation_rate': 1e-4
    }

    ann_mutation_args = {
        'neuron_mutation_rate': 0.5,
        'edge_mutation_rate': 0.0,
        'weight_mutation_rate': 1e-3
    }

    base_ann.print_info()
    evolved_ann, ann_evolution_rewards = genetic.evolve(base_ann, env, ann_mutation_args, episodes=20, mutations=30, graph=False, render=False, save=False, plot=False)
    evolved_ann_rewards = evolved_ann.metrics['rewards']

    base_ant.print_info()
    evolved_ant, ant_evolution_rewards = genetic.evolve(base_ant, env, mutation_args, episodes=20, mutations=30, graph=False, render=False, save=False, plot=False)
    evolved_ant_rewards = evolved_ant.metrics['rewards']

    evolved_ant_rewards = moving_average(evolved_ant_rewards, 50)
    evolved_ann_rewards = moving_average(evolved_ann_rewards, 50)

    plt.figure(figsize=(8, 6), dpi=150)
    for rewards in ann_evolution_rewards:
        plt.plot(rewards, color='#ff8819', alpha=0.2)
    plt.plot(evolved_ann_rewards, color='#ff8819', label='ANN (50 episode moving average)')

    for rewards in ant_evolution_rewards:
        plt.plot(rewards, color='#0e68cf', alpha=0.2)
    plt.plot(evolved_ant_rewards, color='#0e68cf', label='ANT (50 episode moving average)')

    ax = plt.gca()
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['left'].set_visible(True)
    ax.spines['bottom'].set_visible(True)
    plt.title(f'ANN vs. ANT Evolution in {env.name}', fontsize=12)
    plt.grid(False)
    plt.legend()
    plt.xlabel('Episode')
    plt.ylabel('Total reward')
    plt.tight_layout()
    # plt.ylim(bottom=-500)
    plt.show()


def convergence_test(tests=24):
    env = CARL_ACROBOT
    episodes = 100

    ann_small_sgd_convergence = np.zeros(tests)
    ant_small_sgd_convergence = np.zeros(tests)
    ann_large_sgd_convergence = np.zeros(tests)
    ant_large_sgd_convergence = np.zeros(tests)
    ann_small_evolution_convergence = np.zeros(tests)
    ant_small_evolution_convergence = np.zeros(tests)
    ann_large_evolution_convergence = np.zeros(tests)
    ant_large_evolution_convergence = np.zeros(tests)
    
    for i in range(tests):
        debug_print([f'Starting small test {i}'])
        sgd_ant = ANT(
            num_neurons         = 40,
            edge_probability    = 2,
            num_input_neurons   = env.observation_space,
            num_output_neurons  = env.action_space
        )
        env.configure_newtork(sgd_ant)
        sgd_ant.optimizer = RMSProp(alpha=1e-6, beta=0.99)

        sgd_ann = ANN([env.observation_space, 14, 14, env.action_space])
        sgd_ann.optimizer = RMSProp(alpha=3e-3, beta=0.99)

        sgd_ann_rewards = discrete_rl.train(sgd_ann, env, episodes=episodes, time=200, render=False, plot=False, gif=False)
        sgd_ant_rewards = discrete_rl.train(sgd_ant, env, episodes=episodes, time=200, render=False, plot=False, gif=False)

        ann_small_sgd_convergence[i] = np.mean(sgd_ann_rewards[-10:])
        ant_small_sgd_convergence[i] = np.mean(sgd_ant_rewards[-10:])

        mutation_args = {
            'neuron_mutation_rate': 0.5,
            'edge_mutation_rate': 0.05,
            'weight_mutation_rate': 1e-4
        }

        ann, _ = genetic.evolve(sgd_ann, env, mutation_args, episodes=episodes//10, mutations=36, graph=False, render=False, save=False, plot=False)
        ant, _ = genetic.evolve(sgd_ant, env, mutation_args, episodes=episodes//10, mutations=36, graph=False, render=False, save=False, plot=False)

        ann_small_evolution_convergence[i] = np.mean(ann.metrics['rewards'][-10:])
        ant_small_evolution_convergence[i] = np.mean(ant.metrics['rewards'][-10:])

        debug_print([f'Starting large test {i}'])
        sgd_ant = ANT(
            num_neurons         = 70,
            edge_probability    = 2,
            num_input_neurons   = env.observation_space,
            num_output_neurons  = env.action_space
        )
        env.configure_newtork(sgd_ant)
        sgd_ant.optimizer = RMSProp(alpha=1e-6, beta=0.99)

        sgd_ann = ANN([env.observation_space, 30, 30, env.action_space])
        sgd_ann.optimizer = RMSProp(alpha=3e-3, beta=0.99)

        sgd_ann_rewards = discrete_rl.train(sgd_ann, env, episodes=episodes, time=200, render=False, plot=False, gif=False)
        sgd_ant_rewards = discrete_rl.train(sgd_ant, env, episodes=episodes, time=200, render=False, plot=False, gif=False)

        ann_large_sgd_convergence[i] = np.mean(sgd_ann_rewards[-10:])
        ant_large_sgd_convergence[i] = np.mean(sgd_ant_rewards[-10:])

        mutation_args = {
            'neuron_mutation_rate': 0.5,
            'edge_mutation_rate': 0.05,
            'weight_mutation_rate': 1e-4
        }

        ann, _ = genetic.evolve(sgd_ann, env, mutation_args, episodes=episodes//10, mutations=36, graph=False, render=False, save=False, plot=False)
        ant, _ = genetic.evolve(sgd_ant, env, mutation_args, episodes=episodes//10, mutations=36, graph=False, render=False, save=False, plot=False)

        ann_large_evolution_convergence[i] = np.mean(ann.metrics['rewards'][-10:])
        ant_large_evolution_convergence[i] = np.mean(ant.metrics['rewards'][-10:])
    
    
    # plt.figure(figsize=(8, 6), dpi=150)


    groups = ['ANN-small', 'ANT-small', 'ANN-large', 'ANT-large']
    average_ann_small_sgd_convergence = 200 + np.mean(ann_small_sgd_convergence)
    average_ant_small_sgd_convergence = 200 + np.mean(ant_small_sgd_convergence)
    average_ann_large_sgd_convergence = 200 + np.mean(ann_large_sgd_convergence)
    average_ant_large_sgd_convergence = 200 + np.mean(ant_large_sgd_convergence)

    average_ann_small_evolution_convergence = 200 + np.mean(ann_small_evolution_convergence)
    average_ant_small_evolution_convergence = 200 + np.mean(ant_small_evolution_convergence)
    average_ann_large_evolution_convergence = 200 + np.mean(ann_large_evolution_convergence)
    average_ant_large_evolution_convergence = 200 + np.mean(ant_large_evolution_convergence)

    std_ann_small_sgd = np.std(ann_small_sgd_convergence)
    std_ant_small_sgd = np.std(ant_small_sgd_convergence)
    std_ann_large_sgd = np.std(ann_large_sgd_convergence)
    std_ant_large_sgd = np.std(ant_large_sgd_convergence)

    std_ann_small_evolution = np.std(ann_small_evolution_convergence)
    std_ant_small_evolution = np.std(ant_small_evolution_convergence)
    std_ann_large_evolution = np.std(ann_large_evolution_convergence)
    std_ant_large_evolution = np.std(ant_large_evolution_convergence)

    sgd = [average_ann_small_sgd_convergence, average_ant_small_sgd_convergence, average_ann_large_sgd_convergence, average_ant_large_sgd_convergence]
    evolution = [average_ann_small_evolution_convergence, average_ant_small_evolution_convergence, average_ann_large_evolution_convergence, average_ant_large_evolution_convergence]
    sgd_errors = [std_ann_small_sgd, std_ant_small_sgd, std_ann_large_sgd, std_ant_large_sgd]
    evolution_errors = [std_ann_small_evolution, std_ant_small_evolution, std_ann_large_evolution, std_ant_large_evolution]

    x = [0, 0.4, 0.8, 1.2]
    width = 0.15

    fig, ax = plt.subplots(layout='constrained')
    bars1 = ax.bar(x, sgd, width, label='SGD', color='#6f69e0', yerr=sgd_errors, capsize=3, error_kw={'ecolor':'#363636', 'capthick':2, 'elinewidth':0.8})
    bars2 = ax.bar([p + width for p in x], evolution, width, label='SGD & Evolution', color='#e09169', yerr=evolution_errors, capsize=3, error_kw={'ecolor':'#363636', 'capthick':2, 'elinewidth':0.8})
    ax.set_xticks([p + width / 2 for p in x])
    ax.set_xticklabels(groups)
    ax.legend()

    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['left'].set_visible(True)
    ax.spines['bottom'].set_visible(True)
    plt.title(f'Mean Convergence after {episodes} Episodes in {env.name}', fontsize=12)
    plt.grid(axis='x')
    plt.legend()
    plt.xlabel('Network')
    plt.ylabel('Average total reward')
    plt.tight_layout()
    plt.show()


def convergence_test_ant(tests=24):
    env = ACROBOT
    episodes = 100

    small_ant_sgd_convergence = np.zeros(tests)
    small_ant_evolution_convergence = np.zeros(tests)
    small_ant_sgd_evolution_convergence = np.zeros(tests)

    large_ant_sgd_convergence = np.zeros(tests)
    large_ant_evolution_convergence = np.zeros(tests)
    large_ant_sgd_evolution_convergence = np.zeros(tests)

    mutation_args = {
        'neuron_mutation_rate': 0.5,
        'edge_mutation_rate': 0.05,
        'weight_mutation_rate': 1e-4
    }

    for test in range(tests):
        debug_print([f'Starting small test {test}'])
        sgd_ant = ANT(
            num_neurons         = 40,
            edge_probability    = 2,
            num_input_neurons   = env.observation_space,
            num_output_neurons  = env.action_space
        )
        env.configure_newtork(sgd_ant)
        sgd_ant.optimizer = RMSProp(alpha=1e-6, beta=0.99)

        evolution_ant = ANT(
            num_neurons         = 40,
            edge_probability    = 2,
            num_input_neurons   = env.observation_space,
            num_output_neurons  = env.action_space
        )
        env.configure_newtork(evolution_ant)
        evolution_ant.optimizer = RMSProp(alpha=0, beta=0.99)

        sgd_evolution_ant = ANT(
            num_neurons         = 40,
            edge_probability    = 2,
            num_input_neurons   = env.observation_space,
            num_output_neurons  = env.action_space
        )
        env.configure_newtork(sgd_evolution_ant)
        sgd_evolution_ant.optimizer = RMSProp(alpha=1e-6, beta=0.99)

        a = discrete_rl.train(sgd_ant, env, episodes=episodes, time=200, render=False, plot=False, gif=False)
        b, _ = genetic.evolve(evolution_ant, env, mutation_args, episodes=episodes//10, mutations=36, graph=False, render=False, save=False, plot=False)
        c, _ = genetic.evolve(sgd_evolution_ant, env, mutation_args, episodes=episodes//10, mutations=36, graph=False, render=False, save=False, plot=False)

        small_ant_sgd_convergence[test] = 200 + np.mean(a[-10:])
        small_ant_evolution_convergence[test] = 200 + np.mean(b.metrics['rewards'][-10:])
        small_ant_sgd_evolution_convergence[test] = 200 + np.mean(c.metrics['rewards'][-10:])

        debug_print([f'Starting large test {test}'])
        sgd_ant = ANT(
            num_neurons         = 70,
            edge_probability    = 2,
            num_input_neurons   = env.observation_space,
            num_output_neurons  = env.action_space
        )
        env.configure_newtork(sgd_ant)
        sgd_ant.optimizer = RMSProp(alpha=1e-6, beta=0.99)

        evolution_ant = ANT(
            num_neurons         = 70,
            edge_probability    = 2,
            num_input_neurons   = env.observation_space,
            num_output_neurons  = env.action_space
        )
        env.configure_newtork(evolution_ant)
        evolution_ant.optimizer = RMSProp(alpha=0, beta=0.99)

        sgd_evolution_ant = ANT(
            num_neurons         = 70,
            edge_probability    = 2,
            num_input_neurons   = env.observation_space,
            num_output_neurons  = env.action_space
        )
        env.configure_newtork(sgd_evolution_ant)
        sgd_evolution_ant.optimizer = RMSProp(alpha=1e-6, beta=0.99)

        a = discrete_rl.train(sgd_ant, env, episodes=episodes, time=200, render=False, plot=False, gif=False)
        b, _ = genetic.evolve(evolution_ant, env, mutation_args, episodes=episodes//10, mutations=36, graph=False, render=False, save=False, plot=False)
        c, _ = genetic.evolve(sgd_evolution_ant, env, mutation_args, episodes=episodes//10, mutations=36, graph=False, render=False, save=False, plot=False)

        large_ant_sgd_convergence[test] = 200 + np.mean(a[-10:])
        large_ant_evolution_convergence[test] = 200 + np.mean(b.metrics['rewards'][-10:])
        large_ant_sgd_evolution_convergence[test] = 200 + np.mean(c.metrics['rewards'][-10:])
    

    average_small_ant_sgd_convergence = np.mean(small_ant_sgd_convergence)
    average_small_ant_evolution_convergence = np.mean(small_ant_evolution_convergence)
    average_small_ant_sgd_evolution_convergence = np.mean(small_ant_sgd_evolution_convergence)

    std_small_ant_sgd = np.std(small_ant_sgd_convergence)
    std_small_ant_evolution = np.std(small_ant_evolution_convergence)
    std_small_ant_sgd_evolution = np.std(small_ant_sgd_evolution_convergence)

    average_large_ant_sgd_convergence = np.mean(large_ant_sgd_convergence)
    average_large_ant_evolution_convergence = np.mean(large_ant_evolution_convergence)
    average_large_ant_sgd_evolution_convergence = np.mean(large_ant_sgd_evolution_convergence)

    std_large_ant_sgd = np.std(large_ant_sgd_convergence)
    std_large_ant_evolution = np.std(large_ant_evolution_convergence)
    std_large_ant_sgd_evolution = np.std(large_ant_sgd_evolution_convergence)

    sgd = [average_small_ant_sgd_convergence, average_large_ant_sgd_convergence]
    evolution = [average_small_ant_evolution_convergence, average_large_ant_evolution_convergence]
    sgd_evolution = [average_small_ant_sgd_evolution_convergence, average_large_ant_sgd_evolution_convergence]    

    std_ant_sgd = [std_small_ant_sgd, std_large_ant_sgd]
    std_ant_evolution = [std_small_ant_evolution, std_large_ant_evolution]
    std_ant_sgd_evolution = [std_small_ant_sgd_evolution, std_large_ant_sgd_evolution]

    groups = ['ANT-small', 'ANT-large']
    x = [0, 0.3]
    width = 0.05

    fig, ax = plt.subplots(layout='constrained')
    bars1 = ax.bar(x, sgd, width, label='SGD', color='#6f69e0', yerr=std_ant_sgd, capsize=3, error_kw={'ecolor':'#363636', 'capthick':2, 'elinewidth':0.8})
    bars2 = ax.bar([p + width for p in x], evolution, width, label='Evolution', color='#e84f2c', yerr=std_ant_evolution, capsize=3, error_kw={'ecolor':'#363636', 'capthick':2, 'elinewidth':0.8})
    bars3 = ax.bar([p + 2 * width for p in x], sgd_evolution, width, label='SGD & Evolution', color='#e09169', yerr=std_ant_sgd_evolution, capsize=3, error_kw={'ecolor':'#363636', 'capthick':2, 'elinewidth':0.8})
    ax.legend()
    ax.set_xticks([p + width for p in x])
    ax.set_xticklabels(groups)

    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['left'].set_visible(True)
    ax.spines['bottom'].set_visible(True)
    plt.title(f'Mean Convergence after {episodes} Episodes in {env.name}', fontsize=12)
    plt.grid(axis='x')
    plt.legend()
    plt.xlabel('Network')
    plt.ylabel('Average total reward')
    plt.tight_layout()
    plt.show()

def planet_test(ann, ant, env, episodes=100):
    mutation_args = {
        'neuron_mutation_rate': 0.5,
        'edge_mutation_rate': 0,
        'weight_mutation_rate': 2e-5
    }
    ant_ = copy.deepcopy(ant)
    ann_ = copy.deepcopy(ann)
    ant_.mutate(**mutation_args)
    ann_.mutate(**mutation_args)

    ann_rewards = discrete_rl.train(ann_, env, episodes=episodes-1, time=200, render=False, plot=False, gif=False)
    ant_rewards = discrete_rl.train(ant_, env, episodes=episodes-1, time=200, render=False, plot=False, gif=False)

    ann_rewards = np.array([ann.metrics['rewards'][-1]] + ann_rewards) - ann.metrics['rewards'][-1]
    ant_rewards = np.array([ant.metrics['rewards'][-1]] + ant_rewards) - ant.metrics['rewards'][-1]

    print('done')

    return ann_rewards, ant_rewards


def carl_planet(tests=256):
    episodes = 100
    ant_earth = ANT(num_neurons=5, edge_probability=10, num_input_neurons=2, num_output_neurons=1)
    ant_earth.load('saved_networks/ant_earth.pkl')

    ann_earth = ANN(layers=[1, 2, 3, 4])
    ann_earth.load('saved_networks/ann_earth.pkl')

    mutation_args = {
        'neuron_mutation_rate': 0.5,
        'edge_mutation_rate': 0,
        'weight_mutation_rate': 1e-5
    }

    mars = LUNARLANDER_MARS

    ann_rewards = np.empty((tests, episodes))
    ant_rewards = np.empty((tests, episodes))

    with multiprocessing.Pool(processes=12) as pool:
        results = pool.starmap(planet_test, [(ann_earth, ant_earth, mars, episodes) for _ in range(tests)])
        for i, (ann, ant) in enumerate(results):
            ann_rewards[i] = ann
            ant_rewards[i] = ant

        mean_ant_rewards = np.mean(ant_rewards, axis=0)
        mean_ann_rewards = np.mean(ann_rewards, axis=0)
        std_ant_rewards = np.std(ant_rewards, axis=0)
        std_ann_rewards = np.std(ann_rewards, axis=0)

        plt.figure(figsize=(8, 6), dpi=150)
        ax = plt.gca()
        plt.fill_between(range(episodes), mean_ant_rewards - std_ant_rewards, mean_ant_rewards + std_ant_rewards, color='#0e68cf', alpha=0.2, edgecolor='none')
        plt.plot(mean_ant_rewards, color='#0e68cf', label='ANT')

        plt.fill_between(range(episodes), mean_ann_rewards - std_ann_rewards, mean_ann_rewards + std_ann_rewards, color='#ff8819', alpha=0.2, edgecolor='none')
        plt.plot(mean_ann_rewards, color='#ff8819', label='ANN')

        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.spines['left'].set_visible(True)
        ax.spines['bottom'].set_visible(True)
        ax.fill_betweenx(ax.get_ylim(), -2, 1, color='grey', alpha=0.5, edgecolor='none')
        plt.title(f'Earth-ANN vs. Earth-ANT on Mars', fontsize=12)
        plt.grid(False)
        plt.legend()
        plt.xlabel('Episode')
        plt.ylabel('Difference in total reward from Earth')
        plt.tight_layout()
        plt.show()


def carl_acrobot(tests=30):
    episodes = 100
    base_env = ACROBOT
    carl_env = CARL_ACROBOT

    mutation_args = {
        'neuron_mutation_rate': 0.5,
        'edge_mutation_rate': 0.05,
        'weight_mutation_rate': 1e-4
    }

    ann_small_base_convergence = np.zeros(tests)
    ant_small_base_convergence = np.zeros(tests)
    ann_large_base_convergence = np.zeros(tests)
    ant_large_base_convergence = np.zeros(tests)
    ann_small_carl_convergence = np.zeros(tests)
    ant_small_carl_convergence = np.zeros(tests)
    ann_large_carl_convergence = np.zeros(tests)
    ant_large_carl_convergence = np.zeros(tests)
    
    for i in range(tests):
        debug_print([f'Starting small test {i}'])
        base_ant = ANT(
            num_neurons         = 40,
            edge_probability    = 2,
            num_input_neurons   = base_env.observation_space,
            num_output_neurons  = base_env.action_space
        )
        base_env.configure_newtork(base_ant)
        base_ant.optimizer = RMSProp(alpha=1e-6, beta=0.99)

        base_ann = ANN([base_env.observation_space, 14, 14, base_env.action_space])
        base_ann.optimizer = RMSProp(alpha=3e-3, beta=0.99)

        ann, _ = genetic.evolve(base_ann, base_env, mutation_args, episodes=episodes//10, mutations=36, graph=False, render=False, save=False, plot=False)
        ant, _ = genetic.evolve(base_ant, base_env, mutation_args, episodes=episodes//10, mutations=36, graph=False, render=False, save=False, plot=False)

        ann_small_base_convergence[i] = np.mean(ann.metrics['rewards'][-10:])
        ant_small_base_convergence[i] = np.mean(ant.metrics['rewards'][-10:])

        base_ant = ANT(
            num_neurons         = 40,
            edge_probability    = 2,
            num_input_neurons   = base_env.observation_space,
            num_output_neurons  = base_env.action_space
        )
        base_env.configure_newtork(base_ant)
        base_ant.optimizer = RMSProp(alpha=1e-6, beta=0.99)

        base_ann = ANN([base_env.observation_space, 14, 14, base_env.action_space])
        base_ann.optimizer = RMSProp(alpha=3e-3, beta=0.99)

        ann, _ = genetic.evolve(base_ann, carl_env, mutation_args, episodes=episodes//10, mutations=36, graph=False, render=False, save=False, plot=False)
        ant, _ = genetic.evolve(base_ant, carl_env, mutation_args, episodes=episodes//10, mutations=36, graph=False, render=False, save=False, plot=False)

        ann_small_carl_convergence[i] = np.mean(ann.metrics['rewards'][-10:])
        ant_small_carl_convergence[i] = np.mean(ant.metrics['rewards'][-10:])

        debug_print([f'Starting large test {i}'])
        base_ant = ANT(
            num_neurons         = 70,
            edge_probability    = 2,
            num_input_neurons   = base_env.observation_space,
            num_output_neurons  = base_env.action_space
        )
        base_env.configure_newtork(base_ant)
        base_ant.optimizer = RMSProp(alpha=1e-6, beta=0.99)

        base_ann = ANN([base_env.observation_space, 30, 30, base_env.action_space])
        base_ann.optimizer = RMSProp(alpha=3e-3, beta=0.99)

        ann, _ = genetic.evolve(base_ann, base_env, mutation_args, episodes=episodes//10, mutations=36, graph=False, render=False, save=False, plot=False)
        ant, _ = genetic.evolve(base_ant, base_env, mutation_args, episodes=episodes//10, mutations=36, graph=False, render=False, save=False, plot=False)

        ann_large_base_convergence[i] = np.mean(ann.metrics['rewards'][-10:])
        ant_large_base_convergence[i] = np.mean(ant.metrics['rewards'][-10:])

        base_ant = ANT(
            num_neurons         = 70,
            edge_probability    = 2,
            num_input_neurons   = base_env.observation_space,
            num_output_neurons  = base_env.action_space
        )
        base_env.configure_newtork(base_ant)
        base_ant.optimizer = RMSProp(alpha=1e-6, beta=0.99)

        base_ann = ANN([base_env.observation_space, 30, 30, base_env.action_space])
        base_ann.optimizer = RMSProp(alpha=3e-3, beta=0.99)

        ann, _ = genetic.evolve(base_ann, carl_env, mutation_args, episodes=episodes//10, mutations=36, graph=False, render=False, save=False, plot=False)
        ant, _ = genetic.evolve(base_ant, carl_env, mutation_args, episodes=episodes//10, mutations=36, graph=False, render=False, save=False, plot=False)

        ann_large_carl_convergence[i] = np.mean(ann.metrics['rewards'][-10:])
        ant_large_carl_convergence[i] = np.mean(ant.metrics['rewards'][-10:])
    
    
    # plt.figure(figsize=(8, 6), dpi=150)


    groups = ['ANN-small', 'ANT-small', 'ANN-large', 'ANT-large']
    average_ann_small_base_convergence = 200 + np.mean(ann_small_base_convergence)
    average_ant_small_base_convergence = 200 + np.mean(ant_small_base_convergence)
    average_ann_large_base_convergence = 200 + np.mean(ann_large_base_convergence)
    average_ant_large_base_convergence = 200 + np.mean(ant_large_base_convergence)

    average_ann_small_carl_convergence = 200 + np.mean(ann_small_carl_convergence)
    average_ant_small_carl_convergence = 200 + np.mean(ant_small_carl_convergence)
    average_ann_large_carl_convergence = 200 + np.mean(ann_large_carl_convergence)
    average_ant_large_carl_convergence = 200 + np.mean(ant_large_carl_convergence)

    std_ann_small_base = np.std(ann_small_base_convergence)
    std_ant_small_base = np.std(ant_small_base_convergence)
    std_ann_large_base = np.std(ann_large_base_convergence)
    std_ant_large_base = np.std(ant_large_base_convergence)

    std_ann_small_carl = np.std(ann_small_carl_convergence)
    std_ant_small_carl = np.std(ant_small_carl_convergence)
    std_ann_large_carl = np.std(ann_large_carl_convergence)
    std_ant_large_carl = np.std(ant_large_carl_convergence)

    base = [average_ann_small_base_convergence, average_ant_small_base_convergence, average_ann_large_base_convergence, average_ant_large_base_convergence]
    carl = [average_ann_small_carl_convergence, average_ant_small_carl_convergence, average_ann_large_carl_convergence, average_ant_large_carl_convergence]
    base_errors = [std_ann_small_base, std_ant_small_base, std_ann_large_base, std_ant_large_base]
    carl_errors = [std_ann_small_carl, std_ant_small_carl, std_ann_large_carl, std_ant_large_carl]

    x = [0, 0.4, 0.8, 1.2]
    width = 0.15

    fig, ax = plt.subplots(layout='constrained')
    bars1 = ax.bar(x, base, width, label='Base Acrobot', color='#6f69e0', yerr=base_errors, capsize=3, error_kw={'ecolor':'#363636', 'capthick':2, 'elinewidth':0.8})
    bars2 = ax.bar([p + width for p in x], carl, width, label='CARL Acrobot', color='#e09169', yerr=carl_errors, capsize=3, error_kw={'ecolor':'#363636', 'capthick':2, 'elinewidth':0.8})
    ax.set_xticks([p + width / 2 for p in x])
    ax.set_xticklabels(groups)
    ax.legend()

    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['left'].set_visible(True)
    ax.spines['bottom'].set_visible(True)
    plt.title(f'Mean Convergence after {episodes} Episodes in Acrobot: Base vs. CARL', fontsize=12)
    plt.grid(axis='x')
    plt.legend()
    plt.xlabel('Network')
    plt.ylabel('Average total reward')
    plt.tight_layout()
    plt.show()




if __name__ == '__main__':
    start_time = time.time()
    # convergence(CARTPOLE, reinforce_online.train, genetic.train, episodes=1000, trials=5)
    # convergence(CARTPOLE, discrete_rl.convergence_train, discrete_rl.convergence_train, episodes=100, trials=5)
    # evolution()
    # average_evolution(iterations=20)
    # parallelization_performance()
    # evolution2()
    # convergence_test_ant(tests=20)
    # carl_planet(tests=256)
    carl_acrobot(tests=12)
    end_time = time.time()

    debug_print([f'Test execution time: {end_time - start_time : 9.2f} s'])