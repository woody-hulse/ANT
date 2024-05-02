import discrete_rl
import genetic
import reinforce_online
from environments import *

from tqdm import tqdm

def test_train(env, size, learning_rate, episodes):
    return np.random.normal(loc=learning_rate, scale=size/5, size=(episodes,))

def convergence_experiment(env, ann_train, ant_train2, episodes=1000, trials=5):
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



if __name__ == '__main__':
    # convergence_experiment(CARTPOLE, reinforce_online.train, genetic.train, episodes=1000, trials=5)
    convergence_experiment(CARTPOLE, discrete_rl.convergence_train, discrete_rl.convergence_train, episodes=100, trials=5)