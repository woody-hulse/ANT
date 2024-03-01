import gym
# import tensorflow as tf
import copy
import time
import cv2
from PIL import Image
import IPython as ip

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

def continuous_rl_train(network, env, episodes=1000, time=10000):
    debug_print([f'Training {network.name}'])

    loss_layer = RL()
    for episode in range(episodes):
        state = env.reset()[0]
        total_reward = 0

        pbar = tqdm(range(time))
        p_reward = 0
        p_action = np.zeros(network.num_output_neurons)

        for t in pbar:
            action = network.forward_pass(state)
            state, reward, done, info, _ = env.step(action)
            total_reward += reward

            if done:
                break

            y_hat = arr([reward for _ in range(network.num_output_neurons)])
            '''
            y_hat = np.zeros(network.num_output_neurons)
            for i, a in enumerate(action):
                action_a = p_action.copy()
                action_a[i] = a

                env_copy = copy.deepcopy(env)
                _, reward, _, _, _ = env_copy.step(action_a)
                y_hat[i] = reward
            '''

            # Compute the expected Q values
            # next_state = network.forward_pass(state)
            # expected_state_action_values = (next_state * 0.99) + reward

            # print(expected_state_action_values.shape)

            # Compute Huber loss
            # loss = loss_layer(state, expected_state_action_values)
            # print(p_reward, y_hat)
            network.backward_pass(
                loss_layer, 
                y_hat * 0.99, 
                arr([p_reward for _ in range(network.num_output_neurons)]),
                decay=0.0, 
                update_metrics=True
            )

            p_reward = reward

            pbar.set_description(f't={t+1:05}; reward={round(reward, 3)}; action={round(np.sum(action), 3)}; ' + network.get_metrics_string(metrics=['loss', 'energy', 'grad_energy', 'prop_input', 'prop_grad']))


def continuous_rl_train_2(network, env, episodes=1000, time=1000, render=False):
    gamma = 0.99  # Discount factor for future rewards

    loss_fn = MSE()

    for episode in range(episodes):
        state = env.reset()[0]

        image_frames = []
        G = 0
        total_reward = 0
        pbar = tqdm(range(time))
        for t in pbar:
            if render:
                image_array = env.render()
                image_frame = Image.fromarray(image_array)
                image_frames.append(image_frame)
            action = network.forward_pass(state)
            next_state, reward, done, _, _ = env.step(action)

            G = reward + gamma * G
            network.backward_pass(loss_fn, action, G)

            max_chars = 160
            total_reward += reward
            pbar_string = f'Episode {episode : 05}: reward={total_reward : 9.3f}; action={np.sum(action) : 7.3f}; ' + \
                network.get_metrics_string(metrics=['loss', 'energy', 'grad_energy', 'prop_input', 'prop_grad'])
            
            if len(pbar_string) > max_chars: pbar_string = pbar_string[:max_chars - 3] + '...'

            pbar.set_description(pbar_string)

            state = next_state

            if done: break

        if render:
            image_frames[0].save('rl_results/episode' + str(episode) + '.gif', 
                            save_all = True, 
                            duration = 20,
                            loop = 0,
                            append_images = image_frames[1:])



def main():
    # 'Pendulum-v1'
    # 'MountainCarContinuous-v0'
    # 'BipedalWalker-v3'
    # 'Ant-v2'
    env = gym.make('Pendulum-v1', render_mode='rgb_array')

    network = ContinuousNetwork(
    num_neurons         = 64,
    edge_probability    = 1.6,
    num_input_neurons   = env.observation_space.shape[0],
    num_output_neurons  = env.action_space.shape[0]
    )
    for neuron in network.output_neurons:
        neuron.activation = Sigmoid2()
    # visualize_network(network)
    # visualize_network(network, show_weight=True)

    network.print_info()
    rmsprop = RMSProp(alpha=1e-5, beta=0.99, reg=0)
    network.set_optimizer(rmsprop)

    continuous_rl_train_2(network, env, episodes=200, render=True)
    # watch_rl_episode(network, env)
    visualize_episode(env, network, name='prelim_rl_results')


if __name__ == '__main__':
    main()