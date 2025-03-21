import gym
from gym import spaces
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Arrow

class Watermelon(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self):
        super(Watermelon, self).__init__()
        self.action_space = spaces.Discrete(6)
        # self.observation_space = spaces.Tuple((spaces.Box(low=0, high=np.inf, shape=(1,)),
        #                                        spaces.Discrete(2)))
        self.observation_space = np.array([0, 0])
        self.max_distance = 250
        self.random_seed = 42
        # self.reset()

    def set_seed(self, seed):
        self.random_seed = seed

    def generate_obstacles(self):
        np.random.seed(self.random_seed)
        return [(np.random.uniform(8,142, 2), np.random.uniform(2,8)) for _ in range(np.random.randint(20, 80))]

    def generate_targets(self):
        np.random.seed(self.random_seed)
        return [np.random.uniform(8, 142, 2) for _ in range(np.random.randint(3,6))]

    def check_collision(self, position, radius=3):
        return any(np.linalg.norm(center - position) <= obstacle_radius + radius for center, obstacle_radius in self.obstacles)

    def random_position(self):
        position = np.array([20 + 110/2, 20 + 110/2])
        return position
        # np.random.seed(self.random_seed)
        # positions = np.random.uniform(20, 130, size=(100, 2))
        # for i in range(100):
        #     position = positions[i]
        #     if not self.check_collision(position):
        #         return position

    def raycast(self, direction):
        step_size, distance = 1, 0
        current_position = self.position.copy()
        while distance < self.max_distance:
            current_position += direction * step_size
            distance += np.linalg.norm(direction * step_size)
            if self.check_collision(current_position): # Check for obstacles
                return int(distance), 0  # distance to obstacle, target not visible
            for target in self.targets:
                target_distance = np.linalg.norm(target - current_position)
                if target_distance < 3: # check for targets within 3 units
                    return int(target_distance), 1 # distance to target, target visible
        return (250, 0) # max distance, target not visible

    def step(self, action):
        direction = np.array([np.cos(np.radians(self.angle)), np.sin(np.radians(self.angle))])
        reward = 0

        if action == 0: self.angle += 10
        elif action == 1: self.angle -= 10
        elif action in [2, 3, 4, 5]:
            step_length = 1 if action in [2, 3] else 5
            movement = step_length * direction if action in [2, 4] else -step_length * direction
            new_position = self.position + movement
            for i in range(1, step_length + 1):
                intermediate_position = self.position + (i * direction if action in [2, 4] else -i * direction)
                if self.check_collision(intermediate_position):
                    new_position = intermediate_position - (direction if action in [2, 4] else -direction)
                    reward -= 10
                    # print('Agent Collision')
                    break
            self.position = new_position

        distance, visible_target = self.raycast(direction)
        distance_to_nearest_target = min(np.linalg.norm(self.position - target) for target in self.targets)
        reward += 1000 if distance_to_nearest_target < 5 else 0
        return (distance, visible_target), reward, distance_to_nearest_target < 5, {}

    def reset(self):
        # self.random_seed = seed
        self.obstacles = self.generate_obstacles()
        self.targets = self.generate_targets()
        self.position = self.random_position()
        self.angle = 0
        return [self.raycast(np.array([np.cos(np.radians(self.angle)), np.sin(np.radians(self.angle))]))]

    def render(self, observation=None, action=None, mode='human', close=False):
        self.figure, self.axis = plt.subplots()
        self.axis.clear()
        self.axis.set_xlim(0, 150)
        self.axis.set_ylim(0, 150)
        for center, radius in self.obstacles:
            self.axis.add_patch(Circle(center, radius, color='gray'))
        for target in self.targets:
            self.axis.add_patch(Circle(target, 3, color='red'))
        self.axis.add_patch(Circle(self.position, 3, color='blue'))
        self.axis.add_patch(Arrow(self.position[0], self.position[1],
                                  5 * np.cos(np.radians(self.angle)),
                                  5 * np.sin(np.radians(self.angle)),
                                  width=2,
                                  color='black'))
        plt.pause(0.01)

        width, height = self.figure.canvas.get_width_height()
        img = np.frombuffer(self.figure.canvas.tostring_rgb(), dtype=np.uint8)
        img = img.reshape(1280, 960, 3)

        self.close()
        return img

    def close(self): plt.close()

'''
# Testing the environment with random actions
env = ANT_ENV()
observations = env.reset()

for _ in range(200):
    action = env.action_space.sample()
    observations, reward, done, info = env.step(action)
    print(f"Obs: {observations}, Action: {action}, Reward: {reward}")
    env.render(observations, action)
    if done:
        print("Reached Target")
        break

env.close()
'''