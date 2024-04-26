import gym
from gym import spaces
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Arrow

class ANT_ENV(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self):
        super(ANT_ENV, self).__init__()
        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.Tuple((spaces.Box(low=0, high=np.inf, shape=(1,)),
                                               spaces.Discrete(2)))
        self.max_distance = 250
        self.figure, self.axis = plt.subplots()
        self.reset()

    def generate_obstacles(self):
        return [(np.random.uniform(8,142, 2), np.random.uniform(2,8)) for _ in range(np.random.randint(20, 80))]

    def generate_targets(self):
        return [self.random_position() for _ in range(np.random.randint(3,6))]

    def check_collision(self, position, radius=3):
        return any(np.linalg.norm(center - position) <= obstacle_radius + radius for center, obstacle_radius in self.obstacles)

    def random_position(self):
        while True:
            position = np.random.uniform(20, 130, 2)
            if not self.check_collision(position):
                return position

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
        elif action == 2 or action == 3:
            new_position = self.position + direction if action == 2 else self.position - direction
            self.position = new_position
            if self.check_collision(new_position):
                reward -= 10
                print('Agent Collision')

        distance, visible_target = self.raycast(direction)
        distance_to_nearest_target = min(np.linalg.norm(self.position - target) for target in self.targets)
        reward += 1000 if distance_to_nearest_target < 5 else 0
        return (distance, visible_target), reward, distance_to_nearest_target < 5, {}

    def reset(self):
        self.obstacles = self.generate_obstacles()
        self.targets = self.generate_targets()
        self.position = self.random_position()
        self.angle = 0
        return self.raycast(np.array([np.cos(np.radians(self.angle)), np.sin(np.radians(self.angle))]))

    def render(self, observation, action, mode='human', close=False):
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

    def close(self): plt.close()

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