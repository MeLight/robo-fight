import numpy as np


class SimpleGameEnv:
    def __init__(self):
        self.state_size = 2
        self.action_size = 2
        self.goal_state = np.array([0.2, 0.4])
        self.min_position = -1
        self.max_position = 1
        self.min_velocity = -0.07
        self.max_velocity = 0.07
        self.position = 0
        self.velocity = 0

    def reset(self):
        self.position = np.random.uniform(-0.6, -0.4)
        self.velocity = 0
        return np.array([self.position, self.velocity])

    def step(self, action):
        # Map action to force
        force = -1.0 if action == 0 else 1.0

        # Update velocity and position
        self.velocity += force * 0.001 - 0.0025 * np.cos(3 * self.position)
        self.velocity = np.clip(self.velocity, self.min_velocity, self.max_velocity)
        self.position += self.velocity
        self.position = np.clip(self.position, self.min_position, self.max_position)

        # Determine if the game is over
        done = bool(self.position >= self.goal_state[0] and self.velocity >= self.goal_state[1])

        # Determine the reward
        reward = -1.0
        if done:
            reward = 0.0
        elif abs(self.position - self.goal_state[0]) < 0.1:
            reward = -0.5

        return np.array([self.position, self.velocity]), reward, done, {}