# env_2048_wrapper.py
import gym
from gym import spaces
import numpy as np
from game import Game  # Thay 'your_module' bằng tên tệp hoặc module của bạn chứa lớp Game

class Game2048Wrapper(gym.Env):
    """Custom Environment for 2048 game using Gym interface."""
    metadata = {"render.modes": ["human"]}

    def __init__(self):
        super(Game2048Wrapper, self).__init__()
        self.game = Game()
        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.Box(low=0, high=2**17, shape=(4, 4), dtype=int)  # Sửa np.int thành int

    def step(self, action):
        reward = self.game.do_action(action)
        done = self.game.game_over()
        obs = self.game.state()
        return obs, reward, done, {}

    def reset(self):
        self.game = Game()
        return self.game.state()

    def render(self, mode="human"):
        self.game.print_state()

    def close(self):
        pass
