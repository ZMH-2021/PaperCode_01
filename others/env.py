import gym
from gym import spaces
import numpy as np


class GridWorldEnv(gym.Env):
    metadata = {'render.modes': ['human', 'ansi']}

    def __init__(self):
        self.action_space = spaces.Discrete(4)  # 动作空间：上、下、左、右
        self.observation_space = spaces.Box(low=0, high=4, shape=(6, 6), dtype=np.int)  # 状态空间：6x6的网格
        self._seed()  # 用于生成随机数
        self._reset()

    def _seed(self, seed=None):
        self.np_random, seed = gym.utils.seeding.np_random(seed)
        return [seed]

    def _reset(self):
        # 初始化智能体位置
        self.player_pos = np.array([1, 1])
        # 初始化苹果位置
        self.apple_pos = np.array([4, 4])
        # 初始化环境状态
        self.state = self._create_grid()
        return self.state

    def _create_grid(self):
        grid = np.zeros((6, 6), dtype=int)
        grid[1, 1] = 1  # 智能体
        grid[4, 4] = 2  # 苹果
        grid[2, 2] = 3  # 终点
        return grid

    def _step(self, action):
        # 根据动作更新智能体位置
        self.player_pos += self._get_action_delta(action)

        # 确保智能体不会走出网格边界
        self.player_pos = np.clip(self.player_pos, 0, 5)

        # 检查是否到达终点
        done = self._is_goal_reached()

        # 检查是否吃到苹果
        reward = self._get_apple_reward()

        # 更新状态
        self.state = self._create_grid()

        return self.state, reward, done, {}

    def _render(self, mode='human', close=False):
        # 可视化当前环境状态
        pass

    def _get_action_delta(self, action):
        # 根据动作返回智能体位置的增量
        delta = np.array([0, 0])
        if action == 0:  # 上
            delta[0] = -1
        elif action == 1:  # 下
            delta[0] = 1
        elif action == 2:  # 左
            delta[1] = -1
        elif action == 3:  # 右
            delta[1] = 1
        return delta

    def _is_goal_reached(self):
        # 检查智能体是否到达终点
        return self.player_pos.all() == self.apple_pos.all()

    def _get_apple_reward(self):
        # 检查智能体是否吃到苹果
        if self.player_pos.all() == self.apple_pos.all():
            return 1.0
        else:
            return -0.1


# 注册自定义环境
gym.register(
    id='GridWorld-v0',
    entry_point='path.to.module:GridWorldEnv',  # 指定模块和类名
    max_episode_steps=100,
    reward_threshold=1.0
)