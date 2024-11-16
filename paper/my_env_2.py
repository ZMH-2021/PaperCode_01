from typing import Optional, Union
import gym
from gym import spaces
import numpy as np
from models.Communication_model import Communication_model
from tool_Class import *
from Hyperparameters import *

# 仿真参数
df = Simulation().run()
df = df.sort_values(by='time').reset_index(drop=True)


# 计算奖励函数
def discount_reward(t, sigma1, sigma2, price):
    # sigma1, sigma2 是毫秒
    # t 是微秒
    sigma1 *= 1000
    sigma2 *= 1000
    if t < sigma1:
        return price
    elif sigma1 <= t <= sigma2:
        return price * (sigma2 - t) / (sigma2 - sigma1)
    else:
        return 0


def df_select(dataframe, time, is_Service_Provider):
    return dataframe[(dataframe['time'] == round(time, 1)) & (dataframe['is_Service_Provider'] == is_Service_Provider)]


# 封装计算最大通信时间的函数
def calculate_max_communication_time(n):
    return max(Communication_model(100, 32, 64, 9, 16, 34, 27, 32, 5) for _ in range(n))


# 自定义环境类
class MyEnv(gym.Env[np.ndarray, Union[int, np.ndarray]]):
    def __init__(self, render_mode: Optional[str] = None):
        self.price = 100  # 每个任务的奖励，单位：u/task
        self.sigma1 = 10  # 第一阶段的时间阈值，单位：ms
        self.sigma2 = 100  # 第二阶段的时间阈值，单位：ms
        self.tau = Simulation_step_time  # 单个服务提供商的任务卸载时间步，单位：秒
        self.current_time = 0
        self.total_time = Simulation_time

        self.number_1 = len(df_select(df, self.current_time, True)) + 1
        self.number_2 = len(df_select(df, self.current_time, False)) + 1

        # 动作空间设计
        self.action_space = spaces.MultiDiscrete([self.number_1] * self.number_2)

        # 观测空间设计
        self.observation_space = spaces.Box(
            low=0,
            high=np.array([self.total_time, self.number_1, self.number_2], dtype=np.float32),
            dtype=np.float32
        )

        self.state = None
        self.render_mode = render_mode
        self.total_reward = 0.0  # 初始化总奖励

    def step(self, action):
        assert self.state is not None, "调用 step 方法前请先调用 reset 方法。"

        reward = 0
        terminated = False

        self.current_time = self.state[0] + self.tau

        self.number_1 = len(df_select(df, self.current_time, True)) + 1
        self.number_2 = len(df_select(df, self.current_time, False)) + 1

        # 动作空间根据有计算需求的车辆数量动态设置
        self.action_space = spaces.MultiDiscrete([self.number_1] * self.number_2)

        # 确保 task_actions 的长度和 number_2 一致
        if len(action) != self.number_2:
            action = np.zeros(self.number_2, dtype=int)  # 如果长度不一致，则初始化为全零数组

        # 对每个被服务的车辆分别计算
        for i in range(self.number_2):
            vehicle_action = action[i]
            if vehicle_action == 0:
                pass
            if vehicle_action > 0:  # 分配到 vehicle_action 辆车
                t = calculate_max_communication_time(vehicle_action)
                r = discount_reward(t, self.sigma1, self.sigma2, self.price)
                reward += r
                # print(f"vehicle_action: {vehicle_action}, t: {t}, reward: {r}")

        if self.current_time >= self.total_time:
            terminated = True
        self.state = np.array([self.current_time, self.number_1, self.number_2], dtype=np.float32)
        self.total_reward += reward  # 累加奖励

        return np.array(self.state, dtype=np.float32), reward, terminated, False, {}

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)
        self.total_reward = 0.0  # 重置总奖励
        self.current_time = 0

        self.number_1 = len(df_select(df, self.current_time, True)) + 1
        self.number_2 = len(df_select(df, self.current_time, False)) + 1

        self.state = np.array([self.current_time, self.number_1, self.number_2], dtype=np.float32)

        # 动作空间根据有计算需求的车辆数量动态设置
        self.action_space = spaces.MultiDiscrete([self.number_1] * self.number_2)

        return np.array(self.state, dtype=np.float32), {}

    def render(self):
        pass

    def close(self):
        pass


# 示例运行
if __name__ == "__main__":
    env = MyEnv()
    state, _ = env.reset()
    print(f"Initial State: {state}")

    for _ in range(int(Simulation_time / Simulation_step_time)):  # 运行指定的时间步数
        action = env.action_space.sample()  # 随机选择一个动作
        next_state, reward, terminated, truncated, info = env.step(action)
        print(
            f"Action: {action}, Next State: {next_state}, Reward: {reward}, Terminated: {terminated}, Truncated: {truncated}, Info: {info}")
        if terminated:
            break

    # 输出所有服务提供商车辆的总奖励
    print(f"Total Reward: {env.total_reward}")
