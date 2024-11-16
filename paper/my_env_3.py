from typing import Optional, Union
import gym
from gym import spaces
import numpy as np
import random

# 仿真参数和自定义模块
Simulation_time = 600  # 总仿真时间（秒）
Simulation_step_time = 1  # 每步时间（秒）

# 通信模型
class Communication_model:
    def __init__(self, *args):
        # 初始化通信模型的参数
        pass

    def run(self):
        # 返回一个模拟的通信时间
        return random.uniform(0, 100)  # 这里随机生成一个通信时间（微秒）

# 工具类
def df_select(dataframe, time, is_Service_Provider):
    # 模拟数据选择操作
    return dataframe

def discount_reward(t, sigma1, sigma2, price):
    sigma1 *= 1000  # 转换为微秒
    sigma2 *= 1000
    if t < sigma1:
        return price
    elif sigma1 <= t <= sigma2:
        return price * (sigma2 - t) / (sigma2 - sigma1)
    else:
        return 0

def calculate_max_communication_time(n):
    return max(Communication_model().run() for _ in range(n))

class MyEnv(gym.Env[np.ndarray, Union[int, np.ndarray]]):
    def __init__(self, render_mode: Optional[str] = None):
        self.price = 100  # 每个任务的奖励，单位：u/task
        self.sigma1 = 10  # 第一阶段的时间阈值，单位：ms
        self.sigma2 = 100  # 第二阶段的时间阈值，单位：ms
        self.tau = Simulation_step_time  # 单个服务提供商的任务卸载时间步，单位：秒
        self.current_time = 0
        self.total_time = Simulation_time

        # 假设初始车辆数量（可以用数据帧模拟）
        self.number_1 = random.randint(1, 10)  # 服务提供商车辆数量
        self.number_2 = random.randint(1, 10)  # 被服务车辆数量

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

        # 统计性能指标
        self.total_energy_consumption = 0.0
        self.total_tasks = 0
        self.successful_tasks = 0
        self.total_completion_time = 0.0

    def step(self, action):
        assert self.state is not None, "调用 step 方法前请先调用 reset 方法。"

        reward = 0
        terminated = False

        self.current_time = self.state[0] + self.tau

        # 模拟车辆数量变化
        self.number_1 = random.randint(1, 10)
        self.number_2 = random.randint(1, 10)

        # 动作空间根据车辆数量动态设置
        self.action_space = spaces.MultiDiscrete([self.number_1] * self.number_2)

        # 确保 task_actions 的长度和 number_2 一致
        if len(action) != self.number_2:
            action = np.zeros(self.number_2, dtype=int)  # 如果长度不一致，则初始化为全零数组

        # 对每个被服务的车辆分别计算
        for i in range(self.number_2):
            vehicle_action = action[i]
            if vehicle_action > 0:
                t = calculate_max_communication_time(vehicle_action)
                r = discount_reward(t, self.sigma1, self.sigma2, self.price)
                reward += r
                self.total_tasks += 1
                if r > 0:
                    self.successful_tasks += 1
                    self.total_completion_time += t / 1000.0  # 转换为毫秒
                self.total_energy_consumption += vehicle_action * t  # 简单假设能耗与通信时间成正比

        if self.current_time >= self.total_time:
            terminated = True
        self.state = np.array([self.current_time, self.number_1, self.number_2], dtype=np.float32)
        self.total_reward += reward  # 累加奖励

        return np.array(self.state, dtype=np.float32), reward, terminated, False, {}

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)
        self.total_reward = 0.0  # 重置总奖励
        self.current_time = 0

        self.number_1 = random.randint(1, 10)
        self.number_2 = random.randint(1, 10)

        self.state = np.array([self.current_time, self.number_1, self.number_2], dtype=np.float32)

        # 动作空间根据有计算需求的车辆数量动态设置
        self.action_space = spaces.MultiDiscrete([self.number_1] * self.number_2)

        return np.array(self.state, dtype=np.float32), {}

    def render(self):
        pass

    def close(self):
        # 输出实验的几个性能指标
        if self.total_tasks > 0:
            avg_completion_time = self.total_completion_time / self.successful_tasks
            task_success_rate = self.successful_tasks / self.total_tasks
        else:
            avg_completion_time = 0.0
            task_success_rate = 0.0
        print(f"Total Reward: {self.total_reward}")
        print(f"Task Success Rate: {task_success_rate}")
        print(f"Average Task Completion Time: {avg_completion_time} ms")
        print(f"Total Energy Consumption: {self.total_energy_consumption} units")

# 示例运行
if __name__ == "__main__":
    env = MyEnv()
    state, _ = env.reset()
    print(f"Initial State: {state}")

    for _ in range(int(Simulation_time / Simulation_step_time)):  # 运行指定的时间步数
        action = env.action_space.sample()  # 随机选择一个动作
        next_state, reward, terminated, truncated, info = env.step(action)
        print(f"Action: {action}, Next State: {next_state}, Reward: {reward}, Terminated: {terminated}, Truncated: {truncated}, Info: {info}")
        if terminated:
            break

    # 输出所有服务提供商车辆的总奖励和性能指标
    env.close()
