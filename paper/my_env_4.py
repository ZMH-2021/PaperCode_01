import gym
import numpy as np
from gym import spaces


class MyVehicularEnv(gym.Env):
    """
    自定义的强化学习环境，基于论文中的SMDP模型。
    """

    def __init__(self):
        super(MyVehicularEnv, self).__init__()

        # 环境参数
        self.total_time = 600  # 总仿真时间（秒）
        self.time_step = 1  # 每步时间（秒）
        self.current_time = 0

        # 任务卸载相关参数
        self.sigma1 = 10  # 第一阶段的时间阈值，单位：ms
        self.sigma2 = 100  # 第二阶段的时间阈值，单位：ms
        self.price = 100  # 每个任务的奖励，单位：u/task

        # 动态车辆数量
        self.max_vehicles = 10  # 假设最大车辆数量
        self.num_providers = np.random.randint(1, self.max_vehicles)  # 服务提供商车辆数量
        self.num_consumers = np.random.randint(1, self.max_vehicles)  # 被服务车辆数量

        # 动作空间：为每个被服务车辆选择一个服务提供商（或不提供服务）
        self.action_space = spaces.MultiDiscrete([self.num_providers + 1] * self.num_consumers)

        # 观测空间：当前时间、服务提供商数量、被服务车辆数量
        self.observation_space = spaces.Box(
            low=0,
            high=np.array([self.total_time, self.max_vehicles, self.max_vehicles]),
            dtype=np.float32
        )

        # 初始化状态和奖励
        self.state = None
        self.total_reward = 0.0

    def calculate_communication_time(self, num_providers):
        """
        计算最大通信时间，根据服务提供商数量模拟通信延迟。
        """
        # 这里简单模拟一个通信时间，实际可以根据更复杂的模型计算
        return np.random.uniform(1, 100) / num_providers

    def calculate_reward(self, t):
        """
        根据通信时间计算奖励。
        """
        if t < self.sigma1 * 1000:  # 转换为微秒
            return self.price
        elif self.sigma1 * 1000 <= t <= self.sigma2 * 1000:
            return self.price * (self.sigma2 * 1000 - t) / (self.sigma2 * 1000 - self.sigma1 * 1000)
        else:
            return 0

    def step(self, action):
        """
        执行一步，返回新的状态、奖励、是否结束等信息。
        """
        # 初始化奖励和是否结束标志
        reward = 0
        done = False

        # 更新当前时间
        self.current_time += self.time_step

        # 遍历每个被服务的车辆
        for i in range(self.num_consumers):
            provider_action = action[i]
            if provider_action > 0:  # 如果选择了服务提供商
                t = self.calculate_communication_time(provider_action)
                reward += self.calculate_reward(t)

        # 更新状态
        self.num_providers = np.random.randint(1, self.max_vehicles)
        self.num_consumers = np.random.randint(1, self.max_vehicles)
        self.state = np.array([self.current_time, self.num_providers, self.num_consumers], dtype=np.float32)

        # 累积总奖励
        self.total_reward += reward

        # 判断是否达到仿真时间上限
        if self.current_time >= self.total_time:
            done = True

        return self.state, reward, done, {}

    def reset(self):
        """
        重置环境状态。
        """
        self.current_time = 0
        self.total_reward = 0.0
        self.num_providers = np.random.randint(1, self.max_vehicles)
        self.num_consumers = np.random.randint(1, self.max_vehicles)
        self.state = np.array([self.current_time, self.num_providers, self.num_consumers], dtype=np.float32)
        return self.state

    def render(self, mode='human'):
        """
        可视化环境状态。
        """
        print(
            f"Current Time: {self.current_time}, Providers: {self.num_providers}, Consumers: {self.num_consumers}, Total Reward: {self.total_reward}")

    def close(self):
        """
        关闭环境，释放资源。
        """
        print("Environment closed.")
