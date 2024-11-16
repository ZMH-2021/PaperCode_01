from typing import Optional, Union
import numpy as np
import gym
from gym import spaces
from paper.tool_Class import Simulation
from models.Communication_model import Communication_model

# df_data = Simulation().run()

time = 0.0
time_gap = 0.1


def discount_reward(t, sigma1, sigma2, price):
    if t < sigma1:
        reward = price
    elif sigma1 <= t <= sigma2:
        reward = price * (sigma2 - t) / (sigma2 - sigma1)
    else:
        reward = 0
    return reward


def df_select(dataframe, time, is_Service_Provider):
    return dataframe[(dataframe['time'] == time) & (dataframe['is_Service_Provider'] == is_Service_Provider)]


class MyEnv(gym.Env[np.ndarray, Union[int, np.ndarray]]):
    # metadata = {
    #     "render_modes": ["human", "rgb_array"],
    #     "render_fps": 50,
    # }

    def __init__(self, render_mode: Optional[str] = None):
        self.df_data = Simulation().run()
        self.price = 100  # u/task
        self.sigma1 = 10  # ms
        self.sigma2 = 100  # ms
        self.tau = 0.01  # seconds between state updates
        self.max_latency = 1000  # ms
        self.max_energy_consumption = 1000  # j
        self.max_dependency = None
        self.Maximum_VFC_search_distance = 100
        self.Maximum_number_VFC_vehicles = 5

        self.action_space = spaces.Discrete(4)  # (0,1,2,3)->(2,3,4,5)
        max = np.array(
            [
                self.max_latency*10000,
                self.max_energy_consumption*10000
                # self.max_dependency
            ],
            dtype=np.float32,
        )
        self.observation_space = spaces.Box(0, max, dtype=np.float32)

        self.state = None
        self.render_mode = render_mode

    def step(self, action):
        err_msg = f"{action!r} ({type(action)}) invalid"
        assert self.action_space.contains(action), err_msg
        assert self.state is not None, "Call reset before using step method."

        return_data1 = df_select(self.df_data, time, True)
        return_data2 = df_select(self.df_data, time, False)
        number_1 = return_data1.shape[0]
        number_2 = return_data2.shape[0]

        reward = 0.0
        if number_1 >= 2 and action == 0:  # 分配到2辆车
            t2 = max(Communication_model(100, 32, 64, 9, 16, 34, 27, 32, 5),
                     Communication_model(100, 32, 64, 9, 16, 34, 27, 32, 5))
            reward = discount_reward(t2, self.sigma1, self.sigma2, self.price)
        if number_1 >= 3 and action == 1:  # 分配到3辆车
            t3 = max(Communication_model(100, 32, 64, 9, 16, 34, 27, 32, 5),
                     Communication_model(100, 32, 64, 9, 16, 34, 27, 32, 5),
                     Communication_model(100, 32, 64, 9, 16, 34, 27, 32, 5))
            reward = discount_reward(t3, self.sigma1, self.sigma2, self.price)
        if number_1 >= 4 and action == 2:  # 分配到4辆车
            t4 = max(Communication_model(100, 32, 64, 9, 16, 34, 27, 32, 5),
                     Communication_model(100, 32, 64, 9, 16, 34, 27, 32, 5),
                     Communication_model(100, 32, 64, 9, 16, 34, 27, 32, 5),
                     Communication_model(100, 32, 64, 9, 16, 34, 27, 32, 5))
            reward = discount_reward(t4, self.sigma1, self.sigma2, self.price)
        if number_1 >= 5 and action == 3:  # 分配到5辆车
            t5 = max(Communication_model(100, 32, 64, 9, 16, 34, 27, 32, 5),
                     Communication_model(100, 32, 64, 9, 16, 34, 27, 32, 5),
                     Communication_model(100, 32, 64, 9, 16, 34, 27, 32, 5),
                     Communication_model(100, 32, 64, 9, 16, 34, 27, 32, 5),
                     Communication_model(100, 32, 64, 9, 16, 34, 27, 32, 5))
            reward = discount_reward(t5, self.sigma1, self.sigma2, self.price)

        terminated = bool(
            return_data1.empty or return_data2.empty
        )

        return np.array(self.state, dtype=np.float32), reward, terminated, False, {}

    def reset(self,time_gap=time_gap, time=time):
        time += time_gap
        return np.array(self.state, dtype=np.float32), {}

    def render(self):
        pass

    def close(self):
        pass
