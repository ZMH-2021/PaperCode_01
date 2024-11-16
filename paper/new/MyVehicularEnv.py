import gym
import numpy as np
from gym import spaces

class MyVehicularEnv(gym.Env):
    """
    Custom environment for vehicular task offloading based on SMDP model.
    """

    def __init__(self):
        super(MyVehicularEnv, self).__init__()

        # Environment parameters
        self.total_time = 600  # Total simulation time (seconds)
        self.time_step = 1  # Time step (seconds)
        self.current_time = 0

        # Task offloading parameters
        self.sigma1 = 10  # Stage 1 time threshold, in ms
        self.sigma2 = 100  # Stage 2 time threshold, in ms
        self.price = 100  # Reward per task, in units

        # Vehicle numbers
        self.max_vehicles = 10  # Maximum number of vehicles
        self.num_providers = np.random.randint(1, self.max_vehicles)  # Number of service providers
        self.num_consumers = np.random.randint(1, self.max_vehicles)  # Number of consumers

        # Action space: Choose a provider for each consumer or no service
        self.action_space = spaces.MultiDiscrete([self.num_providers + 1] * self.num_consumers)

        # Observation space: Current time, number of providers, number of consumers
        self.observation_space = spaces.Box(
            low=0,
            high=np.array([self.total_time, self.max_vehicles, self.max_vehicles]),
            dtype=np.float32
        )

        # Initialize state and reward
        self.state = None
        self.total_reward = 0.0

    def calculate_communication_time(self, num_providers):
        """
        Calculate communication time based on the number of providers.
        """
        return np.random.uniform(1, 100) / num_providers

    def calculate_reward(self, t):
        """
        Calculate reward based on communication time.
        """
        if t < self.sigma1 * 1000:
            return self.price
        elif self.sigma1 * 1000 <= t <= self.sigma2 * 1000:
            return self.price * (self.sigma2 * 1000 - t) / (self.sigma2 * 1000 - self.sigma1 * 1000)
        else:
            return 0

    def step(self, action):
        """
        Execute one step in the environment.
        """
        reward = 0
        done = False

        self.current_time += self.time_step

        for i in range(self.num_consumers):
            provider_action = action[i]
            if provider_action > 0:  # If a provider is chosen
                t = self.calculate_communication_time(provider_action)
                reward += self.calculate_reward(t)

        # Update state
        self.num_providers = np.random.randint(1, self.max_vehicles)
        self.num_consumers = np.random.randint(1, self.max_vehicles)
        self.state = np.array([self.current_time, self.num_providers, self.num_consumers], dtype=np.float32)

        # Accumulate total reward
        self.total_reward += reward

        if self.current_time >= self.total_time:
            done = True

        return self.state, reward, done, {}

    def reset(self):
        """
        Reset the environment.
        """
        self.current_time = 0
        self.total_reward = 0.0
        self.num_providers = np.random.randint(1, self.max_vehicles)
        self.num_consumers = np.random.randint(1, self.max_vehicles)
        self.state = np.array([self.current_time, self.num_providers, self.num_consumers], dtype=np.float32)
        return self.state

    def render(self, mode='human'):
        """
        Render the environment.
        """
        print(f"Current Time: {self.current_time}, Providers: {self.num_providers}, Consumers: {self.num_consumers}, Total Reward: {self.total_reward}")

    def close(self):
        """
        Close the environment.
        """
        print("Environment closed.")
