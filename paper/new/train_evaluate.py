import gym
import numpy as np
from DQN import train_dqn
from AC import ActorCriticAgent
from DDPG import DDPGAgent


def train_evaluate(env_name='MyVehicularEnv-v0', episodes=1000):
    env = gym.make(env_name)

    # DQN
    print("Training DQN...")
    dqn_rewards = train_dqn(env, episodes=episodes, save_path='dqn_model.pth')

    # Actor-Critic
    print("Training Actor-Critic...")
    ac_agent = ActorCriticAgent(state_dim=env.observation_space.shape[0], action_dim=env.action_space.n)

