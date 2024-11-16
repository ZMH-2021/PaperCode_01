import gym
from stable_baselines3 import DQN

from paper.my_env_4 import MyVehicularEnv

# 创建并包装环境
env = MyVehicularEnv()

# 定义DQN模型
model = DQN('MlpPolicy', env, verbose=1)

# 训练模型
model.learn(total_timesteps=10000)

# 测试模型
obs = env.reset()
for _ in range(1000):
    action, _ = model.predict(obs)
    obs, reward, done, _ = env.step(action)
    env.render()
    if done:
        obs = env.reset()

env.close()
