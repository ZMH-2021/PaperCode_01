import ray
from ray import tune
from ray.rllib.agents.dqn import DQNTrainer
from ray.rllib.agents.a3c import A3CTrainer

# 初始化Ray
ray.init()

# 选择训练算法，这里以DQN为例
trainer = DQNTrainer(env='GridWorld-v0')

# 开始训练
while True:
    result = trainer.train()
    print(result)

    # 如果达到奖励阈值，停止训练
    if result["episode_reward_max"] >= 1.0:
        break

# 保存模型
checkpoint = trainer.save()

# 关闭Ray
ray.shutdown()