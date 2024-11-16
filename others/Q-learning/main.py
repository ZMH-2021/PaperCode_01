# 导入所需库
from Q_learning import QLearningAgent, evaluate_policy
from torch.utils.tensorboard import SummaryWriter
from gymnasium.wrappers import TimeLimit
from datetime import datetime
import gymnasium as gym
import numpy as np
import os, shutil


def main():
    # 是否开启TensorBoard记录训练过程
    write = True
    # 是否加载模型
    Loadmodel = False
    # 设置最大训练步数
    Max_train_steps = 20000
    # 设置随机种子
    seed = 0
    np.random.seed(seed)
    print(f"随机种子: {seed}")

    ''' 创建并配置环境 '''
    # 设置环境名称
    EnvName = "CliffWalking-v0"
    # 创建环境
    env = gym.make(EnvName)
    # 限制每个episode的最大步数
    env = TimeLimit(env, max_episode_steps=500)
    # 创建用于评估的环境
    eval_env = gym.make(EnvName)
    eval_env = TimeLimit(eval_env, max_episode_steps=100)

    ''' 使用TensorBoard记录训练曲线 '''
    if write:
        # 获取当前时间并创建记录路径
        timenow = str(datetime.now())[0:-7]
        timenow = ' ' + timenow[0:13] + '_' + timenow[14:16] + '_' + timenow[-2::]
        writepath = 'runs/{}'.format(EnvName) + timenow
        # 清除旧记录（如果存在）
        if os.path.exists(writepath): shutil.rmtree(writepath)
        # 初始化SummaryWriter实例
        writer = SummaryWriter(log_dir=writepath)

    ''' 创建Q-learning智能体 '''
    # 创建模型存储目录
    if not os.path.exists('model'): os.mkdir('model')
    # 初始化智能体
    agent = QLearningAgent(
        s_dim=env.observation_space.n,
        a_dim=env.action_space.n,
        lr=0.2,
        gamma=0.9,
        exp_noise=0.1)
    # 加载模型（如果需要）
    if Loadmodel: agent.restore()

    ''' 训练循环 '''
    total_steps = 0
    while total_steps < Max_train_steps:
        # 重置环境并获取初始状态
        s, info = env.reset(seed=seed)
        seed += 1
        done, steps = False, 0

        # 在每个episode内进行训练
        while not done:
            steps += 1
            # 选择动作
            a = agent.select_action(s, deterministic=False)
            # 执行动作并获取下一状态、奖励、done标志及额外信息
            s_next, r, dw, tr, info = env.step(a)
            # 更新Q值表
            agent.train(s, a, r, s_next, dw)

            # 判断是否结束当前episode
            done = (dw or tr)
            s = s_next

            # 更新总步数
            total_steps += 1
            # 定期评估并记录结果
            if total_steps % 100 == 0:
                ep_r = evaluate_policy(eval_env, agent)
                if write: writer.add_scalar('ep_r', ep_r, global_step=total_steps)
                print(f'环境名称: {EnvName}, 随机种子: {seed}, 总步数: {total_steps}, 累计奖励: {ep_r}')

            # 定期保存模型
            if total_steps % Max_train_steps == 0:
                agent.save()

    # 关闭环境
    env.close()
    eval_env.close()


if __name__ == '__main__':
    main()
