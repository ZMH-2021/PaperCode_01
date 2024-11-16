import numpy as np

# 定义Q-learning智能体类
class QLearningAgent():
    def __init__(self, s_dim, a_dim, lr=0.01, gamma=0.9, exp_noise=0.1):
        """
        初始化Q-learning智能体，设置状态维度、动作维度、学习率、折扣因子γ和探索噪声ε
        """
        self.a_dim = a_dim
        self.lr = lr
        self.gamma = gamma
        self.epsilon = exp_noise
        self.Q = np.zeros((s_dim, a_dim))  # 初始化Q表为全零矩阵

    def select_action(self, s, deterministic):
        """
        根据策略选择动作
        :param s: 当前状态
        :param deterministic: 是否采用确定性策略（默认False，即ε-greedy策略）
        :return: 动作
        """
        if deterministic:
            # 确定性策略，选取当前状态下Q值最大的动作
            return np.argmax(self.Q[s, :])
        else:
            # ε-greedy策略，有一定概率随机探索
            if np.random.uniform(0, 1) < self.epsilon:
                return np.random.choice(self.a_dim)  # 随机选取动作
            else:
                return np.argmax(self.Q[s, :])

    def train(self, s, a, r, s_next, dw):
        """
        更新Q表
        :param s: 上一状态
        :param a: 上一步骤的动作
        :param r: 上一步骤的即时奖励
        :param s_next: 下一状态
        :param dw: 是否结束（done标志位）
        """
        Q_sa = self.Q[s, a]
        target_Q = r + (1 - dw) * self.gamma * np.max(self.Q[s_next, :])  # 计算目标Q值
        self.Q[s, a] += self.lr * (target_Q - Q_sa)  # 更新Q表

    def save(self):
        """
        保存Q表到.npy文件
        """
        npy_file = 'model/q_table.npy'
        np.save(npy_file, self.Q)
        print(npy_file + ' 已保存.')

    def restore(self, npy_file='model/q_table.npy'):
        """
        从.npy文件加载Q表
        """
        self.Q = np.load(npy_file)
        print(npy_file + ' 已加载.')

def evaluate_policy(env, agent):
    """
    评估策略性能
    :param env: 环境对象
    :param agent: Q-learning智能体对象
    :return: 评估期间获得的平均累积奖励
    """
    s, info = env.reset()
    done, ep_r, steps = False, 0, 0
    while not done:
        a = agent.select_action(s, deterministic=True)
        s_next, r, dw, tr, info = env.step(a)
        done = (dw or tr)

        ep_r += r
        steps += 1
        s = s_next
    return ep_r