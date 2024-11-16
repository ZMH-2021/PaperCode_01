import torch.nn.functional as F  # 导入torch.nn.functional模块，用于实现激活函数等操作
import torch.nn as nn  # 导入torch.nn模块，用于构建神经网络层
import argparse  # 导入argparse模块，用于处理命令行参数
import torch  # 导入torch模块，用于构建和训练神经网络

class Actor(nn.Module):  # 定义Actor类，继承自nn.Module
    def __init__(self, state_dim, action_dim, net_width, maxaction):  # 初始化函数，输入状态维度、动作维度、网络宽度和最大动作值
        super(Actor, self).__init__()  # 调用父类的初始化函数

        self.l1 = nn.Linear(state_dim, net_width)  # 定义第一层线性层，输入为状态维度，输出为网络宽度
        self.l2 = nn.Linear(net_width, 300)  # 定义第二层线性层，输入为网络宽度，输出为300
        self.l3 = nn.Linear(300, action_dim)  # 定义第三层线性层，输入为300，输出为动作维度

        self.maxaction = maxaction  # 定义最大动作值

    def forward(self, state):  # 定义前向传播函数，输入为状态
        a = torch.relu(self.l1(state))  # 通过第一层线性层并使用ReLU激活函数
        a = torch.relu(self.l2(a))  # 通过第二层线性层并使用ReLU激活函数
        a = torch.tanh(self.l3(a)) * self.maxaction  # 通过第三层线性层并使用tanh激活函数，然后乘以最大动作值
        return a  # 返回计算得到的动作值

class Q_Critic(nn.Module):  # 定义Q_Critic类，继承自nn.Module
    def __init__(self, state_dim, action_dim, net_width):  # 初始化函数，输入状态维度、动作维度和网络宽度
        super(Q_Critic, self).__init__()  # 调用父类的初始化函数

        self.l1 = nn.Linear(state_dim + action_dim, net_width)  # 定义第一层线性层，输入为状态维度+动作维度，输出为网络宽度
        self.l2 = nn.Linear(net_width, 300)  # 定义第二层线性层，输入为网络宽度，输出为300
        self.l3 = nn.Linear(300, 1)  # 定义第三层线性层，输入为300，输出为1

    def forward(self, state, action):  # 定义前向传播函数，输入为状态和动作
        sa = torch.cat([state, action], 1)  # 将状态和动作拼接在一起
        q = F.relu(self.l1(sa))  # 通过第一层线性层并使用ReLU激活函数
        q = F.relu(self.l2(q))  # 通过第二层线性层并使用ReLU激活函数
        q = self.l3(q)  # 通过第三层线性层
        return q  # 返回计算得到的Q值

def evaluate_policy(env, agent, turns=3):  # 定义评估策略的函数，输入为环境、智能体和回合数
    total_scores = 0  # 初始化总得分
    for j in range(turns):  # 进行指定回合数的游戏
        s, info = env.reset()  # 重置环境并获取初始状态
        done = False  # 初始化游戏结束标志
        while not done:  # 当游戏未结束时
            # Take deterministic actions at test time
            a = agent.select_action(s, deterministic=True)  # 根据当前状态选择确定性动作
            s_next, r, dw, tr, info = env.step(a)  # 执行动作并获取下一个状态、奖励、是否结束等信息
            done = (dw or tr)  # 更新游戏结束标志

            total_scores += r  # 累加奖励
            s = s_next  # 更新当前状态
    return int(total_scores / turns)  # 返回平均得分

# Just ignore this function~
def str2bool(v):  # 定义将字符串转换为布尔值的函数，输入为字符串
    '''transfer str to bool for argparse'''
    if isinstance(v, bool):  # 如果输入已经是布尔值，则直接返回
        return v
    if v.lower() in ('yes', 'True','true','TRUE', 't', 'y', '1'):  # 如果输入表示真，则返回True
        return True
    elif v.lower() in ('no', 'False','false','FALSE', 'f', 'n', '0'):  # 如果输入表示假，则返回False
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')  # 如果输入无法识别，则抛出异常
