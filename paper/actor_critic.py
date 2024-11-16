import gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
from my_env_3 import MyEnv  # 导入你的自定义环境

# Hyperparameters
learning_rate = 0.0002
gamma = 0.98
n_rollout = 10
entropy_coef = 0.01  # 熵正则化系数
reward_scale = 100.0  # 奖励归一化系数
target_update_interval = 10  # 目标网络更新间隔
learning_rate_decay = 0.999  # 学习率衰减
buffer_size = 1000  # 经验回放缓冲区大小
batch_size = 32  # 训练批次大小

class ActorCritic(nn.Module):
    def __init__(self, input_dim, action_dims):
        super(ActorCritic, self).__init__()
        self.data = []

        self.fc1 = nn.Linear(input_dim, 256)  # 输入层，调整以匹配观察形状
        self.fc_pi = nn.ModuleList([nn.Linear(256, dim) for dim in action_dims])
        self.fc_v = nn.Linear(256, 1)

        # 引入目标网络
        self.target_v = nn.Linear(256, 1)
        self.update_target()

        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)
        self.scheduler = torch.optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=learning_rate_decay)

    def pi(self, x, softmax_dim=0):
        x = F.relu(self.fc1(x))
        probs = [F.softmax(fc(x), dim=softmax_dim) for fc in self.fc_pi]
        return probs

    def v(self, x):
        x = F.relu(self.fc1(x))
        v = self.fc_v(x)
        return v

    def target_v(self, x):
        x = F.relu(self.fc1(x))
        v = self.target_v(x)
        return v

    def update_target(self):
        self.target_v.load_state_dict(self.fc_v.state_dict())

    def put_data(self, transition):
        self.data.append(transition)
        if len(self.data) > buffer_size:
            self.data.pop(0)

    def make_batch(self):
        batch = random.sample(self.data, batch_size)
        s_lst, a_lst, r_lst, s_prime_lst, done_lst = [], [], [], [], []
        for transition in batch:
            s, a, r, s_prime, done = transition
            s_lst.append(s)
            a_lst.append(a)  # 调整以添加整个动作
            r_lst.append([r / reward_scale])  # 奖励归一化
            s_prime_lst.append(s_prime)
            done_mask = 0.0 if done else 1.0
            done_lst.append([done_mask])

        s_batch, a_batch, r_batch, s_prime_batch, done_batch = torch.tensor(s_lst, dtype=torch.float), torch.tensor(a_lst, dtype=torch.long), torch.tensor(r_lst, dtype=torch.float), torch.tensor(s_prime_lst, dtype=torch.float), torch.tensor(done_lst, dtype=torch.float)
        return s_batch, a_batch, r_batch, s_prime_batch, done_batch

    def train_net(self):
        if len(self.data) < batch_size:
            return  # 当数据不足时不进行训练

        s, a, r, s_prime, done = self.make_batch()
        td_target = r + gamma * self.target_v(s_prime) * done
        delta = td_target - self.v(s)

        pi = self.pi(s, softmax_dim=1)
        pi_a = [pi[i].gather(1, a[:, i].unsqueeze(1)).squeeze(1) for i in range(len(pi))]
        pi_a = torch.stack(pi_a, dim=1)

        # 计算熵以进行熵正则化
        entropy = -torch.sum(pi * torch.log(pi + 1e-10), dim=1).mean()

        # 增加熵正则化的损失项
        loss = -torch.sum(torch.log(pi_a) * delta.detach(), dim=1) + F.smooth_l1_loss(self.v(s), td_target.detach()) - entropy_coef * entropy

        self.optimizer.zero_grad()
        loss.mean().backward()
        self.optimizer.step()
        self.scheduler.step()  # 学习率衰减

    def train_target_net(self, n_epi):
        if n_epi % target_update_interval == 0:
            self.update_target()

def main():
    env = MyEnv()
    input_dim = env.observation_space.shape[0]
    action_dims = env.action_space.nvec
    model = ActorCritic(input_dim, action_dims)
    print_interval = 20
    score = 0.0

    for n_epi in range(10000):
        done = False
        s, _ = env.reset()
        while not done:
            for t in range(n_rollout):
                prob = model.pi(torch.from_numpy(s).float())
                a = [Categorical(prob[i]).sample().item() for i in range(len(prob))]  # 生成多离散动作列表
                s_prime, r, done, truncated, info = env.step(a)
                model.put_data((s, a, r, s_prime, done))

                s = s_prime
                score += r

                if done:
                    break

            model.train_net()

        # 每隔一段时间更新目标网络
        model.train_target_net(n_epi)

        if n_epi % print_interval == 0 and n_epi != 0:
            print("# of episode :{}, avg score : {:.1f}".format(n_epi, score / print_interval))
            score = 0.0
    env.close()

if __name__ == '__main__':
    main()
