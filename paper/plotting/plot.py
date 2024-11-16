import matplotlib.pyplot as plt
import numpy as np
import random


# 辅助函数：生成指定范围内的随机数据点
def generate_random_data(low, high, count):
    return [random.uniform(low, high) for _ in range(count)]


# 辅助函数：为每种算法生成数据
def generate_data(algorithms, lanes, densities):
    random.seed(42)  # 设置随机种子以保证结果的可重复性
    data = {alg: {} for alg in algorithms}

    for alg in algorithms:
        data[alg]['Average Latency (ms)'] = {lane: generate_random_data(90, 110, len(densities)) for lane in lanes}
        data[alg]['Task Success Rate (%)'] = {density: generate_random_data(80, 100, len(lanes)) for density in
                                              densities}
        data[alg]['System Throughput (tasks/s)'] = {lane: generate_random_data(30, 70, len(densities)) for lane in
                                                    lanes}
        data[alg]['Energy Efficiency (J/task)'] = {density: generate_random_data(0.5, 1.5, len(lanes)) for density in
                                                   densities}

    return data


# 绘制平均时延
def plot_average_latency(data):
    plt.figure(figsize=(10, 6))
    for algorithm, latency_data in data.items():
        lanes = list(latency_data['Average Latency (ms)'].keys())
        latencies = [np.mean(density_data) for density_data in latency_data['Average Latency (ms)'].values()]
        plt.plot(lanes, latencies, label=algorithm, marker='o')
    plt.xlabel('Number of Lanes')
    plt.ylabel('Average Latency (ms)')
    plt.title('Average Latency Comparison')
    plt.legend()
    plt.grid(True)
    plt.show()


# 绘制任务成功率
def plot_task_success_rate(data):
    plt.figure(figsize=(10, 6))
    densities = list(data[algorithms[0]]['Task Success Rate (%)'].keys())
    for algorithm, success_rate_data in data.items():
        success_rates = [np.mean(lane_data) for lane_data in success_rate_data['Task Success Rate (%)'].values()]
        plt.plot(densities, success_rates, label=algorithm, marker='o')
    plt.xlabel('Vehicle Density')
    plt.ylabel('Average Task Success Rate (%)')
    plt.title('Task Success Rate Comparison')
    plt.legend()
    plt.grid(True)
    plt.show()


# 绘制系统吞吐量
def plot_system_throughput(data):
    plt.figure(figsize=(10, 6))
    for algorithm, throughput_data in data.items():
        lanes = list(throughput_data['System Throughput (tasks/s)'].keys())
        throughputs = [np.mean(density_data) for density_data in
                       throughput_data['System Throughput (tasks/s)'].values()]
        plt.plot(lanes, throughputs, label=algorithm, marker='o')
    plt.xlabel('Number of Lanes')
    plt.ylabel('Average System Throughput (tasks/s)')
    plt.title('System Throughput Comparison')
    plt.legend()
    plt.grid(True)
    plt.show()


# 绘制能耗效率
def plot_energy_efficiency(data):
    plt.figure(figsize=(10, 6))
    densities = list(data[algorithms[0]]['Energy Efficiency (J/task)'].keys())
    for algorithm, efficiency_data in data.items():
        efficiencies = [np.mean(lane_data) for lane_data in efficiency_data['Energy Efficiency (J/task)'].values()]
        plt.plot(densities, efficiencies, label=algorithm, marker='o')
    plt.xlabel('Vehicle Density')
    plt.ylabel('Average Energy Efficiency (J/task)')
    plt.title('Energy Efficiency Comparison')
    plt.legend()
    plt.grid(True)
    plt.show()


# 主函数入口
if __name__ == '__main__':
    # 定义算法名称、车道数量和车辆密度范围
    algorithms = ['Actor-Critic', 'DDPG', 'DQN', 'Greedy']
    lanes = [1, 2, 4, 6, 8]  # 车道数量
    densities = [50, 100, 150, 200]  # 车辆密度

    # 生成数据
    data = generate_data(algorithms, lanes, densities)

    # 绘制平均时延图表
    plot_average_latency(data)

    # 绘制任务成功率图表
    plot_task_success_rate(data)

    # 绘制系统吞吐量图表
    plot_system_throughput(data)

    # 绘制能耗效率图表
    plot_energy_efficiency(data)