import matplotlib.pyplot as plt
import numpy as np

def plot_results():
    # Example data; replace with your actual results
    algorithms = ['Actor-Critic', 'DQN', 'DDPG']
    task_success_rates = [0.85, 0.78, 0.80]
    avg_completion_times = [30, 35, 32]
    energy_consumptions = [50, 55, 52]

    x = np.arange(len(algorithms))

    fig, ax = plt.subplots(3, 1, figsize=(10, 15))

    ax[0].bar(x, task_success_rates, align='center')
    ax[0].set_xticks(x)
    ax[0].set_xticklabels(algorithms)
    ax[0].set_title('Task Success Rate')
    ax[0].set_ylabel('Success Rate')

    ax[1].bar(x, avg_completion_times, align='center')
    ax[1].set_xticks(x)
    ax[1].set_xticklabels(algorithms)
    ax[1].set_title('Average Task Completion Time')
    ax[1].set_ylabel('Completion Time (s)')

    ax[2].bar(x, energy_consumptions, align='center')
    ax[2].set_xticks(x)
    ax[2].set_xticklabels(algorithms)
    ax[2].set_title('Energy Consumption')
    ax[2].set_ylabel('Energy (units)')

    plt.tight_layout()
    plt.savefig('results.png')
    plt.show()

if __name__ == "__main__":
    plot_results()

