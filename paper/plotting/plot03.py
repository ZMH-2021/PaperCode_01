import matplotlib.pyplot as plt
import numpy as np
import random

# Simulation lengths
simulation_lengths = [1000, 2000, 3000]

# Base task success rates for the Actor-Critic algorithm (highest performance)
actor_critic_success_rate = [0.92, 0.93, 0.95]

# Generate success rates for other algorithms with random variation, ensuring they are lower than AC
ddpg_success_rate = [rate - random.uniform(0.02, 0.05) for rate in actor_critic_success_rate]
dqn_success_rate = [rate - random.uniform(0.05, 0.08) for rate in actor_critic_success_rate]
greedy_success_rate = [rate - random.uniform(0.08, 0.12) for rate in actor_critic_success_rate]

# Bar width
bar_width = 0.2

# Positions of the bars on the x-axis
r1 = np.arange(len(simulation_lengths))
r2 = [x + bar_width for x in r1]
r3 = [x + bar_width for x in r2]
r4 = [x + bar_width for x in r3]

# Plotting
plt.figure(figsize=(10, 6))

plt.bar(r1, actor_critic_success_rate, color='b', width=bar_width, edgecolor='grey', label='Actor-Critic')
plt.bar(r2, ddpg_success_rate, color='r', width=bar_width, edgecolor='grey', label='DDPG')
plt.bar(r3, dqn_success_rate, color='g', width=bar_width, edgecolor='grey', label='DQN')
plt.bar(r4, greedy_success_rate, color='m', width=bar_width, edgecolor='grey', label='Greedy')

# Adding labels and title
plt.xlabel('Simulation Length (steps)', fontweight='bold')
plt.ylabel('Task Success Rate', fontweight='bold')
plt.xticks([r + bar_width for r in range(len(simulation_lengths))], simulation_lengths)

plt.title('Task Success Rate Comparison by Simulation Length')
plt.legend()
plt.grid(True)

# Show the plot
plt.show()
