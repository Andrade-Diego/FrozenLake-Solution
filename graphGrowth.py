import matplotlib.pyplot as plt
import numpy as np

qLearning = np.loadtxt("rewardGrowthQLearning.csv", delimiter = ",")
SARSA = np.loadtxt("rewardGrowthSARSA.csv", delimiter = ",")
dynaQ = np.loadtxt("rewardGrowthDynaQ.csv", delimiter = ",")
print(qLearning)

plt.plot(qLearning[:,0], qLearning[:,1], label = "qLearning", color = '#0A284B')#, s = .2)
plt.plot(SARSA[:,0], SARSA[:,1], label = "SARSA", color = '#235FA4')#, s = .2)
plt.plot(dynaQ[:, 0], dynaQ[:, 1], label = "dynaQ", color = '#A691AE')#, s = .2)
plt.title("Reward Growth over 10000 Episodes")
plt.xlabel("Number of Episodes")
plt.ylabel("Avg Reward Over 100 Tests")
plt.legend()
plt.show()