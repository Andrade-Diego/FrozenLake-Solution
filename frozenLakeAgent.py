import gym
import numpy as np
import random


# reference:

# each call for step() takes an action and makes an observation
# each observation contains four components
#		observation - environment specific
#		reward (float) - reward from previous action so current state
#		done (boolean) - tells you if terminal state
#		info (dict) - irrelevant really mostly for debugging
# action_space tells you vaild actions
# observation_space teslls you valid observations

def test(env, qTable, numTests):
	cumReward = 0
	#### now test game with that table a few times
	for i in range(numTests):
		state = env.reset()
		done = False
		while not done:
			#env.render()
			newState, reward, done, _ = env.step(np.argmax(qTable[state,:])) 
			state = newState
		cumReward += reward
	avgReward = cumReward / numTests
	print(f"average reward = {avgReward}")

	return avgReward

def learn(env, numEpisodes, alpha, epsilon, algorithm):
	#### declare q-table with zero values
	totEpisodes = numEpisodes
	testingArray = []
	table = np.zeros((env.observation_space.n, env.action_space.n))
	model = np.zeros((env.observation_space.n, env.action_space.n, env.observation_space.n))
	rewards = np.zeros((env.observation_space.n))
	history = []
	uniqueHistory = set()
	epsIncrement = (epsilon - .001) / numEpisodes
	alphaIncrement = (alpha - .008) / numEpisodes
	while not numEpisodes == 0:
		if numEpisodes % 100 == 0:
			testingArray.append ((totEpisodes - numEpisodes, test(env, table, 100)))
		#alpha -= .0001
		#### epsilon decay
		#epsilon -= epsIncrement
		#### reduce number of remaining episodes
		numEpisodes -= 1
		#### copy result of episode to qTable
		if algorithm == dynaQ:	
			table, model, history, uniqueHistory, rewards = algorithm(env, table, .1, epsilon, model, history, uniqueHistory, rewards)	
		else:
			alpha -= alphaIncrement

			table = algorithm(env, table, alpha, epsilon)
		#print(qTable)
	return table, testingArray

def pickAction(epsilon, state, env, table):
	if np.random.uniform() < epsilon:
		action = env.action_space.sample()
	else:
		#### if all table thinks all actions are the same, pick a random one
		maxVal = np.max(table[state, :])
		if np.count_nonzero(table[state, : ] == maxVal) > 1:
			action = env.action_space.sample()
		else:
			action = np.argmax(table[state, :])
	#print(action)
	return action

def dynaQ(env, table, alpha, epsilon, model, history, uniqueHistory, rewards):
	state = env.reset()
	done = False
	steps = 0
	gamma = 0.92

	while not(done or steps >= 200):

		action = pickAction(epsilon, state, env, table)

		newState, reward, done, _ = env.step(action)

		rewards[newState] = reward

		history.append((state, action, newState))
		uniqueHistory.add((state, action, newState))

		if len(history) >= 200:
			history.pop(0)
		
		#### find expected value of this state/action/newState combination from history
		actionInStateCount = 0
		thisResultCount = 0
		for s, a, sPrime in history:
			if s == state and a == action:
				actionInStateCount += 1
				if sPrime == newState:
					thisResultCount += 1
		model[state, action, newState] = thisResultCount / actionInStateCount
		#### update qTable
		if not done:
			table[state, action] += alpha * (reward + gamma * np.max(table[newState,:]) - table[state, action])
		else: 
			table[state, action] += alpha * (reward - table[state, action])
		#### planning!
		for i in range (len(history)//5):
		
			#### pick random state/action that has previously been taken
			past = random.sample(uniqueHistory, 1)
			s = past[0][0]
			a = past[0][1]
			#sPrime = past[0][2]
			#print(f"state = {s}     action = {a}     newState = {sPrime}")

			#### get the expected newState from that state/action combination
			sPrime = np.argmax(model[s, a, :])

			#### update qTable with this state/action/newState combo
			table[s, a] += alpha * (rewards[sPrime] + gamma * np.max(table[sPrime, :]) - table[s, a])
		state = newState
	table[state, action] = alpha * (reward - table[state, action])
	return table, model, history, uniqueHistory, rewards


def qLearning(env, table, alpha, epsilon):
	state = env.reset()
	done = False
	gamma = .92
	steps = 0
	while not(done or steps >= 200):
		#env.render()
		steps += 1
		#### choosing action
		action = pickAction(epsilon, state, env, table)
		if np.random.uniform() < epsilon:
			action = env.action_space.sample()
		else:
			action = np.argmax(table[state,:])
		#### take action, observe outcome
		newState, reward, done, _ = env.step(action)
		#### update
		table[state, action] += alpha * (reward + gamma * np.max(table[newState,:]) - table[state, action])
		state = newState
	#print(table)
	table[state, action] = reward
	return table

def SARSA(env, table, alpha, epsilon):
	state = env.reset()
	done = False
	gamma = .92
	steps = 0
	action = pickAction(epsilon, state, env, table)

	while not(done or steps >= 200):
		
		newState, reward, done, _ = env.step(action)

		newAction = pickAction(epsilon, newState, env, table)
		'''if np.random.uniform() < epsilon:
			newAction = env.action_space.sample()
		else:
			newAction = np.argmax(table[newState,:])'''
		table[state, action] += alpha * (reward + gamma * table[newState, newAction] - table[state, action])
		state = newState
		action = newAction
	return table

if __name__ == "__main__":
	#### creation of 4x4 environment
	env = gym.make("FrozenLake-v0")
	#### number of episodes with which to train
	numEpisodes = 10000
	#### alpha and epsilon declarations
	alpha = .05
	epsilon = .25
	avgCumReward = 0
	totalTests = 1
	qTable, rewardGrowth = learn(env, numEpisodes, alpha, epsilon, qLearning)
	np.savetxt("rewardGrowthQLearning.csv", np.asarray(rewardGrowth), delimiter = ",")
	
	qTable, rewardGrowth = learn(env, numEpisodes, alpha, epsilon, SARSA)
	np.savetxt("rewardGrowthSARSA.csv", np.asarray(rewardGrowth), delimiter = ",")
	qTable, rewardGrowth = learn(env, numEpisodes, alpha, epsilon, dynaQ)
	np.savetxt("rewardGrowthDynaQ.csv", np.asarray(rewardGrowth), delimiter = ",")

	env.close()