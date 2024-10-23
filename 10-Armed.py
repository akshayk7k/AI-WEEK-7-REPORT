import numpy as np

class NonStationaryBandit:
    def __init__(self, n_arms=10):
        self.reward_means = np.zeros(n_arms)

    def pull(self, action):
        self.reward_means += np.random.normal(0, 0.01, len(self.reward_means))
        return max(self.reward_means[action], 0)

class EpsilonGreedyAgent:
    def __init__(self, n_arms=10, epsilon=0.1):
        self.epsilon = epsilon
        self.q_values = np.zeros(n_arms)
        self.action_counts = np.zeros(n_arms)

    def select_action(self):
        if np.random.random() < self.epsilon:
            return np.random.choice(len(self.q_values))
        else:
            return np.argmax(self.q_values)

    def update_q_values(self, action, reward):
        self.action_counts[action] += 1
        self.q_values[action] += (reward - self.q_values[action]) / self.action_counts[action]

bandit = NonStationaryBandit(n_arms=10)
agent = EpsilonGreedyAgent(n_arms=10, epsilon=0.1)

for i in range(10000):
    action = agent.select_action()  
    reward = bandit.pull(action)  
    agent.update_q_values(action, reward)  

print(f"Q-values: {agent.q_values}")
