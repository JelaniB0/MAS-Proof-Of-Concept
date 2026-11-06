import numpy as np
import math
#constructor - parameter for num of arms and the difficulty prediction as they are given
class LinUCB:
    def __init__(self, n_arms, d,alpha):
        self.n_arms = n_arms
        self.d=d
        self.alpha = alpha
        self.A = [np.eye(d) for _ in range(n_arms)]
        self.b = [np.zeros(d) for _ in range(n_arms)]

#testing
#bandit = LinUCB(n_arms=3, d=2)
#print(bandit.A)
#print(bandit.b)
#expected reward formula => theta = matrix inverse (vector)
    def predict(self, arm, context):
        theta = np.linalg.solve(self.A[arm], self.b[arm])
        return theta
#predicted reward for a particular task => reward = difficulty*context dot product 
    def predict_reward(self,arm, context):
        theta = self.predict(arm,context)
        product = np.dot(context, theta)           
        return product
    def exploration_bonus(self, arm, context):
        uncertainty = np.dot(np.linalg.solve(self.A[arm], context), context)
        return self.alpha * math.sqrt(uncertainty)

#testing
#bandit = LinUCB(n_arms=3, d=2, alpha=2.0)
#context = np.array([1.0, 0.7])

#print("Theta:", bandit.predict(0, context))
#print("Predicted reward:", bandit.predict_reward(0, context))
#print("Exploration bonus:", bandit.exploration_bonus(0, context))
    def calculate_ucb(self, arm, context):
        predicted_reward = self.predict_reward(arm, context)
        bonus = self.exploration_bonus(arm, context)
        return predicted_reward + bonus
    def select_arm(self, context):
        scores = []
        for arm in range(self.n_arms):
            score = self.calculate_ucb(arm, context)
            scores.append(score)
        return np.argmax(scores)
    def update(self, arm, context, reward):
            self.A[arm] += np.outer(context, context)
            self.b[arm] += reward * context

bandit = LinUCB(n_arms=3, d=2, alpha=1.0)
context = np.array([1.0, 0.7])

selected_arm = bandit.select_arm(context)
print("selected pool:", selected_arm)

bandit.update(selected_arm, context, 850)

print("updated A:", bandit.A[selected_arm])
print("updated b:", bandit.b[selected_arm])
print("updated theta:", bandit.predict(selected_arm, context))

print("New UCB scores:")

for arm in range(bandit.n_arms):
    print(f"Pool {arm + 1}: {bandit.calculate_ucb(arm, context)}")

print("New selected pool:", bandit.select_arm(context))
#deciding which agent gets sophisticated task
