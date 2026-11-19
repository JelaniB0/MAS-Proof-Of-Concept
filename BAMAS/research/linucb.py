
import numpy as np
import math

class LinUCB:
    def __init__(self, n_arms, d, alpha):
        self.n_arms = n_arms
        self.d = d
        self.alpha = alpha
        self.A = [np.eye(d) for _ in range(n_arms)]
        self.b = [np.zeros(d) for _ in range(n_arms)]

    def predict(self, arm, context):
        theta = np.linalg.solve(self.A[arm], self.b[arm])
        return theta

    def predict_reward(self, arm, context):
        theta = self.predict(arm, context)
        product = np.dot(context, theta)
        return product

    def exploration_bonus(self, arm, context):
        uncertainty = np.dot(np.linalg.solve(self.A[arm], context), context)
        return self.alpha * math.sqrt(uncertainty)

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
