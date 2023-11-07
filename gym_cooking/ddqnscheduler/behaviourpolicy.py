import rl
import numpy as np

class SoftEpsilonGreedyPolicy(object):
    def __init__(self,n_total_operations, epsilon=1, n_actions=4):
        super(SoftEpsilonGreedyPolicy, self).__init__()
        self.epsilon = epsilon
        self.n_actions = n_actions
        self.n_total_operations = n_total_operations
        self.e_greedy_decrement = 1/pow(self.n_total_operations, 1.8)

    def select_action(self, q_values):
        """
        Select an action based on the Q-values.

        Parameters:
            q_values (np.ndarray): Q-values for each action.

        Returns:
            int: The selected action.
        """
        if np.random.rand() < self.epsilon:
            # Randomly choose an action with epsilon probability
            return np.random.randint(self.n_actions)
        else:
            # Choose the action with the highest Q-value
            self.epsilon = max(0.1, self.epsilon - self.e_greedy_decrement)
            return np.argmax(q_values)
