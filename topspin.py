import numpy as np


class TopSpinState:
    def __init__(self, state, k=4):
        self.state = state
        self.k = k
        self.n = len(state)

    def is_goal(self):
        # Check if the state is in ascending order
        return self.state == list(range(1, self.n + 1))

    def get_state_as_list(self):
        return list(self.state)

    def get_neighbors(self):
        neighbors = []
        neighbors.append((list(np.roll(self.state, 1)), 1))
        neighbors.append((list(np.roll(self.state, -1)), 1))
        neighbors.append((self.state[:self.k][::-1] + self.state[self.k:], 1))

        return neighbors
