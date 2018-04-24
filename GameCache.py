import numpy as np

class GameCache:
    def __init__(self):
        self._sequential = False
        self._values = []
        self._states = []
        self._actions = []
        self._rewards = []
        self._len = 0

    def __len__(self):
        return self._len

    def clear(self):
        self._sequential = False
        self._states = []
        self._actions = []
        self._rewards = []
        self._values = []
        self._len = 0

    def state(self, index):
        return self._from_array(self._states, index)

    def action(self, index):
        return self._from_array(self._actions, index)

    def reward(self, index):
        return self._from_array(self._rewards, index)

    def values(self, index):
        return self._from_array(self._values, index)

    def optimal_values(self, gamma):
        """
        Returns a numpy array of optimal values given a gamma, calculated
        as based on taking the optimal action at each step, and what the
        cumulative weighted rewards will be.
        Return value is of shape (N, actions).
        """
        if len(self) == 0:
            return np.zeros(0)

        self.make_sequential()

        optimal_values = self._values.copy()
        
        # set the last value
        action = self.action(-1)
        optimal_values[-1, action] = self.reward(-1)

        # iterate backwards through all previous values
        for i in range(optimal_values.shape[0] - 2, -1, -1):
            action = self.action(i)
            optimal_values[i, action] = self.reward(i) + gamma * np.max(optimal_values[i+1, :])

        return optimal_values

    def _from_array(self, array, index):
        if self._sequential:
            return array[index,...]
        else:
            return array[index]

    def push(self, state, values, action, reward):
        if self._sequential:
            self._sequential = False
            self._states = [self._states, state[np.newaxis,...]]
            self._values = [self._values, values[np.newaxis,...]]
            self._actions = [self._actions, action]
            self._rewards = [self._rewards, reward]
        else:
            self._states += [state[np.newaxis,...]]
            self._values += [values[np.newaxis,...]]
            self._actions += [action]
            self._rewards += [reward]
        self._len += 1

    def make_sequential(self):
        """
        Make the game cache store all its data sequentially, to make
        it appropriate for saving to a file.
        """
        if self._sequential:
            return
        else:
            self._sequential = True
            if len(self) > 0:
                self._states = np.concatenate(self._states, axis=0)
                self._values = np.concatenate(self._values, axis=0)
                self._actions = np.stack(self._actions)
                self._rewards = np.stack(self._rewards)
            else:
                self._states = np.zeros(0)
                self._values = np.zeros(0)
                self._actions = np.zeros(0)
                self.rewards = np.zeros(0)

    def save(self, filename):
        self.make_sequential()
        np.savez(filename, states=self._states, values=self._values, actions=self._actions, rewards=self._rewards)

    def load(self, filename):
        self._sequential = True
        data = np.load(filename)
        self._states = data['states']
        self._values = data['values']
        self._actions = data['actions']
        self._rewards = data['rewards']
        self._len = self._states.shape[0]
