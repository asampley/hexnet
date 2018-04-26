import numpy as np

class GameCache:
    def __init__(self):
        self._sequential = False
        self._values = []
        self._states = []
        self._actions = []
        self._rewards = []
        self._terminal = []
        self._len = 0

    def __len__(self):
        return self._len

    def clear(self):
        self._sequential = False
        self._values = []
        self._states = []
        self._actions = []
        self._rewards = []
        self._terminal = []
        self._len = 0

    def state(self, index):
        return self._from_array(self._states, index)

    def action(self, index):
        return self._from_array(self._actions, index)

    def reward(self, index):
        return self._from_array(self._rewards, index)

    def values(self, index):
        return self._from_array(self._values, index)

    def terminal(self, index):
        return self._from_array(self._terminal, index)

    def _from_array(self, array, index):
        self.make_sequential()
        return array[index,...]

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
            terminal = self.terminal(i)
            if terminal:
                optimal_values[i, action] = self.reward(i)
            else:
                optimal_values[i, action] = self.reward(i) + gamma * np.max(optimal_values[i+1, :])

        return optimal_values

    def push(self, state, values, action, reward, terminal):
        state_append = state[np.newaxis,...]
        values_append = values[np.newaxis,...]
        action_append = np.array([action])
        reward_append = np.array([reward])
        terminal_append = np.array([terminal])

        if self._sequential:
            self._sequential = False
            if len(self._states) == 0:
                self._states = [state_append]
            else:
                self._states = [self._states, state_append]
            if len(self._values) == 0:
                self._values = [values_append]
            else:
                self._values = [self._values, values_append]
            if len(self._actions) == 0:
                self._actions = [action_append]
            else:
                self._actions = [self._actions, action_append]
            if len(self._rewards) == 0:
                self._rewards = [reward_append]
            else:
                self._rewards = [self._rewards, reward_append]
            if len(self._terminal) == 0:
                self._terminal = [terminal_append]
            else:
                self._terminal = [self._terminal, terminal_append]
        else:
            self._states += [state_append]
            self._values += [values_append]
            self._actions += [action_append]
            self._rewards += [reward_append]
            self._terminal += [terminal_append]
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
                self._actions = np.concatenate(self._actions, axis=0)
                self._rewards = np.concatenate(self._rewards, axis=0)
                self._terminal = np.concatenate(self._terminal, axis=0)
            else:
                self._states = np.zeros(0)
                self._values = np.zeros(0)
                self._actions = np.zeros(0)
                self._rewards = np.zeros(0)
                self._terminal = np.zeros(0)

    def save(self, filename):
        self.make_sequential()
        np.savez(filename, states=self._states, values=self._values, actions=self._actions, rewards=self._rewards, terminal=self._terminal)

    def load(self, filename):
        self._sequential = True
        data = np.load(filename)
        self._states = data['states']
        self._values = data['values']
        self._actions = data['actions']
        self._rewards = data['rewards']
        self._terminal = data['terminal']
        self._len = self._states.shape[0]
