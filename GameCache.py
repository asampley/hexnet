import numpy as np

class GameCache:
    def __init__(self, fileprefix, max_size):
        self._fileprefix = fileprefix
        self._allocated = False
        self._values = None
        self._states = None
        self._actions = None
        self._rewards = None
        self._terminal = None
        self._len = 0
        self._max_size = max_size
        self._push_index = 0

    def __len__(self):
        return self._len

    def clear(self):
        self._len = 0
        self._push_index = 0

    def _reallocate(self, state_shape, num_actions):
        self._allocated = True
        if self._states is not None:
            del self._states
        self._states = np.memmap(self._fileprefix + '_states.npy', dtype=np.uint8, mode='w+', shape=(self._max_size,) + state_shape)
        self._values = np.zeros((self._max_size, num_actions), np.float32)
        self._actions = np.zeros((self._max_size,), np.int32)
        self._rewards = np.zeros((self._max_size,), np.float32)
        self._terminal = np.zeros((self._max_size,), np.bool)
        self._len = 0
        self._push_index = 0

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

        if not self._allocated:
            self._reallocate(state.shape, values.size)
            
        self._states[self._push_index,...] = state_append
        self._values[self._push_index,...] = values_append
        self._actions[self._push_index,...] = action_append
        self._rewards[self._push_index,...] = reward_append
        self._terminal[self._push_index,...] = terminal_append
            
        self._push_index = (self._push_index + 1) % self._max_size
        self._len = min(self._len + 1, self._max_size)

    def save(self):
        np.savez(self._fileprefix + '_other.npz', 
                values=self._values, 
                actions=self._actions, 
                rewards=self._rewards, 
                terminal=self._terminal,
                max_size=self._max_size,
                len=self._len)

    def load(self, state_shape):
        with np.load(self._fileprefix + '_other.npz') as data:
            self._sequential = True
            self._values = data['values']
            self._actions = data['actions']
            self._rewards = data['rewards']
            self._terminal = data['terminal']
            self._max_size = data['max_size']
            self._len = data['len']
            self._push_index = self._len if self._len < self._max_size else 0
        if self._states is not None:
            del self._states
        self._states = np.memmap(self._fileprefix + '_states.npy', dtype=np.uint8, mode='r+', shape=(self._max_size,) + state_shape)
