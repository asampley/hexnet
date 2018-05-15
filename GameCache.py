import numpy as np
import os

class GameCache:
    def __init__(self, fileprefix, image_shape, num_actions, max_size):
        self._fileprefix = fileprefix
        self._image_shape = image_shape
        self._values = None
        self._images = None
        self._actions = None
        self._rewards = None
        self._terminal = None
        self._len = 0
        self._max_size = max_size
        self._push_index = 0

        # attempt to load old data (will override image_shape, num_actions, and max_size if necessary)
        # allocate memory if old cache cannot be loaded
        try:
            self._load()
        except FileNotFoundError:
            self._reallocate(num_actions)



    def __len__(self):
        return self._len

    def clear(self):
        self._len = 0
        self._push_index = 0

    def _reallocate(self, num_actions):
        # create directory if necessary
        dirname = os.path.dirname(self._fileprefix)
        if len(dirname) > 0 and not os.path.exists(dirname):
            os.makedirs(dirname)

        # allocate numpy arrays
        if self._images is not None:
            del self._images
        self._images = np.memmap(self._fileprefix + 'images.npy', dtype=np.uint8, mode='w+', shape=(self._max_size,) + self._image_shape)
        self._values = np.zeros((self._max_size, num_actions), np.float32)
        self._actions = np.zeros((self._max_size,), np.int32)
        self._rewards = np.zeros((self._max_size,), np.float32)
        self._terminal = np.zeros((self._max_size,), np.bool)
        self._len = 0
        self._push_index = 0

    def state(self, index, images_bck, images_fwd):
        # work on either slices or ints or numpy arrays, though not efficiently
        if type(index) is int:
            index = np.arange(index, index + 1)
        elif type(index) is slice:
            pass

        state = np.zeros((len(index),) + self._image_shape + (images_bck + images_fwd + 1,), self._images.dtype)

        for indexi in range(len(index)):
            # only return as far back and forward as non-terminal states (but the state at index can be terminal)
            indices = index[indexi] + np.arange(-images_bck, images_fwd + 1, dtype=np.int64)

            # make sure no previous images are terminal
            for i in range(images_bck):
                # if any indices are less than 0, make it and previous indices 0
                if indices[i] < 0:
                    indices[:i+1] = 0
                # if this index is terminal, make it and previous indices point to the last non-terminal image
                if np.any(self.terminal(indices[i])):
                    indices[:i+1] = indices[i] + 1

            # make sure only one of current and future images is terminal
            for i in range(-images_fwd - 1, 0):
                # if any indices are greater than or equal to len, make it and future indices len-1
                if indices[i] >= len(self):
                    indices[i:] = len(self) - 1
                # if this index is terminal, make it and future indices point to it
                if np.any(self.terminal(indices[i])):
                    indices[i:] = indices[i]
           
            state[indexi, ...] = np.moveaxis(self._from_array(self._images, indices), 0, -1)

        return state # shaped (index,) + image_shape + (state_time_steps,)

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

    def push(self, image, values, action, reward, terminal):
        image_append = image[np.newaxis,...]
        values_append = values[np.newaxis,...]
        action_append = np.array([action])
        reward_append = np.array([reward])
        terminal_append = np.array([terminal])

        self._images[self._push_index,...] = image_append
        self._values[self._push_index,...] = values_append
        self._actions[self._push_index,...] = action_append
        self._rewards[self._push_index,...] = reward_append
        self._terminal[self._push_index,...] = terminal_append
            
        self._push_index = (self._push_index + 1) % self._max_size
        self._len = min(self._len + 1, self._max_size)

    def save(self):
        np.savez(self._fileprefix + 'other.npz', 
                values=self._values, 
                actions=self._actions, 
                rewards=self._rewards, 
                terminal=self._terminal,
                max_size=self._max_size,
                image_shape=self._image_shape,
                len=self._len)

    def _load(self):
        # check for existance of files
        for filesuffix in ['other.npz', 'images.npy']:
            filename = self._fileprefix + filesuffix
            if not os.path.isfile(filename):
                raise FileNotFoundError('Unable to find ' + filename)

        # load previous cache
        with np.load(self._fileprefix + 'other.npz') as data:
            self._values = data['values']
            self._actions = data['actions']
            self._rewards = data['rewards']
            self._terminal = data['terminal']
            self._max_size = np.asscalar(data['max_size'])
            self._image_shape = tuple(data['image_shape'])
            self._len = np.asscalar(data['len'])
            self._push_index = self._len if self._len < self._max_size else 0
        if self._images is not None:
            del self._images
        self._images = np.memmap(self._fileprefix + 'images.npy', dtype=np.uint8, mode='r+', shape=(self._max_size,) + self._image_shape)
