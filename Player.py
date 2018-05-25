from GameController import GameController
from model import Net
import numpy as np

class Player:
    def __init__(self, learning_rate=1e-4, width=256, height=128, time_steps=4, actions=3
            , conv_kernels=[[16,16],[8,8]], conv_strides=[[8,8],[4,4]], conv_channels=[16,32], dense_channels=[256]
            , save_dir='model/'):
        # used to control the game
        self.gc = GameController()

        # used to predict values of state
        params = {
                'LEARNING_RATE': learning_rate,
                'WIDTH': width,
                'HEIGHT':height,
                'TIME_STEPS': time_steps,
                'ACTIONS': actions,
                'CONV_KERNELS': conv_kernels,
                'CONV_STRIDES': conv_strides,
                'CONV_CHANNELS': conv_channels,
                'DENSE_CHANNELS': dense_channels,
                'SAVE_DIR': save_dir
                }
        self.net = Net(params)
 
        # action map maps action keys to functions of the form () -> ()
        self.actionMap = {
                0: self.gc.go_left,
                1: self.gc.stop,
                2: self.gc.go_right
                }

        # policy is a function of the form (state) -> (action)
        self.policy = self.best_action

    def save(self):
        self.net.save()

    def restore(self):
        try:
            self.net.restore()
        except:
            pass

    def set_window_to_focus(self):
        self.gc.set_window_to_focus()

    def value(self, state, action=None):
        """
        Returns a value for a state action pair.
        state is a (width, height, time_steps) size numpy array.
        If action is None, return a numpy array of size (actions).
        """
        if action is None:
            return self.net.predict(state[np.newaxis,...]).reshape((3,))
        else:
            return self.net.predict(state[np.newaxis,...]).reshape((3,))[action]

    def act(self, action):
        """
        Take action.
        """
        self.actionMap[action]()

    def policy_action(self, state):
        """
        Return the action selected by the policy
        """
        return self.policy(state)
    
    def epsilon_greedy_action(self, state, epsilon):
        if np.random.rand() < epsilon:
            return self.random_action(state)
        else:
            return self.best_action(state)

    def best_action(self, state):
        """
        Get the best action for a state
        """
        action = np.argmax(self.value(state))
        return action

    def random_action(self, state):
        """
        Get a random action based on the size of actionMap
        """
        action = np.random.randint(len(self.actionMap))
        return action

    def learn(self, states, actions, rewards, next_terminal, gamma):
        """
        Learn action values for states.
        states is a (N, width, height, time_steps + 1) size numpy array.
        action_values is a (N, actions) size numpy array.
        """
        self.net.train(states, actions, rewards, next_terminal, gamma)

    def update_target_function(self):
        """
        Updates the function for computing the target for training
        """
        self.net.train_update_target_weights()
