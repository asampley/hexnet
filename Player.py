from GameController import GameController
import numpy as np

class Player:
    def __init__(self):
        # used to control the game
        self.gc = GameController()
        
        # action map maps action keys to functions of the form () -> ()
        self.actionMap = {
                0: self.gc.go_left,
                1: self.gc.stop,
                2: self.gc.go_right
                }

        # policy is a function of the form (state) -> (action)
        self.policy = self.best_action

    def set_window_to_focus(self):
        self.gc.set_window_to_focus()

    def value(self, state, action=None):
        """
        Returns a value for a state action pair.
        If action is None, return a numpy array of size (actions).
        """
        if action is None:
            return np.zeros((len(self.actionMap),))
        else:
            return np.zeros((1,))

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

    def best_action(self, state):
        """
        Get the best action for a state
        """
        action = np.argmax(self.value(state))
        return action

    def learn(self, states, action_values):
        """
        Learn action values for states.
        states is a (N, width, height, time_steps) size numpy array.
        action_values is a (N, actions) size numpy array.
        """
        pass