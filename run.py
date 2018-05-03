from TemplateMatcher import TemplateMatcher
from WindowGrabber import WindowGrabber
from Player import Player
from GameCache import GameCache
import cv2
import time
import numpy as np
import math
import sys

# constants
LIVE_REWARD = 1
DEAD_REWARD = -10
MAX_CACHE = 1000000
GAMMA = 0.9
BATCH_SIZE = 50
EPSILON_RANGE = (1.0, 0.1)
EPSILON_ITERATION_END = 1e6

# create finite state machine variable
fsm = 'restarting'

# create array to store 'state' of game, which is a sequence of images in the shape (height, width, time_steps + 1)
state = np.zeros((128, 256, 5))
state_previous = np.zeros((128, 256, 5))

# create information for the game
player = Player()
gameover_matcher = TemplateMatcher(cv2.Canny(cv2.imread('res/gameover.png'), 50, 200), 0.5)
window_grabber = WindowGrabber()
game_cache = GameCache('cache', MAX_CACHE)

try:
    game_cache.load(state.shape)
    print('Loaded previous cache of length ' + str(len(game_cache)))
except FileNotFoundError:
    pass

# restore player if we can
player.restore()
def calc_epsilon():
    global player

    iterations = player.net.global_step()
    if iterations >= EPSILON_ITERATION_END:
        epsilon = EPSILON_RANGE[1]
    else:
        epsilon = ((EPSILON_ITERATION_END - iterations) * EPSILON_RANGE[0] + iterations * EPSILON_RANGE[1]) / EPSILON_ITERATION_END

    return epsilon
player.policy = lambda state: player.epsilon_greedy_action(state, calc_epsilon())

# create array to store 'state' of game, which is a sequence of images in the shape (height, width, time_steps + 1)
state = np.zeros((128, 256, 5))
state_previous = np.zeros((128, 256, 5))

# select window
print("Select window in 3 seconds")
time.sleep(3)
window_grabber.set_window_to_focus()
player.set_window_to_focus()

# create variable to keep track of what time_step in each game we are at
time_step = 0

# define necessary variables, just to be clear
image_state = None
image_state_previous = None
values = None
values_previous = None
action = None
action_previous = None
reward_previous = None

def train(print_bar=True, bar_length=20):
    global player, game_cache

    # randomly permute game caches and iterate
    gc = game_cache

    if len(gc) == 0:
        return
    
    # randomly permute states and iterate
    indices = np.random.permutation(len(gc))

    batches = math.ceil(indices.shape[0] / BATCH_SIZE)
    for batch_i in range(0, batches):
        batch_indices = indices[batch_i * BATCH_SIZE : min(len(indices), (batch_i+1) * BATCH_SIZE)]
       
        training_states = gc.state(batch_indices)
        training_rewards = gc.reward(batch_indices)
        training_actions = gc.action(batch_indices)
        training_next_terminal = gc.terminal(batch_indices)

        player.learn(training_states, training_actions, training_rewards, training_next_terminal, GAMMA)

        # print progress bar
        if print_bar:
            fraction_done = batch_i / (batches - 1)
            bar_segments = math.floor(bar_length * fraction_done)
            sys.stdout.write('\r')
            sys.stdout.write(('\r[%-' + str(bar_length) + 's] %d%%') % ('=' * bar_segments, fraction_done * 100))
            sys.stdout.flush()

    # print new line after progress bar after complete
    if print_bar:
        print()

    # do summary on whatever the last batch was
    player.net.summarize(training_states, training_actions, training_rewards, training_next_terminal, GAMMA)
    
# run player
newgame = False
while True:
    try:
        # grab the frame of the game
        image_raw = window_grabber.grab()

        # create display image
        image_display = image_raw.copy()

        # check if the game is over
        image_canny = cv2.Canny(image_raw, 50, 200)
        gameover = gameover_matcher.isMatch(image_canny)

        # create image for adding to the state
        image_state = cv2.cvtColor(cv2.resize(image_raw, (state.shape[1], state.shape[0])), cv2.COLOR_RGBA2GRAY)

        # add previous image to the game cache
        if time_step != 0:
            # calculate the reward for the previous image and the action taken
            reward_previous = DEAD_REWARD if gameover else LIVE_REWARD

            # push previous state
            game_cache.push(state_previous, values_previous, action_previous, reward_previous, gameover)

        # update finite state machine
        if fsm == 'playing':
            if gameover:
                print('Game over')
                fsm = 'gameover'
        elif fsm == 'restarting':
            if not gameover:
                print('New game started')
                fsm = 'playing'

        # evalutate finite state machine (and sometimes update)
        if fsm == 'restarting':
            # restart the game
            player.gc.restart()
        elif fsm == 'gameover':
            fsm = 'restarting'

            # write game over on display image
            cv2.putText(image_display, "Game Over", (0, image_display.shape[0]), cv2.FONT_HERSHEY_PLAIN, 1, (0,0,255))

            # train the player
            print ('Training on ' + str(len(game_cache)) + ' states')
            train()
            print ('Training finished for step ' + str(player.net.global_step()))

            time_step = 0
            
        elif fsm == 'playing':
            # if the game not over, add the new image to the state
            if time_step == 0:
                # repeat the image for each time_step
                state = np.repeat(image_state[...,None], state.shape[-1], axis=-1)
            else:
                # roll state over, and add new image to the end
                state = np.roll(state, -1, axis=-1)
                state[...,-1] = image_state

            # get player's action values
            values = player.value(state)
            # get player to take action
            action = player.policy_action(state)
            player.act(action)

            # write values on display image
            cv2.putText(image_display, "Epsilon: " + str(calc_epsilon()), (0, image_display.shape[0] - 48), cv2.FONT_HERSHEY_PLAIN, 1, (0,255,0))
            cv2.putText(image_display, "Left:    " + str(values[0])     , (0, image_display.shape[0] - 32), cv2.FONT_HERSHEY_PLAIN, 1, (0,255,0))
            cv2.putText(image_display, "Stop:    " + str(values[1])     , (0, image_display.shape[0] - 16), cv2.FONT_HERSHEY_PLAIN, 1, (0,255,0))
            cv2.putText(image_display, "Right:   " + str(values[2])     , (0, image_display.shape[0]     ), cv2.FONT_HERSHEY_PLAIN, 1, (0,255,0))

            time_step += 1
            image_state_previous = image_state
            action_previous = action
            values_previous = values
            state_previous = state

        cv2.imshow("Player", image_display)
        cv2.waitKey(1)
    except KeyboardInterrupt:
        print('Saving player')
        player.save()
        print('Saving cache')
        game_cache.save()

        break
