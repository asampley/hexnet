from TemplateMatcher import TemplateMatcher
from WindowGrabber import WindowGrabber
from Player import Player
from GameCache import GameCache
import cv2
import time
import numpy as np
import math

# constants
LIVE_REWARD = 1
DEAD_REWARD = -10
MAX_CACHES = 100
GAMMA = 0.9
BATCH_SIZE = 50
EPSILON_RANGE = (1.0, 0.1)
EPSILON_ITERATION_END = 1e6

# create finite state machine variable
fsm = 'gameover'

# create information for the game
player = Player()
gameover_matcher = TemplateMatcher(cv2.Canny(cv2.imread('res/gameover.png'), 50, 200), 0.5)
window_grabber = WindowGrabber()
game_caches = [GameCache()]
game_cache = game_caches[0]

# restore player if we can
player.restore()
def policy(state):
    global player

    iterations = player.net.global_step()
    if iterations >= EPSILON_ITERATION_END:
        epsilon = EPSILON_RANGE[1]
    else:
        epsilon = ((EPSILON_ITERATION_END - iterations) * EPSILON_RANGE[0] + iterations * EPSILON_RANGE[1]) / EPSILON_ITERATION_END

    return player.epsilon_greedy_action(state, epsilon)
player.policy = policy

# create array to store 'state' of game, which is a sequence of images in the shape (height, width, time_steps)
state = np.zeros((128, 256, 4))

# select window
print("Select window in 3 seconds")
time.sleep(3)
window_grabber.set_window_to_focus()
player.set_window_to_focus()

# create variable to keep track of what time_step in each game we are at
time_step = 0
game_index = 0

# define necessary variables, just to be clear
image_state = None
image_state_previous = None
values = None
values_previous = None
action = None
action_previous = None
reward_previous = None

def train():
    global player, game_caches

    # randomly permute game caches and iterate
    for gci in np.random.permutation(len(game_caches)):
        gc = game_caches[gci]

        if len(gc) == 0:
            continue
        
        # randomly permute states and iterate
        indices = np.random.permutation(np.arange(state.shape[-1] - 1, len(gc) - 1))
        
        for batch_i in range(0, math.ceil(indices.shape[0] / BATCH_SIZE)):
            training_states = []
            training_next_ims = []
            training_rewards = []
            training_actions = []
            for im_i in range(0, BATCH_SIZE):
                i = batch_i * BATCH_SIZE + im_i
                if i >= len(indices):
                    break
                training_states += [np.moveaxis(gc.state(slice(indices[i] - state.shape[-1] + 1, indices[i] + 1)), 0, -1)]
                training_next_ims += [gc.state(indices[i] + 1)[...,np.newaxis]]
                training_rewards += [gc.reward(indices[i])]
                training_actions += [gc.action(indices[i])]
            training_states = np.stack(training_states)
            training_next_ims = np.stack(training_next_ims)
            training_rewards = np.stack(training_rewards)
            training_actions = np.stack(training_actions)

            player.learn(training_states, training_next_ims, training_actions, training_rewards, GAMMA)

        # do summary on whatever the last batch was
        player.net.summarize(training_states, training_next_ims, training_actions, training_rewards, GAMMA)
    
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

            game_cache.push(image_state_previous, values_previous, action_previous, reward_previous)

        # update finite state machine
        if fsm == 'playing':
            if gameover:
                fsm = 'gameover'
        elif fsm == 'restarting':
            if not gameover:
                fsm = 'playing'

        # evalutate finite state machine (and sometimes update)
        if fsm == 'restarting':
            # restart the game
            player.gc.restart()
            time_step = 0
        elif fsm == 'gameover':
            fsm = 'restarting'

            # write game over on display image
            cv2.putText(image_display, "Game Over", (0, image_display.shape[0]), cv2.FONT_HERSHEY_PLAIN, 1, (0,0,255))

            # save game_cache
            game_cache.save('cache/' + str(game_index) + '.npz')

            # train the player
            train()

            # switch game caches and clear it (in case it is an old one)
            game_index = game_index + 1 if game_index < MAX_CACHES - 1 else 0
            if len(game_caches) <= game_index:
                game_caches += [GameCache()]
            game_cache = game_caches[game_index]
            game_cache.clear()
            
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
            cv2.putText(image_display, "Left:  " + str(values[0]), (0, image_display.shape[0] - 32), cv2.FONT_HERSHEY_PLAIN, 1, (0,255,0))
            cv2.putText(image_display, "Stop:  " + str(values[1]), (0, image_display.shape[0] - 16), cv2.FONT_HERSHEY_PLAIN, 1, (0,255,0))
            cv2.putText(image_display, "Right: " + str(values[2]), (0, image_display.shape[0]     ), cv2.FONT_HERSHEY_PLAIN, 1, (0,255,0))

            time_step += 1
            image_state_previous = image_state
            action_previous = action
            values_previous = values

        cv2.imshow("Player", image_display)
        cv2.waitKey(1)
    except KeyboardInterrupt:
        print('Saving player')
        player.save()
        break
