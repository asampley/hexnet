from TemplateMatcher import TemplateMatcher
from WindowGrabber import WindowGrabber
from Player import Player
from GameCache import GameCache
import cv2
import time
import numpy as np
import math
import sys
import argparse

# read arguments
parser = argparse.ArgumentParser()

group = parser.add_argument_group('rewards')
group.add_argument('--rewardlive', type=float, help='Reward for every frame the agent is alive. Default: 1', default=1)
group.add_argument('--rewarddead', type=float, help='Reward for every frame the agent is dead. Default: -10', default=-10)

group = parser.add_argument_group('cache')
group.add_argument('--cacheprefix', type=str, help='Prefix for all cache files. Default: cache/', default='cache/')
group.add_argument('--cachemin', type=int, help='Minimum cache size before training starts. Default: 50000', default=50000)
group.add_argument('--cachemax', type=int, help='Maximum cache size before overwriting images. Default: 1000000', default=1000000)

group = parser.add_argument_group('neural network')
group.add_argument('--batchsize', type=int, help='Size of each batch for training the neural network. Default: 50', default=50)
group.add_argument('--imagesize', type=int, nargs=2, help='Height and width of the image. Default: 128 256', default=[128, 256])
group.add_argument('--imagecount', type=int, help='Number of images for each prediction by the network. Default: 4', default=4)
group.add_argument('--updatebatches', type=int, help='Number of training batches to do before updating the target weights. Default: 10000', default=10000)
group.add_argument('-k', '--kernel', dest='kernels', metavar=('height', 'width'), type=int, nargs=2, action='append'
        , help='Add a convolution kernel. Can be used multiple times.')
group.add_argument('-s', '--stride', dest='strides', metavar=('height', 'width'), type=int, nargs=2, action='append'
        , help='Specify the stride (height, width) for each convolution. Can be used multiple times. Must be used the same number of times as --kernel.')
group.add_argument('-c', '--channels', dest='channels', metavar='channels', type=int, action='append'
        , help='Specify the number of channels for each convolution. Can be used multiple times. Must be used the same number of times as --kernel.')
group.add_argument('-d', '--dense', dest='denses', metavar='neurons', type=int, action='append'
        , help='Add a dense layer after the convolution layers with the specified number of neurons. Can be used multiple times.')
group.add_argument('--savedir', type=str, help='Directory to save weights and training metrics. Default: model/', default='model/')

group = parser.add_argument_group('machine learning')
group.add_argument('--learningrate', type=float, help='Learning rate. Default: 1e-4', default=1e-4)
group.add_argument('-g', '--gamma', type=float, help='Discount rate gamma. Default: 0.99', default=0.99)

# parse arguments
args = parser.parse_args()
print(args)
assert len(args.kernels) == len(args.strides), 'Must specify a --stride for each --kernel'
assert len(args.kernels) == len(args.channels), 'Must specify a --channels for each --kernel'

# constants
LIVE_REWARD = args.rewardlive
DEAD_REWARD = args.rewarddead
MIN_CACHE = args.cachemin # only train if we have this many states
MAX_CACHE = args.cachemax
GAMMA = args.gamma
BATCH_SIZE = args.batchsize
EPSILON_RANGE = (1.0, 0.1)
EPSILON_ITERATION_END = 1e6
UPDATE_TARGET_ITERATIONS = args.updatebatches

IMAGE_SHAPE = args.imagesize # (height, width)
IMAGES_BCK = args.imagecount - 1 # DO NOT CHANGE
IMAGES_FWD = 1 # DO NOT CHANGE
IMAGES_PER_STATE = 1 + IMAGES_BCK + IMAGES_FWD # DO NOT CHANGE
STATE_SHAPE = IMAGE_SHAPE + [IMAGES_PER_STATE] # DO NOT CHANGE
NUM_ACTIONS = 3

# create finite state machine variable
fsm = 'restarting'

# create array to store 'state' of game, which is a sequence of images in the shape (height, width, time_steps + 1)
state = np.zeros(STATE_SHAPE)
state_previous = np.zeros(STATE_SHAPE)

# create information for the game
player = Player(
        learning_rate = args.learningrate
        , width = IMAGE_SHAPE[1]
        , height = IMAGE_SHAPE[0]
        , time_steps = args.imagecount
        , actions = NUM_ACTIONS
        , conv_kernels = args.kernels
        , conv_strides = args.strides
        , conv_channels = args.channels
        , dense_channels = args.denses
        , save_dir = args.savedir
        )
gameover_matcher = TemplateMatcher(cv2.Canny(cv2.imread('res/gameover.png'), 50, 200), 0.5)
window_grabber = WindowGrabber()
game_cache = GameCache(args.cacheprefix, IMAGE_SHAPE, NUM_ACTIONS, MAX_CACHE)

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
        # update target weights every so many iterations
        if player.net.global_step() % UPDATE_TARGET_ITERATIONS == 0:
            print('\nUpdated target function weights') # newline to print nicely with bar
            player.update_target_function()

        batch_indices = indices[batch_i * BATCH_SIZE : min(len(indices), (batch_i+1) * BATCH_SIZE)]
        
        training_states = gc.state(batch_indices, IMAGES_BCK, IMAGES_FWD)
        training_rewards = gc.reward(batch_indices)
        training_actions = gc.action(batch_indices)
        training_next_terminal = gc.terminal(batch_indices)

        player.learn(training_states, training_actions, training_rewards, training_next_terminal, GAMMA)

        # print progress bar
        if print_bar:
            if batches == 1:
                fraction_done = 1
            else:
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

        # check if the game is over
        image_canny = cv2.Canny(image_raw, 50, 200)
        gameover = gameover_matcher.isMatch(image_canny)

        # create image for adding to the state
        image_state = cv2.cvtColor(cv2.resize(image_raw, (state.shape[1], state.shape[0])), cv2.COLOR_RGBA2GRAY)

        # create display image
        image_display = cv2.cvtColor(cv2.resize(image_state, (image_raw.shape[1], image_raw.shape[0]), interpolation=cv2.INTER_NEAREST), cv2.COLOR_GRAY2BGR)

        # add previous image to the game cache
        if time_step != 0:
            # calculate the reward for the previous image and the action taken
            reward_previous = DEAD_REWARD if gameover else LIVE_REWARD

            # push previous state
            game_cache.push(image_state_previous, values_previous, action_previous, reward_previous, gameover)

        # update finite state machine
        if fsm == 'playing':
            if gameover:
                print('Game over with cache of size ' + str(len(game_cache)))
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
            cv2.waitKey(1)

            if len(game_cache) >= MIN_CACHE:
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
