from TemplateMatcher import TemplateMatcher
from WindowGrabber import WindowGrabber
from Player import Player
import cv2
import time
import numpy as np

# create information for the game
player = Player()
gameover_matcher = TemplateMatcher(cv2.Canny(cv2.imread('res/gameover.png'), 50, 200), 0.5)
window_grabber = WindowGrabber()

# create array to store 'state' of game, which is a sequence of images in the shape (width, height, colors, time_steps)
state = np.zeros((100, 200, 3, 4))

# select window
print("Select window in 3 seconds")
time.sleep(3)
window_grabber.set_window_to_focus()
player.set_window_to_focus()

# create variable to keep track of what time_step in each game we are at
time_step = -1

# run player
while True:
    time_step += 1

    # grab the frame of the game
    image_raw = window_grabber.grab()

    # create display image
    image_display = image_raw.copy()

    # check if the game is over
    image_canny = cv2.Canny(image_raw, 50, 200)
    gameover = gameover_matcher.isMatch(image_canny)

    if gameover:
        # if the game is over, restart
        player.gc.restart()
        time_step = 0

        # write game over on display image
        cv2.putText(image_display, "Game Over", (0, image_display.shape[0]), cv2.FONT_HERSHEY_PLAIN, 1, (0,0,255))
    else:
        # if the game not over, add the new image to the state
        image_state = cv2.cvtColor(cv2.resize(image_raw, (state.shape[1], state.shape[0])), cv2.COLOR_RGBA2RGB)
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
        player.act(player.policy_action(state))

        # write values on display image
        cv2.putText(image_display, "Left:  " + str(values[0]), (0, image_display.shape[0] - 32), cv2.FONT_HERSHEY_PLAIN, 1, (0,255,0))
        cv2.putText(image_display, "Stop:  " + str(values[1]), (0, image_display.shape[0] - 16), cv2.FONT_HERSHEY_PLAIN, 1, (0,255,0))
        cv2.putText(image_display, "Right: " + str(values[2]), (0, image_display.shape[0]     ), cv2.FONT_HERSHEY_PLAIN, 1, (0,255,0))

    cv2.imshow("Player", image_display)
    cv2.waitKey(1)
