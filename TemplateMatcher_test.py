from WindowGrabber import WindowGrabber
from TemplateMatcher import TemplateMatcher
import numpy as np
import cv2
import time

wg = WindowGrabber()
template_image = cv2.Canny(cv2.imread('res/gameover.png'), 50, 200)
tm = TemplateMatcher(template_image, 0.5)

print("Select window in 3 seconds")
time.sleep(3)
wg.set_window_to_focus()

while True:
    image_raw = wg.grab()
    image_cmp = cv2.Canny(image_raw, 50, 200)
    match = tm.isMatch(image_cmp)

    display = cv2.cvtColor(image_cmp, cv2.COLOR_GRAY2BGR)

    if match:
        border_color = (0, 255, 0)
    else:
        border_color = (0, 0, 255)

    cv2.rectangle(display, (0,0), (display.shape[1], display.shape[0]), border_color, 4)

    cv2.imshow('Image', display)
    cv2.waitKey(1)
