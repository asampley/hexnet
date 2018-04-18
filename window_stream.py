from Xlib import display as Xdisplay
from Xlib.protocol import request as Xrequest
import numpy as np
import mss
import time
import cv2

# get X display
display = Xdisplay.Display()

# wait 3 seconds, and select the window focussed
print("Select window in 3 seconds")
time.sleep(3)
window = display.get_input_focus().focus

while True:
    with mss.mss() as screenshot:
        # start timing
        t1 = time.time()

        # get the geometry of the window
        geometry = window.get_geometry()

        # convert coords to global display coords
        coords = Xrequest.TranslateCoords(display=display.display, src_wid=window, dst_wid=window.query_tree().root, src_x=0, src_y=0)
        monitor = {'left': coords.x, 'top': coords.y, 'width': geometry.width, 'height': geometry.height}

        # print time after converting coordinates
        t2 = time.time()
        print("Converted to screen coordinates in " + str(t2 - t1) + "s")

        # take screenshot
        image = screenshot.grab(monitor)

        # print time after taking screenshot
        t3 = time.time()
        print("Took screen shot in " + str(t3 - t2) + "s")

        arr = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        print(arr.shape)
        print(arr.dtype)

        # print time after converting to numpy
        t4 = time.time()
        print("Converted to numpy array in " + str(t4 - t3) + "s")

        # show image
        cv2.imshow("Image", arr)
        cv2.waitKey(1)

        # print time to show image
        t5 = time.time()
        print("Displayed image in " + str(t5 - t4) + "s")

        # end timing
        print("Total time: " + str(t5 - t1) + "s")

