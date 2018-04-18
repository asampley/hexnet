from WindowGrabber import WindowGrabber
import time
import cv2

grabber = WindowGrabber()
print("Select window in 3 seconds")
time.sleep(3)
grabber.set_window_to_focus()

while True:
    cv2.imshow("Image", grabber.grab())
    cv2.waitKey(1)

