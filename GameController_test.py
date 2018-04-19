from GameController import GameController
import time

controller = GameController()

print("Select window to send keypresses to in 3 seconds")
time.sleep(3)
controller.set_window_to_focus()

while True:
    controller.restart()
    time.sleep(0.25)
    controller.go_left()
    time.sleep(0.25)
    controller.stop()
    time.sleep(0.25)
    controller.go_right()
    time.sleep(0.25)
