from KeySender import KeySender
from Xlib import XK
import time

sender = KeySender()
print("Select window to recieve key presses in 3 seconds")
time.sleep(3)
sender.set_window_to_focus()

sender.key_down(XK.XK_space)
sender.key_up(XK.XK_space)
time.sleep(0.5)
sender.key_down(XK.XK_Left)
time.sleep(0.5)
sender.key_up(XK.XK_Left)
sender.key_down(XK.XK_Right)
time.sleep(0.5)
sender.key_up(XK.XK_Right)

