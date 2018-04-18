from Xlib import display as Xdisplay
from Xlib.protocol import request as Xrequest
import numpy as np
import mss

class WindowGrabber:
    def __init__(self):
        self.display = Xdisplay.Display()
        self.screenshot = mss.mss()
        self.window = self.display.get_input_focus().focus

    def set_window_to_focus(self):
        self.window = self.display.get_input_focus().focus

    def grab(self):
        # get the geometry of the window
        geometry = self.window.get_geometry()

        # convert coords to global display coords
        coords = Xrequest.TranslateCoords(display=self.display.display, src_wid=self.window, dst_wid=self.window.query_tree().root, src_x=0, src_y=0)
        monitor = {'left': coords.x, 'top': coords.y, 'width': geometry.width, 'height': geometry.height}

        # take screenshot
        return np.array(self.screenshot.grab(monitor))
