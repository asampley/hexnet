from Xlib import display as Xdisplay, X
import Xlib.protocol as Xprotocol
import Xlib.protocol.event as Xevent
import time

class KeySender:
    def __init__(self):
        self.display = Xdisplay.Display()
        self.window = self.display.get_input_focus().focus

    def set_window_to_focus(self):
        self.window = self.display.get_input_focus().focus

    def key_down(self, keysym, modifiers=0):
        self.window.send_event(self._create_key_event(True, keysym, modifiers))
        self.window.display.flush()

    def key_up(self, keysym, modifiers=0):
        self.window.send_event(self._create_key_event(False, keysym, modifiers))
        self.window.display.flush()

    def _create_key_event(self, press, keysym, modifiers):
        if press:
            XeventConstructor = Xevent.KeyPress
        else:
            XeventConstructor = Xevent.KeyRelease

        event = XeventConstructor(
            time = X.CurrentTime,
            root = self.window.query_tree().root,
            window = self.window,
            same_screen = 1,
            child = X.NONE,
            root_x = 1,
            root_y = 1,
            event_x = 1,
            event_y = 1,
            state = modifiers,
            detail = self.display.keysym_to_keycode(keysym)
            )
        
        return event
