from KeySender import KeySender
from Xlib import XK

class GameController:
    def __init__(self):
        self._last_key = None
        self._key_sender = KeySender()

    def set_window_to_focus(self):
        self._key_sender.set_window_to_focus()

    def restart(self):
        self._switch_key(XK.XK_space)

    def go_left(self):
        self._switch_key(XK.XK_Left)

    def go_right(self):
        self._switch_key(XK.XK_Right)

    def stop(self):
        self._release_last_key()

    def _switch_key(self, key):
        if self._last_key is not None:
            self._key_sender.key_up(self._last_key)

        if key is not None:
            self._key_sender.key_down(key)

        self._last_key = key

    def _release_last_key(self):
        if self._last_key is not None:
            self._key_sender.key_up(self._last_key)
        self._last_key = None

