import pyautogui
import time

print("Select window to recieve key presses in 3 seconds")
time.sleep(3)

pyautogui.press('space')
time.sleep(0.5)
pyautogui.keyDown('left')
time.sleep(0.5)
pyautogui.keyUp('left')
pyautogui.keyDown('right')
time.sleep(0.5)
pyautogui.keyUp('right')

