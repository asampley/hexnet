from GameCache import GameCache
import numpy as np
import cv2
import argparse

parser = argparse.ArgumentParser(description="View a game cache file")
parser.add_argument("file", type=str, help="game cache file")

args = parser.parse_args()

cache = GameCache()
cache.load(args.file)
optimal_values = cache.optimal_values(0.9)

def show_index(index):
    global cache, optimal_values
    image = cv2.cvtColor(cache.state(index), cv2.COLOR_GRAY2BGR)
    cv2.putText(image, 
            "Values: " + np.array2string(cache.values(index), floatmode='fixed', precision=1), 
            (0, image.shape[0]     ), cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 0), 2)
    cv2.putText(image, 
            "V*s   : " + np.array2string(optimal_values[index,:], floatmode='fixed', precision=1), 
            (0, image.shape[0] - 16), cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 0), 2)
    cv2.putText(image, "Action: " + str(cache.action(index)), (0, image.shape[0] - 32), cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 0), 2)
    cv2.putText(image, "Reward: " + str(cache.reward(index)), (0, image.shape[0] - 48), cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 0), 2)
    cv2.imshow("Game Cache", image)

cv2.namedWindow("Game Cache")
cv2.createTrackbar("Index", "Game Cache", 0, len(cache) - 1, show_index)
show_index(0)

while True:
    k = cv2.waitKey(100)

    # run until escape or window is closed
    if k == 27 or cv2.getWindowProperty("Game Cache", cv2.WND_PROP_VISIBLE) < 1:
        break
