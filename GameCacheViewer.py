from GameCache import GameCache
import numpy as np
import cv2
import argparse

parser = argparse.ArgumentParser(description="View a game cache file")
parser.add_argument("file", type=str, help="game cache file")
parser.add_argument("--scale", type=float, default=None, help="scale factor of window")
parser.add_argument("--size", type=float, nargs=3, default=(128,256,5), help="size of the state in the cache")

args = parser.parse_args()

cache = GameCache(args.file, 0)
cache.load(args.size)
print("Loaded cache of length " + str(len(cache)))
optimal_values = cache.optimal_values(0.9)

index = 0
frame = 0

def set_index(i):
    global index
    index = i
    show()

def set_frame(i):
    global frame
    frame = i
    show()

def show():
    global cache, optimal_values
    image = cv2.cvtColor(cache.state(index)[...,frame], cv2.COLOR_GRAY2BGR)
    if args.scale is not None:
        image = cv2.resize(image, (0, 0), fx=args.scale, fy=args.scale, interpolation=cv2.INTER_NEAREST)
    cv2.putText(image, 
            "Values: " + np.array2string(cache.values(index), floatmode='fixed', precision=1), 
            (0, image.shape[0]     ), cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 0), 2)
    cv2.putText(image, 
            "V*s   : " + np.array2string(optimal_values[index,:], floatmode='fixed', precision=1), 
            (0, image.shape[0] - 16), cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 0), 2)
    cv2.putText(image, "Action: " + str(cache.action(index)), (0, image.shape[0] - 32), cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 0), 2)
    cv2.putText(image, "Reward: " + str(cache.reward(index)), (0, image.shape[0] - 48), cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 0), 2)
    cv2.putText(image, "Terminal: " + str(cache.terminal(index)), (0, image.shape[0] - 64), cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 0), 2)
    cv2.imshow("Game Cache", image)

cv2.namedWindow("Game Cache")
cv2.createTrackbar("Index", "Game Cache", 0, len(cache) - 1, set_index)
cv2.createTrackbar("Frame", "Game Cache", 0, cache.state(0).shape[-1] - 1, set_frame)
show()

while True:
    k = cv2.waitKey(100)

    # run until escape or window is closed
    if k == 27 or cv2.getWindowProperty("Game Cache", cv2.WND_PROP_VISIBLE) < 1:
        break
