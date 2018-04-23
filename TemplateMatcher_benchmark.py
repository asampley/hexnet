from TemplateMatcher import TemplateMatcher
from WindowGrabber import WindowGrabber
import cv2
import time

wg = WindowGrabber()
print("Select window to compare to in 3 seconds")
time.sleep(3)
wg.set_window_to_focus()

template = cv2.Canny(cv2.imread('res/gameover.png'), 50, 200)
tm = TemplateMatcher(template, 0.8)

def preprocess( image ):
    return cv2.Canny(image, 50, 200)

methods = {
        'sqdiff': cv2.TM_SQDIFF,
        'sqdiff_normed': cv2.TM_SQDIFF_NORMED,
        'ccorr': cv2.TM_CCORR,
        'ccorr_normed': cv2.TM_CCORR_NORMED,
        'ccoeff': cv2.TM_CCOEFF,
        'ccoeff_normed': cv2.TM_CCOEFF_NORMED
        }

for method_name, method in methods.items():
    tm = TemplateMatcher(template, 0.5, method)
    t0 = time.time()
    matchRating = 0
    for i in range(100):
        matchRating += tm.matchRating(preprocess(wg.grab()))
    t1 = time.time()
    print("100 iterations using " + method_name + " in " + str(t1 - t0) + ". Average rating: " + str(matchRating / 100))
