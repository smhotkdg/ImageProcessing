import sys, cv2, numpy as np
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent.parent))
import common as C

img = C.load("tl_day_red.png")
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

h,s,v = cv2.split(hsv)


# 1번 범위 (H: 0 ~ 10)
lower_red1 = np.array([0, 50, 50])
upper_red1 = np.array([10, 255, 255])

# 2번 범위 (H: 170 ~ 179)
lower_red2 = np.array([170, 50, 50])
upper_red2 = np.array([179, 255, 255])

mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
mask2 = cv2.inRange(hsv, lower_red2, upper_red2)

red_mask = cv2.bitwise_or(mask1, mask2)
h = cv2.inRange(h, 8, 20)

C.show("TL", img)
C.show("HSV", hsv)
C.show("h_red", red_mask)
cv2.waitKey(0)