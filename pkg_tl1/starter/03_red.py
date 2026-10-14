"""미션 3 (20분) — 빨강은 범위가 둘입니다
먼저 한 범위만 써 보고, 몇 화소를 놓치는지 확인한 뒤 고칩니다.
"""
import sys, cv2, numpy as np
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent.parent))
import common as C


img = C.load("tl_day_red.png")
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

# TODO 1) 낮은 쪽 범위 H 0~10 으로 마스크를 만드세요
m1 = np.zeros(hsv.shape[:2], np.uint8)

# TODO 2) 높은 쪽 범위 H 170~179 로 마스크를 만드세요
m2 = np.zeros(hsv.shape[:2], np.uint8)

# TODO 3) 두 마스크를 bitwise_or 로 합치세요
mask = m1

print("H   0-10  :", cv2.countNonZero(m1), "화소")
print("H 170-179 :", cv2.countNonZero(m2), "화소")
print("둘을 합치면:", cv2.countNonZero(mask), "화소")

night = C.load("tl_night.png")
nmask = C.color_mask(night, "red")

result = C.grid([C.tag(img, "DAY"), C.tag(mask, "red mask"),
                 C.tag(night, "NIGHT"), C.tag(nmask, "red mask")],
                cols=2, size=(320, 180))
C.show("RED", result)
C.wait(0)
C.close()
C.save("03_red.png", result)
