"""미션 3 해설 — 빨강은 범위가 둘입니다"""
import sys, cv2, numpy as np
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent.parent))
import common as C


img = C.load("tl_day_red.png")
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

m1 = cv2.inRange(hsv, np.array((0, 120, 120)), np.array((10, 255, 255)))
m2 = cv2.inRange(hsv, np.array((170, 120, 120)), np.array((179, 255, 255)))
mask = cv2.bitwise_or(m1, m2)

print("H   0-10  :", cv2.countNonZero(m1), "화소")       # 1110
print("H 170-179 :", cv2.countNonZero(m2), "화소")       # 1131
print("둘을 합치면:", cv2.countNonZero(mask), "화소")     # 2241
# 어느 한쪽만 쓰면 전구의 절반이 사라집니다.

patch = hsv[62:71, 394:403, 0]
print("전구 주변 H 값:", sorted(set(int(v) for v in patch.ravel())))

night = C.load("tl_night.png")
nmask = C.color_mask(night, "red")
print("야간 빨강 :", cv2.countNonZero(nmask), "화소")     # 2317

result = C.grid([C.tag(img, "DAY"), C.tag(mask, "red mask"),
                 C.tag(night, "NIGHT"), C.tag(nmask, "red mask")],
                cols=2, size=(320, 180))
C.show("RED", result)
C.wait(0)
C.close()
C.save("03_red.png", result)
