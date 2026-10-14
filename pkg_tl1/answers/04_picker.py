"""미션 4 해설 — 트랙바로 내 범위 찾기"""
import sys, cv2, numpy as np
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent.parent))
import common as C


img = C.load("tl_day_red.png")
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

cv2.namedWindow("PICKER")
for name, init, mx in (("Hmin", 0, 179), ("Hmax", 10, 179),
                       ("Smin", 120, 255), ("Smax", 255, 255),
                       ("Vmin", 120, 255), ("Vmax", 255, 255)):
    cv2.createTrackbar(name, "PICKER", init, mx, C.nothing)

while True:
    lo = np.array([cv2.getTrackbarPos(n, "PICKER") for n in ("Hmin", "Smin", "Vmin")])
    hi = np.array([cv2.getTrackbarPos(n, "PICKER") for n in ("Hmax", "Smax", "Vmax")])
    mask = cv2.inRange(hsv, lo, hi)
    view = np.hstack([C.apply_mask(img, mask), cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)])
    C.show("PICKER", cv2.resize(view, (960, 270)))
    key = C.wait(30)
    if key == ord("q"):
        break
    if key == ord("s"):
        print("찾은 범위  lo =", tuple(lo), " hi =", tuple(hi),
              " 화소", cv2.countNonZero(mask))
        C.save("04_picker.png", view)
C.close()

# headless 환경에서는 트랙바가 모두 0 이므로 기본 범위로 한 장만 남깁니다.
if C.HEADLESS:
    mask = C.color_mask(img, "red")
    view = np.hstack([C.apply_mask(img, mask), cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)])
    print("기본 범위로 저장  화소", cv2.countNonZero(mask))
    C.save("04_picker.png", view)
# S 하한을 올리면 회색이 빠지고, V 하한을 올리면 어두운 곳이 빠집니다.
# 간판처럼 밝고 선명한 빨강은 끝까지 남습니다. 색만으로는 지울 수 없습니다.
