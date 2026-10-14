"""미션 4 (20분) — 트랙바로 내 범위 찾기
s 저장 / q 종료.  찾은 여섯 값을 반드시 적어 두세요. 다음 주에 씁니다.
"""
import sys, cv2, numpy as np
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent.parent))
import common as C


img = C.load("tl_day_red.png")
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

cv2.namedWindow("PICKER")
# TODO) 트랙바 여섯 개를 만드세요. H 의 최대는 179, S 와 V 는 255 입니다
#       ('Hmin', 0) ('Hmax', 10) ('Smin', 120) ('Smax', 255) ('Vmin', 120) ('Vmax', 255)

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
