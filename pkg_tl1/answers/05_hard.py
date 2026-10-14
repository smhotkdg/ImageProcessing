"""미션 5 해설 — 나쁜 조건에서도 되는가"""
import sys, cv2, numpy as np
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent.parent))
import common as C


names = ["tl_night.png", "tl_dusk.png", "tl_rain.png"]
cells = []

for name in names:
    img = C.load(name)
    mask = C.color_mask(img, "red")
    print(f"{name:<14} 전체 {cv2.countNonZero(mask):5d} 화소 "
          f"/ 전구 영역 {C.lamp_count(mask):4d} 화소")
    cells.append(C.tag(img, name.replace(".png", "")))
    cells.append(C.tag(mask, "red mask"))
# 야간 401 / 역광 373 / 비 353 — 조건이 달라도 전구는 안정적으로 잡힙니다.

# 같은 장면을 BGR 범위로 처리하면 어떻게 되는지 비교합니다.
print()
for name in names:
    img = C.load(name)
    bgr = cv2.inRange(img, np.array((0, 0, 170)), np.array((90, 90, 255)))
    print(f"{name:<14} BGR 범위로는 전구 영역 {C.lamp_count(bgr):4d} 화소")

result = C.grid(cells, cols=2, size=(320, 120))
C.show("HARD", result)
C.wait(0)
C.close()
C.save("05_hard.png", result)
