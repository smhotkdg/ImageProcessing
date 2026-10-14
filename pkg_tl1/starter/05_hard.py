"""미션 5 (15분) — 나쁜 조건에서도 되는가
같은 범위 한 벌로 야간 · 역광 · 비 세 장을 처리합니다.
"""
import sys, cv2, numpy as np
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent.parent))
import common as C


names = ["tl_night.png", "tl_dusk.png", "tl_rain.png"]
cells = []

for name in names:
    img = C.load(name)

    # TODO) C.color_mask 로 빨강 마스크를 만드세요
    mask = np.zeros(img.shape[:2], np.uint8)

    print(f"{name:<14} 전체 {cv2.countNonZero(mask):5d} 화소 "
          f"/ 전구 영역 {C.lamp_count(mask):4d} 화소")
    cells.append(C.tag(img, name.replace(".png", "")))
    cells.append(C.tag(mask, "red mask"))

result = C.grid(cells, cols=2, size=(320, 120))
C.show("HARD", result)
C.wait(0)
C.close()
C.save("05_hard.png", result)
