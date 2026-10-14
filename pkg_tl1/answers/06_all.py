"""종합 미션 해설 — 세 색을 한 번에 세기"""
import sys, cv2, numpy as np
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent.parent))
import common as C



def detect(img):
    """세 색의 마스크 화소 수를 돌려줍니다. 2차시의 입력이 되는 함수입니다."""
    return {color: C.count(C.color_mask(img, color))
            for color in ("red", "yellow", "green")}


def annotate(img, counts):
    out = img.copy()
    y = 266
    for name, n in counts.items():
        cv2.putText(out, f"{name:<7}{n:5d} px", (26, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.62, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(out, f"{name:<7}{n:5d} px", (26, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.62, (255, 255, 255), 2, cv2.LINE_AA)
        y += 32
    return out


cells = []
for name in ("tl_day_red.png", "tl_day_green.png", "tl_night.png", "tl_none.png"):
    img = C.load(name)
    counts = detect(img)
    shown = annotate(img, counts)
    print(f"{name:<18} {counts}")
    cells.append(C.tag(shown, name.replace(".png", "")))

result = C.grid(cells, cols=2, size=(320, 180))
C.show("ALL", result)
C.wait(0)
C.close()
C.save("06_all.png", result)

print()
print("tl_none.png 에는 신호등이 없습니다. 그런데도 빨강이 1872 화소 나옵니다.")
print("정체는 OPEN 간판 1490 화소와 차량 제동등 네 개입니다.")
print("'화소 수가 가장 많은 색이 켜진 신호' 라는 규칙은 여기서 틀립니다.")
print("색 다음에 모양을 봐야 하는 이유입니다. (2차시)")
