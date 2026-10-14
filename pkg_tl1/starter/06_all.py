"""종합 미션 (20분) — 세 색을 한 번에 세기
이 함수가 다음 주(2차시)의 입력이 됩니다. 반드시 완성하세요.
"""
import sys, cv2, numpy as np
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent.parent))
import common as C



def detect(img):
    """세 색의 마스크 화소 수를 {'red': n, 'yellow': n, 'green': n} 으로 돌려줍니다."""
    result = {}
    for color in ("red", "yellow", "green"):
        # TODO 1) C.color_mask 로 마스크를 만들고 화소 수를 세세요
        result[color] = 0
    return result


def annotate(img, counts):
    """화면 왼쪽 아래에 숫자를 올립니다."""
    out = img.copy()
    y = 266
    for name, n in counts.items():
        cv2.putText(out, f"{name:<7}{n:5d} px", (26, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.62, (255, 255, 255), 2, cv2.LINE_AA)
        y += 32
    return out


cells = []
for name in ("tl_day_red.png", "tl_day_green.png", "tl_night.png", "tl_none.png"):
    img = C.load(name)

    # TODO 2) detect 로 세어서 annotate 로 표시하세요
    counts = detect(img)
    shown = annotate(img, counts)

    print(f"{name:<18} {counts}")
    cells.append(C.tag(shown, name.replace(".png", "")))

result = C.grid(cells, cols=2, size=(320, 180))
C.show("ALL", result)
C.wait(0)
C.close()
C.save("06_all.png", result)
