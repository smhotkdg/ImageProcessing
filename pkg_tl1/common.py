"""공통 도우미 — 신호등 인식 실습 패키지 (1~3차시 공용)

경로, 입출력, 처리 함수를 모아 둔 제공 코드입니다.
학생이 수정할 필요는 없습니다.

headless 환경(서버·CI)에서 창 없이 확인만 하려면
환경변수 CV_HEADLESS=1 을 설정하고 실행합니다.
"""
import os
from pathlib import Path
import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent
IMG = ROOT / "img"
OUT = ROOT / "output"
EXPECTED = ROOT / "expected"
OUT.mkdir(exist_ok=True)

HEADLESS = os.environ.get("CV_HEADLESS") == "1"

if HEADLESS:
    # 창을 띄울 수 없는 환경(서버·자동 채점)에서 GUI 호출을 무시합니다.
    # 학생용 코드는 순수 OpenCV 호출 그대로 두고, 여기서만 막습니다.
    cv2.namedWindow = lambda *a, **k: None
    cv2.createTrackbar = lambda *a, **k: None
    cv2.getTrackbarPos = lambda *a, **k: 0
    cv2.setMouseCallback = lambda *a, **k: None
    cv2.destroyAllWindows = lambda *a, **k: None
    cv2.imshow = lambda *a, **k: None


# ── 입출력 ────────────────────────────────────────────────
def load(name, gray=False):
    """img 폴더의 사진을 읽습니다. 실패하면 바로 알려 줍니다."""
    path = IMG / name
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise SystemExit(f"[오류] 사진을 열지 못했습니다: {path}")
    # 흑백은 항상 cvtColor 로 만듭니다. imread 의 흑백 모드와 값이 미세하게 다릅니다.
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if gray else img


def open_video(name):
    """img 폴더의 동영상을 엽니다. fps 가 이상하면 30 으로 대체합니다."""
    path = IMG / name
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise SystemExit(f"[오류] 동영상을 열지 못했습니다: {path}")
    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or fps != fps or fps <= 1:
        print("[안내] fps 를 읽지 못해 30 으로 사용합니다.")
        fps = 30.0
    return cap, fps


def frame_of(name, index, gray=False):
    """동영상의 index 번째 프레임을 한 장 꺼냅니다 (1부터 셉니다)."""
    cap, _ = open_video(name)
    img, i = None, 0
    while True:
        ret, f = cap.read()
        if not ret:
            break
        i += 1
        if i == index:
            img = f.copy()
            break
    cap.release()
    if img is None:
        raise SystemExit(f"[오류] {name} 에 {index}번 프레임이 없습니다")
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if gray else img


def make_writer(name, fps, size):
    """MJPG/AVI 저장기를 만듭니다. size 는 (너비, 높이) 순서입니다."""
    target = OUT / name
    fourcc = cv2.VideoWriter_fourcc(*"MJPG")
    writer = cv2.VideoWriter(str(target), fourcc, fps, size)
    if not writer.isOpened():
        raise SystemExit(f"[오류] 저장기를 만들지 못했습니다: {target}")
    return writer, target


def save(name, img):
    """output 폴더에 저장하고 경로를 출력합니다."""
    path = OUT / name
    cv2.imwrite(str(path), img)
    print(f"[저장] {path}")
    return path


def show(title, img):
    """headless 환경에서는 창을 띄우지 않습니다."""
    if not HEADLESS:
        cv2.imshow(title, img)


_TICK = [0]


def wait(delay=0):
    """headless 환경에서는 키 입력이 없는 것으로 보고 즉시 반환합니다.
    무한 루프도 끝나도록 140번째 호출부터는 q 를 누른 것으로 처리합니다."""
    if HEADLESS:
        _TICK[0] += 1
        return ord("q") if _TICK[0] > 140 else 255
    return cv2.waitKey(delay) & 0xFF


def close():
    if not HEADLESS:
        cv2.destroyAllWindows()


def nothing(value):
    pass


def stats(name, img):
    """min · max · mean 을 한 줄로 출력합니다."""
    print(f"{name:<14} min {int(img.min()):3d}  max {int(img.max()):3d}  "
          f"mean {float(img.mean()):5.1f}  (폭 {int(img.max()) - int(img.min())})")


def grid(cells, cols=2, size=(320, 180)):
    """결과들을 격자로 이어 붙여 한 장으로 만듭니다."""
    cw, ch = size
    cells = [c if c.ndim == 3 else cv2.cvtColor(c, cv2.COLOR_GRAY2BGR) for c in cells]
    rows = []
    for i in range(0, len(cells), cols):
        rows.append(np.hstack([cv2.resize(c, (cw, ch)) for c in cells[i:i + cols]]))
    return np.vstack(rows)


def tag(img, text, color=(255, 255, 255)):
    """결과 위에 영문 라벨을 얹습니다 (한글은 기본 글꼴로 표시되지 않습니다)."""
    out = img.copy()
    cv2.putText(out, text, (10, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (0, 0, 0), 4, cv2.LINE_AA)
    cv2.putText(out, text, (10, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.62, color, 1, cv2.LINE_AA)
    return out


# ── 처리 함수 (이미지와 프레임에 똑같이 사용) ────────────────
# ── 신호등 색 범위 ────────────────────────────────────────
# 수업에서 실제로 재서 정한 값입니다. 영상이 바뀌면 다시 재야 합니다.
RANGES = {
    "red":    [((0, 120, 120), (10, 255, 255)),        # H 가 0 에서 끊기므로
               ((170, 120, 120), (179, 255, 255))],    # 범위가 둘입니다
    "yellow": [((18, 120, 150), (35, 255, 255))],
    "green":  [((60, 90, 120), (90, 255, 255))],
}
# 전구를 둘러싼 21 x 21 영역 (합성 장면에서 전구 위치는 고정되어 있습니다)
LAMP_BOXES = {
    "red":    (slice(56, 77), slice(388, 409)),
    "yellow": (slice(84, 105), slice(388, 409)),
    "green":  (slice(111, 132), slice(388, 409)),
}


def to_hsv(img):
    """BGR 영상을 HSV 로 바꿉니다."""
    return cv2.cvtColor(img, cv2.COLOR_BGR2HSV)


def color_mask(img, color):
    """'red' · 'yellow' · 'green' 중 하나의 마스크를 만듭니다.
    빨강은 두 범위를 bitwise_or 로 합칩니다."""
    hsv = to_hsv(img)
    mask = None
    for lo, hi in RANGES[color]:
        m = cv2.inRange(hsv, np.array(lo, np.uint8), np.array(hi, np.uint8))
        mask = m if mask is None else cv2.bitwise_or(mask, m)
    return mask


def count(mask):
    """마스크에서 흰 화소의 개수."""
    return int(cv2.countNonZero(mask))


def lamp_count(mask, color="red"):
    """해당 전구 영역(21 x 21) 안에서만 센 화소 수. 배경의 가짜를 뺀 값입니다."""
    return int(cv2.countNonZero(mask[LAMP_BOXES[color]]))


def pixel_hsv(img, x, y):
    """한 점의 HSV 를 (H, S, V) 로 돌려줍니다. 순서가 [y, x] 인 점에 주의."""
    return tuple(int(v) for v in to_hsv(img)[y, x])


def apply_mask(img, mask):
    """마스크가 흰 곳만 원본에서 남깁니다."""
    return cv2.bitwise_and(img, img, mask=mask)


