"""미션 1 해설 — H · S · V 를 나눠서 보기"""
import sys, cv2, numpy as np
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent.parent))
import common as C


img = C.load("tl_day_red.png")
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
h, s, v = cv2.split(hsv)

print("원본 모양 ", img.shape)              # (360, 640, 3)
print("HSV 모양  ", hsv.shape)              # (360, 640, 3)
print("채널 모양 ", h.shape)                # (360, 640)

y, x = 66, 398                               # 빨강 전구 중심
print(f"전구 BGR  {tuple(int(c) for c in img[y, x])}")    # (90, 87, 203)
print(f"전구 HSV  {tuple(int(c) for c in hsv[y, x])}")    # (179, 146, 203)

result = C.grid([C.tag(img, "ORIGINAL"), C.tag(h, "H  hue 0-179"),
                 C.tag(s, "S  saturation"), C.tag(v, "V  value")],
                cols=2, size=(320, 180))
C.show("CHANNELS", result)
C.wait(0)
C.close()
C.save("01_channels.png", result)
# H 를 회색으로 보면 의미를 알기 어렵습니다. 0 과 179 가 둘 다 빨강인데
# 한쪽은 까맣고 한쪽은 하얗게 보이기 때문입니다.
