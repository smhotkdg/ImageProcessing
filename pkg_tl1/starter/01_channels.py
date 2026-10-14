"""미션 1 (15분) — H · S · V 를 나눠서 보기
같은 사진을 다른 자로 재면 무엇이 보이는지 확인합니다.
"""
import sys, cv2, numpy as np
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent.parent))
import common as C


img = C.load("tl_day_red.png")

# TODO 1) BGR 을 HSV 로 바꾸세요 (cv2.cvtColor, cv2.COLOR_BGR2HSV)
hsv = img

# TODO 2) cv2.split 으로 H, S, V 세 장으로 나누세요
h, s, v = None, None, None

print("원본 모양 ", img.shape)
print("HSV 모양  ", hsv.shape)

result = C.grid([C.tag(img, "ORIGINAL"), C.tag(h, "H  hue 0-179"),
                 C.tag(s, "S  saturation"), C.tag(v, "V  value")],
                cols=2, size=(320, 180))
C.show("CHANNELS", result)
C.wait(0)
C.close()
C.save("01_channels.png", result)
