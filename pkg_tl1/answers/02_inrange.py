"""미션 2 해설 — inRange 로 초록과 노랑 찾기"""
import sys, cv2, numpy as np
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent.parent))
import common as C


green_img = C.load("tl_day_green.png")
yellow_img = C.load("tl_day_yellow.png")

hsv_g = cv2.cvtColor(green_img, cv2.COLOR_BGR2HSV)
hsv_y = cv2.cvtColor(yellow_img, cv2.COLOR_BGR2HSV)

gmask = cv2.inRange(hsv_g, np.array((60, 90, 120)), np.array((90, 255, 255)))
ymask = cv2.inRange(hsv_y, np.array((18, 120, 150)), np.array((35, 255, 255)))

print("노랑 마스크", cv2.countNonZero(ymask), "화소")      # 376
print("초록 마스크", cv2.countNonZero(gmask), "화소")      # 1542
print("초록 중 전구 영역만", C.lamp_count(gmask, "green"), "화소")

# 전구 하나는 대략 350~400 화소입니다. 초록 1542 중 나머지는 가로수입니다.
result = C.grid([C.tag(yellow_img, "yellow lamp"), C.tag(ymask, "YELLOW mask"),
                 C.tag(green_img, "green lamp"), C.tag(gmask, "GREEN mask")],
                cols=2, size=(320, 180))
C.show("INRANGE", result)
C.wait(0)
C.close()
C.save("02_inrange.png", result)
