"""미션 2 (20분) — inRange 로 초록과 노랑 찾기"""
import sys, cv2, numpy as np
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent.parent))
import common as C


green_img = C.load("tl_day_green.png")
yellow_img = C.load("tl_day_yellow.png")

hsv_g = cv2.cvtColor(green_img, cv2.COLOR_BGR2HSV)
hsv_y = cv2.cvtColor(yellow_img, cv2.COLOR_BGR2HSV)

# TODO 1) 초록 범위 H 60~90, S 90 이상, V 120 이상으로 마스크를 만드세요
gmask = np.zeros(hsv_g.shape[:2], np.uint8)

# TODO 2) 노랑 범위 H 18~35, S 120 이상, V 150 이상으로 마스크를 만드세요
ymask = np.zeros(hsv_y.shape[:2], np.uint8)

print("노랑 마스크", cv2.countNonZero(ymask), "화소")
print("초록 마스크", cv2.countNonZero(gmask), "화소")

result = C.grid([C.tag(yellow_img, "yellow lamp"), C.tag(ymask, "YELLOW mask"),
                 C.tag(green_img, "green lamp"), C.tag(gmask, "GREEN mask")],
                cols=2, size=(320, 180))
C.show("INRANGE", result)
C.wait(0)
C.close()
C.save("02_inrange.png", result)
