import cv2
import matplotlib.pyplot as plt

color = cv2.imread('road.jpg')
gray = cv2.cvtColor(color, cv2.COLOR_BGR2GRAY)
hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
eq = cv2.equalizeHist(gray)
eq_hist = cv2.calcHist([eq], [0], None, [256], [0, 256])

cv2.imwrite('road_eq.jpg', eq)
plt.plot(eq_hist, color='r')
plt.xlim([0, 256])
plt.show()