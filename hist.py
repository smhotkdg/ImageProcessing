import cv2  
import matplotlib.pyplot as plt

color = cv2.imread('road.jpg')
gray = cv2.cvtColor(color, cv2.COLOR_BGR2GRAY)

hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
print('hist shape:', hist.shape)
eq = cv2.equalizeHist(gray)

cv2.imshow('GRAY', gray)
cv2.imshow('EQUALIZE', eq)
hist_eq = cv2.calcHist([eq], [0], None, [256], [0, 256])
cv2.waitKey(0)
cv2.destroyAllWindows()
plt.plot(hist_eq, color='red')
plt.plot(hist, color='blue')
plt.xlim([0, 256])
plt.show()
