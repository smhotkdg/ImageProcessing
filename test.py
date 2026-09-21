import cv2

color = cv2.imread('road.jpg')
gray = cv2.imread('road.jpg', cv2.IMREAD_GRAYSCALE)

cv2.imshow('COLOR', color)
cv2.imshow('GRAY', gray)

cv2.waitKey(0)
cv2.destroyAllWindows()
