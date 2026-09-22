import cv2


path = str('road.jpg')
color = cv2.imread(path)
gray = cv2.cvtColor(color, cv2.COLOR_BGR2GRAY)
edge = cv2.Canny(gray, 100, 200)
if color is None or gray is None:
    raise FileNotFoundError(path)
print(color.shape, gray.shape)
cv2.imshow('COLOR', color)
cv2.imshow('GRAY', gray)
cv2.waitKey(0)
cv2.destroyAllWindows()
