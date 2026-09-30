import cv2

path = str('road.jpg')
color = cv2.imread(path)

gray = cv2.cvtColor(color, cv2.COLOR_BGR2GRAY)
threshold_value = gray.mean()
print(gray.min(), gray.max(), gray.mean())   
_, binary = cv2.threshold(gray, threshold_value, 255, cv2.THRESH_BINARY)


tv, otsu = cv2.threshold(gray, 0, 255,
                         cv2.THRESH_BINARY + cv2.THRESH_OTSU)

cv2.imshow('GRAY', gray)
cv2.imshow('BINARY', binary)
cv2.waitKey(0)
cv2.destroyAllWindows()