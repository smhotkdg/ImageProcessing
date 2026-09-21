import cv2

img = cv2.imread('road.jpg')
if img is None:
    #여기 이미지 없는 경우 예외처리 해줘야함
    raise FileNotFoundError('road.jpg')
print(img.shape, img.dtype)
cv2.imshow('IMG', img)
cv2.waitKey(0)
cv2.destroyAllWindows()
