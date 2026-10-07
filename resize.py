import cv2

img = cv2.imread('test.png')         
small = img[140:185, 300:345]        

for flag in [cv2.INTER_NEAREST, cv2.INTER_LINEAR,
             cv2.INTER_CUBIC,   cv2.INTER_AREA]:
    up = cv2.resize(small, (360, 360), interpolation=flag)
    cv2.imshow(f'flag = {flag}', up)

cv2.waitKey(0)
cv2.destroyAllWindows()

