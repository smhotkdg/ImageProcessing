import cv2

color = cv2.imread('road.jpg')
gray = cv2.cvtColor(color, cv2.COLOR_BGR2GRAY)

grayPath = 'road_gray.jpg'
saveResult = cv2.imwrite(grayPath, gray)

if saveResult:
    print('Image saved successfully.')
    gray2 = cv2.imread(grayPath)
    #gray2 = cv2.imread(grayPath,cv2.IMREAD_GRAYSCALE)
    print('shape:', gray2.shape)
    print('gray shape:', gray.shape)

    cv2.imshow('GRAY', gray2)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
else:
    print('Failed to save image.')

