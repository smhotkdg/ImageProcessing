import cv2


path = str('road.jpg')
color = cv2.imread(path)
gray = cv2.cvtColor(color, cv2.COLOR_BGR2GRAY)
print(gray[40, 60], gray[170, 330], gray[260, 80])   # 210 132 113
print(gray.min(), gray.max(), gray.mean())           
