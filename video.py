from pathlib import Path
from unittest import result
import cv2

def drawImg(img):
    result = img.copy()
    cv2.line(result, (30, 60), (220, 60), (255, 0, 0), 3)
    cv2.rectangle(result, (80, 90), (1000, 600), (0, 255, 0), 3)
    cv2.circle(result, (640, 360), 30, (0, 0, 255), -1)

    cv2.putText(result, 'test', (30, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                (255, 255, 255), 2, cv2.LINE_AA)
    return result

SOURCE = str('road.mp4')
# 동영상 파일 열기
cap = cv2.VideoCapture(SOURCE)
# 동영상이 정상적으로 열렸는지 확인
if not cap.isOpened():
    raise RuntimeError('Camera/video open failed')


# 저장할 이미지 파일 경로 설정
target = Path('output/photo.png')
# output 폴더가 없으면 생성
target.parent.mkdir(exist_ok=True)
# 재생 속도 설정 (카메라: 1ms, 동영상: 42ms ≈ 24fps)
delay = 1 if isinstance(SOURCE, int) else 42
while True:
    # 프레임 한 장 읽기
    ret, frame = cap.read()
    # 더 이상 프레임이 없으면 반복 종료
    if not ret:
        break
    # 현재 프레임을 화면에 표시
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    result = drawImg(frame)
    
    #print(gray.min(), gray.max(), gray.mean())           
    gray = cv2.resize(gray, (640, 360))
    #threshold_value = gray.mean()
    threshold_value = gray.mean()
    print(gray.min(), gray.max(), gray.mean())   
    _, binary = cv2.threshold(gray, threshold_value, 255, cv2.THRESH_BINARY)
    tv, otsu = cv2.threshold(gray, 0, 255,
                         cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    a = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                              cv2.THRESH_BINARY_INV, 51, 8)

    
    cv2.imshow('CAPTURE', gray)
    cv2.imshow('threshold', binary)
    cv2.imshow('otsu', otsu)
    cv2.imshow('adaptive', a)
    # 키 입력 대기
    key = cv2.waitKey(delay) & 0xFF
    
    # 's' 키 → 현재 프레임을 이미지로 저장
    if key == ord('s'):
        print('saved:', cv2.imwrite(str(target), gray))
    if key == ord('r'):
        print('saved:', cv2.imwrite(str(target), result))
    # 'q' 키 → 프로그램 종료
    if key == ord('q'):
        break
# 동영상 자원 해
cap.release()
cv2.destroyAllWindows()
