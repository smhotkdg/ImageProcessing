from pathlib import Path
import cv2
# 재생할 동영상 파일 경로 설정 (웹캠 사용 시 0으로 변경)
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
    cv2.imshow('CAPTURE', frame)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) 
    cv2.imshow('gray',gray)
    # 키 입력 대기
    key = cv2.waitKey(delay) & 0xFF
    # 's' 키 → 현재 프레임을 이미지로 저장
    if key == ord('s'):
        print('saved:', cv2.imwrite(str(target), frame))
    # 'q' 키 → 프로그램 종료
    if key == ord('q'):
        break
# 동영상 자원 해
cap.release()
# 열린 모든 창 닫기
cv2.destroyAllWindows()
