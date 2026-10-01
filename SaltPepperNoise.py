import cv2
import numpy as np


def salt_pepper_noise(image, amount=0.05, salt_ratio=0.5, seed=None):
    """
    image: uint8 이미지 (흑백 또는 컬러)
    amount: 잡음을 적용할 픽셀 비율 (0~1)
    salt_ratio: 잡음 중 흰색 비율 (0~1)
    seed: 결과 재현을 위한 난수 시드
    """
    if not 0 <= amount <= 1 or not 0 <= salt_ratio <= 1:
        raise ValueError("amount와 salt_ratio는 0~1이어야 합니다.")
    if image.dtype != np.uint8:
        raise ValueError("uint8 이미지를 입력하세요.")

    rng = np.random.default_rng(seed)
    noisy = image.copy()

    height, width = image.shape[:2]
    count = int(height * width * amount)

    # 중복 없이 잡음 위치 선택
    indices = rng.choice(height * width, size=count, replace=False)
    rows, cols = np.unravel_index(indices, (height, width))

    salt_count = int(count * salt_ratio)
    noisy[rows[:salt_count], cols[:salt_count]] = 255  # 소금: 흰색
    noisy[rows[salt_count:], cols[salt_count:]] = 0    # 후추: 검은색

    return noisy


image = cv2.imread("road.jpg")
if image is None:
    raise FileNotFoundError("road.jpg 파일을 찾을 수 없습니다.")

noisy_image = salt_pepper_noise(image, amount=0.05, seed=42)
cv2.imwrite("noisy_image.png", noisy_image)