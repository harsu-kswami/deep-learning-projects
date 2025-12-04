# TODO: GPU augmentations
import random
import cv2
import numpy as np

def random_brightness(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    factor = 0.5 + random.random()  # 0.5–1.5
    hsv[:, :, 2] = np.clip(hsv[:, :, 2] * factor, 0, 255)
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)

def random_shift(img, steering, max_shift=40, steer_corr=0.002):
    h, w, _ = img.shape
    dx = random.randint(-max_shift, max_shift)
    M = np.float32([[1, 0, dx], [0, 1, 0]])
    img = cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_REPLICATE)
    steering = steering + dx * steer_corr
    return img, steering

def augment(img, steering):
    # img: RGB uint8, steering: float
    if random.random() < 0.5:
        img = random_brightness(img)
    if random.random() < 0.5:
        img, steering = random_shift(img, steering)
    return img, steering
