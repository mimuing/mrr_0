import cv2
import numpy as np
import matplotlib.pyplot as plt

class ImageAugmentor:
    @classmethod    
    def augment_hsv(cls, image):
        h, s, v = cv2.split(image)
        h = h * np.random.uniform(low=1, high=6)
        s = s * np.random.uniform(low=1, high=4)
        v = v * np.random.uniform(low=1, high=6)
        # HSV 채널 병합
        merge_image = cv2.merge((h, s, v))
        # 이미지 값을 [0, 1] 범위로 클리핑
        merge_image = np.clip(merge_image / merge_image.max(), 0, 1)
        # 이미지를 [0, 255] 범위로 변환
        merge_image_uint8 = (merge_image * 255).astype(np.uint8)
        # 이미지를 BGR로 변환
        merge_image_bgr = cv2.cvtColor(merge_image_uint8, cv2.COLOR_HSV2BGR)
        # 이미지를 회색조로 변환
        merge_image_gray = cv2.cvtColor(merge_image_bgr, cv2.COLOR_BGR2GRAY)
        return merge_image_gray
    
    @staticmethod
    def rgb_norm(image):
        r, g, b = cv2.split(image)
        r = (r - np.mean(r)) / np.std(r)
        g = (g - np.mean(g)) / np.std(g)
        b = (b - np.mean(b)) / np.std(b)
        r = ((r - r.min()) / (r.max() - r.min()) * 255).astype(np.uint8)
        g = ((g - g.min()) / (g.max() - g.min()) * 255).astype(np.uint8)
        b = ((b - b.min()) / (b.max() - b.min()) * 255).astype(np.uint8)
        return cv2.merge((r, g, b))
        
