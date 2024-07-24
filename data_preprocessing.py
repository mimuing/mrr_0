import os
import numpy as np
import cv2
from sklearn.model_selection import train_test_split

def load_images_from_folder(folder):
    images = []
    labels = []
    for filename in os.listdir(folder):
        img = cv2.imread(os.path.join(folder, filename), cv2.IMREAD_GRAYSCALE)
        if img is not None:
            images.append(img)
            # 레이블 추출 (파일명에 따라 다르게 설정 가능)
            label = filename.split('_')[0]  # 예: 'cat_001.jpg' -> 'cat'
            labels.append(label)
    return images, labels

def preprocess_images(images):
    processed_images = []
    for img in images:
        img = cv2.resize(img, (224, 224))  # 크기 조정
        img = img / 255.0  # 정규화
        processed_images.append(img)
    return np.array(processed_images)

if __name__ == "__main__":
    images, labels = load_images_from_folder('data/images')
    images = preprocess_images(images)
    X_train, X_val, y_train, y_val = train_test_split(images, labels, test_size=0.2)
