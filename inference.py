import torch
from model import get_model
from data_preprocessing import preprocess_images
import numpy as np
import cv2

def predict(model, image_path):
    model.eval()
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    image = cv2.resize(image, (224, 224)) / 255.0
    image = np.expand_dims(image, axis=0)  # 배치 차원 추가
    with torch.no_grad():
        output = model(torch.tensor(image).float())
        _, predicted = torch.max(output.data, 1)
    return predicted.item()

if __name__ == "__main__":
    model = get_model()
    prediction = predict(model, 'data/test_image.jpg')
    print(f'Predicted Class: {prediction}')
