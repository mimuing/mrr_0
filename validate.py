import torch
from model import get_model
from data_preprocessing import load_images_from_folder, preprocess_images

def validate(model, val_loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in val_loader:
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    accuracy = 100 * correct / total
    print(f'Validation Accuracy: {accuracy:.2f}%')

if __name__ == "__main__":
    images, labels = load_images_from_folder('data/images')
    images = preprocess_images(images)
    model = get_model()
    validate(model, images)
