import torch
import torch.optim as optim
import torch.nn.functional as F
from model import get_model
from data_preprocessing import load_images_from_folder, preprocess_images

def train(model, train_loader, optimizer, epochs):
    model.train()
    for epoch in range(epochs):
        for images, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(images)
            loss = F.cross_entropy(outputs, labels)
            loss.backward()
            optimizer.step()
            print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}')

if __name__ == "__main__":
    images, labels = load_images_from_folder('data/images')
    images = preprocess_images(images)
    model = get_model()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    train(model, images, optimizer, epochs=10)
