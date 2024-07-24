from train import train
from validate import validate
from model import get_model
from data_preprocessing import load_images_from_folder, preprocess_images
from config import BATCH_SIZE, EPOCHS

if __name__ == "__main__":
    images, labels = load_images_from_folder('data/images')
    images = preprocess_images(images)
