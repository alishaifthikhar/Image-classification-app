# utils.py
import torch
from torchvision import datasets, transforms
from PIL import Image
import numpy as np
import os

def load_data(data_dir):
    """Loads datasets and returns train, validation, and test dataloaders."""
    train_dir = os.path.join(data_dir, 'train')
    valid_dir = os.path.join(data_dir, 'valid')
    test_dir = os.path.join(data_dir, 'test')

    # Define transforms
    train_transforms = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    valid_transforms = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    test_transforms = valid_transforms  # Use the same transforms for validation and testing

    # Load datasets
    try:
        train_data = datasets.ImageFolder(train_dir, transform=train_transforms)
        valid_data = datasets.ImageFolder(valid_dir, transform=valid_transforms)
        test_data = datasets.ImageFolder(test_dir, transform=test_transforms)
    except FileNotFoundError as e:
        raise FileNotFoundError(f"Error: {e}. Make sure the dataset path is correct.")

    # Define dataloaders
    trainloader = torch.utils.data.DataLoader(train_data, batch_size=64, shuffle=True)
    validloader = torch.utils.data.DataLoader(valid_data, batch_size=32)
    testloader = torch.utils.data.DataLoader(test_data, batch_size=32)

    print(f"Data loaded successfully!\n"
          f"Train samples: {len(train_data)}, Validation samples: {len(valid_data)}, Test samples: {len(test_data)}")

    return trainloader, validloader, testloader

def process_image(image_path):
    """Processes an image for a PyTorch model."""
    try:
        image = Image.open(image_path)
    except Exception as e:
        raise ValueError(f"Error loading image: {image_path}. Details: {e}")

    # Resize and crop image
    image = image.resize((256, 256))
    left = (256 - 224) / 2
    top = (256 - 224) / 2
    right = left + 224
    bottom = top + 224
    image = image.crop((left, top, right, bottom))

    # Convert to numpy array and normalize
    np_image = np.array(image) / 255
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    np_image = (np_image - mean) / std

    # Transpose color channels and convert to tensor
    np_image = np_image.transpose((2, 0, 1))
    return torch.from_numpy(np_image).float()
