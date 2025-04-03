# train.py
import argparse
import torch
from torch import nn, optim
from torchvision import datasets, transforms, models
from model import build_model, train_model
from utils import load_data
import os

def main():
    # Define command line arguments
    parser = argparse.ArgumentParser(description="Train a neural network on flower data.")
    parser.add_argument("data_directory", type=str, help="Path to the dataset")
    parser.add_argument("--save_dir", type=str, default="saved_models", help="Directory to save checkpoints")
    parser.add_argument("--arch", type=str, choices=["vgg16", "resnet18"], default="vgg16", help="Model architecture (vgg16 or resnet18)")
    parser.add_argument("--learning_rate", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--hidden_units", type=int, default=512, help="Number of hidden units")
    parser.add_argument("--epochs", type=int, default=5, help="Number of epochs")
    parser.add_argument("--gpu", action="store_true", help="Use GPU for training")
    args = parser.parse_args()

    # Ensure save directory exists
    os.makedirs(args.save_dir, exist_ok=True)

    print(f"Loading data from: {args.data_directory}")
    trainloader, validloader, _ = load_data(args.data_directory)

    print(f"Building model using {args.arch} architecture...")
    model = build_model(args.arch, args.hidden_units)

    print("Starting training...")
    train_model(model, trainloader, validloader, args.epochs, args.learning_rate, args.gpu)

    # Save the checkpoint
    checkpoint = {
        'arch': args.arch,
        'hidden_units': args.hidden_units,
        'state_dict': model.state_dict(),
        'class_to_idx': trainloader.dataset.class_to_idx
    }

    checkpoint_path = os.path.join(args.save_dir, "checkpoint.pth")
    torch.save(checkpoint, checkpoint_path)
    print(f"Model checkpoint saved at: {checkpoint_path}")

if __name__ == '__main__':
    main()
