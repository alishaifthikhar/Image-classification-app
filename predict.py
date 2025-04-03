# predict.py
import argparse
import torch
from torchvision import transforms
from PIL import Image
import json
import os
from model import load_checkpoint, predict
from utils import process_image

def main():
    # Define command line arguments
    parser = argparse.ArgumentParser(description="Predict flower name from an image using a trained network.")
    parser.add_argument("input", type=str, help="Path to the input image")
    parser.add_argument("checkpoint", type=str, help="Path to the model checkpoint")
    parser.add_argument("--top_k", type=int, default=5, help="Return top K most likely classes")
    parser.add_argument("--category_names", type=str, help="Path to JSON file mapping categories to real names")
    parser.add_argument("--gpu", action="store_true", help="Use GPU for inference")
    args = parser.parse_args()

    # Ensure the input image exists
    if not os.path.exists(args.input):
        print(f"Error: Image file '{args.input}' not found.")
        return

    # Ensure the checkpoint file exists
    if not os.path.exists(args.checkpoint):
        print(f"Error: Checkpoint file '{args.checkpoint}' not found.")
        return

    print(f"Loading model from checkpoint: {args.checkpoint}")
    model = load_checkpoint(args.checkpoint)

    print(f"Processing image: {args.input}")
    image = process_image(args.input)

    print("Performing inference...")
    probs, classes = predict(model, image, args.top_k, args.gpu)

    # Map categories to real names (if JSON file is provided)
    if args.category_names:
        if not os.path.exists(args.category_names):
            print(f"Warning: Category names file '{args.category_names}' not found. Using raw class labels.")
        else:
            with open(args.category_names, 'r') as f:
                try:
                    cat_to_name = json.load(f)
                    classes = [cat_to_name.get(str(cls), f"Class {cls}") for cls in classes]
                except json.JSONDecodeError:
                    print("Error: Unable to parse JSON file. Using raw class labels.")

    # Display results
    print("\n--- Prediction Results ---")
    for i in range(len(classes)):
        print(f"{i+1}: {classes[i]} ({probs[i]*100:.2f}%)")

if __name__ == '__main__':
    main()
