# model.py
import torch
from torch import nn, optim
from torchvision import models

def build_model(arch='vgg16', hidden_units=512):
    """Builds a pre-trained model with a custom classifier."""
    # Load a pre-trained model
    if arch == 'vgg16':
        model = models.vgg16(pretrained=True)
        input_size = 25088
        classifier = nn.Sequential(
            nn.Linear(input_size, hidden_units),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(hidden_units, 102),
            nn.LogSoftmax(dim=1)
        )
        model.classifier = classifier

    elif arch == 'resnet18':
        model = models.resnet18(pretrained=True)
        input_size = 512
        classifier = nn.Sequential(
            nn.Linear(input_size, hidden_units),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(hidden_units, 102),
            nn.LogSoftmax(dim=1)
        )
        model.fc = classifier  # ResNet uses `fc` instead of `classifier`

    else:
        raise ValueError(f"Unsupported architecture: {arch}")

    # Freeze parameters
    for param in model.parameters():
        param.requires_grad = False

    return model

def train_model(model, trainloader, validloader, epochs, learning_rate, gpu):
    """Trains the model and validates on a validation set."""
    device = torch.device("cuda" if gpu and torch.cuda.is_available() else "cpu")
    model.to(device)

    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.classifier.parameters() if hasattr(model, 'classifier') else model.fc.parameters(), lr=learning_rate)

    print(f"Training on {'GPU' if gpu and torch.cuda.is_available() else 'CPU'}...")

    for epoch in range(epochs):
        model.train()
        running_loss = 0

        for inputs, labels in trainloader:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        # Validation step
        model.eval()
        valid_loss = 0
        accuracy = 0

        with torch.no_grad():
            for inputs, labels in validloader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                valid_loss += criterion(outputs, labels).item()

                # Calculate accuracy
                ps = torch.exp(outputs)
                top_p, top_class = ps.topk(1, dim=1)
                equals = top_class == labels.view(*top_class.shape)
                accuracy += torch.mean(equals.type(torch.FloatTensor)).item()

        print(f"Epoch {epoch+1}/{epochs} | "
              f"Train Loss: {running_loss/len(trainloader):.3f} | "
              f"Validation Loss: {valid_loss/len(validloader):.3f} | "
              f"Validation Accuracy: {accuracy/len(validloader):.3f}")

def load_checkpoint(filepath):
    """Loads a trained model from a checkpoint file."""
    checkpoint = torch.load(filepath, map_location=torch.device("cuda" if torch.cuda.is_available() else "cpu"))

    model = build_model(checkpoint['arch'], checkpoint['hidden_units'])
    model.load_state_dict(checkpoint['state_dict'])
    model.class_to_idx = checkpoint['class_to_idx']

    print(f"Model loaded from {filepath} (Architecture: {checkpoint['arch']})")
    return model

def predict(model, image, topk=5, gpu=False):
    """Predicts the top K most probable classes for an image."""
    device = torch.device("cuda" if gpu and torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    with torch.no_grad():
        image = image.to(device)
        output = model(image)
        ps = torch.exp(output)
        top_p, top_class = ps.topk(topk, dim=1)

    return top_p.cpu().numpy()[0], top_class.cpu().numpy()[0]
